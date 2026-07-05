/* oxidize-c finetune: port of oxidize-finetuning's SFT trainer — frozen
 * quantized base, trainable LoRA on the LM head, AdamW, windowed batched
 * forward, EOS-packed chunks. Exports adapter_manifest.json in the same
 * schema as the Rust crate.
 *
 * ponytail: SFT LM-head LoRA only (matches the Rust SftTrainer). DPO/RLHF/
 * QLoRA-merge stay Rust-side; port them when someone actually trains with
 * them on this engine. JSONL rows use the "text" field or
 * instruction/input/output; "messages" arrays are not parsed. */
#include "oc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

/* ---- LoRA adapter (mirrors oxidize-finetuning lora.rs) ---- */

oc_lora *oc_lora_new(size_t in_dim, size_t out_dim, size_t rank, float alpha,
                     uint64_t seed) {
  if (rank == 0) rank = 1;
  oc_lora *l = calloc(1, sizeof(*l));
  l->in_dim = in_dim;
  l->out_dim = out_dim;
  l->rank = rank;
  l->scale = alpha / (float)rank;
  size_t an = rank * in_dim, bn = out_dim * rank;
  l->a = malloc(an * sizeof(float));
  l->b = calloc(bn, sizeof(float));
  l->grad_a = calloc(an, sizeof(float));
  l->grad_b = calloc(bn, sizeof(float));
  l->m_a = calloc(an, sizeof(float));
  l->v_a = calloc(an, sizeof(float));
  l->m_b = calloc(bn, sizeof(float));
  l->v_b = calloc(bn, sizeof(float));
  /* xorshift init of A, B starts at zero (same scheme as init_lora_a) */
  float s = 1.0f / sqrtf((float)rank);
  uint64_t state = (seed * 0x9E3779B97F4A7C15ull) | 1;
  for (size_t i = 0; i < an; ++i) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    float u = (float)(uint32_t)(state >> 32) / 4294967295.0f * 2.0f - 1.0f;
    l->a[i] = u * s;
  }
  return l;
}

void oc_lora_free(oc_lora *l) {
  if (!l) return;
  free(l->a); free(l->b); free(l->grad_a); free(l->grad_b);
  free(l->m_a); free(l->v_a); free(l->m_b); free(l->v_b);
  free(l);
}

static void lora_down(const oc_lora *l, const float *xs, float *hidden,
                      size_t count) {
  size_t rank = l->rank, in = l->in_dim;
#pragma omp parallel for schedule(static)
  for (size_t t = 0; t < count; ++t)
    for (size_t r = 0; r < rank; ++r)
      hidden[t * rank + r] = oc_dot_f32(l->a + r * in, xs + t * in, in);
}

void oc_lora_forward(const oc_lora *l, const float *xs, float *outs,
                     size_t count) {
  size_t rank = l->rank, out = l->out_dim;
  float *hidden = malloc(count * rank * sizeof(float));
  lora_down(l, xs, hidden, count);
#pragma omp parallel for schedule(static)
  for (size_t t = 0; t < count; ++t)
    for (size_t o = 0; o < out; ++o)
      outs[t * out + o] +=
          l->scale * oc_dot_f32(l->b + o * rank, hidden + t * rank, rank);
  free(hidden);
}

void oc_lora_backward(oc_lora *l, const float *xs, const float *grad_outs,
                      size_t count) {
  size_t rank = l->rank, in = l->in_dim, out = l->out_dim;
  float scale = l->scale;
  float *hidden = malloc(count * rank * sizeof(float));
  lora_down(l, xs, hidden, count);

  /* grad_b[o][r] += scale * sum_t g[t][o] * hidden[t][r] */
#pragma omp parallel for schedule(static)
  for (size_t o = 0; o < out; ++o) {
    float *gb = l->grad_b + o * rank;
    for (size_t t = 0; t < count; ++t) {
      float g = scale * grad_outs[t * out + o];
      if (g == 0.0f) continue;
      const float *h = hidden + t * rank;
      for (size_t r = 0; r < rank; ++r) gb[r] += g * h[r];
    }
  }

  /* grad_hidden[t][r] = scale * sum_o g[t][o] * b[o][r] */
  float *gh = calloc(count * rank, sizeof(float));
#pragma omp parallel for schedule(static)
  for (size_t t = 0; t < count; ++t) {
    float *ghr = gh + t * rank;
    const float *g = grad_outs + t * out;
    for (size_t o = 0; o < out; ++o) {
      if (g[o] == 0.0f) continue;
      float gs = scale * g[o];
      const float *br = l->b + o * rank;
      for (size_t r = 0; r < rank; ++r) ghr[r] += gs * br[r];
    }
  }

  /* grad_a[r][i] += sum_t gh[t][r] * xs[t][i] */
#pragma omp parallel for schedule(static)
  for (size_t r = 0; r < rank; ++r) {
    float *ga = l->grad_a + r * in;
    for (size_t t = 0; t < count; ++t) {
      float g = gh[t * rank + r];
      if (g == 0.0f) continue;
      const float *x = xs + t * in;
      for (size_t i = 0; i < in; ++i) ga[i] += g * x[i];
    }
  }
  free(gh);
  free(hidden);
}

static void adamw(float *p, const float *g, float *m, float *v, size_t n,
                  float lr, float wd, size_t step) {
  const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  float bc1 = 1.0f - powf(b1, (float)step), bc2 = 1.0f - powf(b2, (float)step);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    p[i] *= 1.0f - lr * wd;
    m[i] = b1 * m[i] + (1.0f - b1) * g[i];
    v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
    p[i] -= lr * (m[i] / bc1) / (sqrtf(v[i] / bc2) + eps);
  }
}

void oc_lora_step(oc_lora *l, float lr, float weight_decay, size_t step) {
  adamw(l->a, l->grad_a, l->m_a, l->v_a, l->rank * l->in_dim, lr, weight_decay, step);
  adamw(l->b, l->grad_b, l->m_b, l->v_b, l->out_dim * l->rank, lr, weight_decay, step);
}

void oc_lora_zero_grad(oc_lora *l) {
  memset(l->grad_a, 0, l->rank * l->in_dim * sizeof(float));
  memset(l->grad_b, 0, l->out_dim * l->rank * sizeof(float));
}

/* logits -> grad_scale * (softmax - onehot) in place; returns summed loss */
float oc_ce_grad(float *logits, const uint32_t *targets, size_t count,
                 size_t vocab, float grad_scale, size_t *n_out) {
  double loss = 0;
  size_t n = 0;
#pragma omp parallel for schedule(static) reduction(+ : loss, n)
  for (size_t t = 0; t < count; ++t) {
    float *row = logits + t * vocab;
    uint32_t tgt = targets[t];
    if (tgt >= vocab) oc_die("finetune: target %u out of vocab %zu", tgt, vocab);
    float mx = row[0];
    for (size_t i = 1; i < vocab; ++i)
      if (row[i] > mx) mx = row[i];
    double es = 0;
    for (size_t i = 0; i < vocab; ++i) es += exp((double)row[i] - mx);
    float lse = mx + (float)log(es);
    loss += (double)(lse - row[tgt]);
    n += 1;
    for (size_t i = 0; i < vocab; ++i) {
      float p = expf(row[i] - lse);
      row[i] = (p - (i == tgt ? 1.0f : 0.0f)) * grad_scale;
    }
  }
  *n_out = n;
  return (float)loss;
}

/* ---- dataset: JSONL ("text" or instruction/input/output) or plain text ---- */

/* extract "key": "..." string value with basic escape decoding; NULL if absent */
static char *json_str_field(const char *line, const char *key) {
  char pat[64];
  snprintf(pat, sizeof pat, "\"%s\"", key);
  const char *p = strstr(line, pat);
  if (!p) return NULL;
  p += strlen(pat);
  while (*p == ' ' || *p == '\t') ++p;
  if (*p != ':') return NULL;
  ++p;
  while (*p == ' ' || *p == '\t') ++p;
  if (*p != '"') return NULL;
  ++p;
  size_t cap = 256, n = 0;
  char *out = malloc(cap);
  while (*p && *p != '"') {
    char c = *p++;
    if (c == '\\' && *p) {
      char e = *p++;
      c = e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e;
      if (e == 'u') { /* skip \uXXXX (rare in SFT text; emit '?') */
        for (int k = 0; k < 4 && *p; ++k) ++p;
        c = '?';
      }
    }
    if (n + 1 >= cap) out = realloc(out, cap *= 2);
    out[n++] = c;
  }
  out[n] = 0;
  return out;
}

typedef struct { char **texts; size_t n; } dataset_t;

static void ds_push(dataset_t *d, char *text) {
  d->texts = realloc(d->texts, (d->n + 1) * sizeof(char *));
  d->texts[d->n++] = text;
}

static dataset_t load_dataset(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) oc_die("finetune: cannot open dataset %s", path);
  dataset_t d = {0};
  char *line = NULL;
  size_t cap = 0;
  ssize_t len;
  while ((len = getline(&line, &cap, f)) > 0) {
    const char *p = line;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\n' || *p == 0) continue;
    if (*p != '{') { /* plain-text: one example per line */
      char *t = strdup(p);
      t[strcspn(t, "\n")] = 0;
      ds_push(&d, t);
      continue;
    }
    char *text = json_str_field(p, "text");
    if (text && *text) {
      ds_push(&d, text);
      continue;
    }
    free(text);
    char *inst = json_str_field(p, "instruction");
    char *in = json_str_field(p, "input");
    char *out = json_str_field(p, "output");
    if ((inst && *inst) || (out && *out)) {
      size_t need = 96 + (inst ? strlen(inst) : 0) + (in ? strlen(in) : 0) +
                    (out ? strlen(out) : 0);
      char *t = malloc(need);
      if (in && *in)
        snprintf(t, need, "<|im_start|>user\n%s\n%s\n<|im_start|>assistant\n%s\n",
                 inst ? inst : "", in, out ? out : "");
      else
        snprintf(t, need, "<|im_start|>user\n%s\n<|im_start|>assistant\n%s\n",
                 inst ? inst : "", out ? out : "");
      ds_push(&d, t);
    }
    free(inst); free(in); free(out);
  }
  free(line);
  fclose(f);
  if (d.n == 0) oc_die("finetune: dataset %s is empty", path);
  return d;
}

/* ---- chunk packing (EOS-separated, same as dataset.rs pack_chunks) ---- */

typedef struct { uint32_t *ids; size_t n; } chunk_t;

static double now_s_(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void finetune_usage(void) {
  fprintf(stderr,
          "Usage: oxidize-c finetune --model m.gguf --data data.jsonl [options]\n"
          "  --out DIR           adapter output dir (default: lora-out)\n"
          "  --epochs N          default 1\n"
          "  --lr F              learning rate (default 2e-4)\n"
          "  --rank N            LoRA rank (default 16)\n"
          "  --alpha F           LoRA alpha (default 32)\n"
          "  --weight-decay F    default 0\n"
          "  --max-seq-len N     packed chunk length (default 512)\n"
          "  --window N          batched forward window (default 64)\n"
          "  --tokens-per-step N optimizer cadence in tokens (default 256)\n"
          "  --warmup N          LR warmup steps (default 10)\n"
          "  --seed N            default 42\n"
          "  --no-pack           one chunk per example (no EOS packing)\n"
          "  --eval              report loss only, no training\n");
  exit(2);
}

static void export_manifest(const char *dir, const oc_lora *l) {
  if (mkdir(dir, 0755) != 0) { /* ok if it exists */ }
  char path[1024];
  snprintf(path, sizeof path, "%s/adapter_manifest.json", dir);
  FILE *f = fopen(path, "w");
  if (!f) oc_die("finetune: cannot write %s", path);
  fprintf(f, "{\n  \"rank\": %zu,\n  \"alpha_scale\": %g,\n  \"adapters\": [\n",
          l->rank, (double)l->scale);
  fprintf(f, "    {\n      \"target\": \"OutputHead\",\n      \"in_dim\": %zu,\n"
             "      \"out_dim\": %zu,\n      \"lora_a\": [", l->in_dim, l->out_dim);
  for (size_t i = 0; i < l->rank * l->in_dim; ++i)
    fprintf(f, "%s%.7g", i ? "," : "", (double)l->a[i]);
  fprintf(f, "],\n      \"lora_b\": [");
  for (size_t i = 0; i < l->out_dim * l->rank; ++i)
    fprintf(f, "%s%.7g", i ? "," : "", (double)l->b[i]);
  fprintf(f, "]\n    }\n  ]\n}\n");
  if (fclose(f) != 0) oc_die("finetune: write failed for %s", path);
  printf("adapter -> %s\n", path);
}

int oc_finetune_main(int argc, char **argv) {
  const char *model_path = NULL, *data_path = NULL, *out_dir = "lora-out";
  size_t rank = 16, epochs = 1, max_seq_len = 512, window = 64,
         tokens_per_step = 256, warmup = 10;
  float alpha = 32.0f, lr = 2e-4f, weight_decay = 0.0f;
  uint64_t seed = 42;
  bool pack = true, eval_only = false;

  for (int i = 0; i < argc; ++i) {
#define FVAL() (i + 1 < argc ? argv[++i] : (finetune_usage(), (char *)0))
    if (!strcmp(argv[i], "--model")) model_path = FVAL();
    else if (!strcmp(argv[i], "--data")) data_path = FVAL();
    else if (!strcmp(argv[i], "--out")) out_dir = FVAL();
    else if (!strcmp(argv[i], "--epochs")) epochs = strtoull(FVAL(), 0, 10);
    else if (!strcmp(argv[i], "--lr")) lr = strtof(FVAL(), NULL);
    else if (!strcmp(argv[i], "--rank")) rank = strtoull(FVAL(), 0, 10);
    else if (!strcmp(argv[i], "--alpha")) alpha = strtof(FVAL(), NULL);
    else if (!strcmp(argv[i], "--weight-decay")) weight_decay = strtof(FVAL(), NULL);
    else if (!strcmp(argv[i], "--max-seq-len")) max_seq_len = strtoull(FVAL(), 0, 10);
    else if (!strcmp(argv[i], "--window")) window = strtoull(FVAL(), 0, 10);
    else if (!strcmp(argv[i], "--tokens-per-step")) tokens_per_step = strtoull(FVAL(), 0, 10);
    else if (!strcmp(argv[i], "--warmup")) warmup = strtoull(FVAL(), 0, 10);
    else if (!strcmp(argv[i], "--seed")) seed = strtoull(FVAL(), 0, 10);
    else if (!strcmp(argv[i], "--no-pack")) pack = false;
    else if (!strcmp(argv[i], "--eval")) eval_only = true;
    else finetune_usage();
#undef FVAL
  }
  if (!model_path || !data_path) finetune_usage();
  if (max_seq_len < 2) max_seq_len = 2;
  if (window < 2) window = 2;
  if (tokens_per_step < 1) tokens_per_step = 1;

  oc_model *m = oc_model_load(model_path, max_seq_len);
  oc_tokenizer *tok = oc_tokenizer_load(m->g);
  if (!tok) oc_die("finetune: unsupported tokenizer");
  size_t h = m->cfg.hidden_size, vocab = m->cfg.vocab_size;

  /* EOS id: first token flagged end-of-generation */
  uint32_t eos = 0;
  for (uint32_t i = 0; i < (uint32_t)vocab; ++i)
    if (oc_is_eog(tok, i)) { eos = i; break; }

  dataset_t ds = load_dataset(data_path);
  printf("dataset: %zu examples\n", ds.n);

  /* tokenize + pack into chunks of max_seq_len */
  chunk_t *chunks = NULL;
  size_t n_chunks = 0;
  uint32_t *cur = malloc(max_seq_len * sizeof(uint32_t));
  size_t cur_n = 0;
  for (size_t e = 0; e < ds.n; ++e) {
    size_t tn = 0;
    uint32_t *ids = oc_tokenize(tok, ds.texts[e], true, &tn);
    size_t cap = max_seq_len * 4 > 2 ? max_seq_len * 4 : 2;
    if (tn > cap) tn = cap;
    if (tn < 2) { free(ids); continue; }
    if (!pack) {
      size_t n = tn > max_seq_len ? max_seq_len : tn;
      chunks = realloc(chunks, (n_chunks + 1) * sizeof(chunk_t));
      chunks[n_chunks].ids = malloc(n * sizeof(uint32_t));
      memcpy(chunks[n_chunks].ids, ids, n * sizeof(uint32_t));
      chunks[n_chunks++].n = n;
      free(ids);
      continue;
    }
    for (size_t i = 0; i <= tn; ++i) { /* <= tn: trailing EOS separator */
      uint32_t id = i < tn ? ids[i] : eos;
      cur[cur_n++] = id;
      if (cur_n == max_seq_len) {
        chunks = realloc(chunks, (n_chunks + 1) * sizeof(chunk_t));
        chunks[n_chunks].ids = malloc(cur_n * sizeof(uint32_t));
        memcpy(chunks[n_chunks].ids, cur, cur_n * sizeof(uint32_t));
        chunks[n_chunks++].n = cur_n;
        cur_n = 0;
      }
    }
    free(ids);
  }
  if (pack && cur_n >= 2) {
    chunks = realloc(chunks, (n_chunks + 1) * sizeof(chunk_t));
    chunks[n_chunks].ids = malloc(cur_n * sizeof(uint32_t));
    memcpy(chunks[n_chunks].ids, cur, cur_n * sizeof(uint32_t));
    chunks[n_chunks++].n = cur_n;
  }
  free(cur);
  if (n_chunks == 0) oc_die("finetune: no usable chunks after tokenization");
  printf("chunks: %zu x <=%zu tokens\n", n_chunks, max_seq_len);

  oc_lora *lora = oc_lora_new(h, vocab, rank, alpha, seed);
  printf("lora: rank %zu, %zu params\n", lora->rank,
         lora->rank * (h + vocab));

  float *normed = malloc(window * h * sizeof(float));
  float *logits = malloc(window * vocab * sizeof(float));
  float grad_scale = 1.0f / (float)tokens_per_step;
  size_t opt_step = 0, accum = 0, total_tokens = 0;
  double total_loss = 0, started = now_s_(), last_report = started;
  size_t tokens_since = 0;

  for (size_t epoch = 0; epoch < (eval_only ? 1 : epochs); ++epoch) {
    double epoch_loss = 0;
    size_t epoch_tokens = 0;
    for (size_t c = 0; c < n_chunks; ++c) {
      if (chunks[c].n < 2) continue;
      oc_reset_state(m);
      const uint32_t *inputs = chunks[c].ids;
      const uint32_t *targets = chunks[c].ids + 1;
      size_t len = chunks[c].n - 1;
      for (size_t pos = 0; pos < len; ) {
        size_t kk = len - pos < window ? len - pos : window;
        oc_forward_train(m, inputs + pos, kk, pos, normed, logits);
        oc_lora_forward(lora, normed, logits, kk);
        size_t n = 0;
        float loss = oc_ce_grad(logits, targets + pos, kk, vocab,
                                eval_only ? 0.0f : grad_scale, &n);
        if (!eval_only) {
          oc_lora_backward(lora, normed, logits, kk);
          accum += n;
          if (accum >= tokens_per_step) {
            opt_step++;
            float wlr = (warmup && opt_step < warmup)
                            ? lr * (float)opt_step / (float)warmup : lr;
            oc_lora_step(lora, wlr, weight_decay, opt_step);
            oc_lora_zero_grad(lora);
            accum = 0;
          }
        }
        epoch_loss += loss;
        epoch_tokens += n;
        total_loss += loss;
        total_tokens += n;
        tokens_since += n;
        double now = now_s_();
        if (now - last_report >= 10.0) {
          printf("  epoch %zu step %zu tokens %zu loss %.4f | %.2f tok/s\n",
                 epoch + 1, opt_step, total_tokens,
                 epoch_tokens ? epoch_loss / epoch_tokens : 0.0,
                 tokens_since / (now - last_report));
          fflush(stdout);
          last_report = now;
          tokens_since = 0;
        }
        pos += kk;
      }
    }
    printf("epoch %zu: loss %.4f over %zu tokens\n", epoch + 1,
           epoch_tokens ? epoch_loss / epoch_tokens : 0.0, epoch_tokens);
  }

  if (!eval_only) {
    if (accum > 0) { /* flush trailing partial accumulation */
      opt_step++;
      float wlr = (warmup && opt_step < warmup)
                      ? lr * (float)opt_step / (float)warmup : lr;
      oc_lora_step(lora, wlr, weight_decay, opt_step);
    }
    double elapsed = now_s_() - started;
    printf("done: %zu steps, %zu tokens, mean loss %.4f, %.2f tok/s\n",
           opt_step, total_tokens,
           total_tokens ? total_loss / total_tokens : 0.0,
           elapsed > 0 ? total_tokens / elapsed : 0.0);
    export_manifest(out_dir, lora);
  }

  free(normed);
  free(logits);
  oc_lora_free(lora);
  for (size_t i = 0; i < n_chunks; ++i) free(chunks[i].ids);
  free(chunks);
  for (size_t i = 0; i < ds.n; ++i) free(ds.texts[i]);
  free(ds.texts);
  oc_tokenizer_free(tok);
  oc_model_free(m);
  return 0;
}
