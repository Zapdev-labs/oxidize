/* oxidize-c CLI: one-shot generation or HTTP/WebSocket server. */
#include "oc.h"
#include "gen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int oc_serve(oc_model *m, oc_tokenizer *tok, const char *host, int port,
             float temperature, size_t draft_k, int spec_mode);

/* SMT threads thrash the memory-bound gemv; default to physical cores.
 * ponytail: single-socket assumption */
static int physical_cores(void) {
  int cores = 0;
  FILE *f = fopen("/proc/cpuinfo", "r");
  if (f) {
    char line[256];
    while (fgets(line, sizeof line, f)) {
      if (strncmp(line, "cpu cores", 9) == 0) {
        const char *c = strchr(line, ':');
        if (c) cores = atoi(c + 1);
        break;
      }
    }
    fclose(f);
  }
  return cores > 0 ? cores : 4;
}

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct { oc_tokenizer *tok; } stream_ud;
static void print_token(uint32_t id, void *ud) {
  stream_ud *s = ud;
  char frag[256];
  size_t fn = oc_detokenize(s->tok, id, frag, sizeof frag);
  fwrite(frag, 1, fn, stdout);
  fflush(stdout);
}

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s --model <path.gguf> [options]\n"
          "       %s prune --input in.gguf --output out.gguf [options]\n"
          "       %s finetune --model m.gguf --data data.jsonl [options]\n",
          prog, prog, prog);
  fprintf(stderr,
          "  --prompt <str>      prompt text (raw)\n"
          "  --chat              wrap prompt in ChatML (qwen chat models)\n"
          "  --max-tokens N      decode budget (default 64)\n"
          "  --ctx N             context/KV limit (default 8192)\n"
          "  --kv-int8           int8 KV cache (4x smaller, faster full ctx)\n"
          "  --draft N           speculative draft length (default 4, 0=off)\n"
          "  --spec MODE         drafting: ngram (default) | mtp | off\n"
          "  --threads N         worker threads (default: physical cores)\n"
          "  --temperature F     sampling temperature (default 0 = greedy)\n"
          "  --top-k N --top-p F sampling filters\n"
          "  --seed N            RNG seed\n"
          "  --no-bos            don't prepend BOS\n"
          "  --stream            stream text to stdout\n"
          "  --serve             run HTTP+WebSocket server instead of one-shot\n"
          "  --port N            server port (default 8090)\n"
          "  --host A            bind address (default 127.0.0.1; 0.0.0.0 for LAN)\n");
  exit(2);
}

int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "prune"))
    return oc_prune_main(argc - 2, argv + 2);
  if (argc > 1 && !strcmp(argv[1], "finetune"))
    return oc_finetune_main(argc - 2, argv + 2);

  const char *model_path = NULL, *prompt = "Hello", *host = "127.0.0.1";
  size_t max_tokens = 64, top_k = 0, ctx = 8192, draft = 4;
  int kv_int8 = 0;
  int spec_mode = 2; /* off by default: spec pays only at high acceptance on CPU */
  float temperature = 0.0f, top_p = 1.0f;
  int threads = 0, port = 8090;
  bool no_bos = false, stream = false, serve = false, chat = false;
  uint64_t seed = 0;

  for (int i = 1; i < argc; ++i) {
#define VAL() (i + 1 < argc ? argv[++i] : (usage(argv[0]), (char *)0))
    if (!strcmp(argv[i], "--model")) model_path = VAL();
    else if (!strcmp(argv[i], "--prompt")) prompt = VAL();
    else if (!strcmp(argv[i], "--max-tokens")) max_tokens = strtoull(VAL(), 0, 10);
    else if (!strcmp(argv[i], "--ctx")) ctx = strtoull(VAL(), 0, 10);
    else if (!strcmp(argv[i], "--draft")) draft = strtoull(VAL(), 0, 10);
    else if (!strcmp(argv[i], "--spec")) {
      const char *v = VAL();
      spec_mode = !strcmp(v, "mtp") ? 1 : !strcmp(v, "off") ? 2 : 0;
    }
    else if (!strcmp(argv[i], "--threads")) threads = atoi(VAL());
    else if (!strcmp(argv[i], "--temperature")) temperature = strtof(VAL(), 0);
    else if (!strcmp(argv[i], "--top-k")) top_k = strtoull(VAL(), 0, 10);
    else if (!strcmp(argv[i], "--top-p")) top_p = strtof(VAL(), 0);
    else if (!strcmp(argv[i], "--seed")) seed = strtoull(VAL(), 0, 10);
    else if (!strcmp(argv[i], "--kv-int8")) kv_int8 = 1;
    else if (!strcmp(argv[i], "--no-bos")) no_bos = true;
    else if (!strcmp(argv[i], "--stream")) stream = true;
    else if (!strcmp(argv[i], "--serve")) serve = true;
    else if (!strcmp(argv[i], "--chat")) chat = true;
    else if (!strcmp(argv[i], "--port")) port = atoi(VAL());
    else if (!strcmp(argv[i], "--host")) host = VAL();
    else usage(argv[0]);
#undef VAL
  }
  if (!model_path) usage(argv[0]);
  oc_gen_seed(seed);
#ifdef _OPENMP
  if (threads <= 0 && !getenv("OMP_NUM_THREADS")) {
    threads = physical_cores();
    /* server mode: leave two cores for the rest of the system */
    if (serve && threads > 4) threads -= 2;
  }
  if (threads > 0) omp_set_num_threads(threads);
#else
  (void)threads;
#endif

  double t0 = now_s();
  oc_model *m = oc_model_load(model_path, ctx, kv_int8);
  double t_load = now_s() - t0;
  const oc_config *c = &m->cfg;
  size_t n_gdn = 0;
  for (size_t l = 0; l < c->layer_count; ++l)
    if (m->layers[l].is_gdn) n_gdn++;
  fprintf(stderr,
          "loaded in %.1fs: layers=%zu (%zu GDN + %zu attn) hidden=%zu heads=%zu/"
          "%zu head_dim=%zu vocab=%zu ctx=%zu (kv %zu) inter=%zu mtp=%s\n",
          t_load, c->layer_count, n_gdn, c->layer_count - n_gdn, c->hidden_size,
          c->n_heads, c->kv_heads, c->head_dim, c->vocab_size, c->context_size,
          m->kv_ctx, c->intermediate_size, m->mtp ? "yes" : "no");

  oc_tokenizer *tok = oc_tokenizer_load(m->g);
  if (!tok) oc_die("no usable tokenizer in GGUF");

  if (serve) return oc_serve(m, tok, host, port, temperature, draft, spec_mode);

  /* ---- one-shot CLI ---- */
  char *wrapped = NULL;
  const char *ptext = prompt;
  if (chat) {
    size_t need = strlen(prompt) + 128;
    wrapped = malloc(need);
    snprintf(wrapped, need,
             "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
    ptext = wrapped;
  }
  size_t n_prompt;
  uint32_t *ids = oc_tokenize(tok, ptext, !no_bos && !chat, &n_prompt);
  free(wrapped);
  if (n_prompt == 0) oc_die("empty prompt after tokenize");
  if (n_prompt + max_tokens + 2 > m->kv_ctx)
    max_tokens = m->kv_ctx > n_prompt + 2 ? m->kv_ctx - n_prompt - 2 : 0;

  oc_gen g = {0};
  g.m = m;
  g.tok = tok;
  g.temperature = temperature;
  g.top_k = top_k;
  g.top_p = top_p;
  g.draft_k = draft;
  g.spec_mode = spec_mode;
  stream_ud sud = {tok};
  if (stream) { g.on_token = print_token; g.ud = &sud; }

  uint32_t *out = malloc(max_tokens * sizeof(uint32_t));
  double t1 = now_s();
  size_t n_out = oc_generate(&g, ids, n_prompt, max_tokens, out);
  double t_gen = now_s() - t1;
  if (stream) fputc('\n', stdout);

  fprintf(stderr, "gen: %zu prompt + %zu tokens in %.2fs (%.2f tok/s)",
          n_prompt, n_out, t_gen, t_gen > 0 ? (double)n_out / t_gen : 0.0);
  if (g.drafted)
    fprintf(stderr, "  [mtp: drafted %zu accepted %zu = %.0f%%]", g.drafted,
            g.accepted, 100.0 * (double)g.accepted / (double)g.drafted);
  fprintf(stderr, "\n");
  if (!stream) {
    char frag[256];
    printf("text: ");
    for (size_t i = 0; i < n_out; ++i) {
      size_t fn = oc_detokenize(tok, out[i], frag, sizeof frag);
      fwrite(frag, 1, fn, stdout);
    }
    printf("\n");
  }
  free(out);
  free(ids);
  oc_tokenizer_free(tok);
  oc_model_free(m);
  return 0;
}
