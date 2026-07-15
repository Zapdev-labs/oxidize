/* oxidize-c CLI: Gemma 4 GGUF text generation on CPU.
 * Modes: one-shot (--prompt), interactive multi-turn (--chat, KV cache
 * reused across turns), metadata dump (--inspect). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gguf.h"
#include "model.h"
#include "model_qwen36.h" /* speculative decode with the qwen36 MTP draft head */
#include "quant.h"
#include "sampler.h"
#include "tensor.h"
#include "tokenizer.h"
#include "vision.h" /* --image: CLIP/SigLIP mmproj encoder */

#define RECENT_CAP 64 /* repeat-penalty window */

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(const char* argv0) {
  fprintf(stderr,
          "usage: %s --model path.gguf [--prompt \"...\"] [--max-tokens N]\n"
          "          [--temp T] [--top-k K] [--top-p P] [--min-p P]\n"
          "          [--repeat-penalty R] [--ctx N] [--threads N]\n"
          "          [--system \"...\"] [--chat] [--inspect] [--version]\n"
          "          [--bench] [--greedy] [--raw] [--seed N]\n"
          "          [--kv-type f32|f16|q8|q4]  KV cache precision (default f32)\n"
          "          [--kv-quant | --no-kv-quant]  alias for --kv-type q4|f32\n"
          "          [--spec] [--draft-tokens N]  speculative decode via the\n"
          "              qwen36 MTP head (greedy only; default N=4)\n"
          "          --mmproj vision.gguf --image raw.f32  encode an image with a\n"
          "              CLIP/SigLIP mmproj tower; --image is a raw f32 CHW file of\n"
          "              3*S*S values (S = clip.vision.image_size), pre-normalized\n",
          argv0);
}

/* The kernels are selected from cpuid at startup, so the same binary is fast
 * on one machine and slow on another with no visible difference. Print what it
 * picked. OC_ISA=scalar|avx2|avx512 forces it down. */
static void print_isa(void) {
  printf("oxidize-c  isa: %s", oc_isa_active_name());
  if (oc_isa_active() != oc_isa_available()) printf(" (forced; cpu supports more)");
  printf("\n");
}

/* ---------- --inspect: dump GGUF metadata + tensor table ---------- */

static void print_value(const GgufValue* v) {
  switch (v->kind) {
    case GGUF_T_U8: case GGUF_T_U16: case GGUF_T_U32: case GGUF_T_U64:
      printf("%llu", (unsigned long long)v->v.u); break;
    case GGUF_T_BOOL: printf(v->v.u ? "true" : "false"); break;
    case GGUF_T_I8: case GGUF_T_I16: case GGUF_T_I32: case GGUF_T_I64:
      printf("%lld", (long long)v->v.i); break;
    case GGUF_T_F32: case GGUF_T_F64: printf("%g", v->v.f); break;
    case GGUF_T_STRING: {
      size_t len = v->v.str.len;
      if (len > 80) printf("\"%.80s...\" (%zu chars)", v->v.str.ptr, len);
      else printf("\"%.*s\"", (int)len, v->v.str.ptr);
      break;
    }
    case GGUF_T_ARRAY: {
      size_t n = v->v.arr.n, show = n < 4 ? n : 4;
      printf("[");
      for (size_t i = 0; i < show; ++i) {
        if (i) printf(", ");
        print_value(&v->v.arr.items[i]);
      }
      if (n > show) printf(", ... %zu items", n);
      printf("]");
      break;
    }
    default: printf("?kind=%d", v->kind);
  }
}

static int inspect(const char* path) {
  char err[256] = {0};
  GgufFile g;
  if (gguf_open(&g, path, err, sizeof(err)) != 0) {
    fprintf(stderr, "error: %s\n", err);
    return 1;
  }
  print_isa();
  printf("gguf v%u  %zu KVs  %zu tensors  %.2f MiB\n", g.version, g.n_kv,
         g.n_tensors, (double)g.size / (1024.0 * 1024.0));
  printf("\nmetadata:\n");
  for (size_t i = 0; i < g.n_kv; ++i) {
    printf("  %-48s = ", g.kvs[i].key);
    print_value(&g.kvs[i].val);
    printf("\n");
  }
  printf("\ntensors:\n");
  for (size_t i = 0; i < g.n_tensors; ++i) {
    const GgufTensorInfo* t = &g.tensors[i];
    printf("  %-48s type=%-3u [", t->name, t->ggml_type);
    for (uint32_t d = 0; d < t->n_dims; ++d)
      printf("%s%llu", d ? " x " : "", (unsigned long long)t->dims[d]);
    printf("]\n");
  }
  gguf_close(&g);
  return 0;
}

/* ---------- --image: CLIP/SigLIP mmproj vision encoder ----------
 * Loads a vision GGUF (usually mmproj-*.gguf), reads a raw f32 CHW image
 * (already resized to clip.vision.image_size and normalized), runs the tower,
 * and reports the [n_image_tokens x projection_dim] embedding. The projection
 * dim equals the target LM hidden size: those rows are what a LLaVA decoder
 * substitutes for the <image> placeholder in its INPUT-embedding stream. The
 * text models in this build accept token ids (not input embeddings), so the
 * splice itself is documented, not executed — this path produces + reports the
 * embeddings that would splice. */
static int run_vision(const char* mmproj_path, const char* image_path) {
  char err[256] = {0};
  GgufFile g;
  if (gguf_open(&g, mmproj_path, err, sizeof(err)) != 0) {
    fprintf(stderr, "error: %s\n", err);
    return 1;
  }
  oc_pool_init(0);
  int rc = 1;
  float *img = NULL, *emb = NULL;
  VisionEncoder* v = NULL;
  if (vision_load(&v, &g, err, sizeof(err)) != 0) {
    fprintf(stderr, "error: %s\n", err);
    goto done;
  }
  const VisionConfig* C = vision_config(v);
  print_isa();
  printf("vision tower: image %zux%zu  patch %zu  hidden %zu  heads %zu  layers %zu\n"
         "  ffn %zu  patches %zu  projector->%zu  act %s  class-token %s\n",
         C->image_size, C->image_size, C->patch_size, C->hidden, C->n_head,
         C->n_layer, C->inter, C->n_patches, C->proj_dim,
         C->use_gelu ? "gelu" : "quick-gelu", C->has_class_token ? "yes" : "no");

  size_t need = 3 * C->image_size * C->image_size;
  img = malloc(need * sizeof(float));
  if (!img) { fprintf(stderr, "error: out of memory\n"); goto done; }
  FILE* f = fopen(image_path, "rb");
  if (!f) { fprintf(stderr, "error: cannot open %s\n", image_path); goto done; }
  size_t rd = fread(img, sizeof(float), need, f);
  int extra = fgetc(f) != EOF; /* file should hold exactly `need` f32 values */
  fclose(f);
  if (rd != need || extra) {
    fprintf(stderr,
            "error: %s must be exactly %zu f32 values (3 x %zu x %zu CHW), got %zu%s\n",
            image_path, need, C->image_size, C->image_size, rd, extra ? "+" : "");
    goto done;
  }

  size_t nt = 0, dim = 0;
  emb = vision_encode(v, img, &nt, &dim, err, sizeof(err));
  if (!emb) { fprintf(stderr, "error: %s\n", err); goto done; }

  double n2 = 0.0;
  float lo = emb[0], hi = emb[0];
  for (size_t i = 0; i < nt * dim; ++i) {
    n2 += (double)emb[i] * (double)emb[i];
    if (emb[i] < lo) lo = emb[i];
    if (emb[i] > hi) hi = emb[i];
  }
  printf("image embedding: [%zu tokens x %zu dims]  L2 %.4f  range [%.4f, %.4f]\n",
         nt, dim, sqrt(n2), (double)lo, (double)hi);
  printf("  row0:");
  for (size_t i = 0; i < dim && i < 6; ++i) printf(" %.4f", (double)emb[i]);
  printf("%s\n", dim > 6 ? " ..." : "");
  printf("splice: substitute these %zu rows for the <image> placeholder token in the\n"
         "  LM's input-embedding sequence (dim %zu must equal the LM hidden size).\n"
         "  Decoder splice is documented only: the text models here take token ids.\n",
         nt, dim);
  rc = 0;

done:
  free(emb);
  free(img);
  vision_free(v);
  gguf_close(&g);
  oc_pool_free();
  return rc;
}

/* ---------- generation ---------- */

/* generate() drives the model through the family-agnostic hooks from model.h,
 * so it never branches on architecture. */
typedef struct {
  void* model;
  ModelForwardFn forward;
  ModelForwardBatchFn forward_batch;
  size_t vocab, ctx;
  int gemma_stops; /* apply gemma4 hardcoded stop ids (1, 106) */
  Tokenizer* tok;
  SamplerConfig* sampler;
  size_t pos; /* next KV position */
  int32_t eot_tok;
  int32_t recent[RECENT_CAP];
  size_t n_recent;
  /* Speculative decode (qwen36 MTP head only). qm is non-NULL only when the
   * loaded model is qwen36 WITH a usable MTP block AND the sampler is pure
   * greedy with no penalties (the path is bit-exact only there). */
  Qwen36Model* qm;
  int spec;
  size_t draft_tokens;
  size_t spec_steps, spec_accept; /* per-generate stats, filled by decode_spec */
} GenState;

static void push_recent(GenState* gs, int32_t id) {
  if (gs->n_recent < RECENT_CAP) {
    gs->recent[gs->n_recent++] = id;
  } else {
    memmove(gs->recent, gs->recent + 1, (RECENT_CAP - 1) * sizeof(int32_t));
    gs->recent[RECENT_CAP - 1] = id;
  }
}

static int is_stop(const GenState* gs, int32_t next) {
  const Tokenizer* tok = gs->tok;
  return next < 0 || (gs->gemma_stops && (next == 1 || next == 106)) ||
         next == (int32_t)tok->eos_id ||
         (tok->eot_id >= 0 && next == (int32_t)tok->eot_id) ||
         (gs->eot_tok >= 0 && next == gs->eot_tok);
}

/* Speculative decode loop (qwen36 MTP). Each step drafts k tokens, verifies
 * them in one batched target forward, commits the argmax-matching prefix plus
 * one target token, and streams the committed tokens. Bit-exact to the plain
 * greedy loop below (every committed token is the target's own argmax). `seed`
 * is the last prefill token; on entry gs->qm->logits/normed hold the pending
 * target distribution + hidden (left by prefill). Returns generated count. */
static int decode_spec(GenState* gs, int32_t seed, int max_tokens, char* buf,
                       size_t bufn) {
  Qwen36Model* qm = gs->qm;
  size_t k = gs->draft_tokens ? gs->draft_tokens : 4;
  int32_t draft[64], out[66];
  if (k > 63) k = 63; /* draft[]/out[] bound */
  if (k + 1 > qm->batch_cap) k = qm->batch_cap - 1; /* verify k + commit k+1 rows */
  if (k < 1) k = 1; /* main gates batch_cap >= 2, so this holds */
  int produced = 0, stop = 0;
  gs->spec_steps = 0;
  gs->spec_accept = 0;
  while (produced < max_tokens && !stop) {
    if (gs->pos + k + 1 > gs->ctx) break; /* not enough room for a full step */
    qwen36_mtp_draft(qm, seed, draft, k);
    size_t accepted = 0;
    size_t len = qwen36_spec_step(qm, draft, k, gs->pos, out, &accepted);
    if (len == 0) break; /* spec unavailable near ctx end */
    gs->spec_steps++;
    gs->spec_accept += accepted;
    for (size_t j = 0; j < len; ++j) {
      if (produced >= max_tokens || is_stop(gs, out[j])) {
        stop = 1;
        break;
      }
      size_t w = tokenizer_decode_token(gs->tok, out[j], buf, bufn);
      fwrite(buf, 1, w, stdout);
      fflush(stdout);
      push_recent(gs, out[j]);
      produced++;
    }
    /* Commit the full step's positions (spec_step already forwarded them); any
     * post-stop token stays in the KV as harmless context, keeping pos and the
     * recurrent state consistent. */
    gs->pos += len;
    seed = out[len - 1];
  }
  return produced;
}

/* Prefill ids then decode up to max_tokens, streaming to stdout.
 * Returns generated count, or -1 on error. */
static int generate(GenState* gs, const int32_t* ids, size_t n_ids,
                    int max_tokens, double* prefill_s, double* decode_s) {
  if (gs->pos + n_ids >= gs->ctx) {
    fprintf(stderr, "error: context full (%zu + %zu tokens > ctx %zu)\n",
            gs->pos, n_ids, gs->ctx);
    return -1;
  }
  /* Prefill the whole prompt in ONE batched pass. Feeding it token by token
   * made a prompt cost exactly as much as generating it: every weight was read
   * from DRAM once per prompt token instead of once per batch. */
  double t0 = now_sec();
  float* logits = gs->forward_batch(gs->model, ids, n_ids, gs->pos, true);
  for (size_t i = 0; i < n_ids; ++i) push_recent(gs, ids[i]);
  gs->pos += n_ids;
  *prefill_s = now_sec() - t0;
  if (!logits) { fprintf(stderr, "error: forward failed\n"); return -1; }

  char buf[512];
  int produced = 0;
  double t1 = now_sec();
  if (gs->qm && gs->spec) {
    produced = decode_spec(gs, ids[n_ids - 1], max_tokens, buf, sizeof(buf));
    *decode_s = now_sec() - t1;
    return produced;
  }
  for (; produced < max_tokens && gs->pos < gs->ctx; ++produced) {
    sampler_penalize(gs->sampler, logits, gs->vocab, gs->recent, gs->n_recent);
    int32_t next = sample_token(gs->sampler, logits, gs->vocab);
    if (is_stop(gs, next)) break;
    size_t w = tokenizer_decode_token(gs->tok, next, buf, sizeof(buf));
    fwrite(buf, 1, w, stdout);
    fflush(stdout);
    push_recent(gs, next);
    logits = gs->forward(gs->model, next, gs->pos++, true);
    if (!logits) break;
  }
  *decode_s = now_sec() - t1;
  return produced;
}

int main(int argc, char** argv) {
  const char* model_path = NULL;
  const char* mmproj_path = NULL; /* --mmproj: CLIP/SigLIP vision GGUF */
  const char* image_path = NULL;  /* --image: raw f32 CHW image for the tower */
  const char* prompt = NULL;
  const char* system_prompt = NULL;
  int max_tokens = 128;
  size_t ctx = 4096;
  int threads = 0;
  int bench = 0;
  int raw = 0;  /* --raw: skip chat template (base-LM diagnostic) */
  int chat = 0; /* --chat: interactive multi-turn REPL on stdin */
  int do_inspect = 0;
  int spec = 0;          /* --spec: speculative decode with the MTP draft head */
  int draft_tokens = 4;  /* --draft-tokens N: candidates proposed per spec step */
  /* KV cache precision. f32 default; f16 ~lossless half-size; q8 int8/head;
   * q4 the gemma4 rotoquant. --kv-quant/--no-kv-quant kept as q4/f32 aliases. */
  OcKvType kv_type = OC_KV_F32;
  SamplerConfig sampler = {.temperature = 0.0f, .top_k = 0, .top_p = 1.0f,
                           .rng = 0x9E3779B97F4A7C15ull, .min_p = 0.0f,
                           .repeat_penalty = 0.0f};

  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    const char* v = i + 1 < argc ? argv[i + 1] : NULL;
    if (strcmp(a, "--model") == 0 && v) model_path = argv[++i];
    else if (strcmp(a, "--mmproj") == 0 && v) mmproj_path = argv[++i];
    else if (strcmp(a, "--image") == 0 && v) image_path = argv[++i];
    else if (strcmp(a, "--prompt") == 0 && v) prompt = argv[++i];
    else if (strcmp(a, "--system") == 0 && v) system_prompt = argv[++i];
    else if (strcmp(a, "--max-tokens") == 0 && v) max_tokens = atoi(argv[++i]);
    else if (strcmp(a, "--temp") == 0 && v) sampler.temperature = (float)atof(argv[++i]);
    else if (strcmp(a, "--top-k") == 0 && v) sampler.top_k = atoi(argv[++i]);
    else if (strcmp(a, "--top-p") == 0 && v) sampler.top_p = (float)atof(argv[++i]);
    else if (strcmp(a, "--min-p") == 0 && v) sampler.min_p = (float)atof(argv[++i]);
    else if (strcmp(a, "--repeat-penalty") == 0 && v) sampler.repeat_penalty = (float)atof(argv[++i]);
    else if (strcmp(a, "--ctx") == 0 && v) ctx = (size_t)atoll(argv[++i]);
    else if (strcmp(a, "--threads") == 0 && v) threads = atoi(argv[++i]);
    else if (strcmp(a, "--seed") == 0 && v) sampler.rng = (uint64_t)atoll(argv[++i]) | 1;
    else if (strcmp(a, "--bench") == 0) bench = 1;
    else if (strcmp(a, "--raw") == 0) raw = 1;
    else if (strcmp(a, "--chat") == 0) chat = 1;
    else if (strcmp(a, "--inspect") == 0) do_inspect = 1;
    else if (strcmp(a, "--kv-type") == 0 && v) {
      const char* t = argv[++i];
      if (strcmp(t, "f32") == 0) kv_type = OC_KV_F32;
      else if (strcmp(t, "f16") == 0) kv_type = OC_KV_F16;
      else if (strcmp(t, "q8") == 0) kv_type = OC_KV_Q8;
      else if (strcmp(t, "q4") == 0) kv_type = OC_KV_Q4;
      else { fprintf(stderr, "unknown --kv-type: %s\n", t); usage(argv[0]); return 1; }
    }
    else if (strcmp(a, "--kv-quant") == 0) kv_type = OC_KV_Q4;
    else if (strcmp(a, "--no-kv-quant") == 0) kv_type = OC_KV_F32;
    else if (strcmp(a, "--spec") == 0) spec = 1;
    else if (strcmp(a, "--draft-tokens") == 0 && v) draft_tokens = atoi(argv[++i]);
    else if (strcmp(a, "--greedy") == 0) sampler.temperature = 0.0f;
    else if (strcmp(a, "--version") == 0) { print_isa(); return 0; }
    else if (strcmp(a, "--help") == 0) { usage(argv[0]); return 0; }
    else { fprintf(stderr, "unknown argument: %s\n", a); usage(argv[0]); return 1; }
  }
  /* Vision path: --image needs --mmproj; runs the tower and exits (the text LM
   * is not involved — see run_vision). */
  if (image_path || mmproj_path) {
    if (!image_path || !mmproj_path) {
      fprintf(stderr, "error: --image and --mmproj must be given together\n");
      usage(argv[0]);
      return 1;
    }
    return run_vision(mmproj_path, image_path);
  }
  if (!model_path) { usage(argv[0]); return 1; }
  if (do_inspect) return inspect(model_path);
  if (!prompt && !chat) prompt = "Hello";

  char err[256] = {0};
  GgufFile g;
  if (gguf_open(&g, model_path, err, sizeof(err)) != 0) {
    fprintf(stderr, "error: %s\n", err);
    return 1;
  }

  oc_pool_init(threads);

  /* KV cache precision is process-wide (the loaders read it); the bool arg still
   * forces q4 so gemma4's rotoquant path fires exactly as before. */
  oc_kv_set_type(kv_type);

  /* Arch dispatch: gemma4 / qwen36 / generic dense llama (model.c). */
  Model model;
  if (model_load(&model, &g, ctx, kv_type == OC_KV_Q4, err, sizeof(err)) != 0) {
    fprintf(stderr, "error: %s\n", err);
    gguf_close(&g);
    oc_pool_free();
    return 1;
  }
  GgufFile* owned_g = model.g;

  /* Dual-socket runs: node-local weight replicas + pinned workers. */
  oc_numa_replicate(owned_g->map, owned_g->size);

  Tokenizer tok;
  if (tokenizer_init(&tok, owned_g) != 0) {
    model_free(&model);
    oc_pool_free();
    return 1;
  }

  /* Chat template: detect the family from tokenizer.ggml.chat_template (and, if
   * absent, from the vocab's special tokens), then format turns from a fixed
   * per-family control-token table (the llama.cpp approach). --raw bypasses it. */
  char* tmpl = gguf_get_str(owned_g, "tokenizer.chat_template");
  ChatFamily fam = chat_detect(&tok, tmpl);
  free(tmpl);
  if (!raw) fprintf(stderr, "chat template: %s\n", chat_family_name(fam));
  const char* stop_tok = chat_stop_token(fam);

  GenState gs = {model.handle,
                 model.forward,
                 model.forward_batch,
                 model.vocab,
                 model.ctx,
                 model.gemma_stops,
                 &tok, &sampler, 0,
                 tokenizer_piece_id(&tok, stop_tok, strlen(stop_tok)),
                 {0}, 0, NULL, 0, 0, 0, 0};

  /* Speculative decode is enabled only where it is provably bit-exact to plain
   * decode: qwen36 with a usable MTP draft head, pure greedy, no penalties.
   * Anywhere else --spec is a no-op (with a one-line reason). */
  if (spec) {
    Qwen36Model* qm = model.family == MODEL_QWEN36 ? (Qwen36Model*)model.handle : NULL;
    int greedy_no_pen = sampler.temperature <= 0.0f && sampler.repeat_penalty <= 1.0f &&
                        sampler.frequency_penalty == 0.0f &&
                        sampler.presence_penalty == 0.0f;
    if (qm && qm->has_mtp && greedy_no_pen && qm->batch_cap >= 2) {
      gs.qm = qm;
      gs.spec = 1;
      gs.draft_tokens = draft_tokens > 0 ? (size_t)draft_tokens : 4;
    } else if (!qm || !qm->has_mtp) {
      fprintf(stderr, "note: --spec ignored (model has no qwen36 MTP draft head)\n");
    } else {
      fprintf(stderr, "note: --spec ignored (needs pure greedy: temp 0, no penalties)\n");
    }
  }

  int rc = 0;
  char* line = NULL;
  size_t line_cap = 0;
  for (;;) {
    const char* user_msg = prompt;
    if (chat) {
      ssize_t r = getline(&line, &line_cap, stdin);
      if (r < 0) break; /* EOF */
      while (r > 0 && (line[r - 1] == '\n' || line[r - 1] == '\r')) line[--r] = 0;
      if (r == 0) continue;
      if (strcmp(line, "/bye") == 0 || strcmp(line, "/exit") == 0) break;
      user_msg = line;
    }

    /* Build turn text. First turn optionally carries a system preamble;
     * follow-up turns start by closing the model's previous turn. */
    size_t flen = strlen(user_msg) + (system_prompt ? strlen(system_prompt) : 0) + 320;
    char* full = malloc(flen);
    if (raw) {
      snprintf(full, flen, "%s", user_msg);
    } else if (chat_format_turn(fam, gs.pos == 0 ? system_prompt : NULL,
                                user_msg, gs.pos == 0, full, flen) == 0) {
      fprintf(stderr, "error: chat template overflow (message too long)\n");
      free(full);
      rc = 1;
      break;
    }

    size_t n_prompt = 0;
    int32_t* ids = tokenizer_encode(&tok, full, gs.pos == 0, &n_prompt);
    free(full);
    if (!ids || n_prompt == 0) {
      fprintf(stderr, "error: prompt tokenization failed\n");
      rc = 1;
      break;
    }

    double t_prefill = 0, t_decode = 0;
    int produced = generate(&gs, ids, n_prompt, max_tokens, &t_prefill, &t_decode);
    free(ids);
    if (produced < 0) { rc = 1; break; }
    printf("\n");
    if (bench || chat)
      fprintf(stderr, "[prefill %zu tok %.2f tok/s | decode %d tok %.2f tok/s | ctx %zu/%zu]\n",
              n_prompt, t_prefill > 0 ? (double)n_prompt / t_prefill : 0.0,
              produced, t_decode > 0 ? (double)produced / t_decode : 0.0,
              gs.pos, gs.ctx);
    /* Speculative stats: mean draft tokens accepted per step and the resulting
     * tokens-committed-per-target-batch (≈ the decode speedup vs 1.0 for plain
     * autoregressive decode, which commits one token per target forward). */
    if (gs.spec && gs.spec_steps)
      fprintf(stderr, "[spec: %zu steps | %.2f/%zu draft accepted/step | "
              "%.2f tok/target-batch (plain = 1.00)]\n",
              gs.spec_steps, (double)gs.spec_accept / (double)gs.spec_steps,
              gs.draft_tokens, (double)produced / (double)gs.spec_steps);
    if (chat) {
      /* record separator: end-of-turn marker for wrapping UIs */
      fputs("\x1e\n", stdout);
      fflush(stdout);
    } else {
      break;
    }
  }
  free(line);

  tokenizer_free(&tok);
  model_free(&model);
  oc_pool_free();
  return rc;
}
