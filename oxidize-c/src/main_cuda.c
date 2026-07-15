/* oxidize-c CLI, CUDA backend: GPU-resident text generation.
 * Dispatches on general.architecture: "gemma*" -> the gemma4 resident forward
 * (src/cuda/gemma4_cuda.cu), everything else -> the generic llama-family dense
 * forward (src/cuda/llama_cuda.cu). Deliberate near-copy of main.c structure;
 * the deltas are --gpus N / --ngl N and the *_cuda_step() decode loop. Both
 * backends drive one family-agnostic generation loop through a step fn pointer. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cuda/gemma4_cuda.h"
#include "cuda/llama_cuda.h"
#include "gguf.h"
#include "model_gemma4.h"
#include "model_llama.h"
#include "sampler.h"
#include "tensor.h"
#include "tokenizer.h"

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(const char* argv0) {
  fprintf(stderr,
          "usage: %s --model path.gguf [--prompt \"...\"] [--max-tokens N]\n"
          "          [--temp T] [--top-k K] [--top-p P] [--ctx N] [--gpus N]\n"
          "          [--ngl N]   layers to offload (default: all)\n"
          "          [--bench] [--greedy] [--seed N]\n"
          "          [--kv-quant | --no-kv-quant]  rotated int4 KV cache (gemma4)\n",
          argv0);
}

/* Family-agnostic decode step: run one token through whichever backend is live
 * and return the (need_logits) logits or NULL; *failed on error. */
typedef float* (*cuda_step_fn)(void* backend, void* model, int32_t tok,
                               size_t pos, bool need_logits, int* failed);

static float* step_gemma(void* be, void* mdl, int32_t t, size_t p, bool nl, int* f) {
  return gemma4_cuda_step((Gemma4Cuda*)be, (Gemma4Model*)mdl, t, p, nl, f);
}
static float* step_llama(void* be, void* mdl, int32_t t, size_t p, bool nl, int* f) {
  return llama_cuda_step((LlamaCuda*)be, (LlamaModel*)mdl, t, p, nl, f);
}

int main(int argc, char** argv) {
  const char* model_path = NULL;
  const char* prompt = "Hello";
  int max_tokens = 128;
  size_t ctx = 4096;
  int n_gpus = 1;
  int n_gpu_layers = -1; /* -1 = every layer */
  int bench = 0;
  int kv_quant = 0; /* gemma4 rotoquant; ignored by llama (f16 KV) */
  SamplerConfig sampler = {.temperature = 0.0f, .top_k = 0, .top_p = 1.0f,
                           .rng = 0x9E3779B97F4A7C15ull, .min_p = 0.0f,
                           .repeat_penalty = 0.0f}; /* greedy default */

  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    const char* v = i + 1 < argc ? argv[i + 1] : NULL;
    if (strcmp(a, "--model") == 0 && v) model_path = argv[++i];
    else if (strcmp(a, "--prompt") == 0 && v) prompt = argv[++i];
    else if (strcmp(a, "--max-tokens") == 0 && v) max_tokens = atoi(argv[++i]);
    else if (strcmp(a, "--temp") == 0 && v) sampler.temperature = (float)atof(argv[++i]);
    else if (strcmp(a, "--top-k") == 0 && v) sampler.top_k = atoi(argv[++i]);
    else if (strcmp(a, "--top-p") == 0 && v) sampler.top_p = (float)atof(argv[++i]);
    else if (strcmp(a, "--ctx") == 0 && v) ctx = (size_t)atoll(argv[++i]);
    else if (strcmp(a, "--gpus") == 0 && v) n_gpus = atoi(argv[++i]);
    else if (strcmp(a, "--ngl") == 0 && v) n_gpu_layers = atoi(argv[++i]);
    else if (strcmp(a, "--seed") == 0 && v) sampler.rng = (uint64_t)atoll(argv[++i]) | 1;
    else if (strcmp(a, "--bench") == 0) bench = 1;
    else if (strcmp(a, "--kv-quant") == 0) kv_quant = 1;
    else if (strcmp(a, "--no-kv-quant") == 0) kv_quant = 0;
    else if (strcmp(a, "--greedy") == 0) sampler.temperature = 0.0f;
    else if (strcmp(a, "--help") == 0) { usage(argv[0]); return 0; }
    else { fprintf(stderr, "unknown argument: %s\n", a); usage(argv[0]); return 1; }
  }
  if (!model_path) { usage(argv[0]); return 1; }

  char err[256] = {0};
  GgufFile g;
  if (gguf_open(&g, model_path, err, sizeof(err)) != 0) {
    fprintf(stderr, "error: %s\n", err);
    return 1;
  }

  /* The CPU thread pool serves the partial-offload (-ngl) tail's *_forward_from;
   * harmless when the GPU owns the whole stack. */
  oc_pool_init(0);

  char* arch = gguf_architecture(&g);
  int is_gemma = arch && strncmp(arch, "gemma", 5) == 0;
  free(arch);

  /* One of these is used; the other stays zeroed. The GgufFile ownership moves
   * into whichever loader succeeds (its .g), so cleanup goes through that. */
  Gemma4Model gmodel;
  LlamaModel lmodel;
  Gemma4Cuda* gcuda = NULL;
  LlamaCuda* lcuda = NULL;
  void* backend = NULL;
  void* model_handle = NULL;
  cuda_step_fn step = NULL;
  size_t vocab = 0;
  int gemma_stops = 0;
  GgufFile* owned_g = NULL;

  if (is_gemma) {
    if (gemma4_load(&gmodel, &g, ctx, kv_quant != 0, err, sizeof(err)) != 0) {
      fprintf(stderr, "error: %s\n", err);
      gguf_close(&g);
      oc_pool_free();
      return 1;
    }
    if (gemma4_cuda_init(&gcuda, &gmodel, n_gpus, n_gpu_layers, err, sizeof(err)) != 0) {
      fprintf(stderr, "error: %s\n", err);
      gemma4_free(&gmodel);
      oc_pool_free();
      return 1;
    }
    backend = gcuda; model_handle = &gmodel; step = step_gemma;
    vocab = gmodel.vocab; owned_g = &gmodel.g; gemma_stops = 1;
  } else {
    oc_kv_set_type(OC_KV_F16); /* llama GPU cache is f16; keep the CPU tail matched */
    if (llama_load(&lmodel, &g, ctx, err, sizeof(err)) != 0) {
      fprintf(stderr, "error: %s\n", err);
      gguf_close(&g);
      oc_pool_free();
      return 1;
    }
    if (llama_cuda_init(&lcuda, &lmodel, n_gpus, n_gpu_layers, err, sizeof(err)) != 0) {
      fprintf(stderr, "error: %s\n", err);
      llama_free(&lmodel);
      oc_pool_free();
      return 1;
    }
    backend = lcuda; model_handle = &lmodel; step = step_llama;
    vocab = lmodel.vocab; owned_g = &lmodel.g; gemma_stops = 0;
  }

  Tokenizer tok;
  if (tokenizer_init(&tok, owned_g) != 0) {
    if (is_gemma) { gemma4_cuda_free(gcuda); gemma4_free(&gmodel); }
    else { llama_cuda_free(lcuda); llama_free(&lmodel); }
    oc_pool_free();
    return 1;
  }

  /* Proper per-family chat template (same path main.c uses), so llama and gemma
   * both get the right control tokens rather than a hardcoded gemma turn. */
  char* tmpl = gguf_get_str(owned_g, "tokenizer.chat_template");
  ChatFamily fam = chat_detect(&tok, tmpl);
  free(tmpl);
  const char* stop_tok_str = chat_stop_token(fam);
  int32_t stop_tok = tokenizer_piece_id(&tok, stop_tok_str, strlen(stop_tok_str));

  size_t flen = strlen(prompt) + 320;
  char* full = malloc(flen);
  if (chat_format_turn(fam, NULL, prompt, true, full, flen) == 0) {
    fprintf(stderr, "error: chat template overflow (prompt too long)\n");
    free(full);
    if (is_gemma) { gemma4_cuda_free(gcuda); gemma4_free(&gmodel); }
    else { llama_cuda_free(lcuda); llama_free(&lmodel); }
    tokenizer_free(&tok);
    oc_pool_free();
    return 1;
  }

  size_t n_prompt = 0;
  int32_t* ids = tokenizer_encode(&tok, full, true, &n_prompt);
  free(full);
  if (!ids || n_prompt == 0) {
    fprintf(stderr, "error: prompt tokenization failed\n");
    return 1;
  }
  fprintf(stderr, "prompt: %zu tokens (chat template: %s)\n", n_prompt,
          chat_family_name(fam));

  int failed = 0;

  /* Prefill: fully async (no per-token sync); last token produces output. */
  double t0 = now_sec();
  for (size_t i = 0; i + 1 < n_prompt; ++i) {
    step(backend, model_handle, ids[i], i, false, &failed);
    if (failed) return 1;
  }
  float* logits = step(backend, model_handle, ids[n_prompt - 1], n_prompt - 1, true, &failed);
  if (failed || !logits) return 1;
  double t_prefill = now_sec() - t0;

  /* Decode: one forward + one vocab-float D2H copy per token. */
  char buf[512];
  size_t pos = n_prompt;
  int produced = 0;
  double t1 = now_sec();
  for (; produced < max_tokens; ++produced) {
    int32_t next = sample_token(&sampler, logits, vocab); /* temp 0 == argmax */
    if (next < 0 || (gemma_stops && (next == 1 || next == 106)) ||
        next == (int32_t)tok.eos_id ||
        (tok.eot_id >= 0 && next == (int32_t)tok.eot_id) ||
        (stop_tok >= 0 && next == stop_tok))
      break;
    size_t w = tokenizer_decode_token(&tok, next, buf, sizeof(buf));
    fwrite(buf, 1, w, stdout);
    fflush(stdout);
    logits = step(backend, model_handle, next, pos++, true, &failed);
    if (failed || !logits) break;
  }
  double t_decode = now_sec() - t1;
  printf("\n");

  if (bench) {
    fprintf(stderr, "prefill: %zu tokens in %.2fs (%.2f tok/s)\n", n_prompt,
            t_prefill, t_prefill > 0 ? (double)n_prompt / t_prefill : 0.0);
    fprintf(stderr, "decode:  %d tokens in %.2fs (%.2f tok/s)\n", produced,
            t_decode, t_decode > 0 ? (double)produced / t_decode : 0.0);
  }

  free(ids);
  tokenizer_free(&tok);
  if (is_gemma) { gemma4_cuda_free(gcuda); gemma4_free(&gmodel); }
  else { llama_cuda_free(lcuda); llama_free(&lmodel); }
  oc_pool_free();
  return 0;
}
