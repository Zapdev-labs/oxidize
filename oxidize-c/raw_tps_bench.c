#include "oc.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  double prefill_seconds;
  double decode_seconds;
  size_t committed_target_tokens;
  uint64_t committed_hash;
} raw_result;

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint32_t argmax_token(const float *logits, size_t vocab) {
  size_t best = 0;
  for (size_t i = 1; i < vocab; ++i)
    if (logits[i] > logits[best]) best = i;
  return (uint32_t)best;
}

static uint64_t hash_token(uint64_t hash, uint32_t token) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    hash ^= (uint8_t)(token >> shift);
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static raw_result run_target_sequence(oc_model *model, const oc_tokenizer *tok,
                                      const uint32_t *prompt, size_t n_prompt,
                                      size_t max_tokens) {
  raw_result result = {.committed_hash = UINT64_C(1469598103934665603)};
  const size_t vocab = model->cfg.vocab_size;
  float *logits = malloc(vocab * sizeof(*logits));
  if (!logits) oc_die("raw-tps-bench: logits allocation failed");

  oc_reset_state(model);
  double prefill_start = now_s();
  size_t pos = 0;
  for (size_t off = 0; off < n_prompt;) {
    size_t count = n_prompt - off;
    if (count > 64) count = 64;
    oc_forward(model, prompt + off, count, pos, logits);
    off += count;
    pos += count;
  }
  result.prefill_seconds = now_s() - prefill_start;

  double decode_start = now_s();
  while (result.committed_target_tokens < max_tokens && pos < model->kv_ctx) {
    uint32_t token = argmax_token(logits, vocab);
    if (oc_is_eog(tok, token)) break;
    oc_forward(model, &token, 1, pos, logits);
    ++pos;
    ++result.committed_target_tokens;
    result.committed_hash = hash_token(result.committed_hash, token);
  }
  result.decode_seconds = now_s() - decode_start;
  free(logits);
  return result;
}

static void print_result(const raw_result *result) {
  double tps = result->decode_seconds > 0.0
                   ? (double)result->committed_target_tokens / result->decode_seconds
                   : 0.0;
  printf("{\"sequences\":1,\"counted_tokens\":\"target_verified_committed\","
         "\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f,"
         "\"committed_target_tokens\":%zu,\"raw_committed_tps\":%.9f,"
         "\"committed_token_hash\":\"%016" PRIx64 "\","
         "\"draft_proposed_tokens\":0,\"draft_rejected_tokens\":0}\n",
         result->prefill_seconds, result->decode_seconds,
         result->committed_target_tokens, tps, result->committed_hash);
}

static void usage(const char *program) {
  fprintf(stderr,
          "Usage: %s --model MODEL.gguf [--prompt TEXT] [--max-tokens N] "
          "[--ctx N] [--warmup-tokens N] [--no-bos]\n"
          "       %s --dry-run\n"
          "Measures exactly one causal sequence.  The JSON metric excludes "
          "model load, warmup, and prompt prefill.\n",
          program, program);
}

static size_t parse_size(const char *flag, const char *value) {
  char *end = NULL;
  errno = 0;
  unsigned long long number = strtoull(value, &end, 10);
  if (!value[0] || errno || !end || *end || number > SIZE_MAX)
    oc_die("raw-tps-bench: bad %s %s", flag, value);
  return (size_t)number;
}

int main(int argc, char **argv) {
  const char *model_path = NULL;
  const char *prompt_text = "Hello";
  size_t max_tokens = 64, context = 8192, warmup_tokens = 0;
  int add_bos = 1, dry_run = 0;

  for (int i = 1; i < argc; ++i) {
    const char *argument = argv[i];
    if (!strcmp(argument, "--help")) {
      usage(argv[0]);
      return 0;
    }
    if (!strcmp(argument, "--dry-run")) {
      dry_run = 1;
      continue;
    }
    if (!strcmp(argument, "--no-bos")) {
      add_bos = 0;
      continue;
    }
    if (!strcmp(argument, "--model") || !strcmp(argument, "--prompt") ||
        !strcmp(argument, "--max-tokens") || !strcmp(argument, "--ctx") ||
        !strcmp(argument, "--warmup-tokens")) {
      if (++i == argc) {
        usage(argv[0]);
        return 2;
      }
      if (!strcmp(argument, "--model")) model_path = argv[i];
      else if (!strcmp(argument, "--prompt")) prompt_text = argv[i];
      else if (!strcmp(argument, "--max-tokens")) max_tokens = parse_size(argument, argv[i]);
      else if (!strcmp(argument, "--ctx")) context = parse_size(argument, argv[i]);
      else warmup_tokens = parse_size(argument, argv[i]);
      continue;
    }
    if (!strcmp(argument, "--concurrency") || !strcmp(argument, "--batch") ||
        !strcmp(argument, "--max-seqs") || !strcmp(argument, "--gpu-list"))
      oc_die("raw-tps-bench: %s is forbidden; this benchmark has one sequence", argument);
    fprintf(stderr, "raw-tps-bench: unknown option %s\n", argument);
    usage(argv[0]);
    return 2;
  }

  if (dry_run) {
    raw_result empty = {.committed_hash = UINT64_C(1469598103934665603)};
    print_result(&empty);
    return 0;
  }
  if (!model_path) {
    usage(argv[0]);
    return 2;
  }
  if (context == 0) oc_die("raw-tps-bench: --ctx must be positive");

  oc_model *model = oc_model_load(model_path, context, 0);
  oc_tokenizer *tok = oc_tokenizer_load(model->g);
  if (!tok) oc_die("raw-tps-bench: no usable tokenizer in GGUF");
  size_t n_prompt = 0;
  uint32_t *prompt = oc_tokenize(tok, prompt_text, add_bos != 0, &n_prompt);
  if (!prompt || n_prompt == 0) oc_die("raw-tps-bench: empty prompt after tokenize");
  if (n_prompt >= model->kv_ctx) oc_die("raw-tps-bench: prompt exhausts context");
  if (max_tokens > model->kv_ctx - n_prompt) max_tokens = model->kv_ctx - n_prompt;

  if (warmup_tokens) (void)run_target_sequence(model, tok, prompt, n_prompt, warmup_tokens);
  raw_result result = run_target_sequence(model, tok, prompt, n_prompt, max_tokens);
  print_result(&result);

  free(prompt);
  oc_tokenizer_free(tok);
  oc_model_free(model);
  return 0;
}
