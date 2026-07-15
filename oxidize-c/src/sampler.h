/* Greedy + temperature/top-k/top-p/min-p sampling with repetition penalties
 * (ported from oxidize-core/src/model/sampling.rs).
 *
 * sample_token holds a scratch candidate buffer in SamplerConfig and grows it
 * on demand, so the decode hot loop never mallocs per token. Greedy
 * (temperature <= 0) is a plain argmax and allocates nothing. Call
 * sampler_free() once when done to release the scratch buffer. */
#ifndef OC_SAMPLER_H
#define OC_SAMPLER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
  float temperature;    /* <= 0 => greedy */
  int top_k;            /* <= 0 => disabled */
  float top_p;          /* >= 1 => disabled */
  uint64_t rng;         /* xorshift64* state */
  float min_p;          /* <= 0 => disabled; keep tokens with p >= min_p*max_p */
  float repeat_penalty; /* <= 1 => disabled (llama.cpp style divide/multiply) */
  /* OpenAI-style penalties over recent tokens, applied in sampler_penalize.
   * Default 0 => disabled. frequency scales with occurrence count; presence is
   * a flat penalty applied once per distinct token. */
  float frequency_penalty;
  float presence_penalty;
  /* Reused candidate scratch (opaque). Do not touch; sampler_free() releases. */
  void* scratch;
  size_t scratch_cap; /* capacity in candidate entries */
} SamplerConfig;

int32_t sample_token(SamplerConfig* s, const float* logits, size_t n);

/* Applies repeat_penalty then frequency/presence penalties in-place to the
 * logits of the `n_recent` token ids. Any of the three off (repeat<=1,
 * penalties==0) is skipped. */
void sampler_penalize(const SamplerConfig* s, float* logits, size_t n,
                      const int32_t* recent, size_t n_recent);

/* Releases the internal scratch buffer. Safe to call multiple times / on a
 * zero-initialised config. */
void sampler_free(SamplerConfig* s);

#endif
