/*
 * sampling.h — token sampling from logits.
 *
 * Port of oxidize-core/src/model/sampling.rs (greedy / temperature / top-k /
 * top-p / repeat-penalty) to C11. Operates on the logits produced by
 * oc_llama_forward. Deterministic given the same (logits, config, seed,
 * recent_tokens).
 *
 * Scope of the `cpu-llama-sampling` feature: the standard samplers. Mirostat
 * and min-p are deferred (sampler framework is extensible via OcSamplerConfig).
 */
#ifndef OXIDIZE_SAMPLING_H
#define OXIDIZE_SAMPLING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_SAMPLER_GREEDY       = 0,   /* argmax                              */
    OC_SAMPLER_TEMPERATURE  = 1,   /* softmax(logits/T) then sample       */
    OC_SAMPLER_TOP_K        = 2,   /* keep top-K logits, then sample      */
    OC_SAMPLER_TOP_P        = 3,   /* nucleus: smallest set with cum >= p */
} OcSamplerType;

typedef struct OcSamplerConfig {
    OcSamplerType type;
    float  temperature;      /* >0; 1.0 = identity. T=0 falls back to greedy. */
    uint32_t top_k;          /* 0 = disabled (all vocab).                     */
    float  top_p;            /* (0,1]; 1.0 = disabled.                       */
    float  repeat_penalty;   /* 1.0 = no penalty; >1 penalizes recent tokens. */
    uint64_t seed;           /* RNG seed (xorshift64). 0 = greedy always.     */
} OcSamplerConfig;

/* Default config: greedy, no penalty. */
#define OC_SAMPLER_DEFAULT ((OcSamplerConfig){ \
    OC_SAMPLER_GREEDY, 1.0f, 0u, 1.0f, 1.0f, 0ull })

/* Sample one token from `logits` (length vocab_size).
 *
 *   - `recent_tokens` (length `n_recent`, may be NULL/0): tokens to apply
 *     repeat-penalty to. Penalties divide the logit (penalty>1 lowers prob)
 *     of each recent token before sampling.
 *   - For GREEDY or temperature==0: returns argmax deterministically.
 *   - For stochastic samplers: uses xorshift64 seeded by cfg->seed (caller
 *     should advance the seed between calls for non-repeated sequences).
 *
 * Returns the sampled token id (< vocab_size). Never returns an out-of-range
 * id even on degenerate input (all -inf logits → argmax of finite). */
uint32_t oc_sample(const float *logits, size_t vocab_size,
                   const OcSamplerConfig *cfg,
                   const uint32_t *recent_tokens, size_t n_recent);

/* Convenience: argmax over logits. */
uint32_t oc_argmax(const float *logits, size_t vocab_size);

/* Apply repeat-penalty in place: for each token in `recent`, divide its logit
 * by `penalty` (if >0). Mirrors the Rust reference (penalty is a divisor, not
 * a subtractor). */
void oc_apply_repeat_penalty(float *logits, size_t vocab_size,
                             const uint32_t *recent, size_t n_recent,
                             float penalty);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SAMPLING_H */
