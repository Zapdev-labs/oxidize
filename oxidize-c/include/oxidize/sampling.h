/* sampling.h — token sampling from logits. */
#ifndef OXIDIZE_SAMPLING_H
#define OXIDIZE_SAMPLING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_SAMPLER_GREEDY        = 0,   /* argmax                              */
    OC_SAMPLER_TEMPERATURE   = 1,   /* softmax(logits/T) then sample       */
    OC_SAMPLER_TOP_K         = 2,   /* keep top-K logits, then sample      */
    OC_SAMPLER_TOP_P         = 3,   /* nucleus: smallest set with cum >= p */
    OC_SAMPLER_MIROSTAT_V2   = 4,   /* Mirostat v2 (surprise-based)        */
    OC_SAMPLER_MIN_P         = 5,   /* min-p: filter by min probability ratio */
    OC_SAMPLER_TYPICAL_P     = 6,   /* locally typical sampling             */
    OC_SAMPLER_TAIL_FREE     = 7,   /* tail-free sampling (TFS)             */
} OcSamplerType;

typedef struct OcSamplerConfig {
    OcSamplerType type;
    float  temperature;      /* >0; 1.0 = identity. T=0 falls back to greedy. */
    uint32_t top_k;          /* 0 = disabled (all vocab).                     */
    float  top_p;            /* (0,1]; 1.0 = disabled.                       */
    float  repeat_penalty;   /* 1.0 = no penalty; >1 penalizes recent tokens. */
    uint64_t seed;           /* RNG seed; 0 selects a fixed nonzero seed.      */
    float  mu;               /* initial running surprise estimate (default 2*tau). */
    float  tau;              /* target surprise (default 5.0).                */
    float  eta;              /* learning rate for mu updates (default 0.1).   */
    float  learning_rate;    /* optional override; 0 = use `eta`.             */
    float  min_p;            /* min-p ratio (0=disabled, 0.05=keep tokens with p >= 0.05*max_p) */
    float  typical_p;         /* typical-p threshold (0=disabled, 0.95=keep top 95% of typical mass) */
    float  tail_free_z;       /* TFS z-threshold (0=disabled, 0.95=keep tokens above z cutoff) */
} OcSamplerConfig;

/* Default config: greedy, no penalty. Mirostat fields default to the
 * canonical v2 values (mu=10.0, tau=5.0, eta=0.1). With Mirostat, oc_sample
 * updates cfg->mu; callers must also advance cfg->seed between calls. */
#define OC_SAMPLER_DEFAULT ((OcSamplerConfig){ \
    OC_SAMPLER_GREEDY, 1.0f, 0u, 1.0f, 1.0f, 0ull, \
    10.0f, 5.0f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f })

/* Sample one token from `logits` (length vocab_size). */
uint32_t oc_sample(const float *logits, size_t vocab_size,
                   OcSamplerConfig *cfg,
                   const uint32_t *recent_tokens, size_t n_recent);

/* Convenience: argmax over logits. */
uint32_t oc_argmax(const float *logits, size_t vocab_size);

/* Apply repeat-penalty in place: positive logits are divided by `penalty` and
 * non-positive logits are multiplied by it. */
void oc_apply_repeat_penalty(float *logits, size_t vocab_size,
                             const uint32_t *recent, size_t n_recent,
                             float penalty);

/* Mirostat v2 sampler (surprise-balanced truncation). */
uint32_t oc_mirostat_v2_sample(const float *logits, size_t vocab_size,
                               const OcSamplerConfig *cfg,
                               float *mu, uint64_t *rng_state);


/* ─── Repetition penalties ────────────────────────────────────────────── */
typedef struct {
    uint32_t *tokens;  /* owned, caller must free */
    size_t    n_tokens;
    float     score;
} OcBeamSearchResult;

/* Run beam search over per-step logits. */
OcError oc_beam_search(float * const *logits_per_step, size_t n_steps,
                        size_t vocab_size, size_t beam_width,
                        uint32_t eos_token, OcBeamSearchResult *out);

void oc_beam_search_result_free(OcBeamSearchResult *result);


/* Compute softmax(logits/T) into out. Returns OC_ERR_INVALID_ARG for bad args
 * or empty logits. out must have vocab_size floats. */
OcError oc_softmax_probs(const float *logits, size_t vocab_size,
                          float temperature, float *out);


/* Compute residual probs: max(0, target - draft), normalized.
 * target_probs and draft_probs must be vocab_size floats.
 * out must be vocab_size floats. */
void oc_residual_probs(const float *target_probs, const float *draft_probs,
                        float *out, size_t vocab_size);

/* Sample from a probability distribution via CDF.
 * Returns the sampled index, or vocab_size on error.
 * If sum <= 0, returns argmax of probs. */
size_t oc_sample_probabilities(const float *probs, size_t vocab_size, float random);


typedef struct {
    float frequency_penalty;   /* subtracted per occurrence (0=no-op) */
    float presence_penalty;     /* subtracted once if token appears (0=no-op) */
    uint32_t newline_token_id; /* 0xFFFFFFFF = no newline penalty */
    float newline_penalty;     /* subtracted from newline_token_id */
} OcRepetitionPenaltyConfig;

/* Apply repetition penalties in place: subtracts frequency*freq + presence
 * for each recent token, and newline penalty if configured. */
void oc_apply_repetition_penalties(float *logits, size_t vocab_size,
                                    const uint32_t *recent, size_t n_recent,
                                    const OcRepetitionPenaltyConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SAMPLING_H */
