/*
 * sampling.h — token sampling from logits.
 *
 * Port of oxidize-core/src/model/sampling.rs (greedy / temperature / top-k /
 * top-p / repeat-penalty) to C11. Operates on the logits produced by
 * oc_llama_forward. Deterministic given the same (logits, config, seed,
 * recent_tokens).
 *
 * Scope of the `cpu-llama-sampling` feature: the standard samplers plus
 * Mirostat v2. min-p is deferred (sampler framework is extensible via
 * OcSamplerConfig).
 */
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
    /* ── Mirostat v2 parameters ────────────────────────────────────────
     * `mu` is the running target surprise estimate; it is mutated between
     * calls by oc_mirostat_v2_sample. `tau` is the target surprise (entropy
     * ceiling). `eta` is the per-step learning rate applied to mu updates.
     * `learning_rate` is reserved as an explicit override hook (when >0 it
     * takes precedence over `eta`); 0.0 means use `eta`.              */
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

/* Sample one token from `logits` (length vocab_size).
 *
 *   - `recent_tokens` (length `n_recent`, may be NULL/0): tokens to apply
 *     repeat-penalty to. Penalties divide the logit (penalty>1 lowers prob)
 *     of each recent token before sampling.
 *   - For GREEDY or temperature==0: returns argmax deterministically.
 *   - For stochastic samplers: uses xorshift64 seeded by cfg->seed (caller
 *     should advance the seed between calls for non-repeated sequences).
 *
 * Returns the sampled token id (< vocab_size), or UINT32_MAX when vocab_size
 * is zero. */
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

/* Mirostat v2 sampler (surprise-balanced truncation).
 *
 * Implements the algorithm from Basu et al. (2020), "Mirostat: a Neural
 * Text Decoding Algorithm that Directly Controls Perplexity":
 *
 *   1. Compute softmax probabilities from `logits` (length vocab_size).
 *      No temperature scaling is applied here; the caller should pre-scale
 *      logits if desired.
 *   2. Keep tokens whose surprise is no greater than `*mu`, then sample from
 *      that renormalized distribution using `*rng_state`.
 *   3. Compute the observed surprise s = -log2(p_sampled).
 *   4. Update the running estimate: *mu = *mu - eta * (s - tau), where
 *      `eta` is taken from cfg->learning_rate when it is > 0, else
 *      cfg->eta.
 *   5. Clamp *mu to [0, 2*tau].
 *   6. Return the sampled token id.
 *
 * `mu` is read AND written (state is carried across calls by the caller).
 * `cfg` supplies tau, eta, and learning_rate. Returns the sampled token;
 * returns UINT32_MAX for an empty vocabulary. */
uint32_t oc_mirostat_v2_sample(const float *logits, size_t vocab_size,
                               const OcSamplerConfig *cfg,
                               float *mu, uint64_t *rng_state);

/* ─── Beam search ─────────────────────────────────────────────────────── */

/* Beam search result. tokens is caller-freed. */
typedef struct {
    uint32_t *tokens;  /* owned, caller must free */
    size_t    n_tokens;
    float     score;
} OcBeamSearchResult;

/* Run beam search over per-step logits.
 *
 * logits_per_step: array of pointers to logit arrays (one per step).
 * n_steps: number of steps.
 * vocab_size: vocabulary dimension.
 * beam_width: number of beams to keep (must be > 0).
 * eos_token: 0xFFFFFFFF = no EOS.
 * out: result (tokens are malloc'd, caller frees).
 */
OcError oc_beam_search(float * const *logits_per_step, size_t n_steps,
                        size_t vocab_size, size_t beam_width,
                        uint32_t eos_token, OcBeamSearchResult *out);

void oc_beam_search_result_free(OcBeamSearchResult *result);

/* ─── Softmax probabilities ────────────────────────────────────────────── */

/* Compute softmax(logits/T) into out. Returns OC_ERR_INVALID_ARG for bad args
 * or empty logits. out must have vocab_size floats. */
OcError oc_softmax_probs(const float *logits, size_t vocab_size,
                          float temperature, float *out);

/* ─── Speculative decode probability helpers ──────────────────────────── */

/* Compute residual probs: max(0, target - draft), normalized.
 * target_probs and draft_probs must be vocab_size floats.
 * out must be vocab_size floats. */
void oc_residual_probs(const float *target_probs, const float *draft_probs,
                        float *out, size_t vocab_size);

/* Sample from a probability distribution via CDF.
 * Returns the sampled index, or vocab_size on error.
 * If sum <= 0, returns argmax of probs. */
size_t oc_sample_probabilities(const float *probs, size_t vocab_size, float random);

/* ─── Repetition penalties ────────────────────────────────────────────── */

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
