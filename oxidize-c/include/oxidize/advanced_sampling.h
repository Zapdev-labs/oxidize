/*
 * advanced_sampling.h — Advanced sampling strategies for LLM generation.
 *
 * Extends the basic sampling.h with more sophisticated strategies:
 *   - Mirostat (v1 + v2): adaptive entropy control
 *   - Tail-free sampling (TFS)
 *   - Locally typical sampling
 *   - Top-A sampling (adaptive top-k)
 *   - Eta-cutoff sampling
 *   - Penalty sampling (frequency + presence)
 *   - Beam search
 *   - Contrastive search
 *
 * All samplers take a logits array [vocab_size] and return a single token.
 * They can be composed (e.g., temperature → top_k → top_p → mirostat).
 */
#ifndef OXIDIZE_ADVANCED_SAMPLING_H
#define OXIDIZE_ADVANCED_SAMPLING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Mirostat ──────────────────────────────────────────────────────────── */

/* Mirostat v1: maintains target surprise (entropy) via adaptive k.
 * Uses the learned error feedback to adjust k each step.
 *
 * mu: current surprise budget (start at 2*tau)
 * tau: target surprise (typically 5.0 for balanced output)
 * eta: learning rate (typically 0.1)
 * Returns the sampled token and updates *mu. */
uint32_t oc_sample_mirostat_v1(const float *logits, size_t vocab_size,
                                float *mu, float tau, float eta,
                                uint32_t *state);

/* Mirostat v2: directly estimates surprise and adjusts.
 * Simpler than v1, uses top-p instead of top-k. */
uint32_t oc_sample_mirostat_v2(const float *logits, size_t vocab_size,
                                float *mu, float tau, float eta,
                                uint32_t *state);

/* ─── Tail-free sampling (TFS) ───────────────────────────────────────────── */

/* TFS: compute second derivative of sorted probabilities, cut off at
 * the tail where the second derivative falls below a threshold. */
uint32_t oc_sample_tfs(const float *logits, size_t vocab_size,
                       float z, float temperature);

/* ─── Locally typical sampling ──────────────────────────────────────────── */

/* Select tokens whose entropy is close to the conditional expectation. */
uint32_t oc_sample_typical(const float *logits, size_t vocab_size,
                            float p, float temperature);

/* ─── Top-A sampling (adaptive top-k) ────────────────────────────────────── */

/* Top-A: remove all tokens whose probability is less than max_prob * a^2.
 * More aggressive than top-p when one token dominates. */
uint32_t oc_sample_top_a(const float *logits, size_t vocab_size,
                          float a, float temperature);

/* ─── Eta cutoff ─────────────────────────────────────────────────────────── */

/* Eta cutoff: keep tokens whose probability exceeds epsilon.
 * epsilon = min(eta, sqrt(eta/vocab_size)). */
uint32_t oc_sample_eta_cutoff(const float *logits, size_t vocab_size,
                                float epsilon, float temperature);

/* ─── Penalty sampling ──────────────────────────────────────────────────── */

/* Apply frequency and presence penalties to logits.
 * frequency_penalty: penalize tokens proportional to their count
 * presence_penalty: penalize any token that has appeared at all
 * Modifies logits in-place. */
void oc_apply_penalties(float *logits, size_t vocab_size,
                        const uint32_t *recent_tokens, size_t n_recent,
                        float frequency_penalty, float presence_penalty);

/* ─── Beam search ────────────────────────────────────────────────────────── */

typedef struct OcBeam {
    uint32_t *tokens;       /* token sequence (owned)                     */
    size_t    n_tokens;     /* current length                              */
    float     log_prob;     /* cumulative log probability                   */
    bool      finished;     /* reached EOS                                 */
} OcBeam;

typedef struct OcBeamSearchState {
    OcBeam *beams;          /* array of beam_width beams                   */
    size_t  beam_width;     /* number of beams to maintain                  */
    size_t  max_length;     /* maximum tokens to generate                   */
    float   length_penalty; /* 1.0 = no penalty, >1 = prefer longer        */
    uint32_t eos_token;     /* end-of-sequence token (0xFFFFFFFF = none)    */
    size_t  n_finished;     /* count of finished beams                      */
} OcBeamSearchState;

/* Initialize beam search state. */
OcError oc_beam_search_init(OcBeamSearchState *st, size_t beam_width,
                             size_t max_length, float length_penalty,
                             uint32_t eos_token, uint32_t first_token);

/* Expand beams: given logits for each beam, produce new candidate beams. */
OcError oc_beam_search_step(OcBeamSearchState *st,
                             const float *logits_per_beam, /* [beam_width × vocab] */
                             size_t vocab_size);

/* Check if beam search is complete (all beams finished or max length). */
bool oc_beam_search_done(const OcBeamSearchState *st);

/* Get the best (highest log_prob) beam. */
const OcBeam *oc_beam_search_best(const OcBeamSearchState *st);

/* Free beam search state. */
void oc_beam_search_free(OcBeamSearchState *st);

/* ─── Contrastive search ────────────────────────────────────────────────── */

/* Contrastive search: balances similarity to previous tokens with
 * a penalty for repetition.
 *
 * logits: [vocab_size]
 * past_keys: [seq_len × hidden_dim] (previous token representations)
 * current_key: [hidden_dim] (representation of the candidate being scored)
 * Returns the selected token. */
uint32_t oc_sample_contrastive(const float *logits, size_t vocab_size,
                                const float *past_keys,
                                const float *current_key,
                                size_t seq_len, size_t hidden_dim,
                                float alpha, float beta);

/* ─── Sampler chain ─────────────────────────────────────────────────────── */

typedef enum {
    OC_SAMPLER_STEP_TEMPERATURE = 0,
    OC_SAMPLER_STEP_TOP_K,
    OC_SAMPLER_STEP_TOP_P,
    OC_SAMPLER_STEP_TFS,
    OC_SAMPLER_STEP_TYPICAL,
    OC_SAMPLER_STEP_TOP_A,
    OC_SAMPLER_STEP_ETA,
    OC_SAMPLER_STEP_PENALTIES,
    OC_SAMPLER_STEP_MIROSTAT_V1,
    OC_SAMPLER_STEP_MIROSTAT_V2,
} OcSamplerStepType;

typedef struct OcSamplerStep {
    OcSamplerStepType type;
    float  param1;   /* e.g., temperature, k, p, z, tau */
    float  param2;   /* e.g., eta, a, epsilon, penalty */
} OcSamplerStep;

typedef struct OcSamplerChain {
    OcSamplerStep *steps;
    size_t n_steps;
    size_t capacity;
    uint32_t rng_state;
} OcSamplerChain;

/* Initialize an empty sampler chain. */
void oc_sampler_chain_init(OcSamplerChain *chain);

/* Add a step to the chain. */
OcError oc_sampler_chain_add(OcSamplerChain *chain,
                              OcSamplerStepType type,
                              float param1, float param2);

/* Run the full chain on logits and return a token. */
uint32_t oc_sampler_chain_sample(OcSamplerChain *chain,
                                  float *logits, size_t vocab_size,
                                  const uint32_t *recent_tokens,
                                  size_t n_recent,
                                  float *mirostat_mu);

/* Free the chain. */
void oc_sampler_chain_free(OcSamplerChain *chain);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ADVANCED_SAMPLING_H */
