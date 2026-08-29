#ifndef OXIDIZE_SPECULATIVE_H
#define OXIDIZE_SPECULATIVE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/llama.h"
#include "oxidize/sampling.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_SPEC_MAX_DRAFT 32   /* hard cap on draft_tokens_per_step */

typedef struct OcSpeculativeConfig {
    uint32_t draft_tokens_per_step;  /* K (default 4)                       */
    uint32_t max_new_tokens;         /* 0 = unlimited until stop           */
    float    min_acceptance_rate;    /* below this, fall back to plain (0.3) */
    bool     greedy;                 /* true = greedy verify, false = stochastic */
    float    temperature;            /* verify temperature (stochastic)     */
    uint32_t stop_token;             /* 0xFFFFFFFF = no stop               */
    uint64_t seed;                   /* RNG seed for stochastic sampling   */
} OcSpeculativeConfig;

#define OC_SPECULATIVE_DEFAULT ((OcSpeculativeConfig){ \
    4u, 0u, 0.3f, true, 1.0f, 0xFFFFFFFFu, 0ull })

typedef struct OcSpeculativeStats {
    uint64_t total_draft_tokens;
    uint64_t accepted_draft_tokens;
    uint64_t target_forward_passes;
    uint64_t draft_forward_passes;
    uint64_t emitted_tokens;
} OcSpeculativeStats;

typedef struct OcSpeculativeResult {
    uint32_t tokens[OC_SPEC_MAX_DRAFT + 1]; /* accepted + residual/bonus   */
    uint32_t count;                          /* number of tokens emitted    */
    uint32_t accepted;                        /* draft tokens accepted       */
    bool     used_residual;                   /* last token was residual     */
} OcSpeculativeResult;

/* Verification kernel: compare draft vs target logits, return accepted */
OcError oc_speculative_decode(
    const uint32_t *draft_tokens,
    float * const *draft_logits,
    float * const *target_logits,
    uint32_t k,
    size_t vocab_size,
    const OcSpeculativeConfig *cfg,
    uint64_t *seed_state,
    OcSpeculativeResult *out);

/* Full speculative generation loop. */
OcError oc_speculative_generate(
    OcLlamaModel *target, OcLlamaSession *target_sess,
    OcLlamaModel *draft, OcLlamaSession *draft_sess,
    const uint32_t *prompt, size_t prompt_len,
    const OcSpeculativeConfig *cfg,
    uint32_t *out_tokens, size_t *out_len, size_t out_cap,
    OcSpeculativeStats *stats);


/* Acceptance rate: accepted / total_draft (0.0 if no drafts). */
double oc_speculative_acceptance_rate(const OcSpeculativeStats *stats);

/* Tokens per target forward: emitted_tokens / target_forward_passes (0.0 if none). */
double oc_speculative_tokens_per_target_forward(const OcSpeculativeStats *stats);

/* Estimated speedup vs plain decoding (1.0 = no speedup). */
double oc_speculative_estimated_speedup(const OcSpeculativeStats *stats);


/* Load a GGUF file as a draft model for speculative decoding.
 * The caller owns the returned OcLlamaModel and must free it.
 * Returns OC_ERR_MODEL if the GGUF is not a valid draft model. */
OcError oc_speculative_load_draft(const char *path,
                                    OcLlamaModel **out_model,
                                    OcLlamaSession **out_sess);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SPECULATIVE_H */
