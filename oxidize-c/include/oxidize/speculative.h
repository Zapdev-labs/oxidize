/*
 * speculative.h — speculative decoding (draft model + target verification).
 *
 * Port of oxidize-core/src/model/speculative.rs + sampling.rs::speculative_decode.
 * Two-phase generation: a small draft model proposes K tokens, the target
 * model verifies them in a single batched forward, and accepted tokens are
 * emitted. On rejection, a residual-sampled token (stochastic) or the
 * target argmax (greedy) is emitted instead. When all K draft tokens are
 * accepted, one bonus token is sampled from the target's logits — so every
 * step emits at least 1 token and at most K+1.
 *
 * Requires two loaded models (target + draft) sharing the same vocabulary.
 * The draft model can be any OcLlamaModel (typically a smaller checkpoint).
 */
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

/* Verification kernel: compare draft vs target logits, return accepted
 * tokens + one residual/bonus token.
 *
 *   draft_tokens  : [k] proposed token ids
 *   draft_logits  : [k] pointers to vocab_size-length logit arrays
 *   target_logits : [k+1] pointers (one per position, including bonus)
 *   k             : number of draft tokens
 *   vocab_size    : vocabulary dimension
 *   cfg           : config (greedy/stochastic, temperature, seed)
 *   seed_state    : in/out RNG state (xorshift64); advanced per use
 *   out           : result (tokens, count, accepted, used_residual)
 */
OcError oc_speculative_decode(
    const uint32_t *draft_tokens,
    float * const *draft_logits,
    float * const *target_logits,
    uint32_t k,
    size_t vocab_size,
    const OcSpeculativeConfig *cfg,
    uint64_t *seed_state,
    OcSpeculativeResult *out);

/* Full speculative generation loop.
 *
 *   target/draft        : loaded models (same vocab)
 *   target_sess/draft_sess : sessions (will be prefilled + advanced)
 *   prompt / prompt_len : input token ids
 *   cfg                 : config
 *   out_tokens          : output buffer (caller-allocated)
 *   out_len             : set to number of tokens written
 *   out_cap             : capacity of out_tokens
 *   stats               : statistics (may be NULL)
 */
OcError oc_speculative_generate(
    OcLlamaModel *target, OcLlamaSession *target_sess,
    OcLlamaModel *draft, OcLlamaSession *draft_sess,
    const uint32_t *prompt, size_t prompt_len,
    const OcSpeculativeConfig *cfg,
    uint32_t *out_tokens, size_t *out_len, size_t out_cap,
    OcSpeculativeStats *stats);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SPECULATIVE_H */
