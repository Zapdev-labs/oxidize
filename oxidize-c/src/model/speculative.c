/*
 * speculative.c — speculative decoding implementation.
 *
 * Port of oxidize-core/src/model/speculative.rs + sampling.rs::speculative_decode.
 *
 * Algorithm:
 *   1. Prefill prompt on both target + draft models, saving the last
 *      token's logits as the "seed" logits for the first step.
 *   2. Per step:
 *      a. Draft generates K tokens autoregressively (forward + sample),
 *         saving K sets of draft logits.
 *      b. Target forwards the K draft tokens, saving K sets of target
 *         logits (plus the saved seed → K+1 total).
 *      c. Verification kernel accepts/rejects draft tokens.
 *      d. Target rewinds + replays accepted tokens, saving last logits.
 *      e. Draft rewinds + replays accepted tokens, saving last logits.
 *      f. Emit accepted + residual/bonus tokens.
 */
#include "oxidize/speculative.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ─── RNG helpers ──────────────────────────────────────────────────────── */

static uint64_t xorshift64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float xorshift64_uniform(uint64_t *state)
{
    return (float)(xorshift64(state) >> 11) * (1.0f / 9007199254740992.0f);
}

/* ─── Softmax + sampling helpers ────────────────────────────────────────── */

static void softmax(float *probs, const float *logits, size_t n, float temp)
{
    if (n == 0) return;
    float inv_t = (temp > 0.0f) ? (1.0f / temp) : 1.0f;
    float mx = logits[0] * inv_t;
    for (size_t i = 1; i < n; i++) {
        float v = logits[i] * inv_t;
        if (v > mx) mx = v;
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        probs[i] = expf(logits[i] * inv_t - mx);
        sum += (double)probs[i];
    }
    if (sum > 0.0) {
        float inv = (float)(1.0 / sum);
        for (size_t i = 0; i < n; i++) probs[i] *= inv;
    }
}

static uint32_t sample_from_probs(const float *probs, size_t n, float r)
{
    float cum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        cum += probs[i];
        if (r < cum) return (uint32_t)i;
    }
    return (uint32_t)(n - 1);
}

/* ─── Verification kernel ──────────────────────────────────────────────── */

OcError oc_speculative_decode(
    const uint32_t *draft_tokens,
    float * const *draft_logits,
    float * const *target_logits,
    uint32_t k,
    size_t vocab_size,
    const OcSpeculativeConfig *cfg,
    uint64_t *seed_state,
    OcSpeculativeResult *out)
{
    if (draft_tokens == NULL || draft_logits == NULL || target_logits == NULL ||
        cfg == NULL || seed_state == NULL || out == NULL)
        return OC_ERR_INVALID_ARG;
    if (k == 0 || k > OC_SPEC_MAX_DRAFT || vocab_size == 0)
        return OC_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < k; i++) {
        if (draft_tokens[i] >= vocab_size) return OC_ERR_INVALID_ARG;
    }

    out->count = 0;
    out->accepted = 0;
    out->used_residual = false;

    float *draft_probs  = malloc(vocab_size * sizeof(float));
    float *target_probs = malloc(vocab_size * sizeof(float));
    if (!draft_probs || !target_probs) {
        free(draft_probs); free(target_probs);
        return OC_ERR_OOM;
    }

    for (uint32_t step = 0; step < k; step++) {
        uint32_t token = draft_tokens[step];

        if (cfg->greedy) {
            uint32_t amax = oc_argmax(target_logits[step], vocab_size);
            if (token == amax) {
                out->tokens[out->count++] = token;
                out->accepted++;
                continue;
            }
            out->tokens[out->count++] = amax;
            out->used_residual = true;
            goto done;
        }

        /* Stochastic: accept with prob min(1, p/q). */
        softmax(draft_probs,  draft_logits[step],  vocab_size, cfg->temperature);
        softmax(target_probs, target_logits[step], vocab_size, cfg->temperature);

        float q = draft_probs[token];
        if (q < 1e-38f) q = 1e-38f;
        float p = target_probs[token];
        float accept_prob = (p / q < 1.0f) ? (p / q) : 1.0f;

        if (xorshift64_uniform(seed_state) <= accept_prob) {
            out->tokens[out->count++] = token;
            out->accepted++;
            continue;
        }

        /* Reject: sample from residual = normalize(max(0, p - q)). */
        double res_sum = 0.0;
        for (size_t i = 0; i < vocab_size; i++) {
            float diff = target_probs[i] - draft_probs[i];
            target_probs[i] = (diff > 0.0f) ? diff : 0.0f;
            res_sum += (double)target_probs[i];
        }
        if (res_sum > 0.0 && isfinite((float)res_sum)) {
            float inv = (float)(1.0 / res_sum);
            for (size_t i = 0; i < vocab_size; i++) target_probs[i] *= inv;
            out->tokens[out->count++] = sample_from_probs(
                target_probs, vocab_size, xorshift64_uniform(seed_state));
        } else {
            out->tokens[out->count++] = oc_argmax(target_logits[step], vocab_size);
        }
        out->used_residual = true;
        goto done;
    }

    /* All K accepted → bonus token from target_logits[k]. */
    if (cfg->greedy) {
        out->tokens[out->count++] = oc_argmax(target_logits[k], vocab_size);
    } else {
        softmax(target_probs, target_logits[k], vocab_size, cfg->temperature);
        out->tokens[out->count++] = sample_from_probs(
            target_probs, vocab_size, xorshift64_uniform(seed_state));
    }

done:
    free(draft_probs); free(target_probs);
    return OC_OK;
}

/* ─── Full speculative generation loop ──────────────────────────────────── */

OcError oc_speculative_generate(
    OcLlamaModel *target, OcLlamaSession *target_sess,
    OcLlamaModel *draft, OcLlamaSession *draft_sess,
    const uint32_t *prompt, size_t prompt_len,
    const OcSpeculativeConfig *cfg,
    uint32_t *out_tokens, size_t *out_len, size_t out_cap,
    OcSpeculativeStats *stats)
{
    if (!target || !target_sess || !draft || !draft_sess || !prompt ||
        !cfg || !out_tokens || !out_len)
        return OC_ERR_INVALID_ARG;
    if (prompt_len == 0) return OC_ERR_INVALID_ARG;
    if (out_cap == 0) return OC_ERR_INVALID_ARG;
    if (target->cfg.vocab_size != draft->cfg.vocab_size)
        return OC_ERR_INVALID_ARG;

    uint32_t k = cfg->draft_tokens_per_step;
    if (k == 0) k = 4;
    if (k > OC_SPEC_MAX_DRAFT) k = OC_SPEC_MAX_DRAFT;

    size_t vocab = target->cfg.vocab_size;
    uint64_t seed = cfg->seed ? cfg->seed : 0x9E3779B97F4A7C15ULL;

    /* Logits scratch: [K+1] arrays of vocab_size for target, [K] for draft.
     * We use a flat allocation and set up pointer arrays.
     * target_ptrs[0] is the "seed" logits (from prefill or previous step's
     * last replay). target_ptrs[1..K] are from forwarding draft tokens.
     * draft_ptrs[0..K-1] are from draft generation. */
    float *target_buf = malloc((size_t)(k + 1) * vocab * sizeof(float));
    float *draft_buf  = malloc((size_t)k * vocab * sizeof(float));
    if (!target_buf || !draft_buf) {
        free(target_buf); free(draft_buf);
        return OC_ERR_OOM;
    }

    float *target_ptrs[OC_SPEC_MAX_DRAFT + 1];
    float *draft_ptrs[OC_SPEC_MAX_DRAFT];
    for (uint32_t i = 0; i <= k; i++)
        target_ptrs[i] = target_buf + (size_t)i * vocab;
    for (uint32_t i = 0; i < k; i++)
        draft_ptrs[i] = draft_buf + (size_t)i * vocab;

    uint32_t draft_tokens[OC_SPEC_MAX_DRAFT];
    OcSamplerConfig scfg = OC_SAMPLER_DEFAULT;
    scfg.seed = seed;

    if (stats) memset(stats, 0, sizeof(*stats));
    *out_len = 0;
    OcError status = OC_OK;

    /* 1. Prefill prompt on target. Forward all but last with NULL logits,
     *    then forward last token with logits → target_ptrs[0]. */
    oc_llama_session_reset(target_sess);
    for (size_t i = 0; i < prompt_len - 1; i++) {
        status = oc_llama_forward(target_sess, prompt[i], NULL);
        if (status != OC_OK) goto cleanup;
    }
    status = oc_llama_forward(target_sess, prompt[prompt_len - 1], target_ptrs[0]);
    if (status != OC_OK) goto cleanup;

    /* Prefill prompt on draft, same pattern → draft_ptrs[0]. */
    oc_llama_session_reset(draft_sess);
    for (size_t i = 0; i < prompt_len - 1; i++) {
        status = oc_llama_forward(draft_sess, prompt[i], NULL);
        if (status != OC_OK) goto cleanup;
    }
    status = oc_llama_forward(draft_sess, prompt[prompt_len - 1], draft_ptrs[0]);
    if (status != OC_OK) goto cleanup;

    /* Sample the first token from target seed logits. */
    uint32_t current_token;
    if (cfg->greedy) {
        current_token = oc_argmax(target_ptrs[0], vocab);
    } else {
        scfg.type = OC_SAMPLER_TEMPERATURE;
        scfg.temperature = cfg->temperature;
        scfg.seed = xorshift64(&seed);
        current_token = oc_sample(target_ptrs[0], vocab, &scfg, NULL, 0);
    }
    if (current_token == cfg->stop_token) goto cleanup;

    out_tokens[(*out_len)++] = current_token;
    if (stats) stats->emitted_tokens++;
    status = oc_llama_forward(target_sess, current_token, target_ptrs[0]);
    if (status != OC_OK) goto cleanup;
    status = oc_llama_forward(draft_sess, current_token, draft_ptrs[0]);
    if (status != OC_OK) goto cleanup;

    /* 2. Speculative loop. */
    while (*out_len < out_cap) {
        uint32_t max_new = cfg->max_new_tokens;
        if (max_new > 0 && stats && stats->emitted_tokens >= max_new) break;
        if (max_new > 0 && !stats && *out_len >= max_new) break;

        /* --- Draft generation: K tokens autoregressively. ---
         * draft_ptrs[0] already holds logits from the start position.
         * Generate draft_tokens[0] from draft_ptrs[0], forward it to get
         * draft_ptrs[1], etc. */
        uint32_t draft_ckpt = (uint32_t)draft_sess->pos;
        for (uint32_t i = 0; i < k; i++) {
            if (cfg->greedy) {
                draft_tokens[i] = oc_argmax(draft_ptrs[i], vocab);
            } else {
                scfg.type = OC_SAMPLER_TEMPERATURE;
                scfg.temperature = cfg->temperature;
                scfg.seed = xorshift64(&seed);
                draft_tokens[i] = oc_sample(draft_ptrs[i], vocab, &scfg, NULL, 0);
            }
            status = oc_llama_forward(draft_sess, draft_tokens[i],
                                      (i + 1 < k) ? draft_ptrs[i + 1] : NULL);
            if (status != OC_OK) goto cleanup;
        }
        if (stats) {
            stats->total_draft_tokens += k;
            stats->draft_forward_passes += k;
        }

        /* --- Target verification: forward K draft tokens, collecting
         *     target_ptrs[1..K]. target_ptrs[0] was saved from the
         *     previous step (or prefill). --- */
        uint32_t target_ckpt = target_sess->pos;
        for (uint32_t i = 0; i < k; i++) {
            status = oc_llama_forward(target_sess, draft_tokens[i],
                                      target_ptrs[i + 1]);
            if (status != OC_OK) goto cleanup;
        }
        if (stats) stats->target_forward_passes++;

        /* --- Verification kernel. --- */
        OcSpeculativeResult result;
        uint64_t vs = xorshift64(&seed);
        OcError e = oc_speculative_decode(
            draft_tokens, draft_ptrs, target_ptrs,
            k, vocab, cfg, &vs, &result);
        if (e != OC_OK) { status = e; goto cleanup; }
        if (stats) stats->accepted_draft_tokens += result.accepted;

        size_t emit_count = result.count;
        if (emit_count > out_cap - *out_len) emit_count = out_cap - *out_len;
        if (cfg->max_new_tokens > 0 &&
            emit_count > cfg->max_new_tokens - *out_len)
            emit_count = cfg->max_new_tokens - *out_len;

        oc_llama_session_rewind(target_sess, target_ckpt);
        oc_llama_session_rewind(draft_sess, draft_ckpt);
        for (size_t i = 0; i < emit_count; i++) {
            float *target_logits = i + 1 == emit_count ? target_ptrs[0] : NULL;
            float *draft_logits = i + 1 == emit_count ? draft_ptrs[0] : NULL;
            status = oc_llama_forward(target_sess, result.tokens[i], target_logits);
            if (status != OC_OK) goto cleanup;
            status = oc_llama_forward(draft_sess, result.tokens[i], draft_logits);
            if (status != OC_OK) goto cleanup;
        }

        for (size_t i = 0; i < emit_count; i++) {
            out_tokens[(*out_len)++] = result.tokens[i];
            if (stats) stats->emitted_tokens++;
            if (result.tokens[i] == cfg->stop_token) goto cleanup;
        }
        if (emit_count == 0) break;
        current_token = result.tokens[emit_count - 1];
        if (cfg->max_new_tokens > 0 && *out_len >= cfg->max_new_tokens) break;

        /* --- Fallback if acceptance rate too low. --- */
        if (stats && cfg->min_acceptance_rate > 0.0f &&
            stats->total_draft_tokens >= 10) {
            float rate = (float)stats->accepted_draft_tokens /
                         (float)stats->total_draft_tokens;
            if (rate < cfg->min_acceptance_rate) {
                /* Plain target sampling for this step. */
                uint32_t fb;
                if (cfg->greedy) {
                    fb = oc_argmax(target_ptrs[0], vocab);
                } else {
                    scfg.type = OC_SAMPLER_TEMPERATURE;
                    scfg.temperature = cfg->temperature;
                    scfg.seed = xorshift64(&seed);
                    fb = oc_sample(target_ptrs[0], vocab, &scfg, NULL, 0);
                }
                if (*out_len < out_cap) {
                    out_tokens[(*out_len)++] = fb;
                    if (stats) stats->emitted_tokens++;
                }
                if (fb == cfg->stop_token) goto cleanup;
                /* Forward through both models. */
                status = oc_llama_forward(target_sess, fb, target_ptrs[0]);
                if (status != OC_OK) goto cleanup;
                status = oc_llama_forward(draft_sess, fb, draft_ptrs[0]);
                if (status != OC_OK) goto cleanup;
                current_token = fb;
            }
        }
    }

cleanup:
    free(target_buf); free(draft_buf);
    return status;
}
