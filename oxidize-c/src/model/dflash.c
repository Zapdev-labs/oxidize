/*
 * dflash.c — DFlash speculative decoding implementation.
 */
#include "oxidize/dflash.h"

#include <math.h>
#include <string.h>

OcError oc_dflash_config_init(OcDFlashConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->max_draft_tokens = 8;
    cfg->verification_window = 16;
    cfg->acceptance_threshold = 0.0f;
    cfg->adaptive = true;
    return OC_OK;
}

OcError oc_dflash_state_init(OcDFlashState *state, const OcDFlashConfig *cfg)
{
    if (!state) return OC_ERR_INVALID_ARG;
    memset(state, 0, sizeof(*state));
    if (cfg) state->config = *cfg;
    else oc_dflash_config_init(&state->config);
    return OC_OK;
}

OcError oc_dflash_set_draft(OcDFlashState *state, const uint32_t *tokens,
                           const float *logprobs, uint32_t n)
{
    if (!state || !tokens || !logprobs) return OC_ERR_INVALID_ARG;
    if (n > OC_DFLASH_MAX_DRAFT) n = OC_DFLASH_MAX_DRAFT;
    if (n > state->config.max_draft_tokens) n = state->config.max_draft_tokens;
    memcpy(state->draft_tokens, tokens, n * sizeof(uint32_t));
    memcpy(state->draft_logprobs, logprobs, n * sizeof(float));
    state->n_draft = n;
    state->n_accepted = 0;
    return OC_OK;
}

OcError oc_dflash_set_target(OcDFlashState *state, const uint32_t *tokens,
                            const float *logprobs, uint32_t n)
{
    if (!state || !tokens || !logprobs) return OC_ERR_INVALID_ARG;
    if (n > OC_DFLASH_MAX_DRAFT) n = OC_DFLASH_MAX_DRAFT;
    memcpy(state->target_tokens, tokens, n * sizeof(uint32_t));
    memcpy(state->target_logprobs, logprobs, n * sizeof(float));
    return OC_OK;
}

OcError oc_dflash_verify(OcDFlashState *state, uint32_t *out_accepted, uint32_t *out_n)
{
    if (!state || !out_accepted || !out_n) return OC_ERR_INVALID_ARG;
    *out_n = 0;
    uint32_t accepted = 0;

    for (uint32_t i = 0; i < state->n_draft; i++) {
        if (state->draft_tokens[i] == state->target_tokens[i]) {
            out_accepted[accepted++] = state->target_tokens[i];
            /* Check if we should continue (stochastic acceptance). */
            if (state->config.acceptance_threshold > 0.0f) {
                float ratio = expf(state->target_logprobs[i] - state->draft_logprobs[i]);
                if (ratio < state->config.acceptance_threshold) break;
            }
        } else {
            /* Mismatch: replace with target token and stop. */
            out_accepted[accepted++] = state->target_tokens[i];
            break;
        }
    }

    state->n_accepted = accepted;
    state->total_accepted += accepted;
    state->total_proposed += state->n_draft;
    *out_n = accepted;

    /* Adaptive: adjust max_draft_tokens based on acceptance rate. */
    if (state->config.adaptive && state->total_proposed > 20) {
        float rate = (float)state->total_accepted / (float)state->total_proposed;
        if (rate > 0.7f && state->config.max_draft_tokens < OC_DFLASH_MAX_DRAFT) {
            state->config.max_draft_tokens++;
        } else if (rate < 0.3f && state->config.max_draft_tokens > 1) {
            state->config.max_draft_tokens--;
        }
    }

    return OC_OK;
}

OcError oc_dflash_get_accepted(const OcDFlashState *state, const uint32_t **out_tokens, uint32_t *out_n)
{
    if (!state || !out_tokens || !out_n) return OC_ERR_INVALID_ARG;
    *out_tokens = state->target_tokens; /* accepted tokens are at the start */
    *out_n = state->n_accepted;
    return OC_OK;
}

float oc_dflash_acceptance_rate(const OcDFlashState *state)
{
    if (!state || state->total_proposed == 0) return 0.0f;
    return (float)state->total_accepted / (float)state->total_proposed;
}

uint32_t oc_dflash_avg_acceptance(const OcDFlashState *state)
{
    if (!state || state->total_proposed == 0) return 0;
    return (uint32_t)(100.0f * (float)state->total_accepted / (float)state->total_proposed);
}

void oc_dflash_state_free(OcDFlashState *state)
{
    if (!state) return;
    memset(state, 0, sizeof(*state));
}
