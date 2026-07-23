/*
 * generation.c — Token generation engine implementation.
 */
#include "oxidize/generation.h"

#include <stdlib.h>
#include <string.h>

OcError oc_gen_config_init(OcGenConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->max_tokens = 256;
    cfg->temperature = 0.8f;
    cfg->top_p = 0.9f;
    cfg->top_k = 40;
    cfg->repeat_penalty = 1.1f;
    cfg->repeat_last_n = 64;
    cfg->seed = 0;
    cfg->stream = true;
    cfg->stop_on_eos = true;
    return OC_OK;
}

OcError oc_gen_state_init(OcGenState *state, const uint32_t *context, size_t n)
{
    if (!state) return OC_ERR_INVALID_ARG;
    memset(state, 0, sizeof(*state));
    if (context && n > 0) {
        if (n > 4096) n = 4096;
        state->context = malloc(n * sizeof(uint32_t));
        if (!state->context) return OC_ERR_OOM;
        memcpy(state->context, context, n * sizeof(uint32_t));
        state->n_context = n;
        state->pos = (uint32_t)n;
        /* Seed recent tokens from context. */
        size_t start = n > 64 ? n - 64 : 0;
        for (size_t i = start; i < n; i++)
            state->recent_tokens[state->n_recent++] = context[i];
    }
    memset(&state->sampler, 0, sizeof(state->sampler));
    state->sampler.type = OC_SAMPLER_TOP_P;
    state->sampler.top_p = 0.9f;
    state->sampler.top_k = 40;
    return OC_OK;
}

OcError oc_gen_state_add_token(OcGenState *state, uint32_t token)
{
    if (!state) return OC_ERR_INVALID_ARG;
    if (state->n_recent < 64) {
        state->recent_tokens[state->n_recent++] = token;
    } else {
        memmove(state->recent_tokens, state->recent_tokens + 1, 63 * sizeof(uint32_t));
        state->recent_tokens[63] = token;
    }
    state->pos++;
    return OC_OK;
}

OcError oc_gen_result_init(OcGenResult *result, size_t max_tokens)
{
    if (!result) return OC_ERR_INVALID_ARG;
    memset(result, 0, sizeof(*result));
    if (max_tokens == 0) max_tokens = OC_GEN_MAX_TOKENS;
    if (max_tokens > OC_GEN_MAX_TOKENS) max_tokens = OC_GEN_MAX_TOKENS;
    result->tokens = malloc(max_tokens * sizeof(uint32_t));
    if (!result->tokens) return OC_ERR_OOM;
    result->n_tokens = 0;
    return OC_OK;
}

void oc_gen_result_free(OcGenResult *result)
{
    if (!result) return;
    free(result->tokens);
    memset(result, 0, sizeof(*result));
}

OcError oc_gen_config_from_cli(const OcGenConfig *cfg, OcSamplerConfig *out)
{
    if (!cfg || !out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));
    if (cfg->top_k > 0) {
        out->type = OC_SAMPLER_TOP_K;
        out->top_k = cfg->top_k;
        out->top_p = cfg->top_p;
    } else {
        out->type = OC_SAMPLER_TOP_P;
        out->top_p = cfg->top_p;
        out->top_k = 0;
    }
    out->temperature = cfg->temperature;
    out->repeat_penalty = cfg->repeat_penalty;
    out->seed = cfg->seed;
    return OC_OK;
}

const char *oc_gen_stop_reason(const OcGenResult *result)
{
    if (!result) return "unknown";
    if (result->stopped_on_eos) return "stop";
    if (result->n_tokens > 0) return "length";
    return "error";
}

uint64_t oc_gen_total_tokens(const OcGenResult *result)
{
    if (!result) return 0;
    return (uint64_t)result->n_prompt_tokens + (uint64_t)result->n_tokens;
}
