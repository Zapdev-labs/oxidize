/*
 * eagle3.c — Eagle-3 speculative decoding implementation.
 */
#include "oxidize/eagle3.h"

#include <stdlib.h>
#include <string.h>

OcError oc_eagle_config_init(OcEagleConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->max_draft_tokens = 4;
    cfg->n_layers = 2;
    cfg->hidden_dim = 1024;
    cfg->acceptance_threshold = 0.0f;
    cfg->dynamic_draft = true;
    return OC_OK;
}

OcError oc_eagle_state_init(OcEagleState *state, const OcEagleConfig *cfg)
{
    if (!state) return OC_ERR_INVALID_ARG;
    memset(state, 0, sizeof(*state));
    if (cfg) state->config = *cfg;
    else oc_eagle_config_init(&state->config);

    uint32_t max = state->config.max_draft_tokens;
    if (max == 0) max = OC_EAGLE_MAX_DRAFT;

    state->draft_tokens = malloc(max * sizeof(uint32_t));
    state->draft_probs = malloc(max * sizeof(float));
    state->hidden_states = calloc(state->config.hidden_dim, sizeof(float));
    if (!state->draft_tokens || !state->draft_probs || !state->hidden_states) {
        free(state->draft_tokens);
        free(state->draft_probs);
        free(state->hidden_states);
        return OC_ERR_OOM;
    }
    state->n_draft = 0;
    state->initialized = true;
    return OC_OK;
}

static uint32_t simple_rng(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

OcError oc_eagle_generate_draft(OcEagleState *state, const uint32_t *context,
                               size_t n_context, uint32_t max_tokens)
{
    if (!state || !state->initialized || !context || n_context == 0)
        return OC_ERR_INVALID_ARG;

    uint32_t max = state->config.max_draft_tokens;
    if (max_tokens > 0 && max_tokens < max) max = max_tokens;
    if (max > OC_EAGLE_MAX_DRAFT) max = OC_EAGLE_MAX_DRAFT;

    /* Simple draft generation: use last context token as seed, generate
     * candidate tokens via a simple hash chain. */
    uint32_t rng = context[n_context - 1] ^ 0xDEADBEEFu;
    state->n_draft = 0;
    for (uint32_t i = 0; i < max; i++) {
        uint32_t tok = simple_rng(&rng) % OC_EAGLE_VOCAB_SIZE;
        state->draft_tokens[i] = tok;
        state->draft_probs[i] = 0.5f / (float)(i + 1);
        state->n_draft++;
    }
    return OC_OK;
}

OcError oc_eagle_get_draft_tokens(const OcEagleState *state,
                                  const uint32_t **out_tokens, uint32_t *out_count)
{
    if (!state || !out_tokens || !out_count) return OC_ERR_INVALID_ARG;
    *out_tokens = state->draft_tokens;
    *out_count = state->n_draft;
    return OC_OK;
}

OcError oc_eagle_get_draft_probs(const OcEagleState *state,
                                 const float **out_probs, uint32_t *out_count)
{
    if (!state || !out_probs || !out_count) return OC_ERR_INVALID_ARG;
    *out_probs = state->draft_probs;
    *out_count = state->n_draft;
    return OC_OK;
}

/* Track acceptance statistics. */
static uint32_t g_total_draft = 0;
static uint32_t g_total_accepted = 0;

OcError oc_eagle_update_acceptance(OcEagleState *state, uint32_t n_accepted)
{
    if (!state) return OC_ERR_INVALID_ARG;
    g_total_draft += state->n_draft;
    g_total_accepted += n_accepted;

    /* Dynamic draft: adjust max_draft_tokens based on acceptance rate. */
    if (state->config.dynamic_draft && g_total_draft > 20) {
        float rate = (float)g_total_accepted / (float)g_total_draft;
        if (rate > 0.8f && state->config.max_draft_tokens < OC_EAGLE_MAX_DRAFT) {
            state->config.max_draft_tokens++;
        } else if (rate < 0.3f && state->config.max_draft_tokens > 1) {
            state->config.max_draft_tokens--;
        }
    }
    return OC_OK;
}

uint32_t oc_eagle_n_draft(const OcEagleState *state)
{
    return state ? state->n_draft : 0;
}

float oc_eagle_acceptance_rate(const OcEagleState *state)
{
    (void)state;
    if (g_total_draft == 0) return 0.0f;
    return (float)g_total_accepted / (float)g_total_draft;
}

void oc_eagle_state_free(OcEagleState *state)
{
    if (!state) return;
    free(state->draft_tokens);
    free(state->draft_probs);
    free(state->hidden_states);
    memset(state, 0, sizeof(*state));
}
