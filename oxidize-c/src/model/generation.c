#define _POSIX_C_SOURCE 200809L
/*
 * generation.c — Token generation engine implementation.
 */
#include "oxidize/generation.h"
#include "oxidize/inf_model.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

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

/* ─── Full generation loop ────────────────────────────────────────────── */

/* Get monotonic time in seconds. */
static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

OcError oc_gen_run(void *model_ptr,
                    const uint32_t *prompt, size_t n_prompt,
                    const OcGenConfig *cfg,
                    OcGenResult *result,
                    OcGenTokenCb callback, void *user)
{
    OcInferenceModel *model = (OcInferenceModel *)model_ptr;
    if (!model || !cfg || !result) return OC_ERR_INVALID_ARG;
    if (n_prompt > 0 && !prompt) return OC_ERR_INVALID_ARG;

    OcSamplerConfig sampler;
    oc_gen_config_from_cli(cfg, &sampler);

    /* Track recent tokens for repeat penalty. */
    uint32_t recent[64];
    size_t n_recent = 0;

    /* Seed recent tokens from prompt. */
    if (prompt && n_prompt > 0) {
        size_t start = n_prompt > 64 ? n_prompt - 64 : 0;
        for (size_t i = start; i < n_prompt; i++) {
            if (n_recent < 64)
                recent[n_recent++] = prompt[i];
        }
    }

    result->n_prompt_tokens = n_prompt;
    result->stopped_on_eos = false;
    result->n_tokens = 0;

    /* 1. Prefill: forward all prompt tokens. */
    double prefill_start = 0.0;
    float *logits = NULL;
    size_t logits_len = 0;

    if (n_prompt > 0) {
        prefill_start = get_time_sec();
        for (size_t i = 0; i < n_prompt; i++) {
            OcError e = oc_inf_model_forward_token(model, prompt[i], i);
            if (e != OC_OK) return e;
        }
        /* Get logits from the last prompt token. */
        OcError e = oc_inf_model_final_head_from_workspace(model, &logits, &logits_len);
        if (e != OC_OK) return e;
        result->prefill_time_sec = get_time_sec() - prefill_start;
        result->prefill_tokens_per_sec = n_prompt / result->prefill_time_sec;
    } else {
        /* No prompt: embed token 0 and get logits. */
        prefill_start = get_time_sec();
        OcError e = oc_inf_model_forward_token(model, 0, 0);
        if (e != OC_OK) return e;
        e = oc_inf_model_final_head_from_workspace(model, &logits, &logits_len);
        if (e != OC_OK) return e;
        result->prefill_time_sec = get_time_sec() - prefill_start;
        result->prefill_tokens_per_sec = 1.0 / result->prefill_time_sec;
        n_prompt = 1;
    }

    /* 2. Sample first token. */
    uint32_t token = oc_sample(logits, logits_len, &sampler, recent, n_recent);
    if (n_recent < 64) recent[n_recent++] = token;
    result->tokens[result->n_tokens++] = token;
    result->eval_time_sec = 0.0;

    if (callback) callback(token, NULL, user);

    /* Check stop. */
    if (cfg->stop_on_eos && cfg->top_k == 0) {
        /* No explicit stop token; just check max_tokens. */
    }
    if (cfg->stop_on_eos && token == 0) {
        result->stopped_on_eos = true;
        result->tokens_per_sec = 1.0 / result->prefill_time_sec;
        return OC_OK;
    }

    /* 3. Decode loop. */
    double decode_start = get_time_sec();
    size_t pos = n_prompt;

    for (size_t i = 1; i < cfg->max_tokens; i++) {
        OcError e = oc_inf_model_forward_token_logits(model, token, pos,
                                                        &logits, &logits_len);
        if (e != OC_OK) return e;

        token = oc_sample(logits, logits_len, &sampler, recent, n_recent);
        if (n_recent < 64) recent[n_recent++] = token;

        result->tokens[result->n_tokens++] = token;
        pos++;

        if (callback) callback(token, NULL, user);

        if (cfg->stop_on_eos && token == 0) {
            result->stopped_on_eos = true;
            break;
        }
    }

    double total_decode = get_time_sec() - decode_start;
    result->eval_time_sec = total_decode + result->prefill_time_sec;
    result->tokens_per_sec = (double)result->n_tokens / total_decode;

    return OC_OK;
}

OcError oc_gen_run_mtp(void *model_ptr,
                         const uint32_t *prompt, size_t n_prompt,
                         const OcGenConfig *cfg,
                         size_t draft_tokens_per_step,
                         OcGenResult *result,
                         OcGenTokenCb callback, void *user)
{
    OcInferenceModel *model = (OcInferenceModel *)model_ptr;
    if (!model || !cfg || !result) return OC_ERR_INVALID_ARG;
    if (n_prompt > 0 && !prompt) return OC_ERR_INVALID_ARG;

    /* If no MTP block, fall back to standard generation. */
    if (!oc_inf_model_has_mtp(model)) {
        return oc_gen_run(model, prompt, n_prompt, cfg, result, callback, user);
    }

    if (draft_tokens_per_step == 0)
        draft_tokens_per_step = 4;

    OcSamplerConfig sampler;
    oc_gen_config_from_cli(cfg, &sampler);

    uint32_t recent[64];
    size_t n_recent = 0;

    if (prompt && n_prompt > 0) {
        size_t start = n_prompt > 64 ? n_prompt - 64 : 0;
        for (size_t i = start; i < n_prompt; i++) {
            if (n_recent < 64)
                recent[n_recent++] = prompt[i];
        }
    }

    result->n_prompt_tokens = n_prompt;
    result->stopped_on_eos = false;
    result->n_tokens = 0;

    /* Prefill. */
    double prefill_start = get_time_sec();
    float *logits = NULL;
    size_t logits_len = 0;

    if (n_prompt > 0) {
        for (size_t i = 0; i < n_prompt; i++) {
            OcError e = oc_inf_model_forward_token(model, prompt[i], i);
            if (e != OC_OK) return e;
        }
        oc_inf_model_final_head_from_workspace(model, &logits, &logits_len);
        result->prefill_time_sec = get_time_sec() - prefill_start;
        result->prefill_tokens_per_sec = n_prompt / result->prefill_time_sec;
    } else {
        oc_inf_model_forward_token(model, 0, 0);
        oc_inf_model_final_head_from_workspace(model, &logits, &logits_len);
        result->prefill_time_sec = get_time_sec() - prefill_start;
        n_prompt = 1;
    }

    /* Sample first token. */
    uint32_t token = oc_sample(logits, logits_len, &sampler, recent, n_recent);
    if (n_recent < 64) recent[n_recent++] = token;
    result->tokens[result->n_tokens++] = token;
    if (callback) callback(token, NULL, user);

    /* Decode loop with MTP speculative decoding. */
    double decode_start = get_time_sec();
    size_t pos = n_prompt;
    size_t vocab = oc_inf_model_config(model)->vocab_size;

    /* Buffers for draft tokens. */
    uint32_t draft_tokens[32];
    float *draft_logits = malloc(32 * vocab * sizeof(float));
    if (!draft_logits) return OC_ERR_OOM;

    const float *last_hidden = oc_inf_model_last_output_hidden(model);

    while (result->n_tokens < cfg->max_tokens) {
        size_t n_draft = 0;
        size_t max_draft = result->n_tokens + draft_tokens_per_step < cfg->max_tokens
                           ? draft_tokens_per_step
                           : (cfg->max_tokens - result->n_tokens);
        if (max_draft > 32) max_draft = 32;
        if (max_draft == 0) break;

        /* Generate draft tokens. */
        OcError e = oc_inf_model_draft_mtp_tokens(model, token,
                                                     last_hidden,
                                                     oc_inf_model_config(model)->hidden_size,
                                                     max_draft,
                                                     draft_tokens,
                                                     draft_logits,
                                                     &n_draft);
        if (e != OC_OK) {
            free(draft_logits);
            return e;
        }

        /* Verify each draft token with the target model. */
        for (size_t d = 0; d < n_draft && result->n_tokens < cfg->max_tokens; d++) {
            e = oc_inf_model_forward_token_logits(model, token, pos,
                                                    &logits, &logits_len);
            if (e != OC_OK) {
                free(draft_logits);
                return e;
            }

            uint32_t target_token = oc_sample(logits, logits_len, &sampler,
                                               recent, n_recent);

            if (target_token == draft_tokens[d]) {
                /* Accept draft token. */
                token = draft_tokens[d];
            } else {
                /* Reject: use target token. */
                token = target_token;
            }

            if (n_recent < 64) recent[n_recent++] = token;
            result->tokens[result->n_tokens++] = token;
            pos++;

            if (callback) callback(token, NULL, user);

            if (cfg->stop_on_eos && token == 0) {
                result->stopped_on_eos = true;
                free(draft_logits);
                goto mtp_done;
            }
        }

        /* If all draft tokens were rejected, still need to advance.
         * The target token was already sampled above. */
        last_hidden = oc_inf_model_last_output_hidden(model);
    }

mtp_done:
    free(draft_logits);

    double total_decode = get_time_sec() - decode_start;
    result->eval_time_sec = total_decode + result->prefill_time_sec;
    result->tokens_per_sec = (double)result->n_tokens / total_decode;

    return OC_OK;
}
