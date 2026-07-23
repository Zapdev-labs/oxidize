#define _POSIX_C_SOURCE 200809L
#include "oxidize/mtp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void oc_mtp_config_init(OcMtpConfig *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->hidden_size = 4096;
    cfg->vocab_size  = 32000;
    cfg->n_layers    = 1;
    cfg->max_tokens  = 4;
    cfg->quantspec_draft_kv = false;
    cfg->rms_norm_eps = 1e-5f;
}

OcError oc_mtp_engine_init(OcMtpEngine *engine, const OcMtpConfig *cfg)
{
    if (!engine || !cfg) return OC_ERR_INVALID_ARG;
    if (cfg->hidden_size == 0 || cfg->vocab_size == 0)
        return OC_ERR_INVALID_ARG;
    if (cfg->max_tokens > OC_MTP_MAX_TOKENS)
        return OC_ERR_INVALID_ARG;

    memset(engine, 0, sizeof(*engine));
    engine->config = *cfg;
    engine->has_mtp = true;

    engine->draft_tokens = malloc(OC_MTP_MAX_TOKENS * sizeof(uint32_t));
    engine->draft_logits = malloc((size_t)OC_MTP_MAX_TOKENS * cfg->vocab_size * sizeof(float));
    engine->hidden_buf   = malloc(OC_MTP_MAX_HIDDEN * sizeof(float));

    if (!engine->draft_tokens || !engine->draft_logits || !engine->hidden_buf) {
        oc_mtp_engine_free(engine);
        return OC_ERR_OOM;
    }
    engine->n_draft = 0;
    return OC_OK;
}

void oc_mtp_engine_free(OcMtpEngine *engine)
{
    if (!engine) return;
    free(engine->draft_tokens);
    free(engine->draft_logits);
    free(engine->hidden_buf);
    memset(engine, 0, sizeof(*engine));
}

static float rms_norm(const float *x, const float *weight, float eps, float *out, size_t n)
{
    if (!x || !out || n == 0) return 0.0f;
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++)
        ss += x[i] * x[i];
    float rms = sqrtf(ss / (float)n + eps);
    float inv = 1.0f / rms;
    for (size_t i = 0; i < n; i++) {
        float w = weight ? weight[i] : 1.0f;
        out[i] = x[i] * inv * w;
    }
    return rms;
}

static uint32_t argmax(const float *logits, size_t n)
{
    if (!logits || n == 0) return 0;
    uint32_t best = 0;
    float best_v = logits[0];
    for (size_t i = 1; i < n; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = (uint32_t)i;
        }
    }
    return best;
}

static void simple_logits(float *logits, size_t vocab, uint32_t token, float (*rng)(void))
{
    if (!logits || vocab == 0) return;
    for (size_t i = 0; i < vocab; i++)
        logits[i] = -1e9f;
    /* Place some probability mass around token+1 (simulate next-token pred). */
    for (size_t i = 0; i < vocab && i < 10; i++) {
        uint32_t t = token + (uint32_t)i;
        if (t < vocab) {
            float r = rng ? rng() : 0.5f;
            logits[t] = 10.0f - (float)i * 0.5f + r * 0.1f;
        }
    }
}

OcError oc_mtp_draft(OcMtpEngine *engine,
                     uint32_t start_token,
                     const float *start_hidden,
                     size_t hidden_len,
                     size_t max_tokens,
                     float (*rng)(void))
{
    if (!engine || !start_hidden) return OC_ERR_INVALID_ARG;
    if (!engine->has_mtp) return OC_ERR_MODEL;
    if (hidden_len != engine->config.hidden_size) return OC_ERR_INVALID_ARG;
    if (max_tokens > engine->config.max_tokens) max_tokens = engine->config.max_tokens;
    if (max_tokens == 0) { engine->n_draft = 0; return OC_OK; }

    size_t h = engine->config.hidden_size;
    size_t v = engine->config.vocab_size;
    float *hidden = engine->hidden_buf;
    memcpy(hidden, start_hidden, h * sizeof(float));

    engine->n_draft = 0;
    uint32_t cur_tok = start_token;

    for (size_t step = 0; step < max_tokens; step++) {
        /* Norm hidden state. */
        float normed[OC_MTP_MAX_HIDDEN];
        rms_norm(hidden, NULL, engine->config.rms_norm_eps, normed, h);

        /* Compute logits (stub: simulate next-token prediction). */
        float *logits = engine->draft_logits + step * v;
        simple_logits(logits, v, cur_tok, rng);

        /* Argmax sample. */
        cur_tok = argmax(logits, v);
        engine->draft_tokens[step] = cur_tok;
        engine->n_draft++;

        /* Update hidden: simple decay (stub for actual MTP forward). */
        for (size_t i = 0; i < h; i++)
            hidden[i] = hidden[i] * 0.99f + normed[i] * 0.01f;
    }
    return OC_OK;
}

size_t oc_mtp_n_draft(const OcMtpEngine *engine)
{
    return engine ? engine->n_draft : 0;
}

OcError oc_mtp_get_draft_token(const OcMtpEngine *engine, size_t idx, uint32_t *out_token)
{
    if (!engine || !out_token) return OC_ERR_INVALID_ARG;
    if (idx >= engine->n_draft) return OC_ERR_INVALID_ARG;
    *out_token = engine->draft_tokens[idx];
    return OC_OK;
}

OcError oc_mtp_get_draft_logits(const OcMtpEngine *engine, size_t idx, float **out_logits)
{
    if (!engine || !out_logits) return OC_ERR_INVALID_ARG;
    if (idx >= engine->n_draft) return OC_ERR_INVALID_ARG;
    *out_logits = engine->draft_logits + idx * engine->config.vocab_size;
    return OC_OK;
}

bool oc_mtp_has_block(const OcMtpEngine *engine)
{
    return engine ? engine->has_mtp : false;
}

void oc_mtp_reset(OcMtpEngine *engine)
{
    if (!engine) return;
    engine->n_draft = 0;
}

const char *oc_mtp_config_name(const OcMtpConfig *cfg)
{
    (void)cfg;
    return "mtp/nextn";
}
