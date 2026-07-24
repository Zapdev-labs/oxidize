/*
 * diffusion_gemma.c — Gemma diffusion model implementation.
 */
#include "oxidize/diffusion_gemma.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

OcError oc_diff_gemma_config_init(OcDiffGemmaConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_layers = 18;
    cfg->hidden_dim = 2048;
    cfg->vocab_size = 256000;
    cfg->n_diffusion_steps = 50;
    cfg->sigma_min = 0.002f;
    cfg->sigma_max = 80.0f;
    cfg->rho = 7.0f;
    cfg->sampler = OC_DIFF_GEMMA_FLOW_MATCH;
    cfg->use_classifier_free_guidance = false;
    cfg->guidance_scale = 3.5f;
    return OC_OK;
}

OcError oc_diff_gemma_model_init(OcDiffGemmaModel *model, const OcDiffGemmaConfig *cfg)
{
    if (!model) return OC_ERR_INVALID_ARG;
    OcDiffGemmaConfig defaults;
    if (!cfg) {
        oc_diff_gemma_config_init(&defaults);
        cfg = &defaults;
    }
    if (cfg->n_layers == 0 || cfg->hidden_dim == 0 || cfg->vocab_size == 0)
        return OC_ERR_INVALID_ARG;

    memset(model, 0, sizeof(*model));
    model->config = *cfg;

    model->embedding = calloc((size_t)cfg->vocab_size * cfg->hidden_dim, sizeof(float));
    if (!model->embedding) return OC_ERR_OOM;

    model->output = calloc((size_t)cfg->vocab_size * cfg->hidden_dim, sizeof(float));
    if (!model->output) {
        free(model->embedding);
        return OC_ERR_OOM;
    }

    model->initialized = true;
    return OC_OK;
}

OcError oc_diff_gemma_forward(OcDiffGemmaModel *model, uint32_t token, float sigma, float *logits)
{
    if (!model || !logits) return OC_ERR_INVALID_ARG;
    if (!model->initialized) return OC_ERR_MODEL;

    OcDiffGemmaConfig *cfg = &model->config;
    size_t h = cfg->hidden_dim;
    size_t vocab = cfg->vocab_size;
    size_t tok_idx = (size_t)token;
    if (tok_idx >= vocab) tok_idx = vocab - 1;

    /* Embed token, scale by sigma (noise level conditioning). */
    float *hidden = malloc(h * sizeof(float));
    if (!hidden) return OC_ERR_OOM;

    if (model->embedding)
        memcpy(hidden, model->embedding + tok_idx * h, h * sizeof(float));
    else
        memset(hidden, 0, h * sizeof(float));

    /* Scale hidden by sigma (simple noise-level conditioning). */
    float sigma_val = (sigma > 0.0f) ? sigma : 1.0f;
    for (size_t i = 0; i < h; i++)
        hidden[i] *= sigma_val;

    /* RMSNorm + output projection (simplified: no transformer layers). */
    float ss = 0.0f;
    for (size_t i = 0; i < h; i++) ss += hidden[i] * hidden[i];
    float rms = 1.0f / sqrtf(ss / h + 1e-6f);
    for (size_t i = 0; i < h; i++)
        hidden[i] *= rms;

    if (model->output) {
        for (size_t r = 0; r < vocab; r++) {
            float dot = 0.0f;
            const float *orow = model->output + r * h;
            for (size_t c = 0; c < h; c++)
                dot += orow[c] * hidden[c];
            logits[r] = dot;
        }
    } else {
        memset(logits, 0, vocab * sizeof(float));
    }

    free(hidden);
    return OC_OK;
}

OcError oc_diff_gemma_sample(OcDiffGemmaModel *model, float *logits, size_t n, uint32_t step)
{
    if (!model || !logits) return OC_ERR_INVALID_ARG;
    if (!model->initialized) return OC_ERR_MODEL;

    OcDiffGemmaConfig *cfg = &model->config;
    if (n == 0) return OC_OK;

    /* Temperature based on diffusion step: colder at later steps. */
    float progress = (cfg->n_diffusion_steps > 0)
        ? (float)step / (float)cfg->n_diffusion_steps : 0.5f;
    float temperature = 1.0f - progress * 0.9f; /* 1.0 → 0.1 */
    if (temperature < 0.01f) temperature = 0.01f;

    /* Softmax with temperature, then argmax. */
    float max_logit = -INFINITY;
    for (size_t i = 0; i < n; i++)
        if (logits[i] > max_logit) max_logit = logits[i];

    float sum_exp = 0.0f;
    for (size_t i = 0; i < n; i++) {
        logits[i] = expf((logits[i] - max_logit) / temperature);
        sum_exp += logits[i];
    }

    if (sum_exp > 0.0f) {
        float inv = 1.0f / sum_exp;
        /* Argmax (greedy sampling). */
        size_t best = 0;
        float best_prob = 0.0f;
        for (size_t i = 0; i < n; i++) {
            logits[i] *= inv;
            if (logits[i] > best_prob) {
                best_prob = logits[i];
                best = i;
            }
        }
        /* Set logits to one-hot for the argmax. */
        memset(logits, 0, n * sizeof(float));
        logits[best] = 1.0f;
    }

    return OC_OK;
}

OcError oc_diff_gemma_denoise(OcDiffGemmaModel *model, float *tokens, size_t n,
                             float sigma_from, float sigma_to)
{
    if (!model || !tokens) return OC_ERR_INVALID_ARG;
    if (!model->initialized) return OC_ERR_MODEL;

    /* Euler denoising step: tokens = tokens + (sigma_to - sigma_from) * velocity.
     * In a real diffusion model, velocity comes from the model forward pass.
     * Here we implement the Euler step structure. */
    float delta = sigma_to - sigma_from;

    if (delta == 0.0f || n == 0) return OC_OK;

    /* Compute velocity = forward(tokens, sigma_from) - tokens (simplified). */
    float *logits = malloc(model->config.vocab_size * sizeof(float));
    if (!logits) return OC_ERR_OOM;

    for (size_t i = 0; i < n; i++) {
        uint32_t tok = (uint32_t)tokens[i];
        if (tok >= model->config.vocab_size) tok = 0;

        OcError e = oc_diff_gemma_forward(model, tok, sigma_from, logits);
        if (e != OC_OK) { free(logits); return e; }

        /* Sample to get denoised token. */
        e = oc_diff_gemma_sample(model, logits, model->config.vocab_size, 0);
        if (e != OC_OK) { free(logits); return e; }

        /* Find the argmax (one-hot in logits). */
        for (size_t j = 0; j < model->config.vocab_size; j++) {
            if (logits[j] > 0.5f) {
                /* Euler step: blend old and new token. */
                if (delta < 0.0f) {
                    /* Denoising: move toward new token. */
                    tokens[i] = (float)j;
                }
                break;
            }
        }
    }

    free(logits);
    return OC_OK;
}

const char *oc_diff_gemma_sampler_name(OcDiffGemmaSampler s)
{
    switch (s) {
    case OC_DIFF_GEMMA_DDIM:       return "ddim";
    case OC_DIFF_GEMMA_DPM2M:     return "dpm2m";
    case OC_DIFF_GEMMA_EULER:      return "euler";
    case OC_DIFF_GEMMA_FLOW_MATCH: return "flow_match";
    default: return "unknown";
    }
}

void oc_diff_gemma_free(OcDiffGemmaModel *model)
{
    if (!model) return;
    free(model->embedding);
    free(model->output);
    memset(model, 0, sizeof(*model));
}
