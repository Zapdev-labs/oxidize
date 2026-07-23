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
    (void)token;
    (void)sigma;
    memset(logits, 0, model->config.vocab_size * sizeof(float));
    return OC_OK;
}

OcError oc_diff_gemma_sample(OcDiffGemmaModel *model, float *logits, size_t n, uint32_t step)
{
    if (!model || !logits) return OC_ERR_INVALID_ARG;
    if (!model->initialized) return OC_ERR_MODEL;
    (void)n;
    (void)step;
    /* Stub: no sampling. */
    return OC_OK;
}

OcError oc_diff_gemma_denoise(OcDiffGemmaModel *model, float *tokens, size_t n,
                             float sigma_from, float sigma_to)
{
    if (!model || !tokens) return OC_ERR_INVALID_ARG;
    if (!model->initialized) return OC_ERR_MODEL;
    (void)n;
    (void)sigma_from;
    (void)sigma_to;
    /* Stub: no denoising. */
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
