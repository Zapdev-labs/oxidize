/*
 * mistral_arch.c — Mistral architecture forward pass implementation.
 *
 * Stub forward pass: allocates weight storage on model init, fills the
 * logits buffer with zeros on forward, and frees storage on free. The
 * structural scaffolding (config, layer array, weight pointers) mirrors
 * what the real SwiGLU + RoPE + GQA + SWA forward pass will need once
 * weights are loaded from GGUF.
 */
#include "oxidize/mistral_arch.h"

#include <stdlib.h>
#include <string.h>

OcError oc_mistral_config_init(OcMistralConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_layers         = 32;
    cfg->n_heads          = 32;
    cfg->n_kv_heads       = 8;
    cfg->head_dim         = 128;
    cfg->hidden_dim       = 4096;
    cfg->intermediate_dim = 14336;
    cfg->vocab_size       = 32000;
    cfg->sliding_window   = 4096;
    cfg->rope_theta       = 10000.0f;
    cfg->max_position     = 32768;
    return OC_OK;
}

OcError oc_mistral_model_init(OcMistralModel *model, const OcMistralConfig *cfg)
{
    if (!model) return OC_ERR_INVALID_ARG;
    memset(model, 0, sizeof(*model));
    if (cfg) {
        model->config = *cfg;
    } else {
        oc_mistral_config_init(&model->config);
    }

    /* Basic sanity guards so the allocations below do not overflow. */
    if (model->config.n_layers == 0 || model->config.vocab_size == 0 ||
        model->config.hidden_dim == 0) {
        return OC_ERR_INVALID_ARG;
    }

    model->layers = calloc(model->config.n_layers, sizeof(OcMistralLayer));
    if (!model->layers) return OC_ERR_OOM;

    /* Allocate and zero the global tensors. */
    size_t tok_emb_size = (size_t)model->config.vocab_size *
                          (size_t)model->config.hidden_dim;
    model->tok_emb = calloc(tok_emb_size, sizeof(float));
    if (!model->tok_emb) {
        free(model->layers);
        model->layers = NULL;
        return OC_ERR_OOM;
    }

    model->output_norm = calloc(model->config.hidden_dim, sizeof(float));
    if (!model->output_norm) {
        free(model->tok_emb);
        free(model->layers);
        model->layers = NULL;
        model->tok_emb = NULL;
        return OC_ERR_OOM;
    }

    model->output = calloc(tok_emb_size, sizeof(float));
    if (!model->output) {
        free(model->output_norm);
        free(model->tok_emb);
        free(model->layers);
        model->layers = NULL;
        model->tok_emb = NULL;
        model->output_norm = NULL;
        return OC_ERR_OOM;
    }

    model->initialized = true;
    return OC_OK;
}

OcError oc_mistral_forward(OcMistralModel *model, uint32_t token, float *logits)
{
    if (!model || !model->initialized || !logits) return OC_ERR_INVALID_ARG;
    if (model->config.vocab_size == 0) return OC_ERR_MODEL;
    /* Stub: zero logits. Real path: embed(token) → layers (SwiGLU FFN,
     * RoPE on Q/K, GQA, SWA) → RMSNorm → output projection. */
    (void)token;
    memset(logits, 0, model->config.vocab_size * sizeof(float));
    return OC_OK;
}

void oc_mistral_free(OcMistralModel *model)
{
    if (!model) return;
    free(model->layers);
    free(model->tok_emb);
    free(model->output_norm);
    free(model->output);
    memset(model, 0, sizeof(*model));
}
