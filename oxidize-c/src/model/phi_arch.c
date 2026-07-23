/*
 * phi_arch.c — Phi-2 / Phi-3 architecture forward pass implementation.
 *
 * Stub forward pass: allocates weight storage on model init, fills the
 * logits buffer with zeros on forward, and frees storage on free. The
 * structural scaffolding (config, layer array, weight pointers) mirrors
 * what the real GeGLU + RoPE + dense-MHA forward pass will need once
 * weights are loaded from GGUF.
 */
#include "oxidize/phi_arch.h"

#include <stdlib.h>
#include <string.h>

OcError oc_phi_config_init(OcPhiConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_layers         = 24;
    cfg->n_heads          = 32;
    cfg->head_dim         = 80;
    cfg->hidden_dim       = 2560;
    cfg->intermediate_dim = 10240;
    cfg->vocab_size       = 51200;
    cfg->rope_theta       = 10000.0f;
    return OC_OK;
}

OcError oc_phi_model_init(OcPhiModel *model, const OcPhiConfig *cfg)
{
    if (!model) return OC_ERR_INVALID_ARG;
    memset(model, 0, sizeof(*model));
    if (cfg) {
        model->config = *cfg;
    } else {
        oc_phi_config_init(&model->config);
    }

    if (model->config.n_layers == 0 || model->config.vocab_size == 0 ||
        model->config.hidden_dim == 0) {
        return OC_ERR_INVALID_ARG;
    }

    model->layers = calloc(model->config.n_layers, sizeof(OcPhiLayer));
    if (!model->layers) return OC_ERR_OOM;

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

OcError oc_phi_forward(OcPhiModel *model, uint32_t token, float *logits)
{
    if (!model || !model->initialized || !logits) return OC_ERR_INVALID_ARG;
    if (model->config.vocab_size == 0) return OC_ERR_MODEL;
    /* Stub: zero logits. Real path: embed(token) → layers (GeGLU FFN,
     * RoPE on Q/K, dense multi-head attention) → norm → output proj. */
    (void)token;
    memset(logits, 0, model->config.vocab_size * sizeof(float));
    return OC_OK;
}

void oc_phi_free(OcPhiModel *model)
{
    if (!model) return;
    free(model->layers);
    free(model->tok_emb);
    free(model->output_norm);
    free(model->output);
    memset(model, 0, sizeof(*model));
}
