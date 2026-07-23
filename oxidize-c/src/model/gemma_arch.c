/*
 * gemma_arch.c — Gemma architecture forward pass implementation.
 *
 * Stub forward pass: allocates weight storage on model init, fills the
 * logits buffer with zeros on forward, and frees storage on free. The
 * structural scaffolding (config, layer array, weight pointers) mirrors
 * what the real GeGLU + RoPE + GQA + embedding-scale forward pass will
 * need once weights are loaded from GGUF.
 */
#include "oxidize/gemma_arch.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

OcError oc_gemma_config_init(OcGemmaConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_layers         = 18;
    cfg->n_heads          = 8;
    cfg->n_kv_heads       = 1;
    cfg->head_dim         = 256;
    cfg->hidden_dim       = 2048;
    cfg->intermediate_dim = 16384;
    cfg->vocab_size       = 256000;
    cfg->embedding_scale  = sqrtf((float)cfg->hidden_dim);
    cfg->rope_theta       = 10000.0f;
    return OC_OK;
}

OcError oc_gemma_model_init(OcGemmaModel *model, const OcGemmaConfig *cfg)
{
    if (!model) return OC_ERR_INVALID_ARG;
    memset(model, 0, sizeof(*model));
    if (cfg) {
        model->config = *cfg;
    } else {
        oc_gemma_config_init(&model->config);
    }

    if (model->config.n_layers == 0 || model->config.vocab_size == 0 ||
        model->config.hidden_dim == 0) {
        return OC_ERR_INVALID_ARG;
    }

    /* If embedding_scale was not set (caller passed a partial config),
     * default to sqrt(hidden_dim). */
    if (model->config.embedding_scale <= 0.0f) {
        model->config.embedding_scale = sqrtf((float)model->config.hidden_dim);
    }

    model->layers = calloc(model->config.n_layers, sizeof(OcGemmaLayer));
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

OcError oc_gemma_forward(OcGemmaModel *model, uint32_t token, float *logits)
{
    if (!model || !model->initialized || !logits) return OC_ERR_INVALID_ARG;
    if (model->config.vocab_size == 0) return OC_ERR_MODEL;
    /* Stub: zero logits. Real path multiplies the embedding lookup by
     * config.embedding_scale before the first layer. */
    (void)token;
    memset(logits, 0, model->config.vocab_size * sizeof(float));
    return OC_OK;
}

void oc_gemma_free(OcGemmaModel *model)
{
    if (!model) return;
    free(model->layers);
    free(model->tok_emb);
    free(model->output_norm);
    free(model->output);
    memset(model, 0, sizeof(*model));
}
