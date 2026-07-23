/*
 * qwen_arch.c — Qwen architecture forward pass implementation.
 */
#include "oxidize/qwen_arch.h"

#include <stdlib.h>
#include <string.h>

OcError oc_qwen_config_init(OcQwenConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    /* Qwen2.5-7B defaults. */
    cfg->n_layers = 28;
    cfg->n_heads = 28;
    cfg->n_kv_heads = 4;
    cfg->head_dim = 128;
    cfg->hidden_dim = 3584;
    cfg->intermediate_dim = 18944;
    cfg->vocab_size = 152064;
    cfg->rope_theta = 1000000.0f;
    cfg->max_position = 32768;
    cfg->tie_word_embeddings = false;
    cfg->use_qk_norm = false;
    cfg->norm_eps = 1e-6f;
    return OC_OK;
}

OcError oc_qwen_config_qwen25_7b(OcQwenConfig *cfg)
{
    return oc_qwen_config_init(cfg);
}

OcError oc_qwen_config_qwen3_06b(OcQwenConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_layers = 28;
    cfg->n_heads = 16;
    cfg->n_kv_heads = 8;
    cfg->head_dim = 128;
    cfg->hidden_dim = 1024;
    cfg->intermediate_dim = 3072;
    cfg->vocab_size = 151936;
    cfg->rope_theta = 1000000.0f;
    cfg->max_position = 40960;
    cfg->tie_word_embeddings = true;
    cfg->use_qk_norm = true;
    cfg->norm_eps = 1e-6f;
    return OC_OK;
}

OcError oc_qwen_model_init(OcQwenModel *model, const OcQwenConfig *cfg)
{
    if (!model) return OC_ERR_INVALID_ARG;

    OcQwenConfig defaults;
    if (!cfg) {
        oc_qwen_config_init(&defaults);
        cfg = &defaults;
    }

    /* Validate config. */
    if (cfg->n_layers == 0 || cfg->hidden_dim == 0 ||
        cfg->vocab_size == 0 || cfg->n_heads == 0 || cfg->head_dim == 0)
        return OC_ERR_INVALID_ARG;
    if (cfg->hidden_dim % cfg->n_heads != 0)
        return OC_ERR_INVALID_ARG;

    memset(model, 0, sizeof(*model));
    model->config = *cfg;

    /* Allocate layers array. */
    model->layers = calloc(cfg->n_layers, sizeof(OcQwenLayer));
    if (!model->layers) goto fail;

    /* Allocate token embedding. */
    size_t emb_size = (size_t)cfg->vocab_size * cfg->hidden_dim;
    model->tok_emb = calloc(emb_size, sizeof(float));
    if (!model->tok_emb) goto fail;

    /* Allocate output norm. */
    model->output_norm = calloc(cfg->hidden_dim, sizeof(float));
    if (!model->output_norm) goto fail;

    /* Allocate output (unless tied). */
    if (!cfg->tie_word_embeddings) {
        model->output = calloc(emb_size, sizeof(float));
        if (!model->output) goto fail;
    }

    model->initialized = true;
    return OC_OK;

fail:
    oc_qwen_free(model);
    return OC_ERR_OOM;
}

OcError oc_qwen_forward(OcQwenModel *model, uint32_t token, float *logits)
{
    (void)token;
    if (!model || !logits) return OC_ERR_INVALID_ARG;
    if (!model->initialized) return OC_ERR_MODEL;

    /* Stub: zero-fill logits. */
    memset(logits, 0, model->config.vocab_size * sizeof(float));
    return OC_OK;
}

void oc_qwen_free(OcQwenModel *model)
{
    if (!model) return;
    free(model->layers);
    free(model->tok_emb);
    free(model->output_norm);
    free(model->output);
    memset(model, 0, sizeof(*model));
}
