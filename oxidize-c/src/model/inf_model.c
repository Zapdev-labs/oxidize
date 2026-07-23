#define _POSIX_C_SOURCE 200809L
#include "oxidize/inf_model.h"

#include <stdlib.h>
#include <string.h>

OcError oc_inf_model_init(OcInferenceModel *m, const OcInferenceConfig *cfg)
{
    if (!m || !cfg) return OC_ERR_INVALID_ARG;

    memset(m, 0, sizeof(*m));
    m->config = *cfg;

    /* Allocate workspace. */
    OcError e = oc_workspace_for_config(&m->workspace, cfg);
    if (e != OC_OK) goto fail;

    /* Allocate layers array. */
    m->layers_cap = cfg->layer_count > 0 ? cfg->layer_count : 32;
    m->layers = calloc(m->layers_cap, sizeof(OcLayerWeights));
    if (!m->layers) { e = OC_ERR_OOM; goto fail; }
    m->n_layers = 0;

    /* Initialize weight storage. */
    oc_weight_storage_init(&m->tok_embeddings);
    oc_weight_storage_init(&m->output_weight);

    /* Initialize KV cache. */
    OcKvCacheConfig kv_cfg;
    oc_kv_cache_config_init(&kv_cfg);
    kv_cfg.n_layers = cfg->layer_count;
    kv_cfg.n_heads = cfg->num_key_value_heads;
    kv_cfg.head_dim = oc_inference_config_kv_head_dim(cfg);
    kv_cfg.max_seq_len = cfg->context_size;
    e = oc_kv_cache_init(&m->kv_cache, &kv_cfg);
    if (e != OC_OK) goto fail;

    /* KV layer map (all layers are attention layers by default). */
    m->kv_layer_map_len = cfg->layer_count;
    m->kv_layer_map = malloc(cfg->layer_count * sizeof(int32_t));
    if (!m->kv_layer_map) { e = OC_ERR_OOM; goto fail; }
    for (size_t i = 0; i < cfg->layer_count; i++)
        m->kv_layer_map[i] = (int32_t)i;

    /* SSM engine (only for Mamba/LFM2 architectures). */
    /* For standard transformer, n_layers=0 SSM state is fine. */
    m->ssm_engine.ssm_states = NULL;
    m->ssm_engine.conv_buffers = NULL;

    /* Last output hidden. */
    m->last_output_hidden = calloc(cfg->hidden_size, sizeof(float));
    m->last_output_hidden_len = cfg->hidden_size;
    if (!m->last_output_hidden) { e = OC_ERR_OOM; goto fail; }

    /* Initialize MTP to NULL. */
    m->mtp = NULL;

    m->loaded = false;
    return OC_OK;

fail:
    oc_inf_model_free(m);
    return e;
}

void oc_inf_model_free(OcInferenceModel *m)
{
    if (!m) return;

    /* Free layers. */
    if (m->layers) {
        for (size_t i = 0; i < m->n_layers; i++)
            oc_layer_weights_free(&m->layers[i]);
        free(m->layers);
    }

    /* Free MTP. */
    if (m->mtp) {
        oc_mtp_weights_free(m->mtp);
        free(m->mtp);
    }

    /* Free weight storage. */
    oc_weight_storage_free(&m->tok_embeddings);
    oc_weight_storage_free(&m->output_weight);
    free(m->norm_weight);

    /* Free KV cache. */
    oc_kv_cache_free(&m->kv_cache);

    /* Free KV layer map. */
    free(m->kv_layer_map);

    /* Free SSM engine. */
    oc_ssm_engine_free(&m->ssm_engine);

    /* Free workspace. */
    oc_workspace_free(&m->workspace);

    /* Free last output hidden. */
    free(m->last_output_hidden);

    /* Free EAGLE3 capture. */
    free(m->eagle3_capture_layers);
    if (m->eagle3_layer_hiddens) {
        for (size_t i = 0; i < m->eagle3_n_hiddens; i++)
            free(m->eagle3_layer_hiddens[i]);
        free(m->eagle3_layer_hiddens);
    }

    memset(m, 0, sizeof(*m));
}

OcError oc_inf_model_add_layer(OcInferenceModel *m, OcLayerWeights *layer)
{
    if (!m || !layer) return OC_ERR_INVALID_ARG;

    if (m->n_layers >= m->layers_cap) {
        size_t new_cap = m->layers_cap * 2;
        OcLayerWeights *nl = realloc(m->layers, new_cap * sizeof(OcLayerWeights));
        if (!nl) return OC_ERR_OOM;
        m->layers = nl;
        m->layers_cap = new_cap;
    }

    m->layers[m->n_layers] = *layer;
    m->n_layers++;
    return OC_OK;
}

OcError oc_inf_model_set_mtp(OcInferenceModel *m, OcMtpWeights *mtp)
{
    if (!m) return OC_ERR_INVALID_ARG;
    if (m->mtp) {
        oc_mtp_weights_free(m->mtp);
        free(m->mtp);
    }
    m->mtp = mtp;
    return OC_OK;
}

const OcInferenceConfig *oc_inf_model_config(const OcInferenceModel *m)
{
    return m ? &m->config : NULL;
}

size_t oc_inf_model_kv_layer_count(const OcInferenceModel *m)
{
    if (!m) return 0;
    return m->kv_cache.config.n_layers;
}

size_t oc_inf_model_kv_row_len(const OcInferenceModel *m)
{
    if (!m) return 0;
    return m->kv_cache.config.n_heads * m->kv_cache.config.head_dim;
}

bool oc_inf_model_is_loaded(const OcInferenceModel *m)
{
    return m ? m->loaded : false;
}

bool oc_inf_model_batched_decode_enabled(void)
{
    /* Check OX_BATCHED_DECODE env var. */
    const char *env = getenv("OX_BATCHED_DECODE");
    if (env && (env[0] == '1' || env[0] == 't' || env[0] == 'T'))
        return true;
    return false;
}
