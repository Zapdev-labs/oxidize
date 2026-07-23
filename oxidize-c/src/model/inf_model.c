#define _POSIX_C_SOURCE 200809L
#include "oxidize/inf_model.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "oxidize/activation.h"
#include "oxidize/weight_ops.h"

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

/* ─── Forward pass methods ────────────────────────────────────────────── */

void oc_inf_model_embed_token(OcInferenceModel *m, uint32_t token)
{
    if (!m) return;
    size_t h = m->config.hidden_size;
    float *x = m->workspace.x;
    if (!x || h == 0) return;

    memset(x, 0, h * sizeof(float));

    uint32_t token_idx = token;
    if (token_idx >= m->config.vocab_size && m->config.vocab_size > 0)
        token_idx = m->config.vocab_size - 1;

    oc_weight_storage_lookup_embedding(&m->tok_embeddings, h,
                                       m->config.vocab_size,
                                       token_idx, x);

    if (m->config.embedding_scale != 1.0f) {
        for (size_t i = 0; i < h; i++)
            x[i] *= m->config.embedding_scale;
    }
}

const float *oc_inf_model_hidden_state(const OcInferenceModel *m)
{
    if (!m) return NULL;
    return m->workspace.x;
}

size_t oc_inf_model_config_hidden_size(const OcInferenceModel *m)
{
    return m ? m->config.hidden_size : 0;
}

OcError oc_inf_model_set_hidden_state(OcInferenceModel *m,
                                       const float *hidden, size_t len)
{
    if (!m || !hidden) return OC_ERR_INVALID_ARG;
    size_t h = m->config.hidden_size;
    if (len != h) return OC_ERR_INVALID_ARG;
    memcpy(m->workspace.x, hidden, h * sizeof(float));
    return OC_OK;
}

OcError oc_inf_model_apply_final_norm(const OcInferenceModel *m,
                                       const float *hidden, float *out, size_t len)
{
    if (!m || !hidden || !out) return OC_ERR_INVALID_ARG;
    size_t h = m->config.hidden_size;
    if (len != h) return OC_ERR_INVALID_ARG;
    if (!m->norm_weight) return OC_ERR_MODEL;
    oc_rms_norm_f32(hidden, m->norm_weight, out, h, m->config.rms_norm_eps);
    return OC_OK;
}

const float *oc_inf_model_final_norm_weight(const OcInferenceModel *m)
{
    return m ? m->norm_weight : NULL;
}

bool oc_inf_model_has_mtp(const OcInferenceModel *m)
{
    if (!m) return false;
    return m->mtp != NULL && oc_mtp_weights_is_usable(m->mtp, &m->config);
}

size_t oc_inf_model_nextn_predict_layers(const OcInferenceModel *m)
{
    return m ? m->config.nextn_predict_layers : 0;
}

const float *oc_inf_model_last_output_hidden(const OcInferenceModel *m)
{
    return m ? m->last_output_hidden : NULL;
}

OcError oc_inf_model_set_eagle3_capture_layers(OcInferenceModel *m,
                                                  const size_t *layers, size_t n)
{
    if (!m) return OC_ERR_INVALID_ARG;
    if (!layers || n == 0) {
        free(m->eagle3_capture_layers);
        m->eagle3_capture_layers = NULL;
        m->eagle3_n_capture_layers = 0;
        if (m->eagle3_layer_hiddens) {
            for (size_t i = 0; i < m->eagle3_n_hiddens; i++)
                free(m->eagle3_layer_hiddens[i]);
            free(m->eagle3_layer_hiddens);
            m->eagle3_layer_hiddens = NULL;
        }
        m->eagle3_n_hiddens = 0;
        return OC_OK;
    }

    /* Free old. */
    free(m->eagle3_capture_layers);
    if (m->eagle3_layer_hiddens) {
        for (size_t i = 0; i < m->eagle3_n_hiddens; i++)
            free(m->eagle3_layer_hiddens[i]);
        free(m->eagle3_layer_hiddens);
    }

    m->eagle3_capture_layers = malloc(n * sizeof(size_t));
    if (!m->eagle3_capture_layers) return OC_ERR_OOM;
    memcpy(m->eagle3_capture_layers, layers, n * sizeof(size_t));
    m->eagle3_n_capture_layers = n;

    /* Allocate per-capture-layer hidden buffers (initially NULL). */
    m->eagle3_layer_hiddens = calloc(n, sizeof(float *));
    if (!m->eagle3_layer_hiddens) return OC_ERR_OOM;
    m->eagle3_n_hiddens = n;
    for (size_t i = 0; i < n; i++)
        m->eagle3_layer_hiddens[i] = NULL;

    return OC_OK;
}

OcError oc_inf_model_concat_eagle3_features(const OcInferenceModel *m,
                                             float *out, size_t out_len)
{
    if (!m || !out) return OC_ERR_INVALID_ARG;
    size_t h = m->config.hidden_size;
    size_t needed = m->eagle3_n_capture_layers * h;
    if (out_len < needed) return OC_ERR_INVALID_ARG;

    for (size_t i = 0; i < m->eagle3_n_capture_layers; i++) {
        float *hidden = m->eagle3_layer_hiddens[i];
        if (!hidden) return OC_ERR_MODEL;
        memcpy(out + i * h, hidden, h * sizeof(float));
    }
    return OC_OK;
}

OcError oc_inf_model_lm_head_logits_from_normed(const OcInferenceModel *m,
                                                  const float *normed, size_t normed_len,
                                                  float *logits, size_t logits_len)
{
    if (!m || !normed || !logits) return OC_ERR_INVALID_ARG;
    size_t h = m->config.hidden_size;
    if (normed_len != h) return OC_ERR_INVALID_ARG;
    if (logits_len != m->config.vocab_size) return OC_ERR_INVALID_ARG;

    memset(logits, 0, logits_len * sizeof(float));
    return oc_gemv_weight(&m->output_weight, m->config.vocab_size, h, normed, logits);
}

OcError oc_inf_model_final_head_from_workspace(OcInferenceModel *m,
                                                 float **out, size_t *out_len)
{
    if (!m || !out || !out_len) return OC_ERR_INVALID_ARG;
    size_t h = m->config.hidden_size;
    size_t vocab = m->config.vocab_size;
    float *normed = m->workspace.hidden_a;
    float *logits = m->workspace.logits;

    if (!normed || !logits || !m->norm_weight) return OC_ERR_MODEL;

    /* Final RMSNorm. */
    oc_rms_norm_f32(m->workspace.x, m->norm_weight, normed, h, m->config.rms_norm_eps);

    /* Save last output hidden. */
    if (m->last_output_hidden && m->last_output_hidden_len >= h)
        memcpy(m->last_output_hidden, normed, h * sizeof(float));

    /* lm_head GEMV. */
    memset(logits, 0, vocab * sizeof(float));
    OcError e = oc_gemv_weight(&m->output_weight, vocab, h, normed, logits);
    if (e != OC_OK) return e;

    *out = logits;
    *out_len = vocab;
    return OC_OK;
}

/* ─── Single-token forward pass ────────────────────────────────────────── */

OcError oc_inf_model_forward_token(OcInferenceModel *m, uint32_t token, size_t position)
{
    if (!m) return OC_ERR_INVALID_ARG;
    if (position >= m->config.context_size) return OC_ERR_INVALID_ARG;

    OcInferenceConfig *cfg = &m->config;
    size_t h = cfg->hidden_size;
    uint32_t n_heads = cfg->num_attention_heads;
    uint32_t kvh = cfg->num_key_value_heads;
    uint32_t head_dim = oc_inference_config_head_dim(cfg);
    uint32_t kv_head_dim = oc_inference_config_kv_head_dim(cfg);
    uint32_t rope_len = oc_inference_config_effective_rope_dim(cfg);
    float eps = cfg->rms_norm_eps;

    /* 1. Embed token into workspace.x. */
    oc_inf_model_embed_token(m, token);

    float *normed = m->workspace.hidden_a;
    float *q_vec = m->workspace.q_full;
    float *k_vec = m->workspace.k_vec;
    float *v_vec = m->workspace.v_vec;
    float *attn_out = m->workspace.attn_result;
    float *ffn_gate = m->workspace.intermediate_a;
    float *ffn_up = m->workspace.intermediate_b;
    float *ffn_down = m->workspace.hidden_b;

    /* 2. Process each layer. */
    for (size_t li = 0; li < m->n_layers && li < cfg->layer_count; li++) {
        OcLayerWeights *layer = &m->layers[li];
        float rope_theta = oc_inference_config_layer_rope_theta(cfg, (uint32_t)li);
        uint32_t layer_window = oc_inference_config_layer_sliding_window(cfg, (uint32_t)li);

        /* --- Attention block --- */
        if (oc_layer_weights_has_attention(layer)) {
            /* Attn RMSNorm. */
            if (layer->attn_norm)
                oc_rms_norm_f32(m->workspace.x, layer->attn_norm, normed, h, eps);
            else
                memcpy(normed, m->workspace.x, h * sizeof(float));

            /* Q projection. */
            uint32_t q_len = n_heads * head_dim;
            OcError e = oc_gemv_weight(&layer->attn_q, q_len, h, normed, q_vec);
            if (e != OC_OK) return e;
            if (layer->attn_q_bias)
                oc_add_repeating_bias(q_vec, q_len, layer->attn_q_bias, q_len);

            /* K projection. */
            uint32_t kv_len = kvh * kv_head_dim;
            if (!oc_weight_storage_is_empty(&layer->attn_k)) {
                e = oc_gemv_weight(&layer->attn_k, kv_len, h, normed, k_vec);
                if (e != OC_OK) return e;
                if (layer->attn_k_bias)
                    oc_add_repeating_bias(k_vec, kv_len, layer->attn_k_bias, kv_len);
            }

            /* V projection. */
            if (!oc_weight_storage_is_empty(&layer->attn_v)) {
                e = oc_gemv_weight(&layer->attn_v, kv_len, h, normed, v_vec);
                if (e != OC_OK) return e;
                if (layer->attn_v_bias)
                    oc_add_repeating_bias(v_vec, kv_len, layer->attn_v_bias, kv_len);
            }

            /* Per-head Q norm (Qwen). */
            if (layer->attn_q_norm) {
                for (uint32_t hd = 0; hd < n_heads; hd++) {
                    float *qh = q_vec + hd * head_dim;
                    float tmp[256];
                    if (head_dim <= 256) {
                        oc_rms_norm_f32(qh, layer->attn_q_norm, tmp, head_dim, eps);
                        memcpy(qh, tmp, head_dim * sizeof(float));
                    }
                }
            }

            /* Per-head K norm (Qwen). */
            if (layer->attn_k_norm) {
                for (uint32_t hd = 0; hd < kvh; hd++) {
                    float *kh = k_vec + hd * kv_head_dim;
                    float tmp[256];
                    if (kv_head_dim <= 256) {
                        oc_rms_norm_f32(kh, layer->attn_k_norm, tmp, kv_head_dim, eps);
                        memcpy(kh, tmp, kv_head_dim * sizeof(float));
                    }
                }
            }

            /* Apply RoPE to Q heads. */
            for (uint32_t hd = 0; hd < n_heads; hd++) {
                float *qh = q_vec + hd * head_dim;
                oc_inference_config_apply_rope_head(cfg, qh, qh, head_dim,
                                                     rope_len, (int64_t)position,
                                                     rope_theta);
            }

            /* Apply RoPE to K heads. */
            for (uint32_t hd = 0; hd < kvh; hd++) {
                float *kh = k_vec + hd * kv_head_dim;
                oc_inference_config_apply_rope_head(cfg, kh, kh, kv_head_dim,
                                                     rope_len, (int64_t)position,
                                                     rope_theta);
            }

            /* KV cache append. */
            int32_t kv_idx = -1;
            if (li < m->kv_layer_map_len)
                kv_idx = m->kv_layer_map[li];

            if (kv_idx >= 0 && kv_len > 0) {
                e = oc_kv_cache_append(&m->kv_cache, (uint32_t)kv_idx,
                                       k_vec, v_vec, 1);
                if (e != OC_OK) return e;
            }

            /* Attention: for each Q head, compute scaled dot-product attention
             * against the KV cache (with sliding window). */
            memset(attn_out, 0, q_len * sizeof(float));
            if (kv_idx >= 0 && kv_len > 0) {
                uint32_t seq_len = oc_kv_cache_n_tokens(&m->kv_cache);
                uint32_t eff_seq = seq_len;
                const float *key_cache = NULL;
                const float *val_cache = NULL;

                if (layer_window > 0 && seq_len > layer_window) {
                    uint32_t skip = seq_len - layer_window;
                    oc_kv_cache_get(&m->kv_cache, (uint32_t)kv_idx, skip,
                                    &key_cache, &val_cache);
                    eff_seq = layer_window;
                } else {
                    oc_kv_cache_get(&m->kv_cache, (uint32_t)kv_idx, 0,
                                    &key_cache, &val_cache);
                }

                if (key_cache && val_cache) {
                    /* GQA: each Q head attends against kv_head_dim slices.
                     * For simplicity, use SDPA per Q head with its matching
                     * KV head (repeating KV heads for GQA). */
                    uint32_t n_rep = n_heads / (kvh > 0 ? kvh : 1);
                    for (uint32_t hd = 0; hd < n_heads; hd++) {
                        uint32_t kv_hd = hd / n_rep;
                        const float *q_head = q_vec + hd * head_dim;
                        const float *k_head = key_cache + kv_hd * kv_head_dim;
                        const float *v_head = val_cache + kv_hd * kv_head_dim;
                        float *out_head = attn_out + hd * head_dim;
                        oc_scaled_dot_product_attention_f32(
                            q_head, k_head, v_head, eff_seq,
                            kv_len, out_head);
                    }
                }
            }

            /* Attn output projection. */
            if (!oc_weight_storage_is_empty(&layer->attn_output)) {
                float *proj = m->workspace.hidden_b;
                e = oc_gemv_weight(&layer->attn_output, h, q_len, attn_out, proj);
                if (e != OC_OK) return e;
                if (layer->attn_output_bias)
                    oc_add_repeating_bias(proj, h, layer->attn_output_bias, h);

                /* Gemma sandwich norm. */
                if (cfg->sandwich_norm && layer->post_attention_norm) {
                    oc_rms_norm_f32(proj, layer->post_attention_norm, attn_out, h, eps);
                    memcpy(proj, attn_out, h * sizeof(float));
                }

                /* Residual add. */
                for (size_t i = 0; i < h; i++)
                    m->workspace.x[i] += proj[i];
            }
        }

        /* --- FFN block --- */
        float *ffn_norm_weight;
        if (cfg->sandwich_norm)
            ffn_norm_weight = layer->ffn_norm;
        else if (layer->post_attention_norm)
            ffn_norm_weight = layer->post_attention_norm;
        else if (layer->ffn_norm)
            ffn_norm_weight = layer->ffn_norm;
        else
            ffn_norm_weight = NULL;

        if (oc_layer_weights_has_moe(layer)) {
            /* MoE FFN. */
            if (ffn_norm_weight) {
                oc_rms_norm_f32(m->workspace.x, ffn_norm_weight, normed, h, eps);
            } else {
                memcpy(normed, m->workspace.x, h * sizeof(float));
            }
            float ffn_out[4096];
            memset(ffn_out, 0, h * sizeof(float));
            if (h <= 4096) {
                float *gate_scratch = m->workspace.intermediate_c;
                OcError e = oc_moe_ffn_forward(
                    &layer->ffn_gate_inp, &layer->ffn_gate_exps,
                    &layer->ffn_up_exps, &layer->ffn_down_exps,
                    layer->ffn_exp_probs_b, cfg, normed, ffn_out,
                    ffn_gate, ffn_up, gate_scratch,
                    m->workspace.moe_router_logits,
                    (OcExpertScore *)m->workspace.moe_gate_all);
                if (e != OC_OK) return e;
                for (size_t i = 0; i < h; i++)
                    m->workspace.x[i] += ffn_out[i];
            }
        } else if (oc_layer_weights_has_dense_ffn(layer) && ffn_norm_weight) {
            /* Dense FFN. */
            oc_rms_norm_f32(m->workspace.x, ffn_norm_weight, normed, h, eps);

            size_t i_size = cfg->intermediate_size;
            OcError e = oc_gemv_weight(&layer->ffn_gate, i_size, h, normed, ffn_gate);
            if (e != OC_OK) return e;
            e = oc_gemv_weight(&layer->ffn_up, i_size, h, normed, ffn_up);
            if (e != OC_OK) return e;

            /* SwiGLU or GeGLU. */
            if (cfg->gelu_ffn)
                oc_geglu_inplace_f32(ffn_gate, ffn_up, i_size);
            else
                oc_swiglu_inplace_f32(ffn_gate, ffn_up, i_size);

            e = oc_gemv_weight(&layer->ffn_down, h, i_size, ffn_gate, ffn_down);
            if (e != OC_OK) return e;
            if (layer->ffn_down_bias)
                oc_add_repeating_bias(ffn_down, h, layer->ffn_down_bias, h);

            /* Gemma sandwich norm (post-FFN). */
            if (cfg->sandwich_norm && layer->post_ffn_norm) {
                float tmp[4096];
                if (h <= 4096) {
                    oc_rms_norm_f32(ffn_down, layer->post_ffn_norm, tmp, h, eps);
                    memcpy(ffn_down, tmp, h * sizeof(float));
                }
            }

            /* Residual add. */
            for (size_t i = 0; i < h; i++)
                m->workspace.x[i] += ffn_down[i];
        }

        /* EAGLE3 capture. */
        for (size_t ei = 0; ei < m->eagle3_n_capture_layers; ei++) {
            if (m->eagle3_capture_layers[ei] == li) {
                if (!m->eagle3_layer_hiddens[ei]) {
                    m->eagle3_layer_hiddens[ei] = malloc(h * sizeof(float));
                }
                if (m->eagle3_layer_hiddens[ei]) {
                    memcpy(m->eagle3_layer_hiddens[ei], m->workspace.x,
                           h * sizeof(float));
                }
                break;
            }
        }
    }

    return OC_OK;
}

OcError oc_inf_model_forward_token_logits(OcInferenceModel *m, uint32_t token,
                                            size_t position,
                                            float **logits, size_t *logits_len)
{
    if (!m || !logits || !logits_len) return OC_ERR_INVALID_ARG;
    OcError e = oc_inf_model_forward_token(m, token, position);
    if (e != OC_OK) return e;
    return oc_inf_model_final_head_from_workspace(m, logits, logits_len);
}
