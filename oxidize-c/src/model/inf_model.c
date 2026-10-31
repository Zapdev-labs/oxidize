#define _POSIX_C_SOURCE 200809L
#include "oxidize/inf_model.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "oxidize/activation.h"
#include "oxidize/weight_ops.h"
#include "oxidize/rope_scaling.h"

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
    return (size_t)m->kv_cache.config.n_heads * m->kv_cache.config.head_dim;
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

/* ─── Attention head dimension helpers ─────────────────────────────────── */

void oc_attention_head_dims(const OcInferenceConfig *cfg,
                             const OcLayerWeights *layer,
                             size_t q_len, size_t kv_len,
                             uint32_t *out_q_head_dim,
                             uint32_t *out_q_heads,
                             uint32_t *out_kv_head_dim,
                             uint32_t *out_kv_heads)
{
    /* q_head_dim: use attn_q_norm len if it divides q_len evenly,
     * else q_len / num_attention_heads, else q_len. */
    uint32_t q_hd;
    if (layer && layer->attn_q_norm && layer->n_q_norm > 0 &&
        q_len > 0 && (q_len % layer->n_q_norm) == 0) {
        q_hd = (uint32_t)layer->n_q_norm;
    } else if (cfg->num_attention_heads > 0 && q_len > 0 &&
               (q_len % cfg->num_attention_heads) == 0) {
        q_hd = (uint32_t)(q_len / cfg->num_attention_heads);
    } else {
        q_hd = (uint32_t)q_len;
    }

    uint32_t q_h = q_hd > 0 ? (uint32_t)(q_len / q_hd) : 1;

    uint32_t kv_hd;
    if (layer && layer->attn_k_norm && layer->n_k_norm > 0 &&
        kv_len > 0 && (kv_len % layer->n_k_norm) == 0) {
        kv_hd = (uint32_t)layer->n_k_norm;
    } else if (cfg->num_key_value_heads > 0 && kv_len > 0 &&
               (kv_len % cfg->num_key_value_heads) == 0) {
        kv_hd = (uint32_t)(kv_len / cfg->num_key_value_heads);
    } else if (kv_len > 0) {
        kv_hd = (uint32_t)kv_len;
    } else {
        kv_hd = q_hd;
    }

    uint32_t kv_h = kv_hd > 0 ? (uint32_t)(kv_len / kv_hd) : 1;

    if (out_q_head_dim) *out_q_head_dim = q_hd;
    if (out_q_heads) *out_q_heads = q_h;
    if (out_kv_head_dim) *out_kv_head_dim = kv_hd;
    if (out_kv_heads) *out_kv_heads = kv_h;
}

OcError oc_gemv_weight_head(const OcWeightStorage *ws,
                             size_t rows, size_t cols,
                             uint32_t head, uint32_t n_heads,
                             const float *input, float *output)
{
    if (!ws || !input || !output || n_heads == 0)
        return OC_ERR_INVALID_ARG;

    /* For F32: per-head slice = data[head * rows*cols .. (head+1)*rows*cols]. */
    if (ws->type == OC_WEIGHT_F32) {
        size_t per_head = ws->f32_len / n_heads;
        size_t start = (size_t)head * per_head;
        if (start + rows * cols > ws->f32_len)
            return OC_ERR_INVALID_ARG;
        return oc_gemv_f32(ws->f32_data + start, rows, cols, input, output);
    }

    /* For quantized: dequantize only this head's rows, not the whole matrix.
     * Each head has `rows` rows of `cols` elements each. The per-head byte
     * offset is head * (rows * bytes_per_row). */
    size_t per_head_elems = rows * cols;
    float *f32_head = malloc(per_head_elems * sizeof(float));
    if (!f32_head) return OC_ERR_OOM;

    OcGgufQuantizationType qt = oc_weight_storage_qtype(ws);
    OcQuantBlockLayout bs = oc_quant_block_size(qt);
    if (bs.elements_per_block == 0 || bs.bytes_per_block == 0) {
        free(f32_head);
        return OC_ERR_QUANT;
    }

    /* Bytes per row = (cols / elements_per_block) * bytes_per_block. */
    if (cols % bs.elements_per_block != 0) {
        free(f32_head);
        return OC_ERR_INVALID_ARG;
    }
    size_t bytes_per_row = (cols / bs.elements_per_block) * bs.bytes_per_block;
    size_t per_head_bytes = rows * bytes_per_row;

    /* Get the raw data pointer and total size. */
    const uint8_t *raw = NULL;
    size_t raw_size = 0;
    if (ws->type == OC_WEIGHT_QUANTIZED) {
        raw = ws->quant_data;
        raw_size = ws->quant_size;
    } else if (ws->type == OC_WEIGHT_MMAP_QUANTIZED) {
        raw = ws->mmap_data + ws->mmap_offset;
        raw_size = ws->mmap_size;
    }

    if (!raw) {
        free(f32_head);
        return OC_ERR_INVALID_ARG;
    }

    /* Compute byte offset for this head. */
    size_t head_byte_offset = (size_t)head * per_head_bytes;
    if (head_byte_offset + per_head_bytes > raw_size) {
        free(f32_head);
        return OC_ERR_INVALID_ARG;
    }

    /* Dequantize only this head's rows. */
    OcError e = oc_quant_dequant_row(qt, raw + head_byte_offset,
                                     per_head_bytes, f32_head, per_head_elems);
    if (e != OC_OK) {
        free(f32_head);
        return e;
    }

    /* GEMV on the dequantized per-head data. */
    e = oc_gemv_f32(f32_head, rows, cols, input, output);
    free(f32_head);
    return e;
}

/* ─── MLA (DeepSeek2) layer forward ────────────────────────────────────── */

static OcError forward_mla_layer(OcInferenceModel *m, OcLayerWeights *layer,
                                  size_t li, size_t pos)
{
    OcInferenceConfig *cfg = &m->config;
    size_t h = cfg->hidden_size;
    uint32_t n_heads = cfg->num_attention_heads;
    float eps = cfg->rms_norm_eps;
    float *x = m->workspace.x;
    float *attn_out = m->workspace.attn_result;

    /* Compute dimensions. */
    size_t q_lora = oc_weight_storage_output_dim(&layer->mla_q_a, h);
    size_t q_len = oc_weight_storage_output_dim(&layer->mla_q_b, q_lora);
    size_t kv_out = oc_weight_storage_output_dim(&layer->mla_kv_a_mqa, h);
    size_t kv_lora = layer->n_mla_kv_a_norm;  /* kv_a_norm len */
    if (kv_lora == 0) kv_lora = kv_out / 2;   /* fallback */
    size_t kv_pe_dim = kv_out > kv_lora ? kv_out - kv_lora : 0;
    uint32_t kv_head_dim = oc_inference_config_kv_head_dim(cfg);
    size_t k_nope_dim = oc_weight_storage_output_dim(&layer->mla_k_b, kv_lora);
    if (n_heads > 0) k_nope_dim /= n_heads;
    size_t v_head_dim = oc_weight_storage_output_dim(&layer->mla_v_b, kv_lora);
    if (n_heads > 0) v_head_dim /= n_heads;
    size_t q_pe_dim = kv_head_dim > k_nope_dim ? kv_head_dim - k_nope_dim : 0;

    /* 1. RMSNorm. */
    float *normed = m->workspace.hidden_b;
    if (layer->attn_norm)
        oc_rms_norm_f32(x, layer->attn_norm, normed, h, eps);
    else
        memcpy(normed, x, h * sizeof(float));

    /* 2. Q low-rank projection: q_a -> q_a_norm -> q_b. */
    float *c_q = m->workspace.intermediate_a;  /* [q_lora] */
    memset(c_q, 0, q_lora * sizeof(float));
    OcError e = oc_gemv_weight(&layer->mla_q_a, q_lora, h, normed, c_q);
    if (e != OC_OK) return e;

    if (layer->mla_q_a_norm) {
        float *tmp = m->workspace.intermediate_b;
        memcpy(tmp, c_q, q_lora * sizeof(float));
        oc_rms_norm_f32(tmp, layer->mla_q_a_norm, c_q, q_lora, eps);
    }

    float *q_vec = m->workspace.q_full;  /* [q_len] */
    memset(q_vec, 0, q_len * sizeof(float));
    e = oc_gemv_weight(&layer->mla_q_b, q_len, q_lora, c_q, q_vec);
    if (e != OC_OK) return e;

    /* 3. KV low-rank projection: kv_a_mqa -> split into c_kv + k_pe. */
    float *kv_pe = m->workspace.intermediate_c;  /* [kv_out] */
    memset(kv_pe, 0, kv_out * sizeof(float));
    e = oc_gemv_weight(&layer->mla_kv_a_mqa, kv_out, h, normed, kv_pe);
    if (e != OC_OK) return e;

    float *c_kv = m->workspace.mamba_scratch;  /* [kv_lora] */
    if (c_kv && kv_lora <= (size_t)m->workspace.hidden_size) {
        memcpy(c_kv, kv_pe, kv_lora * sizeof(float));
        if (layer->mla_kv_a_norm) {
            float *tmp = m->workspace.intermediate_b;
            memcpy(tmp, c_kv, kv_lora * sizeof(float));
            oc_rms_norm_f32(tmp, layer->mla_kv_a_norm, c_kv, kv_lora, eps);
        }
    }

    /* 4. Apply RoPE to k_pe. */
    float *k_pe_rope = m->workspace.flash_q;  /* [kv_pe_dim] */
    if (kv_pe_dim > 0 && k_pe_rope) {
        float rope_amp = -1.0f;
        if (cfg->yarn_factor > 1.0f && cfg->yarn_mscale_all_dim > 0.0f) {
            oc_rope_deepseek_yarn_scales(cfg->yarn_factor, cfg->yarn_mscale,
                                          cfg->yarn_mscale_all_dim,
                                          (uint32_t)kv_head_dim, &rope_amp, NULL);
        }
        if (cfg->yarn_factor > 1.0f) {
            oc_apply_rope_yarn_scaled_f32(kv_pe + kv_lora, k_pe_rope,
                                          kv_pe_dim, kv_pe_dim, (int64_t)pos,
                                          cfg->rope_theta, cfg->yarn_factor,
                                          (uint32_t)cfg->yarn_orig_ctx, rope_amp);
        } else {
            oc_inference_config_apply_rope_head(cfg, kv_pe + kv_lora, k_pe_rope,
                                                 kv_pe_dim, kv_pe_dim,
                                                 (int64_t)pos, cfg->rope_theta);
        }
    }

    /* 5. Compute K and V per head from c_kv. */
    size_t total_k = (size_t)n_heads * kv_head_dim;
    size_t total_v = (size_t)n_heads * v_head_dim;
    float *k_store = m->workspace.k_vec;     /* [total_k] */
    float *v_store = m->workspace.v_vec;     /* [total_v] */
    if (k_store) memset(k_store, 0, total_k * sizeof(float));
    if (v_store) memset(v_store, 0, total_v * sizeof(float));

    for (uint32_t hd = 0; hd < n_heads; hd++) {
        /* K_nope = mla_k_b[head] @ c_kv. */
        size_t k_off = (size_t)hd * kv_head_dim;
        if (!oc_weight_storage_is_empty(&layer->mla_k_b) && k_store) {
            e = oc_gemv_weight_head(&layer->mla_k_b, k_nope_dim, kv_lora,
                                     hd, n_heads, c_kv, k_store + k_off);
            if (e != OC_OK) return e;
        }
        /* Copy k_pe_rope into the rope portion of K. */
        size_t rope_off = k_off + k_nope_dim;
        size_t copy = q_pe_dim < kv_pe_dim ? q_pe_dim : kv_pe_dim;
        if (copy > kv_head_dim - k_nope_dim) copy = kv_head_dim - k_nope_dim;
        if (k_pe_rope && k_store)
            memcpy(k_store + rope_off, k_pe_rope, copy * sizeof(float));

        /* V = mla_v_b[head] @ c_kv.
         *
         * v_b is laid out exactly like k_b: the converter splits HF's
         * kv_b_proj [n_heads*(nope+v), kv_lora] by taking rows
         * [h*256+128, h*256+256) for each head, so head h's block is
         * `v_head_dim` contiguous rows of `kv_lora` columns — head-major,
         * which is precisely what oc_gemv_weight_head expects.
         *
         * This used to hand-roll the GEMV with a [kv_lora, v_dim, n_heads]
         * index AND read through oc_weight_storage_f32_data(), which returns
         * NULL for OC_WEIGHT_MMAP_QUANTIZED. Since BF16 is a quantized type
         * here, that made V identically zero for every real model, not just
         * quantized ones — attention returned nothing but its own bias. */
        size_t v_off = (size_t)hd * v_head_dim;
        if (!oc_weight_storage_is_empty(&layer->mla_v_b) && v_store) {
            e = oc_gemv_weight_head(&layer->mla_v_b, v_head_dim, kv_lora,
                                     hd, n_heads, c_kv, v_store + v_off);
            if (e != OC_OK) return e;
        }

        /* Apply RoPE to Q's pe portion. */
        size_t q_off = (size_t)hd * kv_head_dim;
        if (q_pe_dim > 0 && q_off + k_nope_dim + q_pe_dim <= q_len) {
            float *q_pe = q_vec + q_off + k_nope_dim;
            float rope_amp = -1.0f;
            if (cfg->yarn_factor > 1.0f && cfg->yarn_mscale_all_dim > 0.0f) {
                oc_rope_deepseek_yarn_scales(cfg->yarn_factor, cfg->yarn_mscale,
                                              cfg->yarn_mscale_all_dim,
                                              (uint32_t)kv_head_dim, &rope_amp, NULL);
            }
            if (cfg->yarn_factor > 1.0f) {
                oc_apply_rope_yarn_scaled_f32(q_pe, q_pe, q_pe_dim, q_pe_dim,
                                              (int64_t)pos, cfg->rope_theta,
                                              cfg->yarn_factor,
                                              (uint32_t)cfg->yarn_orig_ctx, rope_amp);
            } else {
                oc_inference_config_apply_rope_head(cfg, q_pe, q_pe,
                                                     q_pe_dim, q_pe_dim,
                                                     (int64_t)pos, cfg->rope_theta);
            }
        }
    }

    /* 6. KV cache: store padded V (v_head_dim -> kv_head_dim). */
    int32_t kv_idx = -1;
    if (li < m->kv_layer_map_len)
        kv_idx = m->kv_layer_map[li];

    float *v_padded = m->workspace.attn_result;  /* reuse as scratch [total_k] */
    if (v_padded) {
        memset(v_padded, 0, total_k * sizeof(float));
        for (uint32_t hd = 0; hd < n_heads; hd++) {
            size_t v_off = (size_t)hd * v_head_dim;
            size_t k_off = (size_t)hd * kv_head_dim;
            memcpy(v_padded + k_off, v_store + v_off, v_head_dim * sizeof(float));
        }
    }

    if (kv_idx >= 0 && k_store && v_padded) {
        e = oc_kv_cache_append(&m->kv_cache, (uint32_t)kv_idx,
                               k_store, v_padded, 1);
        if (e != OC_OK) return e;
    }

    /* 7. Attention: per-head scaled dot-product. */
    memset(attn_out, 0, (size_t)n_heads * v_head_dim * sizeof(float));
    if (kv_idx >= 0 && k_store && v_padded) {
        uint32_t seq_len = oc_kv_cache_n_tokens(&m->kv_cache);
        /* deepseek_yarn folds a get_mscale(factor, mscale_all_dim)^2 term
         * into the attention scale. For LongCat (factor 120, both mscale
         * terms 1) that is 1.4787^2 = 2.1867, so the plain 1/sqrt(192)
         * would leave every logit 2.19x too small and flatten the softmax
         * across all 76 sub-layers. Architectures without deepseek_yarn
         * leave yarn_mscale_all_dim at 0 and get the usual 1/sqrt(d). */
        float scale = 1.0f / sqrtf((float)kv_head_dim);
        if (cfg->yarn_factor > 1.0f && cfg->yarn_mscale_all_dim > 0.0f) {
            oc_rope_deepseek_yarn_scales(cfg->yarn_factor, cfg->yarn_mscale,
                                          cfg->yarn_mscale_all_dim,
                                          (uint32_t)kv_head_dim,
                                          NULL, &scale);
        }

        for (uint32_t hd = 0; hd < n_heads; hd++) {
            size_t q_off = (size_t)hd * kv_head_dim;
            float *q_head = q_vec + q_off;
            /* Q, K and the cached V are strided by kv_head_dim (nope + rope,
             * 192 for LongCat), but the attention OUTPUT is only v_head_dim
             * (128) wide per head and o_proj expects it densely packed.
             * Writing it at the Q stride leaves 64 dead floats per head, so
             * o_proj reads heads interleaved with padding and runs out of
             * input 42 heads in — the last 22 heads never reach it. */
            size_t v_off_out = (size_t)hd * v_head_dim;
            float *out_head = attn_out + v_off_out;

            /* Compute attention scores. */
            float *scores = m->workspace.kv_keys_copy;  /* reuse scratch */
            if (scores && seq_len <= m->workspace.kv_copy_size) {
                float max_s = -INFINITY;
                for (uint32_t t = 0; t < seq_len; t++) {
                    const float *k_t = NULL;
                    const float *v_t = NULL;
                    oc_kv_cache_get(&m->kv_cache, (uint32_t)kv_idx, t, &k_t, &v_t);
                    if (!k_t) continue;
                    float dot = 0.0f;
                    for (size_t i = 0; i < kv_head_dim; i++)
                        dot += q_head[i] * k_t[q_off + i];
                    scores[t] = dot * scale;
                    if (scores[t] > max_s) max_s = scores[t];
                }

                /* Softmax. */
                float sum = 0.0f;
                for (uint32_t t = 0; t < seq_len; t++) {
                    scores[t] = expf(scores[t] - max_s);
                    sum += scores[t];
                }
                if (sum > 0.0f) {
                    for (uint32_t t = 0; t < seq_len; t++)
                        scores[t] /= sum;
                }

                /* Weighted sum of V. */
                for (size_t i = 0; i < v_head_dim; i++) {
                    float acc = 0.0f;
                    for (uint32_t t = 0; t < seq_len; t++) {
                        const float *k_t = NULL;
                        const float *v_t = NULL;
                        oc_kv_cache_get(&m->kv_cache, (uint32_t)kv_idx, t, &k_t, &v_t);
                        if (!v_t) continue;
                        /* The cache holds V re-strided to kv_head_dim by the
                         * v_padded step above, so this reads at the Q stride
                         * even though V is only v_head_dim wide. */
                        acc += scores[t] * v_t[q_off + i];
                    }
                    out_head[i] = acc;
                }
            }
        }
    }

    /* 8. Attn output projection + residual. */
    if (!oc_weight_storage_is_empty(&layer->attn_output)) {
        size_t attn_in_len = oc_weight_storage_output_dim(&layer->attn_output, h);
        float *proj = m->workspace.hidden_b;
        e = oc_gemv_weight(&layer->attn_output, h, attn_in_len, attn_out, proj);
        if (e != OC_OK) return e;
        for (size_t i = 0; i < h; i++)
            x[i] += proj[i];
    }

    return OC_OK;
}

/* ─── ShortConv (LFM2) layer forward ──────────────────────────────────── */

static OcError forward_shortconv_layer(OcInferenceModel *m, OcLayerWeights *layer,
                                        size_t li, size_t pos)
{
    (void)pos;
    OcInferenceConfig *cfg = &m->config;
    size_t h = cfg->hidden_size;
    float eps = cfg->rms_norm_eps;
    float *x = m->workspace.x;

    /* ShortConv: norm -> in_proj(3*d) -> B*x -> conv1d -> C*conv -> out_proj -> residual. */
    uint32_t l_cache = cfg->shortconv_l_cache;
    if (l_cache == 0) l_cache = 1;

    size_t d = oc_weight_storage_output_dim(&layer->shortconv_in_proj, h) / 3;
    if (d == 0) d = h;

    /* 1. RMSNorm. */
    float *normed = m->workspace.hidden_b;
    if (layer->attn_norm)
        oc_rms_norm_f32(x, layer->attn_norm, normed, h, eps);
    else
        memcpy(normed, x, h * sizeof(float));

    /* 2. in_proj: [h] -> [3*d]. */
    float *bcx = m->workspace.shortconv_bcx;  /* [3*d] */
    memset(bcx, 0, 3 * d * sizeof(float));
    OcError e = oc_gemv_weight(&layer->shortconv_in_proj, 3 * d, h, normed, bcx);
    if (e != OC_OK) return e;

    /* 3. Bx = B * x (B = bcx[0..d], C = bcx[d..2d], x_in = bcx[2d..3d]). */
    float *bx = m->workspace.shortconv_bx;  /* [d] */
    for (size_t i = 0; i < d; i++)
        bx[i] = bcx[i] * bcx[2 * d + i];

    /* 4. Causal depthwise conv1d. Weights [d, l_cache] (channel-major, taps per channel).
     * Last tap = current token. Use SSM conv ring buffer for past frames. */
    float *conv_out = m->workspace.conv_out;  /* [d] */
    bool have_conv = layer->shortconv_conv && layer->n_shortconv_conv == l_cache * d;
    if (have_conv) {
        /* Use SSM engine conv ring buffer for past frames if available. */
        OcSsmConvRing *ring = NULL;
        if (m->ssm_engine.conv_buffers && li < m->ssm_engine.n_layers &&
            m->ssm_engine.conv_buffers[li].dim == d) {
            ring = &m->ssm_engine.conv_buffers[li];
        }
        for (size_t c = 0; c < d; c++) {
            float sum = layer->shortconv_conv[c * l_cache + (l_cache - 1)] * bx[c];
            /* Add past frames from conv ring buffer. */
            if (ring) {
                for (size_t t = 1; t < l_cache && t <= ring->len; t++) {
                    const float *past;
                    size_t past_len;
                    if (oc_ssm_conv_ring_past(ring, t, &past, &past_len) == OC_OK) {
                        sum += layer->shortconv_conv[c * l_cache + (l_cache - 1 - t)] * past[c];
                    }
                }
            }
            conv_out[c] = sum;
        }
        /* Push current bx to ring buffer for next token. */
        if (ring) {
            oc_ssm_conv_ring_push(ring, bx, d);
        }
    } else {
        memcpy(conv_out, bx, d * sizeof(float));
    }

    /* 5. y = C * conv_out. */
    for (size_t i = 0; i < d; i++)
        conv_out[i] *= bcx[d + i];

    /* 6. out_proj: [d] -> [h]. */
    float *attn_out = m->workspace.hidden_a;  /* [h] */
    memset(attn_out, 0, h * sizeof(float));
    e = oc_gemv_weight(&layer->shortconv_out_proj, h, d, conv_out, attn_out);
    if (e != OC_OK) return e;

    /* 7. Residual add. */
    for (size_t i = 0; i < h; i++)
        x[i] += attn_out[i];

    return OC_OK;
}

/* ─── Mamba/SSM layer forward ─────────────────────────────────────────── */

static OcError forward_mamba_layer(OcInferenceModel *m, OcLayerWeights *layer,
                                    size_t li, size_t pos)
{
    (void)pos;
    OcInferenceConfig *cfg = &m->config;
    size_t h = cfg->hidden_size;
    float eps = cfg->rms_norm_eps;
    float *x = m->workspace.x;

    /* Mamba: norm -> qkv_proj -> conv1d -> silu -> split(x_ssm, z_gate)
     * -> group_norm(x_ssm) -> selective_scan -> gate -> out_proj -> residual. */

    /* 1. RMSNorm. */
    float *normed = m->workspace.hidden_a;
    if (layer->attn_norm)
        oc_rms_norm_f32(x, layer->attn_norm, normed, h, eps);
    else
        memcpy(normed, x, h * sizeof(float));

    /* 2. QKV projection: [h] -> [qkv_out_len]. */
    size_t qkv_out_len = oc_weight_storage_output_dim(&layer->attn_qkv, h);
    float *x_proj = m->workspace.q_full;
    memset(x_proj, 0, qkv_out_len * sizeof(float));
    OcError e = oc_gemv_weight(&layer->attn_qkv, qkv_out_len, h, normed, x_proj);
    if (e != OC_OK) return e;

    /* 3. Causal conv1d (kernel=4). Use SSM conv ring buffer for past frames. */
    size_t conv_kernel = 4;
    float *conv_out = m->workspace.conv_out;
    memset(conv_out, 0, qkv_out_len * sizeof(float));
    if (layer->ssm_conv1d && layer->n_ssm_conv1d == conv_kernel * qkv_out_len) {
        /* Use SSM engine conv ring buffer for past frames. */
        OcSsmConvRing *ring = NULL;
        if (m->ssm_engine.conv_buffers && li < m->ssm_engine.n_layers &&
            m->ssm_engine.conv_buffers[li].dim == qkv_out_len) {
            ring = &m->ssm_engine.conv_buffers[li];
        }
        /* Tap-major [kernel, channels]; newest uses last tap. */
        for (size_t c = 0; c < qkv_out_len; c++) {
            float sum = layer->ssm_conv1d[(conv_kernel - 1) * qkv_out_len + c] * x_proj[c];
            /* Add past frames from conv ring buffer. */
            if (ring) {
                for (size_t t = 1; t < conv_kernel && t <= ring->len; t++) {
                    const float *past;
                    size_t past_len;
                    if (oc_ssm_conv_ring_past(ring, t, &past, &past_len) == OC_OK) {
                        sum += layer->ssm_conv1d[(conv_kernel - 1 - t) * qkv_out_len + c] * past[c];
                    }
                }
            }
            conv_out[c] = sum;
        }
        /* Push current x_proj to ring buffer for next token. */
        if (ring) {
            oc_ssm_conv_ring_push(ring, x_proj, qkv_out_len);
        }
    } else {
        memcpy(conv_out, x_proj, qkv_out_len * sizeof(float));
    }

    /* 4. SiLU activation. */
    for (size_t i = 0; i < qkv_out_len; i++) {
        float v = conv_out[i];
        conv_out[i] = v * (1.0f / (1.0f + expf(-v)));
    }

    /* 5. Split into x_ssm and z_gate. */
    size_t half = qkv_out_len / 2;
    float *x_ssm = conv_out;          /* [half] */
    float *z_gate = conv_out + half;  /* [half] */

    /* 6. Group RMSNorm on x_ssm. */
    if (layer->ssm_norm && layer->n_ssm_norm > 0 && half > 0 &&
        (half % layer->n_ssm_norm) == 0) {
        size_t group_size = layer->n_ssm_norm;
        size_t num_groups = half / group_size;
        float *normed_group = m->workspace.head_scratch;
        for (size_t g = 0; g < num_groups; g++) {
            size_t start = g * group_size;
            oc_rms_norm_f32(x_ssm + start, layer->ssm_norm, normed_group,
                            group_size, eps);
            memcpy(x_ssm + start, normed_group, group_size * sizeof(float));
        }
    }

    /* 7. Selective scan SSM. */
    size_t state_dim = layer->n_ssm_a;
    float *mamba_out = m->workspace.intermediate_b;  /* [half] */
    memset(mamba_out, 0, half * sizeof(float));

    if (state_dim > 0 && layer->ssm_alpha && layer->ssm_beta) {
        /* Bx = ssm_beta @ x_ssm: [state_dim, x_ssm_len]. */
        float *bx = m->workspace.intermediate_c;  /* [state_dim] */
        memset(bx, 0, state_dim * sizeof(float));
        size_t x_ssm_len = half;
        if (layer->n_ssm_beta == x_ssm_len * state_dim) {
            for (size_t j = 0; j < x_ssm_len; j++) {
                for (size_t i = 0; i < state_dim; i++) {
                    bx[i] += layer->ssm_beta[j * state_dim + i] * x_ssm[j];
                }
            }
        }

        /* Update SSM state: h = h * exp(-A * dt) + Bx * dt.
         * Use SSM engine state if available. */
        float *ssm_state = NULL;
        if (m->ssm_engine.ssm_states && li < m->ssm_engine.n_layers)
            ssm_state = m->ssm_engine.ssm_states + li * state_dim;
        if (!ssm_state) {
            /* Fallback: allocate on workspace scratch. */
            ssm_state = m->workspace.moe_gate_all;  /* reuse scratch */
            memset(ssm_state, 0, state_dim * sizeof(float));
        }

        for (size_t i = 0; i < state_dim; i++) {
            float a = layer->ssm_a[i % layer->n_ssm_a];
            float a_decay = expf(-a);
            float dt = 0.01f;
            if (layer->ssm_dt_bias && layer->n_ssm_dt_bias > 0) {
                float b = layer->ssm_dt_bias[i % layer->n_ssm_dt_bias];
                dt = logf(1.0f + expf(b));
            }
            float decay = expf(a_decay * dt);
            ssm_state[i] = ssm_state[i] * decay + bx[i] * dt;
        }

        /* Output: y = ssm_alpha @ state. */
        size_t y_len = layer->n_ssm_alpha / state_dim;
        if (y_len > half) y_len = half;
        if (layer->n_ssm_alpha == y_len * state_dim) {
            for (size_t j = 0; j < y_len; j++) {
                float sum = 0.0f;
                for (size_t i = 0; i < state_dim; i++) {
                    sum += layer->ssm_alpha[j * state_dim + i] * ssm_state[i];
                }
                mamba_out[j] = sum;
            }
        }
    }

    /* 8. Apply gate: y *= silu(z_gate). */
    for (size_t i = 0; i < half; i++) {
        float z = z_gate[i];
        mamba_out[i] *= z * (1.0f / (1.0f + expf(-z)));
    }

    /* 9. Output projection + residual. */
    if (!oc_weight_storage_is_empty(&layer->ssm_out)) {
        size_t out_len = oc_weight_storage_output_dim(&layer->ssm_out, half);
        float *projected = m->workspace.hidden_b;  /* [h] */
        memset(projected, 0, h * sizeof(float));
        e = oc_gemv_weight(&layer->ssm_out, out_len, half, mamba_out, projected);
        if (e != OC_OK) return e;
        for (size_t i = 0; i < h; i++)
            x[i] += projected[i];
    } else {
        size_t copy = h < half ? h : half;
        for (size_t i = 0; i < copy; i++)
            x[i] += mamba_out[i];
    }

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

        /* --- Dispatch to specialized layer types --- */

        /* MLA (DeepSeek2): has mla_kv_a_mqa. */
        if (oc_layer_weights_has_mla(layer)) {
            OcError e = forward_mla_layer(m, layer, li, position);
            if (e != OC_OK) return e;

            /* FFN block (shared with standard attention). */
            goto ffn_block;
        }

        /* ShortConv (LFM2): has shortconv_in_proj. */
        if (!oc_weight_storage_is_empty(&layer->shortconv_in_proj)) {
            OcError e = forward_shortconv_layer(m, layer, li, position);
            if (e != OC_OK) return e;

            /* ShortConv layers may have FFN too. */
            goto ffn_block;
        }

        /* Mamba/SSM: has attn_qkv but NOT attn_q. */
        if (!oc_weight_storage_is_empty(&layer->attn_qkv) &&
            oc_weight_storage_is_empty(&layer->attn_q)) {
            OcError e = forward_mamba_layer(m, layer, li, position);
            if (e != OC_OK) return e;
            continue;  /* Mamba layers have no separate FFN block. */
        }

        /* --- Standard attention block --- */
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
                            kv_head_dim, out_head);
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
        ffn_block:;
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
                    m->workspace.moe_expert_scores);
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

/* ─── MTP/nextn draft generation ───────────────────────────────────────── */

/* Run one MTP forward step: embed + eh_proj + attn + ffn + lm_head.
 * Writes the next hidden state to out_hidden [hidden_size].
 * Writes logits to out_logits [vocab_size]. */
static OcError mtp_forward_one(OcInferenceModel *m,
                                 const OcMtpWeights *mtp,
                                 uint32_t token,
                                 const float *prev_hidden,
                                 size_t pos,
                                 float *out_hidden, size_t out_hidden_len,
                                 float *out_logits, size_t out_logits_len)
{
    if (!m || !mtp || !prev_hidden || !out_hidden || !out_logits)
        return OC_ERR_INVALID_ARG;

    size_t h = m->config.hidden_size;
    size_t vocab = m->config.vocab_size;
    float eps = m->config.rms_norm_eps;

    if (out_hidden_len < h) return OC_ERR_INVALID_ARG;
    if (out_logits_len < vocab) return OC_ERR_INVALID_ARG;

    /* 1. Token embedding lookup. */
    float *token_emb = m->workspace.hidden_b;  /* [h] */
    if (!token_emb) return OC_ERR_MODEL;
    memset(token_emb, 0, h * sizeof(float));

    const OcWeightStorage *emb_storage = &mtp->embed_tokens;
    if (oc_weight_storage_is_empty(emb_storage))
        emb_storage = &m->tok_embeddings;

    uint32_t tok_idx = token;
    if (tok_idx >= vocab && vocab > 0) tok_idx = vocab - 1;
    oc_weight_storage_lookup_embedding(emb_storage, h, vocab, tok_idx, token_emb);

    /* 2. Embed norm + hidden norm. */
    float *embed_normed = m->workspace.intermediate_a;  /* [h] */
    float *hidden_normed = m->workspace.intermediate_b;  /* [h] */
    if (!embed_normed || !hidden_normed) return OC_ERR_MODEL;

    if (mtp->enorm) {
        oc_rms_norm_f32(token_emb, mtp->enorm, embed_normed, h, eps);
    } else {
        memcpy(embed_normed, token_emb, h * sizeof(float));
    }
    if (mtp->hnorm) {
        oc_rms_norm_f32(prev_hidden, mtp->hnorm, hidden_normed, h, eps);
    } else {
        memcpy(hidden_normed, prev_hidden, h * sizeof(float));
    }

    /* 3. Concat [embed_normed, hidden_normed] -> eh_proj -> fused [h]. */
    float *concat = m->workspace.shortconv_bcx;  /* [2*h] */
    if (!concat) return OC_ERR_MODEL;
    memcpy(concat, embed_normed, h * sizeof(float));
    memcpy(concat + h, hidden_normed, h * sizeof(float));

    float *fused = m->workspace.x;  /* [h] - overwrite workspace.x */
    memset(fused, 0, h * sizeof(float));
    OcError e = oc_gemv_weight(&mtp->eh_proj, h, h * 2, concat, fused);
    if (e != OC_OK) return e;

    /* 4. Run MTP attention + FFN layer in-place on workspace.x. */
    const OcLayerWeights *layer = &mtp->layer;
    uint32_t n_heads = m->config.num_attention_heads;
    uint32_t kvh = m->config.num_key_value_heads;
    uint32_t head_dim = oc_inference_config_head_dim(&m->config);
    uint32_t kv_head_dim = oc_inference_config_kv_head_dim(&m->config);
    uint32_t rope_len = oc_inference_config_effective_rope_dim(&m->config);
    float rope_theta = m->config.rope_theta;

    /* Attn norm. */
    float *normed = m->workspace.hidden_a;
    if (layer->attn_norm)
        oc_rms_norm_f32(fused, layer->attn_norm, normed, h, eps);
    else
        memcpy(normed, fused, h * sizeof(float));

    /* Q/K/V projections. */
    uint32_t q_len = n_heads * head_dim;
    uint32_t kv_len = kvh * kv_head_dim;
    float *q_vec = m->workspace.q_full;
    float *k_vec = m->workspace.k_vec;
    float *v_vec = m->workspace.v_vec;
    float *attn_out = m->workspace.attn_result;

    if (!oc_weight_storage_is_empty(&layer->attn_q)) {
        e = oc_gemv_weight(&layer->attn_q, q_len, h, normed, q_vec);
        if (e != OC_OK) return e;
        if (layer->attn_q_bias)
            oc_add_repeating_bias(q_vec, q_len, layer->attn_q_bias, q_len);
    }
    if (!oc_weight_storage_is_empty(&layer->attn_k)) {
        e = oc_gemv_weight(&layer->attn_k, kv_len, h, normed, k_vec);
        if (e != OC_OK) return e;
        if (layer->attn_k_bias)
            oc_add_repeating_bias(k_vec, kv_len, layer->attn_k_bias, kv_len);
    }
    if (!oc_weight_storage_is_empty(&layer->attn_v)) {
        e = oc_gemv_weight(&layer->attn_v, kv_len, h, normed, v_vec);
        if (e != OC_OK) return e;
        if (layer->attn_v_bias)
            oc_add_repeating_bias(v_vec, kv_len, layer->attn_v_bias, kv_len);
    }

    /* Apply RoPE. */
    for (uint32_t hd = 0; hd < n_heads; hd++) {
        float *qh = q_vec + hd * head_dim;
        oc_inference_config_apply_rope_head(&m->config, qh, qh, head_dim,
                                             rope_len, (int64_t)pos, rope_theta);
    }
    for (uint32_t hd = 0; hd < kvh; hd++) {
        float *kh = k_vec + hd * kv_head_dim;
        oc_inference_config_apply_rope_head(&m->config, kh, kh, kv_head_dim,
                                             rope_len, (int64_t)pos, rope_theta);
    }

    /* Attention: use SDPA against the model's KV cache at the MTP layer slot.
     * For simplicity, we use a separate MTP KV cache appended to the main cache.
     * The MTP block has its own KV context (pos 0..max_tokens-1). */
    memset(attn_out, 0, q_len * sizeof(float));

    /* Simple attention: attend against what we have so far.
     * For the first draft token (pos=0), seq_len=1, attend to self. */
    uint32_t seq_len = (uint32_t)pos + 1;
    /* Use workspace scratch as a mini KV cache for MTP. */
    /* For pos=0, K/V are just the current token's K/V. */
    /* For pos>0, we would need to store previous K/V. Use workspace.kv_keys_copy. */
    size_t kv_total = (size_t)kv_len * seq_len;
    float *mtp_keys = m->workspace.kv_keys_copy;
    float *mtp_vals = m->workspace.kv_values_copy;
    if (mtp_keys && mtp_vals && kv_total <= m->workspace.kv_copy_size) {
        /* Store current K/V at position pos. */
        memcpy(mtp_keys + pos * kv_len, k_vec, kv_len * sizeof(float));
        memcpy(mtp_vals + pos * kv_len, v_vec, kv_len * sizeof(float));

        /* SDPA per Q head. */
        uint32_t n_rep = n_heads / (kvh > 0 ? kvh : 1);
        for (uint32_t hd = 0; hd < n_heads; hd++) {
            uint32_t kv_hd = hd / n_rep;
            const float *q_head = q_vec + hd * head_dim;
            const float *k_head = mtp_keys + kv_hd * kv_head_dim;
            const float *v_head = mtp_vals + kv_hd * kv_head_dim;
            float *out_head = attn_out + hd * head_dim;
            oc_scaled_dot_product_attention_f32(
                q_head, k_head, v_head, seq_len, kv_len, out_head);
        }
    }

    /* Attn output projection + residual. */
    if (!oc_weight_storage_is_empty(&layer->attn_output)) {
        float *proj = m->workspace.hidden_b;
        e = oc_gemv_weight(&layer->attn_output, h, q_len, attn_out, proj);
        if (e != OC_OK) return e;
        if (layer->attn_output_bias)
            oc_add_repeating_bias(proj, h, layer->attn_output_bias, h);
        for (size_t i = 0; i < h; i++)
            fused[i] += proj[i];
    }

    /* FFN. */
    float *ffn_norm_weight = NULL;
    if (layer->post_attention_norm)
        ffn_norm_weight = layer->post_attention_norm;
    else if (layer->ffn_norm)
        ffn_norm_weight = layer->ffn_norm;

    if (ffn_norm_weight && oc_layer_weights_has_dense_ffn(layer)) {
        oc_rms_norm_f32(fused, ffn_norm_weight, normed, h, eps);
        size_t i_size = m->config.intermediate_size;
        float *gate = m->workspace.intermediate_a;
        float *up = m->workspace.intermediate_b;
        e = oc_gemv_weight(&layer->ffn_gate, i_size, h, normed, gate);
        if (e != OC_OK) return e;
        e = oc_gemv_weight(&layer->ffn_up, i_size, h, normed, up);
        if (e != OC_OK) return e;
        if (m->config.gelu_ffn)
            oc_geglu_inplace_f32(gate, up, i_size);
        else
            oc_swiglu_inplace_f32(gate, up, i_size);
        float *ffn_out = m->workspace.hidden_b;
        e = oc_gemv_weight(&layer->ffn_down, h, i_size, gate, ffn_out);
        if (e != OC_OK) return e;
        if (layer->ffn_down_bias)
            oc_add_repeating_bias(ffn_out, h, layer->ffn_down_bias, h);
        for (size_t i = 0; i < h; i++)
            fused[i] += ffn_out[i];
    }

    /* 5. Final norm + lm_head. */
    const float *norm_w = mtp->shared_head_norm ? mtp->shared_head_norm : m->norm_weight;
    const OcWeightStorage *head_w = &mtp->shared_head_head;
    if (oc_weight_storage_is_empty(head_w))
        head_w = &m->output_weight;

    oc_rms_norm_f32(fused, norm_w, out_hidden, h, eps);
    memset(out_logits, 0, vocab * sizeof(float));
    e = oc_gemv_weight(head_w, vocab, h, out_hidden, out_logits);
    if (e != OC_OK) return e;

    return OC_OK;
}

OcError oc_inf_model_draft_mtp_tokens(OcInferenceModel *m,
                                        uint32_t start_token,
                                        const float *start_hidden, size_t hidden_len,
                                        size_t max_tokens,
                                        uint32_t *out_tokens,
                                        float *out_logits,
                                        size_t *out_n)
{
    if (!m || !start_hidden || !out_tokens || !out_logits || !out_n)
        return OC_ERR_INVALID_ARG;
    if (hidden_len != m->config.hidden_size)
        return OC_ERR_INVALID_ARG;
    if (max_tokens == 0) {
        *out_n = 0;
        return OC_OK;
    }
    if (!oc_inf_model_has_mtp(m))
        return OC_ERR_MODEL;

    const OcMtpWeights *mtp = m->mtp;
    size_t h = m->config.hidden_size;
    size_t vocab = m->config.vocab_size;

    /* Buffers for current step. */
    float *cur_hidden = malloc(h * sizeof(float));
    if (!cur_hidden) return OC_ERR_OOM;
    memcpy(cur_hidden, start_hidden, h * sizeof(float));

    uint32_t cur_token = start_token;
    size_t n_generated = 0;

    /* Allocate hidden buffer for step output. */
    float *step_hidden = malloc(h * sizeof(float));
    if (!step_hidden) { free(cur_hidden); return OC_ERR_OOM; }

    for (size_t step = 0; step < max_tokens; step++) {
        float *step_logits = out_logits + n_generated * vocab;

        OcError e = mtp_forward_one(m, mtp, cur_token, cur_hidden, step,
                                      step_hidden, h, step_logits, vocab);
        if (e != OC_OK) {
            free(cur_hidden);
            free(step_hidden);
            *out_n = n_generated;
            return e;
        }

        /* Simple argmax sampling (no temperature/top-k for draft). */
        uint32_t best = 0;
        float best_v = step_logits[0];
        for (size_t i = 1; i < vocab; i++) {
            if (step_logits[i] > best_v) {
                best_v = step_logits[i];
                best = (uint32_t)i;
            }
        }

        out_tokens[n_generated++] = best;
        cur_token = best;

        /* Swap hidden buffers. */
        float *tmp = cur_hidden;
        cur_hidden = step_hidden;
        step_hidden = tmp;
    }

    free(cur_hidden);
    free(step_hidden);
    *out_n = n_generated;
    return OC_OK;
}

/* ─── Run a range of layers ────────────────────────────────────────────── */

OcError oc_inf_model_run_layer_range(OcInferenceModel *m,
                                       size_t start, size_t end,
                                       size_t pos)
{
    if (!m) return OC_ERR_INVALID_ARG;
    if (start >= m->n_layers || end > m->n_layers) return OC_ERR_INVALID_ARG;
    if (start >= end) return OC_OK;

    OcInferenceConfig *cfg = &m->config;
    size_t h = cfg->hidden_size;
    uint32_t n_heads = cfg->num_attention_heads;
    uint32_t kvh = cfg->num_key_value_heads;
    uint32_t head_dim = oc_inference_config_head_dim(cfg);
    uint32_t kv_head_dim = oc_inference_config_kv_head_dim(cfg);
    uint32_t rope_len = oc_inference_config_effective_rope_dim(cfg);
    float eps = cfg->rms_norm_eps;

    float *normed = m->workspace.hidden_a;
    float *q_vec = m->workspace.q_full;
    float *k_vec = m->workspace.k_vec;
    float *v_vec = m->workspace.v_vec;
    float *attn_out = m->workspace.attn_result;
    float *ffn_gate = m->workspace.intermediate_a;
    float *ffn_up = m->workspace.intermediate_b;
    float *ffn_down = m->workspace.hidden_b;

    for (size_t li = start; li < end && li < cfg->layer_count; li++) {
        OcLayerWeights *layer = &m->layers[li];
        float rope_theta = oc_inference_config_layer_rope_theta(cfg, (uint32_t)li);
        uint32_t layer_window = oc_inference_config_layer_sliding_window(cfg, (uint32_t)li);

        /* MLA layer. */
        if (oc_layer_weights_has_mla(layer)) {
            OcError e = forward_mla_layer(m, layer, li, pos);
            if (e != OC_OK) return e;
            goto lr_ffn;
        }

        /* ShortConv layer. */
        if (!oc_weight_storage_is_empty(&layer->shortconv_in_proj)) {
            OcError e = forward_shortconv_layer(m, layer, li, pos);
            if (e != OC_OK) return e;
            goto lr_ffn;
        }

        /* Mamba/SSM layer. */
        if (!oc_weight_storage_is_empty(&layer->attn_qkv) &&
            oc_weight_storage_is_empty(&layer->attn_q)) {
            OcError e = forward_mamba_layer(m, layer, li, pos);
            if (e != OC_OK) return e;
            continue;
        }

        /* Standard attention. */
        if (oc_layer_weights_has_attention(layer)) {
            if (layer->attn_norm)
                oc_rms_norm_f32(m->workspace.x, layer->attn_norm, normed, h, eps);
            else
                memcpy(normed, m->workspace.x, h * sizeof(float));

            uint32_t q_len = n_heads * head_dim;
            uint32_t kv_len = kvh * kv_head_dim;

            OcError e = oc_gemv_weight(&layer->attn_q, q_len, h, normed, q_vec);
            if (e != OC_OK) return e;
            if (layer->attn_q_bias)
                oc_add_repeating_bias(q_vec, q_len, layer->attn_q_bias, q_len);

            if (!oc_weight_storage_is_empty(&layer->attn_k)) {
                e = oc_gemv_weight(&layer->attn_k, kv_len, h, normed, k_vec);
                if (e != OC_OK) return e;
                if (layer->attn_k_bias)
                    oc_add_repeating_bias(k_vec, kv_len, layer->attn_k_bias, kv_len);
            }
            if (!oc_weight_storage_is_empty(&layer->attn_v)) {
                e = oc_gemv_weight(&layer->attn_v, kv_len, h, normed, v_vec);
                if (e != OC_OK) return e;
                if (layer->attn_v_bias)
                    oc_add_repeating_bias(v_vec, kv_len, layer->attn_v_bias, kv_len);
            }

            /* Per-head Q norm. */
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
            /* Per-head K norm. */
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

            /* RoPE. */
            for (uint32_t hd = 0; hd < n_heads; hd++) {
                float *qh = q_vec + hd * head_dim;
                oc_inference_config_apply_rope_head(cfg, qh, qh, head_dim,
                                                     rope_len, (int64_t)pos, rope_theta);
            }
            for (uint32_t hd = 0; hd < kvh; hd++) {
                float *kh = k_vec + hd * kv_head_dim;
                oc_inference_config_apply_rope_head(cfg, kh, kh, kv_head_dim,
                                                     rope_len, (int64_t)pos, rope_theta);
            }

            /* KV cache. */
            int32_t kv_idx = -1;
            if (li < m->kv_layer_map_len)
                kv_idx = m->kv_layer_map[li];

            if (kv_idx >= 0 && kv_len > 0) {
                e = oc_kv_cache_append(&m->kv_cache, (uint32_t)kv_idx,
                                       k_vec, v_vec, 1);
                if (e != OC_OK) return e;
            }

            /* Attention. */
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
                    uint32_t n_rep = n_heads / (kvh > 0 ? kvh : 1);
                    for (uint32_t hd = 0; hd < n_heads; hd++) {
                        uint32_t kv_hd = hd / n_rep;
                        const float *q_head = q_vec + hd * head_dim;
                        const float *k_head = key_cache + kv_hd * kv_head_dim;
                        const float *v_head = val_cache + kv_hd * kv_head_dim;
                        float *out_head = attn_out + hd * head_dim;
                        oc_scaled_dot_product_attention_f32(
                            q_head, k_head, v_head, eff_seq,
                            kv_head_dim, out_head);
                    }
                }
            }

            /* Attn output projection + residual. */
            if (!oc_weight_storage_is_empty(&layer->attn_output)) {
                e = oc_gemv_weight(&layer->attn_output, h, q_len, attn_out, ffn_down);
                if (e != OC_OK) return e;
                if (layer->attn_output_bias)
                    oc_add_repeating_bias(ffn_down, h, layer->attn_output_bias, h);

                if (cfg->sandwich_norm && layer->post_attention_norm) {
                    oc_rms_norm_f32(ffn_down, layer->post_attention_norm, attn_out, h, eps);
                    memcpy(ffn_down, attn_out, h * sizeof(float));
                }

                for (size_t i = 0; i < h; i++)
                    m->workspace.x[i] += ffn_down[i];
            }
        }

    lr_ffn:
        {
            /* FFN block. */
            float *ffn_norm_weight = NULL;
            if (cfg->sandwich_norm)
                ffn_norm_weight = layer->ffn_norm;
            else if (layer->post_attention_norm)
                ffn_norm_weight = layer->post_attention_norm;
            else if (layer->ffn_norm)
                ffn_norm_weight = layer->ffn_norm;

            if (oc_layer_weights_has_moe(layer)) {
                if (ffn_norm_weight) {
                    oc_rms_norm_f32(m->workspace.x, ffn_norm_weight, normed, h, eps);
                } else {
                    memcpy(normed, m->workspace.x, h * sizeof(float));
                }
                float ffn_out_buf[4096];
                memset(ffn_out_buf, 0, h * sizeof(float));
                if (h <= 4096) {
                    float *gate_scratch = m->workspace.intermediate_c;
                    OcError e = oc_moe_ffn_forward(
                        &layer->ffn_gate_inp, &layer->ffn_gate_exps,
                        &layer->ffn_up_exps, &layer->ffn_down_exps,
                        layer->ffn_exp_probs_b, cfg, normed, ffn_out_buf,
                        ffn_gate, ffn_up, gate_scratch,
                        m->workspace.moe_router_logits,
                        m->workspace.moe_expert_scores);
                    if (e != OC_OK) return e;
                    for (size_t i = 0; i < h; i++)
                        m->workspace.x[i] += ffn_out_buf[i];
                }
            } else if (oc_layer_weights_has_dense_ffn(layer) && ffn_norm_weight) {
                oc_rms_norm_f32(m->workspace.x, ffn_norm_weight, normed, h, eps);

                size_t i_size = cfg->intermediate_size;
                OcError e = oc_gemv_weight(&layer->ffn_gate, i_size, h, normed, ffn_gate);
                if (e != OC_OK) return e;
                e = oc_gemv_weight(&layer->ffn_up, i_size, h, normed, ffn_up);
                if (e != OC_OK) return e;

                if (cfg->gelu_ffn)
                    oc_geglu_inplace_f32(ffn_gate, ffn_up, i_size);
                else
                    oc_swiglu_inplace_f32(ffn_gate, ffn_up, i_size);

                e = oc_gemv_weight(&layer->ffn_down, h, i_size, ffn_gate, ffn_down);
                if (e != OC_OK) return e;
                if (layer->ffn_down_bias)
                    oc_add_repeating_bias(ffn_down, h, layer->ffn_down_bias, h);

                if (cfg->sandwich_norm && layer->post_ffn_norm) {
                    float tmp[4096];
                    if (h <= 4096) {
                        oc_rms_norm_f32(ffn_down, layer->post_ffn_norm, tmp, h, eps);
                        memcpy(ffn_down, tmp, h * sizeof(float));
                    }
                }

                for (size_t i = 0; i < h; i++)
                    m->workspace.x[i] += ffn_down[i];
            }
        }

        /* EAGLE3 capture. */
        for (size_t ei = 0; ei < m->eagle3_n_capture_layers; ei++) {
            if (m->eagle3_capture_layers[ei] == li) {
                if (!m->eagle3_layer_hiddens[ei])
                    m->eagle3_layer_hiddens[ei] = malloc(h * sizeof(float));
                if (m->eagle3_layer_hiddens[ei])
                    memcpy(m->eagle3_layer_hiddens[ei], m->workspace.x,
                           h * sizeof(float));
                break;
            }
        }
    }

    return OC_OK;
}

/* ─── Batched forward (prefill + cross-sequence decode) ───────────────── */

bool oc_inf_model_layers_supported_for_batched(const OcInferenceModel *m)
{
    if (!m || m->n_layers == 0) return false;

    for (size_t i = 0; i < m->n_layers && i < m->config.layer_count; i++) {
        const OcLayerWeights *layer = &m->layers[i];

        /* Mamba layers have attn_qkv but no attn_q. */
        bool is_mamba = !oc_weight_storage_is_empty(&layer->attn_qkv) &&
                        oc_weight_storage_is_empty(&layer->attn_q);
        if (is_mamba) return false;

        /* MoE layers. */
        if (oc_layer_weights_has_moe(layer)) return false;

        /* ShortConv layers. */
        if (!oc_weight_storage_is_empty(&layer->shortconv_in_proj)) return false;

        /* MLA layers. */
        if (oc_layer_weights_has_mla(layer)) return false;

        /* Must have standard attention. */
        if (oc_weight_storage_is_empty(&layer->attn_q)) return false;
    }
    return true;
}

OcError oc_inf_model_forward_tokens(OcInferenceModel *m,
                                      const uint32_t *tokens, size_t n_tokens,
                                      size_t start_pos, bool need_logits,
                                      float **out_logits, size_t *out_logits_len)
{
    if (!m || !tokens || n_tokens == 0)
        return OC_ERR_INVALID_ARG;

    /* If not batched-capable, fall back to per-token. */
    if (!oc_inf_model_layers_supported_for_batched(m)) {
        for (size_t i = 0; i < n_tokens; i++) {
            OcError e = oc_inf_model_forward_token(m, tokens[i], start_pos + i);
            if (e != OC_OK) return e;
        }
        if (need_logits)
            return oc_inf_model_final_head_from_workspace(m, out_logits, out_logits_len);
        return OC_OK;
    }

    OcInferenceConfig *cfg = &m->config;
    size_t h = cfg->hidden_size;
    uint32_t n_heads = cfg->num_attention_heads;
    uint32_t kvh = cfg->num_key_value_heads;
    uint32_t head_dim = oc_inference_config_head_dim(cfg);
    uint32_t kv_head_dim = oc_inference_config_kv_head_dim(cfg);
    uint32_t rope_len = oc_inference_config_effective_rope_dim(cfg);
    float eps = cfg->rms_norm_eps;
    size_t i_size = cfg->intermediate_size;
    size_t batch = n_tokens;

    /* Allocate batch buffers. */
    float *x_batch = malloc(batch * h * sizeof(float));
    float *normed_batch = malloc(batch * h * sizeof(float));
    if (!x_batch || !normed_batch) {
        free(x_batch); free(normed_batch);
        return OC_ERR_OOM;
    }

    /* 1. Embedding lookup for each position. */
    for (size_t i = 0; i < batch; i++) {
        uint32_t tok = tokens[i];
        if (tok >= cfg->vocab_size && cfg->vocab_size > 0)
            tok = cfg->vocab_size - 1;
        oc_weight_storage_lookup_embedding(&m->tok_embeddings, h,
                                            cfg->vocab_size, tok,
                                            x_batch + i * h);
        if (cfg->embedding_scale != 1.0f) {
            for (size_t j = 0; j < h; j++)
                x_batch[i * h + j] *= cfg->embedding_scale;
        }
    }

    /* Determine dimensions from first layer. */
    uint32_t q_len = n_heads * head_dim;
    uint32_t kv_len = kvh * kv_head_dim;

    /* Allocate per-layer scratch. */
    float *q_batch = malloc(batch * q_len * sizeof(float));
    float *k_batch = malloc(batch * kv_len * sizeof(float));
    float *v_batch = malloc(batch * kv_len * sizeof(float));
    float *attn_batch = malloc(batch * q_len * sizeof(float));
    float *proj_batch = malloc(batch * h * sizeof(float));
    float *gate_batch = malloc(batch * i_size * sizeof(float));
    float *up_batch = malloc(batch * i_size * sizeof(float));
    float *ffn_batch = malloc(batch * h * sizeof(float));
    if (!q_batch || !k_batch || !v_batch || !attn_batch || !proj_batch ||
        !gate_batch || !up_batch || !ffn_batch) {
        free(x_batch); free(normed_batch); free(q_batch); free(k_batch);
        free(v_batch); free(attn_batch); free(proj_batch);
        free(gate_batch); free(up_batch); free(ffn_batch);
        return OC_ERR_OOM;
    }

    /* 2. Process each layer. */
    for (size_t li = 0; li < m->n_layers && li < cfg->layer_count; li++) {
        OcLayerWeights *layer = &m->layers[li];
        float rope_theta = oc_inference_config_layer_rope_theta(cfg, (uint32_t)li);
        uint32_t layer_window = oc_inference_config_layer_sliding_window(cfg, (uint32_t)li);

        /* --- Attn norm --- */
        for (size_t i = 0; i < batch; i++) {
            if (layer->attn_norm)
                oc_rms_norm_f32(x_batch + i * h, layer->attn_norm,
                                normed_batch + i * h, h, eps);
            else
                memcpy(normed_batch + i * h, x_batch + i * h, h * sizeof(float));
        }

        /* --- Batched Q/K/V GEMM --- */
        memset(q_batch, 0, batch * q_len * sizeof(float));
        memset(k_batch, 0, batch * kv_len * sizeof(float));
        memset(v_batch, 0, batch * kv_len * sizeof(float));

        OcError e = oc_gemm_weight(&layer->attn_q, q_len, h,
                                    normed_batch, q_batch, batch);
        if (e != OC_OK) goto batch_fail;
        if (layer->attn_q_bias)
            oc_add_repeating_bias(q_batch, batch * q_len, layer->attn_q_bias, q_len);

        if (!oc_weight_storage_is_empty(&layer->attn_k)) {
            e = oc_gemm_weight(&layer->attn_k, kv_len, h,
                                normed_batch, k_batch, batch);
            if (e != OC_OK) goto batch_fail;
            if (layer->attn_k_bias)
                oc_add_repeating_bias(k_batch, batch * kv_len, layer->attn_k_bias, kv_len);
        }

        if (!oc_weight_storage_is_empty(&layer->attn_v)) {
            e = oc_gemm_weight(&layer->attn_v, kv_len, h,
                                normed_batch, v_batch, batch);
            if (e != OC_OK) goto batch_fail;
            if (layer->attn_v_bias)
                oc_add_repeating_bias(v_batch, batch * kv_len, layer->attn_v_bias, kv_len);
        }

        /* --- Per-token: RoPE, KV cache, attention --- */
        for (size_t i = 0; i < batch; i++) {
            size_t pos = start_pos + i;
            float *q = q_batch + i * q_len;
            float *k = k_batch + i * kv_len;
            float *v = v_batch + i * kv_len;

            /* Per-head Q norm. */
            if (layer->attn_q_norm && layer->n_q_norm == head_dim) {
                for (uint32_t hd = 0; hd < n_heads; hd++) {
                    float *qh = q + hd * head_dim;
                    float tmp[256];
                    if (head_dim <= 256) {
                        oc_rms_norm_f32(qh, layer->attn_q_norm, tmp, head_dim, eps);
                        memcpy(qh, tmp, head_dim * sizeof(float));
                    }
                }
            }

            /* Per-head K norm. */
            if (layer->attn_k_norm && layer->n_k_norm == kv_head_dim) {
                for (uint32_t hd = 0; hd < kvh; hd++) {
                    float *kh = k + hd * kv_head_dim;
                    float tmp[256];
                    if (kv_head_dim <= 256) {
                        oc_rms_norm_f32(kh, layer->attn_k_norm, tmp, kv_head_dim, eps);
                        memcpy(kh, tmp, kv_head_dim * sizeof(float));
                    }
                }
            }

            /* RoPE on Q heads. */
            for (uint32_t hd = 0; hd < n_heads; hd++) {
                float *qh = q + hd * head_dim;
                oc_inference_config_apply_rope_head(cfg, qh, qh, head_dim,
                                                     rope_len, (int64_t)pos, rope_theta);
            }

            /* RoPE on K heads. */
            for (uint32_t hd = 0; hd < kvh; hd++) {
                float *kh = k + hd * kv_head_dim;
                oc_inference_config_apply_rope_head(cfg, kh, kh, kv_head_dim,
                                                     rope_len, (int64_t)pos, rope_theta);
            }

            /* KV cache append. */
            int32_t kv_idx = -1;
            if (li < m->kv_layer_map_len)
                kv_idx = m->kv_layer_map[li];

            if (kv_idx >= 0 && kv_len > 0) {
                e = oc_kv_cache_append(&m->kv_cache, (uint32_t)kv_idx, k, v, 1);
                if (e != OC_OK) goto batch_fail;
            }

            /* Attention. */
            float *attn_out = attn_batch + i * q_len;
            memset(attn_out, 0, q_len * sizeof(float));

            if (kv_idx >= 0 && kv_len > 0) {
                uint32_t seq_len = (uint32_t)(pos + 1);
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
                    uint32_t n_rep = n_heads / (kvh > 0 ? kvh : 1);
                    for (uint32_t hd = 0; hd < n_heads; hd++) {
                        uint32_t kv_hd = hd / n_rep;
                        const float *q_head = q + hd * head_dim;
                        const float *k_head = key_cache + kv_hd * kv_head_dim;
                        const float *v_head = val_cache + kv_hd * kv_head_dim;
                        float *out_head = attn_out + hd * head_dim;
                        oc_scaled_dot_product_attention_f32(
                            q_head, k_head, v_head, eff_seq,
                            kv_head_dim, out_head);
                    }
                }
            }
        }

        /* --- Batched attn output projection --- */
        if (!oc_weight_storage_is_empty(&layer->attn_output)) {
            e = oc_gemm_weight(&layer->attn_output, h, q_len,
                                attn_batch, proj_batch, batch);
            if (e != OC_OK) goto batch_fail;
            if (layer->attn_output_bias)
                oc_add_repeating_bias(proj_batch, batch * h, layer->attn_output_bias, h);

            /* Sandwich norm. */
            if (cfg->sandwich_norm && layer->post_attention_norm) {
                for (size_t i = 0; i < batch; i++) {
                    float tmp[4096];
                    if (h <= 4096) {
                        oc_rms_norm_f32(proj_batch + i * h, layer->post_attention_norm,
                                        tmp, h, eps);
                        memcpy(proj_batch + i * h, tmp, h * sizeof(float));
                    }
                }
            }
        } else {
            memset(proj_batch, 0, batch * h * sizeof(float));
        }

        /* Residual add. */
        for (size_t i = 0; i < batch * h; i++)
            x_batch[i] += proj_batch[i];

        /* --- FFN --- */
        float *ffn_norm_w = NULL;
        if (cfg->sandwich_norm)
            ffn_norm_w = layer->ffn_norm;
        else if (layer->post_attention_norm)
            ffn_norm_w = layer->post_attention_norm;
        else if (layer->ffn_norm)
            ffn_norm_w = layer->ffn_norm;

        if (oc_layer_weights_has_dense_ffn(layer) && ffn_norm_w) {
            for (size_t i = 0; i < batch; i++)
                oc_rms_norm_f32(x_batch + i * h, ffn_norm_w,
                                normed_batch + i * h, h, eps);

            e = oc_gemm_weight(&layer->ffn_gate, i_size, h,
                                normed_batch, gate_batch, batch);
            if (e != OC_OK) goto batch_fail;
            e = oc_gemm_weight(&layer->ffn_up, i_size, h,
                                normed_batch, up_batch, batch);
            if (e != OC_OK) goto batch_fail;

            for (size_t i = 0; i < batch; i++) {
                if (cfg->gelu_ffn)
                    oc_geglu_inplace_f32(gate_batch + i * i_size,
                                          up_batch + i * i_size, i_size);
                else
                    oc_swiglu_inplace_f32(gate_batch + i * i_size,
                                          up_batch + i * i_size, i_size);
            }

            e = oc_gemm_weight(&layer->ffn_down, h, i_size,
                                gate_batch, ffn_batch, batch);
            if (e != OC_OK) goto batch_fail;
            if (layer->ffn_down_bias)
                oc_add_repeating_bias(ffn_batch, batch * h, layer->ffn_down_bias, h);

            /* Sandwich norm post-FFN. */
            if (cfg->sandwich_norm && layer->post_ffn_norm) {
                for (size_t i = 0; i < batch; i++) {
                    float tmp[4096];
                    if (h <= 4096) {
                        oc_rms_norm_f32(ffn_batch + i * h, layer->post_ffn_norm,
                                        tmp, h, eps);
                        memcpy(ffn_batch + i * h, tmp, h * sizeof(float));
                    }
                }
            }

            /* Residual add. */
            for (size_t i = 0; i < batch * h; i++)
                x_batch[i] += ffn_batch[i];
        }

        /* EAGLE3 capture. */
        for (size_t ei = 0; ei < m->eagle3_n_capture_layers; ei++) {
            if (m->eagle3_capture_layers[ei] == li) {
                if (!m->eagle3_layer_hiddens[ei])
                    m->eagle3_layer_hiddens[ei] = malloc(h * sizeof(float));
                if (m->eagle3_layer_hiddens[ei])
                    memcpy(m->eagle3_layer_hiddens[ei],
                           x_batch + (batch - 1) * h, h * sizeof(float));
                break;
            }
        }
    }

    /* Copy last token's hidden to workspace.x. */
    memcpy(m->workspace.x, x_batch + (batch - 1) * h, h * sizeof(float));

    /* Final head. */
    if (need_logits) {
        oc_rms_norm_f32(m->workspace.x, m->norm_weight,
                        m->workspace.hidden_a, h, eps);
        if (m->last_output_hidden && m->last_output_hidden_len >= h)
            memcpy(m->last_output_hidden, m->workspace.hidden_a, h * sizeof(float));
        memset(m->workspace.logits, 0, cfg->vocab_size * sizeof(float));
        OcError e = oc_gemv_weight(&m->output_weight, cfg->vocab_size, h,
                                    m->workspace.hidden_a, m->workspace.logits);
        if (e != OC_OK) goto batch_fail;
        if (out_logits) *out_logits = m->workspace.logits;
        if (out_logits_len) *out_logits_len = cfg->vocab_size;
    }

    free(x_batch); free(normed_batch); free(q_batch); free(k_batch);
    free(v_batch); free(attn_batch); free(proj_batch);
    free(gate_batch); free(up_batch); free(ffn_batch);
    return OC_OK;

batch_fail:
    free(x_batch); free(normed_batch); free(q_batch); free(k_batch);
    free(v_batch); free(attn_batch); free(proj_batch);
    free(gate_batch); free(up_batch); free(ffn_batch);
    return OC_ERR_INTERNAL;
}

OcError oc_inf_model_forward_batch(OcInferenceModel *m,
                                     const uint32_t *tokens,
                                     const size_t *positions,
                                     OcSeqKv *kvs, size_t n_seqs,
                                     bool need_logits,
                                     float *out_logits)
{
    if (!m || !tokens || !positions || !kvs || n_seqs == 0)
        return OC_ERR_INVALID_ARG;

    if (!oc_inf_model_layers_supported_for_batched(m))
        return OC_ERR_MODEL;

    OcInferenceConfig *cfg = &m->config;
    size_t h = cfg->hidden_size;
    uint32_t n_heads = cfg->num_attention_heads;
    uint32_t kvh = cfg->num_key_value_heads;
    uint32_t head_dim = oc_inference_config_head_dim(cfg);
    uint32_t kv_head_dim = oc_inference_config_kv_head_dim(cfg);
    uint32_t rope_len = oc_inference_config_effective_rope_dim(cfg);
    float eps = cfg->rms_norm_eps;
    size_t i_size = cfg->intermediate_size;
    size_t batch = n_seqs;
    uint32_t q_len = n_heads * head_dim;
    uint32_t kv_len = kvh * kv_head_dim;

    /* Allocate batch buffers. */
    float *x_batch = malloc(batch * h * sizeof(float));
    float *normed_batch = malloc(batch * h * sizeof(float));
    float *q_batch = malloc(batch * q_len * sizeof(float));
    float *k_batch = malloc(batch * kv_len * sizeof(float));
    float *v_batch = malloc(batch * kv_len * sizeof(float));
    float *attn_batch = malloc(batch * q_len * sizeof(float));
    float *proj_batch = malloc(batch * h * sizeof(float));
    float *gate_batch = malloc(batch * i_size * sizeof(float));
    float *up_batch = malloc(batch * i_size * sizeof(float));
    float *ffn_batch = malloc(batch * h * sizeof(float));
    if (!x_batch || !normed_batch || !q_batch || !k_batch || !v_batch ||
        !attn_batch || !proj_batch || !gate_batch || !up_batch || !ffn_batch) {
        free(x_batch); free(normed_batch); free(q_batch); free(k_batch);
        free(v_batch); free(attn_batch); free(proj_batch);
        free(gate_batch); free(up_batch); free(ffn_batch);
        return OC_ERR_OOM;
    }

    /* 1. Embedding lookup for each sequence. */
    for (size_t i = 0; i < batch; i++) {
        uint32_t tok = tokens[i];
        if (tok >= cfg->vocab_size && cfg->vocab_size > 0)
            tok = cfg->vocab_size - 1;
        oc_weight_storage_lookup_embedding(&m->tok_embeddings, h,
                                            cfg->vocab_size, tok,
                                            x_batch + i * h);
        if (cfg->embedding_scale != 1.0f) {
            for (size_t j = 0; j < h; j++)
                x_batch[i * h + j] *= cfg->embedding_scale;
        }
    }

    /* 2. Process each layer. */
    for (size_t li = 0; li < m->n_layers && li < cfg->layer_count; li++) {
        OcLayerWeights *layer = &m->layers[li];
        float rope_theta = oc_inference_config_layer_rope_theta(cfg, (uint32_t)li);
        uint32_t layer_window = oc_inference_config_layer_sliding_window(cfg, (uint32_t)li);

        int32_t kv_idx = -1;
        if (li < m->kv_layer_map_len)
            kv_idx = m->kv_layer_map[li];

        /* Attn norm. */
        for (size_t i = 0; i < batch; i++) {
            if (layer->attn_norm)
                oc_rms_norm_f32(x_batch + i * h, layer->attn_norm,
                                normed_batch + i * h, h, eps);
            else
                memcpy(normed_batch + i * h, x_batch + i * h, h * sizeof(float));
        }

        /* Batched Q/K/V GEMM. */
        memset(q_batch, 0, batch * q_len * sizeof(float));
        memset(k_batch, 0, batch * kv_len * sizeof(float));
        memset(v_batch, 0, batch * kv_len * sizeof(float));

        OcError e = oc_gemm_weight(&layer->attn_q, q_len, h,
                                    normed_batch, q_batch, batch);
        if (e != OC_OK) goto batch2_fail;
        if (!oc_weight_storage_is_empty(&layer->attn_k)) {
            e = oc_gemm_weight(&layer->attn_k, kv_len, h,
                                normed_batch, k_batch, batch);
            if (e != OC_OK) goto batch2_fail;
        }
        if (!oc_weight_storage_is_empty(&layer->attn_v)) {
            e = oc_gemm_weight(&layer->attn_v, kv_len, h,
                                normed_batch, v_batch, batch);
            if (e != OC_OK) goto batch2_fail;
        }

        /* Per-sequence: RoPE, KV write to SeqKv, attention. */
        for (size_t i = 0; i < batch; i++) {
            size_t pos = positions[i];
            float *q = q_batch + i * q_len;
            float *k = k_batch + i * kv_len;
            float *v = v_batch + i * kv_len;

            /* RoPE. */
            for (uint32_t hd = 0; hd < n_heads; hd++) {
                float *qh = q + hd * head_dim;
                oc_inference_config_apply_rope_head(cfg, qh, qh, head_dim,
                                                     rope_len, (int64_t)pos, rope_theta);
            }
            for (uint32_t hd = 0; hd < kvh; hd++) {
                float *kh = k + hd * kv_head_dim;
                oc_inference_config_apply_rope_head(cfg, kh, kh, kv_head_dim,
                                                     rope_len, (int64_t)pos, rope_theta);
            }

            /* KV write to SeqKv. */
            if (kv_idx >= 0 && kv_len > 0) {
                oc_seq_kv_put(&kvs[i], (size_t)kv_idx, kvs[i].len, k, v);
            }
        }

        /* Per-sequence attention against each sequence's own KV. */
        memset(attn_batch, 0, batch * q_len * sizeof(float));
        if (kv_idx >= 0 && kv_len > 0) {
            for (size_t i = 0; i < batch; i++) {
                size_t seq_len = kvs[i].len + 1;  /* current token now written */
                float *q = q_batch + i * q_len;
                float *attn_out = attn_batch + i * q_len;

                const float *key_cache = NULL;
                const float *val_cache = NULL;
                oc_seq_kv_get(&kvs[i], (size_t)kv_idx, 0, &key_cache, &val_cache);

                if (key_cache && val_cache) {
                    uint32_t eff_seq = (uint32_t)seq_len;
                    const float *eff_k = key_cache;
                    const float *eff_v = val_cache;

                    if (layer_window > 0 && seq_len > layer_window) {
                        size_t skip = (seq_len - layer_window) * kv_len;
                        eff_k = key_cache + skip;
                        eff_v = val_cache + skip;
                        eff_seq = layer_window;
                    }

                    uint32_t n_rep = n_heads / (kvh > 0 ? kvh : 1);
                    for (uint32_t hd = 0; hd < n_heads; hd++) {
                        uint32_t kv_hd = hd / n_rep;
                        const float *q_head = q + hd * head_dim;
                        const float *k_head = eff_k + kv_hd * kv_head_dim;
                        const float *v_head = eff_v + kv_hd * kv_head_dim;
                        float *out_head = attn_out + hd * head_dim;
                        oc_scaled_dot_product_attention_f32(
                            q_head, k_head, v_head, eff_seq,
                            kv_head_dim, out_head);
                    }
                }
            }
        }

        /* Attn output projection. */
        if (!oc_weight_storage_is_empty(&layer->attn_output)) {
            e = oc_gemm_weight(&layer->attn_output, h, q_len,
                                attn_batch, proj_batch, batch);
            if (e != OC_OK) goto batch2_fail;
            if (layer->attn_output_bias)
                oc_add_repeating_bias(proj_batch, batch * h, layer->attn_output_bias, h);
        } else {
            memset(proj_batch, 0, batch * h * sizeof(float));
        }

        /* Residual add. */
        for (size_t i = 0; i < batch * h; i++)
            x_batch[i] += proj_batch[i];

        /* FFN. */
        float *ffn_norm_w = NULL;
        if (cfg->sandwich_norm)
            ffn_norm_w = layer->ffn_norm;
        else if (layer->post_attention_norm)
            ffn_norm_w = layer->post_attention_norm;
        else if (layer->ffn_norm)
            ffn_norm_w = layer->ffn_norm;

        if (oc_layer_weights_has_dense_ffn(layer) && ffn_norm_w) {
            for (size_t i = 0; i < batch; i++)
                oc_rms_norm_f32(x_batch + i * h, ffn_norm_w,
                                normed_batch + i * h, h, eps);

            e = oc_gemm_weight(&layer->ffn_gate, i_size, h,
                                normed_batch, gate_batch, batch);
            if (e != OC_OK) goto batch2_fail;
            e = oc_gemm_weight(&layer->ffn_up, i_size, h,
                                normed_batch, up_batch, batch);
            if (e != OC_OK) goto batch2_fail;

            for (size_t i = 0; i < batch; i++) {
                if (cfg->gelu_ffn)
                    oc_geglu_inplace_f32(gate_batch + i * i_size,
                                          up_batch + i * i_size, i_size);
                else
                    oc_swiglu_inplace_f32(gate_batch + i * i_size,
                                          up_batch + i * i_size, i_size);
            }

            e = oc_gemm_weight(&layer->ffn_down, h, i_size,
                                gate_batch, ffn_batch, batch);
            if (e != OC_OK) goto batch2_fail;
            if (layer->ffn_down_bias)
                oc_add_repeating_bias(ffn_batch, batch * h, layer->ffn_down_bias, h);

            for (size_t i = 0; i < batch * h; i++)
                x_batch[i] += ffn_batch[i];
        }
    }

    /* Advance each SeqKv by 1 (after all layers processed). */
    for (size_t i = 0; i < batch; i++)
        oc_seq_kv_advance(&kvs[i], 1);

    /* Final head for each sequence. */
    if (need_logits && out_logits) {
        float *normed = m->workspace.hidden_a;
        for (size_t i = 0; i < batch; i++) {
            oc_rms_norm_f32(x_batch + i * h, m->norm_weight, normed, h, eps);
            memset(out_logits + i * cfg->vocab_size, 0,
                   cfg->vocab_size * sizeof(float));
            oc_gemv_weight(&m->output_weight, cfg->vocab_size, h,
                           normed, out_logits + i * cfg->vocab_size);
        }
    }

    /* Copy last seq's hidden to workspace.x for compatibility. */
    memcpy(m->workspace.x, x_batch + (batch - 1) * h, h * sizeof(float));

    free(x_batch); free(normed_batch); free(q_batch); free(k_batch);
    free(v_batch); free(attn_batch); free(proj_batch);
    free(gate_batch); free(up_batch); free(ffn_batch);
    return OC_OK;

batch2_fail:
    free(x_batch); free(normed_batch); free(q_batch); free(k_batch);
    free(v_batch); free(attn_batch); free(proj_batch);
    free(gate_batch); free(up_batch); free(ffn_batch);
    return OC_ERR_INTERNAL;
}
