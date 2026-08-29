#include "oxidize/qwen_arch.h"
#include "oxidize/flash_attention.h"

#include <math.h>
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
    if (!model || !logits) return OC_ERR_INVALID_ARG;
    if (!model->initialized) return OC_ERR_MODEL;

    OcQwenConfig *cfg = &model->config;
    size_t h = cfg->hidden_dim;
    size_t hd = cfg->head_dim;
    size_t n_heads = cfg->n_heads;
    size_t n_kv_heads = cfg->n_kv_heads;
    size_t q_size = n_heads * hd;
    size_t kv_size = n_kv_heads * hd;
    size_t kv_len = n_kv_heads * hd;
    size_t inter = cfg->intermediate_dim;
    size_t vocab = cfg->vocab_size;
    float eps = cfg->norm_eps;

    float *hidden = malloc(h * sizeof(float));
    if (!hidden) return OC_ERR_OOM;
    size_t tok_idx = (size_t)token;
    if (tok_idx >= vocab) tok_idx = vocab - 1;
    if (model->tok_emb)
        memcpy(hidden, model->tok_emb + tok_idx * h, h * sizeof(float));
    else
        memset(hidden, 0, h * sizeof(float));

    /* Static KV cache. */
    static float *s_keys = NULL;
    static float *s_values = NULL;
    static size_t s_kv_cap = 0;
    static size_t s_seq_len = 0;
    if (s_kv_cap == 0) {
        s_kv_cap = 256;
        s_keys = calloc(s_kv_cap * kv_len, sizeof(float));
        s_values = calloc(s_kv_cap * kv_len, sizeof(float));
        s_seq_len = 0;
    }

    for (uint32_t li = 0; li < cfg->n_layers; li++) {
        OcQwenLayer *layer = &model->layers[li];

        /* Attention norm. */
        float *normed = malloc(h * sizeof(float));
        if (!normed) { free(hidden); return OC_ERR_OOM; }
        if (layer->attn_norm) {
            float ss = 0.0f;
            for (size_t i = 0; i < h; i++) ss += hidden[i] * hidden[i];
            float rms = 1.0f / sqrtf(ss / h + eps);
            for (size_t i = 0; i < h; i++) normed[i] = hidden[i] * rms * layer->attn_norm[i];
        } else {
            memcpy(normed, hidden, h * sizeof(float));
        }

        /* QKV. */
        float *q = calloc(q_size, sizeof(float));
        float *k = calloc(kv_size, sizeof(float));
        float *v = calloc(kv_size, sizeof(float));
        if (!q || !k || !v) { free(hidden); free(normed); free(q); free(k); free(v); return OC_ERR_OOM; }

        if (layer->attn_q)
            for (size_t r = 0; r < q_size; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < h; c++) dot += layer->attn_q[r * h + c] * normed[c];
                q[r] = dot;
            }
        if (layer->attn_k)
            for (size_t r = 0; r < kv_size; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < h; c++) dot += layer->attn_k[r * h + c] * normed[c];
                k[r] = dot;
            }
        if (layer->attn_v)
            for (size_t r = 0; r < kv_size; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < h; c++) dot += layer->attn_v[r * h + c] * normed[c];
                v[r] = dot;
            }

        /* Optional QK-norm (Qwen3): RMSNorm on each Q/K head. */
        if (cfg->use_qk_norm && layer->attn_q_norm) {
            for (size_t hh = 0; hh < n_heads; hh++) {
                float *qh = q + hh * hd;
                float ss = 0.0f;
                for (size_t i = 0; i < hd; i++) ss += qh[i] * qh[i];
                float rms = 1.0f / sqrtf(ss / hd + eps);
                for (size_t i = 0; i < hd; i++) qh[i] = qh[i] * rms * layer->attn_q_norm[i];
            }
        }
        if (cfg->use_qk_norm && layer->attn_k_norm) {
            for (size_t hh = 0; hh < n_kv_heads; hh++) {
                float *kh = k + hh * hd;
                float ss = 0.0f;
                for (size_t i = 0; i < hd; i++) ss += kh[i] * kh[i];
                float rms = 1.0f / sqrtf(ss / hd + eps);
                for (size_t i = 0; i < hd; i++) kh[i] = kh[i] * rms * layer->attn_k_norm[i];
            }
        }

        /* RoPE. */
        size_t pos = s_seq_len;
        float theta = cfg->rope_theta;
        for (size_t hh = 0; hh < n_heads; hh++) {
            float *qh = q + hh * hd;
            for (size_t d = 0; d < hd; d += 2) {
                float freq = pos / powf(theta, (float)(d / 2) / (float)(hd / 2));
                float c = cosf(freq), s = sinf(freq);
                float q0 = qh[d], q1 = qh[d + 1];
                qh[d] = q0 * c - q1 * s; qh[d + 1] = q0 * s + q1 * c;
            }
        }
        for (size_t hh = 0; hh < n_kv_heads; hh++) {
            float *kh = k + hh * hd;
            for (size_t d = 0; d < hd; d += 2) {
                float freq = pos / powf(theta, (float)(d / 2) / (float)(hd / 2));
                float c = cosf(freq), s = sinf(freq);
                float k0 = kh[d], k1 = kh[d + 1];
                kh[d] = k0 * c - k1 * s; kh[d + 1] = k0 * s + k1 * c;
            }
        }

        /* KV cache append. */
        if (s_seq_len < s_kv_cap) {
            size_t off = s_seq_len * kv_len;
            memcpy(s_keys + off, k, kv_size * sizeof(float));
            memcpy(s_values + off, v, kv_size * sizeof(float));
            s_seq_len++;
        }

        /* Attention decode. */
        float *attn_out = calloc(q_size, sizeof(float));
        if (!attn_out) { free(hidden); free(normed); free(q); free(k); free(v); return OC_ERR_OOM; }
        if (s_seq_len > 0)
            oc_flash_attention_decode_heads_f32(q, s_keys, s_values,
                s_seq_len, hd, kv_len, n_heads, n_kv_heads, attn_out);

        /* Output proj + residual. */
        float *attn_resid = calloc(h, sizeof(float));
        if (!attn_resid) { free(hidden); free(normed); free(q); free(k); free(v); free(attn_out); return OC_ERR_OOM; }
        if (layer->attn_output)
            for (size_t r = 0; r < h; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < q_size; c++) dot += layer->attn_output[r * q_size + c] * attn_out[c];
                attn_resid[r] = dot;
            }
        for (size_t i = 0; i < h; i++) hidden[i] += attn_resid[i];

        /* FFN norm. */
        if (layer->ffn_norm) {
            float ss = 0.0f;
            for (size_t i = 0; i < h; i++) ss += hidden[i] * hidden[i];
            float rms = 1.0f / sqrtf(ss / h + eps);
            for (size_t i = 0; i < h; i++) normed[i] = hidden[i] * rms * layer->ffn_norm[i];
        } else {
            memcpy(normed, hidden, h * sizeof(float));
        }

        /* SwiGLU FFN. */
        float *gate_out = malloc(inter * sizeof(float));
        float *up_out = malloc(inter * sizeof(float));
        float *act = malloc(inter * sizeof(float));
        float *mlp_out = calloc(h, sizeof(float));
        if (!gate_out || !up_out || !act || !mlp_out) {
            free(hidden); free(normed); free(q); free(k); free(v); free(attn_out); free(attn_resid);
            free(gate_out); free(up_out); free(act); free(mlp_out);
            return OC_ERR_OOM;
        }
        if (layer->ffn_gate)
            for (size_t r = 0; r < inter; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < h; c++) dot += layer->ffn_gate[r * h + c] * normed[c];
                gate_out[r] = dot;
            }
        if (layer->ffn_up)
            for (size_t r = 0; r < inter; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < h; c++) dot += layer->ffn_up[r * h + c] * normed[c];
                up_out[r] = dot;
            }
        for (size_t i = 0; i < inter; i++) {
            float silu_val = gate_out[i] / (1.0f + expf(-gate_out[i]));
            act[i] = silu_val * up_out[i];
        }
        if (layer->ffn_down)
            for (size_t r = 0; r < h; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < inter; c++) dot += layer->ffn_down[r * inter + c] * act[c];
                mlp_out[r] = dot;
            }
        for (size_t i = 0; i < h; i++) hidden[i] += mlp_out[i];

        free(normed); free(q); free(k); free(v); free(attn_out); free(attn_resid);
        free(gate_out); free(up_out); free(act); free(mlp_out);
    }

    /* Final norm + output. */
    float *normed = malloc(h * sizeof(float));
    if (!normed) { free(hidden); return OC_ERR_OOM; }
    if (model->output_norm) {
        float ss = 0.0f;
        for (size_t i = 0; i < h; i++) ss += hidden[i] * hidden[i];
        float rms = 1.0f / sqrtf(ss / h + eps);
        for (size_t i = 0; i < h; i++) normed[i] = hidden[i] * rms * model->output_norm[i];
    } else {
        memcpy(normed, hidden, h * sizeof(float));
    }

    float *out_w = model->output;
    if (!out_w && cfg->tie_word_embeddings) out_w = model->tok_emb;

    if (out_w) {
        for (size_t r = 0; r < vocab; r++) {
            float dot = 0.0f;
            for (size_t c = 0; c < h; c++) dot += out_w[r * h + c] * normed[c];
            logits[r] = dot;
        }
    } else {
        memset(logits, 0, vocab * sizeof(float));
    }

    free(hidden);
    free(normed);
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
