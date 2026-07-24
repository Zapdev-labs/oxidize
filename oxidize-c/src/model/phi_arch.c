/*
 * phi_arch.c — Phi-2 / Phi-3 architecture forward pass implementation.
 * GeGLU + RoPE + dense MHA (no GQA).
 */
#include "oxidize/phi_arch.h"
#include "oxidize/flash_attention.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

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

static float phi_gelu_tanh(float x)
{
    float c = 0.7978845608028654f;
    float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

OcError oc_phi_forward(OcPhiModel *model, uint32_t token, float *logits)
{
    if (!model || !model->initialized || !logits) return OC_ERR_INVALID_ARG;
    if (model->config.vocab_size == 0) return OC_ERR_MODEL;

    OcPhiConfig *cfg = &model->config;
    size_t h = cfg->hidden_dim;
    size_t hd = cfg->head_dim;
    size_t n_heads = cfg->n_heads;
    /* Phi uses dense attention: n_kv_heads == n_heads */
    size_t n_kv_heads = n_heads;
    size_t q_size = n_heads * hd;
    size_t kv_size = n_kv_heads * hd;
    size_t kv_len = n_kv_heads * hd;
    size_t inter = cfg->intermediate_dim;
    size_t vocab = cfg->vocab_size;
    float eps = 1e-5f;

    /* 1. Token embedding. */
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
        OcPhiLayer *layer = &model->layers[li];

        /* Input norm. */
        float *normed = malloc(h * sizeof(float));
        if (!normed) { free(hidden); return OC_ERR_OOM; }
        if (layer->input_norm) {
            float ss = 0.0f;
            for (size_t i = 0; i < h; i++) ss += hidden[i] * hidden[i];
            float rms = 1.0f / sqrtf(ss / h + eps);
            for (size_t i = 0; i < h; i++) normed[i] = hidden[i] * rms * layer->input_norm[i];
        } else {
            memcpy(normed, hidden, h * sizeof(float));
        }

        /* QKV (dense: all use n_heads * head_dim). */
        float *q = calloc(q_size, sizeof(float));
        float *k = calloc(kv_size, sizeof(float));
        float *v = calloc(kv_size, sizeof(float));
        if (!q || !k || !v) { free(hidden); free(normed); free(q); free(k); free(v); return OC_ERR_OOM; }

        if (layer->wq)
            for (size_t r = 0; r < q_size; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < h; c++) dot += layer->wq[r * h + c] * normed[c];
                q[r] = dot;
            }
        if (layer->wk)
            for (size_t r = 0; r < kv_size; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < h; c++) dot += layer->wk[r * h + c] * normed[c];
                k[r] = dot;
            }
        if (layer->wv)
            for (size_t r = 0; r < kv_size; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < h; c++) dot += layer->wv[r * h + c] * normed[c];
                v[r] = dot;
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

        /* Attention decode (dense: n_kv_heads == n_heads). */
        float *attn_out = calloc(q_size, sizeof(float));
        if (!attn_out) { free(hidden); free(normed); free(q); free(k); free(v); return OC_ERR_OOM; }
        if (s_seq_len > 0)
            oc_flash_attention_decode_heads_f32(q, s_keys, s_values,
                s_seq_len, hd, kv_len, n_heads, n_kv_heads, attn_out);

        /* Output proj + residual. */
        float *attn_resid = calloc(h, sizeof(float));
        if (!attn_resid) { free(hidden); free(normed); free(q); free(k); free(v); free(attn_out); return OC_ERR_OOM; }
        if (layer->wo)
            for (size_t r = 0; r < h; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < q_size; c++) dot += layer->wo[r * q_size + c] * attn_out[c];
                attn_resid[r] = dot;
            }
        for (size_t i = 0; i < h; i++) hidden[i] += attn_resid[i];

        /* Post-attention norm. */
        if (layer->post_attn_norm) {
            float ss = 0.0f;
            for (size_t i = 0; i < h; i++) ss += hidden[i] * hidden[i];
            float rms = 1.0f / sqrtf(ss / h + eps);
            for (size_t i = 0; i < h; i++) normed[i] = hidden[i] * rms * layer->post_attn_norm[i];
        } else {
            memcpy(normed, hidden, h * sizeof(float));
        }

        /* GeGLU FFN. */
        float *gate_out = malloc(inter * sizeof(float));
        float *up_out = malloc(inter * sizeof(float));
        float *act = malloc(inter * sizeof(float));
        float *mlp_out = calloc(h, sizeof(float));
        if (!gate_out || !up_out || !act || !mlp_out) {
            free(hidden); free(normed); free(q); free(k); free(v); free(attn_out); free(attn_resid);
            free(gate_out); free(up_out); free(act); free(mlp_out);
            return OC_ERR_OOM;
        }
        if (layer->w_gate)
            for (size_t r = 0; r < inter; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < h; c++) dot += layer->w_gate[r * h + c] * normed[c];
                gate_out[r] = dot;
            }
        if (layer->w_up)
            for (size_t r = 0; r < inter; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < h; c++) dot += layer->w_up[r * h + c] * normed[c];
                up_out[r] = dot;
            }
        for (size_t i = 0; i < inter; i++) act[i] = phi_gelu_tanh(gate_out[i]) * up_out[i];
        if (layer->w_down)
            for (size_t r = 0; r < h; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < inter; c++) dot += layer->w_down[r * inter + c] * act[c];
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

    if (model->output) {
        for (size_t r = 0; r < vocab; r++) {
            float dot = 0.0f;
            for (size_t c = 0; c < h; c++) dot += model->output[r * h + c] * normed[c];
            logits[r] = dot;
        }
    } else {
        memset(logits, 0, vocab * sizeof(float));
    }

    free(hidden);
    free(normed);
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
