/* mistral_arch.c — Mistral architecture forward pass implementation. */
#include "oxidize/mistral_arch.h"
#include "oxidize/flash_attention.h"

#include <math.h>
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

    OcMistralConfig *cfg = &model->config;
    size_t h = cfg->hidden_dim;
    size_t hd = cfg->head_dim;
    size_t n_heads = cfg->n_heads;
    size_t n_kv_heads = cfg->n_kv_heads;
    size_t q_size = n_heads * hd;
    size_t kv_size = n_kv_heads * hd;
    size_t kv_len = n_kv_heads * hd;
    size_t inter = cfg->intermediate_dim;
    size_t vocab = cfg->vocab_size;
    float eps = 1e-5f;

    float *hidden = malloc(h * sizeof(float));
    if (!hidden) return OC_ERR_OOM;
    size_t tok_idx = (size_t)token;
    if (tok_idx >= vocab) tok_idx = vocab - 1;
    if (model->tok_emb)
        memcpy(hidden, model->tok_emb + tok_idx * h, h * sizeof(float));
    else
        memset(hidden, 0, h * sizeof(float));

    /* Session-scoped KV cache (stored on the model, not static globals). */
    if (model->kv_cache_cap == 0) {
        model->kv_cache_cap = 256;
        size_t kv_bytes = model->kv_cache_cap * kv_len * sizeof(float);
        model->kv_cache_k = calloc(cfg->n_layers, sizeof(float *));
        model->kv_cache_v = calloc(cfg->n_layers, sizeof(float *));
        if (!model->kv_cache_k || !model->kv_cache_v) {
            free(hidden);
            return OC_ERR_OOM;
        }
        for (uint32_t li = 0; li < cfg->n_layers; li++) {
            model->kv_cache_k[li] = malloc(kv_bytes);
            model->kv_cache_v[li] = malloc(kv_bytes);
            if (!model->kv_cache_k[li] || !model->kv_cache_v[li]) {
                free(hidden);
                return OC_ERR_OOM;
            }
            memset(model->kv_cache_k[li], 0, kv_bytes);
            memset(model->kv_cache_v[li], 0, kv_bytes);
        }
        model->kv_seq_len = 0;
    }

    /* Process each layer. */
    for (uint32_t li = 0; li < cfg->n_layers; li++) {
        OcMistralLayer *layer = &model->layers[li];

        /* Attention norm. */
        float *normed = malloc(h * sizeof(float));
        if (!normed) { free(hidden); return OC_ERR_OOM; }
        if (layer->attention_norm) {
            float ss = 0.0f;
            for (size_t i = 0; i < h; i++) ss += hidden[i] * hidden[i];
            float rms = 1.0f / sqrtf(ss / h + eps);
            for (size_t i = 0; i < h; i++) normed[i] = hidden[i] * rms * layer->attention_norm[i];
        } else {
            memcpy(normed, hidden, h * sizeof(float));
        }

        /* QKV projections. */
        float *q = calloc(q_size, sizeof(float));
        float *k = calloc(kv_size, sizeof(float));
        float *v = calloc(kv_size, sizeof(float));
        if (!q || !k || !v) { free(hidden); free(normed); free(q); free(k); free(v); return OC_ERR_OOM; }

        if (layer->wq) {
            for (size_t r = 0; r < q_size; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < h; c++) dot += layer->wq[r * h + c] * normed[c];
                q[r] = dot;
            }
        }
        if (layer->wk) {
            for (size_t r = 0; r < kv_size; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < h; c++) dot += layer->wk[r * h + c] * normed[c];
                k[r] = dot;
            }
        }
        if (layer->wv) {
            for (size_t r = 0; r < kv_size; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < h; c++) dot += layer->wv[r * h + c] * normed[c];
                v[r] = dot;
            }
        }

        /* RoPE on Q and K. */
        size_t pos = model->kv_seq_len;
        float theta = cfg->rope_theta;
        for (size_t hh = 0; hh < n_heads; hh++) {
            float *qh = q + hh * hd;
            for (size_t d = 0; d < hd; d += 2) {
                float freq = pos / powf(theta, (float)(d / 2) / (float)(hd / 2));
                float c = cosf(freq), s = sinf(freq);
                float q0 = qh[d], q1 = qh[d + 1];
                qh[d] = q0 * c - q1 * s;
                qh[d + 1] = q0 * s + q1 * c;
            }
        }
        for (size_t hh = 0; hh < n_kv_heads; hh++) {
            float *kh = k + hh * hd;
            for (size_t d = 0; d < hd; d += 2) {
                float freq = pos / powf(theta, (float)(d / 2) / (float)(hd / 2));
                float c = cosf(freq), s = sinf(freq);
                float k0 = kh[d], k1 = kh[d + 1];
                kh[d] = k0 * c - k1 * s;
                kh[d + 1] = k0 * s + k1 * c;
            }
        }

        /* Append to per-layer KV cache. */
        if (model->kv_seq_len < model->kv_cache_cap) {
            size_t off = model->kv_seq_len * kv_len;
            memcpy(model->kv_cache_k[li] + off, k, kv_size * sizeof(float));
            memcpy(model->kv_cache_v[li] + off, v, kv_size * sizeof(float));
        }

        /* Flash attention decode with sliding window. */
        float *attn_out = calloc(q_size, sizeof(float));
        if (!attn_out) { free(hidden); free(normed); free(q); free(k); free(v); return OC_ERR_OOM; }

        if (model->kv_seq_len > 0) {
            /* Apply sliding window: only attend to positions within window. */
            size_t window_start = 0;
            if (cfg->sliding_window > 0 && model->kv_seq_len > cfg->sliding_window)
                window_start = model->kv_seq_len - cfg->sliding_window;

            /* Use flash attention with offset. */
            size_t attend_len = model->kv_seq_len - window_start;
            const float *k_ptr = model->kv_cache_k[li] + window_start * kv_len;
            const float *v_ptr = model->kv_cache_v[li] + window_start * kv_len;

            oc_flash_attention_decode_heads_f32(
                q, k_ptr, v_ptr, attend_len, hd, kv_len,
                n_heads, n_kv_heads, attn_out);
        }

        /* Output projection + residual. */
        float *attn_resid = calloc(h, sizeof(float));
        if (!attn_resid) { free(hidden); free(normed); free(q); free(k); free(v); free(attn_out); return OC_ERR_OOM; }
        if (layer->wo) {
            for (size_t r = 0; r < h; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < q_size; c++) dot += layer->wo[r * q_size + c] * attn_out[c];
                attn_resid[r] = dot;
            }
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

        /* SwiGLU FFN: down(silu(gate(normed)) * up(normed)). */
        float *gate_out = malloc(inter * sizeof(float));
        float *up_out = malloc(inter * sizeof(float));
        float *act = malloc(inter * sizeof(float));
        float *mlp_out = calloc(h, sizeof(float));
        if (!gate_out || !up_out || !act || !mlp_out) {
            free(hidden); free(normed); free(q); free(k); free(v); free(attn_out); free(attn_resid);
            free(gate_out); free(up_out); free(act); free(mlp_out);
            return OC_ERR_OOM;
        }
        if (layer->w_gate) {
            for (size_t r = 0; r < inter; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < h; c++) dot += layer->w_gate[r * h + c] * normed[c];
                gate_out[r] = dot;
            }
        }
        if (layer->w_up) {
            for (size_t r = 0; r < inter; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < h; c++) dot += layer->w_up[r * h + c] * normed[c];
                up_out[r] = dot;
            }
        }
        for (size_t i = 0; i < inter; i++) {
            float silu_val = gate_out[i] / (1.0f + expf(-gate_out[i]));
            act[i] = silu_val * up_out[i];
        }
        if (layer->w_down) {
            for (size_t r = 0; r < h; r++) {
                float dot = 0.0f;
                for (size_t c = 0; c < inter; c++) dot += layer->w_down[r * inter + c] * act[c];
                mlp_out[r] = dot;
            }
        }
        for (size_t i = 0; i < h; i++) hidden[i] += mlp_out[i];

        free(normed); free(q); free(k); free(v); free(attn_out); free(attn_resid);
        free(gate_out); free(up_out); free(act); free(mlp_out);
    }

    /* Advance sequence position after all layers processed. */
    if (model->kv_seq_len < model->kv_cache_cap)
        model->kv_seq_len++;

    /* Final norm + output projection. */
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

    /* LM head. */
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

void oc_mistral_free(OcMistralModel *model)
{
    if (!model) return;
    free(model->layers);
    free(model->tok_emb);
    free(model->output_norm);
    free(model->output);
    if (model->kv_cache_k) {
        for (uint32_t i = 0; i < model->config.n_layers; i++)
            free(model->kv_cache_k[i]);
        free(model->kv_cache_k);
    }
    if (model->kv_cache_v) {
        for (uint32_t i = 0; i < model->config.n_layers; i++)
            free(model->kv_cache_v[i]);
        free(model->kv_cache_v);
    }
    memset(model, 0, sizeof(*model));
}
