/*
 * dflash.c — DFlash speculative decoding implementation.
 */
#include "oxidize/dflash.h"
#include "arch_ops.h"
#include "oxidize/flash_attention.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

OcError oc_dflash_config_init(OcDFlashConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->max_draft_tokens = 8;
    cfg->verification_window = 16;
    cfg->acceptance_threshold = 0.0f;
    cfg->adaptive = true;
    return OC_OK;
}

OcError oc_dflash_state_init(OcDFlashState *state, const OcDFlashConfig *cfg)
{
    if (!state) return OC_ERR_INVALID_ARG;
    memset(state, 0, sizeof(*state));
    if (cfg) state->config = *cfg;
    else oc_dflash_config_init(&state->config);
    return OC_OK;
}

OcError oc_dflash_set_draft(OcDFlashState *state, const uint32_t *tokens,
                           const float *logprobs, uint32_t n)
{
    if (!state || !tokens || !logprobs) return OC_ERR_INVALID_ARG;
    if (n > OC_DFLASH_MAX_DRAFT) n = OC_DFLASH_MAX_DRAFT;
    if (n > state->config.max_draft_tokens) n = state->config.max_draft_tokens;
    memcpy(state->draft_tokens, tokens, n * sizeof(uint32_t));
    memcpy(state->draft_logprobs, logprobs, n * sizeof(float));
    state->n_draft = n;
    state->n_accepted = 0;
    return OC_OK;
}

OcError oc_dflash_set_target(OcDFlashState *state, const uint32_t *tokens,
                            const float *logprobs, uint32_t n)
{
    if (!state || !tokens || !logprobs) return OC_ERR_INVALID_ARG;
    if (n > OC_DFLASH_MAX_DRAFT) n = OC_DFLASH_MAX_DRAFT;
    memcpy(state->target_tokens, tokens, n * sizeof(uint32_t));
    memcpy(state->target_logprobs, logprobs, n * sizeof(float));
    return OC_OK;
}

OcError oc_dflash_verify(OcDFlashState *state, uint32_t *out_accepted, uint32_t *out_n)
{
    if (!state || !out_accepted || !out_n) return OC_ERR_INVALID_ARG;
    *out_n = 0;
    uint32_t accepted = 0;

    for (uint32_t i = 0; i < state->n_draft; i++) {
        if (state->draft_tokens[i] == state->target_tokens[i]) {
            out_accepted[accepted++] = state->target_tokens[i];
            /* Check if we should continue (stochastic acceptance). */
            if (state->config.acceptance_threshold > 0.0f) {
                float ratio = expf(state->target_logprobs[i] - state->draft_logprobs[i]);
                if (ratio < state->config.acceptance_threshold) break;
            }
        } else {
            /* Mismatch: replace with target token and stop. */
            out_accepted[accepted++] = state->target_tokens[i];
            break;
        }
    }

    state->n_accepted = accepted;
    state->total_accepted += accepted;
    state->total_proposed += state->n_draft;
    *out_n = accepted;

    /* Adaptive: adjust max_draft_tokens based on acceptance rate. */
    if (state->config.adaptive && state->total_proposed > 20) {
        float rate = (float)state->total_accepted / (float)state->total_proposed;
        if (rate > 0.7f && state->config.max_draft_tokens < OC_DFLASH_MAX_DRAFT) {
            state->config.max_draft_tokens++;
        } else if (rate < 0.3f && state->config.max_draft_tokens > 1) {
            state->config.max_draft_tokens--;
        }
    }

    return OC_OK;
}

OcError oc_dflash_get_accepted(const OcDFlashState *state, const uint32_t **out_tokens, uint32_t *out_n)
{
    if (!state || !out_tokens || !out_n) return OC_ERR_INVALID_ARG;
    *out_tokens = state->target_tokens; /* accepted tokens are at the start */
    *out_n = state->n_accepted;
    return OC_OK;
}

float oc_dflash_acceptance_rate(const OcDFlashState *state)
{
    if (!state || state->total_proposed == 0) return 0.0f;
    return (float)state->total_accepted / (float)state->total_proposed;
}

uint32_t oc_dflash_avg_acceptance(const OcDFlashState *state)
{
    if (!state || state->total_proposed == 0) return 0;
    return (uint32_t)(100.0f * (float)state->total_accepted / (float)state->total_proposed);
}

void oc_dflash_state_free(OcDFlashState *state)
{
    if (!state) return;
    memset(state, 0, sizeof(*state));
}

/* ─── Real DFlash draft model (port of dflash.rs) ───────────────────── */

void oc_dflash_model_config_init(OcDFlashModelConfig *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->hidden_size = 2048;
    cfg->num_hidden_layers = 8;
    cfg->num_target_layers = 40;
    cfg->block_size = 16;
    cfg->target_layer_ids[0] = 1;
    cfg->target_layer_ids[1] = 10;
    cfg->target_layer_ids[2] = 19;
    cfg->target_layer_ids[3] = 28;
    cfg->target_layer_ids[4] = 37;
    cfg->n_target_layer_ids = 5;
    cfg->mask_token_id = 248070;
    cfg->vocab_size = 248320;
    cfg->num_attention_heads = 32;
    cfg->num_key_value_heads = 8;
    cfg->intermediate_size = 8192;
    cfg->rms_norm_eps = 1e-5f;
    cfg->rope_theta = 10000.0f;
}

size_t oc_dflash_config_head_dim(const OcDFlashModelConfig *cfg)
{
    if (!cfg || cfg->num_attention_heads == 0) return 0;
    return cfg->hidden_size / cfg->num_attention_heads;
}

size_t oc_dflash_config_target_hidden_width(const OcDFlashModelConfig *cfg)
{
    if (!cfg) return 0;
    return cfg->hidden_size * cfg->num_target_layers;
}

OcError oc_dflash_model_init(OcDFlashDraftModel *m, const OcDFlashModelConfig *cfg)
{
    if (!m || !cfg) return OC_ERR_INVALID_ARG;
    memset(m, 0, sizeof(*m));
    m->config = *cfg;

    size_t n_layers = cfg->num_hidden_layers;
    size_t hd = oc_dflash_config_head_dim(cfg);
    size_t kv_len = cfg->num_key_value_heads * hd;

    m->layers = calloc(n_layers, sizeof(OcDFlashDecoderLayer));
    m->kv_cache = calloc(n_layers, sizeof(OcDFlashKvCache));
    m->n_layers = n_layers;
    if (!m->layers || !m->kv_cache) {
        oc_dflash_model_free(m);
        return OC_ERR_OOM;
    }

    /* Pre-allocate KV caches. */
    size_t init_cap = 256;
    for (size_t i = 0; i < n_layers; i++) {
        m->kv_cache[i].keys = calloc(init_cap * kv_len, sizeof(float));
        m->kv_cache[i].values = calloc(init_cap * kv_len, sizeof(float));
        m->kv_cache[i].capacity = init_cap;
        if (!m->kv_cache[i].keys || !m->kv_cache[i].values) {
            oc_dflash_model_free(m);
            return OC_ERR_OOM;
        }
    }

    m->position_offset = 0;
    m->loaded = true;
    return OC_OK;
}

OcError oc_dflash_cache_target_hidden(OcDFlashDraftModel *m,
                                       const float *hidden, size_t len)
{
    if (!m || !hidden) return OC_ERR_INVALID_ARG;
    size_t expected = oc_dflash_config_target_hidden_width(&m->config);
    if (len != expected) return OC_ERR_INVALID_ARG;

    free(m->target_hidden_cache);
    m->target_hidden_cache = malloc(len * sizeof(float));
    if (!m->target_hidden_cache) return OC_ERR_OOM;
    memcpy(m->target_hidden_cache, hidden, len * sizeof(float));
    m->target_hidden_cache_len = len;
    return OC_OK;
}

void oc_dflash_clear_speculative_caches(OcDFlashDraftModel *m)
{
    if (!m) return;
    free(m->target_hidden_cache);
    m->target_hidden_cache = NULL;
    m->target_hidden_cache_len = 0;
    for (size_t i = 0; i < m->config.num_hidden_layers; i++) {
        m->kv_cache[i].seq_len = 0;
    }
    m->position_offset = 0;
}

OcError oc_dflash_forward_token(OcDFlashDraftModel *m,
                                 uint32_t token,
                                 const float *target_hidden, size_t target_hidden_len,
                                 float *out_hidden, size_t hidden_len)
{
    if (!m || !m->loaded || !out_hidden)
        return OC_ERR_INVALID_ARG;

    OcDFlashModelConfig *cfg = &m->config;
    size_t h = cfg->hidden_size;
    if (hidden_len != h) return OC_ERR_INVALID_ARG;

    size_t hd = oc_dflash_config_head_dim(cfg);
    size_t n_heads = cfg->num_attention_heads;
    size_t n_kv_heads = cfg->num_key_value_heads;
    size_t q_size = n_heads * hd;
    size_t kv_size = n_kv_heads * hd;
    size_t kv_len = n_kv_heads * hd;

    /* 1. Token embedding. */
    float *hidden = malloc(h * sizeof(float));
    if (!hidden) return OC_ERR_OOM;
    if (m->tok_embeddings.data && m->tok_embeddings.rows > token) {
        memcpy(hidden, m->tok_embeddings.data + token * h, h * sizeof(float));
    } else {
        memset(hidden, 0, h * sizeof(float));
    }

    /* 2. Optional target hidden fusion. */
    float *target_context = NULL;
    if (target_hidden && target_hidden_len > 0 && m->fc.data) {
        float *fused = malloc(h * sizeof(float));
        if (!fused) { free(hidden); return OC_ERR_OOM; }
        oc_arch_matvec(m->fc.data, target_hidden, fused, h, target_hidden_len);
        if (m->fc_bias) {
            for (size_t i = 0; i < h; i++) fused[i] += m->fc_bias[i];
        }
        if (m->hidden_norm) {
            float *normed = malloc(h * sizeof(float));
            if (!normed) { free(hidden); free(fused); return OC_ERR_OOM; }
            oc_arch_rms_norm(fused, m->hidden_norm, normed, h, cfg->rms_norm_eps);
            free(fused);
            target_context = normed;
        } else {
            target_context = fused;
        }
    }

    /* 3. Layer loop. */
    for (size_t li = 0; li < m->n_layers; li++) {
        OcDFlashDecoderLayer *layer = &m->layers[li];

        /* Determine head_dim from q_norm if available. */
        size_t layer_hd = hd;
        /* q_norm_weight length gives head_dim */

        /* Attention branch. */
        float *normed = malloc(h * sizeof(float));
        if (!normed) { free(hidden); free(target_context); return OC_ERR_OOM; }
        if (layer->input_layernorm)
            oc_arch_rms_norm(hidden, layer->input_layernorm, normed, h, cfg->rms_norm_eps);
        else
            memcpy(normed, hidden, h * sizeof(float));

        /* QKV. */
        float *q = calloc(q_size, sizeof(float));
        float *k = calloc(kv_size, sizeof(float));
        float *v = calloc(kv_size, sizeof(float));
        if (!q || !k || !v) {
            free(hidden); free(target_context); free(normed);
            free(q); free(k); free(v);
            return OC_ERR_OOM;
        }
        if (layer->attention.q_proj.data)
            oc_arch_matvec(layer->attention.q_proj.data, normed, q, q_size, h);
        if (layer->attention.k_proj.data)
            oc_arch_matvec(layer->attention.k_proj.data, normed, k, kv_size, h);
        if (layer->attention.v_proj.data)
            oc_arch_matvec(layer->attention.v_proj.data, normed, v, kv_size, h);

        /* RoPE. */
        oc_arch_rope_qk(q, k, n_heads, n_kv_heads, layer_hd, m->position_offset, cfg->rope_theta);

        /* KV cache append. */
        OcDFlashKvCache *cache = &m->kv_cache[li];
        if (cache->seq_len < cache->capacity) {
            size_t off = cache->seq_len * kv_len;
            memcpy(cache->keys + off, k, kv_size * sizeof(float));
            memcpy(cache->values + off, v, kv_size * sizeof(float));
            cache->seq_len++;
        }

        /* Attention decode. */
        float *attn_out = calloc(q_size, sizeof(float));
        if (!attn_out) {
            free(hidden); free(target_context); free(normed);
            free(q); free(k); free(v);
            return OC_ERR_OOM;
        }
        if (cache->seq_len > 0) {
            oc_flash_attention_decode_heads_f32(
                q, cache->keys, cache->values,
                cache->seq_len, layer_hd, kv_len,
                n_heads, n_kv_heads, attn_out);
        }

        /* Output proj + residual. */
        float *attn_resid = malloc(h * sizeof(float));
        if (!attn_resid) {
            free(hidden); free(target_context); free(normed);
            free(q); free(k); free(v); free(attn_out);
            return OC_ERR_OOM;
        }
        if (layer->attention.o_proj.data && layer->attention.o_proj.rows == h)
            oc_arch_matvec(layer->attention.o_proj.data, attn_out, attn_resid, h, q_size);
        else if (q_size == h)
            memcpy(attn_resid, attn_out, h * sizeof(float));
        else
            memset(attn_resid, 0, h * sizeof(float));
        for (size_t i = 0; i < h; i++) hidden[i] += attn_resid[i];

        /* FFN. */
        float *normed_ffn = malloc(h * sizeof(float));
        if (!normed_ffn) {
            free(hidden); free(target_context); free(normed);
            free(q); free(k); free(v); free(attn_out); free(attn_resid);
            return OC_ERR_OOM;
        }
        if (layer->post_attention_layernorm)
            oc_arch_rms_norm(hidden, layer->post_attention_layernorm, normed_ffn, h, cfg->rms_norm_eps);
        else
            memcpy(normed_ffn, hidden, h * sizeof(float));

        size_t inter = cfg->intermediate_size;
        float *gate_out = malloc(inter * sizeof(float));
        float *up_out = malloc(inter * sizeof(float));
        float *act = malloc(inter * sizeof(float));
        float *mlp_out = malloc(h * sizeof(float));
        if (!gate_out || !up_out || !act || !mlp_out) {
            free(hidden); free(target_context); free(normed);
            free(q); free(k); free(v); free(attn_out); free(attn_resid); free(normed_ffn);
            free(gate_out); free(up_out); free(act); free(mlp_out);
            return OC_ERR_OOM;
        }
        if (layer->mlp_gate.data)
            oc_arch_matvec(layer->mlp_gate.data, normed_ffn, gate_out, inter, h);
        else memset(gate_out, 0, inter * sizeof(float));
        if (layer->mlp_up.data)
            oc_arch_matvec(layer->mlp_up.data, normed_ffn, up_out, inter, h);
        else memset(up_out, 0, inter * sizeof(float));
        for (size_t i = 0; i < inter; i++) act[i] = oc_arch_silu(gate_out[i]) * up_out[i];
        if (layer->mlp_down.data)
            oc_arch_matvec(layer->mlp_down.data, act, mlp_out, h, inter);
        else memset(mlp_out, 0, h * sizeof(float));
        for (size_t i = 0; i < h; i++) hidden[i] += mlp_out[i];

        free(normed); free(q); free(k); free(v); free(attn_out); free(attn_resid);
        free(normed_ffn); free(gate_out); free(up_out); free(act); free(mlp_out);
    }

    /* 4. Final norm. */
    if (m->norm) {
        float *normed = malloc(h * sizeof(float));
        if (!normed) { free(hidden); free(target_context); return OC_ERR_OOM; }
        oc_arch_rms_norm(hidden, m->norm, normed, h, cfg->rms_norm_eps);
        memcpy(hidden, normed, h * sizeof(float));
        free(normed);
    }

    /* 5. Copy output. */
    memcpy(out_hidden, hidden, h * sizeof(float));
    m->position_offset++;

    free(hidden);
    free(target_context);
    return OC_OK;
}

OcError oc_dflash_logits(const OcDFlashDraftModel *m,
                          const float *hidden, size_t hidden_len,
                          float *out_logits, size_t logits_len)
{
    if (!m || !hidden || !out_logits || !m->loaded)
        return OC_ERR_INVALID_ARG;

    size_t h = m->config.hidden_size;
    if (hidden_len != h) return OC_ERR_INVALID_ARG;

    /* RMSNorm + lm_head. */
    float *normed = malloc(h * sizeof(float));
    if (!normed) return OC_ERR_OOM;
    if (m->norm)
        oc_arch_rms_norm(hidden, m->norm, normed, h, m->config.rms_norm_eps);
    else
        memcpy(normed, hidden, h * sizeof(float));

    size_t vocab = m->config.vocab_size;
    if (m->output.data && m->output.rows == vocab && m->output.cols == h)
        oc_arch_matvec(m->output.data, normed, out_logits, vocab, h);
    else
        memset(out_logits, 0, logits_len * sizeof(float));

    free(normed);
    return OC_OK;
}

void oc_dflash_model_free(OcDFlashDraftModel *m)
{
    if (!m) return;
    free(m->fc.data);
    free(m->fc_bias);
    free(m->hidden_norm);
    for (size_t i = 0; i < m->n_layers; i++) {
        OcDFlashDecoderLayer *l = &m->layers[i];
        free(l->input_layernorm);
        free(l->attention.q_proj.data);
        free(l->attention.k_proj.data);
        free(l->attention.v_proj.data);
        free(l->attention.o_proj.data);
        free(l->attention.q_norm_weight);
        free(l->attention.k_norm_weight);
        free(l->post_attention_layernorm);
        free(l->mlp_gate.data);
        free(l->mlp_up.data);
        free(l->mlp_down.data);
    }
    free(m->layers);
    free(m->norm);
    free(m->output.data);
    free(m->tok_embeddings.data);
    if (m->kv_cache) {
        for (size_t i = 0; i < m->config.num_hidden_layers; i++) {
            free(m->kv_cache[i].keys);
            free(m->kv_cache[i].values);
        }
        free(m->kv_cache);
    }
    free(m->target_hidden_cache);
    memset(m, 0, sizeof(*m));
}
