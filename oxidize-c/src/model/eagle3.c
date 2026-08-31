/*
 * eagle3.c — Eagle-3 speculative decoding implementation.
 */
#include "oxidize/eagle3.h"
#include "oxidize/flash_attention.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

OcError oc_eagle_config_init(OcEagleConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->max_draft_tokens = 4;
    cfg->n_layers = 2;
    cfg->hidden_dim = 1024;
    cfg->acceptance_threshold = 0.0f;
    cfg->dynamic_draft = true;
    return OC_OK;
}

OcError oc_eagle_state_init(OcEagleState *state, const OcEagleConfig *cfg)
{
    if (!state) return OC_ERR_INVALID_ARG;
    memset(state, 0, sizeof(*state));
    if (cfg) state->config = *cfg;
    else oc_eagle_config_init(&state->config);

    uint32_t max = state->config.max_draft_tokens;
    if (max == 0) max = OC_EAGLE_MAX_DRAFT;

    state->draft_tokens = malloc(max * sizeof(uint32_t));
    state->draft_probs = malloc(max * sizeof(float));
    state->hidden_states = calloc(state->config.hidden_dim, sizeof(float));
    if (!state->draft_tokens || !state->draft_probs || !state->hidden_states) {
        free(state->draft_tokens);
        free(state->draft_probs);
        free(state->hidden_states);
        return OC_ERR_OOM;
    }
    state->n_draft = 0;
    state->initialized = true;
    return OC_OK;
}

static uint32_t simple_rng(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

OcError oc_eagle_generate_draft(OcEagleState *state, const uint32_t *context,
                               size_t n_context, uint32_t max_tokens)
{
    if (!state || !state->initialized || !context || n_context == 0)
        return OC_ERR_INVALID_ARG;

    uint32_t max = state->config.max_draft_tokens;
    if (max_tokens > 0 && max_tokens < max) max = max_tokens;
    if (max > OC_EAGLE_MAX_DRAFT) max = OC_EAGLE_MAX_DRAFT;

    /* Simple draft generation: use last context token as seed, generate
     * candidate tokens via a simple hash chain. */
    uint32_t rng = context[n_context - 1] ^ 0xDEADBEEFu;
    state->n_draft = 0;
    for (uint32_t i = 0; i < max; i++) {
        uint32_t tok = simple_rng(&rng) % OC_EAGLE_VOCAB_SIZE;
        state->draft_tokens[i] = tok;
        state->draft_probs[i] = 0.5f / (float)(i + 1);
        state->n_draft++;
    }
    return OC_OK;
}

OcError oc_eagle_get_draft_tokens(const OcEagleState *state,
                                  const uint32_t **out_tokens, uint32_t *out_count)
{
    if (!state || !out_tokens || !out_count) return OC_ERR_INVALID_ARG;
    *out_tokens = state->draft_tokens;
    *out_count = state->n_draft;
    return OC_OK;
}

OcError oc_eagle_get_draft_probs(const OcEagleState *state,
                                 const float **out_probs, uint32_t *out_count)
{
    if (!state || !out_probs || !out_count) return OC_ERR_INVALID_ARG;
    *out_probs = state->draft_probs;
    *out_count = state->n_draft;
    return OC_OK;
}

/* Track acceptance statistics. */
static uint32_t g_total_draft = 0;
static uint32_t g_total_accepted = 0;

OcError oc_eagle_update_acceptance(OcEagleState *state, uint32_t n_accepted)
{
    if (!state) return OC_ERR_INVALID_ARG;
    g_total_draft += state->n_draft;
    g_total_accepted += n_accepted;

    /* Dynamic draft: adjust max_draft_tokens based on acceptance rate. */
    if (state->config.dynamic_draft && g_total_draft > 20) {
        float rate = (float)g_total_accepted / (float)g_total_draft;
        if (rate > 0.8f && state->config.max_draft_tokens < OC_EAGLE_MAX_DRAFT) {
            state->config.max_draft_tokens++;
        } else if (rate < 0.3f && state->config.max_draft_tokens > 1) {
            state->config.max_draft_tokens--;
        }
    }
    return OC_OK;
}

uint32_t oc_eagle_n_draft(const OcEagleState *state)
{
    return state ? state->n_draft : 0;
}

float oc_eagle_acceptance_rate(const OcEagleState *state)
{
    (void)state;
    if (g_total_draft == 0) return 0.0f;
    return (float)g_total_accepted / (float)g_total_draft;
}

void oc_eagle_state_free(OcEagleState *state)
{
    if (!state) return;
    free(state->draft_tokens);
    free(state->draft_probs);
    free(state->hidden_states);
    memset(state, 0, sizeof(*state));
}


void oc_eagle3_config_init(OcEagle3Config *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->hidden_size = 4096;
    cfg->num_hidden_layers = 1;
    cfg->extract_layers[0] = 1;
    cfg->extract_layers[1] = 16;
    cfg->extract_layers[2] = 32;
    cfg->n_extract_layers = 3;
    cfg->target_hidden_size = 4096;
    cfg->norm_before_residual = false;
    cfg->vocab_size = 128256;
    cfg->draft_vocab_size = 128256;
    cfg->num_attention_heads = 32;
    cfg->num_key_value_heads = 32;
    cfg->head_dim = 0;
    cfg->intermediate_size = 4 * 4096;
    cfg->rms_norm_eps = 1e-5f;
    cfg->rope_theta = 10000.0f;
}

size_t oc_eagle3_config_head_dim(const OcEagle3Config *cfg)
{
    if (!cfg) return 0;
    if (cfg->head_dim > 0) return cfg->head_dim;
    if (cfg->num_attention_heads == 0) return 0;
    return cfg->hidden_size / cfg->num_attention_heads;
}

size_t oc_eagle3_config_encoder_input_width(const OcEagle3Config *cfg)
{
    if (!cfg) return 0;
    return cfg->n_extract_layers * cfg->target_hidden_size;
}

/* Simple GEMV: out[m] = W[m,n] * x[n] */
static void eagle3_gemv(const float *w, size_t rows, size_t cols,
                         const float *x, float *out)
{
    for (size_t r = 0; r < rows; r++) {
        const float *wrow = w + r * cols;
        float dot = 0.0f;
        for (size_t c = 0; c < cols; c++)
            dot += wrow[c] * x[c];
        out[r] = dot;
    }
}

/* RMSNorm: out = x / rms(x) * weight */
static void eagle3_rms_norm(const float *x, const float *weight, float *out,
                             size_t n, float eps)
{
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = 1.0f / sqrtf(ss / n + eps);
    for (size_t i = 0; i < n; i++) out[i] = x[i] * rms * weight[i];
}

/* Apply RoPE to q and k for a single position. */
static void eagle3_apply_rope(float *q, float *k, size_t n_heads, size_t n_kv_heads,
                               size_t head_dim, size_t pos, float theta)
{
    for (size_t h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
        for (size_t d = 0; d < head_dim; d += 2) {
            float freq = pos / powf(theta, (float)(d / 2) / (float)(head_dim / 2));
            float c = cosf(freq), s = sinf(freq);
            float q0 = qh[d], q1 = qh[d + 1];
            qh[d]     = q0 * c - q1 * s;
            qh[d + 1] = q0 * s + q1 * c;
        }
    }
    for (size_t h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (size_t d = 0; d < head_dim; d += 2) {
            float freq = pos / powf(theta, (float)(d / 2) / (float)(head_dim / 2));
            float c = cosf(freq), s = sinf(freq);
            float k0 = kh[d], k1 = kh[d + 1];
            kh[d]     = k0 * c - k1 * s;
            kh[d + 1] = k0 * s + k1 * c;
        }
    }
}

/* SiLU: x * sigmoid(x) */
static float silu(float x)
{
    return x / (1.0f + expf(-x));
}

OcError oc_eagle3_model_init(OcEagle3DraftModel *m, const OcEagle3Config *cfg)
{
    if (!m || !cfg) return OC_ERR_INVALID_ARG;
    memset(m, 0, sizeof(*m));
    m->config = *cfg;

    size_t h = cfg->hidden_size;
    size_t hd = oc_eagle3_config_head_dim(cfg);
    size_t n_kv = cfg->num_hidden_layers;

    m->g_embeddings = calloc(h, sizeof(float));
    m->kv_caches = calloc(n_kv, sizeof(OcEagle3KvCache));
    if (!m->g_embeddings || !m->kv_caches) {
        oc_eagle3_model_free(m);
        return OC_ERR_OOM;
    }

    /* Pre-allocate KV cache capacity for 256 tokens. */
    size_t kv_len = cfg->num_key_value_heads * hd;
    size_t init_cap = 256;
    for (size_t i = 0; i < n_kv; i++) {
        m->kv_caches[i].keys = calloc(init_cap * kv_len, sizeof(float));
        m->kv_caches[i].values = calloc(init_cap * kv_len, sizeof(float));
        m->kv_caches[i].capacity = init_cap;
        if (!m->kv_caches[i].keys || !m->kv_caches[i].values) {
            oc_eagle3_model_free(m);
            return OC_ERR_OOM;
        }
    }

    m->position_offset = 0;
    m->loaded = true;
    return OC_OK;
}

OcError oc_eagle3_encode_features(OcEagle3DraftModel *m,
                                   const float *target_features, size_t n_features)
{
    if (!m || !target_features || !m->loaded)
        return OC_ERR_INVALID_ARG;

    size_t expected = oc_eagle3_config_encoder_input_width(&m->config);
    if (n_features != expected) return OC_ERR_INVALID_ARG;

    size_t h = m->config.hidden_size;
    if (m->fc.data && m->fc.rows == h && m->fc.cols == expected) {
        eagle3_gemv(m->fc.data, h, expected, target_features, m->g_embeddings);
    } else {
        /* No fc weight loaded - just copy first h elements. */
        size_t copy_len = n_features < h ? n_features : h;
        memcpy(m->g_embeddings, target_features, copy_len * sizeof(float));
    }
    return OC_OK;
}

OcError oc_eagle3_logits_from_hidden(const OcEagle3DraftModel *m,
                                      const float *hidden, size_t hidden_len,
                                      float *out_logits, size_t logits_len)
{
    if (!m || !hidden || !out_logits || !m->loaded)
        return OC_ERR_INVALID_ARG;

    size_t h = m->config.hidden_size;
    if (hidden_len != h) return OC_ERR_INVALID_ARG;

    /* RMSNorm with output_norm. */
    float *normed = malloc(h * sizeof(float));
    if (!normed) return OC_ERR_OOM;

    if (m->output_norm) {
        eagle3_rms_norm(hidden, m->output_norm, normed, h, m->config.rms_norm_eps);
    } else {
        memcpy(normed, hidden, h * sizeof(float));
    }

    /* LM head GEMV. */
    size_t dv = m->config.draft_vocab_size;
    if (m->output.data && m->output.rows == dv && m->output.cols == h) {
        eagle3_gemv(m->output.data, dv, h, normed, out_logits);
    } else {
        memset(out_logits, 0, logits_len * sizeof(float));
    }

    free(normed);

    if (m->d2t && m->n_d2t > 0 && logits_len >= m->config.vocab_size) {
        float *scattered = malloc(m->config.vocab_size * sizeof(float));
        if (!scattered) return OC_ERR_OOM;
        for (size_t i = 0; i < m->config.vocab_size; i++)
            scattered[i] = -INFINITY;
        for (size_t i = 0; i < m->n_d2t && i < dv; i++) {
            uint64_t target_idx = m->d2t[i];
            if (target_idx < m->config.vocab_size)
                scattered[target_idx] = out_logits[i];
        }
        memcpy(out_logits, scattered, m->config.vocab_size * sizeof(float));
        free(scattered);
    }

    return OC_OK;
}

OcError oc_eagle3_forward_decoder(OcEagle3DraftModel *m,
                                   uint32_t token,
                                   float *out_hidden, size_t hidden_len,
                                   float *out_logits, size_t logits_len)
{
    if (!m || !m->loaded || !out_hidden)
        return OC_ERR_INVALID_ARG;

    OcEagle3Config *cfg = &m->config;
    size_t h = cfg->hidden_size;
    if (hidden_len != h) return OC_ERR_INVALID_ARG;

    size_t hd = oc_eagle3_config_head_dim(cfg);
    size_t n_heads = cfg->num_attention_heads;
    size_t n_kv_heads = cfg->num_key_value_heads;
    size_t q_size = n_heads * hd;
    size_t kv_size = n_kv_heads * hd;
    size_t kv_len = n_kv_heads * hd;

    float *embd = malloc(h * sizeof(float));
    if (!embd) return OC_ERR_OOM;
    if (m->tok_embeddings.data && m->tok_embeddings.rows > token) {
        memcpy(embd, m->tok_embeddings.data + token * h, h * sizeof(float));
    } else {
        memset(embd, 0, h * sizeof(float));
    }

    float *embd_norm = malloc(h * sizeof(float));
    if (!embd_norm) { free(embd); return OC_ERR_OOM; }
    if (m->layer.attn_norm)
        eagle3_rms_norm(embd, m->layer.attn_norm, embd_norm, h, cfg->rms_norm_eps);
    else
        memcpy(embd_norm, embd, h * sizeof(float));

    float *g_norm = malloc(h * sizeof(float));
    if (!g_norm) { free(embd); free(embd_norm); return OC_ERR_OOM; }
    if (m->layer.attn_norm_2)
        eagle3_rms_norm(m->g_embeddings, m->layer.attn_norm_2, g_norm, h, cfg->rms_norm_eps);
    else
        memcpy(g_norm, m->g_embeddings, h * sizeof(float));

    float *residual = cfg->norm_before_residual ? g_norm : m->g_embeddings;

    size_t concat_len = 2 * h;
    float *concat = malloc(concat_len * sizeof(float));
    if (!concat) { free(embd); free(embd_norm); free(g_norm); return OC_ERR_OOM; }
    memcpy(concat, embd_norm, h * sizeof(float));
    memcpy(concat + h, g_norm, h * sizeof(float));

    float *q = malloc(q_size * sizeof(float));
    float *k = malloc(kv_size * sizeof(float));
    float *v = malloc(kv_size * sizeof(float));
    if (!q || !k || !v) {
        free(embd); free(embd_norm); free(g_norm); free(concat);
        free(q); free(k); free(v);
        return OC_ERR_OOM;
    }

    if (m->layer.attention.q_proj.data)
        eagle3_gemv(m->layer.attention.q_proj.data, q_size, concat_len, concat, q);
    else memset(q, 0, q_size * sizeof(float));
    if (m->layer.attention.k_proj.data)
        eagle3_gemv(m->layer.attention.k_proj.data, kv_size, concat_len, concat, k);
    else memset(k, 0, kv_size * sizeof(float));
    if (m->layer.attention.v_proj.data)
        eagle3_gemv(m->layer.attention.v_proj.data, kv_size, concat_len, concat, v);
    else memset(v, 0, kv_size * sizeof(float));

    eagle3_apply_rope(q, k, n_heads, n_kv_heads, hd, m->position_offset, cfg->rope_theta);

    OcEagle3KvCache *cache = &m->kv_caches[0];
    if (cache->seq_len < cache->capacity) {
        size_t offset = cache->seq_len * kv_len;
        memcpy(cache->keys + offset, k, kv_size * sizeof(float));
        memcpy(cache->values + offset, v, kv_size * sizeof(float));
        cache->seq_len++;
    }

    float *attn_out = malloc(q_size * sizeof(float));
    if (!attn_out) {
        free(embd); free(embd_norm); free(g_norm); free(concat);
        free(q); free(k); free(v);
        return OC_ERR_OOM;
    }
    if (cache->seq_len > 0) {
        oc_flash_attention_decode_heads_f32(
            q, cache->keys, cache->values,
            cache->seq_len, hd, kv_len,
            n_heads, n_kv_heads, attn_out);
    } else {
        memset(attn_out, 0, q_size * sizeof(float));
    }

    float *hidden = malloc(h * sizeof(float));
    if (!hidden) {
        free(embd); free(embd_norm); free(g_norm); free(concat);
        free(q); free(k); free(v); free(attn_out);
        return OC_ERR_OOM;
    }
    if (m->layer.attention.o_proj.data && m->layer.attention.o_proj.rows == h)
        eagle3_gemv(m->layer.attention.o_proj.data, h, q_size, attn_out, hidden);
    else if (q_size == h)
        memcpy(hidden, attn_out, h * sizeof(float));
    else
        memset(hidden, 0, h * sizeof(float));

    for (size_t i = 0; i < h; i++) hidden[i] += residual[i];

    float *normed_ffn = malloc(h * sizeof(float));
    if (!normed_ffn) {
        free(embd); free(embd_norm); free(g_norm); free(concat);
        free(q); free(k); free(v); free(attn_out); free(hidden);
        return OC_ERR_OOM;
    }
    if (m->layer.ffn_norm)
        eagle3_rms_norm(hidden, m->layer.ffn_norm, normed_ffn, h, cfg->rms_norm_eps);
    else
        memcpy(normed_ffn, hidden, h * sizeof(float));

    size_t inter = cfg->intermediate_size;
    float *gate_out = malloc(inter * sizeof(float));
    float *up_out = malloc(inter * sizeof(float));
    float *act = malloc(inter * sizeof(float));
    if (!gate_out || !up_out || !act) {
        free(embd); free(embd_norm); free(g_norm); free(concat);
        free(q); free(k); free(v); free(attn_out); free(hidden); free(normed_ffn);
        free(gate_out); free(up_out); free(act);
        return OC_ERR_OOM;
    }
    if (m->layer.mlp_gate.data)
        eagle3_gemv(m->layer.mlp_gate.data, inter, h, normed_ffn, gate_out);
    else memset(gate_out, 0, inter * sizeof(float));
    if (m->layer.mlp_up.data)
        eagle3_gemv(m->layer.mlp_up.data, inter, h, normed_ffn, up_out);
    else memset(up_out, 0, inter * sizeof(float));
    for (size_t i = 0; i < inter; i++) act[i] = silu(gate_out[i]) * up_out[i];

    float *mlp_out = malloc(h * sizeof(float));
    if (!mlp_out) {
        free(embd); free(embd_norm); free(g_norm); free(concat);
        free(q); free(k); free(v); free(attn_out); free(hidden); free(normed_ffn);
        free(gate_out); free(up_out); free(act);
        return OC_ERR_OOM;
    }
    if (m->layer.mlp_down.data)
        eagle3_gemv(m->layer.mlp_down.data, h, inter, act, mlp_out);
    else memset(mlp_out, 0, h * sizeof(float));
    for (size_t i = 0; i < h; i++) hidden[i] += mlp_out[i];

    memcpy(m->g_embeddings, hidden, h * sizeof(float));
    m->position_offset++;

    /* Copy hidden to output. */
    memcpy(out_hidden, hidden, h * sizeof(float));

    if (out_logits && logits_len > 0) {
        oc_eagle3_logits_from_hidden(m, hidden, h, out_logits, logits_len);
    }

    /* Cleanup. */
    free(embd); free(embd_norm); free(g_norm); free(concat);
    free(q); free(k); free(v); free(attn_out);
    free(hidden); free(normed_ffn);
    free(gate_out); free(up_out); free(act); free(mlp_out);

    return OC_OK;
}

void oc_eagle3_reset_cache(OcEagle3DraftModel *m)
{
    if (!m) return;
    for (size_t i = 0; i < m->config.num_hidden_layers; i++) {
        m->kv_caches[i].seq_len = 0;
    }
    m->position_offset = 0;
}

OcError oc_eagle3_reserve_cache_tokens(OcEagle3DraftModel *m, size_t n_tokens)
{
    if (!m) return OC_ERR_INVALID_ARG;
    size_t hd = oc_eagle3_config_head_dim(&m->config);
    size_t kv_len = m->config.num_key_value_heads * hd;
    for (size_t i = 0; i < m->config.num_hidden_layers; i++) {
        OcEagle3KvCache *c = &m->kv_caches[i];
        if (n_tokens > c->capacity) {
            float *new_k = realloc(c->keys, n_tokens * kv_len * sizeof(float));
            float *new_v = realloc(c->values, n_tokens * kv_len * sizeof(float));
            if (!new_k || !new_v) {
                free(new_k); free(new_v);
                return OC_ERR_OOM;
            }
            c->keys = new_k;
            c->values = new_v;
            c->capacity = n_tokens;
        }
    }
    return OC_OK;
}

void oc_eagle3_model_free(OcEagle3DraftModel *m)
{
    if (!m) return;
    free(m->fc.data);
    free(m->d2t);
    free(m->layer.attn_norm);
    free(m->layer.attn_norm_2);
    free(m->layer.attention.q_proj.data);
    free(m->layer.attention.k_proj.data);
    free(m->layer.attention.v_proj.data);
    free(m->layer.attention.o_proj.data);
    free(m->layer.ffn_norm);
    free(m->layer.mlp_gate.data);
    free(m->layer.mlp_up.data);
    free(m->layer.mlp_down.data);
    free(m->output_norm);
    free(m->output.data);
    free(m->tok_embeddings.data);
    free(m->g_embeddings);
    if (m->kv_caches) {
        for (size_t i = 0; i < m->config.num_hidden_layers; i++) {
            free(m->kv_caches[i].keys);
            free(m->kv_caches[i].values);
        }
        free(m->kv_caches);
    }
    memset(m, 0, sizeof(*m));
}
