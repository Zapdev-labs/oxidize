/*
 * arch_session_ops.h — private single-token helpers over an OcLlamaSession,
 * shared by the non-Llama forward passes (arch_forward.c, glm_arch.c).
 */
#ifndef OXIDIZE_C_MODEL_ARCH_SESSION_OPS_H
#define OXIDIZE_C_MODEL_ARCH_SESSION_OPS_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "oxidize/llama.h"
#include "oxidize/matvec.h"
#include "oxidize/quant.h"

/* Dequantize row `token` of tok_embeddings into the session's x. */
static inline void oc_arch_embed_token(OcLlamaSession *s, uint32_t token)
{
    OcWeightView *w = &s->model->tok_embeddings;
    if (token >= s->model->cfg.vocab_size) token = s->model->cfg.vocab_size - 1;
    if (w->qtype == OC_QUANT_F32) {
        memcpy(s->x, w->data + (size_t)token * w->row_bytes,
               s->model->cfg.n_embd * sizeof(float));
    } else {
        oc_quant_dequant_row(w->qtype,
            w->data + (size_t)token * w->row_bytes, w->row_bytes,
            s->x, s->model->cfg.n_embd);
    }
}

/* Pick the f32 or quantized matvec from the weight's qtype. */
static inline void oc_arch_weight_matvec(const OcWeightView *w, const float *in, float *out,
                        float *temp)
{
    if (w->qtype == OC_QUANT_F32) {
        oc_matvec_f32((const float *)w->data, w->rows, w->cols, in, out);
    } else {
        oc_matvec_quantized(w->qtype, w->data, w->rows, w->cols,
                            w->row_bytes, in, out, temp);
    }
}

/*
 * Computes attention for a single query head against all cached K/V up to
 * `sess->pos`. Uses the online (streaming) softmax algorithm (Milakov &
 * Gimelshein 2018) so we never need to materialize the full score vector.
 *
 * For multi-query attention (Falcon) or GQA, the `kv_head` index selects
 * which KV head's cache to attend to. For GPT-2 (n_head_kv == n_head) each
 * Q head has its own KV head. For Falcon (n_head_kv == 1) all Q heads
 * share the single KV head (kv_head = 0 for all heads).
 *
 * Parameters:
 *   s        — session (provides KV cache, pos, config)
 *   head     — query head index (0..n_head-1)
 *   layer    — layer index (selects the KV cache slice)
 *   q_vec    — query vector for this head (length head_dim)
 *   out_vec  — output vector (length head_dim), written in place */
static inline void oc_arch_attention_head(const OcLlamaSession *s, uint32_t head,
                                uint32_t layer, const float *q_vec,
                                float *out_vec)
{
    const OcLlamaConfig *c = &s->model->cfg;
    size_t hd = c->head_dim;
    uint32_t group = c->n_head / c->n_head_kv;   /* GQA group size */
    uint32_t kv_head = head / group;              /* which KV head */
    size_t kv_off = ((size_t)layer * c->n_ctx + 0) * s->kv_row_floats
                  + (size_t)kv_head * hd;
    const float *k_layer = s->kv_k;
    const float *v_layer = s->kv_v;
    float scale = 1.0f / sqrtf((float)hd);

    /* Online softmax (Milakov & Gimelshein 2018). seq_len = pos+1. */
    float run_max = -INFINITY;
    float run_sum = 0.0f;
    for (size_t i = 0; i < hd; i++) out_vec[i] = 0.0f;

    int64_t seq_len = s->pos + 1;

    for (int64_t t = 0; t < seq_len; t++) {
        const float *k_t = k_layer + kv_off + (size_t)t * s->kv_row_floats;
        float dot = 0.0f;
        for (size_t i = 0; i < hd; i++) dot += q_vec[i] * k_t[i];
        float score = dot * scale;
        float new_max = (score > run_max) ? score : run_max;
        float exp_factor = expf(run_max - new_max);
        float exp_score = expf(score - new_max);
        for (size_t i = 0; i < hd; i++) out_vec[i] *= exp_factor;
        const float *v_t = v_layer + kv_off + (size_t)t * s->kv_row_floats;
        for (size_t i = 0; i < hd; i++) out_vec[i] += exp_score * v_t[i];
        run_sum = run_sum * exp_factor + exp_score;
        run_max = new_max;
    }
    float inv = 1.0f / run_sum;
    for (size_t i = 0; i < hd; i++) out_vec[i] *= inv;
}

/* The session, its model, and its position within the context window. */
static inline OcError oc_arch_validate_session(OcLlamaSession *sess)
{
    if (sess == NULL || sess->model == NULL) return OC_ERR_INVALID_ARG;
    if ((uint64_t)sess->pos >= sess->model->cfg.n_ctx) return OC_ERR_INVALID_ARG;
    return OC_OK;
}

#endif /* OXIDIZE_C_MODEL_ARCH_SESSION_OPS_H */
