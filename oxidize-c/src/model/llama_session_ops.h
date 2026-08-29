/*
 * llama_session_ops.h — private helpers shared by the OcLlamaSession
 * forward implementations (llama.c, arch_forward.c, glm_arch.c).
 *
 * These previously existed as byte-identical static copies in each
 * translation unit. They are internal to src/model/ — not installed,
 * not part of the public API.
 */
#ifndef OXIDIZE_C_MODEL_LLAMA_SESSION_OPS_H
#define OXIDIZE_C_MODEL_LLAMA_SESSION_OPS_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "oxidize/llama.h"
#include "oxidize/matvec.h"
#include "oxidize/quant.h"

/* Dequantize row `token` of tok_embeddings into `dst` (length n_embd),
 * clamping the token id to the vocabulary. */
static inline void oc_llama_embed_token_into(OcLlamaSession *s,
                                             uint32_t token, float *dst)
{
    OcWeightView *w = &s->model->tok_embeddings;
    if (token >= s->model->cfg.vocab_size) token = s->model->cfg.vocab_size - 1;
    if (w->qtype == OC_QUANT_F32) {
        memcpy(dst, w->data + (size_t)token * w->row_bytes,
               s->model->cfg.n_embd * sizeof(float));
    } else {
        oc_quant_dequant_row(w->qtype,
            w->data + (size_t)token * w->row_bytes, w->row_bytes,
            dst, s->model->cfg.n_embd);
    }
}

/* s->x variant used by the single-token forward paths. */
static inline void oc_llama_embed_token(OcLlamaSession *s, uint32_t token)
{
    oc_llama_embed_token_into(s, token, s->x);
}

/* matvec wrapper: pick f32 or quantized path based on qtype. */
static inline void oc_llama_matvec(const OcWeightView *w, const float *in,
                                   float *out, float *temp)
{
    if (w->qtype == OC_QUANT_F32) {
        oc_matvec_f32((const float *)w->data, w->rows, w->cols, in, out);
    } else {
        oc_matvec_quantized(w->qtype, w->data, w->rows, w->cols, w->row_bytes,
                            in, out, temp);
    }
}

/* One attention head with online softmax (Milakov & Gimelshein 2018),
 * GQA head → KV-head mapping, fp32 KV cache. seq_len = pos + 1. */
static inline void oc_llama_attention_head(const OcLlamaSession *s,
                                           uint32_t head, uint32_t layer,
                                           const float *q_vec,
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

#endif /* OXIDIZE_C_MODEL_LLAMA_SESSION_OPS_H */
