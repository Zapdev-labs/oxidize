#include "oxidize/arch_forward.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "oxidize/activation.h"
#include "oxidize/llama.h"
#include "oxidize/log.h"
#include "oxidize/matvec.h"
#include "oxidize/model.h"
#include "oxidize/quant.h"
#include "oxidize/tensor_ops.h"


/* Dequantize row `token` of tok_embeddings into `dst` (length n_embd). */
static void arch_embed_token(OcLlamaSession *s, uint32_t token)
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

/* matvec wrapper: pick f32 or quantized path based on qtype. */
static void arch_matvec(const OcWeightView *w, const float *in, float *out,
                        float *temp)
{
    if (w->qtype == OC_QUANT_F32) {
        oc_matvec_f32((const float *)w->data, w->rows, w->cols, in, out);
    } else {
        oc_matvec_quantized(w->qtype, w->data, w->rows, w->cols,
                            w->row_bytes, in, out, temp);
    }
}

static void arch_layer_norm(const float *x, const float *weight,
                            const float *bias, float *out, size_t n, float eps)
{
    /* Compute mean. */
    float mean = 0.0f;
    for (size_t i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;

    /* Compute variance (population variance: divide by n, not n-1). */
    float var = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;

    float inv_std = 1.0f / sqrtf(var + eps);

    /* Normalize, scale, shift. */
    for (size_t i = 0; i < n; i++) {
        out[i] = (x[i] - mean) * inv_std * weight[i]
               + (bias ? bias[i] : 0.0f);
    }
}

static void arch_gelu_inplace_f32(float *buf, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        buf[i] = oc_gelu_approx_f32(buf[i]);
    }
}

static void arch_attention_head(const OcLlamaSession *s, uint32_t head,
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

static void arch_gpt2_layer(OcLlamaSession *s, uint32_t layer)
{
    const OcLlamaConfig *c = &s->model->cfg;
    OcLlamaLayer *L = &s->model->layers[layer];
    size_t hd = c->head_dim;
    size_t n_embd = c->n_embd;

    arch_layer_norm(s->x, L->attn_norm, L->attn_norm_bias, s->normed,
                    n_embd, c->rms_norm_eps);

    arch_matvec(&L->attn_q, s->normed, s->q, s->dequant_temp);
    arch_matvec(&L->attn_k, s->normed, s->k, s->dequant_temp);
    arch_matvec(&L->attn_v, s->normed, s->v, s->dequant_temp);


    size_t kv_off = ((size_t)layer * c->n_ctx + (size_t)s->pos)
                  * s->kv_row_floats;
    memcpy(s->kv_k + kv_off, s->k, s->kv_row_floats * sizeof(float));
    memcpy(s->kv_v + kv_off, s->v, s->kv_row_floats * sizeof(float));

    for (uint32_t h = 0; h < c->n_head; h++) {
        arch_attention_head(s, h, layer, s->q + h * hd,
                            s->attn_out + h * hd);
    }

    arch_matvec(&L->attn_output, s->attn_out, s->normed, s->dequant_temp);
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];

    arch_layer_norm(s->x, L->ffn_norm, L->ffn_norm_bias, s->normed,
                    n_embd, c->rms_norm_eps);

    arch_matvec(&L->ffn_up, s->normed, s->ffn_gate, s->dequant_temp);
    arch_gelu_inplace_f32(s->ffn_gate, c->n_ff);
    arch_matvec(&L->ffn_down, s->ffn_gate, s->normed, s->dequant_temp);
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];
}

static void arch_gptj_layer(OcLlamaSession *s, uint32_t layer)
{
    const OcLlamaConfig *c = &s->model->cfg;
    OcLlamaLayer *L = &s->model->layers[layer];
    size_t hd = c->head_dim;
    size_t n_embd = c->n_embd;
    float rope_base = c->rope_theta > 0.0f ? c->rope_theta : 10000.0f;

    arch_layer_norm(s->x, L->attn_norm, L->attn_norm_bias, s->normed,
                    n_embd, c->rms_norm_eps);

    arch_matvec(&L->attn_q, s->normed, s->q, s->dequant_temp);
    arch_matvec(&L->attn_k, s->normed, s->k, s->dequant_temp);
    arch_matvec(&L->attn_v, s->normed, s->v, s->dequant_temp);

    for (uint32_t h = 0; h < c->n_head; h++) {
        oc_tensor_rope_gptj_f32(s->q + h * hd, hd, s->pos, rope_base);
    }
    for (uint32_t h = 0; h < c->n_head_kv; h++) {
        oc_tensor_rope_gptj_f32(s->k + h * hd, hd, s->pos, rope_base);
    }

    size_t kv_off = ((size_t)layer * c->n_ctx + (size_t)s->pos)
                  * s->kv_row_floats;
    memcpy(s->kv_k + kv_off, s->k, s->kv_row_floats * sizeof(float));
    memcpy(s->kv_v + kv_off, s->v, s->kv_row_floats * sizeof(float));

    for (uint32_t h = 0; h < c->n_head; h++) {
        arch_attention_head(s, h, layer, s->q + h * hd,
                            s->attn_out + h * hd);
    }

    arch_matvec(&L->attn_output, s->attn_out, s->normed, s->dequant_temp);
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];

    arch_layer_norm(s->x, L->ffn_norm, L->ffn_norm_bias, s->normed,
                    n_embd, c->rms_norm_eps);

    arch_matvec(&L->ffn_up, s->normed, s->ffn_gate, s->dequant_temp);
    arch_gelu_inplace_f32(s->ffn_gate, c->n_ff);
    arch_matvec(&L->ffn_down, s->ffn_gate, s->normed, s->dequant_temp);
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];
}

static void arch_neox_layer(OcLlamaSession *s, uint32_t layer)
{
    const OcLlamaConfig *c = &s->model->cfg;
    OcLlamaLayer *L = &s->model->layers[layer];
    size_t hd = c->head_dim;
    size_t n_embd = c->n_embd;

    arch_layer_norm(s->x, L->attn_norm, L->attn_norm_bias, s->normed,
                    n_embd, c->rms_norm_eps);

    arch_matvec(&L->attn_q, s->normed, s->q, s->dequant_temp);
    arch_matvec(&L->attn_k, s->normed, s->k, s->dequant_temp);
    arch_matvec(&L->attn_v, s->normed, s->v, s->dequant_temp);

    for (uint32_t h = 0; h < c->n_head; h++) {
        oc_apply_rope_f32(s->q + h * hd, s->q + h * hd, hd, c->rope_dim,
                          s->pos, c->rope_theta);
    }
    for (uint32_t h = 0; h < c->n_head_kv; h++) {
        oc_apply_rope_f32(s->k + h * hd, s->k + h * hd, hd, c->rope_dim,
                          s->pos, c->rope_theta);
    }

    size_t kv_off = ((size_t)layer * c->n_ctx + (size_t)s->pos)
                  * s->kv_row_floats;
    memcpy(s->kv_k + kv_off, s->k, s->kv_row_floats * sizeof(float));
    memcpy(s->kv_v + kv_off, s->v, s->kv_row_floats * sizeof(float));

    for (uint32_t h = 0; h < c->n_head; h++) {
        arch_attention_head(s, h, layer, s->q + h * hd,
                            s->attn_out + h * hd);
    }

    /* Output projection + residual. */
    arch_matvec(&L->attn_output, s->attn_out, s->normed, s->dequant_temp);
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];

    arch_layer_norm(s->x, L->ffn_norm, L->ffn_norm_bias, s->normed,
                    n_embd, c->rms_norm_eps);

    if (L->ffn_gate.data != NULL) {
        arch_matvec(&L->ffn_gate, s->normed, s->ffn_gate, s->dequant_temp);
        arch_matvec(&L->ffn_up,   s->normed, s->ffn_up,   s->dequant_temp);
        oc_swiglu_inplace_f32(s->ffn_gate, s->ffn_up, c->n_ff);
        arch_matvec(&L->ffn_down, s->ffn_gate, s->normed, s->dequant_temp);
    } else {
        arch_matvec(&L->ffn_up, s->normed, s->ffn_up, s->dequant_temp);
        arch_gelu_inplace_f32(s->ffn_up, c->n_ff);
        arch_matvec(&L->ffn_down, s->ffn_up, s->normed, s->dequant_temp);
    }
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];
}

/* [n_embd + 2*head_dim, n_embd] because of multi-query attention: */
static void arch_falcon_layer(OcLlamaSession *s, uint32_t layer)
{
    const OcLlamaConfig *c = &s->model->cfg;
    OcLlamaLayer *L = &s->model->layers[layer];
    size_t hd = c->head_dim;
    size_t n_embd = c->n_embd;

    arch_layer_norm(s->x, L->attn_norm, L->attn_norm_bias, s->normed,
                    n_embd, c->rms_norm_eps);

    arch_matvec(&L->attn_q, s->normed, s->q, s->dequant_temp);
    arch_matvec(&L->attn_k, s->normed, s->k, s->dequant_temp);
    arch_matvec(&L->attn_v, s->normed, s->v, s->dequant_temp);

    if (L->ffn_norm != NULL) {
        arch_layer_norm(s->x, L->ffn_norm, L->ffn_norm_bias, s->normed,
                        n_embd, c->rms_norm_eps);
    }
    if (L->ffn_gate.data != NULL) {
        arch_matvec(&L->ffn_gate, s->normed, s->ffn_gate, s->dequant_temp);
        arch_matvec(&L->ffn_up,   s->normed, s->ffn_up,   s->dequant_temp);
        oc_swiglu_inplace_f32(s->ffn_gate, s->ffn_up, c->n_ff);
        arch_matvec(&L->ffn_down, s->ffn_gate, s->normed, s->dequant_temp);
    } else {
        arch_matvec(&L->ffn_up, s->normed, s->ffn_up, s->dequant_temp);
        arch_gelu_inplace_f32(s->ffn_up, c->n_ff);
        arch_matvec(&L->ffn_down, s->ffn_up, s->normed, s->dequant_temp);
    }
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];

    for (uint32_t h = 0; h < c->n_head; h++) {
        oc_apply_rope_f32(s->q + h * hd, s->q + h * hd, hd, c->rope_dim,
                          s->pos, c->rope_theta);
    }
    for (uint32_t h = 0; h < c->n_head_kv; h++) {
        oc_apply_rope_f32(s->k + h * hd, s->k + h * hd, hd, c->rope_dim,
                          s->pos, c->rope_theta);
    }

    size_t kv_off = ((size_t)layer * c->n_ctx + (size_t)s->pos)
                  * s->kv_row_floats;
    memcpy(s->kv_k + kv_off, s->k, s->kv_row_floats * sizeof(float));
    memcpy(s->kv_v + kv_off, s->v, s->kv_row_floats * sizeof(float));

    for (uint32_t h = 0; h < c->n_head; h++) {
        arch_attention_head(s, h, layer, s->q + h * hd,
                            s->attn_out + h * hd);
    }

    arch_matvec(&L->attn_output, s->attn_out, s->normed, s->dequant_temp);
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];
}

static OcError arch_final_norm_and_logits(OcLlamaSession *s, float *logits_out)
{
    OcLlamaModel *m = s->model;
    size_t n_embd = m->cfg.n_embd;

    /* Final LayerNorm (GPT-2/NeoX/Falcon use LayerNorm, not RMSNorm). */
    arch_layer_norm(s->x, m->final_norm, m->final_norm_bias, s->normed,
                    n_embd, m->cfg.rms_norm_eps);

    /* lm_head projection → logits. */
    if (logits_out != NULL) {
        if (m->output.qtype == OC_QUANT_F32) {
            oc_matvec_f32((const float *)m->output.data, m->output.rows,
                          m->output.cols, s->normed, logits_out);
        } else {
            oc_matvec_quantized(m->output.qtype, m->output.data,
                                m->output.rows, m->output.cols,
                                m->output.row_bytes, s->normed, logits_out,
                                s->dequant_temp);
        }
    }
    return OC_OK;
}


/* Resolve the GPT-2 position embedding tensor from the model's GGUF.
 * Returns a pointer to the weight view, or NULL if not found. */
static OcWeightView *arch_gpt2_get_pos_embed(OcLlamaModel *m)
{
    if (m->gpt2_pos_resolved) {
        return (m->gpt2_pos_embed.data != NULL) ? &m->gpt2_pos_embed : NULL;
    }
    m->gpt2_pos_resolved = true;

    /* Scan the GGUF tensor list for the position embedding tensor. */
    OcArena *arena = oc_arena_new(1u << 16);
    if (arena == NULL) return NULL;

    OcGgufTensorInfo *infos = NULL;
    size_t n = 0;
    OcError e = oc_gguf_map_mapped_tensor_infos(&m->gguf, arena, &infos, &n);
    if (e != OC_OK) {
        oc_arena_free(arena);
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        const char *cname = infos[i].name;
        if (cname == NULL) continue;
        if (strcmp(cname, "position_embeddings.weight") == 0 ||
            strcmp(cname, "wpe.weight") == 0 ||
            strcmp(cname, "position_embd.weight") == 0) {
            m->gpt2_pos_embed = (OcWeightView){
                .data = oc_gguf_map_tensor_data(&m->gguf, &infos[i]),
                .qtype = oc_quant_type_from_ggml_id(infos[i].ggml_type),
                .rows = (infos[i].n_dims >= 2) ? infos[i].dims[1] : 1,
                .cols = infos[i].dims[0],
            };
            m->gpt2_pos_embed.row_bytes =
                oc_quantized_size(m->gpt2_pos_embed.qtype, m->gpt2_pos_embed.cols);
            if (m->gpt2_pos_embed.row_bytes == 0) {
                m->gpt2_pos_embed.row_bytes =
                    m->gpt2_pos_embed.cols * sizeof(float);
            }
            break;
        }
    }
    oc_arena_free(arena);

    if (m->gpt2_pos_embed.data == NULL) {
        oc_log(OC_LOG_WARN,
               "arch_forward: GPT-2 position_embeddings.weight not found; "
               "multi-token sequences will be incorrect");
        return NULL;
    }
    return &m->gpt2_pos_embed;
}

/* Add the GPT-2 learned positional embedding for position `pos` to `s->x`. */
static void arch_gpt2_add_pos_embed(OcLlamaSession *s, int64_t pos)
{
    OcWeightView *wpe = arch_gpt2_get_pos_embed(s->model);
    if (wpe == NULL) return;
    if (pos >= (int64_t)wpe->rows) pos = (int64_t)wpe->rows - 1;
    if (pos < 0) pos = 0;

    size_t n_embd = s->model->cfg.n_embd;
    if (wpe->qtype == OC_QUANT_F32) {
        const float *row = (const float *)(wpe->data + (size_t)pos * wpe->row_bytes);
        for (size_t i = 0; i < n_embd; i++) s->x[i] += row[i];
    } else {
        /* Dequantize the position embedding row into the temp buffer, then
         * add. We reuse the ffn_up buffer as temp (it's n_ff >= n_embd in
         * practice; if n_ff < n_embd we use dequant_temp which is max-sized). */
        float *temp = (s->model->cfg.n_ff >= n_embd) ? s->ffn_up
                                                      : s->dequant_temp;
        oc_quant_dequant_row(wpe->qtype,
            wpe->data + (size_t)pos * wpe->row_bytes, wpe->row_bytes,
            temp, n_embd);
        for (size_t i = 0; i < n_embd; i++) s->x[i] += temp[i];
    }
}

static OcError arch_validate_session(OcLlamaSession *sess)
{
    if (sess == NULL || sess->model == NULL) return OC_ERR_INVALID_ARG;
    if ((uint64_t)sess->pos >= sess->model->cfg.n_ctx) return OC_ERR_INVALID_ARG;
    return OC_OK;
}

/* Check that the layer weights required for a forward pass are present.
 * Returns OC_ERR_MODEL if critical weights are missing. */
static OcError arch_validate_layers(OcLlamaModel *m)
{
    if (m->layers == NULL) return OC_ERR_MODEL;
    if (m->tok_embeddings.data == NULL) return OC_ERR_MODEL;
    if (m->final_norm == NULL) return OC_ERR_MODEL;
    for (uint32_t l = 0; l < m->cfg.n_layer; l++) {
        OcLlamaLayer *L = &m->layers[l];
        if (L->attn_norm == NULL) return OC_ERR_MODEL;
        if (L->attn_q.data == NULL) return OC_ERR_MODEL;
        if (L->attn_k.data == NULL) return OC_ERR_MODEL;
        if (L->attn_v.data == NULL) return OC_ERR_MODEL;
        if (L->attn_output.data == NULL) return OC_ERR_MODEL;
        if (L->ffn_up.data == NULL) return OC_ERR_MODEL;
        if (L->ffn_down.data == NULL) return OC_ERR_MODEL;
    }
    return OC_OK;
}

OcError oc_arch_forward_gpt2(OcLlamaSession *sess, uint32_t token,
                              float *logits_out)
{
    OcError e = arch_validate_session(sess);
    if (e != OC_OK) return e;
    e = arch_validate_layers(sess->model);
    if (e != OC_OK) return e;

    arch_embed_token(sess, token);

    arch_gpt2_add_pos_embed(sess, sess->pos);

    for (uint32_t l = 0; l < sess->model->cfg.n_layer; l++) {
        arch_gpt2_layer(sess, l);
    }

    /* 4-5. Final LayerNorm + lm_head projection → logits. */
    e = arch_final_norm_and_logits(sess, logits_out);
    if (e != OC_OK) return e;

    sess->pos++;
    return OC_OK;
}

OcError oc_arch_forward_gptj(OcLlamaSession *sess, uint32_t token,
                              float *logits_out)
{
    OcError e = arch_validate_session(sess);
    if (e != OC_OK) return e;
    e = arch_validate_layers(sess->model);
    if (e != OC_OK) return e;

    arch_embed_token(sess, token);

    for (uint32_t l = 0; l < sess->model->cfg.n_layer; l++) {
        arch_gptj_layer(sess, l);
    }

    e = arch_final_norm_and_logits(sess, logits_out);
    if (e != OC_OK) return e;

    sess->pos++;
    return OC_OK;
}

OcError oc_arch_forward_gpt_neox(OcLlamaSession *sess, uint32_t token,
                                  float *logits_out)
{
    OcError e = arch_validate_session(sess);
    if (e != OC_OK) return e;
    e = arch_validate_layers(sess->model);
    if (e != OC_OK) return e;

    arch_embed_token(sess, token);


    for (uint32_t l = 0; l < sess->model->cfg.n_layer; l++) {
        arch_neox_layer(sess, l);
    }

    /* 4-5. Final LayerNorm + lm_head projection → logits. */
    e = arch_final_norm_and_logits(sess, logits_out);
    if (e != OC_OK) return e;

    sess->pos++;
    return OC_OK;
}

OcError oc_arch_forward_falcon(OcLlamaSession *sess, uint32_t token,
                                float *logits_out)
{
    OcError e = arch_validate_session(sess);
    if (e != OC_OK) return e;
    e = arch_validate_layers(sess->model);
    if (e != OC_OK) return e;

    arch_embed_token(sess, token);


    for (uint32_t l = 0; l < sess->model->cfg.n_layer; l++) {
        arch_falcon_layer(sess, l);
    }

    /* 4-5. Final LayerNorm + lm_head projection → logits. */
    e = arch_final_norm_and_logits(sess, logits_out);
    if (e != OC_OK) return e;

    sess->pos++;
    return OC_OK;
}

#ifdef OC_ARCH_FORWARD_TEST_STUBS
#include <stdio.h>

static void arch_forward_test_stubs(void)
{
    /* Just reference the functions to ensure they're not optimized away. */
    (void)oc_arch_forward_gpt2;
    (void)oc_arch_forward_gptj;
    (void)oc_arch_forward_gpt_neox;
    (void)oc_arch_forward_falcon;
    (void)arch_layer_norm;
    (void)arch_gelu_inplace_f32;
    (void)arch_attention_head;
    (void)arch_gpt2_layer;
    (void)arch_gptj_layer;
    (void)arch_neox_layer;
    (void)arch_falcon_layer;
    (void)arch_final_norm_and_logits;
    (void)arch_gpt2_get_pos_embed;
    (void)arch_gpt2_add_pos_embed;
    (void)arch_embed_token;
    (void)arch_matvec;
    (void)arch_validate_session;
    (void)arch_validate_layers;
    printf("arch_forward test stubs OK\n");
}
#endif /* OC_ARCH_FORWARD_TEST_STUBS */
