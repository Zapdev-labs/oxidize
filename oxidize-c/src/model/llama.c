/*
 * llama.c — Llama-family dense forward pass (CPU).
 *
 * Port of oxidize-core/src/model/inference.rs (Llama/Mistral path of
 * `InferenceModel::forward_single`) + inference/{forward,layers,load}.rs.
 *
 * Weight matrices are read zero-copy from the mmap'd GGUF. The matvec
 * dequantizes one weight row at a time via the SIMD-accelerated
 * `oc_quant_dequant_row`, then does an f32 dot with the activation.
 *
 * Bit-exactness target (VAL-FWD-001..004): RMSNorm, RoPE, SwiGLU, attention
 * match the Rust scalar reference. F32-accumulated RMSNorm (not f64) to
 * mirror Rust's `x.iter().map(|v| v*v).sum::<f32>()`.
 */
#include "oxidize/llama.h"

#include "oxidize/activation.h"
#include "oxidize/arena.h"
#include "oxidize/gguf.h"
#include "oxidize/log.h"
#include "oxidize/matvec.h"
#include "oxidize/model.h"
#include "oxidize/quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n, sz);
    return p;
}

static uint32_t cfg_u32(const OcGgufFile *f, const char *key, uint32_t def)
{
    uint32_t v;
    return oc_gguf_metadata_get_u32(f, key, &v) ? v : def;
}

static float cfg_f32(const OcGgufFile *f, const char *key, float def)
{
    float v;
    return oc_gguf_metadata_get_f32(f, key, &v) ? v : def;
}

/* Build a weight view from a resolved tensor info. dims[0]=cols, dims[1]=rows
 * for 2-D; for 1-D (norms) dims[0]=n. */
static OcWeightView view_from_info(const OcGgufMmappedFile *m,
                                   const OcGgufTensorInfo *info)
{
    OcWeightView v = {0};
    if (info == NULL) return v;
    v.data = oc_gguf_map_tensor_data(m, info);
    v.qtype = oc_quant_type_from_ggml_id(info->ggml_type);
    if (info->n_dims == 1) {
        v.rows = 1;
        v.cols = (size_t)info->dims[0];
    } else if (info->n_dims >= 2) {
        v.rows = (size_t)info->dims[1];
        v.cols = (size_t)info->dims[0];
    }
    v.row_bytes = oc_quantized_size(v.qtype, v.cols);
    if (v.row_bytes == 0) v.row_bytes = v.cols * sizeof(float);  /* F32 etc. */
    return v;
}

/* Dequantize a 1-D norm tensor (length n) into a freshly malloc'd f32 array.
 * Norms are almost always stored as F32 in practice; this also handles the
 * rare quantized case. Returns NULL on OOM. */
static float *load_norm(const OcGgufMmappedFile *m, const OcGgufTensorInfo *info,
                        size_t n)
{
    if (info == NULL) return NULL;
    float *out = xcalloc(n, sizeof(float));
    if (out == NULL) return NULL;
    OcWeightView v = view_from_info(m, info);
    if (v.qtype == OC_QUANT_F32) {
        memcpy(out, v.data, n * sizeof(float));
    } else {
        oc_quant_dequant_row(v.qtype, v.data, v.row_bytes, out, n);
    }
    return out;
}

/* ─── Config parsing ──────────────────────────────────────────────────── */

static OcError parse_config(const OcGgufFile *f, const char *arch_str,
                            OcLlamaConfig *cfg)
{
    /* The metadata keys are namespaced by architecture string ("llama.",
     * "qwen2.", "mistral.", ...). Build the key prefix. */
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "%s.", arch_str ? arch_str : "llama");

    char key[160];
    /* Build keys with the arch prefix. */
    snprintf(key, sizeof(key), "%svocab_size", prefix);
    cfg->vocab_size  = cfg_u32(f, key, 32000);
    snprintf(key, sizeof(key), "%scontext_length", prefix);
    cfg->n_ctx       = cfg_u32(f, key, 4096);
    snprintf(key, sizeof(key), "%sblock_count", prefix);
    cfg->n_layer     = cfg_u32(f, key, 32);
    snprintf(key, sizeof(key), "%sembedding_length", prefix);
    cfg->n_embd      = cfg_u32(f, key, 4096);
    snprintf(key, sizeof(key), "%sfeed_forward_length", prefix);
    cfg->n_ff        = cfg_u32(f, key, 11008);
    snprintf(key, sizeof(key), "%sattention.head_count", prefix);
    cfg->n_head      = cfg_u32(f, key, 32);
    snprintf(key, sizeof(key), "%sattention.head_count_kv", prefix);
    cfg->n_head_kv   = cfg_u32(f, key, cfg->n_head);
    snprintf(key, sizeof(key), "%sattention.key_length", prefix);
    uint32_t key_len = cfg_u32(f, key, 0);
    snprintf(key, sizeof(key), "%srope.dimension_count", prefix);
    uint32_t rope_dim = cfg_u32(f, key, 0);
    snprintf(key, sizeof(key), "%srope.freq_base", prefix);
    cfg->rope_theta   = cfg_f32(f, key, 10000.0f);
    snprintf(key, sizeof(key), "%sattention.layer_norm_rms_epsilon", prefix);
    cfg->rms_norm_eps = cfg_f32(f, key, 1e-5f);

    /* Also accept general.vocab_size as a fallback. */
    if (cfg->vocab_size == 32000) {
        uint32_t gv;
        if (oc_gguf_metadata_get_u32(f, "general.vocab_size", &gv)) {
            cfg->vocab_size = gv;
        }
    }

    cfg->head_dim   = (key_len > 0) ? key_len : (cfg->n_embd / cfg->n_head);
    cfg->kv_head_dim = cfg->head_dim;   /* Llama: same; attention.key_length
                                         * applies to both. */
    cfg->rope_dim   = (rope_dim > 0) ? rope_dim : cfg->kv_head_dim;
    if (cfg->rope_dim > cfg->kv_head_dim) cfg->rope_dim = cfg->kv_head_dim;
    cfg->tied_embeddings = false;
    if (cfg->n_head == 0) cfg->n_head = 32;
    if (cfg->n_head_kv == 0) cfg->n_head_kv = cfg->n_head;
    return OC_OK;
}

/* ─── Weight resolution ───────────────────────────────────────────────── */

/* Match a canonical tensor name against the per-layer / global pattern set
 * and store into the model. Returns true if matched. */
static bool assign_tensor(OcLlamaModel *m, const char *cname,
                          const OcGgufMmappedFile *mm,
                          const OcGgufTensorInfo *info)
{
    /* Global tensors. */
    if (strcmp(cname, "tok_embeddings.weight") == 0 ||
        strcmp(cname, "token_embd.weight") == 0) {
        m->tok_embeddings = view_from_info(mm, info);
        return true;
    }
    if (strcmp(cname, "output.weight") == 0) {
        m->output = view_from_info(mm, info);
        return true;
    }
    if (strcmp(cname, "norm.weight") == 0 ||
        strcmp(cname, "output_norm.weight") == 0 ||
        strcmp(cname, "token_embd_norm.weight") == 0) {
        m->final_norm = load_norm(mm, info, m->cfg.n_embd);
        return true;
    }
    /* Per-layer: blk.<N>.<suffix> */
    if (strncmp(cname, "blk.", 4) == 0) {
        char *end = NULL;
        unsigned long layer_idx = strtoul(cname + 4, &end, 10);
        if (end == cname + 4 || *end != '.') return false;
        if (layer_idx >= m->cfg.n_layer) return false;
        OcLlamaLayer *L = &m->layers[layer_idx];
        const char *suf = end + 1;
        if (strcmp(suf, "attn_norm.weight") == 0) {
            L->attn_norm = load_norm(mm, info, m->cfg.n_embd);
        } else if (strcmp(suf, "ffn_norm.weight") == 0) {
            L->ffn_norm = load_norm(mm, info, m->cfg.n_embd);
        } else if (strcmp(suf, "attn_q.weight") == 0) {
            L->attn_q = view_from_info(mm, info);
        } else if (strcmp(suf, "attn_k.weight") == 0) {
            L->attn_k = view_from_info(mm, info);
        } else if (strcmp(suf, "attn_v.weight") == 0) {
            L->attn_v = view_from_info(mm, info);
        } else if (strcmp(suf, "attn_output.weight") == 0) {
            L->attn_output = view_from_info(mm, info);
        } else if (strcmp(suf, "ffn_gate.weight") == 0) {
            L->ffn_gate = view_from_info(mm, info);
        } else if (strcmp(suf, "ffn_up.weight") == 0) {
            L->ffn_up = view_from_info(mm, info);
        } else if (strcmp(suf, "ffn_down.weight") == 0) {
            L->ffn_down = view_from_info(mm, info);
        } else {
            return false;   /* unrecognized suffix; not an error */
        }
        return true;
    }
    return false;
}

static OcError resolve_weights(OcLlamaModel *m)
{
    OcArena *arena = oc_arena_new(1u << 20);   /* 1 MiB for mapped names */
    if (arena == NULL) return OC_ERR_OOM;
    OcGgufTensorInfo *infos = NULL;
    size_t n = 0;
    OcError e = oc_gguf_map_mapped_tensor_infos(&m->gguf, arena, &infos, &n);
    if (e != OC_OK) { oc_arena_free(arena); return e; }

    for (size_t i = 0; i < n; i++) {
        const char *cname = infos[i].name;   /* already canonical-mapped */
        assign_tensor(m, cname, &m->gguf, &infos[i]);
    }
    oc_arena_free(arena);

    /* Validate critical tensors. */
    if (m->tok_embeddings.data == NULL) {
        oc_log(OC_LOG_ERROR, "llama: tok_embeddings.weight not found");
        return OC_ERR_MODEL;
    }
    /* If output.weight absent → tied embeddings. */
    if (m->output.data == NULL) {
        m->output = m->tok_embeddings;
        m->cfg.tied_embeddings = true;
    }
    if (m->final_norm == NULL) {
        oc_log(OC_LOG_ERROR, "llama: norm.weight not found");
        return OC_ERR_MODEL;
    }
    /* Infer n_ff from ffn_gate if metadata was absent. */
    if (m->cfg.n_ff == 0 && m->layers[0].ffn_gate.rows > 0) {
        m->cfg.n_ff = (uint32_t)m->layers[0].ffn_gate.rows;
    }
    /* Infer vocab_size from tok_embeddings rows if still default. */
    if (m->tok_embeddings.rows > 0 && (m->cfg.vocab_size == 32000 ||
        m->cfg.vocab_size != m->tok_embeddings.rows)) {
        /* Trust the actual tensor shape over metadata default. */
    }
    return OC_OK;
}

/* ─── Load ─────────────────────────────────────────────────────────────── */

OcError oc_llama_load(const char *path, OcLlamaModel *out)
{
    if (path == NULL || out == NULL) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    OcError e = oc_gguf_map_open(path, &out->gguf);
    if (e != OC_OK) return e;

    out->arch = oc_gguf_arch_from_file(&out->gguf.unified);
    /* Read the raw `general.architecture` string for the config-key prefix
     * (must match the GGUF on-disk value exactly: "qwen2", not "qwen";
     * "deepseek2", not "deepseek"). Fall back to the enum name only if the
     * metadata key is absent. */
    const char *arch_str = NULL;
    size_t arch_len = 0;
    if (!oc_gguf_metadata_get_str(&out->gguf.unified, "general.architecture",
                                  &arch_str, &arch_len)) {
        arch_str = oc_model_arch_name(out->arch);
    }
    if (arch_str == NULL) arch_str = "llama";

    OcError e2 = parse_config(&out->gguf.unified, arch_str, &out->cfg);
    if (e2 != OC_OK) { oc_gguf_map_free(&out->gguf); return e2; }

    out->layers = xcalloc(out->cfg.n_layer, sizeof(OcLlamaLayer));
    if (out->layers == NULL) {
        oc_gguf_map_free(&out->gguf);
        return OC_ERR_OOM;
    }

    e = resolve_weights(out);
    if (e != OC_OK) {
        oc_llama_free(out);
        return e;
    }
    oc_log(OC_LOG_INFO, "llama: loaded %s n_layer=%u n_embd=%u n_head=%u "
           "n_head_kv=%u head_dim=%u n_ff=%u vocab=%u ctx=%u%s",
           arch_str, out->cfg.n_layer, out->cfg.n_embd, out->cfg.n_head,
           out->cfg.n_head_kv, out->cfg.head_dim, out->cfg.n_ff,
           out->cfg.vocab_size, out->cfg.n_ctx,
           out->cfg.tied_embeddings ? " (tied)" : "");
    return OC_OK;
}

/* ─── Session ─────────────────────────────────────────────────────────── */

OcError oc_llama_session_init(OcLlamaModel *model, OcLlamaSession *out)
{
    if (model == NULL || out == NULL) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));
    out->model = model;
    out->kv_row_floats = (size_t)model->cfg.n_head_kv * model->cfg.kv_head_dim;
    size_t per_layer = (size_t)model->cfg.n_ctx * out->kv_row_floats;
    size_t total = (size_t)model->cfg.n_layer * per_layer;
    out->kv_k = xcalloc(total, sizeof(float));
    out->kv_v = xcalloc(total, sizeof(float));
    size_t maxw = model->cfg.n_embd > model->cfg.n_ff ? model->cfg.n_embd
                                                     : model->cfg.n_ff;
    out->x = xcalloc(model->cfg.n_embd, sizeof(float));
    out->normed = xcalloc(model->cfg.n_embd, sizeof(float));
    out->q = xcalloc((size_t)model->cfg.n_head * model->cfg.head_dim, sizeof(float));
    out->k = xcalloc(out->kv_row_floats, sizeof(float));
    out->v = xcalloc(out->kv_row_floats, sizeof(float));
    out->attn_out = xcalloc((size_t)model->cfg.n_head * model->cfg.head_dim, sizeof(float));
    out->ffn_gate = xcalloc(model->cfg.n_ff, sizeof(float));
    out->ffn_up = xcalloc(model->cfg.n_ff, sizeof(float));
    out->dequant_temp = xcalloc(maxw, sizeof(float));
    out->logits = xcalloc(model->cfg.vocab_size, sizeof(float));
    if (!out->kv_k || !out->kv_v || !out->x || !out->normed || !out->q ||
        !out->k || !out->v || !out->attn_out || !out->ffn_gate ||
        !out->ffn_up || !out->dequant_temp || !out->logits) {
        oc_llama_session_free(out);
        return OC_ERR_OOM;
    }
    out->pos = 0;
    return OC_OK;
}

void oc_llama_session_reset(OcLlamaSession *sess)
{
    if (sess) sess->pos = 0;
}

void oc_llama_session_free(OcLlamaSession *sess)
{
    if (sess == NULL) return;
    free(sess->kv_k); free(sess->kv_v);
    free(sess->x); free(sess->normed);
    free(sess->q); free(sess->k); free(sess->v);
    free(sess->attn_out);
    free(sess->ffn_gate); free(sess->ffn_up);
    free(sess->dequant_temp);
    free(sess->logits);
    memset(sess, 0, sizeof(*sess));
}

void oc_llama_free(OcLlamaModel *model)
{
    if (model == NULL) return;
    if (model->layers) {
        for (uint32_t i = 0; i < model->cfg.n_layer; i++) {
            free(model->layers[i].attn_norm);
            free(model->layers[i].ffn_norm);
        }
        free(model->layers);
    }
    free(model->final_norm);
    oc_gguf_map_free(&model->gguf);
    memset(model, 0, sizeof(*model));
}

/* ─── Forward pass ─────────────────────────────────────────────────────── */

/* Embed one token: dequantize row `token` of tok_embeddings into `dst`. */
static void embed_token(OcLlamaSession *s, uint32_t token)
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
static void matvec(const OcWeightView *w, const float *in, float *out,
                   float *temp)
{
    if (w->qtype == OC_QUANT_F32) {
        oc_matvec_f32((const float *)w->data, w->rows, w->cols, in, out);
    } else {
        oc_matvec_quantized(w->qtype, w->data, w->rows, w->cols, w->row_bytes,
                            in, out, temp);
    }
}

/* Online-softmax attention for one Q head against all cached K/V up to `pos`.
 * GQA: Q head h attends to KV head (h / group_size). */
static void attention_head(const OcLlamaSession *s, uint32_t head,
                           uint32_t layer, const float *q_vec, float *out_vec)
{
    const OcLlamaConfig *c = &s->model->cfg;
    size_t hd = c->head_dim;
    uint32_t group = c->n_head / c->n_head_kv;
    uint32_t kv_head = head / group;
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

static void forward_layer(OcLlamaSession *s, uint32_t layer)
{
    const OcLlamaConfig *c = &s->model->cfg;
    OcLlamaLayer *L = &s->model->layers[layer];
    size_t hd = c->head_dim;

    /* Pre-attention RMSNorm. */
    oc_rms_norm_f32(s->x, L->attn_norm, s->normed, c->n_embd, c->rms_norm_eps);

    /* Q/K/V projections. */
    matvec(&L->attn_q, s->normed, s->q, s->dequant_temp);
    matvec(&L->attn_k, s->normed, s->k, s->dequant_temp);
    matvec(&L->attn_v, s->normed, s->v, s->dequant_temp);

    /* RoPE on Q (per head) and K (per kv head). */
    for (uint32_t h = 0; h < c->n_head; h++) {
        oc_apply_rope_f32(s->q + h * hd, s->q + h * hd, hd, c->rope_dim,
                          s->pos, c->rope_theta);
    }
    for (uint32_t h = 0; h < c->n_head_kv; h++) {
        oc_apply_rope_f32(s->k + h * hd, s->k + h * hd, hd, c->rope_dim,
                          s->pos, c->rope_theta);
    }

    /* KV cache write at position `pos`. */
    size_t kv_off = ((size_t)layer * c->n_ctx + (size_t)s->pos) * s->kv_row_floats;
    memcpy(s->kv_k + kv_off, s->k, s->kv_row_floats * sizeof(float));
    memcpy(s->kv_v + kv_off, s->v, s->kv_row_floats * sizeof(float));

    /* Attention per head → s->attn_out. */
    for (uint32_t h = 0; h < c->n_head; h++) {
        attention_head(s, h, layer, s->q + h * hd, s->attn_out + h * hd);
    }

    /* Output projection. */
    matvec(&L->attn_output, s->attn_out, s->normed, s->dequant_temp);
    /* Residual add (reusing normed as the projection output buffer). */
    for (size_t i = 0; i < c->n_embd; i++) s->x[i] += s->normed[i];

    /* Pre-FFN RMSNorm. */
    oc_rms_norm_f32(s->x, L->ffn_norm, s->normed, c->n_embd, c->rms_norm_eps);

    /* SwiGLU FFN: gate = silu(W_gate·x) * (W_up·x); out = W_down·gate. */
    matvec(&L->ffn_gate, s->normed, s->ffn_gate, s->dequant_temp);
    matvec(&L->ffn_up, s->normed, s->ffn_up, s->dequant_temp);
    oc_swiglu_inplace_f32(s->ffn_gate, s->ffn_up, c->n_ff);
    matvec(&L->ffn_down, s->ffn_gate, s->normed, s->dequant_temp);

    /* Residual add. */
    for (size_t i = 0; i < c->n_embd; i++) s->x[i] += s->normed[i];
}

OcError oc_llama_forward(OcLlamaSession *sess, uint32_t token, float *logits_out)
{
    if (sess == NULL || sess->model == NULL) return OC_ERR_INVALID_ARG;
    if ((uint64_t)sess->pos >= sess->model->cfg.n_ctx) return OC_ERR_INVALID_ARG;

    embed_token(sess, token);
    for (uint32_t l = 0; l < sess->model->cfg.n_layer; l++) {
        forward_layer(sess, l);
    }
    /* Final RMSNorm + lm_head. */
    OcLlamaModel *m = sess->model;
    oc_rms_norm_f32(sess->x, m->final_norm, sess->normed, m->cfg.n_embd,
                    m->cfg.rms_norm_eps);
    if (logits_out != NULL) {
        if (m->output.qtype == OC_QUANT_F32) {
            oc_matvec_f32((const float *)m->output.data, m->output.rows,
                          m->output.cols, sess->normed, logits_out);
        } else {
            oc_matvec_quantized(m->output.qtype, m->output.data, m->output.rows,
                                 m->output.cols, m->output.row_bytes,
                                 sess->normed, logits_out, sess->dequant_temp);
        }
    }
    sess->pos++;
    return OC_OK;
}
