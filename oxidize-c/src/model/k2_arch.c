/*
 * k2_arch.c — K2-Horizon building blocks (see include/oxidize/k2_arch.h).
 *
 * Reference semantics: llama.cpp-k2 src/models/k2-horizon.cpp and the HF
 * modeling_k2_horizon.py. Everything here is glue around the existing
 * matvec kernels; the only arithmetic of note is the router and the gate.
 */
#include "oxidize/k2_arch.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oxidize/activation.h"
#include "oxidize/gguf.h"
#include "oxidize/log.h"
#include "oxidize/matvec.h"
#include "oxidize/parallel.h"

bool oc_k2_arch_match(const char *arch_str)
{
    return arch_str != NULL &&
        (strcmp(arch_str, "k2-horizon") == 0 ||
         strcmp(arch_str, "k2_horizon") == 0);
}

/* ─── Header peek ───────────────────────────────────────────────────────── */

/* Standard GGUF metadata value type ids. */
enum {
    K2_GT_U8 = 0, K2_GT_I8 = 1, K2_GT_U16 = 2, K2_GT_I16 = 3,
    K2_GT_U32 = 4, K2_GT_I32 = 5, K2_GT_F32 = 6, K2_GT_BOOL = 7,
    K2_GT_STR = 8, K2_GT_ARR = 9, K2_GT_U64 = 10, K2_GT_I64 = 11,
    K2_GT_F64 = 12,
};

static size_t k2_scalar_size(uint32_t t)
{
    switch (t) {
    case K2_GT_U8: case K2_GT_I8: case K2_GT_BOOL: return 1;
    case K2_GT_U16: case K2_GT_I16: return 2;
    case K2_GT_U32: case K2_GT_I32: case K2_GT_F32: return 4;
    case K2_GT_U64: case K2_GT_I64: case K2_GT_F64: return 8;
    default: return 0;
    }
}

static bool k2_rd(FILE *fp, void *dst, size_t n)
{
    return fread(dst, 1, n, fp) == n;
}

static bool k2_skip(FILE *fp, uint64_t n)
{
    if (n > (uint64_t)1 << 40) return false;
    return fseek(fp, (long)n, SEEK_CUR) == 0;
}

/* Skip one value of type t. Arrays of strings are walked; nested arrays are
 * refused (no GGUF writer emits them before general.architecture). */
static bool k2_skip_value(FILE *fp, uint32_t t)
{
    size_t sz = k2_scalar_size(t);
    if (sz) return k2_skip(fp, sz);
    if (t == K2_GT_STR) {
        uint64_t len;
        return k2_rd(fp, &len, 8) && k2_skip(fp, len);
    }
    if (t == K2_GT_ARR) {
        uint32_t et;
        uint64_t cnt;
        if (!k2_rd(fp, &et, 4) || !k2_rd(fp, &cnt, 8)) return false;
        size_t esz = k2_scalar_size(et);
        if (esz) {
            if (cnt > ((uint64_t)1 << 40) / esz) return false;
            return k2_skip(fp, cnt * esz);
        }
        if (et != K2_GT_STR) return false;
        for (uint64_t i = 0; i < cnt; i++) {
            uint64_t len;
            if (!k2_rd(fp, &len, 8) || !k2_skip(fp, len)) return false;
        }
        return true;
    }
    return false;
}

bool oc_k2_gguf_path_is_k2(const char *path)
{
    if (path == NULL) return false;
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) return false;
    bool is_k2 = false;
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!k2_rd(fp, &magic, 4) || magic != 0x46554747u /* "GGUF" */ ||
        !k2_rd(fp, &version, 4) || version < 2 ||
        !k2_rd(fp, &n_tensors, 8) || !k2_rd(fp, &n_kv, 8))
        goto done;
    /* general.architecture is the first key in every writer we know of;
     * bound the walk so a hostile header cannot make this slow. */
    for (uint64_t i = 0; i < n_kv && i < 64; i++) {
        uint64_t klen;
        char key[64];
        uint32_t type;
        if (!k2_rd(fp, &klen, 8)) goto done;
        const bool want = (klen == 20);
        if (want) {
            if (!k2_rd(fp, key, 20)) goto done;
        } else if (!k2_skip(fp, klen)) {
            goto done;
        }
        if (!k2_rd(fp, &type, 4)) goto done;
        if (want && memcmp(key, "general.architecture", 20) == 0) {
            uint64_t vlen;
            char val[16];
            if (type != K2_GT_STR || !k2_rd(fp, &vlen, 8)) goto done;
            if (vlen != 10 || !k2_rd(fp, val, 10)) goto done;
            is_k2 = memcmp(val, "k2-horizon", 10) == 0 ||
                    memcmp(val, "k2_horizon", 10) == 0;
            goto done;
        }
        if (!k2_skip_value(fp, type)) goto done;
    }
done:
    fclose(fp);
    return is_k2;
}

/* ─── Config ────────────────────────────────────────────────────────────── */

OcError oc_k2_parse_config(const OcGgufFile *f, const char *prefix,
                           OcLlamaConfig *cfg)
{
    if (f == NULL || prefix == NULL || cfg == NULL) return OC_ERR_INVALID_ARG;
    char key[160];
    uint32_t u;
    bool b;

    cfg->is_k2 = true;
    /* The gate function is not in the metadata: the presence of attn_gate
     * implies softplus(beta = ln2) for this architecture. */
    cfg->attn_out_gate = true;
    cfg->attn_gate_kind = OC_ATTN_GATE_SOFTPLUS_LN2;

    snprintf(key, sizeof key, "%sattention.group_norm_groups", prefix);
    cfg->norm_groups = oc_gguf_metadata_get_u32(f, key, &u) ? u : 1u;
    if (cfg->norm_groups == 0) cfg->norm_groups = 1;
    if (cfg->n_embd % cfg->norm_groups != 0) {
        oc_log(OC_LOG_ERROR, "k2: n_embd %u not divisible by %u norm groups",
               cfg->n_embd, cfg->norm_groups);
        return OC_ERR_MODEL;
    }

    snprintf(key, sizeof key, "%sattention.value_expert_count", prefix);
    cfg->value_expert_count = oc_gguf_metadata_get_u32(f, key, &u) ? u : 0u;
    snprintf(key, sizeof key, "%sattention.value_expert_used_count", prefix);
    cfg->value_expert_used = oc_gguf_metadata_get_u32(f, key, &u) ? u : 0u;
    if (cfg->value_expert_count > 0) {
        if (cfg->value_expert_used == 0) cfg->value_expert_used = 1;
        if (cfg->value_expert_used > cfg->value_expert_count)
            cfg->value_expert_used = cfg->value_expert_count;
        if (cfg->value_expert_used > OC_K2_MAX_VALUE_USED) {
            oc_log(OC_LOG_ERROR, "k2: value_expert_used_count %u exceeds %u",
                   cfg->value_expert_used, OC_K2_MAX_VALUE_USED);
            return OC_ERR_MODEL;
        }
    } else {
        cfg->value_expert_used = 0;
    }

    snprintf(key, sizeof key, "%sexpert_weights_norm", prefix);
    cfg->expert_weights_norm = oc_gguf_metadata_get_bool(f, key, &b) ? b : true;
    snprintf(key, sizeof key, "%sleading_dense_block_count", prefix);
    cfg->leading_dense_block_count =
        oc_gguf_metadata_get_u32(f, key, &u) ? u : 0u;
    return OC_OK;
}

/* ─── Elementwise pieces ────────────────────────────────────────────────── */

void oc_k2_grouped_rms_norm(const float *x, const float *weight, float *out,
                            size_t n, uint32_t groups, float eps)
{
    if (groups <= 1 || n % groups != 0) {
        oc_rms_norm_f32(x, weight, out, n, eps);
        return;
    }
    const size_t g = n / groups;
    for (uint32_t i = 0; i < groups; i++)
        oc_rms_norm_f32(x + (size_t)i * g, weight + (size_t)i * g,
                        out + (size_t)i * g, g, eps);
}

#define K2_LN2     0.69314718055994530942f
#define K2_INV_LN2 1.44269504088896340736f

float oc_k2_softplus_ln2(float g)
{
    const float z = K2_LN2 * g;
    if (z > 20.0f) return g;
    return log1pf(expf(z)) * K2_INV_LN2;
}

void oc_k2_softplus_gate_apply(float *out, const float *gate, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] *= oc_k2_softplus_ln2(gate[i]);
}

void oc_k2_rope_table(int64_t pos, float theta, uint32_t rope_dim,
                      float *cos_out, float *sin_out)
{
    const uint32_t half = rope_dim / 2;
    const double lt = log((double)theta);
    for (uint32_t j = 0; j < half; j++) {
        const double freq = exp(-2.0 * (double)j / (double)rope_dim * lt);
        const double a = (double)pos * freq;
        cos_out[j] = (float)cos(a);
        sin_out[j] = (float)sin(a);
    }
}

void oc_k2_rope_apply(float *x, uint32_t rope_dim, const float *cos_t,
                      const float *sin_t)
{
    const uint32_t half = rope_dim / 2;
    for (uint32_t j = 0; j < half; j++) {
        const float x0 = x[j];
        const float x1 = x[half + j];
        x[j]        = x0 * cos_t[j] - x1 * sin_t[j];
        x[half + j] = x0 * sin_t[j] + x1 * cos_t[j];
    }
}

static inline float k2_silu(float v)
{
    return v / (1.0f + expf(-v));
}

void oc_k2_route(float *probs, const float *bias, uint32_t n, uint32_t k,
                 bool normalize, float scale, uint32_t *sel, float *w,
                 uint32_t *idx)
{
    if (k > n) k = n;
    for (uint32_t i = 0; i < n; i++) {
        probs[i] = 1.0f / (1.0f + expf(-probs[i]));
        idx[i] = i;
    }
    for (uint32_t i = 0; i < k; i++) {
        uint32_t best = i;
        float sb = probs[idx[i]] + (bias ? bias[idx[i]] : 0.0f);
        for (uint32_t j = i + 1; j < n; j++) {
            const float sj = probs[idx[j]] + (bias ? bias[idx[j]] : 0.0f);
            if (sj > sb) { best = j; sb = sj; }
        }
        const uint32_t t = idx[i]; idx[i] = idx[best]; idx[best] = t;
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < k; i++) {
        sel[i] = idx[i];
        w[i] = probs[idx[i]];
        sum += w[i];
    }
    float mul = scale;
    if (normalize) {
        if (sum < OC_K2_WEIGHT_SUM_MIN) sum = OC_K2_WEIGHT_SUM_MIN;
        mul = scale / sum;
    }
    for (uint32_t i = 0; i < k; i++) w[i] *= mul;
}

/* ─── MoVA ─────────────────────────────────────────────────────────────── */

static void k2_router(const OcWeightView *r, const float *in, float *out,
                      float *temp)
{
    if (r->qtype == OC_QUANT_F32)
        oc_matvec_f32((const float *)r->data, r->rows, r->cols, in, out);
    else
        oc_matvec_quantized(r->qtype, r->data, r->rows, r->cols, r->row_bytes,
                            in, out, temp);
}

/* v_out[d] = sum_i w[i] * silu(out_i[d]), summed in expert order — split
 * over d across the pool (bit-identical to the serial loop; the 4 x 1024
 * expf per layer were ~1% of decode on the calling thread). */
typedef struct {
    float       *v_out;
    const float *out_all;
    const float *w;
    uint32_t     k;
    size_t       kv_row;
} K2MixJob;

static void k2_mix_slice(size_t begin, size_t end, size_t tid, void *ud)
{
    (void)tid;
    const K2MixJob *j = (const K2MixJob *)ud;
    for (size_t d = begin; d < end; d++) j->v_out[d] = 0.0f;
    for (uint32_t i = 0; i < j->k; i++) {
        const float *o = j->out_all + (size_t)i * j->kv_row;
        const float wi = j->w[i];
        for (size_t d = begin; d < end; d++) j->v_out[d] += wi * k2_silu(o[d]);
    }
}

void oc_k2_mova_value(const OcLlamaConfig *c, const OcLlamaLayer *L,
                      const float *normed, float *v_out, float *probs,
                      uint32_t *idx, float *w, float *out_all, float *temp)
{
    const uint32_t n = c->value_expert_count;
    const uint32_t k = c->value_expert_used;
    const OcWeightView *ve = &L->attn_v_exps;
    const size_t kv_row = ve->rows;
    const size_t stride = ve->rows * ve->row_bytes;

    uint32_t sel[OC_K2_MAX_VALUE_USED];
    if (k == 0 || k > OC_K2_MAX_VALUE_USED) {
        memset(v_out, 0, kv_row * sizeof(float));
        return;
    }
    k2_router(&L->attn_v_gate, normed, probs, temp);
    oc_k2_route(probs, L->attn_v_gate_b, n, k, c->expert_weights_norm,
                c->expert_weights_scale, sel, w, idx);

    if (ve->qtype != OC_QUANT_F32) {
        /* One parallel region for all selected experts, sharing one
         * activation quantization. */
        OcGgufQuantizationType qt[OC_K2_MAX_VALUE_USED];
        const uint8_t *data[OC_K2_MAX_VALUE_USED];
        size_t rows[OC_K2_MAX_VALUE_USED], rb[OC_K2_MAX_VALUE_USED];
        float *outs[OC_K2_MAX_VALUE_USED];
        for (uint32_t i = 0; i < k; i++) {
            qt[i] = ve->qtype;
            data[i] = ve->data + (size_t)sel[i] * stride;
            rows[i] = kv_row;
            rb[i] = ve->row_bytes;
            outs[i] = out_all + (size_t)i * kv_row;
        }
        oc_matvec_quantized_fused(qt, data, rows, ve->cols, rb, k, normed,
                                  outs, temp);
    } else {
        for (uint32_t i = 0; i < k; i++)
            oc_matvec_f32((const float *)(ve->data + (size_t)sel[i] * stride),
                          kv_row, ve->cols, normed,
                          out_all + (size_t)i * kv_row);
    }

    K2MixJob job = { v_out, out_all, w, k, kv_row };
    oc_parallel_for(kv_row, k2_mix_slice, &job);
}

void oc_k2_mova_batch_free(OcK2MovaBatch *mb)
{
    if (mb == NULL) return;
    free(mb->probs); free(mb->idx); free(mb->sel); free(mb->w);
    free(mb->ex_tok); free(mb->ex_w); free(mb->ex_off); free(mb->ex_fill);
    free(mb->gath); free(mb->eout);
    memset(mb, 0, sizeof(*mb));
}

OcError oc_k2_mova_batch_init(OcK2MovaBatch *mb, const OcLlamaConfig *c,
                              size_t cap)
{
    if (mb == NULL || c == NULL || cap == 0) return OC_ERR_INVALID_ARG;
    memset(mb, 0, sizeof(*mb));
    if (c->value_expert_count == 0) return OC_OK;
    mb->cap = cap;
    mb->n_exp = c->value_expert_count;
    mb->k = c->value_expert_used ? c->value_expert_used : 1;
    mb->n_embd = c->n_embd;
    mb->kv_row = (size_t)c->n_head_kv * c->kv_head_dim;
    mb->probs   = calloc(cap * mb->n_exp, sizeof(float));
    mb->idx     = calloc(mb->n_exp, sizeof(uint32_t));
    mb->sel     = calloc(cap * mb->k, sizeof(uint32_t));
    mb->w       = calloc(cap * mb->k, sizeof(float));
    mb->ex_tok  = calloc(cap * mb->k, sizeof(uint32_t));
    mb->ex_w    = calloc(cap * mb->k, sizeof(float));
    mb->ex_off  = calloc((size_t)mb->n_exp + 1, sizeof(uint32_t));
    mb->ex_fill = calloc(mb->n_exp, sizeof(uint32_t));
    mb->gath    = calloc(cap * mb->n_embd, sizeof(float));
    mb->eout    = calloc(cap * mb->kv_row, sizeof(float));
    if (!mb->probs || !mb->idx || !mb->sel || !mb->w || !mb->ex_tok ||
        !mb->ex_w || !mb->ex_off || !mb->ex_fill || !mb->gath || !mb->eout) {
        oc_k2_mova_batch_free(mb);
        return OC_ERR_OOM;
    }
    return OC_OK;
}

static void k2_mm_batch(const OcWeightView *wv, const uint8_t *data,
                        size_t rows, const float *in, size_t in_stride,
                        float *out, size_t out_stride, size_t n, float *temp,
                        uint8_t *act, size_t act_bytes)
{
    if (wv->qtype == OC_QUANT_F32)
        oc_matvec_f32_batch((const float *)data, rows, wv->cols, in,
                            in_stride, out, out_stride, n);
    else
        oc_matvec_quantized_batch(wv->qtype, data, rows, wv->cols,
                                  wv->row_bytes, in, in_stride, out,
                                  out_stride, n, temp, act, act_bytes);
}

void oc_k2_mova_value_batch(OcK2MovaBatch *mb, const OcLlamaConfig *c,
                            const OcLlamaLayer *L, const float *normed,
                            size_t in_stride, float *v_out, size_t out_stride,
                            size_t n, float *temp, uint8_t *act,
                            size_t act_bytes)
{
    const uint32_t ne = mb->n_exp;
    const uint32_t k = mb->k;
    const OcWeightView *ve = &L->attn_v_exps;
    const size_t kv_row = ve->rows;
    const size_t stride = ve->rows * ve->row_bytes;

    /* Router for every token in one pass. */
    k2_mm_batch(&L->attn_v_gate, L->attn_v_gate.data, L->attn_v_gate.rows,
                normed, in_stride, mb->probs, ne, n, temp, act, act_bytes);

    memset(mb->ex_off, 0, ((size_t)ne + 1) * sizeof(uint32_t));
    memset(mb->ex_fill, 0, (size_t)ne * sizeof(uint32_t));
    for (size_t j = 0; j < n; j++) {
        uint32_t *sel = mb->sel + j * k;
        oc_k2_route(mb->probs + j * ne, L->attn_v_gate_b, ne, k,
                    c->expert_weights_norm, c->expert_weights_scale, sel,
                    mb->w + j * k, mb->idx);
        for (uint32_t i = 0; i < k; i++) mb->ex_off[sel[i] + 1]++;
        memset(v_out + j * out_stride, 0, kv_row * sizeof(float));
    }
    for (uint32_t e = 0; e < ne; e++) mb->ex_off[e + 1] += mb->ex_off[e];
    for (size_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < k; i++) {
            const uint32_t e = mb->sel[j * k + i];
            const uint32_t at = mb->ex_off[e] + mb->ex_fill[e]++;
            mb->ex_tok[at] = (uint32_t)j;
            mb->ex_w[at] = mb->w[j * k + i];
        }
    }

    for (uint32_t e = 0; e < ne; e++) {
        const uint32_t m = mb->ex_off[e + 1] - mb->ex_off[e];
        if (m == 0) continue;
        const uint32_t *toks = mb->ex_tok + mb->ex_off[e];
        const float *ws = mb->ex_w + mb->ex_off[e];
        for (uint32_t t = 0; t < m; t++)
            memcpy(mb->gath + (size_t)t * mb->n_embd,
                   normed + (size_t)toks[t] * in_stride,
                   mb->n_embd * sizeof(float));
        k2_mm_batch(ve, ve->data + (size_t)e * stride, kv_row, mb->gath,
                    mb->n_embd, mb->eout, kv_row, m, temp, act, act_bytes);
        for (uint32_t t = 0; t < m; t++) {
            const float *o = mb->eout + (size_t)t * kv_row;
            float *dst = v_out + (size_t)toks[t] * out_stride;
            const float wt = ws[t];
            for (size_t d = 0; d < kv_row; d++) dst[d] += wt * k2_silu(o[d]);
        }
    }
}

/* ─── Validation ───────────────────────────────────────────────────────── */

OcError oc_k2_validate_layers(const OcLlamaModel *m)
{
    const OcLlamaConfig *c = &m->cfg;
    const size_t kv_row = (size_t)c->n_head_kv * c->kv_head_dim;
    for (uint32_t l = 0; l < c->n_layer; l++) {
        const OcLlamaLayer *L = &m->layers[l];
        if (L->attn_v_exps.data != NULL) {
            if (c->value_expert_count == 0 ||
                L->attn_v_exps.cols != c->n_embd ||
                L->attn_v_exps.rows != kv_row ||
                L->attn_v_gate.data == NULL ||
                L->attn_v_gate.rows != c->value_expert_count ||
                L->attn_v_gate.cols != c->n_embd ||
                L->attn_v_gate_b == NULL) {
                oc_log(OC_LOG_ERROR,
                       "k2: blk.%u MoVA tensors missing or mis-shaped", l);
                return OC_ERR_TENSOR;
            }
        } else if (L->attn_v.data == NULL || L->attn_v.rows != kv_row ||
                   L->attn_v.cols != c->n_embd) {
            oc_log(OC_LOG_ERROR, "k2: blk.%u has neither attn_v nor "
                   "attn_v_exps", l);
            return OC_ERR_TENSOR;
        }
        if (L->attn_gate.data != NULL &&
            (L->attn_gate.cols != c->n_embd ||
             L->attn_gate.rows != (size_t)c->n_head * c->head_dim)) {
            oc_log(OC_LOG_ERROR, "k2: blk.%u attn_gate mis-shaped", l);
            return OC_ERR_TENSOR;
        }
    }
    return OC_OK;
}
