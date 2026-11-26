/*
 * inkling_convert.c — Inkling-2 (inkling2_loop) safetensors -> oxidize GGUF.
 *
 * Writes two files from a Hugging Face checkpoint directory:
 *   - the target model: NVFP4 experts copied bit for bit from the ModelOpt
 *     checkpoint (MXFP8 experts are requantized to NVFP4), dense projections
 *     quantized to NVFP4 (or kept BF16), small tensors as F32;
 *   - the DSpark drafter from mtp.safetensors.
 *
 * The writer streams tensor data, so peak memory stays at one expert or one
 * dense matrix no matter how large the model is.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/inkling.h"

#include "oxidize/arena.h"
#include "oxidize/json.h"
#include "oxidize/log.h"
#include "oxidize/nv3.h"
#include "oxidize/nvfp4.h"
#include "oxidize/parallel.h"
#include "oxidize/safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { GT_F32 = 0, GT_BF16 = 30, GT_NVFP4 = 40, GT_NV3 = OC_NV3_GGML_TYPE };
enum { MV_U32 = 4, MV_I32 = 5, MV_F32 = 6, MV_BOOL = 7, MV_STR = 8, MV_ARR = 9 };

/* ─── small helpers ─────────────────────────────────────────────────────── */

static float bf16f(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static float e4m3fn(uint8_t b)
{
    int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if ((b & 0x7F) == 0x7F) return NAN;
    float v = e == 0 ? ldexpf((float)m, -9) : ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return s ? -v : v;
}

typedef struct {
    OcSafetensorsFile model, mtp;
    bool have_model, have_mtp;
} Src;

static const OcSafetensorsTensor *find_t(const OcSafetensorsFile *f, const char *name)
{
    const OcSafetensorsTensor *t = NULL;
    return oc_safetensors_get_tensor(f, name, &t) == OC_OK ? t : NULL;
}

static const void *tdata(const OcSafetensorsFile *f, const OcSafetensorsTensor *t)
{
    const void *p = NULL;
    return oc_safetensors_get_tensor_data(f, t, &p) == OC_OK ? p : NULL;
}

/* Element i of a BF16/F32 tensor as f32. */
static float elem(const OcSafetensorsTensor *t, const void *p, size_t i)
{
    if (strcmp(t->dtype, "F32") == 0) return ((const float *)p)[i];
    return bf16f(((const uint16_t *)p)[i]);
}

static size_t numel(const OcSafetensorsTensor *t)
{
    size_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) n *= (size_t)t->shape[i];
    return n;
}

/* ─── GGUF streaming writer ─────────────────────────────────────────────── */

typedef struct {
    FILE    *fp;
    uint64_t pos;
} W;

static void w_bytes(W *w, const void *p, size_t n) { fwrite(p, 1, n, w->fp); w->pos += n; }
static void w_u32(W *w, uint32_t v) { w_bytes(w, &v, 4); }
static void w_u64(W *w, uint64_t v) { w_bytes(w, &v, 8); }
static void w_f32(W *w, float v) { w_bytes(w, &v, 4); }
static void w_str(W *w, const char *s) { uint64_t n = strlen(s); w_u64(w, n); w_bytes(w, s, n); }
static void w_strn(W *w, const char *s, size_t n) { w_u64(w, n); w_bytes(w, s, n); }
static void w_pad(W *w)
{
    static const uint8_t z[32] = { 0 };
    size_t pad = (size_t)((32 - w->pos % 32) % 32);
    w_bytes(w, z, pad);
}

typedef struct {
    const char *key;
    int type;          /* MV_* or MV_ARR */
    int arr_type;      /* element type for MV_ARR */
    uint32_t u;
    float f;
    bool b;
    const char *s;
    size_t n;          /* array length */
    const void *arr;   /* u32/i32/f32/bool(uint8) array, or const char ** for strings */
    const size_t *slens; /* string lengths for string arrays (NULL = strlen) */
} Kv;

static void w_kv(W *w, const Kv *kv)
{
    w_str(w, kv->key);
    w_u32(w, (uint32_t)kv->type);
    switch (kv->type) {
    case MV_U32: w_u32(w, kv->u); break;
    case MV_F32: w_f32(w, kv->f); break;
    case MV_BOOL: { uint8_t b = kv->b; w_bytes(w, &b, 1); } break;
    case MV_STR: w_str(w, kv->s); break;
    case MV_ARR:
        w_u32(w, (uint32_t)kv->arr_type);
        w_u64(w, kv->n);
        if (kv->arr_type == MV_STR) {
            const char *const *ss = kv->arr;
            for (size_t i = 0; i < kv->n; i++) {
                if (kv->slens) w_strn(w, ss[i], kv->slens[i]); else w_str(w, ss[i]);
            }
        } else if (kv->arr_type == MV_BOOL) {
            w_bytes(w, kv->arr, kv->n);
        } else {
            w_bytes(w, kv->arr, kv->n * 4);
        }
        break;
    }
}

/* ─── tensor plan ───────────────────────────────────────────────────────── */

typedef enum {
    K_F32,          /* BF16/F32 source -> F32 (rows limited by dims)              */
    K_F32_T,        /* 2D source transposed -> F32                                 */
    K_CONV,         /* [C,1,4] conv weight -> F32 [C][4]                           */
    K_CONV_STATIC,  /* [4,C] lag-major static taps -> F32 [C][4] oldest first     */
    K_BF16,         /* BF16 copy (rows limited by dims)                            */
    K_DENSE,        /* BF16 2D -> NVFP4 rows (+ scale tensor next)                 */
    K_SCALE,        /* the scale(s) recorded by the previous tensor                 */
    K_EXP_NVFP4,    /* ModelOpt NVFP4 experts -> ggml NVFP4 (+ scale tensor next)  */
    K_EXP_MXFP8,    /* MXFP8 experts -> NVFP4 requant (+ scale tensor next)        */
    K_LOOP_QUERY,   /* stacked cross-pass queries                                   */
    K_SOFTMAX_ROW,  /* softmax of one row of a small matrix                         */
} Kind;

typedef struct {
    char     name[96];
    uint32_t n_dims;
    uint64_t dims[3];
    uint32_t gtype;
    uint64_t nbytes;
    uint64_t offset;
    Kind     kind;
    const OcSafetensorsFile *f;
    const OcSafetensorsTensor *t, *t2;
    int      arg;          /* w13 flag, row index, ...                             */
} PT;

typedef struct {
    PT    *v;
    size_t n, cap;
    float  scales[256];
    size_t n_scales;
} Plan;

static PT *plan_add(Plan *p, const char *name, Kind kind, uint32_t gtype, uint32_t n_dims,
                    uint64_t d0, uint64_t d1, uint64_t d2)
{
    if (p->n == p->cap) {
        p->cap = p->cap ? p->cap * 2 : 256;
        p->v = realloc(p->v, p->cap * sizeof *p->v);
    }
    PT *t = &p->v[p->n++];
    memset(t, 0, sizeof *t);
    snprintf(t->name, sizeof t->name, "%s", name);
    t->kind = kind;
    t->gtype = gtype;
    t->n_dims = n_dims;
    t->dims[0] = d0; t->dims[1] = d1; t->dims[2] = d2;
    uint64_t n = d0 * (n_dims > 1 ? d1 : 1) * (n_dims > 2 ? d2 : 1);
    t->nbytes = gtype == GT_F32 ? n * 4 : gtype == GT_BF16 ? n * 2 : gtype == GT_NV3 ? n / 64 * 28 : n / 64 * 36;
    return t;
}

/* ─── producers ─────────────────────────────────────────────────────────── */

typedef struct {
    const float *src;
    uint8_t *dst;
    size_t cols;
    float gscale;
} QJob;

static void q_rows(size_t b, size_t e, size_t tid, void *u)
{
    (void)tid;
    QJob *j = u;
    const size_t rb = oc_nvfp4_row_bytes(j->cols);
    for (size_t r = b; r < e; r++)
        oc_nvfp4_quantize_row(j->src + r * j->cols, j->cols, j->gscale, j->dst + r * rb);
}

static void quantize_matrix(const float *x, size_t rows, size_t cols, float gscale, uint8_t *out)
{
    QJob j = { x, out, cols, gscale };
    oc_parallel_for(rows, q_rows, &j);
}

static void q3_rows(size_t b, size_t e, size_t tid, void *u)
{
    (void)tid;
    QJob *j = u;
    const size_t rb = oc_nv3_row_bytes(j->cols);
    for (size_t r = b; r < e; r++) oc_nv3_quantize_row(j->src + r * j->cols, NULL, j->cols, j->gscale, j->dst + r * rb);
}

/* One matrix in NV3 (gtype GT_NV3) or NVFP4; returns its global scale. */
static float quantize_weights(const float *x, size_t rows, size_t cols, uint32_t gtype, uint8_t *out)
{
    if (gtype != GT_NV3) {
        const float g = oc_nvfp4_global_scale(x, rows * cols);
        quantize_matrix(x, rows, cols, g, out);
        return g;
    }
    QJob j = { x, out, cols, oc_nv3_global_scale(x, rows * cols) };
    oc_parallel_for(rows, q3_rows, &j);
    return j.gscale;
}

/* ModelOpt NVFP4 row (pairs packed low-first, E4M3 scale per 16) -> ggml block row. */
static void repack_nvfp4_row(const uint8_t *packed, const uint8_t *scales, size_t cols, uint8_t *out)
{
    for (size_t b = 0; b < cols / 64; b++) {
        uint8_t *blk = out + b * 36;
        uint8_t nib[64];
        for (size_t i = 0; i < 32; i++) {
            nib[2 * i]     = packed[b * 32 + i] & 15;
            nib[2 * i + 1] = packed[b * 32 + i] >> 4;
        }
        for (size_t s = 0; s < 4; s++) {
            blk[s] = scales[b * 4 + s] & 0x7F;
            for (size_t j = 0; j < 8; j++)
                blk[4 + s * 8 + j] = (uint8_t)(nib[s * 16 + j] | (nib[s * 16 + 8 + j] << 4));
        }
    }
}

static OcError produce(Plan *p, PT *t, W *w)
{
    const OcSafetensorsTensor *st = t->t;
    const void *d = st ? tdata(t->f, st) : NULL;
    if (st && !d) return OC_ERR_IO;
    uint64_t n_el = t->dims[0] * (t->n_dims > 1 ? t->dims[1] : 1) * (t->n_dims > 2 ? t->dims[2] : 1);

    switch (t->kind) {
    case K_F32: {
        for (uint64_t i = 0; i < n_el; i++) w_f32(w, elem(st, d, i));
        return OC_OK;
    }
    case K_F32_T: {
        size_t r = (size_t)st->shape[0], c = (size_t)st->shape[1];
        for (size_t j = 0; j < c; j++)
            for (size_t i = 0; i < r; i++) w_f32(w, elem(st, d, i * c + j));
        return OC_OK;
    }
    case K_CONV: {
        for (uint64_t i = 0; i < n_el; i++) w_f32(w, elem(st, d, i));
        return OC_OK;
    }
    case K_CONV_STATIC: {
        size_t k = (size_t)st->shape[0], c = (size_t)st->shape[1];
        for (size_t ch = 0; ch < c; ch++)
            for (size_t i = 0; i < k; i++) w_f32(w, elem(st, d, (k - 1 - i) * c + ch));
        return OC_OK;
    }
    case K_BF16: {
        w_bytes(w, d, (size_t)t->nbytes);
        return OC_OK;
    }
    case K_DENSE: {
        size_t cols = (size_t)t->dims[0], rows = (size_t)t->dims[1];
        float *x = malloc(rows * cols * sizeof(float));
        uint8_t *q = malloc((size_t)t->nbytes);
        if (!x || !q) { free(x); free(q); return OC_ERR_OOM; }
        for (size_t i = 0; i < rows * cols; i++) x[i] = elem(st, d, i);
        p->scales[0] = quantize_weights(x, rows, cols, t->gtype, q);
        w_bytes(w, q, (size_t)t->nbytes);
        p->n_scales = 1;
        free(x); free(q);
        return OC_OK;
    }
    case K_SCALE: {
        for (size_t i = 0; i < p->n_scales; i++) w_f32(w, p->scales[i]);
        return OC_OK;
    }
    case K_EXP_NVFP4: {
        const bool w13 = t->arg == 1;
        size_t cols = (size_t)t->dims[0], rows = (size_t)t->dims[1], ne = (size_t)t->dims[2];
        const OcSafetensorsTensor *sc = t->t2;
        const uint8_t *sd = tdata(t->f, sc);
        char nm[160];
        snprintf(nm, sizeof nm, "%s_amax", st->name);
        const OcSafetensorsTensor *am = find_t(t->f, nm);
        snprintf(nm, sizeof nm, "%s_outer_scale_denominator", st->name);
        const OcSafetensorsTensor *dn = find_t(t->f, nm);
        if (!sd || !am || !dn) return OC_ERR_FORMAT;
        const float *amax = tdata(t->f, am);
        const float denom = *(const float *)tdata(t->f, dn);
        const size_t rb = oc_nvfp4_row_bytes(cols);
        uint8_t *buf = malloc(rows * rb);
        float *x = t->gtype == GT_NV3 ? malloc(rows * cols * sizeof(float)) : NULL;
        uint8_t *q = t->gtype == GT_NV3 ? malloc(rows * oc_nv3_row_bytes(cols)) : NULL;
        if (!buf || (t->gtype == GT_NV3 && (!x || !q))) { free(buf); free(x); free(q); return OC_ERR_OOM; }
        const uint8_t *wd = d;
        for (size_t e = 0; e < ne; e++) {
            for (size_t r = 0; r < rows; r++) {
                size_t sr = w13 ? (r < rows / 2 ? 2 * r : 2 * (r - rows / 2) + 1) : r;
                repack_nvfp4_row(wd + (e * rows + sr) * (cols / 2),
                                 sd + (e * rows + sr) * (cols / 16), cols, buf + r * rb);
            }
            p->scales[e] = amax[e] / denom;
            if (t->gtype != GT_NV3) {
                w_bytes(w, buf, rows * rb);
                continue;
            }
            /* NV3: requantize the checkpoint's NVFP4 values */
            for (size_t r = 0; r < rows; r++) {
                oc_nvfp4_dequant_row(buf + r * rb, cols, x + r * cols);
                for (size_t c = 0; c < cols; c++) x[r * cols + c] *= p->scales[e];
            }
            p->scales[e] = quantize_weights(x, rows, cols, GT_NV3, q);
            w_bytes(w, q, rows * oc_nv3_row_bytes(cols));
        }
        p->n_scales = ne;
        free(buf); free(x); free(q);
        return OC_OK;
    }
    case K_EXP_MXFP8: {
        const bool w13 = t->arg == 1;
        size_t cols = (size_t)t->dims[0], rows = (size_t)t->dims[1], ne = (size_t)t->dims[2];
        const uint8_t *sd = tdata(t->f, t->t2);
        if (!sd) return OC_ERR_FORMAT;
        const size_t rb = t->gtype == GT_NV3 ? oc_nv3_row_bytes(cols) : oc_nvfp4_row_bytes(cols);
        float *x = malloc(rows * cols * sizeof(float));
        uint8_t *q = malloc(rows * rb);
        if (!x || !q) { free(x); free(q); return OC_ERR_OOM; }
        const uint8_t *wd = d;
        for (size_t e = 0; e < ne; e++) {
            for (size_t r = 0; r < rows; r++) {
                size_t sr = w13 ? (r < rows / 2 ? 2 * r : 2 * (r - rows / 2) + 1) : r;
                const uint8_t *src = wd + (e * rows + sr) * cols;
                const uint8_t *ss = sd + (e * rows + sr) * (cols / 32);
                for (size_t c = 0; c < cols; c++)
                    x[r * cols + c] = e4m3fn(src[c]) * ldexpf(1.0f, (int)ss[c / 32] - 127);
            }
            p->scales[e] = quantize_weights(x, rows, cols, t->gtype, q);
            w_bytes(w, q, rows * rb);
        }
        p->n_scales = ne;
        free(x); free(q);
        return OC_OK;
    }
    case K_LOOP_QUERY: {
        /* rows [q[2i], kv[2i], kv[2i+1], q[2i+1]] for the q, k, v and ffn messages */
        const void *kv = tdata(t->f, t->t2);
        size_t c = (size_t)st->shape[1], i = (size_t)t->arg;
        size_t rows[4] = { 2 * i, 2 * i, 2 * i + 1, 2 * i + 1 };
        for (int k = 0; k < 4; k++) {
            const OcSafetensorsTensor *src = (k == 0 || k == 3) ? st : t->t2;
            const void *sd = (k == 0 || k == 3) ? d : kv;
            for (size_t j = 0; j < c; j++) w_f32(w, elem(src, sd, rows[k] * c + j));
        }
        return OC_OK;
    }
    case K_SOFTMAX_ROW: {
        size_t c = (size_t)st->shape[1], r = (size_t)t->arg;
        float m = -INFINITY, s = 0.0f, v[64];
        for (size_t j = 0; j < c; j++) { v[j] = elem(st, d, r * c + j); if (v[j] > m) m = v[j]; }
        for (size_t j = 0; j < c; j++) { v[j] = expf(v[j] - m); s += v[j]; }
        for (size_t j = 0; j < c; j++) w_f32(w, v[j] / s);
        return OC_OK;
    }
    }
    return OC_ERR_INTERNAL;
}

static OcError write_gguf(const char *path, const char *arch, const Kv *kvs, size_t n_kv, Plan *p)
{
    W w = { fopen(path, "wb"), 0 };
    if (!w.fp) return OC_ERR_IO;
    w_u32(&w, 0x46554747u);
    w_u32(&w, 3);
    w_u64(&w, p->n);
    w_u64(&w, n_kv + 1);
    Kv a = { .key = "general.architecture", .type = MV_STR, .s = arch };
    w_kv(&w, &a);
    for (size_t i = 0; i < n_kv; i++) w_kv(&w, &kvs[i]);
    uint64_t off = 0;
    for (size_t i = 0; i < p->n; i++) {
        PT *t = &p->v[i];
        t->offset = off;
        off += (t->nbytes + 31) / 32 * 32;
        w_str(&w, t->name);
        w_u32(&w, t->n_dims);
        for (uint32_t k = 0; k < t->n_dims; k++) w_u64(&w, t->dims[k]);
        w_u32(&w, t->gtype);
        w_u64(&w, t->offset);
    }
    w_pad(&w);
    const uint64_t base = w.pos;
    for (size_t i = 0; i < p->n; i++) {
        PT *t = &p->v[i];
        uint64_t start = w.pos;
        OcError e = produce(p, t, &w);
        if (e != OC_OK || w.pos - start != t->nbytes || start - base != t->offset) {
            oc_log(OC_LOG_ERROR, "inkling-convert: %s failed (%s, wrote %llu of %llu)", t->name,
                   oc_error_msg(e), (unsigned long long)(w.pos - start), (unsigned long long)t->nbytes);
            fclose(w.fp);
            return e != OC_OK ? e : OC_ERR_INTERNAL;
        }
        w_pad(&w);
        if (i % 64 == 0 || t->nbytes > (64u << 20))
            oc_log(OC_LOG_INFO, "inkling-convert: [%zu/%zu] %s", i + 1, p->n, t->name);
    }
    return fclose(w.fp) == 0 ? OC_OK : OC_ERR_IO;
}

/* ─── config + tokenizer ────────────────────────────────────────────────── */

static char *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); b = NULL; }
    fclose(f);
    if (b) { b[n] = 0; if (len) *len = (size_t)n; }
    return b;
}

static double jnum(const OcJson *o, const char *k, double def)
{
    const OcJson *v = oc_json_get(o, k);
    return v && v->type == OC_JSON_NUMBER ? v->number : def;
}

typedef struct {
    const char **tokens;
    size_t      *tok_lens;
    int32_t     *types;
    size_t       n_tokens;
    const char **merges;
    size_t       n_merges;
} Tok;

static OcError load_tokenizer(const char *dir, OcArena *arena, size_t n_vocab, Tok *tk)
{
    char path[4096];
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);
    size_t len = 0;
    char *js = slurp(path, &len);
    if (!js) return OC_ERR_IO;
    OcJson *root = NULL;
    OcError e = oc_json_parse(js, len, arena, &root);
    free(js);
    if (e != OC_OK) return e;
    const OcJson *model = oc_json_get(root, "model");
    const OcJson *vocab = model ? oc_json_get(model, "vocab") : NULL;
    const OcJson *merges = model ? oc_json_get(model, "merges") : NULL;
    if (!vocab || !merges) return OC_ERR_FORMAT;
    tk->n_tokens = n_vocab;
    tk->tokens = calloc(n_vocab, sizeof *tk->tokens);
    tk->tok_lens = calloc(n_vocab, sizeof *tk->tok_lens);
    tk->types = calloc(n_vocab, sizeof *tk->types);
    for (const OcJson *v = vocab->child; v; v = v->next) {
        size_t id = (size_t)v->number;
        if (id < n_vocab) { tk->tokens[id] = v->key; tk->types[id] = 1; }
    }
    const OcJson *added = oc_json_get(root, "added_tokens");
    for (const OcJson *a = added ? added->child : NULL; a; a = a->next) {
        const OcJson *id = oc_json_get(a, "id"), *c = oc_json_get(a, "content");
        if (id && c && (size_t)id->number < n_vocab) {
            tk->tokens[(size_t)id->number] = c->string;
            tk->types[(size_t)id->number] = 3;
        }
    }
    for (size_t i = 0; i < n_vocab; i++) {
        if (!tk->tokens[i]) {
            char *s = oc_arena_alloc(arena, 32, 1);
            snprintf(s, 32, "<|pad_%zu|>", i);
            tk->tokens[i] = s;
            tk->types[i] = 5;
        }
        tk->tok_lens[i] = strlen(tk->tokens[i]);
    }
    tk->n_merges = merges->count;
    tk->merges = calloc(merges->count ? merges->count : 1, sizeof *tk->merges);
    size_t i = 0;
    for (const OcJson *m = merges->child; m; m = m->next, i++) {
        if (m->type == OC_JSON_STRING) {
            tk->merges[i] = m->string;
        } else if (m->type == OC_JSON_ARRAY && m->child && m->child->next) {
            size_t la = strlen(m->child->string), lb = strlen(m->child->next->string);
            char *s = oc_arena_alloc(arena, la + lb + 2, 1);
            memcpy(s, m->child->string, la);
            s[la] = ' ';
            memcpy(s + la + 1, m->child->next->string, lb + 1);
            tk->merges[i] = s;
        } else {
            return OC_ERR_FORMAT;
        }
    }
    return OC_OK;
}

/* ─── target model plan ─────────────────────────────────────────────────── */

typedef struct {
    const OcSafetensorsFile *f;
    Plan *p;
    OcInklingConvertOptions o;
    OcError err;
} Ctx;

static const OcSafetensorsTensor *need(Ctx *c, const OcSafetensorsFile *f, const char *name)
{
    const OcSafetensorsTensor *t = find_t(f, name);
    if (!t && c->err == OC_OK) {
        oc_log(OC_LOG_ERROR, "inkling-convert: missing tensor %s", name);
        c->err = OC_ERR_FORMAT;
    }
    return t;
}

static void add_f32(Ctx *c, const OcSafetensorsFile *f, const char *dst, const char *src, Kind kind)
{
    const OcSafetensorsTensor *t = need(c, f, src);
    if (!t) return;
    uint64_t d0, d1 = 1;
    uint32_t nd = 1;
    if (kind == K_F32_T) { d0 = t->shape[0]; d1 = t->shape[1]; nd = 2; }
    else if (kind == K_CONV) { d0 = t->shape[t->n_dims - 1]; d1 = t->shape[0]; nd = 2; }
    else if (kind == K_CONV_STATIC) { d0 = t->shape[0]; d1 = t->shape[1]; nd = 2; }
    else if (t->n_dims == 1) { d0 = t->shape[0]; }
    else { d0 = t->shape[t->n_dims - 1]; d1 = numel(t) / d0; nd = 2; }
    PT *pt = plan_add(c->p, dst, kind, GT_F32, nd, d0, d1, 1);
    pt->f = f; pt->t = t;
}

static void add_dense(Ctx *c, const OcSafetensorsFile *f, const char *dst, const char *src,
                      OcInklingWeightType type, uint64_t max_rows)
{
    const OcSafetensorsTensor *t = need(c, f, src);
    if (!t) return;
    uint64_t cols = t->shape[1], rows = t->shape[0];
    if (max_rows && rows > max_rows) rows = max_rows;
    char nm[96];
    if (type == OC_INK_W_BF16 && strcmp(t->dtype, "BF16") == 0) {
        PT *pt = plan_add(c->p, dst, K_BF16, GT_BF16, 2, cols, rows, 1);
        pt->f = f; pt->t = t;
        return;
    }
    PT *pt = plan_add(c->p, dst, K_DENSE, type == OC_INK_W_NV3 ? GT_NV3 : GT_NVFP4, 2, cols, rows, 1);
    pt->f = f; pt->t = t;
    snprintf(nm, sizeof nm, "%.*s.scale", (int)(strlen(dst) - 7), dst);
    plan_add(c->p, nm, K_SCALE, GT_F32, 1, 1, 1, 1);
}

static void add_experts(Ctx *c, int il, bool w13)
{
    char src[120], scale[160], dst[96], nm[96];
    snprintf(src, sizeof src, "model.llm.layers.%d.mlp.experts.%s", il, w13 ? "w13_weight" : "w2_weight");
    snprintf(scale, sizeof scale, "%s_scale", src);
    const OcSafetensorsTensor *t = need(c, c->f, src), *s = need(c, c->f, scale);
    if (!t || !s) return;
    const bool fp4 = strcmp(t->dtype, "U8") == 0;
    uint64_t ne = t->shape[0], rows = t->shape[1], cols = fp4 ? t->shape[2] * 2 : t->shape[2];
    snprintf(dst, sizeof dst, "blk.%d.%s.weight", il, w13 ? "ffn_gate_up_exps" : "ffn_down_exps");
    PT *pt = plan_add(c->p, dst, fp4 ? K_EXP_NVFP4 : K_EXP_MXFP8, c->o.expert_type == OC_INK_W_NV3 ? GT_NV3 : GT_NVFP4,
                      3, cols, rows, ne);
    pt->f = c->f; pt->t = t; pt->t2 = s; pt->arg = w13;
    snprintf(nm, sizeof nm, "blk.%d.%s.scale", il, w13 ? "ffn_gate_up_exps" : "ffn_down_exps");
    plan_add(c->p, nm, K_SCALE, GT_F32, 1, ne, 1, 1);
}

static OcError plan_target(Ctx *c, const OcJson *tc, bool *swa, int n_layer, int core_start, int core_end)
{
    Plan *p = c->p;
    const OcSafetensorsFile *f = c->f;
    const uint64_t n_vocab = (uint64_t)jnum(tc, "unpadded_vocab_size", 0);
    char a[160], b[160];

    const OcSafetensorsTensor *emb = need(c, f, "model.llm.embed.weight");
    if (emb) {
        PT *pt = plan_add(p, "token_embd.weight", K_BF16, GT_BF16, 2, emb->shape[1], n_vocab, 1);
        pt->f = f; pt->t = emb;
    }
    add_f32(c, f, "token_embd_norm.weight", "model.llm.embed_norm.weight", K_F32);
    add_dense(c, f, "output.weight", "model.llm.unembed.weight", c->o.output_type, n_vocab);
    add_f32(c, f, "output_norm.weight", "model.llm.norm.weight", K_F32);
    add_f32(c, f, "output_res_query.weight", "model.llm.attn_res.output_query", K_F32);

    for (int i = 0; i < n_layer && c->err == OC_OK; i++) {
#define L(fmt) (snprintf(a, sizeof a, "model.llm.layers.%d." fmt, i), a)
#define D(fmt) (snprintf(b, sizeof b, "blk.%d." fmt, i), b)
        add_dense(c, f, D("attn_q.weight"), L("attn.wq_du.weight"), c->o.dense_type, 0);
        add_dense(c, f, D("attn_k.weight"), L("attn.wk_dv.weight"), c->o.dense_type, 0);
        add_dense(c, f, D("attn_v.weight"), L("attn.wv_dv.weight"), c->o.dense_type, 0);
        add_dense(c, f, D("attn_output.weight"), L("attn.wo_ud.weight"), c->o.dense_type, 0);
        add_dense(c, f, D("attn_gate.weight"), L("attn.gate_attn.weight"), c->o.dense_type, 0);
        add_dense(c, f, D("attn_rel.weight"), L("attn.wr_du.weight"), c->o.dense_type, 0);
        add_f32(c, f, D("attn_sink_proj.weight"), L("attn.attn_sink_proj.weight"), K_F32);
        add_f32(c, f, D("attn_q_clip.weight"), L("attn.q_l2_clip.scale"), K_F32);
        add_f32(c, f, D("attn_rel_clip.weight"), L("attn.rel_l2_clip.scale"), K_F32);
        add_f32(c, f, D("attn_rel_bias.weight"), L("attn.rel_logits_proj.proj"), K_F32_T);
        add_f32(c, f, D("attn_post_conv.weight"), L("attn_sconv.weight"), K_CONV);
        add_f32(c, f, D("ffn_post_conv.weight"), L("mlp_sconv.weight"), K_CONV);
        if (swa[i]) {
            add_f32(c, f, D("attn_k_conv.weight"), L("attn.k_sconv.weight"), K_CONV);
            add_f32(c, f, D("attn_v_conv.weight"), L("attn.v_sconv.weight"), K_CONV);
            add_f32(c, f, D("attn_norm.weight"), L("attn_norm.weight"), K_F32);
        } else {
            add_f32(c, f, D("attn_k_conv.weight"), L("attn.k_sconv.static_w"), K_CONV_STATIC);
            add_f32(c, f, D("attn_v_conv.weight"), L("attn.v_sconv.static_w"), K_CONV_STATIC);
            add_f32(c, f, D("attn_k_conv_proj.weight"), L("attn.k_sconv.weight_proj"), K_F32);
            add_f32(c, f, D("attn_v_conv_proj.weight"), L("attn.v_sconv.weight_proj"), K_F32);
            add_f32(c, f, D("attn_q_inp_norm.weight"), L("q_input_norm.weight"), K_F32);
            add_f32(c, f, D("attn_k_inp_norm.weight"), L("k_input_norm.weight"), K_F32);
            add_f32(c, f, D("attn_v_inp_norm.weight"), L("v_input_norm.weight"), K_F32);
        }
        if (i > 0) {
            snprintf(a, sizeof a, "model.llm.attn_res.attn_queries.%d", i - 1);
            add_f32(c, f, D("attn_res_query.weight"), a, K_F32);
        }
        snprintf(a, sizeof a, "model.llm.attn_res.mlp_queries.%d", i);
        add_f32(c, f, D("ffn_res_query.weight"), a, K_F32);
        add_f32(c, f, D("ffn_norm.weight"), L("mlp_norm.weight"), K_F32);
        add_f32(c, f, D("ffn_gate_inp.weight"), L("mlp.gate.weight"), K_F32);
        add_f32(c, f, D("ffn_exp_probs_b.bias"), L("mlp.gate.bias"), K_F32);
        add_experts(c, i, true);
        add_experts(c, i, false);

        if (i >= core_start && i < core_end) {
            const OcSafetensorsTensor *q = need(c, f, "model.llm.middle_cycle_cross_pass_queries");
            const OcSafetensorsTensor *kv = need(c, f, "model.llm.middle_cycle_cross_pass_kv_queries");
            if (q && kv) {
                PT *pt = plan_add(p, D("loop_msg_query.weight"), K_LOOP_QUERY, GT_F32, 2, q->shape[1], 4, 1);
                pt->f = f; pt->t = q; pt->t2 = kv; pt->arg = i - core_start;
            }
            static const char *ctl[4] = { "attn", "k", "v", "mlp" };
            static const char *dst[4] = { "q", "k", "v", "ffn" };
            static const char *part[3] = { "down", "gain", "forget" };
            for (int k = 0; k < 4; k++) {
                for (int m = 0; m < 3; m++) {
                    char s2[160], d2[96];
                    snprintf(s2, sizeof s2, "model.llm.layers.%d.controllers.%s.%s_weight", i, ctl[k], part[m]);
                    snprintf(d2, sizeof d2, "blk.%d.loop_%s_ctrl_%s.weight", i, dst[k], part[m]);
                    const OcSafetensorsTensor *t = need(c, f, s2);
                    if (t) {
                        PT *pt = plan_add(p, d2, K_BF16, GT_BF16, 2, t->shape[1], t->shape[0], 1);
                        pt->f = f; pt->t = t;
                    }
                }
            }
        }
#undef L
#undef D
    }
    return c->err;
}

/* ─── draft (DSpark) plan ───────────────────────────────────────────────── */

static OcError plan_draft(Ctx *c, int n_layer, int n_fuse, uint64_t n_vocab)
{
    const OcSafetensorsFile *f = c->f;
    char a[160], b[160];
    for (int i = 0; i < n_layer && c->err == OC_OK; i++) {
#define L(fmt) (snprintf(a, sizeof a, "layers.%d." fmt, i), a)
#define D(fmt) (snprintf(b, sizeof b, "blk.%d." fmt, i), b)
        add_f32(c, f, D("attn_norm.weight"), L("input_layernorm.weight"), K_F32);
        add_dense(c, f, D("attn_q.weight"), L("self_attn.q_proj.weight"), c->o.draft_type, 0);
        add_dense(c, f, D("attn_k.weight"), L("self_attn.k_proj.weight"), c->o.draft_type, 0);
        add_dense(c, f, D("attn_v.weight"), L("self_attn.v_proj.weight"), c->o.draft_type, 0);
        add_dense(c, f, D("attn_output.weight"), L("self_attn.o_proj.weight"), c->o.draft_type, 0);
        add_f32(c, f, D("attn_q_norm.weight"), L("self_attn.q_norm.weight"), K_F32);
        add_f32(c, f, D("attn_k_norm.weight"), L("self_attn.k_norm.weight"), K_F32);
        add_dense(c, f, D("attn_k_ctx.weight"), L("self_attn.k_proj_ctx.weight"), c->o.draft_type, 0);
        add_dense(c, f, D("attn_v_ctx.weight"), L("self_attn.v_proj_ctx.weight"), c->o.draft_type, 0);
        add_f32(c, f, D("attn_k_ctx_norm.weight"), L("self_attn.k_norm_ctx.weight"), K_F32);
        add_f32(c, f, D("ctx_norm.weight"), L("dflare_norm.weight"), K_F32);
        add_f32(c, f, D("ffn_norm.weight"), L("post_attention_layernorm.weight"), K_F32);
        add_dense(c, f, D("ffn_gate.weight"), L("mlp.gate_proj.weight"), c->o.draft_type, 0);
        add_dense(c, f, D("ffn_up.weight"), L("mlp.up_proj.weight"), c->o.draft_type, 0);
        add_dense(c, f, D("ffn_down.weight"), L("mlp.down_proj.weight"), c->o.draft_type, 0);
        const OcSafetensorsTensor *fw = need(c, f, "fuse_weight");
        if (fw) {
            PT *pt = plan_add(c->p, D("ctx_mix.weight"), K_SOFTMAX_ROW, GT_F32, 1, fw->shape[1], 1, 1);
            pt->f = f; pt->t = fw; pt->arg = i;
        }
#undef L
#undef D
    }
    const OcSafetensorsTensor *fw = need(c, f, "fuse_weight");
    if (fw) {
        if ((int)fw->shape[0] != n_layer + 1 || (int)fw->shape[1] != n_fuse) return OC_ERR_FORMAT;
        PT *pt = plan_add(c->p, "markov_ctx_mix.weight", K_SOFTMAX_ROW, GT_F32, 1, fw->shape[1], 1, 1);
        pt->f = f; pt->t = fw; pt->arg = n_layer;
    }
    add_f32(c, f, "output_norm.weight", "norm.weight", K_F32);
    add_f32(c, f, "mask_embd.weight", "mask_embed", K_F32);
    add_f32(c, f, "markov_ctx_norm.weight", "markov_head.dflare_norm.weight", K_F32);
    add_f32(c, f, "markov_hidden_out.weight", "markov_head.hidden_out.weight", K_F32);
    add_f32(c, f, "markov_hidden_out.bias", "markov_head.hidden_out.bias", K_F32);
    add_f32(c, f, "markov_rnn_proj.weight", "markov_head.rnn_proj.weight", K_F32);
    add_f32(c, f, "markov_rnn_proj.bias", "markov_head.rnn_proj.bias", K_F32);
    const OcSafetensorsTensor *w1 = need(c, f, "markov_head.markov_w1.weight");
    if (w1) {
        PT *pt = plan_add(c->p, "markov_w1.weight", K_BF16, GT_BF16, 2, w1->shape[1], n_vocab, 1);
        pt->f = f; pt->t = w1;
    }
    return c->err;
}

/* ─── entry point ───────────────────────────────────────────────────────── */

OcError oc_inkling_convert(const OcInklingConvertOptions *opt)
{
    if (!opt || !opt->hf_dir || (!opt->out_path && !opt->draft_out_path)) return OC_ERR_INVALID_ARG;
    if (opt->output_type == OC_INK_W_NV3 || opt->draft_type == OC_INK_W_NV3) {
        /* the DSpark drafter reads the lm head and its own weights as NVFP4 or BF16 only */
        oc_log(OC_LOG_ERROR, "inkling-convert: NV3 is for the experts and attention, not the lm head or drafter");
        return OC_ERR_INVALID_ARG;
    }
    char path[4096];
    OcArena *arena = oc_arena_new(64u << 20);
    if (!arena) return OC_ERR_OOM;
    OcError e = OC_OK;

    snprintf(path, sizeof path, "%s/config.json", opt->hf_dir);
    size_t len = 0;
    char *cfg_text = slurp(path, &len);
    OcJson *cfg = NULL;
    if (!cfg_text || oc_json_parse(cfg_text, len, arena, &cfg) != OC_OK) {
        free(cfg_text);
        oc_arena_free(arena);
        return OC_ERR_FORMAT;
    }
    free(cfg_text);
    const OcJson *tc = oc_json_get(cfg, "text_config");
    const OcJson *mtp = oc_json_get(cfg, "mtp_config");
    const OcJson *loop = tc ? oc_json_get(tc, "loop_config") : NULL;
    if (!tc || !loop) { oc_arena_free(arena); return OC_ERR_FORMAT; }

    const int n_layer = (int)jnum(tc, "num_hidden_layers", 0);
    const int core_start = (int)jnum(loop, "core_start", 0), core_end = (int)jnum(loop, "core_end", 0);
    const int n_passes = (int)jnum(loop, "num_passes", 0);
    const uint32_t n_vocab = (uint32_t)jnum(tc, "unpadded_vocab_size", 0);
    bool swa[OC_INK_MAX_LAYERS] = { 0 };
    uint8_t swa_u8[OC_INK_MAX_LAYERS] = { 0 };
    const OcJson *local = oc_json_get(tc, "local_layer_ids");
    for (const OcJson *v = local ? local->child : NULL; v; v = v->next)
        if (v->number >= 0 && v->number < OC_INK_MAX_LAYERS) swa[(int)v->number] = true;
    for (int i = 0; i < OC_INK_MAX_LAYERS; i++) swa_u8[i] = swa[i];
    if (n_layer <= 0 || n_layer > OC_INK_MAX_LAYERS || n_vocab == 0) { oc_arena_free(arena); return OC_ERR_FORMAT; }

    Src s = { 0 };
    snprintf(path, sizeof path, "%s/model-00001-of-00001.safetensors", opt->hf_dir);
    e = oc_safetensors_open(path, &s.model);
    if (e != OC_OK) { oc_arena_free(arena); return e; }
    s.have_model = true;

    if (opt->out_path) {
        Tok tk = { 0 };
        e = load_tokenizer(opt->hf_dir, arena, n_vocab, &tk);
        if (e != OC_OK) {
            oc_log(OC_LOG_ERROR, "inkling-convert: tokenizer.json: %s", oc_error_msg(e));
            goto done;
        }
        snprintf(path, sizeof path, "%s/chat_template.jinja", opt->hf_dir);
        char *tmpl = slurp(path, NULL);

        Plan p = { 0 };
        Ctx c = { &s.model, &p, *opt, OC_OK };
        e = plan_target(&c, tc, swa, n_layer, core_start, core_end);
        if (e == OC_OK) {
#define A "inkling2_loop."
            Kv kv[] = {
                { .key = A "block_count", .type = MV_U32, .u = (uint32_t)n_layer },
                { .key = A "embedding_length", .type = MV_U32, .u = (uint32_t)jnum(tc, "hidden_size", 0) },
                { .key = A "vocab_size", .type = MV_U32, .u = n_vocab },
                { .key = A "attention.head_count", .type = MV_U32, .u = (uint32_t)jnum(tc, "num_attention_heads", 0) },
                { .key = A "attention.head_count_kv", .type = MV_U32, .u = (uint32_t)jnum(tc, "num_key_value_heads", 0) },
                { .key = A "attention.key_length", .type = MV_U32, .u = (uint32_t)jnum(tc, "head_dim", 0) },
                { .key = A "attention.layer_norm_rms_epsilon", .type = MV_F32, .f = (float)jnum(tc, "rms_norm_eps", 1e-6) },
                { .key = A "attention.sliding_window", .type = MV_U32, .u = (uint32_t)jnum(tc, "sliding_window_size", 0) },
                { .key = A "attention.sliding_window_pattern", .type = MV_ARR, .arr_type = MV_BOOL, .n = (size_t)n_layer, .arr = swa_u8 },
                { .key = A "attention.relative_length", .type = MV_U32, .u = (uint32_t)jnum(tc, "d_rel", 0) },
                { .key = A "attention.relative_extent", .type = MV_U32, .u = (uint32_t)jnum(tc, "rel_extent", 0) },
                { .key = A "attention.qk_clip", .type = MV_F32, .f = (float)jnum(tc, "qk_l2_clip_threshold", 0) },
                { .key = A "attention.relative_clip", .type = MV_F32, .f = (float)jnum(tc, "rel_l2_clip_threshold", 0) },
                { .key = A "attention.log_scale_floor", .type = MV_U32, .u = (uint32_t)jnum(tc, "log_scaling_n_floor", 0) },
                { .key = A "attention.log_scale_alpha", .type = MV_F32, .f = (float)jnum(tc, "log_scaling_alpha", 0) },
                { .key = A "expert_count", .type = MV_U32, .u = (uint32_t)jnum(tc, "n_routed_experts", 0) },
                { .key = A "expert_used_count", .type = MV_U32, .u = (uint32_t)jnum(tc, "num_experts_per_tok", 0) },
                { .key = A "expert_feed_forward_length", .type = MV_U32, .u = (uint32_t)jnum(tc, "intermediate_size", 0) },
                { .key = A "expert_weights_scale", .type = MV_F32, .f = (float)jnum(tc, "route_scale", 1) },
                { .key = A "activation.situ_beta", .type = MV_F32, .f = (float)jnum(tc, "mlp_gate_softcap", 0) },
                { .key = A "activation.situ_linear_beta", .type = MV_F32, .f = (float)jnum(tc, "mlp_up_softcap", 0) },
                { .key = A "final_logit_softcapping", .type = MV_F32, .f = (float)jnum(tc, "final_logit_softcapping", 0) },
                { .key = A "attn_res.block_size", .type = MV_U32, .u = (uint32_t)jnum(tc, "n_layers_per_attn_res_block", 0) },
                { .key = A "shortconv.l_cache", .type = MV_U32, .u = (uint32_t)jnum(tc, "sconv_kernel_size", 0) },
                { .key = A "loop.start", .type = MV_U32, .u = (uint32_t)core_start },
                { .key = A "loop.end", .type = MV_U32, .u = (uint32_t)core_end },
                { .key = A "loop.count", .type = MV_U32, .u = (uint32_t)n_passes },
                { .key = A "loop.gate_rank", .type = MV_U32, .u = (uint32_t)jnum(loop, "gate_rank", 0) },
                { .key = A "context_length", .type = MV_U32, .u = 262144 },
                { .key = "tokenizer.ggml.model", .type = MV_STR, .s = "gpt2" },
                { .key = "tokenizer.ggml.pre", .type = MV_STR, .s = "gpt-4o" },
                { .key = "tokenizer.ggml.tokens", .type = MV_ARR, .arr_type = MV_STR, .n = tk.n_tokens, .arr = tk.tokens, .slens = tk.tok_lens },
                { .key = "tokenizer.ggml.token_type", .type = MV_ARR, .arr_type = MV_I32, .n = tk.n_tokens, .arr = tk.types },
                { .key = "tokenizer.ggml.merges", .type = MV_ARR, .arr_type = MV_STR, .n = tk.n_merges, .arr = tk.merges },
                { .key = "tokenizer.ggml.eos_token_id", .type = MV_U32, .u = (uint32_t)jnum(cfg, "eos_token_id", 0) },
                { .key = "tokenizer.ggml.padding_token_id", .type = MV_U32, .u = 199999 },
                { .key = "tokenizer.ggml.add_bos_token", .type = MV_BOOL, .b = false },
                { .key = "tokenizer.chat_template", .type = MV_STR, .s = tmpl ? tmpl : "" },
            };
#undef A
            e = write_gguf(opt->out_path, "inkling2_loop", kv, sizeof kv / sizeof kv[0], &p);
        }
        free(tmpl);
        free(p.v);
        free(tk.tokens); free(tk.tok_lens); free(tk.types); free(tk.merges);
        if (e != OC_OK) goto done;
        oc_log(OC_LOG_INFO, "inkling-convert: wrote %s", opt->out_path);
    }

    if (opt->draft_out_path) {
        if (!mtp) { e = OC_ERR_FORMAT; goto done; }
        snprintf(path, sizeof path, "%s/mtp.safetensors", opt->hf_dir);
        e = oc_safetensors_open(path, &s.mtp);
        if (e != OC_OK) goto done;
        s.have_mtp = true;
        const OcJson *tl = oc_json_get(mtp, "target_layer_ids");
        int32_t targets[8] = { 0 };
        size_t n_targets = 0;
        for (const OcJson *v = tl ? tl->child : NULL; v && n_targets < 8; v = v->next) targets[n_targets++] = (int32_t)v->number;
        const int d_layers = (int)jnum(mtp, "num_hidden_layers", 0);
        uint8_t dswa[8] = { 0 };
        const OcJson *dl = oc_json_get(mtp, "local_layer_ids");
        for (const OcJson *v = dl ? dl->child : NULL; v; v = v->next)
            if (v->number >= 0 && v->number < 8) dswa[(int)v->number] = 1;

        Plan p = { 0 };
        Ctx c = { &s.mtp, &p, *opt, OC_OK };
        e = plan_draft(&c, d_layers, (int)n_targets, n_vocab);
        if (e == OC_OK) {
#define A "inkling2_dspark."
            Kv kv[] = {
                { .key = A "block_count", .type = MV_U32, .u = (uint32_t)d_layers },
                { .key = A "embedding_length", .type = MV_U32, .u = (uint32_t)jnum(mtp, "hidden_size", 0) },
                { .key = A "feed_forward_length", .type = MV_U32, .u = (uint32_t)jnum(mtp, "intermediate_size", 0) },
                { .key = A "attention.head_count", .type = MV_U32, .u = (uint32_t)jnum(mtp, "num_attention_heads", 0) },
                { .key = A "attention.head_count_kv", .type = MV_U32, .u = (uint32_t)jnum(mtp, "num_key_value_heads", 0) },
                { .key = A "attention.key_length", .type = MV_U32, .u = (uint32_t)jnum(mtp, "head_dim", 0) },
                { .key = A "attention.layer_norm_rms_epsilon", .type = MV_F32, .f = (float)jnum(mtp, "rms_norm_eps", 1e-6) },
                { .key = A "attention.sliding_window", .type = MV_U32, .u = (uint32_t)jnum(mtp, "sliding_window_size", 0) },
                { .key = A "attention.sliding_window_pattern", .type = MV_ARR, .arr_type = MV_BOOL, .n = (size_t)d_layers, .arr = dswa },
                { .key = A "rope.freq_base", .type = MV_F32, .f = (float)jnum(mtp, "rope_theta", 1e6) },
                { .key = A "block_size", .type = MV_U32, .u = (uint32_t)jnum(mtp, "block_size", 0) },
                { .key = A "markov_rank", .type = MV_U32, .u = (uint32_t)jnum(mtp, "markov_rank", 0) },
                { .key = A "target_layers", .type = MV_ARR, .arr_type = MV_I32, .n = n_targets, .arr = targets },
                { .key = A "vocab_size", .type = MV_U32, .u = n_vocab },
            };
#undef A
            e = write_gguf(opt->draft_out_path, "inkling2_dspark", kv, sizeof kv / sizeof kv[0], &p);
        }
        free(p.v);
        if (e == OC_OK) oc_log(OC_LOG_INFO, "inkling-convert: wrote %s", opt->draft_out_path);
    }

done:
    if (s.have_mtp) oc_safetensors_close(&s.mtp);
    if (s.have_model) oc_safetensors_close(&s.model);
    oc_arena_free(arena);
    return e;
}
