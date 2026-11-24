/*
 * inkling_dspark.c — DSpark drafter for Inkling-2 and greedy speculative
 * generation.
 *
 * The drafter is a 3-layer block model. Each committed target token is
 * injected once: its target features (layer visits 5, 23, 41, final norm)
 * are mixed per draft layer into context K/V, and mixed once more into the
 * Markov head's context state f_rnn. A draft runs [anchor, mask x n] through
 * the layers with non-causal attention over the context and the block,
 * takes base logits from the target's lm head, then walks the block left to
 * right: an RNN over [state, w1[prev], hidden_i, f_rnn] yields a Markov
 * vector whose projection through the lm head biases position i. The bias
 * is evaluated on the top candidates of the base logits, which is where it
 * can change the argmax.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/inkling.h"

#include "inkling_gpu.h"
#include "inkling_simd.h"
#include "oxidize/log.h"
#include "oxidize/nvfp4.h"
#include "oxidize/parallel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GT_F32   0u
#define GT_BF16  30u
#define GT_NVFP4 40u

#define DS_MAX_BLOCK 16
#define DS_FRNN_RING 64

OcError oc_ink_load_mat(const OcGgufMmappedFile *map, const char *name, OcInkMat *out, bool required);

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline float bf16f(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static float dotf(const float *a, const float *b, size_t n) { return ink_dot(a, b, n); }

static void rms_norm(const float *x, const float *w, size_t n, float eps, float *out)
{
    double ss = 0.0;
    for (size_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float r = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (size_t i = 0; i < n; i++) out[i] = x[i] * r * (w ? w[i] : 1.0f);
}

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ─── loading ───────────────────────────────────────────────────────────── */

static const float *f32t(const OcGgufMmappedFile *map, const char *name, OcError *e)
{
    const OcGgufTensorInfo *ti = oc_gguf_map_tensor_get(map, name);
    if (!ti || ti->ggml_type != GT_F32) {
        oc_log(OC_LOG_ERROR, "inkling-draft: missing F32 tensor %s", name);
        *e = OC_ERR_FORMAT;
        return NULL;
    }
    return (const float *)oc_gguf_map_tensor_data(map, ti);
}

static uint32_t du32(const OcGgufMmappedFile *map, const char *key, OcError *e)
{
    char k[128];
    uint32_t v = 0;
    snprintf(k, sizeof k, "inkling2_dspark.%s", key);
    if (!oc_gguf_metadata_get_u32(&map->unified, k, &v)) {
        oc_log(OC_LOG_ERROR, "inkling-draft: missing metadata %s", k);
        *e = OC_ERR_FORMAT;
    }
    return v;
}

static float df32(const OcGgufMmappedFile *map, const char *key, OcError *e)
{
    char k[128];
    float v = 0.0f;
    snprintf(k, sizeof k, "inkling2_dspark.%s", key);
    if (!oc_gguf_metadata_get_f32(&map->unified, k, &v)) {
        oc_log(OC_LOG_ERROR, "inkling-draft: missing metadata %s", k);
        *e = OC_ERR_FORMAT;
    }
    return v;
}

/* ─── Markov bias fold ──────────────────────────────────────────────────── */

#define FOLD_ROWS 8

typedef struct {
    const OcInkMat *U;            /* target lm head [V][E]  */
    const float    *Ht, *hb;      /* hidden_out^T [R][E], hidden_out_b [E] */
    size_t          V, E, R;
    float          *w2f, *b2;     /* [V][R], [V]            */
    uint8_t        *w2;
    float           gscale;
} FoldJob;

static void lm_row(const OcInkMat *U, size_t v, float *out)
{
    const uint8_t *row = U->data + v * U->row_bytes;
    if (U->gtype == GT_NVFP4) {
        oc_nvfp4_dequant_row(row, U->cols, out);
        const float s = U->scale ? U->scale[0] : 1.0f;
        for (size_t i = 0; i < U->cols; i++) out[i] *= s;
    } else if (U->gtype == GT_BF16) {
        for (size_t i = 0; i < U->cols; i++) out[i] = bf16f(((const uint16_t *)row)[i]);
    } else {
        memcpy(out, row, U->cols * sizeof(float));
    }
}

static void fold_rows(size_t b, size_t e, size_t tid, void *u)
{
    const FoldJob *j = u;
    float *buf = oc_parallel_scratch(tid, FOLD_ROWS * j->E * sizeof(float));
    if (!buf) return;
    for (size_t c = b; c < e; c++) {
        const size_t v0 = c * FOLD_ROWS, nv = j->V - v0 < FOLD_ROWS ? j->V - v0 : FOLD_ROWS;
        for (size_t k = 0; k < nv; k++) {
            lm_row(j->U, v0 + k, buf + k * j->E);
            j->b2[v0 + k] = ink_dot(buf + k * j->E, j->hb, j->E);
        }
        for (size_t r = 0; r < j->R; r++)
            for (size_t k = 0; k < nv; k++) j->w2f[(v0 + k) * j->R + r] = ink_dot(buf + k * j->E, j->Ht + r * j->E, j->E);
    }
}

static void quant_rows(size_t b, size_t e, size_t tid, void *u)
{
    (void)tid;
    const FoldJob *j = u;
    const size_t rb = oc_nvfp4_row_bytes(j->R);
    for (size_t v = b; v < e; v++) oc_nvfp4_quantize_row(j->w2f + v * j->R, j->R, j->gscale, j->w2 + v * rb);
}

/* w2 = U . hidden_out and w2_b = U . hidden_out_b, the reference fold of
 * the Markov head's hidden projection into the lm head. */
static OcError fold_markov(OcInklingDraft *d, const OcInklingModel *t)
{
    const size_t V = t->output.rows, E = d->n_embd, R = d->rank;
    if (t->output.cols != E || V < d->n_vocab) return OC_ERR_MODEL;
    FoldJob j = { &t->output, NULL, d->hidden_out_b, V, E, R, NULL, NULL, NULL, 0.0f };
    float *Ht = malloc(R * E * sizeof(float));
    j.w2f = malloc(V * R * sizeof(float));
    d->w2_b = malloc(V * sizeof(float));
    d->w2 = malloc(V * oc_nvfp4_row_bytes(R));
    if (!Ht || !j.w2f || !d->w2_b || !d->w2) { free(Ht); free(j.w2f); return OC_ERR_OOM; }
    for (size_t e = 0; e < E; e++)
        for (size_t r = 0; r < R; r++) Ht[r * E + e] = d->hidden_out[e * R + r];
    j.Ht = Ht;
    j.b2 = d->w2_b;
    oc_parallel_for((V + FOLD_ROWS - 1) / FOLD_ROWS, fold_rows, &j);
    j.gscale = oc_nvfp4_global_scale(j.w2f, V * R);
    j.w2 = d->w2;
    oc_parallel_for(V, quant_rows, &j);
    d->w2_scale = j.gscale;
    free(Ht);
    free(j.w2f);
    return OC_OK;
}

OcError oc_inkling_draft_load(const char *path, const OcInklingModel *target, OcInklingDraft *d)
{
    memset(d, 0, sizeof *d);
    OcError e = oc_gguf_map_open(path, &d->map);
    if (e != OC_OK) return e;
    const OcGgufMmappedFile *map = &d->map;
    d->n_layer = du32(map, "block_count", &e);
    d->n_embd = du32(map, "embedding_length", &e);
    d->n_ff = du32(map, "feed_forward_length", &e);
    d->n_head = du32(map, "attention.head_count", &e);
    d->n_head_kv = du32(map, "attention.head_count_kv", &e);
    d->head_dim = du32(map, "attention.key_length", &e);
    d->n_swa = du32(map, "attention.sliding_window", &e);
    d->block_size = du32(map, "block_size", &e);
    d->rank = du32(map, "markov_rank", &e);
    d->n_vocab = du32(map, "vocab_size", &e);
    d->eps = df32(map, "attention.layer_norm_rms_epsilon", &e);
    d->rope_base = df32(map, "rope.freq_base", &e);
    if (e != OC_OK) goto fail;
    if (d->n_layer > 4 || d->n_embd != target->n_embd || d->n_vocab != target->n_vocab ||
        d->block_size > DS_MAX_BLOCK || d->rank != 256 || d->head_dim % 2) {
        oc_log(OC_LOG_ERROR, "inkling-draft: drafter does not match the target model");
        e = OC_ERR_MODEL;
        goto fail;
    }
    const OcGgufMetadataValue *tl = oc_gguf_metadata_get(&map->unified, "inkling2_dspark.target_layers");
    const OcGgufMetadataValue *pat = oc_gguf_metadata_get(&map->unified, "inkling2_dspark.attention.sliding_window_pattern");
    if (!tl || tl->v.arr.len == 0 || tl->v.arr.len > OC_INK_MAX_TAPS || !pat || pat->v.arr.len != d->n_layer) {
        e = OC_ERR_FORMAT;
        goto fail;
    }
    d->n_targets = (uint32_t)tl->v.arr.len;
    for (uint32_t i = 0; i < d->n_targets; i++) d->target_layers[i] = tl->v.arr.values[i].v.i32;

    char n[128];
    for (uint32_t i = 0; i < d->n_layer && e == OC_OK; i++) {
        OcInkDraftLayer *L = &d->layers[i];
        L->swa = pat->v.arr.values[i].v.b;
#define N(s) (snprintf(n, sizeof n, "blk.%u." s, i), n)
        L->attn_norm = f32t(map, N("attn_norm.weight"), &e);
        L->q_norm = f32t(map, N("attn_q_norm.weight"), &e);
        L->k_norm = f32t(map, N("attn_k_norm.weight"), &e);
        L->k_ctx_norm = f32t(map, N("attn_k_ctx_norm.weight"), &e);
        L->ctx_norm = f32t(map, N("ctx_norm.weight"), &e);
        L->ctx_mix = f32t(map, N("ctx_mix.weight"), &e);
        L->ffn_norm = f32t(map, N("ffn_norm.weight"), &e);
        OcInkMat *mats[] = { &L->wq, &L->wk, &L->wv, &L->wo, &L->wk_ctx, &L->wv_ctx, &L->ffn_gate, &L->ffn_up, &L->ffn_down };
        const char *names[] = { "attn_q", "attn_k", "attn_v", "attn_output", "attn_k_ctx", "attn_v_ctx", "ffn_gate", "ffn_up", "ffn_down" };
        for (size_t k = 0; k < 9 && e == OC_OK; k++) {
            char nn[128];
            snprintf(nn, sizeof nn, "blk.%u.%s.weight", i, names[k]);
            e = oc_ink_load_mat(map, nn, mats[k], true);
        }
#undef N
    }
    if (e != OC_OK) goto fail;
    d->output_norm = f32t(map, "output_norm.weight", &e);
    d->mask_embd = f32t(map, "mask_embd.weight", &e);
    d->markov_ctx_mix = f32t(map, "markov_ctx_mix.weight", &e);
    d->markov_ctx_norm = f32t(map, "markov_ctx_norm.weight", &e);
    d->hidden_out = f32t(map, "markov_hidden_out.weight", &e);
    d->hidden_out_b = f32t(map, "markov_hidden_out.bias", &e);
    d->rnn_proj = f32t(map, "markov_rnn_proj.weight", &e);
    d->rnn_proj_b = f32t(map, "markov_rnn_proj.bias", &e);
    const OcGgufTensorInfo *w1 = oc_gguf_map_tensor_get(map, "markov_w1.weight");
    if (!w1 || w1->ggml_type != GT_BF16 || w1->dims[0] != d->rank) { e = OC_ERR_FORMAT; goto fail; }
    d->markov_w1 = (const uint16_t *)oc_gguf_map_tensor_data(map, w1);
    if (e == OC_OK) e = fold_markov(d, target);
    if (e == OC_OK) return OC_OK;
fail:
    free(d->w2);
    free(d->w2_b);
    oc_gguf_map_free(&d->map);
    return e;
}

void oc_inkling_draft_free(OcInklingDraft *d)
{
    if (!d) return;
    free(d->w2);
    free(d->w2_b);
    oc_gguf_map_free(&d->map);
    memset(d, 0, sizeof *d);
}

/* ─── linear helper (same row-chunked kernels as the target) ────────────── */

typedef struct {
    const OcInkMat   *W;
    const float      *x;
    size_t            x_stride;
    const OcNvfp4Act *acts;
    size_t            n;
    float            *y;
    size_t            y_stride;
} DJob;

static void d_rows(size_t b, size_t e, size_t tid, void *u)
{
    (void)tid;
    const DJob *j = u;
    const OcInkMat *W = j->W;
    size_t r0 = b * 16, r1 = e * 16;
    if (r1 > W->rows) r1 = W->rows;
    if (W->gtype == GT_NVFP4) {
        oc_nvfp4_gemm_rows(W->data, W->cols, r0, r1, W->scale ? W->scale[0] : 1.0f, j->acts, j->n, j->y, j->y_stride);
        return;
    }
    for (size_t r = r0; r < r1; r++)
        for (size_t t = 0; t < j->n; t++) {
            const float *x = j->x + t * j->x_stride;
            float s = 0.0f;
            if (W->gtype == GT_BF16) {
                s = ink_dot_bf16((const uint16_t *)(W->data + r * W->row_bytes), x, W->cols);
            } else {
                s = dotf((const float *)(W->data + r * W->row_bytes), x, W->cols);
            }
            j->y[t * j->y_stride + r] = s;
        }
}

typedef struct {
    OcNvfp4Act acts[DS_MAX_BLOCK + OC_INK_MAX_BATCH];
    int8_t    *q;
    float     *d;
    size_t     cols;
} DAct;

static void dlinear(DAct *a, const OcInkMat *W, const float *x, size_t x_stride, size_t n, float *y, size_t y_stride)
{
    if (W->gtype == GT_NVFP4) {
        for (size_t t = 0; t < n; t++) {
            a->acts[t].q = a->q + t * a->cols;
            a->acts[t].d = a->d + t * (a->cols / 64);
            oc_nvfp4_quantize_act(x + t * x_stride, W->cols, a->acts[t]);
        }
    }
    DJob j = { W, x, x_stride, a->acts, n, y, y_stride };
    oc_parallel_for((W->rows + 15) / 16, d_rows, &j);
}

/* ─── spec state ────────────────────────────────────────────────────────── */

struct OcInklingSpec {
    const OcInklingDraft *d;
    OcInklingSession     *s;
    const OcInklingModel *m;
    uint32_t n_ctx;
    float   *k[4], *v[4];
    uint32_t cap[4];
    uint32_t n_ctx_pos;                 /* context positions injected       */
    float    p_min;                     /* stop drafting below this top prob */
    float   *frnn;                      /* [DS_FRNN_RING][E]                */
    int64_t  frnn_pos[DS_FRNN_RING];
    DAct     act;
    double   prof[5];                   /* OC_INK_PROF: layers, base, walk, p_min, inject */
    InkGpuDraft *gd;                    /* drafter on the target session's GPU */
    uint64_t prof_rounds;
    /* scratch */
    float *x, *h, *xn, *q, *kb, *vb, *o, *a, *g, *u, *base, *rope_cos, *rope_sin;
};

static void rope_neox(float *x, size_t hd, uint32_t pos, float base)
{
    const size_t half = hd / 2;
    for (size_t i = 0; i < half; i++) {
        const float th = (float)pos * powf(base, -2.0f * (float)i / (float)hd);
        const float c = cosf(th), s = sinf(th);
        const float a = x[i], b = x[i + half];
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

OcError oc_inkling_spec_new(const OcInklingDraft *d, OcInklingSession *target, uint32_t n_ctx, OcInklingSpec **out)
{
    OcInklingSpec *sp = calloc(1, sizeof *sp);
    if (!sp) return OC_ERR_OOM;
    sp->d = d;
    sp->s = target;
    sp->n_ctx = n_ctx;
    const size_t E = d->n_embd, KV = (size_t)d->n_head_kv * d->head_dim, QD = (size_t)d->n_head * d->head_dim;
    const size_t B = OC_INK_MAX_BATCH > DS_MAX_BLOCK ? OC_INK_MAX_BATCH : DS_MAX_BLOCK;
    OcError e = OC_OK;
    for (uint32_t l = 0; l < d->n_layer; l++) {
        uint32_t cap = d->layers[l].swa ? d->n_swa + 64 : n_ctx;
        if (cap > n_ctx) cap = n_ctx;
        sp->cap[l] = cap;
        sp->k[l] = malloc((size_t)cap * KV * sizeof(float));
        sp->v[l] = malloc((size_t)cap * KV * sizeof(float));
        if (!sp->k[l] || !sp->v[l]) e = OC_ERR_OOM;
    }
    for (int i = 0; i < DS_FRNN_RING; i++) sp->frnn_pos[i] = -1;
    sp->act.cols = d->n_ff > E ? d->n_ff : (QD > E ? QD : E);
    sp->act.q = aligned_alloc(64, B * sp->act.cols);
    sp->act.d = calloc(B * (sp->act.cols / 64), sizeof(float));
#define AL(p, n) do { (p) = calloc((n), sizeof(float)); if (!(p)) e = OC_ERR_OOM; } while (0)
    AL(sp->frnn, DS_FRNN_RING * E);
    AL(sp->x, B * E); AL(sp->h, B * E); AL(sp->xn, B * E); AL(sp->q, B * QD); AL(sp->kb, B * KV);
    AL(sp->vb, B * KV); AL(sp->o, B * QD); AL(sp->a, B * E); AL(sp->g, B * d->n_ff); AL(sp->u, B * d->n_ff);
    AL(sp->base, (size_t)(DS_MAX_BLOCK + 1) * d->n_vocab);
#undef AL
    sp->m = NULL;
    if (!sp->act.q || !sp->act.d) e = OC_ERR_OOM;
    if (e != OC_OK) { oc_inkling_spec_free(sp); return e; }
    InkGpu *g = ink_session_gpu(target);
    if (g && !getenv("OC_INK_DS_CPU") && ink_gpu_draft_new(g, d, n_ctx, &sp->gd) != OC_OK) {
        oc_log(OC_LOG_WARN, "inkling-draft: drafter does not fit the GPU path, drafting on the CPU");
        sp->gd = NULL;
    }
    *out = sp;
    return OC_OK;
}

void oc_inkling_spec_free(OcInklingSpec *sp)
{
    if (!sp) return;
    if (getenv("OC_INK_PROF") && sp->prof_rounds) {
        static const char *const names[5] = { "layers", "base_logits", "markov_walk", "p_min", "inject" };
        for (int i = 0; i < 5; i++)
            fprintf(stderr, "[dspark-prof] %-12s %7.3f s  %7.3f ms/round\n", names[i], sp->prof[i],
                    sp->prof[i] * 1e3 / (double)sp->prof_rounds);
    }
    ink_gpu_draft_free(sp->gd);
    for (int l = 0; l < 4; l++) { free(sp->k[l]); free(sp->v[l]); }
    float *bufs[] = { sp->frnn, sp->x, sp->h, sp->xn, sp->q, sp->kb, sp->vb, sp->o, sp->a, sp->g, sp->u, sp->base };
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) free(bufs[i]);
    free(sp->act.q);
    free(sp->act.d);
    free(sp);
}

/* Mix the target taps of one token with weights mix[n_targets]. */
static void mix_taps(const float *taps, uint32_t n_targets, size_t E, const float *mix, float *out)
{
    memset(out, 0, E * sizeof(float));
    for (uint32_t i = 0; i < n_targets; i++)
        for (size_t c = 0; c < E; c++) out[c] += mix[i] * taps[i * E + c];
}

/* The Markov head's context state f_rnn of taps t0..t0+n of the last target
 * call, at positions pos0... */
static void inject_frnn(OcInklingSpec *sp, size_t t0, size_t n, uint32_t pos0)
{
    const OcInklingDraft *d = sp->d;
    const size_t E = d->n_embd;
    for (size_t t = 0; t < n; t++) {
        const uint32_t p = pos0 + (uint32_t)t;
        float *f = sp->frnn + (size_t)(p % DS_FRNN_RING) * E;
        mix_taps(oc_inkling_taps(sp->s, t0 + t), d->n_targets, E, d->markov_ctx_mix, f);
        rms_norm(f, d->markov_ctx_norm, E, d->eps, f);
        sp->frnn_pos[p % DS_FRNN_RING] = p;
    }
    sp->n_ctx_pos = pos0 + (uint32_t)n;
}

/* Taps t0..t0+n of the last target call, at positions pos0... */
static void inject_part(OcInklingSpec *sp, size_t t0, size_t n, uint32_t pos0)
{
    const OcInklingDraft *d = sp->d;
    const size_t E = d->n_embd, KV = (size_t)d->n_head_kv * d->head_dim, HD = d->head_dim;
    for (uint32_t l = 0; l < d->n_layer; l++) {
        const OcInkDraftLayer *L = &d->layers[l];
        for (size_t t = 0; t < n; t++) {
            mix_taps(oc_inkling_taps(sp->s, t0 + t), d->n_targets, E, L->ctx_mix, sp->x + t * E);
            rms_norm(sp->x + t * E, L->ctx_norm, E, d->eps, sp->x + t * E);
        }
        dlinear(&sp->act, &L->wk_ctx, sp->x, E, n, sp->kb, KV);
        dlinear(&sp->act, &L->wv_ctx, sp->x, E, n, sp->vb, KV);
        for (size_t t = 0; t < n; t++) {
            const uint32_t p = pos0 + (uint32_t)t;
            for (size_t g = 0; g < d->n_head_kv; g++) {
                float *k = sp->kb + t * KV + g * HD;
                rms_norm(k, L->k_ctx_norm, HD, d->eps, k);
                rope_neox(k, HD, p, d->rope_base);
            }
            const uint32_t slot = L->swa ? p % sp->cap[l] : p;
            memcpy(sp->k[l] + (size_t)slot * KV, sp->kb + t * KV, KV * sizeof(float));
            memcpy(sp->v[l] + (size_t)slot * KV, sp->vb + t * KV, KV * sizeof(float));
        }
    }
    inject_frnn(sp, t0, n, pos0);
}

OcError oc_inkling_spec_inject(OcInklingSpec *sp, size_t n)
{
    if (n > oc_inkling_session_max_batch(sp->s)) return OC_ERR_INVALID_ARG;
    const double t0 = now_s();
    const uint32_t pos0 = oc_inkling_session_pos(sp->s) - (uint32_t)n;
    if (sp->gd) {
        const OcError e = ink_gpu_draft_inject(sp->gd, (uint32_t)n, pos0);
        if (e != OC_OK) return e;
        inject_frnn(sp, 0, n, pos0);
    } else {
        for (size_t t = 0; t < n; t += OC_INK_MAX_BATCH) {
            const size_t c = n - t < OC_INK_MAX_BATCH ? n - t : OC_INK_MAX_BATCH;
            inject_part(sp, t, c, pos0 + (uint32_t)t);
        }
    }
    sp->prof[4] += now_s() - t0;
    return OC_OK;
}

typedef struct {
    OcInklingSpec *sp;
    uint32_t l, n_blk, pos0, lo;
} DAttn;

static void d_attn(size_t b, size_t e, size_t tid, void *u)
{
    const DAttn *j = u;
    OcInklingSpec *sp = j->sp;
    const OcInklingDraft *d = sp->d;
    const OcInkDraftLayer *L = &d->layers[j->l];
    const size_t HD = d->head_dim, KV = (size_t)d->n_head_kv * HD, QD = (size_t)d->n_head * HD;
    const size_t grp = d->n_head / d->n_head_kv;
    const uint32_t n_ctx_keys = j->pos0 - j->lo;
    float *sc = oc_parallel_scratch(tid, ((size_t)n_ctx_keys + DS_MAX_BLOCK + HD) * sizeof(float));
    if (!sc) return;
    float *acc = sc + n_ctx_keys + DS_MAX_BLOCK;
    const float ks = 1.0f / sqrtf((float)HD);
    for (size_t it = b; it < e; it++) {
        const size_t t = it / d->n_head, h = it % d->n_head, g = h / grp;
        const float *q = sp->q + t * QD + h * HD;
        float mx = -INFINITY;
        for (uint32_t k = 0; k < n_ctx_keys; k++) {
            const uint32_t p = j->lo + k, slot = L->swa ? p % sp->cap[j->l] : p;
            sc[k] = dotf(q, sp->k[j->l] + (size_t)slot * KV + g * HD, HD) * ks;
            if (sc[k] > mx) mx = sc[k];
        }
        for (uint32_t k = 0; k < j->n_blk; k++) {
            float v = dotf(q, sp->kb + k * KV + g * HD, HD) * ks;
            sc[n_ctx_keys + k] = v;
            if (v > mx) mx = v;
        }
        float den = 0.0f;
        memset(acc, 0, HD * sizeof(float));
        for (uint32_t k = 0; k < n_ctx_keys + j->n_blk; k++) {
            const float w = expf(sc[k] - mx);
            den += w;
            const float *vv;
            if (k < n_ctx_keys) {
                const uint32_t p = j->lo + k, slot = L->swa ? p % sp->cap[j->l] : p;
                vv = sp->v[j->l] + (size_t)slot * KV + g * HD;
            } else {
                vv = sp->vb + (k - n_ctx_keys) * KV + g * HD;
            }
            for (size_t i = 0; i < HD; i++) acc[i] += w * vv[i];
        }
        float *o = sp->o + t * QD + h * HD;
        for (size_t i = 0; i < HD; i++) o[i] = acc[i] / den;
    }
}

/* Drafter layers on the CPU: final normed hidden states of the nb block slots
 * into sp->xn, target lm head logits of slots 1..nb-1 into sp->base. */
static void draft_block_cpu(OcInklingSpec *sp, const OcInklingModel *m, uint32_t nb, uint32_t pos0)
{
    const OcInklingDraft *d = sp->d;
    const size_t E = d->n_embd, HD = d->head_dim, KV = (size_t)d->n_head_kv * HD, QD = (size_t)d->n_head * HD;
    const uint32_t n_max = nb - 1;
    double tp = now_s();

    for (uint32_t l = 0; l < d->n_layer; l++) {
        const OcInkDraftLayer *L = &d->layers[l];
        for (uint32_t t = 0; t < nb; t++) rms_norm(sp->h + t * E, L->attn_norm, E, d->eps, sp->xn + t * E);
        dlinear(&sp->act, &L->wq, sp->xn, E, nb, sp->q, QD);
        dlinear(&sp->act, &L->wk, sp->xn, E, nb, sp->kb, KV);
        dlinear(&sp->act, &L->wv, sp->xn, E, nb, sp->vb, KV);
        for (uint32_t t = 0; t < nb; t++) {
            for (size_t h = 0; h < d->n_head; h++) {
                float *q = sp->q + t * QD + h * HD;
                rms_norm(q, L->q_norm, HD, d->eps, q);
                rope_neox(q, HD, pos0 + t, d->rope_base);
            }
            for (size_t g = 0; g < d->n_head_kv; g++) {
                float *k = sp->kb + t * KV + g * HD;
                rms_norm(k, L->k_norm, HD, d->eps, k);
                rope_neox(k, HD, pos0 + t, d->rope_base);
            }
        }
        const uint32_t lo = L->swa && pos0 > d->n_swa ? pos0 - d->n_swa : 0;
        DAttn aj = { sp, l, nb, pos0, lo };
        oc_parallel_for((size_t)nb * d->n_head, d_attn, &aj);
        dlinear(&sp->act, &L->wo, sp->o, QD, nb, sp->a, E);
        for (size_t i = 0; i < nb * E; i++) sp->h[i] += sp->a[i];
        for (uint32_t t = 0; t < nb; t++) rms_norm(sp->h + t * E, L->ffn_norm, E, d->eps, sp->xn + t * E);
        dlinear(&sp->act, &L->ffn_gate, sp->xn, E, nb, sp->g, d->n_ff);
        dlinear(&sp->act, &L->ffn_up, sp->xn, E, nb, sp->u, d->n_ff);
        for (size_t i = 0; i < nb * d->n_ff; i++) sp->g[i] = sp->g[i] * sigmoidf(sp->g[i]) * sp->u[i];
        dlinear(&sp->act, &L->ffn_down, sp->g, d->n_ff, nb, sp->a, E);
        for (size_t i = 0; i < nb * E; i++) sp->h[i] += sp->a[i];
    }
    for (uint32_t t = 0; t < nb; t++) rms_norm(sp->h + t * E, d->output_norm, E, d->eps, sp->xn + t * E);
    sp->prof[0] += now_s() - tp;
    tp = now_s();

    /* base logits of the draft slots from the target lm head (no softcap yet) */
    const size_t V = d->n_vocab;
    {
        OcNvfp4Act acts[DS_MAX_BLOCK];
        for (uint32_t t = 0; t < n_max; t++) {
            acts[t].q = sp->act.q + t * sp->act.cols;
            acts[t].d = sp->act.d + t * (sp->act.cols / 64);
        }
        if (m->output.gtype == GT_NVFP4)
            for (uint32_t t = 0; t < n_max; t++) oc_nvfp4_quantize_act(sp->xn + (t + 1) * E, E, acts[t]);
        DJob j = { &m->output, sp->xn + E, E, acts, n_max, sp->base, V };
        oc_parallel_for((m->output.rows + 15) / 16, d_rows, &j);
    }
    sp->prof[1] += now_s() - tp;
}

OcError oc_inkling_spec_draft(OcInklingSpec *sp, uint32_t anchor, uint32_t n_max, uint32_t *out, uint32_t *n_out)
{
    const OcInklingDraft *d = sp->d;
    const OcInklingModel *m = NULL;
    *n_out = 0;
    const uint32_t pos0 = oc_inkling_session_pos(sp->s);
    if (n_max + 1 > d->block_size) n_max = d->block_size - 1;
    if (n_max == 0 || pos0 == 0 || sp->n_ctx_pos != pos0) return OC_OK;
    if (sp->frnn_pos[(pos0 - 1) % DS_FRNN_RING] != (int64_t)(pos0 - 1)) return OC_OK;
    const float *frnn = sp->frnn + (size_t)((pos0 - 1) % DS_FRNN_RING) * d->n_embd;
    m = oc_inkling_session_model(sp->s);
    double tp = now_s();
    sp->prof_rounds++;

    const size_t E = d->n_embd;
    const uint32_t nb = n_max + 1;

    /* block input: normed anchor embedding, then mask embeddings */
    for (size_t i = 0; i < E; i++) sp->h[i] = bf16f(m->tok_embd[(size_t)anchor * E + i]);
    rms_norm(sp->h, m->tok_norm, E, m->eps, sp->h);
    for (uint32_t t = 1; t < nb; t++) memcpy(sp->h + t * E, d->mask_embd, E * sizeof(float));

    if (sp->gd) {
        uint32_t tok[DS_MAX_BLOCK];
        float ptop[DS_MAX_BLOCK];
        const OcError e = ink_gpu_draft_forward(sp->gd, sp->h, nb, pos0, anchor, frnn, tok, ptop);
        if (e != OC_OK) return e;
        uint32_t n_ok = n_max;
        for (uint32_t i = 0; i < n_max; i++) {
            if (sp->p_min > 0.0f && ptop[i] < sp->p_min) { n_ok = i; break; }
            out[i] = tok[i];
        }
        sp->prof[0] += now_s() - tp;
        *n_out = n_ok;
        return OC_OK;
    }
    draft_block_cpu(sp, m, nb, pos0);
    tp = now_s();
    const size_t V = d->n_vocab;

    /* Markov walk: an RNN over [state, w1[prev], hidden_i, f_rnn] gives a
     * Markov vector per position; its bias w2 . markov + w2_b on the base
     * logits picks the token, which becomes prev of the next position. */
    const size_t R = d->rank;
    const OcInkMat w2m = { GT_NVFP4, d->w2, V, R, 1, oc_nvfp4_row_bytes(R), V * oc_nvfp4_row_bytes(R), &d->w2_scale };
    float *bias = sp->base + (size_t)n_max * V;
    float state[256] = { 0 }, joint[2 * 256 + 2 * 4096], markov[256];
    uint32_t prev = anchor, n_ok = n_max;
    const float cap = m->softcap;
    for (uint32_t i = 1; i <= n_max; i++) {
        memcpy(joint, state, R * sizeof(float));
        for (size_t r = 0; r < R; r++) joint[R + r] = bf16f(d->markov_w1[(size_t)prev * R + r]);
        memcpy(joint + 2 * R, sp->xn + i * E, E * sizeof(float));
        memcpy(joint + 2 * R + E, frnn, E * sizeof(float));
        const OcInkMat rp = { GT_F32, (const uint8_t *)d->rnn_proj, 3 * R, 2 * R + 2 * E, 1,
                              (2 * R + 2 * E) * sizeof(float), 3 * R * (2 * R + 2 * E) * sizeof(float), NULL };
        float jo[768];
        DJob rj = { &rp, joint, 2 * R + 2 * E, NULL, 1, jo, 3 * R };
        oc_parallel_for((3 * R + 15) / 16, d_rows, &rj);
        for (size_t r = 0; r < R; r++) {
            const float gate = sigmoidf(jo[r] + d->rnn_proj_b[r]), cand = tanhf(jo[R + r] + d->rnn_proj_b[R + r]);
            state[r] = cand + gate * (state[r] - cand);
            markov[r] = tanhf(jo[2 * R + r] + d->rnn_proj_b[2 * R + r]);
        }
        OcNvfp4Act ma = { sp->act.q, sp->act.d };
        oc_nvfp4_quantize_act(markov, R, ma);
        DJob bj = { &w2m, markov, R, &ma, 1, bias, V };
        oc_parallel_for((V + 15) / 16, d_rows, &bj);

        /* softcap is monotonic: the greedy pick is the argmax of base + bias */
        const float *base = sp->base + (size_t)(i - 1) * V;
        float bv = -INFINITY;
        uint32_t bi = 0;
        for (size_t v = 0; v < V; v++) {
            const float x = base[v] + bias[v] + d->w2_b[v];
            bias[v] = x;
            if (x > bv) { bv = x; bi = (uint32_t)v; }
        }
        if (sp->p_min > 0.0f) {
            /* p(top) = 1 / sum_v exp(c_v - c_top), c the softcapped logit.
             * Terms more than 20 below the top add < 2e-9 each, so only raw
             * logits above the matching threshold are visited. */
            const double tz = now_s();
            const float cm = cap > 0.0f ? cap * tanhf(bv / cap) : bv;
            float thr = bv - 20.0f;
            if (cap > 0.0f) thr = cm - 20.0f > -cap ? cap * atanhf((cm - 20.0f) / cap) : -INFINITY;
            double z = 0.0;
            for (size_t v = 0; v < V; v++) {
                if (bias[v] < thr) continue;
                const float c = cap > 0.0f ? cap * tanhf(bias[v] / cap) : bias[v];
                z += exp((double)(c - cm));
            }
            sp->prof[3] += now_s() - tz;
            if (1.0 / z < (double)sp->p_min) { n_ok = i - 1; break; }
        }
        out[i - 1] = bi;
        prev = bi;
    }
    sp->prof[2] += now_s() - tp;
    *n_out = n_ok;
    return OC_OK;
}

/* ─── generation ────────────────────────────────────────────────────────── */

static uint32_t argmax(const float *x, size_t n)
{
    size_t b = 0;
    for (size_t i = 1; i < n; i++) if (x[i] > x[b]) b = i;
    return (uint32_t)b;
}

static bool is_eog(const OcInklingModel *m, uint32_t t)
{
    return m->have_tok ? oc_tokenizer_is_eog(&m->tok, t) : false;
}

OcError oc_inkling_generate(OcInklingSession *s, OcInklingSpec *sp, const uint32_t *prompt, size_t n_prompt,
                            uint32_t n_predict, uint32_t draft_max, float p_min,
                            OcInklingTokenCb cb, void *user, OcInklingSpecStats *st)
{
    const OcInklingModel *m = oc_inkling_session_model(s);
    const size_t V = m->n_vocab;
    OcInklingSpecStats local = { 0 };
    if (!st) st = &local;
    memset(st, 0, sizeof *st);
    if (n_prompt == 0) return OC_ERR_INVALID_ARG;
    if (sp) sp->p_min = p_min;
    if (draft_max + 1 > OC_INK_MAX_BATCH) draft_max = OC_INK_MAX_BATCH - 1;
    if (draft_max > 15) draft_max = 15;
    float *logits = malloc((size_t)(draft_max + 1) * V * sizeof(float));
    if (!logits) return OC_ERR_OOM;
    OcError e = OC_OK;

    double t0 = now_s();
    const size_t chunk = oc_inkling_session_max_batch(s);
    for (size_t i = 0; i < n_prompt && e == OC_OK; i += chunk) {
        size_t c = n_prompt - i < chunk ? n_prompt - i : chunk;
        e = oc_inkling_forward(s, prompt + i, c, logits, false);
        if (e == OC_OK && sp) e = oc_inkling_spec_inject(sp, c);
    }
    st->n_prompt = n_prompt;
    st->t_prefill = now_s() - t0;
    if (e != OC_OK) { free(logits); return e; }

    t0 = now_s();
    uint32_t tok = argmax(logits, V);
    uint32_t emitted = 0;
    bool stop = false;
    if (!cb(tok, user) || is_eog(m, tok)) stop = true;
    emitted++;

    while (!stop && emitted < n_predict) {
        uint32_t batch[OC_INK_MAX_BATCH], nd = 0;
        batch[0] = tok;
        if (sp && draft_max > 0) {
            double td = now_s();
            uint32_t want = draft_max;
            if (want > n_predict - emitted) want = n_predict - emitted;
            e = oc_inkling_spec_draft(sp, tok, want, batch + 1, &nd);
            st->t_draft += now_s() - td;
            if (e != OC_OK) break;
        }
        double tv = now_s();
        e = oc_inkling_forward(s, batch, nd + 1, logits, true);
        st->t_verify += now_s() - tv;
        if (e != OC_OK) break;
        uint32_t acc = 0;
        while (acc < nd && argmax(logits + (size_t)acc * V, V) == batch[acc + 1]) acc++;
        if (getenv("OC_INK_DEBUG")) {
            fprintf(stderr, "\n[round] anchor %u draft:", batch[0]);
            for (uint32_t k = 0; k < nd; k++) fprintf(stderr, " %u", batch[k + 1]);
            fprintf(stderr, " | target:");
            for (uint32_t k = 0; k <= nd; k++) fprintf(stderr, " %u", argmax(logits + (size_t)k * V, V));
            fprintf(stderr, " | acc %u\n", acc);
        }
        uint32_t bonus = argmax(logits + (size_t)acc * V, V);
        if (acc < nd) {
            e = oc_inkling_rollback(s, acc + 1);
            if (e != OC_OK) break;
        }
        if (sp) {
            e = oc_inkling_spec_inject(sp, acc + 1);
            if (e != OC_OK) break;
        }
        st->rounds++;
        st->drafted += nd;
        st->accepted += acc;
        if (acc < 16) st->accept_hist[acc]++;
        for (uint32_t k = 0; k <= acc && !stop; k++) {
            const uint32_t t = k < acc ? batch[k + 1] : bonus;
            emitted++;
            if (!cb(t, user) || is_eog(m, t) || emitted >= n_predict) stop = true;
        }
        tok = bonus;
    }
    st->emitted = emitted;
    st->t_decode = now_s() - t0;
    free(logits);
    return e;
}
