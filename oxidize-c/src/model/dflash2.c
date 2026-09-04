/*
 * dflash2.c — DFlash 2 block-diffusion draft model implementation.
 *
 * Port of z-lab/dflash dflash/model.py (DFlash2DraftModel) for the
 * dependency-free C11 port. See dflash2.h for the architecture contract.
 *
 * Reference semantics (dflash/model.py):
 *   - The block forward processes `verify_size` noise rows (embeddings of
 *     [committed, mask, ..., mask]) in ONE parallel pass.
 *   - Attention: k_ctx/v_ctx from the fused target context rows, k_noise/
 *     v_noise from the (conv'd) block hidden; both k streams share one
 *     per-head k_norm; RoPE over true positions; bidirectional sliding
 *     window (is_causal=false, window 2048).
 *   - Grouped dynamic conv is block-local: x[i-t] zero-pads at the start
 *     of THIS forward's token span, not the absolute sequence.
 *   - Selector: top-k over target-lm_head logits, then a greedy (or
 *     temperature-sampled) path via
 *     edge(p->c) = <A[p] o proj(h), B[c]> + unary[c].
 */
/* clock_gettime (phase timing + benchmark clocks). */
#define _POSIX_C_SOURCE 200809L

#include "oxidize/dflash2.h"
#include "oxidize/parallel.h"
#include "oxidize/safetensors.h"
#include "oxidize/simd.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ─── Hugepage-aware allocation ─────────────────────────────────────── */
/*
 * The step's two big working sets (BF16 weights ~2.35 GB, target lm_head
 * ~1.18 GB) are fully streamed every propose step. With transparent
 * hugepages in madvise mode those plain mallocs land on 4 KB pages: a
 * 1.18 GB sequential scan walks ~300k pages and misses the dTLB on most
 * of them. Backing the big buffers with MADV_HUGEPAGE 2 MB pages keeps the
 * scan inside the L2 TLB. Rounds size up to 2 MB so madvise can coalesce. */
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>

static size_t df2_hp_pagesize(void)
{
    static long ps = -1;
    if (ps < 0) {
        ps = sysconf(_SC_PAGESIZE);
        if (ps <= 0) ps = 4096;
    }
    return (size_t)ps;
}

void *oc_dflash2_alloc_huge(size_t n)
{
    if (n < 1u << 20) return malloc(n);       /* small: plain malloc */
    size_t pg = df2_hp_pagesize();
    if (n > SIZE_MAX - (pg - 1)) return NULL;
    size_t n2 = (n + pg - 1) & ~(pg - 1);
    void *p = mmap(NULL, n2, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    /* No malloc fallback: free_huge releases >=1 MiB allocations with
     * munmap, so a fallback pointer would be freed with the wrong
     * deallocator. Callers treat NULL as OOM already. */
#ifdef MADV_HUGEPAGE
    if (madvise(p, n2, MADV_HUGEPAGE) != 0) { /* best effort; keep 4 KB */
    }
#endif
    return p;
}

void oc_dflash2_free_huge(void *p, size_t n)
{
    if (!p) return;
    if (n < 1u << 20) { free(p); return; }
    size_t pg = df2_hp_pagesize();
    if (n > SIZE_MAX - (pg - 1)) return;
    size_t n2 = (n + pg - 1) & ~(pg - 1);
    munmap(p, n2);
}

static void *df2_alloc_huge(size_t n) { return oc_dflash2_alloc_huge(n); }
static void df2_free_huge(void *p, size_t n) { oc_dflash2_free_huge(p, n); }
#else
void *oc_dflash2_alloc_huge(size_t n) { return malloc(n); }
void oc_dflash2_free_huge(void *p, size_t n) { (void)n; free(p); }
#define df2_alloc_huge(n) malloc(n)
#define df2_free_huge(p, n) free(p)
#endif


/* ─── Small helpers ─────────────────────────────────────────────────── */

static void rms_norm_row(const float *x, const float *w, float *out,
                         size_t n, float eps)
{
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / (float)n + eps);
    for (size_t i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

static bool df2_size_mul(size_t a, size_t b, size_t *out);
static bool df2_size_add(size_t a, size_t b, size_t *out);
static bool df2_float_extent(size_t count);

static float df2_dot(const float *a, const float *b, size_t n)
{
    float result;
    if (oc_simd_try_dflash2_dot_f32(a, b, n, &result)) return result;
    float acc = 0.0f;
    for (size_t i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}

/* ── Portable scalar helpers (both x86 and non-x86) ─────────────────── */

/* sum_i ap[i] * b[i] where ap = a * p is precomputed by the caller (it is
 * shared across all k candidates for one position). */
static float df2_dot_pre(const float *ap, const float *b, size_t n)
{
    return df2_dot(ap, b, n);
}

/* BF16 x F32 dot: the weight row is raw BF16 (the high 16 bits of an
 * F32), the activation is F32. Widening via cvtepu16 + 32-bit shift
 * (NOT cvtph_ps, which is IEEE half precision). */
static float df2_dot_bf16(const uint16_t *w, const float *x, size_t n)
{
    float result;
    if (oc_simd_try_dflash2_dot_bf16(w, x, n, &result)) return result;
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)w[i] << 16;
        float wv;
        memcpy(&wv, &bits, 4);
        sum += wv * x[i];
    }
    return sum;
}

/* Dot one weight row (F32 or BF16) against x. */
static float df2_dot_wrow(const OcDFlash2Weight *w, size_t row,
                          const float *x)
{
    if (w->bf16)
        return df2_dot_bf16(w->bf16 + row * w->cols, x, w->cols);
    return df2_dot(w->data + row * w->cols, x, w->cols);
}

/* Widen one weight row to F32 into dst (dst must hold w->cols floats).
 * Used by paths that need contiguous F32 rows (elementwise ops, gather). */
static void df2_widen_row(const OcDFlash2Weight *w, size_t row, float *dst)
{
    if (w->bf16) {
        const uint16_t *src = w->bf16 + row * w->cols;
        for (size_t i = 0; i < w->cols; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            memcpy(&dst[i], &bits, 4);
        }
    } else {
        memcpy(dst, w->data + row * w->cols, w->cols * sizeof(float));
    }
}

static void gemv_rows_dispatch(const float *w, size_t rows, size_t cols,
                               const float *x, float *out)
{
    if (oc_simd_try_dflash2_gemv_rows_f32(w, rows, cols, x, out)) return;
    for (size_t r = 0; r < rows; r++)
        out[r] = df2_dot(w + r * cols, x, cols);
}

/* ─── GEMV (parallel over rows for large matrices) ──────────────────── */

typedef struct OcDFlash2GemvCtx {
    const OcDFlash2Weight *w;
    const float *x;
    float *out;
} OcDFlash2GemvCtx;

static void dflash2_gemv_par_fn(size_t begin, size_t end, size_t tid, void *ud)
{
    (void)tid;
    OcDFlash2GemvCtx *c = (OcDFlash2GemvCtx *)ud;
    for (size_t r = begin; r < end; r++)
        c->out[r] = df2_dot_wrow(c->w, r, c->x);
}

void oc_dflash2_gemv(const float *w, size_t rows, size_t cols,
                     const float *x, float *out)
{
    /* Legacy F32-pointer entry (kept for the public API). */
    OcDFlash2Weight tmp = { (float *)(uintptr_t)w, NULL, rows, cols, 0,
                            NULL, NULL };
    OcDFlash2GemvCtx c = { &tmp, x, out };
    if (rows * cols < (size_t)32768) {
        gemv_rows_dispatch(w, rows, cols, x, out);
        return;
    }
    oc_parallel_for(rows, dflash2_gemv_par_fn, &c);
}

/* Weight-struct GEMV (handles BF16-resident weights). */
static void gemv_w(const OcDFlash2Weight *w, const float *x, float *out)
{
    if (!w->bf16 && w->rows * w->cols < (size_t)32768) {
        gemv_rows_dispatch(w->data, w->rows, w->cols, x, out);
        return;
    }
    OcDFlash2GemvCtx c = { w, x, out };
    oc_parallel_for(w->rows, dflash2_gemv_par_fn, &c);
}

/* GEMM: out[i, :] = W @ x[i, :] for each of `n` input rows.
 * w: [rows, cols]; x: [n, cols]; out: [n, rows]. Weight-stationary: each
 * worker owns a slice of W rows and streams each weight element exactly
 * once per region; the n input rows all reuse the streamed weight row. */
typedef struct OcDFlash2GemmCtx {
    const OcDFlash2Weight *w;
    size_t x_stride;   /* == w->cols */
    const float *x;
    float *out;
    size_t out_row_stride; /* == w->rows */
    size_t n;
} OcDFlash2GemmCtx;

/* out[i, :] = W @ x[i, :] for n input rows; weights streamed per worker. */

/* Batch size cap for the fused BF16 block-GEMM kernel: one W row is
 * loaded/widened once and FMAd against up to 8 x rows (8 accumulator
 * sets fit comfortably in registers; beyond that fall back per-row). */
#define OC_DF2_GEMM_FUSED_MAX_N 8u

static void dflash2_gemm_par_fn(size_t begin, size_t end, size_t tid,
                                void *ud)
{
    (void)tid;
    OcDFlash2GemmCtx *c = (OcDFlash2GemmCtx *)ud;
    /* W-row outer loop: weight streamed once per region, x rows stay in
     * cache (x is at most block*4096*4 = 128 KB for the block sizes here). */
    /* BF16 weights with a small x-row count: the fused kernel loads and
     * widens each W chunk once for ALL x rows (the per-row loop would
     * re-load + re-widen the row once per x row, which measured ~1.6x
     * slower streaming on cold DRAM). Falls back per-row for F32 weights
     * (small/synthetic) and n > 8. */
    if (c->w->bf16 && c->n <= OC_DF2_GEMM_FUSED_MAX_N) {
        const size_t H = c->w->cols;
        float out[OC_DF2_GEMM_FUSED_MAX_N];
        if (begin < end && oc_simd_try_dflash2_dot_bf16_batch(
                c->w->bf16 + begin * H, c->x, H, c->n, out)) {
            for (size_t i = 0; i < c->n; i++)
                c->out[i * c->out_row_stride + begin] = out[i];
            for (size_t r = begin + 1; r < end; r++) {
                if (oc_simd_try_dflash2_dot_bf16_batch(
                        c->w->bf16 + r * H, c->x, H, c->n, out)) {
                    for (size_t i = 0; i < c->n; i++)
                        c->out[i * c->out_row_stride + r] = out[i];
                } else {
                    for (size_t i = 0; i < c->n; i++)
                        c->out[i * c->out_row_stride + r] =
                            df2_dot_bf16(c->w->bf16 + r * H,
                                         c->x + i * H, H);
                }
            }
            return;
        }
    }
    for (size_t r = begin; r < end; r++) {
        for (size_t i = 0; i < c->n; i++)
            c->out[i * c->out_row_stride + r] =
                df2_dot_wrow(c->w, r, c->x + i * c->x_stride);
    }
}

/* n-row x blocking for ctx GEMMs: split x rows across workers instead of
 * W rows when n is large (x per worker then fits in cache while its W-row
 * slice streams once). Each (worker, x-block) pair streams W's full row
 * set once per block, so total W traffic = W * ceil(n / X_BLOCK). */
#define OC_DF2_GEMM_X_BLOCK 256u
static void dflash2_gemm_xblock_par_fn(size_t begin, size_t end, size_t tid,
                                       void *ud)
{
    (void)tid;
    OcDFlash2GemmCtx *c = (OcDFlash2GemmCtx *)ud;
    for (size_t i = begin; i < end; i++)
        for (size_t r = 0; r < c->w->rows; r++)
            c->out[i * c->out_row_stride + r] =
                df2_dot_wrow(c->w, r, c->x + i * c->x_stride);
}

static void gemm(const OcDFlash2Weight *w, const float *x, size_t n,
                 float *out)
{
    if (n == 0) return;
    if (n == 1) {
        gemv_w(w, x, out);
        return;
    }
    OcDFlash2GemmCtx c = { w, w->cols, x, out, w->rows, n };
    if (n >= OC_DF2_GEMM_X_BLOCK) {
        oc_parallel_for(n, dflash2_gemm_xblock_par_fn, &c);
        return;
    }
    oc_parallel_for(w->rows, dflash2_gemm_par_fn, &c);
}

/* Multi-GEMM: several weight matrices — each with its own x rows — run in
 * ONE pool dispatch over a unified row space (each slice maps its rows back
 * to the owning matrix). Saves per-dispatch wake overhead when a phase
 * strings together several small GEMMs (attention projections, mlp
 * gate+up). All matrices must share the same n (x-row count) and cols. */
typedef struct OcDFlash2MultiGemmCtx {
    const OcDFlash2Weight *ws[8]; /* [n_w] */
    const float *xs[8];           /* [n_w] x row blocks, each [n, cols] */
    float *outs[8];               /* [n_w] out matrices */
    size_t n_w;
    size_t row0[9];               /* unified-row offset per matrix (+end) */
    size_t total_rows;
    size_t n;                     /* shared x-row count per matrix */
} OcDFlash2MultiGemmCtx;

static void dflash2_multigemm_par_fn(size_t begin, size_t end, size_t tid,
                                     void *ud)
{
    (void)tid;
    OcDFlash2MultiGemmCtx *c = (OcDFlash2MultiGemmCtx *)ud;
    for (size_t ur = begin; ur < end; ur++) {
        size_t wsel = 0;
        while (wsel + 1 < c->n_w && ur >= c->row0[wsel + 1]) wsel++;
        const OcDFlash2Weight *w = c->ws[wsel];
        const float *x = c->xs[wsel];
        const size_t r = ur - c->row0[wsel];
        const size_t H = w->cols;
        float out[OC_DF2_GEMM_FUSED_MAX_N];
        if (w->bf16 && c->n <= OC_DF2_GEMM_FUSED_MAX_N &&
            oc_simd_try_dflash2_dot_bf16_batch(
                w->bf16 + r * H, x, H, c->n, out)) {
            for (size_t i = 0; i < c->n; i++)
                c->outs[wsel][i * w->rows + r] = out[i];
            continue;
        }
        for (size_t i = 0; i < c->n; i++)
            c->outs[wsel][i * w->rows + r] = df2_dot_wrow(w, r, x + i * H);
    }
}

/* Run up to 8 small GEMMs (same n, same cols) in one dispatch. */
static void gemm_multi(OcDFlash2MultiGemmCtx *c)
{
    c->total_rows = 0;
    for (size_t k = 0; k < c->n_w; k++) {
        c->row0[k] = c->total_rows;
        c->total_rows += c->ws[k]->rows;
    }
    c->row0[c->n_w] = c->total_rows;
    if (c->total_rows == 0 || c->n == 0) return;
    if (c->n == 1) {
        for (size_t k = 0; k < c->n_w; k++)
            gemv_w(c->ws[k], c->xs[k], c->outs[k]);
        return;
    }
    oc_parallel_for(c->total_rows, dflash2_multigemm_par_fn, c);
}

/* ─── Grouped dynamic causal conv ───────────────────────────────────── */

static void conv_mac(float *ob, const float *kb, const float *xb,
                     float dv, size_t n)
{
    if (oc_simd_try_dflash2_conv_mac(ob, kb, xb, dv, n)) return;
    for (size_t c = 0; c < n; c++) ob[c] += (kb[c] + dv) * xb[c];
}

/* Grouped-conv parallelization was measured and rejected: the block convs
 * are 8 rows x 256 groups of 16-lane MACs — ~64 B of writes per job — and
 * a pool dispatch over 2048 such jobs cost more than the serial MAC work
 * (127-129 ms/step vs 120.6 serial). Kept serial. */

/*
 * out[i, c] = sum_t (base[t, c] + dyn[i, t, g(c)]) * x[i - t, c]
 * dyn is [len, kernel * groups] with dyn[i, t*groups + g] the dynamic
 * increment for tap t, group g. x[i - t] zero-pads when i < t.
 */
void oc_dflash2_grouped_conv(const float *x, const float *dyn,
                             const float *base,
                             size_t len, size_t hidden,
                             size_t kernel, size_t group_size,
                             float *out)
{
    const size_t groups = hidden / group_size;
    memset(out, 0, len * hidden * sizeof(float));
    for (size_t i = 0; i < len; i++) {
        const float *drow = dyn + i * (kernel * groups);
        float *orow = out + i * hidden;
        for (size_t t = 0; t < kernel && t <= i; t++) {
            const float *xr = x + (i - t) * hidden;
            const float *kbase = base + t * hidden;
            for (size_t g = 0; g < groups; g++) {
                float dyn_t_g = drow[t * groups + g];
                const float *xb = xr + g * group_size;
                const float *kb = kbase + g * group_size;
                float *ob = orow + g * group_size;
                conv_mac(ob, kb, xb, dyn_t_g, group_size);
            }
        }
    }
}

/* ─── Selector scores ──────────────────────────────────────────────── */

void oc_dflash2_selector_scores(const float *proj_h, const float *A_p,
                                const float *B_k, const float *unary,
                                size_t k, float *scores)
{
    float ap[OC_DFLASH2_RANK];
    for (size_t i = 0; i < OC_DFLASH2_RANK; i++)
        ap[i] = A_p[i] * proj_h[i];
    for (size_t ki = 0; ki < k; ki++)
        scores[ki] = df2_dot_pre(ap, B_k + ki * OC_DFLASH2_RANK,
                                OC_DFLASH2_RANK) + unary[ki];
}

/* ─── Config ───────────────────────────────────────────────────────── */

void oc_dflash2_config_init(OcDFlash2Config *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->hidden_size = 4096;
    cfg->intermediate_size = 12288;
    cfg->num_hidden_layers = 5;
    cfg->num_attention_heads = 32;
    cfg->num_key_value_heads = 8;
    cfg->head_dim = 128;
    cfg->vocab_size = 154880;
    cfg->num_target_layers = 45;
    static const size_t ids[5] = { 5, 14, 24, 33, 42 };
    cfg->n_target_layer_ids = 5;
    for (size_t i = 0; i < 5; i++) cfg->target_layer_ids[i] = ids[i];
    cfg->block_size = 8;
    cfg->conv_kernel_size = 2;
    cfg->conv_group_size = 16;
    cfg->mask_token_id = 154856;
    cfg->selector_rank = 256;
    cfg->selector_top_k = 16;
    cfg->sliding_window = 2048;
    cfg->rope_theta = 10000.0f;
    cfg->rms_norm_eps = 1e-5f;
}

/* ─── KV ring ──────────────────────────────────────────────────────── */

OcError oc_dflash2_kvring_init(OcDFlash2KvRing *ring, size_t capacity,
                               size_t n_kv_heads, size_t head_dim)
{
    if (!ring || capacity == 0 || n_kv_heads == 0 || head_dim == 0)
        return OC_ERR_INVALID_ARG;
    size_t vec, elems;
    if (!df2_size_mul(n_kv_heads, head_dim, &vec) ||
        !df2_size_mul(capacity, vec, &elems) ||
        !df2_float_extent(elems) ||
        capacity > SIZE_MAX / sizeof(int64_t))
        return OC_ERR_INVALID_ARG;
    memset(ring, 0, sizeof(*ring));
    ring->capacity = capacity;
    ring->n_kv_heads = n_kv_heads;
    ring->head_dim = head_dim;
    ring->k = malloc(elems * sizeof(float));
    ring->v = malloc(elems * sizeof(float));
    ring->pos = malloc(capacity * sizeof(int64_t));
    if (!ring->k || !ring->v || !ring->pos) {
        oc_dflash2_kvring_free(ring);
        return OC_ERR_OOM;
    }
    return OC_OK;
}

void oc_dflash2_kvring_free(OcDFlash2KvRing *ring)
{
    if (!ring) return;
    free(ring->k);
    free(ring->v);
    free(ring->pos);
    free(ring->undo_k);
    free(ring->undo_v);
    free(ring->undo_pos);
    memset(ring, 0, sizeof(*ring));
}

void oc_dflash2_kvring_clear(OcDFlash2KvRing *ring)
{
    if (!ring) return;
    free(ring->undo_k);
    free(ring->undo_v);
    free(ring->undo_pos);
    ring->undo_k = NULL;
    ring->undo_v = NULL;
    ring->undo_pos = NULL;
    ring->undo_n = 0;
    ring->len = 0;
    ring->total = 0;
}

OcError oc_dflash2_kvring_append(OcDFlash2KvRing *ring,
                                 const float *k, const float *v,
                                 int64_t pos0, size_t n)
{
    if (!ring || !k || !v) return OC_ERR_INVALID_ARG;
    size_t vec, undo_elems, new_total;
    if (ring->capacity == 0 || n > ring->capacity ||
        !df2_size_mul(ring->n_kv_heads, ring->head_dim, &vec) ||
        !df2_size_mul(n, vec, &undo_elems) ||
        !df2_float_extent(undo_elems) ||
        n > SIZE_MAX / sizeof(int64_t) ||
        !df2_size_add(ring->total, n, &new_total) ||
        (n > 0 && ((n - 1) > (size_t)INT64_MAX ||
                   pos0 > INT64_MAX - (int64_t)(n - 1))))
        return OC_ERR_INVALID_ARG;
    free(ring->undo_k);
    free(ring->undo_v);
    free(ring->undo_pos);
    ring->undo_k = NULL;
    ring->undo_v = NULL;
    ring->undo_pos = NULL;
    ring->undo_n = 0;

    if (n > ring->capacity - ring->len) {
        ring->undo_k = malloc(undo_elems * sizeof(float));
        ring->undo_v = malloc(undo_elems * sizeof(float));
        ring->undo_pos = malloc(n * sizeof(int64_t));
        if (!ring->undo_k || !ring->undo_v || !ring->undo_pos) {
            free(ring->undo_k);
            free(ring->undo_v);
            free(ring->undo_pos);
            ring->undo_k = NULL;
            ring->undo_v = NULL;
            ring->undo_pos = NULL;
            return OC_ERR_OOM;
        }
        ring->undo_total = ring->total;
        ring->undo_n = n;
        for (size_t i = 0; i < n; i++) {
            const size_t slot = (ring->total + i) % ring->capacity;
            memcpy(ring->undo_k + i * vec, ring->k + slot * vec,
                   vec * sizeof(float));
            memcpy(ring->undo_v + i * vec, ring->v + slot * vec,
                   vec * sizeof(float));
            ring->undo_pos[i] = ring->pos[slot];
        }
    }
    for (size_t i = 0; i < n; i++) {
        size_t slot = (ring->total + i) % ring->capacity;
        memcpy(ring->k + slot * vec, k + i * vec, vec * sizeof(float));
        memcpy(ring->v + slot * vec, v + i * vec, vec * sizeof(float));
        ring->pos[slot] = pos0 + (int64_t)i;
    }
    ring->total = new_total;
    ring->len = ring->total < ring->capacity ? ring->total : ring->capacity;
    return OC_OK;
}

void oc_dflash2_kvring_trim(OcDFlash2KvRing *ring, int64_t pos_keep_exclusive)
{
    if (!ring) return;
    const size_t appended_total = ring->total;
    const size_t scan_floor = ring->undo_n > 0 ? ring->undo_total : 0;
    /* Drop the newest entries with pos >= pos_keep_exclusive. Writes are
     * in increasing position order, so scan write indices from the end. */
    while (ring->total > scan_floor) {
        size_t w = ring->total - 1;
        size_t slot = w % ring->capacity;
        if (ring->pos[slot] >= pos_keep_exclusive)
            ring->total--;
        else
            break;
    }
    if (ring->undo_n > 0 && ring->total >= ring->undo_total) {
        const size_t first = ring->total - ring->undo_total;
        const size_t end = appended_total - ring->undo_total;
        const size_t vec = ring->n_kv_heads * ring->head_dim;
        for (size_t i = first; i < end && i < ring->undo_n; i++) {
            const size_t slot = (ring->undo_total + i) % ring->capacity;
            memcpy(ring->k + slot * vec, ring->undo_k + i * vec,
                   vec * sizeof(float));
            memcpy(ring->v + slot * vec, ring->undo_v + i * vec,
                   vec * sizeof(float));
            ring->pos[slot] = ring->undo_pos[i];
        }
    }
    /* If the entire latest append was removed, its overwritten slots are
     * restored now and an older trim can safely continue into history. */
    if (ring->undo_n > 0 && ring->total == ring->undo_total) {
        while (ring->total > 0) {
            const size_t slot = (ring->total - 1) % ring->capacity;
            if (ring->pos[slot] < pos_keep_exclusive) break;
            ring->total--;
        }
    }
    free(ring->undo_k);
    free(ring->undo_v);
    free(ring->undo_pos);
    ring->undo_k = NULL;
    ring->undo_v = NULL;
    ring->undo_pos = NULL;
    ring->undo_n = 0;
    ring->len = ring->total < ring->capacity ? ring->total : ring->capacity;
}

/* ─── Weight loading ────────────────────────────────────────────────── */

/* Validated raw payload for a tensor: routes the access through
 * oc_safetensors_get_tensor_data (offset+length bounds-checked against
 * the file) and additionally requires the descriptor's declared byte
 * range to cover EXACTLY the requested dtype payload — a shorter (or
 * padded) range is malformed metadata and must fail the load, not read
 * past the mapping. Sets *dtype_sz_out (4 = F32, 2 = BF16). */
static const uint8_t *df2_tensor_raw(const OcSafetensorsFile *f,
                                     const OcSafetensorsTensor *t,
                                     size_t expect_elems,
                                     size_t *dtype_sz_out)
{
    const size_t dtype_sz = strcmp(t->dtype, "F32") == 0 ? 4
                          : (strcmp(t->dtype, "BF16") == 0 ? 2 : 0);
    if (dtype_sz == 0) return NULL;
    /* Shape arithmetic can wrap on adversarial dimensions; the
     * byte-count math below must not. */
    if (expect_elems > SIZE_MAX / dtype_sz) return NULL;
    const void *raw = NULL;
    if (oc_safetensors_get_tensor_data(f, t, &raw) != OC_OK) return NULL;
    if (t->data_length != (uint64_t)expect_elems * (uint64_t)dtype_sz)
        return NULL;
    *dtype_sz_out = dtype_sz;
    return (const uint8_t *)raw;
}

static float *load_tensor_f32(const OcSafetensorsFile *f,
                              const char *name, size_t expect_elems)
{
    const OcSafetensorsTensor *t = NULL;
    for (size_t i = 0; i < f->n_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) {
            t = &f->tensors[i];
            break;
        }
    }
    if (!t) return NULL;
    size_t elems = 1;
    for (uint32_t d = 0; d < t->n_dims; d++)
        if (!df2_size_mul(elems, (size_t)t->shape[d], &elems)) return NULL;
    if (elems != expect_elems) return NULL;
    if (expect_elems > SIZE_MAX / sizeof(float)) return NULL;
    size_t dtype_sz = 0;
    const uint8_t *raw = df2_tensor_raw(f, t, expect_elems, &dtype_sz);
    if (!raw) return NULL;
    float *dst = malloc(expect_elems * sizeof(float));
    if (!dst) return NULL;
    if (dtype_sz == 4) {
        memcpy(dst, raw, expect_elems * 4);
    } else {
        for (size_t i = 0; i < expect_elems; i++) {
            uint16_t h;
            memcpy(&h, raw + i * 2, 2);
            uint32_t bits = (uint32_t)h << 16;
            memcpy(&dst[i], &bits, 4);
        }
    }
    return dst;
}

/* Load a [rows, cols] weight; returns OC_ERR_FORMAT when missing/mismatched.
 * Tensors at least BF16_KEEP_MIN_ELEMS elements are kept in raw BF16
 * (halving resident + streamed bytes; BF16->F32 widening is exact so
 * numerics are unchanged); smaller ones are widened to F32 for the plain
 * scalar paths (norms, conv bases, selector projections). */
#define OC_DF2_BF16_KEEP_MIN_ELEMS ((size_t)1 << 18)

static OcError load_w(const OcSafetensorsFile *f, const char *name,
                      OcDFlash2Weight *w, size_t rows, size_t cols)
{
    w->generate = NULL;
    w->gen_user = NULL;
    w->bf16 = NULL;
    size_t elems;
    if (!df2_size_mul(rows, cols, &elems)) return OC_ERR_FORMAT;

    /* Locate the tensor once; reuse the header walk for both paths. */
    const OcSafetensorsTensor *t = NULL;
    for (size_t i = 0; i < f->n_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) {
            t = &f->tensors[i];
            break;
        }
    }
    if (!t) {
        fprintf(stderr, "dflash2: missing tensor %s\n", name);
        w->rows = rows;
        w->cols = cols;
        w->data = NULL;
        return OC_ERR_FORMAT;
    }
    size_t got = 1;
    for (uint32_t d = 0; d < t->n_dims; d++)
        if (!df2_size_mul(got, (size_t)t->shape[d], &got))
            return OC_ERR_FORMAT;
    if (got != elems) {
        fprintf(stderr, "dflash2: bad tensor %s\n", name);
        w->rows = rows;
        w->cols = cols;
        w->data = NULL;
        return OC_ERR_FORMAT;
    }
    /* Range + dtype validation via the shared df2_tensor_raw (see its
     * comment): the BF16-keep path below copies directly from the raw
     * section, so it must not run on a malformed descriptor either.
     * The exact-length check here also catches an F32 payload where
     * BF16 is required. */
    size_t dtype_sz = 0;
    const uint8_t *raw = df2_tensor_raw(f, t, elems, &dtype_sz);
    if (!raw) {
        fprintf(stderr, "dflash2: bad tensor range %s\n", name);
        w->rows = rows;
        w->cols = cols;
        w->data = NULL;
        return OC_ERR_FORMAT;
    }

    if (dtype_sz == 2 && elems >= OC_DF2_BF16_KEEP_MIN_ELEMS) {
        size_t bytes = elems * sizeof(uint16_t);
        w->bf16 = df2_alloc_huge(bytes);
        w->data = NULL;
        w->rows = rows;
        w->cols = cols;
        if (!w->bf16) return OC_ERR_OOM;
        w->alloc_bytes = bytes;
        memcpy(w->bf16, raw, elems * 2);
        return OC_OK;
    }
    w->data = load_tensor_f32(f, name, elems);
    w->rows = rows;
    w->cols = cols;
    if (!w->data) {
        fprintf(stderr, "dflash2: missing/bad tensor %s\n", name);
        return OC_ERR_FORMAT;
    }
    return OC_OK;
}

static OcError load_vec(const OcSafetensorsFile *f, const char *name,
                        float **v, size_t n)
{
    *v = load_tensor_f32(f, name, n);
    if (!*v) fprintf(stderr, "dflash2: missing/bad tensor %s\n", name);
    return *v ? OC_OK : OC_ERR_FORMAT;
}

/* Pull a numeric JSON field (int or float) from a flat config.json buffer.
 * Handles "key":value and "key" : value spacing; returns 1 when found. */
static int cfg_num(const char *json, const char *key, double *out)
{
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++;
    if (*p != ':') return -1;
    p++;
    while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++;
    char *end = NULL;
    double v = strtod(p, &end);
    if (end == p) return -1;
    const char *tail = end;
    while (*tail == ' ' || *tail == '\n' || *tail == '\t' || *tail == '\r')
        tail++;
    if (*tail != ',' && *tail != '}' && *tail != ']') return -1;
    *out = v;
    return 1;
}

static bool cfg_optional_size(const char *json, const char *key, size_t *out)
{
    double v;
    int found = cfg_num(json, key, &v);
    if (found == 0) return true;
    if (found < 0) return false;
    if (!isfinite(v) || v < 0.0 || floor(v) != v ||
        v >= (double)SIZE_MAX)
        return false;
    *out = (size_t)v;
    return true;
}

static bool cfg_optional_u32(const char *json, const char *key, uint32_t *out)
{
    double v;
    int found = cfg_num(json, key, &v);
    if (found == 0) return true;
    if (found < 0) return false;
    if (!isfinite(v) || v < 0.0 || floor(v) != v || v > UINT32_MAX)
        return false;
    *out = (uint32_t)v;
    return true;
}

static bool cfg_optional_float(const char *json, const char *key, float *out)
{
    double v;
    int found = cfg_num(json, key, &v);
    if (found == 0) return true;
    if (found < 0) return false;
    if (!isfinite(v) || v < -FLT_MAX || v > FLT_MAX) return false;
    *out = (float)v;
    return true;
}

static bool df2_size_mul(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool df2_size_add(size_t a, size_t b, size_t *out)
{
    if (b > SIZE_MAX - a) return false;
    *out = a + b;
    return true;
}

static bool df2_float_extent(size_t count)
{
    return count <= SIZE_MAX / sizeof(float);
}

/* Validate all config-derived extents used by tensor shapes, KV storage,
 * and the largest block-forward scratch buffers before opening weights. */
static bool df2_config_extents_valid(const OcDFlash2Config *cfg)
{
    size_t target_width, q_rows, kv_rows, kernel_rows, groups, capacity;
    size_t extent, context_rows, slot_floats;

    if (!df2_size_mul(cfg->n_target_layer_ids, cfg->hidden_size,
                      &target_width) ||
        !df2_size_mul(cfg->num_attention_heads, cfg->head_dim, &q_rows) ||
        !df2_size_mul(cfg->num_key_value_heads, cfg->head_dim, &kv_rows) ||
        !df2_size_mul(2, cfg->conv_kernel_size, &kernel_rows) ||
        !df2_size_add(cfg->sliding_window, cfg->block_size, &capacity))
        return false;
    groups = cfg->hidden_size / cfg->conv_group_size;
    if (!df2_size_mul(kernel_rows, groups, &kernel_rows)) return false;

#define DF2_FLOAT_PRODUCT(a, b) \
    (df2_size_mul((a), (b), &extent) && df2_float_extent(extent))
    if (!DF2_FLOAT_PRODUCT(cfg->hidden_size, target_width) ||
        !DF2_FLOAT_PRODUCT(cfg->vocab_size, cfg->selector_rank) ||
        !DF2_FLOAT_PRODUCT(cfg->selector_rank, cfg->hidden_size) ||
        !DF2_FLOAT_PRODUCT(q_rows, cfg->hidden_size) ||
        !DF2_FLOAT_PRODUCT(kv_rows, cfg->hidden_size) ||
        !DF2_FLOAT_PRODUCT(kernel_rows, cfg->hidden_size) ||
        !DF2_FLOAT_PRODUCT(cfg->intermediate_size, cfg->hidden_size) ||
        !DF2_FLOAT_PRODUCT(capacity, kv_rows) ||
        !DF2_FLOAT_PRODUCT(capacity, cfg->hidden_size) ||
        !DF2_FLOAT_PRODUCT(cfg->block_size, cfg->hidden_size) ||
        !DF2_FLOAT_PRODUCT(cfg->block_size, q_rows) ||
        !DF2_FLOAT_PRODUCT(cfg->block_size, kv_rows) ||
        !DF2_FLOAT_PRODUCT(cfg->block_size, kernel_rows) ||
        !DF2_FLOAT_PRODUCT(cfg->block_size, cfg->intermediate_size))
        return false;
#undef DF2_FLOAT_PRODUCT

    if (!df2_size_add(capacity, cfg->block_size, &context_rows) ||
        !df2_size_mul(context_rows, kv_rows, &extent) ||
        !df2_float_extent(extent))
        return false;

    if (!df2_size_mul(kv_rows, 2, &slot_floats) ||
        !df2_float_extent(slot_floats) ||
        !df2_size_mul(slot_floats, sizeof(float), &extent) ||
        !df2_size_add(extent, sizeof(int64_t), &extent) ||
        !df2_size_mul(extent, cfg->block_size, &extent) ||
        !df2_size_mul(extent, cfg->num_hidden_layers, &extent))
        return false;

    return capacity <= SIZE_MAX / sizeof(int64_t) &&
           cfg->num_hidden_layers <= SIZE_MAX / sizeof(OcDFlash2Layer) &&
           cfg->num_hidden_layers <= SIZE_MAX / sizeof(OcDFlash2KvRing) &&
           cfg->head_dim / 2 <= SIZE_MAX / sizeof(float);
}

/* Parse the model's config.json into cfg. Missing fields keep their
 * defaults from oc_dflash2_config_init, so GLM-only configs work even if
 * they omit fields (e.g. head_dim). */
static bool parse_config_json(OcDFlash2Config *cfg, const char *json)
{
    if (!cfg_optional_size(json, "hidden_size", &cfg->hidden_size) ||
        !cfg_optional_size(json, "intermediate_size", &cfg->intermediate_size) ||
        !cfg_optional_size(json, "num_hidden_layers", &cfg->num_hidden_layers) ||
        !cfg_optional_size(json, "num_attention_heads", &cfg->num_attention_heads) ||
        !cfg_optional_size(json, "num_key_value_heads", &cfg->num_key_value_heads) ||
        !cfg_optional_size(json, "head_dim", &cfg->head_dim) ||
        !cfg_optional_size(json, "vocab_size", &cfg->vocab_size) ||
        !cfg_optional_size(json, "num_target_layers", &cfg->num_target_layers) ||
        !cfg_optional_size(json, "block_size", &cfg->block_size) ||
        !cfg_optional_size(json, "conv_kernel_size", &cfg->conv_kernel_size) ||
        !cfg_optional_size(json, "conv_group_size", &cfg->conv_group_size) ||
        !cfg_optional_u32(json, "mask_token_id", &cfg->mask_token_id) ||
        !cfg_optional_size(json, "selector_rank", &cfg->selector_rank) ||
        !cfg_optional_size(json, "selector_top_k", &cfg->selector_top_k) ||
        !cfg_optional_size(json, "sliding_window", &cfg->sliding_window) ||
        !cfg_optional_float(json, "rope_theta", &cfg->rope_theta) ||
        !cfg_optional_float(json, "rms_norm_eps", &cfg->rms_norm_eps))
        return false;
    /* target_layer_ids lives inside the nested dflash_config object; scan
     * for the array and read up to OC_DFLASH2_MAX_TARGET_LAYERS ints. */
    const char *t = strstr(json, "\"target_layer_ids\"");
    if (t) {
        const char *ob = strchr(t, '[');
        if (ob) {
            size_t n = 0;
            const char *p = ob + 1;
            while (n < OC_DFLASH2_MAX_TARGET_LAYERS) {
                while (*p == ' ' || *p == '\n' || *p == '\t' ||
                       *p == '\r' || *p == ',') p++;
                if (*p == ']' || *p == '\0') break;
                errno = 0;
                char *end = NULL;
                if (*p == '-') return false;
                unsigned long long id = strtoull(p, &end, 10);
                if (end == p || errno == ERANGE || id >= (unsigned long long)SIZE_MAX)
                    return false;
                const char *tail = end;
                while (*tail == ' ' || *tail == '\n' || *tail == '\t' ||
                       *tail == '\r') tail++;
                if (*tail != ',' && *tail != ']') return false;
                p = end;
                cfg->target_layer_ids[n++] = (size_t)id;
            }
            if (n) cfg->n_target_layer_ids = n;
        }
    }
    return true;
}

OcError oc_dflash2_model_load(OcDFlash2Model *m, const char *st_path,
                              const char *config_json_path)
{
    if (!m || !st_path) return OC_ERR_INVALID_ARG;
    memset(m, 0, sizeof(*m));
    oc_dflash2_config_init(&m->cfg);
    if (config_json_path) {
        /* An explicitly supplied config must actually read. A missing or
         * unreadable file silently fell back to the GLM defaults before,
         * which would misinterpret every tensor below. */
        FILE *fh = fopen(config_json_path, "rb");
        if (!fh) {
            fprintf(stderr, "dflash2: cannot open config %s\n",
                    config_json_path);
            return OC_ERR_IO;
        }
        char *buf = malloc(1 << 16);
        if (!buf) { fclose(fh); return OC_ERR_OOM; }
        size_t got = fread(buf, 1, (1 << 16) - 1, fh);
        bool io_err = ferror(fh) != 0;
        fclose(fh);
        if (io_err || got == 0) {
            free(buf);
            fprintf(stderr, "dflash2: config %s is empty/unreadable\n",
                    config_json_path);
            return io_err ? OC_ERR_IO : OC_ERR_FORMAT;
        }
        buf[got] = '\0';
        bool parsed = parse_config_json(&m->cfg, buf);
        /* Sanity: a config that yields no hidden_size at all was not a
         * DFlash2 config.json — garbage JSON would otherwise keep the
         * GLM defaults and load the checkpoint with wrong dimensions. */
        double probe;
        bool has_hidden = cfg_num(buf, "hidden_size", &probe) == 1;
        free(buf);
        if (!parsed || !has_hidden) {
            fprintf(stderr, "dflash2: config %s is invalid or has no hidden_size\n",
                    config_json_path);
            return OC_ERR_FORMAT;
        }
    }

    /* oc_dflash2_selector_scores is specialized to OC_DFLASH2_RANK (the
     * codebook stride everywhere); reject checkpoints that configure a
     * different rank before any tensor is loaded. */
    if (m->cfg.selector_rank != OC_DFLASH2_RANK) {
        fprintf(stderr, "dflash2: unsupported selector_rank %zu (need %d)\n",
                m->cfg.selector_rank, OC_DFLASH2_RANK);
        return OC_ERR_FORMAT;
    }

    /* Validate every dimension the derive step turns into a tensor shape,
     * VLA size, or ring capacity, before anything is derived from it. A
     * malformed config (zero or oversized field) must fail the load here
     * rather than under-allocate downstream. conv_group_size is checked
     * BEFORE the hidden_size % conv_group_size divide. */
    if (m->cfg.hidden_size == 0 ||
        m->cfg.intermediate_size == 0 ||
        m->cfg.num_hidden_layers == 0 ||
        m->cfg.num_hidden_layers > OC_DFLASH2_MAX_LAYERS ||
        m->cfg.num_attention_heads == 0 ||
        m->cfg.num_key_value_heads == 0 ||
        m->cfg.head_dim == 0 ||
        m->cfg.head_dim % 2 != 0 ||   /* RoPE splits head_dim in half */
        m->cfg.n_target_layer_ids == 0 ||
        m->cfg.n_target_layer_ids > OC_DFLASH2_MAX_TARGET_LAYERS ||
        m->cfg.num_attention_heads % m->cfg.num_key_value_heads != 0 ||
        m->cfg.conv_group_size == 0 ||
        m->cfg.hidden_size % m->cfg.conv_group_size != 0 ||
        m->cfg.conv_kernel_size == 0 ||
        m->cfg.block_size == 0 ||
        m->cfg.block_size > OC_DFLASH2_MAX_BLOCK ||
        m->cfg.selector_top_k == 0 ||
        m->cfg.selector_top_k > OC_DFLASH2_MAX_TOP_K ||
        m->cfg.sliding_window == 0 ||
        m->cfg.vocab_size == 0 ||
        !isfinite(m->cfg.rope_theta) || m->cfg.rope_theta <= 0.0f ||
        !isfinite(m->cfg.rms_norm_eps) || m->cfg.rms_norm_eps <= 0.0f ||
        !df2_config_extents_valid(&m->cfg)) {
        fprintf(stderr, "dflash2: invalid config dimensions\n");
        return OC_ERR_FORMAT;
    }

    OcSafetensorsFile f;
    OcError e = oc_safetensors_open(st_path, &f);
    if (e != OC_OK) return e;

    const size_t H = m->cfg.hidden_size;
    const size_t n_layers = m->cfg.num_hidden_layers;
    const size_t n_target_w = m->cfg.n_target_layer_ids * H;

    e = load_w(&f, "fc.weight", &m->fc, H, n_target_w);
    if (e == OC_OK) e = load_vec(&f, "hidden_norm.weight", &m->hidden_norm, H);
    if (e == OC_OK) e = load_vec(&f, "norm.weight", &m->norm, H);
    if (e == OC_OK)
        e = load_w(&f, "candidate_selector.predecessor_codebook",
                   &m->selector.predecessor_codebook,
                   m->cfg.vocab_size, m->cfg.selector_rank);
    if (e == OC_OK)
        e = load_w(&f, "candidate_selector.successor_codebook",
                   &m->selector.successor_codebook,
                   m->cfg.vocab_size, m->cfg.selector_rank);
    if (e == OC_OK)
        e = load_w(&f, "candidate_selector.hidden_projection.weight",
                   &m->selector.hidden_projection,
                   m->cfg.selector_rank, H);

    if (e == OC_OK) {
        m->layers = calloc(n_layers, sizeof(OcDFlash2Layer));
        if (!m->layers) e = OC_ERR_OOM;
    }
    if (e == OC_OK) m->n_layers = n_layers;

    const size_t kvh = m->cfg.num_key_value_heads;
    const size_t hd = m->cfg.head_dim;
    const size_t q_rows = m->cfg.num_attention_heads * hd;
    const size_t kv_rows = kvh * hd;
    const size_t inter = m->cfg.intermediate_size;
    const size_t ksz = m->cfg.conv_kernel_size;
    const size_t groups = H / m->cfg.conv_group_size;

    for (size_t li = 0; e == OC_OK && li < n_layers; li++) {
        char name[160];
        OcDFlash2Layer *L = &m->layers[li];

        snprintf(name, sizeof(name), "layers.%zu.input_layernorm.weight", li);
        e = load_vec(&f, name, &L->input_layernorm, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.post_attention_layernorm.weight", li);
        e = load_vec(&f, name, &L->post_attention_layernorm, H);
        if (e != OC_OK) break;

        snprintf(name, sizeof(name), "layers.%zu.self_attn.q_proj.weight", li);
        e = load_w(&f, name, &L->attn.q_proj, q_rows, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.self_attn.k_proj.weight", li);
        e = load_w(&f, name, &L->attn.k_proj, kv_rows, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.self_attn.v_proj.weight", li);
        e = load_w(&f, name, &L->attn.v_proj, kv_rows, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.self_attn.o_proj.weight", li);
        e = load_w(&f, name, &L->attn.o_proj, H, q_rows);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.self_attn.q_norm.weight", li);
        e = load_vec(&f, name, &L->attn.q_norm, hd);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.self_attn.k_norm.weight", li);
        e = load_vec(&f, name, &L->attn.k_norm, hd);
        if (e != OC_OK) break;

        snprintf(name, sizeof(name), "layers.%zu.attention_conv.base_kernel", li);
        e = load_w(&f, name, &L->attn_conv.base_kernel, 2 * ksz, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name),
                 "layers.%zu.attention_conv.kernel_projection.weight", li);
        e = load_w(&f, name, &L->attn_conv.kernel_proj, 2 * ksz * groups, H);
        if (e != OC_OK) break;

        snprintf(name, sizeof(name), "layers.%zu.mlp.gate_proj.weight", li);
        e = load_w(&f, name, &L->mlp_gate, inter, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.mlp.up_proj.weight", li);
        e = load_w(&f, name, &L->mlp_up, inter, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.mlp.down_proj.weight", li);
        e = load_w(&f, name, &L->mlp_down, H, inter);
        if (e != OC_OK) break;

        snprintf(name, sizeof(name), "layers.%zu.mlp_conv.base_kernel", li);
        e = load_w(&f, name, &L->mlp_conv.base_kernel, 2 * ksz, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name),
                 "layers.%zu.mlp_conv.kernel_projection.weight", li);
        e = load_w(&f, name, &L->mlp_conv.kernel_proj, 2 * ksz * groups, H);
        if (e != OC_OK) break;
    }

    if (e == OC_OK) {
        m->kv = calloc(n_layers, sizeof(OcDFlash2KvRing));
        if (!m->kv) e = OC_ERR_OOM;
    }
    for (size_t li = 0; e == OC_OK && li < n_layers; li++) {
        size_t cap = m->cfg.sliding_window + m->cfg.block_size;
        e = oc_dflash2_kvring_init(&m->kv[li], cap, kvh, hd);
    }
    if (e == OC_OK) {
        m->kv_capacity = m->cfg.sliding_window + m->cfg.block_size;
        m->target_ctx = df2_alloc_huge(m->kv_capacity * H * sizeof(float));
        if (!m->target_ctx) e = OC_ERR_OOM;
    }

    oc_safetensors_close(&f);
    if (e != OC_OK) {
        oc_dflash2_model_free(m);
        return e;
    }
    /* RoPE frequency table: same powf the reference evaluates per call,
     * computed once. Bit-identical values, no per-step transcendentals. */
    m->rope_freq_n = m->cfg.head_dim / 2;
    m->rope_freq = malloc(m->rope_freq_n * sizeof(float));
    if (!m->rope_freq) {
        oc_dflash2_model_free(m);
        return OC_ERR_OOM;
    }
    for (size_t d = 0; d < m->rope_freq_n; d++)
        m->rope_freq[d] = powf(m->cfg.rope_theta,
                               -((float)d / (float)m->rope_freq_n));
    m->loaded = true;
    m->rng_state = 0x9E3779B9u; /* nonzero default seed */
    return OC_OK;
}

/* Free a weight's backing store: hugepage-backed when alloc_bytes is set
 * (see df2_alloc_huge), plain malloc otherwise. */
static void df2_free_w(OcDFlash2Weight *w)
{
    if (w->alloc_bytes) {
        df2_free_huge(w->data ? (void *)w->data : (void *)w->bf16,
                      w->alloc_bytes);
    } else {
        free(w->data);
        free(w->bf16);
    }
    w->data = NULL;
    w->bf16 = NULL;
    w->alloc_bytes = 0;
}

void oc_dflash2_model_free(OcDFlash2Model *m)
{
    if (!m) return;
    df2_free_w(&m->fc);
    free(m->hidden_norm);
    free(m->norm);
    df2_free_w(&m->selector.predecessor_codebook);
    df2_free_w(&m->selector.successor_codebook);
    df2_free_w(&m->selector.hidden_projection);
    for (size_t li = 0; li < m->n_layers; li++) {
        OcDFlash2Layer *L = &m->layers[li];
        free(L->input_layernorm);
        free(L->post_attention_layernorm);
        df2_free_w(&L->attn.q_proj);
        df2_free_w(&L->attn.k_proj);
        df2_free_w(&L->attn.v_proj);
        df2_free_w(&L->attn.o_proj);
        free(L->attn.q_norm);
        free(L->attn.k_norm);
        df2_free_w(&L->attn_conv.base_kernel);
        df2_free_w(&L->attn_conv.kernel_proj);
        df2_free_w(&L->mlp_gate);
        df2_free_w(&L->mlp_up);
        df2_free_w(&L->mlp_down);
        df2_free_w(&L->mlp_conv.base_kernel);
        df2_free_w(&L->mlp_conv.kernel_proj);
    }
    free(m->layers);
    if (m->kv) {
        for (size_t li = 0; li < m->n_layers; li++)
            oc_dflash2_kvring_free(&m->kv[li]);
        free(m->kv);
    }
    free(m->last_hidden);
    df2_free_huge(m->target_ctx,
                  m->kv_capacity * m->cfg.hidden_size * sizeof(float));
    free(m->rope_freq);
    memset(m, 0, sizeof(*m));
}

void oc_dflash2_reset(OcDFlash2Model *m)
{
    if (!m) return;
    for (size_t li = 0; li < m->n_layers; li++)
        oc_dflash2_kvring_clear(&m->kv[li]);
    m->next_noise_pos = 0;
    m->target_ctx_len = 0;
}
const float *oc_dflash2_last_hidden(const OcDFlash2Model *m, size_t *rows)
{
    if (!m) {
        if (rows) *rows = 0;
        return NULL;
    }
    if (rows) *rows = m->last_hidden_len;
    return m->last_hidden;
}

/* ─── Context fusion ────────────────────────────────────────────────── */

OcError oc_dflash2_set_context(OcDFlash2Model *m,
                               const float *target_context,
                               size_t n_ctx_rows)
{
    if (!m || !m->loaded || !target_context || n_ctx_rows == 0)
        return OC_ERR_INVALID_ARG;
    /* The reference passes whatever context the target produced (prefill
     * rows at step 0, then the per-verify rows); only the KV window bounds
     * the usable length. The block forward appends n_ctx + block rows to
     * each ring in one timeline, so the context must leave room for the
     * block: n_ctx + block_size <= capacity. Checked by subtraction —
     * the addition form would wrap for adversarial n_ctx_rows. */
    if (m->cfg.block_size > m->kv_capacity ||
        n_ctx_rows > m->kv_capacity - m->cfg.block_size)
        return OC_ERR_INVALID_ARG;
    const size_t H = m->cfg.hidden_size;

    /* ctx = hidden_norm(fc @ concat(target hiddens)), per row. */
    gemm(&(m->fc), target_context, n_ctx_rows, m->target_ctx);
    for (size_t r = 0; r < n_ctx_rows; r++)
        rms_norm_row(m->target_ctx + r * H, m->hidden_norm,
                     m->target_ctx + r * H, H, m->cfg.rms_norm_eps);
    m->target_ctx_len = n_ctx_rows;
    /* Ctx rows occupy positions [next_noise_pos - n_ctx, next_noise_pos)
     * in the next propose call (reference:
     * position_ids[:, start - produced : start + verify]); they do not
     * advance the noise start position. */
    return OC_OK;
}

/* ─── Propose (one block forward + selector) ─────────────────────────── */

/* Parallel top-k scan over vocab rows, batched over all draft rows: each
 * vocab row is generated/loaded once and dotted against every draft row. */
typedef struct OcDFlash2TopKCtx {
    const OcDFlash2Weight *lm;
    const float *x;          /* draft hidden rows [n_draft, H] */
    size_t n_draft;
    size_t vocab;
    size_t top_k;
    uint32_t *tidx;          /* [n_threads, n_draft, top_k] per-thread best ids */
    float *tval;             /* [n_threads, n_draft, top_k] per-thread best vals */
    size_t H;
    _Atomic int *err;       /* shared worker failure (0 = ok; 1 = OOM) */
} OcDFlash2TopKCtx;

/* K-best insert (descending). Returns early when below current worst. */
static inline void df2_topk_insert(float *pval, uint32_t *pidx,
                                   size_t top_k, float v_, uint32_t id_)
{
    if (v_ > pval[top_k - 1]) {
        size_t k = top_k - 1;
        pval[k] = v_;
        pidx[k] = id_;
        while (k > 0 && pval[k] > pval[k - 1]) {
            float tv = pval[k]; pval[k] = pval[k - 1]; pval[k - 1] = tv;
            uint32_t ti = pidx[k]; pidx[k] = pidx[k - 1]; pidx[k - 1] = ti;
            k--;
        }
    }
}

static void dflash2_topk_par_fn(size_t begin, size_t end, size_t tid,
                                void *ud)
{
    OcDFlash2TopKCtx *c = (OcDFlash2TopKCtx *)ud;
    const size_t top_k = c->top_k;
    const size_t H = c->H;
    const size_t n_draft = c->n_draft;
    uint32_t *idx = c->tidx + tid * n_draft * top_k;
    float *val = c->tval + tid * n_draft * top_k;
    float *gen_buf = oc_parallel_scratch(tid, H * sizeof(float));
    const bool generated = !c->lm->data && !c->lm->bf16;
    /* A failed scratch alloc must not silently drop this worker's vocab
     * slice (the top-k would come back incomplete yet OC_OK); mark the
     * shared error so propose can fail. */
    if (generated && !gen_buf) {
        atomic_store_explicit(c->err, 1, memory_order_relaxed);
        return;
    }

    for (size_t v = begin; v < end; v++) {
        const float *wr;
        if (c->lm->bf16) {
            /* Fused batched BF16 dot: W streamed once, all draft rows
             * accumulated in the same pass. */
            float scores[OC_DFLASH2_MAX_BLOCK];
            if (oc_simd_try_dflash2_dot_bf16_batch(
                    c->lm->bf16 + v * H, c->x, H, n_draft, scores)) {
                for (size_t p = 0; p < n_draft; p++)
                    df2_topk_insert(val + p * top_k, idx + p * top_k,
                                     top_k, scores[p], (uint32_t)v);
                continue;
            }
            for (size_t p = 0; p < n_draft; p++) {
                float v_ = df2_dot_bf16(c->lm->bf16 + v * H,
                                        c->x + p * H, H);
                df2_topk_insert(val + p * top_k, idx + p * top_k,
                                top_k, v_, (uint32_t)v);
            }
            continue;
        }
        if (c->lm->data) {
            for (size_t p = 0; p < n_draft; p++) {
                float v_ = df2_dot(c->lm->data + v * H, c->x + p * H, H);
                df2_topk_insert(val + p * top_k, idx + p * top_k,
                                top_k, v_, (uint32_t)v);
            }
            continue;
        }
        c->lm->generate(v, H, gen_buf, c->lm->gen_user);
        wr = gen_buf;
        for (size_t p = 0; p < n_draft; p++) {
            float v_ = df2_dot(wr, c->x + p * H, H);
            df2_topk_insert(val + p * top_k, idx + p * top_k,
                            top_k, v_, (uint32_t)v);
        }
    }
}

typedef struct OcDFlash2ProposeCtx {
    /* Inputs. */
    OcDFlash2Model *m;
    const float *noise_emb;    /* [block, H] */
    const OcDFlash2Weight *lm_head;
    float temperature;
    /* Scratch shared across layer loop. */
    float *hidden;             /* [block, H] current stream */
    float *normed;             /* [block, H] */
    float *conv_dyn;           /* [block, 2*ksz*groups] per-row kernel proj */
    float *conv_dyn_post;      /* saved dynamic kernels (tap 1) [block, ksz*groups] */
    float *conv_scratch;       /* [block, H] */
    float *conv_out;           /* [block, H] */
    float *q;                  /* [block, n_heads*hd] */
    float *k_all;              /* [n_ctx+block, n_kv*hd] */
    float *v_all;              /* [n_ctx+block, n_kv*hd] */
    float *attn_out;           /* [block, n_heads*hd] */
    float *mlp_gu;              /* [block, inter] gate*up */
    float *mlp_out;            /* [block, H] */
    float *logits;             /* [block-1, vocab_scratch] */
} OcDFlash2ProposeCtx;

/*
 * Bidirectional sliding-window attention for the block queries against the
 * KV ring. q: [q_len, n_heads*hd] (normed + RoPE'd); out:
 * [q_len, n_heads*hd]. Mask: |qpos - kpos| < window. Softmax is computed
 * per (query row, query head) with the online (single-pass) algorithm:
 * each streamed key updates running max, denominator, and the weighted V
 * accumulator, rescaling on new maxima. The reference computes the same
 * softmax with a two-pass max+sum; mathematically identical (the validators
 * pin this within BF16 tolerance).
 *
 * Jobs (q_len * n_heads of them) are fully independent — each owns one
 * output row-head — so the loop nest is split across the thread pool
 * without changing any per-job key order (bit-identical to the serial
 * evaluation; single-thread `oc_parallel_for` falls back inline).
 */
typedef struct OcDFlash2AttnRingCtx {
    const OcDFlash2KvRing *ring;
    const float *q;
    int64_t pos_q0;
    size_t q_len;
    size_t n_q_heads;
    size_t head_dim;
    float scale;
    size_t window;
    float *out;
} OcDFlash2AttnRingCtx;

static void attn_ring_job(size_t job, OcDFlash2AttnRingCtx *c)
{
    const size_t n_q_heads = c->n_q_heads;
    const size_t head_dim = c->head_dim;
    const size_t qi = job / n_q_heads;
    const size_t h = job % n_q_heads;
    const OcDFlash2KvRing *ring = c->ring;
    const size_t n_kv = ring->n_kv_heads;
    const size_t gq = n_q_heads / n_kv;
    const size_t vec = n_kv * head_dim;
    const float *qh = c->q + (qi * n_q_heads + h) * head_dim;
    float *oh = c->out + (qi * n_q_heads + h) * head_dim;
    const int64_t qpos = c->pos_q0 + (int64_t)qi;
    const size_t hg = h / gq;             /* kv group */

    float m = -INFINITY;   /* running max of scaled scores */
    float denom = 0.0f;     /* running softmax denominator */
    /* Live entries are write indices [total - len, total), each at slot
     * (w % capacity). Iterating physical slots [0, len) happens to cover
     * the same SET (the live span is contiguous mod capacity), but in the
     * wrong ORDER once the ring has wrapped — and the online softmax is
     * order-sensitive at the ulp level, so the accumulation order must
     * stay chronological to match the reference. */
    const size_t w0 = ring->total - ring->len;
    for (size_t s = 0; s < ring->len; s++) {
        const size_t slot = (w0 + s) % ring->capacity;
        int64_t dq = qpos - ring->pos[slot];
        if (dq >= (int64_t)c->window || -dq >= (int64_t)c->window) continue;
        const float *kh = ring->k + slot * vec + hg * head_dim;
        const float *vh = ring->v + slot * vec + hg * head_dim;
        float sc = df2_dot(qh, kh, head_dim) * c->scale;
        if (sc > m) {
            /* New max: rescale the accumulator + denominator. */
            float r = (m == -INFINITY) ? 0.0f : expf(m - sc);
            for (size_t d = 0; d < head_dim; d++) oh[d] *= r;
            denom *= r;
            m = sc;
        }
        float w = expf(sc - m);
        denom += w;
        for (size_t d = 0; d < head_dim; d++) oh[d] += w * vh[d];
    }
    if (denom > 0.0f) {
        float inv = 1.0f / denom;
        for (size_t d = 0; d < head_dim; d++) oh[d] *= inv;
    }
}

static void attn_ring_par_fn(size_t begin, size_t end, size_t tid, void *ud)
{
    (void)tid;
    OcDFlash2AttnRingCtx *c = (OcDFlash2AttnRingCtx *)ud;
    for (size_t job = begin; job < end; job++) attn_ring_job(job, c);
}

static void attn_ring(const OcDFlash2KvRing *ring, const float *q,
                      int64_t pos_q0, size_t q_len, size_t n_q_heads,
                      size_t head_dim, float scale, size_t window,
                      float *out)
{
    memset(out, 0, q_len * n_q_heads * head_dim * sizeof(float));
    OcDFlash2AttnRingCtx c = { ring, q, pos_q0, q_len, n_q_heads,
                               head_dim, scale, window, out };
    oc_parallel_for(q_len * n_q_heads, attn_ring_par_fn, &c);
}

/* Scratch bundle shared by the block forward (propose + debug paths). */
typedef struct DFlash2BlockScratch {
    float *hidden, *normed, *conv_dyn, *conv_dyn_post, *conv_scratch;
    float *conv_out, *q, *k_all, *v_all, *k_noise, *v_noise;
    float *attn_out, *attn_proj, *mlp_gu, *mlp_up_out, *mlp_out, *x;
} DFlash2BlockScratch;

static void dflash2_block_scratch_free(DFlash2BlockScratch *s)
{
    free(s->hidden); free(s->normed); free(s->conv_dyn);
    free(s->conv_dyn_post); free(s->conv_scratch); free(s->conv_out);
    free(s->q); free(s->k_all); free(s->v_all); free(s->k_noise);
    free(s->v_noise); free(s->attn_out); free(s->attn_proj);
    free(s->mlp_gu); free(s->mlp_up_out); free(s->mlp_out); free(s->x);
    memset(s, 0, sizeof(*s));
}

static int dflash2_block_scratch_alloc(DFlash2BlockScratch *s,
                                       size_t block, size_t H,
                                       size_t n_heads, size_t n_kv,
                                       size_t hd, size_t ksz, size_t groups,
                                       size_t inter, size_t n_ctx)
{
    memset(s, 0, sizeof(*s));
    s->hidden = malloc(block * H * sizeof(float));
    s->normed = malloc(block * H * sizeof(float));
    s->conv_dyn = malloc(block * 2 * ksz * groups * sizeof(float));
    s->conv_dyn_post = malloc(block * ksz * groups * sizeof(float));
    s->conv_scratch = malloc(block * H * sizeof(float));
    s->conv_out = malloc(block * H * sizeof(float));
    s->q = malloc(block * n_heads * hd * sizeof(float));
    s->k_all = malloc((n_ctx + block) * n_kv * hd * sizeof(float));
    s->v_all = malloc((n_ctx + block) * n_kv * hd * sizeof(float));
    s->k_noise = malloc(block * n_kv * hd * sizeof(float));
    s->v_noise = malloc(block * n_kv * hd * sizeof(float));
    s->attn_out = malloc(block * n_heads * hd * sizeof(float));
    s->attn_proj = malloc(block * H * sizeof(float));
    s->mlp_gu = malloc(block * inter * sizeof(float));
    s->mlp_up_out = malloc(block * inter * sizeof(float));
    s->mlp_out = malloc(block * H * sizeof(float));
    s->x = malloc(block * H * sizeof(float));
    if (!s->hidden || !s->normed || !s->conv_dyn || !s->conv_dyn_post ||
        !s->conv_scratch || !s->conv_out || !s->q || !s->k_all || !s->v_all ||
        !s->k_noise || !s->v_noise || !s->attn_out || !s->attn_proj ||
        !s->mlp_gu || !s->mlp_up_out || !s->mlp_out || !s->x) {
        dflash2_block_scratch_free(s);
        return 0;
    }
    return 1;
}

/* Shared block forward: the layered attention + conv + MLP pass from
 * noise embeddings to final-normed hidden rows, including the per-layer
 * KV ring append. Used by BOTH oc_dflash2_propose and
 * oc_dflash2_forward_debug so the two entry points cannot drift (the
 * review flagged eps handling and OOM behavior diverging between the
 * previously duplicated loops).
 *
 * eps arrives as float — every RMSNorm here (layer, per-head q/k, final)
 * must see the configured fractional epsilon, never a truncated one.
 *
 * out_normed receives the final-normed [block, H] rows (may alias
 * s->normed). */
enum { PT_ACONV, PT_QKV, PT_NORM, PT_ATTN, PT_OCONV, PT_MLP, PT_N };
static const char *df2_pt_names[PT_N] = {
    "attn_convproj", "qkv_gemm", "qk_norm_rope", "attn",
    "o_proj_conv", "mlp"
};
static double df2_pt[PT_N];
static int df2_pt_on;
static double df2_pt_t0;

static void df2_pt_mark(int idx)
{
    if (!df2_pt_on) return;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double now = ts.tv_sec + ts.tv_nsec * 1e-9;
    df2_pt[idx] += now - df2_pt_t0;
    df2_pt_t0 = now;
}

static OcError dflash2_forward_block(OcDFlash2Model *m,
                                     const float *noise_emb, size_t block,
                                     DFlash2BlockScratch *s, float *out_normed)
{
    const size_t H = m->cfg.hidden_size;
    const size_t hd = m->cfg.head_dim;
    const size_t n_heads = m->cfg.num_attention_heads;
    const size_t n_kv = m->cfg.num_key_value_heads;
    const size_t ksz = m->cfg.conv_kernel_size;
    const size_t gs = m->cfg.conv_group_size;
    const size_t groups = H / gs;
    const size_t inter = m->cfg.intermediate_size;
    const float eps = m->cfg.rms_norm_eps;
    const size_t n_ctx = m->target_ctx_len;
    const int64_t start = m->next_noise_pos;
    const int64_t ctx_pos0 = start - (int64_t)n_ctx;

    float *hidden = s->hidden, *normed = s->normed, *conv_dyn = s->conv_dyn;
    float *conv_dyn_post = s->conv_dyn_post, *conv_scratch = s->conv_scratch;
    float *conv_out = s->conv_out, *q = s->q, *k_all = s->k_all;
    float *v_all = s->v_all, *k_noise = s->k_noise, *v_noise = s->v_noise;
    float *attn_out = s->attn_out, *attn_proj = s->attn_proj;
    float *mlp_gu = s->mlp_gu, *mlp_up_out = s->mlp_up_out;
    float *mlp_out = s->mlp_out, *x = s->x;
    OcError ae;   /* ring-append result per layer */

    memcpy(hidden, noise_emb, block * H * sizeof(float));

    for (size_t li = 0; li < m->n_layers; li++) {
        OcDFlash2Layer *L = &m->layers[li];
        OcDFlash2KvRing *ring = &m->kv[li];

        /* ── Attention block ──────────────────────────────────── */
        memcpy(x, hidden, block * H * sizeof(float));
        for (size_t i = 0; i < block; i++)
            rms_norm_row(x + i * H, L->input_layernorm, normed + i * H, H, eps);

        gemm(&L->attn_conv.kernel_proj, normed, block, conv_dyn);
        df2_pt_mark(PT_ACONV);
        /* dyn view: [block, 2][ksz][groups]; tap-0 kernels at
         * conv_dyn[i, 0*ksz*groups + t*groups + g], tap-1 at +ksz*groups. */
        for (size_t i = 0; i < block; i++) {
            memcpy(conv_scratch + i * ksz * groups,
                   conv_dyn + i * 2 * ksz * groups,
                   ksz * groups * sizeof(float));
            memcpy(conv_dyn_post + i * ksz * groups,
                   conv_dyn + i * 2 * ksz * groups + ksz * groups,
                   ksz * groups * sizeof(float));
        }
        oc_dflash2_grouped_conv(normed, conv_scratch,
                                L->attn_conv.base_kernel.data,
                                block, H, ksz, gs, conv_out);

        /* Q from conv'd hidden; K/V from BOTH context and conv'd hidden.
         * One dispatch per x-row group: {q, k_noise, v_noise} over conv'd
         * rows, {k_ctx, v_ctx} over the fused target context. */
        {
            OcDFlash2MultiGemmCtx mg = { 0 };
            mg.ws[0] = &L->attn.q_proj;   mg.xs[0] = conv_out;  mg.outs[0] = q;
            mg.ws[1] = &L->attn.k_proj;   mg.xs[1] = conv_out;  mg.outs[1] = k_noise;
            mg.ws[2] = &L->attn.v_proj;   mg.xs[2] = conv_out;  mg.outs[2] = v_noise;
            mg.n_w = 3;
            mg.n = block;
            gemm_multi(&mg);
            if (n_ctx >= 1) {
                OcDFlash2MultiGemmCtx mc = { 0 };
                mc.ws[0] = &L->attn.k_proj; mc.xs[0] = m->target_ctx; mc.outs[0] = k_all;
                mc.ws[1] = &L->attn.v_proj; mc.xs[1] = m->target_ctx; mc.outs[1] = v_all;
                mc.n_w = 2;
                mc.n = n_ctx;
                gemm_multi(&mc);
            }
        }
        df2_pt_mark(PT_QKV);

        memcpy(k_all + n_ctx * n_kv * hd, k_noise, block * n_kv * hd * sizeof(float));
        memcpy(v_all + n_ctx * n_kv * hd, v_noise, block * n_kv * hd * sizeof(float));
        const size_t n_k = n_ctx + block;

        /* Per-head RMSNorm: q [block, n_heads, hd]; k_all [n_k, n_kv, hd]. */
        for (size_t i = 0; i < block; i++)
            for (size_t h = 0; h < n_heads; h++) {
                float *qh = q + (i * n_heads + h) * hd;
                float ss = 0.0f;
                for (size_t d = 0; d < hd; d++) ss += qh[d] * qh[d];
                float r = 1.0f / sqrtf(ss / (float)hd + eps);
                for (size_t d = 0; d < hd; d++) qh[d] = qh[d] * r * L->attn.q_norm[d];
            }
        for (size_t i = 0; i < n_k; i++)
            for (size_t h = 0; h < n_kv; h++) {
                float *kh = k_all + (i * n_kv + h) * hd;
                float ss = 0.0f;
                for (size_t d = 0; d < hd; d++) ss += kh[d] * kh[d];
                float r = 1.0f / sqrtf(ss / (float)hd + eps);
                for (size_t d = 0; d < hd; d++) kh[d] = kh[d] * r * L->attn.k_norm[d];
            }

        /* RoPE: k over [ctx_pos0, start + block), q over
         * [start, start + block). theta 10k, head_dim 128. */
        {
            const size_t half = hd / 2;
            const float *freq = m->rope_freq;
            for (size_t i = 0; i < n_k; i++) {
                int64_t pos = ctx_pos0 + (int64_t)i;
                for (size_t h = 0; h < n_kv; h++) {
                    float *kh = k_all + (i * n_kv + h) * hd;
                    for (size_t d = 0; d < half; d++) {
                        float ang = (float)pos * freq[d];
                        float c = cosf(ang), s = sinf(ang);
                        float k0 = kh[d], k1 = kh[d + half];
                        kh[d] = k0 * c - k1 * s;
                        kh[d + half] = k0 * s + k1 * c;
                    }
                }
            }
            for (size_t i = 0; i < block; i++) {
                int64_t pos = start + (int64_t)i;
                for (size_t h = 0; h < n_heads; h++) {
                    float *qh = q + (i * n_heads + h) * hd;
                    for (size_t d = 0; d < half; d++) {
                        float ang = (float)pos * freq[d];
                        float c = cosf(ang), s = sinf(ang);
                        float q0 = qh[d], q1 = qh[d + half];
                        qh[d] = q0 * c - q1 * s;
                        qh[d + half] = q0 * s + q1 * c;
                    }
                }
            }
        }

        df2_pt_mark(PT_NORM);

        /* Append K/V (already RoPE'd) to the ring, then attend. A failed
         * append (n_k > capacity) must abort, not attend over stale
         * entries and return tokens from wrong attention state. */
        ae = oc_dflash2_kvring_append(ring, k_all, v_all, ctx_pos0, n_k);
        if (ae != OC_OK) return ae;

        attn_ring(ring, q, start, block, n_heads, hd,
                  1.0f / sqrtf((float)hd), m->cfg.sliding_window, attn_out);
        df2_pt_mark(PT_ATTN);

        /* o_proj + conv finish + residual. */
        gemm(&L->attn.o_proj, attn_out, block, attn_proj);
        oc_dflash2_grouped_conv(attn_proj, conv_dyn_post,
                                L->attn_conv.base_kernel.data + ksz * H,
                                block, H, ksz, gs, conv_out);
        for (size_t i = 0; i < block * H; i++) hidden[i] += conv_out[i];
        df2_pt_mark(PT_OCONV);

        /* ── MLP block ────────────────────────────────────────── */
        memcpy(x, hidden, block * H * sizeof(float));
        for (size_t i = 0; i < block; i++)
            rms_norm_row(x + i * H, L->post_attention_layernorm,
                         normed + i * H, H, eps);

        gemm(&L->mlp_conv.kernel_proj, normed, block, conv_dyn);
        for (size_t i = 0; i < block; i++) {
            memcpy(conv_scratch + i * ksz * groups,
                   conv_dyn + i * 2 * ksz * groups,
                   ksz * groups * sizeof(float));
            memcpy(conv_dyn_post + i * ksz * groups,
                   conv_dyn + i * 2 * ksz * groups + ksz * groups,
                   ksz * groups * sizeof(float));
        }
        oc_dflash2_grouped_conv(normed, conv_scratch,
                                L->mlp_conv.base_kernel.data,
                                block, H, ksz, gs, conv_out);

        {
            OcDFlash2MultiGemmCtx mg = { 0 };
            mg.ws[0] = &L->mlp_gate; mg.xs[0] = conv_out; mg.outs[0] = mlp_gu;
            mg.ws[1] = &L->mlp_up;   mg.xs[1] = conv_out; mg.outs[1] = mlp_up_out;
            mg.n_w = 2;
            mg.n = block;
            gemm_multi(&mg);
        }
        for (size_t i = 0; i < block * inter; i++)
            mlp_gu[i] = silu(mlp_gu[i]) * mlp_up_out[i];
        gemm(&L->mlp_down, mlp_gu, block, mlp_out);
        df2_pt_mark(PT_MLP);
        oc_dflash2_grouped_conv(mlp_out, conv_dyn_post,
                                L->mlp_conv.base_kernel.data + ksz * H,
                                block, H, ksz, gs, conv_out);
        for (size_t i = 0; i < block * H; i++) hidden[i] += conv_out[i];
    }

    /* Final norm on all block rows. */
    for (size_t i = 0; i < block; i++)
        rms_norm_row(hidden + i * H, m->norm, out_normed + i * H, H, eps);
    return OC_OK;
}

OcError oc_dflash2_propose(OcDFlash2Model *m,
                           const uint32_t *anchor_ids, size_t n_anchor,
                           const float *noise_emb, size_t block,
                           const uint32_t *block_ids,
                           const OcDFlash2Weight *lm_head,
                           float temperature,
                           uint32_t *out_tokens,
                           uint32_t *out_top_k,
                           float *out_top_k_probs)
{
    if (!m || !m->loaded) return OC_ERR_INVALID_ARG;
    if (!anchor_ids || n_anchor == 0) return OC_ERR_INVALID_ARG;
    if (anchor_ids[n_anchor - 1] >= m->cfg.vocab_size)
        return OC_ERR_INVALID_ARG; /* codebook row would be OOB */
    if (!noise_emb || block == 0 || block > OC_DFLASH2_MAX_BLOCK)
        return OC_ERR_INVALID_ARG;
    if (!lm_head || (!lm_head->data && !lm_head->bf16 &&
                      !lm_head->generate))
        return OC_ERR_INVALID_ARG;
    /* The lm_head must actually be [vocab, hidden]: a mismatched cols
     * would be read with inconsistent strides (wrong logits), and rows
     * below selector_top_k (or zero) cannot fill the candidate lattice. */
    if (lm_head->cols != m->cfg.hidden_size || lm_head->rows == 0 ||
        lm_head->rows < m->cfg.selector_top_k)
        return OC_ERR_INVALID_ARG;
    if (!out_tokens) return OC_ERR_INVALID_ARG;

    const size_t H = m->cfg.hidden_size;
    const size_t top_k = m->cfg.selector_top_k;
    const size_t rank = m->cfg.selector_rank;
    const size_t vocab = lm_head->rows;

    /* Absolute positions: context rows [start - n_ctx, start), noise rows
     * [start, start + block) where start = next_noise_pos. */
    const int64_t start = m->next_noise_pos;

    /* Optional phase timing (DF2_PHASE_TIMING=1): wall time per segment of
     * the layer loop, printed to stderr at propose end. Off by default. */
    df2_pt_on = getenv("DF2_PHASE_TIMING") != NULL;
    for (int k = 0; k < PT_N; k++) df2_pt[k] = 0.0;
    if (df2_pt_on) {
        struct timespec df2_ts_;
        clock_gettime(CLOCK_MONOTONIC, &df2_ts_);
        df2_pt_t0 = df2_ts_.tv_sec + df2_ts_.tv_nsec * 1e-9;
    }

    /* Shared block forward (same helper the debug path runs, so the two
     * entry points cannot drift on eps handling or OOM behavior). */
    DFlash2BlockScratch s;
    if (!dflash2_block_scratch_alloc(&s, block, H,
                                     m->cfg.num_attention_heads,
                                     m->cfg.num_key_value_heads,
                                     m->cfg.head_dim, m->cfg.conv_kernel_size,
                                     H / m->cfg.conv_group_size,
                                     m->cfg.intermediate_size,
                                     m->target_ctx_len)) {
        return OC_ERR_OOM;
    }
    OcError e = dflash2_forward_block(m, noise_emb, block, &s, s.normed);
    if (df2_pt_on) {
        fprintf(stderr, "df2 phases ms/step:");
        for (int k = 0; k < PT_N; k++)
            fprintf(stderr, " %s=%.2f", df2_pt_names[k], df2_pt[k] * 1e3);
        fprintf(stderr, "\n");
    }
    if (e != OC_OK) {
        dflash2_block_scratch_free(&s);
        return e;
    }

    /* Retain the normed rows for oc_dflash2_last_hidden callers. */
    if (!m->last_hidden || m->last_hidden_len != block) {
        free(m->last_hidden);
        m->last_hidden = malloc(block * H * sizeof(float));
        if (!m->last_hidden) {
            m->last_hidden_len = 0;
            dflash2_block_scratch_free(&s);
            goto fail_after_fwd;
        }
        m->last_hidden_len = block;
    }
    memcpy(m->last_hidden, s.normed, block * H * sizeof(float));

    float *normed = s.normed;

    /* Draft hidden = rows 1..block-1 (skip the anchor row). */
    const size_t n_draft = block - 1;
    const float *draft_hidden = normed + H;

    /* ── Selector ─────────────────────────────────────────────── */
    /* Logits via the target lm_head for each draft row (parallel over
     * rows for large lm_heads). */
    float *unary = malloc(n_draft * top_k * sizeof(float));
    uint32_t *cand = malloc(n_draft * top_k * sizeof(uint32_t));
    float *proj_h = malloc(n_draft * rank * sizeof(float));
    float *scores = malloc(top_k * sizeof(float));
    float *path_probs = out_top_k_probs;
    if (!unary || !cand || !proj_h || !scores) {
        free(unary); free(cand); free(proj_h); free(scores);
        dflash2_block_scratch_free(&s);
        goto fail_after_fwd;
    }

    /* logits: one pass over the vocab, dotting each row against all n_draft
     * hidden rows at once (generated rows are materialized once and reused;
     * a real lm_head is streamed once from memory). */
    const size_t vocab_eff = vocab < m->cfg.vocab_size ? vocab : m->cfg.vocab_size;
    int logit_err = 0;
    if ((lm_head->data || lm_head->bf16) && vocab <= (size_t)1 << 16) {
        /* Small real lm_head: full logits per draft row, serial top-k.
         * lrow holds every lm_head row (gemv_w writes rows floats); the
         * candidate scan below is bounded by vocab_eff so ids stay valid
         * codebook rows even when the head is larger than cfg.vocab_size. */
        float *lrow = malloc(vocab * sizeof(float));
        if (!lrow) logit_err = 1;
        for (size_t p = 0; p < n_draft && !logit_err; p++) {
            gemv_w(lm_head, draft_hidden + p * H, lrow);
            for (size_t k = 0; k < top_k; k++) {
                size_t best = 0;
                for (size_t v = 1; v < vocab_eff; v++)
                    if (lrow[v] > lrow[best]) best = v;
                cand[p * top_k + k] = (uint32_t)best;
                unary[p * top_k + k] = lrow[best];
                lrow[best] = -INFINITY;
            }
        }
        free(lrow);
    } else {
        /* Large-vocab path: parallel streaming top-k over vocab slices.
         * Each worker generates/loads each vocab row once and dots it
         * against every draft row, keeping a per-(thread, draft) k-best. */
        const size_t max_t = oc_parallel_n_threads();
        uint32_t *tidx = calloc(max_t * n_draft * top_k, sizeof(uint32_t));
        float *tval = malloc(max_t * n_draft * top_k * sizeof(float));
        if (!tidx || !tval) {
            free(tidx); free(tval);
            logit_err = 1;
        } else {
            for (size_t i = 0; i < max_t * n_draft * top_k; i++)
                tval[i] = -INFINITY;

            OcDFlash2TopKCtx c;
            _Atomic int wk_err = 0;
            c.lm = lm_head;
            c.x = draft_hidden;
            c.n_draft = n_draft;
            c.vocab = vocab_eff;
            c.top_k = top_k;
            c.tidx = tidx;
            c.tval = tval;
            c.H = H;
            c.err = &wk_err;
            oc_parallel_for(vocab_eff, dflash2_topk_par_fn, &c);
            if (atomic_load_explicit(&wk_err, memory_order_relaxed) != 0) {
                free(tidx);
                free(tval);
                free(unary); free(cand); free(proj_h); free(scores);
                dflash2_block_scratch_free(&s);
                goto fail_after_fwd;
            }

            /* Merge per-thread k-best lists into the final top-k. */
            for (size_t p = 0; p < n_draft; p++) {
                size_t best_idx[top_k];
                float best_val[top_k];
                for (size_t k = 0; k < top_k; k++) {
                    best_val[k] = -INFINITY;
                    best_idx[k] = 0;
                }
                for (size_t t = 0; t < max_t; t++) {
                    const uint32_t *ti =
                        tidx + (t * n_draft + p) * top_k;
                    const float *tv = tval + (t * n_draft + p) * top_k;
                    for (size_t k = 0; k < top_k; k++) {
                        float v = tv[k];
                        if (v == -INFINITY) continue;
                        if (v > best_val[top_k - 1]) {
                            size_t j = top_k - 1;
                            best_val[j] = v;
                            best_idx[j] = ti[k];
                            while (j > 0 && best_val[j] > best_val[j - 1]) {
                                float tvv = best_val[j];
                                best_val[j] = best_val[j-1];
                                best_val[j-1] = tvv;
                                size_t tii = best_idx[j];
                                best_idx[j] = best_idx[j-1];
                                best_idx[j-1] = tii;
                                j--;
                            }
                        }
                    }
                }
                for (size_t k = 0; k < top_k; k++) {
                    cand[p * top_k + k] = (uint32_t)best_idx[k];
                    unary[p * top_k + k] = best_val[k];
                }
            }
            free(tidx);
            free(tval);
        }
    }
    if (logit_err) {
        free(unary); free(cand); free(proj_h); free(scores);
        dflash2_block_scratch_free(&s);
        goto fail_after_fwd;
    }

    /* hidden_projection for each draft row. */
    gemm(&(m->selector.hidden_projection), draft_hidden, n_draft, proj_h);

    /* Greedy (or temperature) path trace. */
    uint32_t predecessor = anchor_ids[n_anchor - 1];
    float *A_buf = malloc(rank * sizeof(float));
    if (!A_buf) {
        free(unary); free(cand); free(proj_h); free(scores);
        dflash2_block_scratch_free(&s);
        goto fail_after_fwd;
    }
    for (size_t p = 0; p < n_draft; p++) {
        df2_widen_row(&m->selector.predecessor_codebook, predecessor, A_buf);
        const float *A_p = A_buf;
        float *un = unary + p * top_k;
        /* B rows for the candidates of this position. */
        float *Brows = malloc(top_k * rank * sizeof(float));
        if (!Brows) {
            free(A_buf);
            free(unary); free(cand); free(proj_h); free(scores);
            dflash2_block_scratch_free(&s);
            goto fail_after_fwd;
        }
        for (size_t k = 0; k < top_k; k++)
            df2_widen_row(&m->selector.successor_codebook,
                          cand[p * top_k + k], Brows + k * rank);
        oc_dflash2_selector_scores(proj_h + p * rank, A_p, Brows, un,
                                   top_k, scores);

        uint32_t choice = 0;
        if (temperature > 0.0f) {
            float mx = scores[0];
            for (size_t k = 1; k < top_k; k++)
                if (scores[k] > mx) mx = scores[k];
            float denom = 0.0f;
            for (size_t k = 0; k < top_k; k++) {
                scores[k] = expf((scores[k] - mx) / temperature);
                denom += scores[k];
            }
            m->rng_state = m->rng_state * 1664525u + 1013904223u;
            float z = ((float)(m->rng_state >> 8) / 16777216.0f) * denom;
            float run = 0.0f;
            for (size_t k = 0; k < top_k; k++) {
                run += scores[k];
                if (z <= run) { choice = (uint32_t)k; break; }
            }
            if (out_top_k_probs) {
                for (size_t k = 0; k < top_k; k++)
                    path_probs[p * top_k + k] = scores[k] / denom;
            }
        } else {
            for (size_t k = 1; k < top_k; k++)
                if (scores[k] > scores[choice]) choice = (uint32_t)k;
            if (out_top_k_probs) {
                float mx = scores[choice];
                float denom = 0.0f;
                for (size_t k = 0; k < top_k; k++)
                    denom += expf(scores[k] - mx);
                for (size_t k = 0; k < top_k; k++)
                    path_probs[p * top_k + k] =
                        expf(scores[k] - mx) / denom;
            }
        }

        predecessor = cand[p * top_k + choice];
        out_tokens[p] = predecessor;
        if (out_top_k)
            memcpy(out_top_k + p * top_k, cand + p * top_k,
                   top_k * sizeof(uint32_t));
        free(Brows);
    }

    /* Match the reference cache contract: after each draft forward the
     * cache is cropped to `start` entries (dflash_generate's
     * _crop_to(past_key_values_draft, start)). All noise rows — including
     * the anchor row at `start` — are discarded; committed positions
     * re-enter the ring as target-derived ctx rows on the next step's
     * set_context + append. The caller advances next_noise_pos by the
     * number of accepted tokens. */
    for (size_t li = 0; li < m->n_layers; li++)
        oc_dflash2_kvring_trim(&m->kv[li], start);

    free(unary); free(cand); free(proj_h); free(scores); free(A_buf);
    dflash2_block_scratch_free(&s);
    (void)block_ids;
    return OC_OK;

fail_after_fwd:
    /* Selector-stage failure AFTER the block forward already appended
     * this step's speculative rows to the rings: roll them back to the
     * last committed position so the caller's retry / next proposal does
     * not start from speculative cache state. */
    for (size_t li = 0; li < m->n_layers; li++)
        oc_dflash2_kvring_trim(&m->kv[li], start);
    return OC_ERR_OOM;
}

/* ─── Debug / validation entry points ────────────────────────────────── */

/*
 * Validation backbone-only forward: same math as propose's forward stage
 * but with KV ring management bypassed (ring entries are written, and the
 * ring is restored to its prior state after), and hidden rows returned.
 */
OcError oc_dflash2_forward_debug(OcDFlash2Model *m,
                                 const float *noise_emb, size_t block,
                                 float *out_hidden)
{
    if (!m || !m->loaded || !noise_emb || !out_hidden)
        return OC_ERR_INVALID_ARG;
    if (block == 0 || block > OC_DFLASH2_MAX_BLOCK) return OC_ERR_INVALID_ARG;


    /* Save ring state so validation is side-effect free. The debug append
     * writes slots (total + i) % capacity; once the ring has wrapped those
     * slots hold the oldest live entries, so their K/V/pos must be captured
     * too, not just the counters. All rings share one row geometry, so one
     * stride covers every layer. */
    const size_t n_layers = m->n_layers;
    if (n_layers == 0 || !m->kv || n_layers > SIZE_MAX / sizeof(size_t))
        return OC_ERR_INVALID_ARG;
    size_t saved_total[n_layers];
    const size_t n_ctx_dbg = m->target_ctx_len;
    size_t n_overwrite, vec_save, float_bytes, slot_bytes, saved_bytes;
    if (!df2_size_add(n_ctx_dbg, block, &n_overwrite) ||
        !df2_size_mul(m->kv[0].n_kv_heads, m->kv[0].head_dim, &vec_save) ||
        !df2_size_mul(vec_save, 2, &float_bytes) ||
        !df2_size_mul(float_bytes, sizeof(float), &float_bytes) ||
        !df2_size_add(float_bytes, sizeof(int64_t), &slot_bytes) ||
        !df2_size_mul(n_layers, n_overwrite, &saved_bytes) ||
        !df2_size_mul(saved_bytes, slot_bytes, &saved_bytes))
        return OC_ERR_INVALID_ARG;
    uint8_t *saved_slots = malloc(saved_bytes);
    if (!saved_slots) return OC_ERR_OOM;
    for (size_t li = 0; li < n_layers; li++) {
        OcDFlash2KvRing *r = &m->kv[li];
        saved_total[li] = r->total;
        for (size_t i = 0; i < n_overwrite; i++) {
            size_t slot = (saved_total[li] + i) % r->capacity;
            uint8_t *dst = saved_slots + (li * n_overwrite + i) * slot_bytes;
            memcpy(dst, r->k + slot * vec_save, vec_save * sizeof(float));
            memcpy(dst + vec_save * sizeof(float),
                   r->v + slot * vec_save, vec_save * sizeof(float));
            memcpy(dst + 2 * vec_save * sizeof(float),
                   &r->pos[slot], sizeof(int64_t));
        }
    }

    const size_t H = m->cfg.hidden_size;

    DFlash2BlockScratch s;
    if (!dflash2_block_scratch_alloc(&s, block, H,
                                     m->cfg.num_attention_heads,
                                     m->cfg.num_key_value_heads,
                                     m->cfg.head_dim, m->cfg.conv_kernel_size,
                                     H / m->cfg.conv_group_size,
                                     m->cfg.intermediate_size,
                                     m->target_ctx_len)) {
        free(saved_slots);
        return OC_ERR_OOM;
    }
    /* Same shared layer loop as propose: identical eps handling and OOM
     * behavior by construction (the review flagged the duplicated loops
     * already diverging on eps). */
    df2_pt_on = 0; /* no phase timing on the debug path */
    OcError e = dflash2_forward_block(m, noise_emb, block, &s, out_hidden);
    if (e != OC_OK) {
        /* Restore the ring even on failure so the debug path stays
         * side-effect free; earlier layers may have appended. */
        for (size_t li = 0; li < n_layers; li++) {
            OcDFlash2KvRing *r = &m->kv[li];
            for (size_t i = 0; i < n_overwrite; i++) {
                size_t slot = (saved_total[li] + i) % r->capacity;
                const uint8_t *src = saved_slots +
                    (li * n_overwrite + i) * slot_bytes;
                memcpy(r->k + slot * vec_save, src, vec_save * sizeof(float));
                memcpy(r->v + slot * vec_save,
                       src + vec_save * sizeof(float),
                       vec_save * sizeof(float));
                memcpy(&r->pos[slot],
                       src + 2 * vec_save * sizeof(float), sizeof(int64_t));
            }
            r->total = saved_total[li];
            r->len = saved_total[li] < r->capacity
                         ? saved_total[li] : r->capacity;
        }
        dflash2_block_scratch_free(&s);
        free(saved_slots);
        return e;
    }

    /* Restore ring state: counters and the slots the debug append
     * overwrote (only live while wrapped; harmless otherwise). */
    for (size_t li = 0; li < n_layers; li++) {
        OcDFlash2KvRing *r = &m->kv[li];
        for (size_t i = 0; i < n_overwrite; i++) {
            size_t slot = (saved_total[li] + i) % r->capacity;
            const uint8_t *src = saved_slots +
                (li * n_overwrite + i) * slot_bytes;
            memcpy(r->k + slot * vec_save, src, vec_save * sizeof(float));
            memcpy(r->v + slot * vec_save,
                   src + vec_save * sizeof(float),
                   vec_save * sizeof(float));
            memcpy(&r->pos[slot],
                   src + 2 * vec_save * sizeof(float), sizeof(int64_t));
        }
        r->total = saved_total[li];
        r->len = saved_total[li] < r->capacity ? saved_total[li] : r->capacity;
    }
    free(saved_slots);

    dflash2_block_scratch_free(&s);
    return OC_OK;
}

/*
 * Validation selector-only: mirror CandidateSelector.select with
 * temperature 0 on precomputed draft hidden rows. out_tokens is [n_rows].
 */
OcError oc_dflash2_selector_debug(OcDFlash2Model *m,
                                  const float *draft_hidden, size_t n_rows,
                                  const uint32_t *anchor_ids, size_t n_anchor,
                                  const OcDFlash2Weight *lm_head,
                                  uint32_t *out_tokens,
                                  uint32_t *out_cand)
{
    if (!m || !m->loaded || !draft_hidden || !anchor_ids || n_anchor == 0 ||
        !lm_head || (!lm_head->data && !lm_head->bf16 &&
                     !lm_head->generate) || !out_tokens)
        return OC_ERR_INVALID_ARG;
    if (n_rows == 0 || anchor_ids[n_anchor - 1] >= m->cfg.vocab_size)
        return OC_ERR_INVALID_ARG; /* codebook row would be OOB */
    /* Same [vocab, hidden] contract as oc_dflash2_propose. */
    if (lm_head->cols != m->cfg.hidden_size || lm_head->rows == 0 ||
        lm_head->rows < m->cfg.selector_top_k)
        return OC_ERR_INVALID_ARG;

    const size_t H = m->cfg.hidden_size;
    const size_t top_k = m->cfg.selector_top_k;
    const size_t rank = m->cfg.selector_rank;
    size_t unary_count, proj_count;
    if (!df2_size_mul(n_rows, top_k, &unary_count) ||
        !df2_float_extent(unary_count) ||
        unary_count > SIZE_MAX / sizeof(uint32_t) ||
        !df2_size_mul(n_rows, rank, &proj_count) ||
        !df2_float_extent(proj_count) ||
        !df2_float_extent(top_k) || !df2_float_extent(rank) ||
        !df2_float_extent(H) || !df2_float_extent(lm_head->rows))
        return OC_ERR_INVALID_ARG;
    /* Candidate ids index the selector codebooks [cfg.vocab_size, rank];
     * an lm_head with more rows must not produce ids past that. */
    const size_t vocab = lm_head->rows < m->cfg.vocab_size
                             ? lm_head->rows : m->cfg.vocab_size;

    float *unary = malloc(unary_count * sizeof(float));
    uint32_t *cand = malloc(unary_count * sizeof(uint32_t));
    float *proj_h = malloc(proj_count * sizeof(float));
    float *scores = malloc(top_k * sizeof(float));
    if (!unary || !cand || !proj_h || !scores) {
        free(unary); free(cand); free(proj_h); free(scores);
        return OC_ERR_OOM;
    }

    /* Generated lm_heads need one materialized row per step (the propose
     * path streams them via the same generate contract). */
    float *gen_row = NULL;
    if (!lm_head->data && !lm_head->bf16 && lm_head->generate) {
        gen_row = malloc(H * sizeof(float));
        if (!gen_row) {
            free(unary); free(cand); free(proj_h); free(scores);
            return OC_ERR_OOM;
        }
    }

    for (size_t p = 0; p < n_rows; p++) {
        /* full logits row then top-k selection. lrow holds every lm_head
         * row (gemv_w writes rows floats); vocab only bounds the scan so
         * candidate ids index valid codebook rows. */
        float *lrow = malloc(lm_head->rows * sizeof(float));
        if (!lrow) {
            free(gen_row);
            free(unary); free(cand); free(proj_h); free(scores);
            return OC_ERR_OOM;
        }
        if (lm_head->data || lm_head->bf16) {
            gemv_w(lm_head, draft_hidden + p * H, lrow);
        } else {
            for (size_t v = 0; v < vocab; v++) {
                lm_head->generate(v, H, gen_row, lm_head->gen_user);
                lrow[v] = df2_dot(gen_row, draft_hidden + p * H, H);
            }
        }
        for (size_t k = 0; k < top_k; k++) {
            size_t best = 0;
            for (size_t v = 1; v < vocab; v++)
                if (lrow[v] > lrow[best]) best = v;
            cand[p * top_k + k] = (uint32_t)best;
            unary[p * top_k + k] = lrow[best];
            lrow[best] = -INFINITY;
        }
        free(lrow);
    }
    free(gen_row);

    gemm(&(m->selector.hidden_projection), draft_hidden, n_rows, proj_h);

    uint32_t predecessor = anchor_ids[n_anchor - 1];
    float *A_buf = malloc(rank * sizeof(float));
    float *B_buf = malloc(rank * sizeof(float));
    if (!A_buf || !B_buf) {
        free(A_buf); free(B_buf);
        free(unary); free(cand); free(proj_h); free(scores);
        return OC_ERR_OOM;
    }
    for (size_t p = 0; p < n_rows; p++) {
        df2_widen_row(&m->selector.predecessor_codebook, predecessor, A_buf);
        const float *A_p = A_buf;
        float *un = unary + p * top_k;
        for (size_t k = 0; k < top_k; k++) {
            df2_widen_row(&m->selector.successor_codebook,
                          cand[p * top_k + k], B_buf);
            float acc = 0.0f;
            for (size_t r = 0; r < rank; r++)
                acc += (A_p[r] * proj_h[p * rank + r]) * B_buf[r];
            scores[k] = acc + un[k];
        }
        uint32_t choice = 0;
        for (size_t k = 1; k < top_k; k++)
            if (scores[k] > scores[choice]) choice = (uint32_t)k;
        predecessor = cand[p * top_k + choice];
        out_tokens[p] = predecessor;
        if (out_cand)
            memcpy(out_cand + p * top_k, cand + p * top_k,
                   top_k * sizeof(uint32_t));
    }

    free(A_buf); free(B_buf);
    free(unary); free(cand); free(proj_h); free(scores);
    return OC_OK;
}
