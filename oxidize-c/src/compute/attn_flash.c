/* attn_flash.c — flash-decoding partials and blocked prefill attention over
 * one kv head of an f32 / int8 / RotorQuant cache. See attn_flash.h. */
#include "oxidize/attn_flash.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define T_ OC_FLASH_TILE

static int g_fl_isa = -1;
static pthread_once_t g_fl_once = PTHREAD_ONCE_INIT;
static void fl_detect_once(void)
{
#if defined(__x86_64__) || defined(__i386__)
    g_fl_isa = (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")
                && getenv("OC_KVRQ_SCALAR") == NULL) ? 1 : 0;
#else
    g_fl_isa = 0;
#endif
}
static int fl_isa(void)
{
    pthread_once(&g_fl_once, fl_detect_once);
    return g_fl_isa;
}

/* ─── exp ─────────────────────────────────────────────────────────────── */

static void exp_scalar(float *x, size_t n)
{
    for (size_t i = 0; i < n; i++) x[i] = x[i] < -87.0f ? 0.0f : expf(x[i]);
}

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define FL_TGT __attribute__((target("avx2,fma")))

static inline FL_TGT __m256 exp256(__m256 x)
{
    const __m256 lo = _mm256_set1_ps(-87.0f);
    const __m256 keep = _mm256_cmp_ps(x, lo, _CMP_GE_OQ);
    x = _mm256_min_ps(_mm256_max_ps(x, lo), _mm256_set1_ps(88.0f));
    const __m256 t = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f));
    const __m256 nf = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT |
                                         _MM_FROUND_NO_EXC);
    const __m256 y = _mm256_mul_ps(_mm256_sub_ps(t, nf),
                                   _mm256_set1_ps(0.69314718055994531f));
    __m256 p = _mm256_set1_ps(1.0f / 720.0f);
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.0f / 120.0f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.0f / 24.0f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.0f / 6.0f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.0f));
    const __m256i e = _mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvtps_epi32(nf), _mm256_set1_epi32(127)), 23);
    return _mm256_and_ps(_mm256_mul_ps(p, _mm256_castsi256_ps(e)), keep);
}

FL_TGT static void exp_avx2(float *x, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x + i, exp256(_mm256_loadu_ps(x + i)));
    if (i < n) {
        float tmp[8] = {0};
        for (size_t j = 0; j < n - i; j++) tmp[j] = x[i + j];
        _mm256_storeu_ps(tmp, exp256(_mm256_loadu_ps(tmp)));
        for (size_t j = 0; j < n - i; j++) x[i + j] = tmp[j];
    }
}

/* S[r*T_ + c] = sum_k Q[r*d + k] * KT[k*T_ + c], r in [0,4), c in [0,T_) */
FL_TGT static void mk_qk_avx2(const float *Q, size_t d, const float *KT,
                              float *S)
{
    for (size_t cb = 0; cb < T_; cb += 16) {
        __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
        __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
        __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
        __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
        const float *q0 = Q, *q1 = Q + d, *q2 = Q + 2 * d, *q3 = Q + 3 * d;
        for (size_t k = 0; k < d; k++) {
            const __m256 b0 = _mm256_loadu_ps(KT + k * T_ + cb);
            const __m256 b1 = _mm256_loadu_ps(KT + k * T_ + cb + 8);
            __m256 a = _mm256_broadcast_ss(q0 + k);
            a00 = _mm256_fmadd_ps(a, b0, a00); a01 = _mm256_fmadd_ps(a, b1, a01);
            a = _mm256_broadcast_ss(q1 + k);
            a10 = _mm256_fmadd_ps(a, b0, a10); a11 = _mm256_fmadd_ps(a, b1, a11);
            a = _mm256_broadcast_ss(q2 + k);
            a20 = _mm256_fmadd_ps(a, b0, a20); a21 = _mm256_fmadd_ps(a, b1, a21);
            a = _mm256_broadcast_ss(q3 + k);
            a30 = _mm256_fmadd_ps(a, b0, a30); a31 = _mm256_fmadd_ps(a, b1, a31);
        }
        _mm256_storeu_ps(S + 0 * T_ + cb, a00); _mm256_storeu_ps(S + 0 * T_ + cb + 8, a01);
        _mm256_storeu_ps(S + 1 * T_ + cb, a10); _mm256_storeu_ps(S + 1 * T_ + cb + 8, a11);
        _mm256_storeu_ps(S + 2 * T_ + cb, a20); _mm256_storeu_ps(S + 2 * T_ + cb + 8, a21);
        _mm256_storeu_ps(S + 3 * T_ + cb, a30); _mm256_storeu_ps(S + 3 * T_ + cb + 8, a31);
    }
}

/* O[r*d + i] += sum_t P[r*T_ + t] * V[t*d + i], r in [0,4) */
FL_TGT static void mk_pv_avx2(const float *P, size_t n, const float *V,
                              size_t d, float *O)
{
    for (size_t db = 0; db < d; db += 16) {
        float *o0 = O + db, *o1 = O + d + db, *o2 = O + 2 * d + db,
              *o3 = O + 3 * d + db;
        __m256 a00 = _mm256_loadu_ps(o0), a01 = _mm256_loadu_ps(o0 + 8);
        __m256 a10 = _mm256_loadu_ps(o1), a11 = _mm256_loadu_ps(o1 + 8);
        __m256 a20 = _mm256_loadu_ps(o2), a21 = _mm256_loadu_ps(o2 + 8);
        __m256 a30 = _mm256_loadu_ps(o3), a31 = _mm256_loadu_ps(o3 + 8);
        for (size_t t = 0; t < n; t++) {
            const __m256 v0 = _mm256_loadu_ps(V + t * d + db);
            const __m256 v1 = _mm256_loadu_ps(V + t * d + db + 8);
            __m256 p = _mm256_broadcast_ss(P + t);
            a00 = _mm256_fmadd_ps(p, v0, a00); a01 = _mm256_fmadd_ps(p, v1, a01);
            p = _mm256_broadcast_ss(P + T_ + t);
            a10 = _mm256_fmadd_ps(p, v0, a10); a11 = _mm256_fmadd_ps(p, v1, a11);
            p = _mm256_broadcast_ss(P + 2 * T_ + t);
            a20 = _mm256_fmadd_ps(p, v0, a20); a21 = _mm256_fmadd_ps(p, v1, a21);
            p = _mm256_broadcast_ss(P + 3 * T_ + t);
            a30 = _mm256_fmadd_ps(p, v0, a30); a31 = _mm256_fmadd_ps(p, v1, a31);
        }
        _mm256_storeu_ps(o0, a00); _mm256_storeu_ps(o0 + 8, a01);
        _mm256_storeu_ps(o1, a10); _mm256_storeu_ps(o1 + 8, a11);
        _mm256_storeu_ps(o2, a20); _mm256_storeu_ps(o2 + 8, a21);
        _mm256_storeu_ps(o3, a30); _mm256_storeu_ps(o3 + 8, a31);
    }
}
#define FL_HAVE_AVX2 1
#else
#define FL_HAVE_AVX2 0
#endif

void oc_attn_flash_exp(float *x, size_t n)
{
#if FL_HAVE_AVX2
    if (fl_isa()) { exp_avx2(x, n); return; }
#endif
    exp_scalar(x, n);
}

static void mk_qk(const float *Q, size_t d, const float *KT, float *S)
{
#if FL_HAVE_AVX2
    if (fl_isa() && d % 16 == 0) { mk_qk_avx2(Q, d, KT, S); return; }
#endif
    for (size_t r = 0; r < 4; r++)
        for (size_t c = 0; c < T_; c++) {
            float a = 0.0f;
            for (size_t k = 0; k < d; k++) a += Q[r * d + k] * KT[k * T_ + c];
            S[r * T_ + c] = a;
        }
}

static void mk_pv(const float *P, size_t n, const float *V, size_t d, float *O)
{
#if FL_HAVE_AVX2
    if (fl_isa() && d % 16 == 0) { mk_pv_avx2(P, n, V, d, O); return; }
#endif
    for (size_t r = 0; r < 4; r++)
        for (size_t t = 0; t < n; t++) {
            const float p = P[r * T_ + t];
            if (p == 0.0f) continue;
            for (size_t i = 0; i < d; i++) O[r * d + i] += p * V[t * d + i];
        }
}

/* ─── Segments: runs of positions with one storage source ────────────── */

typedef struct {
    int64_t t;      /* first position */
    size_t  n;
    int64_t slot;   /* >= 0: RQ exact slot of the first position; -1: main */
} Seg;

static size_t make_segs(const OcKvView *v, int64_t t0, size_t n, Seg *seg)
{
    if (v->kind != OC_KVV_RQ || v->rq->n_slots == 0) {
        seg[0] = (Seg){ t0, n, -1 };
        return 1;
    }
    const OcKvRqCache *c = v->rq;
    const int64_t safe_lo = (int64_t)c->p.n_sink;
    const int64_t safe_hi = v->hi_written - (int64_t)c->p.window;
    if (t0 >= safe_lo && t0 + (int64_t)n - 1 <= safe_hi) {
        seg[0] = (Seg){ t0, n, -1 };
        return 1;
    }
    size_t ns = 0;
    for (size_t i = 0; i < n; i++) {
        const int64_t t = t0 + (int64_t)i;
        const int64_t s = oc_kvrq_slot(c, v->layer, t);
        if (ns > 0) {
            Seg *p = &seg[ns - 1];
            if ((s < 0 && p->slot < 0) ||
                (s >= 0 && p->slot >= 0 && s == p->slot + (int64_t)p->n)) {
                p->n++;
                continue;
            }
        }
        seg[ns++] = (Seg){ t, 1, s };
    }
    return ns;
}

static void seg_score(const OcKvView *v, const Seg *sg, const float *q,
                      size_t G, float *S, size_t ss)
{
    const size_t d = v->d;
    switch (v->kind) {
    case OC_KVV_F32:
        oc_kvf32_score(v->kf + (size_t)sg->t * v->fs, v->fs, d, sg->n, q, G,
                       S, ss);
        return;
    case OC_KVV_Q8:
        oc_kvq8_score(v->kq + (size_t)sg->t * v->qs, v->qs,
                      v->ksc + (size_t)sg->t * v->ss, v->ss, d, sg->n, q, G,
                      S, ss);
        return;
    default: {
        const OcKvRqCache *c = v->rq;
        if (sg->slot >= 0) {
            oc_kvq8_score(oc_kvrq_xq(c, v->layer, 0, v->head) +
                          (size_t)sg->slot * d, d,
                          oc_kvrq_xs(c, v->layer, 0, v->head) + sg->slot, 1,
                          d, sg->n, q, G, S, ss);
        } else {
            oc_kvrq_score(&c->kc, oc_kvrq_kblocks(c, v->layer, v->head) +
                          (size_t)sg->t * c->kc.block_bytes, sg->n, q, G, S,
                          ss);
        }
        return;
    }
    }
}

static void seg_accum(const OcKvView *v, const Seg *sg, const float *w,
                      size_t ws, size_t G, float *acc)
{
    const size_t d = v->d;
    switch (v->kind) {
    case OC_KVV_F32:
        oc_kvf32_accum(v->vf + (size_t)sg->t * v->fs, v->fs, d, sg->n, w, ws,
                       G, acc);
        return;
    case OC_KVV_Q8:
        oc_kvq8_accum(v->vq + (size_t)sg->t * v->qs, v->qs,
                      v->vsc + (size_t)sg->t * v->ss, v->ss, d, sg->n, w, ws,
                      G, acc);
        return;
    default: {
        const OcKvRqCache *c = v->rq;
        if (sg->slot >= 0) {
            oc_kvq8_accum(oc_kvrq_xq(c, v->layer, 1, v->head) +
                          (size_t)sg->slot * d, d,
                          oc_kvrq_xs(c, v->layer, 1, v->head) + sg->slot, 1,
                          d, sg->n, w, ws, G, acc);
        } else {
            oc_kvrq_accum(&c->vc, oc_kvrq_vblocks(c, v->layer, v->head) +
                          (size_t)sg->t * c->vc.block_bytes, sg->n, w, ws, G,
                          acc);
        }
        return;
    }
    }
}

/* Rows [n][d] of K (kind 0) or V (kind 1) for one segment. */
static void seg_rows(const OcKvView *v, const Seg *sg, int kind, float *rows)
{
    const size_t d = v->d;
    switch (v->kind) {
    case OC_KVV_F32: {
        const float *src = (kind ? v->vf : v->kf) + (size_t)sg->t * v->fs;
        for (size_t t = 0; t < sg->n; t++)
            memcpy(rows + t * d, src + t * v->fs, d * sizeof(float));
        return;
    }
    case OC_KVV_Q8:
        oc_kvq8_decode_rows((kind ? v->vq : v->kq) + (size_t)sg->t * v->qs,
                            v->qs,
                            (kind ? v->vsc : v->ksc) + (size_t)sg->t * v->ss,
                            v->ss, d, sg->n, rows);
        return;
    default: {
        const OcKvRqCache *c = v->rq;
        if (sg->slot >= 0) {
            oc_kvq8_decode_rows(oc_kvrq_xq(c, v->layer, (size_t)kind, v->head)
                                + (size_t)sg->slot * d, d,
                                oc_kvrq_xs(c, v->layer, (size_t)kind, v->head)
                                + sg->slot, 1, d, sg->n, rows);
        } else if (kind == 0) {
            oc_kvrq_decode_rows(&c->kc, oc_kvrq_kblocks(c, v->layer, v->head)
                                + (size_t)sg->t * c->kc.block_bytes, sg->n,
                                rows);
        } else {
            oc_kvrq_decode_rows(&c->vc, oc_kvrq_vblocks(c, v->layer, v->head)
                                + (size_t)sg->t * c->vc.block_bytes, sg->n,
                                rows);
        }
        return;
    }
    }
}

/* The dense caches are [pos][kv_head][d]: one head's rows sit a whole
 * cache row apart (1 KB for q8 on K2), so the hardware prefetchers lose the
 * stream at every page boundary and decode turns latency-bound. Prefetch
 * the next block's rows while this one is computed. RQ is [head][pos] and
 * streams sequentially without help. */
static void dense_prefetch(const OcKvView *v, int64_t t0, size_t n)
{
    if (v->kind == OC_KVV_F32) {
        const size_t bytes = v->d * sizeof(float);
        for (size_t t = 0; t < n; t++) {
            const char *k = (const char *)(v->kf + (size_t)(t0 + (int64_t)t) * v->fs);
            const char *vv = (const char *)(v->vf + (size_t)(t0 + (int64_t)t) * v->fs);
            for (size_t o = 0; o < bytes; o += 64) {
                __builtin_prefetch(k + o, 0, 3);
                __builtin_prefetch(vv + o, 0, 3);
            }
        }
    } else if (v->kind == OC_KVV_Q8) {
        for (size_t t = 0; t < n; t++) {
            const size_t r = (size_t)(t0 + (int64_t)t);
            const char *k = (const char *)(v->kq + r * v->qs);
            const char *vv = (const char *)(v->vq + r * v->qs);
            for (size_t o = 0; o < v->d; o += 64) {
                __builtin_prefetch(k + o, 0, 3);
                __builtin_prefetch(vv + o, 0, 3);
            }
            __builtin_prefetch(v->ksc + r * v->ss, 0, 3);
            __builtin_prefetch(v->vsc + r * v->ss, 0, 3);
        }
    }
}

/* ─── Decode ──────────────────────────────────────────────────────────── */

size_t oc_attn_flash_decode_scratch(size_t G)
{
    return G * T_ + T_;
}

void oc_attn_flash_decode_range(const OcKvView *v, const float *q, size_t G,
                                int64_t t0, int64_t t1, float *m, float *l,
                                float *acc, float *scratch)
{
    const size_t d = v->d;
    float *S = scratch;
    Seg seg[T_];
    for (size_t g = 0; g < G; g++) { m[g] = -INFINITY; l[g] = 0.0f; }
    memset(acc, 0, G * d * sizeof(float));
    const int dense = v->kind != OC_KVV_RQ;
    if (dense)
        dense_prefetch(v, t0, (size_t)(t1 - t0 < (int64_t)T_ ? t1 - t0
                                                              : (int64_t)T_));
    for (int64_t tb = t0; tb < t1; tb += T_) {
        const size_t n = (size_t)(t1 - tb < (int64_t)T_ ? t1 - tb : (int64_t)T_);
        if (dense && tb + (int64_t)T_ < t1) {
            const int64_t nb = tb + (int64_t)T_;
            dense_prefetch(v, nb, (size_t)(t1 - nb < (int64_t)T_ ? t1 - nb
                                                                  : (int64_t)T_));
        }
        const size_t ns = make_segs(v, tb, n, seg);
        for (size_t s = 0; s < ns; s++)
            seg_score(v, &seg[s], q, G, S + (size_t)(seg[s].t - tb), T_);
        for (size_t g = 0; g < G; g++) {
            float *sg = S + g * T_;
            float bm = sg[0];
            for (size_t i = 1; i < n; i++) if (sg[i] > bm) bm = sg[i];
            const float mn = bm > m[g] ? bm : m[g];
            for (size_t i = 0; i < n; i++) sg[i] -= mn;
            oc_attn_flash_exp(sg, n);
            float sum = 0.0f;
            for (size_t i = 0; i < n; i++) sum += sg[i];
            if (m[g] != -INFINITY && mn != m[g]) {
                const float alpha = expf(m[g] - mn);
                float *a = acc + g * d;
                for (size_t i = 0; i < d; i++) a[i] *= alpha;
                l[g] *= alpha;
            }
            l[g] += sum;
            m[g] = mn;
        }
        for (size_t s = 0; s < ns; s++)
            seg_accum(v, &seg[s], S + (size_t)(seg[s].t - tb), T_, G, acc);
    }
}

void oc_attn_flash_merge(size_t G, size_t d, size_t n_parts, const float *m,
                         const float *l, const float *acc, float *out)
{
    for (size_t g = 0; g < G; g++) {
        float M = -INFINITY;
        for (size_t p = 0; p < n_parts; p++)
            if (m[p * G + g] > M) M = m[p * G + g];
        float *o = out + g * d;
        memset(o, 0, d * sizeof(float));
        if (M == -INFINITY) continue;
        float L = 0.0f;
        for (size_t p = 0; p < n_parts; p++) {
            const float mp = m[p * G + g];
            if (mp == -INFINITY) continue;
            const float w = expf(mp - M);
            L += w * l[p * G + g];
            const float *a = acc + (p * G + g) * d;
            for (size_t i = 0; i < d; i++) o[i] += w * a[i];
        }
        if (L > 0.0f) {
            const float inv = 1.0f / L;
            for (size_t i = 0; i < d; i++) o[i] *= inv;
        }
    }
}

/* ─── Prefill ─────────────────────────────────────────────────────────── */

size_t oc_attn_flash_prefill_scratch(size_t R, size_t d)
{
    const size_t Rp = (R + 3) & ~(size_t)3;
    /* Qp, O, S, m, l, pos(as float slots x2), KT, Kt, Vt */
    return Rp * d * 2 + Rp * T_ + Rp * 4 + d * T_ + 2 * T_ * d + 16;
}

void oc_attn_flash_prefill(const OcKvView *v, const float *q, size_t R,
                           const int64_t *row_pos, int64_t sw, float *out,
                           float *scratch)
{
    if (R == 0) return;
    const size_t d = v->d;
    const size_t Rp = (R + 3) & ~(size_t)3;
    float *Qp = scratch;
    float *O = Qp + Rp * d;
    float *S = O + Rp * d;
    float *rm = S + Rp * T_;
    float *rl = rm + Rp;
    int64_t *rp = (int64_t *)(void *)(rl + Rp);   /* Rp int64 = 2*Rp floats */
    float *KT = rl + Rp + 2 * Rp;
    float *Kt = KT + d * T_;
    float *Vt = Kt + T_ * d;
    Seg seg[T_];

    memcpy(Qp, q, R * d * sizeof(float));
    for (size_t r = 0; r < Rp; r++) {
        if (r >= R) memcpy(Qp + r * d, q + (R - 1) * d, d * sizeof(float));
        rp[r] = row_pos[r < R ? r : R - 1];
        rm[r] = -INFINITY;
        rl[r] = 0.0f;
    }
    memset(O, 0, Rp * d * sizeof(float));
    const int64_t max_pos = rp[Rp - 1];
    int64_t first = 0;
    if (sw > 0 && rp[0] - sw + 1 > 0) first = rp[0] - sw + 1;

    for (int64_t tb = first; tb <= max_pos; tb += T_) {
        const size_t n = (size_t)(max_pos + 1 - tb < (int64_t)T_
                                  ? max_pos + 1 - tb : (int64_t)T_);
        size_t r0 = 0;
        while (r0 < Rp && rp[r0] < tb) r0++;
        r0 &= ~(size_t)3;

        const size_t ns = make_segs(v, tb, n, seg);
        for (size_t s = 0; s < ns; s++)
            seg_rows(v, &seg[s], 0, Kt + (size_t)(seg[s].t - tb) * d);
        if (n < T_) memset(Kt + n * d, 0, (T_ - n) * d * sizeof(float));
        for (size_t t = 0; t < T_; t++)
            for (size_t k = 0; k < d; k++) KT[k * T_ + t] = Kt[t * d + k];

        for (size_t rb = r0; rb < Rp; rb += 4)
            mk_qk(Qp + rb * d, d, KT, S + rb * T_);

        for (size_t r = r0; r < Rp; r++) {
            float *sr = S + r * T_;
            const int64_t lim64 = rp[r] - tb;   /* last valid column */
            const size_t valid = lim64 < 0 ? 0
                               : (size_t)(lim64 + 1 < (int64_t)n ? lim64 + 1
                                                                 : (int64_t)n);
            /* Sliding window: columns before the row's window start. */
            size_t lo = 0;
            if (sw > 0) {
                const int64_t ws = rp[r] - sw + 1 - tb;
                if (ws > 0) lo = ws > (int64_t)valid ? valid : (size_t)ws;
            }
            if (valid <= lo) {
                memset(sr, 0, T_ * sizeof(float));
                continue;
            }
            float bm = sr[lo];
            for (size_t i = lo + 1; i < valid; i++) if (sr[i] > bm) bm = sr[i];
            const float mn = bm > rm[r] ? bm : rm[r];
            for (size_t i = 0; i < lo; i++) sr[i] = -INFINITY;
            for (size_t i = lo; i < valid; i++) sr[i] -= mn;
            for (size_t i = valid; i < T_; i++) sr[i] = -INFINITY;
            oc_attn_flash_exp(sr, T_);
            float sum = 0.0f;
            for (size_t i = lo; i < valid; i++) sum += sr[i];
            if (rm[r] != -INFINITY && mn != rm[r]) {
                const float alpha = expf(rm[r] - mn);
                float *o = O + r * d;
                for (size_t i = 0; i < d; i++) o[i] *= alpha;
                rl[r] *= alpha;
            }
            rl[r] += sum;
            rm[r] = mn;
        }

        for (size_t s = 0; s < ns; s++)
            seg_rows(v, &seg[s], 1, Vt + (size_t)(seg[s].t - tb) * d);
        for (size_t rb = r0; rb < Rp; rb += 4)
            mk_pv(S + rb * T_, n, Vt, d, O + rb * d);
    }

    for (size_t r = 0; r < R; r++) {
        const float inv = rl[r] > 0.0f ? 1.0f / rl[r] : 0.0f;
        for (size_t i = 0; i < d; i++) out[r * d + i] = O[r * d + i] * inv;
    }
}
