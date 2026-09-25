/* test_attn_flash.c — flash-decoding split/merge and blocked prefill vs a
 * double-precision single-pass reference, for f32, int8 and RQ caches. */
#include <criterion/criterion.h>
#include "oxidize/attn_flash.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t g_s;
static void rs(uint64_t s) { g_s = s; }
static double ru(void)
{
    g_s = g_s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((g_s >> 11) & 0x1FFFFFFFFFFFFFULL) / 9007199254740992.0;
}
static float rg(void)
{
    double u1 = ru(), u2 = ru();
    if (u1 < 1e-300) u1 = 1e-300;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

Test(attn_flash, exp_matches_libm)
{
    float x[203], y[203];
    for (int i = 0; i < 203; i++) x[i] = y[i] = -90.0f + (float)i * 0.47f;
    oc_attn_flash_exp(y, 203);
    for (int i = 0; i < 203; i++) {
        if (x[i] < -87.0f) { cr_assert_eq(y[i], 0.0f); continue; }
        const double r = exp((double)x[i]);
        /* float x*log2(e) loses ~|x|*6e-8 relative; softmax inputs are
         * max-subtracted, so what matters is the small-|x| accuracy. */
        const double tol = fabs(x[i]) < 20.0f ? 1e-6 : 6e-6;
        cr_assert(fabs(y[i] - r) <= tol * r, "exp(%f) = %g vs %g", x[i], y[i], r);
    }
    float m[9] = { -INFINITY, 0, 0, 0, 0, 0, 0, 0, -INFINITY };
    oc_attn_flash_exp(m, 9);
    cr_assert_eq(m[0], 0.0f);
    cr_assert_eq(m[8], 0.0f);
    cr_assert_float_eq(m[1], 1.0f, 1e-7);
}

/* A test cache with materialized K/V rows (what every kind decodes to). */
typedef struct {
    OcKvView v;
    size_t n, d;
    float *K, *V;          /* decoded rows the kernels must reproduce */
    float *MUK;            /* RQ: page mean of K for positions coded as RQ */
    float *kf, *vf;
    int8_t *kq, *vq;
    float *ks, *vsc;
    OcKvRqCache rq;
} TCache;

static void tcache_make(TCache *tc, OcKvViewKind kind, size_t n, size_t d,
                        unsigned kb, unsigned vb, uint32_t sinks,
                        uint32_t window)
{
    memset(tc, 0, sizeof(*tc));
    tc->n = n; tc->d = d;
    tc->K = malloc(n * d * sizeof(float));
    tc->MUK = calloc(n * d, sizeof(float));
    tc->V = malloc(n * d * sizeof(float));
    const size_t stride = 2 * d;  /* two heads per row, test head 1 */
    tc->v.kind = kind;
    tc->v.d = d;
    if (kind == OC_KVV_F32) {
        tc->kf = malloc(n * stride * sizeof(float));
        tc->vf = malloc(n * stride * sizeof(float));
        for (size_t t = 0; t < n; t++)
            for (size_t i = 0; i < d; i++) {
                tc->K[t * d + i] = tc->kf[t * stride + d + i] = rg();
                tc->V[t * d + i] = tc->vf[t * stride + d + i] = rg();
            }
        tc->v.kf = tc->kf + d; tc->v.vf = tc->vf + d; tc->v.fs = stride;
    } else if (kind == OC_KVV_Q8) {
        tc->kq = malloc(n * stride);
        tc->vq = malloc(n * stride);
        tc->ks = malloc(n * 2 * sizeof(float));
        tc->vsc = malloc(n * 2 * sizeof(float));
        for (size_t t = 0; t < n; t++) {
            tc->ks[t * 2 + 1] = 0.02f + (float)ru() * 0.02f;
            tc->vsc[t * 2 + 1] = 0.02f + (float)ru() * 0.02f;
            for (size_t i = 0; i < d; i++) {
                tc->kq[t * stride + d + i] = (int8_t)((int)(ru() * 254) - 127);
                tc->vq[t * stride + d + i] = (int8_t)((int)(ru() * 254) - 127);
                tc->K[t * d + i] = tc->ks[t * 2 + 1] * tc->kq[t * stride + d + i];
                tc->V[t * d + i] = tc->vsc[t * 2 + 1] * tc->vq[t * stride + d + i];
            }
        }
        tc->v.kq = tc->kq + d; tc->v.vq = tc->vq + d; tc->v.qs = stride;
        tc->v.ksc = tc->ks + 1; tc->v.vsc = tc->vsc + 1; tc->v.ss = 2;
    } else {
        OcKvRqParams p = { .k_bits = kb, .v_bits = vb, .n_sink = sinks,
                           .window = window, .rot = OC_KVRQ_ROT_HADAMARD,
                           .seed = 9 };
        cr_assert_eq(oc_kvrq_cache_init(&tc->rq, 1, 2, d, n + 8, &p), OC_OK);
        float k[2 * 256], v[2 * 256], scr[512];
        for (size_t t = 0; t < n; t++) {
            for (size_t i = 0; i < 2 * d; i++) { k[i] = rg() * 3; v[i] = rg(); }
            oc_kvrq_store(&tc->rq, 0, (int64_t)t, k, v, scr);
        }
        /* Decoded (rotated-domain) rows, choosing the exact slot when set. */
        for (size_t t = 0; t < n; t++) {
            const int64_t s = oc_kvrq_slot(&tc->rq, 0, (int64_t)t);
            for (int kind2 = 0; kind2 < 2; kind2++) {
                float *dst = (kind2 ? tc->V : tc->K) + t * d;
                if (s >= 0) {
                    const int8_t *xq = oc_kvrq_xq(&tc->rq, 0, kind2, 1) + s * d;
                    const float xs = oc_kvrq_xs(&tc->rq, 0, kind2, 1)[s];
                    for (size_t i = 0; i < d; i++) dst[i] = xs * xq[i];
                } else {
                    /* RQ block + the per-head mean (centered cache). */
                    oc_kvrq_decode_pos(&tc->rq, 0, (size_t)kind2, 1,
                                       (int64_t)t, dst);
                    const float *mu = NULL;
                    if (oc_kvrq_centered(&tc->rq)) {
                        const size_t pg = oc_kvrq_page_of(&tc->rq, (int64_t)t);
                        if (tc->rq.mu_fixed[pg])
                            mu = oc_kvrq_mu(&tc->rq, 0, pg, 0, 1);
                    }
                    if (kind2 == 0 && mu != NULL)
                        memcpy(tc->MUK + t * d, mu, d * sizeof(float));
                }
            }
        }
        tc->v.rq = &tc->rq; tc->v.layer = 0; tc->v.head = 1;
        tc->v.hi_written = (int64_t)n - 1;
    }
}

static void tcache_free(TCache *tc)
{
    free(tc->K); free(tc->V); free(tc->MUK); free(tc->kf); free(tc->vf);
    free(tc->kq); free(tc->vq); free(tc->ks); free(tc->vsc);
    if (tc->v.kind == OC_KVV_RQ) oc_kvrq_cache_free(&tc->rq);
}

/* qb (when not NULL) scores positions without an RQ exact slot: decode
 * scores RQ blocks with the int8-rounded query (oc_kvrq_prep_q). */
static void ref_attn2(const TCache *tc, const float *q, const float *qb,
                      int64_t lo, int64_t hi, double *out)
{
    const size_t d = tc->d;
    double m = -INFINITY, l = 0;
    double *s = malloc((size_t)(hi - lo + 1) * sizeof(double));
    for (int64_t t = lo; t <= hi; t++) {
        double a = 0;
        const float *qq = q;
        if (qb != NULL && oc_kvrq_slot(&tc->rq, 0, t) < 0) qq = qb;
        for (size_t i = 0; i < d; i++) a += (double)qq[i] * tc->K[t * d + i];
        /* Centered RQ: the kernel scores the residual with the rounded q
         * and adds the page mean term with the exact q. */
        if (qq == qb)
            for (size_t i = 0; i < d; i++)
                a += ((double)q[i] - qb[i]) * tc->MUK[t * d + i];
        s[t - lo] = a;
        if (a > m) m = a;
    }
    for (size_t i = 0; i < d; i++) out[i] = 0;
    for (int64_t t = lo; t <= hi; t++) {
        const double w = exp(s[t - lo] - m);
        l += w;
        for (size_t i = 0; i < d; i++) out[i] += w * tc->V[t * d + i];
    }
    for (size_t i = 0; i < d; i++) out[i] /= l;
    free(s);
}

static void ref_attn(const TCache *tc, const float *q, int64_t lo, int64_t hi,
                     double *out)
{
    ref_attn2(tc, q, NULL, lo, hi, out);
}

static void check_decode(OcKvViewKind kind, unsigned kb, unsigned vb,
                         uint32_t sinks, uint32_t window)
{
    const size_t n = 517, d = 128, G = 4;
    TCache tc;
    rs(100 + kind * 10 + kb);
    tcache_make(&tc, kind, n, d, kb, vb, sinks, window);
    float q[4 * 128];
    for (size_t i = 0; i < G * d; i++) q[i] = rg() * 0.15f;
    float *scr = malloc(oc_attn_flash_decode_scratch(G) * sizeof(float));
    /* Split into uneven parts (incl. an empty one) and merge. */
    const int64_t cuts[6] = { 0, 1, 130, 130, 400, (int64_t)n };
    float m[5 * 4], l[5 * 4], acc[5 * 4 * 128], out[4 * 128], one[4 * 128];
    for (int p = 0; p < 5; p++) {
        if (cuts[p] == cuts[p + 1]) {
            for (size_t g = 0; g < G; g++) { m[p * G + g] = -INFINITY; l[p * G + g] = 0; }
            continue;
        }
        oc_attn_flash_decode_range(&tc.v, q, G, cuts[p], cuts[p + 1],
                                   m + p * G, l + p * G, acc + p * G * d, scr);
    }
    oc_attn_flash_merge(G, d, 5, m, l, acc, out);
    /* Single range = the single-thread reference. */
    oc_attn_flash_decode_range(&tc.v, q, G, 0, (int64_t)n, m, l, acc, scr);
    oc_attn_flash_merge(G, d, 1, m, l, acc, one);
    /* RQ blocks are scored with the int8-rounded query. */
    float qd[4 * 128];
    int8_t q8[4 * 128];
    float qsc[4];
    int32_t qsum[4];
    oc_kvrq_prep_q(q, G, d, q8, qsc, qsum);
    for (size_t i = 0; i < G * d; i++) qd[i] = (float)q8[i] * qsc[i / d];
    for (size_t g = 0; g < G; g++) {
        double ref[128];
        ref_attn2(&tc, q + g * d, kind == OC_KVV_RQ ? qd + g * d : NULL, 0,
                  (int64_t)n - 1, ref);
        for (size_t i = 0; i < d; i++) {
            cr_assert(fabs(out[g * d + i] - ref[i]) < 2e-4 * (1 + fabs(ref[i])),
                      "kind %d g %zu i %zu: split %f ref %f", kind, g, i,
                      out[g * d + i], ref[i]);
            cr_assert(fabs(out[g * d + i] - one[g * d + i]) < 1e-5,
                      "kind %d: split vs single-range differ", kind);
        }
    }
    free(scr);
    tcache_free(&tc);
}

Test(attn_flash, decode_split_matches_reference_f32) { check_decode(OC_KVV_F32, 0, 0, 0, 0); }
Test(attn_flash, decode_split_matches_reference_q8) { check_decode(OC_KVV_Q8, 0, 0, 0, 0); }
Test(attn_flash, decode_split_matches_reference_rq)
{
    check_decode(OC_KVV_RQ, 3, 2, 0, 0);
    check_decode(OC_KVV_RQ, 4, 4, 0, 0);
    check_decode(OC_KVV_RQ, 2, 3, 0, 0);
}
Test(attn_flash, decode_rq_sinks_and_window)
{
    check_decode(OC_KVV_RQ, 2, 2, 4, 100);
    check_decode(OC_KVV_RQ, 3, 2, 4, 0);
}

static void check_prefill(OcKvViewKind kind, int64_t sw, uint32_t sinks,
                          uint32_t window)
{
    const size_t n = 301, d = 128, G = 4, ntok = 13;
    TCache tc;
    rs(500 + kind + (uint64_t)sw);
    tcache_make(&tc, kind, n, d, 3, 2, sinks, window);
    /* The last ntok positions are the chunk; rows = tokens x G. */
    const size_t R = ntok * G;
    float *q = malloc(R * d * sizeof(float));
    float *out = malloc(R * d * sizeof(float));
    int64_t *pos = malloc(R * sizeof(int64_t));
    for (size_t r = 0; r < R; r++) {
        pos[r] = (int64_t)(n - ntok + r / G);
        for (size_t i = 0; i < d; i++) q[r * d + i] = rg() * 0.15f;
    }
    float *scr = malloc(oc_attn_flash_prefill_scratch(R, d) * sizeof(float));
    oc_attn_flash_prefill(&tc.v, q, R, pos, sw, out, scr);
    for (size_t r = 0; r < R; r++) {
        double ref[128];
        int64_t lo = 0;
        if (sw > 0 && pos[r] - sw + 1 > 0) lo = pos[r] - sw + 1;
        ref_attn(&tc, q + r * d, lo, pos[r], ref);
        for (size_t i = 0; i < d; i++)
            cr_assert(fabs(out[r * d + i] - ref[i]) < 2e-4 * (1 + fabs(ref[i])),
                      "kind %d sw %lld row %zu i %zu: %f vs %f", kind,
                      (long long)sw, r, i, out[r * d + i], ref[i]);
    }
    free(q); free(out); free(pos); free(scr);
    tcache_free(&tc);
}

Test(attn_flash, prefill_matches_reference)
{
    check_prefill(OC_KVV_F32, 0, 0, 0);
    check_prefill(OC_KVV_Q8, 0, 0, 0);
    check_prefill(OC_KVV_RQ, 0, 0, 0);
    check_prefill(OC_KVV_RQ, 0, 4, 64);
}

Test(attn_flash, prefill_sliding_window)
{
    check_prefill(OC_KVV_F32, 70, 0, 0);
    check_prefill(OC_KVV_Q8, 5, 0, 0);
}
