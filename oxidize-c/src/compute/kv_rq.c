/* kv_rq.c — RotorQuant KV cache: codec, lazily-committed cache and fused
 * AVX2 attention kernels. See include/oxidize/kv_rq.h for the format.
 *
 * Like attn_kernels.c, the AVX2 bodies are compiled with target attributes
 * and picked at runtime, so the default (no -march) build still gets them and
 * a non-AVX2 CPU falls back to the scalar reference, which the tests compare
 * against. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE   /* MAP_ANONYMOUS / MAP_NORESERVE */
#endif

#include "oxidize/kv_rq.h"
#include "oxidize/flash_attention.h"   /* oc_f16/f32 bit conversions */

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif

/* ─── Rotation ─────────────────────────────────────────────────────────── */

static uint64_t kvrq_splitmix(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static int is_pow2(size_t d) { return d != 0 && (d & (d - 1)) == 0; }

OcError oc_kvrq_rot_init(OcKvRqRot *r, size_t d, OcKvRqRotKind kind,
                         uint64_t seed)
{
    if (r == NULL || d == 0 || d > OC_KVRQ_DIM_MAX) return OC_ERR_INVALID_ARG;
    memset(r, 0, sizeof(*r));
    r->d = d;
    r->kind = kind;
    if (kind == OC_KVRQ_ROT_HADAMARD) {
        if (!is_pow2(d)) return OC_ERR_INVALID_ARG;
        uint64_t s = seed ? seed : 0x5EEDu;
        for (size_t i = 0; i < d; i++)
            r->sign[i] = (kvrq_splitmix(&s) >> 63) ? -1.0f : 1.0f;
        return OC_OK;
    }
    if (kind == OC_KVRQ_ROT_ISO) {
        if (d % 4 != 0) return OC_ERR_INVALID_ARG;
        return oc_rotorquant_init(&r->iso, OC_RQ_ISO, d, 2, seed ? seed : 42u);
    }
    if (kind == OC_KVRQ_ROT_NONE) return OC_OK;
    return OC_ERR_INVALID_ARG;
}

void oc_kvrq_rot_free(OcKvRqRot *r)
{
    if (r == NULL) return;
    if (r->kind == OC_KVRQ_ROT_ISO) oc_rotorquant_free(&r->iso);
    memset(r, 0, sizeof(*r));
}

/* In-place unnormalized fast Walsh-Hadamard transform. */
static void fwht_scalar(float *v, size_t d)
{
    for (size_t h = 1; h < d; h <<= 1) {
        for (size_t i = 0; i < d; i += 2 * h) {
            for (size_t j = i; j < i + h; j++) {
                float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
}

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

__attribute__((target("avx2,fma")))
static void fwht_avx2(float *v, size_t d)
{
    /* Within-register stages h = 1, 2, 4. Lane i gets x_i + x_{i^h} when bit
     * h of i is clear and x_{i^h} - x_i when set. */
    for (size_t i = 0; i < d; i += 8) {
        __m256 x = _mm256_loadu_ps(v + i);
        __m256 b = _mm256_permute_ps(x, 0xB1);
        x = _mm256_blend_ps(_mm256_add_ps(x, b), _mm256_sub_ps(b, x), 0xAA);
        b = _mm256_permute_ps(x, 0x4E);
        x = _mm256_blend_ps(_mm256_add_ps(x, b), _mm256_sub_ps(b, x), 0xCC);
        b = _mm256_permute2f128_ps(x, x, 1);
        x = _mm256_blend_ps(_mm256_add_ps(x, b), _mm256_sub_ps(b, x), 0xF0);
        _mm256_storeu_ps(v + i, x);
    }
    for (size_t h = 8; h < d; h <<= 1) {
        for (size_t i = 0; i < d; i += 2 * h) {
            for (size_t j = i; j < i + h; j += 8) {
                __m256 a = _mm256_loadu_ps(v + j);
                __m256 b = _mm256_loadu_ps(v + j + h);
                _mm256_storeu_ps(v + j, _mm256_add_ps(a, b));
                _mm256_storeu_ps(v + j + h, _mm256_sub_ps(a, b));
            }
        }
    }
}
#endif

static int g_kvrq_isa = -1;
static pthread_once_t g_kvrq_once = PTHREAD_ONCE_INIT;
static void kvrq_detect_once(void)
{
#if defined(__x86_64__) || defined(__i386__)
    g_kvrq_isa = (__builtin_cpu_supports("avx2") &&
                  __builtin_cpu_supports("fma") &&
                  __builtin_cpu_supports("f16c") && getenv("OC_KVRQ_SCALAR") == NULL)
                 ? 1 : 0;
#else
    g_kvrq_isa = 0;
#endif
}
static int kvrq_isa(void)
{
    pthread_once(&g_kvrq_once, kvrq_detect_once);
    return g_kvrq_isa;
}

static void fwht(float *v, size_t d)
{
#if defined(__x86_64__) || defined(__i386__)
    if (d >= 8 && kvrq_isa()) { fwht_avx2(v, d); return; }
#endif
    fwht_scalar(v, d);
}

void oc_kvrq_rotate(const OcKvRqRot *r, const float *in, float *out)
{
    const size_t d = r->d;
    if (r->kind == OC_KVRQ_ROT_HADAMARD) {
        const float inv = 1.0f / sqrtf((float)d);
        for (size_t i = 0; i < d; i++) out[i] = in[i] * r->sign[i];
        fwht(out, d);
        for (size_t i = 0; i < d; i++) out[i] *= inv;
    } else {
        if (out != in) memcpy(out, in, d * sizeof(float));
        if (r->kind == OC_KVRQ_ROT_ISO) oc_rotorquant_rotate(&r->iso, out);
    }
}

void oc_kvrq_unrotate(const OcKvRqRot *r, const float *in, float *out)
{
    const size_t d = r->d;
    if (r->kind == OC_KVRQ_ROT_HADAMARD) {
        const float inv = 1.0f / sqrtf((float)d);
        if (out != in) memcpy(out, in, d * sizeof(float));
        fwht(out, d);
        for (size_t i = 0; i < d; i++) out[i] *= inv * r->sign[i];
    } else {
        if (out != in) memcpy(out, in, d * sizeof(float));
        if (r->kind == OC_KVRQ_ROT_ISO) oc_rotorquant_unrotate(&r->iso, out);
    }
}

/* ─── Codec ───────────────────────────────────────────────────────────── */

OcError oc_kvrq_codec_init(OcKvRqCodec *c, size_t d, unsigned bits)
{
    if (c == NULL || bits < 2 || bits > 4) return OC_ERR_INVALID_ARG;
    if (!(d == 64 || d == 128 || d == 256)) return OC_ERR_INVALID_ARG;
    memset(c, 0, sizeof(*c));
    c->d = d;
    c->bits = bits;
    c->n_levels = 1u << bits;
    c->block_bytes = 2u + d * bits / 8u;
    OcError e = oc_rotorquant_lloyd_max(d, bits, c->centroids);
    if (e != OC_OK) return e;
    float mx = 0.0f;
    for (unsigned i = 0; i < c->n_levels; i++)
        if (fabsf(c->centroids[i]) > mx) mx = fabsf(c->centroids[i]);
    if (!(mx > 0.0f)) return OC_ERR_INVALID_ARG;
    c->cscale = mx / 63.0f;
    for (unsigned i = 0; i < 16; i++) {
        long v = 0;
        if (i < c->n_levels) {
            v = lrintf(c->centroids[i] / c->cscale);
            if (v > 63) v = 63;
            if (v < -63) v = -63;
            c->centroids[i] = (float)v * c->cscale;
        }
        c->ci[i] = (int8_t)v;
        c->ciu[i] = (uint8_t)(v + 64);
    }
    for (unsigned i = 0; i + 1 < c->n_levels; i++)
        c->bounds[i] = 0.5f * (c->centroids[i] + c->centroids[i + 1]);
    return OC_OK;
}

static inline unsigned code_get(const OcKvRqCodec *c, const uint8_t *codes,
                                size_t e)
{
    const size_t d = c->d;
    switch (c->bits) {
    case 2: {
        const size_t q = d / 4;
        return (codes[e % q] >> (2 * (e / q))) & 3u;
    }
    case 3: {
        const size_t q = d / 4, o = d / 8;
        unsigned lo = (codes[e % q] >> (2 * (e / q))) & 3u;
        unsigned hi = (codes[q + e % o] >> (e / o)) & 1u;
        return lo | (hi << 2);
    }
    default: {
        const size_t h = d / 2;
        return e < h ? (codes[e] & 15u) : (codes[e - h] >> 4);
    }
    }
}

static inline void code_set(const OcKvRqCodec *c, uint8_t *codes, size_t e,
                            unsigned v)
{
    const size_t d = c->d;
    switch (c->bits) {
    case 2: {
        const size_t q = d / 4;
        codes[e % q] |= (uint8_t)((v & 3u) << (2 * (e / q)));
        break;
    }
    case 3: {
        const size_t q = d / 4, o = d / 8;
        codes[e % q] |= (uint8_t)((v & 3u) << (2 * (e / q)));
        codes[q + e % o] |= (uint8_t)(((v >> 2) & 1u) << (e / o));
        break;
    }
    default: {
        const size_t h = d / 2;
        if (e < h) codes[e] |= (uint8_t)(v & 15u);
        else codes[e - h] |= (uint8_t)((v & 15u) << 4);
        break;
    }
    }
}

static inline uint16_t load_u16(const uint8_t *p)
{
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

void oc_kvrq_encode(const OcKvRqCodec *c, const float *xr, uint8_t *blk)
{
    const size_t d = c->d;
    uint8_t *codes = blk + 2;
    memset(codes, 0, c->block_bytes - 2);
    double ss = 0.0;
    for (size_t i = 0; i < d; i++) ss += (double)xr[i] * xr[i];
    const double norm = sqrt(ss);
    if (norm == 0.0 || !isfinite(norm)) {
        const uint16_t z = 0;
        memcpy(blk, &z, 2);
        return;
    }
    const float inv = (float)(1.0 / norm);
    double cc = 0.0;
    for (size_t i = 0; i < d; i++) {
        const float u = xr[i] * inv;
        unsigned k = 0;
        while (k + 1 < c->n_levels && u > c->bounds[k]) k++;
        code_set(c, codes, i, k);
        cc += (double)c->centroids[k] * c->centroids[k];
    }
    /* Length-preserving scale: ||s c[i]|| == ||x||. */
    const float s = (float)(norm / sqrt(cc > 0.0 ? cc : 1.0));
    const uint16_t h = oc_f32_to_f16_bits(s);
    memcpy(blk, &h, 2);
}

void oc_kvrq_decode(const OcKvRqCodec *c, const uint8_t *blk, float *xr)
{
    const float s = oc_f16_to_f32_bits(load_u16(blk));
    for (size_t i = 0; i < c->d; i++)
        xr[i] = s * c->centroids[code_get(c, blk + 2, i)];
}

void oc_kvq8_encode_row(const float *x, size_t d, int8_t *codes, float *scale)
{
    float amax = 0.0f;
    for (size_t i = 0; i < d; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f || !isfinite(amax)) {
        memset(codes, 0, d);
        *scale = 0.0f;
        return;
    }
    const float sc = amax / 127.0f, inv = 127.0f / amax;
    for (size_t i = 0; i < d; i++) {
        float q = roundf(x[i] * inv);
        if (q > 127.0f) q = 127.0f;
        if (q < -127.0f) q = -127.0f;
        codes[i] = (int8_t)q;
    }
    *scale = sc;
}

/* ─── Scalar reference kernels ────────────────────────────────────────── */

static void rq_score_scalar(const OcKvRqCodec *c, const uint8_t *blocks,
                            size_t n, const float *q, size_t G, float *scores,
                            size_t ss)
{
    float row[OC_KVRQ_DIM_MAX];
    for (size_t t = 0; t < n; t++) {
        oc_kvrq_decode(c, blocks + t * c->block_bytes, row);
        for (size_t g = 0; g < G; g++) {
            float a = 0.0f;
            for (size_t i = 0; i < c->d; i++) a += q[g * c->d + i] * row[i];
            scores[g * ss + t] = a;
        }
    }
}

static void rq_accum_scalar(const OcKvRqCodec *c, const uint8_t *blocks,
                            size_t n, const float *w, size_t ws, size_t G,
                            float *acc)
{
    float row[OC_KVRQ_DIM_MAX];
    for (size_t t = 0; t < n; t++) {
        oc_kvrq_decode(c, blocks + t * c->block_bytes, row);
        for (size_t g = 0; g < G; g++) {
            const float wt = w[g * ws + t];
            if (wt == 0.0f) continue;
            for (size_t i = 0; i < c->d; i++) acc[g * c->d + i] += wt * row[i];
        }
    }
}

void oc_kvrq_prep_q(const float *q, size_t G, size_t d, int8_t *q8,
                    float *qscale, int32_t *qsum)
{
    for (size_t g = 0; g < G; g++) {
        const float *x = q + g * d;
        int8_t *o = q8 + g * d;
        float amax = 0.0f;
        for (size_t i = 0; i < d; i++)
            if (fabsf(x[i]) > amax) amax = fabsf(x[i]);
        int32_t sum = 0;
        if (!(amax > 0.0f) || !isfinite(amax)) {
            memset(o, 0, d);
            qscale[g] = 0.0f;
        } else {
            const float inv = 127.0f / amax;
            for (size_t i = 0; i < d; i++) {
                long v = lrintf(x[i] * inv);
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                o[i] = (int8_t)v;
                sum += (int32_t)v;
            }
            qscale[g] = amax / 127.0f;
        }
        qsum[g] = sum;
    }
}

static void rq_score_i8_scalar(const OcKvRqCodec *c, const uint8_t *blocks,
                               size_t n, const int8_t *q8,
                               const float *qscale, const int32_t *qsum,
                               size_t G, float *scores, size_t ss)
{
    const size_t d = c->d;
    for (size_t t = 0; t < n; t++) {
        const uint8_t *blk = blocks + t * c->block_bytes;
        const float s = oc_f16_to_f32_bits(load_u16(blk));
        for (size_t g = 0; g < G; g++) {
            int32_t a = 0;
            for (size_t i = 0; i < d; i++)
                a += (int32_t)c->ciu[code_get(c, blk + 2, i)] *
                     (int32_t)q8[g * d + i];
            a -= 64 * qsum[g];
            scores[g * ss + t] = ((float)a * (qscale[g] * c->cscale)) * s;
        }
    }
}

static void q8_score_scalar(const int8_t *codes, size_t cs,
                            const float *scales, size_t sst, size_t d,
                            size_t n, const float *q, size_t G, float *scores,
                            size_t ss)
{
    for (size_t t = 0; t < n; t++) {
        const int8_t *r = codes + t * cs;
        for (size_t g = 0; g < G; g++) {
            float a = 0.0f;
            for (size_t i = 0; i < d; i++) a += q[g * d + i] * (float)r[i];
            scores[g * ss + t] = a * scales[t * sst];
        }
    }
}

static void q8_accum_scalar(const int8_t *codes, size_t cs,
                            const float *scales, size_t sst, size_t d,
                            size_t n, const float *w, size_t ws, size_t G,
                            float *acc)
{
    for (size_t t = 0; t < n; t++) {
        const int8_t *r = codes + t * cs;
        for (size_t g = 0; g < G; g++) {
            const float wt = w[g * ws + t] * scales[t * sst];
            if (wt == 0.0f) continue;
            for (size_t i = 0; i < d; i++) acc[g * d + i] += wt * (float)r[i];
        }
    }
}

static void f32_score_scalar(const float *x, size_t xs, size_t d, size_t n,
                             const float *q, size_t G, float *scores,
                             size_t ss)
{
    for (size_t t = 0; t < n; t++) {
        const float *r = x + t * xs;
        for (size_t g = 0; g < G; g++) {
            float a = 0.0f;
            for (size_t i = 0; i < d; i++) a += q[g * d + i] * r[i];
            scores[g * ss + t] = a;
        }
    }
}

static void f32_accum_scalar(const float *x, size_t xs, size_t d, size_t n,
                             const float *w, size_t ws, size_t G, float *acc)
{
    for (size_t t = 0; t < n; t++) {
        const float *r = x + t * xs;
        for (size_t g = 0; g < G; g++) {
            const float wt = w[g * ws + t];
            if (wt == 0.0f) continue;
            for (size_t i = 0; i < d; i++) acc[g * d + i] += wt * r[i];
        }
    }
}

/* ─── AVX2 kernels ────────────────────────────────────────────────────── */

#if defined(__x86_64__) || defined(__i386__)

#define KVRQ_TGT __attribute__((target("avx2,fma,f16c"), always_inline)) inline

static KVRQ_TGT __m256i load8_u8(const uint8_t *p)
{
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)p));
}

/* Eight code indices for coordinates 8m..8m+7. `bits` is a literal at every
 * call site, so each specialization folds to a load + shift + mask. */
static KVRQ_TGT __m256i rq_idx8(const uint8_t *codes, size_t d, unsigned bits,
                                size_t m)
{
    const size_t e = 8 * m;
    if (bits == 2) {
        const size_t q = d / 4;
        __m256i b = load8_u8(codes + e % q);
        b = _mm256_srlv_epi32(b, _mm256_set1_epi32((int)(2 * (e / q))));
        return _mm256_and_si256(b, _mm256_set1_epi32(3));
    }
    if (bits == 3) {
        const size_t q = d / 4, o = d / 8;
        __m256i lo = load8_u8(codes + e % q);
        lo = _mm256_and_si256(
            _mm256_srlv_epi32(lo, _mm256_set1_epi32((int)(2 * (e / q)))),
            _mm256_set1_epi32(3));
        __m256i hi = load8_u8(codes + q + e % o);
        hi = _mm256_and_si256(
            _mm256_srlv_epi32(hi, _mm256_set1_epi32((int)(e / o))),
            _mm256_set1_epi32(1));
        return _mm256_or_si256(lo, _mm256_slli_epi32(hi, 2));
    }
    {
        const size_t h = d / 2;
        __m256i b = load8_u8(codes + e % h);
        b = _mm256_srlv_epi32(b, _mm256_set1_epi32((int)(4 * (e / h))));
        return _mm256_and_si256(b, _mm256_set1_epi32(15));
    }
}

static KVRQ_TGT __m256 rq_lut(__m256i idx, __m256 c_lo, __m256 c_hi,
                              unsigned bits)
{
    __m256 lo = _mm256_permutevar8x32_ps(c_lo, idx);
    if (bits < 4) return lo;
    __m256 hi = _mm256_permutevar8x32_ps(c_hi, idx);
    return _mm256_blendv_ps(lo, hi,
                            _mm256_castsi256_ps(_mm256_slli_epi32(idx, 28)));
}

static KVRQ_TGT float hsum8(__m256 v)
{
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v),
                          _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/* Horizontal sums of four vectors -> one __m128 {sum a, sum b, sum c, sum d}. */
static KVRQ_TGT __m128 hsum4x8(__m256 a, __m256 b, __m256 c, __m256 d)
{
    __m256 ab = _mm256_hadd_ps(a, b);
    __m256 cd = _mm256_hadd_ps(c, d);
    __m256 abcd = _mm256_hadd_ps(ab, cd);
    return _mm_add_ps(_mm256_castps256_ps128(abcd),
                      _mm256_extractf128_ps(abcd, 1));
}

/* Scores for G heads (in groups of 4) against n blocks. Two positions per
 * iteration: eight independent FMA chains keep both FMA pipes busy (four
 * chains of 16 dependent FMAs per position would be latency-bound). */
static KVRQ_TGT void rq_score_avx2_b(const OcKvRqCodec *c,
                                     const uint8_t *blocks, size_t n,
                                     const float *q, size_t G, float *scores,
                                     size_t ss, const unsigned bits,
                                     const size_t d)
{
    const size_t bb = 2 + d * bits / 8, nm = d / 8;
    const __m256 c_lo = _mm256_loadu_ps(c->centroids);
    const __m256 c_hi = _mm256_loadu_ps(c->centroids + 8);
    for (size_t g0 = 0; g0 < G; g0 += 4) {
        const size_t gc = G - g0 < 4 ? G - g0 : 4;
        const float *q0 = q + (g0 + 0) * d;
        const float *q1 = q + (g0 + (gc > 1 ? 1 : 0)) * d;
        const float *q2 = q + (g0 + (gc > 2 ? 2 : 0)) * d;
        const float *q3 = q + (g0 + (gc > 3 ? 3 : 0)) * d;
        size_t t = 0;
        for (; t + 2 <= n; t += 2) {
            const uint8_t *b0 = blocks + t * bb, *b1 = b0 + bb;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            __m256 e0 = _mm256_setzero_ps(), e1 = _mm256_setzero_ps();
            __m256 e2 = _mm256_setzero_ps(), e3 = _mm256_setzero_ps();
            for (size_t m = 0; m < nm; m++) {
                const __m256 v0 = rq_lut(rq_idx8(b0 + 2, d, bits, m), c_lo,
                                         c_hi, bits);
                const __m256 v1 = rq_lut(rq_idx8(b1 + 2, d, bits, m), c_lo,
                                         c_hi, bits);
                __m256 qv = _mm256_loadu_ps(q0 + 8 * m);
                a0 = _mm256_fmadd_ps(v0, qv, a0); e0 = _mm256_fmadd_ps(v1, qv, e0);
                qv = _mm256_loadu_ps(q1 + 8 * m);
                a1 = _mm256_fmadd_ps(v0, qv, a1); e1 = _mm256_fmadd_ps(v1, qv, e1);
                qv = _mm256_loadu_ps(q2 + 8 * m);
                a2 = _mm256_fmadd_ps(v0, qv, a2); e2 = _mm256_fmadd_ps(v1, qv, e2);
                qv = _mm256_loadu_ps(q3 + 8 * m);
                a3 = _mm256_fmadd_ps(v0, qv, a3); e3 = _mm256_fmadd_ps(v1, qv, e3);
            }
            float r0[4], r1[4];
            _mm_storeu_ps(r0, _mm_mul_ps(hsum4x8(a0, a1, a2, a3),
                                         _mm_set1_ps(_cvtsh_ss(load_u16(b0)))));
            _mm_storeu_ps(r1, _mm_mul_ps(hsum4x8(e0, e1, e2, e3),
                                         _mm_set1_ps(_cvtsh_ss(load_u16(b1)))));
            for (size_t g = 0; g < gc; g++) {
                scores[(g0 + g) * ss + t] = r0[g];
                scores[(g0 + g) * ss + t + 1] = r1[g];
            }
        }
        for (; t < n; t++) {
            const uint8_t *blk = blocks + t * bb;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            for (size_t m = 0; m < nm; m++) {
                const __m256 cv = rq_lut(rq_idx8(blk + 2, d, bits, m), c_lo,
                                         c_hi, bits);
                a0 = _mm256_fmadd_ps(cv, _mm256_loadu_ps(q0 + 8 * m), a0);
                a1 = _mm256_fmadd_ps(cv, _mm256_loadu_ps(q1 + 8 * m), a1);
                a2 = _mm256_fmadd_ps(cv, _mm256_loadu_ps(q2 + 8 * m), a2);
                a3 = _mm256_fmadd_ps(cv, _mm256_loadu_ps(q3 + 8 * m), a3);
            }
            float r[4];
            _mm_storeu_ps(r, _mm_mul_ps(hsum4x8(a0, a1, a2, a3),
                                        _mm_set1_ps(_cvtsh_ss(load_u16(blk)))));
            for (size_t g = 0; g < gc; g++) scores[(g0 + g) * ss + t] = r[g];
        }
    }
}

#define KVRQ_CHUNK 64u

/* acc_g += sum_t w_gt s_t c[idx_t]. Accumulators split by position parity
 * (eight chains) for the same latency reason as the score kernel. */
static KVRQ_TGT void rq_accum_avx2_b(const OcKvRqCodec *c,
                                     const uint8_t *blocks, size_t n,
                                     const float *w, size_t ws, size_t G,
                                     float *acc, const unsigned bits,
                                     const size_t d)
{
    const size_t bb = 2 + d * bits / 8, nm = d / 8;
    const __m256 c_lo = _mm256_loadu_ps(c->centroids);
    const __m256 c_hi = _mm256_loadu_ps(c->centroids + 8);
    float wsc[4][KVRQ_CHUNK + 1];
    for (size_t t0 = 0; t0 < n; t0 += KVRQ_CHUNK) {
        const size_t tn = n - t0 < KVRQ_CHUNK ? n - t0 : KVRQ_CHUNK;
        const uint8_t *bl = blocks + t0 * bb;
        for (size_t g0 = 0; g0 < G; g0 += 4) {
            const size_t gc = G - g0 < 4 ? G - g0 : 4;
            for (size_t t = 0; t < tn; t++) {
                const float s = _cvtsh_ss(load_u16(bl + t * bb));
                for (size_t g = 0; g < 4; g++)
                    wsc[g][t] = g < gc ? w[(g0 + g) * ws + t0 + t] * s : 0.0f;
            }
            float *o0 = acc + g0 * d;
            for (size_t m = 0; m < nm; m++) {
                __m256 a0 = _mm256_loadu_ps(o0 + 8 * m);
                __m256 a1 = gc > 1 ? _mm256_loadu_ps(o0 + d + 8 * m)
                                   : _mm256_setzero_ps();
                __m256 a2 = gc > 2 ? _mm256_loadu_ps(o0 + 2 * d + 8 * m)
                                   : _mm256_setzero_ps();
                __m256 a3 = gc > 3 ? _mm256_loadu_ps(o0 + 3 * d + 8 * m)
                                   : _mm256_setzero_ps();
                __m256 e0 = _mm256_setzero_ps(), e1 = _mm256_setzero_ps();
                __m256 e2 = _mm256_setzero_ps(), e3 = _mm256_setzero_ps();
                size_t t = 0;
                for (; t + 2 <= tn; t += 2) {
                    const __m256 v0 = rq_lut(rq_idx8(bl + t * bb + 2, d, bits,
                                                     m), c_lo, c_hi, bits);
                    const __m256 v1 = rq_lut(rq_idx8(bl + (t + 1) * bb + 2, d,
                                                     bits, m), c_lo, c_hi, bits);
                    a0 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(&wsc[0][t]), a0);
                    e0 = _mm256_fmadd_ps(v1, _mm256_broadcast_ss(&wsc[0][t + 1]), e0);
                    a1 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(&wsc[1][t]), a1);
                    e1 = _mm256_fmadd_ps(v1, _mm256_broadcast_ss(&wsc[1][t + 1]), e1);
                    a2 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(&wsc[2][t]), a2);
                    e2 = _mm256_fmadd_ps(v1, _mm256_broadcast_ss(&wsc[2][t + 1]), e2);
                    a3 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(&wsc[3][t]), a3);
                    e3 = _mm256_fmadd_ps(v1, _mm256_broadcast_ss(&wsc[3][t + 1]), e3);
                }
                if (t < tn) {
                    const __m256 v0 = rq_lut(rq_idx8(bl + t * bb + 2, d, bits,
                                                     m), c_lo, c_hi, bits);
                    a0 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(&wsc[0][t]), a0);
                    a1 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(&wsc[1][t]), a1);
                    a2 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(&wsc[2][t]), a2);
                    a3 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(&wsc[3][t]), a3);
                }
                _mm256_storeu_ps(o0 + 8 * m, _mm256_add_ps(a0, e0));
                if (gc > 1) _mm256_storeu_ps(o0 + d + 8 * m, _mm256_add_ps(a1, e1));
                if (gc > 2) _mm256_storeu_ps(o0 + 2 * d + 8 * m, _mm256_add_ps(a2, e2));
                if (gc > 3) _mm256_storeu_ps(o0 + 3 * d + 8 * m, _mm256_add_ps(a3, e3));
            }
        }
    }
}

static KVRQ_TGT void rq_decode_rows_avx2_b(const OcKvRqCodec *c,
                                           const uint8_t *blocks, size_t n,
                                           float *rows, const unsigned bits,
                                           const size_t d)
{
    const size_t bb = 2 + d * bits / 8, nm = d / 8;
    const __m256 c_lo = _mm256_loadu_ps(c->centroids);
    const __m256 c_hi = _mm256_loadu_ps(c->centroids + 8);
    for (size_t t = 0; t < n; t++) {
        const uint8_t *blk = blocks + t * bb;
        const __m256 s = _mm256_set1_ps(_cvtsh_ss(load_u16(blk)));
        for (size_t m = 0; m < nm; m++)
            _mm256_storeu_ps(rows + t * d + 8 * m,
                             _mm256_mul_ps(s, rq_lut(rq_idx8(blk + 2, d, bits,
                                                             m),
                                                     c_lo, c_hi, bits)));
    }
}

static KVRQ_TGT __m256 load8_i8f(const int8_t *p)
{
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
        _mm_loadl_epi64((const __m128i *)(const void *)p)));
}

/* 32 code indices (one per byte) for coordinates 32v..32v+31, d >= 128.
 * Every plane is laid out so these are 32 consecutive bytes (or, for the
 * 3-bit high plane at d = 128, one 16-byte row seen by both lanes). */
static KVRQ_TGT __m256i rq_idx32(const uint8_t *codes, size_t d,
                                 unsigned bits, size_t v)
{
    const size_t e = 32 * v;
    if (bits == 4) {
        const size_t h = d / 2;
        const __m256i x = _mm256_loadu_si256((const __m256i *)(const void *)
                                             (codes + e % h));
        return _mm256_and_si256(
            _mm256_srli_epi16(x, (int)(4 * (e / h))), _mm256_set1_epi8(15));
    }
    const size_t q = d / 4;
    const __m256i x = _mm256_loadu_si256((const __m256i *)(const void *)
                                         (codes + e % q));
    const __m256i lo = _mm256_and_si256(
        _mm256_srli_epi16(x, (int)(2 * (e / q))), _mm256_set1_epi8(3));
    if (bits == 2) return lo;
    const size_t o = d / 8;
    __m256i hb;
    if (o >= 32) {
        const __m256i y = _mm256_loadu_si256((const __m256i *)(const void *)
                                             (codes + q + e % o));
        hb = _mm256_srli_epi16(y, (int)(e / o));
    } else {
        /* o == 16: byte j bit k holds coordinate j + 16k; this vector
         * needs bit 2v in the low lane and bit 2v+1 in the high lane. */
        const __m256i y = _mm256_broadcastsi128_si256(
            _mm_loadu_si128((const __m128i *)(const void *)(codes + q)));
        hb = _mm256_srlv_epi32(y, _mm256_setr_epi32(
            (int)(2 * v), (int)(2 * v), (int)(2 * v), (int)(2 * v),
            (int)(2 * v + 1), (int)(2 * v + 1), (int)(2 * v + 1),
            (int)(2 * v + 1)));
    }
    hb = _mm256_and_si256(hb, _mm256_set1_epi8(1));
    return _mm256_or_si256(lo, _mm256_slli_epi16(hb, 2));
}

/* Integer scores for G heads (groups of 4) against n blocks. The centroid
 * lookup is one pshufb per 32 coordinates; the dot product runs on
 * maddubs (u8 ci+64 x s8 q) -> madd -> int32, twice the MAC rate of f32
 * FMA and exact. */
static KVRQ_TGT void rq_score_i8_avx2_b(const OcKvRqCodec *c,
                                        const uint8_t *blocks, size_t n,
                                        const int8_t *q8, const float *qscale,
                                        const int32_t *qsum, size_t G,
                                        float *scores, size_t ss,
                                        const unsigned bits, const size_t d)
{
    const size_t bb = 2 + d * bits / 8, nv = d / 32;
    const __m256i lut = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i *)(const void *)c->ciu));
    const __m256i ones = _mm256_set1_epi16(1);
    for (size_t g0 = 0; g0 < G; g0 += 4) {
        const size_t gc = G - g0 < 4 ? G - g0 : 4;
        size_t gi[4];
        for (size_t g = 0; g < 4; g++) gi[g] = g0 + (g < gc ? g : 0);
        const int8_t *q0 = q8 + gi[0] * d, *q1 = q8 + gi[1] * d;
        const int8_t *q2 = q8 + gi[2] * d, *q3 = q8 + gi[3] * d;
        const __m128 qs = _mm_setr_ps(qscale[gi[0]] * c->cscale,
                                      qscale[gi[1]] * c->cscale,
                                      qscale[gi[2]] * c->cscale,
                                      qscale[gi[3]] * c->cscale);
        const __m128i qb = _mm_setr_epi32(64 * qsum[gi[0]], 64 * qsum[gi[1]],
                                          64 * qsum[gi[2]], 64 * qsum[gi[3]]);
        for (size_t t = 0; t < n; t++) {
            const uint8_t *blk = blocks + t * bb;
            __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
            __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
            for (size_t v = 0; v < nv; v++) {
                const __m256i u = _mm256_shuffle_epi8(
                    lut, rq_idx32(blk + 2, d, bits, v));
#define RQ_DOT_(A, Q)                                                         \
    A = _mm256_add_epi32(A, _mm256_madd_epi16(_mm256_maddubs_epi16(          \
            u, _mm256_loadu_si256((const __m256i *)(const void *)            \
                                  ((Q) + 32 * v))), ones))
                RQ_DOT_(a0, q0); RQ_DOT_(a1, q1); RQ_DOT_(a2, q2); RQ_DOT_(a3, q3);
#undef RQ_DOT_
            }
            const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(a0, a1),
                                                _mm256_hadd_epi32(a2, a3));
            const __m128i sum = _mm_sub_epi32(
                _mm_add_epi32(_mm256_castsi256_si128(h),
                              _mm256_extracti128_si256(h, 1)), qb);
            float r[4];
            _mm_storeu_ps(r, _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum), qs),
                                        _mm_set1_ps(_cvtsh_ss(load_u16(blk)))));
            for (size_t g = 0; g < gc; g++) scores[(g0 + g) * ss + t] = r[g];
        }
    }
}

/* ci values of n blocks as int8 rows (row t at out + t*d). */
static KVRQ_TGT void rq_unpack_i8_b(const OcKvRqCodec *c,
                                    const uint8_t *blocks, size_t n,
                                    int8_t *out, const unsigned bits,
                                    const size_t d)
{
    const size_t bb = 2 + d * bits / 8, nv = d / 32;
    const __m256i lut = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i *)(const void *)c->ci));
    for (size_t t = 0; t < n; t++)
        for (size_t v = 0; v < nv; v++)
            _mm256_storeu_si256(
                (__m256i *)(void *)(out + t * d + 32 * v),
                _mm256_shuffle_epi8(lut, rq_idx32(blocks + t * bb + 2, d,
                                                  bits, v)));
}

/* o_g += sum_{t<n} W[g][t] * rows[t], g < 4 (rows of d int8, stride rs;
 * rows of unused heads have W == 0 and are not stored). n <= KVRQ_CHUNK.
 * Weights come from memory as broadcasts, so the FP pipes only see the
 * int8->f32 conversion and the FMAs. */
static KVRQ_TGT void accum4_i8(const int8_t *rows, size_t rs, size_t d,
                               size_t n, const float *W, size_t wst,
                               size_t gc, float *o0)
{
    const size_t nm = d / 8;
    const float *w0 = W, *w1 = W + wst, *w2 = W + 2 * wst, *w3 = W + 3 * wst;
    for (size_t m = 0; m < nm; m++) {
        __m256 a0 = _mm256_loadu_ps(o0 + 8 * m);
        __m256 a1 = gc > 1 ? _mm256_loadu_ps(o0 + d + 8 * m) : _mm256_setzero_ps();
        __m256 a2 = gc > 2 ? _mm256_loadu_ps(o0 + 2 * d + 8 * m) : _mm256_setzero_ps();
        __m256 a3 = gc > 3 ? _mm256_loadu_ps(o0 + 3 * d + 8 * m) : _mm256_setzero_ps();
        __m256 e0 = _mm256_setzero_ps(), e1 = _mm256_setzero_ps();
        __m256 e2 = _mm256_setzero_ps(), e3 = _mm256_setzero_ps();
        size_t t = 0;
        for (; t + 2 <= n; t += 2) {
            const __m256 v0 = load8_i8f(rows + t * rs + 8 * m);
            const __m256 v1 = load8_i8f(rows + (t + 1) * rs + 8 * m);
            a0 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(w0 + t), a0);
            e0 = _mm256_fmadd_ps(v1, _mm256_broadcast_ss(w0 + t + 1), e0);
            a1 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(w1 + t), a1);
            e1 = _mm256_fmadd_ps(v1, _mm256_broadcast_ss(w1 + t + 1), e1);
            a2 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(w2 + t), a2);
            e2 = _mm256_fmadd_ps(v1, _mm256_broadcast_ss(w2 + t + 1), e2);
            a3 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(w3 + t), a3);
            e3 = _mm256_fmadd_ps(v1, _mm256_broadcast_ss(w3 + t + 1), e3);
        }
        if (t < n) {
            const __m256 v0 = load8_i8f(rows + t * rs + 8 * m);
            a0 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(w0 + t), a0);
            a1 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(w1 + t), a1);
            a2 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(w2 + t), a2);
            a3 = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(w3 + t), a3);
        }
        _mm256_storeu_ps(o0 + 8 * m, _mm256_add_ps(a0, e0));
        if (gc > 1) _mm256_storeu_ps(o0 + d + 8 * m, _mm256_add_ps(a1, e1));
        if (gc > 2) _mm256_storeu_ps(o0 + 2 * d + 8 * m, _mm256_add_ps(a2, e2));
        if (gc > 3) _mm256_storeu_ps(o0 + 3 * d + 8 * m, _mm256_add_ps(a3, e3));
    }
}

/* V accumulate: unpack a chunk of blocks to int8 ci rows once, fold the
 * per-block scale and cscale into the weights, then accum4_i8. */
static KVRQ_TGT void rq_accum_i8_avx2_b(const OcKvRqCodec *c,
                                        const uint8_t *blocks, size_t n,
                                        const float *w, size_t ws, size_t G,
                                        float *acc, const unsigned bits,
                                        const size_t d)
{
    const size_t bb = 2 + d * bits / 8;
    int8_t rows[KVRQ_CHUNK * OC_KVRQ_DIM_MAX] __attribute__((aligned(32)));
    float sc[KVRQ_CHUNK];
    float W[4][KVRQ_CHUNK];
    for (size_t t0 = 0; t0 < n; t0 += KVRQ_CHUNK) {
        const size_t tn = n - t0 < KVRQ_CHUNK ? n - t0 : KVRQ_CHUNK;
        const uint8_t *bl = blocks + t0 * bb;
        rq_unpack_i8_b(c, bl, tn, rows, bits, d);
        for (size_t t = 0; t < tn; t++)
            sc[t] = _cvtsh_ss(load_u16(bl + t * bb)) * c->cscale;
        for (size_t g0 = 0; g0 < G; g0 += 4) {
            const size_t gc = G - g0 < 4 ? G - g0 : 4;
            for (size_t g = 0; g < 4; g++)
                for (size_t t = 0; t < tn; t++)
                    W[g][t] = g < gc ? w[(g0 + g) * ws + t0 + t] * sc[t] : 0.0f;
            accum4_i8(rows, d, d, tn, &W[0][0], KVRQ_CHUNK, gc, acc + g0 * d);
        }
    }
}

/* Specialize on (bits, d) so every offset/shift in rq_idx8 folds to a
 * constant: a runtime d turns them into integer divisions per group. */
#define KVRQ_SPECIALIZE(CALL)                                              \
    switch (c->d * 8 + c->bits) {                                          \
    case 128 * 8 + 2: CALL(2, 128); break;                                 \
    case 128 * 8 + 3: CALL(3, 128); break;                                 \
    case 128 * 8 + 4: CALL(4, 128); break;                                 \
    case 64 * 8 + 2: CALL(2, 64); break;                                   \
    case 64 * 8 + 3: CALL(3, 64); break;                                   \
    case 64 * 8 + 4: CALL(4, 64); break;                                   \
    case 256 * 8 + 2: CALL(2, 256); break;                                 \
    case 256 * 8 + 3: CALL(3, 256); break;                                 \
    default: CALL(4, 256); break;                                          \
    }

__attribute__((target("avx2,fma,f16c")))
static void rq_score_avx2(const OcKvRqCodec *c, const uint8_t *b, size_t n,
                          const float *q, size_t G, float *s, size_t ss)
{
#define C_(B, D) rq_score_avx2_b(c, b, n, q, G, s, ss, B, D)
    KVRQ_SPECIALIZE(C_)
#undef C_
}

__attribute__((target("avx2,fma,f16c")))
static void rq_accum_avx2(const OcKvRqCodec *c, const uint8_t *b, size_t n,
                          const float *w, size_t ws, size_t G, float *acc)
{
#define C_(B, D) rq_accum_avx2_b(c, b, n, w, ws, G, acc, B, D)
    KVRQ_SPECIALIZE(C_)
#undef C_
}

#define KVRQ_SPECIALIZE_WIDE(CALL)                                         \
    switch (c->d * 8 + c->bits) {                                          \
    case 128 * 8 + 2: CALL(2, 128); break;                                 \
    case 128 * 8 + 3: CALL(3, 128); break;                                 \
    case 128 * 8 + 4: CALL(4, 128); break;                                 \
    case 256 * 8 + 2: CALL(2, 256); break;                                 \
    case 256 * 8 + 3: CALL(3, 256); break;                                 \
    default: CALL(4, 256); break;                                          \
    }

__attribute__((target("avx2,fma,f16c")))
static void rq_score_i8_avx2(const OcKvRqCodec *c, const uint8_t *b, size_t n,
                             const int8_t *q8, const float *qscale,
                             const int32_t *qsum, size_t G, float *s,
                             size_t ss)
{
#define C_(B, D) rq_score_i8_avx2_b(c, b, n, q8, qscale, qsum, G, s, ss, B, D)
    KVRQ_SPECIALIZE_WIDE(C_)
#undef C_
}

__attribute__((target("avx2,fma,f16c")))
static void rq_accum_i8_avx2(const OcKvRqCodec *c, const uint8_t *b, size_t n,
                             const float *w, size_t ws, size_t G, float *acc)
{
#define C_(B, D) rq_accum_i8_avx2_b(c, b, n, w, ws, G, acc, B, D)
    KVRQ_SPECIALIZE_WIDE(C_)
#undef C_
}

__attribute__((target("avx2,fma,f16c")))
static void rq_decode_rows_avx2(const OcKvRqCodec *c, const uint8_t *b,
                                size_t n, float *rows)
{
#define C_(B, D) rq_decode_rows_avx2_b(c, b, n, rows, B, D)
    KVRQ_SPECIALIZE(C_)
#undef C_
}


/* Generic two-position score / parity-split accumulate for rows that
 * load as eight floats per 8-dim group (int8 codes or f32). ROW(t, m)
 * yields the __m256 for row t, coordinates 8m..8m+7. */
#define DENSE_SCORE_BODY(ROW, SCALE)                                          \
    const size_t nm = d / 8;                                                  \
    for (size_t g0 = 0; g0 < G; g0 += 4) {                                    \
        const size_t gc = G - g0 < 4 ? G - g0 : 4;                            \
        const float *q0 = q + (g0 + 0) * d;                                   \
        const float *q1 = q + (g0 + (gc > 1 ? 1 : 0)) * d;                    \
        const float *q2 = q + (g0 + (gc > 2 ? 2 : 0)) * d;                    \
        const float *q3 = q + (g0 + (gc > 3 ? 3 : 0)) * d;                    \
        size_t t = 0;                                                         \
        for (; t + 2 <= n; t += 2) {                                          \
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();        \
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();        \
            __m256 e0 = _mm256_setzero_ps(), e1 = _mm256_setzero_ps();        \
            __m256 e2 = _mm256_setzero_ps(), e3 = _mm256_setzero_ps();        \
            for (size_t m = 0; m < nm; m++) {                                 \
                const __m256 v0 = ROW(t, m), v1 = ROW(t + 1, m);              \
                __m256 qv = _mm256_loadu_ps(q0 + 8 * m);                      \
                a0 = _mm256_fmadd_ps(v0, qv, a0); e0 = _mm256_fmadd_ps(v1, qv, e0); \
                qv = _mm256_loadu_ps(q1 + 8 * m);                             \
                a1 = _mm256_fmadd_ps(v0, qv, a1); e1 = _mm256_fmadd_ps(v1, qv, e1); \
                qv = _mm256_loadu_ps(q2 + 8 * m);                             \
                a2 = _mm256_fmadd_ps(v0, qv, a2); e2 = _mm256_fmadd_ps(v1, qv, e2); \
                qv = _mm256_loadu_ps(q3 + 8 * m);                             \
                a3 = _mm256_fmadd_ps(v0, qv, a3); e3 = _mm256_fmadd_ps(v1, qv, e3); \
            }                                                                 \
            float r0[4], r1[4];                                               \
            _mm_storeu_ps(r0, _mm_mul_ps(hsum4x8(a0, a1, a2, a3),             \
                                         _mm_set1_ps(SCALE(t))));             \
            _mm_storeu_ps(r1, _mm_mul_ps(hsum4x8(e0, e1, e2, e3),             \
                                         _mm_set1_ps(SCALE(t + 1))));         \
            for (size_t g = 0; g < gc; g++) {                                 \
                scores[(g0 + g) * ss + t] = r0[g];                            \
                scores[(g0 + g) * ss + t + 1] = r1[g];                        \
            }                                                                 \
        }                                                                     \
        for (; t < n; t++) {                                                  \
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();        \
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();        \
            for (size_t m = 0; m < nm; m++) {                                 \
                const __m256 v0 = ROW(t, m);                                  \
                a0 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(q0 + 8 * m), a0);    \
                a1 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(q1 + 8 * m), a1);    \
                a2 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(q2 + 8 * m), a2);    \
                a3 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(q3 + 8 * m), a3);    \
            }                                                                 \
            float r[4];                                                       \
            _mm_storeu_ps(r, _mm_mul_ps(hsum4x8(a0, a1, a2, a3),              \
                                        _mm_set1_ps(SCALE(t))));              \
            for (size_t g = 0; g < gc; g++) scores[(g0 + g) * ss + t] = r[g]; \
        }                                                                     \
    }

#define DENSE_ACCUM_BODY(ROW, WT)                                             \
    const size_t nm = d / 8;                                                  \
    for (size_t g0 = 0; g0 < G; g0 += 4) {                                    \
        const size_t gc = G - g0 < 4 ? G - g0 : 4;                            \
        const float *w0 = w + (g0 + 0) * ws;                                  \
        const float *w1 = w + (g0 + (gc > 1 ? 1 : 0)) * ws;                   \
        const float *w2 = w + (g0 + (gc > 2 ? 2 : 0)) * ws;                   \
        const float *w3 = w + (g0 + (gc > 3 ? 3 : 0)) * ws;                   \
        float *o0 = acc + g0 * d;                                             \
        for (size_t m = 0; m < nm; m++) {                                     \
            __m256 a0 = _mm256_loadu_ps(o0 + 8 * m);                          \
            __m256 a1 = gc > 1 ? _mm256_loadu_ps(o0 + d + 8 * m)              \
                               : _mm256_setzero_ps();                         \
            __m256 a2 = gc > 2 ? _mm256_loadu_ps(o0 + 2 * d + 8 * m)          \
                               : _mm256_setzero_ps();                         \
            __m256 a3 = gc > 3 ? _mm256_loadu_ps(o0 + 3 * d + 8 * m)          \
                               : _mm256_setzero_ps();                         \
            __m256 e0 = _mm256_setzero_ps(), e1 = _mm256_setzero_ps();        \
            __m256 e2 = _mm256_setzero_ps(), e3 = _mm256_setzero_ps();        \
            size_t t = 0;                                                     \
            for (; t + 2 <= n; t += 2) {                                      \
                const __m256 v0 = ROW(t, m), v1 = ROW(t + 1, m);              \
                a0 = _mm256_fmadd_ps(v0, _mm256_set1_ps(WT(w0, t)), a0);      \
                e0 = _mm256_fmadd_ps(v1, _mm256_set1_ps(WT(w0, t + 1)), e0);  \
                a1 = _mm256_fmadd_ps(v0, _mm256_set1_ps(WT(w1, t)), a1);      \
                e1 = _mm256_fmadd_ps(v1, _mm256_set1_ps(WT(w1, t + 1)), e1);  \
                a2 = _mm256_fmadd_ps(v0, _mm256_set1_ps(WT(w2, t)), a2);      \
                e2 = _mm256_fmadd_ps(v1, _mm256_set1_ps(WT(w2, t + 1)), e2);  \
                a3 = _mm256_fmadd_ps(v0, _mm256_set1_ps(WT(w3, t)), a3);      \
                e3 = _mm256_fmadd_ps(v1, _mm256_set1_ps(WT(w3, t + 1)), e3);  \
            }                                                                 \
            if (t < n) {                                                      \
                const __m256 v0 = ROW(t, m);                                  \
                a0 = _mm256_fmadd_ps(v0, _mm256_set1_ps(WT(w0, t)), a0);      \
                a1 = _mm256_fmadd_ps(v0, _mm256_set1_ps(WT(w1, t)), a1);      \
                a2 = _mm256_fmadd_ps(v0, _mm256_set1_ps(WT(w2, t)), a2);      \
                a3 = _mm256_fmadd_ps(v0, _mm256_set1_ps(WT(w3, t)), a3);      \
            }                                                                 \
            _mm256_storeu_ps(o0 + 8 * m, _mm256_add_ps(a0, e0));              \
            if (gc > 1) _mm256_storeu_ps(o0 + d + 8 * m, _mm256_add_ps(a1, e1)); \
            if (gc > 2) _mm256_storeu_ps(o0 + 2 * d + 8 * m, _mm256_add_ps(a2, e2)); \
            if (gc > 3) _mm256_storeu_ps(o0 + 3 * d + 8 * m, _mm256_add_ps(a3, e3)); \
        }                                                                     \
    }

#define Q8_ROW(t, m) load8_i8f(codes + (t) * cs + 8 * (m))
#define Q8_SCALE(t) scales[(t) * sst]
#define F32_ROW(t, m) _mm256_loadu_ps(x + (t) * xs + 8 * (m))
#define F32_SCALE(t) 1.0f
#define F32_WT(wp, t) ((wp)[t])

__attribute__((target("avx2,fma,f16c")))
static void q8_score_avx2(const int8_t *codes, size_t cs, const float *scales,
                          size_t sst, size_t d, size_t n, const float *q,
                          size_t G, float *scores, size_t ss)
{
    DENSE_SCORE_BODY(Q8_ROW, Q8_SCALE)
}

__attribute__((target("avx2,fma,f16c")))
static void q8_accum_avx2(const int8_t *codes, size_t cs, const float *scales,
                          size_t sst, size_t d, size_t n, const float *w,
                          size_t ws, size_t G, float *acc)
{
    float W[4][KVRQ_CHUNK];
    for (size_t t0 = 0; t0 < n; t0 += KVRQ_CHUNK) {
        const size_t tn = n - t0 < KVRQ_CHUNK ? n - t0 : KVRQ_CHUNK;
        for (size_t g0 = 0; g0 < G; g0 += 4) {
            const size_t gc = G - g0 < 4 ? G - g0 : 4;
            for (size_t g = 0; g < 4; g++)
                for (size_t t = 0; t < tn; t++)
                    W[g][t] = g < gc ? w[(g0 + g) * ws + t0 + t] *
                                       scales[(t0 + t) * sst] : 0.0f;
            accum4_i8(codes + t0 * cs, cs, d, tn, &W[0][0], KVRQ_CHUNK, gc,
                      acc + g0 * d);
        }
    }
}

__attribute__((target("avx2,fma,f16c")))
static void f32_score_avx2(const float *x, size_t xs, size_t d, size_t n,
                           const float *q, size_t G, float *scores, size_t ss)
{
    DENSE_SCORE_BODY(F32_ROW, F32_SCALE)
}

__attribute__((target("avx2,fma,f16c")))
static void f32_accum_avx2(const float *x, size_t xs, size_t d, size_t n,
                           const float *w, size_t ws, size_t G, float *acc)
{
    DENSE_ACCUM_BODY(F32_ROW, F32_WT)
}

#define KVRQ_HAVE_AVX2 1
#else
#define KVRQ_HAVE_AVX2 0
#endif

/* ─── Dispatch ────────────────────────────────────────────────────────── */

void oc_kvrq_score(const OcKvRqCodec *c, const uint8_t *blocks, size_t n,
                   const float *q, size_t G, float *scores, size_t ss)
{
#if KVRQ_HAVE_AVX2
    if (kvrq_isa()) { rq_score_avx2(c, blocks, n, q, G, scores, ss); return; }
#endif
    rq_score_scalar(c, blocks, n, q, G, scores, ss);
}

void oc_kvrq_score_i8(const OcKvRqCodec *c, const uint8_t *blocks, size_t n,
                      const int8_t *q8, const float *qscale,
                      const int32_t *qsum, size_t G, float *scores, size_t ss)
{
#if KVRQ_HAVE_AVX2
    if (kvrq_isa() && c->d >= 128) {
        rq_score_i8_avx2(c, blocks, n, q8, qscale, qsum, G, scores, ss);
        return;
    }
#endif
    rq_score_i8_scalar(c, blocks, n, q8, qscale, qsum, G, scores, ss);
}

void oc_kvrq_accum(const OcKvRqCodec *c, const uint8_t *blocks, size_t n,
                   const float *w, size_t ws, size_t G, float *acc)
{
#if KVRQ_HAVE_AVX2
    if (kvrq_isa()) {
        if (c->d >= 128) rq_accum_i8_avx2(c, blocks, n, w, ws, G, acc);
        else rq_accum_avx2(c, blocks, n, w, ws, G, acc);
        return;
    }
#endif
    rq_accum_scalar(c, blocks, n, w, ws, G, acc);
}

void oc_kvrq_decode_rows(const OcKvRqCodec *c, const uint8_t *blocks,
                         size_t n, float *rows)
{
#if KVRQ_HAVE_AVX2
    if (kvrq_isa()) { rq_decode_rows_avx2(c, blocks, n, rows); return; }
#endif
    for (size_t t = 0; t < n; t++)
        oc_kvrq_decode(c, blocks + t * c->block_bytes, rows + t * c->d);
}

void oc_kvq8_score(const int8_t *codes, size_t cs, const float *scales,
                   size_t sst, size_t d, size_t n, const float *q, size_t G,
                   float *scores, size_t ss)
{
#if KVRQ_HAVE_AVX2
    if (kvrq_isa() && d % 8 == 0) {
        q8_score_avx2(codes, cs, scales, sst, d, n, q, G, scores, ss);
        return;
    }
#endif
    q8_score_scalar(codes, cs, scales, sst, d, n, q, G, scores, ss);
}

void oc_kvq8_accum(const int8_t *codes, size_t cs, const float *scales,
                   size_t sst, size_t d, size_t n, const float *w, size_t ws,
                   size_t G, float *acc)
{
#if KVRQ_HAVE_AVX2
    if (kvrq_isa() && d % 8 == 0) {
        q8_accum_avx2(codes, cs, scales, sst, d, n, w, ws, G, acc);
        return;
    }
#endif
    q8_accum_scalar(codes, cs, scales, sst, d, n, w, ws, G, acc);
}

void oc_kvq8_decode_rows(const int8_t *codes, size_t cs, const float *scales,
                         size_t sst, size_t d, size_t n, float *rows)
{
    for (size_t t = 0; t < n; t++) {
        const int8_t *r = codes + t * cs;
        const float s = scales[t * sst];
        float *o = rows + t * d;
        for (size_t i = 0; i < d; i++) o[i] = s * (float)r[i];
    }
}

void oc_kvf32_score(const float *x, size_t xs, size_t d, size_t n,
                    const float *q, size_t G, float *scores, size_t ss)
{
#if KVRQ_HAVE_AVX2
    if (kvrq_isa() && d % 8 == 0) {
        f32_score_avx2(x, xs, d, n, q, G, scores, ss);
        return;
    }
#endif
    f32_score_scalar(x, xs, d, n, q, G, scores, ss);
}

void oc_kvf32_accum(const float *x, size_t xs, size_t d, size_t n,
                    const float *w, size_t ws, size_t G, float *acc)
{
#if KVRQ_HAVE_AVX2
    if (kvrq_isa() && d % 8 == 0) {
        f32_accum_avx2(x, xs, d, n, w, ws, G, acc);
        return;
    }
#endif
    f32_accum_scalar(x, xs, d, n, w, ws, G, acc);
}

/* ─── Cache ───────────────────────────────────────────────────────────── */

static void *map_lazy(size_t bytes)
{
    if (bytes == 0) return NULL;
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return NULL;
#ifdef MADV_NOHUGEPAGE
    /* With THP "always", every one of the 2 x n_kv per-head streams of a
     * layer would fault in 2 MB at a time (1.5 GB for K2 before the first
     * token). Opt out so the cache grows 4 KB at a time with the context;
     * the streams are read sequentially, so small pages cost little. */
    if (getenv("OC_KVRQ_THP") == NULL) madvise(p, bytes, MADV_NOHUGEPAGE);
#endif
    return p;
}

static void unmap_lazy(void *p, size_t bytes)
{
    if (p != NULL && bytes != 0) munmap(p, bytes);
}

OcError oc_kvrq_cache_init(OcKvRqCache *c, size_t n_layers, size_t n_kv,
                           size_t d, size_t n_ctx, const OcKvRqParams *p)
{
    if (c == NULL || p == NULL || n_layers == 0 || n_kv == 0 || n_ctx == 0)
        return OC_ERR_INVALID_ARG;
    memset(c, 0, sizeof(*c));
    c->n_layers = n_layers;
    c->n_kv = n_kv;
    c->d = d;
    c->n_ctx = n_ctx;
    c->p = *p;
    OcError e = oc_kvrq_codec_init(&c->kc, d, p->k_bits);
    if (e == OC_OK) e = oc_kvrq_codec_init(&c->vc, d, p->v_bits);
    if (e == OC_OK) e = oc_kvrq_rot_init(&c->rot, d, p->rot, p->seed);
    if (e != OC_OK) {
        oc_kvrq_rot_free(&c->rot);
        memset(c, 0, sizeof(*c));
        return e;
    }
    c->layer_bytes = n_kv * n_ctx * (c->kc.block_bytes + c->vc.block_bytes);
    c->layer = calloc(n_layers, sizeof(*c->layer));
    if (c->layer == NULL) { oc_kvrq_cache_free(c); return OC_ERR_OOM; }
    for (size_t l = 0; l < n_layers; l++) {
        c->layer[l] = map_lazy(c->layer_bytes);
        if (c->layer[l] == NULL) { oc_kvrq_cache_free(c); return OC_ERR_OOM; }
    }
    c->n_slots = (size_t)p->n_sink + p->window;
    if (c->n_slots > 0) {
        c->xq_bytes = n_layers * 2u * n_kv * c->n_slots * d;
        c->xs_bytes = n_layers * 2u * n_kv * c->n_slots * sizeof(float);
        c->tag_bytes = n_layers * c->n_slots * sizeof(int64_t);
        c->xq = map_lazy(c->xq_bytes);
        c->xs = map_lazy(c->xs_bytes);
        c->tag = malloc(c->tag_bytes);
        if (c->xq == NULL || c->xs == NULL || c->tag == NULL) {
            oc_kvrq_cache_free(c);
            return OC_ERR_OOM;
        }
    }
    if (p->window > 0 && getenv("OC_KVRQ_NO_CENTER") == NULL) {
        c->page = p->window;
        c->n_pages = (n_ctx + c->page - 1) / c->page;
        c->mu_bytes = n_layers * c->n_pages * 2u * n_kv * d * sizeof(float);
        c->mu = map_lazy(c->mu_bytes);
        c->mu_fixed = calloc(n_layers * c->n_pages, 1);
        if (c->mu == NULL || c->mu_fixed == NULL) {
            oc_kvrq_cache_free(c);
            return OC_ERR_OOM;
        }
    }
    oc_kvrq_cache_clear(c);
    return OC_OK;
}

void oc_kvrq_cache_free(OcKvRqCache *c)
{
    if (c == NULL) return;
    if (c->layer != NULL) {
        for (size_t l = 0; l < c->n_layers; l++)
            unmap_lazy(c->layer[l], c->layer_bytes);
        free(c->layer);
    }
    unmap_lazy(c->xq, c->xq_bytes);
    unmap_lazy(c->xs, c->xs_bytes);
    free(c->tag);
    unmap_lazy(c->mu, c->mu_bytes);
    free(c->mu_fixed);
    oc_kvrq_rot_free(&c->rot);
    memset(c, 0, sizeof(*c));
}

void oc_kvrq_cache_clear(OcKvRqCache *c)
{
    if (c == NULL) return;
    if (c->tag != NULL)
        for (size_t i = 0; i < c->n_layers * c->n_slots; i++) c->tag[i] = -1;
    if (c->mu_fixed != NULL) memset(c->mu_fixed, 0, c->n_layers * c->n_pages);
}

void oc_kvrq_cache_rewind(OcKvRqCache *c, int64_t pos)
{
    if (c == NULL) return;
    if (c->tag != NULL)
        for (size_t i = 0; i < c->n_layers * c->n_slots; i++)
            if (c->tag[i] >= pos) c->tag[i] = -1;
    /* Pages that start at or after pos are gone; the page holding pos
     * keeps its mean (its older positions are encoded with it or still in
     * the ring, and the page re-encodes the ring with it when it fills). */
    if (c->mu_fixed != NULL && pos >= 0)
        for (size_t l = 0; l < c->n_layers; l++)
            for (size_t pg = 0; pg < c->n_pages; pg++)
                if ((int64_t)(pg * c->page) >= pos)
                    c->mu_fixed[l * c->n_pages + pg] = 0;
}

static uint8_t *rq_blk(OcKvRqCache *c, size_t layer, size_t kind, size_t h,
                       int64_t pos)
{
    uint8_t *base = c->layer[layer];
    if (kind == 0)
        return base + (h * c->n_ctx + (size_t)pos) * c->kc.block_bytes;
    return base + c->n_kv * c->n_ctx * c->kc.block_bytes +
           (h * c->n_ctx + (size_t)pos) * c->vc.block_bytes;
}

/* Encode page pg of `layer` from the ring (positions [lo, hi) of the page
 * that have an exact slot), fixing the page mean first if needed. */
static void page_encode(OcKvRqCache *c, size_t layer, size_t pg)
{
    const size_t d = c->d;
    int64_t lo = (int64_t)(pg * c->page);
    const int64_t hi = lo + (int64_t)c->page;
    if (lo < (int64_t)c->p.n_sink) lo = (int64_t)c->p.n_sink;
    uint8_t *fixed = &c->mu_fixed[layer * c->n_pages + pg];
    float x[OC_KVRQ_DIM_MAX];
    for (size_t kind = 0; kind < 2; kind++) {
        for (size_t h = 0; h < c->n_kv; h++) {
            float *mu = (float *)oc_kvrq_mu(c, layer, pg, kind, h);
            if (!*fixed) {
                double acc[OC_KVRQ_DIM_MAX] = {0};
                size_t m = 0;
                for (int64_t t = lo; t < hi; t++) {
                    const int64_t s = oc_kvrq_slot(c, layer, t);
                    if (s < 0) continue;
                    const int8_t *q = oc_kvrq_xq(c, layer, kind, h) + (size_t)s * d;
                    const float sc = oc_kvrq_xs(c, layer, kind, h)[s];
                    for (size_t i = 0; i < d; i++) acc[i] += (double)sc * q[i];
                    m++;
                }
                for (size_t i = 0; i < d; i++)
                    mu[i] = m > 0 ? (float)(acc[i] / (double)m) : 0.0f;
            }
            for (int64_t t = lo; t < hi; t++) {
                const int64_t s = oc_kvrq_slot(c, layer, t);
                if (s < 0) continue;
                const int8_t *q = oc_kvrq_xq(c, layer, kind, h) + (size_t)s * d;
                const float sc = oc_kvrq_xs(c, layer, kind, h)[s];
                for (size_t i = 0; i < d; i++) x[i] = sc * (float)q[i] - mu[i];
                oc_kvrq_encode(kind ? &c->vc : &c->kc, x,
                               rq_blk(c, layer, kind, h, t));
            }
        }
    }
    *fixed = 1;
}

void oc_kvrq_flush(OcKvRqCache *c, size_t layer)
{
    if (c == NULL || c->page == 0) return;
    /* The newest tagged position decides the current page. */
    int64_t hi = -1;
    for (size_t s = 0; s < c->n_slots; s++)
        if (c->tag[layer * c->n_slots + s] > hi)
            hi = c->tag[layer * c->n_slots + s];
    if (hi < (int64_t)c->p.n_sink) return;
    page_encode(c, layer, oc_kvrq_page_of(c, hi));
}

void oc_kvrq_decode_pos(const OcKvRqCache *c, size_t layer, size_t kind,
                        size_t head, int64_t t, float *out)
{
    const OcKvRqCodec *cd = kind ? &c->vc : &c->kc;
    const uint8_t *b = (kind ? oc_kvrq_vblocks(c, layer, head)
                             : oc_kvrq_kblocks(c, layer, head)) +
                       (size_t)t * cd->block_bytes;
    oc_kvrq_decode(cd, b, out);
    if (c->page == 0) return;
    const size_t pg = oc_kvrq_page_of(c, t);
    if (!c->mu_fixed[layer * c->n_pages + pg]) return;
    const float *mu = oc_kvrq_mu(c, layer, pg, kind, head);
    for (size_t i = 0; i < c->d; i++) out[i] += mu[i];
}

/* ── Speculative-row checkpoint ─────────────────────────────────────────
 * Per layer: n slot records {tag, then for kind K,V and every head: d int8
 * codes + f32 scale}, then the mean flags of pages page_of(pos0) ..
 * page_of(pos0+n-1) (at most 2 since n <= page). */
static int64_t ckpt_slot_index(const OcKvRqCache *c, int64_t pos)
{
    if (pos < 0) return -1;
    if ((uint64_t)pos < c->p.n_sink) return pos;
    if (c->p.window == 0) return -1;
    return (int64_t)c->p.n_sink + pos % (int64_t)c->p.window;
}

static size_t ckpt_rec_bytes(const OcKvRqCache *c)
{
    return sizeof(int64_t) + 2u * c->n_kv * (c->d + sizeof(float));
}

static size_t ckpt_layer_bytes(const OcKvRqCache *c, size_t n)
{
    return n * ckpt_rec_bytes(c) + 2u;
}

size_t oc_kvrq_ckpt_bytes(const OcKvRqCache *c, size_t n)
{
    if (c == NULL) return 0;
    return c->n_layers * ckpt_layer_bytes(c, n);
}

void oc_kvrq_ckpt_save(const OcKvRqCache *c, int64_t pos0, size_t n,
                       void *buf)
{
    if (c == NULL || buf == NULL || n == 0) return;
    const size_t d = c->d, rec = ckpt_rec_bytes(c);
    for (size_t l = 0; l < c->n_layers; l++) {
        uint8_t *b = (uint8_t *)buf + l * ckpt_layer_bytes(c, n);
        for (size_t j = 0; j < n; j++) {
            uint8_t *r = b + j * rec;
            const int64_t s = ckpt_slot_index(c, pos0 + (int64_t)j);
            int64_t tag = -1;
            if (s >= 0 && c->n_slots > 0) {
                tag = c->tag[l * c->n_slots + (size_t)s];
                uint8_t *w = r + sizeof(int64_t);
                for (size_t kind = 0; kind < 2; kind++)
                    for (size_t h = 0; h < c->n_kv; h++) {
                        memcpy(w, oc_kvrq_xq(c, l, kind, h) + (size_t)s * d, d);
                        w += d;
                        memcpy(w, oc_kvrq_xs(c, l, kind, h) + s, sizeof(float));
                        w += sizeof(float);
                    }
            }
            memcpy(r, &tag, sizeof tag);
        }
        uint8_t *fl = b + n * rec;
        fl[0] = fl[1] = 0;
        if (c->page != 0) {
            const size_t p0 = oc_kvrq_page_of(c, pos0);
            const size_t p1 = oc_kvrq_page_of(c, pos0 + (int64_t)n - 1);
            for (size_t pg = p0; pg <= p1 && pg < p0 + 2 && pg < c->n_pages; pg++)
                fl[pg - p0] = c->mu_fixed[l * c->n_pages + pg];
        }
    }
}

void oc_kvrq_ckpt_restore(OcKvRqCache *c, int64_t pos0, size_t n,
                          const void *buf, size_t layer, int64_t keep)
{
    if (c == NULL || buf == NULL || n == 0 || layer >= c->n_layers) return;
    const size_t d = c->d, rec = ckpt_rec_bytes(c);
    const uint8_t *b = (const uint8_t *)buf + layer * ckpt_layer_bytes(c, n);
    if (c->n_slots > 0) {
        for (size_t j = 0; j < n; j++) {
            const int64_t pos = pos0 + (int64_t)j;
            if (pos < keep) continue;
            const int64_t s = ckpt_slot_index(c, pos);
            if (s < 0) continue;
            int64_t *tag = &c->tag[layer * c->n_slots + (size_t)s];
            if (*tag < keep) continue;          /* already real data */
            const uint8_t *r = b + j * rec;
            memcpy(tag, r, sizeof(int64_t));
            const uint8_t *w = r + sizeof(int64_t);
            for (size_t kind = 0; kind < 2; kind++)
                for (size_t h = 0; h < c->n_kv; h++) {
                    memcpy((int8_t *)oc_kvrq_xq(c, layer, kind, h) +
                           (size_t)s * d, w, d);
                    w += d;
                    memcpy((float *)oc_kvrq_xs(c, layer, kind, h) + s, w,
                           sizeof(float));
                    w += sizeof(float);
                }
        }
    }
    if (c->page != 0) {
        const uint8_t *fl = b + n * rec;
        const size_t p0 = oc_kvrq_page_of(c, pos0);
        const size_t p1 = oc_kvrq_page_of(c, pos0 + (int64_t)n - 1);
        for (size_t pg = p0; pg <= p1 && pg < p0 + 2 && pg < c->n_pages; pg++) {
            const int64_t last = (int64_t)((pg + 1) * c->page) - 1;
            if (last >= keep)
                c->mu_fixed[layer * c->n_pages + pg] = fl[pg - p0];
        }
    }
}

size_t oc_kvrq_bytes_per_token(const OcKvRqCache *c)
{
    return c->n_layers * c->n_kv * (c->kc.block_bytes + c->vc.block_bytes);
}

void oc_kvrq_store(OcKvRqCache *c, size_t layer, int64_t pos, const float *k,
                   const float *v, float *scratch)
{
    const size_t d = c->d;
    float *kr = scratch, *vr = scratch + d;
    int64_t slot = -1;
    if ((uint64_t)pos < c->p.n_sink) slot = pos;
    else if (c->p.window > 0)
        slot = (int64_t)c->p.n_sink + pos % (int64_t)c->p.window;
    /* Centered: the ring holds the current page; its RQ blocks are
     * written when the page completes (page_encode). */
    const int direct = c->page == 0 && (uint64_t)pos >= c->p.n_sink;
    for (size_t h = 0; h < c->n_kv; h++) {
        oc_kvrq_rotate(&c->rot, k + h * d, kr);
        oc_kvrq_rotate(&c->rot, v + h * d, vr);
        if (direct) {
            oc_kvrq_encode(&c->kc, kr, rq_blk(c, layer, 0, h, pos));
            oc_kvrq_encode(&c->vc, vr, rq_blk(c, layer, 1, h, pos));
        }
        if (slot >= 0) {
            for (size_t kind = 0; kind < 2; kind++) {
                int8_t *xq = (int8_t *)oc_kvrq_xq(c, layer, kind, h) +
                             (size_t)slot * d;
                float *xs = (float *)oc_kvrq_xs(c, layer, kind, h) + slot;
                oc_kvq8_encode_row(kind ? vr : kr, d, xq, xs);
            }
        }
    }
    if (slot >= 0) c->tag[layer * c->n_slots + (size_t)slot] = pos;
    if (c->page != 0 && ((size_t)pos + 1) % c->page == 0)
        page_encode(c, layer, oc_kvrq_page_of(c, pos));
}
