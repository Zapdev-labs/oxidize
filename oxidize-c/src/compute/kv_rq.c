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

static KVRQ_TGT void rq_score_avx2_b(const OcKvRqCodec *c,
                                     const uint8_t *blocks, size_t n,
                                     const float *q, size_t G, float *scores,
                                     size_t ss, const unsigned bits)
{
    const size_t d = c->d, bb = c->block_bytes, nm = d / 8;
    const __m256 c_lo = _mm256_loadu_ps(c->centroids);
    const __m256 c_hi = _mm256_loadu_ps(c->centroids + 8);
    for (size_t g0 = 0; g0 < G; g0 += 4) {
        const size_t gc = G - g0 < 4 ? G - g0 : 4;
        const float *q0 = q + (g0 + 0) * d;
        const float *q1 = q + (g0 + (gc > 1 ? 1 : 0)) * d;
        const float *q2 = q + (g0 + (gc > 2 ? 2 : 0)) * d;
        const float *q3 = q + (g0 + (gc > 3 ? 3 : 0)) * d;
        for (size_t t = 0; t < n; t++) {
            const uint8_t *blk = blocks + t * bb;
            const float s = _cvtsh_ss(load_u16(blk));
            const uint8_t *codes = blk + 2;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            for (size_t m = 0; m < nm; m++) {
                const __m256 cv = rq_lut(rq_idx8(codes, d, bits, m), c_lo,
                                         c_hi, bits);
                a0 = _mm256_fmadd_ps(cv, _mm256_loadu_ps(q0 + 8 * m), a0);
                a1 = _mm256_fmadd_ps(cv, _mm256_loadu_ps(q1 + 8 * m), a1);
                a2 = _mm256_fmadd_ps(cv, _mm256_loadu_ps(q2 + 8 * m), a2);
                a3 = _mm256_fmadd_ps(cv, _mm256_loadu_ps(q3 + 8 * m), a3);
            }
            float r[4];
            _mm_storeu_ps(r, _mm_mul_ps(hsum4x8(a0, a1, a2, a3),
                                        _mm_set1_ps(s)));
            for (size_t g = 0; g < gc; g++) scores[(g0 + g) * ss + t] = r[g];
        }
    }
}

#define KVRQ_CHUNK 64u

static KVRQ_TGT void rq_accum_avx2_b(const OcKvRqCodec *c,
                                     const uint8_t *blocks, size_t n,
                                     const float *w, size_t ws, size_t G,
                                     float *acc, const unsigned bits)
{
    const size_t d = c->d, bb = c->block_bytes, nm = d / 8;
    const __m256 c_lo = _mm256_loadu_ps(c->centroids);
    const __m256 c_hi = _mm256_loadu_ps(c->centroids + 8);
    float wsc[4][KVRQ_CHUNK];
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
            float *o0 = acc + (g0 + 0) * d;
            for (size_t m = 0; m < nm; m++) {
                __m256 a0 = _mm256_loadu_ps(o0 + 8 * m);
                __m256 a1 = gc > 1 ? _mm256_loadu_ps(o0 + d + 8 * m)
                                   : _mm256_setzero_ps();
                __m256 a2 = gc > 2 ? _mm256_loadu_ps(o0 + 2 * d + 8 * m)
                                   : _mm256_setzero_ps();
                __m256 a3 = gc > 3 ? _mm256_loadu_ps(o0 + 3 * d + 8 * m)
                                   : _mm256_setzero_ps();
                for (size_t t = 0; t < tn; t++) {
                    const __m256 cv = rq_lut(rq_idx8(bl + t * bb + 2, d, bits,
                                                     m), c_lo, c_hi, bits);
                    a0 = _mm256_fmadd_ps(cv, _mm256_broadcast_ss(&wsc[0][t]), a0);
                    a1 = _mm256_fmadd_ps(cv, _mm256_broadcast_ss(&wsc[1][t]), a1);
                    a2 = _mm256_fmadd_ps(cv, _mm256_broadcast_ss(&wsc[2][t]), a2);
                    a3 = _mm256_fmadd_ps(cv, _mm256_broadcast_ss(&wsc[3][t]), a3);
                }
                _mm256_storeu_ps(o0 + 8 * m, a0);
                if (gc > 1) _mm256_storeu_ps(o0 + d + 8 * m, a1);
                if (gc > 2) _mm256_storeu_ps(o0 + 2 * d + 8 * m, a2);
                if (gc > 3) _mm256_storeu_ps(o0 + 3 * d + 8 * m, a3);
            }
        }
    }
}

static KVRQ_TGT void rq_decode_rows_avx2_b(const OcKvRqCodec *c,
                                           const uint8_t *blocks, size_t n,
                                           float *rows, const unsigned bits)
{
    const size_t d = c->d, bb = c->block_bytes, nm = d / 8;
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

__attribute__((target("avx2,fma,f16c")))
static void rq_score_avx2(const OcKvRqCodec *c, const uint8_t *b, size_t n,
                          const float *q, size_t G, float *s, size_t ss)
{
    switch (c->bits) {
    case 2: rq_score_avx2_b(c, b, n, q, G, s, ss, 2); break;
    case 3: rq_score_avx2_b(c, b, n, q, G, s, ss, 3); break;
    default: rq_score_avx2_b(c, b, n, q, G, s, ss, 4); break;
    }
}

__attribute__((target("avx2,fma,f16c")))
static void rq_accum_avx2(const OcKvRqCodec *c, const uint8_t *b, size_t n,
                          const float *w, size_t ws, size_t G, float *acc)
{
    switch (c->bits) {
    case 2: rq_accum_avx2_b(c, b, n, w, ws, G, acc, 2); break;
    case 3: rq_accum_avx2_b(c, b, n, w, ws, G, acc, 3); break;
    default: rq_accum_avx2_b(c, b, n, w, ws, G, acc, 4); break;
    }
}

__attribute__((target("avx2,fma,f16c")))
static void rq_decode_rows_avx2(const OcKvRqCodec *c, const uint8_t *b,
                                size_t n, float *rows)
{
    switch (c->bits) {
    case 2: rq_decode_rows_avx2_b(c, b, n, rows, 2); break;
    case 3: rq_decode_rows_avx2_b(c, b, n, rows, 3); break;
    default: rq_decode_rows_avx2_b(c, b, n, rows, 4); break;
    }
}

static KVRQ_TGT __m256 load8_i8f(const int8_t *p)
{
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
        _mm_loadl_epi64((const __m128i *)(const void *)p)));
}

__attribute__((target("avx2,fma,f16c")))
static void q8_score_avx2(const int8_t *codes, size_t cs, const float *scales,
                          size_t sst, size_t d, size_t n, const float *q,
                          size_t G, float *scores, size_t ss)
{
    const size_t nm = d / 8;
    for (size_t g0 = 0; g0 < G; g0 += 4) {
        const size_t gc = G - g0 < 4 ? G - g0 : 4;
        const float *q0 = q + (g0 + 0) * d;
        const float *q1 = q + (g0 + (gc > 1 ? 1 : 0)) * d;
        const float *q2 = q + (g0 + (gc > 2 ? 2 : 0)) * d;
        const float *q3 = q + (g0 + (gc > 3 ? 3 : 0)) * d;
        for (size_t t = 0; t < n; t++) {
            const int8_t *r = codes + t * cs;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            for (size_t m = 0; m < nm; m++) {
                const __m256 kv = load8_i8f(r + 8 * m);
                a0 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q0 + 8 * m), a0);
                a1 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q1 + 8 * m), a1);
                a2 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q2 + 8 * m), a2);
                a3 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q3 + 8 * m), a3);
            }
            float rr[4];
            _mm_storeu_ps(rr, _mm_mul_ps(hsum4x8(a0, a1, a2, a3),
                                         _mm_set1_ps(scales[t * sst])));
            for (size_t g = 0; g < gc; g++) scores[(g0 + g) * ss + t] = rr[g];
        }
    }
}

__attribute__((target("avx2,fma,f16c")))
static void q8_accum_avx2(const int8_t *codes, size_t cs, const float *scales,
                          size_t sst, size_t d, size_t n, const float *w,
                          size_t ws, size_t G, float *acc)
{
    const size_t nm = d / 8;
    float wsc[4][KVRQ_CHUNK];
    for (size_t t0 = 0; t0 < n; t0 += KVRQ_CHUNK) {
        const size_t tn = n - t0 < KVRQ_CHUNK ? n - t0 : KVRQ_CHUNK;
        for (size_t g0 = 0; g0 < G; g0 += 4) {
            const size_t gc = G - g0 < 4 ? G - g0 : 4;
            for (size_t t = 0; t < tn; t++) {
                const float s = scales[(t0 + t) * sst];
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
                for (size_t t = 0; t < tn; t++) {
                    const __m256 kv = load8_i8f(codes + (t0 + t) * cs + 8 * m);
                    a0 = _mm256_fmadd_ps(kv, _mm256_broadcast_ss(&wsc[0][t]), a0);
                    a1 = _mm256_fmadd_ps(kv, _mm256_broadcast_ss(&wsc[1][t]), a1);
                    a2 = _mm256_fmadd_ps(kv, _mm256_broadcast_ss(&wsc[2][t]), a2);
                    a3 = _mm256_fmadd_ps(kv, _mm256_broadcast_ss(&wsc[3][t]), a3);
                }
                _mm256_storeu_ps(o0 + 8 * m, a0);
                if (gc > 1) _mm256_storeu_ps(o0 + d + 8 * m, a1);
                if (gc > 2) _mm256_storeu_ps(o0 + 2 * d + 8 * m, a2);
                if (gc > 3) _mm256_storeu_ps(o0 + 3 * d + 8 * m, a3);
            }
        }
    }
}

__attribute__((target("avx2,fma,f16c")))
static void f32_score_avx2(const float *x, size_t xs, size_t d, size_t n,
                           const float *q, size_t G, float *scores, size_t ss)
{
    const size_t nm = d / 8;
    for (size_t g0 = 0; g0 < G; g0 += 4) {
        const size_t gc = G - g0 < 4 ? G - g0 : 4;
        const float *q0 = q + (g0 + 0) * d;
        const float *q1 = q + (g0 + (gc > 1 ? 1 : 0)) * d;
        const float *q2 = q + (g0 + (gc > 2 ? 2 : 0)) * d;
        const float *q3 = q + (g0 + (gc > 3 ? 3 : 0)) * d;
        for (size_t t = 0; t < n; t++) {
            const float *r = x + t * xs;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            for (size_t m = 0; m < nm; m++) {
                const __m256 kv = _mm256_loadu_ps(r + 8 * m);
                a0 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q0 + 8 * m), a0);
                a1 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q1 + 8 * m), a1);
                a2 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q2 + 8 * m), a2);
                a3 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q3 + 8 * m), a3);
            }
            float rr[4];
            _mm_storeu_ps(rr, hsum4x8(a0, a1, a2, a3));
            for (size_t g = 0; g < gc; g++) scores[(g0 + g) * ss + t] = rr[g];
        }
    }
}

__attribute__((target("avx2,fma,f16c")))
static void f32_accum_avx2(const float *x, size_t xs, size_t d, size_t n,
                           const float *w, size_t ws, size_t G, float *acc)
{
    const size_t nm = d / 8;
    for (size_t g0 = 0; g0 < G; g0 += 4) {
        const size_t gc = G - g0 < 4 ? G - g0 : 4;
        const float *w0 = w + (g0 + 0) * ws;
        const float *w1 = w + (g0 + (gc > 1 ? 1 : 0)) * ws;
        const float *w2 = w + (g0 + (gc > 2 ? 2 : 0)) * ws;
        const float *w3 = w + (g0 + (gc > 3 ? 3 : 0)) * ws;
        float *o0 = acc + g0 * d;
        for (size_t m = 0; m < nm; m++) {
            __m256 a0 = _mm256_loadu_ps(o0 + 8 * m);
            __m256 a1 = gc > 1 ? _mm256_loadu_ps(o0 + d + 8 * m)
                               : _mm256_setzero_ps();
            __m256 a2 = gc > 2 ? _mm256_loadu_ps(o0 + 2 * d + 8 * m)
                               : _mm256_setzero_ps();
            __m256 a3 = gc > 3 ? _mm256_loadu_ps(o0 + 3 * d + 8 * m)
                               : _mm256_setzero_ps();
            for (size_t t = 0; t < n; t++) {
                const __m256 kv = _mm256_loadu_ps(x + t * xs + 8 * m);
                a0 = _mm256_fmadd_ps(kv, _mm256_broadcast_ss(w0 + t), a0);
                a1 = _mm256_fmadd_ps(kv, _mm256_broadcast_ss(w1 + t), a1);
                a2 = _mm256_fmadd_ps(kv, _mm256_broadcast_ss(w2 + t), a2);
                a3 = _mm256_fmadd_ps(kv, _mm256_broadcast_ss(w3 + t), a3);
            }
            _mm256_storeu_ps(o0 + 8 * m, a0);
            if (gc > 1) _mm256_storeu_ps(o0 + d + 8 * m, a1);
            if (gc > 2) _mm256_storeu_ps(o0 + 2 * d + 8 * m, a2);
            if (gc > 3) _mm256_storeu_ps(o0 + 3 * d + 8 * m, a3);
        }
    }
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

void oc_kvrq_accum(const OcKvRqCodec *c, const uint8_t *blocks, size_t n,
                   const float *w, size_t ws, size_t G, float *acc)
{
#if KVRQ_HAVE_AVX2
    if (kvrq_isa()) { rq_accum_avx2(c, blocks, n, w, ws, G, acc); return; }
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
    return p == MAP_FAILED ? NULL : p;
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
    oc_kvrq_rot_free(&c->rot);
    memset(c, 0, sizeof(*c));
}

void oc_kvrq_cache_clear(OcKvRqCache *c)
{
    if (c == NULL || c->tag == NULL) return;
    for (size_t i = 0; i < c->n_layers * c->n_slots; i++) c->tag[i] = -1;
}

void oc_kvrq_cache_rewind(OcKvRqCache *c, int64_t pos)
{
    if (c == NULL || c->tag == NULL) return;
    for (size_t i = 0; i < c->n_layers * c->n_slots; i++)
        if (c->tag[i] >= pos) c->tag[i] = -1;
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
    uint8_t *base = c->layer[layer];
    for (size_t h = 0; h < c->n_kv; h++) {
        oc_kvrq_rotate(&c->rot, k + h * d, kr);
        oc_kvrq_rotate(&c->rot, v + h * d, vr);
        oc_kvrq_encode(&c->kc, kr,
                       base + (h * c->n_ctx + (size_t)pos) * c->kc.block_bytes);
        oc_kvrq_encode(&c->vc, vr,
                       base + c->n_kv * c->n_ctx * c->kc.block_bytes +
                       (h * c->n_ctx + (size_t)pos) * c->vc.block_bytes);
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
}
