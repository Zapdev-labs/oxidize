/*
 * inkling_simd.h — vector math for the Inkling-2 forward (private).
 *
 * AVX2/FMA kernels behind a runtime check, scalar fallbacks otherwise. The
 * exp is the Cephes single-precision polynomial (relative error ~2e-7 on
 * the clamped range); tanh and sigmoid are built on it.
 */
#ifndef OC_INKLING_SIMD_H
#define OC_INKLING_SIMD_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "oxidize/nvfp4.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define INK_X86 1
#endif

static inline float ink_bf16f(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline float ink_sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

#ifdef INK_X86
__attribute__((target("avx2,fma")))
static inline __m256 ink_exp8(__m256 x)
{
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3f)), _mm256_set1_ps(88.0f));
    __m256 fx = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f)));
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

/* tanh(x) = 1 - 2 / (exp(2x) + 1) */
__attribute__((target("avx2,fma")))
static inline __m256 ink_tanh8(__m256 x)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 e = ink_exp8(_mm256_add_ps(x, x));
    return _mm256_sub_ps(one, _mm256_div_ps(_mm256_set1_ps(2.0f), _mm256_add_ps(e, one)));
}

__attribute__((target("avx2,fma")))
static inline __m256 ink_sigmoid8(__m256 x)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, ink_exp8(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}

__attribute__((target("avx2,fma")))
static inline float ink_hsum8(__m256 v)
{
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

__attribute__((target("avx2,fma")))
static inline float ink_dot_avx2(const float *a, const float *b, size_t n)
{
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    float acc = ink_hsum8(_mm256_add_ps(s0, s1));
    for (; i < n; i++) acc += a[i] * b[i];
    return acc;
}

__attribute__((target("avx2,fma")))
static inline float ink_dot_bf16_avx2(const uint16_t *a, const float *b, size_t n)
{
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i w0 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(a + i))), 16);
        __m256i w1 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(a + i + 8))), 16);
        s0 = _mm256_fmadd_ps(_mm256_castsi256_ps(w0), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_castsi256_ps(w1), _mm256_loadu_ps(b + i + 8), s1);
    }
    float acc = ink_hsum8(_mm256_add_ps(s0, s1));
    for (; i < n; i++) acc += ink_bf16f(a[i]) * b[i];
    return acc;
}

__attribute__((target("avx2,fma")))
static inline float ink_sumsq_avx2(const float *x, size_t n)
{
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 a = _mm256_loadu_ps(x + i), b = _mm256_loadu_ps(x + i + 8);
        s0 = _mm256_fmadd_ps(a, a, s0);
        s1 = _mm256_fmadd_ps(b, b, s1);
    }
    float acc = ink_hsum8(_mm256_add_ps(s0, s1));
    for (; i < n; i++) acc += x[i] * x[i];
    return acc;
}

/* out = x * s * (w ? w : 1) */
__attribute__((target("avx2,fma")))
static inline void ink_scale_avx2(const float *x, const float *w, float s, size_t n, float *out)
{
    const __m256 vs = _mm256_set1_ps(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x + i), vs);
        if (w) v = _mm256_mul_ps(v, _mm256_loadu_ps(w + i));
        _mm256_storeu_ps(out + i, v);
    }
    for (; i < n; i++) out[i] = x[i] * s * (w ? w[i] : 1.0f);
}

__attribute__((target("avx2,fma")))
static inline void ink_axpy_avx2(float *y, float a, const float *x, size_t n)
{
    const __m256 va = _mm256_set1_ps(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += a * x[i];
}

/* x[i] = c * tanh(x[i] * ic) */
__attribute__((target("avx2,fma")))
static inline void ink_softcap_avx2(float *x, size_t n, float c)
{
    const __m256 vc = _mm256_set1_ps(c), vic = _mm256_set1_ps(1.0f / c);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, _mm256_mul_ps(vc, ink_tanh8(_mm256_mul_ps(_mm256_loadu_ps(x + i), vic))));
    for (; i < n; i++) x[i] = c * tanhf(x[i] / c);
}

/* out = beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb) */
__attribute__((target("avx2,fma")))
static inline void ink_situ_avx2(const float *g, const float *u, float *out, size_t n, float beta, float lb)
{
    const __m256 vb = _mm256_set1_ps(beta), vib = _mm256_set1_ps(1.0f / beta);
    const __m256 vl = _mm256_set1_ps(lb), vil = _mm256_set1_ps(lb > 0.0f ? 1.0f / lb : 0.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 gv = _mm256_loadu_ps(g + i), uv = _mm256_loadu_ps(u + i);
        __m256 ga = _mm256_mul_ps(_mm256_mul_ps(vb, ink_tanh8(_mm256_mul_ps(gv, vib))), ink_sigmoid8(gv));
        __m256 up = lb > 0.0f ? _mm256_mul_ps(vl, ink_tanh8(_mm256_mul_ps(uv, vil))) : uv;
        _mm256_storeu_ps(out + i, _mm256_mul_ps(ga, up));
    }
    for (; i < n; i++) {
        const float ga = beta * tanhf(g[i] / beta) * ink_sigmoidf(g[i]);
        out[i] = ga * (lb > 0.0f ? lb * tanhf(u[i] / lb) : u[i]);
    }
}

/* x[i] = exp(x[i] - m); returns the sum */
__attribute__((target("avx2,fma")))
static inline float ink_exp_sub_sum_avx2(float *x, size_t n, float m)
{
    const __m256 vm = _mm256_set1_ps(m);
    __m256 s = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 e = ink_exp8(_mm256_sub_ps(_mm256_loadu_ps(x + i), vm));
        _mm256_storeu_ps(x + i, e);
        s = _mm256_add_ps(s, e);
    }
    float acc = ink_hsum8(s);
    for (; i < n; i++) { x[i] = expf(x[i] - m); acc += x[i]; }
    return acc;
}

/* out = a * sigmoid(g) * s */
__attribute__((target("avx2,fma")))
static inline void ink_gate_avx2(const float *a, const float *g, float s, size_t n, float *out)
{
    const __m256 vs = _mm256_set1_ps(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(a + i), vs), ink_sigmoid8(_mm256_loadu_ps(g + i))));
    for (; i < n; i++) out[i] = a[i] * s * ink_sigmoidf(g[i]);
}

/* x = x - tanh(f) * x + tanh(gn) * m */
__attribute__((target("avx2,fma")))
static inline void ink_ctrl_mix_avx2(float *x, const float *gn, const float *f, const float *m, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        const __m256 fv = ink_tanh8(_mm256_loadu_ps(f + i)), gv = ink_tanh8(_mm256_loadu_ps(gn + i));
        _mm256_storeu_ps(x + i, _mm256_fmadd_ps(gv, _mm256_loadu_ps(m + i), _mm256_fnmadd_ps(fv, xv, xv)));
    }
    for (; i < n; i++) x[i] = x[i] - tanhf(f[i]) * x[i] + tanhf(gn[i]) * m[i];
}
#endif

static inline bool ink_avx2(void) { return oc_nvfp4_have_avx2(); }

static inline float ink_dot(const float *a, const float *b, size_t n)
{
#ifdef INK_X86
    if (ink_avx2()) return ink_dot_avx2(a, b, n);
#endif
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline float ink_dot_bf16(const uint16_t *a, const float *b, size_t n)
{
#ifdef INK_X86
    if (ink_avx2()) return ink_dot_bf16_avx2(a, b, n);
#endif
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += ink_bf16f(a[i]) * b[i];
    return s;
}

static inline float ink_sumsq(const float *x, size_t n)
{
#ifdef INK_X86
    if (ink_avx2()) return ink_sumsq_avx2(x, n);
#endif
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += (double)x[i] * x[i];
    return (float)s;
}

static inline float ink_rstd(const float *x, size_t n, float eps)
{
    return 1.0f / sqrtf(ink_sumsq(x, n) / (float)n + eps);
}

static inline void ink_scale(const float *x, const float *w, float s, size_t n, float *out)
{
#ifdef INK_X86
    if (ink_avx2()) { ink_scale_avx2(x, w, s, n, out); return; }
#endif
    for (size_t i = 0; i < n; i++) out[i] = x[i] * s * (w ? w[i] : 1.0f);
}

static inline void ink_rms_norm(const float *x, const float *w, size_t n, float eps, float *out)
{
    ink_scale(x, w, ink_rstd(x, n, eps), n, out);
}

static inline void ink_axpy(float *y, float a, const float *x, size_t n)
{
#ifdef INK_X86
    if (ink_avx2()) { ink_axpy_avx2(y, a, x, n); return; }
#endif
    for (size_t i = 0; i < n; i++) y[i] += a * x[i];
}

static inline void ink_softcap(float *x, size_t n, float c)
{
#ifdef INK_X86
    if (ink_avx2()) { ink_softcap_avx2(x, n, c); return; }
#endif
    for (size_t i = 0; i < n; i++) x[i] = c * tanhf(x[i] / c);
}

static inline void ink_situ(const float *g, const float *u, float *out, size_t n, float beta, float lb)
{
#ifdef INK_X86
    if (ink_avx2()) { ink_situ_avx2(g, u, out, n, beta, lb); return; }
#endif
    for (size_t i = 0; i < n; i++) {
        const float ga = beta * tanhf(g[i] / beta) * ink_sigmoidf(g[i]);
        out[i] = ga * (lb > 0.0f ? lb * tanhf(u[i] / lb) : u[i]);
    }
}

static inline float ink_exp_sub_sum(float *x, size_t n, float m)
{
#ifdef INK_X86
    if (ink_avx2()) return ink_exp_sub_sum_avx2(x, n, m);
#endif
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    return s;
}

static inline void ink_gate(const float *a, const float *g, float s, size_t n, float *out)
{
#ifdef INK_X86
    if (ink_avx2()) { ink_gate_avx2(a, g, s, n, out); return; }
#endif
    for (size_t i = 0; i < n; i++) out[i] = a[i] * s * ink_sigmoidf(g[i]);
}

static inline void ink_ctrl_mix(float *x, const float *gn, const float *f, const float *m, size_t n)
{
#ifdef INK_X86
    if (ink_avx2()) { ink_ctrl_mix_avx2(x, gn, f, m, n); return; }
#endif
    for (size_t i = 0; i < n; i++) x[i] = x[i] - tanhf(f[i]) * x[i] + tanhf(gn[i]) * m[i];
}

#endif
