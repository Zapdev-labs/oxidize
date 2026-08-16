#include "oxidize/attn_kernels.h"

#include <stdint.h>

static float dot_f32_scalar(const float *a, const float *b, size_t n)
{
    float acc = 0.0f;
    for (size_t i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}

static void axpy_f32_scalar(float *y, const float *x, float alpha, size_t n)
{
    for (size_t i = 0; i < n; i++) y[i] += alpha * x[i];
}

static void scale_f32_scalar(float *y, float scale, size_t n)
{
    for (size_t i = 0; i < n; i++) y[i] *= scale;
}

static void add_f32_scalar(float *y, const float *x, size_t n)
{
    for (size_t i = 0; i < n; i++) y[i] += x[i];
}

static void rms_apply_scalar(const float *x, const float *weight, float inv,
                             float *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = x[i] * inv * weight[i];
}

static float dot_q8_scalar(const float *a, const int8_t *b, size_t n)
{
    float acc = 0.0f;
    for (size_t i = 0; i < n; i++) acc += a[i] * (float)b[i];
    return acc;
}

static void axpy_q8_scalar(float *y, const int8_t *x, float alpha, size_t n)
{
    for (size_t i = 0; i < n; i++) y[i] += alpha * (float)x[i];
}

#if defined(__x86_64__) || defined(__i386__)

#include <immintrin.h>

static inline float __attribute__((target("avx2,fma")))
hsum256(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s);
    s = _mm_add_ps(s, sh);
    sh = _mm_movehl_ps(s, s);
    s = _mm_add_ss(s, sh);
    return _mm_cvtss_f32(s);
}

__attribute__((target("avx2,fma")))
static float dot_f32_avx2(const float *a, const float *b, size_t n)
{
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                              _mm256_loadu_ps(b + i), acc);
    }
    float sum = hsum256(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

__attribute__((target("avx2,fma")))
static void axpy_f32_avx2(float *y, const float *x, float alpha, size_t n)
{
    const __m256 va = _mm256_set1_ps(alpha);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i,
            _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i),
                            _mm256_loadu_ps(y + i)));
    }
    for (; i < n; i++) y[i] += alpha * x[i];
}

__attribute__((target("avx2,fma")))
static void scale_f32_avx2(float *y, float scale, size_t n)
{
    const __m256 vs = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), vs));
    for (; i < n; i++) y[i] *= scale;
}

__attribute__((target("avx2,fma")))
static void add_f32_avx2(float *y, const float *x, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(y + i),
                                              _mm256_loadu_ps(x + i)));
    for (; i < n; i++) y[i] += x[i];
}

__attribute__((target("avx2,fma")))
static void rms_apply_avx2(const float *x, const float *weight, float inv,
                           float *out, size_t n)
{
    const __m256 vinv = _mm256_set1_ps(inv);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(v, _mm256_loadu_ps(weight + i)));
    }
    for (; i < n; i++) out[i] = x[i] * inv * weight[i];
}

__attribute__((target("avx2,fma")))
static float dot_q8_avx2(const float *a, const int8_t *b, size_t n)
{
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i b8 = _mm_loadl_epi64((const __m128i *)(b + i));
        __m256 bf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b8));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), bf, acc);
    }
    float sum = hsum256(acc);
    for (; i < n; i++) sum += a[i] * (float)b[i];
    return sum;
}

__attribute__((target("avx2,fma")))
static void axpy_q8_avx2(float *y, const int8_t *x, float alpha, size_t n)
{
    const __m256 va = _mm256_set1_ps(alpha);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i x8 = _mm_loadl_epi64((const __m128i *)(x + i));
        __m256 xf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(x8));
        _mm256_storeu_ps(y + i,
            _mm256_fmadd_ps(va, xf, _mm256_loadu_ps(y + i)));
    }
    for (; i < n; i++) y[i] += alpha * (float)x[i];
}

__attribute__((target("avx512f")))
static float dot_f32_avx512(const float *a, const float *b, size_t n)
{
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),
                              _mm512_loadu_ps(b + i), acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

__attribute__((target("avx512f")))
static void axpy_f32_avx512(float *y, const float *x, float alpha, size_t n)
{
    const __m512 va = _mm512_set1_ps(alpha);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(y + i,
            _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i),
                            _mm512_loadu_ps(y + i)));
    }
    for (; i < n; i++) y[i] += alpha * x[i];
}

__attribute__((target("avx512f")))
static void scale_f32_avx512(float *y, float scale, size_t n)
{
    const __m512 vs = _mm512_set1_ps(scale);
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_loadu_ps(y + i), vs));
    for (; i < n; i++) y[i] *= scale;
}

__attribute__((target("avx512f")))
static void add_f32_avx512(float *y, const float *x, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(y + i, _mm512_add_ps(_mm512_loadu_ps(y + i),
                                              _mm512_loadu_ps(x + i)));
    for (; i < n; i++) y[i] += x[i];
}

__attribute__((target("avx512f")))
static void rms_apply_avx512(const float *x, const float *weight, float inv,
                             float *out, size_t n)
{
    const __m512 vinv = _mm512_set1_ps(inv);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_mul_ps(_mm512_loadu_ps(x + i), vinv);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(v, _mm512_loadu_ps(weight + i)));
    }
    for (; i < n; i++) out[i] = x[i] * inv * weight[i];
}

__attribute__((target("avx512f")))
static float dot_q8_avx512(const float *a, const int8_t *b, size_t n)
{
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i b16 = _mm_loadu_si128((const __m128i *)(b + i));
        __m512 bf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b16));
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), bf, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) sum += a[i] * (float)b[i];
    return sum;
}

__attribute__((target("avx512f")))
static void axpy_q8_avx512(float *y, const int8_t *x, float alpha, size_t n)
{
    const __m512 va = _mm512_set1_ps(alpha);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i x16 = _mm_loadu_si128((const __m128i *)(x + i));
        __m512 xf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(x16));
        _mm512_storeu_ps(y + i,
            _mm512_fmadd_ps(va, xf, _mm512_loadu_ps(y + i)));
    }
    for (; i < n; i++) y[i] += alpha * (float)x[i];
}

static int g_isa = -1;

static int detect_isa(void)
{
    if (g_isa >= 0) return g_isa;
    if (__builtin_cpu_supports("avx512f")) {
        g_isa = 2;
        return g_isa;
    }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        g_isa = 1;
        return g_isa;
    }
    g_isa = 0;
    return g_isa;
}

float oc_attn_dot_f32(const float *a, const float *b, size_t n)
{
    switch (detect_isa()) {
    case 2: return dot_f32_avx512(a, b, n);
    case 1: return dot_f32_avx2(a, b, n);
    default: return dot_f32_scalar(a, b, n);
    }
}

void oc_attn_axpy_f32(float *y, const float *x, float alpha, size_t n)
{
    switch (detect_isa()) {
    case 2: axpy_f32_avx512(y, x, alpha, n); return;
    case 1: axpy_f32_avx2(y, x, alpha, n); return;
    default: axpy_f32_scalar(y, x, alpha, n); return;
    }
}

void oc_attn_scale_f32(float *y, float scale, size_t n)
{
    if (scale == 1.0f) return;
    switch (detect_isa()) {
    case 2: scale_f32_avx512(y, scale, n); return;
    case 1: scale_f32_avx2(y, scale, n); return;
    default: scale_f32_scalar(y, scale, n); return;
    }
}

void oc_attn_add_f32(float *y, const float *x, size_t n)
{
    switch (detect_isa()) {
    case 2: add_f32_avx512(y, x, n); return;
    case 1: add_f32_avx2(y, x, n); return;
    default: add_f32_scalar(y, x, n); return;
    }
}

void oc_attn_rms_apply_f32(const float *x, const float *weight, float inv,
                           float *out, size_t n)
{
    switch (detect_isa()) {
    case 2: rms_apply_avx512(x, weight, inv, out, n); return;
    case 1: rms_apply_avx2(x, weight, inv, out, n); return;
    default: rms_apply_scalar(x, weight, inv, out, n); return;
    }
}

float oc_attn_dot_q8(const float *a, const int8_t *b, size_t n)
{
    switch (detect_isa()) {
    case 2: return dot_q8_avx512(a, b, n);
    case 1: return dot_q8_avx2(a, b, n);
    default: return dot_q8_scalar(a, b, n);
    }
}

void oc_attn_axpy_q8(float *y, const int8_t *x, float alpha, size_t n)
{
    switch (detect_isa()) {
    case 2: axpy_q8_avx512(y, x, alpha, n); return;
    case 1: axpy_q8_avx2(y, x, alpha, n); return;
    default: axpy_q8_scalar(y, x, alpha, n); return;
    }
}

#else

float oc_attn_dot_f32(const float *a, const float *b, size_t n)
{
    return dot_f32_scalar(a, b, n);
}
void oc_attn_axpy_f32(float *y, const float *x, float alpha, size_t n)
{
    axpy_f32_scalar(y, x, alpha, n);
}
void oc_attn_scale_f32(float *y, float scale, size_t n)
{
    if (scale == 1.0f) return;
    scale_f32_scalar(y, scale, n);
}
void oc_attn_add_f32(float *y, const float *x, size_t n)
{
    add_f32_scalar(y, x, n);
}
void oc_attn_rms_apply_f32(const float *x, const float *weight, float inv,
                           float *out, size_t n)
{
    rms_apply_scalar(x, weight, inv, out, n);
}
float oc_attn_dot_q8(const float *a, const int8_t *b, size_t n)
{
    return dot_q8_scalar(a, b, n);
}
void oc_attn_axpy_q8(float *y, const int8_t *x, float alpha, size_t n)
{
    axpy_q8_scalar(y, x, alpha, n);
}

#endif
