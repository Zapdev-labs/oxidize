#include "oxk_common.hpp"

#include <cmath>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define OXK_X86 1
#endif

namespace oxk {

bool has_avx2();

float dot_f32_scalar(const float *a, const float *b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

#ifdef OXK_X86
float dot_f32_avx2(const float *a, const float *b, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t i = 0;
    const size_t block = 32;
    for (; i + block <= n; i += block) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

int rms_norm_f32_avx2(const float *x, const float *weight, size_t n, float eps, float *out) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    float sum_sq = _mm_cvtss_f32(lo);
    for (; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
    const float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(n) + eps);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        const __m256 wv = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(xv, wv), _mm256_set1_ps(scale)));
    }
    for (; i < n; ++i) {
        out[i] = x[i] * weight[i] * scale;
    }
    return 0;
}
#endif

float dot_f32(const float *a, const float *b, size_t n) {
#ifdef OXK_X86
    if (has_avx2()) {
        return dot_f32_avx2(a, b, n);
    }
#endif
    return dot_f32_scalar(a, b, n);
}

int rms_norm_f32(const float *x, const float *weight, size_t n, float eps, float *out) {
#ifdef OXK_X86
    if (has_avx2()) {
        return rms_norm_f32_avx2(x, weight, n, eps, out);
    }
#endif
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
    const float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(n) + eps);
    for (size_t i = 0; i < n; ++i) {
        out[i] = x[i] * weight[i] * scale;
    }
    return 0;
}

} // namespace oxk
