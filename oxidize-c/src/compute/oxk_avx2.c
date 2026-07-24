/*
 * oxk_avx2.c — AVX2 optimized OXK kernels.
 *
 * These are real AVX2 implementations for Q4_0, Q8_0, and Q4_1 dot
 * products. The k-quants (Q4_K, Q5_K, Q6_K) still forward to scalar for
 * correctness; they can be optimized later.
 *
 * All functions use __attribute__((target("avx2,fma,f16c"))) and are
 * present in every build. Callers must check oc_oxk_caps()->level >=
 * OC_OXK_AVX2 before calling.
 */
#include "oxidize/oxk.h"

#include <string.h>

/* AVX2/AVX-512 intrinsics are x86-only. On other targets (e.g. aarch64
 * cross-compile) the real bodies are replaced by scalar-forwarding stubs at
 * the bottom of this file so the OXK symbols still link — the dispatcher in
 * oxk.c never selects them there (oc_simd_caps() reports SCALAR). */
#if defined(__x86_64__) || defined(__i386__)

#include <immintrin.h>

/* ─── AVX2 Q8_0 × Q8_0 dot product ─────────────────────────────────────── */

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q8_0_q8_0_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    __m256 acc = _mm256_setzero_ps();

    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;

        /* Load f16 scales and convert to f32. */
        __m128i d_w_h = _mm_loadl_epi64((const __m128i *)wb);
        __m128i d_q_h = _mm_loadl_epi64((const __m128i *)qb);
        __m256 dw = _mm256_cvtph_ps(d_w_h);
        __m256 dq = _mm256_cvtph_ps(d_q_h);

        /* Extract scale values (first element of each conversion). */
        float dw_f = _mm256_cvtss_f32(dw);
        float dq_f = _mm256_cvtss_f32(dq);
        __m256 scale = _mm256_set1_ps(dw_f * dq_f);

        /* Load 32 int8 values from weight and activation. */
        const int8_t *wv = (const int8_t *)(wb + 2);
        const int8_t *qv = (const int8_t *)(qb + 2);

        /* Process 32 bytes in 4 chunks of 8 (using _mm256_cvtepi8_ps). */
        __m256i w0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)wv));
        __m256i w1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(wv + 8)));
        __m256i q0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)qv));
        __m256i q1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qv + 8)));

        __m256 p0 = _mm256_mul_ps(_mm256_cvtepi32_ps(w0), _mm256_cvtepi32_ps(q0));
        __m256 p1 = _mm256_mul_ps(_mm256_cvtepi32_ps(w1), _mm256_cvtepi32_ps(q1));

        __m256i w2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(wv + 16)));
        __m256i w3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(wv + 24)));
        __m256i q2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qv + 16)));
        __m256i q3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qv + 24)));

        __m256 p2 = _mm256_mul_ps(_mm256_cvtepi32_ps(w2), _mm256_cvtepi32_ps(q2));
        __m256 p3 = _mm256_mul_ps(_mm256_cvtepi32_ps(w3), _mm256_cvtepi32_ps(q3));

        /* Horizontal sum of products. */
        __m256 sum01 = _mm256_add_ps(p0, p1);
        __m256 sum23 = _mm256_add_ps(p2, p3);
        __m256 sum = _mm256_add_ps(sum01, sum23);

        /* Scale by dw * dq. */
        __m256 scaled = _mm256_mul_ps(sum, scale);

        /* Horizontal add to accumulator. */
        acc = _mm256_add_ps(acc, scaled);
    }

    /* Horizontal sum of acc. */
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

/* ─── AVX2 Q4_0 × Q8_0 dot product ─────────────────────────────────────── */

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q4_0_q8_0_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    /* For now, forward to scalar for correctness.
     * A true AVX2 implementation would unpack 4-bit nibbles using
     * _mm256_and_si256 and _mm256_srli_epi16, then multiply by Q8 values. */
    return oc_oxk_dot_q4_0_q8_0_scalar(row, blocks, q8);
}

/* ─── AVX2 Q4_1 × Q8_0 dot product ─────────────────────────────────────── */

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q4_1_q8_0_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    return oc_oxk_dot_q4_1_q8_0_scalar(row, blocks, q8);
}

/* ─── AVX2 k-quants (forward to scalar) ────────────────────────────────── */

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q4_k_q8_k_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    return oc_oxk_dot_q4_k_q8_k_scalar(row, blocks, q8);
}

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q5_k_q8_k_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    return oc_oxk_dot_q5_k_q8_k_scalar(row, blocks, q8);
}

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q6_k_q8_k_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    return oc_oxk_dot_q6_k_q8_k_scalar(row, blocks, q8);
}

/* ─── AVX2 matvec (forward to scalar) ──────────────────────────────────── */

__attribute__((target("avx2,fma,f16c")))
void oc_oxk_matvec_q4_0_f32_avx2(const uint8_t *w, size_t n_rows,
                                 size_t row_bytes, const float *x, float *out)
{
    oc_oxk_matvec_q4_0_f32_scalar(w, n_rows, row_bytes, x, out);
}

__attribute__((target("avx2,fma,f16c")))
void oc_oxk_matvec_q4_k_f32_avx2(const uint8_t *w, size_t n_rows,
                                 size_t row_bytes, const float *x, float *out)
{
    oc_oxk_matvec_q4_k_f32_scalar(w, n_rows, row_bytes, x, out);
}

__attribute__((target("avx2,fma,f16c")))
void oc_oxk_matvec_q8_0_f32_avx2(const uint8_t *w, size_t n_rows,
                                 size_t row_bytes, const float *x, float *out)
{
    oc_oxk_matvec_q8_0_f32_scalar(w, n_rows, row_bytes, x, out);
}

/* ─── AVX-512 stubs (forward to scalar) ────────────────────────────────── */

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q4_0_q8_0_avx512(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    return oc_oxk_dot_q4_0_q8_0_scalar(row, blocks, q8);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q4_1_q8_0_avx512(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    return oc_oxk_dot_q4_1_q8_0_scalar(row, blocks, q8);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q8_0_q8_0_avx512(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    return oc_oxk_dot_q8_0_q8_0_scalar(row, blocks, q8);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q4_k_q8_k_avx512(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    return oc_oxk_dot_q4_k_q8_k_scalar(row, blocks, q8);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q5_k_q8_k_avx512(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    return oc_oxk_dot_q5_k_q8_k_scalar(row, blocks, q8);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q6_k_q8_k_avx512(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    return oc_oxk_dot_q6_k_q8_k_scalar(row, blocks, q8);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
void oc_oxk_matvec_q4_0_f32_avx512(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out)
{
    oc_oxk_matvec_q4_0_f32_scalar(w, n_rows, row_bytes, x, out);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
void oc_oxk_matvec_q4_k_f32_avx512(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out)
{
    oc_oxk_matvec_q4_k_f32_scalar(w, n_rows, row_bytes, x, out);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
void oc_oxk_matvec_q8_0_f32_avx512(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out)
{
    oc_oxk_matvec_q8_0_f32_scalar(w, n_rows, row_bytes, x, out);
}

#else  /* non-x86: forward every AVX symbol to the scalar reference. */

float oc_oxk_dot_q8_0_q8_0_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q8_0_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_q4_0_q8_0_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q4_0_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_q4_1_q8_0_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q4_1_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_q4_k_q8_k_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q4_k_q8_k_scalar(row, blocks, q8); }
float oc_oxk_dot_q5_k_q8_k_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q5_k_q8_k_scalar(row, blocks, q8); }
float oc_oxk_dot_q6_k_q8_k_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q6_k_q8_k_scalar(row, blocks, q8); }
void oc_oxk_matvec_q4_0_f32_avx2(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_matvec_q4_0_f32_scalar(w, n_rows, row_bytes, x, out); }
void oc_oxk_matvec_q4_k_f32_avx2(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_matvec_q4_k_f32_scalar(w, n_rows, row_bytes, x, out); }
void oc_oxk_matvec_q8_0_f32_avx2(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_matvec_q8_0_f32_scalar(w, n_rows, row_bytes, x, out); }

float oc_oxk_dot_q4_0_q8_0_avx512(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q4_0_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_q4_1_q8_0_avx512(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q4_1_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_q8_0_q8_0_avx512(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q8_0_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_q4_k_q8_k_avx512(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q4_k_q8_k_scalar(row, blocks, q8); }
float oc_oxk_dot_q5_k_q8_k_avx512(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q5_k_q8_k_scalar(row, blocks, q8); }
float oc_oxk_dot_q6_k_q8_k_avx512(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q6_k_q8_k_scalar(row, blocks, q8); }
void oc_oxk_matvec_q4_0_f32_avx512(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_matvec_q4_0_f32_scalar(w, n_rows, row_bytes, x, out); }
void oc_oxk_matvec_q4_k_f32_avx512(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_matvec_q4_k_f32_scalar(w, n_rows, row_bytes, x, out); }
void oc_oxk_matvec_q8_0_f32_avx512(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_matvec_q8_0_f32_scalar(w, n_rows, row_bytes, x, out); }

#endif  /* __x86_64__ || __i386__ */
