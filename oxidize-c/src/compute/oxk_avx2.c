/*
 * oxk_avx2.c — AVX2 optimized OXK kernels.
 *
 * Real AVX2 implementations for Q8_0 and Q4_K; Q4_0/Q4_1/Q5_K/Q6_K still
 * forward to scalar. Q4_K matters most in practice — it is what most models
 * ship as — and running it on the scalar path made Q4_K_M slower than the
 * physically larger Q8_0.
 *
 * Every vectorized kernel here must be BIT-EXACT against its scalar
 * counterpart, not merely close; test_oxk_avx2_parity.c enforces that with a
 * raw float comparison.
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

/* ─── AVX2 Q4_K × Q8_K dot product ─────────────────────────────────────── */

/* Horizontal sum of eight int32 lanes. */
__attribute__((target("avx2,f16c")))
static inline int32_t hsum_i32_8(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/* Q4_K is the format most models ship in, so this is the kernel that decides
 * throughput in practice. It was forwarding to scalar, which is why Q4_K_M ran
 * slower than the larger Q8_0 model.
 *
 * The nibble products are computed with _mm256_maddubs_epi16: unsigned first
 * operand (nibbles are 0..15), signed second (the int8 activation), giving
 * int16 pairwise sums, which _mm256_madd_epi16 against ones widens to int32.
 * Everything stays integer until one multiply-add per block, exactly as the
 * scalar reference does since it was restructured to accumulate pos/min_acc —
 * so this is bit-exact against it, not merely close. */
__attribute__((target("avx2,f16c")))
float oc_oxk_dot_q4_k_q8_k_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    const __m256i lownib = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    float sum = 0.0f;

    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_K_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_K_SIZE;
        const float dw   = oc_oxk_f16_le_to_f32(wb);
        const float dmin = oc_oxk_f16_le_to_f32(wb + 2);
        const uint8_t *scales = wb + 4;
        const uint8_t *qs     = wb + 16;
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + 256;

        /* Scale each group's lane-wise products in vector form and reduce ONCE
         * per block. Reducing per group instead costs eight horizontal sums
         * per block, and a horizontal sum is a serial dependency chain that
         * stalls the pipeline — it was the dominant cost in the first version
         * of this kernel.
         *
         * Summation order still matches the scalar reference exactly: the
         * lane totals are integers, so regrouping them cannot change the
         * result the way it would in floating point. That is what keeps this
         * bit-exact while being reassociated. */
        __m256i pos_v = _mm256_setzero_si256();
        int32_t min_acc = 0;
        for (int gp = 0; gp < 4; gp++) {
            uint8_t sc1, m1, sc2, m2;
            oc_oxk_get_scale_min_k4((unsigned)(gp * 2),     scales, &sc1, &m1);
            oc_oxk_get_scale_min_k4((unsigned)(gp * 2 + 1), scales, &sc2, &m2);

            /* 32 packed bytes = 64 nibbles = both halves of this group. */
            const __m256i packed = _mm256_loadu_si256((const __m256i *)(qs + gp * 32));
            const __m256i nib_lo = _mm256_and_si256(packed, lownib);
            const __m256i nib_hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), lownib);

            const __m256i a_lo = _mm256_loadu_si256((const __m256i *)(q8v + gp * 64));
            const __m256i a_hi = _mm256_loadu_si256((const __m256i *)(q8v + gp * 64 + 32));

            const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(nib_lo, a_lo), ones16);
            const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(nib_hi, a_hi), ones16);

            pos_v = _mm256_add_epi32(pos_v,
                        _mm256_mullo_epi32(p1, _mm256_set1_epi32((int32_t)sc1)));
            pos_v = _mm256_add_epi32(pos_v,
                        _mm256_mullo_epi32(p2, _mm256_set1_epi32((int32_t)sc2)));

            const int32_t bs1 = oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4)) +
                                oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4 + 1));
            const int32_t bs2 = oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4 + 2)) +
                                oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4 + 3));

            min_acc += (int32_t)m1 * bs1 + (int32_t)m2 * bs2;
        }
        sum += dw * dq * (float)hsum_i32_8(pos_v) - dmin * dq * (float)min_acc;
    }
    return sum;
}

/* ─── AVX2 k-quants (forward to scalar) ────────────────────────────────── */

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
/* AVX-512 hosts run the AVX2 kernel.
 *
 * A VNNI version (dpbusd, which folds the maddubs+madd pair into a single
 * instruction) was written and measured: bit-exact, and about 6% faster on a
 * Cascade Lake Xeon. It is not kept because clang rejects the intrinsic even
 * with a matching target attribute -- it is stricter than GCC about inlining
 * intrinsics whose own target string differs from the caller's -- and a 6%
 * gain on one compiler is not worth failing the clang build. Q4_K is no
 * longer the bottleneck regardless; what remains of the decode gap is in the
 * still-scalar attention, norm and RoPE code, not here.
 *
 * The point of this entry point is that it no longer forwards to *scalar*,
 * which is what it did before and which made an AVX-512 host run the slowest
 * kernel available. */
__attribute__((target("avx512bw,avx512dq")))
float oc_oxk_dot_q4_k_q8_k_avx512(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    return oc_oxk_dot_q4_k_q8_k_avx2(row, blocks, q8);
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
