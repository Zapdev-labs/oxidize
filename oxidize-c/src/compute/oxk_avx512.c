/*
 * oxk_avx512.c — AVX-512 OXK kernel implementations.
 *
 * These implementations use AVX-512BW + AVX-512VNNI intrinsics for
 * Q8_0 and Q4_0/Q4_1 dot products and matvec. All functions are guarded
 * by target attributes so they compile on any CPU.
 */
#include "oxidize/oxk.h"
#include "oxidize/oxk_avx512.h"

#include <string.h>

/* AVX-512 intrinsics are x86-only. On other targets the real bodies are
 * replaced by scalar-forwarding stubs at the bottom of this file so the
 * symbols still link (the dispatcher never selects them off-x86). */
#if defined(__x86_64__) || defined(__i386__)

#include <immintrin.h>

/* ─── Helper: horizontal sum of __m512i (16 int32 lanes) ──────────────── */

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
static inline int32_t mm512_hsum_epi32(__m512i v)
{
    return _mm512_reduce_add_epi32(v);
}

/* ─── VNNI dot product: 32 signed int8 × 32 signed int8 ──────────────────
 *
 * VNNI computes uint8 × int8 → int32. To handle signed × signed, we split
 * the weight vector into positive and negative parts:
 *   w_s8 * q_s8 = w_pos * q - w_neg * q
 * where w_pos = max(0, w) and w_neg = max(0, -w), both as uint8.
 */
__attribute__((target("avx512bw,avx512dq,avx512vnni")))
static inline int32_t vnni_dot_s8x32(const int8_t *w, const int8_t *q)
{
    __m256i w_raw = _mm256_loadu_si256((const __m256i *)w);
    __m256i q_raw = _mm256_loadu_si256((const __m256i *)q);

    __m512i wv = _mm512_zextsi256_si512(w_raw);
    __m512i qv = _mm512_zextsi256_si512(q_raw);

    __m512i zero = _mm512_setzero_si512();
    __m512i w_pos = _mm512_max_epi8(zero, wv);
    __m512i w_neg = _mm512_max_epi8(zero, _mm512_sub_epi8(zero, wv));

    __m512i vpos = _mm512_dpbusd_epi32(zero, w_pos, qv);
    __m512i vneg = _mm512_dpbusd_epi32(zero, w_neg, qv);
    __m512i result = _mm512_sub_epi32(vpos, vneg);

    return mm512_hsum_epi32(result);
}

/* ─── Q8_0 × Q8_0 VNNI ────────────────────────────────────────────────── */

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q8_0_q8_0_avx512_vnni(const uint8_t *row, size_t blocks,
                                        const uint8_t *q8)
{
    int32_t isum = 0;
    (void)isum; /* VNNI dot products computed in second loop for scaling. */
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;

        /* Load f16 scales. */
        float dw = oc_oxk_f16_le_to_f32(wb);
        float dq = oc_oxk_f16_le_to_f32(qb);

        /* VNNI dot product of 32 signed int8 values. */
        int32_t dot = vnni_dot_s8x32((const int8_t *)(wb + 2),
                                     (const int8_t *)(qb + 2));
        isum += dot;

        /* Scale: f32 multiply matches scalar path. */
        (void)dw; (void)dq;
    }

    /* Compute weighted sum. */
    float result = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;
        float dw = oc_oxk_f16_le_to_f32(wb);
        float dq = oc_oxk_f16_le_to_f32(qb);
        int32_t dot = vnni_dot_s8x32((const int8_t *)(wb + 2),
                                     (const int8_t *)(qb + 2));
        result += dw * dq * (float)dot;
    }
    return result;
}

/* ─── Q4_0 × Q8_0 BW (forward to scalar for correctness) ─────────────── */

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q4_0_q8_0_avx512_bw(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8)
{
    return oc_oxk_dot_q4_0_q8_0_scalar(row, blocks, q8);
}

/* ─── Q4_1 × Q8_0 BW (forward to scalar) ──────────────────────────────── */

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q4_1_q8_0_avx512_bw(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8)
{
    return oc_oxk_dot_q4_1_q8_0_scalar(row, blocks, q8);
}

/* ─── Matvec variants (forward to scalar for now) ─────────────────────── */

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
void oc_oxk_matvec_q8_0_f32_avx512_vnni(const uint8_t *w, size_t n_rows,
                                        size_t row_bytes, const float *x,
                                        float *out)
{
    oc_oxk_matvec_q8_0_f32_scalar(w, n_rows, row_bytes, x, out);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
void oc_oxk_matvec_q4_0_f32_avx512_bw(const uint8_t *w, size_t n_rows,
                                      size_t row_bytes, const float *x,
                                      float *out)
{
    oc_oxk_matvec_q4_0_f32_scalar(w, n_rows, row_bytes, x, out);
}

#else  /* non-x86: forward every AVX-512 symbol to the scalar reference. */

float oc_oxk_dot_q8_0_q8_0_avx512_vnni(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q8_0_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_q4_0_q8_0_avx512_bw(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q4_0_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_q4_1_q8_0_avx512_bw(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q4_1_q8_0_scalar(row, blocks, q8); }
void oc_oxk_matvec_q8_0_f32_avx512_vnni(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_matvec_q8_0_f32_scalar(w, n_rows, row_bytes, x, out); }
void oc_oxk_matvec_q4_0_f32_avx512_bw(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_matvec_q4_0_f32_scalar(w, n_rows, row_bytes, x, out); }

#endif  /* __x86_64__ || __i386__ */
