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

__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
static inline int32_t hsum_i32_8_avx512(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/* Vectorized replacement for the per-(block, activation) scalar loop that
 * folds the activation's 16 block sums against the row's per-group scales.
 *
 * That loop was the real cost of these kernels: 16 scalar iterations of
 * load / sign-extend / multiply / add against just FOUR vpdpbusd of actual
 * work per block, so the epilogue outweighed the dot by an order of
 * magnitude. vpmaddwd does the same reduction in one instruction — and
 * because the whole term is exact int32 arithmetic, reassociating it cannot
 * change the result, so the float accumulation below stays bit-identical to
 * the scalar reference.
 *
 * `w16` holds the 16 per-group weights already widened to int16, laid out to
 * match the bsums: for the 16-group types (Q6_K, Q3_K, Q2_K) that is one
 * weight per group; for Q4_K's 8 groups of 32 each weight is duplicated, so
 * lane k of the result is w_k * (bsum_2k + bsum_2k+1). */
__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
static inline int32_t fold_bsums(__m256i w16, const uint8_t *bsums)
{
    const __m256i bs = _mm256_loadu_si256((const __m256i *)bsums);
    return hsum_i32_8_avx512(_mm256_madd_epi16(w16, bs));
}






/* ─── Helper: horizontal sum of __m512i (16 int32 lanes) ──────────────── */

__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
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
__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
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

__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
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

__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
float oc_oxk_dot_q4_0_q8_0_avx512_bw(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8)
{
    return oc_oxk_dot_q4_0_q8_0_scalar(row, blocks, q8);
}

/* ─── Q4_1 × Q8_0 BW (forward to scalar) ──────────────────────────────── */

__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
float oc_oxk_dot_q4_1_q8_0_avx512_bw(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8)
{
    return oc_oxk_dot_q4_1_q8_0_scalar(row, blocks, q8);
}

/* ─── Matvec variants (forward to scalar for now) ─────────────────────── */

__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
void oc_oxk_matvec_q8_0_f32_avx512_vnni(const uint8_t *w, size_t n_rows,
                                        size_t row_bytes, const float *x,
                                        float *out)
{
    oc_oxk_matvec_q8_0_f32_scalar(w, n_rows, row_bytes, x, out);
}

__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
void oc_oxk_matvec_q4_0_f32_avx512_bw(const uint8_t *w, size_t n_rows,
                                      size_t row_bytes, const float *x,
                                      float *out)
{
    oc_oxk_matvec_q4_0_f32_scalar(w, n_rows, row_bytes, x, out);
}

/* ─── VNNI prepared-Q4_K multi-activation dot ─────────────────────────────
 *
 * Pointer math must mirror oxk.c::q4k_prep_ptrs exactly; the layout is part
 * of oc_oxk_q4_k_prep_row()'s contract and test_oxk_gguf_layout.c catches
 * drift. */
__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
void oc_oxk_dot_q4_k_prepped_multi_avx512(const void *prep, size_t blocks,
                                          const uint8_t *acts,
                                          size_t act_stride, size_t n_act,
                                          float *out)
{
    const float   *d     = (const float *)prep;
    const float   *dmin  = d + blocks;
    const uint8_t *codes = (const uint8_t *)(d + 2 * blocks);
    const uint8_t *sc    = codes + blocks * 256;
    const uint8_t *mn    = sc + blocks * 8;

    /* Lane j of chunk gp holds sub-group 2gp (lanes 0-7) / 2gp+1 (8-15). */
    const __m512i idx0 = _mm512_set_epi32(1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0);
    const __m512i idx1 = _mm512_set_epi32(3,3,3,3,3,3,3,3, 2,2,2,2,2,2,2,2);
    const __m512i idx2 = _mm512_set_epi32(5,5,5,5,5,5,5,5, 4,4,4,4,4,4,4,4);
    const __m512i idx3 = _mm512_set_epi32(7,7,7,7,7,7,7,7, 6,6,6,6,6,6,6,6);

    size_t a0 = 0;
    for (; a0 + 4 <= n_act; a0 += 4) {
        const uint8_t *act0 = acts + (a0 + 0) * act_stride;
        const uint8_t *act1 = acts + (a0 + 1) * act_stride;
        const uint8_t *act2 = acts + (a0 + 2) * act_stride;
        const uint8_t *act3 = acts + (a0 + 3) * act_stride;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *cb  = codes + b * 256;
            const uint8_t *scb = sc + b * 8;
            const uint8_t *mnb = mn + b * 8;

            /* 16-byte load spans this block's sc and the mn array head /
             * next block's sc — always inside the prep scratch. Only the
             * low 8 bytes are used. */
            __m512i scz = _mm512_cvtepu8_epi32(
                _mm_loadu_si128((const __m128i *)scb));

            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            const __m512i idx[4] = { idx0, idx1, idx2, idx3 };
            for (int gp = 0; gp < 4; gp++) {
                const __m512i cz = _mm512_loadu_si512(
                    (const void *)(cb + (size_t)gp * 64));
                const __m512i scv = _mm512_permutexvar_epi32(idx[gp], scz);
                const size_t qoff = b * OC_OXK_BLOCK_Q8_K_SIZE + 4 +
                                    (size_t)gp * 64;
                __m512i p;
                p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), cz,
                        _mm512_loadu_si512((const void *)(act0 + qoff)));
                acc0 = _mm512_add_epi32(acc0, _mm512_mullo_epi32(p, scv));
                p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), cz,
                        _mm512_loadu_si512((const void *)(act1 + qoff)));
                acc1 = _mm512_add_epi32(acc1, _mm512_mullo_epi32(p, scv));
                p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), cz,
                        _mm512_loadu_si512((const void *)(act2 + qoff)));
                acc2 = _mm512_add_epi32(acc2, _mm512_mullo_epi32(p, scv));
                p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), cz,
                        _mm512_loadu_si512((const void *)(act3 + qoff)));
                acc3 = _mm512_add_epi32(acc3, _mm512_mullo_epi32(p, scv));
            }

            const uint8_t *qb[4] = {
                act0 + b * OC_OXK_BLOCK_Q8_K_SIZE,
                act1 + b * OC_OXK_BLOCK_Q8_K_SIZE,
                act2 + b * OC_OXK_BLOCK_Q8_K_SIZE,
                act3 + b * OC_OXK_BLOCK_Q8_K_SIZE,
            };
            int32_t pos[4] = {
                _mm512_reduce_add_epi32(acc0), _mm512_reduce_add_epi32(acc1),
                _mm512_reduce_add_epi32(acc2), _mm512_reduce_add_epi32(acc3),
            };
            float *sums[4] = { &s0, &s1, &s2, &s3 };
            /* Duplicate each of the 8 mins so lane k of the fold is
             * mn_k * (bsum_2k + bsum_2k+1) — Q4_K's groups are 32 wide and
             * so span two of the activation's 16-element block sums. */
            const __m256i mn16 = _mm256_cvtepu8_epi16(
                _mm_shuffle_epi8(_mm_loadl_epi64((const __m128i *)mnb),
                    _mm_setr_epi8(0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7)));
            for (int a = 0; a < 4; a++) {
                const uint8_t *bsums = qb[a] + 4 + OC_OXK_QK_K;
                const int32_t min_acc = fold_bsums(mn16, bsums);
                float dq;
                memcpy(&dq, qb[a], 4);
                *sums[a] += d[b] * dq * (float)pos[a]
                          - dmin[b] * dq * (float)min_acc;
            }
        }
        out[a0 + 0] = s0; out[a0 + 1] = s1;
        out[a0 + 2] = s2; out[a0 + 3] = s3;
    }
    for (; a0 < n_act; a0++) {
        out[a0] = oc_oxk_dot_q4_k_prepped(prep, blocks,
                                          acts + a0 * act_stride);
    }
}

/* ─── VNNI prepared-Q6_K multi-activation dot ─────────────────────────────
 *
 * Layout must mirror oxk.c::q6k_prep_ptrs. Each 64-code chunk covers four
 * 16-element scale groups: dpbusd lane i sums elements 4i..4i+3, so lanes
 * 0-3 belong to group 4gp, 4-7 to 4gp+1, 8-11 to 4gp+2, 12-15 to 4gp+3. */
__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
void oc_oxk_dot_q6_k_prepped_multi_avx512(const void *prep, size_t blocks,
                                          const uint8_t *acts,
                                          size_t act_stride, size_t n_act,
                                          float *out)
{
    const float   *d     = (const float *)prep;
    const uint8_t *codes = (const uint8_t *)(d + blocks);
    const int8_t  *sc    = (const int8_t *)(codes + blocks * 256);

    const __m512i idx0 = _mm512_set_epi32(3,3,3,3, 2,2,2,2, 1,1,1,1, 0,0,0,0);
    const __m512i idx1 = _mm512_set_epi32(7,7,7,7, 6,6,6,6, 5,5,5,5, 4,4,4,4);
    const __m512i idx2 = _mm512_set_epi32(11,11,11,11, 10,10,10,10,
                                          9,9,9,9, 8,8,8,8);
    const __m512i idx3 = _mm512_set_epi32(15,15,15,15, 14,14,14,14,
                                          13,13,13,13, 12,12,12,12);

    size_t a0 = 0;
    for (; a0 + 4 <= n_act; a0 += 4) {
        const uint8_t *act[4] = {
            acts + (a0 + 0) * act_stride, acts + (a0 + 1) * act_stride,
            acts + (a0 + 2) * act_stride, acts + (a0 + 3) * act_stride,
        };
        float sums[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *cb  = codes + b * 256;
            const int8_t  *scb = sc + b * 16;
            /* Signed 8 → 32 widen of all 16 group scales. */
            const __m512i scz = _mm512_cvtepi8_epi32(
                _mm_loadu_si128((const __m128i *)scb));

            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            const __m512i idx[4] = { idx0, idx1, idx2, idx3 };
            for (int gp = 0; gp < 4; gp++) {
                const __m512i cz = _mm512_loadu_si512(
                    (const void *)(cb + (size_t)gp * 64));
                const __m512i scv = _mm512_permutexvar_epi32(idx[gp], scz);
                const size_t qoff = b * OC_OXK_BLOCK_Q8_K_SIZE + 4 +
                                    (size_t)gp * 64;
                __m512i p;
                p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), cz,
                        _mm512_loadu_si512((const void *)(act[0] + qoff)));
                acc0 = _mm512_add_epi32(acc0, _mm512_mullo_epi32(p, scv));
                p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), cz,
                        _mm512_loadu_si512((const void *)(act[1] + qoff)));
                acc1 = _mm512_add_epi32(acc1, _mm512_mullo_epi32(p, scv));
                p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), cz,
                        _mm512_loadu_si512((const void *)(act[2] + qoff)));
                acc2 = _mm512_add_epi32(acc2, _mm512_mullo_epi32(p, scv));
                p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), cz,
                        _mm512_loadu_si512((const void *)(act[3] + qoff)));
                acc3 = _mm512_add_epi32(acc3, _mm512_mullo_epi32(p, scv));
            }

            const int32_t pos[4] = {
                _mm512_reduce_add_epi32(acc0), _mm512_reduce_add_epi32(acc1),
                _mm512_reduce_add_epi32(acc2), _mm512_reduce_add_epi32(acc3),
            };
            const __m256i sc16 = _mm256_cvtepi8_epi16(
                _mm_loadu_si128((const __m128i *)scb));
            for (int a = 0; a < 4; a++) {
                const uint8_t *qb = act[a] + b * OC_OXK_BLOCK_Q8_K_SIZE;
                const int32_t minc = fold_bsums(sc16, qb + 4 + OC_OXK_QK_K);
                float dq;
                memcpy(&dq, qb, 4);
                sums[a] += d[b] * dq * (float)(pos[a] - 32 * minc);
            }
        }
        out[a0 + 0] = sums[0]; out[a0 + 1] = sums[1];
        out[a0 + 2] = sums[2]; out[a0 + 3] = sums[3];
    }
    for (; a0 < n_act; a0++) {
        out[a0] = oc_oxk_dot_q6_k_prepped(prep, blocks,
                                          acts + a0 * act_stride);
    }
}


/* Single-activation VNNI form of the bodies above.
 *
 * Decode has exactly one activation per matmul, so it never reaches the
 * 4-wide loop and used to drop to the scalar prepared dot — 256 scalar
 * multiply-adds per block against four vpdpbusd. Sharing the same prepared
 * layout means decode and prefill still agree bit-for-bit on the integer
 * terms; only the float accumulation order (identical here) matters. */
__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
static inline float dot_q3q6_prepped_one(const void *prep, size_t blocks,
                                         const uint8_t *act, int32_t offset)
{
    const float   *d     = (const float *)prep;
    const uint8_t *codes = (const uint8_t *)(d + blocks);
    const int8_t  *sc    = (const int8_t *)(codes + blocks * OC_OXK_QK_K);

    const __m512i idx[4] = {
        _mm512_set_epi32(3,3,3,3, 2,2,2,2, 1,1,1,1, 0,0,0,0),
        _mm512_set_epi32(7,7,7,7, 6,6,6,6, 5,5,5,5, 4,4,4,4),
        _mm512_set_epi32(11,11,11,11, 10,10,10,10, 9,9,9,9, 8,8,8,8),
        _mm512_set_epi32(15,15,15,15, 14,14,14,14, 13,13,13,13, 12,12,12,12),
    };

    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *cb  = codes + b * OC_OXK_QK_K;
        const int8_t  *scb = sc + b * 16;
        const __m512i scz = _mm512_cvtepi8_epi32(
            _mm_loadu_si128((const __m128i *)scb));
        __m512i acc = _mm512_setzero_si512();
        for (int gp = 0; gp < 4; gp++) {
            const __m512i cz = _mm512_loadu_si512(
                (const void *)(cb + (size_t)gp * 64));
            const __m512i p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), cz,
                _mm512_loadu_si512((const void *)(act +
                    b * OC_OXK_BLOCK_Q8_K_SIZE + 4 + (size_t)gp * 64)));
            acc = _mm512_add_epi32(acc,
                _mm512_mullo_epi32(p, _mm512_permutexvar_epi32(idx[gp], scz)));
        }
        const uint8_t *qb = act + b * OC_OXK_BLOCK_Q8_K_SIZE;
        const int32_t off = fold_bsums(
            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)scb)),
            qb + 4 + OC_OXK_QK_K);
        float dq;
        memcpy(&dq, qb, 4);
        sum += d[b] * dq * (float)(_mm512_reduce_add_epi32(acc) -
                                   offset * off);
    }
    return sum;
}

__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
float oc_oxk_dot_q3_k_prepped_avx512(const void *prep, size_t blocks,
                                     const uint8_t *act)
{
    return dot_q3q6_prepped_one(prep, blocks, act, 4);
}

__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
float oc_oxk_dot_q6_k_prepped_avx512(const void *prep, size_t blocks,
                                     const uint8_t *act)
{
    return dot_q3q6_prepped_one(prep, blocks, act, 32);
}

__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
float oc_oxk_dot_q2_k_prepped_avx512(const void *prep, size_t blocks,
                                     const uint8_t *act)
{
    const float   *d     = (const float *)prep;
    const float   *dmin  = d + blocks;
    const uint8_t *codes = (const uint8_t *)(d + 2 * blocks);
    const uint8_t *sc    = codes + blocks * OC_OXK_QK_K;
    const uint8_t *mn    = sc + blocks * 16;

    const __m512i idx[4] = {
        _mm512_set_epi32(3,3,3,3, 2,2,2,2, 1,1,1,1, 0,0,0,0),
        _mm512_set_epi32(7,7,7,7, 6,6,6,6, 5,5,5,5, 4,4,4,4),
        _mm512_set_epi32(11,11,11,11, 10,10,10,10, 9,9,9,9, 8,8,8,8),
        _mm512_set_epi32(15,15,15,15, 14,14,14,14, 13,13,13,13, 12,12,12,12),
    };

    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *cb  = codes + b * OC_OXK_QK_K;
        const uint8_t *scb = sc + b * 16;
        const uint8_t *mnb = mn + b * 16;
        const __m512i scz = _mm512_cvtepu8_epi32(
            _mm_loadu_si128((const __m128i *)scb));
        __m512i acc = _mm512_setzero_si512();
        for (int gp = 0; gp < 4; gp++) {
            const __m512i cz = _mm512_loadu_si512(
                (const void *)(cb + (size_t)gp * 64));
            const __m512i p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), cz,
                _mm512_loadu_si512((const void *)(act +
                    b * OC_OXK_BLOCK_Q8_K_SIZE + 4 + (size_t)gp * 64)));
            acc = _mm512_add_epi32(acc,
                _mm512_mullo_epi32(p, _mm512_permutexvar_epi32(idx[gp], scz)));
        }
        const uint8_t *qb = act + b * OC_OXK_BLOCK_Q8_K_SIZE;
        const int32_t min_acc = fold_bsums(
            _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)mnb)),
            qb + 4 + OC_OXK_QK_K);
        float dq;
        memcpy(&dq, qb, 4);
        sum += dq * (d[b] * (float)_mm512_reduce_add_epi32(acc) -
                     dmin[b] * (float)min_acc);
    }
    return sum;
}

/* ─── VNNI Q3_K / Q2_K prepared-row multi dots ───────────────────────────
 *
 * Both are 16 groups of 16 weights, so one 64-byte code load spans FOUR
 * groups — lanes 0-3 of the vpdpbusd result belong to group 4gp, lanes 4-7 to
 * 4gp+1, and so on. That is the only structural difference from the Q4_K
 * kernel above, whose groups are 32 wide and so span two per load.
 *
 * Q3_K reuses the Q6_K prepared layout, so the body is shared and the code
 * offset (-4 for Q3_K, -32 for Q6_K) is a parameter the compiler folds. */
__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
static inline void dot_q3q6_prepped_multi_body(const void *prep, size_t blocks,
                                               const uint8_t *acts,
                                               size_t act_stride, size_t n_act,
                                               float *out, int32_t offset)
{
    const float   *d     = (const float *)prep;
    const uint8_t *codes = (const uint8_t *)(d + blocks);
    const int8_t  *sc    = (const int8_t *)(codes + blocks * OC_OXK_QK_K);

    /* Lane j of chunk gp holds group 4gp + j/4. */
    const __m512i idx0 = _mm512_set_epi32(3,3,3,3, 2,2,2,2, 1,1,1,1, 0,0,0,0);
    const __m512i idx1 = _mm512_set_epi32(7,7,7,7, 6,6,6,6, 5,5,5,5, 4,4,4,4);
    const __m512i idx2 = _mm512_set_epi32(11,11,11,11, 10,10,10,10,
                                          9,9,9,9, 8,8,8,8);
    const __m512i idx3 = _mm512_set_epi32(15,15,15,15, 14,14,14,14,
                                          13,13,13,13, 12,12,12,12);
    const __m512i idx[4] = { idx0, idx1, idx2, idx3 };

    size_t a0 = 0;
    for (; a0 + 4 <= n_act; a0 += 4) {
        const uint8_t *act[4] = {
            acts + (a0 + 0) * act_stride, acts + (a0 + 1) * act_stride,
            acts + (a0 + 2) * act_stride, acts + (a0 + 3) * act_stride,
        };
        float sums[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *cb  = codes + b * OC_OXK_QK_K;
            const int8_t  *scb = sc + b * 16;
            const __m512i scz = _mm512_cvtepi8_epi32(
                _mm_loadu_si128((const __m128i *)scb));

            __m512i acc[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(),
                               _mm512_setzero_si512(), _mm512_setzero_si512() };
            for (int gp = 0; gp < 4; gp++) {
                const __m512i cz = _mm512_loadu_si512(
                    (const void *)(cb + (size_t)gp * 64));
                const __m512i scv = _mm512_permutexvar_epi32(idx[gp], scz);
                const size_t qoff = b * OC_OXK_BLOCK_Q8_K_SIZE + 4 +
                                    (size_t)gp * 64;
                for (int a = 0; a < 4; a++) {
                    const __m512i p = _mm512_dpbusd_epi32(
                        _mm512_setzero_si512(), cz,
                        _mm512_loadu_si512((const void *)(act[a] + qoff)));
                    acc[a] = _mm512_add_epi32(acc[a],
                                              _mm512_mullo_epi32(p, scv));
                }
            }

            const __m256i sc16 = _mm256_cvtepi8_epi16(
                _mm_loadu_si128((const __m128i *)scb));
            for (int a = 0; a < 4; a++) {
                const uint8_t *qb = act[a] + b * OC_OXK_BLOCK_Q8_K_SIZE;
                const int32_t off = fold_bsums(sc16, qb + 4 + OC_OXK_QK_K);
                float dq;
                memcpy(&dq, qb, 4);
                sums[a] += d[b] * dq *
                           (float)(_mm512_reduce_add_epi32(acc[a]) -
                                   offset * off);
            }
        }
        for (int a = 0; a < 4; a++) out[a0 + a] = sums[a];
    }
    for (; a0 < n_act; a0++) {
        out[a0] = dot_q3q6_prepped_one(prep, blocks, acts + a0 * act_stride,
                                       offset);
    }
}

__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
void oc_oxk_dot_q3_k_prepped_multi_avx512(const void *prep, size_t blocks,
                                          const uint8_t *acts,
                                          size_t act_stride, size_t n_act,
                                          float *out)
{
    dot_q3q6_prepped_multi_body(prep, blocks, acts, act_stride, n_act, out, 4);
}

__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
void oc_oxk_dot_q2_k_prepped_multi_avx512(const void *prep, size_t blocks,
                                          const uint8_t *acts,
                                          size_t act_stride, size_t n_act,
                                          float *out)
{
    const float   *d     = (const float *)prep;
    const float   *dmin  = d + blocks;
    const uint8_t *codes = (const uint8_t *)(d + 2 * blocks);
    const uint8_t *sc    = codes + blocks * OC_OXK_QK_K;
    const uint8_t *mn    = sc + blocks * 16;

    const __m512i idx0 = _mm512_set_epi32(3,3,3,3, 2,2,2,2, 1,1,1,1, 0,0,0,0);
    const __m512i idx1 = _mm512_set_epi32(7,7,7,7, 6,6,6,6, 5,5,5,5, 4,4,4,4);
    const __m512i idx2 = _mm512_set_epi32(11,11,11,11, 10,10,10,10,
                                          9,9,9,9, 8,8,8,8);
    const __m512i idx3 = _mm512_set_epi32(15,15,15,15, 14,14,14,14,
                                          13,13,13,13, 12,12,12,12);
    const __m512i idx[4] = { idx0, idx1, idx2, idx3 };

    size_t a0 = 0;
    for (; a0 + 4 <= n_act; a0 += 4) {
        const uint8_t *act[4] = {
            acts + (a0 + 0) * act_stride, acts + (a0 + 1) * act_stride,
            acts + (a0 + 2) * act_stride, acts + (a0 + 3) * act_stride,
        };
        float sums[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *cb  = codes + b * OC_OXK_QK_K;
            const uint8_t *scb = sc + b * 16;
            const uint8_t *mnb = mn + b * 16;
            const __m512i scz = _mm512_cvtepu8_epi32(
                _mm_loadu_si128((const __m128i *)scb));

            __m512i acc[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(),
                               _mm512_setzero_si512(), _mm512_setzero_si512() };
            for (int gp = 0; gp < 4; gp++) {
                const __m512i cz = _mm512_loadu_si512(
                    (const void *)(cb + (size_t)gp * 64));
                const __m512i scv = _mm512_permutexvar_epi32(idx[gp], scz);
                const size_t qoff = b * OC_OXK_BLOCK_Q8_K_SIZE + 4 +
                                    (size_t)gp * 64;
                for (int a = 0; a < 4; a++) {
                    const __m512i p = _mm512_dpbusd_epi32(
                        _mm512_setzero_si512(), cz,
                        _mm512_loadu_si512((const void *)(act[a] + qoff)));
                    acc[a] = _mm512_add_epi32(acc[a],
                                              _mm512_mullo_epi32(p, scv));
                }
            }

            const __m256i mn16 = _mm256_cvtepu8_epi16(
                _mm_loadu_si128((const __m128i *)mnb));
            for (int a = 0; a < 4; a++) {
                const uint8_t *qb = act[a] + b * OC_OXK_BLOCK_Q8_K_SIZE;
                const int32_t min_acc = fold_bsums(mn16, qb + 4 + OC_OXK_QK_K);
                float dq;
                memcpy(&dq, qb, 4);
                sums[a] += dq *
                    (d[b] * (float)_mm512_reduce_add_epi32(acc[a]) -
                     dmin[b] * (float)min_acc);
            }
        }
        for (int a = 0; a < 4; a++) out[a0 + a] = sums[a];
    }
    for (; a0 < n_act; a0++) {
        out[a0] = oc_oxk_dot_q2_k_prepped_avx512(prep, blocks,
                                                 acts + a0 * act_stride);
    }
}

/* ─── VNNI Q6_K × Q8_K dot ──────────────────────────────────────────────── */


__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
float oc_oxk_dot_q6_k_q8_k_avx512vnni(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8)
{
    const __m256i lownib = _mm256_set1_epi8(0x0F);
    const __m256i two    = _mm256_set1_epi8(0x03);
    float sum = 0.0f;

    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q6_K_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_K_SIZE;
        const uint8_t *ql = wb;
        const uint8_t *qh = wb + 128;
        const int8_t  *sc = (const int8_t *)(wb + 192);
        const float dw = oc_oxk_f16_le_to_f32(wb + 208);
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + 256;

        /* -32 offset folded out through the activation's block sums:
         * sum(sc*(q-32)*a) = sum(sc*q*a) - 32*sum(sc*bsum). Integers, so
         * this matches the scalar reference exactly. */
        int32_t minc = 0;
        for (unsigned g = 0; g < 16; g++) {
            minc += (int32_t)sc[g] *
                    (int32_t)oc_oxk_read_q8_k_bsum(bsums, g);
        }

        __m256i acc = _mm256_setzero_si256();
        for (int n = 0; n < 2; n++) {
            const __m256i ql0 = _mm256_loadu_si256(
                (const __m256i *)(ql + n * 64));
            const __m256i ql1 = _mm256_loadu_si256(
                (const __m256i *)(ql + n * 64 + 32));
            const __m256i qhv = _mm256_loadu_si256(
                (const __m256i *)(qh + n * 32));

            /* q1..q4 as unsigned 0..63, exactly the scalar unpack minus
             * the -32. */
            const __m256i q1 = _mm256_or_si256(
                _mm256_and_si256(ql0, lownib),
                _mm256_slli_epi16(_mm256_and_si256(qhv, two), 4));
            const __m256i q2 = _mm256_or_si256(
                _mm256_and_si256(ql1, lownib),
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(qhv, 2), two), 4));
            const __m256i q3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql0, 4), lownib),
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(qhv, 4), two), 4));
            const __m256i q4 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql1, 4), lownib),
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(qhv, 6), two), 4));

            const __m256i qv[4] = { q1, q2, q3, q4 };
            for (int k = 0; k < 4; k++) {
                const int base = n * 128 + k * 32;
                const __m256i a = _mm256_loadu_si256(
                    (const __m256i *)(q8v + base));
                const __m256i p = _mm256_dpbusd_epi32(
                    _mm256_setzero_si256(), qv[k], a);
                /* dpbusd lane i covers elements 4i..4i+3: lanes 0-3 are the
                 * first 16 elements (scale group base/16), lanes 4-7 the
                 * next. */
                const int g0 = base / 16;
                const __m256i scv = _mm256_set_m128i(
                    _mm_set1_epi32((int32_t)sc[g0 + 1]),
                    _mm_set1_epi32((int32_t)sc[g0]));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(p, scv));
            }
        }
        const int32_t pos = hsum_i32_8_avx512(acc);
        sum += dw * dq * (float)(pos - 32 * minc);
    }
    return sum;
}

/* ─── VNNI Q5_K × Q8_K dot ──────────────────────────────────────────────── */

__attribute__((target("avx512bw,avx512dq,avx512vl,avx512vnni")))
float oc_oxk_dot_q5_k_q8_k_avx512vnni(const uint8_t *row, size_t blocks,
                                      const uint8_t *q8)
{
    const __m256i lownib = _mm256_set1_epi8(0x0F);
    float sum = 0.0f;

    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q5_K_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_K_SIZE;
        const float dw   = oc_oxk_f16_le_to_f32(wb);
        const float dmin = oc_oxk_f16_le_to_f32(wb + 2);
        const uint8_t *scales = wb + 4;
        const uint8_t *qh     = wb + 16;
        const uint8_t *qs     = wb + 48;
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + 256;

        __m256i acc = _mm256_setzero_si256();
        int32_t min_acc = 0;
        for (int gp = 0; gp < 4; gp++) {
            uint8_t sc1, m1, sc2, m2;
            oc_oxk_get_scale_min_k4((unsigned)(gp * 2),     scales, &sc1, &m1);
            oc_oxk_get_scale_min_k4((unsigned)(gp * 2 + 1), scales, &sc2, &m2);

            const __m256i packed = _mm256_loadu_si256(
                (const __m256i *)(qs + gp * 32));

            /* qh[l] carries one high bit per 64-element group: bit 2*gp for
             * the low-nibble half, bit 2*gp+1 for the high-nibble half —
             * the u1/u2 stepping masks of dequant_q5_k, NOT a flat 256-bit
             * field. */
            const __m256i qhv = _mm256_loadu_si256((const __m256i *)qh);
            const __m256i one = _mm256_set1_epi8(0x01);
            const __m256i h_lo = _mm256_and_si256(
                _mm256_srli_epi16(qhv, 2 * gp), one);
            const __m256i h_hi = _mm256_and_si256(
                _mm256_srli_epi16(qhv, 2 * gp + 1), one);
            const __m256i w_lo = _mm256_or_si256(
                _mm256_and_si256(packed, lownib),
                _mm256_slli_epi16(h_lo, 4));
            const __m256i w_hi = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(packed, 4), lownib),
                _mm256_slli_epi16(h_hi, 4));

            const __m256i a_lo = _mm256_loadu_si256(
                (const __m256i *)(q8v + gp * 64));
            const __m256i a_hi = _mm256_loadu_si256(
                (const __m256i *)(q8v + gp * 64 + 32));

            const __m256i p1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(),
                                                   w_lo, a_lo);
            const __m256i p2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(),
                                                   w_hi, a_hi);
            acc = _mm256_add_epi32(acc,
                      _mm256_mullo_epi32(p1, _mm256_set1_epi32((int32_t)sc1)));
            acc = _mm256_add_epi32(acc,
                      _mm256_mullo_epi32(p2, _mm256_set1_epi32((int32_t)sc2)));

            const int32_t bs1 = oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4)) +
                                oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4 + 1));
            const int32_t bs2 = oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4 + 2)) +
                                oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4 + 3));
            min_acc += (int32_t)m1 * bs1 + (int32_t)m2 * bs2;
        }
        sum += dw * dq * (float)hsum_i32_8_avx512(acc)
             - dmin * dq * (float)min_acc;
    }
    return sum;
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
void oc_oxk_dot_q4_k_prepped_multi_avx512(const void *prep, size_t blocks, const uint8_t *acts, size_t act_stride, size_t n_act, float *out)
{
    for (size_t v = 0; v < n_act; v++)
        out[v] = oc_oxk_dot_q4_k_prepped(prep, blocks, acts + v * act_stride);
}
float oc_oxk_dot_q6_k_q8_k_avx512vnni(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q6_k_q8_k_scalar(row, blocks, q8); }
void oc_oxk_dot_q6_k_prepped_multi_avx512(const void *prep, size_t blocks, const uint8_t *acts, size_t act_stride, size_t n_act, float *out)
{
    for (size_t v = 0; v < n_act; v++)
        out[v] = oc_oxk_dot_q6_k_prepped(prep, blocks, acts + v * act_stride);
}
float oc_oxk_dot_q2_k_prepped_avx512(const void *prep, size_t blocks, const uint8_t *act)
{ return oc_oxk_dot_q2_k_prepped(prep, blocks, act); }
float oc_oxk_dot_q3_k_prepped_avx512(const void *prep, size_t blocks, const uint8_t *act)
{ return oc_oxk_dot_q3_k_prepped(prep, blocks, act); }
float oc_oxk_dot_q6_k_prepped_avx512(const void *prep, size_t blocks, const uint8_t *act)
{ return oc_oxk_dot_q6_k_prepped(prep, blocks, act); }
void oc_oxk_dot_q3_k_prepped_multi_avx512(const void *prep, size_t blocks, const uint8_t *acts, size_t act_stride, size_t n_act, float *out)
{
    for (size_t v = 0; v < n_act; v++)
        out[v] = oc_oxk_dot_q3_k_prepped(prep, blocks, acts + v * act_stride);
}
void oc_oxk_dot_q2_k_prepped_multi_avx512(const void *prep, size_t blocks, const uint8_t *acts, size_t act_stride, size_t n_act, float *out)
{
    for (size_t v = 0; v < n_act; v++)
        out[v] = oc_oxk_dot_q2_k_prepped(prep, blocks, acts + v * act_stride);
}
float oc_oxk_dot_q5_k_q8_k_avx512vnni(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q5_k_q8_k_scalar(row, blocks, q8); }

#endif  /* __x86_64__ || __i386__ */
