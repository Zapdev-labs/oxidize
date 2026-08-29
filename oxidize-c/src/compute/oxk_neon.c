/* oxk_neon.c — AArch64 Advanced SIMD (NEON) OXK kernels. bit-exactness contract; the short version is that every integer */
#include "oxidize/oxk_neon.h"

#if defined(__aarch64__)

#include "oxidize/oxk.h"

#include <arm_neon.h>
#include <string.h>

static inline int32_t oc_neon_dot16(int8x16_t a, int8x16_t b)
{
    int16x8_t p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t p1 = vmull_high_s8(a, b);
    int32x4_t acc = vpadalq_s16(vdupq_n_s32(0), p0);
    acc = vpadalq_s16(acc, p1);
    return vaddvq_s32(acc);
}

/* Exact int32 sum of 16 int8 lanes. */
static inline int32_t oc_neon_hsum_s8(int8x16_t a)
{
    return vaddlvq_s16(vpaddlq_s8(a));
}


float oc_oxk_dot_q8_0_q8_0_neon(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;
        float dw = oc_oxk_f16_le_to_f32(wb);
        float dq = oc_oxk_f16_le_to_f32(qb);

        const int8_t *wv = (const int8_t *)(wb + 2);
        const int8_t *qv = (const int8_t *)(qb + 2);

        int32_t isum = oc_neon_dot16(vld1q_s8(wv), vld1q_s8(qv)) +
                       oc_neon_dot16(vld1q_s8(wv + 16), vld1q_s8(qv + 16));

        sum += dw * dq * (float)isum;
    }
    return sum;
}


float oc_oxk_dot_q4_0_q8_0_neon(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    const uint8x16_t nib = vdupq_n_u8(0x0F);
    const int8x16_t  bias = vdupq_n_s8(8);

    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_0_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;
        float dw = oc_oxk_f16_le_to_f32(wb);
        float dq = oc_oxk_f16_le_to_f32(qb);

        /* 16 packed bytes → 32 nibbles. Nibble i's low half pairs with the
         * even-indexed activation qv[2i], the high half with qv[2i+1], so
         * de-interleave the 32 activations into even/odd halves. */
        uint8x16_t packed = vld1q_u8(wb + 2);
        int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed, nib)), bias);
        int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), bias);

        int8x16x2_t qv = vld2q_s8((const int8_t *)(qb + 2));

        int32_t isum = oc_neon_dot16(lo, qv.val[0]) +
                       oc_neon_dot16(hi, qv.val[1]);

        sum += dw * dq * (float)isum;
    }
    return sum;
}


float oc_oxk_dot_q4_1_q8_0_neon(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    const uint8x16_t nib = vdupq_n_u8(0x0F);

    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_1_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;
        float dw = oc_oxk_f16_le_to_f32(wb);
        float mw = oc_oxk_f16_le_to_f32(wb + 2);
        float dq = oc_oxk_f16_le_to_f32(qb);

        uint8x16_t packed = vld1q_u8(wb + 4);
        /* Unbiased nibbles (0..15) — safe to reinterpret as int8. */
        int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(packed, nib));
        int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));

        const int8_t *qraw = (const int8_t *)(qb + 2);
        int8x16x2_t qv = vld2q_s8(qraw);

        int32_t dot_prod = oc_neon_dot16(lo, qv.val[0]) +
                           oc_neon_dot16(hi, qv.val[1]);
        int32_t q8_sum = oc_neon_hsum_s8(vld1q_s8(qraw)) +
                         oc_neon_hsum_s8(vld1q_s8(qraw + 16));

        sum += dw * dq * (float)dot_prod + mw * dq * (float)q8_sum;
    }
    return sum;
}


float oc_oxk_dot_q4_k_q8_k_neon(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    const uint8x16_t nib = vdupq_n_u8(0x0F);

    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_K_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float dw   = oc_oxk_f16_le_to_f32(wb);
        float dmin = oc_oxk_f16_le_to_f32(wb + 2);
        const uint8_t *scales = wb + 4;
        const uint8_t *qs     = wb + 16;

        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + 256;

        for (int gp = 0; gp < 4; gp++) {
            uint8_t sc1, m1, sc2, m2;
            oc_oxk_get_scale_min_k4((unsigned)(gp * 2),     scales, &sc1, &m1);
            oc_oxk_get_scale_min_k4((unsigned)(gp * 2 + 1), scales, &sc2, &m2);

            uint8x16_t p0 = vld1q_u8(qs + gp * 32);
            uint8x16_t p1 = vld1q_u8(qs + gp * 32 + 16);

            const int8_t *a = q8v + gp * 64;       /* low-nibble partners  */
            const int8_t *c = q8v + gp * 64 + 32;  /* high-nibble partners */

            int32_t sum1 =
                oc_neon_dot16(vreinterpretq_s8_u8(vandq_u8(p0, nib)), vld1q_s8(a)) +
                oc_neon_dot16(vreinterpretq_s8_u8(vandq_u8(p1, nib)), vld1q_s8(a + 16));
            int32_t sum2 =
                oc_neon_dot16(vreinterpretq_s8_u8(vshrq_n_u8(p0, 4)), vld1q_s8(c)) +
                oc_neon_dot16(vreinterpretq_s8_u8(vshrq_n_u8(p1, 4)), vld1q_s8(c + 16));

            int32_t bs1 = oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4)) +
                          oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4 + 1));
            int32_t bs2 = oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4 + 2)) +
                          oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4 + 3));

            /* Same f32 op sequence as the scalar reference (no FMA folding:
             * -ffp-contract is not enabled for this build). */
            float term1 = dw * dq * (float)sc1 * (float)sum1 - dw * dmin * dq * (float)m1 * (float)bs1;
            float term2 = dw * dq * (float)sc2 * (float)sum2 - dw * dmin * dq * (float)m2 * (float)bs2;
            sum += term1 + term2;
        }
    }
    return sum;
}


float oc_oxk_dot_q5_k_q8_k_neon(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    const uint8x16_t nib = vdupq_n_u8(0x0F);

    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q5_K_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float dw   = oc_oxk_f16_le_to_f32(wb);
        float dmin = oc_oxk_f16_le_to_f32(wb + 2);
        const uint8_t *scales = wb + 4;
        const uint8_t *qh     = wb + 16;
        const uint8_t *qs     = wb + 48;

        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + 256;

        for (int gp = 0; gp < 4; gp++) {
            uint8_t sc1, m1, sc2, m2;
            oc_oxk_get_scale_min_k4((unsigned)(gp * 2),     scales, &sc1, &m1);
            oc_oxk_get_scale_min_k4((unsigned)(gp * 2 + 1), scales, &sc2, &m2);

            uint8x16_t p0 = vld1q_u8(qs + gp * 32);
            uint8x16_t p1 = vld1q_u8(qs + gp * 32 + 16);

            const uint8x16_t qh0v = vld1q_u8(qh);
            const uint8x16_t qh1v = vld1q_u8(qh + 16);
            const uint8x16_t onev = vdupq_n_u8(1);
            const int8x16_t sh_lo = vdupq_n_s8((int8_t)-(2 * gp));
            const int8x16_t sh_hi = vdupq_n_s8((int8_t)-(2 * gp + 1));
            uint8x16_t h_lo0 = vshlq_n_u8(
                vandq_u8(vshlq_u8(qh0v, sh_lo), onev), 4);
            uint8x16_t h_lo1 = vshlq_n_u8(
                vandq_u8(vshlq_u8(qh1v, sh_lo), onev), 4);
            uint8x16_t h_hi0 = vshlq_n_u8(
                vandq_u8(vshlq_u8(qh0v, sh_hi), onev), 4);
            uint8x16_t h_hi1 = vshlq_n_u8(
                vandq_u8(vshlq_u8(qh1v, sh_hi), onev), 4);

            /* 0..31 — still representable as a non-negative int8. */
            int8x16_t v_lo0 = vreinterpretq_s8_u8(vaddq_u8(vandq_u8(p0, nib), h_lo0));
            int8x16_t v_lo1 = vreinterpretq_s8_u8(vaddq_u8(vandq_u8(p1, nib), h_lo1));
            int8x16_t v_hi0 = vreinterpretq_s8_u8(vaddq_u8(vshrq_n_u8(p0, 4), h_hi0));
            int8x16_t v_hi1 = vreinterpretq_s8_u8(vaddq_u8(vshrq_n_u8(p1, 4), h_hi1));

            const int8_t *a = q8v + gp * 64;
            const int8_t *c = q8v + gp * 64 + 32;

            int32_t sum1 = oc_neon_dot16(v_lo0, vld1q_s8(a)) +
                           oc_neon_dot16(v_lo1, vld1q_s8(a + 16));
            int32_t sum2 = oc_neon_dot16(v_hi0, vld1q_s8(c)) +
                           oc_neon_dot16(v_hi1, vld1q_s8(c + 16));

            int32_t bs1 = oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4)) +
                          oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4 + 1));
            int32_t bs2 = oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4 + 2)) +
                          oc_oxk_read_q8_k_bsum(bsums, (size_t)(gp * 4 + 3));

            float term1 = dw * dq * (float)sc1 * (float)sum1 - dw * dmin * dq * (float)m1 * (float)bs1;
            float term2 = dw * dq * (float)sc2 * (float)sum2 - dw * dmin * dq * (float)m2 * (float)bs2;
            sum += term1 + term2;
        }
    }
    return sum;
}


/* PARITY CAVEAT (the one deviation in this file): the scalar reference adds one f32 term per element. */
/* accurate of the two (fewer roundings), but it is NOT bit-exact. */
float oc_oxk_dot_q6_k_q8_k_neon(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    const uint8x16_t nib  = vdupq_n_u8(0x0F);
    const uint8x16_t mask2 = vdupq_n_u8(0x03);
    const int8x16_t  bias = vdupq_n_s8(32);

    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q6_K_SIZE;
        const uint8_t *ql = wb;
        const uint8_t *qh = wb + 128;
        const int8_t  *sc = (const int8_t *)(wb + 192);
        float dw = oc_oxk_f16_le_to_f32(wb + 208);

        const uint8_t *qb = q8 + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t *q8v = (const int8_t *)(qb + 4);

        for (int n = 0; n < 2; n++) {
            const uint8_t *ql_chunk = ql + n * 64;
            const uint8_t *qh_chunk = qh + n * 32;
            const int8_t  *sc_chunk = sc + n * 8;

            /* `is` (= l / 16) is constant across each 16-lane group, so the
             * scale is loop-invariant and the group reduces to int32. */
            for (int is = 0; is < 2; is++) {
                int l0 = is * 16;
                uint8x16_t a = vld1q_u8(ql_chunk + l0);
                uint8x16_t c = vld1q_u8(ql_chunk + 32 + l0);
                uint8x16_t h = vld1q_u8(qh_chunk + l0);

                int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(a, nib), vshlq_n_u8(vandq_u8(h, mask2), 4))), bias);
                int8x16_t q2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(c, nib), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 2), mask2), 4))), bias);
                int8x16_t q3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(a, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 4), mask2), 4))), bias);
                int8x16_t q4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(c, 4), vshlq_n_u8(vshrq_n_u8(h, 6), 4))), bias);

                int base = n * 128 + l0;
                int32_t s1 = oc_neon_dot16(q1, vld1q_s8(q8v + base));
                int32_t s2 = oc_neon_dot16(q2, vld1q_s8(q8v + base + 32));
                int32_t s3 = oc_neon_dot16(q3, vld1q_s8(q8v + base + 64));
                int32_t s4 = oc_neon_dot16(q4, vld1q_s8(q8v + base + 96));

                sum += dw * dq * (float)sc_chunk[is + 0] * (float)s1;
                sum += dw * dq * (float)sc_chunk[is + 2] * (float)s2;
                sum += dw * dq * (float)sc_chunk[is + 4] * (float)s3;
                sum += dw * dq * (float)sc_chunk[is + 6] * (float)s4;
            }
        }
    }
    return sum;
}

#else  /* !__aarch64__ — empty translation unit guard. */

typedef int oc_oxk_neon_translation_unit_not_empty;

#endif /* __aarch64__ */
