/* test_oxk.c — OXK kernel tests. */
#include "framework.h"
#include "oxidize/oxk.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

Test(oxk, f16_to_f32_zero)
{
    uint8_t p[2] = {0, 0};
    cr_assert_float_eq(oc_oxk_f16_le_to_f32(p), 0.0f, 1e-7f);
}

Test(oxk, f16_to_f32_one)
{
    /* f16 1.0 = 0x3C00 → bytes [0x00, 0x3C] */
    uint8_t p[2] = {0x00, 0x3C};
    cr_assert_float_eq(oc_oxk_f16_le_to_f32(p), 1.0f, 1e-6f);
}

Test(oxk, f16_to_f32_neg_one)
{
    /* f16 -1.0 = 0xBC00 → bytes [0x00, 0xBC] */
    uint8_t p[2] = {0x00, 0xBC};
    cr_assert_float_eq(oc_oxk_f16_le_to_f32(p), -1.0f, 1e-6f);
}

Test(oxk, f16_to_f32_half)
{
    /* f16 0.5 = 0x3800 → bytes [0x00, 0x38] */
    uint8_t p[2] = {0x00, 0x38};
    cr_assert_float_eq(oc_oxk_f16_le_to_f32(p), 0.5f, 1e-6f);
}

Test(oxk, f16_to_f32_two)
{
    /* f16 2.0 = 0x4000 → bytes [0x00, 0x40] */
    uint8_t p[2] = {0x00, 0x40};
    cr_assert_float_eq(oc_oxk_f16_le_to_f32(p), 2.0f, 1e-6f);
}

Test(oxk, f16_to_f32_large)
{
    /* f16 1024 = 0x6400 → bytes [0x00, 0x64] */
    uint8_t p[2] = {0x00, 0x64};
    cr_assert_float_eq(oc_oxk_f16_le_to_f32(p), 1024.0f, 1e-3f);
}

Test(oxk, caps_init)
{
    const OcOxkCaps *caps = oc_oxk_caps();
    cr_assert_not_null(caps);
    cr_assert(caps->level >= OC_OXK_SCALAR && caps->level <= OC_OXK_NEON);
    cr_assert_not_null(caps->name);
}

Test(oxk, init_returns_context)
{
    const OcOxkContext *ctx = oc_oxk_init();
    cr_assert_not_null(ctx);
    cr_assert_not_null(ctx->dot_q4_0_q8_0);
    cr_assert_not_null(ctx->dot_q4_1_q8_0);
    cr_assert_not_null(ctx->dot_q8_0_q8_0);
    cr_assert_not_null(ctx->dot_q4_k_q8_k);
    cr_assert_not_null(ctx->dot_q5_k_q8_k);
    cr_assert_not_null(ctx->dot_q6_k_q8_k);
    cr_assert_not_null(ctx->matvec_q4_0_f32);
    cr_assert_not_null(ctx->matvec_q4_k_f32);
    cr_assert_not_null(ctx->matvec_q8_0_f32);
}

Test(oxk, get_scale_min_k4_low)
{
    uint8_t scales[12] = {0};
    scales[0] = 42;  /* scale for j=0 */
    scales[4] = 17;  /* min for j=0 */
    uint8_t sc, m;
    oc_oxk_get_scale_min_k4(0, scales, &sc, &m);
    cr_assert_eq(sc, 42);
    cr_assert_eq(m, 17);
}

/* j >= 4 packs the 6-bit scale and min across three bytes, per ggml:
 *   scale = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
 *   min   = ((scales[j+4] >> 4) & 0x0F) | ((scales[j] >> 6) << 4)
 *
 * This previously asserted a different assembly — the wrong bit positions and
 * the wrong source byte — which is why every Q4_K and Q5_K block decoded its
 * upper four scale/min pairs incorrectly while the test still passed. The
 * expected values below are computed from the formula above, and match
 * quantization.c, which is bit-identical to the ggml reference. */
Test(oxk, get_scale_min_k4_high)
{
    uint8_t scales[12] = {0};
    scales[0] = 0xC0;  /* j-4 = 0: >> 6 = 3  -> scale bits [5:4] */
    scales[4] = 0x80;  /* j   = 4: >> 6 = 2  -> min   bits [5:4] */
    scales[8] = 0x9A;  /* j+4 = 8: low nibble 0xA -> scale [3:0]
                        *          high nibble 0x9 -> min   [3:0] */
    uint8_t sc, m;
    oc_oxk_get_scale_min_k4(4, scales, &sc, &m);
    cr_assert_eq(sc, 0x0A | (3 << 4), "scale: got %u want %u", sc, 0x0A | (3 << 4));
    cr_assert_eq(m,  0x09 | (2 << 4), "min: got %u want %u",  m,  0x09 | (2 << 4));
}

Test(oxk, dot_q8_0_q8_0_basic)
{
    /* One block: Q8_0 weight = [f16 d=1.0][32 × int8=1] */
    uint8_t w[34], q[34];
    /* f16 1.0 = 0x3C00 */
    w[0] = 0x00; w[1] = 0x3C;
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) {
        w[2 + i] = 1;  /* int8 1 */
        q[2 + i] = 1;  /* int8 1 */
    }
    float result = oc_oxk_dot_q8_0_q8_0_scalar(w, 1, q);
    /* d_w * d_q * sum(1*1) = 1.0 * 1.0 * 32 = 32.0 */
    cr_assert_float_eq(result, 32.0f, 0.01f);
}

Test(oxk, dot_q8_0_q8_0_dispatched)
{
    uint8_t w[34], q[34];
    w[0] = 0x00; w[1] = 0x3C;
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) {
        w[2 + i] = 2;
        q[2 + i] = 3;
    }
    float result = oc_oxk_dot_q8_0_q8_0(w, 1, q);
    /* 1.0 * 1.0 * sum(2*3) = 32 * 6 = 192 */
    cr_assert_float_eq(result, 192.0f, 0.01f);
}

Test(oxk, dot_q4_0_q8_0_basic)
{
    /* Q4_0 block: f16 d=1.0, 16 bytes of packed nibbles all 8 (=0 after -8 offset) */
    uint8_t w[18], q[34];
    w[0] = 0x00; w[1] = 0x3C;  /* f16 1.0 */
    for (int i = 0; i < 16; i++) w[2 + i] = 0x88;  /* nibbles all 8 */
    q[0] = 0x00; q[1] = 0x3C;  /* f16 1.0 */
    for (int i = 0; i < 32; i++) q[2 + i] = 1;  /* int8 1 */
    float result = oc_oxk_dot_q4_0_q8_0_scalar(w, 1, q);
    /* (8-8)*1 = 0 for all 32 elements → dot = 0 */
    cr_assert_float_eq(result, 0.0f, 0.01f);
}

Test(oxk, dot_q4_0_q8_0_nonzero)
{
    /* nibble = 9 → (9-8)=1, q8=1 → dot = 32 * 1 * 1 * 1 = 32 */
    uint8_t w[18], q[34];
    w[0] = 0x00; w[1] = 0x3C;
    for (int i = 0; i < 16; i++) w[2 + i] = 0x99;  /* nibbles all 9 */
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) q[2 + i] = 1;
    float result = oc_oxk_dot_q4_0_q8_0_scalar(w, 1, q);
    /* (9-8)*1 = 1 for 32 elements → 1.0*1.0*32 = 32 */
    cr_assert_float_eq(result, 32.0f, 0.01f);
}

Test(oxk, dot_q4_1_q8_0_basic)
{
    /* Q4_1: f16 d=1.0, f16 m=0.0, nibbles all 0 */
    uint8_t w[20], q[34];
    w[0] = 0x00; w[1] = 0x3C;  /* d=1.0 */
    w[2] = 0x00; w[3] = 0x00;  /* m=0.0 */
    for (int i = 0; i < 16; i++) w[4 + i] = 0x00;  /* nibbles all 0 */
    q[0] = 0x00; q[1] = 0x3C;  /* d=1.0 */
    for (int i = 0; i < 32; i++) q[2 + i] = 1;
    float result = oc_oxk_dot_q4_1_q8_0_scalar(w, 1, q);
    /* d*dq*sum(0*1) + m*dq*sum(1) = 0 + 0 = 0 */
    cr_assert_float_eq(result, 0.0f, 0.01f);
}

Test(oxk, matvec_q8_0_f32_basic)
{
    /* 2 rows, 1 block each, Q8_0 d=1.0, int8=1, f32 input=2.0 */
    uint8_t w[68];  /* 2 × 34 bytes */
    float x[32];
    float out[2];
    for (int r = 0; r < 2; r++) {
        w[r * 34] = 0x00; w[r * 34 + 1] = 0x3C;
        for (int i = 0; i < 32; i++) w[r * 34 + 2 + i] = 1;
    }
    for (int i = 0; i < 32; i++) x[i] = 2.0f;
    oc_oxk_matvec_q8_0_f32_scalar(w, 2, 34, x, out);
    /* d=1.0, w=1, x=2.0 → sum = 32 * 1 * 2 = 64 */
    cr_assert_float_eq(out[0], 64.0f, 0.01f);
    cr_assert_float_eq(out[1], 64.0f, 0.01f);
}

Test(oxk, read_q8_k_bsum)
{
    uint8_t bsums[32] = {0};
    /* bsums[0] = 100 (little-endian) */
    bsums[0] = 100; bsums[1] = 0;
    /* bsums[1] = -50 (little-endian, two's complement) */
    bsums[2] = 0xCE; bsums[3] = 0xFF;  /* 0xFFCE = -50 */
    cr_assert_eq(oc_oxk_read_q8_k_bsum(bsums, 0), 100);
    cr_assert_eq(oc_oxk_read_q8_k_bsum(bsums, 1), -50);
}

Test(oxk, parity_scalar_dispatched)
{
    /* Verify dispatched == scalar for Q8_0. */
    uint8_t w[34], q[34];
    w[0] = 0x00; w[1] = 0x3C;
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) {
        w[2 + i] = (int8_t)(i - 16);
        q[2 + i] = (int8_t)(16 - i);
    }
    float s = oc_oxk_dot_q8_0_q8_0_scalar(w, 1, q);
    float d = oc_oxk_dot_q8_0_q8_0(w, 1, q);
    cr_assert_float_eq(s, d, 0.0f, "dispatched should match scalar exactly");
}
