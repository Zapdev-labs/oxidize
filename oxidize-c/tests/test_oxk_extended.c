/* test_oxk_extended.c — Comprehensive OXK kernel tests. */
#include <criterion/criterion.h>
#include "oxidize/oxk.h"
#include "oxidize/oxk_avx512.h"

#include <string.h>
#include <math.h>
#include <stdlib.h>


Test(oxk_ext, block_size_q4_0)
{
    cr_assert_eq(OC_OXK_BLOCK_Q4_0_SIZE, 18u);
    cr_assert_eq(OC_OXK_QK4_0, 32u);
}

Test(oxk_ext, block_size_q4_1)
{
    cr_assert_eq(OC_OXK_BLOCK_Q4_1_SIZE, 20u);
    cr_assert_eq(OC_OXK_QK4_1, 32u);
}

Test(oxk_ext, block_size_q8_0)
{
    cr_assert_eq(OC_OXK_BLOCK_Q8_0_SIZE, 34u);
    cr_assert_eq(OC_OXK_QK8_0, 32u);
}

Test(oxk_ext, block_size_k_quants)
{
    cr_assert_eq(OC_OXK_QK_K, 256u);
    cr_assert_eq(OC_OXK_BLOCK_Q4_K_SIZE, 144u);
    cr_assert_eq(OC_OXK_BLOCK_Q5_K_SIZE, 176u);
    cr_assert_eq(OC_OXK_BLOCK_Q6_K_SIZE, 210u);
    cr_assert_eq(OC_OXK_BLOCK_Q8_K_SIZE, 292u);
}


Test(oxk_ext, caps_valid_level)
{
    const OcOxkCaps *caps = oc_oxk_caps();
    cr_assert_not_null(caps);
    cr_assert(caps->level >= OC_OXK_SCALAR && caps->level <= OC_OXK_AVX512,
              "level %d out of range", (int)caps->level);
    cr_assert_not_null(caps->name);
    cr_assert(strlen(caps->name) > 0, "caps name empty");
}

Test(oxk_ext, caps_name_matches_level)
{
    const OcOxkCaps *caps = oc_oxk_caps();
    switch (caps->level) {
    case OC_OXK_SCALAR:
        cr_assert_str_eq(caps->name, "scalar");
        break;
    case OC_OXK_AVX2:
        cr_assert_str_eq(caps->name, "avx2");
        break;
    case OC_OXK_AVX512:
        cr_assert_str_eq(caps->name, "avx512");
        break;
    default:
        cr_assert(false, "unknown level %d", (int)caps->level);
    }
}


Test(oxk_ext, dispatcher_q8_0_matches_scalar)
{
    uint8_t w[34], q[34];
    w[0] = 0x00; w[1] = 0x3C;  /* f16 1.0 */
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) {
        w[2 + i] = (int8_t)(i * 2 - 31);
        q[2 + i] = (int8_t)(31 - i);
    }
    float s = oc_oxk_dot_q8_0_q8_0_scalar(w, 1, q);
    float d = oc_oxk_dot_q8_0_q8_0(w, 1, q);
    cr_assert_float_eq(s, d, 0.0f,
                       "dispatcher %.6f != scalar %.6f", d, s);
}

Test(oxk_ext, dispatcher_q4_0_matches_scalar)
{
    uint8_t w[18], q[34];
    w[0] = 0x00; w[1] = 0x3C;
    for (int i = 0; i < 16; i++) w[2 + i] = (uint8_t)(i * 17);
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) q[2 + i] = (int8_t)(i - 16);
    float s = oc_oxk_dot_q4_0_q8_0_scalar(w, 1, q);
    float d = oc_oxk_dot_q4_0_q8_0(w, 1, q);
    cr_assert_float_eq(s, d, 0.0f);
}

Test(oxk_ext, dispatcher_q4_1_matches_scalar)
{
    uint8_t w[20], q[34];
    w[0] = 0x00; w[1] = 0x3C;  /* d=1.0 */
    w[2] = 0x00; w[3] = 0x3C;  /* m=1.0 */
    for (int i = 0; i < 16; i++) w[4 + i] = (uint8_t)(i * 17);
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) q[2 + i] = (int8_t)(i - 16);
    float s = oc_oxk_dot_q4_1_q8_0_scalar(w, 1, q);
    float d = oc_oxk_dot_q4_1_q8_0(w, 1, q);
    cr_assert_float_eq(s, d, 0.0f);
}


Test(oxk_ext, dot_q8_0_known_result)
{
    /* d_w=2.0, d_q=3.0, all weights=4, all activations=5
     * result = 2*3 * sum(4*5) = 6 * 32 * 20 = 6 * 640 = 3840 */
    uint8_t w[34], q[34];
    /* f16 2.0 = 0x4000 → [0x00, 0x40] */
    w[0] = 0x00; w[1] = 0x40;
    /* f16 3.0 = 0x4200 → [0x00, 0x42] */
    q[0] = 0x00; q[1] = 0x42;
    for (int i = 0; i < 32; i++) {
        w[2 + i] = 4;
        q[2 + i] = 5;
    }
    float result = oc_oxk_dot_q8_0_q8_0_scalar(w, 1, q);
    cr_assert_float_eq(result, 3840.0f, 0.1f);
}

Test(oxk_ext, dot_q8_0_negative_values)
{
    /* All weights = -3, activations = 7, d_w = d_q = 1.0
     * result = sum(-3 * 7) = 32 * (-21) = -672 */
    uint8_t w[34], q[34];
    w[0] = 0x00; w[1] = 0x3C;  /* f16 1.0 */
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) {
        w[2 + i] = (int8_t)(-3);
        q[2 + i] = 7;
    }
    float result = oc_oxk_dot_q8_0_q8_0_scalar(w, 1, q);
    cr_assert_float_eq(result, -672.0f, 0.1f);
}

Test(oxk_ext, dot_q8_0_zero_input)
{
    uint8_t w[34], q[34];
    w[0] = 0x00; w[1] = 0x3C;
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) {
        w[2 + i] = 0;
        q[2 + i] = 0;
    }
    float result = oc_oxk_dot_q8_0_q8_0_scalar(w, 1, q);
    cr_assert_float_eq(result, 0.0f, 0.0f);
}


Test(oxk_ext, dot_q4_0_known_result)
{
    /* d_w = 1.0, d_q = 1.0, all nibbles = 9 (value = 9-8 = 1), q8 = 3
     * result = sum(1 * 3) = 32 * 3 = 96 */
    uint8_t w[18], q[34];
    w[0] = 0x00; w[1] = 0x3C;  /* f16 1.0 */
    for (int i = 0; i < 16; i++) w[2 + i] = 0x99;  /* nibbles all 9 */
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) q[2 + i] = 3;
    float result = oc_oxk_dot_q4_0_q8_0_scalar(w, 1, q);
    cr_assert_float_eq(result, 96.0f, 0.1f);
}

Test(oxk_ext, dot_q4_0_zero_when_nibble_is_8)
{
    uint8_t w[18], q[34];
    w[0] = 0x00; w[1] = 0x3C;
    for (int i = 0; i < 16; i++) w[2 + i] = 0x88;  /* nibbles all 8 → 0 */
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) q[2 + i] = 100;
    float result = oc_oxk_dot_q4_0_q8_0_scalar(w, 1, q);
    cr_assert_float_eq(result, 0.0f, 0.0f);
}


Test(oxk_ext, dot_q4_1_known_result)
{
    /* d=1.0, m=2.0, nibbles all 5, q8 all 3, d_q=1.0 */
    uint8_t w[20], q[34];
    w[0] = 0x00; w[1] = 0x3C;  /* d=1.0 */
    w[2] = 0x00; w[3] = 0x40;  /* m=2.0 (f16 0x4000) */
    for (int i = 0; i < 16; i++) w[4 + i] = 0x55;  /* nibbles all 5 */
    q[0] = 0x00; q[1] = 0x3C;  /* d_q=1.0 */
    for (int i = 0; i < 32; i++) q[2 + i] = 3;
    float result = oc_oxk_dot_q4_1_q8_0_scalar(w, 1, q);
    cr_assert_float_eq(result, 672.0f, 0.5f);
}


Test(oxk_ext, dot_q4_k_basic)
{
    /* Zero out a Q4_K block + Q8_K block, verify result is 0. */
    uint8_t w[OC_OXK_BLOCK_Q4_K_SIZE];
    uint8_t q[OC_OXK_BLOCK_Q8_K_SIZE];
    memset(w, 0, sizeof(w));
    memset(q, 0, sizeof(q));
    float result = oc_oxk_dot_q4_k_q8_k_scalar(w, 1, q);
    cr_assert_float_eq(result, 0.0f, 0.0f,
                       "Q4_K zero block should give 0 dot product");
}

Test(oxk_ext, dot_q4_k_dispatched_matches_scalar)
{
    uint8_t w[OC_OXK_BLOCK_Q4_K_SIZE];
    uint8_t q[OC_OXK_BLOCK_Q8_K_SIZE];
    /* Fill with a deterministic pattern. */
    for (size_t i = 0; i < sizeof(w); i++) w[i] = (uint8_t)(i * 7 + 3);
    for (size_t i = 0; i < sizeof(q); i++) q[i] = (uint8_t)(i * 5 + 11);
    float s = oc_oxk_dot_q4_k_q8_k_scalar(w, 1, q);
    float d = oc_oxk_dot_q4_k_q8_k(w, 1, q);
    cr_assert_float_eq(s, d, 0.0f);
}


Test(oxk_ext, dot_q5_k_basic)
{
    uint8_t w[OC_OXK_BLOCK_Q5_K_SIZE];
    uint8_t q[OC_OXK_BLOCK_Q8_K_SIZE];
    memset(w, 0, sizeof(w));
    memset(q, 0, sizeof(q));
    float result = oc_oxk_dot_q5_k_q8_k_scalar(w, 1, q);
    cr_assert_float_eq(result, 0.0f, 0.0f);
}

Test(oxk_ext, dot_q5_k_dispatched_matches_scalar)
{
    uint8_t w[OC_OXK_BLOCK_Q5_K_SIZE];
    uint8_t q[OC_OXK_BLOCK_Q8_K_SIZE];
    for (size_t i = 0; i < sizeof(w); i++) w[i] = (uint8_t)(i * 3 + 1);
    for (size_t i = 0; i < sizeof(q); i++) q[i] = (uint8_t)(i * 11 + 7);
    float s = oc_oxk_dot_q5_k_q8_k_scalar(w, 1, q);
    float d = oc_oxk_dot_q5_k_q8_k(w, 1, q);
    cr_assert_float_eq(s, d, 0.0f);
}


Test(oxk_ext, dot_q6_k_basic)
{
    uint8_t w[OC_OXK_BLOCK_Q6_K_SIZE];
    uint8_t q[OC_OXK_BLOCK_Q8_K_SIZE];
    memset(w, 0, sizeof(w));
    memset(q, 0, sizeof(q));
    float result = oc_oxk_dot_q6_k_q8_k_scalar(w, 1, q);
    cr_assert_float_eq(result, 0.0f, 0.0f);
}

Test(oxk_ext, dot_q6_k_dispatched_matches_scalar)
{
    uint8_t w[OC_OXK_BLOCK_Q6_K_SIZE];
    uint8_t q[OC_OXK_BLOCK_Q8_K_SIZE];
    for (size_t i = 0; i < sizeof(w); i++) w[i] = (uint8_t)(i * 13 + 2);
    for (size_t i = 0; i < sizeof(q); i++) q[i] = (uint8_t)(i * 7 + 5);
    float s = oc_oxk_dot_q6_k_q8_k_scalar(w, 1, q);
    float d = oc_oxk_dot_q6_k_q8_k(w, 1, q);
    cr_assert_float_eq(s, d, 0.0f);
}


Test(oxk_ext, matvec_q8_0_known_result)
{
    /* 1 row, 1 block: d=1.0, weights all 3, input all 2.0
     * result = 1.0 * sum(3 * 2.0) = 32 * 6 = 192 */
    uint8_t w[34];
    float x[32];
    float out[1];
    w[0] = 0x00; w[1] = 0x3C;  /* f16 1.0 */
    for (int i = 0; i < 32; i++) {
        w[2 + i] = 3;
        x[i] = 2.0f;
    }
    oc_oxk_matvec_q8_0_f32_scalar(w, 1, 34, x, out);
    cr_assert_float_eq(out[0], 192.0f, 0.1f);
}

Test(oxk_ext, matvec_q8_0_multiple_rows)
{
    /* 3 rows, 1 block each, different weights per row. */
    uint8_t w[3 * 34];
    float x[32];
    float out[3];
    for (int i = 0; i < 32; i++) x[i] = 1.0f;
    for (int r = 0; r < 3; r++) {
        w[r * 34] = 0x00; w[r * 34 + 1] = 0x3C;  /* d=1.0 */
        for (int i = 0; i < 32; i++) w[r * 34 + 2 + i] = (int8_t)(r + 1);
    }
    oc_oxk_matvec_q8_0_f32_scalar(w, 3, 34, x, out);
    cr_assert_float_eq(out[0], 32.0f, 0.1f);
    cr_assert_float_eq(out[1], 64.0f, 0.1f);
    cr_assert_float_eq(out[2], 96.0f, 0.1f);
}


Test(oxk_ext, matvec_q4_0_known_result)
{
    /* d=1.0, nibbles all 9 (value=1), input all 3.0
     * result = 1.0 * sum(1 * 3.0) = 32 * 3 = 96 */
    uint8_t w[18];
    float x[32];
    float out[1];
    w[0] = 0x00; w[1] = 0x3C;  /* f16 1.0 */
    for (int i = 0; i < 16; i++) w[2 + i] = 0x99;  /* nibbles all 9 */
    for (int i = 0; i < 32; i++) x[i] = 3.0f;
    oc_oxk_matvec_q4_0_f32_scalar(w, 1, 18, x, out);
    cr_assert_float_eq(out[0], 96.0f, 0.1f);
}


Test(oxk_ext, matvec_q4_k_zero_block)
{
    uint8_t w[OC_OXK_BLOCK_Q4_K_SIZE];
    float x[256];
    float out[1];
    memset(w, 0, sizeof(w));
    for (int i = 0; i < 256; i++) x[i] = 1.0f;
    oc_oxk_matvec_q4_k_f32_scalar(w, 1, OC_OXK_BLOCK_Q4_K_SIZE, x, out);
    cr_assert_float_eq(out[0], 0.0f, 0.0f);
}

Test(oxk_ext, matvec_q4_k_dispatched_matches_scalar)
{
    uint8_t w[OC_OXK_BLOCK_Q4_K_SIZE];
    float x[256];
    float out_s[1], out_d[1];
    for (size_t i = 0; i < sizeof(w); i++) w[i] = (uint8_t)(i * 7 + 3);
    for (int i = 0; i < 256; i++) x[i] = (float)(i % 10) * 0.5f;
    oc_oxk_matvec_q4_k_f32_scalar(w, 1, OC_OXK_BLOCK_Q4_K_SIZE, x, out_s);
    oc_oxk_matvec_q4_k_f32(w, 1, OC_OXK_BLOCK_Q4_K_SIZE, x, out_d);
    cr_assert_float_eq(out_s[0], out_d[0], 0.0f);
}


Test(oxk_ext, dot_q8_0_large_256_blocks)
{
    /* 256 blocks of Q8_0, each contributing a known value. */
    enum { N = 256 };
    uint8_t *w = malloc(N * OC_OXK_BLOCK_Q8_0_SIZE);
    uint8_t *q = malloc(N * OC_OXK_BLOCK_Q8_0_SIZE);
    cr_assert_not_null(w);
    cr_assert_not_null(q);

    for (size_t b = 0; b < N; b++) {
        /* d_w = 1.0, d_q = 1.0 */
        w[b * 34] = 0x00; w[b * 34 + 1] = 0x3C;
        q[b * 34] = 0x00; q[b * 34 + 1] = 0x3C;
        for (int i = 0; i < 32; i++) {
            w[b * 34 + 2 + i] = 1;
            q[b * 34 + 2 + i] = 1;
        }
    }
    /* Each block: 1.0 * 1.0 * sum(1*1) = 32. Total = 256 * 32 = 8192. */
    float result = oc_oxk_dot_q8_0_q8_0_scalar(w, N, q);
    cr_assert_float_eq(result, 8192.0f, 1.0f,
                       "expected 8192.0, got %.2f", result);

    /* Dispatcher should match scalar. */
    float d = oc_oxk_dot_q8_0_q8_0(w, N, q);
    cr_assert_float_eq(d, result, 0.0f);

    free(w);
    free(q);
}

Test(oxk_ext, dot_q4_0_large_300_blocks)
{
    /* 300 blocks (> 256) of Q4_0. */
    enum { N = 300 };
    uint8_t *w = malloc(N * OC_OXK_BLOCK_Q4_0_SIZE);
    uint8_t *q = malloc(N * OC_OXK_BLOCK_Q8_0_SIZE);
    cr_assert_not_null(w);
    cr_assert_not_null(q);

    for (size_t b = 0; b < N; b++) {
        w[b * 18] = 0x00; w[b * 18 + 1] = 0x3C;  /* d=1.0 */
        q[b * 34] = 0x00; q[b * 34 + 1] = 0x3C;  /* d=1.0 */
        for (int i = 0; i < 16; i++) w[b * 18 + 2 + i] = 0x99;  /* nibble=9 → 1 */
        for (int i = 0; i < 32; i++) q[b * 34 + 2 + i] = 2;
    }
    /* Each block: 1*1*sum(1*2) = 64. Total = 300 * 64 = 19200. */
    float result = oc_oxk_dot_q4_0_q8_0_scalar(w, N, q);
    cr_assert_float_eq(result, 19200.0f, 2.0f);

    free(w);
    free(q);
}


Test(oxk_ext, parity_q8_0_scalar_vs_avx2)
{
    const OcOxkCaps *caps = oc_oxk_caps();
    if (caps->level < OC_OXK_AVX2) {
        cr_skip_test("AVX2 not available on this CPU");
        return;
    }
    uint8_t w[34], q[34];
    w[0] = 0x00; w[1] = 0x3C;
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) {
        w[2 + i] = (int8_t)(i - 16);
        q[2 + i] = (int8_t)(16 - i);
    }
    float s = oc_oxk_dot_q8_0_q8_0_scalar(w, 1, q);
    float a = oc_oxk_dot_q8_0_q8_0_avx2(w, 1, q);
    cr_assert_float_eq(s, a, 0.001f,
                       "AVX2 %.6f != scalar %.6f", a, s);
}

Test(oxk_ext, parity_q4_0_scalar_vs_avx2)
{
    const OcOxkCaps *caps = oc_oxk_caps();
    if (caps->level < OC_OXK_AVX2) {
        cr_skip_test("AVX2 not available on this CPU");
        return;
    }
    uint8_t w[18], q[34];
    w[0] = 0x00; w[1] = 0x3C;
    for (int i = 0; i < 16; i++) w[2 + i] = (uint8_t)(i * 17);
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) q[2 + i] = (int8_t)(i - 16);
    float s = oc_oxk_dot_q4_0_q8_0_scalar(w, 1, q);
    float a = oc_oxk_dot_q4_0_q8_0_avx2(w, 1, q);
    cr_assert_float_eq(s, a, 0.0f);
}

Test(oxk_ext, parity_q4_1_scalar_vs_avx2)
{
    const OcOxkCaps *caps = oc_oxk_caps();
    if (caps->level < OC_OXK_AVX2) {
        cr_skip_test("AVX2 not available on this CPU");
        return;
    }
    uint8_t w[20], q[34];
    w[0] = 0x00; w[1] = 0x3C;
    w[2] = 0x00; w[3] = 0x3C;
    for (int i = 0; i < 16; i++) w[4 + i] = (uint8_t)(i * 17);
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) q[2 + i] = (int8_t)(i - 16);
    float s = oc_oxk_dot_q4_1_q8_0_scalar(w, 1, q);
    float a = oc_oxk_dot_q4_1_q8_0_avx2(w, 1, q);
    cr_assert_float_eq(s, a, 0.0f);
}


Test(oxk_ext, parity_q8_0_scalar_vs_avx512_vnni)
{
    const OcOxkCaps *caps = oc_oxk_caps();
    if (caps->level < OC_OXK_AVX512 || !caps->has_vnni) {
        cr_skip_test("AVX-512 VNNI not available on this CPU");
        return;
    }
    uint8_t w[34], q[34];
    w[0] = 0x00; w[1] = 0x3C;
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) {
        w[2 + i] = (int8_t)(i - 16);
        q[2 + i] = (int8_t)(16 - i);
    }
    float s = oc_oxk_dot_q8_0_q8_0_scalar(w, 1, q);
    float a = oc_oxk_dot_q8_0_q8_0_avx512_vnni(w, 1, q);
    cr_assert_float_eq(s, a, 0.001f,
                       "AVX-512 VNNI %.6f != scalar %.6f", a, s);
}

Test(oxk_ext, parity_q4_0_scalar_vs_avx512_bw)
{
    const OcOxkCaps *caps = oc_oxk_caps();
    if (caps->level < OC_OXK_AVX512 || !caps->has_vnni) {
        cr_skip_test("AVX-512 not available on this CPU");
        return;
    }
    uint8_t w[18], q[34];
    w[0] = 0x00; w[1] = 0x3C;
    for (int i = 0; i < 16; i++) w[2 + i] = (uint8_t)(i * 17);
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) q[2 + i] = (int8_t)(i - 16);
    float s = oc_oxk_dot_q4_0_q8_0_scalar(w, 1, q);
    float a = oc_oxk_dot_q4_0_q8_0_avx512_bw(w, 1, q);
    cr_assert_float_eq(s, a, 0.001f,
                       "AVX-512 BW %.6f != scalar %.6f", a, s);
}

Test(oxk_ext, parity_q4_1_scalar_vs_avx512_bw)
{
    const OcOxkCaps *caps = oc_oxk_caps();
    if (caps->level < OC_OXK_AVX512 || !caps->has_vnni) {
        cr_skip_test("AVX-512 not available on this CPU");
        return;
    }
    uint8_t w[20], q[34];
    w[0] = 0x00; w[1] = 0x3C;  /* d=1.0 */
    w[2] = 0x00; w[3] = 0x3C;  /* m=1.0 */
    for (int i = 0; i < 16; i++) w[4 + i] = (uint8_t)(i * 17);
    q[0] = 0x00; q[1] = 0x3C;
    for (int i = 0; i < 32; i++) q[2 + i] = (int8_t)(i - 16);
    float s = oc_oxk_dot_q4_1_q8_0_scalar(w, 1, q);
    float a = oc_oxk_dot_q4_1_q8_0_avx512_bw(w, 1, q);
    cr_assert_float_eq(s, a, 0.5f,
                       "AVX-512 BW Q4_1 %.6f != scalar %.6f", a, s);
}

Test(oxk_ext, parity_q4_0_matvec_scalar_vs_avx512_bw)
{
    const OcOxkCaps *caps = oc_oxk_caps();
    if (caps->level < OC_OXK_AVX512 || !caps->has_vnni) {
        cr_skip_test("AVX-512 not available on this CPU");
        return;
    }
    /* 2 rows, 4 blocks each. */
    enum { ROWS = 2, BLOCKS = 4 };
    size_t row_bytes = BLOCKS * OC_OXK_BLOCK_Q4_0_SIZE;
    uint8_t w[ROWS * BLOCKS * OC_OXK_BLOCK_Q4_0_SIZE];
    float x[BLOCKS * 32];
    float out_s[ROWS], out_a[ROWS];

    /* Fill with deterministic data, then overwrite each block's f16 scale
     * with 1.0 (0x3C00) — arbitrary bytes can decode to NaN/Inf scales,
     * and NaN outputs make the parity assertions below meaningless. */
    for (size_t i = 0; i < sizeof(w); i++) w[i] = (uint8_t)(i * 7 + 3);
    for (size_t b = 0; b < ROWS * BLOCKS; b++) {
        w[b * OC_OXK_BLOCK_Q4_0_SIZE]     = 0x00;
        w[b * OC_OXK_BLOCK_Q4_0_SIZE + 1] = 0x3C;
    }
    for (size_t i = 0; i < BLOCKS * 32; i++) x[i] = (float)((int)i % 10) * 0.5f;

    oc_oxk_matvec_q4_0_f32_scalar(w, ROWS, row_bytes, x, out_s);
    oc_oxk_matvec_q4_0_f32_avx512_bw(w, ROWS, row_bytes, x, out_a);

    /* Allow small tolerance due to FP summation order differences. */
    cr_assert_float_eq(out_s[0], out_a[0], 1.0f,
                       "row 0: AVX-512 BW %.4f != scalar %.4f",
                       out_a[0], out_s[0]);
    cr_assert_float_eq(out_s[1], out_a[1], 1.0f,
                       "row 1: AVX-512 BW %.4f != scalar %.4f",
                       out_a[1], out_s[1]);
}


Test(oxk_ext, matvec_q8_0_zero_input)
{
    uint8_t w[34];
    float x[32];
    float out[1];
    memset(w, 0, sizeof(w));
    for (int i = 0; i < 32; i++) x[i] = 0.0f;
    oc_oxk_matvec_q8_0_f32_scalar(w, 1, 34, x, out);
    cr_assert_float_eq(out[0], 0.0f, 0.0f);
}
