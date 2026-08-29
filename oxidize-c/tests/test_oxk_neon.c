/* test_oxk_neon.c — AArch64 NEON ↔ scalar parity for the OXK dot products. */
/* - Q4_0, Q4_1, Q8_0, Q4_K, Q5_K → bit-exact (tolerance 0.0f). */
/* - Q6_K → not bit-exact by construction (the scalar reference adds one f32 */
#include <criterion/criterion.h>

#include "oxidize/oxk.h"
#include "oxidize/oxk_neon.h"

#if defined(__aarch64__)

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Deterministic xorshift so failures are reproducible without a seed file. */
static uint32_t g_state = 0x9E3779B9u;

static void rng_reset(uint32_t seed) { g_state = seed ? seed : 1u; }

static uint32_t rng_next(void)
{
    uint32_t x = g_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_state = x;
    return x;
}

static void fill_random(uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)(rng_next() >> 13);
}

/* f16 scales are written explicitly so the blocks never contain NaN/Inf
 * exponents, which would make "bit-exact" comparisons meaningless. */
static void put_f16(uint8_t *p, uint16_t bits) { p[0] = (uint8_t)(bits & 0xFF); p[1] = (uint8_t)(bits >> 8); }

#define N_BLOCKS 7

static void build_q8_0(uint8_t *buf, size_t blocks)
{
    for (size_t b = 0; b < blocks; b++) {
        uint8_t *p = buf + b * OC_OXK_BLOCK_Q8_0_SIZE;
        put_f16(p, 0x3800 + (uint16_t)(b * 3));  /* ~0.5 .. small positives */
        fill_random(p + 2, 32);
    }
}

static void build_q8_k(uint8_t *buf, size_t blocks)
{
    for (size_t b = 0; b < blocks; b++) {
        uint8_t *p = buf + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float d = 0.0125f + 0.001f * (float)b;
        memcpy(p, &d, 4);
        fill_random(p + 4, 256);
        /* bsums must be the real per-16 sums for the Q4_K/Q5_K min term to
         * mean anything, but parity only needs both kernels to read the same
         * bytes — fill them consistently from the quantized values. */
        const int8_t *q = (const int8_t *)(p + 4);
        for (int g = 0; g < 16; g++) {
            int32_t s = 0;
            for (int i = 0; i < 16; i++) s += (int32_t)q[g * 16 + i];
            int16_t s16 = (int16_t)s;
            p[4 + 256 + g * 2]     = (uint8_t)((uint16_t)s16 & 0xFF);
            p[4 + 256 + g * 2 + 1] = (uint8_t)((uint16_t)s16 >> 8);
        }
    }
}

Test(oxk_neon, caps_report_neon)
{
    const OcOxkCaps *caps = oc_oxk_caps();
    cr_assert_eq(caps->level, OC_OXK_NEON, "AArch64 build must detect NEON");
    cr_assert(caps->has_neon);
    cr_assert_str_eq(caps->name, "neon");
}

Test(oxk_neon, dispatch_selects_neon)
{
    const OcOxkContext *ctx = oc_oxk_init();
    cr_assert_eq(ctx->dot_q8_0_q8_0, oc_oxk_dot_q8_0_q8_0_neon);
    cr_assert_eq(ctx->dot_q4_0_q8_0, oc_oxk_dot_q4_0_q8_0_neon);
    cr_assert_eq(ctx->dot_q4_k_q8_k, oc_oxk_dot_q4_k_q8_k_neon);
    /* Matvecs intentionally stay on the scalar reference. */
    cr_assert_eq(ctx->matvec_q8_0_f32, oc_oxk_matvec_q8_0_f32_scalar);
}

Test(oxk_neon, dot_q8_0_q8_0_bit_exact)
{
    rng_reset(1);
    uint8_t w[N_BLOCKS * OC_OXK_BLOCK_Q8_0_SIZE];
    uint8_t q[N_BLOCKS * OC_OXK_BLOCK_Q8_0_SIZE];
    build_q8_0(w, N_BLOCKS);
    build_q8_0(q, N_BLOCKS);
    cr_assert_float_eq(oc_oxk_dot_q8_0_q8_0_neon(w, N_BLOCKS, q),
                       oc_oxk_dot_q8_0_q8_0_scalar(w, N_BLOCKS, q), 0.0f);
}

Test(oxk_neon, dot_q4_0_q8_0_bit_exact)
{
    rng_reset(2);
    uint8_t w[N_BLOCKS * OC_OXK_BLOCK_Q4_0_SIZE];
    uint8_t q[N_BLOCKS * OC_OXK_BLOCK_Q8_0_SIZE];
    for (size_t b = 0; b < N_BLOCKS; b++) {
        uint8_t *p = w + b * OC_OXK_BLOCK_Q4_0_SIZE;
        put_f16(p, 0x3400 + (uint16_t)b);
        fill_random(p + 2, 16);
    }
    build_q8_0(q, N_BLOCKS);
    cr_assert_float_eq(oc_oxk_dot_q4_0_q8_0_neon(w, N_BLOCKS, q),
                       oc_oxk_dot_q4_0_q8_0_scalar(w, N_BLOCKS, q), 0.0f);
}

Test(oxk_neon, dot_q4_1_q8_0_bit_exact)
{
    rng_reset(3);
    uint8_t w[N_BLOCKS * OC_OXK_BLOCK_Q4_1_SIZE];
    uint8_t q[N_BLOCKS * OC_OXK_BLOCK_Q8_0_SIZE];
    for (size_t b = 0; b < N_BLOCKS; b++) {
        uint8_t *p = w + b * OC_OXK_BLOCK_Q4_1_SIZE;
        put_f16(p, 0x3400 + (uint16_t)b);
        put_f16(p + 2, 0xB400 + (uint16_t)b);  /* negative min */
        fill_random(p + 4, 16);
    }
    build_q8_0(q, N_BLOCKS);
    cr_assert_float_eq(oc_oxk_dot_q4_1_q8_0_neon(w, N_BLOCKS, q),
                       oc_oxk_dot_q4_1_q8_0_scalar(w, N_BLOCKS, q), 0.0f);
}

Test(oxk_neon, dot_q4_k_q8_k_bit_exact)
{
    rng_reset(4);
    uint8_t w[N_BLOCKS * OC_OXK_BLOCK_Q4_K_SIZE];
    uint8_t q[N_BLOCKS * OC_OXK_BLOCK_Q8_K_SIZE];
    for (size_t b = 0; b < N_BLOCKS; b++) {
        uint8_t *p = w + b * OC_OXK_BLOCK_Q4_K_SIZE;
        put_f16(p, 0x3000 + (uint16_t)b);
        put_f16(p + 2, 0x3100 + (uint16_t)b);
        fill_random(p + 4, OC_OXK_BLOCK_Q4_K_SIZE - 4);
    }
    build_q8_k(q, N_BLOCKS);
    cr_assert_float_eq(oc_oxk_dot_q4_k_q8_k_neon(w, N_BLOCKS, q),
                       oc_oxk_dot_q4_k_q8_k_scalar(w, N_BLOCKS, q), 0.0f);
}

Test(oxk_neon, dot_q5_k_q8_k_bit_exact)
{
    rng_reset(5);
    uint8_t w[N_BLOCKS * OC_OXK_BLOCK_Q5_K_SIZE];
    uint8_t q[N_BLOCKS * OC_OXK_BLOCK_Q8_K_SIZE];
    for (size_t b = 0; b < N_BLOCKS; b++) {
        uint8_t *p = w + b * OC_OXK_BLOCK_Q5_K_SIZE;
        put_f16(p, 0x3000 + (uint16_t)b);
        put_f16(p + 2, 0x3100 + (uint16_t)b);
        fill_random(p + 4, OC_OXK_BLOCK_Q5_K_SIZE - 4);
    }
    build_q8_k(q, N_BLOCKS);
    cr_assert_float_eq(oc_oxk_dot_q5_k_q8_k_neon(w, N_BLOCKS, q),
                       oc_oxk_dot_q5_k_q8_k_scalar(w, N_BLOCKS, q), 0.0f);
}

Test(oxk_neon, dot_q6_k_q8_k_close)
{
    rng_reset(6);
    uint8_t w[N_BLOCKS * OC_OXK_BLOCK_Q6_K_SIZE];
    uint8_t q[N_BLOCKS * OC_OXK_BLOCK_Q8_K_SIZE];
    for (size_t b = 0; b < N_BLOCKS; b++) {
        uint8_t *p = w + b * OC_OXK_BLOCK_Q6_K_SIZE;
        fill_random(p, 208);
        put_f16(p + 208, 0x3000 + (uint16_t)b);
    }
    build_q8_k(q, N_BLOCKS);
    float s = oc_oxk_dot_q6_k_q8_k_scalar(w, N_BLOCKS, q);
    float n = oc_oxk_dot_q6_k_q8_k_neon(w, N_BLOCKS, q);
    /* Documented non-bit-exact kernel: relative tolerance only. */
    float scale = fabsf(s) > 1.0f ? fabsf(s) : 1.0f;
    cr_assert(fabsf(s - n) <= 1e-4f * scale,
              "q6_k neon %g vs scalar %g", (double)n, (double)s);
}

Test(oxk_neon, dot_zero_blocks_is_zero)
{
    uint8_t dummy[OC_OXK_BLOCK_Q8_K_SIZE] = {0};
    cr_assert_float_eq(oc_oxk_dot_q8_0_q8_0_neon(dummy, 0, dummy), 0.0f, 0.0f);
    cr_assert_float_eq(oc_oxk_dot_q4_k_q8_k_neon(dummy, 0, dummy), 0.0f, 0.0f);
    cr_assert_float_eq(oc_oxk_dot_q6_k_q8_k_neon(dummy, 0, dummy), 0.0f, 0.0f);
}

#else  /* !__aarch64__ */

Test(oxk_neon, skipped_on_non_aarch64)
{
    cr_assert(true, "NEON kernels are AArch64-only; nothing to verify here");
}

#endif /* __aarch64__ */
