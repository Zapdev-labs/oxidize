/*
 * test_oxk_avx2_parity.c — vectorized OXK kernels vs the scalar reference.
 *
 * The OXK invariant is bit-exactness, not approximate agreement: a kernel that
 * merely rounds differently would silently change model output depending on
 * which CPU it ran on. These compare raw float bits over randomized, properly
 * packed blocks. A tolerance here would defeat the purpose.
 *
 * Skipped at runtime on hosts without the relevant ISA — the functions exist
 * in every build, but calling them where unsupported would fault.
 */
#include <criterion/criterion.h>

#include "oxidize/flash_attention.h"
#include "oxidize/oxk.h"
#include "oxidize/quant.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define COLS 2048u
#define NB   (COLS / 256u)

static float frand(uint32_t *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (float)((int32_t)(*s >> 8) % 2000 - 1000) / 1000.0f;
}

/* Build a Q8_K activation matching matvec.c's encoder. */
static void make_act_q8_k(const float *x, size_t n, uint8_t *out)
{
    for (size_t b = 0; b < n / 256; b++) {
        const float *s = x + b * 256;
        float am = 0.0f;
        for (int i = 0; i < 256; i++) {
            float a = s[i] < 0 ? -s[i] : s[i];
            if (a > am) am = a;
        }
        const float d = am / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t *dst = out + b * OC_OXK_BLOCK_Q8_K_SIZE;
        memcpy(dst, &d, 4);
        int8_t *q = (int8_t *)(dst + 4);
        for (int i = 0; i < 256; i++) {
            int v = (int)(s[i] * id + (s[i] >= 0 ? 0.5f : -0.5f));
            q[i] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
        }
        uint8_t *bs = dst + 4 + 256;
        for (int g = 0; g < 16; g++) {
            int32_t t = 0;
            for (int i = 0; i < 16; i++) t += q[g * 16 + i];
            int16_t t16 = (int16_t)t;
            memcpy(bs + g * 2, &t16, 2);
        }
    }
}

Test(oxk_avx2, q4_k_bit_exact_vs_scalar)
{
    if (oc_oxk_caps()->level < OC_OXK_AVX2 ||
        oc_oxk_caps()->level == OC_OXK_NEON) {
        cr_skip("host has no AVX2");
    }

    const size_t rb = oc_quantized_size(OC_QUANT_Q4_K_M, COLS);
    cr_assert_gt(rb, 0);

    for (uint32_t seed = 1; seed <= 32; seed++) {
        uint32_t s = seed;
        float *wf = malloc(COLS * sizeof(float));
        float *x  = malloc(COLS * sizeof(float));
        uint8_t *w = malloc(rb);
        uint8_t *act = malloc(NB * OC_OXK_BLOCK_Q8_K_SIZE);
        cr_assert_not_null(wf); cr_assert_not_null(x);
        cr_assert_not_null(w);  cr_assert_not_null(act);

        for (size_t i = 0; i < COLS; i++) wf[i] = frand(&s);
        cr_assert_eq(oc_quant_pack_row(OC_QUANT_Q4_K_M, wf, COLS, w, rb), OC_OK);
        for (size_t i = 0; i < COLS; i++) x[i] = frand(&s);
        make_act_q8_k(x, COLS, act);

        const float sc = oc_oxk_dot_q4_k_q8_k_scalar(w, NB, act);
        const float av = oc_oxk_dot_q4_k_q8_k_avx2(w, NB, act);
        cr_assert_arr_eq(&av, &sc, sizeof(float),
                         "seed %u: AVX2 %.9g != scalar %.9g", seed, av, sc);

        free(wf); free(x); free(w); free(act);
    }
}

/* The dispatcher must route to a kernel that agrees with scalar, whichever
 * tier it picked — this is what actually runs in production. */
Test(oxk_avx2, dispatched_q4_k_matches_scalar)
{
    const size_t rb = oc_quantized_size(OC_QUANT_Q4_K_M, COLS);
    uint32_t s = 4242;
    float *wf = malloc(COLS * sizeof(float));
    float *x  = malloc(COLS * sizeof(float));
    uint8_t *w = malloc(rb);
    uint8_t *act = malloc(NB * OC_OXK_BLOCK_Q8_K_SIZE);
    cr_assert_not_null(wf); cr_assert_not_null(x);
    cr_assert_not_null(w);  cr_assert_not_null(act);

    for (size_t i = 0; i < COLS; i++) wf[i] = frand(&s);
    cr_assert_eq(oc_quant_pack_row(OC_QUANT_Q4_K_M, wf, COLS, w, rb), OC_OK);
    for (size_t i = 0; i < COLS; i++) x[i] = frand(&s);
    make_act_q8_k(x, COLS, act);

    const float sc = oc_oxk_dot_q4_k_q8_k_scalar(w, NB, act);
    const float dp = oc_oxk_dot_q4_k_q8_k(w, NB, act);
    cr_assert_arr_eq(&dp, &sc, sizeof(float),
                     "dispatched %.9g != scalar %.9g", dp, sc);

    free(wf); free(x); free(w); free(act);
}

/* ─── Raw-byte blocks: every scale/qh bit pattern, not just packer output ── */

static bool host_avx2(void)
{
    const OcOxkLevel l = oc_oxk_caps()->level;
    return l == OC_OXK_AVX2 || l == OC_OXK_AVX512;
}

static uint32_t rnd32(uint32_t *s)
{
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}

/* Fill `blocks` blocks of `bsize` bytes with random bytes, then overwrite the
 * f16 super-scale(s) at `d_off` (and `d_off + 2` when `two_scales`) with
 * finite values so the result is a well-formed block of that type. */
static void rand_blocks(uint8_t *w, size_t blocks, size_t bsize, size_t d_off,
                        bool two_scales, uint32_t *s)
{
    for (size_t b = 0; b < blocks; b++) {
        uint8_t *xb = w + b * bsize;
        for (size_t i = 0; i < bsize; i++) xb[i] = (uint8_t)rnd32(s);
        for (int k = 0; k < (two_scales ? 2 : 1); k++) {
            const float d = 0.0005f + (float)(rnd32(s) % 1000u) * 4e-5f;
            const uint16_t h = oc_f32_to_f16_bits(d);
            xb[d_off + 2 * k] = (uint8_t)(h & 0xFF);
            xb[d_off + 2 * k + 1] = (uint8_t)(h >> 8);
        }
    }
}

/* Q8_K activation over [-127, 127], the range quantize_act_q8_k produces
 * (d = amax/127, so |q| never rounds past 127). -128 is excluded on purpose:
 * IQ3_S's sign flip (xor/sub, as in ggml) wraps -(-128) to -128 in the SIMD
 * form while the scalar reference computes +128. */
static void rand_act_q8_k(uint8_t *act, size_t blocks, uint32_t *s)
{
    for (size_t b = 0; b < blocks; b++) {
        uint8_t *yb = act + b * OC_OXK_BLOCK_Q8_K_SIZE;
        const float d = 0.001f + (float)(rnd32(s) % 1000u) * 2e-5f;
        memcpy(yb, &d, 4);
        int8_t *q = (int8_t *)(yb + 4);
        for (int i = 0; i < 256; i++)
            q[i] = (int8_t)((int32_t)(rnd32(s) % 255u) - 127);
        for (int g = 0; g < 16; g++) {
            int32_t t = 0;
            for (int i = 0; i < 16; i++) t += q[g * 16 + i];
            const int16_t t16 = (int16_t)t;
            memcpy(yb + 4 + 256 + g * 2, &t16, 2);
        }
    }
}

Test(oxk_avx2, q4_k_raw_blocks_bit_exact_vs_scalar)
{
    if (!host_avx2()) cr_skip("host has no AVX2");
    uint32_t s = 0x9E3779B9u;
    const size_t nb = 12;
    uint8_t *w = malloc(nb * OC_OXK_BLOCK_Q4_K_SIZE);
    uint8_t *a = malloc(nb * OC_OXK_BLOCK_Q8_K_SIZE);
    cr_assert(w && a);
    for (int t = 0; t < 64; t++) {
        rand_blocks(w, nb, OC_OXK_BLOCK_Q4_K_SIZE, 0, true, &s);
        rand_act_q8_k(a, nb, &s);
        for (size_t blocks = 1; blocks <= nb; blocks += 11) {
            const float sc = oc_oxk_dot_q4_k_q8_k_scalar(w, blocks, a);
            const float av = oc_oxk_dot_q4_k_q8_k_avx2(w, blocks, a);
            cr_assert(memcmp(&sc, &av, sizeof sc) == 0,
                      "trial %d blocks %zu: avx2 %.9g != scalar %.9g", t,
                      blocks, (double)av, (double)sc);
        }
    }
    free(w);
    free(a);
}

Test(oxk_avx2, q6_k_raw_blocks_bit_exact_vs_scalar)
{
    if (!host_avx2()) cr_skip("host has no AVX2");
    uint32_t s = 0x7F4A7C15u;
    const size_t nb = 12;
    uint8_t *w = malloc(nb * OC_OXK_BLOCK_Q6_K_SIZE);
    uint8_t *a = malloc(nb * OC_OXK_BLOCK_Q8_K_SIZE);
    cr_assert(w && a);
    for (int t = 0; t < 64; t++) {
        rand_blocks(w, nb, OC_OXK_BLOCK_Q6_K_SIZE, 208, false, &s);
        rand_act_q8_k(a, nb, &s);
        for (size_t blocks = 1; blocks <= nb; blocks += 11) {
            const float sc = oc_oxk_dot_q6_k_q8_k_scalar(w, blocks, a);
            const float dp = oc_oxk_dot_q6_k_q8_k(w, blocks, a);
            cr_assert(memcmp(&sc, &dp, sizeof sc) == 0,
                      "trial %d blocks %zu: dispatched %.9g != scalar %.9g",
                      t, blocks, (double)dp, (double)sc);
            if (host_avx2()) {
                const float av = oc_oxk_dot_q6_k_q8_k_avx2(w, blocks, a);
                cr_assert(memcmp(&sc, &av, sizeof sc) == 0,
                          "trial %d blocks %zu: avx2 %.9g != scalar %.9g", t,
                          blocks, (double)av, (double)sc);
            }
        }
    }
    free(w);
    free(a);
}

/* oc_oxk_dot_rows_*: one call over n consecutive rows must equal n
 * single-row calls to the bit, for the dispatcher and the AVX2 entry, across
 * the short rows (3 blocks) the MoE down projection uses. */
typedef float (*OxkDot1)(const uint8_t *, size_t, const uint8_t *);
typedef void (*OxkDotRows)(const uint8_t *, size_t, size_t, size_t,
                           const uint8_t *, float *);

static void check_rows(size_t bsize, size_t d_off, bool two_scales,
                       OxkDot1 scalar, OxkDotRows rows, OxkDotRows rows_avx2,
                       const char *name)
{
    uint32_t s = 0xA5A5F00Du;
    static const size_t k_blocks[] = { 1, 3, 8 };
    enum { MAX_ROWS = 9, MAX_BLOCKS = 8 };
    uint8_t *w = malloc(MAX_ROWS * MAX_BLOCKS * bsize);
    uint8_t *a = malloc(MAX_BLOCKS * OC_OXK_BLOCK_Q8_K_SIZE);
    cr_assert(w && a);
    for (size_t bi = 0; bi < sizeof k_blocks / sizeof k_blocks[0]; bi++) {
        const size_t blocks = k_blocks[bi];
        const size_t rb = blocks * bsize;
        rand_blocks(w, MAX_ROWS * blocks, bsize, d_off, two_scales, &s);
        rand_act_q8_k(a, blocks, &s);
        for (size_t n = 1; n <= MAX_ROWS; n++) {
            float ref[MAX_ROWS], got[MAX_ROWS], gav[MAX_ROWS];
            for (size_t r = 0; r < n; r++) ref[r] = scalar(w + r * rb, blocks, a);
            rows(w, rb, n, blocks, a, got);
            cr_assert(memcmp(ref, got, n * sizeof(float)) == 0,
                      "%s rows dispatch blocks=%zu n=%zu", name, blocks, n);
            if (host_avx2()) {
                rows_avx2(w, rb, n, blocks, a, gav);
                cr_assert(memcmp(ref, gav, n * sizeof(float)) == 0,
                          "%s rows avx2 blocks=%zu n=%zu", name, blocks, n);
            }
        }
    }
    free(w);
    free(a);
}

Test(oxk_avx2, dot_rows_q4_k_bit_exact_vs_single)
{
    check_rows(OC_OXK_BLOCK_Q4_K_SIZE, 0, true, oc_oxk_dot_q4_k_q8_k_scalar,
               oc_oxk_dot_rows_q4_k_q8_k, oc_oxk_dot_rows_q4_k_q8_k_avx2, "q4_k");
}

Test(oxk_avx2, dot_rows_q6_k_bit_exact_vs_single)
{
    check_rows(OC_OXK_BLOCK_Q6_K_SIZE, 208, false, oc_oxk_dot_q6_k_q8_k_scalar,
               oc_oxk_dot_rows_q6_k_q8_k, oc_oxk_dot_rows_q6_k_q8_k_avx2, "q6_k");
}

Test(oxk_avx2, dot_rows_iq3_s_bit_exact_vs_single)
{
    check_rows(OC_OXK_BLOCK_IQ3_S_SIZE, 0, false, oc_oxk_dot_iq3_s_q8_k_scalar,
               oc_oxk_dot_rows_iq3_s_q8_k, oc_oxk_dot_rows_iq3_s_q8_k_avx2,
               "iq3_s");
}

/* ─── The Q8_0-activation family: raw bytes, activations, rows ────────── */

/* Q8_0 activation over the encoder's full [-128, 127] range. The 4-bit and
 * 3-bit weight kernels stay inside their madd bounds there (only the
 * Q8_0 × Q8_0 pair itself cannot, which is why that kernel is float-lane). */
static void rand_act_q8_0(uint8_t *act, size_t blocks, uint32_t *s)
{
    for (size_t b = 0; b < blocks; b++) {
        uint8_t *yb = act + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const float d = 0.001f + (float)(rnd32(s) % 1000u) * 2e-5f;
        const uint16_t h = oc_f32_to_f16_bits(d);
        yb[0] = (uint8_t)(h & 0xFF);
        yb[1] = (uint8_t)(h >> 8);
        int8_t *q = (int8_t *)(yb + 2);
        for (int i = 0; i < 32; i++)
            q[i] = (int8_t)((int32_t)(rnd32(s) % 256u) - 128);
    }
}

/* Raw-byte bit-exact check over one Q8_0-activation dot pair. */
static void check_q80_pair(size_t bsize, size_t d_off, bool two_scales,
                           OxkDot1 scalar, OxkDot1 avx2, OxkDot1 dispatched,
                           const char *name)
{
    uint32_t s = 0x2545F491u;
    const size_t nb = 12;
    uint8_t *w = malloc(nb * bsize);
    uint8_t *a = malloc(nb * OC_OXK_BLOCK_Q8_0_SIZE);
    cr_assert(w && a);
    for (int t = 0; t < 64; t++) {
        rand_blocks(w, nb, bsize, d_off, two_scales, &s);
        rand_act_q8_0(a, nb, &s);
        for (size_t blocks = 1; blocks <= nb; blocks += 11) {
            const float sc = scalar(w, blocks, a);
            const float av = avx2(w, blocks, a);
            cr_assert(memcmp(&sc, &av, sizeof sc) == 0,
                      "%s trial %d blocks %zu: avx2 %.9g != scalar %.9g", name,
                      t, blocks, (double)av, (double)sc);
            const float dp = dispatched(w, blocks, a);
            cr_assert(memcmp(&sc, &dp, sizeof sc) == 0,
                      "%s trial %d blocks %zu: dispatched %.9g != scalar %.9g",
                      name, t, blocks, (double)dp, (double)sc);
        }
    }
    free(w);
    free(a);
}

Test(oxk_avx2, q4_1_raw_blocks_bit_exact_vs_scalar)
{
    if (!host_avx2()) cr_skip("host has no AVX2");
    check_q80_pair(OC_OXK_BLOCK_Q4_1_SIZE, 0, true, oc_oxk_dot_q4_1_q8_0_scalar,
                   oc_oxk_dot_q4_1_q8_0_avx2, oc_oxk_dot_q4_1_q8_0, "q4_1");
}

Test(oxk_avx2, q5_0_raw_blocks_bit_exact_vs_scalar)
{
    if (!host_avx2()) cr_skip("host has no AVX2");
    check_q80_pair(OC_OXK_BLOCK_Q5_0_SIZE, 0, false, oc_oxk_dot_q5_0_q8_0_scalar,
                   oc_oxk_dot_q5_0_q8_0_avx2, oc_oxk_dot_q5_0_q8_0, "q5_0");
}

Test(oxk_avx2, q5_1_raw_blocks_bit_exact_vs_scalar)
{
    if (!host_avx2()) cr_skip("host has no AVX2");
    check_q80_pair(OC_OXK_BLOCK_Q5_1_SIZE, 0, true, oc_oxk_dot_q5_1_q8_0_scalar,
                   oc_oxk_dot_q5_1_q8_0_avx2, oc_oxk_dot_q5_1_q8_0, "q5_1");
}

Test(oxk_avx2, al5_xs_raw_blocks_bit_exact_vs_scalar)
{
    if (!host_avx2()) cr_skip("host has no AVX2");
    check_q80_pair(OC_OXK_BLOCK_AL5_XS_SIZE, 0, false,
                   oc_oxk_dot_al5_xs_q8_0_scalar, oc_oxk_dot_al5_xs_q8_0_avx2,
                   oc_oxk_dot_al5_xs_q8_0, "al5_xs");
}

Test(oxk_avx2, q5_k_raw_blocks_bit_exact_vs_scalar)
{
    if (!host_avx2()) cr_skip("host has no AVX2");
    uint32_t s = 0x85EBCA6Bu;
    const size_t nb = 12;
    uint8_t *w = malloc(nb * OC_OXK_BLOCK_Q5_K_SIZE);
    uint8_t *a = malloc(nb * OC_OXK_BLOCK_Q8_K_SIZE);
    cr_assert(w && a);
    for (int t = 0; t < 64; t++) {
        rand_blocks(w, nb, OC_OXK_BLOCK_Q5_K_SIZE, 0, true, &s);
        rand_act_q8_k(a, nb, &s);
        for (size_t blocks = 1; blocks <= nb; blocks += 11) {
            const float sc = oc_oxk_dot_q5_k_q8_k_scalar(w, blocks, a);
            const float av = oc_oxk_dot_q5_k_q8_k_avx2(w, blocks, a);
            cr_assert(memcmp(&sc, &av, sizeof sc) == 0,
                      "trial %d blocks %zu: avx2 %.9g != scalar %.9g", t,
                      blocks, (double)av, (double)sc);
            const float dp = oc_oxk_dot_q5_k_q8_k(w, blocks, a);
            cr_assert(memcmp(&sc, &dp, sizeof sc) == 0,
                      "trial %d blocks %zu: dispatched %.9g != scalar %.9g",
                      t, blocks, (double)dp, (double)sc);
        }
    }
    free(w);
    free(a);
}

/* Rows forms over the Q8_0-activation family: one call over n consecutive
 * rows equals n single-row calls to the bit, dispatcher and AVX2 entry. */
static void check_rows_q80(size_t bsize, size_t d_off, bool two_scales,
                           OxkDot1 scalar, OxkDotRows rows,
                           OxkDotRows rows_avx2, const char *name)
{
    uint32_t s = 0xD1B54A32u;
    static const size_t k_blocks[] = { 1, 3, 8 };
    enum { MAX_ROWS = 9, MAX_BLOCKS = 8 };
    uint8_t *w = malloc(MAX_ROWS * MAX_BLOCKS * bsize);
    uint8_t *a = malloc(MAX_BLOCKS * OC_OXK_BLOCK_Q8_0_SIZE);
    cr_assert(w && a);
    for (size_t bi = 0; bi < sizeof k_blocks / sizeof k_blocks[0]; bi++) {
        const size_t blocks = k_blocks[bi];
        const size_t rb = blocks * bsize;
        rand_blocks(w, MAX_ROWS * blocks, bsize, d_off, two_scales, &s);
        rand_act_q8_0(a, blocks, &s);
        for (size_t n = 1; n <= MAX_ROWS; n++) {
            float ref[MAX_ROWS], got[MAX_ROWS], gav[MAX_ROWS];
            for (size_t r = 0; r < n; r++) ref[r] = scalar(w + r * rb, blocks, a);
            rows(w, rb, n, blocks, a, got);
            cr_assert(memcmp(ref, got, n * sizeof(float)) == 0,
                      "%s rows dispatch blocks=%zu n=%zu", name, blocks, n);
            if (host_avx2()) {
                rows_avx2(w, rb, n, blocks, a, gav);
                cr_assert(memcmp(ref, gav, n * sizeof(float)) == 0,
                          "%s rows avx2 blocks=%zu n=%zu", name, blocks, n);
            }
        }
    }
    free(w);
    free(a);
}

Test(oxk_avx2, dot_rows_q4_0_bit_exact_vs_single)
{
    check_rows_q80(OC_OXK_BLOCK_Q4_0_SIZE, 0, false,
                   oc_oxk_dot_q4_0_q8_0_scalar, oc_oxk_dot_rows_q4_0_q8_0,
                   oc_oxk_dot_rows_q4_0_q8_0_avx2, "q4_0");
}

Test(oxk_avx2, dot_rows_q4_1_bit_exact_vs_single)
{
    check_rows_q80(OC_OXK_BLOCK_Q4_1_SIZE, 0, true,
                   oc_oxk_dot_q4_1_q8_0_scalar, oc_oxk_dot_rows_q4_1_q8_0,
                   oc_oxk_dot_rows_q4_1_q8_0_avx2, "q4_1");
}

Test(oxk_avx2, dot_rows_q5_0_bit_exact_vs_single)
{
    check_rows_q80(OC_OXK_BLOCK_Q5_0_SIZE, 0, false,
                   oc_oxk_dot_q5_0_q8_0_scalar, oc_oxk_dot_rows_q5_0_q8_0,
                   oc_oxk_dot_rows_q5_0_q8_0_avx2, "q5_0");
}

Test(oxk_avx2, dot_rows_q5_1_bit_exact_vs_single)
{
    check_rows_q80(OC_OXK_BLOCK_Q5_1_SIZE, 0, true,
                   oc_oxk_dot_q5_1_q8_0_scalar, oc_oxk_dot_rows_q5_1_q8_0,
                   oc_oxk_dot_rows_q5_1_q8_0_avx2, "q5_1");
}

Test(oxk_avx2, dot_rows_q8_0_bit_exact_vs_single)
{
    check_rows_q80(OC_OXK_BLOCK_Q8_0_SIZE, 0, false,
                   oc_oxk_dot_q8_0_q8_0_scalar, oc_oxk_dot_rows_q8_0_q8_0,
                   oc_oxk_dot_rows_q8_0_q8_0_avx2, "q8_0");
}

Test(oxk_avx2, dot_rows_al5_xs_bit_exact_vs_single)
{
    check_rows_q80(OC_OXK_BLOCK_AL5_XS_SIZE, 0, false,
                   oc_oxk_dot_al5_xs_q8_0_scalar, oc_oxk_dot_rows_al5_xs_q8_0,
                   oc_oxk_dot_rows_al5_xs_q8_0_avx2, "al5_xs");
}

Test(oxk_avx2, dot_rows_q5_k_bit_exact_vs_single)
{
    check_rows(OC_OXK_BLOCK_Q5_K_SIZE, 0, true, oc_oxk_dot_q5_k_q8_k_scalar,
               oc_oxk_dot_rows_q5_k_q8_k, oc_oxk_dot_rows_q5_k_q8_k_avx2,
               "q5_k");
}

/* The multi-activation IQ3_S rows kernel (speculative verify, small expert
 * tiles) decodes each row once for up to four activations; every output
 * must equal the scalar single-activation dot bit for bit. */
Test(oxk_avx2, dot_rows_iq3_s_multi_bit_exact_vs_single)
{
    uint32_t s = 0x5EED1234u;
    static const size_t k_blocks[] = { 1, 3, 10 };
    enum { MAX_ROWS = 7, MAX_BLOCKS = 10, MAX_ACT = 5 };
    const size_t bsize = OC_OXK_BLOCK_IQ3_S_SIZE;
    uint8_t *w = malloc(MAX_ROWS * MAX_BLOCKS * bsize);
    uint8_t *a = malloc(MAX_ACT * MAX_BLOCKS * OC_OXK_BLOCK_Q8_K_SIZE);
    cr_assert(w && a);
    for (size_t bi = 0; bi < sizeof k_blocks / sizeof k_blocks[0]; bi++) {
        const size_t blocks = k_blocks[bi];
        const size_t rb = blocks * bsize;
        const size_t as = blocks * OC_OXK_BLOCK_Q8_K_SIZE;
        rand_blocks(w, MAX_ROWS * blocks, bsize, 0, false, &s);
        for (size_t v = 0; v < MAX_ACT; v++) rand_act_q8_k(a + v * as, blocks, &s);
        for (size_t na = 1; na <= MAX_ACT; na++) {
            for (size_t n = 1; n <= MAX_ROWS; n++) {
                float ref[MAX_ACT * MAX_ROWS], got[MAX_ACT * MAX_ROWS];
                for (size_t v = 0; v < na; v++)
                    for (size_t r = 0; r < n; r++)
                        ref[v * MAX_ROWS + r] = oc_oxk_dot_iq3_s_q8_k_scalar(
                            w + r * rb, blocks, a + v * as);
                oc_oxk_dot_rows_iq3_s_q8_k_multi(w, rb, n, blocks, a, as, na,
                                                 got, MAX_ROWS);
                for (size_t v = 0; v < na; v++)
                    cr_assert(memcmp(ref + v * MAX_ROWS, got + v * MAX_ROWS,
                                     n * sizeof(float)) == 0,
                              "iq3_s multi blocks=%zu n_act=%zu rows=%zu act=%zu",
                              blocks, na, n, v);
                if (host_avx2() && na <= 4) {
                    oc_oxk_dot_rows_iq3_s_q8_k_multi_avx2(w, rb, n, blocks, a,
                                                          as, na, got, MAX_ROWS);
                    for (size_t v = 0; v < na; v++)
                        cr_assert(memcmp(ref + v * MAX_ROWS, got + v * MAX_ROWS,
                                         n * sizeof(float)) == 0,
                                  "iq3_s multi avx2 n_act=%zu", na);
                }
            }
        }
    }
    free(w);
    free(a);
}

/* ─── Q2_K / Q3_K: raw blocks, rows, and prepared-row forms ─────────────
 *
 * Q2_K is the dominant weight type in the UD-Q2_K_XL class of models; the
 * whole point of the AVX2 port is that matvec.c routes every one of those
 * rows through oc_oxk_dot_q2_k_q8_k. Random bytes exercise every scale,
 * minimum and hmask bit pattern the packer could emit — the kernel must
 * match scalar to the bit on all of them. */

Test(oxk_avx2, q2_k_raw_blocks_bit_exact_vs_scalar)
{
    if (!host_avx2()) cr_skip("host has no AVX2");
    uint32_t s = 0xC2B2AE35u;
    const size_t nb = 12;
    uint8_t *w = malloc(nb * OC_OXK_BLOCK_Q2_K_SIZE);
    uint8_t *a = malloc(nb * OC_OXK_BLOCK_Q8_K_SIZE);
    cr_assert(w && a);
    for (int t = 0; t < 64; t++) {
        rand_blocks(w, nb, OC_OXK_BLOCK_Q2_K_SIZE, 80, true, &s);
        rand_act_q8_k(a, nb, &s);
        for (size_t blocks = 1; blocks <= nb; blocks += 11) {
            const float sc = oc_oxk_dot_q2_k_q8_k_scalar(w, blocks, a);
            const float av = oc_oxk_dot_q2_k_q8_k_avx2(w, blocks, a);
            cr_assert(memcmp(&sc, &av, sizeof sc) == 0,
                      "trial %d blocks %zu: avx2 %.9g != scalar %.9g", t,
                      blocks, (double)av, (double)sc);
            const float dp = oc_oxk_dot_q2_k_q8_k(w, blocks, a);
            cr_assert(memcmp(&sc, &dp, sizeof sc) == 0,
                      "trial %d blocks %zu: dispatched %.9g != scalar %.9g",
                      t, blocks, (double)dp, (double)sc);
        }
    }
    free(w);
    free(a);
}

Test(oxk_avx2, q3_k_raw_blocks_bit_exact_vs_scalar)
{
    if (!host_avx2()) cr_skip("host has no AVX2");
    uint32_t s = 0x27D4EB2Fu;
    const size_t nb = 12;
    uint8_t *w = malloc(nb * OC_OXK_BLOCK_Q3_K_SIZE);
    uint8_t *a = malloc(nb * OC_OXK_BLOCK_Q8_K_SIZE);
    cr_assert(w && a);
    for (int t = 0; t < 64; t++) {
        rand_blocks(w, nb, OC_OXK_BLOCK_Q3_K_SIZE, 108, false, &s);
        rand_act_q8_k(a, nb, &s);
        for (size_t blocks = 1; blocks <= nb; blocks += 11) {
            const float sc = oc_oxk_dot_q3_k_q8_k_scalar(w, blocks, a);
            const float av = oc_oxk_dot_q3_k_q8_k_avx2(w, blocks, a);
            cr_assert(memcmp(&sc, &av, sizeof sc) == 0,
                      "trial %d blocks %zu: avx2 %.9g != scalar %.9g", t,
                      blocks, (double)av, (double)sc);
            const float dp = oc_oxk_dot_q3_k_q8_k(w, blocks, a);
            cr_assert(memcmp(&sc, &dp, sizeof sc) == 0,
                      "trial %d blocks %zu: dispatched %.9g != scalar %.9g",
                      t, blocks, (double)dp, (double)sc);
        }
    }
    free(w);
    free(a);
}

Test(oxk_avx2, dot_rows_q2_k_bit_exact_vs_single)
{
    check_rows(OC_OXK_BLOCK_Q2_K_SIZE, 80, true, oc_oxk_dot_q2_k_q8_k_scalar,
               oc_oxk_dot_rows_q2_k_q8_k, oc_oxk_dot_rows_q2_k_q8_k_avx2,
               "q2_k");
}

Test(oxk_avx2, dot_rows_q3_k_bit_exact_vs_single)
{
    check_rows(OC_OXK_BLOCK_Q3_K_SIZE, 108, false,
               oc_oxk_dot_q3_k_q8_k_scalar, oc_oxk_dot_rows_q3_k_q8_k,
               oc_oxk_dot_rows_q3_k_q8_k_avx2, "q3_k");
}

/* Prepared-row parity: the batched-prefill path decodes each weight row
 * once into scratch and dots it against an activation tile. Both the
 * single-activation and the 4-at-a-time multi kernels must reproduce the
 * scalar prepped dot bit for bit — the multi is deliberately bit-exact per
 * activation, not merely close, because the same activation can hit either
 * path depending on where it lands in the tile. */
typedef size_t (*OxkPrepBytes)(size_t);
typedef void (*OxkPrepRow)(const uint8_t *, size_t, void *);
typedef float (*OxkDotPrepped)(const void *, size_t, const uint8_t *);
typedef void (*OxkDotPreppedMulti)(const void *, size_t, const uint8_t *,
                                   size_t, size_t, float *);

static void check_prepped(size_t bsize, size_t d_off, bool two_scales,
                          OxkPrepBytes prep_bytes, OxkPrepRow prep_row,
                          OxkDotPrepped scalar, OxkDotPrepped avx2,
                          OxkDotPrepped disp1,
                          OxkDotPreppedMulti multi_disp,
                          OxkDotPreppedMulti multi_avx2,
                          const char *name)
{
    uint32_t s = 0xB5297A4Du;
    static const size_t k_blocks[] = { 1, 3, 8 };
    enum { MAX_BLOCKS = 8, MAX_ACT = 9 };
    uint8_t *w = malloc(MAX_BLOCKS * bsize);
    uint8_t *a = malloc(MAX_ACT * MAX_BLOCKS * OC_OXK_BLOCK_Q8_K_SIZE);
    cr_assert(w && a);
    for (size_t bi = 0; bi < sizeof k_blocks / sizeof k_blocks[0]; bi++) {
        const size_t blocks = k_blocks[bi];
        const size_t as = blocks * OC_OXK_BLOCK_Q8_K_SIZE;
        rand_blocks(w, blocks, bsize, d_off, two_scales, &s);
        void *prep = malloc(prep_bytes(blocks));
        cr_assert(prep);
        prep_row(w, blocks, prep);
        for (size_t v = 0; v < MAX_ACT; v++)
            rand_act_q8_k(a + v * as, blocks, &s);
        const float sc = scalar(prep, blocks, a);
        if (host_avx2()) {
            const float av = avx2(prep, blocks, a);
            cr_assert(memcmp(&sc, &av, sizeof sc) == 0,
                      "%s prepped avx2 blocks=%zu: %.9g != %.9g", name,
                      blocks, (double)av, (double)sc);
        }
        if (disp1) {
            /* Whichever kernel the dispatcher picked (AVX2 here, AVX-512
             * where available) must agree with scalar. */
            const float dp = disp1(prep, blocks, a);
            cr_assert(memcmp(&sc, &dp, sizeof sc) == 0,
                      "%s prepped dispatched blocks=%zu", name, blocks);
        }
        for (size_t na = 1; na <= MAX_ACT; na++) {
            float ref[MAX_ACT], got[MAX_ACT], gav[MAX_ACT];
            for (size_t v = 0; v < na; v++)
                ref[v] = scalar(prep, blocks, a + v * as);
            multi_disp(prep, blocks, a, as, na, got);
            cr_assert(memcmp(ref, got, na * sizeof(float)) == 0,
                      "%s prepped multi dispatch blocks=%zu n_act=%zu", name,
                      blocks, na);
            if (host_avx2()) {
                multi_avx2(prep, blocks, a, as, na, gav);
                cr_assert(memcmp(ref, gav, na * sizeof(float)) == 0,
                          "%s prepped multi avx2 blocks=%zu n_act=%zu", name,
                          blocks, na);
            }
        }
        free(prep);
    }
    free(w);
    free(a);
}

Test(oxk_avx2, q2_k_prepped_bit_exact_vs_scalar)
{
    check_prepped(OC_OXK_BLOCK_Q2_K_SIZE, 80, true,
                  oc_oxk_q2_k_prep_bytes, oc_oxk_q2_k_prep_row,
                  oc_oxk_dot_q2_k_prepped, oc_oxk_dot_q2_k_prepped_avx2,
                  oc_oxk_dot_q2_k_prepped_1,
                  oc_oxk_dot_q2_k_prepped_multi,
                  oc_oxk_dot_q2_k_prepped_multi_avx2, "q2_k");
}

Test(oxk_avx2, q3_k_prepped_bit_exact_vs_scalar)
{
    /* Q3_K shares the Q6_K prepared layout (oc_oxk_q3_k_prep_row writes
     * it), hence oc_oxk_q6_k_prep_bytes. */
    check_prepped(OC_OXK_BLOCK_Q3_K_SIZE, 108, false,
                  oc_oxk_q6_k_prep_bytes, oc_oxk_q3_k_prep_row,
                  oc_oxk_dot_q3_k_prepped, oc_oxk_dot_q3_k_prepped_avx2,
                  oc_oxk_dot_q3_k_prepped_1,
                  oc_oxk_dot_q3_k_prepped_multi,
                  oc_oxk_dot_q3_k_prepped_multi_avx2, "q3_k");
}

Test(oxk_avx2, q6_k_prepped_bit_exact_vs_scalar)
{
    /* The Q6_K multi dispatch moved to the shared Q3/Q6 AVX2 body on AVX2
     * hosts, so it needs the same coverage; there is no _1 dispatcher for
     * Q6_K (disp1 = NULL). */
    check_prepped(OC_OXK_BLOCK_Q6_K_SIZE, 208, false,
                  oc_oxk_q6_k_prep_bytes, oc_oxk_q6_k_prep_row,
                  oc_oxk_dot_q6_k_prepped, oc_oxk_dot_q6_k_prepped_avx2,
                  NULL,
                  oc_oxk_dot_q6_k_prepped_multi,
                  oc_oxk_dot_q6_k_prepped_multi_avx2, "q6_k");
}

Test(oxk_avx2, q4_k_prepped_bit_exact_vs_scalar)
{
    /* Q4_K prepped multi got a real AVX2 body (the batched-prefill path
     * runs it for every Q4_K matrix); no single-act AVX2 kernel exists, so
     * the single comparison forwards to scalar and disp1 is NULL. */
    check_prepped(OC_OXK_BLOCK_Q4_K_SIZE, 144, true,
                  oc_oxk_q4_k_prep_bytes, oc_oxk_q4_k_prep_row,
                  oc_oxk_dot_q4_k_prepped, oc_oxk_dot_q4_k_prepped,
                  NULL,
                  oc_oxk_dot_q4_k_prepped_multi,
                  oc_oxk_dot_q4_k_prepped_multi_avx2, "q4_k");
}
