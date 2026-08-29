/* test_oxk_avx2_parity.c — vectorized OXK kernels vs the scalar reference. */
/* The OXK invariant is bit-exactness, not approximate agreement: a kernel that merely rounds differently would silently change model output depending on CPU. These compare raw float bits over packed blocks. */
#include <criterion/criterion.h>

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
