#include <criterion/criterion.h>

#include "oxidize/flash_attention.h"
#include "oxidize/matvec.h"
#include "oxidize/oxk.h"
#include "oxidize/quant.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define COLS 256u
#define ROWS 8u

static float frand(uint32_t *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (float)((int32_t)(*s >> 8) % 2000 - 1000) / 1000.0f;
}

static void act_q8_0(const float *x, size_t n, uint8_t *out)
{
    for (size_t b = 0; b < n / 32; b++) {
        const float *src = x + b * 32;
        float am = 0.0f;
        for (int i = 0; i < 32; i++) {
            float a = fabsf(src[i]);
            if (a > am) am = a;
        }
        const float d = am / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t *dst = out + b * OC_OXK_BLOCK_Q8_0_SIZE;
        uint16_t dh = oc_f32_to_f16_bits(d);
        dst[0] = (uint8_t)(dh & 0xFF);
        dst[1] = (uint8_t)(dh >> 8);
        int8_t *q = (int8_t *)(dst + 2);
        for (int i = 0; i < 32; i++) {
            int v = (int)lrintf(src[i] * id);
            q[i] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
        }
    }
}

static void act_q8_0_deq(const uint8_t *o, size_t n, float *x)
{
    for (size_t b = 0; b < n / 32; b++) {
        const uint8_t *p = o + b * OC_OXK_BLOCK_Q8_0_SIZE;
        float d = oc_oxk_f16_le_to_f32(p);
        const int8_t *q = (const int8_t *)(p + 2);
        for (int i = 0; i < 32; i++)
            x[b * 32 + i] = d * (float)q[i];
    }
}

static double kernel_vs_dequant(OcGgufQuantizationType qt,
                                float (*dot)(const uint8_t *, size_t,
                                             const uint8_t *))
{
    const size_t rb = oc_quantized_size(qt, COLS);
    uint8_t *w = malloc(rb);
    float *wf = malloc(COLS * sizeof(float));
    float *wd = malloc(COLS * sizeof(float));
    float *x = malloc(COLS * sizeof(float));
    float *xq = malloc(COLS * sizeof(float));
    uint8_t *act = malloc((COLS / 32) * OC_OXK_BLOCK_Q8_0_SIZE);
    cr_assert(w && wf && wd && x && xq && act);

    uint32_t s = 91;
    for (size_t i = 0; i < COLS; i++) wf[i] = frand(&s);
    cr_assert_eq(oc_quant_pack_row(qt, wf, COLS, w, rb), OC_OK);
    for (size_t i = 0; i < COLS; i++) x[i] = frand(&s);
    act_q8_0(x, COLS, act);
    act_q8_0_deq(act, COLS, xq);
    cr_assert_eq(oc_quant_dequant_row(qt, w, rb, wd, COLS), OC_OK);

    double ref = 0.0;
    for (size_t i = 0; i < COLS; i++)
        ref += (double)wd[i] * xq[i];
    const double got = (double)dot(w, COLS / 32, act);

    free(w); free(wf); free(wd); free(x); free(xq); free(act);
    return fabs(ref - got) / (fabs(ref) + 1e-6);
}

static void matvec_matches_kernel(OcGgufQuantizationType qt,
                                  float (*dot)(const uint8_t *, size_t,
                                               const uint8_t *))
{
    const size_t rb = oc_quantized_size(qt, COLS);
    uint8_t *w = malloc(ROWS * rb);
    float *wf = malloc(COLS * sizeof(float));
    float *x = malloc(COLS * sizeof(float));
    float *temp = malloc(COLS * sizeof(float));
    float *out = calloc(ROWS, sizeof(float));
    uint8_t *act = malloc((COLS / 32) * OC_OXK_BLOCK_Q8_0_SIZE);
    cr_assert(w && wf && x && temp && out && act);

    uint32_t s = 0xA11u;
    for (size_t r = 0; r < ROWS; r++) {
        for (size_t i = 0; i < COLS; i++) wf[i] = frand(&s);
        cr_assert_eq(oc_quant_pack_row(qt, wf, COLS, w + r * rb, rb), OC_OK);
    }
    for (size_t i = 0; i < COLS; i++) x[i] = frand(&s);
    act_q8_0(x, COLS, act);

    oc_matvec_quantized(qt, w, ROWS, COLS, rb, x, out, temp);
    for (size_t r = 0; r < ROWS; r++) {
        float want = dot(w + r * rb, COLS / 32, act);
        cr_assert_eq(memcmp(&out[r], &want, sizeof(float)), 0,
                     "row %zu fused %g != kernel %g", r, out[r], want);
    }

    free(w); free(wf); free(x); free(temp); free(out); free(act);
}

Test(al_fused, al5_kernel_matches_dequant)
{
    cr_assert_lt(kernel_vs_dequant(OC_QUANT_AL5, oc_oxk_dot_q4_0_q8_0), 1e-4);
}

Test(al_fused, al8_kernel_matches_dequant)
{
    cr_assert_lt(kernel_vs_dequant(OC_QUANT_AL8, oc_oxk_dot_q8_0_q8_0), 1e-4);
}

Test(al_fused, al6_kernel_matches_dequant)
{
    cr_assert_lt(kernel_vs_dequant(OC_QUANT_AL6, oc_oxk_dot_q5_0_q8_0), 1e-4);
}

Test(al_fused, q5_0_kernel_matches_dequant)
{
    cr_assert_lt(kernel_vs_dequant(OC_QUANT_Q5_0, oc_oxk_dot_q5_0_q8_0), 1e-4);
}

Test(al_fused, q5_1_kernel_matches_dequant)
{
    cr_assert_lt(kernel_vs_dequant(OC_QUANT_Q5_1, oc_oxk_dot_q5_1_q8_0), 1e-4);
}

Test(al_fused, al5_xs_kernel_matches_dequant)
{
    cr_assert_lt(kernel_vs_dequant(OC_QUANT_AL5_XS, oc_oxk_dot_al5_xs_q8_0),
                 1e-4);
}

Test(al_fused, al5_matvec_uses_q4_0_kernel)
{
    matvec_matches_kernel(OC_QUANT_AL5, oc_oxk_dot_q4_0_q8_0);
}

Test(al_fused, al8_matvec_uses_q8_0_kernel)
{
    matvec_matches_kernel(OC_QUANT_AL8, oc_oxk_dot_q8_0_q8_0);
}

Test(al_fused, al6_matvec_uses_q5_0_kernel)
{
    matvec_matches_kernel(OC_QUANT_AL6, oc_oxk_dot_q5_0_q8_0);
}

Test(al_fused, al5_xs_matvec_uses_al5_xs_kernel)
{
    matvec_matches_kernel(OC_QUANT_AL5_XS, oc_oxk_dot_al5_xs_q8_0);
}
