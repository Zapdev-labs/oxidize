/* test_oxk_gguf_layout.c — do the OXK kernels agree with the GGUF layout? The existing OXK tests compare each SIMD variant against the OXK scalar reference. */
#include <criterion/criterion.h>

#include "oxidize/flash_attention.h"
#include "oxidize/oxk.h"
#include "oxidize/quant.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define COLS 1024u

static float frand(uint32_t *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (float)((int32_t)(*s >> 8) % 2000 - 1000) / 1000.0f;
}


static void act_q8_k(const float *x, size_t n, uint8_t *out)
{
    for (size_t b = 0; b < n / 256; b++) {
        const float *s = x + b * 256;
        float am = 0.0f;
        for (int i = 0; i < 256; i++) { float a = fabsf(s[i]); if (a > am) am = a; }
        const float d = am / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t *dst = out + b * OC_OXK_BLOCK_Q8_K_SIZE;
        memcpy(dst, &d, 4);
        int8_t *q = (int8_t *)(dst + 4);
        for (int i = 0; i < 256; i++) {
            int v = (int)lrintf(s[i] * id);
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

static void act_q8_k_deq(const uint8_t *o, size_t n, float *x)
{
    for (size_t b = 0; b < n / 256; b++) {
        float d;
        memcpy(&d, o + b * OC_OXK_BLOCK_Q8_K_SIZE, 4);
        const int8_t *q = (const int8_t *)(o + b * OC_OXK_BLOCK_Q8_K_SIZE + 4);
        for (int i = 0; i < 256; i++) x[b * 256 + i] = d * (float)q[i];
    }
}

static void act_q8_0(const float *x, size_t n, uint8_t *out)
{
    for (size_t b = 0; b < n / 32; b++) {
        const float *s = x + b * 32;
        float am = 0.0f;
        for (int i = 0; i < 32; i++) { float a = fabsf(s[i]); if (a > am) am = a; }
        const float d = am / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t *dst = out + b * OC_OXK_BLOCK_Q8_0_SIZE;
        uint16_t dh = oc_f32_to_f16_bits(d);
        dst[0] = (uint8_t)(dh & 0xFF);
        dst[1] = (uint8_t)(dh >> 8);
        int8_t *q = (int8_t *)(dst + 2);
        for (int i = 0; i < 32; i++) {
            int v = (int)lrintf(s[i] * id);
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
        for (int i = 0; i < 32; i++) x[b * 32 + i] = d * (float)q[i];
    }
}

/* Relative difference between the kernel and the dequant reference, with
 * activation quantization error removed. Returns -1.0 if the type cannot be
 * packed on this build. */
static double kernel_vs_gguf(OcGgufQuantizationType qt, int kblock,
                             float (*dot)(const uint8_t *, size_t,
                                          const uint8_t *))
{
    const size_t rb = oc_quantized_size(qt, COLS);
    if (rb == 0) return -1.0;

    uint8_t *w = malloc(rb);
    float *wf = malloc(COLS * sizeof(float));
    float *wd = malloc(COLS * sizeof(float));
    float *x  = malloc(COLS * sizeof(float));
    float *xq = malloc(COLS * sizeof(float));
    cr_assert_not_null(w); cr_assert_not_null(wf); cr_assert_not_null(wd);
    cr_assert_not_null(x); cr_assert_not_null(xq);

    uint32_t s = 77;
    for (size_t i = 0; i < COLS; i++) wf[i] = frand(&s);
    if (oc_quant_pack_row(qt, wf, COLS, w, rb) != OC_OK) {
        free(w); free(wf); free(wd); free(x); free(xq);
        return -1.0;
    }
    for (size_t i = 0; i < COLS; i++) x[i] = frand(&s);

    const size_t nb = kblock ? COLS / 256 : COLS / 32;
    uint8_t *act = malloc(nb * (kblock ? OC_OXK_BLOCK_Q8_K_SIZE
                                       : OC_OXK_BLOCK_Q8_0_SIZE));
    cr_assert_not_null(act);
    if (kblock) { act_q8_k(x, COLS, act); act_q8_k_deq(act, COLS, xq); }
    else        { act_q8_0(x, COLS, act); act_q8_0_deq(act, COLS, xq); }

    cr_assert_eq(oc_quant_dequant_row(qt, w, rb, wd, COLS), OC_OK);
    double ref = 0.0;
    for (size_t i = 0; i < COLS; i++) ref += (double)wd[i] * xq[i];
    const double got = (double)dot(w, nb, act);

    free(w); free(wf); free(wd); free(x); free(xq); free(act);
    return fabs(ref - got) / (fabs(ref) + 1e-6);
}


Test(oxk_gguf, q8_0_matches_dequant_reference)
{
    double rel = kernel_vs_gguf(OC_QUANT_Q8_0, 0, oc_oxk_dot_q8_0_q8_0);
    cr_assert_geq(rel, 0.0, "Q8_0 packing unavailable");
    cr_assert_lt(rel, 1e-4, "Q8_0 kernel disagrees with GGUF layout (rel=%g)", rel);
}

Test(oxk_gguf, q6_k_matches_dequant_reference)
{
    double rel = kernel_vs_gguf(OC_QUANT_Q6_K, 1, oc_oxk_dot_q6_k_q8_k);
    cr_assert_geq(rel, 0.0, "Q6_K packing unavailable");
    cr_assert_lt(rel, 1e-4, "Q6_K kernel disagrees with GGUF layout (rel=%g)", rel);
}


/* Q4_0 and Q4_1 had the wrong nibble-to-element mapping; Q4_K additionally
 * scaled its offset term by dw*dmin instead of dmin, and decoded the upper
 * four scale/min pairs from the wrong bits. All three now agree exactly. */

Test(oxk_gguf, q4_0_matches_dequant_reference)
{
    double rel = kernel_vs_gguf(OC_QUANT_Q4_0, 0, oc_oxk_dot_q4_0_q8_0);
    cr_assert_geq(rel, 0.0, "Q4_0 packing unavailable");
    cr_assert_lt(rel, 1e-4, "Q4_0 kernel disagrees with GGUF layout (rel=%g)", rel);
}

Test(oxk_gguf, q4_1_matches_dequant_reference)
{
    double rel = kernel_vs_gguf(OC_QUANT_Q4_1, 0, oc_oxk_dot_q4_1_q8_0);
    cr_assert_geq(rel, 0.0, "Q4_1 packing unavailable");
    cr_assert_lt(rel, 1e-4, "Q4_1 kernel disagrees with GGUF layout (rel=%g)", rel);
}

Test(oxk_gguf, q4_k_matches_dequant_reference)
{
    double rel = kernel_vs_gguf(OC_QUANT_Q4_K_M, 1, oc_oxk_dot_q4_k_q8_k);
    cr_assert_geq(rel, 0.0, "Q4_K packing unavailable");
    cr_assert_lt(rel, 1e-4, "Q4_K kernel disagrees with GGUF layout (rel=%g)", rel);
}
