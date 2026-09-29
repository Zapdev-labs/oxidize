/*
 * test_oxk_iq3_s.c — IQ3_S x Q8_K fused dot kernels.
 *
 * Two kinds of check:
 *   - numerical: the integer kernel against an independent f64 dot of the
 *     dequantized weights (oc_quant_dequant_row, the verified reference
 *     decoder) and the dequantized activation. Tight relative tolerance —
 *     the only difference is f32 accumulation order.
 *   - bit-exactness: AVX2, dispatched, and prepared multi-activation forms
 *     against the scalar integer reference. These must match to the bit.
 *
 * Plus the matvec dispatch: IQ3_S rows (single, 3D expert slices, batched)
 * go through the fused path and agree with the dequant reference.
 */
#include <criterion/criterion.h>

#include "oxidize/flash_attention.h"
#include "oxidize/matvec.h"
#include "oxidize/oxk.h"
#include "oxidize/parallel.h"
#include "oxidize/quant.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng_next(uint32_t *s)
{
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}

static float frand_sym(uint32_t *s)
{
    return (float)((int32_t)(rng_next(s) % 20001u) - 10000) / 10000.0f;
}

/* Random but well-formed IQ3_S row: any byte pattern is a valid block, the
 * only constraint is a finite, sensible f16 scale. */
static void make_iq3_s_row(uint8_t *row, size_t blocks, uint32_t *s)
{
    for (size_t b = 0; b < blocks; b++) {
        uint8_t *xb = row + b * OC_OXK_BLOCK_IQ3_S_SIZE;
        for (size_t i = 2; i < OC_OXK_BLOCK_IQ3_S_SIZE; i++)
            xb[i] = (uint8_t)rng_next(s);
        const float d = 0.001f + 0.02f * (frand_sym(s) + 1.0f);
        const uint16_t h = oc_f32_to_f16_bits(d);
        xb[0] = (uint8_t)(h & 0xFF);
        xb[1] = (uint8_t)(h >> 8);
    }
}

/* Q8_K activation with values in [-127, 127], matching the quantizer. */
static void make_act_q8_k(uint8_t *act, size_t blocks, uint32_t *s)
{
    for (size_t b = 0; b < blocks; b++) {
        uint8_t *yb = act + b * OC_OXK_BLOCK_Q8_K_SIZE;
        const float d = 0.005f + 0.01f * (frand_sym(s) + 1.0f);
        memcpy(yb, &d, 4);
        int8_t *q = (int8_t *)(yb + 4);
        for (int i = 0; i < 256; i++)
            q[i] = (int8_t)((int32_t)(rng_next(s) % 255u) - 127);
        for (int g = 0; g < 16; g++) {
            int32_t t = 0;
            for (int i = 0; i < 16; i++) t += q[g * 16 + i];
            const int16_t t16 = (int16_t)t;
            memcpy(yb + 4 + 256 + 2 * g, &t16, 2);
        }
    }
}

/* Independent reference: dequantize both sides, dot in f64. Also returns
 * sum |w_i x_i| so the tolerance scales with the magnitude of the terms. */
static double ref_dot(const uint8_t *row, size_t blocks, const uint8_t *act,
                      double *mag)
{
    const size_t cols = blocks * 256u;
    float *w = malloc(cols * sizeof(float));
    cr_assert_not_null(w);
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_IQ3_S, row,
                                      blocks * OC_OXK_BLOCK_IQ3_S_SIZE,
                                      w, cols), OC_OK);
    double acc = 0.0, m = 0.0;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *yb = act + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float yd;
        memcpy(&yd, yb, 4);
        const int8_t *q = (const int8_t *)(yb + 4);
        for (int i = 0; i < 256; i++) {
            const double t = (double)w[b * 256 + i] * (double)yd * (double)q[i];
            acc += t;
            m += fabs(t);
        }
    }
    free(w);
    *mag = m;
    return acc;
}

static const size_t k_cols[] = { 256u, 768u, 2560u, 4096u, 6144u };
#define N_COLS (sizeof(k_cols) / sizeof(k_cols[0]))

static bool have_avx2(void)
{
    const OcOxkCaps *c = oc_oxk_caps();
    return c->level == OC_OXK_AVX2 || c->level == OC_OXK_AVX512;
}

Test(oxk_iq3_s, scalar_matches_dequant_reference)
{
    uint32_t s = 0x1234567u;
    for (size_t ci = 0; ci < N_COLS; ci++) {
        const size_t blocks = k_cols[ci] / 256u;
        uint8_t *row = malloc(blocks * OC_OXK_BLOCK_IQ3_S_SIZE);
        uint8_t *act = malloc(blocks * OC_OXK_BLOCK_Q8_K_SIZE);
        cr_assert(row && act);
        for (int trial = 0; trial < 8; trial++) {
            make_iq3_s_row(row, blocks, &s);
            make_act_q8_k(act, blocks, &s);
            double mag;
            const double ref = ref_dot(row, blocks, act, &mag);
            const float got = oc_oxk_dot_iq3_s_q8_k_scalar(row, blocks, act);
            cr_assert(fabs((double)got - ref) <= 2e-6 * mag + 1e-12,
                      "cols=%zu trial=%d got=%.9g ref=%.9g mag=%.6g",
                      k_cols[ci], trial, (double)got, ref, mag);
        }
        free(row);
        free(act);
    }
}

Test(oxk_iq3_s, avx2_and_dispatch_bit_exact_vs_scalar)
{
    uint32_t s = 0xBEEF01u;
    for (size_t ci = 0; ci < N_COLS; ci++) {
        const size_t blocks = k_cols[ci] / 256u;
        uint8_t *row = malloc(blocks * OC_OXK_BLOCK_IQ3_S_SIZE);
        uint8_t *act = malloc(blocks * OC_OXK_BLOCK_Q8_K_SIZE);
        cr_assert(row && act);
        for (int trial = 0; trial < 16; trial++) {
            make_iq3_s_row(row, blocks, &s);
            make_act_q8_k(act, blocks, &s);
            const float ref = oc_oxk_dot_iq3_s_q8_k_scalar(row, blocks, act);
            const float dis = oc_oxk_dot_iq3_s_q8_k(row, blocks, act);
            cr_assert(memcmp(&ref, &dis, sizeof ref) == 0,
                      "dispatch cols=%zu %.9g vs %.9g", k_cols[ci],
                      (double)dis, (double)ref);
            if (have_avx2()) {
                const float v = oc_oxk_dot_iq3_s_q8_k_avx2(row, blocks, act);
                cr_assert(memcmp(&ref, &v, sizeof ref) == 0,
                          "avx2 cols=%zu %.9g vs %.9g", k_cols[ci],
                          (double)v, (double)ref);
            }
        }
        free(row);
        free(act);
    }
}

Test(oxk_iq3_s, prep_rows_identical_scalar_vs_avx2)
{
    if (!have_avx2()) return;
    uint32_t s = 0x51u;
    const size_t blocks = 2560u / 256u;
    uint8_t *row = malloc(blocks * OC_OXK_BLOCK_IQ3_S_SIZE);
    const size_t pb = oc_oxk_iq3_s_prep_bytes(blocks);
    uint8_t *p1 = malloc(pb), *p2 = malloc(pb);
    cr_assert(row && p1 && p2);
    make_iq3_s_row(row, blocks, &s);
    oc_oxk_iq3_s_prep_row_scalar(row, blocks, p1);
    oc_oxk_iq3_s_prep_row_avx2(row, blocks, p2);
    cr_assert(memcmp(p1, p2, pb) == 0);
    free(row);
    free(p1);
    free(p2);
}

Test(oxk_iq3_s, prepped_multi_bit_exact_vs_scalar)
{
    uint32_t s = 0xC0FFEEu;
    for (size_t ci = 0; ci < N_COLS; ci++) {
        const size_t blocks = k_cols[ci] / 256u;
        const size_t n_act = 11;
        const size_t stride = blocks * OC_OXK_BLOCK_Q8_K_SIZE;
        uint8_t *row  = malloc(blocks * OC_OXK_BLOCK_IQ3_S_SIZE);
        uint8_t *acts = malloc(n_act * stride);
        uint8_t *prep = malloc(oc_oxk_iq3_s_prep_bytes(blocks));
        cr_assert(row && acts && prep);
        make_iq3_s_row(row, blocks, &s);
        for (size_t a = 0; a < n_act; a++)
            make_act_q8_k(acts + a * stride, blocks, &s);
        oc_oxk_iq3_s_prep_row(row, blocks, prep);
        for (size_t n = 1; n <= n_act; n++) {
            float out_d[11], out_s[11], out_v[11];
            oc_oxk_dot_iq3_s_prepped_multi(prep, blocks, acts, stride, n, out_d);
            oc_oxk_dot_iq3_s_prepped_multi_scalar(prep, blocks, acts, stride,
                                                  n, out_s);
            if (have_avx2())
                oc_oxk_dot_iq3_s_prepped_multi_avx2(prep, blocks, acts,
                                                    stride, n, out_v);
            for (size_t a = 0; a < n; a++) {
                const float ref = oc_oxk_dot_iq3_s_q8_k_scalar(
                    row, blocks, acts + a * stride);
                cr_assert(memcmp(&ref, &out_d[a], 4) == 0,
                          "dispatch multi cols=%zu n=%zu a=%zu", k_cols[ci], n, a);
                cr_assert(memcmp(&ref, &out_s[a], 4) == 0,
                          "scalar multi cols=%zu n=%zu a=%zu", k_cols[ci], n, a);
                if (have_avx2())
                    cr_assert(memcmp(&ref, &out_v[a], 4) == 0,
                              "avx2 multi cols=%zu n=%zu a=%zu", k_cols[ci], n, a);
            }
        }
        free(row);
        free(acts);
        free(prep);
    }
}

/* Row-wise dequant reference for a whole matvec. */
static void ref_matvec(const uint8_t *w, size_t rows, size_t cols,
                       const float *x, double *out, double *mag)
{
    const size_t rb = cols / 256u * OC_OXK_BLOCK_IQ3_S_SIZE;
    float *tmp = malloc(cols * sizeof(float));
    cr_assert_not_null(tmp);
    for (size_t r = 0; r < rows; r++) {
        cr_assert_eq(oc_quant_dequant_row(OC_QUANT_IQ3_S, w + r * rb, rb, tmp,
                                          cols), OC_OK);
        double a = 0.0, m = 0.0;
        for (size_t i = 0; i < cols; i++) {
            a += (double)tmp[i] * x[i];
            m += fabs((double)tmp[i] * x[i]);
        }
        out[r] = a;
        mag[r] = m;
    }
    free(tmp);
}

/* The fused path quantizes the activation to int8, so compare with a
 * tolerance sized for Q8_K activation error rather than for accumulation. */
Test(oxk_iq3_s, matvec_fused_matches_dequant_incl_expert_slices)
{
    oc_matvec_set_fused(true);
    oc_parallel_set_threads(4);
    uint32_t s = 0xA5A5u;
    const size_t cols = 2560u, rows = 96u, n_exp = 3u;
    const size_t rb = cols / 256u * OC_OXK_BLOCK_IQ3_S_SIZE;
    uint8_t *w = malloc(n_exp * rows * rb);
    float *x = malloc(cols * sizeof(float));
    float *temp = malloc(cols * sizeof(float));
    float *y = malloc(rows * sizeof(float));
    double *ref = malloc(rows * sizeof(double));
    double *mag = malloc(rows * sizeof(double));
    cr_assert(w && x && temp && y && ref && mag);
    for (size_t r = 0; r < n_exp * rows; r++)
        make_iq3_s_row(w + r * rb, cols / 256u, &s);
    for (size_t i = 0; i < cols; i++) x[i] = frand_sym(&s);

    for (size_t e = 0; e < n_exp; e++) {
        const uint8_t *slice = w + e * rows * rb;   /* 3D expert slice */
        oc_matvec_quantized(OC_QUANT_IQ3_S, slice, rows, cols, rb, x, y, temp);
        ref_matvec(slice, rows, cols, x, ref, mag);
        for (size_t r = 0; r < rows; r++) {
            cr_assert(fabs((double)y[r] - ref[r]) <= 1e-2 * mag[r] + 1e-6,
                      "expert %zu row %zu: %.6g vs %.6g (mag %.6g)", e, r,
                      (double)y[r], ref[r], mag[r]);
        }
    }
    oc_parallel_shutdown();
    free(w); free(x); free(temp); free(y); free(ref); free(mag);
}

/* Batched (prefill) path must equal per-vector matvec bit-for-bit. */
Test(oxk_iq3_s, matvec_batch_bit_exact_vs_single)
{
    oc_matvec_set_fused(true);
    oc_parallel_set_threads(3);
    uint32_t s = 0x777u;
    const size_t cols = 768u, rows = 40u, n_vec = 13u;
    const size_t rb = cols / 256u * OC_OXK_BLOCK_IQ3_S_SIZE;
    uint8_t *w = malloc(rows * rb);
    float *x = malloc(n_vec * cols * sizeof(float));
    float *temp = malloc(cols * sizeof(float));
    float *yb = malloc(n_vec * rows * sizeof(float));
    float *y1 = malloc(rows * sizeof(float));
    const size_t ab = oc_matvec_batch_scratch_bytes(cols);
    uint8_t *act = malloc(ab);
    cr_assert(w && x && temp && yb && y1 && act);
    for (size_t r = 0; r < rows; r++) make_iq3_s_row(w + r * rb, cols / 256u, &s);
    for (size_t i = 0; i < n_vec * cols; i++) x[i] = frand_sym(&s);

    oc_matvec_quantized_batch(OC_QUANT_IQ3_S, w, rows, cols, rb, x, cols,
                              yb, rows, n_vec, temp, act, ab);
    for (size_t v = 0; v < n_vec; v++) {
        oc_matvec_quantized(OC_QUANT_IQ3_S, w, rows, cols, rb, x + v * cols,
                            y1, temp);
        cr_assert(memcmp(y1, yb + v * rows, rows * sizeof(float)) == 0,
                  "vec %zu differs", v);
    }
    oc_parallel_shutdown();
    free(w); free(x); free(temp); free(yb); free(y1); free(act);
}
