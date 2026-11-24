/* test_nvfp4.c — Criterion tests for src/compute/nvfp4.c. */
#include <criterion/criterion.h>

#include "oxidize/nvfp4.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint32_t seed = 7u;
static float rnd(void)
{
    seed = seed * 1664525u + 1013904223u;
    return (float)((int32_t)(seed >> 8) - (1 << 23)) / (float)(1 << 23);
}

Test(nvfp4, ue4m3_decode)
{
    cr_assert_eq(oc_nvfp4_ue4m3(0x00), 0.0f);
    cr_assert_eq(oc_nvfp4_ue4m3(0x7F), 0.0f);
    cr_assert_eq(oc_nvfp4_ue4m3(0x38), 1.0f);
    cr_assert_eq(oc_nvfp4_ue4m3(0x7E), 448.0f);
    cr_assert_eq(oc_nvfp4_ue4m3(0x01), ldexpf(1.0f, -9));
    cr_assert_eq(oc_nvfp4_ue4m3(0xB8), 1.0f, "sign bit ignored");
}

Test(nvfp4, ue4m3_encode_roundtrip)
{
    for (int c = 1; c < 0x7F; c++)
        cr_assert_eq(oc_nvfp4_ue4m3_encode(oc_nvfp4_ue4m3((uint8_t)c)), (uint8_t)c, "code %d", c);
    cr_assert_eq(oc_nvfp4_ue4m3_encode(0.0f), 0);
    cr_assert_eq(oc_nvfp4_ue4m3_encode(1e9f), 0x7E);
}

Test(nvfp4, quantize_roundtrip_error)
{
    enum { N = 64 * 32 };
    float x[N], y[N];
    for (int i = 0; i < N; i++) x[i] = rnd() * (i % 7 == 0 ? 4.0f : 1.0f) * 0.02f;
    float g = oc_nvfp4_global_scale(x, N);
    uint8_t q[N / 64 * 36];
    oc_nvfp4_quantize_row(x, N, g, q);
    oc_nvfp4_dequant_row(q, N, y);
    double e = 0, r = 0;
    for (int i = 0; i < N; i++) { double d = y[i] * g - x[i]; e += d * d; r += (double)x[i] * x[i]; }
    cr_assert_lt(sqrt(e / r), 0.12, "relative rms %f", sqrt(e / r));
}

/* The dispatched (AVX2) kernel must match the scalar integer reference for
 * every tile size, and both must approximate the f32 product. */
Test(nvfp4, gemm_matches_scalar)
{
    enum { ROWS = 37, COLS = 64 * 9, NV = 7 };
    const size_t rb = oc_nvfp4_row_bytes(COLS);
    uint8_t *w = malloc(ROWS * rb);
    float *wf = malloc(ROWS * COLS * sizeof(float));
    for (size_t r = 0; r < ROWS; r++) {
        for (size_t c = 0; c < COLS; c++) wf[r * COLS + c] = rnd();
        oc_nvfp4_quantize_row(wf + r * COLS, COLS, 1.0f / 448.0f, w + r * rb);
        oc_nvfp4_dequant_row(w + r * rb, COLS, wf + r * COLS);
    }
    float x[NV][COLS];
    int8_t q[NV][COLS];
    float d[NV][COLS / 64];
    OcNvfp4Act acts[NV];
    for (int v = 0; v < NV; v++) {
        for (int c = 0; c < COLS; c++) x[v][c] = rnd();
        acts[v].q = q[v];
        acts[v].d = d[v];
        oc_nvfp4_quantize_act(x[v], COLS, acts[v]);
    }
    const float gs = 1.0f / 448.0f;
    float y[NV][ROWS], ys[ROWS];
    oc_nvfp4_gemm_rows(w, COLS, 0, ROWS, gs, acts, NV, &y[0][0], ROWS);
    for (int v = 0; v < NV; v++) {
        oc_nvfp4_gemv_rows_scalar(w, COLS, 0, ROWS, gs, acts[v], ys);
        for (int r = 0; r < ROWS; r++) {
            cr_assert_float_eq(y[v][r], ys[r], 1e-4f * (1.0f + fabsf(ys[r])), "v %d r %d", v, r);
            double ref = 0, mag = 0;
            for (int c = 0; c < COLS; c++) {
                const double p = (double)wf[r * COLS + c] * gs * x[v][c];
                ref += p;
                mag += fabs(p);
            }
            /* int8 activations: error scales with the sum of |w x|, not the result */
            cr_assert_float_eq(y[v][r], (float)ref, (float)(0.01 * mag), "v %d r %d", v, r);
        }
    }
    free(w);
    free(wf);
}

