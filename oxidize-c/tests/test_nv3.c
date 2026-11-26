#define _POSIX_C_SOURCE 200809L
/* test_nv3.c — Criterion tests for src/compute/nv3.c: the NV3 expert
 * format's quantizer, decoder, CPU GEMM and GPU planar layout. */
#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "oxidize/nv3.h"

static void fill(float *x, size_t n, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        float u = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f), v = (float)rand() / (float)RAND_MAX;
        x[i] = 0.02f * sqrtf(-2.0f * logf(u)) * cosf(6.2831853f * v);
    }
}

Test(nv3, roundtrip_error_is_small)
{
    enum { N = 4096 };
    float x[N], y[N];
    uint8_t q[N / 64 * OC_NV3_BLOCK_BYTES];
    fill(x, N, 1);
    const float g = oc_nv3_global_scale(x, N);
    cr_assert_eq(oc_nv3_row_bytes(N), sizeof q);
    oc_nv3_quantize_row(x, NULL, N, g, q);
    oc_nv3_dequant_row(q, N, y);
    double e = 0.0, s = 0.0;
    for (size_t i = 0; i < N; i++) {
        e += (x[i] - g * y[i]) * (x[i] - g * y[i]);
        s += (double)x[i] * x[i];
    }
    cr_assert_lt(e / s, 0.03, "relative error %f", e / s);
}

Test(nv3, levels_and_signs_are_exact)
{
    /* one block whose sub-block maxima sit on the top level: every value a
     * multiple of {1,3,6,10} decodes back exactly */
    static const float L[8] = { 1, 3, 6, 10, -1, -3, -6, -10 };
    float x[64], y[64];
    uint8_t q[OC_NV3_BLOCK_BYTES];
    for (size_t i = 0; i < 64; i++) x[i] = L[(i * 5 + i / 16) % 8] * 0.25f;
    for (size_t s = 0; s < 4; s++) x[s * 16] = 10.0f * 0.25f;
    oc_nv3_quantize_row(x, NULL, 64, 1.0f, q);
    oc_nv3_dequant_row(q, 64, y);
    for (size_t i = 0; i < 64; i++) cr_assert_float_eq(y[i], x[i], 1e-6f, "value %zu", i);
}

Test(nv3, gemm_matches_dequantized_dot)
{
    enum { R = 8, C = 256 };
    float w[R * C], x[C], dq[C];
    uint8_t q[R * C / 64 * OC_NV3_BLOCK_BYTES];
    int8_t aq[C];
    float ad[C / 64];
    fill(w, R * C, 2);
    fill(x, C, 3);
    const float g = oc_nv3_global_scale(w, R * C);
    for (size_t r = 0; r < R; r++) oc_nv3_quantize_row(w + r * C, NULL, C, g, q + r * oc_nv3_row_bytes(C));
    OcNvfp4Act act = { aq, ad };
    oc_nvfp4_quantize_act(x, C, act);
    float y[R];
    oc_nv3_gemm_rows(q, C, 0, R, g, &act, 1, y, R);
    for (size_t r = 0; r < R; r++) {
        oc_nv3_dequant_row(q + r * oc_nv3_row_bytes(C), C, dq);
        double ref = 0.0, mag = 0.0;
        for (size_t c = 0; c < C; c++) {
            ref += (double)g * dq[c] * x[c];
            mag += fabs((double)g * dq[c] * x[c]);
        }
        cr_assert_lt(fabs(y[r] - ref), 0.02 * mag + 1e-9, "row %zu: %f vs %f", r, y[r], ref);
    }
}

Test(nv3, planar_splits_the_planes)
{
    enum { R = 2, C = 128, NB = R * C / 64 };
    uint8_t w[NB * OC_NV3_BLOCK_BYTES], p[sizeof w];
    for (size_t i = 0; i < sizeof w; i++) w[i] = (uint8_t)(i * 7 + 3);
    oc_nv3_planar(w, R, C, p);
    for (size_t b = 0; b < NB; b++) {
        const uint8_t *blk = w + b * OC_NV3_BLOCK_BYTES;
        cr_assert(!memcmp(p + b * 16, blk + 4, 16));
        cr_assert(!memcmp(p + NB * 16 + b * 8, blk + 20, 8));
        cr_assert(!memcmp(p + NB * 24 + b * 4, blk, 4));
    }
}
