/*
 * nv3.c — NV3 quantization, dequantization and CPU GEMM. See nv3.h.
 */
#include "oxidize/nv3.h"

#include <math.h>
#include <string.h>

static const float NV3_MAG[4] = { 1.0f, 3.0f, 6.0f, 10.0f };

size_t oc_nv3_row_bytes(size_t cols) { return cols / OC_NV3_BLOCK * OC_NV3_BLOCK_BYTES; }

float oc_nv3_global_scale(const float *x, size_t n)
{
    float amax = 0.0f;
    for (size_t i = 0; i < n; i++) amax = fmaxf(amax, fabsf(x[i]));
    return amax > 0.0f ? amax / (5.0f * 448.0f) : 1.0f;
}

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }

static void wr32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

/* Value v = 16 s + 4 c + j of a block as a signed multiple of {1,3,6,10}. */
static int nv3_value(const uint8_t *blk, size_t v)
{
    const size_t s = v / 16, c = (v / 4) % 4, j = v % 4;
    const uint32_t m = (rd32(blk + 4 + 4 * s) >> (8 * j + 2 * c)) & 3u;
    const uint32_t neg = (rd32(blk + 20 + 4 * (s / 2)) >> (8 * j + 4 * (s & 1) + c)) & 1u;
    const int q = (int)NV3_MAG[m];
    return neg ? -q : q;
}

static uint8_t nearest_mag(float a)
{
    return a < 2.0f ? 0 : a < 4.5f ? 1 : a < 8.0f ? 2 : 3;
}

void oc_nv3_quantize_row(const float *x, const float *imp, size_t n, float gscale, uint8_t *out)
{
    for (size_t b = 0; b < n / OC_NV3_BLOCK; b++) {
        const float *xb = x + b * OC_NV3_BLOCK;
        uint8_t *blk = out + b * OC_NV3_BLOCK_BYTES;
        uint8_t mag[64], neg[64];
        for (size_t s = 0; s < 4; s++) {
            const float *xs = xb + s * 16;
            const float *ws = imp ? imp + b * OC_NV3_BLOCK + s * 16 : NULL;
            float amax = 0.0f;
            for (size_t j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(xs[j]));
            uint8_t best_code = 0;
            float best_err = INFINITY;
            if (amax > 0.0f) {
                const int c0 = oc_nvfp4_ue4m3_encode(amax / (5.0f * gscale));
                for (int c = c0 - 4; c <= c0 + 2; c++) {
                    if (c < 1 || c > 0x7E) continue;
                    const float d = oc_nvfp4_ue4m3((uint8_t)c) * 0.5f * gscale;
                    const float id = 1.0f / d;
                    float err = 0.0f;
                    for (size_t j = 0; j < 16; j++) {
                        const float r = fabsf(xs[j]) - d * NV3_MAG[nearest_mag(fabsf(xs[j]) * id)];
                        err += (ws ? ws[j] : 1.0f) * r * r;
                    }
                    if (err < best_err) { best_err = err; best_code = (uint8_t)c; }
                }
            }
            blk[s] = best_code;
            const float d = oc_nvfp4_ue4m3(best_code) * 0.5f * gscale;
            const float id = d > 0.0f ? 1.0f / d : 0.0f;
            for (size_t j = 0; j < 16; j++) {
                mag[s * 16 + j] = nearest_mag(fabsf(xs[j]) * id);
                neg[s * 16 + j] = xs[j] < 0.0f;
            }
        }
        uint32_t mw[4] = { 0 }, sw[2] = { 0 };
        for (size_t v = 0; v < 64; v++) {
            const size_t s = v / 16, c = (v / 4) % 4, j = v % 4;
            mw[s] |= (uint32_t)mag[v] << (8 * j + 2 * c);
            sw[s / 2] |= (uint32_t)neg[v] << (8 * j + 4 * (s & 1) + c);
        }
        for (size_t s = 0; s < 4; s++) wr32(blk + 4 + 4 * s, mw[s]);
        wr32(blk + 20, sw[0]);
        wr32(blk + 24, sw[1]);
    }
}

void oc_nv3_dequant_row(const uint8_t *row, size_t cols, float *out)
{
    for (size_t b = 0; b < cols / OC_NV3_BLOCK; b++) {
        const uint8_t *blk = row + b * OC_NV3_BLOCK_BYTES;
        for (size_t v = 0; v < 64; v++)
            out[b * OC_NV3_BLOCK + v] = (float)nv3_value(blk, v) * 0.5f * oc_nvfp4_ue4m3(blk[v / 16]);
    }
}

void oc_nv3_gemm_rows(const uint8_t *w, size_t cols, size_t r0, size_t r1, float gscale,
                      const OcNvfp4Act *acts, size_t n_vec, float *y, size_t y_stride)
{
    const size_t nb = cols / OC_NV3_BLOCK, rb = oc_nv3_row_bytes(cols);
    for (size_t r = r0; r < r1; r++) {
        const uint8_t *row = w + r * rb;
        for (size_t t = 0; t < n_vec; t++) {
            float acc = 0.0f;
            for (size_t b = 0; b < nb; b++) {
                const uint8_t *blk = row + b * OC_NV3_BLOCK_BYTES;
                const int8_t *q = acts[t].q + b * OC_NV3_BLOCK;
                float sb = 0.0f;
                for (size_t s = 0; s < 4; s++) {
                    int32_t si = 0;
                    /* the activation is in the NVFP4 order: x[16s + i] at
                     * s*8 + i for i < 8, 32 + s*8 + i - 8 otherwise */
                    for (size_t i = 0; i < 16; i++)
                        si += nv3_value(blk, 16 * s + i) * q[i < 8 ? s * 8 + i : 32 + s * 8 + i - 8];
                    sb += (float)si * 0.5f * oc_nvfp4_ue4m3(blk[s]);
                }
                acc += sb * acts[t].d[b];
            }
            y[t * y_stride + r] = acc * gscale;
        }
    }
}

void oc_nv3_planar(const uint8_t *w, size_t rows, size_t cols, uint8_t *out)
{
    const size_t n = rows * (cols / OC_NV3_BLOCK);
    uint8_t *sg = out + n * 16, *sc = out + n * 24;
    for (size_t i = 0; i < n; i++) {
        const uint8_t *blk = w + i * OC_NV3_BLOCK_BYTES;
        memcpy(out + i * 16, blk + 4, 16);
        memcpy(sg + i * 8, blk + 20, 8);
        memcpy(sc + i * 4, blk, 4);
    }
}
