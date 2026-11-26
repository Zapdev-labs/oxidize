/*
 * nv3.h — NV3, a 3-bit sibling of NVFP4 for the Inkling expert weights
 * (oxidize GGUF type 250).
 *
 * 64 values per 28-byte block: four UE4M3 scale bytes (one per 16 values),
 * four 32-bit magnitude words (one per 16-value sub-block) and two 32-bit
 * sign words. Value v = 16 s + 4 c + j (s, c, j in 0..3) keeps its 2-bit
 * magnitude index in bits 8 j + 2 c of magnitude word s and its sign in bit
 * 8 j + 4 (s & 1) + c of sign word s / 2, and decodes as
 *
 *     w = ±{1, 3, 6, 10}[m] * ue4m3(scale s) / 2 * global_scale
 *
 * The magnitudes fit the heavy-tailed NVFP4 checkpoint values better than a
 * uniform 3-bit grid, and the bit layout lets GPU kernels expand four values
 * at a time into int8 lanes. Activations use the NVFP4 int8 format
 * (OcNvfp4Act). 3.5 bits per weight against NVFP4's 4.5.
 */
#ifndef OXIDIZE_NV3_H
#define OXIDIZE_NV3_H

#include <stddef.h>
#include <stdint.h>

#include "oxidize/nvfp4.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_NV3_GGML_TYPE   250u
#define OC_NV3_BLOCK       64u
#define OC_NV3_BLOCK_BYTES 28u

/* Bytes of one NV3 row of `cols` values (cols % 64 == 0). */
size_t oc_nv3_row_bytes(size_t cols);

/* Global scale that maps the largest |x| onto the top code at UE4M3 448. */
float oc_nv3_global_scale(const float *x, size_t n);

/* Quantize `n` floats (n % 64 == 0) with a fixed global scale, choosing each
 * sub-block scale to minimize the squared error, weighted by imp[i] when imp
 * is not NULL. */
void oc_nv3_quantize_row(const float *x, const float *imp, size_t n, float gscale, uint8_t *out);

/* Dequantize one row (without the global scale). */
void oc_nv3_dequant_row(const uint8_t *row, size_t cols, float *out);

/* y[r] for vector v at y + v * y_stride = gscale * dot(row r, act v), r in
 * [r0, r1). */
void oc_nv3_gemm_rows(const uint8_t *w, size_t cols, size_t r0, size_t r1, float gscale,
                      const OcNvfp4Act *acts, size_t n_vec, float *y, size_t y_stride);

/* Planar copy of a [rows x cols] matrix for GPU kernels: the rows * cols/64
 * blocks' 16 magnitude bytes, then their 8 sign bytes, then their 4 scale
 * bytes, each plane in row-major block order. Same total size. */
void oc_nv3_planar(const uint8_t *w, size_t rows, size_t cols, uint8_t *out);

#ifdef __cplusplus
}
#endif

#endif
