/*
 * nvfp4.h — NVFP4 linear kernels for the Inkling path.
 *
 * Weights use the ggml NVFP4 block layout (GGUF type 40): 64 values per
 * 36-byte block, four UE4M3 scale bytes (one per 16-value sub-block) then
 * 32 bytes of E2M1 nibbles, where byte s*8+j holds value s*16+j in its low
 * nibble and value s*16+j+8 in its high nibble. A weight decodes as
 *
 *     w = e2m1(nibble) * ue4m3(scale byte) * global_scale
 *
 * with e2m1 in {0, .5, 1, 1.5, 2, 3, 4, 6} (bit 3 = sign) and the per-matrix
 * f32 global_scale stored beside the tensor. This is the ModelOpt NVFP4
 * convention, so checkpoint experts are copied bit for bit.
 *
 * Activations are quantized once per vector to int8 per 64-value block with
 * one f32 scale, and stored permuted into the weight nibble order so the AVX2
 * kernel needs no shuffles on the activation side.
 */
#ifndef OXIDIZE_NVFP4_H
#define OXIDIZE_NVFP4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OC_NVFP4_BLOCK       64u
#define OC_NVFP4_BLOCK_BYTES 36u

/* Quantized activation of one vector: cols int8 values (block-permuted) and
 * cols/64 scales. */
typedef struct OcNvfp4Act {
    int8_t *q;
    float  *d;
} OcNvfp4Act;

/* Bytes of one NVFP4 row of `cols` values (cols % 64 == 0). */
size_t oc_nvfp4_row_bytes(size_t cols);

/* UE4M3 byte -> f32 (no sign bit; 0x7F and 0 decode to 0). */
float oc_nvfp4_ue4m3(uint8_t b);

/* Nearest UE4M3 code for a positive value (round to nearest, ties up). */
uint8_t oc_nvfp4_ue4m3_encode(float v);

/* Quantize `cols` floats into `out` (q: cols bytes, d: cols/64 floats). */
void oc_nvfp4_quantize_act(const float *x, size_t cols, OcNvfp4Act out);

/* Dequantize one row (without the global scale). */
void oc_nvfp4_dequant_row(const uint8_t *row, size_t cols, float *out);

/* y[r] = gscale * dot(row r, act) for r in [r0, r1). Scalar reference. */
void oc_nvfp4_gemv_rows_scalar(const uint8_t *w, size_t cols, size_t r0, size_t r1,
                               float gscale, OcNvfp4Act act, float *y);

/* Dispatched kernel: AVX2 when available, scalar otherwise.
 * n_vec activations; y for vector v at y + v*y_stride. */
void oc_nvfp4_gemm_rows(const uint8_t *w, size_t cols, size_t r0, size_t r1,
                        float gscale, const OcNvfp4Act *acts, size_t n_vec,
                        float *y, size_t y_stride);

/* Quantize `n` floats (n % 64 == 0) to NVFP4 with a fixed global scale,
 * choosing each sub-block scale to minimize squared error among the
 * nearest UE4M3 codes. */
void oc_nvfp4_quantize_row(const float *x, size_t n, float gscale, uint8_t *out);

/* Global scale that maps the largest |x| onto 6 * 448 (full UE4M3 range). */
float oc_nvfp4_global_scale(const float *x, size_t n);

/* Planar copy of a [rows x cols] matrix for GPU kernels: rows * cols/64
 * blocks of 32 nibble bytes, then the rows * cols/64 four-byte scale words,
 * both in row-major block order. Same total size as the block layout. */
void oc_nvfp4_planar(const uint8_t *w, size_t rows, size_t cols, uint8_t *out);

bool oc_nvfp4_have_avx2(void);

#ifdef __cplusplus
}
#endif

#endif
