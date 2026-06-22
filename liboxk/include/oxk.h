#ifndef OXK_H
#define OXK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OXK_QK_K 256
#define OXK_BLOCK_Q4_K_SIZE 144
#define OXK_BLOCK_Q8_K_BYTES 292

/* Initialize CPU feature detection. Returns 0 on success. */
int oxk_init(void);

const char *oxk_version(void);

/* Non-zero if AVX2+FMA kernels are available on this CPU. */
int oxk_has_avx2(void);

/* Quantize f32 vector into n_blocks Q8_K blocks (llama.cpp layout). */
int oxk_quantize_q8k(const float *vector, size_t n_blocks, uint8_t *out);

/* Fused Q4_K GEMV: output[i] = dot(row_i, vector) for n_rows. */
int oxk_gemv_q4k_range(const uint8_t *rows, size_t n_rows, size_t blocks_per_row,
                       const uint8_t *q8k, float *out);

/* Full Q4_K GEMV from f32 activation: quantizes vector internally. */
int oxk_gemv_q4k_f32(const uint8_t *weight_rows, size_t n_rows, size_t in_dim,
                     const float *vector, float *output);

/* General quantized GEMV — mirrors oxidize_gemv_quantized type codes. */
int oxk_gemv_quantized(uint32_t quant_type, const uint8_t *qbytes, size_t qbytes_len,
                       size_t rows, size_t cols, const float *vector, float *output);

/* f32 dot product with SIMD when available. */
float oxk_dot_f32(const float *a, const float *b, size_t n);

/* RMS normalization: out[i] = x[i] / rms * weight[i]. */
int oxk_rms_norm_f32(const float *x, const float *weight, size_t n, float eps,
                     float *out);

#ifdef __cplusplus
}
#endif

#endif /* OXK_H */
