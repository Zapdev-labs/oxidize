/*
 * weight_ops.h — Weight matrix operations for inference.
 *
 * Port of oxidize-core/src/model/inference.rs GEMV/GEMM functions:
 *   gemv_weight, gemv_expert_weight, gemv_weight_fused, gemm_weight,
 *   add_repeating_bias
 *
 * These operate on OcWeightStorage to perform matrix-vector products
 * against f32 or quantized weight matrices in GGUF natural row-major layout.
 */
#ifndef OXIDIZE_WEIGHT_OPS_H
#define OXIDIZE_WEIGHT_OPS_H

#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/weight_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GEMV: y = W @ x, where W is [rows, cols] in natural row-major layout.
 * For F32 storage: direct dot product per row.
 * For Quantized/Mmap: dequantizes blocks on the fly.
 * input must have cols floats, output must have rows floats. */
OcError oc_gemv_weight(const OcWeightStorage *ws,
                       size_t rows, size_t cols,
                       const float *input, float *output);

/* GEMV for a single expert in a MoE weight matrix.
 * The expert's weights start at expert_idx * (rows * cols) into the storage. */
OcError oc_gemv_expert_weight(const OcWeightStorage *ws,
                               size_t expert_idx, size_t n_experts,
                               size_t rows, size_t cols,
                               const float *input, float *output);

/* Fused GEMV: run multiple projections (q/k/v, gate/up) sharing the same
 * input vector. Each part specifies (storage, rows, output_buffer).
 * n_parts is the number of entries. cols and input are shared. */
typedef struct {
    const OcWeightStorage *storage;
    size_t                  rows;
    float                  *output;
} OcGemvPart;

OcError oc_gemv_weight_fused(OcGemvPart *parts, size_t n_parts,
                              size_t cols, const float *input);

/* Batched GEMM: Y = W @ X, where W is [rows, cols] and X is [batch, cols].
 * output is [batch, rows] row-major. */
OcError oc_gemm_weight(const OcWeightStorage *ws,
                        size_t rows, size_t cols,
                        const float *inputs, float *outputs, size_t batch);

/* Add a per-row bias, repeating modulo bias_len when shorter than a row.
 * buf is [batch * row_len] or [len]. bias is [bias_len]. */
void oc_add_repeating_bias(float *buf, size_t buf_len,
                            const float *bias, size_t bias_len);

/* Simple f32 GEMV: y[j] = sum_i W[j*cols + i] * x[i] for j in 0..rows. */
OcError oc_gemv_f32(const float *weights, size_t rows, size_t cols,
                     const float *input, float *output);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_WEIGHT_OPS_H */
