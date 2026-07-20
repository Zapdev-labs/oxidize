/*
 * matvec.h — quantized/f32 matrix-vector products (weight × activation).
 *
 * Port of the Rust scalar-fallback `gemv_weight` / `gemv_quantized_f32`
 * (dequant-one-row-then-f32-dot) path from oxidize-core/src/compute/tensor/
 * kernels/gemv.rs. The fused q8_k int-dot fast paths are a future
 * optimization; this path is correct for ALL quant types and reuses the
 * SIMD-accelerated `oc_quant_dequant_row` for the dequant step.
 *
 * Weight layout (GGUF row-major, dims[0]=cols innermost, dims[1]=rows):
 *   weight[j] occupies bytes [j*row_bytes .. (j+1)*row_bytes)
 *   output[j] = sum_i weight[j,i] * input[i]
 * where `row_bytes = oc_quantized_size(qtype, cols)` (or cols*sizeof(f32)).
 */
#ifndef OXIDIZE_MATVEC_H
#define OXIDIZE_MATVEC_H

#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/quant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* f32 weight × f32 input → f32 output. `data` is row-major [rows][cols].
 * `input` length == cols, `output` length == rows. */
void oc_matvec_f32(const float *data, size_t rows, size_t cols,
                   const float *input, float *output);

/* Quantized weight × f32 input → f32 output. Dequantizes one weight row at
 * a time into `temp` (caller-provided, length >= cols) and dots with `input`.
 * `data` is the packed quant buffer; `row_bytes` is the per-row byte stride
 * (must equal `oc_quantized_size(qtype, cols)`). Mirrors the Rust scalar
 * `gemv_dequant_scalar_fallback`. */
void oc_matvec_quantized(OcGgufQuantizationType qtype, const uint8_t *data,
                         size_t rows, size_t cols, size_t row_bytes,
                         const float *input, float *output, float *temp);

/* Fused multi-projection variant: computes several matvecs that share the
 * SAME input vector in one call (so the input is read once from cache).
 * Used for the fused Q/K/V (or gate/up) projections. `n_outs` is the number
 * of outputs; `datas[i]`, `rows[i]`, `row_bytes[i]` describe each weight
 * matrix; results are written to `outs[i]`. `temp` must hold `max_cols` f32
 * where `max_cols` is the largest `cols[i]`. All matrices share `cols[0]`
 * (the input dimension) in the Llama case; callers pass a single `cols`. */
void oc_matvec_quantized_fused(OcGgufQuantizationType qtype,
                               const uint8_t *const *datas, const size_t *rows,
                               size_t cols, const size_t *row_bytes,
                               size_t n_outs, const float *input,
                               float *const *outs, float *temp);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MATVEC_H */
