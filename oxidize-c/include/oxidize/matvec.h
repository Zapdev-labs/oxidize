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

#include <stdbool.h>

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
 * matrix; results are written to `outs[i]`. `temp` must hold `cols` f32.
 * All matrices share that input dimension. */
void oc_matvec_quantized_fused(OcGgufQuantizationType qtype,
                               const uint8_t *const *datas, const size_t *rows,
                               size_t cols, const size_t *row_bytes,
                               size_t n_outs, const float *input,
                               float *const *outs, float *temp);

/* ─── Batched matvec (prefill) ───────────────────────────────────────────
 *
 * Same product as oc_matvec_quantized(), but against `n_vec` activation
 * vectors at once. This is what makes prompt prefill fast: a single-vector
 * GEMV re-reads the entire weight matrix from DRAM for every token, so a
 * 500-token prompt streamed 500 copies of the model through memory. Sharing
 * one pass over the weights across a tile of tokens turns that into
 * ceil(n_vec / tile) passes.
 *
 * The activations are tiled (not the whole batch at once) so that the tile
 * stays resident in L2 while the weight rows stream past it — with an
 * untiled inner loop the activations, not the weights, become the streamed
 * operand and the amortization is lost.
 *
 * Layout: `inputs` is n_vec vectors of `cols` floats at stride `in_stride`;
 * `outputs` is n_vec vectors of `rows` floats at stride `out_stride`.
 * Strides are in floats and may exceed the vector length (so callers can
 * point into a padded [n_vec][max_dim] scratch block).
 *
 * `act_scratch` / `act_bytes` describe the fused-path activation buffer;
 * pass NULL/0 to force the dequant path. The tile is clamped to whatever
 * `act_bytes` allows, so an undersized buffer costs speed, never memory
 * safety. Size it with oc_matvec_batch_scratch_bytes(). `temp` is the usual
 * per-call dequant buffer of `cols` floats.
 *
 * Results are bit-identical to calling oc_matvec_quantized() once per vector:
 * each output element is still one row dotted with one activation, in the
 * same order, and rows are still split across threads without reduction. */
void oc_matvec_quantized_batch(OcGgufQuantizationType qtype,
                               const uint8_t *data, size_t rows, size_t cols,
                               size_t row_bytes,
                               const float *inputs, size_t in_stride,
                               float *outputs, size_t out_stride,
                               size_t n_vec, float *temp,
                               uint8_t *act_scratch, size_t act_bytes);

/* Worst-case `act_scratch` size for ANY oc_matvec_quantized_batch() call whose
 * input width is at most `max_cols`. One buffer sized by this covers every
 * matmul in a forward pass — including narrow ones: the tile count is derived
 * from `cols` by integer division, so a narrower matrix can need marginally
 * MORE scratch than a wide one, and taking the max over a few sampled widths
 * is not a bound. This returns a true bound. */
size_t oc_matvec_batch_scratch_bytes(size_t max_cols);

/* f32-weight counterpart of oc_matvec_quantized_batch(). */
void oc_matvec_f32_batch(const float *data, size_t rows, size_t cols,
                         const float *inputs, size_t in_stride,
                         float *outputs, size_t out_stride, size_t n_vec);

/* Enable or disable the fused integer GEMV path (default: enabled).
 *
 * When enabled, oc_matvec_quantized() quantizes the activation to Q8 once and
 * uses the OXK integer kernels instead of dequantizing each weight row to f32.
 * That is substantially faster but introduces int8 activation quantization
 * error, so results differ from the dequant reference. Disable it to get the
 * exact dequant-path numbers (used by the parity tests). */
void oc_matvec_set_fused(bool enabled);
bool oc_matvec_fused_enabled(void);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MATVEC_H */
