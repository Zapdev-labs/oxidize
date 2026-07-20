/*
 * matvec.c — quantized/f32 matrix-vector products (scalar reference).
 *
 * Port of oxidize-core/src/compute/tensor/kernels/gemv.rs
 * (`gemv_f32`, `gemv_dequant_scalar_fallback`). The dequant step reuses
 * the SIMD-accelerated `oc_quant_dequant_row`, so this path inherits the
 * AVX2/AVX-512 speedups on capable hosts.
 */
#include "oxidize/matvec.h"
#include "oxidize/quant.h"

void oc_matvec_f32(const float *data, size_t rows, size_t cols,
                   const float *input, float *output)
{
    for (size_t j = 0; j < rows; j++) {
        const float *row = data + j * cols;
        float acc = 0.0f;
        for (size_t i = 0; i < cols; i++) {
            acc += row[i] * input[i];
        }
        output[j] = acc;
    }
}

void oc_matvec_quantized(OcGgufQuantizationType qtype, const uint8_t *data,
                         size_t rows, size_t cols, size_t row_bytes,
                         const float *input, float *output, float *temp)
{
    for (size_t j = 0; j < rows; j++) {
        const uint8_t *row = data + j * row_bytes;
        /* Dequantize this weight row into `temp` (SIMD on capable hosts). */
        oc_quant_dequant_row(qtype, row, row_bytes, temp, cols);
        float acc = 0.0f;
        for (size_t i = 0; i < cols; i++) {
            acc += temp[i] * input[i];
        }
        output[j] = acc;
    }
}

void oc_matvec_quantized_fused(OcGgufQuantizationType qtype,
                               const uint8_t *const *datas, const size_t *rows,
                               size_t cols, const size_t *row_bytes,
                               size_t n_outs, const float *input,
                               float *const *outs, float *temp)
{
    for (size_t k = 0; k < n_outs; k++) {
        oc_matvec_quantized(qtype, datas[k], rows[k], cols, row_bytes[k],
                            input, outs[k], temp);
    }
}
