#define _POSIX_C_SOURCE 200809L
#include "oxidize/weight_ops.h"
#include "oxidize/quant.h"

#include <stdlib.h>
#include <string.h>

/* ─── f32 GEMV ────────────────────────────────────────────────────────── */

OcError oc_gemv_f32(const float *weights, size_t rows, size_t cols,
                     const float *input, float *output)
{
    if (!weights || !input || !output) return OC_ERR_INVALID_ARG;
    if (rows == 0 || cols == 0) return OC_ERR_INVALID_ARG;

    for (size_t j = 0; j < rows; j++) {
        float sum = 0.0f;
        const float *row = &weights[j * cols];
        for (size_t i = 0; i < cols; i++)
            sum += row[i] * input[i];
        output[j] = sum;
    }
    return OC_OK;
}

/* ─── GEMV weight ─────────────────────────────────────────────────────── */

OcError oc_gemv_weight(const OcWeightStorage *ws,
                       size_t rows, size_t cols,
                       const float *input, float *output)
{
    if (!ws || !input || !output) return OC_ERR_INVALID_ARG;
    if (rows == 0 || cols == 0) return OC_ERR_INVALID_ARG;

    switch (ws->type) {
    case OC_WEIGHT_F32:
        if (ws->f32_len < rows * cols) return OC_ERR_INVALID_ARG;
        return oc_gemv_f32(ws->f32_data, rows, cols, input, output);

    case OC_WEIGHT_QUANTIZED:
    case OC_WEIGHT_MMAP_QUANTIZED: {
        size_t sz;
        const uint8_t *data = oc_weight_storage_quant_bytes(ws, &sz);
        if (!data) return OC_ERR_INVALID_ARG;
        /* Dequantize the full weight matrix row by row. */
        size_t row_bytes = oc_quantized_size(ws->qtype, cols);
        if (row_bytes == 0) return OC_ERR_INVALID_ARG;
        if (row_bytes * rows > sz) return OC_ERR_INVALID_ARG;

        float *row_f32 = malloc(cols * sizeof(float));
        if (!row_f32) return OC_ERR_OOM;

        for (size_t j = 0; j < rows; j++) {
            OcError e = oc_quant_dequant_row_scalar(ws->qtype,
                                                     data + j * row_bytes,
                                                     row_bytes,
                                                     row_f32, cols);
            if (e != OC_OK) {
                free(row_f32);
                return e;
            }
            float sum = 0.0f;
            for (size_t i = 0; i < cols; i++)
                sum += row_f32[i] * input[i];
            output[j] = sum;
        }
        free(row_f32);
        return OC_OK;
    }
    }
    return OC_ERR_INVALID_ARG;
}

/* ─── GEMV expert weight ──────────────────────────────────────────────── */

OcError oc_gemv_expert_weight(const OcWeightStorage *ws,
                               size_t expert_idx, size_t n_experts,
                               size_t rows, size_t cols,
                               const float *input, float *output)
{
    if (!ws || !input || !output) return OC_ERR_INVALID_ARG;
    if (expert_idx >= n_experts) return OC_ERR_INVALID_ARG;
    if (rows == 0 || cols == 0) return OC_ERR_INVALID_ARG;

    switch (ws->type) {
    case OC_WEIGHT_F32: {
        size_t values_per_expert = rows * cols;
        size_t start = expert_idx * values_per_expert;
        if (start + values_per_expert > ws->f32_len) return OC_ERR_INVALID_ARG;
        return oc_gemv_f32(&ws->f32_data[start], rows, cols, input, output);
    }
    case OC_WEIGHT_QUANTIZED:
    case OC_WEIGHT_MMAP_QUANTIZED: {
        size_t sz;
        const uint8_t *data = oc_weight_storage_quant_bytes(ws, &sz);
        if (!data) return OC_ERR_INVALID_ARG;
        size_t row_bytes = oc_quantized_size(ws->qtype, cols);
        if (row_bytes == 0) return OC_ERR_INVALID_ARG;
        size_t expert_size = row_bytes * rows;
        size_t start = expert_idx * expert_size;
        if (start + expert_size > sz) return OC_ERR_INVALID_ARG;

        float *row_f32 = malloc(cols * sizeof(float));
        if (!row_f32) return OC_ERR_OOM;

        for (size_t j = 0; j < rows; j++) {
            OcError e = oc_quant_dequant_row_scalar(ws->qtype,
                                                     data + start + j * row_bytes,
                                                     row_bytes,
                                                     row_f32, cols);
            if (e != OC_OK) {
                free(row_f32);
                return e;
            }
            float sum = 0.0f;
            for (size_t i = 0; i < cols; i++)
                sum += row_f32[i] * input[i];
            output[j] = sum;
        }
        free(row_f32);
        return OC_OK;
    }
    }
    return OC_ERR_INVALID_ARG;
}

/* ─── Fused GEMV ──────────────────────────────────────────────────────── */

OcError oc_gemv_weight_fused(OcGemvPart *parts, size_t n_parts,
                              size_t cols, const float *input)
{
    if (!parts || !input) return OC_ERR_INVALID_ARG;
    if (n_parts == 0) return OC_OK;

    for (size_t p = 0; p < n_parts; p++) {
        if (parts[p].rows == 0) continue;
        OcError e = oc_gemv_weight(parts[p].storage, parts[p].rows, cols,
                                    input, parts[p].output);
        if (e != OC_OK) return e;
    }
    return OC_OK;
}

/* ─── Batched GEMM ────────────────────────────────────────────────────── */

OcError oc_gemm_weight(const OcWeightStorage *ws,
                        size_t rows, size_t cols,
                        const float *inputs, float *outputs, size_t batch)
{
    if (!ws || !inputs || !outputs) return OC_ERR_INVALID_ARG;
    if (rows == 0 || cols == 0 || batch == 0) return OC_ERR_INVALID_ARG;

    /* For batch=1, use GEMV. */
    if (batch == 1)
        return oc_gemv_weight(ws, rows, cols, inputs, outputs);

    switch (ws->type) {
    case OC_WEIGHT_F32: {
        if (ws->f32_len < rows * cols) return OC_ERR_INVALID_ARG;
        for (size_t b = 0; b < batch; b++) {
            const float *in = &inputs[b * cols];
            float *out = &outputs[b * rows];
            for (size_t j = 0; j < rows; j++) {
                float sum = 0.0f;
                const float *row = &ws->f32_data[j * cols];
                for (size_t i = 0; i < cols; i++)
                    sum += row[i] * in[i];
                out[j] = sum;
            }
        }
        return OC_OK;
    }
    case OC_WEIGHT_QUANTIZED:
    case OC_WEIGHT_MMAP_QUANTIZED: {
        size_t sz;
        const uint8_t *data = oc_weight_storage_quant_bytes(ws, &sz);
        if (!data) return OC_ERR_INVALID_ARG;
        size_t row_bytes = oc_quantized_size(ws->qtype, cols);
        if (row_bytes == 0) return OC_ERR_INVALID_ARG;
        if (row_bytes * rows > sz) return OC_ERR_INVALID_ARG;

        float *row_f32 = malloc(cols * sizeof(float));
        if (!row_f32) return OC_ERR_OOM;

        for (size_t b = 0; b < batch; b++) {
            const float *in = &inputs[b * cols];
            float *out = &outputs[b * rows];
            for (size_t j = 0; j < rows; j++) {
                OcError e = oc_quant_dequant_row_scalar(ws->qtype,
                                                         data + j * row_bytes,
                                                         row_bytes,
                                                         row_f32, cols);
                if (e != OC_OK) {
                    free(row_f32);
                    return e;
                }
                float sum = 0.0f;
                for (size_t i = 0; i < cols; i++)
                    sum += row_f32[i] * in[i];
                out[j] = sum;
            }
        }
        free(row_f32);
        return OC_OK;
    }
    }
    return OC_ERR_INVALID_ARG;
}

/* ─── Add repeating bias ─────────────────────────────────────────────── */

void oc_add_repeating_bias(float *buf, size_t buf_len,
                            const float *bias, size_t bias_len)
{
    if (!buf || !bias || bias_len == 0) return;
    for (size_t i = 0; i < buf_len; i++)
        buf[i] += bias[i % bias_len];
}
