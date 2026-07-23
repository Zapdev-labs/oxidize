#define _POSIX_C_SOURCE 200809L
#include "oxidize/weight_storage.h"
#include "oxidize/quant.h"

#include <stdlib.h>
#include <string.h>

void oc_weight_storage_init(OcWeightStorage *ws)
{
    if (!ws) return;
    memset(ws, 0, sizeof(*ws));
    ws->type = OC_WEIGHT_F32;
}

OcError oc_weight_storage_f32(OcWeightStorage *ws, float *data, size_t len)
{
    if (!ws) return OC_ERR_INVALID_ARG;
    oc_weight_storage_free(ws);
    ws->type = OC_WEIGHT_F32;
    ws->f32_data = data;
    ws->f32_len = len;
    return OC_OK;
}

OcError oc_weight_storage_quantized(OcWeightStorage *ws,
                                     OcGgufQuantizationType qtype,
                                     uint8_t *data, size_t size)
{
    if (!ws || qtype == OC_QUANT_UNKNOWN) return OC_ERR_INVALID_ARG;
    oc_weight_storage_free(ws);
    ws->type = OC_WEIGHT_QUANTIZED;
    ws->qtype = qtype;
    ws->quant_data = data;
    ws->quant_size = size;
    return OC_OK;
}

OcError oc_weight_storage_mmap(OcWeightStorage *ws,
                                OcGgufQuantizationType qtype,
                                const uint8_t *mmap_base,
                                size_t offset, size_t size)
{
    if (!ws || !mmap_base || qtype == OC_QUANT_UNKNOWN) return OC_ERR_INVALID_ARG;
    oc_weight_storage_free(ws);
    ws->type = OC_WEIGHT_MMAP_QUANTIZED;
    ws->qtype = qtype;
    ws->mmap_data = mmap_base;
    ws->mmap_offset = offset;
    ws->mmap_size = size;
    return OC_OK;
}

bool oc_weight_storage_is_empty(const OcWeightStorage *ws)
{
    if (!ws) return true;
    switch (ws->type) {
    case OC_WEIGHT_F32:
        return ws->f32_len == 0;
    case OC_WEIGHT_QUANTIZED:
        return ws->quant_size == 0;
    case OC_WEIGHT_MMAP_QUANTIZED:
        return ws->mmap_size == 0;
    }
    return true;
}

size_t oc_weight_storage_output_dim(const OcWeightStorage *ws, size_t input_dim)
{
    if (!ws || input_dim == 0) return 0;
    switch (ws->type) {
    case OC_WEIGHT_F32:
        return ws->f32_len / input_dim;
    case OC_WEIGHT_QUANTIZED:
    case OC_WEIGHT_MMAP_QUANTIZED: {
        OcQuantBlockLayout bl = oc_quant_block_size(ws->qtype);
        if (bl.elements_per_block == 0 || bl.bytes_per_block == 0) return 0;
        size_t bytes_per_row = (input_dim / bl.elements_per_block) * bl.bytes_per_block;
        if (bytes_per_row == 0) return 0;
        size_t total_bytes = (ws->type == OC_WEIGHT_MMAP_QUANTIZED)
                             ? ws->mmap_size : ws->quant_size;
        return total_bytes / bytes_per_row;
    }
    }
    return 0;
}

const uint8_t *oc_weight_storage_quant_bytes(const OcWeightStorage *ws,
                                              size_t *out_size)
{
    if (!ws || !out_size) return NULL;
    switch (ws->type) {
    case OC_WEIGHT_QUANTIZED:
        *out_size = ws->quant_size;
        return ws->quant_data;
    case OC_WEIGHT_MMAP_QUANTIZED:
        *out_size = ws->mmap_size;
        if (!ws->mmap_data) return NULL;
        return ws->mmap_data + ws->mmap_offset;
    case OC_WEIGHT_F32:
        return NULL;
    }
    return NULL;
}

const float *oc_weight_storage_f32_data(const OcWeightStorage *ws,
                                         size_t *out_len)
{
    if (!ws || ws->type != OC_WEIGHT_F32) return NULL;
    if (out_len) *out_len = ws->f32_len;
    return ws->f32_data;
}

OcGgufQuantizationType oc_weight_storage_qtype(const OcWeightStorage *ws)
{
    if (!ws) return OC_QUANT_UNKNOWN;
    if (ws->type == OC_WEIGHT_F32) return OC_QUANT_UNKNOWN;
    return ws->qtype;
}

void oc_weight_storage_free(OcWeightStorage *ws)
{
    if (!ws) return;
    switch (ws->type) {
    case OC_WEIGHT_F32:
        free(ws->f32_data);
        break;
    case OC_WEIGHT_QUANTIZED:
        free(ws->quant_data);
        break;
    case OC_WEIGHT_MMAP_QUANTIZED:
        /* Do not free mmap data (caller owns it). */
        break;
    }
    memset(ws, 0, sizeof(*ws));
}

OcError oc_weight_storage_lookup_embedding(const OcWeightStorage *ws,
                                            size_t hidden_size,
                                            size_t vocab_size,
                                            uint32_t token_idx,
                                            float *out)
{
    if (!ws || !out || hidden_size == 0 || vocab_size == 0)
        return OC_ERR_INVALID_ARG;

    /* Zero output. */
    memset(out, 0, hidden_size * sizeof(float));

    /* Clamp token to valid range. */
    if (token_idx >= vocab_size)
        token_idx = (uint32_t)(vocab_size - 1);

    switch (ws->type) {
    case OC_WEIGHT_F32: {
        size_t start = (size_t)token_idx * hidden_size;
        size_t end = start + hidden_size;
        if (end <= ws->f32_len)
            memcpy(out, &ws->f32_data[start], hidden_size * sizeof(float));
        return OC_OK;
    }
    case OC_WEIGHT_QUANTIZED:
    case OC_WEIGHT_MMAP_QUANTIZED: {
        size_t sz;
        const uint8_t *data = oc_weight_storage_quant_bytes(ws, &sz);
        if (!data) return OC_ERR_INVALID_ARG;
        /* Dequantize the specific token row. */
        size_t offset = (size_t)token_idx * oc_quantized_size(ws->qtype, hidden_size);
        /* Use the quant dequant API if the row fits. */
        /* For Q8_0: block_width=32, block_size=34. */
        /* Fall back to scalar dequant for the row. */
        /* We call oc_quant_dequant_row_scalar which expects the full row. */
        if (offset + oc_quantized_size(ws->qtype, hidden_size) <= sz) {
            size_t row_sz = oc_quantized_size(ws->qtype, hidden_size);
            return oc_quant_dequant_row_scalar(ws->qtype,
                                                data + offset, row_sz,
                                                out, hidden_size);
        }
        return OC_ERR_INVALID_ARG;
    }
    }
    return OC_ERR_INVALID_ARG;
}
