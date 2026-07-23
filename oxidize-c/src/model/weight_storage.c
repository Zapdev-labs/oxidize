#define _POSIX_C_SOURCE 200809L
#include "oxidize/weight_storage.h"

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
