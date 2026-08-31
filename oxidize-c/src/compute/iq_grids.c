/* iq_grids.c — Importance matrix quantization grid implementation. */
#include "oxidize/iq_grids.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


/* IQ4_NL: 16 single-element entries, non-linear scale.
 * Values approximate the IQ4_NL dequant table from ggml. */
static const float IQ4_NL_VALUES[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f,
    -49.0f,  -35.0f,  -22.0f, -10.0f,
      1.0f,   13.0f,   25.0f,  38.0f,
     53.0f,   69.0f,   89.0f, 113.0f,
};

/* IQ2_XXS grid: 256 entries × 8 dims. Each byte nibble takes one of three values: 0x08, 0x19, 0x2b (dequantized as -0.5, 0.0, 0.5 by convention). We generate all combinations procedurally rather than hard-coding 256×8 floats. The three dequant levels: */
#define IQ2_LEVELS   3
static const float IQ2_DEQUANT[3] = { -0.5f, 0.0f, 0.5f };

/* The byte values used in the IQ2_XXS grid (each nibble is one of these). */
static const uint8_t IQ2_BYTES[3] = { 0x08, 0x19, 0x2b };

/* IQ3_XXS grid: 256 entries × 4 dims. Each nibble takes one of 16 values
 * (0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e). Dequantized to a
 * uniform-ish scale. We generate procedurally using the nibble value. */
#define IQ3_NIBBLES 8
static const uint8_t IQ3_NIBBLE_BYTES[IQ3_NIBBLES] = {
    0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e,
};

/* Dequantize an IQ3 nibble byte to a float. Maps the 8 byte values to a
 * linear scale from -1.0 to 1.0. */
static float iq3_nibble_dequant(uint8_t nibble)
{
    /* Find index in the nibble table, then map to [-1, 1]. */
    for (int i = 0; i < IQ3_NIBBLES; i++) {
        if (IQ3_NIBBLE_BYTES[i] == nibble) {
            /* Map index 0..7 to -1.0..1.0 (8 steps of 2/7). */
            return -1.0f + (2.0f * (float)i) / 7.0f;
        }
    }
    return 0.0f;
}

/* Generate the 2-bit (IQ2_XXS) grid: 256 entries × 8 dims. */
static void generate_iq2_grid(float *data)
{
    /* Each of 256 entries corresponds to an 8-nibble pattern.
     * We map index 0..255 to a base-3 encoding of 8 nibbles. */
    for (size_t entry = 0; entry < 256; entry++) {
        float *row = data + entry * 8;
        size_t code = entry;
        for (size_t dim = 0; dim < 8; dim++) {
            /* Each nibble value: 0, 1, or 2 (mod 3 cycling). */
            size_t level = code % 3;
            code /= 3;
            row[dim] = IQ2_DEQUANT[level];
        }
    }
}

/* Generate the 3-bit (IQ3_XXS) grid: 256 entries × 4 dims.
 * Each entry is a 4-nibble pattern where each nibble is one of 8 values. */
static void generate_iq3_grid(float *data)
{
    for (size_t entry = 0; entry < 256; entry++) {
        float *row = data + entry * 4;
        size_t code = entry;
        for (size_t dim = 0; dim < 4; dim++) {
            size_t nibble_idx = code % 8;
            code /= 8;
            row[dim] = iq3_nibble_dequant(IQ3_NIBBLE_BYTES[nibble_idx]);
        }
    }
}


OcError oc_iq_grid_init(OcIqGrid *grid, OcIqGridType type)
{
    if (!grid) return OC_ERR_INVALID_ARG;
    memset(grid, 0, sizeof(*grid));
    grid->type = type;

    switch (type) {
    case OC_IQ_GRID_2BIT:
        grid->n_dims    = 8;
        grid->n_entries = 256;
        break;
    case OC_IQ_GRID_3BIT:
        grid->n_dims    = 4;
        grid->n_entries = 256;
        break;
    case OC_IQ_GRID_4BIT:
        grid->n_dims    = 1;
        grid->n_entries = 16;
        break;
    default:
        return OC_ERR_INVALID_ARG;
    }

    grid->data = malloc(grid->n_entries * grid->n_dims * sizeof(float));
    if (!grid->data) return OC_ERR_OOM;

    switch (type) {
    case OC_IQ_GRID_2BIT:
        generate_iq2_grid(grid->data);
        /* Build lookup table from IQ2 byte values. */
        grid->lookup_size = 3;
        grid->lookup_table = malloc(3);
        if (!grid->lookup_table) { free(grid->data); grid->data = NULL; return OC_ERR_OOM; }
        memcpy(grid->lookup_table, IQ2_BYTES, 3);
        break;
    case OC_IQ_GRID_3BIT:
        generate_iq3_grid(grid->data);
        grid->lookup_size = IQ3_NIBBLES;
        grid->lookup_table = malloc(IQ3_NIBBLES);
        if (!grid->lookup_table) { free(grid->data); grid->data = NULL; return OC_ERR_OOM; }
        memcpy(grid->lookup_table, IQ3_NIBBLE_BYTES, IQ3_NIBBLES);
        break;
    case OC_IQ_GRID_4BIT:
        memcpy(grid->data, IQ4_NL_VALUES, 16 * sizeof(float));
        grid->lookup_table = NULL;
        grid->lookup_size = 0;
        break;
    }

    return OC_OK;
}

OcError oc_iq_grid_nearest(const OcIqGrid *grid, const float *vec,
                           size_t n_dims, const float *importance,
                           size_t *out_idx)
{
    if (!grid || !vec || !out_idx) return OC_ERR_INVALID_ARG;
    if (n_dims != grid->n_dims) return OC_ERR_INVALID_ARG;

    size_t best_idx = 0;
    float best_dist = 0.0f;
    bool first = true;

    for (size_t e = 0; e < grid->n_entries; e++) {
        const float *entry = grid->data + e * grid->n_dims;
        float dist = 0.0f;
        for (size_t d = 0; d < n_dims; d++) {
            float diff = vec[d] - entry[d];
            float w = (importance && importance[d] > 0.0f) ? importance[d] : 1.0f;
            dist += w * diff * diff;
        }
        if (first || dist < best_dist) {
            best_dist = dist;
            best_idx = e;
            first = false;
        }
    }
    *out_idx = best_idx;
    return OC_OK;
}

OcError oc_iq_grid_get(const OcIqGrid *grid, size_t idx, float *out_vec,
                       size_t n_dims)
{
    if (!grid || !out_vec) return OC_ERR_INVALID_ARG;
    if (idx >= grid->n_entries) return OC_ERR_INVALID_ARG;
    if (n_dims != grid->n_dims) return OC_ERR_INVALID_ARG;
    memcpy(out_vec, grid->data + idx * grid->n_dims,
           n_dims * sizeof(float));
    return OC_OK;
}

size_t oc_iq_grid_size(const OcIqGrid *grid)
{
    if (!grid) return 0;
    return grid->n_entries;
}

void oc_iq_grid_free(OcIqGrid *grid)
{
    if (!grid) return;
    free(grid->data);
    free(grid->lookup_table);
    memset(grid, 0, sizeof(*grid));
}

float oc_iq_grid_importance_weight(float error, float importance)
{
    if (importance <= 0.0f) return error;
    return error * importance;
}

OcError oc_iq_quantize_block(const OcIqGrid *grid, const float *input,
                             size_t n, const float *importance,
                             uint32_t *out_indices)
{
    if (!grid || !input || !out_indices) return OC_ERR_INVALID_ARG;
    if (grid->n_dims == 0) return OC_ERR_INVALID_ARG;
    if (n % grid->n_dims != 0) return OC_ERR_INVALID_ARG;

    size_t n_blocks = n / grid->n_dims;
    for (size_t b = 0; b < n_blocks; b++) {
        const float *block_in = input + b * grid->n_dims;
        const float *imp = importance ? importance + b * grid->n_dims : NULL;
        size_t idx;
        OcError err = oc_iq_grid_nearest(grid, block_in, grid->n_dims, imp, &idx);
        if (err != OC_OK) return err;
        out_indices[b] = (uint32_t)idx;
    }
    return OC_OK;
}

OcError oc_iq_dequantize_block(const OcIqGrid *grid, const uint32_t *indices,
                               size_t count, float *out)
{
    if (!grid || !indices || !out) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < count; i++) {
        uint32_t idx = indices[i];
        if (idx >= grid->n_entries) return OC_ERR_INVALID_ARG;
        const float *entry = grid->data + idx * grid->n_dims;
        memcpy(out + i * grid->n_dims, entry, grid->n_dims * sizeof(float));
    }
    return OC_OK;
}
