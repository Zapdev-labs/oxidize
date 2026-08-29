/* iq_grids.h — Importance matrix quantization grids for IQ2/IQ3/IQ4 types. */
#ifndef OXIDIZE_IQ_GRIDS_H
#define OXIDIZE_IQ_GRIDS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
    OC_IQ_GRID_2BIT = 0,   /* 256 entries × 8 dims (IQ2_XXS codebook)   */
    OC_IQ_GRID_3BIT = 1,   /* 256 entries × 4 dims (IQ3_XXS codebook)   */
    OC_IQ_GRID_4BIT = 2,   /*  16 entries × 1 dim   (IQ4_NL  codebook)   */
} OcIqGridType;


/* A precomputed codebook grid. `data` is a flat [n_entries × n_dims] float
 * array (row-major). `lookup_table` holds auxiliary sign/mask bytes used by
 * some IQ variants (e.g. KSIGNS_IQ2XS); may be NULL for simple grids. */
typedef struct OcIqGrid {
    OcIqGridType type;
    size_t       n_dims;          /* dimensionality of each entry         */
    size_t       n_entries;       /* number of codebook entries           */
    float       *data;            /* [n_entries * n_dims] dequantized vals */
    uint8_t     *lookup_table;    /* auxiliary byte table (or NULL)        */
    size_t       lookup_size;     /* number of bytes in lookup_table      */
} OcIqGrid;


/* Allocate and initialize a grid for the given bit width. Returns OC_OK and
 * sets `*grid` on success, OC_ERR_OOM on allocation failure,
 * OC_ERR_INVALID_ARG if `grid` is NULL. */
OcError oc_iq_grid_init(OcIqGrid *grid, OcIqGridType type);

/* Find the nearest grid entry to `vec` (length `n_dims`). If `importance` */
OcError oc_iq_grid_nearest(const OcIqGrid *grid, const float *vec,
                           size_t n_dims, const float *importance,
                           size_t *out_idx);

/* Copy grid entry `idx` into `out_vec` (length `n_dims`). Returns
 * OC_ERR_INVALID_ARG if idx is out of range. */
OcError oc_iq_grid_get(const OcIqGrid *grid, size_t idx, float *out_vec,
                       size_t n_dims);

/* Return the number of entries in the grid. Returns 0 if `grid` is NULL. */
size_t oc_iq_grid_size(const OcIqGrid *grid);

/* Free grid buffers and zero the struct. Safe on NULL. */
void oc_iq_grid_free(OcIqGrid *grid);

/* Weight an error value by an importance scalar. Returns error * importance.
 * If importance <= 0, returns error unchanged (no weighting). */
float oc_iq_grid_importance_weight(float error, float importance);

/* Quantize a block of `n` f32 values using the grid. `n` must be a multiple */
OcError oc_iq_quantize_block(const OcIqGrid *grid, const float *input,
                             size_t n, const float *importance,
                             uint32_t *out_indices);

/* Dequantize `count` grid indices into f32 values. `indices` has `count`
 * entries; `out` must hold `count * grid->n_dims` floats. Returns
 * OC_OK or OC_ERR_INVALID_ARG. */
OcError oc_iq_dequantize_block(const OcIqGrid *grid, const uint32_t *indices,
                               size_t count, float *out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_IQ_GRIDS_H */
