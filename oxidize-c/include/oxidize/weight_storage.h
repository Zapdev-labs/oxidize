/* weight_storage.h — Weight storage abstraction for inference. */
#ifndef OXIDIZE_WEIGHT_STORAGE_H
#define OXIDIZE_WEIGHT_STORAGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/quant.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_WEIGHT_F32 = 0,          /* owning f32 vector */
    OC_WEIGHT_QUANTIZED = 1,    /* owning quantized byte buffer */
    OC_WEIGHT_MMAP_QUANTIZED = 2, /* zero-copy mmap-backed quantized */
} OcWeightStorageType;

typedef struct {
    OcWeightStorageType type;

    /* F32 storage. */
    float   *f32_data;
    size_t   f32_len;

    /* Quantized storage (owning). */
    OcGgufQuantizationType qtype;
    uint8_t *quant_data;
    size_t   quant_size;

    /* MmapQuantized storage (zero-copy). */
    const uint8_t *mmap_data;   /* base pointer of mmap region */
    size_t   mmap_offset;       /* offset within the mmap */
    size_t   mmap_size;         /* size in bytes */
} OcWeightStorage;

/* Initialize as empty F32 storage. */
void oc_weight_storage_init(OcWeightStorage *ws);

/* Create F32 storage (takes ownership of data, which must be malloc'd). */
OcError oc_weight_storage_f32(OcWeightStorage *ws, float *data, size_t len);

/* Create quantized storage (takes ownership of data, which must be malloc'd). */
OcError oc_weight_storage_quantized(OcWeightStorage *ws,
                                     OcGgufQuantizationType qtype,
                                     uint8_t *data, size_t size);

/* Create mmap-backed quantized storage (zero-copy, does not own data). */
OcError oc_weight_storage_mmap(OcWeightStorage *ws,
                                OcGgufQuantizationType qtype,
                                const uint8_t *mmap_base,
                                size_t offset, size_t size);

/* Check if storage is empty. */
bool oc_weight_storage_is_empty(const OcWeightStorage *ws);

/* Compute output dimension given input dimension.
 * F32: len / input_dim.
 * Quantized: size / bytes_per_row. */
size_t oc_weight_storage_output_dim(const OcWeightStorage *ws, size_t input_dim);

/* Get raw quantized bytes (returns NULL for F32 storage). */
const uint8_t *oc_weight_storage_quant_bytes(const OcWeightStorage *ws,
                                              size_t *out_size);

/* Get F32 data pointer (returns NULL for quantized storage). */
const float *oc_weight_storage_f32_data(const OcWeightStorage *ws,
                                         size_t *out_len);

/* Get the quant type (returns OC_QUANT_UNKNOWN for F32). */
OcGgufQuantizationType oc_weight_storage_qtype(const OcWeightStorage *ws);

/* Free owned buffers (f32_data and quant_data). Does not free mmap data. */
void oc_weight_storage_free(OcWeightStorage *ws);

/* Lookup a token embedding from weight storage. */
OcError oc_weight_storage_lookup_embedding(const OcWeightStorage *ws,
                                            size_t hidden_size,
                                            size_t vocab_size,
                                            uint32_t token_idx,
                                            float *out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_WEIGHT_STORAGE_H */
