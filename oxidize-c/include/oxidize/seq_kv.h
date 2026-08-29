/* seq_kv.h — Per-sequence KV buffer for continuous batching. */
#ifndef OXIDIZE_SEQ_KV_H
#define OXIDIZE_SEQ_KV_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float   *key;             /* [kv_layer_count * capacity_tokens * kv_len] */
    float   *value;           /* same size as key */
    size_t   len;             /* number of KV positions written (== next decode pos) */
    size_t   capacity_tokens; /* max positions per layer */
    size_t   kv_layer_count;  /* number of attention layers */
    size_t   kv_len;          /* channels per position (n_kv_heads * head_dim) */
} OcSeqKv;

/* Allocate a zeroed KV buffer for one sequence. */
OcError oc_seq_kv_init(OcSeqKv *kv, size_t kv_layer_count,
                       size_t capacity_tokens, size_t kv_len);

/* Free the KV buffer. Safe on NULL. */
void oc_seq_kv_free(OcSeqKv *kv);

/* Get pointers to K and V at (layer, pos). */
OcError oc_seq_kv_get(const OcSeqKv *kv, size_t layer, size_t pos,
                       const float **out_k, const float **out_v);

/* Write K and V at (layer, pos). k and v must each have kv_len floats. */
OcError oc_seq_kv_put(OcSeqKv *kv, size_t layer, size_t pos,
                      const float *k, const float *v);

/* Advance the write position by n (after writing n tokens). */
OcError oc_seq_kv_advance(OcSeqKv *kv, size_t n);

/* Truncate to exactly n positions. If n >= len, no-op. */
OcError oc_seq_kv_truncate(OcSeqKv *kv, size_t n);

/* Clear all data (zero fill, reset len to 0). */
void oc_seq_kv_clear(OcSeqKv *kv);

/* Total allocated bytes (K + V). */
size_t oc_seq_kv_size_bytes(const OcSeqKv *kv);

/* Check if the KV buffer is empty (len == 0). */
bool oc_seq_kv_is_empty(const OcSeqKv *kv);

/* Total capacity in positions. */
size_t oc_seq_kv_capacity(const OcSeqKv *kv);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SEQ_KV_H */
