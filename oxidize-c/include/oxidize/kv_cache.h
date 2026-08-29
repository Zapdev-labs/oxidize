/* kv_cache.h — simple per-layer KV cache for transformer inference. */
#ifndef OXIDIZE_KV_CACHE_H
#define OXIDIZE_KV_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OC_KV_CACHE_DEFAULT_N_LAYERS    32u
#define OC_KV_CACHE_DEFAULT_N_HEADS     32u
#define OC_KV_CACHE_DEFAULT_HEAD_DIM    128u
#define OC_KV_CACHE_DEFAULT_MAX_SEQ_LEN 4096u
#define OC_KV_CACHE_DTYPE_F32           0u
#define OC_KV_CACHE_DTYPE_F16           1u


typedef struct OcKvCacheConfig {
    uint32_t n_layers;     /* number of model layers (default 32)        */
    uint32_t n_heads;      /* number of attention heads (default 32)     */
    uint32_t head_dim;     /* dimension per head (default 128)            */
    uint32_t max_seq_len;  /* max tokens storable (default 4096)          */
    uint32_t dtype;        /* 0=f32, 1=f16 (default 0)                    */
} OcKvCacheConfig;


typedef struct OcKvCache {
    OcKvCacheConfig config;
    float    *k_data;     /* [n_layers * max_seq_len * n_heads * head_dim] */
    float    *v_data;     /* [n_layers * max_seq_len * n_heads * head_dim] */
    uint32_t  n_tokens;   /* current token count                          */
    uint32_t  capacity;   /* max_seq_len                                  */
} OcKvCache;

/* Fill *cfg with sensible defaults. */
void oc_kv_cache_config_init(OcKvCacheConfig *cfg);

/* Allocate the cache buffers. Returns OC_OK on success, OC_ERR_INVALID_ARG
 * for bad arguments, OC_ERR_OOM on allocation failure. */
OcError oc_kv_cache_init(OcKvCache *cache, const OcKvCacheConfig *cfg);

/* Append n KV pairs for a given layer. k and v must each have
 * n * n_heads * head_dim floats. Returns OC_ERR_INVALID_ARG for bad
 * arguments or if the append would exceed capacity. */
OcError oc_kv_cache_append(OcKvCache *cache, uint32_t layer,
                            const float *k, const float *v, uint32_t n);

/* Get pointers to the K and V vectors at (layer, pos). Sets *k and *v to
 * the storage address. Returns OC_ERR_INVALID_ARG for bad arguments or
 * out-of-range indices. */
OcError oc_kv_cache_get(const OcKvCache *cache, uint32_t layer,
                         uint32_t pos, const float **k, const float **v);

/* Reset the cache to zero tokens (does not free memory). */
void oc_kv_cache_clear(OcKvCache *cache);

/* Truncate the cache to exactly n tokens. If n >= current count this is a
 * no-op. Returns OC_ERR_INVALID_ARG for bad args or n > capacity. */
OcError oc_kv_cache_truncate(OcKvCache *cache, uint32_t n);

/* Current token count. Returns 0 on NULL. */
uint32_t oc_kv_cache_n_tokens(const OcKvCache *cache);

/* Max token capacity. Returns 0 on NULL. */
uint32_t oc_kv_cache_capacity(const OcKvCache *cache);

/* Total allocated bytes (K + V). Returns 0 on NULL. */
size_t oc_kv_cache_size_bytes(const OcKvCache *cache);

/* Free the cache buffers and zero the struct. Safe on NULL. */
void oc_kv_cache_free(OcKvCache *cache);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_KV_CACHE_H */
