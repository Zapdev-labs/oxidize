/*
 * kv_cache.c — simple per-layer KV cache implementation.
 *
 * See include/oxidize/kv_cache.h for design notes.
 */
#include "oxidize/kv_cache.h"

#include <stdlib.h>
#include <string.h>

/* ─── Helpers ──────────────────────────────────────────────────────────── */

/* Per-layer element count for K or V:
 *   max_seq_len * n_heads * head_dim
 */
static size_t kv_layer_count(const OcKvCacheConfig *cfg)
{
    return (size_t)cfg->max_seq_len *
           (size_t)cfg->n_heads *
           (size_t)cfg->head_dim;
}

/* Total element count for K or V across all layers. */
static size_t kv_total_count(const OcKvCacheConfig *cfg)
{
    return (size_t)cfg->n_layers * kv_layer_count(cfg);
}

/* ─── Config ───────────────────────────────────────────────────────────── */

void oc_kv_cache_config_init(OcKvCacheConfig *cfg)
{
    if (!cfg) return;
    cfg->n_layers    = OC_KV_CACHE_DEFAULT_N_LAYERS;
    cfg->n_heads     = OC_KV_CACHE_DEFAULT_N_HEADS;
    cfg->head_dim    = OC_KV_CACHE_DEFAULT_HEAD_DIM;
    cfg->max_seq_len = OC_KV_CACHE_DEFAULT_MAX_SEQ_LEN;
    cfg->dtype       = OC_KV_CACHE_DTYPE_F32;
}

/* ─── Init / Free ─────────────────────────────────────────────────────── */

OcError oc_kv_cache_init(OcKvCache *cache, const OcKvCacheConfig *cfg)
{
    if (!cache || !cfg) return OC_ERR_INVALID_ARG;
    if (cfg->n_layers == 0 || cfg->n_heads == 0 ||
        cfg->head_dim == 0 || cfg->max_seq_len == 0) {
        return OC_ERR_INVALID_ARG;
    }
    if (cfg->dtype != OC_KV_CACHE_DTYPE_F32 &&
        cfg->dtype != OC_KV_CACHE_DTYPE_F16) {
        return OC_ERR_INVALID_ARG;
    }

    memset(cache, 0, sizeof(*cache));
    cache->config   = *cfg;
    cache->capacity = cfg->max_seq_len;
    cache->n_tokens = 0;

    size_t count = kv_total_count(cfg);
    cache->k_data = malloc(count * sizeof(float));
    if (!cache->k_data) {
        return OC_ERR_OOM;
    }
    cache->v_data = malloc(count * sizeof(float));
    if (!cache->v_data) {
        free(cache->k_data);
        cache->k_data = NULL;
        return OC_ERR_OOM;
    }
    memset(cache->k_data, 0, count * sizeof(float));
    memset(cache->v_data, 0, count * sizeof(float));
    return OC_OK;
}

void oc_kv_cache_free(OcKvCache *cache)
{
    if (!cache) return;
    free(cache->k_data);
    free(cache->v_data);
    cache->k_data   = NULL;
    cache->v_data   = NULL;
    cache->n_tokens = 0;
    cache->capacity = 0;
}

/* ─── Append / Get ────────────────────────────────────────────────────── */

OcError oc_kv_cache_append(OcKvCache *cache, uint32_t layer,
                            const float *k, const float *v, uint32_t n)
{
    if (!cache || !k || !v) return OC_ERR_INVALID_ARG;
    if (layer >= cache->config.n_layers) return OC_ERR_INVALID_ARG;
    if (n == 0) return OC_OK;
    if (cache->n_tokens + n > cache->capacity) return OC_ERR_INVALID_ARG;

    const OcKvCacheConfig *cfg = &cache->config;
    size_t per_layer    = kv_layer_count(cfg);
    size_t row_size     = (size_t)cfg->n_heads * (size_t)cfg->head_dim;
    size_t layer_offset = (size_t)layer * per_layer;

    /* Write position: for layer 0, the current n_tokens (before advance).
     * For other layers, n_tokens has already been advanced by the layer 0
     * call for this step, so back up by n to write at the same position.
     * This matches the typical inference loop where layer 0 is written
     * first, then layers 1..N-1 for the same token(s). */
    size_t write_token;
    if (layer == 0) {
        write_token = cache->n_tokens;
    } else {
        write_token = (cache->n_tokens >= n) ? (cache->n_tokens - n) : 0;
    }
    size_t token_offset = write_token * row_size;

    memcpy(cache->k_data + layer_offset + token_offset, k,
           (size_t)n * row_size * sizeof(float));
    memcpy(cache->v_data + layer_offset + token_offset, v,
           (size_t)n * row_size * sizeof(float));

    /* Advance the global token count once per step (on the layer 0 call). */
    if (layer == 0) {
        cache->n_tokens += n;
    }
    return OC_OK;
}

OcError oc_kv_cache_get(const OcKvCache *cache, uint32_t layer,
                         uint32_t pos, const float **k, const float **v)
{
    if (!cache || !k || !v) return OC_ERR_INVALID_ARG;
    if (layer >= cache->config.n_layers) return OC_ERR_INVALID_ARG;
    if (pos >= cache->n_tokens) return OC_ERR_INVALID_ARG;

    const OcKvCacheConfig *cfg = &cache->config;
    size_t per_layer    = kv_layer_count(cfg);
    size_t row_size     = (size_t)cfg->n_heads * (size_t)cfg->head_dim;
    size_t layer_offset = (size_t)layer * per_layer;
    size_t token_offset = (size_t)pos * row_size;

    *k = cache->k_data + layer_offset + token_offset;
    *v = cache->v_data + layer_offset + token_offset;
    return OC_OK;
}

/* ─── Clear / Truncate ────────────────────────────────────────────────── */

void oc_kv_cache_clear(OcKvCache *cache)
{
    if (!cache) return;
    cache->n_tokens = 0;
}

OcError oc_kv_cache_truncate(OcKvCache *cache, uint32_t n)
{
    if (!cache) return OC_ERR_INVALID_ARG;
    /* n must be within capacity. */
    if (n > cache->capacity) return OC_ERR_INVALID_ARG;
    /* If n >= n_tokens, it's a no-op (truncating to >= current is OK). */
    if (n >= cache->n_tokens) return OC_OK;
    /* n < n_tokens: actually truncate. */
    cache->n_tokens = n;
    return OC_OK;
}

/* ─── Accessors ────────────────────────────────────────────────────────── */

uint32_t oc_kv_cache_n_tokens(const OcKvCache *cache)
{
    if (!cache) return 0;
    return cache->n_tokens;
}

uint32_t oc_kv_cache_capacity(const OcKvCache *cache)
{
    if (!cache) return 0;
    return cache->capacity;
}

size_t oc_kv_cache_size_bytes(const OcKvCache *cache)
{
    if (!cache || !cache->k_data) return 0;
    /* K + V, both sized kv_total_count(config) floats. */
    size_t count = kv_total_count(&cache->config);
    return 2u * count * sizeof(float);
}
