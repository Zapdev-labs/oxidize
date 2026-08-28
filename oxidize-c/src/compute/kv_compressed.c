/*
 * kv_compressed.c — facade over RotorQuantCache / HelixCache.
 *
 * P1: attention validates query length before RoPE so a short buffer cannot
 * walk off the end of query_pre_rope (the C++ facade indexed by head_dim
 * unconditionally).
 */
#include "oxidize/kv_compressed.h"

#include "oxidize/activation.h"

#include <stdlib.h>
#include <string.h>

OcError oc_compressed_kv_init(OcCompressedKvCache *cache, size_t head_dim,
                              OcKvScheme scheme, size_t page_size,
                              float rope_theta)
{
    if (!cache || head_dim == 0) return OC_ERR_INVALID_ARG;
    if (page_size == 0) page_size = 64;
    if (rope_theta <= 0.0f) rope_theta = 10000.0f;
    memset(cache, 0, sizeof(*cache));
    cache->scheme = scheme;
    cache->rope_layout = OC_KV_ROPE_INTERLEAVED;
    cache->head_dim = head_dim;
    cache->page_size = page_size;
    cache->rope_theta = rope_theta;
    if (scheme == OC_KV_SCHEME_HELIX) {
        OcHelixCacheConfig cfg;
        oc_helix_cache_config_init(&cfg);
        cfg.head_dim = head_dim;
        cfg.page_size = page_size;
        if (oc_helix_cache_init(&cache->helix, &cfg) != OC_OK)
            return OC_ERR_INVALID_ARG;
        cache->has_helix = 1;
    } else {
        OcRotorQuantCacheConfig cfg;
        oc_rotorquant_cache_config_init(&cfg);
        cfg.head_dim = head_dim;
        if (oc_rotorquant_cache_init(&cache->rotor, &cfg) != OC_OK)
            return OC_ERR_INVALID_ARG;
        cache->has_rotor = 1;
    }
    return OC_OK;
}

void oc_compressed_kv_free(OcCompressedKvCache *cache)
{
    if (!cache) return;
    if (cache->has_helix) oc_helix_cache_free(&cache->helix);
    if (cache->has_rotor) oc_rotorquant_cache_free(&cache->rotor);
    memset(cache, 0, sizeof(*cache));
}

void oc_compressed_kv_set_rope_layout(OcCompressedKvCache *cache,
                                      OcKvRopeLayout layout)
{
    if (cache) cache->rope_layout = layout;
}

OcKvScheme oc_compressed_kv_scheme(const OcCompressedKvCache *cache)
{
    return cache ? cache->scheme : OC_KV_SCHEME_ROTOR;
}

static OcError apply_rope_row(const OcCompressedKvCache *cache, const float *row,
                              size_t position, float *out)
{
    if (cache->rope_layout == OC_KV_ROPE_SPLIT_HALVES) {
        oc_apply_rope_f32(row, out, cache->head_dim, cache->head_dim,
                          (int64_t)position, cache->rope_theta);
    } else {
        oc_apply_rope_norm_f32(row, out, cache->head_dim, cache->head_dim,
                               (int64_t)position, cache->rope_theta);
    }
    return OC_OK;
}

OcError oc_compressed_kv_store_page(OcCompressedKvCache *cache,
                                    size_t layer, size_t kv_head,
                                    const float *pre_rope_keys,
                                    const float *values,
                                    const size_t *positions,
                                    size_t n_tokens)
{
    size_t t;
    float *roped, *row;
    OcError e;
    if (!cache || !pre_rope_keys || !values || !positions || n_tokens == 0)
        return OC_ERR_INVALID_ARG;
    if (cache->has_helix) {
        return oc_helix_cache_store_cold_page(&cache->helix, layer, kv_head,
                                              positions[0],
                                              pre_rope_keys, values, positions,
                                              n_tokens);
    }
    roped = (float *)malloc(n_tokens * cache->head_dim * sizeof(float));
    row = (float *)malloc(cache->head_dim * sizeof(float));
    if (!roped || !row) {
        free(roped);
        free(row);
        return OC_ERR_OOM;
    }
    for (t = 0; t < n_tokens; t++) {
        memcpy(row, pre_rope_keys + t * cache->head_dim,
               cache->head_dim * sizeof(float));
        apply_rope_row(cache, row, positions[t],
                       roped + t * cache->head_dim);
    }
    e = oc_rotorquant_cache_store_page(&cache->rotor, layer, kv_head, roped,
                                       values, n_tokens, positions[0]);
    free(roped);
    free(row);
    cache->next_page_id += 1;
    return e;
}

OcError oc_compressed_kv_attention(OcCompressedKvCache *cache,
                                   size_t layer, size_t kv_head,
                                   const float *query_pre_rope, size_t query_n,
                                   size_t query_position, float *out)
{
    float *rq;
    OcError e;
    if (!cache || !query_pre_rope || !out) return OC_ERR_INVALID_ARG;
    if (query_n != cache->head_dim) return OC_ERR_INVALID_ARG;
    if (cache->has_helix) {
        return oc_helix_cache_attention(&cache->helix, layer, kv_head,
                                        query_pre_rope, query_n,
                                        query_position,
                                        cache->rope_theta, out);
    }
    rq = (float *)malloc(cache->head_dim * sizeof(float));
    if (!rq) return OC_ERR_OOM;
    apply_rope_row(cache, query_pre_rope, query_position, rq);
    e = oc_rotorquant_cache_attention(&cache->rotor, layer, kv_head, rq,
                                      query_n, query_position, out);
    free(rq);
    return e;
}

const OcHelixCache *oc_compressed_kv_helix(const OcCompressedKvCache *cache)
{
    if (!cache || !cache->has_helix) return NULL;
    return &cache->helix;
}

const OcRotorQuantCache *oc_compressed_kv_rotor(const OcCompressedKvCache *cache)
{
    if (!cache || !cache->has_rotor) return NULL;
    return &cache->rotor;
}

float oc_compressed_kv_compression_ratio(const OcCompressedKvCache *cache)
{
    if (!cache) return 1.0f;
    if (cache->has_helix) {
        OcHelixCacheStats st;
        if (oc_helix_cache_stats(&cache->helix, &st) != OC_OK) return 1.0f;
        return oc_helix_cache_compression_ratio(&st);
    }
    {
        OcRotorQuantCacheStats st;
        if (oc_rotorquant_cache_stats(&cache->rotor, &st) != OC_OK) return 1.0f;
        return oc_rotorquant_cache_compression_ratio(&st);
    }
}

OcError oc_compressed_kv_rewind(OcCompressedKvCache *cache, size_t n_keep)
{
    OcError e;
    if (!cache) return OC_ERR_INVALID_ARG;
    if (cache->has_helix) {
        e = oc_helix_cache_rewind(&cache->helix, n_keep);
        if (e != OC_OK) return e;
    }
    if (cache->has_rotor) {
        e = oc_rotorquant_cache_rewind(&cache->rotor, n_keep);
        if (e != OC_OK) return e;
    }
    cache->next_page_id = n_keep;
    return OC_OK;
}

void oc_compressed_kv_clear(OcCompressedKvCache *cache)
{
    if (!cache) return;
    if (cache->has_helix) oc_helix_cache_clear(&cache->helix);
    if (cache->has_rotor) oc_rotorquant_cache_clear(&cache->rotor);
    cache->next_page_id = 0;
}
