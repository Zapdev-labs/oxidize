/*
 * rotorquant_cache.h — page-oriented RotorQuant KV cache (fused int4 decode).
 *
 * Distinct from rotorquant.h (Planar/Iso Lloyd-Max of ONE vector). This is
 * the C port of oxidize-cpp RotorQuantCache: blockwise 3D rotors plus
 * per-block int4 scalar quant. Decode rotates the query once (rotation is
 * orthogonal, so q·k == Rq·Rk), int4-dots the cache, and unrotates the
 * value accumulator once at the end.
 *
 * Keys passed to store_page are POST-RoPE.
 */
#ifndef OXIDIZE_ROTORQUANT_CACHE_H
#define OXIDIZE_ROTORQUANT_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_ROTORQUANT_CACHE_DEFAULT_BLOCK 32u
#define OC_ROTORQUANT_CACHE_DEFAULT_SEED  0x5EED0Fu

typedef struct OcRotorQuantCacheConfig {
    size_t   head_dim;
    size_t   block_size;
    uint32_t seed;
} OcRotorQuantCacheConfig;

typedef struct OcRotorQuantCacheStats {
    size_t token_count;
    size_t key_bytes;
    size_t value_bytes;
    size_t metadata_bytes;
    size_t f32_baseline_bytes;
    float  total_bits_per_coord;
} OcRotorQuantCacheStats;

/* Pointers remain valid until the matching page is dropped (including
 * rewind of pages whose first_position >= n_keep), replaced by
 * store_page() for the same (layer, kv_head, first_position), or the
 * cache is freed. Rewind of a straddling page truncates `tokens` without
 * freeing buffers; a held view's snapshotted `tokens` is then stale.
 * Re-fetch the view after rewind or store_page() before iterating. */
typedef struct OcRotorQuantPageView {
    size_t         layer;
    size_t         kv_head;
    size_t         tokens;
    size_t         first_position;
    const uint8_t *key_codes;
    const float   *key_scales;
    const uint8_t *value_codes;
    const float   *value_scales;
} OcRotorQuantPageView;

typedef struct OcRotorQuantPage OcRotorQuantPage;

typedef struct OcRotorQuantCache {
    OcRotorQuantCacheConfig config;
    float *rotors;          /* n_groups * 9, row-major 3x3 */
    size_t n_groups;
    OcRotorQuantPage *pages;
    size_t n_pages;
    size_t cap_pages;
} OcRotorQuantCache;

void oc_rotorquant_cache_config_init(OcRotorQuantCacheConfig *cfg);

OcError oc_rotorquant_cache_init(OcRotorQuantCache *cache,
                                 const OcRotorQuantCacheConfig *cfg);
void oc_rotorquant_cache_free(OcRotorQuantCache *cache);

/* `first_position` is the upsert key and the position of token 0. It must
 * be distinct per page for the same (layer, kv_head). Token t lives at
 * first_position + t. Replacing an existing page frees the previous
 * code/scale buffers and invalidates any OcRotorQuantPageView of it. */
OcError oc_rotorquant_cache_store_page(OcRotorQuantCache *cache,
                                       size_t layer, size_t kv_head,
                                       const float *keys, const float *values,
                                       size_t n_tokens, size_t first_position);

/* Token t of a page is at first_position + t. Keys with position >
 * query_position are dropped (causal). Pass SIZE_MAX to include every
 * stored token. `out_cap` is checked against the unfiltered token count
 * (every stored token for this head), not the post-mask visible count. */
OcError oc_rotorquant_cache_logits(const OcRotorQuantCache *cache,
                                   size_t layer, size_t kv_head,
                                   const float *query, size_t query_n,
                                   size_t query_position,
                                   float *out, size_t out_cap, size_t *n_out);

/* `out` must hold config.head_dim floats. */
OcError oc_rotorquant_cache_attention(const OcRotorQuantCache *cache,
                                      size_t layer, size_t kv_head,
                                      const float *query, size_t query_n,
                                      size_t query_position,
                                      float *out);

OcError oc_rotorquant_cache_rotate(const OcRotorQuantCache *cache,
                                   const float *v, size_t n, float *out);
OcError oc_rotorquant_cache_unrotate(const OcRotorQuantCache *cache,
                                     const float *v, size_t n, float *out);

size_t oc_rotorquant_cache_n_logits(const OcRotorQuantCache *cache,
                                    size_t layer, size_t kv_head);
size_t oc_rotorquant_cache_page_count(const OcRotorQuantCache *cache);
bool oc_rotorquant_cache_page_view(const OcRotorQuantCache *cache, size_t index,
                                   OcRotorQuantPageView *view);

OcError oc_rotorquant_cache_stats(const OcRotorQuantCache *cache,
                                  OcRotorQuantCacheStats *out);
float oc_rotorquant_cache_compression_ratio(const OcRotorQuantCacheStats *st);

/* Drop pages whose first_position >= n_keep. Pages that straddle n_keep
 * are truncated so tokens at first_position + t >= n_keep are dropped
 * without reallocating: a view of that page keeps valid pointers, but
 * its snapshotted `tokens` is stale. Pages wholly past n_keep are freed
 * and any held view of them dangles. Call page_view() again after rewind
 * before iterating. Unused tail rows on a truncated page are reclaimed
 * on the next store_page of this slot, or when the page is cleared/freed. */
OcError oc_rotorquant_cache_rewind(OcRotorQuantCache *cache, size_t n_keep);
void oc_rotorquant_cache_clear(OcRotorQuantCache *cache);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ROTORQUANT_CACHE_H */
