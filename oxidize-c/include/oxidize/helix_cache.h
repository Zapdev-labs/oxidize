/*
 * helix_cache.h — polar 4-bit keys + Hadamard 3-bit values KV cache.
 *
 * C port of oxidize-cpp HelixCache. Keys are stored PRE-RoPE; RoPE is
 * applied incrementally to the query at decode. Scalar path only.
 */
#ifndef OXIDIZE_HELIX_CACHE_H
#define OXIDIZE_HELIX_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_HELIX_PAGE_COLD = 0,
    OC_HELIX_PAGE_HOT  = 1,
} OcHelixPageTier;

typedef struct OcHelixCacheConfig {
    size_t  page_size;
    size_t  head_dim;
    size_t  rope_dim; /* 0 = head_dim; only this many coords get incremental RoPE */
    uint8_t key_radius_bits;
    uint8_t key_phase_bits;
    uint8_t value_bits;
    float   inactive_threshold;
    float   promotion_epsilon;
    uint32_t promotion_budget;
} OcHelixCacheConfig;

typedef struct OcHelixCacheStats {
    size_t cold_pages;
    size_t hot_pages;
    size_t token_count;
    size_t key_bytes;
    size_t value_bytes;
    size_t hot_bytes;
    size_t metadata_bytes;
    size_t key_metadata_bytes;
    size_t value_metadata_bytes;
    size_t page_metadata_bytes;
    size_t f32_baseline_bytes;
    float  key_bits_per_coord;
    float  value_bits_per_coord;
    float  total_bits_per_coord;
} OcHelixCacheStats;

typedef struct OcHelixColdPageView {
    size_t         layer;
    size_t         kv_head;
    size_t         tokens;
    const size_t  *positions;
    const uint8_t *key_codes;
    const uint8_t *active_mask;
    const float   *mu_phi;
    const float   *log_rho_min;
    const float   *log_rho_step;
    const uint8_t *value_codes;
    const float   *value_scales;
} OcHelixColdPageView;

typedef struct OcHelixPage OcHelixPage;

typedef struct OcHelixCache {
    OcHelixCacheConfig config;
    OcHelixPage *pages;
    size_t n_pages;
    size_t cap_pages;
} OcHelixCache;

void oc_helix_cache_config_init(OcHelixCacheConfig *cfg);

OcError oc_helix_cache_init(OcHelixCache *cache, const OcHelixCacheConfig *cfg);
void oc_helix_cache_free(OcHelixCache *cache);

OcError oc_helix_cache_store_cold_page(OcHelixCache *cache,
                                       size_t layer, size_t kv_head,
                                       size_t page_id,
                                       const float *pre_rope_keys,
                                       const float *values,
                                       const size_t *positions,
                                       size_t n_tokens);
OcError oc_helix_cache_store_hot_page(OcHelixCache *cache,
                                      size_t layer, size_t kv_head,
                                      size_t page_id,
                                      const float *pre_rope_keys,
                                      const float *values,
                                      const size_t *positions,
                                      size_t n_tokens);

/* Append tokens onto a per-(layer, kv_head) hot page. A full page is frozen
 * to cold so per-page polar metadata is amortized. n_tokens may span pages. */
OcError oc_helix_cache_append(OcHelixCache *cache,
                              size_t layer, size_t kv_head,
                              const float *pre_rope_keys,
                              const float *values,
                              const size_t *positions,
                              size_t n_tokens);

OcError oc_helix_cache_logits(const OcHelixCache *cache,
                              size_t layer, size_t kv_head,
                              const float *query_pre_rope, size_t query_n,
                              size_t query_position, float rope_theta,
                              float *out, size_t out_cap, size_t *n_out);
OcError oc_helix_cache_attention(OcHelixCache *cache,
                                 size_t layer, size_t kv_head,
                                 const float *query_pre_rope, size_t query_n,
                                 size_t query_position, float rope_theta,
                                 float *out);

size_t oc_helix_cache_n_logits(const OcHelixCache *cache,
                               size_t layer, size_t kv_head);
size_t oc_helix_cache_page_count(const OcHelixCache *cache);
bool oc_helix_cache_cold_page_view(const OcHelixCache *cache, size_t index,
                                   OcHelixColdPageView *view);

OcError oc_helix_cache_bump_uncertainty(OcHelixCache *cache,
                                        size_t layer, size_t kv_head,
                                        size_t page_id,
                                        float interval_overlap);
OcError oc_helix_cache_should_promote(const OcHelixCache *cache,
                                      size_t layer, size_t kv_head,
                                      size_t page_id, bool *out);

OcError oc_helix_cache_stats(const OcHelixCache *cache, OcHelixCacheStats *out);
float oc_helix_cache_compression_ratio(const OcHelixCacheStats *st);

OcError oc_helix_cache_rewind(OcHelixCache *cache, size_t n_keep);
void oc_helix_cache_clear(OcHelixCache *cache);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_HELIX_CACHE_H */
