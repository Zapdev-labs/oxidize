/*
 * prefix_cache.h — LRU prefix cache for KV cache reuse.
 *
 * Caches KV cache snapshots keyed by token sequence hashes. When multiple
 * requests share a common prefix (e.g., system prompt), the KV cache for
 * that prefix can be reused instead of recomputing.
 *
 * Uses FNV-1a cumulative hashing: the hash of tokens[0..n] can be computed
 * incrementally from the hash of tokens[0..n-1].
 */
#ifndef OXIDIZE_PREFIX_CACHE_H
#define OXIDIZE_PREFIX_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_PREFIX_CACHE_MAX_ENTRIES 64
#define OC_PREFIX_CACHE_MAX_TOKENS 32768

/* FNV-1a hash of a token sequence. */
uint64_t oc_prefix_hash_tokens(const uint32_t *tokens, size_t n);

/* Incremental FNV-1a: continue hashing from a previous hash. */
uint64_t oc_prefix_hash_continue(uint64_t prev_hash, uint32_t token);

/* A cached KV snapshot for a prefix. */
typedef struct OcCachedPrefix {
    uint64_t hash;            /* FNV-1a hash of token sequence           */
    size_t   n_tokens;        /* number of tokens in the prefix          */
    uint32_t *tokens;
    void    *kv_data;         /* opaque KV cache snapshot (caller-owned format) */
    size_t   kv_size;         /* size of kv_data in bytes                */
    uint64_t last_used;       /* LRU timestamp (higher = more recent)    */
    bool     active;          /* is this slot in use?                    */
} OcCachedPrefix;

typedef struct OcPrefixCache {
    OcCachedPrefix entries[OC_PREFIX_CACHE_MAX_ENTRIES];
    size_t n_entries;
    uint64_t clock;           /* monotonic counter for LRU                */
    size_t max_tokens;        /* max tokens per entry to cache            */
    size_t n_hits;
    size_t n_misses;
} OcPrefixCache;

/* Initialize the prefix cache. */
void oc_prefix_cache_init(OcPrefixCache *c, size_t max_tokens);

/* Look up a cached prefix by token hash. Returns the cached entry if found
 * (updating LRU), or NULL if not found. */
const OcCachedPrefix *oc_prefix_cache_lookup(OcPrefixCache *c, uint64_t hash,
                                              const uint32_t *tokens,
                                              size_t n_tokens);

/* Store a KV snapshot for the given token hash. Evicts the LRU entry if full.
 * Takes ownership of `kv_data` (freed on eviction). */
OcError oc_prefix_cache_store(OcPrefixCache *c, uint64_t hash,
                               const uint32_t *tokens, size_t n_tokens,
                               void *kv_data, size_t kv_size);

/* Evict entries that haven't been used since `threshold`. */
size_t oc_prefix_cache_evict(OcPrefixCache *c, uint64_t threshold);

/* Clear all entries. */
void oc_prefix_cache_clear(OcPrefixCache *c);

/* Get cache statistics. */
typedef struct OcPrefixCacheStats {
    size_t n_entries;
    size_t n_hits;
    size_t n_misses;
    size_t total_kv_bytes;
} OcPrefixCacheStats;

void oc_prefix_cache_stats(const OcPrefixCache *c, OcPrefixCacheStats *out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_PREFIX_CACHE_H */
