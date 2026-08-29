/* context_cache.h — Persistent KV cache storage for fast session resume. */
#ifndef OXIDIZE_CONTEXT_CACHE_H
#define OXIDIZE_CONTEXT_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Magic + version for the on-disk serialization format. */
#define OC_CONTEXT_CACHE_MAGIC    0x4F434343u   /* 'OCCC' */
#define OC_CONTEXT_CACHE_VERSION  1u

/* Maximum length of a session id string (including NUL terminator). */
#define OC_CONTEXT_CACHE_SESSION_ID_LEN 64

/* Default configuration values used when OcContextCacheConfig fields are
 * zero-initialized via oc_context_cache_config_default(). */
#define OC_CONTEXT_CACHE_DEFAULT_MAX_ENTRIES   128
#define OC_CONTEXT_CACHE_DEFAULT_MAX_SIZE     (1ULL << 30)   /* 1 GiB */
#define OC_CONTEXT_CACHE_DEFAULT_TTL          (3600ULL)      /* 1 hour  */

/* Configuration for a cache instance. All fields are value types. */
typedef struct OcContextCacheConfig {
    const char *cache_dir;       /* directory for disk files; NULL = no disk  */
    size_t      max_entries;     /* max in-memory entries (0 = default)      */
    uint64_t    max_size_bytes;  /* max total in-memory bytes (0 = default)   */
    uint64_t    ttl_seconds;     /* 0 disables TTL expiration                 */
} OcContextCacheConfig;

/* A single cache entry. `data` is owned by the entry (malloc'd) and is the
 * raw KV cache bytes. `session_id` is NUL-terminated and stored inline. */
typedef struct OcContextCacheEntry {
    char     session_id[OC_CONTEXT_CACHE_SESSION_ID_LEN];
    uint64_t model_hash;         /* hash of model identity (file size + tensor count) */
    uint64_t n_tokens;           /* number of tokens the KV cache covers           */
    uint32_t n_layers;            /* depth of the model                              */
    uint32_t n_head_kv;           /* KV heads                                        */
    uint32_t head_dim;            /* per-head dimension                             */
    uint64_t created_at;          /* epoch seconds when stored                       */
    uint64_t last_accessed;       /* epoch seconds of last load/store                */
    uint64_t size_bytes;          /* size of `data` in bytes                         */
    uint8_t *data;                /* raw KV cache bytes (owned)                      */
} OcContextCacheEntry;

/* Cache statistics. */
typedef struct OcContextCacheStats {
    size_t   n_entries;       /* current entry count                       */
    uint64_t total_bytes;     /* sum of entry size_bytes                   */
    uint64_t hits;             /* successful loads                          */
    uint64_t misses;           /* failed loads                              */
    uint64_t evictions;        /* entries removed by LRU or TTL            */
    double   hit_rate;         /* hits / (hits + misses); 0.0 if no lookups */
} OcContextCacheStats;

/* Opaque cache handle. Thread-safe via internal mutex. */
typedef struct OcContextCache OcContextCache;

/* Produce a default config. `cache_dir` remains NULL (caller assigns). */
OcContextCacheConfig oc_context_cache_config_default(void);

/* Create a new cache. `cfg.cache_dir` is copied (malloc'd) if non-NULL; the
 * directory is created (recursively, 0700) if it does not exist. Returns NULL
 * on OOM or if directory creation fails. */
OcContextCache *oc_context_cache_init(const OcContextCacheConfig *cfg);

/* Free the cache and all entries (memory + on-disk files are NOT removed;
 * use oc_context_cache_clear first if you want to wipe the disk too). Safe
 * on NULL. */
void oc_context_cache_free(OcContextCache *cc);

/* Compute a model hash from a GGUF file size + tensor count. This is a cheap */
uint64_t oc_context_cache_model_hash(uint64_t file_size, uint32_t tensor_count);

/* Generate a session id from a prompt + model hash. Writes a NUL-terminated */
OcError oc_context_cache_session_id(const char *prompt, uint64_t model_hash,
                                    char *out, size_t out_len);

/* Store a KV cache snapshot for `session_id`. */
OcError oc_context_cache_store(OcContextCache *cc, const char *session_id,
                               uint64_t model_hash, uint64_t n_tokens,
                               uint32_t n_layers, uint32_t n_head_kv,
                               uint32_t head_dim, uint8_t *data,
                               uint64_t size_bytes);

/* Load a KV cache snapshot for `session_id`. */
OcError oc_context_cache_load(OcContextCache *cc, const char *session_id,
                              uint64_t model_hash, OcContextCacheEntry *out,
                              bool *found);

/* Evict the least-recently-used entry. Returns true if an entry was evicted,
 * false if the cache was empty. */
bool oc_context_cache_evict(OcContextCache *cc);

/* Remove all entries (memory + on-disk files in cache_dir). Returns OC_OK on
 * success. */
OcError oc_context_cache_clear(OcContextCache *cc);

/* Fill `*out` with current statistics. Safe on NULL (zeroes `*out`). */
void oc_context_cache_get_stats(const OcContextCache *cc,
                                OcContextCacheStats *out);

/* Format statistics as a JSON string into `buf` (up to `cap-1` chars,
 * NUL-terminated). Returns the number of bytes written (excluding NUL). If
 * `buf` is NULL or cap==0, returns the length that would have been written. */
size_t oc_context_cache_format_stats(const OcContextCache *cc, char *buf,
                                     size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_CONTEXT_CACHE_H */
