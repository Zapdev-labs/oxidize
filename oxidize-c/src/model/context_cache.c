/* context_cache.c — Persistent KV cache storage for fast session resume. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "oxidize/context_cache.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

/* Constants                                                           */

#define OC_CC_MIN_CAP    16u     /* minimum hash table capacity          */
#define OC_CC_MAX_CAP    (1u << 20)   /* hard cap to avoid pathological allocs */
#define OC_CC_FNV_OFFSET  0xcbf29ce484222325ULL
#define OC_CC_FNV_PRIME   0x100000001b3ULL

/* Internal types                                                      */

/* A slot in the open-addressing table. `used' distinguishes occupied from
 * empty; tombstones are tracked by `deleted' (set when an entry is removed
 * and the slot becomes available for reuse). */
typedef struct OcCcSlot {
    OcContextCacheEntry *entry;   /* NULL if empty                       */
    bool                 deleted; /* true if a tombstone                 */
} OcCcSlot;

struct OcContextCache {
    OcContextCacheConfig  cfg;
    OcCcSlot             *slots;
    size_t                cap;       /* buckets (power of two)          */
    size_t                n_entries; /* live entries                     */
    uint64_t              total_bytes;
    uint64_t              hits;
    uint64_t              misses;
    uint64_t              evictions;
    pthread_mutex_t       mu;
};

/* Small helpers                                                       */

static uint64_t oc_cc_now(void) {
    return (uint64_t)time(NULL);
}

static uint64_t oc_cc_fnv1a(const char *s) {
    uint64_t h = OC_CC_FNV_OFFSET;
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
        h ^= (uint64_t)*p++;
        h *= OC_CC_FNV_PRIME;
    }
    return h;
}

static size_t oc_cc_round_up_pow2(size_t v) {
    if (v < OC_CC_MIN_CAP) {
        return OC_CC_MIN_CAP;
    }
    if (v >= OC_CC_MAX_CAP) {
        return OC_CC_MAX_CAP;
    }
    /* Classic bit-twiddle: round up to next power of two. */
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
#if SIZE_MAX > 0xFFFFFFFFu
    v |= v >> 32;
#endif
    v++;
    return v;
}

/* Copy `src` into a freshly malloc'd buffer. Returns NULL on OOM or if
 * `size == 0` with `src == NULL`. */
static uint8_t *oc_cc_dup_data(const uint8_t *src, uint64_t size) {
    if (size == 0) {
        /* Allocate a 1-byte sentinel so free() is always valid. */
        uint8_t *p = (uint8_t *)malloc(1);
        return p;
    }
    if (src == NULL) {
        return NULL;
    }
    uint8_t *p = (uint8_t *)malloc((size_t)size);
    if (p == NULL) {
        return NULL;
    }
    memcpy(p, src, (size_t)size);
    return p;
}

static void oc_cc_entry_free(OcContextCacheEntry *e) {
    if (e == NULL) {
        return;
    }
    free(e->data);
    free(e);
}

/* Recursively create a directory (like mkdir -p) with mode 0700. Returns 0
 * on success or -1 on failure (errno set). */
static int oc_cc_mkdir_p(const char *path) {
    if (path == NULL || path[0] == '\0') {
        return -1;
    }
    char buf[4096];
    size_t len = strlen(path);
    if (len >= sizeof(buf)) {
        return -1;
    }
    memcpy(buf, path, len + 1);
    /* Strip trailing slashes. */
    while (len > 1 && buf[len - 1] == '/') {
        buf[--len] = '\0';
    }
    for (size_t i = 1; i < len; i++) {
        if (buf[i] == '/') {
            buf[i] = '\0';
            if (mkdir(buf, 0700) != 0 && errno != EEXIST) {
                return -1;
            }
            buf[i] = '/';
        }
    }
    if (mkdir(buf, 0700) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

/* Build the on-disk filename for a session id. Returns 0 on success, -1 if
 * the path would overflow `out`. */
static int oc_cc_path_for(const char *cache_dir, const char *session_id,
                          char *out, size_t out_sz) {
    if (cache_dir == NULL || session_id == NULL || out == NULL) {
        return -1;
    }
    int n = snprintf(out, out_sz, "%s/%s.bin", cache_dir, session_id);
    if (n < 0 || (size_t)n >= out_sz) {
        return -1;
    }
    return 0;
}

/* Serialization                                                       */

/* All multi-byte fields are written in host byte order (the format is
 * intended for single-host use; cross-host portability would require
 * explicit little-endian I/O). Layout matches context_cache.h. */

static const uint32_t OC_CC_RESERVED_ZERO = 0u;

static OcError oc_cc_write_entry(const OcContextCacheConfig *cfg,
                                 const OcContextCacheEntry *e) {
    if (cfg->cache_dir == NULL) {
        return OC_OK;   /* memory-only mode */
    }
    char path[4096];
    if (oc_cc_path_for(cfg->cache_dir, e->session_id, path, sizeof(path)) != 0) {
        return OC_ERR_INVALID_ARG;
    }
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        return OC_ERR_IO;
    }
    uint32_t magic   = OC_CONTEXT_CACHE_MAGIC;
    uint32_t version = OC_CONTEXT_CACHE_VERSION;
    /* Use a local buffer for the session id to write exactly the fixed
     * width, NUL-padded. */
    char sid_buf[OC_CONTEXT_CACHE_SESSION_ID_LEN];
    memset(sid_buf, 0, sizeof(sid_buf));
    size_t sid_len = strlen(e->session_id);
    if (sid_len >= sizeof(sid_buf)) {
        sid_len = sizeof(sid_buf) - 1;
    }
    memcpy(sid_buf, e->session_id, sid_len);
    uint32_t reserved = OC_CC_RESERVED_ZERO;

    int ok = 1;
    ok &= (fwrite(&magic,         sizeof(magic),   1, f) == 1);
    ok &= (fwrite(&version,       sizeof(version), 1, f) == 1);
    ok &= (fwrite(sid_buf,        sizeof(sid_buf), 1, f) == 1);
    ok &= (fwrite(&e->model_hash, sizeof(e->model_hash), 1, f) == 1);
    ok &= (fwrite(&e->n_tokens,   sizeof(e->n_tokens),   1, f) == 1);
    ok &= (fwrite(&e->n_layers,   sizeof(e->n_layers),   1, f) == 1);
    ok &= (fwrite(&e->n_head_kv,  sizeof(e->n_head_kv),  1, f) == 1);
    ok &= (fwrite(&e->head_dim,   sizeof(e->head_dim),   1, f) == 1);
    ok &= (fwrite(&reserved, sizeof(reserved), 1, f) == 1);
    ok &= (fwrite(&e->created_at,    sizeof(e->created_at),    1, f) == 1);
    ok &= (fwrite(&e->last_accessed, sizeof(e->last_accessed), 1, f) == 1);
    ok &= (fwrite(&e->size_bytes,    sizeof(e->size_bytes),    1, f) == 1);
    if (e->size_bytes > 0 && e->data != NULL) {
        ok &= (fwrite(e->data, 1, (size_t)e->size_bytes, f)
               == (size_t)e->size_bytes);
    }
    if (fclose(f) != 0) {
        ok = 0;
    }
    return ok ? OC_OK : OC_ERR_IO;
}

/* Read an entry from disk into a freshly-allocated OcContextCacheEntry. The
 * caller owns the result (free with oc_cc_entry_free). Returns NULL on any
 * error. */
static OcContextCacheEntry *oc_cc_read_entry(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        return NULL;
    }
    uint32_t magic = 0, version = 0;
    char sid_buf[OC_CONTEXT_CACHE_SESSION_ID_LEN];
    OcContextCacheEntry e;
    memset(&e, 0, sizeof(e));
    uint32_t reserved = 0;

    int ok = 1;
    ok &= (fread(&magic,   sizeof(magic),   1, f) == 1);
    ok &= (fread(&version, sizeof(version), 1, f) == 1);
    ok &= (fread(sid_buf,  sizeof(sid_buf), 1, f) == 1);
    ok &= (fread(&e.model_hash, sizeof(e.model_hash), 1, f) == 1);
    ok &= (fread(&e.n_tokens,   sizeof(e.n_tokens),   1, f) == 1);
    ok &= (fread(&e.n_layers,   sizeof(e.n_layers),   1, f) == 1);
    ok &= (fread(&e.n_head_kv,  sizeof(e.n_head_kv),  1, f) == 1);
    ok &= (fread(&e.head_dim,   sizeof(e.head_dim),   1, f) == 1);
    ok &= (fread(&reserved, sizeof(reserved), 1, f) == 1);
    (void)reserved;
    ok &= (fread(&e.created_at,    sizeof(e.created_at),    1, f) == 1);
    ok &= (fread(&e.last_accessed, sizeof(e.last_accessed), 1, f) == 1);
    ok &= (fread(&e.size_bytes,    sizeof(e.size_bytes),    1, f) == 1);
    if (!ok || magic != OC_CONTEXT_CACHE_MAGIC ||
        version != OC_CONTEXT_CACHE_VERSION) {
        fclose(f);
        return NULL;
    }
    memcpy(e.session_id, sid_buf, sizeof(sid_buf) - 1);
    e.session_id[sizeof(e.session_id) - 1] = '\0';
    if (e.size_bytes > 0) {
        e.data = (uint8_t *)malloc((size_t)e.size_bytes);
        if (e.data == NULL) {
            fclose(f);
            return NULL;
        }
        if (fread(e.data, 1, (size_t)e.size_bytes, f)
            != (size_t)e.size_bytes) {
            free(e.data);
            fclose(f);
            return NULL;
        }
    } else {
        e.data = NULL;
    }
    if (fclose(f) != 0) {
        free(e.data);
        return NULL;
    }
    OcContextCacheEntry *out = (OcContextCacheEntry *)malloc(sizeof(*out));
    if (out == NULL) {
        free(e.data);
        return NULL;
    }
    *out = e;
    return out;
}

/* Delete the on-disk file for a session id (silent no-op if absent). */
static void oc_cc_unlink_entry(const OcContextCacheConfig *cfg,
                               const char *session_id) {
    if (cfg->cache_dir == NULL || session_id == NULL) {
        return;
    }
    char path[4096];
    if (oc_cc_path_for(cfg->cache_dir, session_id, path, sizeof(path)) != 0) {
        return;
    }
    (void)remove(path);
}

/* Hash table internals                                                */

/* Find the slot index for `session_id`. If `find_tombstone` is true and the
 * key is absent, returns the index of the first tombstone encountered (for
 * insertion reuse). Otherwise returns -1 on miss. */
static ssize_t oc_cc_find_slot(OcContextCache *cc, const char *session_id,
                               bool find_tombstone) {
    if (cc->cap == 0 || session_id == NULL) {
        return -1;
    }
    uint64_t h = oc_cc_fnv1a(session_id);
    size_t mask = cc->cap - 1;
    size_t idx = (size_t)h & mask;
    ssize_t first_tomb = -1;
    for (size_t probes = 0; probes < cc->cap; probes++) {
        OcCcSlot *s = &cc->slots[idx];
        if (s->entry == NULL) {
            if (s->deleted) {
                if (first_tomb < 0) {
                    first_tomb = (ssize_t)idx;
                }
            } else {
                /* genuinely empty — key absent */
                if (find_tombstone && first_tomb >= 0) {
                    return first_tomb;
                }
                return (ssize_t)idx;
            }
        } else if (strcmp(s->entry->session_id, session_id) == 0) {
            return (ssize_t)idx;
        }
        idx = (idx + 1) & mask;
    }
    return find_tombstone ? first_tomb : -1;
}

/* Remove an entry at slot index `i` (frees memory + on-disk file). */
static void oc_cc_drop_slot(OcContextCache *cc, size_t i) {
    OcCcSlot *s = &cc->slots[i];
    if (s->entry == NULL) {
        return;
    }
    cc->total_bytes -= s->entry->size_bytes;
    oc_cc_unlink_entry(&cc->cfg, s->entry->session_id);
    oc_cc_entry_free(s->entry);
    s->entry = NULL;
    s->deleted = true;
    cc->n_entries--;
}

/* Find the index of the least-recently-used live entry, or -1 if empty. */
static ssize_t oc_cc_lru_index(OcContextCache *cc) {
    ssize_t best = -1;
    uint64_t best_ts = 0;
    for (size_t i = 0; i < cc->cap; i++) {
        OcCcSlot *s = &cc->slots[i];
        if (s->entry == NULL) {
            continue;
        }
        if (best < 0 || s->entry->last_accessed < best_ts) {
            best = (ssize_t)i;
            best_ts = s->entry->last_accessed;
        }
    }
    return best;
}

/* Sweep TTL-expired entries. Called under lock. */
static void oc_cc_sweep_ttl(OcContextCache *cc) {
    if (cc->cfg.ttl_seconds == 0) {
        return;
    }
    uint64_t now = oc_cc_now();
    for (size_t i = 0; i < cc->cap; i++) {
        OcCcSlot *s = &cc->slots[i];
        if (s->entry == NULL) {
            continue;
        }
        if (now > s->entry->last_accessed &&
            (now - s->entry->last_accessed) >= cc->cfg.ttl_seconds) {
            oc_cc_drop_slot(cc, i);
            cc->evictions++;
        }
    }
}

/* Enforce capacity limits by evicting LRU entries until both limits are
 * satisfied. */
static void oc_cc_enforce_limits(OcContextCache *cc) {
    while (cc->n_entries > 0 &&
           (cc->n_entries > cc->cfg.max_entries ||
            cc->total_bytes > cc->cfg.max_size_bytes)) {
        ssize_t idx = oc_cc_lru_index(cc);
        if (idx < 0) {
            break;
        }
        oc_cc_drop_slot(cc, (size_t)idx);
        cc->evictions++;
    }
}

/* Public API                                                          */

OcContextCacheConfig oc_context_cache_config_default(void) {
    OcContextCacheConfig c;
    c.cache_dir       = NULL;
    c.max_entries     = OC_CONTEXT_CACHE_DEFAULT_MAX_ENTRIES;
    c.max_size_bytes  = OC_CONTEXT_CACHE_DEFAULT_MAX_SIZE;
    c.ttl_seconds     = OC_CONTEXT_CACHE_DEFAULT_TTL;
    return c;
}

OcContextCache *oc_context_cache_init(const OcContextCacheConfig *cfg) {
    if (cfg == NULL) {
        return NULL;
    }
    OcContextCache *cc = (OcContextCache *)malloc(sizeof(*cc));
    if (cc == NULL) {
        return NULL;
    }
    memset(cc, 0, sizeof(*cc));
    cc->cfg = *cfg;
    /* Normalize defaults. */
    if (cc->cfg.max_entries == 0) {
        cc->cfg.max_entries = OC_CONTEXT_CACHE_DEFAULT_MAX_ENTRIES;
    }
    if (cc->cfg.max_size_bytes == 0) {
        cc->cfg.max_size_bytes = OC_CONTEXT_CACHE_DEFAULT_MAX_SIZE;
    }
    /* ttl_seconds == 0 means "no TTL" (preserved). */

    /* Copy cache_dir if provided. */
    if (cc->cfg.cache_dir != NULL) {
        size_t dlen = strlen(cc->cfg.cache_dir);
        char *dcopy = (char *)malloc(dlen + 1);
        if (dcopy == NULL) {
            free(cc);
            return NULL;
        }
        memcpy(dcopy, cc->cfg.cache_dir, dlen + 1);
        cc->cfg.cache_dir = dcopy;
        if (oc_cc_mkdir_p(cc->cfg.cache_dir) != 0) {
            free((void *)cc->cfg.cache_dir);
            free(cc);
            return NULL;
        }
    }

    cc->cap = oc_cc_round_up_pow2(cc->cfg.max_entries);
    cc->slots = (OcCcSlot *)calloc(cc->cap, sizeof(OcCcSlot));
    if (cc->slots == NULL) {
        free((void *)cc->cfg.cache_dir);
        free(cc);
        return NULL;
    }
    if (pthread_mutex_init(&cc->mu, NULL) != 0) {
        free(cc->slots);
        free((void *)cc->cfg.cache_dir);
        free(cc);
        return NULL;
    }
    cc->n_entries    = 0;
    cc->total_bytes  = 0;
    cc->hits         = 0;
    cc->misses       = 0;
    cc->evictions    = 0;
    return cc;
}

void oc_context_cache_free(OcContextCache *cc) {
    if (cc == NULL) {
        return;
    }
    pthread_mutex_lock(&cc->mu);
    for (size_t i = 0; i < cc->cap; i++) {
        if (cc->slots[i].entry != NULL) {
            oc_cc_entry_free(cc->slots[i].entry);
            cc->slots[i].entry = NULL;
        }
    }
    free(cc->slots);
    cc->slots = NULL;
    pthread_mutex_unlock(&cc->mu);
    pthread_mutex_destroy(&cc->mu);
    free((void *)cc->cfg.cache_dir);
    cc->cfg.cache_dir = NULL;
    free(cc);
}

uint64_t oc_context_cache_model_hash(uint64_t file_size, uint32_t tensor_count) {
    /* FNV-1a over the byte representation of (file_size, tensor_count). */
    uint64_t h = OC_CC_FNV_OFFSET;
    for (int i = 0; i < 8; i++) {
        h ^= (file_size >> (i * 8)) & 0xFF;
        h *= OC_CC_FNV_PRIME;
    }
    for (int i = 0; i < 4; i++) {
        h ^= (uint64_t)((tensor_count >> (i * 8)) & 0xFF);
        h *= OC_CC_FNV_PRIME;
    }
    /* Avoid the degenerate zero hash (which we use as a "no model" sentinel). */
    return h == 0 ? 1 : h;
}

OcError oc_context_cache_session_id(const char *prompt, uint64_t model_hash,
                                    char *out, size_t out_len) {
    if (prompt == NULL || out == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    if (out_len < OC_CONTEXT_CACHE_SESSION_ID_LEN) {
        return OC_ERR_INVALID_ARG;
    }
    /* Hash prompt (FNV-1a) then fold in model_hash. */
    uint64_t h = oc_cc_fnv1a(prompt);
    h ^= model_hash;
    h *= OC_CC_FNV_PRIME;
    /* Format 16 hex chars (64-bit) + NUL. */
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 16; i++) {
        out[i] = hex[(h >> (60 - i * 4)) & 0xF];
    }
    out[16] = '\0';
    return OC_OK;
}

OcError oc_context_cache_store(OcContextCache *cc, const char *session_id,
                               uint64_t model_hash, uint64_t n_tokens,
                               uint32_t n_layers, uint32_t n_head_kv,
                               uint32_t head_dim, uint8_t *data,
                               uint64_t size_bytes) {
    if (cc == NULL || session_id == NULL ||
        (data == NULL && size_bytes > 0)) {
        return OC_ERR_INVALID_ARG;
    }
    if (strlen(session_id) >= OC_CONTEXT_CACHE_SESSION_ID_LEN) {
        return OC_ERR_INVALID_ARG;
    }
    pthread_mutex_lock(&cc->mu);

    /* Sweep TTL-expired entries first. */
    oc_cc_sweep_ttl(cc);

    /* Replace existing entry if present. */
    ssize_t found = oc_cc_find_slot(cc, session_id, false);
    if (found >= 0 && cc->slots[(size_t)found].entry != NULL) {
        oc_cc_drop_slot(cc, (size_t)found);
    }

    /* Allocate the entry. We take ownership of `data` as-is (no copy). */
    OcContextCacheEntry *e = (OcContextCacheEntry *)malloc(sizeof(*e));
    if (e == NULL) {
        pthread_mutex_unlock(&cc->mu);
        return OC_ERR_OOM;
    }
    memset(e, 0, sizeof(*e));
    strncpy(e->session_id, session_id, sizeof(e->session_id) - 1);
    e->model_hash     = model_hash;
    e->n_tokens       = n_tokens;
    e->n_layers       = n_layers;
    e->n_head_kv      = n_head_kv;
    e->head_dim       = head_dim;
    e->created_at     = oc_cc_now();
    e->last_accessed  = e->created_at;
    e->size_bytes     = size_bytes;
    e->data           = data;   /* take ownership */

    /* Find an insertion slot (reuse tombstone if possible). */
    ssize_t slot = oc_cc_find_slot(cc, session_id, true);
    if (slot < 0) {
        /* Table is full of tombstones — fall back to scanning for any empty. */
        slot = -1;
        for (size_t i = 0; i < cc->cap; i++) {
            if (cc->slots[i].entry == NULL) {
                slot = (ssize_t)i;
                break;
            }
        }
        if (slot < 0) {
            /* Truly full — evict LRU to make room. */
            ssize_t lru = oc_cc_lru_index(cc);
            if (lru < 0) {
                oc_cc_entry_free(e);
                pthread_mutex_unlock(&cc->mu);
                return OC_ERR_OOM;
            }
            oc_cc_drop_slot(cc, (size_t)lru);
            cc->evictions++;
            slot = lru;
        }
    }
    cc->slots[(size_t)slot].entry    = e;
    cc->slots[(size_t)slot].deleted  = false;
    cc->n_entries++;
    cc->total_bytes += size_bytes;

    /* Write to disk before enforcing limits (so we can clean up the file if
     * the entry is immediately evicted). */
    OcError werr = oc_cc_write_entry(&cc->cfg, e);

    oc_cc_enforce_limits(cc);

    pthread_mutex_unlock(&cc->mu);
    return werr;
}

OcError oc_context_cache_load(OcContextCache *cc, const char *session_id,
                              uint64_t model_hash, OcContextCacheEntry *out,
                              bool *found) {
    if (found != NULL) {
        *found = false;
    }
    if (cc == NULL || session_id == NULL || out == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));

    pthread_mutex_lock(&cc->mu);

    /* Sweep TTL-expired entries first so a stale entry can't be returned. */
    oc_cc_sweep_ttl(cc);

    ssize_t slot = oc_cc_find_slot(cc, session_id, false);
    OcContextCacheEntry *e = NULL;
    if (slot >= 0) {
        e = cc->slots[(size_t)slot].entry;
    }

    /* On a memory miss, attempt to hydrate from disk. */
    if (e == NULL && cc->cfg.cache_dir != NULL) {
        char path[4096];
        if (oc_cc_path_for(cc->cfg.cache_dir, session_id, path, sizeof(path))
            == 0) {
            OcContextCacheEntry *disk = oc_cc_read_entry(path);
            if (disk != NULL) {
                /* Insert into the in-memory table. */
                ssize_t ins = oc_cc_find_slot(cc, session_id, true);
                if (ins >= 0) {
                    cc->slots[(size_t)ins].entry   = disk;
                    cc->slots[(size_t)ins].deleted = false;
                    cc->n_entries++;
                    cc->total_bytes += disk->size_bytes;
                    oc_cc_enforce_limits(cc);
                    e = disk;
                } else {
                    oc_cc_entry_free(disk);
                }
            }
        }
    }

    if (e == NULL) {
        cc->misses++;
        pthread_mutex_unlock(&cc->mu);
        return OC_OK;
    }

    /* Model hash mismatch → treat as miss (but keep the entry; it may match
     * a different model in the future). */
    if (model_hash != 0 && e->model_hash != model_hash) {
        cc->misses++;
        pthread_mutex_unlock(&cc->mu);
        return OC_OK;
    }

    /* TTL re-check (defensive — sweep_ttl should have caught it, but a
     * zero-time window is possible). */
    if (cc->cfg.ttl_seconds != 0) {
        uint64_t now = oc_cc_now();
        if (now > e->last_accessed &&
            (now - e->last_accessed) >= cc->cfg.ttl_seconds) {
            cc->misses++;
            pthread_mutex_unlock(&cc->mu);
            return OC_OK;
        }
    }

    /* Copy out a fresh buffer (caller frees out->data). */
    uint8_t *copy = oc_cc_dup_data(e->data, e->size_bytes);
    if (copy == NULL && e->size_bytes > 0) {
        pthread_mutex_unlock(&cc->mu);
        return OC_ERR_OOM;
    }
    *out = *e;
    out->data = copy;

    /* Update LRU timestamp. */
    e->last_accessed = oc_cc_now();
    /* Persist updated access time to disk (best-effort). */
    (void)oc_cc_write_entry(&cc->cfg, e);

    cc->hits++;
    if (found != NULL) {
        *found = true;
    }
    pthread_mutex_unlock(&cc->mu);
    return OC_OK;
}

bool oc_context_cache_evict(OcContextCache *cc) {
    if (cc == NULL) {
        return false;
    }
    pthread_mutex_lock(&cc->mu);
    ssize_t idx = oc_cc_lru_index(cc);
    bool evicted = false;
    if (idx >= 0) {
        oc_cc_drop_slot(cc, (size_t)idx);
        cc->evictions++;
        evicted = true;
    }
    pthread_mutex_unlock(&cc->mu);
    return evicted;
}

OcError oc_context_cache_clear(OcContextCache *cc) {
    if (cc == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    pthread_mutex_lock(&cc->mu);
    for (size_t i = 0; i < cc->cap; i++) {
        if (cc->slots[i].entry != NULL) {
            oc_cc_drop_slot(cc, i);
        }
        cc->slots[i].deleted = false;
    }
    cc->n_entries   = 0;
    cc->total_bytes = 0;
    pthread_mutex_unlock(&cc->mu);
    return OC_OK;
}

void oc_context_cache_get_stats(const OcContextCache *cc,
                                OcContextCacheStats *out) {
    if (out == NULL) {
        return;
    }
    memset(out, 0, sizeof(*out));
    if (cc == NULL) {
        return;
    }
    /* const_cast away for the mutex lock — the public API presents a read-only
     * view, but the mutex is a synchronization primitive, not logical state. */
    OcContextCache *mut = (OcContextCache *)cc;
    pthread_mutex_lock(&mut->mu);
    out->n_entries   = cc->n_entries;
    out->total_bytes = cc->total_bytes;
    out->hits        = cc->hits;
    out->misses      = cc->misses;
    out->evictions   = cc->evictions;
    uint64_t lookups = cc->hits + cc->misses;
    out->hit_rate    = (lookups == 0) ? 0.0
                                      : (double)cc->hits / (double)lookups;
    pthread_mutex_unlock(&mut->mu);
}

size_t oc_context_cache_format_stats(const OcContextCache *cc, char *buf,
                                     size_t cap) {
    OcContextCacheStats s;
    oc_context_cache_get_stats(cc, &s);
    int n = 0;
    if (buf == NULL || cap == 0) {
        n = snprintf(NULL, 0,
                     "{"
                     "\"n_entries\":%zu,"
                     "\"total_bytes\":%llu,"
                     "\"hits\":%llu,"
                     "\"misses\":%llu,"
                     "\"evictions\":%llu,"
                     "\"hit_rate\":%.4f"
                     "}",
                     s.n_entries,
                     (unsigned long long)s.total_bytes,
                     (unsigned long long)s.hits,
                     (unsigned long long)s.misses,
                     (unsigned long long)s.evictions,
                     s.hit_rate);
        return (n < 0) ? 0 : (size_t)n;
    }
    n = snprintf(buf, cap,
                 "{"
                 "\"n_entries\":%zu,"
                 "\"total_bytes\":%llu,"
                 "\"hits\":%llu,"
                 "\"misses\":%llu,"
                 "\"evictions\":%llu,"
                 "\"hit_rate\":%.4f"
                 "}",
                 s.n_entries,
                 (unsigned long long)s.total_bytes,
                 (unsigned long long)s.hits,
                 (unsigned long long)s.misses,
                 (unsigned long long)s.evictions,
                 s.hit_rate);
    if (n < 0) {
        buf[0] = '\0';
        return 0;
    }
    if ((size_t)n >= cap) {
        buf[cap - 1] = '\0';
        return cap - 1;
    }
    return (size_t)n;
}
