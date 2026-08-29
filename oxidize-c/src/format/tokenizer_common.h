/*
 * tokenizer_common.h — helpers shared by the BPE / tiktoken / SP / WP
 * tokenizer implementations (private to src/format/).
 *
 * Contents:
 *   - OcU64Map: open-addressing u64→u32 hash map used for merge ranks
 *     and merged-token ids (FNV-1a 64, linear probing, 0.7 load factor,
 *     UINT64_MAX empty sentinel, no tombstones).
 *   - oc_pair_key(): pack two u32 token ids into a u64 map key.
 */
#ifndef OXIDIZE_C_FORMAT_TOKENIZER_COMMON_H
#define OXIDIZE_C_FORMAT_TOKENIZER_COMMON_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "oxidize/arena.h"
#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/tokenizer.h"
#include "oxidize/vector.h"

/* ─── u64 → u32 open-addressing map ──────────────────────────────────── */

/* FNV-1a 64-bit hash of a u64 key (mixes the bits; the raw key is already
 * well-distributed from token-id pairs so this mainly avoids pathological
 * clustering for sequential ids). */
static inline uint64_t oc_fnv1a_u64(uint64_t key)
{
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 8; ++i) {
        h ^= (key >> (i * 8)) & 0xFF;
        h *= 1099511628211ull;
    }
    return h;
}

#define OC_U64_EMPTY UINT64_MAX

typedef struct {
    uint64_t *keys;     /* malloc'd, OC_U64_EMPTY marks empty */
    uint32_t *values;   /* malloc'd, parallel to keys */
    size_t    cap;      /* power of two */
    size_t    count;    /* live entries */
} OcU64Map;

/* Open-addressing map: values stored in-line (no pointer indirection).
 * Grows when load factor > 0.7. Linear probing. Tombstone-free: entries
 * are never deleted during the tokenizer's lifetime. */
static inline OcU64Map *oc_u64map_new(size_t initial_cap)
{
    if (initial_cap < 16) initial_cap = 16;
    size_t cap = 1;
    while (cap < initial_cap) cap <<= 1;
    OcU64Map *m = (OcU64Map *)calloc(1, sizeof(OcU64Map));
    if (!m) return NULL;
    m->keys = (uint64_t *)malloc(cap * sizeof(uint64_t));
    m->values = (uint32_t *)malloc(cap * sizeof(uint32_t));
    if (!m->keys || !m->values) {
        free(m->keys); free(m->values); free(m);
        return NULL;
    }
    for (size_t i = 0; i < cap; ++i) m->keys[i] = OC_U64_EMPTY;
    m->cap = cap;
    m->count = 0;
    return m;
}

static inline void oc_u64map_free(OcU64Map *m)
{
    if (!m) return;
    free(m->keys);
    free(m->values);
    free(m);
}

static inline bool oc_u64map_grow(OcU64Map *m, size_t new_cap)
{
    uint64_t *old_keys = m->keys;
    uint32_t *old_values = m->values;
    size_t old_cap = m->cap;

    m->keys = (uint64_t *)malloc(new_cap * sizeof(uint64_t));
    m->values = (uint32_t *)malloc(new_cap * sizeof(uint32_t));
    if (!m->keys || !m->values) {
        free(m->keys); free(m->values);
        m->keys = old_keys; m->values = old_values;
        return false;
    }
    for (size_t i = 0; i < new_cap; ++i) m->keys[i] = OC_U64_EMPTY;
    m->cap = new_cap;
    m->count = 0;
    for (size_t i = 0; i < old_cap; ++i) {
        if (old_keys[i] != OC_U64_EMPTY) {
            uint64_t key = old_keys[i];
            uint32_t hash = (uint32_t)oc_fnv1a_u64(key) & (m->cap - 1);
            while (m->keys[hash] != OC_U64_EMPTY) {
                hash = (hash + 1) & (m->cap - 1);
            }
            m->keys[hash] = key;
            m->values[hash] = old_values[i];
            m->count++;
        }
    }
    free(old_keys);
    free(old_values);
    return true;
}

/* Insert key→value (updates on duplicate key).
 * Returns OC_OK, OC_ERR_INVALID_ARG for the reserved empty sentinel, or
 * OC_ERR_OOM. */
static inline OcError oc_u64map_put(OcU64Map *m, uint64_t key, uint32_t value)
{
    if (key == OC_U64_EMPTY) {
        /* 0xFFFF...F is reserved as the empty sentinel; remapping never
         * happens in practice because token ids are u32 and packed pairs
         * never reach UINT64_MAX. */
        return OC_ERR_INVALID_ARG;
    }
    if ((m->count + 1) * 10 >= m->cap * 7) {
        if (!oc_u64map_grow(m, m->cap << 1)) {
            return OC_ERR_OOM;
        }
    }
    uint32_t hash = (uint32_t)oc_fnv1a_u64(key) & (m->cap - 1);
    while (m->keys[hash] != OC_U64_EMPTY) {
        if (m->keys[hash] == key) {
            m->values[hash] = value;
            return OC_OK;
        }
        hash = (hash + 1) & (m->cap - 1);
    }
    m->keys[hash] = key;
    m->values[hash] = value;
    m->count++;
    return OC_OK;
}

/* Lookup. Returns true and writes *out if found, false otherwise. */
static inline bool oc_u64map_get(const OcU64Map *m, uint64_t key, uint32_t *out)
{
    if (key == OC_U64_EMPTY) return false;
    uint32_t hash = (uint32_t)oc_fnv1a_u64(key) & (m->cap - 1);
    while (m->keys[hash] != OC_U64_EMPTY) {
        if (m->keys[hash] == key) {
            *out = m->values[hash];
            return true;
        }
        hash = (hash + 1) & (m->cap - 1);
    }
    return false;
}

/* Pack two u32 token ids into a u64 key (left in the high 32 bits). */
static inline uint64_t oc_pair_key(uint32_t left, uint32_t right)
{
    return ((uint64_t)left << 32) | (uint64_t)right;
}


/* Read a metadata string array (e.g. tokenizer.ggml.tokens) into an
 * OcVector of arena-owned char* copies. Shared by BPE / SP / WP loaders. */
static inline OcError oc_tokenizer_string_array(const OcGgufFile *gguf,
                                                 const char *key,
                                                 OcArena *arena,
                                                 OcVector *out)
{
    const OcGgufMetadataValue *v = oc_gguf_metadata_get(gguf, key);
    if (!v || v->type != OC_GGUF_MT_ARRAY) {
        return OC_ERR_TOKENIZER;
    }
    OcError e = oc_vector_init(out, sizeof(char *));
    if (e != OC_OK) return e;
    for (size_t i = 0; i < v->v.arr.len; ++i) {
        const OcGgufMetadataValue *elem = &v->v.arr.values[i];
        if (elem->type != OC_GGUF_MT_STRING) {
            oc_vector_free(out);
            return OC_ERR_TOKENIZER;
        }
        char *dup = oc_arena_dup_n(arena, elem->v.str.data, elem->v.str.len);
        if (!dup) { oc_vector_free(out); return OC_ERR_OOM; }
        e = oc_vector_push(out, &dup);
        if (e != OC_OK) { oc_vector_free(out); return e; }
    }
    return OC_OK;
}

#endif /* OXIDIZE_C_FORMAT_TOKENIZER_COMMON_H */
