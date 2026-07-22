/*
 * prefix_cache.c — LRU prefix cache implementation.
 */
#include "oxidize/prefix_cache.h"

#include <stdlib.h>
#include <string.h>

/* FNV-1a 64-bit offset basis and prime. */
#define FNV1A_OFFSET 0xcbf29ce484222325ULL
#define FNV1A_PRIME  0x100000001b3ULL

uint64_t oc_prefix_hash_tokens(const uint32_t *tokens, size_t n)
{
    uint64_t h = FNV1A_OFFSET;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)tokens[i];
        h *= FNV1A_PRIME;
    }
    return h;
}

uint64_t oc_prefix_hash_continue(uint64_t prev_hash, uint32_t token)
{
    uint64_t h = prev_hash;
    h ^= (uint64_t)token;
    h *= FNV1A_PRIME;
    return h;
}

void oc_prefix_cache_init(OcPrefixCache *c, size_t max_tokens)
{
    if (!c) return;
    memset(c, 0, sizeof(*c));
    c->max_tokens = (max_tokens > 0 && max_tokens <= OC_PREFIX_CACHE_MAX_TOKENS)
                    ? max_tokens : OC_PREFIX_CACHE_MAX_TOKENS;
}

const OcCachedPrefix *oc_prefix_cache_lookup(OcPrefixCache *c, uint64_t hash,
                                              const uint32_t *tokens,
                                              size_t n_tokens)
{
    if (!c || (n_tokens > 0 && !tokens)) return NULL;
    for (size_t i = 0; i < c->n_entries; i++) {
        if (c->entries[i].active && c->entries[i].hash == hash &&
            c->entries[i].n_tokens == n_tokens &&
            (n_tokens == 0 || memcmp(c->entries[i].tokens, tokens,
                                     n_tokens * sizeof(*tokens)) == 0)) {
            c->entries[i].last_used = ++c->clock;
            c->n_hits++;
            return &c->entries[i];
        }
    }
    c->n_misses++;
    return NULL;
}

OcError oc_prefix_cache_store(OcPrefixCache *c, uint64_t hash,
                               const uint32_t *tokens, size_t n_tokens,
                               void *kv_data, size_t kv_size)
{
    if (!c || !kv_data || (n_tokens > 0 && !tokens)) return OC_ERR_INVALID_ARG;
    if (n_tokens > c->max_tokens) return OC_ERR_INVALID_ARG;

    uint32_t *token_copy = NULL;
    if (n_tokens > 0) {
        token_copy = malloc(n_tokens * sizeof(*token_copy));
        if (!token_copy) return OC_ERR_OOM;
        memcpy(token_copy, tokens, n_tokens * sizeof(*token_copy));
    }

    /* Check if hash already exists (update in place). */
    for (size_t i = 0; i < c->n_entries; i++) {
        if (c->entries[i].active && c->entries[i].hash == hash &&
            c->entries[i].n_tokens == n_tokens &&
            (n_tokens == 0 || memcmp(c->entries[i].tokens, tokens,
                                     n_tokens * sizeof(*tokens)) == 0)) {
            free(c->entries[i].kv_data);
            free(c->entries[i].tokens);
            c->entries[i].kv_data = kv_data;
            c->entries[i].tokens = token_copy;
            c->entries[i].kv_size = kv_size;
            c->entries[i].n_tokens = n_tokens;
            c->entries[i].last_used = ++c->clock;
            return OC_OK;
        }
    }

    /* Find a free slot or evict LRU. */
    size_t slot = c->n_entries;
    if (slot >= OC_PREFIX_CACHE_MAX_ENTRIES) {
        /* Find LRU entry. */
        uint64_t oldest = UINT64_MAX;
        for (size_t i = 0; i < c->n_entries; i++) {
            if (c->entries[i].last_used < oldest) {
                oldest = c->entries[i].last_used;
                slot = i;
            }
        }
        free(c->entries[slot].kv_data);
        free(c->entries[slot].tokens);
    } else {
        c->n_entries++;
    }

    c->entries[slot].hash = hash;
    c->entries[slot].n_tokens = n_tokens;
    c->entries[slot].tokens = token_copy;
    c->entries[slot].kv_data = kv_data;
    c->entries[slot].kv_size = kv_size;
    c->entries[slot].last_used = ++c->clock;
    c->entries[slot].active = true;
    return OC_OK;
}

size_t oc_prefix_cache_evict(OcPrefixCache *c, uint64_t threshold)
{
    if (!c) return 0;
    size_t evicted = 0;
    for (size_t i = 0; i < c->n_entries; i++) {
        if (c->entries[i].active && c->entries[i].last_used < threshold) {
            free(c->entries[i].kv_data);
            free(c->entries[i].tokens);
            c->entries[i].active = false;
            c->entries[i].kv_data = NULL;
            c->entries[i].tokens = NULL;
            evicted++;
        }
    }
    return evicted;
}

void oc_prefix_cache_clear(OcPrefixCache *c)
{
    if (!c) return;
    for (size_t i = 0; i < c->n_entries; i++) {
        free(c->entries[i].kv_data);
        free(c->entries[i].tokens);
    }
    memset(c, 0, sizeof(*c));
}

void oc_prefix_cache_stats(const OcPrefixCache *c, OcPrefixCacheStats *out)
{
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!c) return;
    out->n_hits = c->n_hits;
    out->n_misses = c->n_misses;
    for (size_t i = 0; i < c->n_entries; i++) {
        if (c->entries[i].active) {
            out->n_entries++;
            out->total_kv_bytes += c->entries[i].kv_size;
        }
    }
}
