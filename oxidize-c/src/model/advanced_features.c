/*
 * advanced_features.c — Advanced inference features implementation.
 */
#include "oxidize/advanced_features.h"

#include <stdlib.h>
#include <string.h>

/* ─── Prompt Cache ─────────────────────────────────────────────────── */

static uint64_t hash_tokens(const uint32_t *tokens, size_t n)
{
    uint64_t h = 1469598103934665603ULL; /* FNV-1a offset */
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)tokens[i];
        h *= 1099511628211ULL;
    }
    return h;
}

OcError oc_prompt_cache_init(OcPromptCache *cache)
{
    if (!cache) return OC_ERR_INVALID_ARG;
    memset(cache, 0, sizeof(*cache));
    return OC_OK;
}

OcError oc_prompt_cache_set_prefix(OcPromptCache *cache, const uint32_t *tokens, size_t n)
{
    if (!cache || !tokens || n == 0) return OC_ERR_INVALID_ARG;
    if (n > OC_AF_MAX_PREFIX) n = OC_AF_MAX_PREFIX;

    free(cache->prefix_tokens);
    cache->prefix_tokens = malloc(n * sizeof(uint32_t));
    if (!cache->prefix_tokens) return OC_ERR_OOM;
    memcpy(cache->prefix_tokens, tokens, n * sizeof(uint32_t));
    cache->n_prefix = n;
    cache->n_cached = n;
    cache->hash = hash_tokens(tokens, n);
    cache->valid = true;
    return OC_OK;
}

OcError oc_prompt_cache_match(const OcPromptCache *cache, const uint32_t *tokens, size_t n, size_t *out_match_len)
{
    if (!cache || !tokens || !out_match_len) return OC_ERR_INVALID_ARG;
    *out_match_len = 0;
    if (!cache->valid || cache->n_prefix == 0) return OC_OK;

    size_t max = n < cache->n_prefix ? n : cache->n_prefix;
    size_t match = 0;
    for (size_t i = 0; i < max; i++) {
        if (tokens[i] == cache->prefix_tokens[i]) match++;
        else break;
    }
    *out_match_len = match;
    return OC_OK;
}

bool oc_prompt_cache_is_valid(const OcPromptCache *cache)
{
    return cache ? cache->valid : false;
}

void oc_prompt_cache_free(OcPromptCache *cache)
{
    if (!cache) return;
    free(cache->prefix_tokens);
    memset(cache, 0, sizeof(*cache));
}

/* ─── Speculative Decoding Stats ──────────────────────────────────── */

OcError oc_spec_stats_init(OcSpecStats *stats, OcSpecMode mode, uint32_t n_draft)
{
    if (!stats) return OC_ERR_INVALID_ARG;
    memset(stats, 0, sizeof(*stats));
    stats->mode = mode;
    stats->n_draft_tokens = n_draft;
    return OC_OK;
}

OcError oc_spec_stats_record(OcSpecStats *stats, uint32_t n_accepted, uint32_t n_total)
{
    if (!stats) return OC_ERR_INVALID_ARG;
    stats->n_accepted += n_accepted;
    stats->n_total += n_total;
    if (stats->n_total > 0)
        stats->acceptance_rate = (float)stats->n_accepted / (float)stats->n_total;
    return OC_OK;
}

float oc_spec_stats_acceptance_rate(const OcSpecStats *stats)
{
    return stats ? stats->acceptance_rate : 0.0f;
}

const char *oc_spec_mode_name(OcSpecMode mode)
{
    switch (mode) {
    case OC_SPEC_MODE_NONE:   return "none";
    case OC_SPEC_MODE_EAGLE:  return "eagle";
    case OC_SPEC_MODE_DFLASH: return "dflash";
    case OC_SPEC_MODE_TREE:   return "tree";
    case OC_SPEC_MODE_LOOKUP: return "lookup";
    default: return "unknown";
    }
}

/* ─── Multi-Model Serving ─────────────────────────────────────────── */

OcError oc_multi_model_init(OcMultiModelServer *srv, const OcMultiModelConfig *cfg)
{
    if (!srv) return OC_ERR_INVALID_ARG;
    memset(srv, 0, sizeof(*srv));

    if (cfg) {
        srv->config = *cfg;
    } else {
        memset(&srv->config, 0, sizeof(srv->config));
        srv->config.max_models = 4;
        srv->config.enable_hot_swap = true;
    }

    uint32_t max = srv->config.max_models > 0 ? srv->config.max_models : 4;
    srv->slots = calloc(max, sizeof(OcModelSlot));
    if (!srv->slots) return OC_ERR_OOM;
    srv->n_slots = max;
    return OC_OK;
}

static OcModelSlot *find_slot(OcMultiModelServer *srv, const char *model_id)
{
    for (uint32_t i = 0; i < srv->n_slots; i++)
        if (srv->slots[i].loaded && strcmp(srv->slots[i].model_id, model_id) == 0)
            return &srv->slots[i];
    return NULL;
}

static OcModelSlot *find_free_slot(OcMultiModelServer *srv)
{
    for (uint32_t i = 0; i < srv->n_slots; i++)
        if (!srv->slots[i].loaded)
            return &srv->slots[i];
    return NULL;
}

static OcModelSlot *find_lru_slot(OcMultiModelServer *srv)
{
    OcModelSlot *lru = NULL;
    for (uint32_t i = 0; i < srv->n_slots; i++)
        if (srv->slots[i].loaded)
            if (!lru || srv->slots[i].last_used_ms < lru->last_used_ms)
                lru = &srv->slots[i];
    return lru;
}

OcError oc_multi_model_load(OcMultiModelServer *srv, const char *model_id, const char *path, uint64_t size_bytes)
{
    if (!srv || !model_id || !path) return OC_ERR_INVALID_ARG;

    /* Check if already loaded. */
    if (find_slot(srv, model_id)) return OC_OK;

    /* Find free slot or evict. */
    OcModelSlot *slot = find_free_slot(srv);
    if (!slot) {
        if (!srv->config.enable_hot_swap) return OC_ERR_OOM;
        slot = find_lru_slot(srv);
        if (!slot) return OC_ERR_OOM;
        srv->used_memory_bytes -= slot->memory_bytes;
        memset(slot, 0, sizeof(*slot));
    }

    strncpy(slot->model_id, model_id, sizeof(slot->model_id) - 1);
    strncpy(slot->model_path, path, sizeof(slot->model_path) - 1);
    slot->memory_bytes = size_bytes;
    slot->loaded = true;
    slot->last_used_ms = 0; /* caller should update */

    srv->n_loaded++;
    srv->used_memory_bytes += size_bytes;
    return OC_OK;
}

OcError oc_multi_model_unload(OcMultiModelServer *srv, const char *model_id)
{
    if (!srv || !model_id) return OC_ERR_INVALID_ARG;
    OcModelSlot *slot = find_slot(srv, model_id);
    if (!slot) return OC_ERR_MODEL;

    srv->used_memory_bytes -= slot->memory_bytes;
    srv->n_loaded--;
    memset(slot, 0, sizeof(*slot));
    return OC_OK;
}

OcError oc_multi_model_find(const OcMultiModelServer *srv, const char *model_id, const OcModelSlot **out)
{
    if (!srv || !model_id || !out) return OC_ERR_INVALID_ARG;
    *out = NULL;
    for (uint32_t i = 0; i < srv->n_slots; i++)
        if (srv->slots[i].loaded && strcmp(srv->slots[i].model_id, model_id) == 0) {
            *out = &srv->slots[i];
            return OC_OK;
        }
    return OC_ERR_MODEL;
}

OcError oc_multi_model_evict_lru(OcMultiModelServer *srv)
{
    if (!srv) return OC_ERR_INVALID_ARG;
    OcModelSlot *lru = find_lru_slot(srv);
    if (!lru) return OC_ERR_MODEL;
    srv->used_memory_bytes -= lru->memory_bytes;
    srv->n_loaded--;
    memset(lru, 0, sizeof(*lru));
    return OC_OK;
}

bool oc_multi_model_is_loaded(const OcMultiModelServer *srv, const char *model_id)
{
    if (!srv || !model_id) return false;
    for (uint32_t i = 0; i < srv->n_slots; i++)
        if (srv->slots[i].loaded && strcmp(srv->slots[i].model_id, model_id) == 0)
            return true;
    return false;
}

uint32_t oc_multi_model_n_loaded(const OcMultiModelServer *srv)
{
    return srv ? srv->n_loaded : 0;
}

uint64_t oc_multi_model_used_memory(const OcMultiModelServer *srv)
{
    return srv ? srv->used_memory_bytes : 0;
}

void oc_multi_model_free(OcMultiModelServer *srv)
{
    if (!srv) return;
    free(srv->slots);
    memset(srv, 0, sizeof(*srv));
}
