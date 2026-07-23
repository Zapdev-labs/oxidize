/*
 * advanced_features.h — Advanced inference features.
 *
 * Higher-level features like prompt caching, prefix matching,
 * speculative decoding orchestration, and multi-model serving.
 * Port from oxidize-core/src/model/advanced_features.rs.
 */
#ifndef OXIDIZE_ADVANCED_FEATURES_H
#define OXIDIZE_ADVANCED_FEATURES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_AF_MAX_PREFIX 4096

/* Prompt caching: cache KV for common prefixes. */
typedef struct {
    uint32_t *prefix_tokens;
    size_t n_prefix;
    size_t n_cached;
    uint64_t hash;
    bool valid;
} OcPromptCache;

/* Speculative decoding orchestration. */
typedef enum {
    OC_SPEC_MODE_NONE = 0,
    OC_SPEC_MODE_EAGLE = 1,
    OC_SPEC_MODE_DFLASH = 2,
    OC_SPEC_MODE_TREE = 3,
    OC_SPEC_MODE_LOOKUP = 4,
} OcSpecMode;

typedef struct {
    OcSpecMode mode;
    uint32_t n_draft_tokens;
    float acceptance_rate;
    uint32_t n_accepted;
    uint32_t n_rejected;
    uint32_t n_total;
} OcSpecStats;

/* Multi-model serving. */
typedef struct {
    uint32_t max_models;
    uint32_t n_loaded;
    uint64_t total_memory_bytes;
    uint64_t used_memory_bytes;
    bool enable_hot_swap;
} OcMultiModelConfig;

typedef struct {
    char model_id[128];
    char model_path[512];
    uint64_t memory_bytes;
    bool loaded;
    uint64_t last_used_ms;
} OcModelSlot;

typedef struct {
    OcMultiModelConfig config;
    OcModelSlot *slots;
    uint32_t n_slots;
    uint32_t n_loaded;
    uint64_t used_memory_bytes;
} OcMultiModelServer;

/* Prompt cache API. */
OcError oc_prompt_cache_init(OcPromptCache *cache);
OcError oc_prompt_cache_set_prefix(OcPromptCache *cache, const uint32_t *tokens, size_t n);
OcError oc_prompt_cache_match(const OcPromptCache *cache, const uint32_t *tokens, size_t n, size_t *out_match_len);
bool oc_prompt_cache_is_valid(const OcPromptCache *cache);
void oc_prompt_cache_free(OcPromptCache *cache);

/* Speculative decoding API. */
OcError oc_spec_stats_init(OcSpecStats *stats, OcSpecMode mode, uint32_t n_draft);
OcError oc_spec_stats_record(OcSpecStats *stats, uint32_t n_accepted, uint32_t n_total);
float oc_spec_stats_acceptance_rate(const OcSpecStats *stats);
const char *oc_spec_mode_name(OcSpecMode mode);

/* Multi-model serving API. */
OcError oc_multi_model_init(OcMultiModelServer *srv, const OcMultiModelConfig *cfg);
OcError oc_multi_model_load(OcMultiModelServer *srv, const char *model_id, const char *path, uint64_t size_bytes);
OcError oc_multi_model_unload(OcMultiModelServer *srv, const char *model_id);
OcError oc_multi_model_find(const OcMultiModelServer *srv, const char *model_id, const OcModelSlot **out);
OcError oc_multi_model_evict_lru(OcMultiModelServer *srv);
bool oc_multi_model_is_loaded(const OcMultiModelServer *srv, const char *model_id);
uint32_t oc_multi_model_n_loaded(const OcMultiModelServer *srv);
uint64_t oc_multi_model_used_memory(const OcMultiModelServer *srv);
void oc_multi_model_free(OcMultiModelServer *srv);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ADVANCED_FEATURES_H */
