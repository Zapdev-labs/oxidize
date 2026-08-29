#ifndef OXIDIZE_MODEL_REGISTRY_H
#define OXIDIZE_MODEL_REGISTRY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include "oxidize/error.h"
#include "oxidize/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of entries the registry will hold before refusing new */
#define OC_MODEL_REGISTRY_MAX_ENTRIES 256

/* Maximum length (including NUL) of a stored path or model name. Paths
 * longer than this are skipped during scan (the registry is intended for
 * human-managed cache directories, not arbitrary filesystem traversal). */
#define OC_MODEL_REGISTRY_MAX_PATH 512

/* A single registered model entry. All strings are NUL-terminated and owned */
typedef struct OcModelEntry {
    char               path[OC_MODEL_REGISTRY_MAX_PATH];   /* absolute or relative file path */
    char               name[OC_MODEL_REGISTRY_MAX_PATH];   /* basename without .gguf extension */
    OcModelArchitecture arch;                              /* from general.architecture      */
    char               quant_type[32];                     /* e.g. "Q4_K_M", "F16"           */
    uint64_t           size_bytes;                         /* from stat()                     */
    uint64_t           n_params;                           /* general.parameter_count        */
    uint32_t           n_layers;                           /* block_count                     */
    uint32_t           n_embd;                             /* embedding_length                */
    uint32_t           vocab_size;                         /* from tokenizer.ggml.tokens      */
    time_t             loaded_at;                          /* discovery time (time(NULL))    */
} OcModelEntry;

/* Sort keys for oc_model_registry_list(). */
typedef enum {
    OC_MODEL_SORT_BY_SIZE = 0,   /* ascending size_bytes                    */
    OC_MODEL_SORT_BY_NAME = 1,   /* lexicographic by name (case-insensitive) */
    OC_MODEL_SORT_BY_DATE = 2,   /* newest first (descending loaded_at)      */
} OcModelSortKey;

/* Aggregate stats computed by oc_model_registry_stats(). */
typedef struct OcModelRegistryStats {
    size_t   total_models;
    uint64_t total_size;
    /* counts indexed by OcModelArchitecture; element [OC_ARCH_UNKNOWN] holds
     * the count of entries whose architecture could not be determined. The
     * array is sized to OC_ARCH__COUNT so callers can index directly. */
    size_t   by_arch[OC_ARCH__COUNT];
    /* Per-quantization-type count. The registry does not enumerate quant
     * types ahead of time, so this is a small fixed table of (type, count)
     * pairs. `n_quant_types` is the number of populated slots. */
    struct { char type[32]; size_t count; } by_quant[32];
    size_t   n_quant_types;
} OcModelRegistryStats;

/* The registry itself. `entries` is a malloc'd array of `count` entries
 * (capacity `cap`). `cache_dir` is the directory passed to
 * oc_model_registry_init (NUL-terminated, owned by the registry). */
typedef struct OcModelRegistry {
    OcModelEntry *entries;
    size_t        count;
    size_t        cap;            /* allocated capacity (<= max_entries)    */
    size_t        max_entries;    /* hard cap (default OC_MODEL_REGISTRY_MAX_ENTRIES) */
    char          cache_dir[OC_MODEL_REGISTRY_MAX_PATH];
} OcModelRegistry;

/* Initialize a registry with capacity `max_entries` (clamped to OC_MODEL_REGISTRY_MAX_ENTRIES). */
OcError oc_model_registry_init(OcModelRegistry *reg, const char *cache_dir,
                               size_t max_entries);

/* Free all registry-owned allocations (entries array). Safe on NULL or
 * already-freed registry. Does NOT free `reg` itself. After this call the
 * registry is zeroed. */
void oc_model_registry_free(OcModelRegistry *reg);

/* Scan `dir` for `*.gguf` files (non-recursive) and add each one via */
OcError oc_model_registry_scan(OcModelRegistry *reg, const char *dir);

/* Add a single model file. Parses the GGUF header to populate arch, */
OcError oc_model_registry_add(OcModelRegistry *reg, const char *path);

/* Remove the entry whose path matches `path` exactly. Shifts subsequent
 * entries down. Returns OC_OK if removed, OC_ERR_INVALID_ARG if not found
 * (or NULL args). */
OcError oc_model_registry_remove(OcModelRegistry *reg, const char *path);

/* Find the first entry whose `name` fuzzy-matches `query`. Fuzzy match: 2. otherwise compute Levenshtein distance to name; return the entry */
const OcModelEntry *oc_model_registry_find(const OcModelRegistry *reg,
                                           const char *query);

/* List all entries sorted by `key`. Writes pointers to registry entries */
size_t oc_model_registry_list(const OcModelRegistry *reg,
                              OcModelSortKey key,
                              const OcModelEntry **out, size_t cap);

/* Format the registry as a JSON array into `buf` (cap bytes). Returns */
size_t oc_model_registry_format(const OcModelRegistry *reg,
                                char *buf, size_t cap);

/* Compute aggregate stats. `out` is zeroed then populated. Safe on NULL
 * registry (writes zeros). */
void oc_model_registry_stats(const OcModelRegistry *reg,
                             OcModelRegistryStats *out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MODEL_REGISTRY_H */
