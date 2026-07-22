/*
 * model_registry.h — on-disk model registry for the C port.
 *
 * Tracks .gguf files discovered in a cache directory (default
 * ~/.cache/oxidize/models). Each entry caches metadata parsed from the GGUF
 * header (architecture, layer count, embedding dim, vocab size) plus file
 * size and discovery time, so CLI commands like `oxidize list` and
 * `oxidize inspect` can render a table without re-parsing every file on
 * every invocation.
 *
 * The registry is intentionally process-local: it does not persist across
 * runs and does not watch for filesystem changes. Callers re-scan with
 * oc_model_registry_scan() when they know the directory may have changed
 * (e.g. after a download completes). The scan reads only the GGUF header
 * (first few KiB), so it is cheap even for multi-GB model files.
 *
 * Concurrency: the registry is NOT thread-safe. Callers that share one
 * registry across threads must serialize access externally.
 */
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

/* Maximum number of entries the registry will hold before refusing new
 * additions. Sized to cover a typical local cache (a handful of quantized
 * variants per base model). The cap is a soft limit: oc_model_registry_scan
 * stops adding entries when it hits the cap rather than erroring. */
#define OC_MODEL_REGISTRY_MAX_ENTRIES 256

/* Maximum length (including NUL) of a stored path or model name. Paths
 * longer than this are skipped during scan (the registry is intended for
 * human-managed cache directories, not arbitrary filesystem traversal). */
#define OC_MODEL_REGISTRY_MAX_PATH 512

/* A single registered model entry. All strings are NUL-terminated and owned
 * by the registry (strdup'd on add, freed on remove/free). Numeric fields
 * are populated from the GGUF header during scan/add; if the GGUF parse
 * fails they are left at 0 and `arch` is set to OC_ARCH_UNKNOWN. */
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

/* Initialize a registry with capacity `max_entries` (clamped to
 * OC_MODEL_REGISTRY_MAX_ENTRIES). `cache_dir` is copied (NUL-truncated to
 * fit); it may be NULL for an empty default. Returns OC_OK or
 * OC_ERR_INVALID_ARG (NULL `out`) / OC_ERR_OOM. */
OcError oc_model_registry_init(OcModelRegistry *reg, const char *cache_dir,
                               size_t max_entries);

/* Free all registry-owned allocations (entries array). Safe on NULL or
 * already-freed registry. Does NOT free `reg` itself. After this call the
 * registry is zeroed. */
void oc_model_registry_free(OcModelRegistry *reg);

/* Scan `dir` for `*.gguf` files (non-recursive) and add each one via
 * oc_model_registry_add. Entries already present (matched by path) are
 * skipped. Stops at max_entries. Returns OC_OK, OC_ERR_IO (opendir failed),
 * OC_ERR_OOM, or OC_ERR_INVALID_ARG. Even on error the registry may contain
 * a partial set of entries scanned before the failure. */
OcError oc_model_registry_scan(OcModelRegistry *reg, const char *dir);

/* Add a single model file. Parses the GGUF header to populate arch,
 * n_layers, n_embd, vocab_size, n_params; reads file size from stat(). If
 * the path is already present the existing entry is updated in place.
 * Returns OC_OK, OC_ERR_IO, OC_ERR_FORMAT, OC_ERR_OOM, or
 * OC_ERR_INVALID_ARG. GGUF parse failures are NOT fatal: the entry is added
 * with zeroed metadata and arch=OC_ARCH_UNKNOWN. */
OcError oc_model_registry_add(OcModelRegistry *reg, const char *path);

/* Remove the entry whose path matches `path` exactly. Shifts subsequent
 * entries down. Returns OC_OK if removed, OC_ERR_INVALID_ARG if not found
 * (or NULL args). */
OcError oc_model_registry_remove(OcModelRegistry *reg, const char *path);

/* Find the first entry whose `name` fuzzy-matches `query`. Fuzzy match:
 *   1. case-insensitive substring of name; if that matches, return it;
 *   2. otherwise compute Levenshtein distance to name; return the entry
 *      with the smallest distance provided distance <= strlen(query)/2.
 * Returns a pointer into `reg` (valid until the next mutating call), or
 * NULL if no match. */
const OcModelEntry *oc_model_registry_find(const OcModelRegistry *reg,
                                           const char *query);

/* List all entries sorted by `key`. Writes pointers to registry entries
 * (in sorted order) into `out` (cap slots). Returns the number of entries
 * written (min(count, cap)). `out` may be NULL to just retrieve the count.
 * The pointers alias into `reg` and are valid until the next mutating call. */
size_t oc_model_registry_list(const OcModelRegistry *reg,
                              OcModelSortKey key,
                              const OcModelEntry **out, size_t cap);

/* Format the registry as a JSON array into `buf` (cap bytes). Returns
 * bytes written excluding NUL, or 0 on overflow / NULL args. Each entry is
 * an object with path, name, arch, quant_type, size_bytes, n_params,
 * n_layers, n_embd, vocab_size, loaded_at (unix epoch). */
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
