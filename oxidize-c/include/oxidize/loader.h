/*
 * loader.h — Universal model loader with architecture detection.
 *
 * Provides a high-level interface for loading GGUF models: detects the
 * architecture from metadata, validates support, and reports tensor/layer
 * counts. Port from oxidize-core/src/model/loader.rs.
 *
 * This is a thin orchestrator over the existing OcModelLoader (see
 * model_loader.h) and OcGgufFile (see gguf.h). It adds architecture
 * detection, supported-arch queries, and a uniform OcLoaderResult.
 */
#ifndef OXIDIZE_LOADER_H
#define OXIDIZE_LOADER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_LOADER_RESULT_ARCH_LEN  64
#define OC_LOADER_RESULT_ERR_LEN  256

/* Result of a loader operation: detected architecture, tensor/layer
 * counts, model size, and load state. `error_msg` is NUL-terminated
 * when `loaded` is false. */
typedef struct OcLoaderResult {
    char arch_name[OC_LOADER_RESULT_ARCH_LEN];
    uint32_t n_tensors;
    uint32_t n_layers;
    uint64_t model_size;
    bool loaded;
    char error_msg[OC_LOADER_RESULT_ERR_LEN];
} OcLoaderResult;

/* Initialize loader state. Idempotent — safe to call before each load.
 * Returns OC_ERR_INVALID_ARG if `state` is NULL. In the current
 * dependency-free port, loader state is process-wide and this function
 * is a no-op placeholder for future registry initialization. */
OcError oc_loader_init(void);

/* Load a model from `path`, detecting the architecture automatically.
 * On success, `result->loaded` is true and `arch_name` / `n_tensors` /
 * `n_layers` / `model_size` are populated. On failure, `loaded` is false
 * and `error_msg` describes the error. Returns OC_OK if the load
 * succeeded (even if the architecture is unknown — check arch_name).
 * Returns OC_ERR_INVALID_ARG if `path` or `result` is NULL. */
OcError oc_loader_load(const char *path, OcLoaderResult *result);

/* Load a model with an explicit architecture string. The string is
 * validated against the supported-architecture list; if unsupported,
 * returns OC_ERR_MODEL and sets `result->error_msg`. */
OcError oc_loader_load_with_arch(const char *path, const char *arch,
                                 OcLoaderResult *result);

/* Detect the architecture from a GGUF file without fully loading it.
 * Writes the architecture name (e.g. "llama", "qwen2") into `out_arch`.
 * Returns OC_OK on success. Returns OC_ERR_IO if the file cannot be
 * opened, OC_ERR_FORMAT if it is not a valid GGUF. */
OcError oc_loader_detect_arch(const char *path, char *out_arch,
                              size_t out_size);

/* List supported architectures. Writes up to `*n` pointers into `out`
 * (pointing to static strings) and sets `*n` to the count. Returns
 * OC_OK on success. `out` may be NULL to query the count only. */
OcError oc_loader_supported_archs(const char **out, uint32_t *n);

/* Check if an architecture name is supported (case-insensitive after
 * '-' → '_' normalization, matching oc_model_arch_from_str). */
bool oc_loader_is_supported(const char *arch);

/* Cleanup a loader result. Zeros the struct. Safe on NULL. */
void oc_loader_unload(OcLoaderResult *result);

/* Get the architecture name by index (0..count-1). Returns NULL for
 * out-of-range indices. Pointers are static and stable. */
const char *oc_loader_arch_name(uint32_t idx);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_LOADER_H */
