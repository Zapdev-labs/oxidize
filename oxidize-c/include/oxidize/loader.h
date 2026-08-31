/* loader.h — Universal model loader with architecture detection. */
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

/* Initialize loader state. Idempotent — safe to call before each load. */
OcError oc_loader_init(void);

/* Load a model from `path`, detecting the architecture automatically. Returns OC_OK even if the architecture is unknown (check arch_name/loaded). Returns OC_ERR_INVALID_ARG if `path` or `result` is NULL. */
OcError oc_loader_load(const char *path, OcLoaderResult *result);

/* Load a model with an explicit architecture string. The string is
 * validated against the supported-architecture list; if unsupported,
 * returns OC_ERR_MODEL and sets `result->error_msg`. */
OcError oc_loader_load_with_arch(const char *path, const char *arch,
                                 OcLoaderResult *result);

/* Detect the architecture from a GGUF file without fully loading it. Writes the name into `out_arch`. Returns OC_ERR_INVALID_ARG on NULL args or out_size == 0, OC_ERR_IO / OC_ERR_FORMAT from gguf open, or OC_OK with an empty string if general.architecture is missing. */
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
