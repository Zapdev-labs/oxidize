#ifndef OXIDIZE_BACKEND_H
#define OXIDIZE_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_BACKEND_MAX_NAME 64

/* Compute backend type. Values 0-4, stable order.
 * Mirrors oxidize-core::ComputeBackend enum variants. */
typedef enum {
    OC_BACKEND_CPU    = 0,
    OC_BACKEND_CUDA   = 1,
    OC_BACKEND_VULKAN = 2,
    OC_BACKEND_METAL  = 3,
    OC_BACKEND_WEBGPU = 4,
    OC_BACKEND__COUNT,
} OcBackendType;

/* Static info about a backend's availability and device capabilities.
 * Populated by oc_backend_detect() / oc_backend_detect_all(). */
typedef struct OcBackendInfo {
    OcBackendType type;
    char name[OC_BACKEND_MAX_NAME];
    bool available;
    uint32_t device_count;
    uint64_t vram_bytes;
    uint16_t compute_capability_major;
    uint16_t compute_capability_minor;
} OcBackendInfo;

/* An initialized backend instance. `user_data` is owned by the backend
 * implementation (e.g., OcCudaContext* for CUDA) and freed by
 * oc_backend_free(). */
typedef struct OcBackend {
    OcBackendType type;
    OcBackendInfo info;
    bool initialized;
    void *user_data;
} OcBackend;

/* Human-readable, NUL-terminated name for the backend type ("cpu", "cuda",
 * "vulkan", "metal", "webgpu"). Returns "unknown" for codes outside the
 * enum range. Never returns NULL. */
const char *oc_backend_type_name(OcBackendType type);

/* Detect if the given backend is available on this system and populate */
OcError oc_backend_detect(OcBackendType type, OcBackendInfo *out);

/* Detect all backends. Writes up to `max` entries into `out` and sets
 * `*n_out` to the number written. Returns OC_OK on success. If `max` is
 * smaller than OC_BACKEND__COUNT, only `max` entries are written. */
OcError oc_backend_detect_all(OcBackendInfo *out, uint32_t *n_out,
                              uint32_t max);

/* Initialize a backend instance for the given type. On success, */
OcError oc_backend_init(OcBackend *backend, OcBackendType type);

/* Cleanup: release any resources held by the backend. Safe on NULL or
 * uninitialized backends. After this call, the backend is zeroed. */
void oc_backend_free(OcBackend *backend);

/* Quick availability check for a backend type. Returns false for unknown
 * types. Does not allocate. */
bool oc_backend_is_available(OcBackendType type);

/* Return the best available backend type, preferring GPU backends (CUDA,
 * Vulkan, Metal) over CPU. Returns OC_BACKEND_CPU if no GPU backend is
 * available. */
OcBackendType oc_backend_best_available(void);

/* Format a human-readable description of the backend info into `out` */
size_t oc_backend_info_print(const OcBackendInfo *info, char *out,
                             size_t out_size);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_BACKEND_H */
