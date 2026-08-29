#include "oxidize/backend.h"
#include "oxidize/cuda.h"

#include <stdio.h>
#include <string.h>


const char *oc_backend_type_name(OcBackendType type)
{
    switch (type) {
    case OC_BACKEND_CPU:    return "cpu";
    case OC_BACKEND_CUDA:   return "cuda";
    case OC_BACKEND_VULKAN: return "vulkan";
    case OC_BACKEND_METAL:  return "metal";
    case OC_BACKEND_WEBGPU: return "webgpu";
    default:                return "unknown";
    }
}

static void fill_info(OcBackendInfo *info, OcBackendType type)
{
    memset(info, 0, sizeof(*info));
    info->type = type;
    /* name defaults to the canonical type name. */
    const char *nm = oc_backend_type_name(type);
    size_t n = strlen(nm);
    if (n >= OC_BACKEND_MAX_NAME) n = OC_BACKEND_MAX_NAME - 1;
    memcpy(info->name, nm, n);
    info->name[n] = '\0';
    info->available = false;
    info->device_count = 0;
    info->vram_bytes = 0;
    info->compute_capability_major = 0;
    info->compute_capability_minor = 0;
}


OcError oc_backend_detect(OcBackendType type, OcBackendInfo *out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    if (type >= OC_BACKEND__COUNT) return OC_ERR_INVALID_ARG;

    fill_info(out, type);

    switch (type) {
    case OC_BACKEND_CPU:
        /* CPU is always available in the C11 port. */
        out->available = true;
        out->device_count = 1;
        break;
    case OC_BACKEND_CUDA:
#ifdef OC_CUDA
        if (oc_cuda_available()) {
            out->available = true;
            out->device_count = 1;
        }
#else
        /* Compiled without OC_CUDA — CUDA is unavailable. */
        out->available = false;
#endif
        break;
    case OC_BACKEND_VULKAN:
        /* Vulkan not linked in the dependency-free C11 port. */
        out->available = false;
        break;
    case OC_BACKEND_METAL:
        /* Metal is macOS-only; the C11 port does not link Metal frameworks. */
        out->available = false;
        break;
    case OC_BACKEND_WEBGPU:
        /* WebGPU requires a runtime (Dawn/wgpu) not linked here. */
        out->available = false;
        break;
    default:
        out->available = false;
        break;
    }

    return OC_OK;
}

OcError oc_backend_detect_all(OcBackendInfo *out, uint32_t *n_out,
                              uint32_t max)
{
    if (!out || !n_out) return OC_ERR_INVALID_ARG;
    uint32_t count = 0;
    for (uint32_t i = 0; i < (uint32_t)OC_BACKEND__COUNT && count < max; i++) {
        OcError e = oc_backend_detect((OcBackendType)i, &out[count]);
        if (e != OC_OK) return e;
        count++;
    }
    *n_out = count;
    return OC_OK;
}


OcError oc_backend_init(OcBackend *backend, OcBackendType type)
{
    if (!backend) return OC_ERR_INVALID_ARG;
    if (type >= OC_BACKEND__COUNT) return OC_ERR_INVALID_ARG;

    memset(backend, 0, sizeof(*backend));
    backend->type = type;

    OcError e = oc_backend_detect(type, &backend->info);
    if (e != OC_OK) return e;

    if (!backend->info.available) return OC_ERR_BACKEND;

    /* CPU needs no allocation; user_data stays NULL. GPU backends would
     * allocate their context here (e.g., OcCudaContext). */
    backend->initialized = true;
    backend->user_data = NULL;
    return OC_OK;
}

void oc_backend_free(OcBackend *backend)
{
    if (!backend) return;
    /* GPU backends would free user_data here. The C11 CPU backend keeps
     * user_data NULL, so nothing to free. */
    if (backend->user_data) {
        /* Future: dispatch on backend->type to free context. For now, no
         * GPU backend allocates user_data in oc_backend_init(). */
    }
    memset(backend, 0, sizeof(*backend));
}


bool oc_backend_is_available(OcBackendType type)
{
    if (type >= OC_BACKEND__COUNT) return false;
    OcBackendInfo info;
    if (oc_backend_detect(type, &info) != OC_OK) return false;
    return info.available;
}

OcBackendType oc_backend_best_available(void)
{
    /* Prefer GPU backends first (CUDA > Vulkan > Metal > WebGPU), then
     * fall back to CPU. */
    OcBackendType preferred[] = {
        OC_BACKEND_CUDA,
        OC_BACKEND_VULKAN,
        OC_BACKEND_METAL,
        OC_BACKEND_WEBGPU,
    };
    for (uint32_t i = 0;
         i < (uint32_t)(sizeof(preferred) / sizeof(preferred[0])); i++) {
        if (oc_backend_is_available(preferred[i])) return preferred[i];
    }
    /* CPU is always available. */
    return OC_BACKEND_CPU;
}


size_t oc_backend_info_print(const OcBackendInfo *info, char *out,
                             size_t out_size)
{
    if (!info) {
        if (out && out_size > 0) out[0] = '\0';
        return 0;
    }
    /* Compute the formatted length without writing. */
    char tmp[256];
    int n = snprintf(tmp, sizeof(tmp),
                     "backend=%s available=%s device_count=%u vram=%llu "
                     "cc=%u.%u",
                     oc_backend_type_name(info->type),
                     info->available ? "yes" : "no",
                     (unsigned)info->device_count,
                     (unsigned long long)info->vram_bytes,
                     (unsigned)info->compute_capability_major,
                     (unsigned)info->compute_capability_minor);
    if (n < 0) {
        if (out && out_size > 0) out[0] = '\0';
        return 0;
    }
    size_t need = (size_t)n;
    if (out && out_size > 0) {
        size_t copy = need < out_size - 1 ? need : out_size - 1;
        memcpy(out, tmp, copy);
        out[copy] = '\0';
    }
    return need;
}
