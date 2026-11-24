/*
 * vk.h — minimal Vulkan compute runtime (OC_VULKAN builds).
 *
 * Descriptor-free: every buffer has a device address, and shaders receive
 * addresses and sizes through push constants (GL_EXT_buffer_reference). One
 * command buffer is recorded at a time; oc_vk_submit runs it and waits.
 *
 * Memory kinds map onto the device's memory types:
 *   OC_VK_MEM_DEVICE  device-local, host-mapped when the device allows it
 *                     (the UMA carve-out on APUs; CPU reads are slow)
 *   OC_VK_MEM_HOST    host-visible, coherent, cached: CPU reads and writes at
 *                     full speed, the GPU reads through the cache snoop
 *   OC_VK_MEM_STREAM  host-visible, write-combined system memory (GTT): GPU
 *                     reads at full speed, CPU should only write sequentially
 *
 * Without OC_VULKAN every entry point reports OC_ERR_BACKEND.
 */
#ifndef OXIDIZE_VK_H
#define OXIDIZE_VK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcVk OcVk;
typedef struct OcVkPipe OcVkPipe;

typedef enum {
    OC_VK_MEM_DEVICE = 0,
    OC_VK_MEM_HOST   = 1,
    OC_VK_MEM_STREAM = 2,
} OcVkMemKind;

typedef struct OcVkBuf {
    void    *map;     /* host pointer, NULL when not host visible */
    uint64_t addr;    /* device address */
    size_t   size;
    void    *impl;
} OcVkBuf;

typedef struct OcVkInfo {
    char     name[256];
    uint32_t subgroup_size;
    uint64_t heap_device, heap_host;        /* bytes */
    uint64_t max_alloc;
    double   timestamp_ns;                  /* ns per timestamp tick, 0 = none */
    bool     integrated;                    /* memory is the host's RAM       */
} OcVkInfo;

/* Opens the first GPU (OC_VK_DEVICE=<index> picks another). */
OcError oc_vk_init(OcVk **out);
void    oc_vk_free(OcVk *vk);
const OcVkInfo *oc_vk_info(const OcVk *vk);

OcError oc_vk_buf_new(OcVk *vk, size_t size, OcVkMemKind kind, OcVkBuf *out);
void    oc_vk_buf_free(OcVk *vk, OcVkBuf *b);

/* A compute pipeline from SPIR-V words with push_bytes of push constants;
 * spec[i] sets specialization constant id i (n_spec may be 0). */
OcError oc_vk_pipe_new(OcVk *vk, const uint32_t *spv, size_t spv_bytes, uint32_t push_bytes,
                       const uint32_t *spec, uint32_t n_spec, OcVkPipe **out);
void    oc_vk_pipe_free(OcVk *vk, OcVkPipe *p);

/* Recording: begin, dispatches separated by barriers, submit (waits). */
OcError oc_vk_begin(OcVk *vk);
void    oc_vk_dispatch(OcVk *vk, const OcVkPipe *p, const void *push, uint32_t gx, uint32_t gy,
                       uint32_t gz);
/* Dispatch with the group counts read from buf + offset (3 x uint32). */
void    oc_vk_dispatch_indirect(OcVk *vk, const OcVkPipe *p, const void *push, const OcVkBuf *buf, size_t offset);
void    oc_vk_barrier(OcVk *vk);
void    oc_vk_fill(OcVk *vk, const OcVkBuf *b, size_t off, size_t bytes, uint32_t value);
/* GPU timestamp into slot (< 4096); read back after submit in ns. */
void    oc_vk_stamp(OcVk *vk, uint32_t slot);
OcError oc_vk_submit(OcVk *vk);
double  oc_vk_stamp_ns(OcVk *vk, uint32_t slot);

#ifdef __cplusplus
}
#endif

#endif
