/*
 * rdma.h — RDMA (Remote Direct Memory Access) transport.
 *
 * Provides a minimal API for RDMA-style registered-memory transfer.
 * Port from oxidize-core/src/mesh/rdma.rs.
 *
 * ponytail: no libibverbs (this port links libc only), so the queue pair is
 * a per-device loopback staging buffer: oc_rdma_send() copies the requested
 * window out of the region, oc_rdma_receive() copies it back into one. That
 * really moves the caller's bytes and is enough to exercise registration,
 * bounds checks, and byte accounting. Swap the staging copy for verbs
 * post_send/post_recv when an InfiniBand build is added.
 */
#ifndef OXIDIZE_RDMA_H
#define OXIDIZE_RDMA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_RDMA_MAX_REGIONS 64

typedef struct {
    void *addr;
    size_t length;
    uint32_t rkey;
    uint32_t lkey;
    bool valid;
} OcRdmaRegion;

typedef struct {
    char device_name[128];
    bool active;
    uint16_t port;
    uint32_t max_mr_size;
    OcRdmaRegion regions[OC_RDMA_MAX_REGIONS];
    uint32_t n_regions;
    uint64_t bytes_sent;
    uint64_t bytes_received;
    /* Loopback staging buffer holding the last posted send. */
    void  *staging;
    size_t staging_len;
    size_t staging_cap;
} OcRdmaDevice;

OcError oc_rdma_init(OcRdmaDevice *dev, const char *device_name);
OcError oc_rdma_register_memory(OcRdmaDevice *dev, void *addr, size_t length,
                                OcRdmaRegion *out_region);
OcError oc_rdma_deregister_memory(OcRdmaDevice *dev, OcRdmaRegion *region);
/* Post a send: copies region[offset .. offset+length) into the device's
 * staging buffer. Returns OC_ERR_INVALID_ARG on an out-of-range window or
 * an invalid region, OC_ERR_OOM if the staging buffer cannot grow. */
OcError oc_rdma_send(OcRdmaDevice *dev, const OcRdmaRegion *region,
                     size_t offset, size_t length);
/* Complete a receive: copies `length` bytes of the staged payload into
 * region[offset ..]. Returns OC_ERR_INVALID_ARG on an out-of-range window
 * or an invalid region, OC_ERR_NETWORK if fewer than `length` bytes were
 * staged by a prior send. */
OcError oc_rdma_receive(OcRdmaDevice *dev, OcRdmaRegion *region,
                        size_t offset, size_t length);
bool oc_rdma_is_active(const OcRdmaDevice *dev);
uint32_t oc_rdma_n_regions(const OcRdmaDevice *dev);
uint64_t oc_rdma_bytes_sent(const OcRdmaDevice *dev);
uint64_t oc_rdma_bytes_received(const OcRdmaDevice *dev);
void oc_rdma_free(OcRdmaDevice *dev);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_RDMA_H */
