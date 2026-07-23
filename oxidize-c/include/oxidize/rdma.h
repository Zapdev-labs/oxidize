/*
 * rdma.h — RDMA (Remote Direct Memory Access) stub.
 *
 * Provides a minimal API for RDMA-based zero-copy data transfer.
 * Stub implementation. Port from oxidize-core/src/mesh/rdma.rs.
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
} OcRdmaDevice;

OcError oc_rdma_init(OcRdmaDevice *dev, const char *device_name);
OcError oc_rdma_register_memory(OcRdmaDevice *dev, void *addr, size_t length,
                                OcRdmaRegion *out_region);
OcError oc_rdma_deregister_memory(OcRdmaDevice *dev, OcRdmaRegion *region);
OcError oc_rdma_send(OcRdmaDevice *dev, const OcRdmaRegion *region,
                     size_t offset, size_t length);
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
