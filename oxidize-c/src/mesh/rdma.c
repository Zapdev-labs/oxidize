/*
 * rdma.c — RDMA transport implementation. See rdma.h for the loopback
 * staging-buffer design note.
 */
#include "oxidize/rdma.h"

#include <stdlib.h>
#include <string.h>

static void copy_str(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0 || !src) { if (dst && cap > 0) dst[0] = '\0'; return; }
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

OcError oc_rdma_init(OcRdmaDevice *dev, const char *device_name)
{
    if (!dev) return OC_ERR_INVALID_ARG;
    memset(dev, 0, sizeof(*dev));
    if (device_name) copy_str(dev->device_name, sizeof(dev->device_name), device_name);
    else strcpy(dev->device_name, "ib0");
    dev->active = false;
    dev->port = 0;
    dev->max_mr_size = 1024 * 1024 * 1024; /* 1 GB */
    return OC_OK;
}

OcError oc_rdma_register_memory(OcRdmaDevice *dev, void *addr, size_t length,
                                OcRdmaRegion *out_region)
{
    if (!dev || !addr || !out_region) return OC_ERR_INVALID_ARG;
    if (dev->n_regions >= OC_RDMA_MAX_REGIONS) return OC_ERR_OOM;

    OcRdmaRegion *r = &dev->regions[dev->n_regions];
    r->addr = addr;
    r->length = length;
    r->lkey = dev->n_regions + 1;
    r->rkey = dev->n_regions + 1;
    r->valid = true;
    dev->n_regions++;

    *out_region = *r;
    return OC_OK;
}

OcError oc_rdma_deregister_memory(OcRdmaDevice *dev, OcRdmaRegion *region)
{
    if (!dev || !region) return OC_ERR_INVALID_ARG;
    region->valid = false;
    region->addr = NULL;
    region->length = 0;
    return OC_OK;
}

/* Bounds check shared by send and receive: the window must be valid and
 * must not wrap size_t. */
static bool region_window_ok(const OcRdmaRegion *region, size_t offset,
                             size_t length)
{
    if (!region->valid || !region->addr) return false;
    if (offset > region->length) return false;
    return length <= region->length - offset;
}

OcError oc_rdma_send(OcRdmaDevice *dev, const OcRdmaRegion *region,
                     size_t offset, size_t length)
{
    if (!dev || !region) return OC_ERR_INVALID_ARG;
    if (!region_window_ok(region, offset, length)) return OC_ERR_INVALID_ARG;

    if (length > dev->staging_cap) {
        void *grown = realloc(dev->staging, length);
        if (!grown) return OC_ERR_OOM;
        dev->staging = grown;
        dev->staging_cap = length;
    }
    if (length > 0)
        memcpy(dev->staging, (const char *)region->addr + offset, length);
    dev->staging_len = length;
    dev->bytes_sent += length;
    return OC_OK;
}

OcError oc_rdma_receive(OcRdmaDevice *dev, OcRdmaRegion *region,
                        size_t offset, size_t length)
{
    if (!dev || !region) return OC_ERR_INVALID_ARG;
    if (!region_window_ok(region, offset, length)) return OC_ERR_INVALID_ARG;
    if (length > dev->staging_len) return OC_ERR_NETWORK;

    if (length > 0)
        memcpy((char *)region->addr + offset, dev->staging, length);
    dev->bytes_received += length;
    return OC_OK;
}

bool oc_rdma_is_active(const OcRdmaDevice *dev)
{
    return dev ? dev->active : false;
}

uint32_t oc_rdma_n_regions(const OcRdmaDevice *dev)
{
    return dev ? dev->n_regions : 0;
}

uint64_t oc_rdma_bytes_sent(const OcRdmaDevice *dev)
{
    return dev ? dev->bytes_sent : 0;
}

uint64_t oc_rdma_bytes_received(const OcRdmaDevice *dev)
{
    return dev ? dev->bytes_received : 0;
}

void oc_rdma_free(OcRdmaDevice *dev)
{
    if (!dev) return;
    free(dev->staging);
    memset(dev, 0, sizeof(*dev));
}
