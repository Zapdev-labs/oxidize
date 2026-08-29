/* ring.h — Ring topology for distributed inference. */
#ifndef OXIDIZE_RING_H
#define OXIDIZE_RING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_RING_MAX_NODES 64

typedef struct {
    uint64_t id;
    char addr[128];
    uint16_t port;
    bool active;
} OcRingNode;

typedef struct {
    OcRingNode nodes[OC_RING_MAX_NODES];
    uint32_t n_nodes;
    uint32_t self_idx;
    uint64_t ring_id;
} OcRing;

typedef struct {
    uint32_t src_idx;
    uint32_t dst_idx;
    uint64_t bytes_transferred;
    uint64_t latency_us;
} OcRingStats;

OcError oc_ring_init(OcRing *ring, uint64_t self_id);
OcError oc_ring_add_node(OcRing *ring, uint64_t id, const char *addr, uint16_t port);
OcError oc_ring_remove_node(OcRing *ring, uint64_t id);
OcError oc_ring_get_next(const OcRing *ring, const OcRingNode **out_next);
OcError oc_ring_get_prev(const OcRing *ring, const OcRingNode **out_prev);
OcError oc_ring_get_node(const OcRing *ring, uint32_t idx, const OcRingNode **out);
uint32_t oc_ring_size(const OcRing *ring);
uint32_t oc_ring_self_idx(const OcRing *ring);
OcError oc_ring_stats(const OcRing *ring, OcRingStats *out_stats);
OcError oc_ring_barrier(OcRing *ring);
void oc_ring_free(OcRing *ring);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_RING_H */
