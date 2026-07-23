/*
 * ring.c — Ring topology implementation.
 */
#include "oxidize/ring.h"

#include <stdlib.h>
#include <string.h>

OcError oc_ring_init(OcRing *ring, uint64_t self_id)
{
    if (!ring) return OC_ERR_INVALID_ARG;
    memset(ring, 0, sizeof(*ring));
    ring->self_idx = 0;
    ring->ring_id = self_id;
    /* Add self as first node. */
    ring->nodes[0].id = self_id;
    ring->nodes[0].active = true;
    ring->n_nodes = 1;
    return OC_OK;
}

OcError oc_ring_add_node(OcRing *ring, uint64_t id, const char *addr, uint16_t port)
{
    if (!ring || !addr) return OC_ERR_INVALID_ARG;
    if (ring->n_nodes >= OC_RING_MAX_NODES) return OC_ERR_OOM;

    /* Check for duplicate. */
    for (uint32_t i = 0; i < ring->n_nodes; i++) {
        if (ring->nodes[i].id == id) return OC_OK;
    }

    OcRingNode *n = &ring->nodes[ring->n_nodes];
    n->id = id;
    n->port = port;
    size_t alen = strlen(addr);
    if (alen >= sizeof(n->addr)) alen = sizeof(n->addr) - 1;
    memcpy(n->addr, addr, alen);
    n->addr[alen] = '\0';
    n->active = true;
    ring->n_nodes++;
    return OC_OK;
}

OcError oc_ring_remove_node(OcRing *ring, uint64_t id)
{
    if (!ring) return OC_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < ring->n_nodes; i++) {
        if (ring->nodes[i].id == id) {
            /* Don't remove self. */
            if (i == ring->self_idx) return OC_ERR_INVALID_ARG;
            /* Shift remaining nodes. */
            for (uint32_t j = i; j < ring->n_nodes - 1; j++)
                ring->nodes[j] = ring->nodes[j + 1];
            ring->n_nodes--;
            /* Adjust self_idx if needed. */
            if (i < ring->self_idx) ring->self_idx--;
            return OC_OK;
        }
    }
    return OC_ERR_MODEL;
}

OcError oc_ring_get_next(const OcRing *ring, const OcRingNode **out_next)
{
    if (!ring || !out_next) return OC_ERR_INVALID_ARG;
    if (ring->n_nodes == 0) return OC_ERR_MODEL;
    uint32_t next = (ring->self_idx + 1) % ring->n_nodes;
    *out_next = &ring->nodes[next];
    return OC_OK;
}

OcError oc_ring_get_prev(const OcRing *ring, const OcRingNode **out_prev)
{
    if (!ring || !out_prev) return OC_ERR_INVALID_ARG;
    if (ring->n_nodes == 0) return OC_ERR_MODEL;
    uint32_t prev = (ring->self_idx + ring->n_nodes - 1) % ring->n_nodes;
    *out_prev = &ring->nodes[prev];
    return OC_OK;
}

OcError oc_ring_get_node(const OcRing *ring, uint32_t idx, const OcRingNode **out)
{
    if (!ring || !out) return OC_ERR_INVALID_ARG;
    if (idx >= ring->n_nodes) return OC_ERR_INVALID_ARG;
    *out = &ring->nodes[idx];
    return OC_OK;
}

uint32_t oc_ring_size(const OcRing *ring)
{
    return ring ? ring->n_nodes : 0;
}

uint32_t oc_ring_self_idx(const OcRing *ring)
{
    return ring ? ring->self_idx : 0;
}

OcError oc_ring_stats(const OcRing *ring, OcRingStats *out_stats)
{
    if (!ring || !out_stats) return OC_ERR_INVALID_ARG;
    memset(out_stats, 0, sizeof(*out_stats));
    uint32_t next = (ring->self_idx + 1) % ring->n_nodes;
    uint32_t prev = (ring->self_idx + ring->n_nodes - 1) % ring->n_nodes;
    out_stats->src_idx = prev;
    out_stats->dst_idx = next;
    return OC_OK;
}

OcError oc_ring_barrier(OcRing *ring)
{
    /* In a real implementation, this would sync across all nodes via TCP. */
    if (!ring) return OC_ERR_INVALID_ARG;
    return OC_OK;
}

void oc_ring_free(OcRing *ring)
{
    if (!ring) return;
    memset(ring, 0, sizeof(*ring));
}
