/*
 * topology.c — Network topology discovery implementation.
 */
#include "oxidize/topology.h"

#include <stdlib.h>
#include <string.h>

static int32_t find_node_idx(const OcTopology *topo, uint64_t node_id)
{
    for (uint32_t i = 0; i < topo->n_nodes; i++)
        if (topo->nodes[i].node_id == node_id) return (int32_t)i;
    return -1;
}

OcError oc_topology_init(OcTopology *topo)
{
    if (!topo) return OC_ERR_INVALID_ARG;
    memset(topo, 0, sizeof(*topo));
    topo->initialized = true;
    return OC_OK;
}

OcError oc_topology_add_node(OcTopology *topo, uint64_t node_id,
                            const char *addr, uint16_t port, uint32_t region_id)
{
    if (!topo || !topo->initialized) return OC_ERR_INVALID_ARG;
    if (topo->n_nodes >= OC_TOPO_MAX_NODES) return OC_ERR_OOM;
    if (find_node_idx(topo, node_id) >= 0) return OC_OK;

    OcTopoNode *n = &topo->nodes[topo->n_nodes];
    memset(n, 0, sizeof(*n));
    n->node_id = node_id;
    n->port = port;
    n->region_id = region_id;
    if (addr) {
        size_t alen = strlen(addr);
        if (alen >= sizeof(n->addr)) alen = sizeof(n->addr) - 1;
        memcpy(n->addr, addr, alen);
        n->addr[alen] = '\0';
    }
    topo->n_nodes++;
    if (region_id + 1 > topo->n_regions) topo->n_regions = region_id + 1;
    return OC_OK;
}

OcError oc_topology_add_link(OcTopology *topo, uint64_t node_a, uint64_t node_b,
                            uint64_t bandwidth_mbps, uint64_t latency_us)
{
    if (!topo) return OC_ERR_INVALID_ARG;
    int32_t ia = find_node_idx(topo, node_a);
    int32_t ib = find_node_idx(topo, node_b);
    if (ia < 0 || ib < 0 || ia == ib) return OC_ERR_INVALID_ARG;

    topo->links[ia][ib].bandwidth_mbps = bandwidth_mbps;
    topo->links[ia][ib].latency_us = latency_us;
    topo->links[ia][ib].connected = true;
    topo->links[ib][ia].bandwidth_mbps = bandwidth_mbps;
    topo->links[ib][ia].latency_us = latency_us;
    topo->links[ib][ia].connected = true;
    return OC_OK;
}

OcError oc_topology_get_link(const OcTopology *topo, uint64_t node_a,
                            uint64_t node_b, const OcTopoLink **out)
{
    if (!topo || !out) return OC_ERR_INVALID_ARG;
    int32_t ia = find_node_idx(topo, node_a);
    int32_t ib = find_node_idx(topo, node_b);
    if (ia < 0 || ib < 0) return OC_ERR_INVALID_ARG;
    *out = &topo->links[ia][ib];
    return OC_OK;
}

OcError oc_topology_get_node(const OcTopology *topo, uint64_t node_id,
                            const OcTopoNode **out)
{
    if (!topo || !out) return OC_ERR_INVALID_ARG;
    int32_t idx = find_node_idx(topo, node_id);
    if (idx < 0) return OC_ERR_MODEL;
    *out = &topo->nodes[idx];
    return OC_OK;
}

OcError oc_topology_get_neighbors(const OcTopology *topo, uint64_t node_id,
                                 uint64_t *out_ids, uint32_t max, uint32_t *out_count)
{
    if (!topo || !out_ids || !out_count) return OC_ERR_INVALID_ARG;
    int32_t idx = find_node_idx(topo, node_id);
    if (idx < 0) return OC_ERR_MODEL;
    uint32_t count = 0;
    for (uint32_t i = 0; i < topo->n_nodes && count < max; i++) {
        if (i != (uint32_t)idx && topo->links[idx][i].connected)
            out_ids[count++] = topo->nodes[i].node_id;
    }
    *out_count = count;
    return OC_OK;
}

OcError oc_topology_stats(const OcTopology *topo, OcTopoStats *out_stats)
{
    if (!topo || !out_stats) return OC_ERR_INVALID_ARG;
    memset(out_stats, 0, sizeof(*out_stats));
    uint64_t total_bw = 0;
    uint64_t total_lat = 0;
    uint32_t n_links = 0;
    for (uint32_t i = 0; i < topo->n_nodes; i++) {
        for (uint32_t j = i + 1; j < topo->n_nodes; j++) {
            if (topo->links[i][j].connected) {
                total_bw += topo->links[i][j].bandwidth_mbps;
                total_lat += topo->links[i][j].latency_us;
                n_links++;
            }
        }
    }
    out_stats->total_bandwidth = total_bw;
    out_stats->avg_latency = n_links > 0 ? total_lat / n_links : 0;
    out_stats->n_links = n_links;
    out_stats->n_regions = topo->n_regions;
    out_stats->diameter = n_links > 0 ? (uint32_t)(topo->n_nodes - 1) : 0;
    return OC_OK;
}

OcError oc_topology_shortest_path(const OcTopology *topo, uint64_t src,
                                  uint64_t dst, uint64_t *out_path,
                                  uint32_t max_hops, uint32_t *out_n_hops)
{
    if (!topo || !out_path || !out_n_hops) return OC_ERR_INVALID_ARG;
    int32_t src_idx = find_node_idx(topo, src);
    int32_t dst_idx = find_node_idx(topo, dst);
    if (src_idx < 0 || dst_idx < 0) return OC_ERR_MODEL;

    /* BFS. */
    if (src_idx == dst_idx) {
        *out_n_hops = 1;
        out_path[0] = src;
        return OC_OK;
    }

    bool visited[OC_TOPO_MAX_NODES] = {false};
    int32_t prev[OC_TOPO_MAX_NODES];
    for (uint32_t i = 0; i < OC_TOPO_MAX_NODES; i++) prev[i] = -1;
    int32_t queue[OC_TOPO_MAX_NODES];
    uint32_t head = 0, tail = 0;
    queue[tail++] = src_idx;
    visited[src_idx] = true;

    while (head < tail) {
        int32_t cur = queue[head++];
        if (cur == dst_idx) {
            /* Reconstruct path. */
            uint32_t n_hops = 0;
            int32_t path_rev[OC_TOPO_MAX_NODES];
            int32_t at = dst_idx;
            while (at != -1) {
                path_rev[n_hops++] = at;
                at = prev[at];
            }
            if (n_hops > max_hops) return OC_ERR_OOM;
            for (uint32_t i = 0; i < n_hops; i++)
                out_path[i] = topo->nodes[path_rev[n_hops - 1 - i]].node_id;
            *out_n_hops = n_hops;
            return OC_OK;
        }
        for (uint32_t i = 0; i < topo->n_nodes; i++) {
            if (!visited[i] && topo->links[cur][i].connected) {
                visited[i] = true;
                prev[i] = cur;
                queue[tail++] = i;
            }
        }
    }
    return OC_ERR_MODEL; /* no path found */
}

uint32_t oc_topology_n_nodes(const OcTopology *topo)
{
    return topo ? topo->n_nodes : 0;
}

uint32_t oc_topology_n_regions(const OcTopology *topo)
{
    return topo ? topo->n_regions : 0;
}

void oc_topology_free(OcTopology *topo)
{
    if (!topo) return;
    memset(topo, 0, sizeof(*topo));
}
