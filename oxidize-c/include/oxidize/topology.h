#ifndef OXIDIZE_TOPOLOGY_H
#define OXIDIZE_TOPOLOGY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_TOPO_MAX_NODES 64

typedef struct {
    uint64_t bandwidth_mbps;  /* measured bandwidth */
    uint64_t latency_us;      /* measured latency */
    bool connected;
} OcTopoLink;

typedef struct {
    uint64_t node_id;
    char addr[128];
    uint16_t port;
    uint32_t region_id;
} OcTopoNode;

typedef struct {
    OcTopoNode nodes[OC_TOPO_MAX_NODES];
    uint32_t n_nodes;
    OcTopoLink links[OC_TOPO_MAX_NODES][OC_TOPO_MAX_NODES];
    uint32_t n_regions;
    bool initialized;
} OcTopology;

typedef struct {
    uint64_t total_bandwidth;
    uint64_t avg_latency;
    uint32_t n_links;
    uint32_t n_regions;
    uint32_t diameter;
} OcTopoStats;

OcError oc_topology_init(OcTopology *topo);
OcError oc_topology_add_node(OcTopology *topo, uint64_t node_id,
                            const char *addr, uint16_t port, uint32_t region_id);
OcError oc_topology_add_link(OcTopology *topo, uint64_t node_a, uint64_t node_b,
                            uint64_t bandwidth_mbps, uint64_t latency_us);
OcError oc_topology_get_link(const OcTopology *topo, uint64_t node_a,
                            uint64_t node_b, const OcTopoLink **out);
OcError oc_topology_get_node(const OcTopology *topo, uint64_t node_id,
                            const OcTopoNode **out);
OcError oc_topology_get_neighbors(const OcTopology *topo, uint64_t node_id,
                                 uint64_t *out_ids, uint32_t max, uint32_t *out_count);
OcError oc_topology_stats(const OcTopology *topo, OcTopoStats *out_stats);
OcError oc_topology_shortest_path(const OcTopology *topo, uint64_t src,
                                  uint64_t dst, uint64_t *out_path,
                                  uint32_t max_hops, uint32_t *out_n_hops);
uint32_t oc_topology_n_nodes(const OcTopology *topo);
uint32_t oc_topology_n_regions(const OcTopology *topo);
void oc_topology_free(OcTopology *topo);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_TOPOLOGY_H */
