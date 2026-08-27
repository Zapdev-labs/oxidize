/*
 * cluster.h — GPU cluster management for distributed inference.
 *
 * Manages a cluster of GPU nodes for distributed inference workloads.
 * Port from oxidize-core/src/cluster/gpu_cluster.rs.
 */
#ifndef OXIDIZE_CLUSTER_H
#define OXIDIZE_CLUSTER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_CLUSTER_MAX_NODES 64
#define OC_CLUSTER_MAX_NAME 128
#define OC_CLUSTER_MAX_ADDR 256

typedef enum {
    OC_CLUSTER_NODE_WORKER = 0,
    OC_CLUSTER_NODE_MASTER = 1,
    OC_CLUSTER_NODE_STANDBY = 2,
} OcClusterNodeType;

typedef struct {
    uint64_t id;
    char name[OC_CLUSTER_MAX_NAME];
    char addr[OC_CLUSTER_MAX_ADDR];
    uint16_t port;
    OcClusterNodeType type;
    uint32_t n_gpus;
    uint64_t total_gpu_memory;
    uint64_t free_gpu_memory;
    bool online;
    uint32_t n_running_tasks;
} OcClusterNode;

typedef struct {
    OcClusterNode nodes[OC_CLUSTER_MAX_NODES];
    uint32_t n_nodes;
    uint64_t self_id;
    uint32_t master_idx;
    uint64_t total_gpu_memory;
    uint64_t free_gpu_memory;
    uint32_t total_gpus;
} OcGpuCluster;

typedef struct {
    uint64_t node_id;
    uint32_t gpu_id;
    uint64_t model_size;
    uint32_t n_layers;
    char model_name[128];
} OcClusterTask;

OcError oc_cluster_init(OcGpuCluster *cluster, uint64_t self_id);
OcError oc_cluster_add_node(OcGpuCluster *cluster, const char *name,
                           const char *addr, uint16_t port,
                           OcClusterNodeType type, uint32_t n_gpus,
                           uint64_t gpu_memory);
OcError oc_cluster_remove_node(OcGpuCluster *cluster, uint64_t node_id);
OcError oc_cluster_get_node(const OcGpuCluster *cluster, uint64_t node_id,
                           const OcClusterNode **out);
OcError oc_cluster_list_nodes(const OcGpuCluster *cluster,
                             const OcClusterNode **out_array, uint32_t *out_count);
OcError oc_cluster_find_best_node(const OcGpuCluster *cluster,
                                  uint64_t required_memory,
                                  const OcClusterNode **out);
uint32_t oc_cluster_n_nodes(const OcGpuCluster *cluster);
uint32_t oc_cluster_n_gpus(const OcGpuCluster *cluster);
uint64_t oc_cluster_total_memory(const OcGpuCluster *cluster);
uint64_t oc_cluster_free_memory(const OcGpuCluster *cluster);
OcError oc_cluster_assign_task(OcGpuCluster *cluster, const OcClusterTask *task,
                              uint64_t *out_node_id);
const char *oc_cluster_node_type_name(OcClusterNodeType type);
void oc_cluster_free(OcGpuCluster *cluster);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_CLUSTER_H */
