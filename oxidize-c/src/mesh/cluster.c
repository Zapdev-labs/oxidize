/*
 * cluster.c — GPU cluster management implementation.
 */
#include "oxidize/cluster.h"

#include <stdlib.h>
#include <string.h>

static void copy_str(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0 || !src) return;
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

OcError oc_cluster_init(OcGpuCluster *cluster, uint64_t self_id)
{
    if (!cluster) return OC_ERR_INVALID_ARG;
    memset(cluster, 0, sizeof(*cluster));
    cluster->self_id = self_id;
    cluster->master_idx = 0;
    return OC_OK;
}

OcError oc_cluster_add_node(OcGpuCluster *cluster, const char *name,
                           const char *addr, uint16_t port,
                           OcClusterNodeType type, uint32_t n_gpus,
                           uint64_t gpu_memory)
{
    if (!cluster || !name) return OC_ERR_INVALID_ARG;
    if (cluster->n_nodes >= OC_CLUSTER_MAX_NODES) return OC_ERR_OOM;

    /* Check for duplicate. */
    for (uint32_t i = 0; i < cluster->n_nodes; i++) {
        if (strcmp(cluster->nodes[i].name, name) == 0) return OC_OK;
    }

    OcClusterNode *n = &cluster->nodes[cluster->n_nodes];
    memset(n, 0, sizeof(*n));
    n->id = (uint64_t)cluster->n_nodes + 1;
    copy_str(n->name, sizeof(n->name), name);
    if (addr) copy_str(n->addr, sizeof(n->addr), addr);
    n->port = port;
    n->type = type;
    n->n_gpus = n_gpus;
    n->total_gpu_memory = gpu_memory;
    n->free_gpu_memory = gpu_memory;
    n->online = true;
    n->n_running_tasks = 0;

    cluster->total_gpu_memory += gpu_memory;
    cluster->free_gpu_memory += gpu_memory;
    cluster->total_gpus += n_gpus;
    cluster->n_nodes++;

    if (type == OC_CLUSTER_NODE_MASTER)
        cluster->master_idx = cluster->n_nodes - 1;

    return OC_OK;
}

OcError oc_cluster_remove_node(OcGpuCluster *cluster, uint64_t node_id)
{
    if (!cluster) return OC_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < cluster->n_nodes; i++) {
        if (cluster->nodes[i].id == node_id) {
            cluster->total_gpu_memory -= cluster->nodes[i].total_gpu_memory;
            cluster->free_gpu_memory -= cluster->nodes[i].free_gpu_memory;
            cluster->total_gpus -= cluster->nodes[i].n_gpus;
            for (uint32_t j = i; j < cluster->n_nodes - 1; j++)
                cluster->nodes[j] = cluster->nodes[j + 1];
            cluster->n_nodes--;
            return OC_OK;
        }
    }
    return OC_ERR_MODEL;
}

OcError oc_cluster_get_node(const OcGpuCluster *cluster, uint64_t node_id,
                           const OcClusterNode **out)
{
    if (!cluster || !out) return OC_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < cluster->n_nodes; i++) {
        if (cluster->nodes[i].id == node_id) {
            *out = &cluster->nodes[i];
            return OC_OK;
        }
    }
    return OC_ERR_MODEL;
}

OcError oc_cluster_list_nodes(const OcGpuCluster *cluster,
                             const OcClusterNode **out_array, uint32_t *out_count)
{
    if (!cluster || !out_array || !out_count) return OC_ERR_INVALID_ARG;
    *out_array = cluster->nodes;
    *out_count = cluster->n_nodes;
    return OC_OK;
}

OcError oc_cluster_find_best_node(const OcGpuCluster *cluster,
                                  uint64_t required_memory,
                                  const OcClusterNode **out)
{
    if (!cluster || !out) return OC_ERR_INVALID_ARG;
    const OcClusterNode *best = NULL;
    uint64_t best_free = 0;
    for (uint32_t i = 0; i < cluster->n_nodes; i++) {
        const OcClusterNode *n = &cluster->nodes[i];
        if (!n->online) continue;
        if (n->free_gpu_memory >= required_memory) {
            if (!best || n->free_gpu_memory < best_free) {
                best = n;
                best_free = n->free_gpu_memory;
            }
        }
    }
    if (!best) return OC_ERR_OOM;
    *out = best;
    return OC_OK;
}

uint32_t oc_cluster_n_nodes(const OcGpuCluster *cluster)
{
    return cluster ? cluster->n_nodes : 0;
}

uint32_t oc_cluster_n_gpus(const OcGpuCluster *cluster)
{
    return cluster ? cluster->total_gpus : 0;
}

uint64_t oc_cluster_total_memory(const OcGpuCluster *cluster)
{
    return cluster ? cluster->total_gpu_memory : 0;
}

uint64_t oc_cluster_free_memory(const OcGpuCluster *cluster)
{
    return cluster ? cluster->free_gpu_memory : 0;
}

OcError oc_cluster_assign_task(OcGpuCluster *cluster, const OcClusterTask *task,
                              uint64_t *out_node_id)
{
    if (!cluster || !task || !out_node_id) return OC_ERR_INVALID_ARG;
    const OcClusterNode *best;
    OcError e = oc_cluster_find_best_node(cluster, task->model_size, &best);
    if (e != OC_OK) return e;

    /* Find mutable version. */
    for (uint32_t i = 0; i < cluster->n_nodes; i++) {
        if (cluster->nodes[i].id == best->id) {
            cluster->nodes[i].free_gpu_memory -= task->model_size;
            cluster->nodes[i].n_running_tasks++;
            cluster->free_gpu_memory -= task->model_size;
            *out_node_id = best->id;
            return OC_OK;
        }
    }
    return OC_ERR_MODEL;
}

const char *oc_cluster_node_type_name(OcClusterNodeType type)
{
    switch (type) {
    case OC_CLUSTER_NODE_WORKER:  return "worker";
    case OC_CLUSTER_NODE_MASTER:  return "master";
    case OC_CLUSTER_NODE_STANDBY: return "standby";
    default:                      return "unknown";
    }
}

void oc_cluster_free(OcGpuCluster *cluster)
{
    if (!cluster) return;
    memset(cluster, 0, sizeof(*cluster));
}
