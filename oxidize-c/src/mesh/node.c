/* node.c — mesh node identity and state. */
#include "oxidize/node.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


OcError oc_node_init(const OcNodeConfig *config, OcNode **out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    *out = NULL;
    if (!config) return OC_ERR_INVALID_ARG;
    if (config->id == 0) return OC_ERR_INVALID_ARG;

    uint32_t max_conn = config->max_connections;
    if (max_conn == 0) max_conn = OC_NODE_DEFAULT_MAX_CONNECTIONS;

    OcNode *n = malloc(sizeof(*n));
    if (!n) return OC_ERR_OOM;
    memset(n, 0, sizeof(*n));

    n->connections = malloc((size_t)max_conn * sizeof(uint64_t));
    if (!n->connections) { free(n); return OC_ERR_OOM; }

    n->info.config = *config;
    n->info.config.max_connections = max_conn;
    n->info.is_online         = true;
    n->info.n_connections     = 0;
    n->info.max_connections   = max_conn;
    n->info.uptime_sec        = 0;
    n->info.bytes_sent        = 0;
    n->info.bytes_received    = 0;
    n->n_connections          = 0;
    n->created_time_sec       = 0;

    /* Ensure addr is NUL-terminated even if the caller filled the whole
     * buffer with non-NUL bytes. */
    n->info.config.addr[OC_NODE_MAX_ADDR_LEN - 1] = '\0';

    *out = n;
    return OC_OK;
}

void oc_node_free(OcNode *node)
{
    if (!node) return;
    free(node->connections);
    memset(node, 0, sizeof(*node));
    free(node);
}


static int32_t node_find_connection(const OcNode *node, uint64_t peer_id)
{
    for (uint32_t i = 0; i < node->n_connections; i++) {
        if (node->connections[i] == peer_id) return (int32_t)i;
    }
    return -1;
}

OcError oc_node_connect(OcNode *node, uint64_t peer_id)
{
    if (!node) return OC_ERR_INVALID_ARG;
    if (peer_id == 0) return OC_ERR_INVALID_ARG;

    /* Idempotent: already connected. */
    if (node_find_connection(node, peer_id) >= 0) {
        return OC_OK;
    }
    if (node->n_connections >= node->info.max_connections) {
        return OC_ERR_OOM;
    }
    node->connections[node->n_connections++] = peer_id;
    node->info.n_connections = node->n_connections;
    return OC_OK;
}

OcError oc_node_disconnect(OcNode *node, uint64_t peer_id)
{
    if (!node) return OC_ERR_INVALID_ARG;
    if (peer_id == 0) return OC_ERR_INVALID_ARG;

    int32_t idx = node_find_connection(node, peer_id);
    if (idx < 0) return OC_OK;

    /* Shift down. */
    for (uint32_t i = (uint32_t)idx; i + 1 < node->n_connections; i++) {
        node->connections[i] = node->connections[i + 1];
    }
    node->n_connections--;
    node->info.n_connections = node->n_connections;
    return OC_OK;
}


OcError oc_node_get_info(const OcNode *node, OcNodeInfo *out_info)
{
    if (!node || !out_info) return OC_ERR_INVALID_ARG;
    *out_info = node->info;
    return OC_OK;
}

bool oc_node_is_online(const OcNode *node)
{
    if (!node) return false;
    return node->info.is_online;
}

uint32_t oc_node_n_connections(const OcNode *node)
{
    if (!node) return 0;
    return node->n_connections;
}

bool oc_node_has_capability(const OcNode *node, uint32_t cap)
{
    if (!node) return false;
    return (node->info.config.capabilities & cap) != 0;
}


OcError oc_node_record_sent(OcNode *node, uint64_t bytes)
{
    if (!node) return OC_ERR_INVALID_ARG;
    node->info.bytes_sent += bytes;
    return OC_OK;
}

OcError oc_node_record_received(OcNode *node, uint64_t bytes)
{
    if (!node) return OC_ERR_INVALID_ARG;
    node->info.bytes_received += bytes;
    return OC_OK;
}


const char *oc_node_capability_name(uint32_t cap)
{
    switch (cap) {
    case OC_NODE_CAP_GPU:     return "gpu";
    case OC_NODE_CAP_CPU:     return "cpu";
    case OC_NODE_CAP_STORAGE: return "storage";
    default:                  return "unknown";
    }
}
