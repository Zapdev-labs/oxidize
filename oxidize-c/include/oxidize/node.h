#ifndef OXIDIZE_NODE_H
#define OXIDIZE_NODE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OC_NODE_DEFAULT_MAX_CONNECTIONS 16u
#define OC_NODE_MAX_ADDR_LEN            256u

/* Capability flags (bitmask). */
#define OC_NODE_CAP_GPU      (1u << 0)
#define OC_NODE_CAP_CPU      (1u << 1)
#define OC_NODE_CAP_STORAGE  (1u << 2)


typedef struct OcNodeConfig {
    uint64_t id;                          /* node id (0 is invalid)              */
    char     addr[OC_NODE_MAX_ADDR_LEN]; /* host[:port]                          */
    uint16_t port;
    uint32_t capabilities;               /* bitmask of OC_NODE_CAP_*            */
    uint32_t max_connections;             /* default OC_NODE_DEFAULT_MAX_CONNECTIONS */
} OcNodeConfig;

typedef struct OcNodeInfo {
    OcNodeConfig config;
    bool     is_online;
    uint32_t n_connections;
    uint32_t max_connections;
    uint64_t uptime_sec;
    uint64_t bytes_sent;
    uint64_t bytes_received;
} OcNodeInfo;

typedef struct OcNode {
    OcNodeInfo info;
    uint64_t  *connections;     /* heap-allocated, capacity = info.max_connections */
    uint32_t   n_connections;   /* current count                                  */
    uint64_t   created_time_sec;
} OcNode;


/* Allocate a node. The config is copied in; `addr` may be empty.
 * Free with oc_node_free. */
OcError oc_node_init(const OcNodeConfig *config, OcNode **out);

/* Free the node and its owned storage. Safe on NULL / already-freed. */
void oc_node_free(OcNode *node);


/* Add a connection to `peer_id`. If the peer is already connected, returns
 * OC_OK without changing the count. If at max_connections, returns
 * OC_ERR_OOM. */
OcError oc_node_connect(OcNode *node, uint64_t peer_id);

/* Remove a connection to `peer_id`. If the peer wasn't connected, returns
 * OC_OK without changing the count. */
OcError oc_node_disconnect(OcNode *node, uint64_t peer_id);


/* Copy the node info into `out_info`. */
OcError oc_node_get_info(const OcNode *node, OcNodeInfo *out_info);

/* True if the node is online. */
bool oc_node_is_online(const OcNode *node);

/* Current number of active connections. */
uint32_t oc_node_n_connections(const OcNode *node);

/* True if the node advertises the given capability flag. */
bool oc_node_has_capability(const OcNode *node, uint32_t cap);


/* Record bytes sent. Updates info.bytes_sent. */
OcError oc_node_record_sent(OcNode *node, uint64_t bytes);

/* Record bytes received. Updates info.bytes_received. */
OcError oc_node_record_received(OcNode *node, uint64_t bytes);


/* Human-readable name for a single capability flag. Returns "unknown"
 * for unrecognized flags. Never returns NULL. */
const char *oc_node_capability_name(uint32_t cap);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_NODE_H */
