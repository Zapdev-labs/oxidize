/* mesh.h — distributed inference mesh (libp2p-style peer discovery + sharding). */
#ifndef OXIDIZE_MESH_H
#define OXIDIZE_MESH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_MESH_MAX_PEERS 64
#define OC_MESH_MAX_ADDR 256

typedef enum {
    OC_PEER_UNKNOWN  = 0,
    OC_PEER_LEADER   = 1,
    OC_PEER_FOLLOWER = 2,
    OC_PEER_WORKER   = 3,
} OcPeerRole;

typedef struct OcPeer {
    uint32_t id;
    char addr[OC_MESH_MAX_ADDR];   /* host:port                           */
    OcPeerRole role;
    uint32_t n_gpus;                /* GPU count on this peer              */
    uint32_t memory_mb;            /* total GPU memory in MB              */
    bool online;
    int socket_fd;                 /* TCP socket to this peer, -1 if none */
} OcPeer;

typedef struct OcMeshConfig {
    uint16_t listen_port;
    const char *bootstrap_addr;    /* leader address to connect to         */
    bool is_leader;
    uint32_t tensor_parallel;       /* TP degree (1 = no sharding)         */
    uint32_t pipeline_parallel;    /* PP degree (1 = no pipelining)       */
} OcMeshConfig;

#define OC_MESH_CONFIG_DEFAULT ((OcMeshConfig){ \
    0, NULL, false, 1u, 1u })

typedef struct OcMesh {
    OcMeshConfig config;
    OcPeer peers[OC_MESH_MAX_PEERS];
    size_t n_peers;
    uint32_t self_id;
    bool initialized;
    int listen_fd;                 /* listening socket, -1 if not listening */
} OcMesh;

/* Initialize the mesh node. */
OcError oc_mesh_init(OcMesh *mesh, const OcMeshConfig *cfg);

/* Start listening for connections (leader only). */
OcError oc_mesh_listen(OcMesh *mesh);

/* Connect to a leader node. */
OcError oc_mesh_connect(OcMesh *mesh, const char *addr);

/* Broadcast a tensor shard to all peers. */
OcError oc_mesh_broadcast(OcMesh *mesh, const void *data, size_t len,
                          uint32_t layer_idx);

/* Gather results from all peers (all-reduce style). */
OcError oc_mesh_allreduce(OcMesh *mesh, float *data, size_t len);

/* Get the number of online peers. */
size_t oc_mesh_peer_count(const OcMesh *mesh);

/* Get a peer by index. */
const OcPeer *oc_mesh_get_peer(const OcMesh *mesh, size_t idx);

/* Shutdown the mesh node. */
void oc_mesh_free(OcMesh *mesh);

/* Compute the shard layout for tensor parallelism. */
typedef struct OcShardLayout {
    uint32_t rank;
    uint32_t world_size;
    uint32_t shard_start;
    uint32_t shard_end;
} OcShardLayout;

OcError oc_mesh_shard_for_layer(const OcMesh *mesh, uint32_t layer_idx,
                                 uint32_t n_rows, OcShardLayout *out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MESH_H */
