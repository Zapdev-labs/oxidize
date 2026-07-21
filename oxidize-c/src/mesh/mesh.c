/*
 * mesh.c — distributed inference mesh stub implementation.
 */
#include "oxidize/mesh.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

OcError oc_mesh_init(OcMesh *mesh, const OcMeshConfig *cfg)
{
    if (!mesh || !cfg) return OC_ERR_INVALID_ARG;
    memset(mesh, 0, sizeof(*mesh));
    mesh->config = *cfg;
    mesh->self_id = 0;
    mesh->n_peers = 1;
    mesh->peers[0].id = 0;
    mesh->peers[0].role = cfg->is_leader ? OC_PEER_LEADER : OC_PEER_FOLLOWER;
    mesh->peers[0].online = true;
    mesh->peers[0].n_gpus = 0;
    mesh->peers[0].memory_mb = 0;
    mesh->initialized = true;
    return OC_OK;
}

OcError oc_mesh_listen(OcMesh *mesh)
{
    if (!mesh || !mesh->initialized) return OC_ERR_INVALID_ARG;
    /* Stub: real implementation would bind a socket. */
    return OC_OK;
}

OcError oc_mesh_connect(OcMesh *mesh, const char *addr)
{
    if (!mesh || !addr) return OC_ERR_INVALID_ARG;
    if (mesh->n_peers >= OC_MESH_MAX_PEERS) return OC_ERR_OOM;
    OcPeer *p = &mesh->peers[mesh->n_peers];
    p->id = (uint32_t)mesh->n_peers;
    snprintf(p->addr, sizeof(p->addr), "%s", addr);
    p->role = OC_PEER_LEADER;
    p->online = true;
    mesh->n_peers++;
    return OC_OK;
}

OcError oc_mesh_broadcast(OcMesh *mesh, const void *data, size_t len,
                          uint32_t layer_idx)
{
    (void)data; (void)len; (void)layer_idx;
    if (!mesh) return OC_ERR_INVALID_ARG;
    return OC_OK;
}

OcError oc_mesh_allreduce(OcMesh *mesh, float *data, size_t len)
{
    (void)data; (void)len;
    if (!mesh) return OC_ERR_INVALID_ARG;
    /* Single-node: no-op (data is already the full result). */
    return OC_OK;
}

size_t oc_mesh_peer_count(const OcMesh *mesh)
{
    return mesh ? mesh->n_peers : 0;
}

const OcPeer *oc_mesh_get_peer(const OcMesh *mesh, size_t idx)
{
    if (!mesh || idx >= mesh->n_peers) return NULL;
    return &mesh->peers[idx];
}

void oc_mesh_free(OcMesh *mesh)
{
    if (!mesh) return;
    memset(mesh, 0, sizeof(*mesh));
}

OcError oc_mesh_shard_for_layer(const OcMesh *mesh, uint32_t layer_idx,
                                 uint32_t n_rows, OcShardLayout *out)
{
    (void)layer_idx;
    if (!mesh || !out) return OC_ERR_INVALID_ARG;
    uint32_t world = mesh->config.tensor_parallel;
    if (world == 0) world = 1;
    uint32_t rank = mesh->self_id % world;
    uint32_t per = n_rows / world;
    uint32_t rem = n_rows % world;
    out->rank = rank;
    out->world_size = world;
    out->shard_start = rank * per + (rank < rem ? rank : rem);
    out->shard_end = out->shard_start + per + (rank < rem ? 1 : 0);
    return OC_OK;
}
