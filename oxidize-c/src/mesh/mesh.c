#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1  /* BSD socket opts (IP_MULTICAST_TTL, etc.) on macOS */
#endif
#include "oxidize/mesh.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>

#define OC_MESH_PORT_DEFAULT 51920
#define OC_MESH_BUF_SIZE (1 << 20)  /* 1 MB send/recv buffer */

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
    mesh->peers[0].socket_fd = -1;
    mesh->listen_fd = -1;
    mesh->initialized = true;
    return OC_OK;
}

/* Resolve host (IPv4 literal or hostname) and connect. Returns fd or -1. */
static int mesh_tcp_connect(const char *host, uint16_t port)
{
    char portstr[8];
    snprintf(portstr, sizeof(portstr), "%u", (unsigned)port);

    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, portstr, &hints, &res) != 0 || !res) return -1;

    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

/* Accept any pending inbound connections and register them as peers. */
static void mesh_accept_pending(OcMesh *mesh)
{
    if (mesh->listen_fd < 0) return;
    for (;;) {
        struct sockaddr_in sa;
        socklen_t sl = sizeof(sa);
        int fd = accept(mesh->listen_fd, (struct sockaddr *)&sa, &sl);
        if (fd < 0) break; /* EAGAIN/EWOULDBLOCK: nothing pending */
        if (mesh->n_peers >= OC_MESH_MAX_PEERS) {
            close(fd);
            break;
        }
        OcPeer *p = &mesh->peers[mesh->n_peers];
        memset(p, 0, sizeof(*p));
        p->id = (uint32_t)mesh->n_peers;
        snprintf(p->addr, sizeof(p->addr), "%s:%u",
                 inet_ntoa(sa.sin_addr), (unsigned)ntohs(sa.sin_port));
        p->role = OC_PEER_FOLLOWER;
        p->online = true;
        p->socket_fd = fd;
        mesh->n_peers++;
    }
}

OcError oc_mesh_listen(OcMesh *mesh)
{
    if (!mesh || !mesh->initialized) return OC_ERR_INVALID_ARG;
    /* Create a TCP listening socket for incoming peer connections. */
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return OC_ERR_IO;

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    uint16_t port = mesh->config.listen_port > 0
        ? mesh->config.listen_port : OC_MESH_PORT_DEFAULT;
    addr.sin_port = htons(port);

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return OC_ERR_IO;
    }
    if (listen(fd, OC_MESH_MAX_PEERS) < 0) {
        close(fd);
        return OC_ERR_IO;
    }
    /* Non-blocking so pending connections can be drained opportunistically. */
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    if (mesh->listen_fd >= 0) close(mesh->listen_fd);
    mesh->listen_fd = fd;
    return OC_OK;
}

OcError oc_mesh_connect(OcMesh *mesh, const char *addr)
{
    if (!mesh || !mesh->initialized || !addr || addr[0] == '\0')
        return OC_ERR_INVALID_ARG;

    /* Parse "host:port" address. */
    char host[256];
    uint16_t port = OC_MESH_PORT_DEFAULT;
    const char *colon = strrchr(addr, ':');
    if (colon) {
        size_t hlen = (size_t)(colon - addr);
        if (hlen >= sizeof(host)) hlen = sizeof(host) - 1;
        memcpy(host, addr, hlen);
        host[hlen] = '\0';
        port = (uint16_t)atoi(colon + 1);
    } else {
        snprintf(host, sizeof(host), "%s", addr);
    }

    /* Retry an existing peer with the same address instead of consuming a
     * new slot on every attempt. */
    OcPeer *p = NULL;
    for (size_t i = 1; i < mesh->n_peers; i++) {
        if (strcmp(mesh->peers[i].addr, addr) == 0) {
            p = &mesh->peers[i];
            break;
        }
    }
    if (!p) {
        if (mesh->n_peers >= OC_MESH_MAX_PEERS) return OC_ERR_OOM;
        p = &mesh->peers[mesh->n_peers];
        memset(p, 0, sizeof(*p));
        p->id = (uint32_t)mesh->n_peers;
        snprintf(p->addr, sizeof(p->addr), "%s", addr);
        p->role = OC_PEER_LEADER;
        p->online = false;
        p->socket_fd = -1;
        mesh->n_peers++;
    }
    if (p->online && p->socket_fd >= 0) return OC_OK; /* already connected */

    int fd = mesh_tcp_connect(host, port);
    if (fd < 0) {
        /* Connection failed; peer stays registered offline for retry. */
        return OC_ERR_IO;
    }

    p->online = true;
    p->socket_fd = fd;
    return OC_OK;
}

OcError oc_mesh_broadcast(OcMesh *mesh, const void *data, size_t len,
                          uint32_t layer_idx)
{
    (void)data; (void)len; (void)layer_idx;
    if (!mesh || !mesh->initialized || (!data && len > 0))
        return OC_ERR_INVALID_ARG;
    mesh_accept_pending(mesh);
    return OC_OK;
}

OcError oc_mesh_allreduce(OcMesh *mesh, float *data, size_t len)
{
    (void)data; (void)len;
    if (!mesh || !mesh->initialized || (!data && len > 0))
        return OC_ERR_INVALID_ARG;
    mesh_accept_pending(mesh);
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
    if (mesh->initialized) {
        for (size_t i = 0; i < mesh->n_peers; i++) {
            if (mesh->peers[i].socket_fd >= 0)
                close(mesh->peers[i].socket_fd);
        }
        if (mesh->listen_fd >= 0) close(mesh->listen_fd);
    }
    memset(mesh, 0, sizeof(*mesh));
    mesh->listen_fd = -1;
}

OcError oc_mesh_shard_for_layer(const OcMesh *mesh, uint32_t layer_idx,
                                 uint32_t n_rows, OcShardLayout *out)
{
    (void)layer_idx;
    if (!mesh || !mesh->initialized || !out) return OC_ERR_INVALID_ARG;
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
