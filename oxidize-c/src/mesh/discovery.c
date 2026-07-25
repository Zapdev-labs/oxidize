/*
 * discovery.c — Mesh node discovery protocol implementation.
 *
 * Uses UDP multicast to announce node presence on the mesh.
 */
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1  /* BSD socket opts (IP_MULTICAST_TTL, etc.) on macOS */
#endif
#include "oxidize/discovery.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

static void copy_str(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0 || !src) { if (dst && cap > 0) dst[0] = '\0'; return; }
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

OcError oc_discovery_init(OcDiscoveryState *state, uint64_t self_id,
                         const char *multicast_addr, uint16_t port)
{
    if (!state) return OC_ERR_INVALID_ARG;
    memset(state, 0, sizeof(*state));
    state->self_id = self_id;
    if (multicast_addr)
        copy_str(state->multicast_addr, sizeof(state->multicast_addr), multicast_addr);
    else
        strcpy(state->multicast_addr, "239.0.0.1");
    state->port = port ? port : 7946;
    state->discovery_interval_ms = 5000;
    state->timeout_ms = 30000;
    return OC_OK;
}

OcError oc_discovery_add_seed(OcDiscoveryState *state, uint64_t id,
                             const char *addr, uint16_t port)
{
    if (!state || !addr) return OC_ERR_INVALID_ARG;
    if (state->n_peers >= OC_DISCOVERY_MAX_PEERS) return OC_ERR_OOM;

    /* Check if peer already exists. */
    for (uint32_t i = 0; i < state->n_peers; i++) {
        if (state->peers[i].id == id) return OC_OK;
    }

    OcPeerInfo *p = &state->peers[state->n_peers];
    memset(p, 0, sizeof(*p));
    p->id = id;
    copy_str(p->addr, sizeof(p->addr), addr);
    p->port = port;
    p->alive = false;
    p->last_seen_ms = 0;
    state->n_peers++;
    return OC_OK;
}

OcError oc_discovery_announce(OcDiscoveryState *state)
{
    if (!state) return OC_ERR_INVALID_ARG;

    /* Create UDP socket. */
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) return OC_ERR_IO;

    /* Set TTL for multicast. */
    unsigned char ttl = 1;
    setsockopt(sock, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl));

    /* Build announce packet: "OXIDIZE:<self_id>:<port>". */
    char buf[128];
    int len = snprintf(buf, sizeof(buf), "OXIDIZE:%llu:%u",
                       (unsigned long long)state->self_id, state->port);
    if (len < 0 || (size_t)len >= sizeof(buf)) {
        close(sock);
        return OC_ERR_INVALID_ARG;
    }

    /* Send to multicast address. */
    struct sockaddr_in dst;
    memset(&dst, 0, sizeof(dst));
    dst.sin_family = AF_INET;
    dst.sin_port = htons(state->port);
    inet_pton(AF_INET, state->multicast_addr, &dst.sin_addr);

    ssize_t sent = sendto(sock, buf, (size_t)len, 0,
                          (struct sockaddr *)&dst, sizeof(dst));
    close(sock);

    if (sent < 0) return OC_ERR_IO;
    return OC_OK;
}

OcError oc_discovery_receive_peer(OcDiscoveryState *state, uint64_t id,
                                  const char *addr, uint16_t port,
                                  uint32_t capabilities)
{
    if (!state || !addr) return OC_ERR_INVALID_ARG;

    /* Check if peer already exists. */
    for (uint32_t i = 0; i < state->n_peers; i++) {
        if (state->peers[i].id == id) {
            state->peers[i].alive = true;
            state->peers[i].last_seen_ms = state->current_ms;
            state->peers[i].capabilities = capabilities;
            copy_str(state->peers[i].addr, sizeof(state->peers[i].addr), addr);
            state->peers[i].port = port;
            return OC_OK;
        }
    }

    if (state->n_peers >= OC_DISCOVERY_MAX_PEERS) return OC_ERR_OOM;

    OcPeerInfo *p = &state->peers[state->n_peers];
    memset(p, 0, sizeof(*p));
    p->id = id;
    copy_str(p->addr, sizeof(p->addr), addr);
    p->port = port;
    p->capabilities = capabilities;
    p->alive = true;
    p->last_seen_ms = state->current_ms;
    state->n_peers++;
    return OC_OK;
}

OcError oc_discovery_tick(OcDiscoveryState *state, uint64_t current_ms)
{
    if (!state) return OC_ERR_INVALID_ARG;
    state->current_ms = current_ms;

    /* Check for timed-out peers. */
    for (uint32_t i = 0; i < state->n_peers; i++) {
        if (state->peers[i].alive) {
            uint64_t elapsed = current_ms - state->peers[i].last_seen_ms;
            if (elapsed > state->timeout_ms) {
                state->peers[i].alive = false;
            }
        }
    }
    return OC_OK;
}

OcError oc_discovery_get_peers(const OcDiscoveryState *state,
                              const OcPeerInfo **out, uint32_t *count)
{
    if (!state || !out || !count) return OC_ERR_INVALID_ARG;
    *out = state->peers;
    *count = state->n_peers;
    return OC_OK;
}

OcError oc_discovery_get_alive_peers(const OcDiscoveryState *state,
                                    const OcPeerInfo **out, uint32_t *count)
{
    if (!state || !out || !count) return OC_ERR_INVALID_ARG;
    *out = state->peers;
    *count = 0;
    for (uint32_t i = 0; i < state->n_peers; i++)
        if (state->peers[i].alive) (*count)++;
    return OC_OK;
}

uint32_t oc_discovery_n_peers(const OcDiscoveryState *state)
{
    return state ? state->n_peers : 0;
}

uint32_t oc_discovery_n_alive(const OcDiscoveryState *state)
{
    if (!state) return 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < state->n_peers; i++)
        if (state->peers[i].alive) count++;
    return count;
}

bool oc_discovery_has_peer(const OcDiscoveryState *state, uint64_t id)
{
    if (!state) return false;
    for (uint32_t i = 0; i < state->n_peers; i++)
        if (state->peers[i].id == id) return true;
    return false;
}

OcError oc_discovery_remove_peer(OcDiscoveryState *state, uint64_t id)
{
    if (!state) return OC_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < state->n_peers; i++) {
        if (state->peers[i].id == id) {
            /* Shift remaining peers down. */
            for (uint32_t j = i; j + 1 < state->n_peers; j++)
                state->peers[j] = state->peers[j + 1];
            state->n_peers--;
            return OC_OK;
        }
    }
    return OC_ERR_MODEL;
}

void oc_discovery_free(OcDiscoveryState *state)
{
    if (!state) return;
    memset(state, 0, sizeof(*state));
}
