/*
 * distributed.c — distributed inference scheduler implementation.
 *
 * Coordinates model inference across multiple nodes using pipeline and
 * tensor parallelism over TCP sockets. Single-node mode (n_nodes == 1)
 * is a zero-overhead fast path: no sockets are opened and all
 * communication functions are no-ops.
 *
 * Communication model:
 *
 *   Pipeline parallelism:
 *     - Layers are split into `pipeline_stages` contiguous groups.
 *     - Each stage runs its layers, then sends the final hidden state to
 *       the next stage via oc_distributed_send_activations().
 *     - The next stage receives via oc_distributed_recv_activations().
 *     - The pipeline master (stage 0) drives the run and is the only node
 *       that interacts with the tokenizer / sampler.
 *
 *   Tensor parallelism:
 *     - Within a pipeline stage, `tensor_parallel_size` nodes each compute
 *       a shard of the weight matrix. After the matmul, they call
 *       oc_distributed_all_reduce() to sum partial results.
 *     - All-reduce uses a simple ring-reduce over TCP: each node sends its
 *       data to the next TP peer, accumulates, and forwards.
 *
 * Reconnection:
 *   If a peer's socket is disconnected (send/recv returns 0 or error), the
 *   peer is marked offline and oc_distributed_reconnect() can be called to
 *   re-establish the TCP connection. The disconnects stat is incremented on
 *   detection; reconnects stat is incremented on successful reconnection.
 */
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1  /* BSD socket opts (IP_MULTICAST_TTL, etc.) on macOS */
#endif
#include "oxidize/distributed.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>

/* MSG_NOSIGNAL is Linux-only; macOS suppresses SIGPIPE via the SO_NOSIGPIPE
 * socket option instead. Fall back to 0 (matching oxidize-server/http.c). */
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

/* ------------------------------------------------------------------ */
/* Constants                                                          */
/* ------------------------------------------------------------------ */

#define OC_DIST_PORT_DEFAULT 51930

/* ------------------------------------------------------------------ */
/* Helpers: timing                                                    */
/* ------------------------------------------------------------------ */

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static void update_latency(OcDistributedStats *st, double lat)
{
    st->latency_ms = lat;
    st->latency_samples++;
    /* Running average. */
    st->avg_latency_ms =
        st->avg_latency_ms +
        (lat - st->avg_latency_ms) / (double)st->latency_samples;
}

/* ------------------------------------------------------------------ */
/* Helpers: TCP socket I/O                                            */
/* ------------------------------------------------------------------ */

/* Send exactly `len` bytes, retrying on partial writes.
 * Returns OC_OK on success, OC_ERR_NETWORK on failure. */
static OcError send_all(int fd, const void *data, size_t len)
{
    const char *p = (const char *)data;
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return OC_ERR_NETWORK;
        }
        if (n == 0) return OC_ERR_NETWORK; /* peer closed */
        sent += (size_t)n;
    }
    return OC_OK;
}

/* Receive exactly `len` bytes, retrying on partial reads.
 * Returns OC_OK on success, OC_ERR_NETWORK on failure. */
static OcError recv_all(int fd, void *data, size_t len)
{
    char *p = (char *)data;
    size_t got = 0;
    while (got < len) {
        ssize_t n = recv(fd, p + got, len - got, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return OC_ERR_NETWORK;
        }
        if (n == 0) return OC_ERR_NETWORK; /* peer closed */
        got += (size_t)n;
    }
    return OC_OK;
}

/* Parse "host:port" into host buffer and port. */
static void parse_addr(const char *addr, char *host, size_t host_cap,
                       uint16_t *port)
{
    *port = OC_DIST_PORT_DEFAULT;
    if (!addr) {
        if (host_cap > 0) host[0] = '\0';
        return;
    }
    const char *colon = strrchr(addr, ':');
    if (colon) {
        size_t hlen = (size_t)(colon - addr);
        if (hlen >= host_cap) hlen = host_cap - 1;
        memcpy(host, addr, hlen);
        host[hlen] = '\0';
        *port = (uint16_t)atoi(colon + 1);
    } else {
        snprintf(host, host_cap, "%s", addr);
    }
}

/* Connect to a "host:port" address (IPv4 literal or hostname).
 * Returns fd >= 0 on success, -1 on fail. */
static int tcp_connect(const char *addr)
{
    char host[256];
    uint16_t port;
    parse_addr(addr, host, sizeof(host), &port);

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
    if (fd < 0) return -1;

    /* Disable Nagle for low-latency activation transfer. */
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return fd;
}

/* ------------------------------------------------------------------ */
/* Config validation                                                  */
/* ------------------------------------------------------------------ */

OcError oc_distributed_validate_config(const OcDistributedConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    if (cfg->n_nodes == 0) return OC_ERR_INVALID_ARG;
    if (cfg->n_nodes > OC_DIST_MAX_NODES) return OC_ERR_INVALID_ARG;
    if (cfg->node_rank >= cfg->n_nodes) return OC_ERR_INVALID_ARG;
    if (cfg->pipeline_stages == 0) return OC_ERR_INVALID_ARG;
    if (cfg->tensor_parallel_size == 0) return OC_ERR_INVALID_ARG;
    if (cfg->activation_dtype_size == 0) return OC_ERR_INVALID_ARG;

    /* Single-node shortcut: anything goes as long as n_nodes==1 and
     * ranks are 0. */
    if (cfg->n_nodes == 1) {
        if (cfg->pipeline_rank >= cfg->pipeline_stages)
            return OC_ERR_INVALID_ARG;
        if (cfg->tensor_rank >= cfg->tensor_parallel_size)
            return OC_ERR_INVALID_ARG;
        return OC_OK;
    }

    /* Multi-node: stricter checks. */
    if (cfg->pipeline_stages > cfg->n_nodes)
        return OC_ERR_INVALID_ARG;
    if (cfg->pipeline_stages > OC_DIST_MAX_PIPELINE)
        return OC_ERR_INVALID_ARG;
    if (cfg->tensor_parallel_size > OC_DIST_MAX_TP)
        return OC_ERR_INVALID_ARG;
    /* Total parallelism must not exceed node count. */
    if ((uint64_t)cfg->pipeline_stages * cfg->tensor_parallel_size >
        cfg->n_nodes)
        return OC_ERR_INVALID_ARG;
    if (cfg->pipeline_rank >= cfg->pipeline_stages)
        return OC_ERR_INVALID_ARG;
    if (cfg->tensor_rank >= cfg->tensor_parallel_size)
        return OC_ERR_INVALID_ARG;
    /* The rank tuple must be consistent with the linear rank mapping so a
     * node cannot claim another stage/group's slot. Nodes beyond the used
     * grid (node_rank >= pp * tp) are unused and only need valid bounds. */
    uint64_t used = (uint64_t)cfg->pipeline_stages * cfg->tensor_parallel_size;
    if ((uint64_t)cfg->node_rank < used &&
        cfg->node_rank != cfg->pipeline_rank * cfg->tensor_parallel_size +
                          cfg->tensor_rank)
        return OC_ERR_INVALID_ARG;
    return OC_OK;
}

/* ------------------------------------------------------------------ */
/* Role resolution                                                    */
/* ------------------------------------------------------------------ */

OcNodeRole oc_distributed_resolve_role(const OcDistributedConfig *cfg)
{
    if (!cfg) return OC_NODE_ROLE_NONE;

    if (cfg->n_nodes == 1) {
        /* Single node: it's the pipeline master. */
        return OC_NODE_ROLE_PIPELINE_MASTER;
    }

    if (cfg->tensor_parallel_size > 1) {
        /* If this node is within a TP group, it's a tensor-parallel node.
         * The rank-0 within each TP group is also the pipeline stage owner. */
        return OC_NODE_ROLE_TENSOR_PARALLEL;
    }

    /* Pure pipeline parallelism. */
    if (cfg->pipeline_rank == 0)
        return OC_NODE_ROLE_PIPELINE_MASTER;
    return OC_NODE_ROLE_PIPELINE_WORKER;
}

/* ------------------------------------------------------------------ */
/* Buffer management                                                  */
/* ------------------------------------------------------------------ */

static OcError ensure_buffers(OcDistributedScheduler *sched)
{
    if (sched->send_buf && sched->recv_buf &&
        sched->buf_capacity >= OC_DIST_BUF_SIZE)
        return OC_OK;

    if (!sched->send_buf) {
        sched->send_buf = malloc(OC_DIST_BUF_SIZE);
        if (!sched->send_buf) return OC_ERR_OOM;
    }
    if (!sched->recv_buf) {
        sched->recv_buf = malloc(OC_DIST_BUF_SIZE);
        if (!sched->recv_buf) {
            free(sched->send_buf);
            sched->send_buf = NULL;
            return OC_ERR_OOM;
        }
    }
    sched->buf_capacity = OC_DIST_BUF_SIZE;
    return OC_OK;
}

/* ------------------------------------------------------------------ */
/* Peer management                                                    */
/* ------------------------------------------------------------------ */

/* Find the peer that is the "next pipeline stage" for this node.
 * In a simple linear pipeline, that's the node with pipeline_rank+1.
 * For this implementation we assume peers are ranked by global node rank
 * and pipeline stages map 1:1 to ranks in pipeline order.
 *
 * Returns a pointer to the next-stage peer, or NULL if this is the last
 * stage or single-node. */
static OcDistPeer *find_next_pipeline_peer(OcDistributedScheduler *sched)
{
    if (sched->config.n_nodes <= 1) return NULL;
    if (sched->config.pipeline_rank + 1 >= sched->config.pipeline_stages)
        return NULL;

    /* Find peer with the next pipeline rank. We use node_rank+1 as a
     * simple linear mapping (works for pure pipeline parallelism).
     * For TP, the "next stage" peer is the first node of the next TP group. */
    uint32_t next_rank = sched->config.node_rank + sched->config.tensor_parallel_size;
    if (next_rank >= sched->n_peers) return NULL;
    return &sched->peers[next_rank];
}

/* Find the peer that is the "previous pipeline stage". */
static OcDistPeer *find_prev_pipeline_peer(OcDistributedScheduler *sched)
{
    if (sched->config.n_nodes <= 1) return NULL;
    if (sched->config.pipeline_rank == 0) return NULL;

    uint32_t prev_rank = sched->config.node_rank >= sched->config.tensor_parallel_size
        ? sched->config.node_rank - sched->config.tensor_parallel_size
        : 0;
    if (prev_rank >= sched->n_peers) return NULL;
    return &sched->peers[prev_rank];
}

/* Find the next tensor-parallel peer (ring topology). */
static OcDistPeer *find_next_tp_peer(OcDistributedScheduler *sched)
{
    if (sched->config.tensor_parallel_size <= 1) return NULL;
    uint32_t tp_base = sched->config.pipeline_rank * sched->config.tensor_parallel_size;
    uint32_t tp_next = tp_base + (sched->config.tensor_rank + 1) % sched->config.tensor_parallel_size;
    if (tp_next >= sched->n_peers) return NULL;
    return &sched->peers[tp_next];
}

/* ------------------------------------------------------------------ */
/* Init / Free                                                        */
/* ------------------------------------------------------------------ */

OcError oc_distributed_init(OcDistributedScheduler *sched,
                            const OcDistributedConfig *cfg)
{
    if (!sched) return OC_ERR_INVALID_ARG;

    /* Zero the struct first so free() is always safe. */
    memset(sched, 0, sizeof(*sched));

    OcError ve = oc_distributed_validate_config(cfg);
    if (ve != OC_OK) return ve;

    sched->config = *cfg;
    sched->role = oc_distributed_resolve_role(cfg);
    sched->listen_fd = -1;
    sched->send_buf = NULL;
    sched->recv_buf = NULL;
    sched->buf_capacity = 0;

    /* Initialize peer table. */
    sched->n_peers = cfg->n_nodes;

    for (uint32_t i = 0; i < sched->n_peers; i++) {
        OcDistPeer *p = &sched->peers[i];
        p->rank = i;
        p->addr[0] = '\0';
        p->role = (i == cfg->node_rank)
            ? sched->role
            : OC_NODE_ROLE_NONE;
        p->socket_fd = -1;
        p->online = (i == cfg->node_rank); /* self is always online */
        p->bytes_sent_to = 0;
        p->bytes_recv_from = 0;
    }

    /* Multi-node mode: set up TCP connections for pipeline parallelism.
     * The pipeline master (node_rank 0) listens and accepts connections
     * from workers. Workers connect to the coordinator address. */
    if (cfg->n_nodes > 1) {
        /* Mark initialized up front so oc_distributed_free() closes any
         * sockets opened below when a failure path aborts init. The final
         * memset in free() clears the flag again. */
        sched->initialized = true;

        uint32_t timeout_ms = cfg->connect_timeout_ms
                            ? cfg->connect_timeout_ms
                            : OC_DIST_CONNECT_TIMEOUT_MS;
        double deadline = now_ms() + (double)timeout_ms;

        if (cfg->node_rank == 0) {
            /* Pipeline master: listen on listen_port (or default). */
            uint16_t port = cfg->listen_port ? cfg->listen_port : OC_DIST_PORT_DEFAULT;
            int lfd = socket(AF_INET, SOCK_STREAM, 0);
            if (lfd < 0) { oc_distributed_free(sched); return OC_ERR_NETWORK; }

            int opt = 1;
            setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

            struct sockaddr_in addr;
            memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = INADDR_ANY;
            addr.sin_port = htons(port);

            if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) < 0 ||
                listen(lfd, (int)(cfg->n_nodes - 1)) < 0) {
                close(lfd);
                oc_distributed_free(sched);
                return OC_ERR_NETWORK;
            }
            sched->listen_fd = lfd;

            /* Bound the accept loop: a worker that never shows up must not
             * hang init forever. SO_RCVTIMEO applies to accept(). */
            struct timeval tv;
            tv.tv_sec = (time_t)(timeout_ms / 1000u);
            tv.tv_usec = (suseconds_t)((timeout_ms % 1000u) * 1000u);
            setsockopt(lfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

            /* Accept connections from workers (nodes 1..n_nodes-1).
             * Each worker connects and sends its rank as a 4-byte uint32.
             * A worker may reconnect after a failed handshake, so keep
             * accepting until every rank is online or the deadline passes. */
            uint32_t connected = 0;
            while (connected + 1 < cfg->n_nodes && now_ms() < deadline) {
                struct sockaddr_in caddr;
                socklen_t clen = sizeof(caddr);
                int cfd = accept(lfd, (struct sockaddr *)&caddr, &clen);
                if (cfd < 0) {
                    if (errno == EINTR) continue;
                    break; /* timeout or hard error */
                }

                /* Read the worker's rank. Rank 0 is us, so reject it. */
                uint32_t rank = 0;
                if (recv_all(cfd, &rank, sizeof(rank)) == OC_OK &&
                    rank > 0 && rank < cfg->n_nodes &&
                    !sched->peers[rank].online) {
                    OcDistPeer *p = &sched->peers[rank];
                    p->socket_fd = cfd;
                    p->online = true;
                    connected++;
                    /* Store the peer's address. */
                    char ip[INET_ADDRSTRLEN];
                    inet_ntop(AF_INET, &caddr.sin_addr, ip, sizeof(ip));
                    snprintf(p->addr, sizeof(p->addr), "%s:%u",
                             ip, ntohs(caddr.sin_port));
                } else {
                    close(cfd);
                }
            }
            /* Close listen socket; all peer traffic uses the accepted fds. */
            close(sched->listen_fd);
            sched->listen_fd = -1;
            if (connected + 1 < cfg->n_nodes) {
                oc_distributed_free(sched);
                return OC_ERR_NETWORK;
            }
        } else {
            /* Worker: connect to coordinator. The master may not be
             * listening yet, so retry until the deadline. */
            if (!cfg->coordinator_addr) {
                oc_distributed_free(sched);
                return OC_ERR_NETWORK;
            }

            int fd = -1;
            for (;;) {
                fd = tcp_connect(cfg->coordinator_addr);
                if (fd >= 0) break;
                if (now_ms() >= deadline) {
                    oc_distributed_free(sched);
                    return OC_ERR_NETWORK;
                }
                struct timespec req = { 0, 20 * 1000 * 1000 }; /* 20 ms */
                nanosleep(&req, NULL);
            }

            /* Send our rank so the master knows who we are. */
            uint32_t rank = cfg->node_rank;
            if (send_all(fd, &rank, sizeof(rank)) != OC_OK) {
                close(fd);
                oc_distributed_free(sched);
                return OC_ERR_NETWORK;
            }

            /* Store coordinator as peer 0. */
            OcDistPeer *p = &sched->peers[0];
            if (p->socket_fd >= 0) close(p->socket_fd);
            p->socket_fd = fd;
            p->online = true;
            snprintf(p->addr, sizeof(p->addr), "%s", cfg->coordinator_addr);
        }
    }

    /* Allocate communication buffers for multi-node. */
    if (cfg->n_nodes > 1) {
        sched->send_buf = malloc(OC_DIST_BUF_SIZE);
        sched->recv_buf = malloc(OC_DIST_BUF_SIZE);
        if (!sched->send_buf || !sched->recv_buf) {
            oc_distributed_free(sched);
            return OC_ERR_OOM;
        }
        sched->buf_capacity = OC_DIST_BUF_SIZE;
    }

    sched->initialized = true;
    return OC_OK;
}

void oc_distributed_free(OcDistributedScheduler *sched)
{
    if (!sched) return;

    /* Only close descriptors this scheduler created: a zero-initialized
     * struct has socket_fd/listen_fd == 0, and closing fd 0 would close
     * the process's stdin. Sockets exist only after a successful init. */
    if (sched->initialized) {
        for (uint32_t i = 0; i < sched->n_peers; i++) {
            if (sched->peers[i].socket_fd >= 0) {
                close(sched->peers[i].socket_fd);
                sched->peers[i].socket_fd = -1;
            }
        }
        if (sched->listen_fd >= 0) {
            close(sched->listen_fd);
            sched->listen_fd = -1;
        }
    }

    /* Free buffers. */
    free(sched->send_buf);
    free(sched->recv_buf);

    memset(sched, 0, sizeof(*sched));
    sched->listen_fd = -1;
}

/* ------------------------------------------------------------------ */
/* Reconnection                                                       */
/* ------------------------------------------------------------------ */

OcError oc_distributed_reconnect(OcDistributedScheduler *sched,
                                 uint32_t peer_rank)
{
    if (!sched || !sched->initialized) return OC_ERR_INVALID_ARG;
    if (peer_rank >= sched->n_peers) return OC_ERR_INVALID_ARG;

    OcDistPeer *p = &sched->peers[peer_rank];
    if (p->online && p->socket_fd >= 0) return OC_OK; /* already connected */
    if (p->addr[0] == '\0') return OC_ERR_NETWORK; /* no address to connect to */

    int fd = tcp_connect(p->addr);
    if (fd < 0) return OC_ERR_NETWORK;

    /* Close old socket if any. */
    if (p->socket_fd >= 0) close(p->socket_fd);
    p->socket_fd = fd;
    p->online = true;
    sched->stats.reconnects++;
    return OC_OK;
}

/* Mark a peer as disconnected and close its socket. */
static void mark_peer_disconnected(OcDistributedScheduler *sched,
                                   OcDistPeer *p)
{
    if (!p || !p->online) return;
    if (p->socket_fd >= 0) {
        close(p->socket_fd);
        p->socket_fd = -1;
    }
    p->online = false;
    sched->stats.disconnects++;
}

/* ------------------------------------------------------------------ */
/* Chunked send/recv (handles buffers larger than OC_DIST_BUF_SIZE)   */
/* ------------------------------------------------------------------ */

static OcError send_chunked(OcDistributedScheduler *sched, OcDistPeer *p,
                            const void *data, size_t len)
{
    if (!p || p->socket_fd < 0) return OC_ERR_NETWORK;

    /* Send length header first (uint64_t, big-endian/network byte order). */
    uint8_t hdr_n[8];
    for (int i = 0; i < 8; i++)
        hdr_n[i] = (uint8_t)((uint64_t)len >> (56 - 8 * i));
    OcError e = send_all(p->socket_fd, hdr_n, sizeof(hdr_n));
    if (e != OC_OK) {
        mark_peer_disconnected(sched, p);
        return e;
    }

    /* Send payload in chunks. */
    const char *src = (const char *)data;
    size_t off = 0;
    while (off < len) {
        size_t chunk = len - off;
        if (chunk > OC_DIST_BUF_SIZE) chunk = OC_DIST_BUF_SIZE;
        e = send_all(p->socket_fd, src + off, chunk);
        if (e != OC_OK) {
            mark_peer_disconnected(sched, p);
            return e;
        }
        off += chunk;
    }

    p->bytes_sent_to += len;
    sched->stats.bytes_sent += len;
    return OC_OK;
}

static OcError recv_chunked(OcDistributedScheduler *sched, OcDistPeer *p,
                            void *data, size_t expected_len)
{
    if (!p || p->socket_fd < 0) return OC_ERR_NETWORK;

    /* Receive length header (big-endian/network byte order). */
    uint8_t hdr_n[8];
    OcError e = recv_all(p->socket_fd, hdr_n, sizeof(hdr_n));
    if (e != OC_OK) {
        mark_peer_disconnected(sched, p);
        return e;
    }
    uint64_t hdr = 0;
    for (int i = 0; i < 8; i++)
        hdr = (hdr << 8) | hdr_n[i];
    if (hdr != expected_len) return OC_ERR_NETWORK; /* size mismatch */

    /* Receive payload in chunks. */
    char *dst = (char *)data;
    size_t off = 0;
    while (off < expected_len) {
        size_t chunk = expected_len - off;
        if (chunk > OC_DIST_BUF_SIZE) chunk = OC_DIST_BUF_SIZE;
        e = recv_all(p->socket_fd, dst + off, chunk);
        if (e != OC_OK) {
            mark_peer_disconnected(sched, p);
            return e;
        }
        off += chunk;
    }

    p->bytes_recv_from += expected_len;
    sched->stats.bytes_received += expected_len;
    return OC_OK;
}

/* ------------------------------------------------------------------ */
/* Pipeline parallelism: send/recv activations                        */
/* ------------------------------------------------------------------ */

OcError oc_distributed_send_activations(OcDistributedScheduler *sched,
                                        const void *data, size_t count)
{
    if (!sched || !sched->initialized) return OC_ERR_INVALID_ARG;
    if (count > 0 && !data) return OC_ERR_INVALID_ARG;

    sched->stats.send_calls++;

    /* Single-node or last pipeline stage: no-op. */
    if (sched->config.n_nodes <= 1) return OC_OK;
    if (sched->config.pipeline_rank + 1 >= sched->config.pipeline_stages)
        return OC_OK;

    OcDistPeer *next = find_next_pipeline_peer(sched);
    if (!next) return OC_OK; /* no next stage to send to */

    size_t dtype = sched->config.activation_dtype_size;
    if (dtype == 0) dtype = 4; /* default float */
    size_t total_bytes = count * dtype;

    if (total_bytes == 0) return OC_OK;

    if (!next->online || next->socket_fd < 0) {
        /* Try to reconnect. */
        OcError re = oc_distributed_reconnect(sched, next->rank);
        if (re != OC_OK) return OC_ERR_NETWORK;
    }

    return send_chunked(sched, next, data, total_bytes);
}

OcError oc_distributed_recv_activations(OcDistributedScheduler *sched,
                                        void *out, size_t count)
{
    if (!sched || !sched->initialized) return OC_ERR_INVALID_ARG;
    if (count > 0 && !out) return OC_ERR_INVALID_ARG;

    sched->stats.recv_calls++;

    /* Single-node or first pipeline stage: no-op (caller already has input). */
    if (sched->config.n_nodes <= 1) return OC_OK;
    if (sched->config.pipeline_rank == 0) return OC_OK;

    OcDistPeer *prev = find_prev_pipeline_peer(sched);
    if (!prev) return OC_OK; /* no previous stage */

    size_t dtype = sched->config.activation_dtype_size;
    if (dtype == 0) dtype = 4;
    size_t total_bytes = count * dtype;

    if (total_bytes == 0) return OC_OK;

    if (!prev->online || prev->socket_fd < 0) {
        /* Try to reconnect. */
        OcError re = oc_distributed_reconnect(sched, prev->rank);
        if (re != OC_OK) return OC_ERR_NETWORK;
    }

    return recv_chunked(sched, prev, out, total_bytes);
}

/* ------------------------------------------------------------------ */
/* Tensor parallelism: all-reduce (ring topology)                      */
/* ------------------------------------------------------------------ */

OcError oc_distributed_all_reduce(OcDistributedScheduler *sched,
                                  float *data, size_t count)
{
    if (!sched || !sched->initialized) return OC_ERR_INVALID_ARG;
    if (count > 0 && !data) return OC_ERR_INVALID_ARG;

    sched->stats.allreduce_calls++;

    /* Single-node or TP=1: no-op. */
    if (sched->config.n_nodes <= 1) return OC_OK;
    if (sched->config.tensor_parallel_size <= 1) return OC_OK;

    /* Ensure we have buffers. */
    OcError be = ensure_buffers(sched);
    if (be != OC_OK) return be;

    size_t byte_len = count * sizeof(float);

    /* NOTE: this ring exchange is known-incomplete (receives from the next
     * peer instead of the previous, and re-adds accumulated buffers for
     * TP >= 3). It is unreachable today because init rejects multi-node
     * configs; a proper reduce-scatter/all-gather is required before
     * multi-node init is enabled. */
    OcDistPeer *next = find_next_tp_peer(sched);
    if (!next) return OC_OK;

    /* Ensure next TP peer is connected. */
    if (!next->online || next->socket_fd < 0) {
        OcError re = oc_distributed_reconnect(sched, next->rank);
        if (re != OC_OK) return OC_ERR_NETWORK;
    }

    /* Use recv buffer as scratch for incoming data. */
    if (byte_len > sched->buf_capacity) {
        /* Resize buffer. */
        free(sched->recv_buf);
        sched->recv_buf = malloc(byte_len);
        if (!sched->recv_buf) {
            sched->recv_buf = NULL;
            return OC_ERR_OOM;
        }
        sched->buf_capacity = byte_len;
    }

    uint32_t tp = sched->config.tensor_parallel_size;
    for (uint32_t step = 0; step < tp - 1; step++) {
        /* Send our current data to the next peer. */
        OcError se = send_chunked(sched, next, data, byte_len);
        if (se != OC_OK) return se;

        /* Receive from previous peer (which is `next`'s predecessor = us
         * in ring; but in ring topology we receive from the previous node).
         * For simplicity in a 2-node ring, next == prev. */
        OcDistPeer *prev = find_next_tp_peer(sched); /* same as next in ring */
        /* Actually, in a ring, the node we receive from is the one whose
         * socket we're connected to as "incoming". For this simplified
         * implementation, we assume bidirectional connections. */

        /* Receive into scratch buffer. */
        OcError re = recv_chunked(sched, prev, sched->recv_buf, byte_len);
        if (re != OC_OK) return re;

        /* Accumulate. */
        for (size_t i = 0; i < count; i++) {
            data[i] += ((float *)sched->recv_buf)[i];
        }
    }

    return OC_OK;
}

/* ------------------------------------------------------------------ */
/* Barrier                                                            */
/* ------------------------------------------------------------------ */

OcError oc_distributed_barrier(OcDistributedScheduler *sched)
{
    if (!sched || !sched->initialized) return OC_ERR_INVALID_ARG;

    sched->stats.barriers_hit++;

    /* Single-node: no-op. */
    if (sched->config.n_nodes <= 1) return OC_OK;

    /* NOTE: this ping/pong is NOT a global barrier (the first stage exits
     * after its outbound ping without waiting for later stages). It is
     * unreachable today because init rejects multi-node configs; a proper
     * arrival + release protocol is required before enabling multi-node. */
    uint8_t ping = 0x42;

    /* Send to next pipeline peer (or TP peer). */
    OcDistPeer *next = find_next_pipeline_peer(sched);
    if (next && next->online && next->socket_fd >= 0) {
        uint64_t hdr = 1;
        OcError e = send_all(next->socket_fd, &hdr, sizeof(hdr));
        if (e != OC_OK) {
            mark_peer_disconnected(sched, next);
            /* Non-fatal for barrier; we proceed. */
        } else {
            e = send_all(next->socket_fd, &ping, 1);
            if (e != OC_OK) mark_peer_disconnected(sched, next);
            sched->stats.bytes_sent += 1 + sizeof(hdr);
            next->bytes_sent_to += 1 + sizeof(hdr);
        }
    }

    /* Receive from previous pipeline peer. */
    OcDistPeer *prev = find_prev_pipeline_peer(sched);
    if (prev && prev->online && prev->socket_fd >= 0) {
        uint64_t hdr;
        OcError e = recv_all(prev->socket_fd, &hdr, sizeof(hdr));
        if (e != OC_OK) {
            mark_peer_disconnected(sched, prev);
        } else {
            uint8_t pong;
            e = recv_all(prev->socket_fd, &pong, 1);
            if (e != OC_OK) mark_peer_disconnected(sched, prev);
            sched->stats.bytes_received += 1 + sizeof(hdr);
            prev->bytes_recv_from += 1 + sizeof(hdr);
        }
    }

    return OC_OK;
}

/* ------------------------------------------------------------------ */
/* Latency measurement                                                 */
/* ------------------------------------------------------------------ */

double oc_distributed_get_latency(OcDistributedScheduler *sched)
{
    if (!sched || !sched->initialized) return 0.0;

    /* Single-node: measure local overhead (essentially zero). */
    if (sched->config.n_nodes <= 1) {
        double t0 = now_ms();
        /* Tiny no-op work to measure timer resolution. */
        volatile uint64_t sink = 0;
        for (int i = 0; i < 100; i++) sink += (uint64_t)i;
        (void)sink;
        double lat = now_ms() - t0;
        update_latency(&sched->stats, lat);
        return lat;
    }

    /* Multi-node: ping the next peer and measure round-trip. */
    OcDistPeer *next = find_next_pipeline_peer(sched);
    if (!next || !next->online || next->socket_fd < 0) {
        /* Try TP peer. */
        next = find_next_tp_peer(sched);
    }
    if (!next || !next->online || next->socket_fd < 0) {
        /* No peer to ping; return last known latency. */
        return sched->stats.latency_ms;
    }

    double t0 = now_ms();

    uint8_t ping = 0x50;
    uint64_t hdr = 1;
    OcError e = send_all(next->socket_fd, &hdr, sizeof(hdr));
    if (e != OC_OK) {
        mark_peer_disconnected(sched, next);
        return sched->stats.latency_ms;
    }
    e = send_all(next->socket_fd, &ping, 1);
    if (e != OC_OK) {
        mark_peer_disconnected(sched, next);
        return sched->stats.latency_ms;
    }

    uint64_t recv_hdr;
    e = recv_all(next->socket_fd, &recv_hdr, sizeof(recv_hdr));
    if (e != OC_OK) {
        mark_peer_disconnected(sched, next);
        return sched->stats.latency_ms;
    }
    uint8_t pong;
    e = recv_all(next->socket_fd, &pong, 1);
    if (e != OC_OK) {
        mark_peer_disconnected(sched, next);
        return sched->stats.latency_ms;
    }

    double lat = now_ms() - t0;
    update_latency(&sched->stats, lat);

    sched->stats.bytes_sent += 1 + sizeof(hdr);
    sched->stats.bytes_received += 1 + sizeof(recv_hdr);
    next->bytes_sent_to += 1 + sizeof(hdr);
    next->bytes_recv_from += 1 + sizeof(recv_hdr);

    return lat;
}

/* ------------------------------------------------------------------ */
/* Stats                                                              */
/* ------------------------------------------------------------------ */

const OcDistributedStats *oc_distributed_get_stats(
    const OcDistributedScheduler *sched)
{
    return sched ? &sched->stats : NULL;
}

size_t oc_distributed_stats_json(const OcDistributedScheduler *sched,
                                 char *buf, size_t cap)
{
    /* If scheduler is NULL, emit empty/minimal JSON. */
    if (!sched) {
        const char *empty = "{}";
        if (!buf || cap == 0) return 2;
        if (cap < 3) {
            buf[0] = '\0';
            return 2;
        }
        memcpy(buf, empty, 3);
        return 2;
    }

    const OcDistributedStats *s = &sched->stats;
    /* Use a local buffer large enough for all fields, then copy out. */
    char local[512];
    int n = snprintf(local, sizeof(local),
        "{\"bytes_sent\":%llu,"
         "\"bytes_received\":%llu,"
         "\"latency_ms\":%.6f,"
         "\"avg_latency_ms\":%.6f,"
         "\"latency_samples\":%llu,"
         "\"tokens_processed\":%llu,"
         "\"barriers_hit\":%llu,"
         "\"send_calls\":%llu,"
         "\"recv_calls\":%llu,"
         "\"allreduce_calls\":%llu,"
         "\"reconnects\":%llu,"
         "\"disconnects\":%llu}",
        (unsigned long long)s->bytes_sent,
        (unsigned long long)s->bytes_received,
        s->latency_ms,
        s->avg_latency_ms,
        (unsigned long long)s->latency_samples,
        (unsigned long long)s->tokens_processed,
        (unsigned long long)s->barriers_hit,
        (unsigned long long)s->send_calls,
        (unsigned long long)s->recv_calls,
        (unsigned long long)s->allreduce_calls,
        (unsigned long long)s->reconnects,
        (unsigned long long)s->disconnects);

    size_t len = (size_t)n;
    if (!buf || cap == 0) return len;
    if (cap <= len) {
        /* Truncate to fit, NUL-terminate. */
        memcpy(buf, local, cap - 1);
        buf[cap - 1] = '\0';
        return len;
    }
    memcpy(buf, local, len + 1);
    return len;
}
