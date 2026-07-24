/*
 * distributed.h — distributed inference scheduler for pipeline and tensor
 * parallelism across multiple nodes.
 *
 * This module coordinates model inference across multiple nodes using:
 *   - Pipeline parallelism: split model layers across nodes; forward pass
 *     sends hidden-state activations between consecutive pipeline stages.
 *   - Tensor parallelism:   split weight matrices across nodes; all-reduce
 *     after each matmul to produce the full output.
 *
 * Communication is TCP-based, reusing the socket helpers from mesh.h when
 * available. Single-node operation (n_nodes == 1) requires no network I/O
 * and all communication functions become no-ops.
 *
 * Design goals:
 *   - Zero-overhead single-node fast path (no sockets, no threads).
 *   - Graceful reconnection on peer disconnect.
 *   - Per-call stats tracking (bytes, latency, tokens).
 */
#ifndef OXIDIZE_DISTRIBUTED_H
#define OXIDIZE_DISTRIBUTED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Limits                                                             */
/* ------------------------------------------------------------------ */

#define OC_DIST_MAX_NODES       64   /* max nodes in a distributed job */
#define OC_DIST_MAX_ADDR        256  /* max "host:port" string length   */
#define OC_DIST_BUF_SIZE        (1u << 20) /* 1 MiB send/recv buffer    */
#define OC_DIST_MAX_PIPELINE    32   /* max pipeline stages            */
#define OC_DIST_MAX_TP          16   /* max tensor-parallel degree     */

/* ------------------------------------------------------------------ */
/* Config                                                             */
/* ------------------------------------------------------------------ */

/*
 * Configuration for a distributed inference scheduler instance.
 *
 *   n_nodes              — total number of nodes participating.
 *   node_rank            — this node's rank in [0, n_nodes).
 *   pipeline_stages      — number of pipeline-parallel stages (>= 1).
 *   tensor_parallel_size  — tensor-parallel degree (>= 1).
 *   pipeline_rank         — this node's pipeline stage index.
 *   tensor_rank           — this node's tensor-parallel rank within a stage.
 *   listen_port           — TCP port to listen on for incoming connections.
 *   coordinator_addr      — "host:port" of the pipeline master (may be NULL
 *                           when this node IS the master).
 *   activation_dtype_size — sizeof the activation element (e.g. 4 for float).
 */
typedef struct OcDistributedConfig {
    uint32_t    n_nodes;
    uint32_t    node_rank;
    uint32_t    pipeline_stages;
    uint32_t    tensor_parallel_size;
    uint32_t    pipeline_rank;
    uint32_t    tensor_rank;
    uint16_t    listen_port;
    const char *coordinator_addr;
    uint32_t    activation_dtype_size;
} OcDistributedConfig;

/* Sensible defaults for a single-node deployment (no communication). */
#define OC_DISTRIBUTED_CONFIG_DEFAULT ((OcDistributedConfig){ \
    .n_nodes              = 1,    \
    .node_rank            = 0,    \
    .pipeline_stages      = 1,    \
    .tensor_parallel_size = 1,    \
    .pipeline_rank        = 0,    \
    .tensor_rank          = 0,    \
    .listen_port          = 0,    \
    .coordinator_addr     = NULL, \
    .activation_dtype_size = 4,   \
})

/* ------------------------------------------------------------------ */
/* Node role                                                          */
/* ------------------------------------------------------------------ */

typedef enum {
    OC_NODE_ROLE_NONE            = 0,
    OC_NODE_ROLE_PIPELINE_MASTER = 1, /* first pipeline stage, drives the run */
    OC_NODE_ROLE_PIPELINE_WORKER = 2, /* intermediate or final pipeline stage  */
    OC_NODE_ROLE_TENSOR_PARALLEL = 3, /* participates in TP within a stage   */
} OcNodeRole;

/* ------------------------------------------------------------------ */
/* Peer descriptor                                                    */
/* ------------------------------------------------------------------ */

typedef struct OcDistPeer {
    uint32_t rank;                       /* global node rank               */
    char     addr[OC_DIST_MAX_ADDR];     /* "host:port"                    */
    OcNodeRole role;
    int       socket_fd;                 /* TCP socket, or -1 if offline   */
    bool      online;
    uint64_t  bytes_sent_to;             /* bytes sent to this peer         */
    uint64_t  bytes_recv_from;           /* bytes received from this peer   */
} OcDistPeer;

/* ------------------------------------------------------------------ */
/* Stats                                                              */
/* ------------------------------------------------------------------ */

typedef struct OcDistributedStats {
    uint64_t bytes_sent;           /* total bytes sent over network        */
    uint64_t bytes_received;       /* total bytes received over network    */
    double   latency_ms;           /* last measured round-trip latency     */
    double   avg_latency_ms;       /* running average latency              */
    uint64_t latency_samples;      /* number of latency measurements       */
    uint64_t tokens_processed;     /* total tokens processed on this node  */
    uint64_t barriers_hit;         /* number of barrier synchronizations   */
    uint64_t send_calls;           /* number of send_activations calls     */
    uint64_t recv_calls;           /* number of recv_activations calls     */
    uint64_t allreduce_calls;      /* number of all_reduce calls           */
    uint64_t reconnects;           /* number of successful reconnects      */
    uint64_t disconnects;          /* number of peer disconnects detected  */
} OcDistributedStats;

/* ------------------------------------------------------------------ */
/* Scheduler                                                          */
/* ------------------------------------------------------------------ */

/*
 * The distributed scheduler holds the resolved config, peer table, TCP
 * sockets for inter-stage communication, reusable send/recv buffers, and
 * accumulated stats. Single-node deployments never open sockets.
 */
typedef struct OcDistributedScheduler {
    OcDistributedConfig config;
    OcNodeRole          role;
    bool                initialized;

    /* Peer table (indexed by node rank). */
    OcDistPeer peers[OC_DIST_MAX_NODES];
    uint32_t   n_peers;

    /* TCP listen socket for accepting incoming pipeline-stage connections
     * (pipeline master and intermediate stages). -1 if not listening. */
    int listen_fd;

    /* Pre-allocated communication buffers (lazily allocated on first use). */
    void  *send_buf;
    void  *recv_buf;
    size_t buf_capacity;

    /* Running stats. */
    OcDistributedStats stats;
} OcDistributedScheduler;

/* ------------------------------------------------------------------ */
/* Lifecycle                                                          */
/* ------------------------------------------------------------------ */

/*
 * Initialize a distributed scheduler from `cfg`. In single-node mode
 * (n_nodes == 1), no sockets are opened and all communication functions
 * are no-ops that return OC_OK.
 *
 * Multi-node mode (n_nodes > 1) is NOT yet supported: peer endpoint
 * configuration, connection acceptance, and rank handshakes are not
 * implemented, so init rejects such configs with OC_ERR_NETWORK rather
 * than advertising a scheduler whose peers can never communicate.
 *
 * Returns:
 *   OC_OK              — scheduler ready.
 *   OC_ERR_INVALID_ARG — NULL scheduler or invalid config (see below).
 *   OC_ERR_NETWORK     — multi-node config (unsupported, see above).
 *
 * Config validation rules:
 *   - n_nodes >= 1 && n_nodes <= OC_DIST_MAX_NODES
 *   - node_rank < n_nodes
 *   - pipeline_stages >= 1 && pipeline_stages <= n_nodes
 *   - tensor_parallel_size >= 1
 *   - pipeline_stages * tensor_parallel_size <= n_nodes (or n_nodes == 1)
 *   - pipeline_rank < pipeline_stages
 *   - tensor_rank < tensor_parallel_size
 *   - activation_dtype_size > 0
 *   - multi-node, node_rank < pipeline_stages * tensor_parallel_size:
 *     node_rank == pipeline_rank * tensor_parallel_size + tensor_rank
 *     (nodes at or beyond the grid are unused and only need valid bounds)
 */
OcError oc_distributed_init(OcDistributedScheduler *sched,
                            const OcDistributedConfig *cfg);

/*
 * Release all resources held by the scheduler: close sockets, free buffers.
 * Safe to call on a zeroed struct or after a failed init. Zeroes the struct.
 */
void oc_distributed_free(OcDistributedScheduler *sched);

/* ------------------------------------------------------------------ */
/* Pipeline parallelism                                               */
/* ------------------------------------------------------------------ */

/*
 * Send hidden-state activations to the next pipeline stage.
 *
 *   sched — initialized scheduler.
 *   data  — pointer to activation buffer (hidden states).
 *   count — number of elements (NOT bytes); element size is
 *           config.activation_dtype_size.
 *
 * On the last pipeline stage (or single-node), this is a no-op returning
 * OC_OK. The bytes_sent stat is incremented by count * dtype_size.
 *
 * Returns OC_OK, OC_ERR_INVALID_ARG, or OC_ERR_NETWORK on send failure.
 */
OcError oc_distributed_send_activations(OcDistributedScheduler *sched,
                                        const void *data, size_t count);

/*
 * Receive hidden-state activations from the previous pipeline stage.
 *
 *   sched — initialized scheduler.
 *   out   — output buffer (caller-allocated, must hold `count` elements).
 *   count — number of elements expected.
 *
 * On the first pipeline stage (or single-node), this is a no-op returning
 * OC_OK without writing to `out` (the caller already has the input).
 * The bytes_received stat is incremented.
 *
 * Returns OC_OK, OC_ERR_INVALID_ARG, or OC_ERR_NETWORK on recv failure.
 */
OcError oc_distributed_recv_activations(OcDistributedScheduler *sched,
                                        void *out, size_t count);

/* ------------------------------------------------------------------ */
/* Tensor parallelism                                                 */
/* ------------------------------------------------------------------ */

/*
 * All-reduce: sum `data` (in-place) across all tensor-parallel peers.
 *
 *   sched — initialized scheduler.
 *   data  — float array, modified in-place to hold the reduced result.
 *   count — number of float elements.
 *
 * In single-node or TP=1 mode, this is a no-op (data is already the full
 * result). Both bytes_sent and bytes_received are incremented.
 *
 * Returns OC_OK, OC_ERR_INVALID_ARG, or OC_ERR_NETWORK.
 */
OcError oc_distributed_all_reduce(OcDistributedScheduler *sched,
                                  float *data, size_t count);

/* ------------------------------------------------------------------ */
/* Synchronization                                                    */
/* ------------------------------------------------------------------ */

/*
 * Barrier: block until all nodes have called this. In single-node mode,
 * returns immediately. Increments barriers_hit stat.
 *
 * Returns OC_OK, OC_ERR_INVALID_ARG, or OC_ERR_NETWORK.
 */
OcError oc_distributed_barrier(OcDistributedScheduler *sched);

/* ------------------------------------------------------------------ */
/* Latency measurement                                                 */
/* ------------------------------------------------------------------ */

/*
 * Measure round-trip communication latency in milliseconds. Performs a
 * small ping/pong with the next peer (or self-loop in single-node mode).
 * Updates stats.latency_ms and stats.avg_latency_ms.
 *
 * Returns the measured latency, or 0.0 on error / single-node.
 */
double oc_distributed_get_latency(OcDistributedScheduler *sched);

/* ------------------------------------------------------------------ */
/* Stats access                                                       */
/* ------------------------------------------------------------------ */

/* Returns a pointer to the live stats struct, or NULL if NULL scheduler. */
const OcDistributedStats *oc_distributed_get_stats(
    const OcDistributedScheduler *sched);

/*
 * Format stats as a JSON string into `buf` (up to `cap-1` chars, NUL-terminated).
 * Returns the number of bytes written (excluding NUL). If `buf` is NULL or
 * cap==0, returns the length that would have been written.
 *
 * Output example:
 *   {"bytes_sent":1024,"bytes_received":512,"latency_ms":0.05,
 *    "avg_latency_ms":0.05,"latency_samples":1,"tokens_processed":10,
 *    "barriers_hit":1,"send_calls":2,"recv_calls":2,
 *    "allreduce_calls":0,"reconnects":0,"disconnects":0}
 */
size_t oc_distributed_stats_json(const OcDistributedScheduler *sched,
                                 char *buf, size_t cap);

/* ------------------------------------------------------------------ */
/* Internals exposed for testing                                      */
/* ------------------------------------------------------------------ */

/* Validate a config struct without initializing a scheduler.
 * Returns OC_OK if valid, OC_ERR_INVALID_ARG otherwise. */
OcError oc_distributed_validate_config(const OcDistributedConfig *cfg);

/* Resolve this node's role from the config. */
OcNodeRole oc_distributed_resolve_role(const OcDistributedConfig *cfg);

/* Attempt to reconnect to an offline peer. Returns OC_OK on success. */
OcError oc_distributed_reconnect(OcDistributedScheduler *sched,
                                 uint32_t peer_rank);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_DISTRIBUTED_H */
