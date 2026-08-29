#ifndef OXIDIZE_DISTRIBUTED_H
#define OXIDIZE_DISTRIBUTED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Limits                                                             */

#define OC_DIST_MAX_NODES       64   /* max nodes in a distributed job */
#define OC_DIST_MAX_ADDR        256  /* max "host:port" string length   */
#define OC_DIST_BUF_SIZE        (1u << 20) /* 1 MiB send/recv buffer    */
#define OC_DIST_MAX_PIPELINE    32   /* max pipeline stages            */
#define OC_DIST_MAX_TP          16   /* max tensor-parallel degree     */

/* Config                                                             */

/* Configuration for a distributed inference scheduler instance. */
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
    /* Bound on how long oc_distributed_init() waits for peers to show up:
     * the master's accept loop and the worker's connect retry both give up
     * after this many milliseconds. 0 = OC_DIST_CONNECT_TIMEOUT_MS. */
    uint32_t    connect_timeout_ms;
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
    .connect_timeout_ms   = 0,    \
})

/* Default peer rendezvous timeout used when connect_timeout_ms == 0. */
#define OC_DIST_CONNECT_TIMEOUT_MS 30000u

/* Node role                                                          */

typedef enum {
    OC_NODE_ROLE_NONE            = 0,
    OC_NODE_ROLE_PIPELINE_MASTER = 1, /* first pipeline stage, drives the run */
    OC_NODE_ROLE_PIPELINE_WORKER = 2, /* intermediate or final pipeline stage  */
    OC_NODE_ROLE_TENSOR_PARALLEL = 3, /* participates in TP within a stage   */
} OcNodeRole;

/* Peer descriptor                                                    */

typedef struct OcDistPeer {
    uint32_t rank;                       /* global node rank               */
    char     addr[OC_DIST_MAX_ADDR];     /* "host:port"                    */
    OcNodeRole role;
    int       socket_fd;                 /* TCP socket, or -1 if offline   */
    bool      online;
    uint64_t  bytes_sent_to;             /* bytes sent to this peer         */
    uint64_t  bytes_recv_from;           /* bytes received from this peer   */
} OcDistPeer;

/* Stats                                                              */

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

/* Scheduler                                                          */

/* The distributed scheduler holds the resolved config, peer table, TCP */
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

/* Lifecycle                                                          */

/* Initialize a distributed scheduler from `cfg`. In single-node mode */
OcError oc_distributed_init(OcDistributedScheduler *sched,
                            const OcDistributedConfig *cfg);

/* Release all resources held by the scheduler: close sockets, free buffers. */
void oc_distributed_free(OcDistributedScheduler *sched);

/* Pipeline parallelism                                               */

/* Send hidden-state activations to the next pipeline stage. */
OcError oc_distributed_send_activations(OcDistributedScheduler *sched,
                                        const void *data, size_t count);

/* Receive hidden-state activations from the previous pipeline stage. */
OcError oc_distributed_recv_activations(OcDistributedScheduler *sched,
                                        void *out, size_t count);

/* Tensor parallelism                                                 */

/* All-reduce: sum `data` (in-place) across all tensor-parallel peers. */
OcError oc_distributed_all_reduce(OcDistributedScheduler *sched,
                                  float *data, size_t count);

/* Synchronization                                                    */

/* Barrier: block until all nodes have called this. In single-node mode, */
OcError oc_distributed_barrier(OcDistributedScheduler *sched);

/* Latency measurement                                                 */

/* Measure round-trip communication latency in milliseconds. */
double oc_distributed_get_latency(OcDistributedScheduler *sched);

/* Stats access                                                       */

/* Returns a pointer to the live stats struct, or NULL if NULL scheduler. */
const OcDistributedStats *oc_distributed_get_stats(
    const OcDistributedScheduler *sched);

/* Format stats as a JSON string into `buf` (up to `cap-1` chars, NUL-terminated). */
size_t oc_distributed_stats_json(const OcDistributedScheduler *sched,
                                 char *buf, size_t cap);

/* Internals exposed for testing                                      */

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
