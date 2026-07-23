/*
 * gossip.h — gossip protocol for cluster node discovery and health checking.
 *
 * A simple membership protocol inspired by oxidize-core/src/mesh/gossip.rs
 * and discovery.rs: each node maintains a list of known peers, periodically
 * refreshes their `last_seen` timestamps via `oc_gossip_tick`, and exchanges
 * peer lists with other nodes through `oc_gossip_serialize` /
 * `oc_gossip_deserialize` / `oc_gossip_merge`.
 *
 * Nodes whose `last_seen` is older than `config.timeout_ms` are marked
 * unhealthy (but retained) so callers can report on them. The protocol is
 * transport-agnostic: callers are responsible for shipping the serialized
 * peer list to a fanout of `config.fanout` peers each tick.
 */
#ifndef OXIDIZE_GOSSIP_H
#define OXIDIZE_GOSSIP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ──────────────────────────────────────────────────────── */

#define OC_GOSSIP_DEFAULT_INTERVAL_MS 1000u
#define OC_GOSSIP_DEFAULT_TIMEOUT_MS  5000u
#define OC_GOSSIP_DEFAULT_MAX_NODES   64u
#define OC_GOSSIP_DEFAULT_FANOUT      3u

#define OC_GOSSIP_MAX_ADDR_LEN    256u
#define OC_GOSSIP_MAX_METADATA    128u

/* ─── Types ─────────────────────────────────────────────────────────── */

typedef struct OcGossipNode {
    uint64_t id;
    char     addr[OC_GOSSIP_MAX_ADDR_LEN]; /* host (no port)               */
    uint16_t port;
    uint64_t last_seen;                    /* ms since epoch               */
    bool     healthy;
    char     metadata[OC_GOSSIP_MAX_METADATA]; /* free-form string tag    */
} OcGossipNode;

typedef struct OcGossipConfig {
    uint32_t interval_ms;  /* default OC_GOSSIP_DEFAULT_INTERVAL_MS        */
    uint32_t timeout_ms;   /* default OC_GOSSIP_DEFAULT_TIMEOUT_MS         */
    uint32_t max_nodes;   /* default OC_GOSSIP_DEFAULT_MAX_NODES           */
    uint32_t fanout;      /* default OC_GOSSIP_DEFAULT_FANOUT              */
} OcGossipConfig;

typedef struct OcGossipState {
    OcGossipConfig config;
    uint64_t       self_id;
    OcGossipNode  *nodes;     /* heap-allocated, capacity = config.max_nodes */
    uint32_t       n_nodes;   /* current count (always includes self)         */
    uint64_t       last_gossip_time;
} OcGossipState;

/* ─── Config helpers ────────────────────────────────────────────────── */

/* Initialize config with defaults. */
OcError oc_gossip_config_init(OcGossipConfig *cfg);

/* ─── Lifecycle ─────────────────────────────────────────────────────── */

/* Allocate a gossip state for the local node `self_id`.
 * `config` may be NULL (defaults are used). The returned state owns a heap
 * allocation; free with oc_gossip_free. */
OcError oc_gossip_init(const OcGossipConfig *config, uint64_t self_id,
                       OcGossipState **out);

/* Free all owned storage and reset state. Safe on NULL / already-freed. */
void oc_gossip_free(OcGossipState *state);

/* ─── Peer management ───────────────────────────────────────────────── */

/* Add (or refresh) a known peer. If a node with the same id already exists,
 * its addr/port/last_seen are updated and `OC_OK` is returned. */
OcError oc_gossip_add_node(OcGossipState *state, uint64_t id,
                           const char *addr, uint16_t port);

/* Remove a node by id. Removing `self_id` is an error. */
OcError oc_gossip_remove_node(OcGossipState *state, uint64_t id);

/* ─── Periodic update ──────────────────────────────────────────────── */

/* Periodic gossip tick. Records `current_time_ms` as the last gossip time,
 * refreshes self's last_seen, and marks any node whose last_seen is older
 * than `config.timeout_ms` as unhealthy. Returns OC_OK on success. */
OcError oc_gossip_tick(OcGossipState *state, uint64_t current_time_ms);

/* ─── Queries ──────────────────────────────────────────────────────── */

/* Copy up to `max` known nodes into `out_nodes`. Sets *out_count to the
 * number written. Safe to pass NULL for out_nodes to just count. */
OcError oc_gossip_get_nodes(const OcGossipState *state,
                            OcGossipNode *out_nodes, uint32_t max,
                            uint32_t *out_count);

/* Number of healthy nodes (including self if healthy). */
uint32_t oc_gossip_get_healthy_count(const OcGossipState *state);

/* ─── Peer exchange ────────────────────────────────────────────────── */

/* Merge a list of remote nodes received from a peer. Updates last_seen for
 * already-known peers (taking the more-recent timestamp) and appends new
 * peers up to `config.max_nodes`. The local node id is preserved. */
OcError oc_gossip_merge(OcGossipState *state, uint64_t current_time_ms,
                        const OcGossipNode *remote_nodes, uint32_t count);

/* Serialize the node list for network transmission. Writes a compact binary
 * representation into `out_buf` (capacity `cap`) and sets *out_len.
 * Format: u32 count, then `count` raw OcGossipNode records (fixed layout).
 * Returns OC_ERR_INVALID_ARG if buf is too small. */
OcError oc_gossip_serialize(const OcGossipState *state, uint8_t *out_buf,
                            size_t cap, size_t *out_len);

/* Deserialize a buffer produced by oc_gossip_serialize and merge it.
 * `current_time_ms` seeds last_seen for newly-learned nodes (since the
 * wire format carries each peer's own last_seen which is preserved). */
OcError oc_gossip_deserialize(OcGossipState *state, uint64_t current_time_ms,
                              const uint8_t *buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_GOSSIP_H */
