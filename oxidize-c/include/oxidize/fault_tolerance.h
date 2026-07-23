/*
 * fault_tolerance.h — Mesh fault tolerance: heartbeat monitoring, failure
 * detection, and failover.
 *
 * Each monitored node periodically emits heartbeats. The fault tolerance
 * manager tracks the last heartbeat timestamp per node and, on each tick,
 * flags nodes whose heartbeats are older than the configured timeout as
 * SUSPECT, then DEAD after `max_missed` consecutive missed heartbeats.
 * Recovered nodes are explicitly promoted back to RECOVERING -> ALIVE.
 *
 * Transport-agnostic: callers are responsible for shipping heartbeat
 * messages and invoking oc_ft_tick periodically.
 */
#ifndef OXIDIZE_FAULT_TOLERANCE_H
#define OXIDIZE_FAULT_TOLERANCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ──────────────────────────────────────────────────────── */

#define OC_FT_MAX_NODES        64u
#define OC_FT_DEFAULT_INTERVAL_MS  1000u
#define OC_FT_DEFAULT_TIMEOUT_MS   5000u
#define OC_FT_DEFAULT_MAX_MISSED   3u
#define OC_FT_DEFAULT_RECOVERY_MS  10000u

/* ─── Types ─────────────────────────────────────────────────────────── */

/* Node lifecycle status. Values are stable for serialization. */
typedef enum {
    OC_FT_ALIVE      = 0,
    OC_FT_SUSPECT    = 1,
    OC_FT_DEAD       = 2,
    OC_FT_RECOVERING = 3,
} OcFtStatus;

typedef struct OcFtConfig {
    uint32_t heartbeat_interval_ms;  /* default OC_FT_DEFAULT_INTERVAL_MS  */
    uint32_t heartbeat_timeout_ms;   /* default OC_FT_DEFAULT_TIMEOUT_MS   */
    uint32_t max_missed;             /* default OC_FT_DEFAULT_MAX_MISSED   */
    uint32_t recovery_timeout_ms;    /* default OC_FT_DEFAULT_RECOVERY_MS  */
} OcFtConfig;

typedef struct OcFtNodeState {
    uint64_t    node_id;
    uint64_t    last_heartbeat_ms;  /* ms since epoch                       */
    uint32_t    missed_count;       /* consecutive missed heartbeats      */
    OcFtStatus  status;
} OcFtNodeState;

typedef struct OcFtManager {
    OcFtConfig    config;
    OcFtNodeState nodes[OC_FT_MAX_NODES];
    uint32_t      n_nodes;
    uint64_t      self_id;
} OcFtManager;

/* ─── Config helpers ────────────────────────────────────────────────── */

/* Initialize config with defaults. */
OcError oc_ft_config_init(OcFtConfig *cfg);

/* ─── Lifecycle ─────────────────────────────────────────────────────── */

/* Allocate a fault tolerance manager for the local node `self_id`.
 * `config` may be NULL (defaults are used). The returned manager owns a
 * heap allocation; free with oc_ft_free. */
OcError oc_ft_init(const OcFtConfig *config, uint64_t self_id,
                   OcFtManager **out);

/* Free all owned storage and reset state. Safe on NULL / already-freed. */
void oc_ft_free(OcFtManager *mgr);

/* ─── Node management ──────────────────────────────────────────────── */

/* Add a node to monitor. If the node already exists, OC_OK is returned.
 * The node starts in ALIVE status with last_heartbeat_ms=0. */
OcError oc_ft_add_node(OcFtManager *mgr, uint64_t node_id);

/* Remove a monitored node. Removing `self_id` is an error. */
OcError oc_ft_remove_node(OcFtManager *mgr, uint64_t node_id);

/* ─── Heartbeats ────────────────────────────────────────────────────── */

/* Record a heartbeat from `node_id` at `current_ms`. Resets the missed
 * counter and promotes the node to ALIVE if it was previously SUSPECT or
 * RECOVERING. Returns OC_ERR_MODEL if the node is not monitored. */
OcError oc_ft_heartbeat(OcFtManager *mgr, uint64_t node_id,
                        uint64_t current_ms);

/* Periodic tick. For each node, computes elapsed = current_ms - last_heartbeat.
 * If elapsed > config.heartbeat_timeout_ms, increments missed_count and
 * promotes to SUSPECT. Once missed_count >= config.max_missed, the node is
 * marked DEAD. Returns OC_OK on success. */
OcError oc_ft_tick(OcFtManager *mgr, uint64_t current_ms);

/* ─── Queries ────────────────────────────────────────────────────────── */

/* Copy the state of `node_id` into `out_state`. Returns OC_ERR_MODEL if
 * the node is not monitored. */
OcError oc_ft_get_node_state(const OcFtManager *mgr, uint64_t node_id,
                             OcFtNodeState *out_state);

/* Count of nodes currently in ALIVE status. */
uint32_t oc_ft_get_alive_count(const OcFtManager *mgr);

/* Copy up to `max` DEAD node ids into `out_ids`. Sets *out_count to the
 * number written. Safe to pass NULL for out_ids to just count. */
OcError oc_ft_get_dead_nodes(const OcFtManager *mgr, uint64_t *out_ids,
                             uint32_t max, uint32_t *out_count);

/* ─── Recovery ──────────────────────────────────────────────────────── */

/* Mark `node_id` as RECOVERING. The node will be promoted to ALIVE on the
 * next heartbeat (or once recovery_timeout_ms elapses without further
 * failure, at the caller's discretion). Returns OC_ERR_MODEL if not
 * monitored. */
OcError oc_ft_recover(OcFtManager *mgr, uint64_t node_id);

/* Human-readable status name (e.g. "ALIVE"). Never NULL. */
const char *oc_ft_status_name(OcFtStatus status);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_FAULT_TOLERANCE_H */
