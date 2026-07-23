/*
 * election.h — leader election protocol for distributed inference clusters.
 *
 * Raft-style leader election: nodes start as FOLLOWER, transition to
 * CANDIDATE when the election timeout expires without a heartbeat, request
 * votes from peers, and become LEADER once they win a majority of votes.
 * Leaders send periodic heartbeats to maintain authority.
 *
 * Port of the conceptual protocol from oxidize-core/src/mesh/election.rs
 * (the Rust port uses a bully-style deterministic election; this C port
 * follows the API specified by the task brief: Raft-style term/vote/heartbeat).
 */
#ifndef OXIDIZE_ELECTION_H
#define OXIDIZE_ELECTION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ──────────────────────────────────────────────────────── */

#define OC_ELECTION_DEFAULT_TIMEOUT_MS      5000u
#define OC_ELECTION_DEFAULT_HEARTBEAT_MS    1000u
#define OC_ELECTION_DEFAULT_NODE_PRIORITY   0u

/* ─── Types ─────────────────────────────────────────────────────────── */

typedef enum {
    OC_ELECTION_FOLLOWER  = 0,
    OC_ELECTION_CANDIDATE  = 1,
    OC_ELECTION_LEADER     = 2,
} OcElectionRole;

typedef struct OcElectionConfig {
    uint32_t election_timeout_ms;     /* default OC_ELECTION_DEFAULT_TIMEOUT_MS   */
    uint32_t heartbeat_interval_ms;   /* default OC_ELECTION_DEFAULT_HEARTBEAT_MS*/
    int32_t  node_priority;          /* default OC_ELECTION_DEFAULT_NODE_PRIORITY*/
} OcElectionConfig;

typedef struct OcElectionState {
    OcElectionConfig config;
    OcElectionRole   role;
    uint64_t          self_id;
    uint64_t          current_term;
    uint64_t          voted_for;          /* 0 = none                       */
    uint64_t          leader_id;          /* 0 = none                        */
    uint32_t          n_votes_received;
    uint64_t          last_heartbeat_ms;
    uint64_t          election_timeout_ms;
} OcElectionState;

/* ─── Config helpers ────────────────────────────────────────────────── */

/* Initialize config with defaults. */
OcError oc_election_config_init(OcElectionConfig *cfg);

/* ─── Lifecycle ─────────────────────────────────────────────────────── */

/* Allocate a new election state for node `self_id`. `config` may be NULL
 * (defaults are used). Free with oc_election_free. */
OcError oc_election_init(const OcElectionConfig *config, uint64_t self_id,
                         OcElectionState **out);

/* Free all owned storage and reset state. Safe on NULL / already-freed. */
void oc_election_free(OcElectionState *state);

/* ─── Periodic tick ─────────────────────────────────────────────────── */

/* Periodic tick. If the follower/candidate election timeout has expired
 * (current_ms - last_heartbeat_ms >= election_timeout_ms), transition to
 * CANDIDATE, increment current_term, vote for self, and reset votes.
 * Leaders are unaffected (they send heartbeats via a separate transport). */
OcError oc_election_tick(OcElectionState *state, uint64_t current_ms);

/* ─── Vote RPC ──────────────────────────────────────────────────────── */

/* Handle a vote request from `candidate_id` at `term`. Grants the vote if:
 *   - term >= current_term (adopt higher term),
 *   - voted_for is 0 (none) or already candidate_id,
 *   - candidate_priority > self_priority OR (equal AND candidate_id > self_id)
 *     — deterministic tie-break so two candidates never deadlock.
 * Returns true via *granted. Resets last_heartbeat_ms on a granted vote. */
OcError oc_election_request_vote(OcElectionState *state,
                                 uint64_t candidate_id, uint64_t term,
                                 bool *granted);

/* Handle a vote response from `voter_id` at `term`. If we are CANDIDATE,
 * term matches, and the vote was for us (granted=true), increment
 * n_votes_received. If n_votes_received reaches majority (n_peers/2 + 1),
 * the caller separately transitions via oc_election_become_leader.
 * Here we accept the count and let the caller decide. */
OcError oc_election_receive_vote(OcElectionState *state,
                                uint64_t voter_id, uint64_t term,
                                bool granted);

/* ─── Heartbeat ─────────────────────────────────────────────────────── */

/* Received a heartbeat from `leader_id` at `term`. If term >= current_term,
 * adopt the term, step down to FOLLOWER (if not already), record leader_id,
 * and refresh last_heartbeat_ms. Returns OC_OK on success. */
OcError oc_election_heartbeat(OcElectionState *state,
                              uint64_t leader_id, uint64_t term);

/* ─── Role transitions ──────────────────────────────────────────────── */

/* Transition to LEADER. Sets role=LEADER, leader_id=self_id,
 * refreshes last_heartbeat_ms (so the new leader doesn't immediately
 * time out). Resets n_votes_received. */
OcError oc_election_become_leader(OcElectionState *state);

/* ─── Queries ───────────────────────────────────────────────────────── */

OcElectionRole oc_election_get_role(const OcElectionState *state);
uint64_t       oc_election_get_leader(const OcElectionState *state);
uint64_t       oc_election_get_term(const OcElectionState *state);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ELECTION_H */
