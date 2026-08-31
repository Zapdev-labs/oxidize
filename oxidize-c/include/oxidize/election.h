/* election.h — leader election protocol for distributed inference clusters. */
#ifndef OXIDIZE_ELECTION_H
#define OXIDIZE_ELECTION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OC_ELECTION_DEFAULT_TIMEOUT_MS      5000u
#define OC_ELECTION_DEFAULT_HEARTBEAT_MS    1000u
#define OC_ELECTION_DEFAULT_NODE_PRIORITY   0u


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


/* Initialize config with defaults. */
OcError oc_election_config_init(OcElectionConfig *cfg);


/* Allocate a new election state for node `self_id`. `config` may be NULL
 * (defaults are used). Free with oc_election_free. */
OcError oc_election_init(const OcElectionConfig *config, uint64_t self_id,
                         OcElectionState **out);

/* Free all owned storage and reset state. Safe on NULL / already-freed. */
void oc_election_free(OcElectionState *state);


/* Periodic tick. If the follower/candidate election timeout has expired (current_ms - last_heartbeat_ms >= election_timeout_ms), transition to CANDIDATE, increment current_term, vote for self, and reset votes. Leaders are unaffected (they send heartbeats via a separate transport). */
OcError oc_election_tick(OcElectionState *state, uint64_t current_ms);


/* Handle a vote request from `candidate_id` at `term`. Grants the vote if term >= current_term, voted_for is 0 or already candidate_id, and candidate_priority > self_priority or equal with candidate_id > self_id. Returns true via *granted. */
OcError oc_election_request_vote(OcElectionState *state,
                                 uint64_t candidate_id, uint64_t term,
                                 bool *granted);

/* Handle a vote response from `voter_id` at `term`. */
OcError oc_election_receive_vote(OcElectionState *state,
                                uint64_t voter_id, uint64_t term,
                                bool granted);


/* Received a heartbeat from `leader_id` at `term`. If term >= current_term,
 * adopt the term, step down to FOLLOWER (if not already), record leader_id,
 * and refresh last_heartbeat_ms. Returns OC_OK on success. */
OcError oc_election_heartbeat(OcElectionState *state,
                              uint64_t leader_id, uint64_t term);


/* Transition to LEADER. Sets role=LEADER, leader_id=self_id,
 * refreshes last_heartbeat_ms (so the new leader doesn't immediately
 * time out). Resets n_votes_received. */
OcError oc_election_become_leader(OcElectionState *state);


OcElectionRole oc_election_get_role(const OcElectionState *state);
uint64_t       oc_election_get_leader(const OcElectionState *state);
uint64_t       oc_election_get_term(const OcElectionState *state);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ELECTION_H */
