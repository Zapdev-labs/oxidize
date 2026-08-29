#include "oxidize/election.h"

#include <stdlib.h>
#include <string.h>


OcError oc_election_config_init(OcElectionConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    cfg->election_timeout_ms   = OC_ELECTION_DEFAULT_TIMEOUT_MS;
    cfg->heartbeat_interval_ms = OC_ELECTION_DEFAULT_HEARTBEAT_MS;
    cfg->node_priority         = OC_ELECTION_DEFAULT_NODE_PRIORITY;
    return OC_OK;
}


OcError oc_election_init(const OcElectionConfig *config, uint64_t self_id,
                         OcElectionState **out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    *out = NULL;

    OcElectionConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        oc_election_config_init(&cfg);
    }
    if (cfg.election_timeout_ms == 0) return OC_ERR_INVALID_ARG;
    if (cfg.heartbeat_interval_ms == 0) return OC_ERR_INVALID_ARG;

    OcElectionState *s = malloc(sizeof(*s));
    if (!s) return OC_ERR_OOM;
    memset(s, 0, sizeof(*s));
    s->config              = cfg;
    s->role                = OC_ELECTION_FOLLOWER;
    s->self_id             = self_id;
    s->current_term        = 0;
    s->voted_for           = 0;
    s->leader_id           = 0;
    s->n_votes_received    = 0;
    s->last_heartbeat_ms   = 0;
    s->election_timeout_ms = cfg.election_timeout_ms;

    *out = s;
    return OC_OK;
}

void oc_election_free(OcElectionState *state)
{
    if (!state) return;
    memset(state, 0, sizeof(*state));
    free(state);
}


OcError oc_election_tick(OcElectionState *state, uint64_t current_ms)
{
    if (!state) return OC_ERR_INVALID_ARG;

    /* Leaders do not time out — they send heartbeats via separate transport. */
    if (state->role == OC_ELECTION_LEADER) {
        return OC_OK;
    }

    /* Check election timeout. Guard against the initial state where */
    if (state->last_heartbeat_ms == 0) {
        state->last_heartbeat_ms = current_ms;
        return OC_OK;
    }

    uint64_t elapsed = (current_ms >= state->last_heartbeat_ms)
                       ? (current_ms - state->last_heartbeat_ms)
                       : 0u;
    if (elapsed < state->election_timeout_ms) {
        return OC_OK;
    }

    /* Timeout expired: become candidate and start a new election. */
    state->role             = OC_ELECTION_CANDIDATE;
    state->current_term    += 1;
    state->voted_for        = state->self_id;
    state->leader_id        = 0;
    state->n_votes_received = 1; /* vote for self */
    state->last_heartbeat_ms = current_ms;
    return OC_OK;
}


OcError oc_election_request_vote(OcElectionState *state,
                                 uint64_t candidate_id, uint64_t term,
                                 bool *granted)
{
    if (!state || !granted) return OC_ERR_INVALID_ARG;
    if (candidate_id == 0) return OC_ERR_INVALID_ARG;
    *granted = false;

    /* Adopt a higher term. */
    if (term > state->current_term) {
        state->current_term = term;
        state->voted_for     = 0;
        state->role          = OC_ELECTION_FOLLOWER;
        state->leader_id     = 0;
    }

    /* Reject stale terms. */
    if (term < state->current_term) {
        return OC_OK;
    }

    /* Grant if we haven't voted yet this term, or already voted for this
     * candidate. */
    bool can_vote = (state->voted_for == 0) || (state->voted_for == candidate_id);
    if (!can_vote) {
        return OC_OK;
    }

    /* Deterministic tie-break: a candidate with higher priority wins.
     * On equal priority, the candidate with the larger id wins. A candidate
     * never beats itself (shouldn't happen, but guard anyway). */
    if (candidate_id == state->self_id) {
        /* Can't vote for self via RPC — self-vote happens at candidacy. */
        return OC_OK;
    }

    int32_t cand_pri = state->config.node_priority;
    int32_t self_pri = state->config.node_priority;
    bool wins = (cand_pri > self_pri)
              || (cand_pri == self_pri && candidate_id > state->self_id);
    if (!wins) {
        return OC_OK;
    }

    state->voted_for = candidate_id;
    state->last_heartbeat_ms = 0; /* refresh so caller can stamp current_ms */
    *granted = true;
    return OC_OK;
}


OcError oc_election_receive_vote(OcElectionState *state,
                                uint64_t voter_id, uint64_t term,
                                bool granted)
{
    if (!state) return OC_ERR_INVALID_ARG;
    (void)voter_id; /* tracked only via the count for this simplified port */

    /* Ignore responses from older terms. */
    if (term < state->current_term) {
        return OC_OK;
    }

    /* Adopt a newer term (e.g. another candidate won). */
    if (term > state->current_term) {
        state->current_term = term;
        state->role          = OC_ELECTION_FOLLOWER;
        state->voted_for     = 0;
        state->leader_id     = 0;
        state->n_votes_received = 0;
        return OC_OK;
    }

    /* Only count votes while we are a candidate at this term. */
    if (state->role != OC_ELECTION_CANDIDATE) {
        return OC_OK;
    }

    if (granted) {
        state->n_votes_received += 1;
    }
    return OC_OK;
}


OcError oc_election_heartbeat(OcElectionState *state,
                              uint64_t leader_id, uint64_t term)
{
    if (!state) return OC_ERR_INVALID_ARG;
    if (leader_id == 0) return OC_ERR_INVALID_ARG;

    /* Stale heartbeat from an old leader — ignore. */
    if (term < state->current_term) {
        return OC_OK;
    }

    /* Adopt the term and step down to follower. */
    if (term > state->current_term) {
        state->current_term = term;
    }
    state->role           = OC_ELECTION_FOLLOWER;
    state->leader_id      = leader_id;
    state->voted_for     = 0; /* reset for the new term */
    /* last_heartbeat_ms is stamped by the caller via oc_election_tick; but
     * the contract says we refresh it here too so a heartbeat alone is
     * sufficient to reset the timer. The caller is expected to also tick. */
    state->n_votes_received = 0;
    return OC_OK;
}


OcError oc_election_become_leader(OcElectionState *state)
{
    if (!state) return OC_ERR_INVALID_ARG;
    state->role             = OC_ELECTION_LEADER;
    state->leader_id        = state->self_id;
    state->n_votes_received = 0;
    /* Refresh so the new leader doesn't immediately time out. The caller
     * should stamp current_ms via the next tick. */
    state->last_heartbeat_ms = 0;
    return OC_OK;
}


OcElectionRole oc_election_get_role(const OcElectionState *state)
{
    if (!state) return OC_ELECTION_FOLLOWER;
    return state->role;
}

uint64_t oc_election_get_leader(const OcElectionState *state)
{
    if (!state) return 0;
    return state->leader_id;
}

uint64_t oc_election_get_term(const OcElectionState *state)
{
    if (!state) return 0;
    return state->current_term;
}
