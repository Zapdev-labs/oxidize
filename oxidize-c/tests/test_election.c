#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>
#include <string.h>
#include "oxidize/election.h"


Test(election, config_init_defaults)
{
    OcElectionConfig cfg;
    cr_assert_eq(oc_election_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.election_timeout_ms,   OC_ELECTION_DEFAULT_TIMEOUT_MS);
    cr_assert_eq(cfg.heartbeat_interval_ms, OC_ELECTION_DEFAULT_HEARTBEAT_MS);
    cr_assert_eq(cfg.node_priority,        OC_ELECTION_DEFAULT_NODE_PRIORITY);
}

Test(election, config_init_null)
{
    cr_assert_eq(oc_election_config_init(NULL), OC_ERR_INVALID_ARG);
}


Test(election, init_creates_follower)
{
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(NULL, 42, &s), OC_OK);
    cr_assert_not_null(s);
    cr_assert_eq(s->self_id, 42u);
    cr_assert_eq(s->role, OC_ELECTION_FOLLOWER);
    cr_assert_eq(s->current_term, 0u);
    cr_assert_eq(s->voted_for, 0u);
    cr_assert_eq(s->leader_id, 0u);
    cr_assert_eq(s->n_votes_received, 0u);
    oc_election_free(s);
}

Test(election, init_with_custom_config)
{
    OcElectionConfig cfg;
    oc_election_config_init(&cfg);
    cfg.election_timeout_ms   = 2000;
    cfg.heartbeat_interval_ms = 500;
    cfg.node_priority         = 7;
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(&cfg, 1, &s), OC_OK);
    cr_assert_eq(s->config.election_timeout_ms,   2000u);
    cr_assert_eq(s->config.heartbeat_interval_ms, 500u);
    cr_assert_eq(s->config.node_priority,         7);
    cr_assert_eq(s->election_timeout_ms, 2000u);
    oc_election_free(s);
}

Test(election, init_zero_timeout_fails)
{
    OcElectionConfig cfg;
    oc_election_config_init(&cfg);
    cfg.election_timeout_ms = 0;
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(&cfg, 1, &s), OC_ERR_INVALID_ARG);
    cr_assert_null(s);
}

Test(election, init_zero_heartbeat_fails)
{
    OcElectionConfig cfg;
    oc_election_config_init(&cfg);
    cfg.heartbeat_interval_ms = 0;
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(&cfg, 1, &s), OC_ERR_INVALID_ARG);
    cr_assert_null(s);
}

Test(election, free_null_is_safe)
{
    oc_election_free(NULL);
    cr_assert(true);
}


Test(election, tick_first_stamp_sets_heartbeat)
{
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_election_tick(s, 100), OC_OK);
    cr_assert_eq(s->last_heartbeat_ms, 100u);
    cr_assert_eq(s->role, OC_ELECTION_FOLLOWER);
    oc_election_free(s);
}

Test(election, tick_triggers_election_on_timeout)
{
    OcElectionConfig cfg;
    oc_election_config_init(&cfg);
    cfg.election_timeout_ms = 100;
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(&cfg, 5, &s), OC_OK);
    /* First tick stamps the heartbeat. */
    cr_assert_eq(oc_election_tick(s, 0), OC_OK);
    cr_assert_eq(s->last_heartbeat_ms, 0u);
    /* Tick before timeout — still follower. */
    cr_assert_eq(oc_election_tick(s, 50), OC_OK);
    cr_assert_eq(s->role, OC_ELECTION_FOLLOWER);
    /* Tick after timeout — becomes candidate. */
    cr_assert_eq(oc_election_tick(s, 200), OC_OK);
    cr_assert_eq(s->role, OC_ELECTION_CANDIDATE);
    cr_assert_eq(s->current_term, 1u);
    cr_assert_eq(s->voted_for, 5u);
    cr_assert_eq(s->n_votes_received, 1u);
    oc_election_free(s);
}

Test(election, tick_leader_does_not_timeout)
{
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_election_become_leader(s), OC_OK);
    cr_assert_eq(oc_election_tick(s, 100000), OC_OK);
    cr_assert_eq(s->role, OC_ELECTION_LEADER);
    oc_election_free(s);
}


Test(election, request_vote_granted_for_higher_priority)
{
    OcElectionConfig cfg;
    oc_election_config_init(&cfg);
    cfg.node_priority = 1;
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(&cfg, 1, &s), OC_OK);
    bool granted = false;
    cr_assert_eq(oc_election_request_vote(s, 2, 1, &granted), OC_OK);
    cr_assert(granted);
    cr_assert_eq(s->voted_for, 2u);
    cr_assert_eq(s->current_term, 1u);
    oc_election_free(s);
}

Test(election, request_vote_rejected_for_lower_priority)
{
    OcElectionConfig cfg;
    oc_election_config_init(&cfg);
    cfg.node_priority = 5;
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(&cfg, 10, &s), OC_OK);
    bool granted = true;
    /* candidate_id 5 < self_id 10, same priority 5 → loses tie-break. */
    cr_assert_eq(oc_election_request_vote(s, 5, 1, &granted), OC_OK);
    cr_assert_not(granted);
    cr_assert_eq(s->voted_for, 0u);
    oc_election_free(s);
}

Test(election, request_vote_rejects_stale_term)
{
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(NULL, 1, &s), OC_OK);
    s->current_term = 5;
    bool granted = true;
    cr_assert_eq(oc_election_request_vote(s, 2, 3, &granted), OC_OK);
    cr_assert_not(granted);
    cr_assert_eq(s->current_term, 5u);
    oc_election_free(s);
}

Test(election, request_vote_adopts_higher_term)
{
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(NULL, 1, &s), OC_OK);
    s->current_term = 3;
    s->voted_for    = 99;
    bool granted = false;
    cr_assert_eq(oc_election_request_vote(s, 2, 7, &granted), OC_OK);
    cr_assert(granted);
    cr_assert_eq(s->current_term, 7u);
    cr_assert_eq(s->voted_for, 2u);
    oc_election_free(s);
}

Test(election, request_vote_double_vote_rejected)
{
    OcElectionConfig cfg;
    oc_election_config_init(&cfg);
    cfg.node_priority = 1;
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(&cfg, 1, &s), OC_OK);
    bool granted = false;
    cr_assert_eq(oc_election_request_vote(s, 2, 1, &granted), OC_OK);
    cr_assert(granted);
    /* Second vote in the same term for a different candidate. */
    granted = true;
    cr_assert_eq(oc_election_request_vote(s, 3, 1, &granted), OC_OK);
    cr_assert_not(granted);
    cr_assert_eq(s->voted_for, 2u);
    oc_election_free(s);
}


Test(election, receive_vote_increments_count)
{
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(NULL, 1, &s), OC_OK);
    s->role             = OC_ELECTION_CANDIDATE;
    s->current_term     = 2;
    s->n_votes_received = 1;
    cr_assert_eq(oc_election_receive_vote(s, 7, 2, true), OC_OK);
    cr_assert_eq(s->n_votes_received, 2u);
    oc_election_free(s);
}

Test(election, receive_vote_ignores_stale_term)
{
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(NULL, 1, &s), OC_OK);
    s->role             = OC_ELECTION_CANDIDATE;
    s->current_term     = 5;
    s->n_votes_received = 1;
    cr_assert_eq(oc_election_receive_vote(s, 7, 3, true), OC_OK);
    cr_assert_eq(s->n_votes_received, 1u);
    oc_election_free(s);
}

Test(election, receive_vote_ignored_when_not_candidate)
{
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(NULL, 1, &s), OC_OK);
    s->role             = OC_ELECTION_FOLLOWER;
    s->current_term     = 2;
    s->n_votes_received = 0;
    cr_assert_eq(oc_election_receive_vote(s, 7, 2, true), OC_OK);
    cr_assert_eq(s->n_votes_received, 0u);
    oc_election_free(s);
}


Test(election, heartbeat_adopts_leader_and_steps_down)
{
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(NULL, 1, &s), OC_OK);
    s->role         = OC_ELECTION_CANDIDATE;
    s->current_term = 1;
    cr_assert_eq(oc_election_heartbeat(s, 9, 3), OC_OK);
    cr_assert_eq(s->role, OC_ELECTION_FOLLOWER);
    cr_assert_eq(s->leader_id, 9u);
    cr_assert_eq(s->current_term, 3u);
    cr_assert_eq(s->voted_for, 0u);
    oc_election_free(s);
}

Test(election, heartbeat_ignores_stale_term)
{
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(NULL, 1, &s), OC_OK);
    s->current_term = 5;
    s->leader_id    = 7;
    cr_assert_eq(oc_election_heartbeat(s, 2, 3), OC_OK);
    cr_assert_eq(s->current_term, 5u);
    cr_assert_eq(s->leader_id, 7u);
    oc_election_free(s);
}

Test(election, heartbeat_rejects_zero_leader)
{
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_election_heartbeat(s, 0, 1), OC_ERR_INVALID_ARG);
    oc_election_free(s);
}


Test(election, become_leader_sets_state)
{
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(NULL, 42, &s), OC_OK);
    cr_assert_eq(oc_election_become_leader(s), OC_OK);
    cr_assert_eq(s->role, OC_ELECTION_LEADER);
    cr_assert_eq(s->leader_id, 42u);
    cr_assert_eq(s->n_votes_received, 0u);
    oc_election_free(s);
}


Test(election, get_role_returns_current_role)
{
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_election_get_role(s), OC_ELECTION_FOLLOWER);
    s->role = OC_ELECTION_LEADER;
    cr_assert_eq(oc_election_get_role(s), OC_ELECTION_LEADER);
    oc_election_free(s);
}

Test(election, get_leader_returns_leader_id)
{
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_election_get_leader(s), 0u);
    s->leader_id = 9;
    cr_assert_eq(oc_election_get_leader(s), 9u);
    oc_election_free(s);
}

Test(election, get_term_returns_current_term)
{
    OcElectionState *s = NULL;
    cr_assert_eq(oc_election_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_election_get_term(s), 0u);
    s->current_term = 7;
    cr_assert_eq(oc_election_get_term(s), 7u);
    oc_election_free(s);
}

Test(election, get_role_null_is_safe)
{
    cr_assert_eq(oc_election_get_role(NULL), OC_ELECTION_FOLLOWER);
    cr_assert_eq(oc_election_get_leader(NULL), 0u);
    cr_assert_eq(oc_election_get_term(NULL), 0u);
}
