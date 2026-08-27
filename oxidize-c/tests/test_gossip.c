/* test_gossip.c — gossip protocol tests. */
#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>
#include <stdio.h>
#include <string.h>
#include "oxidize/gossip.h"

/* ─── Config ──────────────────────────────────────────────────────── */

Test(gossip, config_init_defaults)
{
    OcGossipConfig cfg;
    cr_assert_eq(oc_gossip_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.interval_ms, OC_GOSSIP_DEFAULT_INTERVAL_MS);
    cr_assert_eq(cfg.timeout_ms,  OC_GOSSIP_DEFAULT_TIMEOUT_MS);
    cr_assert_eq(cfg.max_nodes,   OC_GOSSIP_DEFAULT_MAX_NODES);
    cr_assert_eq(cfg.fanout,      OC_GOSSIP_DEFAULT_FANOUT);
}

Test(gossip, config_init_null)
{
    cr_assert_eq(oc_gossip_config_init(NULL), OC_ERR_INVALID_ARG);
}

/* ─── Lifecycle ───────────────────────────────────────────────────── */

Test(gossip, init_creates_self_node)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 42, &s), OC_OK);
    cr_assert_not_null(s);
    cr_assert_eq(s->self_id, 42u);
    cr_assert_eq(s->n_nodes, 1u);
    cr_assert_eq(s->nodes[0].id, 42u);
    cr_assert(s->nodes[0].healthy);
    oc_gossip_free(s);
}

Test(gossip, init_with_custom_config)
{
    OcGossipConfig cfg;
    oc_gossip_config_init(&cfg);
    cfg.max_nodes   = 8;
    cfg.interval_ms = 250;
    cfg.timeout_ms  = 1000;
    cfg.fanout      = 2;
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(&cfg, 1, &s), OC_OK);
    cr_assert_eq(s->config.max_nodes, 8u);
    cr_assert_eq(s->config.timeout_ms, 1000u);
    cr_assert_eq(s->config.fanout, 2u);
    oc_gossip_free(s);
}

Test(gossip, init_zero_max_nodes_fails)
{
    OcGossipConfig cfg;
    oc_gossip_config_init(&cfg);
    cfg.max_nodes = 0;
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(&cfg, 1, &s), OC_ERR_INVALID_ARG);
    cr_assert_null(s);
}

Test(gossip, free_null_is_safe)
{
    oc_gossip_free(NULL);
    cr_assert(true);
}

Test(gossip, init_zero_fanout_normalized)
{
    OcGossipConfig cfg;
    oc_gossip_config_init(&cfg);
    cfg.fanout = 0;
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(&cfg, 1, &s), OC_OK);
    cr_assert_eq(s->config.fanout, 1u);
    oc_gossip_free(s);
}

/* ─── Add / remove ────────────────────────────────────────────────── */

Test(gossip, add_node)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 2, "10.0.0.2", 5001), OC_OK);
    cr_assert_eq(s->n_nodes, 2u);
    cr_assert_str_eq(s->nodes[1].addr, "10.0.0.2");
    cr_assert_eq(s->nodes[1].port, 5001u);
    oc_gossip_free(s);
}

Test(gossip, add_node_refreshes_existing)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 2, "10.0.0.2", 5001), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 2, "10.0.0.20", 6000), OC_OK);
    cr_assert_eq(s->n_nodes, 2u); /* not duplicated */
    cr_assert_str_eq(s->nodes[1].addr, "10.0.0.20");
    cr_assert_eq(s->nodes[1].port, 6000u);
    oc_gossip_free(s);
}

Test(gossip, add_node_overflow)
{
    OcGossipConfig cfg;
    oc_gossip_config_init(&cfg);
    cfg.max_nodes = 2;
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(&cfg, 1, &s), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 2, "10.0.0.2", 5001), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 3, "10.0.0.3", 5002), OC_ERR_OOM);
    cr_assert_eq(s->n_nodes, 2u);
    oc_gossip_free(s);
}

Test(gossip, remove_node)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 2, "10.0.0.2", 5001), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 3, "10.0.0.3", 5002), OC_OK);
    cr_assert_eq(oc_gossip_remove_node(s, 2), OC_OK);
    cr_assert_eq(s->n_nodes, 2u);
    cr_assert_eq(s->nodes[1].id, 3u);
    oc_gossip_free(s);
}

Test(gossip, remove_node_absent_is_ok)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_gossip_remove_node(s, 999), OC_OK);
    cr_assert_eq(s->n_nodes, 1u);
    oc_gossip_free(s);
}

Test(gossip, remove_self_is_error)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 7, &s), OC_OK);
    cr_assert_eq(oc_gossip_remove_node(s, 7), OC_ERR_INVALID_ARG);
    cr_assert_eq(s->n_nodes, 1u);
    oc_gossip_free(s);
}

Test(gossip, add_node_null_args)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 2, NULL, 5001), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_gossip_add_node(NULL, 2, "10.0.0.2", 5001), OC_ERR_INVALID_ARG);
    oc_gossip_free(s);
}

/* ─── Tick / health ───────────────────────────────────────────────── */

Test(gossip, tick_marks_stale_unhealthy)
{
    OcGossipConfig cfg;
    oc_gossip_config_init(&cfg);
    cfg.timeout_ms = 1000;
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(&cfg, 1, &s), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 2, "10.0.0.2", 5001), OC_OK);
    /* Set peer's last_seen to ancient time. */
    s->nodes[1].last_seen = 100;
    cr_assert_eq(oc_gossip_tick(s, 5000), OC_OK);
    /* Self remains healthy. */
    cr_assert(s->nodes[0].healthy);
    /* Peer is stale (5000 - 100 > 1000). */
    cr_assert_not(s->nodes[1].healthy);
    cr_assert_eq(oc_gossip_get_healthy_count(s), 1u);
    oc_gossip_free(s);
}

Test(gossip, tick_keeps_recent_healthy)
{
    OcGossipConfig cfg;
    oc_gossip_config_init(&cfg);
    cfg.timeout_ms = 5000;
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(&cfg, 1, &s), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 2, "10.0.0.2", 5001), OC_OK);
    cr_assert_eq(oc_gossip_tick(s, 1000), OC_OK);
    cr_assert(s->nodes[1].healthy);
    cr_assert_eq(oc_gossip_get_healthy_count(s), 2u);
    oc_gossip_free(s);
}

Test(gossip, tick_null_is_error)
{
    cr_assert_eq(oc_gossip_tick(NULL, 0), OC_ERR_INVALID_ARG);
}

/* ─── Queries ─────────────────────────────────────────────────────── */

Test(gossip, get_nodes_copies)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 2, "10.0.0.2", 5001), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 3, "10.0.0.3", 5002), OC_OK);
    OcGossipNode out[4];
    uint32_t count = 0;
    cr_assert_eq(oc_gossip_get_nodes(s, out, 4, &count), OC_OK);
    cr_assert_eq(count, 3u);
    cr_assert_eq(out[0].id, 1u);
    cr_assert_eq(out[1].id, 2u);
    cr_assert_eq(out[2].id, 3u);
    oc_gossip_free(s);
}

Test(gossip, get_nodes_count_only)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    uint32_t count = 999u;
    cr_assert_eq(oc_gossip_get_nodes(s, NULL, 0, &count), OC_OK);
    cr_assert_eq(count, 1u);
    oc_gossip_free(s);
}

Test(gossip, get_healthy_count_empty_state)
{
    cr_assert_eq(oc_gossip_get_healthy_count(NULL), 0u);
}

/* ─── Merge ───────────────────────────────────────────────────────── */

Test(gossip, merge_adds_new_nodes)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    OcGossipNode remote[2];
    memset(remote, 0, sizeof(remote));
    remote[0].id = 2; snprintf(remote[0].addr, sizeof(remote[0].addr), "10.0.0.2");
    remote[0].port = 5001; remote[0].healthy = true; remote[0].last_seen = 1000;
    remote[1].id = 3; snprintf(remote[1].addr, sizeof(remote[1].addr), "10.0.0.3");
    remote[1].port = 5002; remote[1].healthy = true; remote[1].last_seen = 1000;
    cr_assert_eq(oc_gossip_merge(s, 2000, remote, 2), OC_OK);
    cr_assert_eq(s->n_nodes, 3u);
    cr_assert_eq(s->nodes[1].id, 2u);
    cr_assert_eq(s->nodes[2].id, 3u);
    oc_gossip_free(s);
}

Test(gossip, merge_does_not_add_self)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    OcGossipNode remote[1];
    memset(remote, 0, sizeof(remote));
    remote[0].id = 1; /* same as self */
    snprintf(remote[0].addr, sizeof(remote[0].addr), "10.0.0.99");
    remote[0].healthy = true; remote[0].last_seen = 5000;
    cr_assert_eq(oc_gossip_merge(s, 100, remote, 1), OC_OK);
    cr_assert_eq(s->n_nodes, 1u);
    /* Self addr unchanged. */
    cr_assert_eq(s->nodes[0].addr[0], '\0');
    oc_gossip_free(s);
}

Test(gossip, merge_takes_newer_last_seen)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 2, "10.0.0.2", 5001), OC_OK);
    s->nodes[1].last_seen = 5000;
    OcGossipNode remote[1];
    memset(remote, 0, sizeof(remote));
    remote[0].id = 2; snprintf(remote[0].addr, sizeof(remote[0].addr), "10.0.0.20");
    remote[0].port = 6000; remote[0].healthy = true; remote[0].last_seen = 9000;
    cr_assert_eq(oc_gossip_merge(s, 10000, remote, 1), OC_OK);
    cr_assert_eq(s->nodes[1].last_seen, 9000u);
    cr_assert_str_eq(s->nodes[1].addr, "10.0.0.20");
    cr_assert_eq(s->nodes[1].port, 6000u);
    oc_gossip_free(s);
}

Test(gossip, merge_null_remote_with_zero_count)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_gossip_merge(s, 0, NULL, 0), OC_OK);
    cr_assert_eq(s->n_nodes, 1u);
    oc_gossip_free(s);
}

Test(gossip, merge_null_remote_nonzero_count_is_error)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_gossip_merge(s, 0, NULL, 1), OC_ERR_INVALID_ARG);
    oc_gossip_free(s);
}

/* ─── Serialize / deserialize ────────────────────────────────────── */

Test(gossip, serialize_then_deserialize_roundtrip)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 2, "10.0.0.2", 5001), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 3, "10.0.0.3", 5002), OC_OK);
    snprintf(s->nodes[1].metadata, sizeof(s->nodes[1].metadata), "gpu=4");

    size_t needed = 0;
    cr_assert_eq(oc_gossip_serialize(s, NULL, 0, &needed), OC_ERR_INVALID_ARG);
    cr_assert(needed > 0);

    uint8_t *buf = malloc(needed);
    cr_assert_not_null(buf);
    size_t len = 0;
    cr_assert_eq(oc_gossip_serialize(s, buf, needed, &len), OC_OK);
    cr_assert_eq(len, needed);

    /* Deserialize into a fresh state that only knows itself. */
    OcGossipState *s2 = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s2), OC_OK);
    cr_assert_eq(oc_gossip_deserialize(s2, 12345, buf, len), OC_OK);
    cr_assert_eq(s2->n_nodes, 3u);
    /* Find node 2. */
    bool found2 = false, found3 = false;
    for (uint32_t i = 0; i < s2->n_nodes; i++) {
        if (s2->nodes[i].id == 2u) {
            found2 = true;
            cr_assert_str_eq(s2->nodes[i].addr, "10.0.0.2");
            cr_assert_eq(s2->nodes[i].port, 5001u);
            cr_assert_str_eq(s2->nodes[i].metadata, "gpu=4");
        }
        if (s2->nodes[i].id == 3u) {
            found3 = true;
            cr_assert_str_eq(s2->nodes[i].addr, "10.0.0.3");
        }
    }
    cr_assert(found2);
    cr_assert(found3);
    free(buf);
    oc_gossip_free(s);
    oc_gossip_free(s2);
}

Test(gossip, deserialize_short_buffer_is_error)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    uint8_t tiny[2] = {0, 0};
    cr_assert_eq(oc_gossip_deserialize(s, 0, tiny, 2), OC_ERR_INVALID_ARG);
    oc_gossip_free(s);
}

Test(gossip, serialize_too_small_returns_needed)
{
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    cr_assert_eq(oc_gossip_add_node(s, 2, "10.0.0.2", 5001), OC_OK);
    uint8_t small[4];
    size_t needed = 0;
    cr_assert_eq(oc_gossip_serialize(s, small, 4, &needed), OC_ERR_INVALID_ARG);
    cr_assert(needed > 4);
    oc_gossip_free(s);
}

Test(gossip, deserialize_count_exceeds_max_is_error)
{
    /* Build a buffer claiming 100 nodes — far exceeding the default cap of
     * 64. The deserialize path should reject it before allocating. */
    OcGossipState *s = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &s), OC_OK);
    uint8_t buf[8];
    buf[0] = 100u; buf[1] = 0; buf[2] = 0; buf[3] = 0;
    /* Pad to ensure the length check passes for count=100 is unreachable:
     * we only need to trigger the early cap check. */
    cr_assert_eq(oc_gossip_deserialize(s, 0, buf, 8), OC_ERR_INVALID_ARG);
    oc_gossip_free(s);
}

/* ─── End-to-end ──────────────────────────────────────────────────── */

Test(gossip, full_gossip_cycle)
{
    /* Two nodes exchange peer lists; each ends up knowing about the other. */
    OcGossipState *a = NULL, *b = NULL;
    cr_assert_eq(oc_gossip_init(NULL, 1, &a), OC_OK);
    cr_assert_eq(oc_gossip_init(NULL, 2, &b), OC_OK);

    /* Each node knows about itself; simulate adding the peer's address. */
    cr_assert_eq(oc_gossip_add_node(a, 2, "10.0.0.2", 5001), OC_OK);
    cr_assert_eq(oc_gossip_add_node(b, 1, "10.0.0.1", 5000), OC_OK);

    /* A ticks, serializes, B deserializes. */
    cr_assert_eq(oc_gossip_tick(a, 1000), OC_OK);
    size_t needed = 0;
    cr_assert_eq(oc_gossip_serialize(a, NULL, 0, &needed), OC_ERR_INVALID_ARG);
    uint8_t *buf = malloc(needed);
    size_t len = 0;
    cr_assert_eq(oc_gossip_serialize(a, buf, needed, &len), OC_OK);
    cr_assert_eq(oc_gossip_deserialize(b, 1000, buf, len), OC_OK);

    /* B should still know itself + peer (id 1). Self id (2) not duplicated. */
    cr_assert_eq(b->n_nodes, 2u);
    bool knows_a = false;
    for (uint32_t i = 0; i < b->n_nodes; i++) {
        if (b->nodes[i].id == 1u) knows_a = true;
    }
    cr_assert(knows_a);

    free(buf);
    oc_gossip_free(a);
    oc_gossip_free(b);
}
