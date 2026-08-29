/* test_node.c — mesh node tests. */
#define _POSIX_C_SOURCE 200809L
#include "framework.h"
#include <stdio.h>
#include <string.h>
#include "oxidize/node.h"

/* ─── Lifecycle ─────────────────────────────────────────────────────── */

Test(node, init_creates_online_node)
{
    OcNodeConfig cfg = {0};
    cfg.id = 1;
    cfg.capabilities = OC_NODE_CAP_GPU | OC_NODE_CAP_CPU;
    cfg.max_connections = 8;
    snprintf(cfg.addr, sizeof(cfg.addr), "10.0.0.1");
    cfg.port = 5001;
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(&cfg, &n), OC_OK);
    cr_assert_not_null(n);
    cr_assert_eq(n->info.config.id, 1u);
    cr_assert_str_eq(n->info.config.addr, "10.0.0.1");
    cr_assert_eq(n->info.config.port, 5001u);
    cr_assert_eq(n->info.config.max_connections, 8u);
    cr_assert(n->info.is_online);
    cr_assert_eq(n->info.n_connections, 0u);
    cr_assert_eq(n->info.bytes_sent, 0u);
    cr_assert_eq(n->info.bytes_received, 0u);
    oc_node_free(n);
}

Test(node, init_rejects_zero_id)
{
    OcNodeConfig cfg = {0};
    cfg.id = 0;
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(&cfg, &n), OC_ERR_INVALID_ARG);
    cr_assert_null(n);
}

Test(node, init_uses_default_max_connections)
{
    OcNodeConfig cfg = {0};
    cfg.id = 1;
    cfg.max_connections = 0; /* should default */
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(&cfg, &n), OC_OK);
    cr_assert_eq(n->info.config.max_connections, OC_NODE_DEFAULT_MAX_CONNECTIONS);
    cr_assert_eq(n->info.max_connections, OC_NODE_DEFAULT_MAX_CONNECTIONS);
    oc_node_free(n);
}

Test(node, init_null_out_fails)
{
    OcNodeConfig cfg = {0};
    cfg.id = 1;
    cr_assert_eq(oc_node_init(&cfg, NULL), OC_ERR_INVALID_ARG);
}

Test(node, init_null_config_fails)
{
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(NULL, &n), OC_ERR_INVALID_ARG);
    cr_assert_null(n);
}

Test(node, free_null_is_safe)
{
    oc_node_free(NULL);
    cr_assert(true);
}

/* ─── Connection management ────────────────────────────────────────── */

Test(node, connect_adds_peer)
{
    OcNodeConfig cfg = {0};
    cfg.id = 1;
    cfg.max_connections = 4;
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(&cfg, &n), OC_OK);
    cr_assert_eq(oc_node_connect(n, 2), OC_OK);
    cr_assert_eq(oc_node_n_connections(n), 1u);
    cr_assert_eq(n->info.n_connections, 1u);
    oc_node_free(n);
}

Test(node, connect_is_idempotent)
{
    OcNodeConfig cfg = {0};
    cfg.id = 1;
    cfg.max_connections = 4;
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(&cfg, &n), OC_OK);
    cr_assert_eq(oc_node_connect(n, 2), OC_OK);
    cr_assert_eq(oc_node_connect(n, 2), OC_OK);
    cr_assert_eq(oc_node_n_connections(n), 1u);
    oc_node_free(n);
}

Test(node, connect_rejects_zero_peer)
{
    OcNodeConfig cfg = {0};
    cfg.id = 1;
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(&cfg, &n), OC_OK);
    cr_assert_eq(oc_node_connect(n, 0), OC_ERR_INVALID_ARG);
    oc_node_free(n);
}

Test(node, connect_at_max_returns_oom)
{
    OcNodeConfig cfg = {0};
    cfg.id = 1;
    cfg.max_connections = 1;
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(&cfg, &n), OC_OK);
    cr_assert_eq(oc_node_connect(n, 2), OC_OK);
    cr_assert_eq(oc_node_connect(n, 3), OC_ERR_OOM);
    cr_assert_eq(oc_node_n_connections(n), 1u);
    oc_node_free(n);
}

Test(node, disconnect_removes_peer)
{
    OcNodeConfig cfg = {0};
    cfg.id = 1;
    cfg.max_connections = 4;
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(&cfg, &n), OC_OK);
    cr_assert_eq(oc_node_connect(n, 2), OC_OK);
    cr_assert_eq(oc_node_connect(n, 3), OC_OK);
    cr_assert_eq(oc_node_n_connections(n), 2u);
    cr_assert_eq(oc_node_disconnect(n, 2), OC_OK);
    cr_assert_eq(oc_node_n_connections(n), 1u);
    /* Ensure peer 3 is still connected. */
    cr_assert_eq(n->connections[0], 3u);
    oc_node_free(n);
}

Test(node, disconnect_unknown_peer_is_ok)
{
    OcNodeConfig cfg = {0};
    cfg.id = 1;
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(&cfg, &n), OC_OK);
    cr_assert_eq(oc_node_disconnect(n, 99), OC_OK);
    cr_assert_eq(oc_node_n_connections(n), 0u);
    oc_node_free(n);
}

Test(node, disconnect_rejects_zero_peer)
{
    OcNodeConfig cfg = {0};
    cfg.id = 1;
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(&cfg, &n), OC_OK);
    cr_assert_eq(oc_node_disconnect(n, 0), OC_ERR_INVALID_ARG);
    oc_node_free(n);
}

/* ─── Queries ──────────────────────────────────────────────────────── */

Test(node, get_info_copies_state)
{
    OcNodeConfig cfg = {0};
    cfg.id = 1;
    cfg.capabilities = OC_NODE_CAP_GPU;
    cfg.max_connections = 4;
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(&cfg, &n), OC_OK);
    cr_assert_eq(oc_node_connect(n, 2), OC_OK);
    cr_assert_eq(oc_node_record_sent(n, 1024), OC_OK);
    OcNodeInfo info;
    cr_assert_eq(oc_node_get_info(n, &info), OC_OK);
    cr_assert_eq(info.config.id, 1u);
    cr_assert_eq(info.n_connections, 1u);
    cr_assert_eq(info.bytes_sent, 1024u);
    cr_assert(info.is_online);
    oc_node_free(n);
}

Test(node, is_online_true_after_init)
{
    OcNodeConfig cfg = {0};
    cfg.id = 1;
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(&cfg, &n), OC_OK);
    cr_assert(oc_node_is_online(n));
    oc_node_free(n);
}

OC_TEST_NULL_SAFE(node, is_online_null_returns_false,
        cr_assert_not(oc_node_is_online(NULL));)

OC_TEST_NULL_SAFE(node, n_connections_null_returns_zero,
        cr_assert_eq(oc_node_n_connections(NULL), 0u);)

Test(node, has_capability_checks_flag)
{
    OcNodeConfig cfg = {0};
    cfg.id = 1;
    cfg.capabilities = OC_NODE_CAP_GPU | OC_NODE_CAP_STORAGE;
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(&cfg, &n), OC_OK);
    cr_assert(oc_node_has_capability(n, OC_NODE_CAP_GPU));
    cr_assert(oc_node_has_capability(n, OC_NODE_CAP_STORAGE));
    cr_assert_not(oc_node_has_capability(n, OC_NODE_CAP_CPU));
    oc_node_free(n);
}

OC_TEST_NULL_SAFE(node, has_capability_null_returns_false,
        cr_assert_not(oc_node_has_capability(NULL, OC_NODE_CAP_GPU));)

/* ─── Traffic accounting ───────────────────────────────────────────── */

Test(node, record_sent_accumulates)
{
    OcNodeConfig cfg = {0};
    cfg.id = 1;
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(&cfg, &n), OC_OK);
    cr_assert_eq(oc_node_record_sent(n, 100), OC_OK);
    cr_assert_eq(oc_node_record_sent(n, 200), OC_OK);
    cr_assert_eq(n->info.bytes_sent, 300u);
    oc_node_free(n);
}

Test(node, record_received_accumulates)
{
    OcNodeConfig cfg = {0};
    cfg.id = 1;
    OcNode *n = NULL;
    cr_assert_eq(oc_node_init(&cfg, &n), OC_OK);
    cr_assert_eq(oc_node_record_received(n, 50), OC_OK);
    cr_assert_eq(oc_node_record_received(n, 25), OC_OK);
    cr_assert_eq(n->info.bytes_received, 75u);
    oc_node_free(n);
}

OC_TEST_NULL_SAFE(node, record_sent_null_fails,
        cr_assert_eq(oc_node_record_sent(NULL, 100), OC_ERR_INVALID_ARG);
        cr_assert_eq(oc_node_record_received(NULL, 100), OC_ERR_INVALID_ARG);)

/* ─── Capability names ──────────────────────────────────────────────── */

Test(node, capability_name_returns_string)
{
    cr_assert_str_eq(oc_node_capability_name(OC_NODE_CAP_GPU), "gpu");
    cr_assert_str_eq(oc_node_capability_name(OC_NODE_CAP_CPU), "cpu");
    cr_assert_str_eq(oc_node_capability_name(OC_NODE_CAP_STORAGE), "storage");
    cr_assert_str_eq(oc_node_capability_name(0), "unknown");
    cr_assert_str_eq(oc_node_capability_name(0xFFFFFFFFu), "unknown");
}
