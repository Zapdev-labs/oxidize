/* test_fault_tolerance.c — fault tolerance tests. */
#include "framework.h"
#include "oxidize/fault_tolerance.h"
#include <string.h>

Test(ft, config_init)
{
    OcFtConfig cfg;
    cr_assert_eq(oc_ft_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.heartbeat_interval_ms, OC_FT_DEFAULT_INTERVAL_MS);
    cr_assert_eq(cfg.heartbeat_timeout_ms, OC_FT_DEFAULT_TIMEOUT_MS);
    cr_assert_eq(cfg.max_missed, OC_FT_DEFAULT_MAX_MISSED);
    cr_assert_eq(cfg.recovery_timeout_ms, OC_FT_DEFAULT_RECOVERY_MS);
}

OC_TEST_NULL_SAFE(ft, config_init_null,
        cr_assert_neq(oc_ft_config_init(NULL), OC_OK);)

Test(ft, init_free)
{
    OcFtManager *mgr = NULL;
    cr_assert_eq(oc_ft_init(NULL, 1, &mgr), OC_OK);
    cr_assert_not_null(mgr);
    cr_assert_eq(mgr->self_id, 1);
    cr_assert_eq(mgr->n_nodes, 0);
    oc_ft_free(mgr);
}

OC_TEST_NULL_SAFE(ft, init_null_out,
        cr_assert_neq(oc_ft_init(NULL, 1, NULL), OC_OK);)

Test(ft, add_node)
{
    OcFtManager *mgr = NULL;
    oc_ft_init(NULL, 1, &mgr);
    cr_assert_eq(oc_ft_add_node(mgr, 2), OC_OK);
    cr_assert_eq(mgr->n_nodes, 1);
    cr_assert_eq(mgr->nodes[0].node_id, 2);
    cr_assert_eq(mgr->nodes[0].status, OC_FT_ALIVE);
    oc_ft_free(mgr);
}

Test(ft, add_node_idempotent)
{
    OcFtManager *mgr = NULL;
    oc_ft_init(NULL, 1, &mgr);
    oc_ft_add_node(mgr, 2);
    cr_assert_eq(oc_ft_add_node(mgr, 2), OC_OK);
    cr_assert_eq(mgr->n_nodes, 1);
    oc_ft_free(mgr);
}

Test(ft, remove_node)
{
    OcFtManager *mgr = NULL;
    oc_ft_init(NULL, 1, &mgr);
    oc_ft_add_node(mgr, 2);
    oc_ft_add_node(mgr, 3);
    cr_assert_eq(oc_ft_remove_node(mgr, 2), OC_OK);
    cr_assert_eq(mgr->n_nodes, 1);
    cr_assert_eq(mgr->nodes[0].node_id, 3);
    oc_ft_free(mgr);
}

Test(ft, remove_self)
{
    OcFtManager *mgr = NULL;
    oc_ft_init(NULL, 1, &mgr);
    oc_ft_add_node(mgr, 2);
    cr_assert_neq(oc_ft_remove_node(mgr, 1), OC_OK);
    oc_ft_free(mgr);
}

Test(ft, remove_not_found)
{
    OcFtManager *mgr = NULL;
    oc_ft_init(NULL, 1, &mgr);
    cr_assert_neq(oc_ft_remove_node(mgr, 99), OC_OK);
    oc_ft_free(mgr);
}

Test(ft, heartbeat)
{
    OcFtManager *mgr = NULL;
    oc_ft_init(NULL, 1, &mgr);
    oc_ft_add_node(mgr, 2);
    cr_assert_eq(oc_ft_heartbeat(mgr, 2, 1000), OC_OK);
    OcFtNodeState st;
    cr_assert_eq(oc_ft_get_node_state(mgr, 2, &st), OC_OK);
    cr_assert_eq(st.last_heartbeat_ms, 1000);
    cr_assert_eq(st.status, OC_FT_ALIVE);
    oc_ft_free(mgr);
}

Test(ft, heartbeat_not_found)
{
    OcFtManager *mgr = NULL;
    oc_ft_init(NULL, 1, &mgr);
    cr_assert_neq(oc_ft_heartbeat(mgr, 99, 100), OC_OK);
    oc_ft_free(mgr);
}

Test(ft, tick_suspect)
{
    OcFtManager *mgr = NULL;
    oc_ft_init(NULL, 1, &mgr);
    oc_ft_add_node(mgr, 2);
    oc_ft_heartbeat(mgr, 2, 0);
    /* Advance past timeout (default 5000ms). */
    cr_assert_eq(oc_ft_tick(mgr, 6000), OC_OK);
    OcFtNodeState st;
    oc_ft_get_node_state(mgr, 2, &st);
    cr_assert_eq(st.status, OC_FT_SUSPECT);
    cr_assert_eq(st.missed_count, 1);
    oc_ft_free(mgr);
}

Test(ft, tick_dead)
{
    OcFtConfig cfg;
    oc_ft_config_init(&cfg);
    cfg.heartbeat_timeout_ms = 1000;
    cfg.max_missed = 2;
    OcFtManager *mgr = NULL;
    oc_ft_init(&cfg, 1, &mgr);
    oc_ft_add_node(mgr, 2);
    oc_ft_heartbeat(mgr, 2, 0);
    /* First tick: missed=1 -> SUSPECT. */
    oc_ft_tick(mgr, 1500);
    OcFtNodeState st;
    oc_ft_get_node_state(mgr, 2, &st);
    cr_assert_eq(st.status, OC_FT_SUSPECT);
    /* Second tick: missed=2 -> DEAD. */
    oc_ft_tick(mgr, 2500);
    oc_ft_get_node_state(mgr, 2, &st);
    cr_assert_eq(st.status, OC_FT_DEAD);
    oc_ft_free(mgr);
}

Test(ft, tick_resets_on_heartbeat)
{
    OcFtConfig cfg;
    oc_ft_config_init(&cfg);
    cfg.heartbeat_timeout_ms = 1000;
    cfg.max_missed = 3;
    OcFtManager *mgr = NULL;
    oc_ft_init(&cfg, 1, &mgr);
    oc_ft_add_node(mgr, 2);
    oc_ft_heartbeat(mgr, 2, 0);
    oc_ft_tick(mgr, 1500);
    /* Heartbeat arrives; resets to ALIVE. */
    cr_assert_eq(oc_ft_heartbeat(mgr, 2, 1600), OC_OK);
    OcFtNodeState st;
    oc_ft_get_node_state(mgr, 2, &st);
    cr_assert_eq(st.status, OC_FT_ALIVE);
    cr_assert_eq(st.missed_count, 0);
    oc_ft_free(mgr);
}

Test(ft, alive_count)
{
    OcFtManager *mgr = NULL;
    oc_ft_init(NULL, 1, &mgr);
    oc_ft_add_node(mgr, 2);
    oc_ft_add_node(mgr, 3);
    oc_ft_add_node(mgr, 4);
    cr_assert_eq(oc_ft_get_alive_count(mgr), 3);
    oc_ft_recover(mgr, 3);
    cr_assert_eq(oc_ft_get_alive_count(mgr), 2);
    oc_ft_free(mgr);
}

Test(ft, dead_nodes)
{
    OcFtConfig cfg;
    oc_ft_config_init(&cfg);
    cfg.heartbeat_timeout_ms = 1000;
    cfg.max_missed = 1;
    OcFtManager *mgr = NULL;
    oc_ft_init(&cfg, 1, &mgr);
    oc_ft_add_node(mgr, 2);
    oc_ft_add_node(mgr, 3);
    oc_ft_heartbeat(mgr, 2, 0);
    oc_ft_heartbeat(mgr, 3, 0);
    oc_ft_tick(mgr, 2000);
    uint64_t ids[8];
    uint32_t count = 0;
    cr_assert_eq(oc_ft_get_dead_nodes(mgr, ids, 8, &count), OC_OK);
    cr_assert_eq(count, 2);
    oc_ft_free(mgr);
}

Test(ft, dead_nodes_count_only)
{
    OcFtConfig cfg;
    oc_ft_config_init(&cfg);
    cfg.heartbeat_timeout_ms = 1000;
    cfg.max_missed = 1;
    OcFtManager *mgr = NULL;
    oc_ft_init(&cfg, 1, &mgr);
    oc_ft_add_node(mgr, 2);
    oc_ft_heartbeat(mgr, 2, 0);
    oc_ft_tick(mgr, 2000);
    uint32_t count = 0;
    cr_assert_eq(oc_ft_get_dead_nodes(mgr, NULL, 0, &count), OC_OK);
    cr_assert_eq(count, 1);
    oc_ft_free(mgr);
}

Test(ft, recover)
{
    OcFtManager *mgr = NULL;
    oc_ft_init(NULL, 1, &mgr);
    oc_ft_add_node(mgr, 2);
    cr_assert_eq(oc_ft_recover(mgr, 2), OC_OK);
    OcFtNodeState st;
    oc_ft_get_node_state(mgr, 2, &st);
    cr_assert_eq(st.status, OC_FT_RECOVERING);
    oc_ft_free(mgr);
}

Test(ft, recover_not_found)
{
    OcFtManager *mgr = NULL;
    oc_ft_init(NULL, 1, &mgr);
    cr_assert_neq(oc_ft_recover(mgr, 99), OC_OK);
    oc_ft_free(mgr);
}

Test(ft, status_name)
{
    cr_assert_str_eq(oc_ft_status_name(OC_FT_ALIVE), "ALIVE");
    cr_assert_str_eq(oc_ft_status_name(OC_FT_SUSPECT), "SUSPECT");
    cr_assert_str_eq(oc_ft_status_name(OC_FT_DEAD), "DEAD");
    cr_assert_str_eq(oc_ft_status_name(OC_FT_RECOVERING), "RECOVERING");
}

Test(ft, recovering_skips_tick)
{
    OcFtConfig cfg;
    oc_ft_config_init(&cfg);
    cfg.heartbeat_timeout_ms = 1000;
    cfg.max_missed = 1;
    OcFtManager *mgr = NULL;
    oc_ft_init(&cfg, 1, &mgr);
    oc_ft_add_node(mgr, 2);
    oc_ft_heartbeat(mgr, 2, 0);
    oc_ft_recover(mgr, 2);
    oc_ft_tick(mgr, 99999);
    OcFtNodeState st;
    oc_ft_get_node_state(mgr, 2, &st);
    cr_assert_eq(st.status, OC_FT_RECOVERING);
    cr_assert_eq(st.missed_count, 0);
    oc_ft_free(mgr);
}

Test(ft, self_never_times_out)
{
    OcFtConfig cfg;
    oc_ft_config_init(&cfg);
    cfg.heartbeat_timeout_ms = 1;
    cfg.max_missed = 1;
    OcFtManager *mgr = NULL;
    oc_ft_init(&cfg, 1, &mgr);
    oc_ft_add_node(mgr, 1);  /* self as monitored node */
    oc_ft_heartbeat(mgr, 1, 0);
    oc_ft_tick(mgr, 99999999);
    OcFtNodeState st;
    oc_ft_get_node_state(mgr, 1, &st);
    cr_assert_eq(st.status, OC_FT_ALIVE);
    oc_ft_free(mgr);
}

OC_TEST_NULL_SAFE(ft, free_null_safe,
        oc_ft_free(NULL);)

Test(ft, get_node_state_not_found)
{
    OcFtManager *mgr = NULL;
    oc_ft_init(NULL, 1, &mgr);
    OcFtNodeState st;
    cr_assert_neq(oc_ft_get_node_state(mgr, 99, &st), OC_OK);
    oc_ft_free(mgr);
}

Test(ft, add_many_nodes)
{
    OcFtManager *mgr = NULL;
    oc_ft_init(NULL, 1, &mgr);
    for (uint64_t i = 2; i < 2 + OC_FT_MAX_NODES; i++) {
        oc_ft_add_node(mgr, i);
    }
    cr_assert_eq(mgr->n_nodes, OC_FT_MAX_NODES);
    /* Adding one beyond capacity should fail. */
    cr_assert_neq(oc_ft_add_node(mgr, 9999), OC_OK);
    oc_ft_free(mgr);
}
