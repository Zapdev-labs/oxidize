/* test_discovery.c — Mesh discovery protocol tests. */
#include "framework.h"
#include "oxidize/discovery.h"
#include <string.h>

Test(disc, init)
{
    OcDiscoveryState state;
    cr_assert_eq(oc_discovery_init(&state, 1, "239.0.0.1", 7946), OC_OK);
    cr_assert_eq(state.self_id, 1);
    cr_assert_str_eq(state.multicast_addr, "239.0.0.1");
    cr_assert_eq(state.port, 7946);
    cr_assert_eq(state.n_peers, 0);
    oc_discovery_free(&state);
}

Test(disc, init_defaults)
{
    OcDiscoveryState state;
    cr_assert_eq(oc_discovery_init(&state, 1, NULL, 0), OC_OK);
    cr_assert_str_eq(state.multicast_addr, "239.0.0.1");
    cr_assert_eq(state.port, 7946);
    oc_discovery_free(&state);
}

OC_TEST_NULL_SAFE(disc, init_null,
        cr_assert_neq(oc_discovery_init(NULL, 1, NULL, 0), OC_OK);)

Test(disc, add_seed)
{
    OcDiscoveryState state;
    oc_discovery_init(&state, 1, NULL, 0);
    cr_assert_eq(oc_discovery_add_seed(&state, 2, "10.0.0.2", 7946), OC_OK);
    cr_assert_eq(state.n_peers, 1);
    cr_assert_eq(state.peers[0].id, 2);
    cr_assert_str_eq(state.peers[0].addr, "10.0.0.2");
    oc_discovery_free(&state);
}

Test(disc, add_seed_duplicate)
{
    OcDiscoveryState state;
    oc_discovery_init(&state, 1, NULL, 0);
    oc_discovery_add_seed(&state, 2, "10.0.0.2", 7946);
    cr_assert_eq(oc_discovery_add_seed(&state, 2, "10.0.0.3", 7946), OC_OK);
    cr_assert_eq(state.n_peers, 1);
    oc_discovery_free(&state);
}

OC_TEST_NULL_SAFE(disc, add_seed_null,
        cr_assert_neq(oc_discovery_add_seed(NULL, 0, NULL, 0), OC_OK);)

Test(disc, receive_peer_new)
{
    OcDiscoveryState state;
    oc_discovery_init(&state, 1, NULL, 0);
    cr_assert_eq(oc_discovery_receive_peer(&state, 2, "10.0.0.2", 7946, 0x3), OC_OK);
    cr_assert_eq(state.n_peers, 1);
    cr_assert(state.peers[0].alive);
    cr_assert_eq(state.peers[0].capabilities, 0x3);
    oc_discovery_free(&state);
}

Test(disc, receive_peer_existing)
{
    OcDiscoveryState state;
    oc_discovery_init(&state, 1, NULL, 0);
    oc_discovery_receive_peer(&state, 2, "10.0.0.2", 7946, 0x1);
    oc_discovery_receive_peer(&state, 2, "10.0.0.3", 7947, 0x3);
    cr_assert_eq(state.n_peers, 1);
    cr_assert_str_eq(state.peers[0].addr, "10.0.0.3");
    cr_assert_eq(state.peers[0].port, 7947);
    oc_discovery_free(&state);
}

Test(disc, tick_timeout)
{
    OcDiscoveryState state;
    oc_discovery_init(&state, 1, NULL, 0);
    state.timeout_ms = 1000;
    state.current_ms = 100;
    oc_discovery_receive_peer(&state, 2, "10.0.0.2", 7946, 0);
    cr_assert(state.peers[0].alive);
    /* Advance time beyond timeout. */
    oc_discovery_tick(&state, 2000);
    cr_assert(!state.peers[0].alive);
    oc_discovery_free(&state);
}

Test(disc, tick_no_timeout)
{
    OcDiscoveryState state;
    oc_discovery_init(&state, 1, NULL, 0);
    state.timeout_ms = 10000;
    state.current_ms = 100;
    oc_discovery_receive_peer(&state, 2, "10.0.0.2", 7946, 0);
    oc_discovery_tick(&state, 5000);
    cr_assert(state.peers[0].alive);
    oc_discovery_free(&state);
}

Test(disc, get_peers)
{
    OcDiscoveryState state;
    oc_discovery_init(&state, 1, NULL, 0);
    oc_discovery_receive_peer(&state, 2, "ip2", 80, 0);
    oc_discovery_receive_peer(&state, 3, "ip3", 80, 0);
    const OcPeerInfo *peers;
    uint32_t count;
    cr_assert_eq(oc_discovery_get_peers(&state, &peers, &count), OC_OK);
    cr_assert_eq(count, 2);
    oc_discovery_free(&state);
}

Test(disc, get_alive_peers)
{
    OcDiscoveryState state;
    oc_discovery_init(&state, 1, NULL, 0);
    oc_discovery_receive_peer(&state, 2, "ip2", 80, 0);
    oc_discovery_add_seed(&state, 3, "ip3", 80); /* not alive */
    const OcPeerInfo *peers;
    uint32_t count;
    cr_assert_eq(oc_discovery_get_alive_peers(&state, &peers, &count), OC_OK);
    cr_assert_eq(count, 1);
    oc_discovery_free(&state);
}

Test(disc, n_peers)
{
    OcDiscoveryState state;
    oc_discovery_init(&state, 1, NULL, 0);
    cr_assert_eq(oc_discovery_n_peers(&state), 0);
    oc_discovery_receive_peer(&state, 2, "ip", 80, 0);
    cr_assert_eq(oc_discovery_n_peers(&state), 1);
    oc_discovery_free(&state);
}

Test(disc, n_alive)
{
    OcDiscoveryState state;
    oc_discovery_init(&state, 1, NULL, 0);
    oc_discovery_receive_peer(&state, 2, "ip", 80, 0);
    oc_discovery_add_seed(&state, 3, "ip3", 80);
    cr_assert_eq(oc_discovery_n_alive(&state), 1);
    oc_discovery_free(&state);
}

Test(disc, has_peer)
{
    OcDiscoveryState state;
    oc_discovery_init(&state, 1, NULL, 0);
    oc_discovery_receive_peer(&state, 2, "ip", 80, 0);
    cr_assert(oc_discovery_has_peer(&state, 2));
    cr_assert(!oc_discovery_has_peer(&state, 99));
    oc_discovery_free(&state);
}

OC_TEST_NULL_SAFE(disc, has_peer_null,
        cr_assert(!oc_discovery_has_peer(NULL, 0));)

Test(disc, remove_peer)
{
    OcDiscoveryState state;
    oc_discovery_init(&state, 1, NULL, 0);
    oc_discovery_receive_peer(&state, 2, "ip2", 80, 0);
    oc_discovery_receive_peer(&state, 3, "ip3", 80, 0);
    cr_assert_eq(oc_discovery_remove_peer(&state, 2), OC_OK);
    cr_assert_eq(state.n_peers, 1);
    cr_assert_eq(state.peers[0].id, 3);
    oc_discovery_free(&state);
}

Test(disc, remove_peer_not_found)
{
    OcDiscoveryState state;
    oc_discovery_init(&state, 1, NULL, 0);
    cr_assert_neq(oc_discovery_remove_peer(&state, 99), OC_OK);
    oc_discovery_free(&state);
}

Test(disc, announce)
{
    OcDiscoveryState state;
    oc_discovery_init(&state, 1, NULL, 0);
    cr_assert_eq(oc_discovery_announce(&state), OC_OK);
    oc_discovery_free(&state);
}

OC_TEST_NULL_SAFE(disc, free_null,
        oc_discovery_free(NULL);)
