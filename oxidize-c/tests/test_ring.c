/* test_ring.c — Ring topology tests. */
#include <criterion/criterion.h>
#include "oxidize/ring.h"
#include <string.h>

Test(ring, init_free)
{
    OcRing ring;
    cr_assert_eq(oc_ring_init(&ring, 1), OC_OK);
    cr_assert_eq(ring.n_nodes, 1);
    cr_assert_eq(ring.self_idx, 0);
    cr_assert_eq(ring.nodes[0].id, 1);
    oc_ring_free(&ring);
}

Test(ring, init_null)
{
    cr_assert_neq(oc_ring_init(NULL, 0), OC_OK);
}

Test(ring, add_node)
{
    OcRing ring;
    oc_ring_init(&ring, 1);
    cr_assert_eq(oc_ring_add_node(&ring, 2, "192.168.1.2", 8080), OC_OK);
    cr_assert_eq(ring.n_nodes, 2);
    cr_assert_str_eq(ring.nodes[1].addr, "192.168.1.2");
    cr_assert_eq(ring.nodes[1].port, 8080);
    oc_ring_free(&ring);
}

Test(ring, add_duplicate)
{
    OcRing ring;
    oc_ring_init(&ring, 1);
    oc_ring_add_node(&ring, 2, "addr2", 8080);
    oc_ring_add_node(&ring, 2, "addr2", 8080);
    cr_assert_eq(ring.n_nodes, 2);
    oc_ring_free(&ring);
}

Test(ring, add_null)
{
    cr_assert_neq(oc_ring_add_node(NULL, 0, "a", 0), OC_OK);
}

Test(ring, remove_node)
{
    OcRing ring;
    oc_ring_init(&ring, 1);
    oc_ring_add_node(&ring, 2, "addr2", 8080);
    oc_ring_add_node(&ring, 3, "addr3", 8080);
    cr_assert_eq(oc_ring_remove_node(&ring, 2), OC_OK);
    cr_assert_eq(ring.n_nodes, 2);
    cr_assert_eq(ring.nodes[1].id, 3);
    oc_ring_free(&ring);
}

Test(ring, remove_self)
{
    OcRing ring;
    oc_ring_init(&ring, 1);
    cr_assert_neq(oc_ring_remove_node(&ring, 1), OC_OK);
    oc_ring_free(&ring);
}

Test(ring, remove_not_found)
{
    OcRing ring;
    oc_ring_init(&ring, 1);
    cr_assert_neq(oc_ring_remove_node(&ring, 99), OC_OK);
    oc_ring_free(&ring);
}

Test(ring, get_next)
{
    OcRing ring;
    oc_ring_init(&ring, 1);
    oc_ring_add_node(&ring, 2, "addr2", 8080);
    oc_ring_add_node(&ring, 3, "addr3", 8080);
    const OcRingNode *next;
    cr_assert_eq(oc_ring_get_next(&ring, &next), OC_OK);
    cr_assert_eq(next->id, 2);
    oc_ring_free(&ring);
}

Test(ring, get_next_wrap)
{
    OcRing ring;
    oc_ring_init(&ring, 1);
    oc_ring_add_node(&ring, 2, "addr2", 8080);
    /* Move self to index 1 (node 2). */
    ring.self_idx = 1;
    const OcRingNode *next;
    cr_assert_eq(oc_ring_get_next(&ring, &next), OC_OK);
    /* Next should wrap to index 0 (node 1). */
    cr_assert_eq(next->id, 1);
    oc_ring_free(&ring);
}

Test(ring, get_prev)
{
    OcRing ring;
    oc_ring_init(&ring, 1);
    oc_ring_add_node(&ring, 2, "addr2", 8080);
    oc_ring_add_node(&ring, 3, "addr3", 8080);
    const OcRingNode *prev;
    cr_assert_eq(oc_ring_get_prev(&ring, &prev), OC_OK);
    /* Prev of node 1 (idx 0) is node 3 (idx 2). */
    cr_assert_eq(prev->id, 3);
    oc_ring_free(&ring);
}

Test(ring, get_node)
{
    OcRing ring;
    oc_ring_init(&ring, 1);
    oc_ring_add_node(&ring, 2, "addr2", 8080);
    const OcRingNode *node;
    cr_assert_eq(oc_ring_get_node(&ring, 1, &node), OC_OK);
    cr_assert_eq(node->id, 2);
    oc_ring_free(&ring);
}

Test(ring, get_node_oob)
{
    OcRing ring;
    oc_ring_init(&ring, 1);
    const OcRingNode *node;
    cr_assert_neq(oc_ring_get_node(&ring, 99, &node), OC_OK);
    oc_ring_free(&ring);
}

Test(ring, size)
{
    OcRing ring;
    oc_ring_init(&ring, 1);
    cr_assert_eq(oc_ring_size(&ring), 1);
    oc_ring_add_node(&ring, 2, "a", 80);
    cr_assert_eq(oc_ring_size(&ring), 2);
    oc_ring_free(&ring);
}

Test(ring, self_idx)
{
    OcRing ring;
    oc_ring_init(&ring, 1);
    cr_assert_eq(oc_ring_self_idx(&ring), 0);
    oc_ring_free(&ring);
}

Test(ring, stats)
{
    OcRing ring;
    oc_ring_init(&ring, 1);
    oc_ring_add_node(&ring, 2, "addr2", 8080);
    OcRingStats stats;
    cr_assert_eq(oc_ring_stats(&ring, &stats), OC_OK);
    cr_assert_eq(stats.dst_idx, 1);
    oc_ring_free(&ring);
}

Test(ring, barrier)
{
    OcRing ring;
    oc_ring_init(&ring, 1);
    cr_assert_eq(oc_ring_barrier(&ring), OC_OK);
    oc_ring_free(&ring);
}
