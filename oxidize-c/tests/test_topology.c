/* test_topology.c — Network topology tests. */
#include <criterion/criterion.h>
#include "oxidize/topology.h"
#include <string.h>

Test(topo, init)
{
    OcTopology topo;
    cr_assert_eq(oc_topology_init(&topo), OC_OK);
    cr_assert(topo.initialized);
    cr_assert_eq(topo.n_nodes, 0);
    oc_topology_free(&topo);
}

Test(topo, init_null)
{
    cr_assert_neq(oc_topology_init(NULL), OC_OK);
}

Test(topo, add_node)
{
    OcTopology topo;
    oc_topology_init(&topo);
    cr_assert_eq(oc_topology_add_node(&topo, 1, "192.168.1.1", 8080, 0), OC_OK);
    cr_assert_eq(topo.n_nodes, 1);
    cr_assert_str_eq(topo.nodes[0].addr, "192.168.1.1");
    oc_topology_free(&topo);
}

Test(topo, add_node_duplicate)
{
    OcTopology topo;
    oc_topology_init(&topo);
    oc_topology_add_node(&topo, 1, "a", 80, 0);
    oc_topology_add_node(&topo, 1, "a", 80, 0);
    cr_assert_eq(topo.n_nodes, 1);
    oc_topology_free(&topo);
}

Test(topo, add_link)
{
    OcTopology topo;
    oc_topology_init(&topo);
    oc_topology_add_node(&topo, 1, "a", 80, 0);
    oc_topology_add_node(&topo, 2, "b", 80, 0);
    cr_assert_eq(oc_topology_add_link(&topo, 1, 2, 10000, 100), OC_OK);
    const OcTopoLink *link;
    cr_assert_eq(oc_topology_get_link(&topo, 1, 2, &link), OC_OK);
    cr_assert(link->connected);
    cr_assert_eq(link->bandwidth_mbps, 10000);
    oc_topology_free(&topo);
}

Test(topo, add_link_self)
{
    OcTopology topo;
    oc_topology_init(&topo);
    oc_topology_add_node(&topo, 1, "a", 80, 0);
    cr_assert_neq(oc_topology_add_link(&topo, 1, 1, 1000, 10), OC_OK);
    oc_topology_free(&topo);
}

Test(topo, get_link_not_found)
{
    OcTopology topo;
    oc_topology_init(&topo);
    const OcTopoLink *link;
    cr_assert_neq(oc_topology_get_link(&topo, 99, 98, &link), OC_OK);
    oc_topology_free(&topo);
}

Test(topo, get_node)
{
    OcTopology topo;
    oc_topology_init(&topo);
    oc_topology_add_node(&topo, 42, "addr", 8080, 3);
    const OcTopoNode *n;
    cr_assert_eq(oc_topology_get_node(&topo, 42, &n), OC_OK);
    cr_assert_str_eq(n->addr, "addr");
    cr_assert_eq(n->region_id, 3);
    oc_topology_free(&topo);
}

Test(topo, get_node_not_found)
{
    OcTopology topo;
    oc_topology_init(&topo);
    const OcTopoNode *n;
    cr_assert_neq(oc_topology_get_node(&topo, 99, &n), OC_OK);
    oc_topology_free(&topo);
}

Test(topo, get_neighbors)
{
    OcTopology topo;
    oc_topology_init(&topo);
    oc_topology_add_node(&topo, 1, "a", 80, 0);
    oc_topology_add_node(&topo, 2, "b", 80, 0);
    oc_topology_add_node(&topo, 3, "c", 80, 0);
    oc_topology_add_link(&topo, 1, 2, 1000, 10);
    oc_topology_add_link(&topo, 1, 3, 2000, 20);
    uint64_t ids[10];
    uint32_t count;
    cr_assert_eq(oc_topology_get_neighbors(&topo, 1, ids, 10, &count), OC_OK);
    cr_assert_eq(count, 2);
    oc_topology_free(&topo);
}

Test(topo, stats)
{
    OcTopology topo;
    oc_topology_init(&topo);
    oc_topology_add_node(&topo, 1, "a", 80, 0);
    oc_topology_add_node(&topo, 2, "b", 80, 0);
    oc_topology_add_link(&topo, 1, 2, 10000, 100);
    OcTopoStats stats;
    cr_assert_eq(oc_topology_stats(&topo, &stats), OC_OK);
    cr_assert_eq(stats.n_links, 1);
    cr_assert_eq(stats.total_bandwidth, 10000);
    cr_assert_eq(stats.avg_latency, 100);
    oc_topology_free(&topo);
}

Test(topo, shortest_path_direct)
{
    OcTopology topo;
    oc_topology_init(&topo);
    oc_topology_add_node(&topo, 1, "a", 80, 0);
    oc_topology_add_node(&topo, 2, "b", 80, 0);
    oc_topology_add_link(&topo, 1, 2, 1000, 10);
    uint64_t path[10];
    uint32_t n_hops;
    cr_assert_eq(oc_topology_shortest_path(&topo, 1, 2, path, 10, &n_hops), OC_OK);
    cr_assert_eq(n_hops, 2);
    cr_assert_eq(path[0], 1);
    cr_assert_eq(path[1], 2);
    oc_topology_free(&topo);
}

Test(topo, shortest_path_multi_hop)
{
    OcTopology topo;
    oc_topology_init(&topo);
    oc_topology_add_node(&topo, 1, "a", 80, 0);
    oc_topology_add_node(&topo, 2, "b", 80, 0);
    oc_topology_add_node(&topo, 3, "c", 80, 0);
    oc_topology_add_link(&topo, 1, 2, 1000, 10);
    oc_topology_add_link(&topo, 2, 3, 1000, 10);
    uint64_t path[10];
    uint32_t n_hops;
    cr_assert_eq(oc_topology_shortest_path(&topo, 1, 3, path, 10, &n_hops), OC_OK);
    cr_assert_eq(n_hops, 3);
    oc_topology_free(&topo);
}

Test(topo, shortest_path_no_path)
{
    OcTopology topo;
    oc_topology_init(&topo);
    oc_topology_add_node(&topo, 1, "a", 80, 0);
    oc_topology_add_node(&topo, 2, "b", 80, 0);
    /* No link between 1 and 2. */
    uint64_t path[10];
    uint32_t n_hops;
    cr_assert_neq(oc_topology_shortest_path(&topo, 1, 2, path, 10, &n_hops), OC_OK);
    oc_topology_free(&topo);
}

Test(topo, n_nodes)
{
    OcTopology topo;
    oc_topology_init(&topo);
    cr_assert_eq(oc_topology_n_nodes(&topo), 0);
    oc_topology_add_node(&topo, 1, "a", 80, 0);
    cr_assert_eq(oc_topology_n_nodes(&topo), 1);
    oc_topology_free(&topo);
}

Test(topo, n_regions)
{
    OcTopology topo;
    oc_topology_init(&topo);
    oc_topology_add_node(&topo, 1, "a", 80, 5);
    oc_topology_add_node(&topo, 2, "b", 80, 7);
    cr_assert_eq(oc_topology_n_regions(&topo), 8);
    oc_topology_free(&topo);
}

Test(topo, free_null)
{
    oc_topology_free(NULL);
}
