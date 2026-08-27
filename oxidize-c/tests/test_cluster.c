/* test_cluster.c — GPU cluster tests. */
#include <criterion/criterion.h>
#include "oxidize/cluster.h"
#include <string.h>

Test(cluster, init)
{
    OcGpuCluster c;
    cr_assert_eq(oc_cluster_init(&c, 1), OC_OK);
    cr_assert_eq(c.n_nodes, 0);
    cr_assert_eq(c.self_id, 1);
    oc_cluster_free(&c);
}

Test(cluster, init_null)
{
    cr_assert_neq(oc_cluster_init(NULL, 0), OC_OK);
}

Test(cluster, add_node)
{
    OcGpuCluster c;
    oc_cluster_init(&c, 1);
    cr_assert_eq(oc_cluster_add_node(&c, "node1", "192.168.1.1", 8080,
        OC_CLUSTER_NODE_WORKER, 4, 24ULL * 1024 * 1024 * 1024), OC_OK);
    cr_assert_eq(c.n_nodes, 1);
    cr_assert_str_eq(c.nodes[0].name, "node1");
    cr_assert_eq(c.nodes[0].n_gpus, 4);
    oc_cluster_free(&c);
}

Test(cluster, add_duplicate)
{
    OcGpuCluster c;
    oc_cluster_init(&c, 1);
    oc_cluster_add_node(&c, "node1", "addr1", 80, OC_CLUSTER_NODE_WORKER, 1, 1024);
    oc_cluster_add_node(&c, "node1", "addr1", 80, OC_CLUSTER_NODE_WORKER, 1, 1024);
    cr_assert_eq(c.n_nodes, 1);
    oc_cluster_free(&c);
}

Test(cluster, add_null)
{
    cr_assert_neq(oc_cluster_add_node(NULL, NULL, NULL, 0, 0, 0, 0), OC_OK);
}

Test(cluster, remove_node)
{
    OcGpuCluster c;
    oc_cluster_init(&c, 1);
    oc_cluster_add_node(&c, "node1", "addr", 80, OC_CLUSTER_NODE_WORKER, 2, 1024);
    cr_assert_eq(oc_cluster_remove_node(&c, 1), OC_OK);
    cr_assert_eq(c.n_nodes, 0);
    oc_cluster_free(&c);
}

Test(cluster, get_node)
{
    OcGpuCluster c;
    oc_cluster_init(&c, 1);
    oc_cluster_add_node(&c, "node1", "addr", 80, OC_CLUSTER_NODE_MASTER, 4, 1024);
    const OcClusterNode *n;
    cr_assert_eq(oc_cluster_get_node(&c, 1, &n), OC_OK);
    cr_assert_str_eq(n->name, "node1");
    oc_cluster_free(&c);
}

Test(cluster, get_node_not_found)
{
    OcGpuCluster c;
    oc_cluster_init(&c, 1);
    const OcClusterNode *n;
    cr_assert_neq(oc_cluster_get_node(&c, 999, &n), OC_OK);
    oc_cluster_free(&c);
}

Test(cluster, list_nodes)
{
    OcGpuCluster c;
    oc_cluster_init(&c, 1);
    oc_cluster_add_node(&c, "n1", "a1", 80, OC_CLUSTER_NODE_WORKER, 1, 1024);
    oc_cluster_add_node(&c, "n2", "a2", 80, OC_CLUSTER_NODE_WORKER, 2, 2048);
    const OcClusterNode *arr;
    uint32_t count;
    cr_assert_eq(oc_cluster_list_nodes(&c, &arr, &count), OC_OK);
    cr_assert_eq(count, 2);
    oc_cluster_free(&c);
}

Test(cluster, find_best_node)
{
    OcGpuCluster c;
    oc_cluster_init(&c, 1);
    oc_cluster_add_node(&c, "n1", "a1", 80, OC_CLUSTER_NODE_WORKER, 1, 1024);
    oc_cluster_add_node(&c, "n2", "a2", 80, OC_CLUSTER_NODE_WORKER, 1, 2048);
    const OcClusterNode *best;
    cr_assert_eq(oc_cluster_find_best_node(&c, 512, &best), OC_OK);
    cr_assert_not_null(best);
    /* Best fit: n1 has less free memory but still enough. */
    cr_assert(best->free_gpu_memory >= 512);
    oc_cluster_free(&c);
}

Test(cluster, find_best_no_fit)
{
    OcGpuCluster c;
    oc_cluster_init(&c, 1);
    oc_cluster_add_node(&c, "n1", "a1", 80, OC_CLUSTER_NODE_WORKER, 1, 512);
    const OcClusterNode *best;
    cr_assert_neq(oc_cluster_find_best_node(&c, 2048, &best), OC_OK);
    oc_cluster_free(&c);
}

Test(cluster, n_nodes)
{
    OcGpuCluster c;
    oc_cluster_init(&c, 1);
    cr_assert_eq(oc_cluster_n_nodes(&c), 0);
    oc_cluster_add_node(&c, "n1", "a", 80, 0, 1, 1024);
    cr_assert_eq(oc_cluster_n_nodes(&c), 1);
    oc_cluster_free(&c);
}

Test(cluster, n_gpus)
{
    OcGpuCluster c;
    oc_cluster_init(&c, 1);
    oc_cluster_add_node(&c, "n1", "a", 80, 0, 4, 1024);
    cr_assert_eq(oc_cluster_n_gpus(&c), 4);
    oc_cluster_free(&c);
}

Test(cluster, total_memory)
{
    OcGpuCluster c;
    oc_cluster_init(&c, 1);
    oc_cluster_add_node(&c, "n1", "a", 80, 0, 1, 1024);
    cr_assert_eq(oc_cluster_total_memory(&c), 1024);
    oc_cluster_free(&c);
}

Test(cluster, assign_task)
{
    OcGpuCluster c;
    oc_cluster_init(&c, 1);
    oc_cluster_add_node(&c, "n1", "a", 80, 0, 4, 4096);
    OcClusterTask task = { .model_size = 1024, .n_layers = 32, .gpu_id = 0 };
    strcpy(task.model_name, "test-model");
    uint64_t node_id;
    cr_assert_eq(oc_cluster_assign_task(&c, &task, &node_id), OC_OK);
    cr_assert_eq(c.free_gpu_memory, 3072);
    oc_cluster_free(&c);
}

Test(cluster, assign_task_no_fit)
{
    OcGpuCluster c;
    oc_cluster_init(&c, 1);
    oc_cluster_add_node(&c, "n1", "a", 80, 0, 1, 512);
    OcClusterTask task = { .model_size = 2048 };
    uint64_t node_id;
    cr_assert_neq(oc_cluster_assign_task(&c, &task, &node_id), OC_OK);
    oc_cluster_free(&c);
}

Test(cluster, node_type_name)
{
    cr_assert_str_eq(oc_cluster_node_type_name(OC_CLUSTER_NODE_WORKER), "worker");
    cr_assert_str_eq(oc_cluster_node_type_name(OC_CLUSTER_NODE_MASTER), "master");
    cr_assert_str_eq(oc_cluster_node_type_name(OC_CLUSTER_NODE_STANDBY), "standby");
}

Test(cluster, master_idx)
{
    OcGpuCluster c;
    oc_cluster_init(&c, 1);
    oc_cluster_add_node(&c, "n1", "a", 80, OC_CLUSTER_NODE_MASTER, 1, 1024);
    cr_assert_eq(c.master_idx, 0);
    oc_cluster_free(&c);
}

Test(cluster, free_memory)
{
    OcGpuCluster c;
    oc_cluster_init(&c, 1);
    oc_cluster_add_node(&c, "n1", "a", 80, 0, 1, 1024);
    cr_assert_eq(oc_cluster_free_memory(&c), 1024);
    oc_cluster_free(&c);
}

Test(cluster, free_null)
{
    oc_cluster_free(NULL);
}
