/* test_k8s.c — Kubernetes stub tests. */
#include <criterion/criterion.h>
#include "oxidize/k8s.h"
#include <stdio.h>
#include <string.h>

Test(k8s, init)
{
    OcK8sCluster cluster;
    cr_assert_eq(oc_k8s_init(&cluster, "default", "oxidize"), OC_OK);
    cr_assert_str_eq(cluster.namespace, "default");
    cr_assert_str_eq(cluster.service_name, "oxidize");
    cr_assert_eq(cluster.n_pods, 0);
    cr_assert(!cluster.available);
    oc_k8s_free(&cluster);
}

Test(k8s, init_null_ns)
{
    OcK8sCluster cluster;
    cr_assert_eq(oc_k8s_init(&cluster, NULL, NULL), OC_OK);
    cr_assert_str_eq(cluster.namespace, "default");
    oc_k8s_free(&cluster);
}

Test(k8s, init_null)
{
    cr_assert_neq(oc_k8s_init(NULL, NULL, NULL), OC_OK);
}

Test(k8s, detect)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    cr_assert_eq(oc_k8s_detect(&cluster), OC_OK);
    oc_k8s_free(&cluster);
}

Test(k8s, add_pod)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    cr_assert_eq(oc_k8s_add_pod(&cluster, "pod-1", "10.0.0.1", 8080), OC_OK);
    cr_assert_eq(cluster.n_pods, 1);
    cr_assert_str_eq(cluster.pods[0].name, "pod-1");
    cr_assert_str_eq(cluster.pods[0].ip, "10.0.0.1");
    cr_assert_eq(cluster.pods[0].port, 8080);
    oc_k8s_free(&cluster);
}

Test(k8s, add_pod_null)
{
    cr_assert_neq(oc_k8s_add_pod(NULL, NULL, NULL, 0), OC_OK);
}

Test(k8s, get_pods)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    oc_k8s_add_pod(&cluster, "p1", "ip1", 80);
    oc_k8s_add_pod(&cluster, "p2", "ip2", 80);
    const OcK8sPod *pods;
    uint32_t count;
    cr_assert_eq(oc_k8s_get_pods(&cluster, &pods, &count), OC_OK);
    cr_assert_eq(count, 2);
    oc_k8s_free(&cluster);
}

Test(k8s, get_ready_pods)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    oc_k8s_add_pod(&cluster, "p1", "ip1", 80);
    oc_k8s_mark_pod_ready(&cluster, "p1");
    const OcK8sPod *pods;
    uint32_t count;
    cr_assert_eq(oc_k8s_get_ready_pods(&cluster, &pods, &count), OC_OK);
    cr_assert_eq(count, 1);
    oc_k8s_free(&cluster);
}

Test(k8s, n_pods)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    cr_assert_eq(oc_k8s_n_pods(&cluster), 0);
    oc_k8s_add_pod(&cluster, "p1", "ip", 80);
    cr_assert_eq(oc_k8s_n_pods(&cluster), 1);
    oc_k8s_free(&cluster);
}

Test(k8s, n_ready)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    oc_k8s_add_pod(&cluster, "p1", "ip", 80);
    oc_k8s_add_pod(&cluster, "p2", "ip", 80);
    oc_k8s_mark_pod_ready(&cluster, "p1");
    cr_assert_eq(oc_k8s_n_ready(&cluster), 1);
    oc_k8s_free(&cluster);
}

Test(k8s, is_available)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    cr_assert(!oc_k8s_is_available(&cluster));
    oc_k8s_free(&cluster);
}

Test(k8s, scale)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    /* Not available, should return error. */
    cr_assert_neq(oc_k8s_scale(&cluster, 3), OC_OK);
    cluster.available = true;
    cr_assert_eq(oc_k8s_scale(&cluster, 3), OC_OK);
    oc_k8s_free(&cluster);
}

Test(k8s, mark_pod_ready)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    oc_k8s_add_pod(&cluster, "p1", "ip", 80);
    cr_assert_eq(oc_k8s_mark_pod_ready(&cluster, "p1"), OC_OK);
    cr_assert(cluster.pods[0].ready);
    oc_k8s_free(&cluster);
}

Test(k8s, mark_pod_not_found)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    cr_assert_neq(oc_k8s_mark_pod_ready(&cluster, "nonexistent"), OC_OK);
    oc_k8s_free(&cluster);
}

Test(k8s, free_null)
{
    oc_k8s_free(NULL);
}

Test(k8s, add_multiple_pods)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    for (int i = 0; i < 10; i++) {
        char name[32];
        snprintf(name, sizeof(name), "pod-%d", i);
        oc_k8s_add_pod(&cluster, name, "10.0.0.1", 8080);
    }
    cr_assert_eq(cluster.n_pods, 10);
    oc_k8s_free(&cluster);
}
