/* test_numa.c — NUMA awareness tests. */
#include <criterion/criterion.h>
#include "oxidize/numa.h"
#include <string.h>

Test(numa, detect)
{
    OcNumaTopology topo;
    cr_assert_eq(oc_numa_detect(&topo), OC_OK);
    cr_assert(topo.n_nodes >= 1);
    cr_assert(topo.n_cpus_total >= 1);
}

Test(numa, detect_null)
{
    cr_assert_neq(oc_numa_detect(NULL), OC_OK);
}

Test(numa, available)
{
    /* On any system, this should return true or false without crashing. */
    bool avail = oc_numa_available();
    (void)avail;
}

Test(numa, node_for_cpu)
{
    uint32_t node;
    OcError e = oc_numa_node_for_cpu(0, &node);
    /* This may fail on non-Linux, but on Linux should work. */
    (void)e;
    (void)node;
}

Test(numa, current_node)
{
    uint32_t node;
    OcError e = oc_numa_current_node(&node);
    (void)e;
    (void)node;
}

Test(numa, set_policy)
{
    cr_assert_eq(oc_numa_set_policy(OC_NUMA_POLICY_DEFAULT, 0), OC_OK);
}

Test(numa, describe)
{
    OcNumaTopology topo;
    oc_numa_detect(&topo);
    char buf[1024];
    cr_assert_eq(oc_numa_describe(&topo, buf, sizeof(buf)), OC_OK);
    cr_assert(strlen(buf) > 0);
    cr_assert(strstr(buf, "NUMA") != NULL);
}

Test(numa, describe_null)
{
    cr_assert_neq(oc_numa_describe(NULL, NULL, 0), OC_OK);
}

Test(numa, recommended_threads)
{
    OcNumaTopology topo;
    oc_numa_detect(&topo);
    uint32_t threads = oc_numa_recommended_threads(&topo);
    cr_assert(threads >= 1);
}

Test(numa, recommended_threads_null)
{
    cr_assert_eq(oc_numa_recommended_threads(NULL), 1);
}

Test(numa, alloc_free)
{
    void *ptr = oc_numa_alloc(1024, 0);
    cr_assert_not_null(ptr);
    oc_numa_free(ptr, 1024);
}

Test(numa, alloc_large)
{
    void *ptr = oc_numa_alloc(2 * 1024 * 1024, 0); /* 2 MiB */
    cr_assert_not_null(ptr);
    memset(ptr, 0, 2 * 1024 * 1024);
    oc_numa_free(ptr, 2 * 1024 * 1024);
}

Test(numa, alloc_interleaved)
{
    void *ptr = oc_numa_alloc_interleaved(2 * 1024 * 1024);
    cr_assert_not_null(ptr);
    oc_numa_free(ptr, 2 * 1024 * 1024);
}

Test(numa, alloc_null_free)
{
    oc_numa_free(NULL, 0);
}

Test(numa, addr_node)
{
    uint32_t node;
    char buf[64] = {0};
    OcError e = oc_numa_addr_node(buf, &node);
    (void)e;
    (void)node;
}

Test(numa, bind_thread)
{
    /* This should succeed or be a no-op, not crash. */
    OcError e = oc_numa_bind_thread(0);
    (void)e;
}

Test(numa, pin_cpu)
{
    OcError e = oc_numa_pin_cpu(0);
    (void)e;
}
