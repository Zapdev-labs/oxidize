/* test_numa.c — NUMA awareness tests. */
#include "framework.h"
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

/* Every policy must be accepted, and the process must be left on a usable
 * policy afterwards. On a single-node box these are no-ops that still have to
 * return OC_OK rather than an error. */
Test(numa, set_policy_all_modes)
{
    cr_assert_eq(oc_numa_set_policy(OC_NUMA_POLICY_INTERLEAVE, 0), OC_OK);
    cr_assert_eq(oc_numa_set_policy(OC_NUMA_POLICY_BIND, 0), OC_OK);
    cr_assert_eq(oc_numa_set_policy(OC_NUMA_POLICY_PREFERRED, 0), OC_OK);
    cr_assert_eq(oc_numa_set_policy(OC_NUMA_POLICY_DEFAULT, 0), OC_OK);
}

/* A node id past the end of the topology is rejected, not silently applied to
 * the wrong node. Only meaningful on a real multi-node box; on a single-node
 * system the call short-circuits to OC_OK before validating. */
Test(numa, set_policy_bad_node)
{
    OcNumaTopology topo;
    cr_assert_eq(oc_numa_detect(&topo), OC_OK);
    OcError e = oc_numa_set_policy(OC_NUMA_POLICY_BIND, topo.n_nodes + 8);
    if (topo.available && topo.n_nodes > 1) {
        cr_assert_eq(e, OC_ERR_INVALID_ARG);
    } else {
        cr_assert_eq(e, OC_OK);
    }
    (void)oc_numa_set_policy(OC_NUMA_POLICY_DEFAULT, 0);
}

/* Node-bound and interleaved allocations must be usable memory regardless of
 * whether the kernel honored the placement (mbind is best-effort). */
Test(numa, alloc_bound_and_interleaved_are_writable)
{
    const size_t big = 4u << 20;  /* above the mmap threshold */
    unsigned char *a = (unsigned char *)oc_numa_alloc(big, 0);
    cr_assert_neq(a, NULL);
    memset(a, 0xA5, big);
    cr_assert_eq(a[0], 0xA5);
    cr_assert_eq(a[big - 1], 0xA5);
    oc_numa_free(a, big);

    unsigned char *b = (unsigned char *)oc_numa_alloc_interleaved(big);
    cr_assert_neq(b, NULL);
    memset(b, 0x5A, big);
    cr_assert_eq(b[0], 0x5A);
    cr_assert_eq(b[big - 1], 0x5A);
    oc_numa_free(b, big);
}

/* Small allocations take the malloc path; they must still round-trip. */
Test(numa, alloc_small_uses_malloc_path)
{
    unsigned char *p = (unsigned char *)oc_numa_alloc(128, 0);
    cr_assert_neq(p, NULL);
    memset(p, 0x11, 128);
    cr_assert_eq(p[127], 0x11);
    oc_numa_free(p, 128);
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
