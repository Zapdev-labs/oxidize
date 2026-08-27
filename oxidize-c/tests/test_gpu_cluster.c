/* test_gpu_cluster.c — GPU family profile + manifest generation tests. */
#include <criterion/criterion.h>
#include "oxidize/gpu_cluster.h"
#include <string.h>

/* ----------------------------------------------------------------- */
/* Profile lookups.                                                   */
/* ----------------------------------------------------------------- */

Test(gpu_cluster, profile_b200)
{
    const OcGpuProfile *p = oc_gpu_profile(OC_GPU_FAMILY_B200);
    cr_assert_not_null(p);
    cr_assert_str_eq(p->product, "NVIDIA-B200");
    cr_assert_str_eq(p->generation, "blackwell");
    cr_assert_eq(p->memory_mib, 196608u);
    cr_assert_eq(p->tdp_watts, 1000u);
    cr_assert(p->nvlink);
    cr_assert(!p->mig_capable);
    cr_assert_eq(p->time_slice_replicas, 1u);
    cr_assert_str_eq(p->network_class, "infiniband");
    cr_assert_str_eq(p->workload_type, "training");
}

Test(gpu_cluster, profile_a100)
{
    const OcGpuProfile *p = oc_gpu_profile(OC_GPU_FAMILY_A100);
    cr_assert_not_null(p);
    cr_assert_str_eq(p->product, "NVIDIA-A100-SXM4-80GB");
    cr_assert_str_eq(p->generation, "ampere");
    cr_assert_eq(p->memory_mib, 81920u);
    cr_assert_eq(p->tdp_watts, 400u);
    cr_assert(p->nvlink);
    cr_assert(p->mig_capable);
    cr_assert_eq(p->time_slice_replicas, 2u);
    cr_assert_str_eq(p->network_class, "infiniband");
    cr_assert_str_eq(p->workload_type, "inference");
}

Test(gpu_cluster, profile_rtx_pro_6000)
{
    const OcGpuProfile *p = oc_gpu_profile(OC_GPU_FAMILY_RTX_PRO_6000);
    cr_assert_not_null(p);
    cr_assert_str_eq(p->product, "NVIDIA-RTX-PRO-6000");
    cr_assert_str_eq(p->generation, "ada");
    cr_assert_eq(p->memory_mib, 49152u);
    cr_assert_eq(p->tdp_watts, 300u);
    cr_assert(!p->nvlink);
    cr_assert(!p->mig_capable);
    cr_assert_eq(p->time_slice_replicas, 4u);
    cr_assert_str_eq(p->network_class, "ethernet");
    cr_assert_str_eq(p->workload_type, "edge");
}

Test(gpu_cluster, profile_invalid_returns_null)
{
    cr_assert_null(oc_gpu_profile(OC_GPU_FAMILY__COUNT));
    cr_assert_null(oc_gpu_profile((OcGpuFamily)99));
}

/* ----------------------------------------------------------------- */
/* Slug / name strings.                                               */
/* ----------------------------------------------------------------- */

Test(gpu_cluster, slug_strings)
{
    cr_assert_str_eq(oc_gpu_family_slug(OC_GPU_FAMILY_B200), "b200");
    cr_assert_str_eq(oc_gpu_family_slug(OC_GPU_FAMILY_A100), "a100");
    cr_assert_str_eq(oc_gpu_family_slug(OC_GPU_FAMILY_RTX_PRO_6000), "rtx-pro-6000");
    cr_assert_str_eq(oc_gpu_family_slug(OC_GPU_FAMILY__COUNT), "unknown");
}

Test(gpu_cluster, name_strings)
{
    cr_assert_str_eq(oc_gpu_family_name(OC_GPU_FAMILY_B200), "B200");
    cr_assert_str_eq(oc_gpu_family_name(OC_GPU_FAMILY_A100), "A100");
    cr_assert_str_eq(oc_gpu_family_name(OC_GPU_FAMILY_RTX_PRO_6000), "RTX Pro 6000");
    cr_assert_str_eq(oc_gpu_family_name(OC_GPU_FAMILY__COUNT), "unknown");
}

/* ----------------------------------------------------------------- */
/* Slug parsing.                                                      */
/* ----------------------------------------------------------------- */

Test(gpu_cluster, from_slug_canonical)
{
    cr_assert_eq(oc_gpu_family_from_slug("b200"), OC_GPU_FAMILY_B200);
    cr_assert_eq(oc_gpu_family_from_slug("a100"), OC_GPU_FAMILY_A100);
    cr_assert_eq(oc_gpu_family_from_slug("rtx-pro-6000"), OC_GPU_FAMILY_RTX_PRO_6000);
}

Test(gpu_cluster, from_slug_case_insensitive)
{
    cr_assert_eq(oc_gpu_family_from_slug("B200"), OC_GPU_FAMILY_B200);
    cr_assert_eq(oc_gpu_family_from_slug("A100"), OC_GPU_FAMILY_A100);
    cr_assert_eq(oc_gpu_family_from_slug("RTX-PRO-6000"), OC_GPU_FAMILY_RTX_PRO_6000);
}

Test(gpu_cluster, from_slug_invalid)
{
    cr_assert_eq(oc_gpu_family_from_slug("h100"), OC_GPU_FAMILY__COUNT);
    cr_assert_eq(oc_gpu_family_from_slug(""), OC_GPU_FAMILY__COUNT);
    cr_assert_eq(oc_gpu_family_from_slug(NULL), OC_GPU_FAMILY__COUNT);
}

/* ----------------------------------------------------------------- */
/* Rank ordering.                                                     */
/* ----------------------------------------------------------------- */

Test(gpu_cluster, rank_ordering)
{
    cr_assert(oc_gpu_family_rank(OC_GPU_FAMILY_B200) >
              oc_gpu_family_rank(OC_GPU_FAMILY_A100));
    cr_assert(oc_gpu_family_rank(OC_GPU_FAMILY_A100) >
              oc_gpu_family_rank(OC_GPU_FAMILY_RTX_PRO_6000));
    cr_assert_eq(oc_gpu_family_rank(OC_GPU_FAMILY__COUNT), 0u);
}

Test(gpu_cluster, rank_values)
{
    cr_assert_eq(oc_gpu_family_rank(OC_GPU_FAMILY_B200), 3u);
    cr_assert_eq(oc_gpu_family_rank(OC_GPU_FAMILY_A100), 2u);
    cr_assert_eq(oc_gpu_family_rank(OC_GPU_FAMILY_RTX_PRO_6000), 1u);
}

/* ----------------------------------------------------------------- */
/* Enumeration.                                                       */
/* ----------------------------------------------------------------- */

Test(gpu_cluster, n_families)
{
    cr_assert_eq(oc_gpu_n_families(), 3u);
}

Test(gpu_cluster, family_by_index)
{
    cr_assert_eq(oc_gpu_family_by_index(0), OC_GPU_FAMILY_B200);
    cr_assert_eq(oc_gpu_family_by_index(1), OC_GPU_FAMILY_A100);
    cr_assert_eq(oc_gpu_family_by_index(2), OC_GPU_FAMILY_RTX_PRO_6000);
}

Test(gpu_cluster, family_by_index_out_of_range)
{
    cr_assert_eq(oc_gpu_family_by_index(3), OC_GPU_FAMILY__COUNT);
    cr_assert_eq(oc_gpu_family_by_index(99), OC_GPU_FAMILY__COUNT);
}

Test(gpu_cluster, enumerate_all_families)
{
    size_t n = oc_gpu_n_families();
    cr_assert_eq(n, 3u);
    for (size_t i = 0; i < n; i++) {
        OcGpuFamily f = oc_gpu_family_by_index(i);
        cr_assert_not_null(oc_gpu_profile(f));
        cr_assert_str_neq(oc_gpu_family_slug(f), "unknown");
    }
}

/* ----------------------------------------------------------------- */
/* Label lookup.                                                      */
/* ----------------------------------------------------------------- */

Test(gpu_cluster, label_product)
{
    const char *v = oc_gpu_cluster_label("nvidia.com/gpu.product");
    cr_assert_not_null(v);
    cr_assert_str_eq(v, "NVIDIA-B200");
}

Test(gpu_cluster, label_network)
{
    cr_assert_str_eq(oc_gpu_cluster_label("network"), "infiniband");
    cr_assert_str_eq(oc_gpu_cluster_label("workload"), "training");
}

Test(gpu_cluster, label_unknown)
{
    cr_assert_null(oc_gpu_cluster_label("not.a.real.key"));
    cr_assert_null(oc_gpu_cluster_label(NULL));
}

/* ----------------------------------------------------------------- */
/* Manifest generation.                                               */
/* ----------------------------------------------------------------- */

Test(gpu_cluster, node_pool_yaml_b200)
{
    char buf[2048];
    OcError e = oc_gpu_cluster_node_pool_yaml(OC_GPU_FAMILY_B200, 2, buf, sizeof(buf));
    cr_assert_eq(e, OC_OK);
    cr_assert_gt(strlen(buf), 0u);
    cr_assert_not_null(strstr(buf, "NVIDIA-B200"));
    cr_assert_not_null(strstr(buf, "gpu-b200-pool"));
    cr_assert_not_null(strstr(buf, "replicas: 2"));
    cr_assert_not_null(strstr(buf, "blackwell"));
}

Test(gpu_cluster, node_pool_yaml_a100)
{
    char buf[2048];
    OcError e = oc_gpu_cluster_node_pool_yaml(OC_GPU_FAMILY_A100, 4, buf, sizeof(buf));
    cr_assert_eq(e, OC_OK);
    cr_assert_not_null(strstr(buf, "NVIDIA-A100-SXM4-80GB"));
    cr_assert_not_null(strstr(buf, "ampere"));
    cr_assert_not_null(strstr(buf, "replicas: 4"));
}

Test(gpu_cluster, node_pool_yaml_rtx)
{
    char buf[2048];
    OcError e = oc_gpu_cluster_node_pool_yaml(OC_GPU_FAMILY_RTX_PRO_6000, 8, buf, sizeof(buf));
    cr_assert_eq(e, OC_OK);
    cr_assert_not_null(strstr(buf, "NVIDIA-RTX-PRO-6000"));
    cr_assert_not_null(strstr(buf, "ada"));
    cr_assert_not_null(strstr(buf, "ethernet"));
    cr_assert_not_null(strstr(buf, "edge"));
}

Test(gpu_cluster, node_pool_yaml_invalid_family)
{
    char buf[64];
    cr_assert_neq(oc_gpu_cluster_node_pool_yaml(OC_GPU_FAMILY__COUNT, 1, buf, sizeof(buf)), OC_OK);
}

Test(gpu_cluster, node_pool_yaml_null_out)
{
    cr_assert_neq(oc_gpu_cluster_node_pool_yaml(OC_GPU_FAMILY_B200, 1, NULL, 0), OC_OK);
}

Test(gpu_cluster, node_pool_yaml_overflow)
{
    char buf[8];
    cr_assert_neq(oc_gpu_cluster_node_pool_yaml(OC_GPU_FAMILY_B200, 1, buf, sizeof(buf)), OC_OK);
}

Test(gpu_cluster, device_plugin_yaml_b200)
{
    char buf[4096];
    OcError e = oc_gpu_cluster_device_plugin_yaml(OC_GPU_FAMILY_B200, buf, sizeof(buf));
    cr_assert_eq(e, OC_OK);
    cr_assert_gt(strlen(buf), 0u);
    cr_assert_not_null(strstr(buf, "nvidia-device-plugin-b200"));
    cr_assert_not_null(strstr(buf, "NVIDIA-B200"));
    cr_assert_not_null(strstr(buf, "196608"));
}

Test(gpu_cluster, device_plugin_yaml_a100)
{
    char buf[4096];
    OcError e = oc_gpu_cluster_device_plugin_yaml(OC_GPU_FAMILY_A100, buf, sizeof(buf));
    cr_assert_eq(e, OC_OK);
    cr_assert_not_null(strstr(buf, "nvidia-device-plugin-a100"));
    cr_assert_not_null(strstr(buf, "NVIDIA-A100-SXM4-80GB"));
    cr_assert_not_null(strstr(buf, "NVIDIA_MIG_ENABLED"));
    cr_assert_not_null(strstr(buf, "\"true\""));
}

Test(gpu_cluster, device_plugin_yaml_rtx)
{
    char buf[4096];
    OcError e = oc_gpu_cluster_device_plugin_yaml(OC_GPU_FAMILY_RTX_PRO_6000, buf, sizeof(buf));
    cr_assert_eq(e, OC_OK);
    cr_assert_not_null(strstr(buf, "nvidia-device-plugin-rtx-pro-6000"));
    cr_assert_not_null(strstr(buf, "NVIDIA-RTX-PRO-6000"));
}

Test(gpu_cluster, device_plugin_yaml_invalid_family)
{
    char buf[64];
    cr_assert_neq(oc_gpu_cluster_device_plugin_yaml(OC_GPU_FAMILY__COUNT, buf, sizeof(buf)), OC_OK);
}

Test(gpu_cluster, device_plugin_yaml_null_out)
{
    cr_assert_neq(oc_gpu_cluster_device_plugin_yaml(OC_GPU_FAMILY_B200, NULL, 0), OC_OK);
}
