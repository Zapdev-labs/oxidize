#define _POSIX_C_SOURCE 200809L

#include "oxidize/gpu_cluster.h"

#include <stdio.h>
#include <string.h>
#include <strings.h> /* strcasecmp */

/* Static GPU family profiles.                                         */

static const OcGpuProfile OC_PROFILES[] = {
    [OC_GPU_FAMILY_B200] = {
        .family = OC_GPU_FAMILY_B200,
        .product = "NVIDIA-B200",
        .generation = "blackwell",
        .memory_mib = 196608u,
        .tdp_watts = 1000u,
        .nvlink = true,
        .mig_capable = false,
        .time_slice_replicas = 1u,
        .network_class = "infiniband",
        .workload_type = "training",
    },
    [OC_GPU_FAMILY_A100] = {
        .family = OC_GPU_FAMILY_A100,
        .product = "NVIDIA-A100-SXM4-80GB",
        .generation = "ampere",
        .memory_mib = 81920u,
        .tdp_watts = 400u,
        .nvlink = true,
        .mig_capable = true,
        .time_slice_replicas = 2u,
        .network_class = "infiniband",
        .workload_type = "inference",
    },
    [OC_GPU_FAMILY_RTX_PRO_6000] = {
        .family = OC_GPU_FAMILY_RTX_PRO_6000,
        .product = "NVIDIA-RTX-PRO-6000",
        .generation = "ada",
        .memory_mib = 49152u,
        .tdp_watts = 300u,
        .nvlink = false,
        .mig_capable = false,
        .time_slice_replicas = 4u,
        .network_class = "ethernet",
        .workload_type = "edge",
    },
};

static const char *const OC_FAMILY_SLUGS[] = {
    [OC_GPU_FAMILY_B200] = "b200",
    [OC_GPU_FAMILY_A100] = "a100",
    [OC_GPU_FAMILY_RTX_PRO_6000] = "rtx-pro-6000",
};

static const char *const OC_FAMILY_NAMES[] = {
    [OC_GPU_FAMILY_B200] = "B200",
    [OC_GPU_FAMILY_A100] = "A100",
    [OC_GPU_FAMILY_RTX_PRO_6000] = "RTX Pro 6000",
};

/* Higher rank = more capable. B200 is the flagship, RTX Pro 6000 the
 * entry-level datacenter option. */
static const uint8_t OC_FAMILY_RANKS[] = {
    [OC_GPU_FAMILY_B200] = 3u,
    [OC_GPU_FAMILY_A100] = 2u,
    [OC_GPU_FAMILY_RTX_PRO_6000] = 1u,
};

#define OC_N_FAMILIES \
    (sizeof(OC_PROFILES) / sizeof(OC_PROFILES[0]))

/* Helpers.                                                            */

static bool family_valid(OcGpuFamily f)
{
    return (size_t)f < OC_N_FAMILIES;
}

/* Public API.                                                         */

const OcGpuProfile *oc_gpu_profile(OcGpuFamily family)
{
    if (!family_valid(family)) return NULL;
    return &OC_PROFILES[family];
}

const char *oc_gpu_family_slug(OcGpuFamily family)
{
    if (!family_valid(family)) return "unknown";
    return OC_FAMILY_SLUGS[family];
}

const char *oc_gpu_family_name(OcGpuFamily family)
{
    if (!family_valid(family)) return "unknown";
    return OC_FAMILY_NAMES[family];
}

OcGpuFamily oc_gpu_family_from_slug(const char *slug)
{
    if (!slug) return OC_GPU_FAMILY__COUNT;
    for (size_t i = 0; i < OC_N_FAMILIES; i++) {
        if (strcasecmp(slug, OC_FAMILY_SLUGS[i]) == 0)
            return (OcGpuFamily)i;
    }
    return OC_GPU_FAMILY__COUNT;
}

uint8_t oc_gpu_family_rank(OcGpuFamily family)
{
    if (!family_valid(family)) return 0u;
    return OC_FAMILY_RANKS[family];
}

size_t oc_gpu_n_families(void)
{
    return OC_N_FAMILIES;
}

OcGpuFamily oc_gpu_family_by_index(size_t idx)
{
    if (idx >= OC_N_FAMILIES) return OC_GPU_FAMILY__COUNT;
    return (OcGpuFamily)idx;
}

const char *oc_gpu_cluster_label(const char *key)
{
    if (!key) return NULL;
    /* The label function returns the value for the B200 family by default,
     * mirroring the Rust "default tier" behavior. Callers that need per-family
     * label values can read the profile directly. */
    const OcGpuProfile *p = &OC_PROFILES[OC_GPU_FAMILY_B200];
    static char mem_buf[32];
    static char count_buf[32];

    if (strcmp(key, "nvidia.com/gpu.product") == 0) return p->product;
    if (strcmp(key, "nvidia.com/gpu.memory") == 0) {
        snprintf(mem_buf, sizeof(mem_buf), "%u", p->memory_mib);
        return mem_buf;
    }
    if (strcmp(key, "nvidia.com/gpu.count") == 0) {
        snprintf(count_buf, sizeof(count_buf), "%u", p->time_slice_replicas);
        return count_buf;
    }
    if (strcmp(key, "network") == 0) return p->network_class;
    if (strcmp(key, "workload") == 0) return p->workload_type;
    return NULL;
}

/* Manifest generation.                                                */

OcError oc_gpu_cluster_node_pool_yaml(OcGpuFamily family, uint32_t replicas,
                                      char *out, size_t out_len)
{
    if (!family_valid(family)) return OC_ERR_INVALID_ARG;
    if (!out || out_len == 0) return OC_ERR_INVALID_ARG;

    const OcGpuProfile *p = &OC_PROFILES[family];
    int n = snprintf(out, out_len,
        "apiVersion: apps/v1\n"
        "kind: NodePool\n"
        "metadata:\n"
        "  name: gpu-%s-pool\n"
        "  labels:\n"
        "    nvidia.com/gpu.product: \"%s\"\n"
        "    nvidia.com/gpu.memory: \"%u\"\n"
        "    oxidize.io/gpu.family: \"%s\"\n"
        "    oxidize.io/gpu.generation: \"%s\"\n"
        "    oxidize.io/network.class: \"%s\"\n"
        "    oxidize.io/workload.type: \"%s\"\n"
        "spec:\n"
        "  replicas: %u\n"
        "  nodeSelector:\n"
        "    nvidia.com/gpu.present: \"true\"\n"
        "  tolerations:\n"
        "    - key: nvidia.com/gpu\n"
        "      operator: Exists\n"
        "  config:\n"
        "    gpu:\n"
        "      memoryMiB: %u\n"
        "      tdpWatts: %u\n"
        "      nvlink: %s\n"
        "      mig: %s\n"
        "      timeSliceReplicas: %u\n",
        OC_FAMILY_SLUGS[family],
        p->product,
        p->memory_mib,
        OC_FAMILY_SLUGS[family],
        p->generation,
        p->network_class,
        p->workload_type,
        replicas,
        p->memory_mib,
        p->tdp_watts,
        p->nvlink ? "true" : "false",
        p->mig_capable ? "true" : "false",
        p->time_slice_replicas);

    if (n < 0) return OC_ERR_INTERNAL;
    if ((size_t)n >= out_len) return OC_ERR_INTERNAL;
    return OC_OK;
}

OcError oc_gpu_cluster_device_plugin_yaml(OcGpuFamily family, char *out, size_t out_len)
{
    if (!family_valid(family)) return OC_ERR_INVALID_ARG;
    if (!out || out_len == 0) return OC_ERR_INVALID_ARG;

    const OcGpuProfile *p = &OC_PROFILES[family];
    int n = snprintf(out, out_len,
        "apiVersion: apps/v1\n"
        "kind: DaemonSet\n"
        "metadata:\n"
        "  name: nvidia-device-plugin-%s\n"
        "  namespace: kube-system\n"
        "  labels:\n"
        "    oxidize.io/gpu.family: \"%s\"\n"
        "spec:\n"
        "  selector:\n"
        "    matchLabels:\n"
        "      name: nvidia-device-plugin-%s\n"
        "  template:\n"
        "    metadata:\n"
        "      labels:\n"
        "        name: nvidia-device-plugin-%s\n"
        "    spec:\n"
        "      nodeSelector:\n"
        "        nvidia.com/gpu.product: \"%s\"\n"
        "      tolerations:\n"
        "        - key: nvidia.com/gpu\n"
        "          operator: Exists\n"
        "      containers:\n"
        "        - name: nvidia-device-plugin\n"
        "          image: nvcr.io/nvidia/k8s-device-plugin:v0.14.5\n"
        "          env:\n"
        "            - name: NVIDIA_GPU_MEMORY_MIB\n"
        "              value: \"%u\"\n"
        "            - name: NVIDIA_TIME_SLICE_REPLICAS\n"
        "              value: \"%u\"\n"
        "            - name: NVIDIA_MIG_ENABLED\n"
        "              value: \"%s\"\n"
        "          securityContext:\n"
        "            privileged: true\n",
        OC_FAMILY_SLUGS[family],
        OC_FAMILY_SLUGS[family],
        OC_FAMILY_SLUGS[family],
        OC_FAMILY_SLUGS[family],
        p->product,
        p->memory_mib,
        p->time_slice_replicas,
        p->mig_capable ? "true" : "false");

    if (n < 0) return OC_ERR_INTERNAL;
    if ((size_t)n >= out_len) return OC_ERR_INTERNAL;
    return OC_OK;
}
