/*
 * gpu_cluster.h — GPU family profiles and Kubernetes manifest generation.
 *
 * Port from oxidize-core/src/cluster/gpu_cluster.rs. Defines a catalog of
 * supported GPU families with their memory, power, and partitioning
 * characteristics, plus helpers to generate Kubernetes manifests (node pool
 * and device plugin YAML) for each family.
 */
#ifndef OXIDIZE_GPU_CLUSTER_H
#define OXIDIZE_GPU_CLUSTER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Supported GPU families. Order MUST stay stable (rank ordering and
 * enumeration depend on it). Append-only. */
typedef enum {
    OC_GPU_FAMILY_B200 = 0,
    OC_GPU_FAMILY_A100 = 1,
    OC_GPU_FAMILY_RTX_PRO_6000 = 2,
    OC_GPU_FAMILY_H100 = 3,
    OC_GPU_FAMILY__COUNT, /* sentinel; not a valid family */
} OcGpuFamily;

/* Hardware profile for a GPU family. All string fields point to static
 * storage and are valid for the lifetime of the program. */
typedef struct {
    OcGpuFamily family;
    const char *product;             /* product identifier string */
    const char *generation;          /* architecture generation */
    uint32_t memory_mib;             /* device memory in MiB */
    uint32_t tdp_watts;              /* thermal design power in watts */
    bool nvlink;                     /* NVLink fabric available */
    bool mig_capable;                /* Multi-Instance GPU support */
    uint32_t time_slice_replicas;   /* recommended time-slice partitions */
    const char *network_class;       /* "infiniband" or "ethernet" */
    const char *workload_type;       /* suggested workload class */
} OcGpuProfile;

/* Look up the static profile for a family. Returns NULL for an invalid
 * family value. The returned pointer is into static storage. */
const OcGpuProfile *oc_gpu_profile(OcGpuFamily family);

/* URL/label-safe slug for a family (e.g. "b200"). Returns "unknown" for
 * invalid values. Never returns NULL. */
const char *oc_gpu_family_slug(OcGpuFamily family);

/* Human-readable family name (e.g. "B200"). Returns "unknown" for invalid
 * values. Never returns NULL. */
const char *oc_gpu_family_name(OcGpuFamily family);

/* Parse a family from a slug. Case-insensitive. Returns OC_GPU_FAMILY__COUNT
 * for an unrecognized slug (including NULL). */
OcGpuFamily oc_gpu_family_from_slug(const char *slug);

/* Performance/tier rank of a family (higher = more capable). Stable for the
 * lifetime of the program. */
uint8_t oc_gpu_family_rank(OcGpuFamily family);

/* Number of supported GPU families (excludes the sentinel). */
size_t oc_gpu_n_families(void);

/* Fetch a family by enumeration index. Returns OC_GPU_FAMILY__COUNT for
 * out-of-range indices. */
OcGpuFamily oc_gpu_family_by_index(size_t idx);

/* Lookup a common Kubernetes label value by key. Recognized keys:
 *   "nvidia.com/gpu.product"  -> product string
 *   "nvidia.com/gpu.memory"  -> memory in MiB (as decimal string)
 *   "nvidia.com/gpu.count"    -> time-slice replicas (as decimal string)
 *   "network"                -> network class
 *   "workload"               -> workload type
 * Returns NULL for unrecognized keys. */
const char *oc_gpu_cluster_label(const char *key);

/* Render a Kubernetes NodePool manifest for the given family into `out`.
 * `replicas` selects the node count. The YAML is NUL-terminated. Returns
 * OC_ERR_INVALID_ARG for invalid families, OC_ERR_INVALID_ARG for NULL out
 * buffer, OC_ERR_INTERNAL if the manifest would overflow `out_len`. */
OcError oc_gpu_cluster_node_pool_yaml(OcGpuFamily family, uint32_t replicas,
                                      char *out, size_t out_len);

/* Render a Kubernetes NVIDIA device plugin manifest for the given family
 * into `out`. The YAML is NUL-terminated. Returns OC_ERR_INVALID_ARG for
 * invalid families, OC_ERR_INVALID_ARG for NULL out buffer, OC_ERR_INTERNAL
 * if the manifest would overflow `out_len`. */
OcError oc_gpu_cluster_device_plugin_yaml(OcGpuFamily family, char *out, size_t out_len);

/* Classify an NVML / nvidia-smi product name into a family. Pure, case-
 * insensitive substring match (H100, H100 80GB, NVIDIA H100 SXM5, …).
 * Returns OC_GPU_FAMILY__COUNT for NULL or unrecognized names. A plain
 * "RTX 6000" (non-Pro) does not match. */
OcGpuFamily oc_gpu_family_from_nvidia_name(const char *name);

/* One GPU reported by nvidia-smi. `family` is OC_GPU_FAMILY__COUNT when
 * the product name is not one of the catalogued families. */
typedef struct OcGpuDevice {
    uint32_t    index;
    char        name[128];
    uint32_t    memory_total_mib;
    bool        mig_enabled;
    OcGpuFamily family;
} OcGpuDevice;

/* Parse CSV from
 *   nvidia-smi --query-gpu=index,name,memory.total,mig.mode.current
 *               --format=csv,noheader,nounits
 * Unparseable lines are skipped. Writes at most `cap` entries into `out`.
 * `out_n` receives the number written. */
OcError oc_gpu_parse_nvidia_smi_csv(const char *output, OcGpuDevice *out,
                                      size_t cap, size_t *out_n);

/* Probe nvidia-smi. Missing binary or a failed probe is not an error:
 * returns OC_OK with *out_n = 0. */
OcError oc_gpu_detect(OcGpuDevice *out, size_t cap, size_t *out_n);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_GPU_CLUSTER_H */
