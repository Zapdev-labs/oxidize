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

/* Lookup a common Kubernetes label value by key. Recognized keys: */
const char *oc_gpu_cluster_label(const char *key);

/* Render a Kubernetes NodePool manifest for the given family into `out`. */
OcError oc_gpu_cluster_node_pool_yaml(OcGpuFamily family, uint32_t replicas,
                                      char *out, size_t out_len);

/* Render a Kubernetes NVIDIA device plugin manifest for the given family */
OcError oc_gpu_cluster_device_plugin_yaml(OcGpuFamily family, char *out, size_t out_len);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_GPU_CLUSTER_H */
