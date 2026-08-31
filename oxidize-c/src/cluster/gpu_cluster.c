/*
 * gpu_cluster.c — GPU family profiles and Kubernetes manifest generation.
 *
 * Port from oxidize-core/src/cluster/gpu_cluster.rs.
 */
#define _POSIX_C_SOURCE 200809L

#include "oxidize/gpu_cluster.h"

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h> /* strcasecmp */

/* ------------------------------------------------------------------ */
/* Static GPU family profiles.                                         */
/* ------------------------------------------------------------------ */

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
    [OC_GPU_FAMILY_H100] = {
        .family = OC_GPU_FAMILY_H100,
        .product = "NVIDIA-H100-SXM5-80GB",
        .generation = "hopper",
        .memory_mib = 81920u,
        .tdp_watts = 700u,
        .nvlink = true,
        .mig_capable = false,
        .time_slice_replicas = 1u,
        .network_class = "infiniband",
        .workload_type = "throughput-inference",
    },
};

static const char *const OC_FAMILY_SLUGS[] = {
    [OC_GPU_FAMILY_B200] = "b200",
    [OC_GPU_FAMILY_A100] = "a100",
    [OC_GPU_FAMILY_RTX_PRO_6000] = "rtx-pro-6000",
    [OC_GPU_FAMILY_H100] = "h100",
};

static const char *const OC_FAMILY_NAMES[] = {
    [OC_GPU_FAMILY_B200] = "B200",
    [OC_GPU_FAMILY_A100] = "A100",
    [OC_GPU_FAMILY_RTX_PRO_6000] = "RTX Pro 6000",
    [OC_GPU_FAMILY_H100] = "H100",
};

/* Higher rank = more capable. B200 is the flagship, then H100, A100,
 * RTX Pro 6000. */
static const uint8_t OC_FAMILY_RANKS[] = {
    [OC_GPU_FAMILY_B200] = 4u,
    [OC_GPU_FAMILY_A100] = 2u,
    [OC_GPU_FAMILY_RTX_PRO_6000] = 1u,
    [OC_GPU_FAMILY_H100] = 3u,
};

#define OC_N_FAMILIES \
    (sizeof(OC_PROFILES) / sizeof(OC_PROFILES[0]))

/* ------------------------------------------------------------------ */
/* Helpers.                                                            */
/* ------------------------------------------------------------------ */

static bool family_valid(OcGpuFamily f)
{
    return (size_t)f < OC_N_FAMILIES;
}

/* ------------------------------------------------------------------ */
/* Public API.                                                         */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/* Manifest generation.                                                */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/* Runtime detection (pure classifier + nvidia-smi probe).           */
/* ------------------------------------------------------------------ */

static void ascii_lower_copy(char *dst, size_t cap, const char *src)
{
    size_t i = 0;
    if (cap == 0) return;
    if (src == NULL) {
        dst[0] = '\0';
        return;
    }
    for (; src[i] != '\0' && i + 1 < cap; i++) {
        char c = src[i];
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        dst[i] = c;
    }
    dst[i] = '\0';
}

OcGpuFamily oc_gpu_family_from_nvidia_name(const char *name)
{
    char n[256];
    if (name == NULL || name[0] == '\0') return OC_GPU_FAMILY__COUNT;
    ascii_lower_copy(n, sizeof(n), name);
    if (strstr(n, "b200") != NULL) return OC_GPU_FAMILY_B200;
    if (strstr(n, "h100") != NULL) return OC_GPU_FAMILY_H100;
    if (strstr(n, "a100") != NULL) return OC_GPU_FAMILY_A100;
    if (strstr(n, "rtx") != NULL && strstr(n, "pro") != NULL &&
        strstr(n, "6000") != NULL)
        return OC_GPU_FAMILY_RTX_PRO_6000;
    return OC_GPU_FAMILY__COUNT;
}

static void trim_copy(char *dst, size_t cap, const char *src, size_t len)
{
    while (len > 0 && (*src == ' ' || *src == '\t')) {
        src++;
        len--;
    }
    while (len > 0 && (src[len - 1] == ' ' || src[len - 1] == '\t' ||
                       src[len - 1] == '\r' || src[len - 1] == '\n')) {
        len--;
    }
    if (len >= cap) len = cap - 1;
    if (len > 0) memcpy(dst, src, len);
    dst[len] = '\0';
}

static bool parse_csv_u32(const char *s, uint32_t *out)
{
    char *end = NULL;
    unsigned long v;

    if (s == NULL || s[0] == '\0' || s[0] == '+' || s[0] == '-')
        return false;
    errno = 0;
    v = strtoul(s, &end, 10);
    if (end == s || *end != '\0') return false;
    if (errno == ERANGE || v > (unsigned long)UINT32_MAX) return false;
    *out = (uint32_t)v;
    return true;
}

OcError oc_gpu_parse_nvidia_smi_csv(const char *output, OcGpuDevice *out,
                                      size_t cap, size_t *out_n)
{
    if (out_n == NULL) return OC_ERR_INVALID_ARG;
    *out_n = 0;
    if (output == NULL) return OC_OK;
    if (cap > 0 && out == NULL) return OC_ERR_INVALID_ARG;

    const char *line = output;
    while (*line != '\0' && *out_n < cap) {
        const char *eol = strchr(line, '\n');
        size_t linelen = eol ? (size_t)(eol - line) : strlen(line);
        size_t t = 0;
        while (t < linelen && (line[t] == ' ' || line[t] == '\t' ||
                                line[t] == '\r'))
            t++;
        if (t == linelen) {
            line = eol ? eol + 1 : line + linelen;
            continue;
        }

        const char *fields[4];
        size_t flen[4];
        size_t nfields = 0;
        const char *p = line;
        const char *end = line + linelen;
        while (nfields < 4) {
            const char *comma = p;
            while (comma < end && *comma != ',') comma++;
            fields[nfields] = p;
            flen[nfields] = (size_t)(comma - p);
            nfields++;
            if (comma >= end) break;
            p = comma + 1;
        }
        if (nfields < 3) {
            line = eol ? eol + 1 : line + linelen;
            continue;
        }

        char idxbuf[32], namebuf[128], membuf[32], migbuf[32];
        trim_copy(idxbuf, sizeof(idxbuf), fields[0], flen[0]);
        trim_copy(namebuf, sizeof(namebuf), fields[1], flen[1]);
        trim_copy(membuf, sizeof(membuf), fields[2], flen[2]);
        if (nfields >= 4)
            trim_copy(migbuf, sizeof(migbuf), fields[3], flen[3]);
        else
            migbuf[0] = '\0';

        uint32_t index = 0, mem = 0;
        if (!parse_csv_u32(idxbuf, &index) || !parse_csv_u32(membuf, &mem)) {
            line = eol ? eol + 1 : line + linelen;
            continue;
        }

        OcGpuDevice *d = &out[*out_n];
        memset(d, 0, sizeof(*d));
        d->index = index;
        memcpy(d->name, namebuf, strlen(namebuf) + 1);
        d->memory_total_mib = mem;
        d->mig_enabled = (strcasecmp(migbuf, "enabled") == 0);
        d->family = oc_gpu_family_from_nvidia_name(d->name);
        (*out_n)++;

        line = eol ? eol + 1 : line + linelen;
    }
    return OC_OK;
}

OcError oc_gpu_detect(OcGpuDevice *out, size_t cap, size_t *out_n)
{
    if (out_n == NULL) return OC_ERR_INVALID_ARG;
    *out_n = 0;
    if (cap > 0 && out == NULL) return OC_ERR_INVALID_ARG;

    FILE *fp = popen("nvidia-smi --query-gpu=index,name,memory.total,mig.mode.current "
                     "--format=csv,noheader,nounits 2>/dev/null",
                     "r");
    if (fp == NULL) return OC_OK;

    char buf[8192];
    size_t n = 0;
    while (n + 1 < sizeof(buf)) {
        if (fgets(buf + n, (int)(sizeof(buf) - n), fp) == NULL) break;
        n += strlen(buf + n);
    }
    int status = pclose(fp);
    if (status != 0 || n == 0) return OC_OK;
    if (n == sizeof(buf) - 1 && buf[n - 1] != '\n') {
        char *last_nl = strrchr(buf, '\n');
        if (last_nl != NULL)
            last_nl[1] = '\0';
        else
            buf[0] = '\0';
    }
    return oc_gpu_parse_nvidia_smi_csv(buf, out, cap, out_n);
}
