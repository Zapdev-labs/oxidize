/*
 * detect.c — Hardware detection implementation.
 */
#include "oxidize/detect.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

OcError oc_detect_cpu(OcDetectInfo *info)
{
    if (!info) return OC_ERR_INVALID_ARG;
    memset(info, 0, sizeof(*info));

#if defined(__x86_64__) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx;

    /* CPUID 0: vendor string. */
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        /* CPUID 1: feature flags. */
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
            info->has_sse42 = (ecx & bit_SSE4_2) != 0;
            info->has_avx   = (ecx & bit_AVX) != 0;
            info->has_fma   = (ecx & bit_FMA) != 0;
        }
    }

    /* CPUID 7: extended features. */
    unsigned int eax7, ebx7, ecx7, edx7;
    if (__get_cpuid_count(7, 0, &eax7, &ebx7, &ecx7, &edx7)) {
        info->has_avx2         = (ebx7 & bit_AVX2) != 0;
        info->has_avx512f      = (ebx7 & bit_AVX512F) != 0;
        info->has_avx512bw     = (ebx7 & bit_AVX512BW) != 0;
        info->has_avx512dq     = (ebx7 & bit_AVX512DQ) != 0;
        info->has_avx512vl     = (ebx7 & bit_AVX512VL) != 0;
        info->has_avx512_vnni  = (ecx7 & bit_AVX512VNNI) != 0;
        info->has_bf16         = (eax7 & (1 << 5)) != 0; /* AVX512_BF16 */
    }

    /* CPU model name from CPUID 0x80000002-0x80000004. */
    char name[49] = {0};
    unsigned int *ni = (unsigned int *)name;
    if (__get_cpuid(0x80000002, &eax, &ebx, &ecx, &edx)) {
        ni[0]=eax; ni[1]=ebx; ni[2]=ecx; ni[3]=edx;
    }
    if (__get_cpuid(0x80000003, &eax, &ebx, &ecx, &edx)) {
        ni[4]=eax; ni[5]=ebx; ni[6]=ecx; ni[7]=edx;
    }
    if (__get_cpuid(0x80000004, &eax, &ebx, &ecx, &edx)) {
        ni[8]=eax; ni[9]=ebx; ni[10]=ecx; ni[11]=edx;
    }
    name[48] = '\0';
    char *p = name;
    while (*p == ' ') p++;
    strncpy(info->model_name, p, sizeof(info->model_name) - 1);

    /* Detect Skylake-SP. */
    info->is_skylake_sp = (strstr(info->model_name, "Skylake") != NULL) ||
                          ((strstr(info->model_name, "Xeon") != NULL) &&
                           (strstr(info->model_name, "Gold") != NULL ||
                            strstr(info->model_name, "Platinum") != NULL));
#elif defined(__aarch64__)
    /* NEON (Advanced SIMD) is mandatory in the AArch64 base architecture, so */
    info->has_neon = true;
    strcpy(info->model_name, "aarch64");
#else
    strcpy(info->model_name, "unknown");
#endif

    return OC_OK;
}

OcError oc_detect_numa(OcDetectInfo *info)
{
    if (!info) return OC_ERR_INVALID_ARG;

    /* Read /sys/devices/system/node/ for NUMA info. */
    info->n_numa_nodes = 1; /* default */
    info->numa_node_cpus[0] = info->n_logical;

#if defined(__linux__)
    /* Count /sys/devices/system/node/nodeN directories. */
    uint32_t count = 0;
    for (int i = 0; i < 8; i++) {
        char path[256];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/node/node%d/cpulist", i);
        FILE *f = fopen(path, "r");
        if (!f) break;
        count++;
        fclose(f);
    }
    if (count > 0) info->n_numa_nodes = count;
#endif

    return OC_OK;
}

OcError oc_detect_all(OcDetectInfo *info)
{
    if (!info) return OC_ERR_INVALID_ARG;
    OcError e = oc_detect_cpu(info);
    if (e != OC_OK) return e;

    /* Count cores via /proc/cpuinfo. */
#if defined(__linux__)
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        uint32_t logical = 0;
        uint32_t cores = 0;
        uint32_t sockets = 0;
        uint32_t last_phys = 0xFFFFFFFF;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "processor", 9) == 0) logical++;
            if (strncmp(line, "cpu cores", 9) == 0) {
                char *colon = strchr(line, ':');
                if (colon) cores = (uint32_t)atoi(colon + 1);
            }
            if (strncmp(line, "physical id", 11) == 0) {
                char *colon = strchr(line, ':');
                if (colon) {
                    uint32_t phys = (uint32_t)atoi(colon + 1);
                    if (phys != last_phys) {
                        sockets++;
                        last_phys = phys;
                    }
                }
            }
        }
        fclose(f);
        if (logical > 0) info->n_logical = logical;
        if (cores > 0 && sockets > 0) info->n_cores = cores * sockets;
        else if (cores > 0) info->n_cores = cores;
        if (sockets > 0) info->n_sockets = sockets;
    }

    /* Read RAM from /proc/meminfo. */
    f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "MemTotal:", 9) == 0) {
                char *colon = strchr(line, ':');
                if (colon) {
                    unsigned long kb = 0;
                    sscanf(colon + 1, "%lu", &kb);
                    info->ram_bytes = (uint64_t)kb * 1024;
                }
                break;
            }
        }
        fclose(f);
    }
#else
    info->n_cores = 4;
    info->n_logical = 8;
    info->n_sockets = 1;
    info->ram_bytes = 16ULL * 1024 * 1024 * 1024;
#endif

    return oc_detect_numa(info);
}

const char *oc_detect_simd_level(const OcDetectInfo *info)
{
    if (!info) return "none";
    if (info->has_avx512_vnni) return "avx512_vnni";
    if (info->has_avx512f) return "avx512";
    if (info->has_avx2) return "avx2";
    if (info->has_avx) return "avx";
    if (info->has_sse42) return "sse4.2";
    if (info->has_neon) return "neon";
    return "scalar";
}

bool oc_detect_supports_vnni(const OcDetectInfo *info)
{
    return info ? info->has_avx512_vnni : false;
}

bool oc_detect_is_server(const OcDetectInfo *info)
{
    if (!info) return false;
    /* Heuristic: >32 logical cores or >1 socket = server. */
    return info->n_logical > 32 || info->n_sockets > 1;
}

uint32_t oc_detect_recommended_threads(const OcDetectInfo *info)
{
    if (!info || info->n_cores == 0) return 4;
    return info->n_cores;
}

void oc_detect_print(const OcDetectInfo *info, char *out, size_t out_size)
{
    if (!info || !out || out_size == 0) return;
    snprintf(out, out_size,
        "=== Hardware Detection ===\n"
        "CPU: %s\n"
        "Cores: %u physical, %u logical, %u sockets\n"
        "RAM: %.1f GB\n"
        "SIMD: %s\n"
        "AVX2: %s, AVX-512: %s, VNNI: %s, FMA: %s, NEON: %s\n"
        "Skylake-SP: %s\n"
        "NUMA nodes: %u\n"
        "Recommended threads: %u\n",
        info->model_name,
        info->n_cores, info->n_logical, info->n_sockets,
        (double)info->ram_bytes / (1024.0 * 1024 * 1024),
        oc_detect_simd_level(info),
        info->has_avx2 ? "yes" : "no",
        info->has_avx512f ? "yes" : "no",
        info->has_avx512_vnni ? "yes" : "no",
        info->has_fma ? "yes" : "no",
        info->has_neon ? "yes" : "no",
        info->is_skylake_sp ? "yes" : "no",
        info->n_numa_nodes,
        oc_detect_recommended_threads(info));
}
