/* detect.h — Hardware detection and fingerprinting. */
#ifndef OXIDIZE_DETECT_H
#define OXIDIZE_DETECT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_DETECT_MAX_NAME 256

typedef struct {
    char model_name[OC_DETECT_MAX_NAME];
    uint32_t n_cores;
    uint32_t n_logical;
    uint32_t n_sockets;
    uint32_t l1_cache_kb;
    uint32_t l2_cache_kb;
    uint32_t l3_cache_mb;
    uint64_t ram_bytes;
    bool has_avx;
    bool has_avx2;
    bool has_avx512f;
    bool has_avx512bw;
    bool has_avx512dq;
    bool has_avx512vl;
    bool has_avx512_vnni;
    bool has_fma;
    bool has_sse42;
    bool has_neon;
    bool has_bf16;
    bool is_skylake_sp;
    uint32_t n_numa_nodes;
    uint32_t numa_node_cpus[8];
} OcDetectInfo;

OcError oc_detect_cpu(OcDetectInfo *info);
OcError oc_detect_numa(OcDetectInfo *info);
OcError oc_detect_all(OcDetectInfo *info);
const char *oc_detect_simd_level(const OcDetectInfo *info);
bool oc_detect_supports_vnni(const OcDetectInfo *info);
bool oc_detect_is_server(const OcDetectInfo *info);
uint32_t oc_detect_recommended_threads(const OcDetectInfo *info);
void oc_detect_print(const OcDetectInfo *info, char *out, size_t out_size);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_DETECT_H */
