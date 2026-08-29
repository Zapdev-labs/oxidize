#ifndef OXIDIZE_AUTOTUNE_RULES_H
#define OXIDIZE_AUTOTUNE_RULES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_PLAN_MAX_RATIONALE 1024

typedef enum {
    OC_NUMA_NONE = 0,
    OC_NUMA_SINGLE = 1,
    OC_NUMA_INTERLEAVE = 2,
} OcNumaStrategy;

typedef enum {
    OC_THREAD_AUTO = 0,
    OC_THREAD_PHYSICAL = 1,
    OC_THREAD_LOGICAL = 2,
} OcThreadStrategy;

typedef struct {
    uint32_t n_cores;
    uint32_t n_logical;
    uint32_t n_sockets;
    bool has_avx2;
    bool has_avx512;
    bool has_avx512_vnni;
    bool has_fma;
    bool has_neon;
    bool is_skylake_sp;
    uint32_t l1_cache_kb;
    uint32_t l2_cache_kb;
    uint32_t l3_cache_mb;
    uint64_t ram_gb;
} OcHwCaps;

typedef struct {
    uint64_t file_size_bytes;
    uint32_t n_layers;
    uint32_t n_params;
    uint32_t quant_type;
    uint32_t hidden_dim;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t n_ctx;
} OcModelFp;

typedef struct {
    uint32_t n_threads;
    uint32_t n_batch;
    uint32_t n_ctx;
    OcNumaStrategy numa;
    OcThreadStrategy thread_strategy;
    bool use_flash_attn;
    bool use_oxk;
    bool use_mlock;
    bool use_mmap;
    uint32_t chunk_size;
    char rationale[OC_PLAN_MAX_RATIONALE];
} OcPlan;

OcError oc_hw_caps_init(OcHwCaps *caps);
OcError oc_model_fp_init(OcModelFp *fp);
OcError oc_plan_compute(const OcHwCaps *hw, const OcModelFp *model, OcPlan *plan);
OcError oc_plan_dump(const OcPlan *plan, const OcHwCaps *hw, const OcModelFp *model,
                    char *out, size_t out_size);
const char *oc_numa_strategy_name(OcNumaStrategy s);
const char *oc_thread_strategy_name(OcThreadStrategy s);
const char *oc_plan_quant_type_name(uint32_t q);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_AUTOTUNE_RULES_H */
