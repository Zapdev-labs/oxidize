/* autotune.h — CPU detection, GGUF fingerprint, and tuning plan. */
#ifndef OXIDIZE_AUTOTUNE_H
#define OXIDIZE_AUTOTUNE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/simd.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcCpuInfo {
    uint32_t physical_cores;     /* packages * cores-per-package         */
    uint32_t logical_cores;      /* SMT threads                          */
    uint32_t numa_nodes;         /* 1 if UMA                             */
    uint64_t total_ram_bytes;    /* MemTotal from /proc/meminfo          */
    uint64_t available_ram_bytes;/* MemAvailable                         */
    OcSimdCaps simd;             /* copy of oc_simd_caps()               */
    char     model_name[128];    /* /proc/cpuinfo model name             */
    bool     is_dual_socket;     /* numa_nodes >= 2                      */
} OcCpuInfo;

typedef struct OcModelFingerprint {
    uint64_t file_bytes;          /* total mmap size                     */
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_ff;
    uint32_t vocab_size;
    uint32_t n_ctx;
    uint64_t param_count;        /* estimated (rows*cols sum)             */
    OcGgufQuantizationType dominant_qtype;  /* most-weight-bytes type    */
    double   dominant_qtype_fraction;       /* 0..1 of weight bytes      */
    OcModelArchitecture arch;
    bool     uses_moe;
} OcModelFingerprint;

typedef enum {
    OC_NUMA_NONE      = 0,
    OC_NUMA_SINGLE    = 1,   /* bind to one socket (interleave off)       */
    OC_NUMA_INTERLEAVE = 2, /* spread across sockets                     */
} OcNumaPolicy;

typedef struct OcTuningPlan {
    uint32_t       threads;          /* 0 = let runtime decide            */
    OcNumaPolicy   numa;             /* NUMA memory policy                */
    OcSimdLevel    simd_level;       /* requested SIMD tier               */
    bool           use_hugepages;     /* MADV_HUGEPAGE if RAM allows       */
    bool           mlock_weights;     /* pin weights in RAM if headroom     */
    bool           use_simd_dequant;  /* route dequant through SIMD path   */
    /* Rationale strings (static, NUL-terminated, do not free). */
    const char    *rationale_threads;
    const char    *rationale_numa;
    const char    *rationale_simd;
    const char    *rationale_memory;
} OcTuningPlan;

/* Detect CPU / NUMA / RAM. Reads /proc and /sys on Linux; fills a
 * conservative scalar-only info on other platforms. Returns OC_OK or
 * OC_ERR_IO (missing /proc). */
OcError oc_autotune_detect_cpu(OcCpuInfo *out);

/* Fingerprint a loaded GGUF: counts tensors, estimates parameters, finds
 * the dominant quant type by weight-byte share. `m` must be a successfully
 * opened mmap'd GGUF. Returns OC_OK or OC_ERR_INVALID_ARG. */
OcError oc_autotune_fingerprint_gguf(const OcGgufMmappedFile *m,
                                     OcModelFingerprint *out);

/* Pure function: combine (cpu, model) → plan. */
OcTuningPlan oc_autotune_plan(const OcCpuInfo *cpu,
                              const OcModelFingerprint *model);

/* ─── Apply (autotune-plan-apply feature) ──────────────────────────────── Apply a plan to a loaded mmap'd GGUF: applies MADV_HUGEPAGE (if plan->use_hugepages) and mlock (if plan->mlock_weights) to every shard. */
OcError oc_autotune_apply(const OcTuningPlan *plan, OcGgufMmappedFile *m);

/* Bind the calling thread to a single NUMA node (Linux only, best-effort).
 * Used by worker pools to honor plan->numa == OC_NUMA_SINGLE. On non-Linux
 * or single-socket hosts, returns OC_OK without doing anything. */
OcError oc_autotune_bind_to_numa_node(uint32_t node);

/* Human-readable name for a NUMA policy. */
const char *oc_autotune_numa_name(OcNumaPolicy p);

/* Pretty-print a plan to stderr (used by `oxidize-c --print-plan`). */
void oc_autotune_plan_dump(const OcTuningPlan *plan,
                           const OcCpuInfo *cpu,
                           const OcModelFingerprint *model);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_AUTOTUNE_H */
