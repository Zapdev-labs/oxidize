/*
 * autotune.h — CPU detection, GGUF fingerprint, and tuning plan.
 *
 * Port of oxidize-core/src/autotune/ (detect.rs + fingerprint.rs + rules.rs)
 * to C11. The autotune subsystem is a PURE PLANNER: every decision is
 * captured in the returned OcTuningPlan with a human-readable rationale
 * (mirrors the Rust `plan()` pure-function contract from AGENTS.md). The
 * plan is then applied to a forward context by the autotune-plan-apply
 * feature (thread/NUMA/KV-cache policy).
 *
 * Scope of the `autotune-detect-fingerprint` feature:
 *   - oc_autotune_detect_cpu(): /proc + getauxval-based CPU + NUMA + RAM.
 *   - oc_autotune_fingerprint_gguf(): model size, layer/embd/head counts,
 *     dominant quant type, total parameter estimate.
 *   - oc_autotune_plan(): pure function combining (cpu, model) → plan.
 *
 * Linux-only for the CPU detection path (uses /proc, /sys, getauxval);
 * other platforms get a conservative scalar-only plan.
 */
#ifndef OXIDIZE_AUTOTUNE_H
#define OXIDIZE_AUTOTUNE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/gpu_cluster.h"
#include "oxidize/llama.h"
#include "oxidize/simd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── CPU info ────────────────────────────────────────────────────────── */
typedef struct OcCpuInfo {
    uint32_t physical_cores;     /* packages * cores-per-package         */
    uint32_t logical_cores;      /* SMT threads                          */
    uint32_t numa_nodes;         /* 1 if UMA                             */
    uint64_t total_ram_bytes;    /* MemTotal from /proc/meminfo          */
    uint64_t available_ram_bytes;/* MemAvailable                         */
    OcSimdCaps simd;             /* copy of oc_simd_caps()               */
    char     model_name[128];    /* /proc/cpuinfo model name             */
    bool     is_dual_socket;     /* numa_nodes >= 2                      */
    /* GPU inventory (best-effort nvidia-smi). has_gpu=false and
     * gpu_family=OC_GPU_FAMILY__COUNT when no device is present. */
    bool         has_gpu;
    OcGpuFamily  gpu_family;
    uint32_t     gpu_count;
    uint64_t     gpu_vram_bytes;
} OcCpuInfo;

/* ─── Model fingerprint ───────────────────────────────────────────────── */
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

/* ─── Tuning plan ────────────────────────────────────────────────────────
 *
 * Mirrors Rust TuningPlan: every field has a rationale string. The plan is
 * APPLIED by autotune-plan-apply (next feature); this struct is the pure
 * output of oc_autotune_plan().
 */
typedef enum {
    OC_NUMA_NONE      = 0,
    OC_NUMA_SINGLE    = 1,   /* bind to one socket (interleave off)       */
    OC_NUMA_INTERLEAVE = 2, /* spread across sockets                     */
} OcNumaPolicy;

typedef enum {
    OC_PIPELINE_SEQUENTIAL = 0,
    OC_PIPELINE_CONTINUOUS = 1,
    OC_PIPELINE_PAGED      = 2,
} OcPipelineMode;

typedef enum {
    OC_WEIGHT_NATIVE = 0,
    OC_WEIGHT_FP8    = 1,
    OC_WEIGHT_W8A8   = 2,
    OC_WEIGHT_W4A16  = 3,
} OcWeightPlan;

typedef enum {
    OC_ATTN_DEFAULT          = 0,
    OC_ATTN_FLASH           = 1,
    OC_ATTN_FLASH_ATTENTION3 = 2,
} OcAttentionKernel;

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
    /* Hopper / GPU throughput fields. Append-only: new fields go at the
     * end so other ports can also append. 0 / false / Native / Sequential
     * are the CPU defaults. chunked_prefill_tokens=0 and
     * max_decode_batch=0 mean "leave the runtime default". */
    OcPipelineMode     pipeline;
    OcWeightPlan       weight_plan;
    OcAttentionKernel  attention_kernel;
    bool               cuda_graphs;
    bool               persistent_decode_kernels;
    uint32_t           n_gpu_layers;           /* 0 = CPU only; UINT32_MAX = all */
    uint32_t           chunked_prefill_tokens; /* 0 = leave default            */
    uint32_t           max_decode_batch;       /* 0 = leave default            */
    OcKvCacheType      kv_cache;
    bool               kv_turboquant;
    float              expected_prompt_tps;
    float              expected_decode_tps;
    const char        *rationale_gpu;         /* hopper-specific, or NULL    */
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

/* Pure function: combine (cpu, model) → plan. Every decision is captured in
 * the rationale strings. Mirrors Rust autotune/rules.rs::plan().
 *
 * Heuristics (from AGENTS.md learned facts, dual-socket Cascade Lake):
 *   - threads: min(logical_cores, 16) for dense <=192 GB; 48 for >192 GB.
 *   - NUMA: SINGLE for <=192 GB; INTERLEAVE for >192 GB on dual-socket.
 *   - hugepages/mlock: only if available_ram >= model_size * 2.
 *   - SIMD: use the detected level; SIMD dequant on for AVX2+. */
OcTuningPlan oc_autotune_plan(const OcCpuInfo *cpu,
                              const OcModelFingerprint *model);

/* ─── Apply (autotune-plan-apply feature) ────────────────────────────────
 *
 * Apply a plan to a loaded mmap'd GGUF: applies MADV_HUGEPAGE (if
 * plan->use_hugepages) and mlock (if plan->mlock_weights) to every shard.
 * Thread and NUMA policy are caller-side (set via pthread_setaffinity /
 * numa_set_bind) and are exposed via the plan fields rather than applied
 * here, because they must be set per-thread by the caller's worker pool.
 *
 * Returns OC_OK if at least one of (hugepages, mlock) succeeded, OC_OK if
 * neither was requested, or OC_ERR_IO if a requested policy failed on all
 * shards. Best-effort: partial failures are logged but do not abort. */
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

/* Hopper throughput tier. Pure. Fires only when gpu_family==H100 and
 * plan->n_gpu_layers > 0. Leaves the plan unchanged otherwise. */
void oc_autotune_tier9_hopper(const OcCpuInfo *cpu,
                              const OcModelFingerprint *model,
                              OcTuningPlan *plan);

/* Copy hopper batch/prefill knobs onto a paged scheduler config.
 * max_decode_batch=0 / chunked_prefill_tokens=0 leave the existing values. */
struct OcSchedConfig;
OcError oc_autotune_apply_sched(const OcTuningPlan *plan,
                                 struct OcSchedConfig *cfg);

/* Drop GPU runtime knobs (chunked prefill, decode batch, turboquant KV,
 * CUDA-graph flags) so a Hopper plan printed from inventory is not applied
 * to a CPU session. Leaves n_gpu_layers / weight_plan / TPS for dump. */
void oc_autotune_clear_gpu_runtime(OcTuningPlan *plan);

const char *oc_autotune_pipeline_name(OcPipelineMode p);
const char *oc_autotune_weight_plan_name(OcWeightPlan p);
const char *oc_autotune_attention_kernel_name(OcAttentionKernel k);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_AUTOTUNE_H */
