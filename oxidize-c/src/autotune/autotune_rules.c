/*
 * autotune_rules.c — Hardware-aware tuning rules implementation.
 */
#include "oxidize/autotune_rules.h"

#include <stdio.h>
#include <string.h>

OcError oc_hw_caps_init(OcHwCaps *caps)
{
    if (!caps) return OC_ERR_INVALID_ARG;
    memset(caps, 0, sizeof(*caps));
    caps->n_cores = 8;
    caps->n_logical = 16;
    caps->n_sockets = 1;
    caps->l1_cache_kb = 32;
    caps->l2_cache_kb = 256;
    caps->l3_cache_mb = 16;
    caps->ram_gb = 16;
    return OC_OK;
}

OcError oc_model_fp_init(OcModelFp *fp)
{
    if (!fp) return OC_ERR_INVALID_ARG;
    memset(fp, 0, sizeof(*fp));
    fp->n_layers = 32;
    fp->quant_type = 2;
    fp->hidden_dim = 4096;
    fp->n_heads = 32;
    fp->n_kv_heads = 32;
    fp->n_ctx = 4096;
    return OC_OK;
}

OcError oc_plan_compute(const OcHwCaps *hw, const OcModelFp *model, OcPlan *plan)
{
    if (!hw || !model || !plan) return OC_ERR_INVALID_ARG;
    memset(plan, 0, sizeof(*plan));

    if (hw->n_cores > 0) {
        plan->n_threads = hw->n_cores;
    } else {
        plan->n_threads = hw->n_logical > 0 ? hw->n_logical / 2 : 4;
    }

    if (hw->has_avx512) {
        plan->n_batch = 512;
    } else if (hw->has_avx2) {
        plan->n_batch = 256;
    } else {
        plan->n_batch = 128;
    }

    plan->n_ctx = model->n_ctx > 0 ? model->n_ctx : 4096;

    double model_gb = (double)model->file_size_bytes / (1024.0 * 1024.0 * 1024.0);
    if (hw->n_sockets > 1 && model_gb > 192.0) {
        plan->numa = OC_NUMA_INTERLEAVE;
        plan->n_threads = hw->n_logical > 0 ? hw->n_logical * 3 / 4 : 48;
    } else if (hw->n_sockets > 1) {
        plan->numa = OC_NUMA_SINGLE;
        if (hw->n_cores >= hw->n_sockets)
            plan->n_threads = hw->n_cores / hw->n_sockets;
    } else {
        plan->numa = OC_NUMA_NONE;
    }

    plan->thread_strategy = OC_THREAD_PHYSICAL;
    plan->use_flash_attn = hw->has_avx512 || plan->n_ctx > 2048;
    plan->use_oxk = (model->quant_type >= 1 && model->quant_type <= 4) &&
                    (hw->has_avx2 || hw->has_avx512);
    plan->use_mlock = model_gb < 64.0 && hw->ram_gb > (uint64_t)model_gb * 2;
    plan->use_mmap = true;
    plan->chunk_size = hw->l2_cache_kb > 0 ? hw->l2_cache_kb / 4 : 64;

    snprintf(plan->rationale, sizeof(plan->rationale),
        "threads=%u (cores=%u, logical=%u, sockets=%u), "
        "batch=%u (avx512=%d, avx2=%d), "
        "numa=%s, flash=%d, oxk=%d, mlock=%d, "
        "model_gb=%.1f, quant=%s, ctx=%u",
        plan->n_threads, hw->n_cores, hw->n_logical, hw->n_sockets,
        plan->n_batch, (int)hw->has_avx512, (int)hw->has_avx2,
        oc_numa_strategy_name(plan->numa),
        (int)plan->use_flash_attn, (int)plan->use_oxk, (int)plan->use_mlock,
        model_gb, oc_plan_quant_type_name(model->quant_type), plan->n_ctx);

    return OC_OK;
}

OcError oc_plan_dump(const OcPlan *plan, const OcHwCaps *hw, const OcModelFp *model,
                    char *out, size_t out_size)
{
    if (!plan || !out || out_size == 0) return OC_ERR_INVALID_ARG;
    snprintf(out, out_size,
        "=== Tuning Plan ===\n"
        "Threads: %u (%s)\n"
        "Batch: %u\n"
        "Context: %u\n"
        "NUMA: %s\n"
        "Flash Attention: %s\n"
        "OXK Kernels: %s\n"
        "mlock: %s\n"
        "mmap: %s\n"
        "Chunk: %u\n"
        "\n"
        "Hardware: cores=%u, logical=%u, sockets=%u, "
        "avx2=%d, avx512=%d, vnni=%d, ram=%lluGB\n"
        "Model: layers=%u, quant=%s, hidden=%u, ctx=%u\n"
        "\n"
        "Rationale: %s\n",
        plan->n_threads, oc_thread_strategy_name(plan->thread_strategy),
        plan->n_batch, plan->n_ctx, oc_numa_strategy_name(plan->numa),
        plan->use_flash_attn ? "yes" : "no",
        plan->use_oxk ? "yes" : "no",
        plan->use_mlock ? "yes" : "no",
        plan->use_mmap ? "yes" : "no",
        plan->chunk_size,
        hw ? hw->n_cores : 0, hw ? hw->n_logical : 0, hw ? hw->n_sockets : 0,
        hw ? (int)hw->has_avx2 : 0, hw ? (int)hw->has_avx512 : 0,
        hw ? (int)hw->has_avx512_vnni : 0, hw ? (unsigned long long)hw->ram_gb : 0,
        model ? model->n_layers : 0, model ? oc_plan_quant_type_name(model->quant_type) : "n/a",
        model ? model->hidden_dim : 0, model ? model->n_ctx : 0,
        plan->rationale);
    return OC_OK;
}

const char *oc_numa_strategy_name(OcNumaStrategy s)
{
    switch (s) {
    case OC_NUMA_NONE:       return "none";
    case OC_NUMA_SINGLE:     return "single";
    case OC_NUMA_INTERLEAVE: return "interleave";
    default: return "unknown";
    }
}

const char *oc_thread_strategy_name(OcThreadStrategy s)
{
    switch (s) {
    case OC_THREAD_AUTO:     return "auto";
    case OC_THREAD_PHYSICAL: return "physical";
    case OC_THREAD_LOGICAL:  return "logical";
    default: return "unknown";
    }
}

const char *oc_plan_quant_type_name(uint32_t q)
{
    switch (q) {
    case 0: return "f16";
    case 1: return "q8_0";
    case 2: return "q4_0";
    case 3: return "q4_k";
    case 4: return "q5_k";
    default: return "unknown";
    }
}
