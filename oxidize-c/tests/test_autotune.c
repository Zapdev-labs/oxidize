/*
 * test_autotune.c — autotune detect + fingerprint + plan tests.
 *
 * VAL-AUTOTUNE-001..006 cover:
 *   1. CPU detection returns sensible values on this host.
 *   2. Plan is pure (same inputs → same outputs).
 *   3. Plan picks threads, NUMA, hugepages per the learned heuristics.
 *   4. Parser fixtures (no weights) fingerprint gracefully (zeros, UNKNOWN
 *      arch, no dominant qtype) — must not crash.
 *   5. SIMD level in plan matches detected caps.
 *   6. NUMA name function covers all enum values.
 */
#include <criterion/criterion.h>

#include "oxidize/autotune.h"
#include "oxidize/gguf.h"
#include "oxidize/gpu_cluster.h"
#include "oxidize/openai.h"
#include "oxidize/scheduler.h"

#include <string.h>

/* ─── CPU detection ────────────────────────────────────────────────────── */

Test(autotune, detect_cpu_returns_sensible_values)
{
    OcCpuInfo cpu;
    OcError e = oc_autotune_detect_cpu(&cpu);
    cr_assert_eq(e, OC_OK, "detect_cpu should succeed");
    cr_assert(cpu.logical_cores > 0, "must have >0 logical cores");
    cr_assert(cpu.numa_nodes >= 1, "must have >=1 NUMA node");
    /* On this dev host logical_cores == 32, RAM > 1 GB. Be lenient. */
    cr_assert(cpu.total_ram_bytes > (1ULL << 30), "RAM > 1 GB");
    cr_assert(cpu.simd.name != NULL && cpu.simd.name[0] != '\0',
              "simd name must be set");
}

Test(autotune, detect_cpu_null_arg_rejected)
{
    cr_assert_eq(oc_autotune_detect_cpu(NULL), OC_ERR_INVALID_ARG);
}

/* ─── Plan purity ──────────────────────────────────────────────────────── */

Test(autotune, plan_is_pure_same_inputs_same_output)
{
    OcCpuInfo cpu;
    oc_autotune_detect_cpu(&cpu);
    OcModelFingerprint model;
    memset(&model, 0, sizeof(model));
    model.file_bytes = 8ULL << 30;   /* 8 GB */
    model.n_layer = 32;

    OcTuningPlan p1 = oc_autotune_plan(&cpu, &model);
    OcTuningPlan p2 = oc_autotune_plan(&cpu, &model);
    cr_assert_eq(p1.threads, p2.threads, "threads must be deterministic");
    cr_assert_eq(p1.numa, p2.numa, "numa must be deterministic");
    cr_assert_eq(p1.use_hugepages, p2.use_hugepages, "hugepages deterministic");
    cr_assert_eq(p1.mlock_weights, p2.mlock_weights, "mlock deterministic");
    cr_assert_eq(p1.simd_level, p2.simd_level, "simd deterministic");
}

Test(autotune, plan_threads_match_learned_heuristics)
{
    OcCpuInfo cpu;
    oc_autotune_detect_cpu(&cpu);

    /* One thread per physical core. The plan used to cap at 16, which was
     * measured against the old dequant-to-f32 forward pass; the fused integer
     * kernels are compute-bound and 16 leaves most of the machine idle
     * (2.94 vs 5.43 tok/s on a 48-core dual socket). SMT threads are excluded
     * deliberately — 96 threads measured worse than 16. */
    OcModelFingerprint small;
    memset(&small, 0, sizeof(small));
    small.file_bytes = 8ULL << 30;
    OcTuningPlan p = oc_autotune_plan(&cpu, &small);
    const uint32_t expected_small = cpu.is_dual_socket
        ? cpu.physical_cores / cpu.numa_nodes : cpu.physical_cores;
    cr_assert_eq(p.threads, expected_small,
                 "expected physical cores in selected NUMA policy (%u), got %u",
                 expected_small, p.threads);
    cr_assert_leq(p.threads, cpu.logical_cores,
                  "never more threads than logical cores");
    if (cpu.is_dual_socket) {
        cr_assert_eq(p.numa, OC_NUMA_SINGLE, "small model → single-socket");
    }

    /* Large model (>192 GB): interleave across sockets, same thread rule.
     * 48 was previously hardcoded here because that is the physical core
     * count of the box it was tuned on; deriving it keeps the plan right on
     * other machines. */
    OcModelFingerprint big;
    memset(&big, 0, sizeof(big));
    big.file_bytes = 256ULL << 30;   /* 256 GB */
    p = oc_autotune_plan(&cpu, &big);
    cr_assert_eq(p.threads, cpu.physical_cores,
                 "large model: one thread per physical core");
    if (cpu.is_dual_socket) {
        cr_assert_eq(p.numa, OC_NUMA_INTERLEAVE, "large model → interleave");
    }
}

Test(autotune, plan_hugepages_only_when_2x_headroom)
{
    OcCpuInfo cpu;
    oc_autotune_detect_cpu(&cpu);

    /* Model larger than available RAM → no hugepages, no mlock. */
    OcModelFingerprint too_big;
    memset(&too_big, 0, sizeof(too_big));
    too_big.file_bytes = cpu.available_ram_bytes + 1;
    OcTuningPlan p = oc_autotune_plan(&cpu, &too_big);
    cr_assert_not(p.use_hugepages, "no hugepages when model > available RAM");
    cr_assert_not(p.mlock_weights, "no mlock when no headroom");
}

Test(autotune, plan_simd_matches_caps)
{
    OcCpuInfo cpu;
    oc_autotune_detect_cpu(&cpu);
    OcModelFingerprint model;
    memset(&model, 0, sizeof(model));
    model.file_bytes = 1ULL << 30;
    OcTuningPlan p = oc_autotune_plan(&cpu, &model);
    cr_assert_eq(p.simd_level, cpu.simd.level, "plan simd == detected");
    if (cpu.simd.level >= OC_SIMD_AVX2) {
        cr_assert(p.use_simd_dequant, "AVX2+ host should enable simd dequant");
    } else {
        cr_assert_not(p.use_simd_dequant, "scalar host should not enable simd dequant");
    }
}

/* ─── NUMA name ────────────────────────────────────────────────────────── */

Test(autotune, numa_name_covers_enum)
{
    cr_assert_str_eq(oc_autotune_numa_name(OC_NUMA_NONE), "none");
    cr_assert_str_eq(oc_autotune_numa_name(OC_NUMA_SINGLE), "single");
    cr_assert_str_eq(oc_autotune_numa_name(OC_NUMA_INTERLEAVE), "interleave");
}

/* ─── Fingerprint gracefully handles parser fixtures ────────────────────
 * The parser fixtures have a valid GGUF header but no weights — fingerprint
 * must return OC_OK with zeroed counts, not crash. */
Test(autotune, fingerprint_handles_parser_fixture)
{
    OcGgufMmappedFile m;
    OcError e = oc_gguf_map_open("../oxidize-core/tests/fixtures/valid-v3.gguf", &m);
    if (e != OC_OK) {
        cr_skip_test("fixture not available at this CWD");
    }
    OcModelFingerprint fp;
    e = oc_autotune_fingerprint_gguf(&m, &fp);
    cr_assert_eq(e, OC_OK, "fingerprint should succeed on valid GGUF");
    cr_assert(fp.file_bytes > 0, "file_bytes should be set");
    oc_gguf_map_free(&m);
}

Test(autotune, fingerprint_null_args_rejected)
{
    OcModelFingerprint fp;
    cr_assert_eq(oc_autotune_fingerprint_gguf(NULL, &fp), OC_ERR_INVALID_ARG);
    OcGgufMmappedFile m = {0};
    cr_assert_eq(oc_autotune_fingerprint_gguf(&m, NULL), OC_ERR_INVALID_ARG);
}

Test(autotune, plan_dump_does_not_crash)
{
    OcCpuInfo cpu;
    oc_autotune_detect_cpu(&cpu);
    OcModelFingerprint model;
    memset(&model, 0, sizeof(model));
    model.file_bytes = 4ULL << 30;
    model.n_layer = 32;
    model.arch = OC_ARCH_LLAMA;
    model.dominant_qtype = OC_QUANT_Q4_K_M;
    model.dominant_qtype_fraction = 0.9;
    OcTuningPlan p = oc_autotune_plan(&cpu, &model);
    /* Dump goes to stderr; just assert no crash. */
    oc_autotune_plan_dump(&p, &cpu, &model);
    cr_assert(true, "plan_dump ran without crashing");
}

/* ─── Apply (autotune-plan-apply) ──────────────────────────────────────── */

Test(autotune, apply_null_args_rejected)
{
    OcGgufMmappedFile m;
    cr_assert_eq(oc_autotune_apply(NULL, &m), OC_ERR_INVALID_ARG);
    OcTuningPlan p;
    memset(&p, 0, sizeof(p));
    cr_assert_eq(oc_autotune_apply(&p, NULL), OC_ERR_INVALID_ARG);
}

Test(autotune, apply_noop_plan_does_not_crash)
{
    /* A plan that requests no hugepages and no mlock should be a no-op. */
    OcGgufMmappedFile m;
    if (oc_gguf_map_open("../oxidize-core/tests/fixtures/valid-v3.gguf", &m) != OC_OK) {
        cr_skip_test("fixture not available at this CWD");
    }
    OcTuningPlan p;
    memset(&p, 0, sizeof(p));
    p.use_hugepages = false;
    p.mlock_weights = false;
    OcError e = oc_autotune_apply(&p, &m);
    cr_assert_eq(e, OC_OK, "noop plan must succeed");
    oc_gguf_map_free(&m);
}

Test(autotune, apply_hugepages_best_effort_on_small_file)
{
    /* Even if the plan requests hugepages, applying to a tiny fixture must
     * not crash; the underlying advise is best-effort. */
    OcGgufMmappedFile m;
    if (oc_gguf_map_open("../oxidize-core/tests/fixtures/valid-v3.gguf", &m) != OC_OK) {
        cr_skip_test("fixture not available");
    }
    OcCpuInfo cpu;
    oc_autotune_detect_cpu(&cpu);
    OcTuningPlan p;
    memset(&p, 0, sizeof(p));
    p.use_hugepages = true;
    p.mlock_weights = false;   /* don't mlock the fixture */
    OcError e = oc_autotune_apply(&p, &m);
    cr_assert_eq(e, OC_OK, "hugepages apply is best-effort, must return OK");
    oc_gguf_map_free(&m);
}

Test(autotune, bind_to_numa_node_does_not_crash)
{
    /* Binding to node 0 should always succeed (or no-op on non-Linux). */
    OcError e = oc_autotune_bind_to_numa_node(0);
    cr_assert_eq(e, OC_OK, "bind_to_numa_node(0) must succeed or no-op");
    /* Binding to a non-existent node must also not crash (best-effort). */
    e = oc_autotune_bind_to_numa_node(999);
    cr_assert_eq(e, OC_OK, "bind to non-existent node is a no-op");
}

/* ─── Hopper throughput tier ─────────────────────────────────────────── */

static void hopper_cpu(OcCpuInfo *cpu, OcGpuFamily family, bool has_gpu)
{
    memset(cpu, 0, sizeof(*cpu));
    cpu->physical_cores = 32;
    cpu->logical_cores = 64;
    cpu->numa_nodes = 2;
    cpu->is_dual_socket = true;
    cpu->total_ram_bytes = 256ULL << 30;
    cpu->available_ram_bytes = 200ULL << 30;
    cpu->has_gpu = has_gpu;
    cpu->gpu_family = has_gpu ? family : OC_GPU_FAMILY__COUNT;
    cpu->gpu_count = has_gpu ? 1u : 0;
    cpu->gpu_vram_bytes = has_gpu ? (80ULL << 30) : 0;
}

static void hopper_fp_31b(OcModelFingerprint *m, uint32_t n_ctx)
{
    memset(m, 0, sizeof(*m));
    m->file_bytes = 20ULL << 30;
    m->n_layer = 48;
    m->n_embd = 4096;
    m->n_head = 32;
    m->n_ctx = n_ctx;
    m->dominant_qtype = OC_QUANT_Q4_K_M;
    m->dominant_qtype_fraction = 0.95;
    m->arch = OC_ARCH_LLAMA;
}

Test(autotune, hopper_h100_q4km_31b)
{
    OcCpuInfo cpu;
    OcModelFingerprint m;
    hopper_cpu(&cpu, OC_GPU_FAMILY_H100, true);
    hopper_fp_31b(&m, 4096);
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    cr_assert(p.n_gpu_layers > 0);
    cr_assert_eq(p.weight_plan, OC_WEIGHT_W4A16);
    cr_assert_eq(p.pipeline, OC_PIPELINE_PAGED);
    cr_assert_eq(p.attention_kernel, OC_ATTN_FLASH_ATTENTION3);
    cr_assert(p.cuda_graphs);
    cr_assert(p.persistent_decode_kernels);
    cr_assert_eq(p.chunked_prefill_tokens, 512u);
    cr_assert_eq(p.max_decode_batch, 16u);
    cr_assert(p.kv_turboquant);
    cr_assert_eq(p.kv_cache, OC_KV_Q8);
    cr_assert_float_eq(p.expected_decode_tps, 1150.0f, 0.1f);
    cr_assert_not_null(p.rationale_gpu);
}

Test(autotune, hopper_h100_long_ctx_prefill_1024)
{
    OcCpuInfo cpu;
    OcModelFingerprint m;
    hopper_cpu(&cpu, OC_GPU_FAMILY_H100, true);
    hopper_fp_31b(&m, 8192);
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    cr_assert_eq(p.chunked_prefill_tokens, 1024u);
}

Test(autotune, hopper_does_not_fire_on_a100)
{
    OcCpuInfo cpu;
    OcModelFingerprint m;
    hopper_cpu(&cpu, OC_GPU_FAMILY_A100, true);
    hopper_fp_31b(&m, 4096);
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    cr_assert_neq(p.attention_kernel, OC_ATTN_FLASH_ATTENTION3);
    cr_assert_not(p.cuda_graphs);
    cr_assert(p.expected_decode_tps != 1150.0f);
    cr_assert_null(p.rationale_gpu);
}

Test(autotune, hopper_does_not_fire_on_b200)
{
    OcCpuInfo cpu;
    OcModelFingerprint m;
    hopper_cpu(&cpu, OC_GPU_FAMILY_B200, true);
    hopper_fp_31b(&m, 4096);
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    cr_assert_neq(p.attention_kernel, OC_ATTN_FLASH_ATTENTION3);
    cr_assert_not(p.cuda_graphs);
    cr_assert_float_eq(p.expected_decode_tps, 200.0f, 0.1f);
}

Test(autotune, hopper_q8_is_w8a8)
{
    OcCpuInfo cpu;
    OcModelFingerprint m;
    hopper_cpu(&cpu, OC_GPU_FAMILY_H100, true);
    hopper_fp_31b(&m, 4096);
    m.dominant_qtype = OC_QUANT_Q8_0;
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    cr_assert_eq(p.weight_plan, OC_WEIGHT_W8A8);
    cr_assert(p.kv_turboquant);
    cr_assert_float_eq(p.expected_decode_tps, 650.0f, 0.1f);
}

Test(autotune, hopper_f16_is_fp8)
{
    OcCpuInfo cpu;
    OcModelFingerprint m;
    hopper_cpu(&cpu, OC_GPU_FAMILY_H100, true);
    hopper_fp_31b(&m, 4096);
    m.dominant_qtype = OC_QUANT_F16;
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    cr_assert_eq(p.weight_plan, OC_WEIGHT_FP8);
    cr_assert(p.kv_turboquant);
    cr_assert_float_eq(p.expected_decode_tps, 650.0f, 0.1f);
}

Test(autotune, hopper_q6k_is_native)
{
    OcCpuInfo cpu;
    OcModelFingerprint m;
    hopper_cpu(&cpu, OC_GPU_FAMILY_H100, true);
    hopper_fp_31b(&m, 4096);
    m.dominant_qtype = OC_QUANT_Q6_K;
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    cr_assert_eq(p.weight_plan, OC_WEIGHT_NATIVE);
    cr_assert_not(p.kv_turboquant);
    cr_assert_float_eq(p.expected_decode_tps, 300.0f, 0.1f);
    cr_assert_eq(p.attention_kernel, OC_ATTN_FLASH_ATTENTION3);
}

Test(autotune, hopper_tiny_vram_skips_offload)
{
    OcCpuInfo cpu;
    OcModelFingerprint m;
    hopper_cpu(&cpu, OC_GPU_FAMILY_H100, true);
    hopper_fp_31b(&m, 4096);
    cpu.gpu_vram_bytes = 1ULL << 30;
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    cr_assert_eq(p.n_gpu_layers, 0);
    cr_assert_neq(p.attention_kernel, OC_ATTN_FLASH_ATTENTION3);
    cr_assert_not(p.cuda_graphs);
}

Test(autotune, hopper_does_not_fire_without_gpu)
{
    OcCpuInfo cpu;
    OcModelFingerprint m;
    hopper_cpu(&cpu, OC_GPU_FAMILY_H100, false);
    hopper_fp_31b(&m, 4096);
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    cr_assert_eq(p.n_gpu_layers, 0);
    cr_assert_neq(p.attention_kernel, OC_ATTN_FLASH_ATTENTION3);
    cr_assert_not(p.cuda_graphs);
    cr_assert(p.expected_decode_tps != 1150.0f);
}

Test(autotune, apply_sched_copies_hopper_batch)
{
    OcCpuInfo cpu;
    OcModelFingerprint m;
    hopper_cpu(&cpu, OC_GPU_FAMILY_H100, true);
    hopper_fp_31b(&m, 8192);
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    OcSchedConfig cfg;
    cr_assert_eq(oc_sched_config_init(&cfg), OC_OK);
    cr_assert_eq(oc_autotune_apply_sched(&p, &cfg), OC_OK);
    cr_assert_eq(cfg.max_batch_size, 16u);
    cr_assert_eq(cfg.prefill_chunk_size, 1024u);
}

Test(autotune, openai_apply_tuning_plan)
{
    OcOpenaiState st;
    memset(&st, 0, sizeof(st));
    OcCpuInfo cpu;
    OcModelFingerprint m;
    hopper_cpu(&cpu, OC_GPU_FAMILY_H100, true);
    hopper_fp_31b(&m, 4096);
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    cr_assert_eq(oc_openai_apply_tuning_plan(&st, &p, 0, NULL), OC_OK);
    cr_assert(st.has_plan);
    cr_assert(st.kv_set);
    cr_assert_eq(st.kv_type, OC_KV_Q8);
    cr_assert_eq(st.sched.max_batch_size, 16u);
    cr_assert_eq(st.sched.prefill_chunk_size, 512u);
    cr_assert_eq(oc_openai_apply_tuning_plan(&st, &p, 2048, NULL), OC_OK);
    cr_assert_eq(st.sched.prefill_chunk_size, 2048u);
}

Test(autotune, openai_apply_explicit_kv_wins_over_hopper)
{
    OcOpenaiState st;
    memset(&st, 0, sizeof(st));
    OcCpuInfo cpu;
    OcModelFingerprint m;
    hopper_cpu(&cpu, OC_GPU_FAMILY_H100, true);
    hopper_fp_31b(&m, 4096);
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    cr_assert(p.kv_turboquant);
    cr_assert_eq(oc_openai_apply_tuning_plan(&st, &p, 0, "f32"), OC_OK);
    cr_assert(st.kv_set);
    cr_assert_eq(st.kv_type, OC_KV_F32);
}

Test(autotune, openai_apply_non_hopper_does_not_force_f32)
{
    OcOpenaiState st;
    memset(&st, 0, sizeof(st));
    OcCpuInfo cpu;
    OcModelFingerprint m;
    hopper_cpu(&cpu, OC_GPU_FAMILY_A100, true);
    hopper_fp_31b(&m, 8192);
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    cr_assert_not(p.kv_turboquant);
    cr_assert_eq(oc_openai_apply_tuning_plan(&st, &p, 0, NULL), OC_OK);
    cr_assert_not(st.kv_set);
}

Test(autotune, openai_apply_prefill_without_plan)
{
    OcOpenaiState st;
    memset(&st, 0, sizeof(st));
    cr_assert_eq(oc_openai_apply_tuning_plan(&st, NULL, 256, "q8"), OC_OK);
    cr_assert_not(st.has_plan);
    cr_assert_eq(st.sched.prefill_chunk_size, 256u);
    cr_assert(st.kv_set);
    cr_assert_eq(st.kv_type, OC_KV_Q8);
}

Test(autotune, hopper_partial_vram_skips_offload)
{
    OcCpuInfo cpu;
    OcModelFingerprint m;
    hopper_cpu(&cpu, OC_GPU_FAMILY_H100, true);
    hopper_fp_31b(&m, 4096);
    m.file_bytes = 100ULL << 30;
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    cr_assert_eq(p.n_gpu_layers, 0);
    cr_assert_neq(p.attention_kernel, OC_ATTN_FLASH_ATTENTION3);
    cr_assert_not(p.cuda_graphs);
    cr_assert_not(p.kv_turboquant);
}

Test(autotune, clear_gpu_runtime_drops_hopper_knobs)
{
    OcCpuInfo cpu;
    OcModelFingerprint m;
    hopper_cpu(&cpu, OC_GPU_FAMILY_H100, true);
    hopper_fp_31b(&m, 8192);
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    cr_assert(p.kv_turboquant);
    cr_assert_eq(p.chunked_prefill_tokens, 1024u);
    oc_autotune_clear_gpu_runtime(&p);
    cr_assert_eq(p.chunked_prefill_tokens, 0);
    cr_assert_eq(p.max_decode_batch, 0);
    cr_assert_not(p.kv_turboquant);
    cr_assert_not(p.cuda_graphs);
    cr_assert_eq(p.n_gpu_layers, 48);
    OcOpenaiState st;
    memset(&st, 0, sizeof(st));
    cr_assert_eq(oc_openai_apply_tuning_plan(&st, &p, 0, NULL), OC_OK);
    cr_assert_not(st.kv_set);
    cr_assert_eq(st.sched.prefill_chunk_size, 0);
}

Test(autotune, plan_and_apply_without_gpu_does_not_crash)
{
    OcCpuInfo cpu;
    oc_autotune_detect_cpu(&cpu);
    cpu.has_gpu = false;
    cpu.gpu_family = OC_GPU_FAMILY__COUNT;
    cpu.gpu_count = 0;
    cpu.gpu_vram_bytes = 0;
    OcModelFingerprint m;
    memset(&m, 0, sizeof(m));
    m.file_bytes = 1ULL << 30;
    m.n_layer = 32;
    OcTuningPlan p = oc_autotune_plan(&cpu, &m);
    OcGgufMmappedFile mf;
    if (oc_gguf_map_open("../oxidize-core/tests/fixtures/valid-v3.gguf", &mf) != OC_OK) {
        cr_skip_test("fixture not available at this CWD");
    }
    cr_assert_eq(oc_autotune_apply(&p, &mf), OC_OK);
    oc_gguf_map_free(&mf);
}

Test(autotune, detect_cpu_gpu_fields_without_device)
{
    OcCpuInfo cpu;
    cr_assert_eq(oc_autotune_detect_cpu(&cpu), OC_OK);
    if (!cpu.has_gpu) {
        cr_assert_eq(cpu.gpu_family, OC_GPU_FAMILY__COUNT);
        cr_assert_eq(cpu.gpu_count, 0);
        cr_assert_eq(cpu.gpu_vram_bytes, 0);
    }
}
