/* test_autotune_rules.c — Autotune rules tests. */
#include "framework.h"
#include "oxidize/autotune_rules.h"
#include <string.h>

Test(ar, hw_caps_init)
{
    OcHwCaps caps;
    cr_assert_eq(oc_hw_caps_init(&caps), OC_OK);
    cr_assert_eq(caps.n_cores, 8);
    cr_assert_eq(caps.n_logical, 16);
    cr_assert_eq(caps.n_sockets, 1);
}

OC_TEST_NULL_SAFE(ar, hw_caps_init_null,
        cr_assert_neq(oc_hw_caps_init(NULL), OC_OK);)

Test(ar, model_fp_init)
{
    OcModelFp fp;
    cr_assert_eq(oc_model_fp_init(&fp), OC_OK);
    cr_assert_eq(fp.n_layers, 32);
    cr_assert_eq(fp.quant_type, 2);
    cr_assert_eq(fp.hidden_dim, 4096);
}

OC_TEST_NULL_SAFE(ar, model_fp_init_null,
        cr_assert_neq(oc_model_fp_init(NULL), OC_OK);)

Test(ar, plan_basic)
{
    OcHwCaps hw; oc_hw_caps_init(&hw);
    OcModelFp fp; oc_model_fp_init(&fp);
    fp.file_size_bytes = 4ULL * 1024 * 1024 * 1024;
    OcPlan plan;
    cr_assert_eq(oc_plan_compute(&hw, &fp, &plan), OC_OK);
    cr_assert_gt(plan.n_threads, 0);
    cr_assert_gt(plan.n_batch, 0);
    cr_assert(plan.use_mmap);
    cr_assert(strlen(plan.rationale) > 0);
}

Test(ar, plan_avx512)
{
    OcHwCaps hw; oc_hw_caps_init(&hw);
    hw.has_avx512 = true;
    OcModelFp fp; oc_model_fp_init(&fp);
    fp.quant_type = 3;
    OcPlan plan;
    oc_plan_compute(&hw, &fp, &plan);
    cr_assert_eq(plan.n_batch, 512);
    cr_assert(plan.use_flash_attn);
    cr_assert(plan.use_oxk);
}

Test(ar, plan_avx2)
{
    OcHwCaps hw; oc_hw_caps_init(&hw);
    hw.has_avx2 = true;
    hw.has_avx512 = false;
    OcModelFp fp; oc_model_fp_init(&fp);
    fp.quant_type = 2;
    OcPlan plan;
    oc_plan_compute(&hw, &fp, &plan);
    cr_assert_eq(plan.n_batch, 256);
    cr_assert(plan.use_oxk);
}

Test(ar, plan_no_simd)
{
    OcHwCaps hw; oc_hw_caps_init(&hw);
    hw.has_avx2 = false;
    hw.has_avx512 = false;
    OcModelFp fp; oc_model_fp_init(&fp);
    fp.quant_type = 2;
    OcPlan plan;
    oc_plan_compute(&hw, &fp, &plan);
    cr_assert_eq(plan.n_batch, 128);
    cr_assert(!plan.use_oxk);
}

Test(ar, plan_numa_interleave)
{
    OcHwCaps hw; oc_hw_caps_init(&hw);
    hw.n_sockets = 2;
    hw.n_logical = 96;
    hw.ram_gb = 376;
    OcModelFp fp; oc_model_fp_init(&fp);
    fp.file_size_bytes = 200ULL * 1024 * 1024 * 1024;
    OcPlan plan;
    oc_plan_compute(&hw, &fp, &plan);
    cr_assert_eq(plan.numa, OC_NUMA_INTERLEAVE);
}

Test(ar, plan_numa_single)
{
    OcHwCaps hw; oc_hw_caps_init(&hw);
    hw.n_sockets = 2;
    hw.n_cores = 48;
    OcModelFp fp; oc_model_fp_init(&fp);
    fp.file_size_bytes = 4ULL * 1024 * 1024 * 1024;
    OcPlan plan;
    oc_plan_compute(&hw, &fp, &plan);
    cr_assert_eq(plan.numa, OC_NUMA_SINGLE);
    cr_assert_eq(plan.n_threads, 24);
}

Test(ar, plan_mlock_small)
{
    OcHwCaps hw; oc_hw_caps_init(&hw);
    hw.ram_gb = 64;
    OcModelFp fp; oc_model_fp_init(&fp);
    fp.file_size_bytes = 4ULL * 1024 * 1024 * 1024;
    OcPlan plan;
    oc_plan_compute(&hw, &fp, &plan);
    cr_assert(plan.use_mlock);
}

Test(ar, plan_no_mlock_large)
{
    OcHwCaps hw; oc_hw_caps_init(&hw);
    hw.ram_gb = 16;
    OcModelFp fp; oc_model_fp_init(&fp);
    fp.file_size_bytes = 100ULL * 1024 * 1024 * 1024;
    OcPlan plan;
    oc_plan_compute(&hw, &fp, &plan);
    cr_assert(!plan.use_mlock);
}

OC_TEST_NULL_SAFE(ar, plan_null,
        cr_assert_neq(oc_plan_compute(NULL, NULL, NULL), OC_OK);)

Test(ar, plan_dump)
{
    OcHwCaps hw; oc_hw_caps_init(&hw);
    OcModelFp fp; oc_model_fp_init(&fp);
    OcPlan plan;
    oc_plan_compute(&hw, &fp, &plan);
    char out[4096];
    cr_assert_eq(oc_plan_dump(&plan, &hw, &fp, out, sizeof(out)), OC_OK);
    cr_assert(strstr(out, "Tuning Plan") != NULL);
    cr_assert(strstr(out, "Threads:") != NULL);
}

OC_TEST_NULL_SAFE(ar, plan_dump_null,
        cr_assert_neq(oc_plan_dump(NULL, NULL, NULL, NULL, 0), OC_OK);)

Test(ar, numa_name)
{
    cr_assert_str_eq(oc_numa_strategy_name(OC_NUMA_NONE), "none");
    cr_assert_str_eq(oc_numa_strategy_name(OC_NUMA_SINGLE), "single");
    cr_assert_str_eq(oc_numa_strategy_name(OC_NUMA_INTERLEAVE), "interleave");
}

Test(ar, thread_name)
{
    cr_assert_str_eq(oc_thread_strategy_name(OC_THREAD_AUTO), "auto");
    cr_assert_str_eq(oc_thread_strategy_name(OC_THREAD_PHYSICAL), "physical");
    cr_assert_str_eq(oc_thread_strategy_name(OC_THREAD_LOGICAL), "logical");
}

Test(ar, quant_name)
{
    cr_assert_str_eq(oc_plan_quant_type_name(0), "f16");
    cr_assert_str_eq(oc_plan_quant_type_name(1), "q8_0");
    cr_assert_str_eq(oc_plan_quant_type_name(2), "q4_0");
    cr_assert_str_eq(oc_plan_quant_type_name(3), "q4_k");
    cr_assert_str_eq(oc_plan_quant_type_name(4), "q5_k");
}

Test(ar, plan_default_ctx)
{
    OcHwCaps hw; oc_hw_caps_init(&hw);
    OcModelFp fp; oc_model_fp_init(&fp);
    fp.n_ctx = 0;
    OcPlan plan;
    oc_plan_compute(&hw, &fp, &plan);
    cr_assert_eq(plan.n_ctx, 4096);
}

Test(ar, plan_chunk_size)
{
    OcHwCaps hw; oc_hw_caps_init(&hw);
    hw.l2_cache_kb = 512;
    OcModelFp fp; oc_model_fp_init(&fp);
    OcPlan plan;
    oc_plan_compute(&hw, &fp, &plan);
    cr_assert_eq(plan.chunk_size, 128);
}
