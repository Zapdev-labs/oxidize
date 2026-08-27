/* test_apply.c — OcApplyResult tests. */
#include <criterion/criterion.h>
#include "oxidize/apply.h"
#include "oxidize/autotune_rules.h"

#include <string.h>

Test(apply, result_init)
{
    OcApplyResult r;
    cr_assert_eq(oc_apply_result_init(&r), OC_OK, "");
    cr_assert_eq(r.n_threads, 0, "");
    cr_assert_eq(r.n_batch, 0, "");
    cr_assert_not(r.flash_attn, "");
    cr_assert_not(r.oxk, "");
    cr_assert_not(r.mlock, "");
    cr_assert_not(r.mmap, "");
    cr_assert_not(r.applied, "");
}

Test(apply, result_init_null)
{
    cr_assert_eq(oc_apply_result_init(NULL), OC_ERR_INVALID_ARG, "");
}

Test(apply, apply_plan_basic)
{
    OcHwCaps hw;
    oc_hw_caps_init(&hw);
    OcModelFp fp;
    oc_model_fp_init(&fp);
    fp.file_size_bytes = 4ULL * 1024 * 1024 * 1024;
    OcPlan plan;
    cr_assert_eq(oc_plan_compute(&hw, &fp, &plan), OC_OK, "");

    OcApplyResult r;
    cr_assert_eq(oc_apply_plan(&plan, &r), OC_OK, "");
    cr_assert(r.applied, "");
    cr_assert_gt(r.n_threads, 0, "");
    cr_assert_gt(r.n_batch, 0, "");
    cr_assert(r.mmap, "plan should enable mmap");
    cr_assert(strlen(r.numa_strategy) > 0, "");
}

Test(apply, apply_plan_null)
{
    cr_assert_eq(oc_apply_plan(NULL, NULL), OC_ERR_INVALID_ARG, "");
    OcApplyResult r;
    cr_assert_eq(oc_apply_plan(NULL, &r), OC_ERR_INVALID_ARG, "");
}

Test(apply, apply_plan_numa_strategy)
{
    OcHwCaps hw;
    oc_hw_caps_init(&hw);
    hw.n_sockets = 2;
    hw.n_logical = 96;
    hw.ram_gb    = 376;
    OcModelFp fp;
    oc_model_fp_init(&fp);
    fp.file_size_bytes = 200ULL * 1024 * 1024 * 1024;
    OcPlan plan;
    oc_plan_compute(&hw, &fp, &plan);

    OcApplyResult r;
    oc_apply_plan(&plan, &r);
    cr_assert_str_eq(r.numa_strategy, "interleave", "");
}

Test(apply, set_threads)
{
    cr_assert_eq(oc_apply_set_threads(16), OC_OK, "");
    cr_assert_eq(oc_apply_set_threads(0), OC_ERR_INVALID_ARG, "");
}

Test(apply, set_batch_size)
{
    cr_assert_eq(oc_apply_set_batch_size(512), OC_OK, "");
    cr_assert_eq(oc_apply_set_batch_size(0), OC_ERR_INVALID_ARG, "");
}

Test(apply, set_numa)
{
    cr_assert_eq(oc_apply_set_numa("single"), OC_OK, "");
    cr_assert_eq(oc_apply_set_numa("interleave"), OC_OK, "");
    cr_assert_eq(oc_apply_set_numa("none"), OC_OK, "");
    cr_assert_eq(oc_apply_set_numa("bogus"), OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_apply_set_numa(NULL), OC_ERR_INVALID_ARG, "");
}

Test(apply, enable_flash_attn)
{
    cr_assert_eq(oc_apply_enable_flash_attn(true), OC_OK, "");
    cr_assert_eq(oc_apply_enable_flash_attn(false), OC_OK, "");
}

Test(apply, enable_oxk)
{
    cr_assert_eq(oc_apply_enable_oxk(true), OC_OK, "");
    cr_assert_eq(oc_apply_enable_oxk(false), OC_OK, "");
}

Test(apply, enable_mlock)
{
    cr_assert_eq(oc_apply_enable_mlock(true), OC_OK, "");
    cr_assert_eq(oc_apply_enable_mlock(false), OC_OK, "");
}

Test(apply, print)
{
    OcHwCaps hw;
    oc_hw_caps_init(&hw);
    OcModelFp fp;
    oc_model_fp_init(&fp);
    OcPlan plan;
    oc_plan_compute(&hw, &fp, &plan);

    OcApplyResult r;
    oc_apply_plan(&plan, &r);

    char buf[1024];
    size_t n = oc_apply_print(&r, buf, sizeof(buf));
    cr_assert_gt(n, 0, "");
    cr_assert(strstr(buf, "ApplyResult") != NULL, "");
    cr_assert(strstr(buf, "threads:") != NULL, "");
    cr_assert(strstr(buf, "applied:") != NULL, "");
}

Test(apply, print_null_and_zero_size)
{
    OcApplyResult r;
    oc_apply_result_init(&r);
    /* NULL out, zero size: should return the would-be length, not crash. */
    size_t n = oc_apply_print(&r, NULL, 0);
    cr_assert_gt(n, 0, "should return would-be length");
    /* NULL result: returns 0. */
    cr_assert_eq(oc_apply_print(NULL, NULL, 0), 0, "");
}
