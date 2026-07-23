/* test_cross_validation.c — cross-validation suite tests. */
#include <criterion/criterion.h>
#include <math.h>
#include <string.h>

#include "oxidize/cross_validation.h"

/* Unique suite name to avoid collision with test_validation.c's "validation". */

/* ─── compare: identical arrays ──────────────────────────────────────── */

Test(cross_validation, compare_identical_arrays)
{
    float buf[] = {1.0f, 2.0f, 3.0f, 4.0f};
    OcValidationResult r;
    OcError e = oc_cross_validation_compare(OC_VAL_SUITE_FULL_PIPELINE,
                                           buf, buf, 4u, 0.001f, &r);
    cr_assert_eq(e, OC_OK);
    cr_assert_float_eq(r.max_abs_diff, 0.0f, 1e-9f);
    cr_assert(oc_cross_validation_passed(&r));
    cr_assert_eq(r.suite, OC_VAL_SUITE_FULL_PIPELINE);
}

Test(cross_validation, compare_identical_zero_tolerance)
{
    float buf[] = {-1.5f, 0.0f, 1.5f};
    OcValidationResult r;
    cr_assert_eq(oc_cross_validation_compare(OC_VAL_SUITE_SMOKE_CHECK,
                                            buf, buf, 3u, 0.0f, &r), OC_OK);
    cr_assert_float_eq(r.max_abs_diff, 0.0f, 1e-9f);
    cr_assert(oc_cross_validation_passed(&r));
}

/* ─── compare: different arrays ─────────────────────────────────────── */

Test(cross_validation, compare_different_arrays)
{
    float expected[] = {1.0f, 2.0f, 3.0f};
    float actual[]   = {1.1f, 2.0f, 3.5f};
    OcValidationResult r;
    cr_assert_eq(oc_cross_validation_compare(OC_VAL_SUITE_FULL_PIPELINE,
                                            expected, actual, 3u, 0.4f, &r),
                OC_OK);
    cr_assert_float_eq(r.max_abs_diff, 0.5f, 1e-6f);
    cr_assert(!oc_cross_validation_passed(&r));
}

Test(cross_validation, compare_within_tolerance)
{
    float expected[] = {1.0f, 2.0f, 3.0f};
    float actual[]   = {1.01f, 2.0f, 2.99f};
    OcValidationResult r;
    cr_assert_eq(oc_cross_validation_compare(OC_VAL_SUITE_VULKAN_DFLASH_CPU,
                                            expected, actual, 3u, 0.1f, &r),
                OC_OK);
    cr_assert_float_eq(r.max_abs_diff, 0.01f, 1e-6f);
    cr_assert(oc_cross_validation_passed(&r));
}

Test(cross_validation, compare_negative_values)
{
    float expected[] = {-1.0f, -2.0f, 3.0f};
    float actual[]   = {-1.5f, -1.5f, 3.5f};
    OcValidationResult r;
    cr_assert_eq(oc_cross_validation_compare(OC_VAL_SUITE_FULL_PIPELINE,
                                            expected, actual, 3u, 0.6f, &r),
                OC_OK);
    cr_assert_float_eq(r.max_abs_diff, 0.5f, 1e-6f);
    cr_assert(oc_cross_validation_passed(&r));
}

Test(cross_validation, compare_inf_difference)
{
    float expected[] = {1.0f};
    float actual[]   = {INFINITY};
    OcValidationResult r;
    cr_assert_eq(oc_cross_validation_compare(OC_VAL_SUITE_FULL_PIPELINE,
                                            expected, actual, 1u, 1e9f, &r),
                OC_OK);
    cr_assert(isinf(r.max_abs_diff));
    cr_assert(!oc_cross_validation_passed(&r));
}

/* ─── null / edge cases ─────────────────────────────────────────────── */

Test(cross_validation, compare_null_out)
{
    float buf[] = {1.0f};
    cr_assert_eq(oc_cross_validation_compare(OC_VAL_SUITE_SMOKE_CHECK,
                                            buf, buf, 1u, 0.0f, NULL),
                OC_ERR_INVALID_ARG);
}

Test(cross_validation, compare_null_buffers_with_len)
{
    OcValidationResult r;
    cr_assert_eq(oc_cross_validation_compare(OC_VAL_SUITE_SMOKE_CHECK,
                                            NULL, NULL, 3u, 0.0f, &r),
                OC_ERR_INVALID_ARG);
}

Test(cross_validation, compare_null_buffers_zero_len_ok)
{
    OcValidationResult r;
    cr_assert_eq(oc_cross_validation_compare(OC_VAL_SUITE_SMOKE_CHECK,
                                            NULL, NULL, 0u, 0.0f, &r),
                OC_OK);
    cr_assert_float_eq(r.max_abs_diff, 0.0f, 1e-9f);
    cr_assert(oc_cross_validation_passed(&r));
}

Test(cross_validation, compare_invalid_suite)
{
    float buf[] = {1.0f};
    OcValidationResult r;
    cr_assert_eq(oc_cross_validation_compare(OC_VAL_SUITE__COUNT,
                                            buf, buf, 1u, 0.0f, &r),
                OC_ERR_INVALID_ARG);
}

Test(cross_validation, compare_single_element)
{
    float expected[] = {5.0f};
    float actual[]   = {5.5f};
    OcValidationResult r;
    cr_assert_eq(oc_cross_validation_compare(OC_VAL_SUITE_SMOKE_CHECK,
                                            expected, actual, 1u, 0.4f, &r),
                OC_OK);
    cr_assert_float_eq(r.max_abs_diff, 0.5f, 1e-6f);
    cr_assert(!oc_cross_validation_passed(&r));
    cr_assert_float_eq(r.tolerance, 0.4f, 1e-9f);
}

/* ─── passed() ──────────────────────────────────────────────────────── */

Test(cross_validation, passed_null_returns_false)
{
    cr_assert(!oc_cross_validation_passed(NULL));
}

Test(cross_validation, passed_exact_tolerance_boundary)
{
    OcValidationResult r = {OC_VAL_SUITE_FULL_PIPELINE, 0.5f, 0.5f};
    cr_assert(oc_cross_validation_passed(&r));
}

Test(cross_validation, passed_just_above_tolerance)
{
    OcValidationResult r = {OC_VAL_SUITE_FULL_PIPELINE, 0.5f + 1e-6f, 0.5f};
    cr_assert(!oc_cross_validation_passed(&r));
}

/* ─── suite name strings ────────────────────────────────────────────── */

Test(cross_validation, suite_name_vulkan_dflash_cpu)
{
    cr_assert_str_eq(oc_cross_validation_suite_name(OC_VAL_SUITE_VULKAN_DFLASH_CPU),
                     "VulkanDflashCpu");
}

Test(cross_validation, suite_name_full_pipeline)
{
    cr_assert_str_eq(oc_cross_validation_suite_name(OC_VAL_SUITE_FULL_PIPELINE),
                     "FullPipeline");
}

Test(cross_validation, suite_name_smoke_check)
{
    cr_assert_str_eq(oc_cross_validation_suite_name(OC_VAL_SUITE_SMOKE_CHECK),
                     "SmokeCheck");
}

Test(cross_validation, suite_name_invalid)
{
    cr_assert_str_eq(oc_cross_validation_suite_name(OC_VAL_SUITE__COUNT),
                     "unknown");
}

/* ─── suite enumeration ─────────────────────────────────────────────── */

Test(cross_validation, n_suites_is_three)
{
    cr_assert_eq(oc_cross_validation_n_suites(), 3u);
}

Test(cross_validation, suite_by_index_in_range)
{
    cr_assert_eq(oc_cross_validation_suite_by_index(0u),
                 OC_VAL_SUITE_VULKAN_DFLASH_CPU);
    cr_assert_eq(oc_cross_validation_suite_by_index(1u),
                 OC_VAL_SUITE_FULL_PIPELINE);
    cr_assert_eq(oc_cross_validation_suite_by_index(2u),
                 OC_VAL_SUITE_SMOKE_CHECK);
}

Test(cross_validation, suite_by_index_out_of_range)
{
    cr_assert_eq(oc_cross_validation_suite_by_index(99u),
                 OC_VAL_SUITE_VULKAN_DFLASH_CPU);
}
