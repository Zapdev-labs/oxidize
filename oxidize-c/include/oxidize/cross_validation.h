/*
 * cross_validation.h — Output cross-validation suites and result reporting.
 *
 * Port of oxidize-core/src/validation/cross_validation.rs. Provides
 * validation suites (VulkanDflashCpu, FullPipeline, SmokeCheck) that
 * compare expected vs actual model output tensors element-wise and report
 * the maximum absolute difference against a caller-supplied tolerance.
 *
 * The module is intentionally side-effect free: callers pass buffers and
 * receive an OcValidationResult. No global state is mutated.
 */
#ifndef OXIDIZE_CROSS_VALIDATION_H
#define OXIDIZE_CROSS_VALIDATION_H

#include <stdbool.h>
#include <stddef.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Validation suites ───────────────────────────────────────────────── */

typedef enum {
    OC_VAL_SUITE_VULKAN_DFLASH_CPU = 0,
    OC_VAL_SUITE_FULL_PIPELINE,
    OC_VAL_SUITE_SMOKE_CHECK,
    /* sentinel for bounds checking; not a valid suite */
    OC_VAL_SUITE__COUNT,
} OcValidationSuite;

/* ─── Result ──────────────────────────────────────────────────────────── */

typedef struct {
    OcValidationSuite suite;
    float             max_abs_diff;
    float             tolerance;
} OcValidationResult;

/* ─── API ─────────────────────────────────────────────────────────────── */

/*
 * Compare `expected` and `actual` buffers element-wise for `len` elements.
 * Computes the maximum absolute difference across all pairs and stores the
 * result, suite, and tolerance in `*out`.
 *
 * Returns:
 *   OC_OK             — comparison performed (regardless of pass/fail)
 *   OC_ERR_INVALID_ARG — `actual`/`expected`/`out` NULL, OR `len>0` with NULL
 *                       buffer, OR `suite` out of range
 *
 * For `len==0` with non-NULL buffers, returns OC_OK with max_abs_diff=0.0f.
 */
OcError oc_cross_validation_compare(OcValidationSuite suite,
                                    const float *expected,
                                    const float *actual,
                                    size_t len,
                                    float tolerance,
                                    OcValidationResult *out);

/* True iff `r` is non-NULL and r->max_abs_diff <= r->tolerance. */
bool oc_cross_validation_passed(const OcValidationResult *r);

/* Human-readable, NUL-terminated suite name. Returns "unknown" for invalid
 * suites. Never returns NULL. */
const char *oc_cross_validation_suite_name(OcValidationSuite s);

/* Number of implemented validation suites (excludes the sentinel). */
size_t oc_cross_validation_n_suites(void);

/* Suite by index. Returns OC_VAL_SUITE_VULKAN_DFLASH_CPU for out-of-range. */
OcValidationSuite oc_cross_validation_suite_by_index(size_t idx);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_CROSS_VALIDATION_H */
