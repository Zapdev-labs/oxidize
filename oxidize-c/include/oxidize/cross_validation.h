/* cross_validation.h — Output cross-validation suites and result reporting. */
#ifndef OXIDIZE_CROSS_VALIDATION_H
#define OXIDIZE_CROSS_VALIDATION_H

#include <stdbool.h>
#include <stddef.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
    OC_VAL_SUITE_VULKAN_DFLASH_CPU = 0,
    OC_VAL_SUITE_FULL_PIPELINE,
    OC_VAL_SUITE_SMOKE_CHECK,
    /* sentinel for bounds checking; not a valid suite */
    OC_VAL_SUITE__COUNT,
} OcValidationSuite;


typedef struct {
    OcValidationSuite suite;
    float             max_abs_diff;
    float             tolerance;
} OcValidationResult;


/* Compare `expected` and `actual` buffers element-wise for `len` elements. */
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
