#define _POSIX_C_SOURCE 200809L

#include "oxidize/cross_validation.h"

#include <math.h>
#include <stddef.h>


static const char *const k_suite_names[] = {
    "VulkanDflashCpu",
    "FullPipeline",
    "SmokeCheck",
};


OcError oc_cross_validation_compare(OcValidationSuite suite,
                                    const float *expected,
                                    const float *actual,
                                    size_t len,
                                    float tolerance,
                                    OcValidationResult *out)
{
    if (out == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    if ((size_t)suite >= (size_t)OC_VAL_SUITE__COUNT) {
        return OC_ERR_INVALID_ARG;
    }
    if (len > 0u && (expected == NULL || actual == NULL)) {
        return OC_ERR_INVALID_ARG;
    }

    float max_abs_diff = 0.0f;
    for (size_t i = 0u; i < len; ++i) {
        float diff = expected[i] - actual[i];
        if (diff < 0.0f) {
            diff = -diff;
        }
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }

    out->suite       = suite;
    out->max_abs_diff = max_abs_diff;
    out->tolerance    = tolerance;
    return OC_OK;
}

bool oc_cross_validation_passed(const OcValidationResult *r)
{
    if (r == NULL) {
        return false;
    }
    return r->max_abs_diff <= r->tolerance;
}

const char *oc_cross_validation_suite_name(OcValidationSuite s)
{
    if ((size_t)s >= (size_t)OC_VAL_SUITE__COUNT) {
        return "unknown";
    }
    return k_suite_names[(size_t)s];
}

size_t oc_cross_validation_n_suites(void)
{
    return (size_t)OC_VAL_SUITE__COUNT;
}

OcValidationSuite oc_cross_validation_suite_by_index(size_t idx)
{
    if (idx >= (size_t)OC_VAL_SUITE__COUNT) {
        return OC_VAL_SUITE_VULKAN_DFLASH_CPU;
    }
    return (OcValidationSuite)idx;
}
