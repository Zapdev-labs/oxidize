/* apply.c — Apply a tuning plan to the runtime. */
#include "oxidize/apply.h"

#include <stdio.h>
#include <string.h>

/* Process-global runtime overrides (models the CLI/server Args struct). */
static OcApplyResult g_runtime = {0, 0, {0}, false, false, false, false, false};

OcError oc_apply_result_init(OcApplyResult *result)
{
    if (!result) return OC_ERR_INVALID_ARG;
    memset(result, 0, sizeof(*result));
    return OC_OK;
}

OcError oc_apply_plan(const OcPlan *plan, OcApplyResult *result)
{
    if (!plan || !result) return OC_ERR_INVALID_ARG;

    memset(result, 0, sizeof(*result));

    result->n_threads = plan->n_threads;
    result->n_batch   = plan->n_batch;
    result->flash_attn = plan->use_flash_attn;
    result->oxk        = plan->use_oxk;
    result->mlock      = plan->use_mlock;
    result->mmap       = plan->use_mmap;

    /* Translate NUMA strategy enum to string. */
    const char *s = oc_numa_strategy_name(plan->numa);
    strncpy(result->numa_strategy, s, sizeof(result->numa_strategy) - 1);
    result->numa_strategy[sizeof(result->numa_strategy) - 1] = '\0';

    result->applied = true;

    /* Also update the process-global runtime state. */
    g_runtime = *result;

    return OC_OK;
}

OcError oc_apply_set_threads(uint32_t n)
{
    if (n == 0) return OC_ERR_INVALID_ARG;
    g_runtime.n_threads = n;
    g_runtime.applied    = true;
    return OC_OK;
}

OcError oc_apply_set_batch_size(uint32_t n)
{
    if (n == 0) return OC_ERR_INVALID_ARG;
    g_runtime.n_batch = n;
    g_runtime.applied  = true;
    return OC_OK;
}

OcError oc_apply_set_numa(const char *strategy)
{
    if (!strategy) return OC_ERR_INVALID_ARG;
    /* Validate against known strategy names. */
    if (strcmp(strategy, "none") != 0 &&
        strcmp(strategy, "single") != 0 &&
        strcmp(strategy, "interleave") != 0) {
        return OC_ERR_INVALID_ARG;
    }
    strncpy(g_runtime.numa_strategy, strategy, sizeof(g_runtime.numa_strategy) - 1);
    g_runtime.numa_strategy[sizeof(g_runtime.numa_strategy) - 1] = '\0';
    g_runtime.applied = true;
    return OC_OK;
}

OcError oc_apply_enable_flash_attn(bool enable)
{
    g_runtime.flash_attn = enable;
    g_runtime.applied     = true;
    return OC_OK;
}

OcError oc_apply_enable_oxk(bool enable)
{
    g_runtime.oxk     = enable;
    g_runtime.applied = true;
    return OC_OK;
}

OcError oc_apply_enable_mlock(bool enable)
{
    g_runtime.mlock   = enable;
    g_runtime.applied = true;
    return OC_OK;
}

size_t oc_apply_print(const OcApplyResult *result, char *out, size_t out_size)
{
    if (!result) return 0;

    /* Format into a local buffer first, then handle truncation. */
    char buf[512];
    int n = snprintf(buf, sizeof(buf),
        "ApplyResult:\n"
        "  threads: %u\n"
        "  batch:   %u\n"
        "  numa:    %s\n"
        "  flash:   %s\n"
        "  oxk:     %s\n"
        "  mlock:   %s\n"
        "  mmap:    %s\n"
        "  applied: %s\n",
        result->n_threads,
        result->n_batch,
        result->numa_strategy,
        result->flash_attn ? "yes" : "no",
        result->oxk ? "yes" : "no",
        result->mlock ? "yes" : "no",
        result->mmap ? "yes" : "no",
        result->applied ? "yes" : "no");

    if (n < 0) return 0;
    size_t len = (size_t)n;

    if (!out || out_size == 0) return len;

    size_t copy = (len < out_size - 1) ? len : (out_size - 1);
    memcpy(out, buf, copy);
    out[copy] = '\0';
    return copy;
}
