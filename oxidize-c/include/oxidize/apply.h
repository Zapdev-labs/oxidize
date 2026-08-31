/* apply.h — Apply a tuning plan to the runtime. */
#ifndef OXIDIZE_APPLY_H
#define OXIDIZE_APPLY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/autotune_rules.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Result of applying a tuning plan. Each field records the value the runtime
 * should use. `applied` is true once oc_apply_plan() has populated the struct. */
typedef struct {
    uint32_t n_threads;
    uint32_t n_batch;
    char     numa_strategy[16];
    bool     flash_attn;
    bool     oxk;
    bool     mlock;
    bool     mmap;
    bool     applied;
} OcApplyResult;

/* Zero-initialize the result (applied=false, all fields zeroed). */
OcError oc_apply_result_init(OcApplyResult *result);

/* Apply a tuning plan: read fields from `plan` and populate `result`. */
OcError oc_apply_plan(const OcPlan *plan, OcApplyResult *result);

/* Set thread count at runtime. */
OcError oc_apply_set_threads(uint32_t n);

/* Set batch size at runtime. */
OcError oc_apply_set_batch_size(uint32_t n);

/* Set NUMA strategy ("none", "single", "interleave"). */
OcError oc_apply_set_numa(const char *strategy);

/* Toggle flash attention. */
OcError oc_apply_enable_flash_attn(bool enable);

/* Toggle OXK kernels. */
OcError oc_apply_enable_oxk(bool enable);

/* Toggle mlock. */
OcError oc_apply_enable_mlock(bool enable);

/* Format `result` as a human-readable string into `out` (NUL-terminated).
 * Returns the number of bytes written (excluding NUL). If `out` is NULL or
 * `out_size` is 0, returns the length that would have been written. */
size_t oc_apply_print(const OcApplyResult *result, char *out, size_t out_size);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_APPLY_H */
