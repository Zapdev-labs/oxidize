/* attn_dump.h — per-layer attention weight dump for debugging/validation. */
#ifndef OXIDIZE_ATTN_DUMP_H
#define OXIDIZE_ATTN_DUMP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcAttnDumper {
    bool enabled;
    char output_dir[256];
    uint32_t step;    /* current forward step                */
    uint32_t layer;   /* current layer                       */
} OcAttnDumper;

/* Initialize the dumper. Checks for OXIDIZE_TRACE_VALS or OXIDIZE_TRACE_FWD
 * env vars. If neither is set, `enabled` is false and all dump calls are
 * no-ops. */
void oc_attn_dump_init(OcAttnDumper *d, const char *output_dir);

/* Dump a float array to `{output_dir}/step{step}_layer{layer}_{name}.f32`. */
OcError oc_attn_dump_f32(const OcAttnDumper *d, const char *name,
                         const float *data, size_t count);

/* Dump logits (special case: always to `logits_step{N}.f32`). */
OcError oc_attn_dump_logits(const OcAttnDumper *d, const float *logits,
                            size_t vocab_size);

/* Set the current step and layer for subsequent dumps. */
void oc_attn_dump_set_context(OcAttnDumper *d, uint32_t step, uint32_t layer);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ATTN_DUMP_H */
