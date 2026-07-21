/*
 * attn_dump.c — per-layer attention weight dump implementation.
 */
#include "oxidize/attn_dump.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void oc_attn_dump_init(OcAttnDumper *d, const char *output_dir)
{
    if (!d) return;
    memset(d, 0, sizeof(*d));
    d->enabled = false;
    /* Check env vars (mirrors Rust OXIDIZE_TRACE_VALS / OXIDIZE_TRACE_FWD). */
    if (getenv("OXIDIZE_TRACE_VALS") || getenv("OXIDIZE_TRACE_FWD")) {
        d->enabled = true;
        if (output_dir) {
            snprintf(d->output_dir, sizeof(d->output_dir), "%s", output_dir);
        } else {
            snprintf(d->output_dir, sizeof(d->output_dir), "/tmp/oxidize_trace");
        }
    }
}

void oc_attn_dump_set_context(OcAttnDumper *d, uint32_t step, uint32_t layer)
{
    if (!d || !d->enabled) return;
    d->step = step;
    d->layer = layer;
}

OcError oc_attn_dump_f32(const OcAttnDumper *d, const char *name,
                         const float *data, size_t count)
{
    if (!d || !d->enabled || !name || !data) return OC_ERR_INVALID_ARG;
    char path[512];
    snprintf(path, sizeof(path), "%s/step%u_layer%u_%s.f32",
             d->output_dir, d->step, d->layer, name);
    FILE *f = fopen(path, "wb");
    if (!f) return OC_ERR_IO;
    fwrite(data, sizeof(float), count, f);
    fclose(f);
    return OC_OK;
}

OcError oc_attn_dump_logits(const OcAttnDumper *d, const float *logits,
                            size_t vocab_size)
{
    if (!d || !d->enabled || !logits) return OC_ERR_INVALID_ARG;
    char path[512];
    snprintf(path, sizeof(path), "%s/logits_step%u.f32",
             d->output_dir, d->step);
    FILE *f = fopen(path, "wb");
    if (!f) return OC_ERR_IO;
    fwrite(logits, sizeof(float), vocab_size, f);
    fclose(f);
    return OC_OK;
}
