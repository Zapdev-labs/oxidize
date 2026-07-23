/*
 * frame_sampler.c — Video frame sampling implementation.
 */
#include "oxidize/frame_sampler.h"

#include <stdlib.h>
#include <string.h>

OcError oc_fs_config_init(OcFsConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->strategy = OC_FS_STRATEGY_UNIFORM;
    cfg->n_frames = 8;
    cfg->fps = 2;
    cfg->seed = 42;
    return OC_OK;
}

OcError oc_fs_sample_uniform(uint32_t n_total, uint32_t n_sample, OcFsResult *out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    if (n_total == 0 || n_sample == 0) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    uint32_t actual = n_sample < n_total ? n_sample : n_total;
    out->indices = malloc(actual * sizeof(uint32_t));
    out->timestamps_ms = malloc(actual * sizeof(uint64_t));
    if (!out->indices || !out->timestamps_ms) {
        free(out->indices); free(out->timestamps_ms);
        return OC_ERR_OOM;
    }

    if (actual == 1) {
        out->indices[0] = n_total / 2;
        out->timestamps_ms[0] = 0;
        out->n_indices = 1;
        return OC_OK;
    }

    for (uint32_t i = 0; i < actual; i++) {
        out->indices[i] = (uint32_t)((uint64_t)i * n_total / actual);
        out->timestamps_ms[i] = (uint64_t)out->indices[i] * 1000 / (n_total > 0 ? n_total : 1);
    }
    out->n_indices = actual;
    return OC_OK;
}

OcError oc_fs_sample_first_n(uint32_t n_total, uint32_t n_sample, OcFsResult *out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    if (n_total == 0 || n_sample == 0) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    uint32_t actual = n_sample < n_total ? n_sample : n_total;
    out->indices = malloc(actual * sizeof(uint32_t));
    out->timestamps_ms = malloc(actual * sizeof(uint64_t));
    if (!out->indices || !out->timestamps_ms) {
        free(out->indices); free(out->timestamps_ms);
        return OC_ERR_OOM;
    }

    for (uint32_t i = 0; i < actual; i++) {
        out->indices[i] = i;
        out->timestamps_ms[i] = (uint64_t)i * 1000;
    }
    out->n_indices = actual;
    return OC_OK;
}

OcError oc_fs_sample_last_n(uint32_t n_total, uint32_t n_sample, OcFsResult *out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    if (n_total == 0 || n_sample == 0) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    uint32_t actual = n_sample < n_total ? n_sample : n_total;
    uint32_t start = n_total - actual;
    out->indices = malloc(actual * sizeof(uint32_t));
    out->timestamps_ms = malloc(actual * sizeof(uint64_t));
    if (!out->indices || !out->timestamps_ms) {
        free(out->indices); free(out->timestamps_ms);
        return OC_ERR_OOM;
    }

    for (uint32_t i = 0; i < actual; i++) {
        out->indices[i] = start + i;
        out->timestamps_ms[i] = (uint64_t)(start + i) * 1000;
    }
    out->n_indices = actual;
    return OC_OK;
}

/* Simple LCG for reproducible random. */
static uint32_t lcg_next(uint32_t *state)
{
    *state = *state * 1103515245u + 12345u;
    return *state;
}

OcError oc_fs_sample_random(uint32_t n_total, uint32_t n_sample, uint32_t seed,
                           OcFsResult *out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    if (n_total == 0 || n_sample == 0) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    uint32_t actual = n_sample < n_total ? n_sample : n_total;
    out->indices = malloc(actual * sizeof(uint32_t));
    out->timestamps_ms = malloc(actual * sizeof(uint64_t));
    if (!out->indices || !out->timestamps_ms) {
        free(out->indices); free(out->timestamps_ms);
        return OC_ERR_OOM;
    }

    /* Fisher-Yates partial shuffle. */
    uint32_t *pool = malloc(n_total * sizeof(uint32_t));
    if (!pool) { free(out->indices); free(out->timestamps_ms); return OC_ERR_OOM; }
    for (uint32_t i = 0; i < n_total; i++) pool[i] = i;

    uint32_t state = seed ? seed : 1;
    for (uint32_t i = 0; i < actual; i++) {
        uint32_t j = i + (lcg_next(&state) % (n_total - i));
        uint32_t tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
        out->indices[i] = pool[i];
        out->timestamps_ms[i] = (uint64_t)pool[i] * 1000;
    }
    free(pool);
    out->n_indices = actual;
    return OC_OK;
}

OcError oc_fs_sample(const OcFsConfig *cfg, uint64_t video_duration_ms,
                    uint32_t total_frames, OcFsResult *out)
{
    if (!cfg || !out) return OC_ERR_INVALID_ARG;
    if (total_frames == 0) return OC_ERR_INVALID_ARG;
    (void)video_duration_ms; /* reserved for future keyframe detection */

    switch (cfg->strategy) {
    case OC_FS_STRATEGY_UNIFORM:
        return oc_fs_sample_uniform(total_frames, cfg->n_frames, out);
    case OC_FS_STRATEGY_FIRST_N:
        return oc_fs_sample_first_n(total_frames, cfg->n_frames, out);
    case OC_FS_STRATEGY_LAST_N:
        return oc_fs_sample_last_n(total_frames, cfg->n_frames, out);
    case OC_FS_STRATEGY_RANDOM:
        return oc_fs_sample_random(total_frames, cfg->n_frames, cfg->seed, out);
    case OC_FS_STRATEGY_KEYFRAME:
        /* Keyframe sampling not implemented; fallback to uniform. */
        return oc_fs_sample_uniform(total_frames, cfg->n_frames, out);
    default:
        return oc_fs_sample_uniform(total_frames, cfg->n_frames, out);
    }
}

uint32_t oc_fs_estimate_n_frames(uint64_t duration_ms, uint32_t fps, uint32_t max_frames)
{
    if (fps == 0 || duration_ms == 0) return 0;
    uint64_t total = (duration_ms * fps) / 1000;
    if (max_frames > 0 && total > max_frames) total = max_frames;
    return (uint32_t)total;
}

const char *oc_fs_strategy_name(OcFsStrategy strategy)
{
    switch (strategy) {
    case OC_FS_STRATEGY_UNIFORM:  return "uniform";
    case OC_FS_STRATEGY_FIRST_N:  return "first_n";
    case OC_FS_STRATEGY_LAST_N:   return "last_n";
    case OC_FS_STRATEGY_RANDOM:   return "random";
    case OC_FS_STRATEGY_KEYFRAME: return "keyframe";
    default: return "unknown";
    }
}

void oc_fs_result_free(OcFsResult *result)
{
    if (!result) return;
    free(result->indices);
    free(result->timestamps_ms);
    memset(result, 0, sizeof(*result));
}
