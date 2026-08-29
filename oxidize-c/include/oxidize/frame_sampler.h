/* frame_sampler.h — Video frame sampling for multimodal models. */
#ifndef OXIDIZE_FRAME_SAMPLER_H
#define OXIDIZE_FRAME_SAMPLER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_FS_MAX_FRAMES 64

typedef enum {
    OC_FS_STRATEGY_UNIFORM = 0,
    OC_FS_STRATEGY_FIRST_N = 1,
    OC_FS_STRATEGY_LAST_N = 2,
    OC_FS_STRATEGY_RANDOM = 3,
    OC_FS_STRATEGY_KEYFRAME = 4,
} OcFsStrategy;

typedef struct {
    OcFsStrategy strategy;
    uint32_t n_frames;
    uint32_t fps;
    uint32_t seed;
} OcFsConfig;

typedef struct {
    uint32_t *indices;
    uint32_t n_indices;
    uint64_t *timestamps_ms;
} OcFsResult;

OcError oc_fs_config_init(OcFsConfig *cfg);
OcError oc_fs_sample(const OcFsConfig *cfg, uint64_t video_duration_ms,
                    uint32_t total_frames, OcFsResult *out);
OcError oc_fs_sample_uniform(uint32_t n_total, uint32_t n_sample, OcFsResult *out);
OcError oc_fs_sample_first_n(uint32_t n_total, uint32_t n_sample, OcFsResult *out);
OcError oc_fs_sample_last_n(uint32_t n_total, uint32_t n_sample, OcFsResult *out);
OcError oc_fs_sample_random(uint32_t n_total, uint32_t n_sample, uint32_t seed,
                           OcFsResult *out);
uint32_t oc_fs_estimate_n_frames(uint64_t duration_ms, uint32_t fps, uint32_t max_frames);
const char *oc_fs_strategy_name(OcFsStrategy strategy);
void oc_fs_result_free(OcFsResult *result);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_FRAME_SAMPLER_H */
