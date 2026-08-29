/* video_config.h — Video pipeline configuration types. */
#ifndef OXIDIZE_VIDEO_CONFIG_H
#define OXIDIZE_VIDEO_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/vision_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_VID_SAMPLE_UNIFORM  = 0,
    OC_VID_SAMPLE_DENSE    = 1,
    OC_VID_SAMPLE_ADAPTIVE = 2,
} OcFrameSamplingStrategy;

typedef enum {
    OC_VID_POOL_MEAN      = 0,
    OC_VID_POOL_MAX       = 1,
    OC_VID_POOL_LAST      = 2,
    OC_VID_POOL_ATTENTION = 3,
    OC_VID_POOL_LSTM      = 4,
} OcTemporalPool;

typedef struct OcVideoConfig {
    OcVisionConfig          vision;          /* vision encoder config         */
    uint32_t                target_frames;    /* desired sampled frame count */
    OcFrameSamplingStrategy sampling;         /* sampling strategy            */
    uint32_t                dense_stride;     /* stride for DENSE strategy    */
    OcTemporalPool          temporal_pool;    /* temporal aggregation mode    */
    uint32_t                temporal_hidden;  /* temporal projection out dim  */
    uint32_t                llm_hidden;       /* LLM hidden size              */
    uint32_t                max_video_tokens; /* hard cap on emitted tokens  */
} OcVideoConfig;

/* Initialize a config with sensible defaults: */
void oc_video_config_init(OcVideoConfig *cfg);

/* Validate a config. Returns: */
OcError oc_video_config_validate(const OcVideoConfig *cfg);

/* Human-readable name for a sampling strategy. Never returns NULL.
 * Returns "unknown" for out-of-range values. */
const char *oc_video_sampling_name(OcFrameSamplingStrategy s);

/* Human-readable name for a temporal pool. Never returns NULL.
 * Returns "unknown" for out-of-range values. */
const char *oc_video_pool_name(OcTemporalPool p);

/* Compute the number of video tokens a config would emit for a single */
uint32_t oc_video_config_n_tokens(const OcVideoConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_VIDEO_CONFIG_H */
