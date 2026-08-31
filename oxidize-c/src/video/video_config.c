/* video_config.c — Video pipeline configuration implementation. */
#define _POSIX_C_SOURCE 200809L

#include "oxidize/video_config.h"

#include <stddef.h>


#define OC_VID_DEFAULT_TARGET_FRAMES    8u
#define OC_VID_DEFAULT_DENSE_STRIDE     1u
#define OC_VID_DEFAULT_TEMPORAL_HIDDEN  768u
#define OC_VID_DEFAULT_LLM_HIDDEN      4096u
#define OC_VID_DEFAULT_MAX_VIDEO_TOKENS 256u


void oc_video_config_init(OcVideoConfig *cfg)
{
    if (cfg == NULL) {
        return;
    }
    /* Vision config init never fails for a non-NULL pointer; ignore rc. */
    (void)oc_vision_cfg_init(&cfg->vision);
    cfg->target_frames    = OC_VID_DEFAULT_TARGET_FRAMES;
    cfg->sampling         = OC_VID_SAMPLE_UNIFORM;
    cfg->dense_stride     = OC_VID_DEFAULT_DENSE_STRIDE;
    cfg->temporal_pool    = OC_VID_POOL_MEAN;
    cfg->temporal_hidden  = OC_VID_DEFAULT_TEMPORAL_HIDDEN;
    cfg->llm_hidden       = OC_VID_DEFAULT_LLM_HIDDEN;
    cfg->max_video_tokens = OC_VID_DEFAULT_MAX_VIDEO_TOKENS;
}


OcError oc_video_config_validate(const OcVideoConfig *cfg)
{
    if (cfg == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    if (cfg->target_frames == 0) {
        return OC_ERR_INVALID_ARG;
    }
    if (cfg->temporal_hidden == 0) {
        return OC_ERR_INVALID_ARG;
    }
    if (cfg->llm_hidden == 0) {
        return OC_ERR_INVALID_ARG;
    }
    if (cfg->max_video_tokens == 0) {
        return OC_ERR_INVALID_ARG;
    }
    /* Sampling enum range check. */
    if ((uint32_t)cfg->sampling > (uint32_t)OC_VID_SAMPLE_ADAPTIVE) {
        return OC_ERR_INVALID_ARG;
    }
    /* Temporal pool enum range check. */
    if ((uint32_t)cfg->temporal_pool > (uint32_t)OC_VID_POOL_LSTM) {
        return OC_ERR_INVALID_ARG;
    }
    /* DENSE strategy must have a non-zero stride. */
    if (cfg->sampling == OC_VID_SAMPLE_DENSE && cfg->dense_stride == 0) {
        return OC_ERR_FORMAT;
    }
    return OC_OK;
}


const char *oc_video_sampling_name(OcFrameSamplingStrategy s)
{
    switch (s) {
    case OC_VID_SAMPLE_UNIFORM:
        return "uniform";
    case OC_VID_SAMPLE_DENSE:
        return "dense";
    case OC_VID_SAMPLE_ADAPTIVE:
        return "adaptive";
    default:
        return "unknown";
    }
}

const char *oc_video_pool_name(OcTemporalPool p)
{
    switch (p) {
    case OC_VID_POOL_MEAN:
        return "mean";
    case OC_VID_POOL_MAX:
        return "max";
    case OC_VID_POOL_LAST:
        return "last";
    case OC_VID_POOL_ATTENTION:
        return "attention";
    case OC_VID_POOL_LSTM:
        return "lstm";
    default:
        return "unknown";
    }
}


uint32_t oc_video_config_n_tokens(const OcVideoConfig *cfg)
{
    if (cfg == NULL) {
        return 0u;
    }
    uint32_t n = cfg->target_frames;
    /* ATTENTION and LSTM pools collapse the frame axis into a learned
     * token stream; cap by max_video_tokens. The other pools emit one
     * token per sampled frame (still capped). */
    if (n > cfg->max_video_tokens) {
        n = cfg->max_video_tokens;
    }
    return n;
}
