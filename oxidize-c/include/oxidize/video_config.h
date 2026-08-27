/*
 * video_config.h — Video pipeline configuration types.
 *
 * Port of oxidize-core/src/video/config.rs. Defines frame sampling
 * strategies, temporal pooling modes, and the top-level OcVideoConfig
 * struct that governs the video multimodal path.
 *
 * NOTE: This is distinct from the older `video.h` frame-extraction
 * config; this module mirrors the Rust `VideoConfig` type used by the
 * encoder + LLM bridge.
 */
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

/* ─── Frame sampling strategy ────────────────────────────────────────────
 *
 * Mirrors the Rust `FrameSamplingStrategy` enum.
 *  - UNIFORM:  evenly spaced N frames across the video duration.
 *  - DENSE:    every `dense_stride`-th frame, producing a denser set.
 *  - ADAPTIVE: scene-change driven selection.
 */
typedef enum {
    OC_VID_SAMPLE_UNIFORM  = 0,
    OC_VID_SAMPLE_DENSE    = 1,
    OC_VID_SAMPLE_ADAPTIVE = 2,
} OcFrameSamplingStrategy;

/* ─── Temporal pooling ───────────────────────────────────────────────────
 *
 * Mirrors the Rust `TemporalPool` enum. Controls how per-frame vision
 * embeddings are collapsed into a fixed-size sequence of LLM tokens.
 */
typedef enum {
    OC_VID_POOL_MEAN      = 0,
    OC_VID_POOL_MAX       = 1,
    OC_VID_POOL_LAST      = 2,
    OC_VID_POOL_ATTENTION = 3,
    OC_VID_POOL_LSTM      = 4,
} OcTemporalPool;

/* ─── Top-level video config ─────────────────────────────────────────────
 *
 * Mirrors the Rust `VideoConfig` struct. `dense_stride` is only used
 * when `sampling == OC_VID_SAMPLE_DENSE`; it is stored here rather than
 * in a sub-union to keep the C ABI flat and POD.
 */
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

/* Initialize a config with sensible defaults:
 *   target_frames=8, UNIFORM sampling, dense_stride=1,
 *   MEAN pool, temporal_hidden=768, llm_hidden=4096,
 *   max_video_tokens=256, vision via oc_vision_cfg_init. */
void oc_video_config_init(OcVideoConfig *cfg);

/* Validate a config. Returns:
 *   OC_OK                — config is sound.
 *   OC_ERR_INVALID_ARG   — cfg is NULL or has zero/overflowing fields.
 *   OC_ERR_FORMAT        — DENSE strategy with dense_stride == 0. */
OcError oc_video_config_validate(const OcVideoConfig *cfg);

/* Human-readable name for a sampling strategy. Never returns NULL.
 * Returns "unknown" for out-of-range values. */
const char *oc_video_sampling_name(OcFrameSamplingStrategy s);

/* Human-readable name for a temporal pool. Never returns NULL.
 * Returns "unknown" for out-of-range values. */
const char *oc_video_pool_name(OcTemporalPool p);

/* Compute the number of video tokens a config would emit for a single
 * video given its target_frames. For most pools this is target_frames
 * (one token per frame); ATTENTION/LSTM pools collapse to a single
 * token stream of size max_video_tokens (bounded by target_frames).
 *
 * Returns 0 if cfg is NULL. */
uint32_t oc_video_config_n_tokens(const OcVideoConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_VIDEO_CONFIG_H */
