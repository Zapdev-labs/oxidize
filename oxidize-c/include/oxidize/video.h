#ifndef OXIDIZE_VIDEO_H
#define OXIDIZE_VIDEO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/vision.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcVideoFrame {
    uint8_t       *rgb;       /* width * height * 3 bytes               */
    uint32_t       width;
    uint32_t       height;
    uint64_t       timestamp_ms; /* presentation timestamp in the video  */
    uint32_t       frame_idx;     /* sequential index                      */
} OcVideoFrame;


typedef enum {
    OC_FRAME_SAMPLE_UNIFORM  = 0, /* N frames evenly spaced               */
    OC_FRAME_SAMPLE_FPS       = 1, /* fixed frames per second              */
    OC_FRAME_SAMPLE_SCENE     = 2, /* scene-change detection               */
    OC_FRAME_SAMPLE_KEYFRAME  = 3, /* I-frames only (requires decoder)     */
} OcFrameSampleStrategy;

typedef struct OcVideoFrameConfig {
    OcFrameSampleStrategy strategy;
    uint32_t n_frames;             /* target frame count (uniform)           */
    uint32_t fps;                  /* frames per second (FPS mode)            */
    float    scene_threshold;      /* histogram diff threshold (0..1, default 0.3) */
    uint32_t max_frames;           /* hard cap (default 32)                   */
    bool     resize_enabled;       /* resize frames before encoding            */
    uint32_t target_width;         /* resize target (default 224)              */
    uint32_t target_height;
} OcVideoFrameConfig;

#define OC_VIDEO_FRAME_CONFIG_DEFAULT ((OcVideoFrameConfig){ \
    OC_FRAME_SAMPLE_UNIFORM, 8, 1, 0.3f, 32, true, 224, 224 })

typedef struct OcVideoFrameSampler {
    OcVideoFrameConfig cfg;
    /* Internal state for scene-change detection */
    uint32_t  prev_hist[64];     /* coarse RGB histogram for previous frame */
    bool      has_prev_hist;
    uint32_t  total_frames_seen;
    uint32_t  frames_selected;
} OcVideoFrameSampler;

/* Initialize a frame sampler. */
void oc_video_sampler_init(OcVideoFrameSampler *s, OcVideoFrameConfig cfg);

/* Process a single decoded frame. Returns true if the frame was selected
 * (i.e., added to the output set), false if it was skipped. */
bool oc_video_sampler_process(OcVideoFrameSampler *s, const OcVideoFrame *frame);

/* Compute the final set of frame indices to sample, given the total frame
 * count. Used for uniform/FPS sampling where we know the total upfront. */
OcError oc_video_sampler_plan(OcVideoFrameSampler *s, uint32_t total_frames,
                               uint32_t *out_indices, size_t *out_count);

/* Reset the sampler for a new video. */
void oc_video_sampler_reset(OcVideoFrameSampler *s);


typedef enum {
    OC_TEMPORAL_MEAN    = 0,  /* mean pool across frames                  */
    OC_TEMPORAL_MAX     = 1,  /* max pool across frames                   */
    OC_TEMPORAL_ATTN    = 2,  /* attention-weighted pooling               */
    OC_TEMPORAL_CONCAT  = 3,  /* concatenate frame embeddings             */
} OcTemporalAggregation;

typedef struct OcVideoEmbedding {
    float   *data;              /* aggregated embedding vector            */
    size_t   dim;               /* embedding dimensionality               */
    uint32_t n_frames;          /* number of frames aggregated            */
    OcTemporalAggregation method;
} OcVideoEmbedding;

/* Aggregate frame embeddings into a single video embedding. */
OcError oc_video_aggregate(const float *frame_embeddings, /* [n_frames * dim] */
                            uint32_t n_frames, size_t dim,
                            OcTemporalAggregation method,
                            OcVideoEmbedding *out);

/* Free a video embedding. */
void oc_video_embedding_free(OcVideoEmbedding *emb);


typedef struct OcVideoPrompt {
    OcVideoEmbedding *video_emb;
    const char       *text_prompt;
    uint32_t          *token_ids;     /* tokenized text (owned by caller) */
    size_t             n_tokens;
} OcVideoPrompt;

/* Construct a multimodal prompt from video embedding + text. */
OcError oc_video_prompt_create(OcVideoEmbedding *video_emb,
                                const char *text,
                                OcVideoPrompt *out);

/* Free a video prompt (does not free video_emb or token_ids). */
void oc_video_prompt_free(OcVideoPrompt *p);


/* Compute a coarse 64-bin RGB histogram from a frame. */
void oc_video_compute_histogram(const OcVideoFrame *frame, uint32_t hist[64]);

/* Compute the difference between two histograms (0..1, 0=identical). */
float oc_video_histogram_diff(const uint32_t a[64], const uint32_t b[64]);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_VIDEO_H */
