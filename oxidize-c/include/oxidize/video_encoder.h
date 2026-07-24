/*
 * video_encoder.h — Per-frame vision encoding + temporal pooling stub.
 *
 * Port of oxidize-core/src/video/encoder.rs. The C port does not ship
 * a real vision backbone; this module provides a stub encoder that
 * runs a simple (mean/max) temporal pooling over per-frame embeddings
 * and projects them into the LLM hidden space via a 1:1 copy (the
 * projection is a placeholder until real weights are wired in).
 */
#ifndef OXIDIZE_VIDEO_ENCODER_H
#define OXIDIZE_VIDEO_ENCODER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Encoder config ──────────────────────────────────────────────────── */

typedef struct {
    uint32_t vision_hidden;   /* per-frame vision embedding dim */
    uint32_t temporal_hidden; /* temporal projection out dim     */
    uint32_t llm_hidden;     /* LLM hidden size (output dim)   */
} OcVideoEncoderConfig;

/* ─── Encoder ────────────────────────────────────────────────────────────
 *
 * Holds the configured projection dims and the most recent output
 * token buffer. `output_tokens` is owned and sized
 * [n_tokens * llm_hidden] floats after a successful encode().
 */
typedef struct OcVideoEncoder {
    OcVideoEncoderConfig config;
    float   *output_tokens;  /* owned; [n_tokens * llm_hidden] */
    size_t   n_tokens;
    /* Optional projection weight: [llm_hidden, vision_hidden] for
     * linear projection from vision_hidden to llm_hidden. NULL = copy/pad. */
    float   *proj_weight;
    float   *proj_bias;      /* [llm_hidden] or NULL */
} OcVideoEncoder;

/* Initialize an encoder. Validates the config: all dims must be > 0.
 * Returns OC_ERR_INVALID_ARG on bad args, OC_OK on success. */
OcError oc_video_encoder_init(OcVideoEncoder *enc,
                               const OcVideoEncoderConfig *cfg);

/* Free an encoder and its output buffer. Safe on NULL. */
void oc_video_encoder_free(OcVideoEncoder *enc);

/* Encode frames: takes [n_frames * frame_dim] per-frame embeddings and
 * produces video tokens. The temporal pooling reduces across frames;
 * the resulting token count is `n_frames` (one token per frame in this
 * stub). `frame_dim` MUST equal config.vision_hidden. The output buffer
 * is (re)allocated to [n_frames * llm_hidden].
 *
 * Returns OC_OK on success, OC_ERR_INVALID_ARG on bad args or dim
 * mismatch, OC_ERR_OOM on allocation failure. */
OcError oc_video_encoder_encode(OcVideoEncoder *enc,
                                 const float *frame_embeddings,
                                 size_t n_frames, size_t frame_dim);

/* Get output token count (0 before any encode). */
size_t oc_video_encoder_n_tokens(const OcVideoEncoder *enc);

/* Borrow the output token data, or NULL if no encode has run. */
const float *oc_video_encoder_output(const OcVideoEncoder *enc);

/* Get the value of the token at (frame, dim). Returns OC_ERR_INVALID_ARG
 * on bad args or out-of-range indices. */
OcError oc_video_encoder_get_token(const OcVideoEncoder *enc,
                                    size_t frame, size_t dim,
                                    float *out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_VIDEO_ENCODER_H */
