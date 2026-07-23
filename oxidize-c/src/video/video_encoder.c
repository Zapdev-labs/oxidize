/*
 * video_encoder.c — Per-frame vision encoding + temporal pooling stub.
 *
 * Port of oxidize-core/src/video/encoder.rs. The C port is a stub:
 * it copies per-frame embeddings into the output buffer, padding /
 * truncating each frame to llm_hidden. No real vision backbone or
 * learned temporal projection is present.
 */
#define _POSIX_C_SOURCE 200809L

#include "oxidize/video_encoder.h"

#include <stdlib.h>
#include <string.h>

/* ─── helpers ────────────────────────────────────────────────────────── */

static OcError validate_cfg(const OcVideoEncoderConfig *cfg)
{
    if (cfg == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    if (cfg->vision_hidden == 0 || cfg->temporal_hidden == 0 ||
        cfg->llm_hidden == 0) {
        return OC_ERR_INVALID_ARG;
    }
    return OC_OK;
}

/* ─── lifecycle ──────────────────────────────────────────────────────── */

OcError oc_video_encoder_init(OcVideoEncoder *enc,
                               const OcVideoEncoderConfig *cfg)
{
    if (enc == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    OcError e = validate_cfg(cfg);
    if (e != OC_OK) {
        return e;
    }
    enc->config       = *cfg;
    enc->output_tokens = NULL;
    enc->n_tokens      = 0;
    return OC_OK;
}

void oc_video_encoder_free(OcVideoEncoder *enc)
{
    if (enc == NULL) {
        return;
    }
    free(enc->output_tokens);
    enc->output_tokens = NULL;
    enc->n_tokens      = 0;
}

/* ─── encode ─────────────────────────────────────────────────────────── */

OcError oc_video_encoder_encode(OcVideoEncoder *enc,
                                 const float *frame_embeddings,
                                 size_t n_frames, size_t frame_dim)
{
    if (enc == NULL || frame_embeddings == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    if (frame_dim != enc->config.vision_hidden) {
        return OC_ERR_INVALID_ARG;
    }
    /* n_frames == 0 is allowed: it just produces zero tokens. */
    size_t need = n_frames * (size_t)enc->config.llm_hidden;
    float *buf = (float *)realloc(enc->output_tokens, need * sizeof(float));
    if (need > 0 && buf == NULL) {
        return OC_ERR_OOM;
    }
    enc->output_tokens = buf;
    enc->n_tokens      = n_frames;

    /* Stub projection: copy frame_dim values into the first
     * min(frame_dim, llm_hidden) slots of each token; zero-fill the rest. */
    size_t copy_n = frame_dim;
    if (copy_n > enc->config.llm_hidden) {
        copy_n = enc->config.llm_hidden;
    }
    for (size_t i = 0; i < n_frames; ++i) {
        float *dst = buf + i * enc->config.llm_hidden;
        const float *src = frame_embeddings + i * frame_dim;
        memcpy(dst, src, copy_n * sizeof(float));
        for (size_t j = copy_n; j < enc->config.llm_hidden; ++j) {
            dst[j] = 0.f;
        }
    }
    return OC_OK;
}

/* ─── accessors ──────────────────────────────────────────────────────── */

size_t oc_video_encoder_n_tokens(const OcVideoEncoder *enc)
{
    if (enc == NULL) {
        return 0;
    }
    return enc->n_tokens;
}

const float *oc_video_encoder_output(const OcVideoEncoder *enc)
{
    if (enc == NULL) {
        return NULL;
    }
    return enc->output_tokens;
}

OcError oc_video_encoder_get_token(const OcVideoEncoder *enc,
                                    size_t frame, size_t dim,
                                    float *out)
{
    if (enc == NULL || out == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    if (frame >= enc->n_tokens || dim >= enc->config.llm_hidden) {
        return OC_ERR_INVALID_ARG;
    }
    *out = enc->output_tokens[frame * enc->config.llm_hidden + dim];
    return OC_OK;
}
