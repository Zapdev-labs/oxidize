/*
 * video.c — Video multimodal frame sampling and temporal aggregation.
 *
 * Port of oxidize-core/src/video/. This module handles frame selection
 * strategies (uniform, FPS, scene-change detection), frame-to-embedding
 * temporal aggregation (mean, max, attention, concat), and video+text
 * multimodal prompt construction.
 *
 * The actual video decoding (demuxing, H.264/VP9 decoding) is handled by
 * the caller — this module receives already-decoded RGB frames. This
 * keeps the C port dependency-free while providing the full frame
 * sampling and aggregation pipeline.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/video.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ─── Frame sampler ────────────────────────────────────────────────────── */

void oc_video_sampler_init(OcVideoFrameSampler *s, OcVideoFrameConfig cfg)
{
    if (!s) return;
    memset(s, 0, sizeof(*s));
    s->cfg = cfg;
    if (s->cfg.max_frames == 0) s->cfg.max_frames = 32;
    if (s->cfg.scene_threshold <= 0.0f) s->cfg.scene_threshold = 0.3f;
    s->has_prev_hist = false;
}

void oc_video_sampler_reset(OcVideoFrameSampler *s)
{
    if (!s) return;
    s->has_prev_hist = false;
    s->total_frames_seen = 0;
    s->frames_selected = 0;
    memset(s->prev_hist, 0, sizeof(s->prev_hist));
}

/* Plan uniform sampling: select n_frames evenly spaced indices. */
OcError oc_video_sampler_plan(OcVideoFrameSampler *s, uint32_t total_frames,
                               uint32_t *out_indices, size_t *out_count)
{
    if (!s || !out_indices || !out_count) return OC_ERR_INVALID_ARG;
    if (total_frames == 0) { *out_count = 0; return OC_OK; }

    uint32_t n = s->cfg.n_frames;
    if (n == 0) n = 8;
    if (n > s->cfg.max_frames) n = s->cfg.max_frames;
    if (n > total_frames) n = total_frames;

    /* Evenly spaced: indices = i * total_frames / n for i in 0..n-1. */
    for (uint32_t i = 0; i < n; i++) {
        out_indices[i] = (uint32_t)((uint64_t)i * total_frames / n);
    }
    *out_count = n;
    return OC_OK;
}

/* Scene-change detection via histogram difference. */
bool oc_video_sampler_process(OcVideoFrameSampler *s, const OcVideoFrame *frame)
{
    if (!s || !frame) return false;
    s->total_frames_seen++;

    if (s->frames_selected >= s->cfg.max_frames) return false;

    switch (s->cfg.strategy) {
    case OC_FRAME_SAMPLE_UNIFORM: {
        /* For uniform sampling, we need the total count upfront.
         * If we don't know it, accept every Nth frame — but never more than
         * the requested target (use oc_video_sampler_plan for exact uniform
         * spacing when the total frame count is known). */
        uint32_t target = s->cfg.n_frames ? s->cfg.n_frames : 8;
        if (target > s->cfg.max_frames) target = s->cfg.max_frames;
        if (s->frames_selected >= target) return false;
        if (s->cfg.n_frames > 0 && s->total_frames_seen > 0) {
            /* Approximate: accept roughly n_frames spread across. */
            uint32_t stride = 1;
            if (s->cfg.n_frames > 0 && s->cfg.n_frames < s->total_frames_seen) {
                stride = s->total_frames_seen / s->cfg.n_frames;
            }
            if (stride == 0) stride = 1;
            if (s->frames_selected == 0) {
                s->frames_selected++;
                return true;
            }
            if (s->total_frames_seen % stride == 0) {
                s->frames_selected++;
                return true;
            }
            return false;
        }
        s->frames_selected++;
        return true;
    }

    case OC_FRAME_SAMPLE_FPS:
        /* Time-based selection: accept a frame when at least 1000/fps ms of
         * presentation time has elapsed since the last selected frame, so
         * the output rate is independent of the source frame rate. */
        if (s->cfg.fps > 0) {
            /* ponytail: FPS state reuses prev_hist[0..1] (unused outside the
             * SCENE strategy) as the last-selected timestamp; move to a
             * dedicated field when video.h can grow one. */
            uint64_t interval_ms = 1000ull / s->cfg.fps;
            if (interval_ms == 0) interval_ms = 1;
            uint64_t last = ((uint64_t)s->prev_hist[1] << 32) | s->prev_hist[0];
            if (!s->has_prev_hist ||
                frame->timestamp_ms >= last + interval_ms) {
                s->prev_hist[0] = (uint32_t)frame->timestamp_ms;
                s->prev_hist[1] = (uint32_t)(frame->timestamp_ms >> 32);
                s->has_prev_hist = true;
                s->frames_selected++;
                return true;
            }
            return false;
        }
        s->frames_selected++;
        return true;

    case OC_FRAME_SAMPLE_SCENE: {
        uint32_t hist[64];
        oc_video_compute_histogram(frame, hist);

        if (!s->has_prev_hist) {
            memcpy(s->prev_hist, hist, sizeof(hist));
            s->has_prev_hist = true;
            s->frames_selected++;
            return true;
        }

        float diff = oc_video_histogram_diff(s->prev_hist, hist);
        memcpy(s->prev_hist, hist, sizeof(hist));

        if (diff >= s->cfg.scene_threshold) {
            s->frames_selected++;
            return true;
        }
        return false;
    }

    case OC_FRAME_SAMPLE_KEYFRAME:
        /* Keyframe detection requires decoder info. Accept all. */
        s->frames_selected++;
        return true;

    default:
        s->frames_selected++;
        return true;
    }
}

/* ─── Histogram utilities ──────────────────────────────────────────────── */

void oc_video_compute_histogram(const OcVideoFrame *frame, uint32_t hist[64])
{
    memset(hist, 0, 64 * sizeof(uint32_t));
    if (!frame || !frame->rgb) return;

    size_t total_pixels = (size_t)frame->width * frame->height;
    /* Sample every Nth pixel for performance (N chosen to process ~4096 pixels). */
    size_t step = total_pixels / 4096;
    if (step == 0) step = 1;

    for (size_t i = 0; i < total_pixels; i += step) {
        uint8_t r = frame->rgb[i * 3];
        uint8_t g = frame->rgb[i * 3 + 1];
        uint8_t b = frame->rgb[i * 3 + 2];

        /* Coarse 4-bit per channel → 64 bins (4*4*4). */
        uint32_t ri = r >> 6;  /* 0..3 */
        uint32_t gi = g >> 6;
        uint32_t bi = b >> 6;
        hist[ri * 16 + gi * 4 + bi]++;
    }
}

float oc_video_histogram_diff(const uint32_t a[64], const uint32_t b[64])
{
    /* Chi-squared distance normalized to [0, 1]. */
    float sum_a = 0, sum_b = 0, diff = 0;
    for (int i = 0; i < 64; i++) {
        sum_a += (float)a[i];
        sum_b += (float)b[i];
    }
    if (sum_a == 0 && sum_b == 0) return 0.0f;
    float norm = (sum_a > sum_b) ? sum_a : sum_b;
    if (norm == 0) norm = 1.0f;

    for (int i = 0; i < 64; i++) {
        float fa = (float)a[i] / norm;
        float fb = (float)b[i] / norm;
        float d = fa - fb;
        if (fa + fb > 0) {
            diff += d * d / (fa + fb);
        }
    }
    /* Normalize: chi-squared distance is at most 2 for disjoint
     * distributions; scale to the documented [0, 1] range. */
    if (diff > 2.0f) diff = 2.0f;
    return diff * 0.5f;
}

/* ─── Frame-to-embedding aggregation ───────────────────────────────────── */

OcError oc_video_aggregate(const float *frame_embeddings,
                            uint32_t n_frames, size_t dim,
                            OcTemporalAggregation method,
                            OcVideoEmbedding *out)
{
    if (!frame_embeddings || !out || n_frames == 0 || dim == 0)
        return OC_ERR_INVALID_ARG;

    out->n_frames = n_frames;
    out->method = method;

    switch (method) {
    case OC_TEMPORAL_MEAN: {
        out->dim = dim;
        out->data = calloc(dim, sizeof(float));
        if (!out->data) return OC_ERR_OOM;
        for (uint32_t f = 0; f < n_frames; f++) {
            for (size_t d = 0; d < dim; d++) {
                out->data[d] += frame_embeddings[f * dim + d];
            }
        }
        float inv = 1.0f / (float)n_frames;
        for (size_t d = 0; d < dim; d++) out->data[d] *= inv;
        break;
    }

    case OC_TEMPORAL_MAX: {
        out->dim = dim;
        out->data = malloc(dim * sizeof(float));
        if (!out->data) return OC_ERR_OOM;
        /* Initialize to -inf. */
        for (size_t d = 0; d < dim; d++) out->data[d] = -INFINITY;
        for (uint32_t f = 0; f < n_frames; f++) {
            for (size_t d = 0; d < dim; d++) {
                float v = frame_embeddings[f * dim + d];
                if (v > out->data[d]) out->data[d] = v;
            }
        }
        break;
    }

    case OC_TEMPORAL_ATTN: {
        /* Attention-weighted pooling: compute attention scores per frame,
         * then weighted sum. */
        out->dim = dim;
        out->data = calloc(dim, sizeof(float));
        if (!out->data) return OC_ERR_OOM;

        /* Compute attention scores (dot product with mean). */
        float *scores = calloc(n_frames, sizeof(float));
        if (!scores) { free(out->data); out->data = NULL; return OC_ERR_OOM; }

        /* Mean as query. */
        float *mean = calloc(dim, sizeof(float));
        if (!mean) { free(scores); free(out->data); out->data = NULL; return OC_ERR_OOM; }
        for (uint32_t f = 0; f < n_frames; f++)
            for (size_t d = 0; d < dim; d++)
                mean[d] += frame_embeddings[f * dim + d];
        float inv_n = 1.0f / (float)n_frames;
        for (size_t d = 0; d < dim; d++) mean[d] *= inv_n;

        /* Dot products. */
        float max_score = -INFINITY;
        for (uint32_t f = 0; f < n_frames; f++) {
            float s = 0.0f;
            for (size_t d = 0; d < dim; d++)
                s += mean[d] * frame_embeddings[f * dim + d];
            scores[f] = s;
            if (s > max_score) max_score = s;
        }

        /* Softmax. */
        float sum_exp = 0.0f;
        for (uint32_t f = 0; f < n_frames; f++) {
            scores[f] = expf(scores[f] - max_score);
            sum_exp += scores[f];
        }
        if (sum_exp == 0) sum_exp = 1.0f;

        /* Weighted sum. */
        for (uint32_t f = 0; f < n_frames; f++) {
            float w = scores[f] / sum_exp;
            for (size_t d = 0; d < dim; d++)
                out->data[d] += w * frame_embeddings[f * dim + d];
        }

        free(scores);
        free(mean);
        break;
    }

    case OC_TEMPORAL_CONCAT: {
        /* Guard dim * n_frames * sizeof(float) against size_t overflow. */
        if (dim > (SIZE_MAX / sizeof(float)) / n_frames)
            return OC_ERR_INVALID_ARG;
        out->dim = dim * n_frames;
        out->data = malloc(out->dim * sizeof(float));
        if (!out->data) return OC_ERR_OOM;
        memcpy(out->data, frame_embeddings, out->dim * sizeof(float));
        break;
    }

    default:
        return OC_ERR_INVALID_ARG;
    }

    return OC_OK;
}

void oc_video_embedding_free(OcVideoEmbedding *emb)
{
    if (!emb) return;
    free(emb->data);
    emb->data = NULL;
    emb->dim = 0;
}

/* ─── Video prompt ──────────────────────────────────────────────────────── */

OcError oc_video_prompt_create(OcVideoEmbedding *video_emb,
                                const char *text,
                                OcVideoPrompt *out)
{
    if (!video_emb || !text || !out) return OC_ERR_INVALID_ARG;
    out->video_emb = video_emb;
    out->text_prompt = text;
    out->token_ids = NULL;
    out->n_tokens = 0;
    return OC_OK;
}

void oc_video_prompt_free(OcVideoPrompt *p)
{
    if (!p) return;
    p->video_emb = NULL;
    p->text_prompt = NULL;
    p->token_ids = NULL;
    p->n_tokens = 0;
}
