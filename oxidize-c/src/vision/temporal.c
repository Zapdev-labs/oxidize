#include "oxidize/temporal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

OcError oc_temporal_config_init(OcTemporalConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    cfg->agg_type    = OC_TEMPORAL_MEAN;
    cfg->n_frames    = 8;
    cfg->feature_dim = 768;
    cfg->hidden_dim  = 0;
    return OC_OK;
}

OcError oc_temporal_init(OcTemporalState *state, const OcTemporalConfig *cfg)
{
    if (!state || !cfg) return OC_ERR_INVALID_ARG;
    if (cfg->feature_dim == 0) return OC_ERR_INVALID_ARG;

    state->config   = *cfg;
    state->n_output = cfg->feature_dim;
    state->output   = (float *)calloc(cfg->feature_dim, sizeof(float));
    if (!state->output) {
        state->n_output = 0;
        return OC_ERR_OOM;
    }
    return OC_OK;
}

OcError oc_temporal_aggregate_mean(const float *features,
                                   uint32_t n_frames,
                                   uint32_t dim,
                                   float *out)
{
    if (!features || !out) return OC_ERR_INVALID_ARG;
    if (n_frames == 0 || dim == 0) return OC_ERR_INVALID_ARG;

    for (uint32_t i = 0; i < dim; i++) {
        double sum = 0.0;
        for (uint32_t f = 0; f < n_frames; f++) {
            sum += (double)features[f * dim + i];
        }
        out[i] = (float)(sum / (double)n_frames);
    }
    return OC_OK;
}

OcError oc_temporal_aggregate_max(const float *features,
                                  uint32_t n_frames,
                                  uint32_t dim,
                                  float *out)
{
    if (!features || !out) return OC_ERR_INVALID_ARG;
    if (n_frames == 0 || dim == 0) return OC_ERR_INVALID_ARG;

    /* Initialize with the first frame. */
    for (uint32_t i = 0; i < dim; i++) {
        out[i] = features[i];
    }
    for (uint32_t f = 1; f < n_frames; f++) {
        for (uint32_t i = 0; i < dim; i++) {
            float v = features[f * dim + i];
            if (v > out[i]) out[i] = v;
        }
    }
    return OC_OK;
}

OcError oc_temporal_aggregate_last(const float *features,
                                   uint32_t n_frames,
                                   uint32_t dim,
                                   float *out)
{
    if (!features || !out) return OC_ERR_INVALID_ARG;
    if (n_frames == 0 || dim == 0) return OC_ERR_INVALID_ARG;

    const float *last = features + (n_frames - 1) * dim;
    memcpy(out, last, dim * sizeof(float));
    return OC_OK;
}

OcError oc_temporal_aggregate(OcTemporalState *state,
                              const float *frame_features,
                              uint32_t n_frames,
                              float *out)
{
    if (!state || !out) return OC_ERR_INVALID_ARG;
    if (!frame_features && n_frames > 0) return OC_ERR_INVALID_ARG;
    if (n_frames == 0) return OC_ERR_INVALID_ARG;
    if (state->config.feature_dim == 0) return OC_ERR_INVALID_ARG;

    uint32_t dim = state->config.feature_dim;
    float *dst = state->output ? state->output : out;

    switch (state->config.agg_type) {
    case OC_TEMPORAL_MEAN:
        oc_temporal_aggregate_mean(frame_features, n_frames, dim, dst);
        break;
    case OC_TEMPORAL_ATTENTION: {
        /* Temporal self-attention: weight each frame by its similarity
         * to the mean feature, then weighted-average. */
        /* Compute mean feature. */
        float *mean = calloc(dim, sizeof(float));
        if (mean) {
            oc_temporal_aggregate_mean(frame_features, n_frames, dim, mean);
            /* Compute attention scores (dot product of each frame with mean). */
            float *scores = calloc(n_frames, sizeof(float));
            if (scores) {
                float max_score = -1e30f;
                for (uint32_t f = 0; f < n_frames; f++) {
                    float dot = 0.0f;
                    const float *frame = frame_features + f * dim;
                    for (uint32_t i = 0; i < dim; i++)
                        dot += frame[i] * mean[i];
                    scores[f] = dot;
                    if (scores[f] > max_score) max_score = scores[f];
                }
                /* Softmax. */
                float sum_exp = 0.0f;
                for (uint32_t f = 0; f < n_frames; f++) {
                    scores[f] = expf(scores[f] - max_score);
                    sum_exp += scores[f];
                }
                /* Weighted average. */
                if (sum_exp > 0.0f) {
                    for (uint32_t i = 0; i < dim; i++) dst[i] = 0.0f;
                    for (uint32_t f = 0; f < n_frames; f++) {
                        float w = scores[f] / sum_exp;
                        const float *frame = frame_features + f * dim;
                        for (uint32_t i = 0; i < dim; i++)
                            dst[i] += w * frame[i];
                    }
                } else {
                    oc_temporal_aggregate_mean(frame_features, n_frames, dim, dst);
                }
                free(scores);
            } else {
                oc_temporal_aggregate_mean(frame_features, n_frames, dim, dst);
            }
            free(mean);
        } else {
            oc_temporal_aggregate_mean(frame_features, n_frames, dim, dst);
        }
        break;
    }
    case OC_TEMPORAL_LSTM: {
        /* Simplified temporal recurrence: weighted exponential decay
         * across frames (emulates LSTM hidden state propagation). */
        float decay = 0.9f;
        for (uint32_t i = 0; i < dim; i++) dst[i] = 0.0f;
        for (uint32_t f = 0; f < n_frames; f++) {
            const float *frame = frame_features + f * dim;
            for (uint32_t i = 0; i < dim; i++) {
                dst[i] = dst[i] * decay + frame[i] * (1.0f - decay);
            }
        }
        break;
    }
    case OC_TEMPORAL_MAX:
        oc_temporal_aggregate_max(frame_features, n_frames, dim, dst);
        break;
    case OC_TEMPORAL_LAST:
        oc_temporal_aggregate_last(frame_features, n_frames, dim, dst);
        break;
    default:
        return OC_ERR_INVALID_ARG;
    }

    /* Copy to caller's output buffer if it differs from state->output. */
    if (out != dst) {
        memcpy(out, dst, dim * sizeof(float));
    }
    return OC_OK;
}

void oc_temporal_free(OcTemporalState *state)
{
    if (!state) return;
    free(state->output);
    state->output   = NULL;
    state->n_output = 0;
}

const char *oc_temporal_agg_type_name(OcTemporalAggType type)
{
    switch (type) {
    case OC_TEMPORAL_MEAN:      return "mean";
    case OC_TEMPORAL_MAX:       return "max";
    case OC_TEMPORAL_LAST:      return "last";
    case OC_TEMPORAL_ATTENTION: return "attention";
    case OC_TEMPORAL_LSTM:      return "lstm";
    default: return "unknown";
    }
}
