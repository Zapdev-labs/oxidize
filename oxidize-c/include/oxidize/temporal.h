#ifndef OXIDIZE_TEMPORAL_H
#define OXIDIZE_TEMPORAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Aggregation strategy. */
typedef enum {
    OC_TEMPORAL_MEAN = 0,
    OC_TEMPORAL_MAX  = 1,
    OC_TEMPORAL_LAST = 2,
    OC_TEMPORAL_ATTENTION = 3,
    OC_TEMPORAL_LSTM = 4,
} OcTemporalAggType;

/* Configuration for temporal aggregation. */
typedef struct {
    OcTemporalAggType agg_type;
    uint32_t          n_frames;    /* expected frame count            */
    uint32_t          feature_dim; /* dimensionality of each frame   */
    uint32_t          hidden_dim;  /* used by attention / LSTM       */
} OcTemporalConfig;

/* Stateful temporal aggregator. Owns `output` (malloc'd). */
typedef struct {
    OcTemporalConfig config;
    float           *output;
    uint32_t         n_output;
} OcTemporalState;

/* Initialize config with defaults: MEAN, n_frames=8, feature_dim=768,
 * hidden_dim=0. */
OcError oc_temporal_config_init(OcTemporalConfig *cfg);

/* Allocate state for the given config. `output` is sized to `feature_dim`
 * floats. */
OcError oc_temporal_init(OcTemporalState *state, const OcTemporalConfig *cfg);

/* Aggregate `n_frames` frames (each `config.feature_dim` floats) into `out`
 * (which must hold at least `config.feature_dim` floats). Also writes into
 * `state->output`. */
OcError oc_temporal_aggregate(OcTemporalState *state,
                              const float *frame_features,
                              uint32_t n_frames,
                              float *out);

/* Mean pooling: `out[i] = mean over frames of features[frame*dim + i]`. */
OcError oc_temporal_aggregate_mean(const float *features,
                                   uint32_t n_frames,
                                   uint32_t dim,
                                   float *out);

/* Max pooling: `out[i] = max over frames`. */
OcError oc_temporal_aggregate_max(const float *features,
                                  uint32_t n_frames,
                                  uint32_t dim,
                                  float *out);

/* Take the last frame: `out[i] = features[(n_frames-1)*dim + i]`. */
OcError oc_temporal_aggregate_last(const float *features,
                                   uint32_t n_frames,
                                   uint32_t dim,
                                   float *out);

/* Free state buffers. Safe on NULL or already-freed state. */
void oc_temporal_free(OcTemporalState *state);

/* Human-readable name for an aggregation type. Never returns NULL. */
const char *oc_temporal_agg_type_name(OcTemporalAggType type);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_TEMPORAL_H */
