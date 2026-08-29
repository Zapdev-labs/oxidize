#ifndef OXIDIZE_ACTIVATION_STATS_H
#define OXIDIZE_ACTIVATION_STATS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Statistics for a single linear layer's input activations. */
typedef struct OcLayerActivationStats {
    size_t  input_dim;        /* number of columns (features)            */
    double *sum_sq;          /* per-column sum of squares (L2^2)         */
    size_t  n_samples;        /* number of activation samples observed   */
    bool    active;           /* is this layer being tracked?            */
} OcLayerActivationStats;

/* Overall stats collector for all layers in a model. */
typedef struct OcActivationStats {
    OcLayerActivationStats *layers;
    size_t n_layers;
} OcActivationStats;

/* Initialize the stats collector for `n_layers` layers. */
OcError oc_activation_stats_init(OcActivationStats *stats, size_t n_layers);

/* Observe a batch of activations for a specific layer.
 * `activations` is [batch_size, input_dim] row-major. */
OcError oc_activation_stats_observe(OcActivationStats *stats, size_t layer_idx,
                                    const float *activations,
                                    size_t batch_size, size_t input_dim);

/* Finalize: compute L2 norms from accumulated sums. */
OcError oc_activation_stats_finalize(OcActivationStats *stats);

/* Get the L2 norm of the input activations for layer `layer_idx`.
 * Writes the norm into `out_norms` (length = input_dim). */
OcError oc_activation_stats_get_l2_norms(const OcActivationStats *stats,
                                         size_t layer_idx,
                                         float *out_norms, size_t dim);

/* Free the stats collector. */
void oc_activation_stats_free(OcActivationStats *stats);

/* Get a summary string of the stats (for debugging). */
void oc_activation_stats_summary(const OcActivationStats *stats,
                                 char *buf, size_t buf_len);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ACTIVATION_STATS_H */
