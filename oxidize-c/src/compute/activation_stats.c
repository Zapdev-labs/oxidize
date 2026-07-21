/*
 * activation_stats.c — activation statistics implementation.
 */
#include "oxidize/activation_stats.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

OcError oc_activation_stats_init(OcActivationStats *stats, size_t n_layers)
{
    if (!stats) return OC_ERR_INVALID_ARG;
    memset(stats, 0, sizeof(*stats));
    stats->n_layers = n_layers;
    stats->layers = calloc(n_layers, sizeof(OcLayerActivationStats));
    if (!stats->layers) return OC_ERR_OOM;
    return OC_OK;
}

OcError oc_activation_stats_observe(OcActivationStats *stats, size_t layer_idx,
                                    const float *activations,
                                    size_t batch_size, size_t input_dim)
{
    if (!stats || !activations) return OC_ERR_INVALID_ARG;
    if (layer_idx >= stats->n_layers) return OC_ERR_INVALID_ARG;

    OcLayerActivationStats *ls = &stats->layers[layer_idx];

    /* Allocate the sum_sq array on first use. */
    if (!ls->active) {
        ls->input_dim = input_dim;
        ls->sum_sq = calloc(input_dim, sizeof(double));
        if (!ls->sum_sq) return OC_ERR_OOM;
        ls->active = true;
    }

    /* Accumulate sum of squares per column. */
    for (size_t b = 0; b < batch_size; b++) {
        const float *row = activations + b * input_dim;
        for (size_t c = 0; c < input_dim; c++) {
            ls->sum_sq[c] += (double)row[c] * row[c];
        }
    }
    ls->n_samples += batch_size;
    return OC_OK;
}

OcError oc_activation_stats_finalize(OcActivationStats *stats)
{
    /* Nothing to do - norms are computed on demand in get_l2_norms. */
    (void)stats;
    return OC_OK;
}

OcError oc_activation_stats_get_l2_norms(const OcActivationStats *stats,
                                         size_t layer_idx,
                                         float *out_norms, size_t dim)
{
    if (!stats || !out_norms) return OC_ERR_INVALID_ARG;
    if (layer_idx >= stats->n_layers) return OC_ERR_INVALID_ARG;
    const OcLayerActivationStats *ls = &stats->layers[layer_idx];
    if (!ls->active) return OC_ERR_INVALID_ARG;
    if (dim != ls->input_dim) return OC_ERR_INVALID_ARG;

    /* L2 norm = sqrt(sum_sq / n_samples). */
    double inv_n = (ls->n_samples > 0) ? 1.0 / (double)ls->n_samples : 1.0;
    for (size_t c = 0; c < dim; c++) {
        out_norms[c] = (float)sqrt(ls->sum_sq[c] * inv_n);
    }
    return OC_OK;
}

void oc_activation_stats_free(OcActivationStats *stats)
{
    if (!stats) return;
    if (stats->layers) {
        for (size_t i = 0; i < stats->n_layers; i++) {
            free(stats->layers[i].sum_sq);
        }
        free(stats->layers);
    }
    memset(stats, 0, sizeof(*stats));
}

void oc_activation_stats_summary(const OcActivationStats *stats,
                                 char *buf, size_t buf_len)
{
    if (!stats || !buf || buf_len == 0) return;
    size_t active_count = 0;
    size_t total_samples = 0;
    for (size_t i = 0; i < stats->n_layers; i++) {
        if (stats->layers[i].active) {
            active_count++;
            total_samples += stats->layers[i].n_samples;
        }
    }
    snprintf(buf, buf_len,
             "activation_stats: %zu/%zu layers tracked, %zu total samples",
             active_count, stats->n_layers, total_samples);
}
