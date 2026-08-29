/* prune.h — Wanda and magnitude pruning utility. */
#ifndef OXIDIZE_PRUNE_H
#define OXIDIZE_PRUNE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/activation_stats.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_PRUNE_WANDA     = 0,  /* |W| * ||X||_2                          */
    OC_PRUNE_MAGNITUDE = 1,  /* |W|                                    */
} OcPruneStrategy;

typedef struct OcPruneConfig {
    OcPruneStrategy strategy;
    float           sparsity;      /* fraction of weights to prune [0,1)  */
    const char     *input_path;    /* input GGUF path                      */
    const char     *output_path;   /* output GGUF path                     */
    /* Calibration activation stats for OC_PRUNE_WANDA (may be NULL, in
     * which case Wanda falls back to magnitude pruning). */
    const OcActivationStats *activation_stats;
    bool            verbose;
} OcPruneConfig;

/* Prune a model using the configured strategy. */
OcError oc_prune_model(const OcPruneConfig *cfg);

/* Prune with Wanda: uses activation stats for importance scoring.
 * `stats` should be collected via oc_activation_stats_observe during calibration. */
OcError oc_prune_wanda(const char *input_path, const char *output_path,
                        float sparsity,
                        const OcActivationStats *stats);

/* Prune with magnitude: simply zero the smallest |W| values. */
OcError oc_prune_magnitude(const char *input_path, const char *output_path,
                            float sparsity);

/* Compute the sparsity of a model (fraction of zero weights). */
OcError oc_prune_compute_sparsity(const char *model_path, float *out_sparsity);

/* Get the name of a prune strategy. */
const char *oc_prune_strategy_name(OcPruneStrategy s);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_PRUNE_H */
