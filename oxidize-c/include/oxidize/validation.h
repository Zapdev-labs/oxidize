/* validation.h — cross-validation and model quality assessment utilities. */
#ifndef OXIDIZE_VALIDATION_H
#define OXIDIZE_VALIDATION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OC_VALIDATION_DEFAULT_N_FOLDS    5u
#define OC_VALIDATION_DEFAULT_MAX_SAMPLES 1000u
#define OC_VALIDATION_DEFAULT_SEED       42u
#define OC_VALIDATION_MAX_FOLDS         10u
#define OC_VALIDATION_MAX_INPUT_TOKENS  256u


typedef struct OcValidationConfig {
    uint32_t n_folds;        /* default OC_VALIDATION_DEFAULT_N_FOLDS      */
    uint32_t max_samples;    /* default OC_VALIDATION_DEFAULT_MAX_SAMPLES  */
    uint32_t seed;           /* default OC_VALIDATION_DEFAULT_SEED         */
} OcValidationConfig;

typedef struct OcValidationResult {
    float    accuracy;                         /* 0.0 .. 1.0               */
    float    loss;                             /* mean cross-entropy loss   */
    uint32_t n_samples;                         /* evaluated sample count    */
    float    per_fold_accuracy[OC_VALIDATION_MAX_FOLDS];
    uint32_t n_folds;                           /* number of folds used      */
} OcValidationResult;

typedef struct OcValidationSample {
    uint32_t input_tokens[OC_VALIDATION_MAX_INPUT_TOKENS];
    uint32_t n_input;
    uint32_t expected_token;
    uint32_t predicted_token;                  /* set by caller (model)    */
    float    logprob;                          /* logprob assigned to
                                                  expected_token (for loss) */
    float    weight;                            /* sample weight (default 1) */
} OcValidationSample;

typedef struct OcValidationState {
    OcValidationConfig    config;
    OcValidationSample   *samples;             /* heap-allocated buffer      */
    uint32_t               n_samples;
    uint32_t               cap_samples;
} OcValidationState;


/* Initialize config with defaults. */
OcError oc_validation_config_init(OcValidationConfig *cfg);


/* Allocate a validation state for the given config (NULL = defaults). */
OcError oc_validation_init(const OcValidationConfig *config,
                           OcValidationState **out);

/* Free all owned storage. Safe on NULL / already-freed. */
void oc_validation_free(OcValidationState *state);


/* Add a validation sample. `n_input` must be <= OC_VALIDATION_MAX_INPUT_TOKENS.
 * Copies the sample into internal storage. */
OcError oc_validation_add_sample(OcValidationState *state,
                                 const OcValidationSample *sample);

/* Remove all samples (config preserved). */
OcError oc_validation_clear(OcValidationState *state);


/* Run k-fold cross-validation over the stored samples. Folds are assigned deterministically from `config.seed`. For each fold, samples in that fold are "held out" — accuracy/loss are computed only over held-out samples (simulating train/test split semantics). Writes per-fold and overall aggregated metrics to `out_result`. */
OcError oc_validation_k_fold(const OcValidationState *state,
                              OcValidationResult *out_result);

/* Run single-pass validation: compute overall accuracy and loss over all
 * stored samples in a single pass (no folding). Sets n_folds=1. */
OcError oc_validation_single(const OcValidationState *state,
                             OcValidationResult *out_result);

/* Compute a confusion matrix of size n_classes x n_classes. Writes a
 * row-major matrix of counts into `out_matrix` (capacity n_classes*n_classes).
 * Element [expected * n_classes + predicted] is incremented per sample. */
OcError oc_validation_confusion_matrix(const OcValidationState *state,
                                       uint32_t n_classes,
                                       uint32_t *out_matrix);

/* Compute perplexity over the stored samples. Perplexity = exp(mean loss)
 * where loss is the per-sample cross-entropy (the sample's `logprob` field
 * is interpreted as the log-probability assigned to the expected token). */
OcError oc_validation_perplexity(const OcValidationState *state,
                                double *out_perplexity);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_VALIDATION_H */
