/*
 * merge.h — checkpoint merging utility.
 *
 * Merges two or more GGUF model checkpoints using various strategies
 * (linear, SLERP, TIES, DARE). Outputs a single merged GGUF file.
 *
 * Port of oxidize-merge/ Rust crate.
 */
#ifndef OXIDIZE_MERGE_H
#define OXIDIZE_MERGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_MERGE_LINEAR  = 0,  /* weighted linear combination             */
    OC_MERGE_SLERP   = 1,  /* spherical linear interpolation          */
    OC_MERGE_TIES    = 2,  /* TIES merging (trim, elect, combine)      */
    OC_MERGE_DARE    = 3,  /* Drop And REscale                        */
} OcMergeStrategy;

typedef struct OcMergeInput {
    const char *path;        /* GGUF file path                          */
    float       weight;      /* blend weight (for LINEAR/SLERP)         */
} OcMergeInput;

typedef struct OcMergeConfig {
    OcMergeStrategy strategy;
    OcMergeInput   *inputs;       /* array of input checkpoints         */
    size_t          n_inputs;     /* number of inputs                    */
    const char     *output_path;  /* output GGUF path                    */
    float           slerp_t;      /* SLERP interpolation parameter        */
    float           ties_density;  /* TIES: fraction of params to keep    */
    bool            verbose;
} OcMergeConfig;

/* Merge multiple GGUF checkpoints into one. */
OcError oc_merge_models(const OcMergeConfig *cfg);

/* Linear merge: out = sum(w_i * model_i) / sum(w_i). */
OcError oc_merge_linear(const OcMergeInput *inputs, size_t n_inputs,
                         const char *output_path);

/* SLERP merge: spherical interpolation between two models. */
OcError oc_merge_slerp(const char *path_a, const char *path_b,
                        float t, const char *output_path);

/* TIES merge: Trim → Elect → Combine. */
OcError oc_merge_ties(const OcMergeInput *inputs, size_t n_inputs,
                       float density, const char *output_path);

/* DARE merge: Drop And REscale. */
OcError oc_merge_dare(const OcMergeInput *inputs, size_t n_inputs,
                      float drop_rate, const char *output_path);

/* Get the name of a merge strategy. */
const char *oc_merge_strategy_name(OcMergeStrategy s);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MERGE_H */
