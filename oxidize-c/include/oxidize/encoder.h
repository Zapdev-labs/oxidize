/* encoder.h — Vision encoder pipeline integration. */
#ifndef OXIDIZE_ENCODER_H
#define OXIDIZE_ENCODER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/vision_config.h"
#include "oxidize/vision_preprocess.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct OcEncoderPipeline {
    OcVisionConfig      config;
    OcPreprocessConfig  preprocessor;
    bool                initialized;
} OcEncoderPipeline;


/* Initialize a pipeline with the given vision config (or CLIP base defaults
 * if NULL). */
OcError oc_encoder_pipeline_init(OcEncoderPipeline *pipe,
                                  const OcVisionConfig *vcfg);

/* Full pipeline: preprocess -> encode -> output features.
 * Writes up to max_features floats into out_features and sets *out_n.
 * Returns OC_ERR_INVALID_ARG on bad args, OC_ERR_OOM if buffer too small. */
OcError oc_encoder_pipeline_process(OcEncoderPipeline *pipe,
                                     const OcImage *image,
                                     float *out_features,
                                     size_t max_features,
                                     size_t *out_n);

/* Batch processing: process n_images images, writing results contiguously */
OcError oc_encoder_pipeline_process_batch(OcEncoderPipeline *pipe,
                                           const OcImage *images,
                                           uint32_t n_images,
                                           float *out_features,
                                           size_t max_features,
                                           size_t *out_n);

/* Get the expected output feature count for a single image. */
size_t oc_encoder_pipeline_n_output_features(const OcEncoderPipeline *pipe);

/* Free the pipeline. Safe on NULL. */
void oc_encoder_pipeline_free(OcEncoderPipeline *pipe);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ENCODER_H */
