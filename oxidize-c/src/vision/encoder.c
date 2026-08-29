/* encoder.c — Vision encoder pipeline integration. */
#include "oxidize/encoder.h"

#include <stdlib.h>
#include <string.h>


/* Compute the number of patches: (image_size / patch_size)^2. */
static uint32_t pipeline_n_patches(const OcVisionConfig *cfg)
{
    if (cfg->patch_size == 0) return 0;
    uint32_t per_side = cfg->image_size / cfg->patch_size;
    return per_side * per_side;
}

/* Compute expected output features = n_patches * projection_dim.
 * Falls back to hidden_dim if projection_dim is 0. */
static size_t pipeline_output_size(const OcVisionConfig *cfg)
{
    uint32_t n_patches = pipeline_n_patches(cfg);
    uint32_t dim = cfg->projection_dim > 0 ? cfg->projection_dim
                                            : cfg->hidden_dim;
    return (size_t)n_patches * dim;
}


OcError oc_encoder_pipeline_init(OcEncoderPipeline *pipe,
                                  const OcVisionConfig *vcfg)
{
    if (!pipe) return OC_ERR_INVALID_ARG;
    memset(pipe, 0, sizeof(*pipe));

    if (vcfg) {
        pipe->config = *vcfg;
    } else {
        OcError e = oc_vision_cfg_init(&pipe->config);
        if (e != OC_OK) return e;
    }

    OcError e = oc_preprocess_config_init(&pipe->preprocessor);
    if (e != OC_OK) return e;

    pipe->initialized = true;
    return OC_OK;
}


OcError oc_encoder_pipeline_process(OcEncoderPipeline *pipe,
                                     const OcImage *image,
                                     float *out_features,
                                     size_t max_features,
                                     size_t *out_n)
{
    if (!pipe || !image || !out_features || !out_n) {
        return OC_ERR_INVALID_ARG;
    }
    if (!pipe->initialized) return OC_ERR_INVALID_ARG;
    if (!image->pixels) return OC_ERR_INVALID_ARG;
    if (image->width == 0 || image->height == 0) {
        return OC_ERR_INVALID_ARG;
    }
    *out_n = 0;

    size_t expected = pipeline_output_size(&pipe->config);
    if (max_features < expected) return OC_ERR_OOM;

    /* Run preprocessing to get normalized floats (we use a temporary
     * buffer, then discard — the real encoder would consume them).
     * The preprocessor resizes to target_size x target_size. */
    size_t preproc_size = (size_t)pipe->preprocessor.target_size
                        * pipe->preprocessor.target_size
                        * pipe->config.n_channels;
    float *preproc_buf = malloc(preproc_size * sizeof(float));
    if (!preproc_buf) return OC_ERR_OOM;

    OcError e = oc_preprocess_full(image, preproc_buf, &pipe->preprocessor);
    if (e != OC_OK) {
        free(preproc_buf);
        return e;
    }

    /* Stub encoding: fill output with a small deterministic constant
     * derived from image dimensions so different images produce different
     * (but reproducible) features. */
    float seed = (float)((image->width * 31u + image->height) & 0xFF) / 255.0f;
    for (size_t i = 0; i < expected; i++) {
        out_features[i] = seed * 1e-3f;
    }

    free(preproc_buf);
    *out_n = expected;
    return OC_OK;
}

OcError oc_encoder_pipeline_process_batch(OcEncoderPipeline *pipe,
                                           const OcImage *images,
                                           uint32_t n_images,
                                           float *out_features,
                                           size_t max_features,
                                           size_t *out_n)
{
    if (!pipe || !images || !out_features || !out_n) {
        return OC_ERR_INVALID_ARG;
    }
    if (n_images == 0) return OC_ERR_INVALID_ARG;
    if (!pipe->initialized) return OC_ERR_INVALID_ARG;
    *out_n = 0;

    size_t per_image = pipeline_output_size(&pipe->config);
    size_t total = per_image * n_images;
    if (max_features < total) return OC_ERR_OOM;

    size_t written = 0;
    for (uint32_t i = 0; i < n_images; i++) {
        size_t n = 0;
        OcError e = oc_encoder_pipeline_process(pipe, &images[i],
                                                out_features + written,
                                                per_image, &n);
        if (e != OC_OK) return e;
        written += n;
    }
    *out_n = written;
    return OC_OK;
}


size_t oc_encoder_pipeline_n_output_features(const OcEncoderPipeline *pipe)
{
    if (!pipe || !pipe->initialized) return 0;
    return pipeline_output_size(&pipe->config);
}


void oc_encoder_pipeline_free(OcEncoderPipeline *pipe)
{
    if (!pipe) return;
    memset(pipe, 0, sizeof(*pipe));
}
