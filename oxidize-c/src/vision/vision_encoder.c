/*
 * vision_encoder.c — CLIP-style vision encoder (stub).
 *
 * Provides the API surface for image → feature encoding. Real computation
 * is deferred; this module returns deterministic placeholder vectors so
 * callers can wire up the multimodal pipeline end-to-end.
 */
#include "oxidize/vision_encoder.h"

#include <stdlib.h>
#include <string.h>

/* ─── Config helpers ────────────────────────────────────────────────── */

OcError oc_vision_config_init(OcVisionEncoderConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    cfg->image_size  = OC_VISION_DEFAULT_IMAGE_SIZE;
    cfg->patch_size  = OC_VISION_DEFAULT_PATCH_SIZE;
    cfg->n_channels  = OC_VISION_DEFAULT_N_CHANNELS;
    cfg->n_layers    = OC_VISION_DEFAULT_N_LAYERS;
    cfg->hidden_dim  = OC_VISION_DEFAULT_HIDDEN_DIM;
    cfg->n_heads     = OC_VISION_DEFAULT_N_HEADS;
    return OC_OK;
}

/* ─── Encoder lifecycle ─────────────────────────────────────────────── */

OcError oc_vision_encoder_init(OcVisionEncoder **encoder,
                               const OcVisionEncoderConfig *config)
{
    if (!encoder) return OC_ERR_INVALID_ARG;
    *encoder = NULL;

    OcVisionEncoderConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        oc_vision_config_init(&cfg);
    }
    if (cfg.image_size == 0 || cfg.patch_size == 0 || cfg.n_channels == 0
        || cfg.hidden_dim == 0) {
        return OC_ERR_INVALID_ARG;
    }
    /* patch_size must evenly divide image_size for a clean patch grid. */
    if (cfg.image_size % cfg.patch_size != 0) {
        return OC_ERR_INVALID_ARG;
    }
    if (cfg.n_heads == 0) {
        return OC_ERR_INVALID_ARG;
    }

    OcVisionEncoder *e = malloc(sizeof(*e));
    if (!e) return OC_ERR_OOM;
    memset(e, 0, sizeof(*e));
    e->config      = cfg;
    e->weight_data  = NULL;
    e->weight_size  = 0;
    e->initialized  = true;

    *encoder = e;
    return OC_OK;
}

OcError oc_vision_encoder_load_weights(OcVisionEncoder *encoder,
                                       const void *data, size_t size)
{
    if (!encoder) return OC_ERR_INVALID_ARG;
    if (!data && size > 0) return OC_ERR_INVALID_ARG;

    /* Free any previously loaded weights. */
    free(encoder->weight_data);
    encoder->weight_data = NULL;
    encoder->weight_size = 0;

    if (size == 0) {
        return OC_OK;
    }

    void *copy = malloc(size);
    if (!copy) return OC_ERR_OOM;
    memcpy(copy, data, size);
    encoder->weight_data = copy;
    encoder->weight_size = size;
    return OC_OK;
}

/* ─── Encoding ──────────────────────────────────────────────────────── */

/* Compute n_patches = (image_size / patch_size)^2. */
static uint32_t vision_n_patches(const OcVisionEncoderConfig *cfg)
{
    uint32_t per_side = cfg->image_size / cfg->patch_size;
    return per_side * per_side;
}

OcError oc_vision_encoder_encode(OcVisionEncoder *encoder,
                                 const OcImagePatch *image,
                                 float **out_features,
                                 size_t *out_n_features)
{
    if (!encoder || !image || !out_features || !out_n_features) {
        return OC_ERR_INVALID_ARG;
    }
    if (!encoder->initialized) return OC_ERR_INVALID_ARG;
    if (!image->pixels) return OC_ERR_INVALID_ARG;
    if (image->width == 0 || image->height == 0 || image->channels == 0) {
        return OC_ERR_INVALID_ARG;
    }
    *out_features = NULL;
    *out_n_features = 0;

    uint32_t n_patches = vision_n_patches(&encoder->config);
    size_t n = (size_t)n_patches * encoder->config.hidden_dim;
    float *out = malloc(n * sizeof(float));
    if (!out) return OC_ERR_OOM;

    /* Stub: fill with a small deterministic constant derived from image
     * dimensions so different images produce different (but reproducible)
     * features. */
    float seed = (float)((image->width * 31u + image->height) & 0xFF) / 255.0f;
    for (size_t i = 0; i < n; i++) {
        out[i] = seed * 1e-3f;
    }

    *out_features = out;
    *out_n_features = n;
    return OC_OK;
}

OcError oc_vision_encoder_patch_embed(OcVisionEncoder *encoder,
                                     const OcImagePatch *image,
                                     float **out_patches,
                                     size_t *out_n_patches)
{
    if (!encoder || !image || !out_patches || !out_n_patches) {
        return OC_ERR_INVALID_ARG;
    }
    if (!encoder->initialized) return OC_ERR_INVALID_ARG;
    if (!image->pixels) return OC_ERR_INVALID_ARG;
    if (image->width == 0 || image->height == 0 || image->channels == 0) {
        return OC_ERR_INVALID_ARG;
    }
    *out_patches = NULL;
    *out_n_patches = 0;

    uint32_t n_patches = vision_n_patches(&encoder->config);
    size_t patch_dim = (size_t)encoder->config.patch_size
                     * encoder->config.patch_size
                     * encoder->config.n_channels;
    size_t n = (size_t)n_patches * patch_dim;
    float *out = malloc(n * sizeof(float));
    if (!out) return OC_ERR_OOM;

    /* Stub: fill with a small deterministic constant. */
    float seed = (float)((image->width + image->height * 7u) & 0xFF) / 255.0f;
    for (size_t i = 0; i < n; i++) {
        out[i] = seed * 1e-3f;
    }

    *out_patches = out;
    *out_n_patches = n;
    return OC_OK;
}

void oc_vision_encoder_free(OcVisionEncoder *encoder)
{
    if (!encoder) return;
    free(encoder->weight_data);
    memset(encoder, 0, sizeof(*encoder));
    free(encoder);
}

/* ─── Image patch helpers ───────────────────────────────────────────── */

OcError oc_image_patch_init(uint32_t width, uint32_t height,
                            uint32_t channels, OcImagePatch **out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    *out = NULL;
    if (width == 0 || height == 0 || channels == 0) return OC_ERR_INVALID_ARG;

    OcImagePatch *p = malloc(sizeof(*p));
    if (!p) return OC_ERR_OOM;
    size_t n = (size_t)width * height * channels;
    p->pixels = calloc(n, sizeof(float));
    if (!p->pixels) { free(p); return OC_ERR_OOM; }
    p->width    = width;
    p->height   = height;
    p->channels = channels;
    *out = p;
    return OC_OK;
}

void oc_image_patch_free(OcImagePatch *patch)
{
    if (!patch) return;
    free(patch->pixels);
    memset(patch, 0, sizeof(*patch));
    free(patch);
}
