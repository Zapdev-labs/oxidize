/*
 * vision_encoder.h — CLIP-style vision encoder for multimodal models.
 *
 * Stub implementation providing the API surface for image → feature encoding
 * via a CLIP-style ViT (patch embedding → optional transformer → projection).
 * Real computation is deferred; this module returns deterministic placeholders
 * so callers can wire up the multimodal pipeline end-to-end.
 *
 * Port of the conceptual API from oxidize-core/src/vision/encoder.rs.
 */
#ifndef OXIDIZE_VISION_ENCODER_H
#define OXIDIZE_VISION_ENCODER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ──────────────────────────────────────────────────────── */

#define OC_VISION_DEFAULT_IMAGE_SIZE   224u
#define OC_VISION_DEFAULT_PATCH_SIZE   16u
#define OC_VISION_DEFAULT_N_CHANNELS   3u
#define OC_VISION_DEFAULT_N_LAYERS     12u
#define OC_VISION_DEFAULT_HIDDEN_DIM   768u
#define OC_VISION_DEFAULT_N_HEADS      12u

/* ─── Config ────────────────────────────────────────────────────────── */

typedef struct OcVisionEncoderConfig {
    uint32_t image_size;   /* default OC_VISION_DEFAULT_IMAGE_SIZE */
    uint32_t patch_size;   /* default OC_VISION_DEFAULT_PATCH_SIZE */
    uint32_t n_channels;  /* default OC_VISION_DEFAULT_N_CHANNELS */
    uint32_t n_layers;    /* default OC_VISION_DEFAULT_N_LAYERS   */
    uint32_t hidden_dim;  /* default OC_VISION_DEFAULT_HIDDEN_DIM */
    uint32_t n_heads;     /* default OC_VISION_DEFAULT_N_HEADS    */
} OcVisionEncoderConfig;

/* ─── Image patch ───────────────────────────────────────────────────── */

typedef struct OcImagePatch {
    float   *pixels;     /* owned, size = width * height * channels */
    uint32_t width;
    uint32_t height;
    uint32_t channels;
} OcImagePatch;

/* ─── Encoder ────────────────────────────────────────────────────────── */

typedef struct OcVisionEncoder {
    OcVisionEncoderConfig config;
    void     *weight_data;     /* owned copy of caller-supplied weights */
    size_t    weight_size;     /* bytes                                            */
    bool      initialized;
} OcVisionEncoder;

/* ─── Config helpers ────────────────────────────────────────────────── */

/* Initialize config with defaults. */
OcError oc_vision_config_init(OcVisionEncoderConfig *cfg);

/* ─── Encoder lifecycle ─────────────────────────────────────────────── */

/* Allocate an encoder (heap-allocated) and store the config.
 * Free with oc_vision_encoder_free. */
OcError oc_vision_encoder_init(OcVisionEncoder **encoder,
                               const OcVisionEncoderConfig *config);

/* Load weights into the encoder. `data` is copied (the encoder owns the
 * copy). If weights were previously loaded, they are freed and replaced. */
OcError oc_vision_encoder_load_weights(OcVisionEncoder *encoder,
                                       const void *data, size_t size);

/* Encode an image patch to a feature vector. The stub returns a
 * deterministic vector of size n_patches * hidden_dim filled with a small
 * constant. `*out_features` is heap-allocated and owned by the caller.
 * `*out_n_features` is the number of floats written. */
OcError oc_vision_encoder_encode(OcVisionEncoder *encoder,
                                 const OcImagePatch *image,
                                 float **out_features,
                                 size_t *out_n_features);

/* Extract patches from an image patch grid. The stub returns a deterministic
 * vector of size n_patches * (patch_size^2 * channels) filled with a small
 * constant. `*out_patches` is heap-allocated and owned by the caller.
 * `*out_n_patches` is the number of patches. */
OcError oc_vision_encoder_patch_embed(OcVisionEncoder *encoder,
                                     const OcImagePatch *image,
                                     float **out_patches,
                                     size_t *out_n_patches);

/* Free the encoder and its owned weights. Safe on NULL / already-freed. */
void oc_vision_encoder_free(OcVisionEncoder *encoder);

/* ─── Image patch helpers ───────────────────────────────────────────── */

/* Allocate an image patch of the given dimensions with zeroed pixel data.
 * Free with oc_image_patch_free. */
OcError oc_image_patch_init(uint32_t width, uint32_t height,
                            uint32_t channels, OcImagePatch **out);

/* Free an image patch owned by the caller. Safe on NULL / already-freed. */
void oc_image_patch_free(OcImagePatch *patch);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_VISION_ENCODER_H */
