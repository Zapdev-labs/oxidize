/* vision_encoder.h — CLIP-style vision encoder for multimodal models. */
#ifndef OXIDIZE_VISION_ENCODER_H
#define OXIDIZE_VISION_ENCODER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OC_VISION_DEFAULT_IMAGE_SIZE   224u
#define OC_VISION_DEFAULT_PATCH_SIZE   16u
#define OC_VISION_DEFAULT_N_CHANNELS   3u
#define OC_VISION_DEFAULT_N_LAYERS     12u
#define OC_VISION_DEFAULT_HIDDEN_DIM   768u
#define OC_VISION_DEFAULT_N_HEADS      12u


typedef struct OcVisionEncoderConfig {
    uint32_t image_size;   /* default OC_VISION_DEFAULT_IMAGE_SIZE */
    uint32_t patch_size;   /* default OC_VISION_DEFAULT_PATCH_SIZE */
    uint32_t n_channels;  /* default OC_VISION_DEFAULT_N_CHANNELS */
    uint32_t n_layers;    /* default OC_VISION_DEFAULT_N_LAYERS   */
    uint32_t hidden_dim;  /* default OC_VISION_DEFAULT_HIDDEN_DIM */
    uint32_t n_heads;     /* default OC_VISION_DEFAULT_N_HEADS    */
} OcVisionEncoderConfig;


typedef struct OcImagePatch {
    float   *pixels;     /* owned, size = width * height * channels */
    uint32_t width;
    uint32_t height;
    uint32_t channels;
} OcImagePatch;


/* Per-layer weights for ViT transformer blocks. */
typedef struct OcViTLayer {
    float *layer_norm1;        /* [hidden_dim] */
    float *q_proj;             /* [hidden_dim, hidden_dim] */
    float *k_proj;             /* [hidden_dim, hidden_dim] */
    float *v_proj;             /* [hidden_dim, hidden_dim] */
    float *out_proj;           /* [hidden_dim, hidden_dim] */
    float *layer_norm2;        /* [hidden_dim] */
    float *mlp_fc1;            /* [4*hidden_dim, hidden_dim] */
    float *mlp_fc2;            /* [hidden_dim, 4*hidden_dim] */
} OcViTLayer;

typedef struct OcVisionEncoder {
    OcVisionEncoderConfig config;
    void     *weight_data;     /* owned copy of caller-supplied weights */
    size_t    weight_size;     /* bytes                                            */
    /* ViT weights (NULL until loaded). */
    float    *patch_embed_weight;  /* [hidden_dim, patch_dim] */
    float    *patch_embed_bias;    /* [hidden_dim] */
    float    *cls_token;           /* [hidden_dim] */
    float    *pos_embed;           /* [1+n_patches, hidden_dim] */
    float    *final_norm;          /* [hidden_dim] */
    float    *proj;               /* [proj_dim, hidden_dim] (optional) */
    OcViTLayer *layers;            /* [n_layers] */
    bool      initialized;
} OcVisionEncoder;


/* Initialize config with defaults. */
OcError oc_vision_config_init(OcVisionEncoderConfig *cfg);


/* Allocate an encoder (heap-allocated) and store the config.
 * Free with oc_vision_encoder_free. */
OcError oc_vision_encoder_init(OcVisionEncoder **encoder,
                               const OcVisionEncoderConfig *config);

/* Load weights into the encoder. `data` is copied (the encoder owns the
 * copy). If weights were previously loaded, they are freed and replaced. */
OcError oc_vision_encoder_load_weights(OcVisionEncoder *encoder,
                                       const void *data, size_t size);

/* Encode an image to a feature vector. Without weights this is a stub: a deterministic placeholder of size n_patches * hidden_dim. `*out_features` is heap-allocated and caller-owned; `*out_n_features` is the number of floats written. */
OcError oc_vision_encoder_encode(OcVisionEncoder *encoder,
                                 const OcImagePatch *image,
                                 float **out_features,
                                 size_t *out_n_features);

/* Extract patches from an image patch grid. The stub returns a deterministic vector of size n_patches * (patch_size^2 * channels) filled with a small constant. `*out_patches` is heap-allocated and owned by the caller. `*out_n_patches` is the number of patches. */
OcError oc_vision_encoder_patch_embed(OcVisionEncoder *encoder,
                                     const OcImagePatch *image,
                                     float **out_patches,
                                     size_t *out_n_patches);

/* Free the encoder and its owned weights. Safe on NULL / already-freed. */
void oc_vision_encoder_free(OcVisionEncoder *encoder);


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
