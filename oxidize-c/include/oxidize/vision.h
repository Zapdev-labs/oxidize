/*
 * vision.h — CLIP-style vision encoder for multimodal inference.
 *
 * Provides image encoding via a simplified CLIP-style vision transformer
 * (ViT): patch embedding via linear projection, positional embeddings,
 * optional layer norm, and mean pooling. Produces image embeddings that
 * can be injected into the LLM's embedding space.
 *
 * Weight matrices (patch_proj, pos_emb, ln_w, ln_b) are set from a GGUF
 * vision model. When weights are absent, the encoder returns mean-pooled
 * patch statistics as a fallback.
 */
#ifndef OXIDIZE_VISION_H
#define OXIDIZE_VISION_H

#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Supported image formats. */
typedef enum {
    OC_IMAGE_RGB  = 0,
    OC_IMAGE_RGBA = 1,
    OC_IMAGE_GRAY = 2,
} OcImageFormat;

/* Image descriptor. */
typedef struct OcImage {
    const uint8_t *data;     /* raw pixel data                      */
    uint32_t width;
    uint32_t height;
    uint32_t channels;        /* 1=gray, 3=RGB, 4=RGBA                */
    OcImageFormat format;
} OcImage;

/* Vision encoder config (CLIP-style). */
typedef struct OcVisionConfig {
    uint32_t image_size;      /* input resolution (e.g. 224, 336)      */
    uint32_t patch_size;      /* ViT patch size (e.g. 14, 16)          */
    uint32_t hidden_size;     /* embedding dimension                   */
    uint32_t n_layers;       /* ViT depth                             */
    uint32_t n_heads;        /* attention heads                       */
    uint32_t n_patches;      /* (image_size / patch_size)^2           */
} OcVisionConfig;

/* Vision encoder state. */
typedef struct OcVisionEncoder {
    OcVisionConfig config;
    /* Weight matrices (all optional; if NULL, uses fallback). */
    const float *patch_proj;     /* [hidden_size, patch_size^2 * channels] */
    const float *pos_emb;       /* [n_patches, hidden_size] (or NULL)      */
    const float *ln_weight;     /* [hidden_size] (or NULL)                  */
    const float *ln_bias;       /* [hidden_size] (or NULL)                  */
    const float *cls_emb;       /* [hidden_size] (optional CLS token)       */
    bool initialized;
} OcVisionEncoder;

/* Initialize the vision encoder with a CLIP model from a GGUF. */
OcError oc_vision_init(OcVisionEncoder *enc, const OcVisionConfig *cfg);

/* Set weight matrices (non-owning; caller must keep them alive). */
OcError oc_vision_set_weights(OcVisionEncoder *enc,
                               const float *patch_proj,
                               const float *pos_emb,
                               const float *ln_weight,
                               const float *ln_bias,
                               const float *cls_emb);

/* Encode an image into a flat embedding vector.
 * `out_embeddings` receives `n_patches * hidden_size` floats.
 * `out_len` receives the number of floats written. */
OcError oc_vision_encode(OcVisionEncoder *enc, const OcImage *img,
                         float *out_embeddings, size_t *out_len);

/* Preprocess: resize an image to the encoder's input size. */
OcError oc_vision_resize(const OcImage *src, uint32_t target_w,
                          uint32_t target_h, uint8_t *out);

/* Preprocess: normalize pixel values to [-1, 1] range. */
OcError oc_vision_normalize(const OcImage *img, float *out);

/* Free the vision encoder. */
void oc_vision_free(OcVisionEncoder *enc);

/* Multimodal prompt: combines text tokens with image embeddings. */
typedef struct OcMultimodalPrompt {
    uint32_t *text_tokens;
    size_t n_text_tokens;
    float *image_embeddings;  /* [n_patches * hidden_size]            */
    size_t n_image_embeddings;
    uint32_t image_token_id;  /* special token marking image position  */
} OcMultimodalPrompt;

/* Create a multimodal prompt by inserting image embeddings at the
 * image_token_id position in the token sequence. */
OcError oc_multimodal_create(const uint32_t *text_tokens, size_t n_text,
                              float *image_embeddings, size_t n_img_emb,
                              uint32_t image_token_id,
                              OcMultimodalPrompt *out);

void oc_multimodal_free(OcMultimodalPrompt *prompt);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_VISION_H */
