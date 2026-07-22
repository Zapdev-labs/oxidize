/*
 * multimodal_proj.h — Projection layer for multimodal inference.
 *
 * Maps vision/audio/video encoder outputs into the LLM embedding space,
 * enabling multimodal inference (vision-language, audio-language, etc.).
 *
 * Architecture: a small MLP (default 2 linear layers) with configurable
 * activation (GELU/ReLU/SiLU) between layers. Weights are loaded from a
 * flat buffer (e.g. GGUF tensor data) or set per-layer.
 *
 * Port of the conceptual oxidize-core/src/vision/multimodal_proj path.
 */
#ifndef OXIDIZE_MULTIMODAL_PROJ_H
#define OXIDIZE_MULTIMODAL_PROJ_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Modalities ────────────────────────────────────────────────────────── */

typedef enum {
    OC_MODALITY_VISION = 0,
    OC_MODALITY_AUDIO  = 1,
    OC_MODALITY_VIDEO  = 2,
    OC_MODALITY_TEXT   = 3,
} OcMultimodalModality;

/* ─── Activation types ──────────────────────────────────────────────────── */

typedef enum {
    OC_MM_ACT_GELU = 0, /* exact GeLU (erf-based)                      */
    OC_MM_ACT_RELU = 1, /* ReLU: max(0, x)                            */
    OC_MM_ACT_SILU = 2, /* SiLU: x * sigmoid(x)                       */
} OcMultimodalActivation;

/* ─── Config ──────────────────────────────────────────────────────────────
 *
 * Defaults: n_layers=2, hidden_dim=0 (=> input_dim), activation=GELU.
 */
typedef struct OcMultimodalProjectionConfig {
    uint32_t input_dim;     /* encoder output dim (e.g. 1024)           */
    uint32_t output_dim;    /* LLM embedding dim (e.g. 4096)            */
    OcMultimodalModality modality;
    uint32_t n_layers;      /* default 2 (min 1)                        */
    uint32_t hidden_dim;   /* intermediate dim; 0 => input_dim         */
    OcMultimodalActivation activation;
} OcMultimodalProjectionConfig;

#define OC_MM_PROJ_CONFIG_DEFAULT ((OcMultimodalProjectionConfig){ \
    .input_dim = 0, \
    .output_dim = 0, \
    .modality = OC_MODALITY_VISION, \
    .n_layers = 2, \
    .hidden_dim = 0, \
    .activation = OC_MM_ACT_GELU, })

/* ─── Projection state ───────────────────────────────────────────────────
 *
 * `weights` stores, for each layer l (0-indexed), a weight matrix
 * [out_dim_l x in_dim_l] and a bias vector [out_dim_l]. The matrices are
 * owned (malloc'd) and freed by oc_mm_proj_free().
 */
typedef struct OcMultimodalProjection {
    OcMultimodalProjectionConfig config;
    float **weights;    /* [n_layers] matrices, row-major [out x in]    */
    float **biases;     /* [n_layers] vectors, length out_dim_l         */
    uint32_t *in_dims;  /* [n_layers] input dim per layer               */
    uint32_t *out_dims; /* [n_layers] output dim per layer              */
    uint32_t n_weights; /* == config.n_layers                            */
    bool initialized;
} OcMultimodalProjection;

/* ─── Lifecycle ─────────────────────────────────────────────────────────── */

/* Allocate a projection from a config. Computes per-layer dims:
 *   layer 0:  in = input_dim,  out = (n_layers > 1) ? hidden_dim : output_dim
 *   layer k:  in = prev_out,   out = (k == n_layers-1) ? output_dim : hidden_dim
 * If hidden_dim == 0, hidden_dim defaults to input_dim.
 * Returns NULL on OOM or bad config. */
OcMultimodalProjection *oc_mm_proj_init(const OcMultimodalProjectionConfig *config);

/* Free a projection and all owned weights/biases. Safe on NULL. */
void oc_mm_proj_free(OcMultimodalProjection *proj);

/* ─── Weight loading ────────────────────────────────────────────────────── */

/* Load all weights from a flat buffer. The buffer must contain, for each
 * layer, the weight matrix (out_dim * in_dim floats, row-major) followed by
 * the bias (out_dim floats). Returns OC_OK on success,
 * OC_ERR_INVALID_ARG if the buffer is too small. */
OcError oc_mm_proj_load_weights(OcMultimodalProjection *proj,
                                  const float *data, size_t data_size);

/* Set a specific layer's weight matrix. `n_elements` must equal
 * out_dims[l] * in_dims[l]. The data is copied (caller may free the input).
 * Returns OC_OK on success. */
OcError oc_mm_proj_set_layer_weight(OcMultimodalProjection *proj,
                                      uint32_t layer_idx,
                                      const float *weight_data,
                                      size_t n_elements);

/* Set a specific layer's bias vector. `n_elements` must equal out_dims[l].
 * The data is copied. Returns OC_OK on success. */
OcError oc_mm_proj_set_layer_bias(OcMultimodalProjection *proj,
                                    uint32_t layer_idx,
                                    const float *bias_data,
                                    size_t n_elements);

/* ─── Forward pass ────────────────────────────────────────────────────────
 *
 * Project `n_tokens` input embeddings (each `input_dim` floats) into the
 * output space. Returns a malloc'd buffer of size n_tokens * output_dim
 * floats, or NULL on error. The caller must free the returned buffer.
 *
 * Applies linear layers with the configured activation between layers
 * (no activation after the final layer).
 */
float *oc_mm_proj_forward(OcMultimodalProjection *proj,
                            const float *input, size_t n_tokens);

/* ─── Prompt concatenation ────────────────────────────────────────────────
 *
 * Concatenate text embeddings (n_text * output_dim) and multimodal embeddings
 * (n_mm * output_dim) into a single buffer. Returns a malloc'd buffer of
 * size (n_text + n_mm) * output_dim floats, or NULL on error.
 * `text_embeds` and `mm_embeds` must each have output_dim columns.
 */
float *oc_mm_proj_concat_prompt(OcMultimodalProjection *proj,
                                  const float *text_embeds, size_t n_text,
                                  const float *mm_embeds, size_t n_mm);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MULTIMODAL_PROJ_H */
