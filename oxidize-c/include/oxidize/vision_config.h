/*
 * vision_config.h — Vision model configuration.
 *
 * Configuration for CLIP-style vision encoders and multimodal projections.
 * Port from oxidize-core/src/vision/config.rs.
 */
#ifndef OXIDIZE_VISION_CONFIG_H
#define OXIDIZE_VISION_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_VISION_MODEL_CLIP = 0,
    OC_VISION_MODEL_SIGLIP = 1,
    OC_VISION_MODEL_DINO = 2,
    OC_VISION_MODEL_INTERN_VIT = 3,
} OcVisionModelType;

typedef struct {
    OcVisionModelType model_type;
    uint32_t image_size;
    uint32_t patch_size;
    uint32_t n_channels;
    uint32_t n_layers;
    uint32_t hidden_dim;
    uint32_t n_heads;
    uint32_t mlp_ratio;
    uint32_t n_positions;
    bool use_abs_pos_emb;
    bool use_rotary_emb;
    bool use_swin;
    float layer_norm_eps;
    uint32_t projection_dim;
} OcVisionConfig;

OcError oc_vision_cfg_init(OcVisionConfig *cfg);
OcError oc_vision_cfg_clip_base(OcVisionConfig *cfg);
OcError oc_vision_cfg_clip_large(OcVisionConfig *cfg);
OcError oc_vision_cfg_siglip(OcVisionConfig *cfg);
OcError oc_vision_cfg_validate(const OcVisionConfig *cfg);
uint32_t oc_vision_cfg_n_patches(const OcVisionConfig *cfg);
uint32_t oc_vision_cfg_n_patches_total(const OcVisionConfig *cfg);
uint32_t oc_vision_cfg_patch_dim(const OcVisionConfig *cfg);
const char *oc_vision_model_type_name(OcVisionModelType type);
bool oc_vision_cfg_is_valid(const OcVisionConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_VISION_CONFIG_H */
