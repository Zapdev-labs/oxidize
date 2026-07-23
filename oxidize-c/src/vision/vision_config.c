/*
 * vision_config.c — Vision model configuration implementation.
 */
#include "oxidize/vision_config.h"

#include <string.h>

OcError oc_vision_cfg_init(OcVisionConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->model_type = OC_VISION_MODEL_CLIP;
    cfg->image_size = 224;
    cfg->patch_size = 16;
    cfg->n_channels = 3;
    cfg->n_layers = 12;
    cfg->hidden_dim = 768;
    cfg->n_heads = 12;
    cfg->mlp_ratio = 4;
    cfg->n_positions = 196; /* (224/16)^2 */
    cfg->use_abs_pos_emb = true;
    cfg->use_rotary_emb = false;
    cfg->use_swin = false;
    cfg->layer_norm_eps = 1e-5f;
    cfg->projection_dim = 512;
    return OC_OK;
}

OcError oc_vision_cfg_clip_base(OcVisionConfig *cfg)
{
    OcError e = oc_vision_cfg_init(cfg);
    if (e != OC_OK) return e;
    cfg->model_type = OC_VISION_MODEL_CLIP;
    cfg->image_size = 224;
    cfg->patch_size = 16;
    cfg->n_layers = 12;
    cfg->hidden_dim = 768;
    cfg->n_heads = 12;
    cfg->projection_dim = 512;
    return OC_OK;
}

OcError oc_vision_cfg_clip_large(OcVisionConfig *cfg)
{
    OcError e = oc_vision_cfg_init(cfg);
    if (e != OC_OK) return e;
    cfg->model_type = OC_VISION_MODEL_CLIP;
    cfg->image_size = 336;
    cfg->patch_size = 14;
    cfg->n_layers = 24;
    cfg->hidden_dim = 1024;
    cfg->n_heads = 16;
    cfg->projection_dim = 768;
    return OC_OK;
}

OcError oc_vision_cfg_siglip(OcVisionConfig *cfg)
{
    OcError e = oc_vision_cfg_init(cfg);
    if (e != OC_OK) return e;
    cfg->model_type = OC_VISION_MODEL_SIGLIP;
    cfg->image_size = 384;
    cfg->patch_size = 16;
    cfg->n_layers = 27;
    cfg->hidden_dim = 1152;
    cfg->n_heads = 16;
    cfg->use_abs_pos_emb = false;
    cfg->use_rotary_emb = true;
    cfg->projection_dim = 768;
    return OC_OK;
}

OcError oc_vision_cfg_validate(const OcVisionConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    if (cfg->image_size == 0) return OC_ERR_INVALID_ARG;
    if (cfg->patch_size == 0) return OC_ERR_INVALID_ARG;
    if (cfg->image_size % cfg->patch_size != 0) return OC_ERR_INVALID_ARG;
    if (cfg->n_channels == 0) return OC_ERR_INVALID_ARG;
    if (cfg->n_layers == 0) return OC_ERR_INVALID_ARG;
    if (cfg->hidden_dim == 0) return OC_ERR_INVALID_ARG;
    if (cfg->n_heads == 0) return OC_ERR_INVALID_ARG;
    if (cfg->hidden_dim % cfg->n_heads != 0) return OC_ERR_INVALID_ARG;
    if (cfg->mlp_ratio == 0) return OC_ERR_INVALID_ARG;
    return OC_OK;
}

uint32_t oc_vision_cfg_n_patches(const OcVisionConfig *cfg)
{
    if (!cfg || cfg->patch_size == 0) return 0;
    return (cfg->image_size / cfg->patch_size) *
           (cfg->image_size / cfg->patch_size);
}

uint32_t oc_vision_cfg_n_patches_total(const OcVisionConfig *cfg)
{
    uint32_t n = oc_vision_cfg_n_patches(cfg);
    return n + 1; /* +1 for CLS token */
}

uint32_t oc_vision_cfg_patch_dim(const OcVisionConfig *cfg)
{
    if (!cfg) return 0;
    return cfg->patch_size * cfg->patch_size * cfg->n_channels;
}

const char *oc_vision_model_type_name(OcVisionModelType type)
{
    switch (type) {
    case OC_VISION_MODEL_CLIP:      return "clip";
    case OC_VISION_MODEL_SIGLIP:    return "siglip";
    case OC_VISION_MODEL_DINO:      return "dino";
    case OC_VISION_MODEL_INTERN_VIT: return "intern_vit";
    default: return "unknown";
    }
}

bool oc_vision_cfg_is_valid(const OcVisionConfig *cfg)
{
    return oc_vision_cfg_validate(cfg) == OC_OK;
}
