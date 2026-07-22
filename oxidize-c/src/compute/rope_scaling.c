/*
 * rope_scaling.c — RoPE position scaling implementation.
 *
 * Implements linear, NTK-aware, YaRN, and dynamic NTK methods.
 */
#include "oxidize/rope_scaling.h"

#include <math.h>
#include <string.h>

/* ─── API ──────────────────────────────────────────────────────────────── */

OcError oc_rope_config_init(OcRopeScalingConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->type = OC_ROPE_NONE;
    cfg->scale_factor = 4.0f;
    cfg->original_max_pos = 4096;
    cfg->extended_max_pos = 16384;
    cfg->beta_fast = 32.0f;
    cfg->beta_slow = 1.0f;
    cfg->attention_factor = 1.0f;
    cfg->mscale = 0.0f;
    cfg->mscale_all_dim = 0.0f;
    return OC_OK;
}

float oc_rope_scale_factor(const OcRopeScalingConfig *cfg, uint32_t pos)
{
    if (!cfg) return 1.0f;
    switch (cfg->type) {
    case OC_ROPE_LINEAR:
        return cfg->scale_factor;
    case OC_ROPE_DYNAMIC_NTK:
        if (pos <= cfg->original_max_pos)
            return 1.0f;
        return cfg->scale_factor * (float)cfg->original_max_pos /
               (float)(pos + 1);
    case OC_ROPE_NTK:
    case OC_ROPE_YARN:
        return cfg->scale_factor;
    default:
        return 1.0f;
    }
}

float oc_rope_apply_linear(float freq, uint32_t pos, float scale)
{
    (void)freq;
    return (float)pos / scale;
}

float oc_rope_apply_ntk(float freq, uint32_t pos, float scale,
                         uint32_t dim, uint32_t max_pos)
{
    (void)pos;
    (void)max_pos;
    /* NTK modifies the base frequency: theta' = theta * scale^(dim / (dim-2))
     * The frequency becomes freq' = theta'^(-2i/dim) */
    float alpha = powf(scale, (float)dim / (float)(dim > 2 ? dim - 2 : 1));
    return freq / alpha;
}

int oc_rope_yarn_find_correction_dim(int dim, const OcRopeScalingConfig *cfg)
{
    if (!cfg) return 0;
    /* correction_dim = (dim * low_freq_wavelength) / (2 * pi * original_max_pos) */
    float low_freq_wavelength = cfg->original_max_pos /
        (cfg->beta_fast * 2.0f * 3.14159265f);
    return (int)((float)dim * low_freq_wavelength /
                 (2.0f * 3.14159265f * (float)cfg->original_max_pos));
}

void oc_rope_yarn_find_correction_range(int *lo, int *hi,
                                       const OcRopeScalingConfig *cfg)
{
    if (!lo || !hi || !cfg) {
        if (lo) *lo = 0;
        if (hi) *hi = 0;
        return;
    }
    *lo = oc_rope_yarn_find_correction_dim(0, cfg);
    /* For the high end, use beta_slow */
    OcRopeScalingConfig tmp = *cfg;
    tmp.beta_fast = cfg->beta_slow;
    *hi = oc_rope_yarn_find_correction_dim(0, &tmp);
    if (*lo > *hi) {
        int t = *lo;
        *lo = *hi;
        *hi = t;
    }
}

float oc_rope_yarn_linear_ramp_factor(float min_val, float max_val,
                                      const OcRopeScalingConfig *cfg)
{
    (void)cfg;
    if (max_val == min_val) return min_val;
    return (max_val - min_val) / (max_val - min_val);
}

float oc_rope_yarn_mscale(float scale)
{
    if (scale <= 1.0f) return 1.0f;
    return 0.1f * logf(scale) + 1.0f;
}

float oc_rope_apply_yarn(float freq, uint32_t pos,
                         const OcRopeScalingConfig *cfg, uint32_t dim)
{
    if (!cfg) return freq * (float)pos;

    /* YaRN: compute the wavelength of this dimension. */
    float wavelength = 2.0f * 3.14159265f / freq;
    float ratio = wavelength / (float)cfg->original_max_pos;

    if (ratio < cfg->beta_fast / (float)cfg->original_max_pos) {
        /* High-frequency: no interpolation. */
        return freq * (float)pos;
    } else if (ratio > cfg->beta_slow / (float)cfg->original_max_pos) {
        /* Low-frequency: interpolate position. */
        return freq * (float)pos / cfg->scale_factor;
    } else {
        /* Middle: blend. */
        int lo, hi;
        oc_rope_yarn_find_correction_range(&lo, &hi, cfg);
        float blend = oc_rope_yarn_linear_ramp_factor(0.0f, 1.0f, cfg);
        (void)dim;
        return freq * (float)pos * (1.0f - blend) / cfg->scale_factor +
               freq * (float)pos * blend;
    }
}

OcError oc_rope_apply(const OcRopeScalingConfig *cfg, uint32_t pos,
                      uint32_t dim, float base_freq,
                      float *out_cos, float *out_sin)
{
    if (!cfg || !out_cos || !out_sin || dim == 0) return OC_ERR_INVALID_ARG;

    for (uint32_t i = 0; i < dim; i++) {
        float freq = base_freq / powf(10000.0f, (float)(2 * i) / (float)dim);
        float scaled_pos;

        switch (cfg->type) {
        case OC_ROPE_LINEAR:
            scaled_pos = oc_rope_apply_linear(freq, pos, cfg->scale_factor);
            break;
        case OC_ROPE_NTK:
            scaled_pos = oc_rope_apply_ntk(freq, pos, cfg->scale_factor,
                                           dim, cfg->original_max_pos);
            break;
        case OC_ROPE_YARN:
            scaled_pos = oc_rope_apply_yarn(freq, pos, cfg, dim);
            break;
        case OC_ROPE_DYNAMIC_NTK: {
            float scale = oc_rope_scale_factor(cfg, pos);
            scaled_pos = oc_rope_apply_ntk(freq, pos, scale, dim, pos);
            break;
        }
        default:
            scaled_pos = freq * (float)pos;
            break;
        }

        out_cos[i] = cosf(scaled_pos);
        out_sin[i] = sinf(scaled_pos);
    }

    return OC_OK;
}

OcError oc_rope_apply_to_tensor(const OcRopeScalingConfig *cfg, uint32_t pos,
                                float *tensor, uint32_t head_dim,
                                uint32_t n_heads, float base_freq)
{
    if (!cfg || !tensor || head_dim == 0 || n_heads == 0)
        return OC_ERR_INVALID_ARG;

    /* Allocate cos/sin arrays on stack (head_dim is typically 64-128). */
    float cos_buf[256];
    float sin_buf[256];
    if (head_dim > 256) return OC_ERR_INVALID_ARG;

    OcError e = oc_rope_apply(cfg, pos, head_dim, base_freq, cos_buf, sin_buf);
    if (e != OC_OK) return e;

    /* Apply rotation to each head. */
    for (uint32_t h = 0; h < n_heads; h++) {
        float *head = tensor + h * head_dim;
        for (uint32_t i = 0; i < head_dim / 2; i++) {
            float c = cos_buf[i];
            float s = sin_buf[i];
            float x0 = head[i];
            float x1 = head[i + head_dim / 2];
            head[i] = x0 * c - x1 * s;
            head[i + head_dim / 2] = x0 * s + x1 * c;
        }
    }

    return OC_OK;
}

const char *oc_rope_scaling_type_name(OcRopeScalingType type)
{
    switch (type) {
    case OC_ROPE_NONE:        return "none";
    case OC_ROPE_LINEAR:      return "linear";
    case OC_ROPE_NTK:         return "ntk";
    case OC_ROPE_YARN:        return "yarn";
    case OC_ROPE_DYNAMIC_NTK: return "dynamic_ntk";
    default:                  return "unknown";
    }
}
