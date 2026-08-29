/* rope_scaling.c — RoPE position scaling implementation. */
#include "oxidize/rope_scaling.h"

#include <math.h>
#include <string.h>


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
    /* Matches Rust: n_dims * ln(orig_ctx / (n_rot * 2*PI)) / (2 * ln(base))
     * where n_rot = beta_fast or beta_slow, base = 10000 (theta). */
    float n_rot = cfg->beta_fast;
    float base = 10000.0f;
    float ratio = (float)cfg->original_max_pos / (n_rot * 2.0f * 3.14159265f);
    if (ratio <= 0.0f) return 0;
    return (int)((float)dim * logf(ratio) / (2.0f * logf(base)));
}

void oc_rope_yarn_find_correction_range(int *lo, int *hi,
                                       const OcRopeScalingConfig *cfg)
{
    if (!lo || !hi || !cfg) {
        if (lo) *lo = 0;
        if (hi) *hi = 0;
        return;
    }
    OcRopeScalingConfig tmp = *cfg;
    tmp.beta_fast = cfg->beta_fast;  /* beta_fast=32 → low-freq dim */
    *lo = oc_rope_yarn_find_correction_dim(0, cfg);
    tmp.beta_fast = cfg->beta_slow;  /* beta_slow=1 → high-freq dim */
    *hi = oc_rope_yarn_find_correction_dim(0, &tmp);
    if (*lo > *hi) {
        int t = *lo;
        *lo = *hi;
        *hi = t;
    }
    /* Clamp to valid range [0, dim-1]. dim is not known here, so clamp >= 0. */
    if (*lo < 0) *lo = 0;
}

float oc_rope_yarn_linear_ramp_factor(float min_val, float max_val,
                                      const OcRopeScalingConfig *cfg)
{
    (void)cfg;
    /* Rust uses: ramp = 1.0 - (i - corr_lo) / (corr_hi - corr_lo), clamped [0,1].
     * This function computes the denominator-independent ramp factor.
     * The actual ramp is computed inline in oc_rope_apply. */
    if (max_val <= min_val) return 0.0f;
    return 1.0f;
}

float oc_rope_yarn_mscale(float scale)
{
    if (scale <= 1.0f) return 1.0f;
    return 0.1f * logf(scale) + 1.0f;
}

float oc_rope_yarn_mscale_m(float scale, float m)
{
    if (scale <= 1.0f) return 1.0f;
    return 0.1f * m * logf(scale) + 1.0f;
}

void oc_rope_deepseek_yarn_scales(float scale_factor, float mscale,
                                  float mscale_all_dim, uint32_t head_dim,
                                  float *rope_attn_factor,
                                  float *softmax_scale)
{
    float m_rope = oc_rope_yarn_mscale_m(scale_factor, mscale);
    float m_all  = oc_rope_yarn_mscale_m(scale_factor, mscale_all_dim);

    if (rope_attn_factor) {
        *rope_attn_factor = (m_all != 0.0f) ? (m_rope / m_all) : 1.0f;
    }
    if (softmax_scale) {
        float base = (head_dim > 0) ? (1.0f / sqrtf((float)head_dim)) : 1.0f;
        *softmax_scale = base * m_all * m_all;
    }
}

float oc_rope_apply_yarn(float freq, uint32_t pos,
                         const OcRopeScalingConfig *cfg, uint32_t dim)
{
    if (!cfg) return freq * (float)pos;
    /* YaRN: blend between extrapolated (high-freq) and interpolated (low-freq)
     * position scaling using a linear ramp. */
    float freq_scale = 1.0f / cfg->scale_factor;
    float mscale = oc_rope_yarn_mscale(cfg->scale_factor);

    /* Compute correction range. */
    int corr_lo, corr_hi;
    corr_lo = oc_rope_yarn_find_correction_dim((int)dim, cfg);
    OcRopeScalingConfig tmp_slow = *cfg;
    tmp_slow.beta_fast = cfg->beta_slow;
    corr_hi = oc_rope_yarn_find_correction_dim((int)dim, &tmp_slow);
    if (corr_lo > corr_hi) { int t = corr_lo; corr_lo = corr_hi; corr_hi = t; }
    if (corr_lo < 0) corr_lo = 0;
    if (corr_hi > (int)dim - 1) corr_hi = (int)dim - 1;

    (void)freq;  /* freq is handled by caller via dim index */

    /* This function returns the effective angle for this freq at this pos.
     * But the actual per-dimension logic is in oc_rope_apply. This helper
     * returns the extrapolated angle (no interpolation) as a fallback. */
    float theta_extrap = freq * (float)pos;
    float theta_interp = theta_extrap * freq_scale;
    (void)mscale;
    (void)corr_lo;
    (void)corr_hi;
    return theta_interp;
}

OcError oc_rope_apply(const OcRopeScalingConfig *cfg, uint32_t pos,
                      uint32_t dim, float base_freq,
                      float *out_cos, float *out_sin)
{
    if (!cfg || !out_cos || !out_sin || dim == 0) return OC_ERR_INVALID_ARG;
    if (dim % 2 != 0) return OC_ERR_INVALID_ARG;

    uint32_t half_dim = dim / 2;

    /* Precompute frequency for each dimension index (0..half_dim-1).
     * freq_i = theta ^ (-(2*i)/dim) = theta ^ (-i/half_dim) */
    /* Rust: freq starts at 1.0, multiplied by theta^(-2/dim) each step. */
    float freq_mult = powf(base_freq, -2.0f / (float)dim);

    if (cfg->type == OC_ROPE_YARN) {
        /* YaRN: blend extrapolated and interpolated angles with a linear ramp. */
        float freq_scale = 1.0f / cfg->scale_factor;
        float mscale = oc_rope_yarn_mscale(cfg->scale_factor);

        /* Compute correction range using the same formula as Rust. */
        float ratio_fast = (float)cfg->original_max_pos /
                           (cfg->beta_fast * 2.0f * 3.14159265f);
        float ratio_slow = (float)cfg->original_max_pos /
                           (cfg->beta_slow * 2.0f * 3.14159265f);
        float base_log = 2.0f * logf(base_freq);
        float corr_lo = floorf((float)dim * logf(ratio_fast) / base_log);
        float corr_hi = ceilf((float)dim * logf(ratio_slow) / base_log);
        if (corr_lo < 0.0f) corr_lo = 0.0f;
        if (corr_hi > (float)dim - 1.0f) corr_hi = (float)dim - 1.0f;
        float denom = corr_hi - corr_lo;

        float freq = 1.0f;
        for (uint32_t i = 0; i < half_dim; i++) {
            float theta_extrap = (float)pos * freq;
            float theta_interp = theta_extrap * freq_scale;

            float ramp = 1.0f - ((float)i - corr_lo) /
                         (denom > 0.001f ? denom : 0.001f);
            if (ramp < 0.0f) ramp = 0.0f;
            if (ramp > 1.0f) ramp = 1.0f;

            float angle = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
            out_cos[i] = cosf(angle) * mscale;
            out_sin[i] = sinf(angle) * mscale;
            freq *= freq_mult;
        }
    } else if (cfg->type == OC_ROPE_DYNAMIC_NTK) {
        /* Dynamic NTK: scale base frequency dynamically based on position. */
        float scale = oc_rope_scale_factor(cfg, pos);
        float alpha = powf(scale, (float)dim / (float)(dim > 2 ? dim - 2 : 1));
        float ntk_base = base_freq * alpha;
        for (uint32_t i = 0; i < half_dim; i++) {
            float freq = powf(ntk_base, -(float)(2 * i) / (float)dim);
            float angle = freq * (float)pos;
            out_cos[i] = cosf(angle);
            out_sin[i] = sinf(angle);
        }
    } else {
        /* None, Linear, NTK: use the standard per-dimension approach. */
        float freq = 1.0f;
        for (uint32_t i = 0; i < half_dim; i++) {
            float scaled_pos;
            switch (cfg->type) {
            case OC_ROPE_LINEAR:
                scaled_pos = (float)pos / cfg->scale_factor;
                break;
            case OC_ROPE_NTK: {
                float alpha = powf(cfg->scale_factor,
                    (float)dim / (float)(dim > 2 ? dim - 2 : 1));
                float ntk_freq = freq / alpha;
                scaled_pos = ntk_freq * (float)pos;
                break;
            }
            default:
                scaled_pos = freq * (float)pos;
                break;
            }
            out_cos[i] = cosf(scaled_pos);
            out_sin[i] = sinf(scaled_pos);
            freq *= freq_mult;
        }
    }

    return OC_OK;
}

OcError oc_rope_apply_to_tensor(const OcRopeScalingConfig *cfg, uint32_t pos,
                                float *tensor, uint32_t head_dim,
                                uint32_t n_heads, float base_freq)
{
    if (!cfg || !tensor || head_dim == 0 || n_heads == 0)
        return OC_ERR_INVALID_ARG;
    if (head_dim % 2 != 0) return OC_ERR_INVALID_ARG;

    /* Allocate cos/sin arrays on stack (head_dim/2 is typically 32-64). */
    float cos_buf[256];
    float sin_buf[256];
    uint32_t half_dim = head_dim / 2;
    if (half_dim > 256) return OC_ERR_INVALID_ARG;

    OcError e = oc_rope_apply(cfg, pos, head_dim, base_freq, cos_buf, sin_buf);
    if (e != OC_OK) return e;

    /* Apply rotation to each head (NeoX/HF split-half style). */
    for (uint32_t h = 0; h < n_heads; h++) {
        float *head = tensor + h * head_dim;
        for (uint32_t i = 0; i < half_dim; i++) {
            float c = cos_buf[i];
            float s = sin_buf[i];
            float x0 = head[i];
            float x1 = head[i + half_dim];
            head[i] = x0 * c - x1 * s;
            head[i + half_dim] = x0 * s + x1 * c;
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
