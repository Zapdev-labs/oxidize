#ifndef OXIDIZE_ROPE_SCALING_H
#define OXIDIZE_ROPE_SCALING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
    OC_ROPE_NONE        = 0,
    OC_ROPE_LINEAR      = 1,
    OC_ROPE_NTK         = 2,
    OC_ROPE_YARN        = 3,
    OC_ROPE_DYNAMIC_NTK = 4,
} OcRopeScalingType;

typedef struct {
    OcRopeScalingType type;
    float scale_factor;           /* default 4.0 */
    uint32_t original_max_pos;    /* default 4096 */
    uint32_t extended_max_pos;    /* default 16384 */
    float beta_fast;             /* YaRN: default 32 */
    float beta_slow;            /* YaRN: default 1 */
    float attention_factor;      /* YaRN: default 1.0 */
    float mscale;
    float mscale_all_dim;
} OcRopeScalingConfig;


/* Initialize config with defaults. */
OcError oc_rope_config_init(OcRopeScalingConfig *cfg);

/* Compute the effective scale factor for a given position. */
float oc_rope_scale_factor(const OcRopeScalingConfig *cfg, uint32_t pos);

/* Linear scaling: pos / scale. */
float oc_rope_apply_linear(float freq, uint32_t pos, float scale);

/* NTK-aware: modifies base frequency. */
float oc_rope_apply_ntk(float freq, uint32_t pos, float scale,
                         uint32_t dim, uint32_t max_pos);

/* YaRN: wavelength-based scaling. */
float oc_rope_apply_yarn(float freq, uint32_t pos,
                         const OcRopeScalingConfig *cfg, uint32_t dim);

/* YaRN helper: find correction dimension. */
int oc_rope_yarn_find_correction_dim(int dim, const OcRopeScalingConfig *cfg);

/* YaRN helper: find correction range. */
void oc_rope_yarn_find_correction_range(int *lo, int *hi,
                                       const OcRopeScalingConfig *cfg);

/* YaRN helper: compute linear ramp factor. */
float oc_rope_yarn_linear_ramp_factor(float min_val, float max_val,
                                      const OcRopeScalingConfig *cfg);

/* YaRN helper: compute mscale. Equivalent to oc_rope_yarn_mscale_m(scale, 1). */
float oc_rope_yarn_mscale(float scale);

/* YaRN helper: mscale with an explicit exponent multiplier `m`
 * (get_mscale in the DeepSeek reference): 0.1*m*ln(scale) + 1, or 1 when
 * scale <= 1. */
float oc_rope_yarn_mscale_m(float scale, float m);

/* deepseek_yarn: resolve the two derived scales from the mscale pair. */
void oc_rope_deepseek_yarn_scales(float scale_factor, float mscale,
                                  float mscale_all_dim, uint32_t head_dim,
                                  float *rope_attn_factor,
                                  float *softmax_scale);

/* Apply RoPE: compute cos/sin for a given position and frequency.
 * Writes to out_cos and out_sin (arrays of `dim` elements). */
OcError oc_rope_apply(const OcRopeScalingConfig *cfg, uint32_t pos,
                      uint32_t dim, float base_freq,
                      float *out_cos, float *out_sin);

/* Apply RoPE to a tensor in-place. The tensor has shape [n_heads, head_dim]
 * and the RoPE is applied to each head. `tensor` is a float array of size
 * n_heads * head_dim. */
OcError oc_rope_apply_to_tensor(const OcRopeScalingConfig *cfg, uint32_t pos,
                                float *tensor, uint32_t head_dim,
                                uint32_t n_heads, float base_freq);

/* Get the string name of a scaling type. */
const char *oc_rope_scaling_type_name(OcRopeScalingType type);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ROPE_SCALING_H */
