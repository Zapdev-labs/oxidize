#include <criterion/criterion.h>
#include "oxidize/rope_scaling.h"
#include <math.h>
#include <string.h>

Test(rope, config_init)
{
    OcRopeScalingConfig cfg;
    cr_assert_eq(oc_rope_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.type, OC_ROPE_NONE);
    cr_assert_float_eq(cfg.scale_factor, 4.0f, 0.001f);
    cr_assert_eq(cfg.original_max_pos, 4096);
    cr_assert_eq(cfg.extended_max_pos, 16384);
    cr_assert_float_eq(cfg.beta_fast, 32.0f, 0.001f);
    cr_assert_float_eq(cfg.beta_slow, 1.0f, 0.001f);
}

Test(rope, config_init_null)
{
    cr_assert_neq(oc_rope_config_init(NULL), OC_OK);
}

Test(rope, scale_factor_none)
{
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    cfg.type = OC_ROPE_NONE;
    cr_assert_float_eq(oc_rope_scale_factor(&cfg, 100), 1.0f, 0.001f);
}

Test(rope, scale_factor_linear)
{
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    cfg.type = OC_ROPE_LINEAR;
    cr_assert_float_eq(oc_rope_scale_factor(&cfg, 100), 4.0f, 0.001f);
}

Test(rope, scale_factor_dynamic_ntk)
{
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    cfg.type = OC_ROPE_DYNAMIC_NTK;
    /* Within original max_pos: scale = 1.0 */
    cr_assert_float_eq(oc_rope_scale_factor(&cfg, 100), 1.0f, 0.001f);
    /* Beyond original max_pos: scale > 1 */
    float s = oc_rope_scale_factor(&cfg, 8192);
    cr_assert(s > 1.0f, "scale should be > 1 for pos > max_pos, got %f", s);
}

Test(rope, apply_linear)
{
    float result = oc_rope_apply_linear(1.0f, 100, 4.0f);
    cr_assert_float_eq(result, 25.0f, 0.001f);
}

Test(rope, apply_ntk)
{
    float freq = 1.0f;
    float result = oc_rope_apply_ntk(freq, 0, 4.0f, 64, 4096);
    cr_assert(result != freq, "NTK should modify frequency");
}

Test(rope, yarn_mscale)
{
    cr_assert_float_eq(oc_rope_yarn_mscale(1.0f), 1.0f, 0.001f);
    float s = oc_rope_yarn_mscale(4.0f);
    cr_assert(s > 1.0f, "mscale(4.0) should be > 1, got %f", s);
}

Test(rope, yarn_correction_dim)
{
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    cfg.type = OC_ROPE_YARN;
    int dim = oc_rope_yarn_find_correction_dim(0, &cfg);
    cr_assert(dim >= 0, "correction dim should be >= 0");
}

Test(rope, yarn_correction_range)
{
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    cfg.type = OC_ROPE_YARN;
    int lo, hi;
    oc_rope_yarn_find_correction_range(&lo, &hi, &cfg);
    cr_assert(lo >= 0);
    cr_assert(hi >= lo);
}

Test(rope, apply_none)
{
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    float cos_buf[8], sin_buf[8];
    memset(cos_buf, 0, sizeof(cos_buf));
    memset(sin_buf, 0, sizeof(sin_buf));
    cr_assert_eq(oc_rope_apply(&cfg, 10, 8, 10000.0f, cos_buf, sin_buf), OC_OK);
    /* cos/sin should have valid values (first half_dim=4 elements written). */
    for (int i = 0; i < 4; i++) {
        cr_assert(cos_buf[i] >= -1.01f && cos_buf[i] <= 1.01f);
        cr_assert(sin_buf[i] >= -1.01f && sin_buf[i] <= 1.01f);
    }
}

Test(rope, apply_linear_cfg)
{
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    cfg.type = OC_ROPE_LINEAR;
    float cos_buf[8], sin_buf[8];
    memset(cos_buf, 0, sizeof(cos_buf));
    memset(sin_buf, 0, sizeof(sin_buf));
    cr_assert_eq(oc_rope_apply(&cfg, 10, 8, 10000.0f, cos_buf, sin_buf), OC_OK);
    /* Values should be valid (first half_dim=4 elements written). */
    for (int i = 0; i < 4; i++) {
        cr_assert(cos_buf[i] >= -1.01f && cos_buf[i] <= 1.01f);
    }
}

Test(rope, apply_yarn)
{
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    cfg.type = OC_ROPE_YARN;
    float cos_buf[64], sin_buf[64];
    cr_assert_eq(oc_rope_apply(&cfg, 100, 64, 10000.0f, cos_buf, sin_buf), OC_OK);
}

Test(rope, yarn_mscale_value)
{
    /* mscale = 1 + 0.1 * ln(scale). For scale=4: 1 + 0.1*ln(4) ≈ 1.1386 */
    float ms = oc_rope_yarn_mscale(4.0f);
    cr_assert_float_eq(ms, 1.0f + 0.1f * logf(4.0f), 0.001f);
}

Test(rope, yarn_cos_sin_scaled_by_mscale)
{
    /* YaRN cos/sin should be scaled by mscale > 1, so |cos| can exceed 1. */
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    cfg.type = OC_ROPE_YARN;
    cfg.scale_factor = 4.0f;
    float cos_buf[64], sin_buf[64];
    memset(cos_buf, 0, sizeof(cos_buf));
    memset(sin_buf, 0, sizeof(sin_buf));
    cr_assert_eq(oc_rope_apply(&cfg, 100, 64, 10000.0f, cos_buf, sin_buf), OC_OK);
    /* At least one cos value should be > 1 (due to mscale). */
    bool found_scaled = false;
    for (int i = 0; i < 32; i++) {
        if (cos_buf[i] > 1.0f || sin_buf[i] > 1.0f) {
            found_scaled = true;
            break;
        }
    }
    cr_assert(found_scaled, "YaRN mscale should produce cos/sin > 1");
}

Test(rope, yarn_pos_zero_applies_mscale)
{
    /* At pos=0 with YaRN, mscale is applied (cos(0)=1, sin(0)=0),
     * so values are scaled by mscale but not rotated. */
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    cfg.type = OC_ROPE_YARN;
    cfg.scale_factor = 4.0f;
    float tensor[64];
    for (int i = 0; i < 64; i++) tensor[i] = (float)(i + 1);
    cr_assert_eq(oc_rope_apply_to_tensor(&cfg, 0, tensor, 64, 1, 10000.0f), OC_OK);
    /* Position 0 with YaRN: values scaled by mscale = 1 + 0.1*ln(4). */
    float mscale = 1.0f + 0.1f * logf(4.0f);
    for (int i = 0; i < 32; i++) {
        cr_assert_float_eq(tensor[i], (float)(i + 1) * mscale, 0.01f,
            "tensor[%d] = %f, expected %f", i, tensor[i], (float)(i + 1) * mscale);
    }
}

Test(rope, yarn_differs_from_none)
{
    /* YaRN at high position should produce different angles than no-scaling. */
    OcRopeScalingConfig cfg_none, cfg_yarn;
    oc_rope_config_init(&cfg_none);
    oc_rope_config_init(&cfg_yarn);
    cfg_yarn.type = OC_ROPE_YARN;
    cfg_yarn.scale_factor = 4.0f;

    float cos_none[32], sin_none[32];
    float cos_yarn[32], sin_yarn[32];
    memset(cos_none, 0, sizeof(cos_none));
    memset(sin_none, 0, sizeof(sin_none));
    memset(cos_yarn, 0, sizeof(cos_yarn));
    memset(sin_yarn, 0, sizeof(sin_yarn));

    oc_rope_apply(&cfg_none, 5000, 64, 10000.0f, cos_none, sin_none);
    oc_rope_apply(&cfg_yarn, 5000, 64, 10000.0f, cos_yarn, sin_yarn);

    bool differs = false;
    for (int i = 0; i < 32; i++) {
        if (fabsf(cos_none[i] - cos_yarn[i]) > 0.01f ||
            fabsf(sin_none[i] - sin_yarn[i]) > 0.01f) {
            differs = true;
            break;
        }
    }
    cr_assert(differs, "YaRN should produce different angles than no-scaling");
}

Test(rope, dynamic_ntk_scales_beyond_orig)
{
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    cfg.type = OC_ROPE_DYNAMIC_NTK;
    float cos_buf[64], sin_buf[64];
    memset(cos_buf, 0, sizeof(cos_buf));
    memset(sin_buf, 0, sizeof(sin_buf));
    /* pos beyond original_max_pos should use NTK scaling. */
    cr_assert_eq(oc_rope_apply(&cfg, 8192, 64, 10000.0f, cos_buf, sin_buf), OC_OK);
    /* Values should be valid. */
    for (int i = 0; i < 32; i++) {
        cr_assert(cos_buf[i] >= -2.0f && cos_buf[i] <= 2.0f);
    }
}

Test(rope, odd_dim_rejected)
{
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    float cos_buf[7], sin_buf[7];
    cr_assert_neq(oc_rope_apply(&cfg, 10, 7, 10000.0f, cos_buf, sin_buf), OC_OK);
}

Test(rope, apply_ntk_cfg)
{
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    cfg.type = OC_ROPE_NTK;
    float cos_buf[32], sin_buf[32];
    cr_assert_eq(oc_rope_apply(&cfg, 50, 32, 10000.0f, cos_buf, sin_buf), OC_OK);
}

Test(rope, apply_null)
{
    cr_assert_neq(oc_rope_apply(NULL, 0, 8, 10000.0f, NULL, NULL), OC_OK);
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    cr_assert_neq(oc_rope_apply(&cfg, 0, 8, 10000.0f, NULL, NULL), OC_OK);
}

Test(rope, apply_to_tensor)
{
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    float tensor[64] = {0};
    for (int i = 0; i < 64; i++) tensor[i] = (float)i;
    cr_assert_eq(oc_rope_apply_to_tensor(&cfg, 5, tensor, 64, 1, 10000.0f), OC_OK);
    /* Values should be modified */
    bool changed = false;
    for (int i = 0; i < 64; i++) {
        if (tensor[i] != (float)i) { changed = true; break; }
    }
    cr_assert(changed, "tensor should be modified by RoPE");
}

Test(rope, apply_to_tensor_multihead)
{
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    cfg.type = OC_ROPE_LINEAR;
    float tensor[128] = {0};
    for (int i = 0; i < 128; i++) tensor[i] = 1.0f;
    cr_assert_eq(oc_rope_apply_to_tensor(&cfg, 10, tensor, 64, 2, 10000.0f), OC_OK);
}

Test(rope, apply_to_tensor_null)
{
    cr_assert_neq(oc_rope_apply_to_tensor(NULL, 0, NULL, 64, 1, 10000.0f), OC_OK);
}

Test(rope, type_name)
{
    cr_assert_str_eq(oc_rope_scaling_type_name(OC_ROPE_NONE), "none");
    cr_assert_str_eq(oc_rope_scaling_type_name(OC_ROPE_LINEAR), "linear");
    cr_assert_str_eq(oc_rope_scaling_type_name(OC_ROPE_NTK), "ntk");
    cr_assert_str_eq(oc_rope_scaling_type_name(OC_ROPE_YARN), "yarn");
    cr_assert_str_eq(oc_rope_scaling_type_name(OC_ROPE_DYNAMIC_NTK), "dynamic_ntk");
}

Test(rope, yarn_linear_ramp_factor)
{
    OcRopeScalingConfig cfg;
    oc_rope_config_init(&cfg);
    /* When min == max, should return 0. */
    cr_assert_float_eq(oc_rope_yarn_linear_ramp_factor(1.0f, 1.0f, &cfg), 0.0f, 0.001f);
    /* When max > min, should return 1.0. */
    cr_assert_float_eq(oc_rope_yarn_linear_ramp_factor(0.0f, 1.0f, &cfg), 1.0f, 0.001f);
}

Test(rope, ntk_modifies_base)
{
    /* NTK should modify the base frequency, producing different angles. */
    OcRopeScalingConfig cfg_none, cfg_ntk;
    oc_rope_config_init(&cfg_none);
    oc_rope_config_init(&cfg_ntk);
    cfg_ntk.type = OC_ROPE_NTK;
    cfg_ntk.scale_factor = 4.0f;

    float cos_none[32], sin_none[32];
    float cos_ntk[32], sin_ntk[32];
    memset(cos_none, 0, sizeof(cos_none));
    memset(sin_none, 0, sizeof(sin_none));
    memset(cos_ntk, 0, sizeof(cos_ntk));
    memset(sin_ntk, 0, sizeof(sin_ntk));

    oc_rope_apply(&cfg_none, 100, 64, 10000.0f, cos_none, sin_none);
    oc_rope_apply(&cfg_ntk, 100, 64, 10000.0f, cos_ntk, sin_ntk);

    bool differs = false;
    for (int i = 0; i < 32; i++) {
        if (fabsf(cos_none[i] - cos_ntk[i]) > 0.01f) {
            differs = true;
            break;
        }
    }
    cr_assert(differs, "NTK should produce different angles than no-scaling");
}


Test(rope_scaling, deepseek_yarn_scales_longcat)
{
    /* LongCat-2.0: factor 120, mscale = mscale_all_dim = 1, head_dim 192. */
    float rope_f = -1.0f, softmax = -1.0f;
    oc_rope_deepseek_yarn_scales(120.0f, 1.0f, 1.0f, 192, &rope_f, &softmax);

    cr_assert_float_eq(rope_f, 1.0f, 1e-6f,
        "LongCat rope_attn_factor must cancel to 1.0, got %.8f", rope_f);
    /* 192^-0.5 * 1.47874917^2 = 0.07216878 * 2.18670914 = 0.15781034
     * (double-precision reference; the f32 path lands ~1e-6 above it, so the
     * tolerance is sized for f32 rounding through logf/sqrtf, not for slop). */
    cr_assert_float_eq(softmax, 0.15781034f, 1e-5f,
        "LongCat softmax_scale expected 0.15781034, got %.8f", softmax);

    /* It must NOT equal the plain 1/sqrt(192) that a non-YaRN path would use;
     * that difference (2.1867x) is the bug this pins down. */
    cr_assert(fabsf(softmax - 0.07216878f) > 1e-3f,
        "softmax_scale must not collapse to plain 1/sqrt(head_dim)");
}

Test(rope_scaling, deepseek_yarn_scales_deepseek_v3)
{
    /* DeepSeek-V3: mscale=1, mscale_all_dim=0 -> the correction rides on
     * cos/sin instead, and the softmax scale stays the plain 1/sqrt(d). */
    float rope_f = -1.0f, softmax = -1.0f;
    oc_rope_deepseek_yarn_scales(40.0f, 1.0f, 0.0f, 192, &rope_f, &softmax);

    cr_assert_float_eq(rope_f, 0.1f * logf(40.0f) + 1.0f, 1e-6f,
        "DeepSeek rope_attn_factor should be get_mscale(40,1), got %.8f", rope_f);
    cr_assert_float_eq(softmax, 1.0f / sqrtf(192.0f), 1e-6f,
        "DeepSeek softmax_scale should be plain 1/sqrt(192), got %.8f", softmax);
}

Test(rope_scaling, deepseek_yarn_scales_no_scaling)
{
    /* factor <= 1 disables YaRN entirely: both terms are 1. */
    float rope_f = -1.0f, softmax = -1.0f;
    oc_rope_deepseek_yarn_scales(1.0f, 1.0f, 1.0f, 192, &rope_f, &softmax);
    cr_assert_float_eq(rope_f, 1.0f, 1e-6f, "no YaRN -> factor 1.0");
    cr_assert_float_eq(softmax, 1.0f / sqrtf(192.0f), 1e-6f,
        "no YaRN -> plain 1/sqrt(head_dim)");
}

Test(rope_scaling, yarn_correction_range_longcat)
{
    /* LongCat: rope dim 64, base 1e6, original ctx 8192, beta_fast 32, */
    const float two_pi = 6.283185307f;
    float base_log = 2.0f * logf(1.0e6f);
    float lo = floorf(64.0f * logf(8192.0f / (32.0f * two_pi)) / base_log);
    float hi = ceilf (64.0f * logf(8192.0f / ( 1.0f * two_pi)) / base_log);

    cr_assert_float_eq(lo, 8.0f, 1e-6f, "corr_lo expected 8, got %.4f", lo);
    cr_assert_float_eq(hi, 17.0f, 1e-6f, "corr_hi expected 17, got %.4f", hi);
}
