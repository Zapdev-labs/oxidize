/* test_rope_scaling.c — RoPE scaling tests. */
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
    cr_assert_eq(oc_rope_apply(&cfg, 10, 8, 10000.0f, cos_buf, sin_buf), OC_OK);
    /* cos/sin should have valid values */
    for (int i = 0; i < 8; i++) {
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
    cr_assert_eq(oc_rope_apply(&cfg, 10, 8, 10000.0f, cos_buf, sin_buf), OC_OK);
    /* Values should be valid */
    for (int i = 0; i < 8; i++) {
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
