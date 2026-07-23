/* test_activation.c — Activation function tests (softmax, layer_norm, swiglu, attention, rms_norm_qwen). */
#include <criterion/criterion.h>
#include "oxidize/activation.h"
#include <math.h>
#include <string.h>

Test(act, softmax_basic)
{
    float in[] = {1.0f, 2.0f, 3.0f};
    float out[3];
    oc_softmax_f32(in, out, 3);
    /* Sum should be 1.0. */
    float sum = out[0] + out[1] + out[2];
    cr_assert_float_eq(sum, 1.0f, 0.001f);
    /* Largest input -> largest output. */
    cr_assert(out[2] > out[1]);
    cr_assert(out[1] > out[0]);
}

Test(act, softmax_uniform)
{
    float in[] = {5.0f, 5.0f, 5.0f};
    float out[3];
    oc_softmax_f32(in, out, 3);
    cr_assert_float_eq(out[0], 1.0f/3.0f, 0.001f);
    cr_assert_float_eq(out[1], 1.0f/3.0f, 0.001f);
    cr_assert_float_eq(out[2], 1.0f/3.0f, 0.001f);
}

Test(act, softmax_large_values)
{
    float in[] = {1000.0f, 1001.0f, 1002.0f};
    float out[3];
    oc_softmax_f32(in, out, 3);
    /* Should not overflow/NaN. */
    cr_assert(!isnan(out[0]));
    cr_assert(!isnan(out[1]));
    cr_assert(!isnan(out[2]));
    float sum = out[0] + out[1] + out[2];
    cr_assert_float_eq(sum, 1.0f, 0.001f);
}

Test(act, softmax_single)
{
    float in[] = {42.0f};
    float out[1];
    oc_softmax_f32(in, out, 1);
    cr_assert_float_eq(out[0], 1.0f, 0.001f);
}

Test(act, layer_norm_basic)
{
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float b[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[4];
    oc_layer_norm_f32(in, w, b, out, 4, 1e-5f);
    /* Mean = 2.5, var = 1.25, inv_std = 1/sqrt(1.25+eps) */
    float mean = 2.5f;
    float inv_std = 1.0f / sqrtf(1.25f + 1e-5f);
    cr_assert_float_eq(out[0], (1.0f - mean) * inv_std, 0.01f);
    cr_assert_float_eq(out[3], (4.0f - mean) * inv_std, 0.01f);
}

Test(act, layer_norm_with_weight_bias)
{
    float in[] = {0.0f, 0.0f};
    float w[] = {2.0f, 3.0f};
    float b[] = {1.0f, 2.0f};
    float out[2];
    oc_layer_norm_f32(in, w, b, out, 2, 1e-5f);
    /* mean=0, var=0, inv_std=1/sqrt(eps) -> out = 0*inv_std*w + b = b */
    cr_assert_float_eq(out[0], 1.0f, 0.1f);
    cr_assert_float_eq(out[1], 2.0f, 0.1f);
}

Test(act, swiglu_non_inplace)
{
    float gate[] = {0.0f, 1.0f, -1.0f};
    float up[] = {2.0f, 3.0f, 4.0f};
    float out[3];
    oc_swiglu_f32(gate, up, out, 3);
    /* silu(0) = 0, silu(1) = 1*sigmoid(1) = 0.7311, silu(-1) = -1*sigmoid(-1) = -0.2689 */
    cr_assert_float_eq(out[0], 0.0f, 0.001f);
    cr_assert_float_eq(out[1], 0.7311f * 3.0f, 0.01f);
    cr_assert_float_eq(out[2], -0.2689f * 4.0f, 0.01f);
}

Test(act, scaled_dot_product_attention_basic)
{
    /* dim=2, seq_len=2.
     * query = [1, 0], keys = [[1,0],[0,1]], values = [[10,20],[30,40]].
     * score0 = 1/sqrt(2) ~ 0.707, score1 = 0.
     * softmax([0.707, 0]) = [0.669, 0.331].
     * output = 0.669*[10,20] + 0.331*[30,40] = [16.65, 26.62]. */
    float q[] = {1.0f, 0.0f};
    float k[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float v[] = {10.0f, 20.0f, 30.0f, 40.0f};
    float out[2];
    oc_scaled_dot_product_attention_f32(q, k, v, 2, 2, out);
    /* Check it's a weighted average (not exactly [10,20]). */
    cr_assert(out[0] > 10.0f && out[0] < 30.0f);
    cr_assert(out[1] > 20.0f && out[1] < 40.0f);
    /* More weight on key0 since score0 > score1. */
    cr_assert(out[0] < 20.0f);  /* closer to 10 than 30 */
}

Test(act, scaled_dot_product_attention_uniform)
{
    /* All keys equal -> uniform attention -> output = mean of values. */
    float q[] = {1.0f, 1.0f};
    float k[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float v[] = {10.0f, 20.0f, 30.0f, 40.0f};
    float out[2];
    oc_scaled_dot_product_attention_f32(q, k, v, 2, 2, out);
    /* Mean of [10,20] and [30,40] = [20, 30]. */
    cr_assert_float_eq(out[0], 20.0f, 0.1f);
    cr_assert_float_eq(out[1], 30.0f, 0.1f);
}

Test(act, scaled_dot_product_attention_empty)
{
    float q[] = {1.0f};
    float k[1];
    float v[1];
    float out[1] = {99.0f};
    oc_scaled_dot_product_attention_f32(q, k, v, 0, 1, out);
    /* Empty seq -> output should be 0. */
    cr_assert_float_eq(out[0], 0.0f, 0.001f);
}

Test(act, rms_norm_qwen_standard)
{
    /* weight_plus_one = false -> same as standard rms_norm. */
    float x[] = {3.0f, 4.0f};
    float w[] = {1.0f, 1.0f};
    float out[2];
    oc_rms_norm_f32_qwen(x, w, out, 2, 1e-5f, false);
    /* Compare with standard. */
    float out_std[2];
    oc_rms_norm_f32(x, w, out_std, 2, 1e-5f);
    cr_assert_float_eq(out[0], out_std[0], 0.001f);
    cr_assert_float_eq(out[1], out_std[1], 0.001f);
}

Test(act, rms_norm_qwen_plus_one)
{
    /* weight_plus_one = true -> uses (1+w) instead of w. */
    float x[] = {3.0f, 4.0f};
    float w[] = {0.5f, 0.5f};
    float out[2];
    oc_rms_norm_f32_qwen(x, w, out, 2, 1e-5f, true);
    /* Standard norm with w=1.5 should match. */
    float w15[] = {1.5f, 1.5f};
    float out_std[2];
    oc_rms_norm_f32(x, w15, out_std, 2, 1e-5f);
    cr_assert_float_eq(out[0], out_std[0], 0.001f);
    cr_assert_float_eq(out[1], out_std[1], 0.001f);
}

Test(act, rms_norm_qwen_zero_weights)
{
    /* weight_plus_one = true with w=0 -> uses (1+0)=1, same as no weight. */
    float x[] = {1.0f, 2.0f, 3.0f};
    float w[] = {0.0f, 0.0f, 0.0f};
    float out[3];
    oc_rms_norm_f32_qwen(x, w, out, 3, 1e-5f, true);
    /* Should just be x * inv_rms. */
    float ss = 1.0f + 4.0f + 9.0f;
    float inv_rms = 1.0f / sqrtf(ss / 3.0f + 1e-5f);
    cr_assert_float_eq(out[0], 1.0f * inv_rms, 0.01f);
    cr_assert_float_eq(out[1], 2.0f * inv_rms, 0.01f);
    cr_assert_float_eq(out[2], 3.0f * inv_rms, 0.01f);
}
