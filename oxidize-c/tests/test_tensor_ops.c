/* test_tensor_ops.c — tensor operations tests. */
#include <criterion/criterion.h>
#include "oxidize/tensor_ops.h"
#include <math.h>
#include <string.h>

Test(tensor, add)
{
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float out[3];
    oc_tensor_add_f32(a, b, out, 3);
    cr_assert_float_eq(out[0], 5.0f, 1e-6f);
    cr_assert_float_eq(out[1], 7.0f, 1e-6f);
    cr_assert_float_eq(out[2], 9.0f, 1e-6f);
}

Test(tensor, mul)
{
    float a[] = {2.0f, 3.0f, 4.0f};
    float b[] = {5.0f, 6.0f, 7.0f};
    float out[3];
    oc_tensor_mul_f32(a, b, out, 3);
    cr_assert_float_eq(out[0], 10.0f, 1e-6f);
    cr_assert_float_eq(out[1], 18.0f, 1e-6f);
    cr_assert_float_eq(out[2], 28.0f, 1e-6f);
}

Test(tensor, scale)
{
    float a[] = {1.0f, 2.0f, 3.0f};
    float out[3];
    oc_tensor_scale_f32(a, 2.5f, out, 3);
    cr_assert_float_eq(out[0], 2.5f, 1e-6f);
    cr_assert_float_eq(out[1], 5.0f, 1e-6f);
    cr_assert_float_eq(out[2], 7.5f, 1e-6f);
}

Test(tensor, sum)
{
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    cr_assert_float_eq(oc_tensor_sum_f32(a, 4), 10.0f, 1e-6f);
}

Test(tensor, max)
{
    float a[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
    cr_assert_float_eq(oc_tensor_max_f32(a, 5), 5.0f, 1e-6f);
}

Test(tensor, argmax)
{
    float a[] = {1.0f, 5.0f, 3.0f, 7.0f, 2.0f};
    cr_assert_eq(oc_tensor_argmax_f32(a, 5), 3);
}

Test(tensor, l2_norm)
{
    float a[] = {3.0f, 4.0f};
    cr_assert_float_eq(oc_tensor_l2_norm_f32(a, 2), 5.0f, 1e-6f);
}

Test(tensor, mean_variance)
{
    float a[] = {2.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 7.0f, 9.0f};
    cr_assert_float_eq(oc_tensor_mean_f32(a, 8), 5.0f, 1e-6f);
    cr_assert_float_eq(oc_tensor_variance_f32(a, 8), 4.0f, 1e-6f);
}

Test(tensor, softmax)
{
    float a[] = {1.0f, 2.0f, 3.0f};
    float out[3];
    oc_tensor_softmax_f32(a, out, 3);
    /* Sum should be 1. */
    float sum = out[0] + out[1] + out[2];
    cr_assert_float_eq(sum, 1.0f, 1e-5f);
    /* Largest input → largest output. */
    cr_assert(out[2] > out[1]);
    cr_assert(out[1] > out[0]);
}

Test(tensor, softmax_temp)
{
    float a[] = {1.0f, 2.0f, 3.0f};
    float out_hot[3], out_cold[3];
    oc_tensor_softmax_temp_f32(a, out_hot, 3, 0.5f);
    oc_tensor_softmax_temp_f32(a, out_cold, 3, 2.0f);
    /* Lower temp → sharper distribution. */
    cr_assert(out_hot[2] > out_cold[2]);
}

Test(tensor, log_softmax)
{
    float a[] = {1.0f, 2.0f, 3.0f};
    float out[3];
    oc_tensor_log_softmax_f32(a, out, 3);
    /* log(softmax) should sum to log(1) = 0. */
    float sum = expf(out[0]) + expf(out[1]) + expf(out[2]);
    cr_assert_float_eq(sum, 1.0f, 1e-5f);
}

Test(tensor, layer_norm)
{
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float b[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[4];
    oc_tensor_layer_norm_f32(a, w, b, out, 4, 1e-5f);
    /* Mean of output should be ~0 (since weights=1, bias=0). */
    float m = oc_tensor_mean_f32(out, 4);
    cr_assert_float_eq(m, 0.0f, 1e-4f);
}

Test(tensor, rms_norm)
{
    float a[] = {3.0f, 4.0f};
    float w[] = {1.0f, 1.0f};
    float out[2];
    oc_tensor_rms_norm_f32(a, w, out, 2, 1e-5f);
    /* RMS of a = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355. */
    /* Output should be a / 3.5355 ≈ {0.8485, 1.1314}. */
    cr_assert_float_eq(out[0], 3.0f / sqrtf(12.5f), 1e-4f);
    cr_assert_float_eq(out[1], 4.0f / sqrtf(12.5f), 1e-4f);
}

Test(tensor, silu)
{
    float a[] = {0.0f, 1.0f, -1.0f};
    float out[3];
    oc_tensor_silu_f32(a, out, 3);
    cr_assert_float_eq(out[0], 0.0f, 1e-6f);
    cr_assert(out[1] > 0.7f && out[1] < 0.8f);
    cr_assert(out[2] > -0.3f && out[2] < -0.2f);
}

Test(tensor, gelu)
{
    float a[] = {0.0f, 1.0f, -1.0f};
    float out[3];
    oc_tensor_gelu_f32(a, out, 3);
    cr_assert_float_eq(out[0], 0.0f, 1e-6f);
    cr_assert(out[1] > 0.8f && out[1] < 0.9f);
    cr_assert(out[2] > -0.2f && out[2] < -0.1f);
}

Test(tensor, transpose)
{
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}; /* 2×3 */
    float out[6]; /* 3×2 */
    oc_tensor_transpose_f32(a, out, 2, 3);
    cr_assert_float_eq(out[0], 1.0f, 1e-6f);
    cr_assert_float_eq(out[1], 4.0f, 1e-6f);
    cr_assert_float_eq(out[2], 2.0f, 1e-6f);
    cr_assert_float_eq(out[5], 6.0f, 1e-6f);
}

Test(tensor, concat)
{
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f}; /* 1×4 */
    float b[] = {5.0f, 6.0f};            /* 1×2 */
    float out[6];
    oc_tensor_concat_f32(a, b, out, 1, 4, 2);
    for (int i = 0; i < 6; i++)
        cr_assert_float_eq(out[i], (float)(i + 1), 1e-6f, "out[%d]", i);
}

Test(tensor, gemm)
{
    /* A: 2×3, B: 3×2, C: 2×2 */
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float B[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float C[4];
    oc_tensor_gemm_f32(A, B, C, 2, 3, 2);
    /* C[0][0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58 */
    cr_assert_float_eq(C[0], 58.0f, 1e-4f);
    /* C[0][1] = 1*8 + 2*10 + 3*12 = 8+20+36 = 64 */
    cr_assert_float_eq(C[1], 64.0f, 1e-4f);
    /* C[1][0] = 4*7 + 5*9 + 6*11 = 28+45+66 = 139 */
    cr_assert_float_eq(C[2], 139.0f, 1e-4f);
}

Test(tensor, fill_zero)
{
    float a[5];
    oc_tensor_fill_f32(a, 42.0f, 5);
    cr_assert_float_eq(a[0], 42.0f, 1e-6f);
    cr_assert_float_eq(a[4], 42.0f, 1e-6f);
    oc_tensor_zero_f32(a, 5);
    cr_assert_float_eq(a[0], 0.0f, 1e-6f);
    cr_assert_float_eq(a[4], 0.0f, 1e-6f);
}

Test(tensor, random)
{
    float a[100];
    oc_tensor_random_f32(a, 100, 42);
    /* All values should be in [0, 1). */
    for (int i = 0; i < 100; i++) {
        cr_assert(a[i] >= 0.0f && a[i] < 1.0f, "value %d out of range: %f", i, a[i]);
    }
    /* Same seed → same output. */
    float b[100];
    oc_tensor_random_f32(b, 100, 42);
    for (int i = 0; i < 100; i++)
        cr_assert_float_eq(a[i], b[i], 1e-6f, "same seed should produce same values");
}

Test(tensor, iadd_imul)
{
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    oc_tensor_iadd_f32(a, b, 3);
    cr_assert_float_eq(a[0], 5.0f, 1e-6f);
    cr_assert_float_eq(a[1], 7.0f, 1e-6f);
    cr_assert_float_eq(a[2], 9.0f, 1e-6f);
    oc_tensor_imul_f32(a, b, 3);
    cr_assert_float_eq(a[0], 20.0f, 1e-6f);
    cr_assert_float_eq(a[1], 35.0f, 1e-6f);
    cr_assert_float_eq(a[2], 54.0f, 1e-6f);
}

Test(tensor, attention_head)
{
    /* Q: [2], K: [3×2], V: [3×2] */
    float Q[] = {1.0f, 0.0f};
    float K[] = {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f};
    float V[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float out[2];
    float scale = 1.0f / sqrtf(2.0f);
    oc_tensor_attention_head_f32(Q, K, V, out, 3, 2, scale);
    /* Expected: scores = {1, 0, 0.5} * scale, softmax weights
     * w = {0.455527, 0.224606, 0.319866}; out = sum(w[i] * V[i]). */
    float s0 = 1.0f * scale, s1 = 0.0f, s2 = 0.5f * scale;
    float e0 = expf(s0), e1 = expf(s1), e2 = expf(s2);
    float wsum = e0 + e1 + e2;
    float w0 = e0 / wsum, w1 = e1 / wsum, w2 = e2 / wsum;
    float exp0 = w0 * V[0] + w1 * V[2] + w2 * V[4];
    float exp1 = w0 * V[1] + w1 * V[3] + w2 * V[5];
    cr_assert_float_eq(out[0], exp0, 1e-5f); /* ≈ 2.72868 */
    cr_assert_float_eq(out[1], exp1, 1e-5f); /* ≈ 3.72868 */
}

Test(tensor, copy)
{
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[3];
    oc_tensor_copy_f32(a, b, 3);
    cr_assert_float_eq(b[0], 1.0f, 1e-6f);
    cr_assert_float_eq(b[2], 3.0f, 1e-6f);
}

Test(tensor, repeat_row)
{
    float row[] = {1.0f, 2.0f, 3.0f};
    float out[9]; /* 3 × 3 */
    oc_tensor_repeat_row_f32(row, out, 3, 3);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            cr_assert_float_eq(out[r * 3 + c], row[c], 1e-6f,
                               "out[%d][%d]", r, c);
}
