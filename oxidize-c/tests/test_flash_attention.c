/* test_flash_attention.c — flash attention kernel tests. */
#include <criterion/criterion.h>
#include "oxidize/flash_attention.h"
#include <math.h>
#include <string.h>

Test(flash_attn, single_position)
{
    /* Q = K = [1, 0, 0, 0], V = [1, 2, 3, 4]
     * score = 1/sqrt(4) * (1*1) = 0.5
     * softmax(0.5) = 1.0 (single element)
     * out = 1.0 * V = [1, 2, 3, 4] */
    float q[] = {1, 0, 0, 0};
    float k[] = {1, 0, 0, 0};
    float v[] = {1, 2, 3, 4};
    float out[4] = {0};
    float temp[4] = {0};

    cr_assert_eq(oc_flash_attention_head(q, k, v, 1, 4, out, temp), OC_OK);
    cr_assert_float_eq(out[0], 1.0f, 1e-5f);
    cr_assert_float_eq(out[1], 2.0f, 1e-5f);
    cr_assert_float_eq(out[2], 3.0f, 1e-5f);
    cr_assert_float_eq(out[3], 4.0f, 1e-5f);
}

Test(flash_attn, two_positions_equal_weight)
{
    /* Two positions with equal scores → output is average of V's. */
    float q[] = {1, 0, 0, 0};
    float k[] = {1, 0, 0, 0, 1, 0, 0, 0};  /* 2 positions */
    float v[] = {2, 0, 0, 0, 4, 0, 0, 0}; /* V0=[2,0,0,0], V1=[4,0,0,0] */
    float out[4] = {0};
    float temp[4] = {0};

    cr_assert_eq(oc_flash_attention_head(q, k, v, 2, 4, out, temp), OC_OK);
    /* Both have the same score, so out = (V0 + V1) / 2 = [3, 0, 0, 0] */
    cr_assert_float_eq(out[0], 3.0f, 1e-5f);
    cr_assert_float_eq(out[1], 0.0f, 1e-5f);
}

Test(flash_attn, attention_scores)
{
    float q[] = {1, 0, 0};
    float k[] = {1, 0, 0, 0, 1, 0}; /* 2 positions, dim 3 */
    float scores[2];
    cr_assert_eq(oc_attention_scores(q, k, 2, 3, scores), OC_OK);
    /* score[0] = 1/sqrt(3) * 1 = 0.5774 */
    cr_assert_float_eq(scores[0], 1.0f / sqrtf(3.0f), 1e-5f);
    cr_assert_float_eq(scores[1], 0.0f, 1e-6f);
}

Test(flash_attn, sliding_window)
{
    /* 3 positions, window starts at position 1 (skip position 0). */
    float q[] = {1, 0};
    float k[] = {1, 0, 2, 0, 1, 0}; /* 3 positions */
    float v[] = {10, 0, 20, 0, 30, 0};
    float out[2] = {0};
    float temp[2] = {0};

    cr_assert_eq(oc_flash_attention_sliding(q, k, v, 3, 2, 1, out, temp), OC_OK);
    /* Positions 1 and 2 are attended. Position 1 has K=[2,0], score=2/sqrt(2).
     * Position 2 has K=[1,0], score=1/sqrt(2).
     * Higher score → more weight on position 1 (V=[20,0]). */
    cr_assert(out[0] > 15.0f, "Should be closer to V1=20 than V2=30");
}

Test(flash_attn, null_safety)
{
    float q[] = {1};
    cr_assert_neq(oc_flash_attention_head(NULL, NULL, NULL, 0, 1, NULL, NULL), OC_OK);
    cr_assert_neq(oc_flash_attention_head(q, NULL, NULL, 0, 1, NULL, NULL), OC_OK);
}

Test(flash_attn, multi_head)
{
    /* 2 heads, 1 KV head (GQA with group_size=2).
     * head_dim=2, seq_len=1. */
    float q[] = {1, 0, 0, 1};  /* head 0: [1,0], head 1: [0,1] */
    float k[] = {1, 0};        /* 1 position, 1 KV head, dim 2 */
    float v[] = {5, 7};
    float out[4] = {0};
    float temp[2] = {0};

    cr_assert_eq(oc_flash_attention_multi_head(q, k, v, 2, 1, 1, 2, out, temp), OC_OK);
    /* Head 0: Q=[1,0], K=[1,0] → score=1/sqrt(2), out=V=[5,7] */
    cr_assert_float_eq(out[0], 5.0f, 1e-5f);
    cr_assert_float_eq(out[1], 7.0f, 1e-5f);
    /* Head 1: Q=[0,1], K=[1,0] → score=0, out=V=[5,7] */
    cr_assert_float_eq(out[2], 5.0f, 1e-5f);
    cr_assert_float_eq(out[3], 7.0f, 1e-5f);
}
