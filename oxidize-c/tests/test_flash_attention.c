/* test_flash_attention.c — flash attention kernel tests. */
#include "framework.h"
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
    /* Head 0: Q=[1,0], K=[1,0] -> score=1/sqrt(2), out=V=[5,7] */
    cr_assert_float_eq(out[0], 5.0f, 1e-5f);
    cr_assert_float_eq(out[1], 7.0f, 1e-5f);
    /* Head 1: Q=[0,1], K=[1,0] -> score=0, out=V=[5,7] */
    cr_assert_float_eq(out[2], 5.0f, 1e-5f);
    cr_assert_float_eq(out[3], 7.0f, 1e-5f);
}

/* ─── decode_heads_f32 tests ────────────────────────────────────────── */

Test(flash_decode_heads, f32_basic)
{
    /* 2 query heads, 1 KV head, head_dim=4, seq_len=2.
     * kv_len = kv_heads * head_dim = 1 * 4 = 4.
     * key_layer layout: [seq_len][kv_len] = [2][4]. */
    float q[8] = {1, 0, 0, 0,  0, 1, 0, 0};  /* 2 heads * 4 dim */
    float k[8] = {1, 0, 0, 0,  1, 0, 0, 0};  /* 2 pos * 4 kv_len */
    float v[8] = {2, 0, 0, 0,  4, 0, 0, 0};
    float out[8] = {0};

    cr_assert_eq(oc_flash_attention_decode_heads_f32(q, k, v, 2, 4, 4, 2, 1, out), OC_OK);
    /* Both positions have same K for kv_head 0, so equal attention.
     * out = avg(V0, V1) = [3, 0, 0, 0] per head. */
    cr_assert_float_eq(out[0], 3.0f, 1e-5f);
    cr_assert_float_eq(out[4], 3.0f, 1e-5f);
}

Test(flash_decode_heads, f32_gqa)
{
    /* 4 query heads, 2 KV heads, head_dim=2.
     * group_size = 4/2 = 2. Heads 0,1 -> kv_head 0; Heads 2,3 -> kv_head 1.
     * kv_len = 2 * 2 = 4. */
    float q[8] = {1, 0, 0, 1, 1, 0, 0, 1};
    float k[4] = {1, 0, 0, 1};  /* 1 pos * kv_len=4 */
    float v[4] = {3, 0, 0, 5};
    float out[8] = {0};

    cr_assert_eq(oc_flash_attention_decode_heads_f32(q, k, v, 1, 2, 4, 4, 2, out), OC_OK);
    /* Head 0 -> kv_head 0: Q=[1,0], K=[1,0], score=1/sqrt(2), out=V=[3,0] */
    cr_assert_float_eq(out[0], 3.0f, 1e-5f);
    /* Head 3 -> kv_head 1: Q=[0,1], K=[0,1], score=1/sqrt(2), out=V=[0,5] */
    cr_assert_float_eq(out[7], 5.0f, 1e-5f);
}

Test(flash_decode_heads, f32_empty_seq)
{
    float q[4] = {1, 0, 0, 0};
    float k[4] = {0};
    float v[4] = {0};
    float out[4] = {0};

    cr_assert_eq(oc_flash_attention_decode_heads_f32(q, k, v, 0, 4, 4, 1, 1, out), OC_OK);
    for (int i = 0; i < 4; i++)
        cr_assert_float_eq(out[i], 0.0f, 1e-6f);
}

OC_TEST_NULL_SAFE(flash_decode_heads, f32_null_safety,
        cr_assert_neq(oc_flash_attention_decode_heads_f32(NULL, NULL, NULL, 0, 1, 1, 1, 1, NULL), OC_OK);)

Test(flash_decode_heads, f32_invalid_heads)
{
    float q[4] = {0};
    float k[4] = {0};
    float v[4] = {0};
    float out[4] = {0};
    /* num_heads not divisible by kv_heads */
    cr_assert_neq(oc_flash_attention_decode_heads_f32(q, k, v, 1, 2, 2, 3, 2, out), OC_OK);
    /* kv_heads = 0 */
    cr_assert_neq(oc_flash_attention_decode_heads_f32(q, k, v, 1, 2, 2, 2, 0, out), OC_OK);
    /* head_dim = 0 */
    cr_assert_neq(oc_flash_attention_decode_heads_f32(q, k, v, 1, 0, 0, 2, 1, out), OC_OK);
}

Test(flash_decode_heads, f32_block_boundary)
{
    /* Test with seq_len = OC_FLASH_BLOCK_SIZE + 1 = 65.
     * Ensures block tiling handles boundary correctly. */
    size_t head_dim = 4, kv_heads = 1, num_heads = 1;
    size_t kv_len = kv_heads * head_dim;
    size_t seq_len = 65;

    float q[4] = {1, 0, 0, 0};
    float k[65 * 4];
    float v[65 * 4];
    float out[4] = {0};

    for (size_t t = 0; t < seq_len; t++) {
        k[t * 4] = 1.0f; k[t * 4 + 1] = 0; k[t * 4 + 2] = 0; k[t * 4 + 3] = 0;
        v[t * 4] = (float)t; v[t * 4 + 1] = 0; v[t * 4 + 2] = 0; v[t * 4 + 3] = 0;
    }

    cr_assert_eq(oc_flash_attention_decode_heads_f32(q, k, v, seq_len, head_dim, kv_len, num_heads, kv_heads, out), OC_OK);
    /* All positions have same K, so equal attention -> average of all V[0].
     * avg(0..64) = 32.0 */
    cr_assert_float_eq(out[0], 32.0f, 0.5f);
}

/* ─── decode_heads_f16 tests ────────────────────────────────────────── */

Test(flash_decode_heads, f16_matches_f32)
{
    /* Small test: 2 heads, 1 KV head, dim=4, seq_len=3. */
    size_t head_dim = 4, kv_heads = 1, num_heads = 2;
    size_t kv_len = kv_heads * head_dim;
    size_t seq_len = 3;

    float q[8] = {0.5f, -0.3f, 0.2f, 0.1f,  -0.1f, 0.4f, 0.3f, -0.2f};
    float kf[12] = {
        0.1f, 0.2f, -0.1f, 0.3f,
        -0.2f, 0.1f, 0.4f, 0.0f,
        0.3f, -0.3f, 0.1f, 0.2f
    };
    float vf[12] = {
        0.5f, -0.5f, 0.3f, 0.1f,
        0.2f, 0.6f, -0.4f, 0.3f,
        -0.1f, 0.4f, 0.2f, -0.5f
    };

    uint16_t k16[12], v16[12];
    for (int i = 0; i < 12; i++) {
        k16[i] = oc_f32_to_f16_bits(kf[i]);
        v16[i] = oc_f32_to_f16_bits(vf[i]);
    }

    float out_f32[8] = {0};
    float out_f16[8] = {0};

    oc_flash_attention_decode_heads_f32(q, kf, vf, seq_len, head_dim, kv_len, num_heads, kv_heads, out_f32);
    oc_flash_attention_decode_heads_f16(q, k16, v16, seq_len, head_dim, kv_len, num_heads, kv_heads, out_f16);

    /* Should match within f16 precision. */
    for (int i = 0; i < 8; i++)
        cr_assert_float_eq(out_f16[i], out_f32[i], 0.02f);
}

OC_TEST_NULL_SAFE(flash_decode_heads, f16_null_safety,
        cr_assert_neq(oc_flash_attention_decode_heads_f16(NULL, NULL, NULL, 0, 1, 1, 1, 1, NULL), OC_OK);)

/* ─── prefill_f32 tests ─────────────────────────────────────────────── */

Test(flash_prefill, basic)
{
    /* 2 queries, 2 KV positions, head_dim=2.
     * Q0=[1,0] attends to K0=[1,0] and K1=[0,1].
     * Score(Q0,K0) = 1/sqrt(2), Score(Q0,K1) = 0.
     * Since softmax of [1/sqrt(2), 0] is not uniform, out is weighted toward K0. */
    float q[4] = {1, 0, 0, 1};
    float k[4] = {1, 0, 0, 1};
    float v[4] = {10, 0, 0, 20};
    float out[4] = {0};

    cr_assert_eq(oc_flash_attention_prefill_f32(q, k, v, 2, 2, 2, out), OC_OK);
    /* Q0=[1,0] should attend more to K0=[1,0] (V=[10,0]) than K1=[0,1] (V=[0,20]). */
    cr_assert(out[0] > out[1], "Q0 should favor V0");
    /* Q1=[0,1] should attend more to K1=[0,1] (V=[0,20]) than K0=[1,0] (V=[10,0]). */
    cr_assert(out[3] > out[2], "Q1 should favor V1");
}

Test(flash_prefill, single_query_single_kv)
{
    float q[2] = {1, 0};
    float k[2] = {1, 0};
    float v[2] = {3, 5};
    float out[2] = {0};

    oc_flash_attention_prefill_f32(q, k, v, 1, 1, 2, out);
    cr_assert_float_eq(out[0], 3.0f, 1e-5f);
    cr_assert_float_eq(out[1], 5.0f, 1e-5f);
}

OC_TEST_NULL_SAFE(flash_prefill, null_safety,
        cr_assert_neq(oc_flash_attention_prefill_f32(NULL, NULL, NULL, 0, 0, 1, NULL), OC_OK);)

Test(flash_prefill, block_boundary)
{
    /* seq_len = 65 to test block tiling. */
    size_t head_dim = 2;
    size_t kv_seq_len = 65;

    float q[2] = {1, 0};
    float k[130];
    float v[130];
    float out[2] = {0};

    for (size_t t = 0; t < kv_seq_len; t++) {
        k[t * 2] = 1.0f; k[t * 2 + 1] = 0.0f;
        v[t * 2] = (float)t; v[t * 2 + 1] = 0.0f;
    }

    oc_flash_attention_prefill_f32(q, k, v, 1, kv_seq_len, head_dim, out);
    /* All K equal, so uniform attention -> avg(0..64) = 32.0 */
    cr_assert_float_eq(out[0], 32.0f, 0.5f);
}

/* ─── f16 conversion tests ──────────────────────────────────────────── */

Test(f16_conv, round_trip)
{
    float values[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 100.0f, 0.001f, -1000.0f};
    for (int i = 0; i < 8; i++) {
        uint16_t h = oc_f32_to_f16_bits(values[i]);
        float restored = oc_f16_to_f32_bits(h);
        /* f16 has ~3 decimal digits precision. */
        cr_assert_float_eq(restored, values[i], fabsf(values[i]) * 0.01f + 0.01f);
    }
}

Test(f16_conv, zero)
{
    cr_assert_eq(oc_f32_to_f16_bits(0.0f), 0);
    cr_assert_float_eq(oc_f16_to_f32_bits(0), 0.0f, 1e-10f);
}

Test(f16_conv, infinity)
{
    uint16_t h = oc_f32_to_f16_bits(INFINITY);
    cr_assert((h & 0x7C00) == 0x7C00, "Inf should have exp=0x1F");
    float restored = oc_f16_to_f32_bits(h);
    cr_assert(isinf(restored));
}
