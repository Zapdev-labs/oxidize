/* test_seq_kv.c — Per-sequence KV buffer tests. */
#include "framework.h"
#include "oxidize/seq_kv.h"
#include <string.h>

Test(seqkv, init)
{
    OcSeqKv kv;
    cr_assert_eq(oc_seq_kv_init(&kv, 32, 2048, 128), OC_OK);
    cr_assert_not_null(kv.key);
    cr_assert_not_null(kv.value);
    cr_assert_eq(kv.len, 0);
    cr_assert_eq(kv.capacity_tokens, 2048);
    cr_assert_eq(kv.kv_layer_count, 32);
    cr_assert_eq(kv.kv_len, 128);
    oc_seq_kv_free(&kv);
}

OC_TEST_NULL_SAFE(seqkv, init_null,
        cr_assert_neq(oc_seq_kv_init(NULL, 1, 1, 1), OC_OK);)

Test(seqkv, init_bad_args)
{
    OcSeqKv kv;
    cr_assert_neq(oc_seq_kv_init(&kv, 0, 1, 1), OC_OK);
    cr_assert_neq(oc_seq_kv_init(&kv, 1, 0, 1), OC_OK);
    cr_assert_neq(oc_seq_kv_init(&kv, 1, 1, 0), OC_OK);
}

Test(seqkv, put_get)
{
    OcSeqKv kv;
    oc_seq_kv_init(&kv, 2, 4, 3);
    float k[3] = {1.0f, 2.0f, 3.0f};
    float v[3] = {4.0f, 5.0f, 6.0f};
    cr_assert_eq(oc_seq_kv_put(&kv, 0, 0, k, v), OC_OK);

    const float *ok, *ov;
    cr_assert_eq(oc_seq_kv_get(&kv, 0, 0, &ok, &ov), OC_OK);
    cr_assert_float_eq(ok[0], 1.0f, 0.001f);
    cr_assert_float_eq(ok[2], 3.0f, 0.001f);
    cr_assert_float_eq(ov[0], 4.0f, 0.001f);
    cr_assert_float_eq(ov[2], 6.0f, 0.001f);
    oc_seq_kv_free(&kv);
}

Test(seqkv, put_get_different_layer)
{
    OcSeqKv kv;
    oc_seq_kv_init(&kv, 2, 4, 3);
    float k1[3] = {1, 2, 3};
    float v1[3] = {4, 5, 6};
    float k2[3] = {7, 8, 9};
    float v2[3] = {10, 11, 12};
    oc_seq_kv_put(&kv, 0, 0, k1, v1);
    oc_seq_kv_put(&kv, 1, 0, k2, v2);

    const float *ok, *ov;
    oc_seq_kv_get(&kv, 0, 0, &ok, &ov);
    cr_assert_float_eq(ok[0], 1.0f, 0.001f);
    oc_seq_kv_get(&kv, 1, 0, &ok, &ov);
    cr_assert_float_eq(ok[0], 7.0f, 0.001f);
    oc_seq_kv_free(&kv);
}

Test(seqkv, get_bad_layer)
{
    OcSeqKv kv;
    oc_seq_kv_init(&kv, 2, 4, 3);
    const float *ok, *ov;
    cr_assert_neq(oc_seq_kv_get(&kv, 99, 0, &ok, &ov), OC_OK);
    oc_seq_kv_free(&kv);
}

Test(seqkv, get_bad_pos)
{
    OcSeqKv kv;
    oc_seq_kv_init(&kv, 2, 4, 3);
    const float *ok, *ov;
    cr_assert_neq(oc_seq_kv_get(&kv, 0, 99, &ok, &ov), OC_OK);
    oc_seq_kv_free(&kv);
}

Test(seqkv, advance)
{
    OcSeqKv kv;
    oc_seq_kv_init(&kv, 2, 4, 3);
    cr_assert_eq(kv.len, 0);
    oc_seq_kv_advance(&kv, 3);
    cr_assert_eq(kv.len, 3);
    oc_seq_kv_free(&kv);
}

Test(seqkv, advance_overflow)
{
    OcSeqKv kv;
    oc_seq_kv_init(&kv, 2, 4, 3);
    oc_seq_kv_advance(&kv, 100);
    cr_assert_eq(kv.len, 4);
    oc_seq_kv_free(&kv);
}

Test(seqkv, truncate)
{
    OcSeqKv kv;
    oc_seq_kv_init(&kv, 2, 4, 3);
    oc_seq_kv_advance(&kv, 3);
    oc_seq_kv_truncate(&kv, 1);
    cr_assert_eq(kv.len, 1);
    oc_seq_kv_free(&kv);
}

Test(seqkv, truncate_noop)
{
    OcSeqKv kv;
    oc_seq_kv_init(&kv, 2, 4, 3);
    oc_seq_kv_advance(&kv, 2);
    oc_seq_kv_truncate(&kv, 5);
    cr_assert_eq(kv.len, 2);
    oc_seq_kv_free(&kv);
}

Test(seqkv, clear)
{
    OcSeqKv kv;
    oc_seq_kv_init(&kv, 2, 4, 3);
    float k[3] = {1, 2, 3};
    float v[3] = {4, 5, 6};
    oc_seq_kv_put(&kv, 0, 0, k, v);
    oc_seq_kv_advance(&kv, 1);
    oc_seq_kv_clear(&kv);
    cr_assert_eq(kv.len, 0);
    const float *ok, *ov;
    oc_seq_kv_get(&kv, 0, 0, &ok, &ov);
    cr_assert_float_eq(ok[0], 0.0f, 0.001f);
    oc_seq_kv_free(&kv);
}

Test(seqkv, size_bytes)
{
    OcSeqKv kv;
    oc_seq_kv_init(&kv, 32, 2048, 128);
    size_t sz = oc_seq_kv_size_bytes(&kv);
    cr_assert_gt(sz, 0);
    /* 32 * 2048 * 128 * 4 * 2 (K+V) = 67108864 */
    cr_assert_eq(sz, 32 * 2048 * 128 * sizeof(float) * 2);
    oc_seq_kv_free(&kv);
}

Test(seqkv, is_empty)
{
    OcSeqKv kv;
    oc_seq_kv_init(&kv, 1, 1, 1);
    cr_assert(oc_seq_kv_is_empty(&kv));
    oc_seq_kv_advance(&kv, 1);
    cr_assert(!oc_seq_kv_is_empty(&kv));
    oc_seq_kv_free(&kv);
}

Test(seqkv, capacity)
{
    OcSeqKv kv;
    oc_seq_kv_init(&kv, 2, 2048, 128);
    cr_assert_eq(oc_seq_kv_capacity(&kv), 2048);
    oc_seq_kv_free(&kv);
}

OC_TEST_NULL_SAFE(seqkv, free_null,
        oc_seq_kv_free(NULL);)
