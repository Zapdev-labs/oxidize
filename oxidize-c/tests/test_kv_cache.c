/* test_kv_cache.c — simple per-layer KV cache tests. */
#define _POSIX_C_SOURCE 200809L
#include "framework.h"
#include "oxidize/kv_cache.h"
#include <stdlib.h>
#include <string.h>

/* Use a small config for tests to keep memory allocations reasonable. */
static OcKvCacheConfig small_config(void)
{
    OcKvCacheConfig c;
    oc_kv_cache_config_init(&c);
    c.n_layers    = 2;
    c.n_heads     = 2;
    c.head_dim    = 4;
    c.max_seq_len = 8;
    c.dtype       = OC_KV_CACHE_DTYPE_F32;
    return c;
}

/* Per-token row size in floats: n_heads * head_dim. */
static size_t row_size(const OcKvCacheConfig *c)
{
    return (size_t)c->n_heads * (size_t)c->head_dim;
}

Test(kv_cache, config_default)
{
    OcKvCacheConfig c;
    oc_kv_cache_config_init(&c);
    cr_assert_eq(c.n_layers, OC_KV_CACHE_DEFAULT_N_LAYERS);
    cr_assert_eq(c.n_heads, OC_KV_CACHE_DEFAULT_N_HEADS);
    cr_assert_eq(c.head_dim, OC_KV_CACHE_DEFAULT_HEAD_DIM);
    cr_assert_eq(c.max_seq_len, OC_KV_CACHE_DEFAULT_MAX_SEQ_LEN);
    cr_assert_eq(c.dtype, OC_KV_CACHE_DTYPE_F32);
}

Test(kv_cache, config_init_null_is_safe)
{
    /* Should not crash on NULL. */
    oc_kv_cache_config_init(NULL);
    cr_assert(true);
}

Test(kv_cache, init_free)
{
    OcKvCache cache;
    OcKvCacheConfig c = small_config();
    cr_assert_eq(oc_kv_cache_init(&cache, &c), OC_OK);
    cr_assert_not_null(cache.k_data);
    cr_assert_not_null(cache.v_data);
    cr_assert_eq(cache.n_tokens, 0u);
    cr_assert_eq(cache.capacity, c.max_seq_len);
    oc_kv_cache_free(&cache);
    cr_assert_null(cache.k_data);
    cr_assert_null(cache.v_data);
    cr_assert_eq(cache.n_tokens, 0u);
}

Test(kv_cache, init_null_args)
{
    OcKvCacheConfig c = small_config();
    cr_assert_eq(oc_kv_cache_init(NULL, &c), OC_ERR_INVALID_ARG);
    OcKvCache cache;
    cr_assert_eq(oc_kv_cache_init(&cache, NULL), OC_ERR_INVALID_ARG);
}

Test(kv_cache, init_invalid_config)
{
    OcKvCache cache;
    OcKvCacheConfig c = small_config();
    c.n_layers = 0;
    cr_assert_eq(oc_kv_cache_init(&cache, &c), OC_ERR_INVALID_ARG);

    c = small_config();
    c.n_heads = 0;
    cr_assert_eq(oc_kv_cache_init(&cache, &c), OC_ERR_INVALID_ARG);

    c = small_config();
    c.head_dim = 0;
    cr_assert_eq(oc_kv_cache_init(&cache, &c), OC_ERR_INVALID_ARG);

    c = small_config();
    c.max_seq_len = 0;
    cr_assert_eq(oc_kv_cache_init(&cache, &c), OC_ERR_INVALID_ARG);
}

Test(kv_cache, init_invalid_dtype)
{
    OcKvCache cache;
    OcKvCacheConfig c = small_config();
    c.dtype = 99u;
    cr_assert_eq(oc_kv_cache_init(&cache, &c), OC_ERR_INVALID_ARG);
}

Test(kv_cache, append_and_get)
{
    OcKvCache cache;
    OcKvCacheConfig c = small_config();
    oc_kv_cache_init(&cache, &c);
    /* row_size = 2 * 4 = 8 floats. */
    float k[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float v[8] = {9, 10, 11, 12, 13, 14, 15, 16};
    cr_assert_eq(oc_kv_cache_append(&cache, 0, k, v, 1), OC_OK);
    cr_assert_eq(cache.n_tokens, 1u);
    const float *pk = NULL;
    const float *pv = NULL;
    cr_assert_eq(oc_kv_cache_get(&cache, 0, 0, &pk, &pv), OC_OK);
    cr_assert_not_null(pk);
    cr_assert_not_null(pv);
    cr_assert_arr_eq(pk, k, sizeof(k));
    cr_assert_arr_eq(pv, v, sizeof(v));
    oc_kv_cache_free(&cache);
}

Test(kv_cache, append_multiple_tokens)
{
    OcKvCache cache;
    OcKvCacheConfig c = small_config();
    oc_kv_cache_init(&cache, &c);
    /* Append 3 tokens at once: k and v each have 3 * 8 = 24 floats. */
    float k[24] = {0};
    float v[24] = {0};
    for (int i = 0; i < 24; i++) {
        k[i] = (float)(i + 1);
        v[i] = (float)(i + 100);
    }
    cr_assert_eq(oc_kv_cache_append(&cache, 0, k, v, 3), OC_OK);
    cr_assert_eq(cache.n_tokens, 3u);
    /* Verify each token. */
    size_t rs = row_size(&c);
    for (uint32_t t = 0; t < 3; t++) {
        const float *pk = NULL;
        const float *pv = NULL;
        cr_assert_eq(oc_kv_cache_get(&cache, 0, t, &pk, &pv), OC_OK);
        cr_assert_arr_eq(pk, k + t * rs, rs * sizeof(float));
        cr_assert_arr_eq(pv, v + t * rs, rs * sizeof(float));
    }
    oc_kv_cache_free(&cache);
}

Test(kv_cache, append_multiple_layers)
{
    OcKvCache cache;
    OcKvCacheConfig c = small_config();
    oc_kv_cache_init(&cache, &c);
    float k0[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float v0[8] = {9, 10, 11, 12, 13, 14, 15, 16};
    float k1[8] = {17, 18, 19, 20, 21, 22, 23, 24};
    float v1[8] = {25, 26, 27, 28, 29, 30, 31, 32};
    cr_assert_eq(oc_kv_cache_append(&cache, 0, k0, v0, 1), OC_OK);
    cr_assert_eq(oc_kv_cache_append(&cache, 1, k1, v1, 1), OC_OK);
    /* n_tokens only advances on layer 0. */
    cr_assert_eq(cache.n_tokens, 1u);
    const float *pk0 = NULL, *pv0 = NULL, *pk1 = NULL, *pv1 = NULL;
    cr_assert_eq(oc_kv_cache_get(&cache, 0, 0, &pk0, &pv0), OC_OK);
    cr_assert_eq(oc_kv_cache_get(&cache, 1, 0, &pk1, &pv1), OC_OK);
    cr_assert_arr_eq(pk0, k0, sizeof(k0));
    cr_assert_arr_eq(pv0, v0, sizeof(v0));
    cr_assert_arr_eq(pk1, k1, sizeof(k1));
    cr_assert_arr_eq(pv1, v1, sizeof(v1));
    /* Layer 0 and layer 1 pointers must be distinct. */
    cr_assert_neq(pk0, pk1);
    cr_assert_neq(pv0, pv1);
    oc_kv_cache_free(&cache);
}

Test(kv_cache, append_exceeds_capacity)
{
    OcKvCache cache;
    OcKvCacheConfig c = small_config();
    oc_kv_cache_init(&cache, &c);
    /* capacity is 8; append 9 tokens at once. */
    size_t rs = row_size(&c);
    float *k = malloc(9 * rs * sizeof(float));
    float *v = malloc(9 * rs * sizeof(float));
    cr_assert_eq(oc_kv_cache_append(&cache, 0, k, v, 9), OC_ERR_INVALID_ARG);
    cr_assert_eq(cache.n_tokens, 0u);
    free(k);
    free(v);
    oc_kv_cache_free(&cache);
}

Test(kv_cache, append_invalid_layer)
{
    OcKvCache cache;
    OcKvCacheConfig c = small_config();
    oc_kv_cache_init(&cache, &c);
    float k[8] = {0};
    float v[8] = {0};
    /* n_layers=2 so layer 2 is out of range. */
    cr_assert_eq(oc_kv_cache_append(&cache, 2, k, v, 1), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_kv_cache_append(&cache, 0, NULL, v, 1), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_kv_cache_append(&cache, 0, k, NULL, 1), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_kv_cache_append(NULL, 0, k, v, 1), OC_ERR_INVALID_ARG);
    oc_kv_cache_free(&cache);
}

Test(kv_cache, get_invalid_pos)
{
    OcKvCache cache;
    OcKvCacheConfig c = small_config();
    oc_kv_cache_init(&cache, &c);
    const float *pk = NULL;
    const float *pv = NULL;
    /* No tokens appended yet. */
    cr_assert_eq(oc_kv_cache_get(&cache, 0, 0, &pk, &pv), OC_ERR_INVALID_ARG);
    /* NULL args. */
    cr_assert_eq(oc_kv_cache_get(&cache, 0, 0, NULL, &pv), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_kv_cache_get(&cache, 0, 0, &pk, NULL), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_kv_cache_get(NULL, 0, 0, &pk, &pv), OC_ERR_INVALID_ARG);
    oc_kv_cache_free(&cache);
}

Test(kv_cache, get_invalid_layer)
{
    OcKvCache cache;
    OcKvCacheConfig c = small_config();
    oc_kv_cache_init(&cache, &c);
    float k[8] = {0};
    float v[8] = {0};
    oc_kv_cache_append(&cache, 0, k, v, 1);
    const float *pk = NULL;
    const float *pv = NULL;
    cr_assert_eq(oc_kv_cache_get(&cache, 2, 0, &pk, &pv), OC_ERR_INVALID_ARG);
    oc_kv_cache_free(&cache);
}

Test(kv_cache, clear)
{
    OcKvCache cache;
    OcKvCacheConfig c = small_config();
    oc_kv_cache_init(&cache, &c);
    float k[8] = {1};
    float v[8] = {2};
    oc_kv_cache_append(&cache, 0, k, v, 1);
    cr_assert_eq(cache.n_tokens, 1u);
    oc_kv_cache_clear(&cache);
    cr_assert_eq(cache.n_tokens, 0u);
    /* Capacity is preserved. */
    cr_assert_eq(cache.capacity, c.max_seq_len);
    oc_kv_cache_free(&cache);
}

Test(kv_cache, truncate)
{
    OcKvCache cache;
    OcKvCacheConfig c = small_config();
    oc_kv_cache_init(&cache, &c);
    /* Append 4 tokens. */
    size_t rs = row_size(&c);
    float *k = calloc(4 * rs, sizeof(float));
    float *v = calloc(4 * rs, sizeof(float));
    oc_kv_cache_append(&cache, 0, k, v, 4);
    cr_assert_eq(cache.n_tokens, 4u);
    /* Truncate to 2. */
    cr_assert_eq(oc_kv_cache_truncate(&cache, 2), OC_OK);
    cr_assert_eq(cache.n_tokens, 2u);
    /* Truncate to a value >= n_tokens but <= capacity is a no-op. */
    cr_assert_eq(oc_kv_cache_truncate(&cache, 4), OC_OK);
    cr_assert_eq(cache.n_tokens, 2u);
    free(k);
    free(v);
    oc_kv_cache_free(&cache);
}

Test(kv_cache, truncate_invalid)
{
    OcKvCache cache;
    OcKvCacheConfig c = small_config();
    oc_kv_cache_init(&cache, &c);
    /* n > capacity (8) is invalid. */
    cr_assert_eq(oc_kv_cache_truncate(&cache, 100), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_kv_cache_truncate(NULL, 0), OC_ERR_INVALID_ARG);
    oc_kv_cache_free(&cache);
}

OC_TEST_NULL_SAFE(kv_cache, accessors_on_null,
        cr_assert_eq(oc_kv_cache_n_tokens(NULL), 0u);
        cr_assert_eq(oc_kv_cache_capacity(NULL), 0u);
        cr_assert_eq(oc_kv_cache_size_bytes(NULL), 0u);)

Test(kv_cache, size_bytes)
{
    OcKvCache cache;
    OcKvCacheConfig c = small_config();
    oc_kv_cache_init(&cache, &c);
    /* K + V = 2 * n_layers * max_seq_len * n_heads * head_dim * sizeof(float)
     *       = 2 * 2 * 8 * 2 * 4 * 4 = 1024. */
    cr_assert_eq(oc_kv_cache_size_bytes(&cache), 1024u);
    oc_kv_cache_free(&cache);
    cr_assert_eq(oc_kv_cache_size_bytes(&cache), 0u);
}

Test(kv_cache, free_null_is_safe)
{
    oc_kv_cache_free(NULL);
    cr_assert(true);
}

Test(kv_cache, append_zero_tokens)
{
    OcKvCache cache;
    OcKvCacheConfig c = small_config();
    oc_kv_cache_init(&cache, &c);
    float k[1] = {0};
    float v[1] = {0};
    cr_assert_eq(oc_kv_cache_append(&cache, 0, k, v, 0), OC_OK);
    cr_assert_eq(cache.n_tokens, 0u);
    oc_kv_cache_free(&cache);
}
