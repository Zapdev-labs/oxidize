/* test_kv_page.c — paged KV cache management tests. */
#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>
#include "oxidize/kv_page.h"
#include <string.h>

/* Use a small config for tests to keep memory allocations reasonable. */
static OcKvPageConfig small_config(void)
{
    OcKvPageConfig c = oc_kv_page_config_default();
    c.page_size = 4;
    c.head_dim = 2;
    c.n_heads = 2;
    c.n_layers = 2;
    c.max_pages = 8;
    return c;
}

Test(kv_page, config_default)
{
    OcKvPageConfig c = oc_kv_page_config_default();
    cr_assert_eq(c.page_size, OC_KV_PAGE_DEFAULT_PAGE_SIZE);
    cr_assert_eq(c.head_dim, OC_KV_PAGE_DEFAULT_HEAD_DIM);
    cr_assert_eq(c.n_heads, OC_KV_PAGE_DEFAULT_N_HEADS);
    cr_assert_eq(c.n_layers, OC_KV_PAGE_DEFAULT_N_LAYERS);
    cr_assert_eq(c.max_pages, OC_KV_PAGE_DEFAULT_MAX_PAGES);
}

Test(kv_page, kv_bytes_nonzero)
{
    OcKvPageConfig c = small_config();
    cr_assert_eq(oc_kv_page_kv_bytes(&c), 128u);
}

Test(kv_page, init_free)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    cr_assert_eq(oc_kv_page_init(&mgr, c), OC_OK);
    cr_assert_not_null(mgr);
    cr_assert_eq(mgr->n_total, c.max_pages);
    cr_assert_eq(mgr->n_free, c.max_pages);
    oc_kv_page_free_mgr(mgr);
}

Test(kv_page, init_invalid_config)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    c.max_pages = 0;
    cr_assert_eq(oc_kv_page_init(&mgr, c), OC_ERR_INVALID_ARG);
}

Test(kv_page, alloc_single)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    oc_kv_page_init(&mgr, c);
    uint32_t pid = 0;
    cr_assert_eq(oc_kv_page_alloc(mgr, 42, 0, &pid), OC_OK);
    cr_assert_eq(mgr->n_free, c.max_pages - 1);
    cr_assert_eq(mgr->stats.used_pages, 1u);
    cr_assert_eq(mgr->stats.allocation_count, 1u);
    oc_kv_page_free_mgr(mgr);
}

Test(kv_page, alloc_sets_seq_and_pos)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    oc_kv_page_init(&mgr, c);
    uint32_t pid = 0;
    oc_kv_page_alloc(mgr, 7, 16, &pid);
    OcKvPage *p = oc_kv_page_get(mgr, pid);
    cr_assert_not_null(p);
    cr_assert_eq(p->seq_id, 7u);
    cr_assert_eq(p->start_pos, 16u);
    cr_assert_eq(p->ref_count, 1u);
    oc_kv_page_free_mgr(mgr);
}

Test(kv_page, alloc_exhaustion)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    oc_kv_page_init(&mgr, c);
    uint32_t pid = 0;
    for (uint32_t i = 0; i < c.max_pages; i++) {
        cr_assert_eq(oc_kv_page_alloc(mgr, i, 0, &pid), OC_OK);
    }
    cr_assert_eq(oc_kv_page_alloc(mgr, 99, 0, &pid), OC_ERR_OOM);
    oc_kv_page_free_mgr(mgr);
}

Test(kv_page, free_returns_to_pool)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    oc_kv_page_init(&mgr, c);
    uint32_t pid = 0;
    oc_kv_page_alloc(mgr, 1, 0, &pid);
    cr_assert_eq(mgr->n_free, c.max_pages - 1);
    cr_assert_eq(oc_kv_page_free(mgr, pid), OC_OK);
    cr_assert_eq(mgr->n_free, c.max_pages);
    cr_assert_eq(mgr->stats.used_pages, 0u);
    cr_assert_eq(mgr->stats.eviction_count, 1u);
    oc_kv_page_free_mgr(mgr);
}

Test(kv_page, free_invalid_page)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    oc_kv_page_init(&mgr, c);
    cr_assert_eq(oc_kv_page_free(mgr, 999), OC_ERR_INVALID_ARG);
    /* page 0 is free (never allocated) */
    cr_assert_eq(oc_kv_page_free(mgr, 0), OC_ERR_INVALID_ARG);
    oc_kv_page_free_mgr(mgr);
}

Test(kv_page, get_invalid_returns_null)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    oc_kv_page_init(&mgr, c);
    cr_assert_null(oc_kv_page_get(mgr, 999));
    /* page 0 is free so should also return NULL */
    cr_assert_null(oc_kv_page_get(mgr, 0));
    oc_kv_page_free_mgr(mgr);
}

Test(kv_page, write_read_kv)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    oc_kv_page_init(&mgr, c);
    uint32_t pid = 0;
    oc_kv_page_alloc(mgr, 1, 0, &pid);
    /* head_dim=2, n_heads=2 -> 4 floats per slot */
    float key[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float val[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    cr_assert_eq(oc_kv_page_write_kv(mgr, pid, 0, 0, key, val), OC_OK);
    float out_key[4] = {0};
    float out_val[4] = {0};
    cr_assert_eq(oc_kv_page_read_kv(mgr, pid, 0, 0, out_key, out_val), OC_OK);
    cr_assert_arr_eq(out_key, key, sizeof(key));
    cr_assert_arr_eq(out_val, val, sizeof(val));
    oc_kv_page_free_mgr(mgr);
}

Test(kv_page, write_read_multiple_layers)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    oc_kv_page_init(&mgr, c);
    uint32_t pid = 0;
    oc_kv_page_alloc(mgr, 1, 0, &pid);
    float key1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float val1[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float key2[4] = {9.0f, 10.0f, 11.0f, 12.0f};
    float val2[4] = {13.0f, 14.0f, 15.0f, 16.0f};
    cr_assert_eq(oc_kv_page_write_kv(mgr, pid, 0, 0, key1, val1), OC_OK);
    cr_assert_eq(oc_kv_page_write_kv(mgr, pid, 1, 0, key2, val2), OC_OK);
    float out_key[4] = {0};
    float out_val[4] = {0};
    cr_assert_eq(oc_kv_page_read_kv(mgr, pid, 0, 0, out_key, out_val), OC_OK);
    cr_assert_arr_eq(out_key, key1, sizeof(key1));
    cr_assert_arr_eq(out_val, val1, sizeof(val1));
    cr_assert_eq(oc_kv_page_read_kv(mgr, pid, 1, 0, out_key, out_val), OC_OK);
    cr_assert_arr_eq(out_key, key2, sizeof(key2));
    cr_assert_arr_eq(out_val, val2, sizeof(val2));
    oc_kv_page_free_mgr(mgr);
}

Test(kv_page, write_read_multiple_tokens)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    oc_kv_page_init(&mgr, c);
    uint32_t pid = 0;
    oc_kv_page_alloc(mgr, 1, 0, &pid);
    float key0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float val0[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float key1[4] = {9.0f, 10.0f, 11.0f, 12.0f};
    float val1[4] = {13.0f, 14.0f, 15.0f, 16.0f};
    cr_assert_eq(oc_kv_page_write_kv(mgr, pid, 0, 0, key0, val0), OC_OK);
    cr_assert_eq(oc_kv_page_write_kv(mgr, pid, 0, 1, key1, val1), OC_OK);
    float out_key[4] = {0};
    float out_val[4] = {0};
    cr_assert_eq(oc_kv_page_read_kv(mgr, pid, 0, 0, out_key, out_val), OC_OK);
    cr_assert_arr_eq(out_key, key0, sizeof(key0));
    cr_assert_eq(oc_kv_page_read_kv(mgr, pid, 0, 1, out_key, out_val), OC_OK);
    cr_assert_arr_eq(out_key, key1, sizeof(key1));
    oc_kv_page_free_mgr(mgr);
}

Test(kv_page, write_invalid_layer)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    oc_kv_page_init(&mgr, c);
    uint32_t pid = 0;
    oc_kv_page_alloc(mgr, 1, 0, &pid);
    float key[4] = {0};
    float val[4] = {0};
    /* n_layers=2 so layer 2 is out of range */
    cr_assert_eq(oc_kv_page_write_kv(mgr, pid, 2, 0, key, val), OC_ERR_INVALID_ARG);
    oc_kv_page_free_mgr(mgr);
}

Test(kv_page, write_invalid_token_offset)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    oc_kv_page_init(&mgr, c);
    uint32_t pid = 0;
    oc_kv_page_alloc(mgr, 1, 0, &pid);
    float key[4] = {0};
    float val[4] = {0};
    /* page_size=4 so token_offset 4 is out of range */
    cr_assert_eq(oc_kv_page_write_kv(mgr, pid, 0, 4, key, val), OC_ERR_INVALID_ARG);
    oc_kv_page_free_mgr(mgr);
}

Test(kv_page, get_seq_pages)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    oc_kv_page_init(&mgr, c);
    uint32_t p1 = 0, p2 = 0, p3 = 0;
    oc_kv_page_alloc(mgr, 10, 0, &p1);
    oc_kv_page_alloc(mgr, 10, 4, &p2);
    oc_kv_page_alloc(mgr, 20, 0, &p3);
    uint32_t out[8] = {0};
    size_t n = 0;
    cr_assert_eq(oc_kv_page_get_seq_pages(mgr, 10, out, 8, &n), OC_OK);
    cr_assert_eq(n, 2u);
    /* Free pages should not be returned. */
    oc_kv_page_free(mgr, p1);
    cr_assert_eq(oc_kv_page_get_seq_pages(mgr, 10, out, 8, &n), OC_OK);
    cr_assert_eq(n, 1u);
    oc_kv_page_free_mgr(mgr);
}

Test(kv_page, stats)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    oc_kv_page_init(&mgr, c);
    uint32_t p1 = 0, p2 = 0;
    oc_kv_page_alloc(mgr, 1, 0, &p1);
    oc_kv_page_alloc(mgr, 2, 0, &p2);
    OcKvPageStats stats;
    cr_assert_eq(oc_kv_page_stats(mgr, &stats), OC_OK);
    cr_assert_eq(stats.total_pages, c.max_pages);
    cr_assert_eq(stats.free_pages, c.max_pages - 2);
    cr_assert_eq(stats.used_pages, 2u);
    cr_assert_eq(stats.allocation_count, 2u);
    cr_assert_eq(stats.eviction_count, 0u);
    oc_kv_page_free(mgr, p1);
    cr_assert_eq(oc_kv_page_stats(mgr, &stats), OC_OK);
    cr_assert_eq(stats.eviction_count, 1u);
    cr_assert_eq(stats.used_pages, 1u);
    oc_kv_page_free_mgr(mgr);
}

Test(kv_page, reuse_after_free)
{
    OcKvPageManager *mgr = NULL;
    OcKvPageConfig c = small_config();
    oc_kv_page_init(&mgr, c);
    uint32_t p1 = 0;
    oc_kv_page_alloc(mgr, 1, 0, &p1);
    oc_kv_page_free(mgr, p1);
    /* After freeing, we should be able to allocate again. */
    uint32_t p2 = 0;
    cr_assert_eq(oc_kv_page_alloc(mgr, 2, 0, &p2), OC_OK);
    /* K/V storage should already be allocated (lazy init persists). */
    OcKvPage *page = oc_kv_page_get(mgr, p2);
    cr_assert_not_null(page);
    cr_assert_not_null(page->layer_k);
    cr_assert_not_null(page->layer_v);
    cr_assert_eq(page->seq_id, 2u);
    oc_kv_page_free_mgr(mgr);
}
