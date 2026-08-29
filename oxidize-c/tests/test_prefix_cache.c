/* test_prefix_cache.c — prefix cache tests. */
#define _POSIX_C_SOURCE 200809L
#include "framework.h"
#include "oxidize/prefix_cache.h"
#include <string.h>

Test(prefix_cache, hash_tokens)
{
    uint32_t tokens[] = {1, 2, 3};
    uint64_t h = oc_prefix_hash_tokens(tokens, 3);
    cr_assert_neq(h, 0);
    /* Same input should produce same hash. */
    cr_assert_eq(h, oc_prefix_hash_tokens(tokens, 3));
}

Test(prefix_cache, hash_incremental)
{
    uint32_t tokens[] = {1, 2, 3};
    uint64_t h_full = oc_prefix_hash_tokens(tokens, 3);
    uint64_t h_inc = oc_prefix_hash_tokens(tokens, 2);
    h_inc = oc_prefix_hash_continue(h_inc, tokens[2]);
    cr_assert_eq(h_full, h_inc);
}

Test(prefix_cache, init_lookup_empty)
{
    OcPrefixCache c;
    oc_prefix_cache_init(&c, 1024);
    cr_assert_eq(c.n_entries, 0);
    cr_assert_eq(oc_prefix_cache_lookup(&c, 123, NULL, 0), NULL);
}

Test(prefix_cache, store_lookup)
{
    OcPrefixCache c;
    oc_prefix_cache_init(&c, 1024);
    uint32_t tokens[] = { 1, 2, 3, 4, 5 };
    char *data = strdup("kv_snapshot");
    oc_prefix_cache_store(&c, 42, tokens, 5, data, strlen(data) + 1);
    const OcCachedPrefix *p = oc_prefix_cache_lookup(&c, 42, tokens, 5);
    cr_assert_not_null(p);
    cr_assert_eq(p->n_tokens, 5);
    cr_assert_str_eq((char *)p->kv_data, "kv_snapshot");
    oc_prefix_cache_clear(&c);
}

Test(prefix_cache, evict_lru)
{
    OcPrefixCache c;
    oc_prefix_cache_init(&c, 1024);
    /* Fill cache. */
    for (size_t i = 0; i < 5; i++) {
        uint32_t token = (uint32_t)i;
        oc_prefix_cache_store(&c, (uint64_t)(i + 1), &token, 1, strdup("x"), 2);
    }
    /* Evict entries with last_used < current clock. */
    size_t evicted = oc_prefix_cache_evict(&c, c.clock);
    /* The first 4 entries should be evicted (last entry was just stored). */
    cr_assert(evicted >= 4);
    oc_prefix_cache_clear(&c);
}

Test(prefix_cache, clear)
{
    OcPrefixCache c;
    oc_prefix_cache_init(&c, 1024);
    uint32_t one = 1, two = 2;
    oc_prefix_cache_store(&c, 1, &one, 1, strdup("a"), 2);
    oc_prefix_cache_store(&c, 2, &two, 1, strdup("b"), 2);
    oc_prefix_cache_clear(&c);
    cr_assert_eq(c.n_entries, 0);
    cr_assert_eq(oc_prefix_cache_lookup(&c, 1, &one, 1), NULL);
}

Test(prefix_cache, stats)
{
    OcPrefixCache c;
    oc_prefix_cache_init(&c, 1024);
    uint32_t first[] = { 1, 2, 3 };
    uint32_t second[] = { 4, 5, 6, 7, 8 };
    oc_prefix_cache_store(&c, 1, first, 3, strdup("abc"), 4);
    oc_prefix_cache_store(&c, 2, second, 5, strdup("xyz"), 4);
    cr_assert_not_null(oc_prefix_cache_lookup(&c, 1, first, 3));
    uint32_t missing[] = { 9 };
    cr_assert_null(oc_prefix_cache_lookup(&c, 9, missing, 1));
    OcPrefixCacheStats stats;
    oc_prefix_cache_stats(&c, &stats);
    cr_assert_eq(stats.n_entries, 2);
    cr_assert_eq(stats.n_hits, 1);
    cr_assert_eq(stats.n_misses, 1);
    cr_assert_eq(stats.total_kv_bytes, 8);
    oc_prefix_cache_clear(&c);
}

Test(prefix_cache, hash_collision_does_not_reuse_snapshot)
{
    OcPrefixCache c;
    oc_prefix_cache_init(&c, 1024);
    uint32_t stored[] = { 10, 20 };
    uint32_t colliding[] = { 30, 40 };
    cr_assert_eq(oc_prefix_cache_store(&c, 42, stored, 2, strdup("kv"), 3), OC_OK);
    cr_assert_null(oc_prefix_cache_lookup(&c, 42, colliding, 2));
    cr_assert_not_null(oc_prefix_cache_lookup(&c, 42, stored, 2));
    oc_prefix_cache_clear(&c);
}
