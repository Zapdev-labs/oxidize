/* test_prefix_cache.c — prefix cache tests. */
#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>
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
    cr_assert_eq(oc_prefix_cache_lookup(&c, 123), NULL);
}

Test(prefix_cache, store_lookup)
{
    OcPrefixCache c;
    oc_prefix_cache_init(&c, 1024);
    char *data = strdup("kv_snapshot");
    oc_prefix_cache_store(&c, 42, 5, data, strlen(data) + 1);
    const OcCachedPrefix *p = oc_prefix_cache_lookup(&c, 42);
    cr_assert_not_null(p);
    cr_assert_eq(p->n_tokens, 5);
    cr_assert_str_eq((char *)p->kv_data, "kv_snapshot");
}

Test(prefix_cache, evict_lru)
{
    OcPrefixCache c;
    oc_prefix_cache_init(&c, 1024);
    /* Fill cache. */
    for (size_t i = 0; i < 5; i++) {
        oc_prefix_cache_store(&c, (uint64_t)(i + 1), 1, strdup("x"), 2);
    }
    /* Evict entries with last_used < current clock. */
    size_t evicted = oc_prefix_cache_evict(&c, c.clock);
    /* The first 4 entries should be evicted (last entry was just stored). */
    cr_assert(evicted >= 4);
}

Test(prefix_cache, clear)
{
    OcPrefixCache c;
    oc_prefix_cache_init(&c, 1024);
    oc_prefix_cache_store(&c, 1, 1, strdup("a"), 2);
    oc_prefix_cache_store(&c, 2, 1, strdup("b"), 2);
    oc_prefix_cache_clear(&c);
    cr_assert_eq(c.n_entries, 0);
    cr_assert_eq(oc_prefix_cache_lookup(&c, 1), NULL);
}

Test(prefix_cache, stats)
{
    OcPrefixCache c;
    oc_prefix_cache_init(&c, 1024);
    oc_prefix_cache_store(&c, 1, 3, strdup("abc"), 4);
    oc_prefix_cache_store(&c, 2, 5, strdup("xyz"), 4);
    OcPrefixCacheStats stats;
    oc_prefix_cache_stats(&c, &stats);
    cr_assert_eq(stats.n_entries, 2);
    cr_assert_eq(stats.total_kv_bytes, 8);
}
