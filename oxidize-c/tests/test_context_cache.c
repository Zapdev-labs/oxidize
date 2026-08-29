/* test_context_cache.c — context cache tests. */
#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>
#include "oxidize/context_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* ------------------------------------------------------------------ Helpers ------------------------------------------------------------------ */

static OcContextCacheConfig make_mem_cfg(void) {
    OcContextCacheConfig c = oc_context_cache_config_default();
    c.cache_dir       = NULL;
    c.max_entries     = 4;
    c.max_size_bytes  = 1ULL << 20;   /* 1 MiB */
    c.ttl_seconds     = 0;
    return c;
}

static OcContextCacheConfig make_disk_cfg(const char *dir) {
    OcContextCacheConfig c = oc_context_cache_config_default();
    c.cache_dir       = dir;
    c.max_entries     = 4;
    c.max_size_bytes  = 1ULL << 20;
    c.ttl_seconds     = 0;
    return c;
}

/* Build a small fake KV buffer; caller frees. */
static uint8_t *make_kv(const char *tag, uint64_t *out_size) {
    size_t n = strlen(tag) + 1;
    uint8_t *buf = (uint8_t *)malloc(n);
    cr_assert_not_null(buf);
    memcpy(buf, tag, n);
    *out_size = (uint64_t)n;
    return buf;
}

/* ------------------------------------------------------------------ config + init/free ------------------------------------------------------------------ */

Test(context_cache, config_default)
{
    OcContextCacheConfig c = oc_context_cache_config_default();
    cr_assert_null(c.cache_dir);
    cr_assert_eq(c.max_entries, OC_CONTEXT_CACHE_DEFAULT_MAX_ENTRIES);
    cr_assert_eq(c.max_size_bytes, OC_CONTEXT_CACHE_DEFAULT_MAX_SIZE);
    cr_assert_eq(c.ttl_seconds, OC_CONTEXT_CACHE_DEFAULT_TTL);
}

Test(context_cache, init_free)
{
    OcContextCacheConfig c = make_mem_cfg();
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);
    OcContextCacheStats s;
    oc_context_cache_get_stats(cc, &s);
    cr_assert_eq(s.n_entries, 0);
    cr_assert_eq(s.total_bytes, 0);
    cr_assert_eq(s.hits, 0);
    cr_assert_eq(s.misses, 0);
    cr_assert_eq(s.evictions, 0);
    cr_assert_eq(s.hit_rate, 0.0);
    oc_context_cache_free(cc);
}

Test(context_cache, init_null_config)
{
    cr_assert_null(oc_context_cache_init(NULL));
}

Test(context_cache, init_creates_cache_dir)
{
    const char *dir = "/tmp/ox_ctx_cache_init_test";
    (void)system("rm -rf /tmp/ox_ctx_cache_init_test");
    OcContextCacheConfig c = make_disk_cfg(dir);
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);
    struct stat st;
    cr_assert_eq(stat(dir, &st), 0, "cache_dir should exist");
    oc_context_cache_free(cc);
    (void)system("rm -rf /tmp/ox_ctx_cache_init_test");
}

Test(context_cache, free_null_safe)
{
    oc_context_cache_free(NULL);
}

/* ------------------------------------------------------------------ store / load roundtrip ------------------------------------------------------------------ */

Test(context_cache, store_load_roundtrip_mem)
{
    OcContextCacheConfig c = make_mem_cfg();
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);

    uint64_t sz = 0;
    uint8_t *kv = make_kv("hello-kv", &sz);
    uint64_t mh = oc_context_cache_model_hash(12345, 7);
    OcError e = oc_context_cache_store(cc, "sess1", mh, 4, 8, 4, 64, kv, sz);
    cr_assert_eq(e, OC_OK);

    OcContextCacheEntry out;
    bool found = false;
    e = oc_context_cache_load(cc, "sess1", mh, &out, &found);
    cr_assert_eq(e, OC_OK);
    cr_assert(found);
    cr_assert_str_eq(out.session_id, "sess1");
    cr_assert_eq(out.model_hash, mh);
    cr_assert_eq(out.n_tokens, 4);
    cr_assert_eq(out.n_layers, 8);
    cr_assert_eq(out.n_head_kv, 4);
    cr_assert_eq(out.head_dim, 64);
    cr_assert_eq(out.size_bytes, sz);
    cr_assert_not_null(out.data);
    cr_assert_str_eq((char *)out.data, "hello-kv");
    free(out.data);
    oc_context_cache_free(cc);
}

Test(context_cache, store_load_roundtrip_disk)
{
    const char *dir = "/tmp/ox_ctx_cache_disk_rt";
    (void)system("rm -rf /tmp/ox_ctx_cache_disk_rt");
    OcContextCacheConfig c = make_disk_cfg(dir);
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);

    uint64_t sz = 0;
    uint8_t *kv = make_kv("disk-kv-payload", &sz);
    uint64_t mh = oc_context_cache_model_hash(9999, 3);
    OcError e = oc_context_cache_store(cc, "diskSess", mh, 10, 6, 2, 32, kv, sz);
    cr_assert_eq(e, OC_OK);

    /* Verify the file exists. */
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, "diskSess");
    struct stat st;
    cr_assert_eq(stat(path, &st), 0, "disk file should exist");

    /* Load it back. */
    OcContextCacheEntry out;
    bool found = false;
    e = oc_context_cache_load(cc, "diskSess", mh, &out, &found);
    cr_assert_eq(e, OC_OK);
    cr_assert(found);
    cr_assert_str_eq(out.session_id, "diskSess");
    cr_assert_eq(out.model_hash, mh);
    cr_assert_str_eq((char *)out.data, "disk-kv-payload");
    free(out.data);
    oc_context_cache_free(cc);
    (void)system("rm -rf /tmp/ox_ctx_cache_disk_rt");
}

Test(context_cache, load_miss)
{
    OcContextCacheConfig c = make_mem_cfg();
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);

    OcContextCacheEntry out;
    bool found = true;
    OcError e = oc_context_cache_load(cc, "nope", 0, &out, &found);
    cr_assert_eq(e, OC_OK);
    cr_assert(!found);
    oc_context_cache_free(cc);
}

Test(context_cache, load_model_hash_mismatch)
{
    OcContextCacheConfig c = make_mem_cfg();
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);

    uint64_t sz = 0;
    uint8_t *kv = make_kv("x", &sz);
    uint64_t mh = oc_context_cache_model_hash(1, 1);
    cr_assert_eq(oc_context_cache_store(cc, "s", mh, 1, 1, 1, 1, kv, sz), OC_OK);

    OcContextCacheEntry out;
    bool found = false;
    /* Different model hash → miss. */
    OcError e = oc_context_cache_load(cc, "s", mh + 1, &out, &found);
    cr_assert_eq(e, OC_OK);
    cr_assert(!found);
    oc_context_cache_free(cc);
}

Test(context_cache, load_null_args)
{
    OcContextCacheConfig c = make_mem_cfg();
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);
    bool found = true;
    cr_assert_eq(oc_context_cache_load(cc, NULL, 0, NULL, &found), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_context_cache_load(NULL, "x", 0, NULL, &found), OC_ERR_INVALID_ARG);
    cr_assert(!found);
    oc_context_cache_free(cc);
}

Test(context_cache, store_replaces_existing)
{
    OcContextCacheConfig c = make_mem_cfg();
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);

    uint64_t sz1 = 0, sz2 = 0;
    uint8_t *kv1 = make_kv("v1", &sz1);
    uint8_t *kv2 = make_kv("v2-longer", &sz2);
    uint64_t mh = oc_context_cache_model_hash(1, 1);
    cr_assert_eq(oc_context_cache_store(cc, "s", mh, 1, 1, 1, 1, kv1, sz1), OC_OK);
    cr_assert_eq(oc_context_cache_store(cc, "s", mh, 1, 1, 1, 1, kv2, sz2), OC_OK);

    OcContextCacheEntry out;
    bool found = false;
    cr_assert_eq(oc_context_cache_load(cc, "s", mh, &out, &found), OC_OK);
    cr_assert(found);
    cr_assert_str_eq((char *)out.data, "v2-longer");
    free(out.data);

    OcContextCacheStats s;
    oc_context_cache_get_stats(cc, &s);
    cr_assert_eq(s.n_entries, 1, "replace should not duplicate");
    oc_context_cache_free(cc);
}

/* ------------------------------------------------------------------ LRU eviction ------------------------------------------------------------------ */

Test(context_cache, lru_evict_on_max_entries)
{
    OcContextCacheConfig c = make_mem_cfg();
    c.max_entries = 3;
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);

    uint64_t mh = oc_context_cache_model_hash(1, 1);
    for (int i = 0; i < 5; i++) {
        char sid[16];
        snprintf(sid, sizeof(sid), "s%d", i);
        uint64_t sz = 0;
        uint8_t *kv = make_kv(sid, &sz);
        OcError e = oc_context_cache_store(cc, sid, mh, 1, 1, 1, 1, kv, sz);
        cr_assert_eq(e, OC_OK);
    }
    OcContextCacheStats s;
    oc_context_cache_get_stats(cc, &s);
    cr_assert_leq(s.n_entries, 3, "should cap at max_entries");
    cr_assert(s.evictions >= 2, "should have evicted at least 2");
    oc_context_cache_free(cc);
}

Test(context_cache, lru_evict_on_max_size)
{
    OcContextCacheConfig c = make_mem_cfg();
    c.max_entries    = 100;
    c.max_size_bytes = 64;   /* tiny: each entry ~16 bytes */
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);

    uint64_t mh = oc_context_cache_model_hash(1, 1);
    for (int i = 0; i < 10; i++) {
        char sid[16];
        snprintf(sid, sizeof(sid), "s%d", i);
        uint64_t sz = 0;
        uint8_t *kv = make_kv("0123456789abcdef", &sz);
        OcError e = oc_context_cache_store(cc, sid, mh, 1, 1, 1, 1, kv, sz);
        cr_assert_eq(e, OC_OK);
    }
    OcContextCacheStats s;
    oc_context_cache_get_stats(cc, &s);
    cr_assert_leq(s.total_bytes, 64, "should respect max_size_bytes");
    oc_context_cache_free(cc);
}

Test(context_cache, manual_evict)
{
    OcContextCacheConfig c = make_mem_cfg();
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);

    uint64_t sz = 0, mh = oc_context_cache_model_hash(1, 1);
    uint8_t *kv = make_kv("e", &sz);
    cr_assert_eq(oc_context_cache_store(cc, "s", mh, 1, 1, 1, 1, kv, sz), OC_OK);

    cr_assert(oc_context_cache_evict(cc));
    OcContextCacheStats s;
    oc_context_cache_get_stats(cc, &s);
    cr_assert_eq(s.n_entries, 0);
    cr_assert_eq(s.evictions, 1);
    /* Evict again on empty cache. */
    cr_assert(!oc_context_cache_evict(cc));
    oc_context_cache_free(cc);
}

Test(context_cache, lru_updates_on_load)
{
    OcContextCacheConfig c = make_mem_cfg();
    c.max_entries = 2;
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);

    uint64_t mh = oc_context_cache_model_hash(1, 1);
    uint64_t sz = 0;
    uint8_t *kv1 = make_kv("a", &sz);
    uint8_t *kv2 = make_kv("b", &sz);
    uint8_t *kv3 = make_kv("c", &sz);
    cr_assert_eq(oc_context_cache_store(cc, "s1", mh, 1, 1, 1, 1, kv1, sz), OC_OK);
    cr_assert_eq(oc_context_cache_store(cc, "s2", mh, 1, 1, 1, 1, kv2, sz), OC_OK);

    /* Touch s1 so s2 becomes LRU. */
    OcContextCacheEntry out;
    bool found = false;
    cr_assert_eq(oc_context_cache_load(cc, "s1", mh, &out, &found), OC_OK);
    cr_assert(found);
    free(out.data);

    /* Insert s3 → evicts LRU (s2). */
    cr_assert_eq(oc_context_cache_store(cc, "s3", mh, 1, 1, 1, 1, kv3, sz), OC_OK);

    /* s2 should be gone; s1 and s3 remain. */
    found = false;
    cr_assert_eq(oc_context_cache_load(cc, "s2", mh, &out, &found), OC_OK);
    cr_assert(!found, "s2 should have been evicted");
    found = false;
    cr_assert_eq(oc_context_cache_load(cc, "s1", mh, &out, &found), OC_OK);
    cr_assert(found, "s1 should survive");
    free(out.data);
    oc_context_cache_free(cc);
}

/* ------------------------------------------------------------------ stats + clear ------------------------------------------------------------------ */

Test(context_cache, stats_tracking)
{
    OcContextCacheConfig c = make_mem_cfg();
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);

    uint64_t mh = oc_context_cache_model_hash(1, 1), sz = 0;
    uint8_t *kv = make_kv("zz", &sz);
    cr_assert_eq(oc_context_cache_store(cc, "s1", mh, 1, 1, 1, 1, kv, sz), OC_OK);

    OcContextCacheEntry out;
    bool found = false;
    /* Hit. */
    cr_assert_eq(oc_context_cache_load(cc, "s1", mh, &out, &found), OC_OK);
    cr_assert(found);
    free(out.data);
    /* Miss. */
    found = false;
    cr_assert_eq(oc_context_cache_load(cc, "s2", mh, &out, &found), OC_OK);
    cr_assert(!found);

    OcContextCacheStats s;
    oc_context_cache_get_stats(cc, &s);
    cr_assert_eq(s.n_entries, 1);
    cr_assert_eq(s.hits, 1);
    cr_assert_eq(s.misses, 1);
    cr_assert(s.hit_rate > 0.49 && s.hit_rate < 0.51, "hit_rate ~0.5");
    oc_context_cache_free(cc);
}

Test(context_cache, clear_all)
{
    OcContextCacheConfig c = make_mem_cfg();
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);

    uint64_t mh = oc_context_cache_model_hash(1, 1), sz = 0;
    uint8_t *kv1 = make_kv("a", &sz);
    uint8_t *kv2 = make_kv("b", &sz);
    cr_assert_eq(oc_context_cache_store(cc, "s1", mh, 1, 1, 1, 1, kv1, sz), OC_OK);
    cr_assert_eq(oc_context_cache_store(cc, "s2", mh, 1, 1, 1, 1, kv2, sz), OC_OK);

    cr_assert_eq(oc_context_cache_clear(cc), OC_OK);
    OcContextCacheStats s;
    oc_context_cache_get_stats(cc, &s);
    cr_assert_eq(s.n_entries, 0);
    cr_assert_eq(s.total_bytes, 0);

    /* Clear on empty is fine. */
    cr_assert_eq(oc_context_cache_clear(cc), OC_OK);
    /* Clear on NULL is an error. */
    cr_assert_eq(oc_context_cache_clear(NULL), OC_ERR_INVALID_ARG);
    oc_context_cache_free(cc);
}

Test(context_cache, clear_removes_disk_files)
{
    const char *dir = "/tmp/ox_ctx_cache_clear_disk";
    (void)system("rm -rf /tmp/ox_ctx_cache_clear_disk");
    OcContextCacheConfig c = make_disk_cfg(dir);
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);

    uint64_t mh = oc_context_cache_model_hash(1, 1), sz = 0;
    uint8_t *kv = make_kv("d", &sz);
    cr_assert_eq(oc_context_cache_store(cc, "s1", mh, 1, 1, 1, 1, kv, sz), OC_OK);

    char path[1024];
    snprintf(path, sizeof(path), "%s/s1.bin", dir);
    struct stat st;
    cr_assert_eq(stat(path, &st), 0, "file should exist before clear");

    cr_assert_eq(oc_context_cache_clear(cc), OC_OK);
    cr_assert_neq(stat(path, &st), 0, "file should be removed after clear");
    oc_context_cache_free(cc);
    (void)system("rm -rf /tmp/ox_ctx_cache_clear_disk");
}

/* ------------------------------------------------------------------ TTL expiration ------------------------------------------------------------------ */

Test(context_cache, ttl_expiration)
{
    OcContextCacheConfig c = make_mem_cfg();
    c.ttl_seconds = 1;
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);

    uint64_t mh = oc_context_cache_model_hash(1, 1), sz = 0;
    uint8_t *kv = make_kv("t", &sz);
    cr_assert_eq(oc_context_cache_store(cc, "s1", mh, 1, 1, 1, 1, kv, sz), OC_OK);

    /* Immediate load should hit. */
    OcContextCacheEntry out;
    bool found = false;
    cr_assert_eq(oc_context_cache_load(cc, "s1", mh, &out, &found), OC_OK);
    cr_assert(found, "should hit before TTL");
    free(out.data);

    /* Wait past TTL. */
    sleep(2);

    /* Load should miss (entry expired by sweep). */
    found = false;
    cr_assert_eq(oc_context_cache_load(cc, "s1", mh, &out, &found), OC_OK);
    cr_assert(!found, "should miss after TTL");

    OcContextCacheStats s;
    oc_context_cache_get_stats(cc, &s);
    cr_assert_eq(s.n_entries, 0, "entry should be evicted");
    cr_assert(s.evictions >= 1, "should count an eviction");
    oc_context_cache_free(cc);
}

Test(context_cache, ttl_zero_disables)
{
    OcContextCacheConfig c = make_mem_cfg();
    c.ttl_seconds = 0;
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);

    uint64_t mh = oc_context_cache_model_hash(1, 1), sz = 0;
    uint8_t *kv = make_kv("t", &sz);
    cr_assert_eq(oc_context_cache_store(cc, "s1", mh, 1, 1, 1, 1, kv, sz), OC_OK);

    OcContextCacheEntry out;
    bool found = false;
    /* With TTL == 0, the entry survives indefinitely (no sleep needed — just
     * confirm the sweep doesn't evict fresh entries). */
    cr_assert_eq(oc_context_cache_load(cc, "s1", mh, &out, &found), OC_OK);
    cr_assert(found);
    free(out.data);
    oc_context_cache_free(cc);
}

/* ------------------------------------------------------------------ helpers: model_hash, session_id, format_stats ------------------------------------------------------------------ */

Test(context_cache, model_hash_stable_and_nonzero)
{
    uint64_t h1 = oc_context_cache_model_hash(100, 5);
    uint64_t h2 = oc_context_cache_model_hash(100, 5);
    cr_assert_eq(h1, h2, "same inputs → same hash");
    cr_assert_neq(h1, 0, "hash should not be zero");
    uint64_t h3 = oc_context_cache_model_hash(101, 5);
    cr_assert_neq(h1, h3, "different file size → different hash");
    uint64_t h4 = oc_context_cache_model_hash(100, 6);
    cr_assert_neq(h1, h4, "different tensor count → different hash");
}

Test(context_cache, session_id_generation)
{
    char sid[OC_CONTEXT_CACHE_SESSION_ID_LEN];
    uint64_t mh = oc_context_cache_model_hash(42, 2);
    OcError e = oc_context_cache_session_id("hello", mh, sid, sizeof(sid));
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(strlen(sid), 16, "session id should be 16 hex chars");
    /* Same inputs → same id. */
    char sid2[OC_CONTEXT_CACHE_SESSION_ID_LEN];
    oc_context_cache_session_id("hello", mh, sid2, sizeof(sid2));
    cr_assert_str_eq(sid, sid2);
    /* Different prompt → different id. */
    char sid3[OC_CONTEXT_CACHE_SESSION_ID_LEN];
    oc_context_cache_session_id("world", mh, sid3, sizeof(sid3));
    cr_assert_neq(strcmp(sid, sid3), 0);
}

Test(context_cache, session_id_bad_args)
{
    char sid[OC_CONTEXT_CACHE_SESSION_ID_LEN];
    cr_assert_eq(oc_context_cache_session_id(NULL, 1, sid, sizeof(sid)),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_context_cache_session_id("x", 1, NULL, sizeof(sid)),
                 OC_ERR_INVALID_ARG);
    /* Too-small buffer. */
    cr_assert_eq(oc_context_cache_session_id("x", 1, sid, 4),
                 OC_ERR_INVALID_ARG);
}

Test(context_cache, format_stats_json)
{
    OcContextCacheConfig c = make_mem_cfg();
    OcContextCache *cc = oc_context_cache_init(&c);
    cr_assert_not_null(cc);

    uint64_t mh = oc_context_cache_model_hash(1, 1), sz = 0;
    uint8_t *kv = make_kv("j", &sz);
    oc_context_cache_store(cc, "s1", mh, 1, 1, 1, 1, kv, sz);

    char buf[512];
    size_t n = oc_context_cache_format_stats(cc, buf, sizeof(buf));
    cr_assert(n > 0);
    cr_assert(n < sizeof(buf));
    cr_assert_eq(buf[0], '{');
    cr_assert_eq(buf[n - 1], '}');
    cr_assert(strstr(buf, "\"n_entries\"") != NULL);
    cr_assert(strstr(buf, "\"hit_rate\"") != NULL);
    cr_assert(strstr(buf, "\"evictions\"") != NULL);

    /* NULL buf → returns length that would be written. */
    size_t expect = oc_context_cache_format_stats(cc, NULL, 0);
    cr_assert(expect > 0);
    oc_context_cache_free(cc);
}

Test(context_cache, get_stats_null_safe)
{
    OcContextCacheStats s;
    oc_context_cache_get_stats(NULL, &s);
    cr_assert_eq(s.n_entries, 0);
    cr_assert_eq(s.total_bytes, 0);
    cr_assert_eq(s.hit_rate, 0.0);
}
