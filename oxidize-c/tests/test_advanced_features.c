/* test_advanced_features.c — Advanced features tests. */
#include "framework.h"
#include "oxidize/advanced_features.h"
#include <string.h>

/* ─── Prompt Cache ────────────────────────────────────────────────── */

Test(af, prompt_cache_init)
{
    OcPromptCache cache;
    cr_assert_eq(oc_prompt_cache_init(&cache), OC_OK);
    cr_assert(!cache.valid);
    cr_assert_eq(cache.n_prefix, 0);
}

Test(af, prompt_cache_set_prefix)
{
    OcPromptCache cache;
    oc_prompt_cache_init(&cache);
    uint32_t tokens[] = {1, 2, 3, 4, 5};
    cr_assert_eq(oc_prompt_cache_set_prefix(&cache, tokens, 5), OC_OK);
    cr_assert(cache.valid);
    cr_assert_eq(cache.n_prefix, 5);
    cr_assert_eq(cache.prefix_tokens[0], 1);
    oc_prompt_cache_free(&cache);
}

Test(af, prompt_cache_match)
{
    OcPromptCache cache;
    oc_prompt_cache_init(&cache);
    uint32_t prefix[] = {1, 2, 3, 4, 5};
    oc_prompt_cache_set_prefix(&cache, prefix, 5);
    uint32_t input[] = {1, 2, 3, 99, 100};
    size_t match;
    cr_assert_eq(oc_prompt_cache_match(&cache, input, 5, &match), OC_OK);
    cr_assert_eq(match, 3);
    oc_prompt_cache_free(&cache);
}

Test(af, prompt_cache_match_all)
{
    OcPromptCache cache;
    oc_prompt_cache_init(&cache);
    uint32_t prefix[] = {1, 2, 3};
    oc_prompt_cache_set_prefix(&cache, prefix, 3);
    uint32_t input[] = {1, 2, 3, 4, 5};
    size_t match;
    oc_prompt_cache_match(&cache, input, 5, &match);
    cr_assert_eq(match, 3);
    oc_prompt_cache_free(&cache);
}

Test(af, prompt_cache_match_none)
{
    OcPromptCache cache;
    oc_prompt_cache_init(&cache);
    uint32_t prefix[] = {1, 2, 3};
    oc_prompt_cache_set_prefix(&cache, prefix, 3);
    uint32_t input[] = {99, 98, 97};
    size_t match;
    oc_prompt_cache_match(&cache, input, 3, &match);
    cr_assert_eq(match, 0);
    oc_prompt_cache_free(&cache);
}

Test(af, prompt_cache_is_valid)
{
    OcPromptCache cache;
    oc_prompt_cache_init(&cache);
    cr_assert(!oc_prompt_cache_is_valid(&cache));
    uint32_t tokens[] = {1};
    oc_prompt_cache_set_prefix(&cache, tokens, 1);
    cr_assert(oc_prompt_cache_is_valid(&cache));
    oc_prompt_cache_free(&cache);
}

OC_TEST_NULL_SAFE(af, prompt_cache_null,
        cr_assert_neq(oc_prompt_cache_init(NULL), OC_OK);
        cr_assert_neq(oc_prompt_cache_set_prefix(NULL, NULL, 0), OC_OK);)

OC_TEST_NULL_SAFE(af, prompt_cache_free_null,
        oc_prompt_cache_free(NULL);)

/* ─── Speculative Decoding Stats ──────────────────────────────────── */

Test(af, spec_stats_init)
{
    OcSpecStats stats;
    cr_assert_eq(oc_spec_stats_init(&stats, OC_SPEC_MODE_EAGLE, 4), OC_OK);
    cr_assert_eq(stats.mode, OC_SPEC_MODE_EAGLE);
    cr_assert_eq(stats.n_draft_tokens, 4);
    cr_assert_eq(stats.n_accepted, 0);
    cr_assert_eq(stats.n_total, 0);
}

Test(af, spec_stats_record)
{
    OcSpecStats stats;
    oc_spec_stats_init(&stats, OC_SPEC_MODE_DFLASH, 8);
    oc_spec_stats_record(&stats, 6, 8);
    cr_assert_eq(stats.n_accepted, 6);
    cr_assert_eq(stats.n_total, 8);
    cr_assert_float_eq(stats.acceptance_rate, 0.75f, 0.01f);
}

Test(af, spec_stats_record_multiple)
{
    OcSpecStats stats;
    oc_spec_stats_init(&stats, OC_SPEC_MODE_TREE, 4);
    oc_spec_stats_record(&stats, 3, 4);
    oc_spec_stats_record(&stats, 2, 4);
    cr_assert_eq(stats.n_accepted, 5);
    cr_assert_eq(stats.n_total, 8);
    cr_assert_float_eq(stats.acceptance_rate, 0.625f, 0.01f);
}

OC_TEST_NULL_SAFE(af, spec_stats_null,
        cr_assert_neq(oc_spec_stats_init(NULL, OC_SPEC_MODE_NONE, 0), OC_OK);
        cr_assert_neq(oc_spec_stats_record(NULL, 0, 0), OC_OK);)

OC_TEST_NULL_SAFE(af, spec_stats_acceptance_rate_null,
        cr_assert_eq(oc_spec_stats_acceptance_rate(NULL), 0.0f);)

Test(af, spec_mode_name)
{
    cr_assert_str_eq(oc_spec_mode_name(OC_SPEC_MODE_NONE), "none");
    cr_assert_str_eq(oc_spec_mode_name(OC_SPEC_MODE_EAGLE), "eagle");
    cr_assert_str_eq(oc_spec_mode_name(OC_SPEC_MODE_DFLASH), "dflash");
    cr_assert_str_eq(oc_spec_mode_name(OC_SPEC_MODE_TREE), "tree");
    cr_assert_str_eq(oc_spec_mode_name(OC_SPEC_MODE_LOOKUP), "lookup");
}

/* ─── Multi-Model Serving ─────────────────────────────────────────── */

Test(af, multi_model_init)
{
    OcMultiModelServer srv;
    cr_assert_eq(oc_multi_model_init(&srv, NULL), OC_OK);
    cr_assert_eq(srv.n_loaded, 0);
    cr_assert_eq(srv.used_memory_bytes, 0);
    cr_assert(srv.config.enable_hot_swap);
    oc_multi_model_free(&srv);
}

Test(af, multi_model_load)
{
    OcMultiModelServer srv;
    oc_multi_model_init(&srv, NULL);
    cr_assert_eq(oc_multi_model_load(&srv, "llama-7b", "/tmp/model.gguf", 4000000000ULL), OC_OK);
    cr_assert_eq(srv.n_loaded, 1);
    cr_assert_eq(srv.used_memory_bytes, 4000000000ULL);
    cr_assert(oc_multi_model_is_loaded(&srv, "llama-7b"));
    oc_multi_model_free(&srv);
}

Test(af, multi_model_load_duplicate)
{
    OcMultiModelServer srv;
    oc_multi_model_init(&srv, NULL);
    oc_multi_model_load(&srv, "model", "/tmp/m.gguf", 1000);
    cr_assert_eq(oc_multi_model_load(&srv, "model", "/tmp/m.gguf", 1000), OC_OK);
    cr_assert_eq(srv.n_loaded, 1);
    oc_multi_model_free(&srv);
}

Test(af, multi_model_unload)
{
    OcMultiModelServer srv;
    oc_multi_model_init(&srv, NULL);
    oc_multi_model_load(&srv, "model", "/tmp/m.gguf", 1000);
    cr_assert_eq(oc_multi_model_unload(&srv, "model"), OC_OK);
    cr_assert_eq(srv.n_loaded, 0);
    cr_assert_eq(srv.used_memory_bytes, 0);
    cr_assert(!oc_multi_model_is_loaded(&srv, "model"));
    oc_multi_model_free(&srv);
}

Test(af, multi_model_find)
{
    OcMultiModelServer srv;
    oc_multi_model_init(&srv, NULL);
    oc_multi_model_load(&srv, "llama", "/tmp/l.gguf", 1000);
    const OcModelSlot *slot;
    cr_assert_eq(oc_multi_model_find(&srv, "llama", &slot), OC_OK);
    cr_assert_str_eq(slot->model_id, "llama");
    oc_multi_model_free(&srv);
}

Test(af, multi_model_find_not_found)
{
    OcMultiModelServer srv;
    oc_multi_model_init(&srv, NULL);
    const OcModelSlot *slot;
    cr_assert_neq(oc_multi_model_find(&srv, "nonexistent", &slot), OC_OK);
    oc_multi_model_free(&srv);
}

Test(af, multi_model_evict_lru)
{
    OcMultiModelServer srv;
    oc_multi_model_init(&srv, NULL);
    oc_multi_model_load(&srv, "m1", "/tmp/m1.gguf", 1000);
    srv.slots[0].last_used_ms = 100;
    oc_multi_model_load(&srv, "m2", "/tmp/m2.gguf", 2000);
    srv.slots[1].last_used_ms = 200;
    cr_assert_eq(oc_multi_model_evict_lru(&srv), OC_OK);
    cr_assert(!oc_multi_model_is_loaded(&srv, "m1")); /* m1 was older */
    cr_assert(oc_multi_model_is_loaded(&srv, "m2"));
    oc_multi_model_free(&srv);
}

Test(af, multi_model_n_loaded)
{
    OcMultiModelServer srv;
    oc_multi_model_init(&srv, NULL);
    cr_assert_eq(oc_multi_model_n_loaded(&srv), 0);
    oc_multi_model_load(&srv, "m1", "/p", 100);
    cr_assert_eq(oc_multi_model_n_loaded(&srv), 1);
    oc_multi_model_load(&srv, "m2", "/p", 200);
    cr_assert_eq(oc_multi_model_n_loaded(&srv), 2);
    oc_multi_model_free(&srv);
}

Test(af, multi_model_used_memory)
{
    OcMultiModelServer srv;
    oc_multi_model_init(&srv, NULL);
    oc_multi_model_load(&srv, "m1", "/p", 1000);
    oc_multi_model_load(&srv, "m2", "/p", 2000);
    cr_assert_eq(oc_multi_model_used_memory(&srv), 3000);
    oc_multi_model_free(&srv);
}

Test(af, multi_model_null)
{
    cr_assert_neq(oc_multi_model_init(NULL, NULL), OC_OK);
    cr_assert_neq(oc_multi_model_load(NULL, NULL, NULL, 0), OC_OK);
    cr_assert(!oc_multi_model_is_loaded(NULL, NULL));
    cr_assert_eq(oc_multi_model_n_loaded(NULL), 0);
    oc_multi_model_free(NULL);
}
