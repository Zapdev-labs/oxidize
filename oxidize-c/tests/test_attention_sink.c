/* test_attention_sink.c — Attention sink tests. */
#include "framework.h"
#include "oxidize/attention_sink.h"
#include <string.h>

Test(sink, config_init)
{
    OcAttentionSinkConfig cfg;
    cr_assert_eq(oc_attn_sink_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.sink_size, 4);
    cr_assert_eq(cfg.window_size, 4096);
}

Test(sink, config_init_null)
{
    cr_assert_neq(oc_attn_sink_config_init(NULL), OC_OK);
}

Test(sink, init_free)
{
    OcAttentionSinkConfig cfg;
    oc_attn_sink_config_init(&cfg);
    OcAttentionSink sink;
    cr_assert_eq(oc_attn_sink_init(&sink, &cfg, 64, 8), OC_OK);
    cr_assert(sink.initialized);
    oc_attn_sink_free(&sink);
    cr_assert(!sink.initialized);
}

Test(sink, init_null)
{
    cr_assert_neq(oc_attn_sink_init(NULL, NULL, 0, 0), OC_OK);
    OcAttentionSinkConfig cfg;
    oc_attn_sink_config_init(&cfg);
    OcAttentionSink sink;
    cr_assert_neq(oc_attn_sink_init(&sink, &cfg, 0, 8), OC_OK);
    cr_assert_neq(oc_attn_sink_init(&sink, &cfg, 64, 0), OC_OK);
}

Test(sink, append_sink)
{
    OcAttentionSinkConfig cfg;
    oc_attn_sink_config_init(&cfg);
    cfg.sink_size = 4;
    cfg.window_size = 16;
    OcAttentionSink sink;
    cr_assert_eq(oc_attn_sink_init(&sink, &cfg, 8, 1), OC_OK);

    float key[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float val[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    uint32_t slot;

    cr_assert_eq(oc_attn_sink_append(&sink, key, val, 0, &slot), OC_OK);
    cr_assert_eq(slot, 0);
    cr_assert_eq(oc_attn_sink_size(&sink), 1);

    oc_attn_sink_free(&sink);
}

Test(sink, append_multiple_sink)
{
    OcAttentionSinkConfig cfg;
    oc_attn_sink_config_init(&cfg);
    cfg.sink_size = 4;
    cfg.window_size = 16;
    OcAttentionSink sink;
    cr_assert_eq(oc_attn_sink_init(&sink, &cfg, 8, 1), OC_OK);

    float key[8] = {1};
    float val[8] = {2};
    uint32_t slot;

    for (uint32_t i = 0; i < 4; i++) {
        cr_assert_eq(oc_attn_sink_append(&sink, key, val, i, &slot), OC_OK);
        cr_assert_eq(slot, i);
    }
    cr_assert_eq(oc_attn_sink_size(&sink), 4);

    oc_attn_sink_free(&sink);
}

Test(sink, append_window)
{
    OcAttentionSinkConfig cfg;
    oc_attn_sink_config_init(&cfg);
    cfg.sink_size = 2;
    cfg.window_size = 8;
    OcAttentionSink sink;
    cr_assert_eq(oc_attn_sink_init(&sink, &cfg, 8, 1), OC_OK);

    float key[8] = {1};
    float val[8] = {2};
    uint32_t slot;

    /* Fill sinks first. */
    for (uint32_t i = 0; i < 2; i++)
        oc_attn_sink_append(&sink, key, val, i, &slot);

    /* Now window entries. */
    cr_assert_eq(oc_attn_sink_append(&sink, key, val, 2, &slot), OC_OK);
    cr_assert_eq(slot, 2); /* slot = sink_size + 0 */
    cr_assert_eq(oc_attn_sink_size(&sink), 3);

    oc_attn_sink_free(&sink);
}

Test(sink, get)
{
    OcAttentionSinkConfig cfg;
    oc_attn_sink_config_init(&cfg);
    cfg.sink_size = 2;
    cfg.window_size = 8;
    OcAttentionSink sink;
    cr_assert_eq(oc_attn_sink_init(&sink, &cfg, 4, 1), OC_OK);

    float key[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float val[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    uint32_t slot;

    oc_attn_sink_append(&sink, key, val, 0, &slot);
    const float *out_k, *out_v;
    cr_assert_eq(oc_attn_sink_get(&sink, 0, &out_k, &out_v), OC_OK);
    cr_assert_float_eq(out_k[0], 1.0f, 0.001f);
    cr_assert_float_eq(out_v[0], 5.0f, 0.001f);

    oc_attn_sink_free(&sink);
}

Test(sink, get_window)
{
    OcAttentionSinkConfig cfg;
    oc_attn_sink_config_init(&cfg);
    cfg.sink_size = 2;
    cfg.window_size = 8;
    OcAttentionSink sink;
    cr_assert_eq(oc_attn_sink_init(&sink, &cfg, 4, 1), OC_OK);

    float key[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float val[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    uint32_t slot;

    /* Fill sinks. */
    oc_attn_sink_append(&sink, key, val, 0, &slot);
    oc_attn_sink_append(&sink, key, val, 1, &slot);

    /* Window entry. */
    float key2[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float val2[4] = {50.0f, 60.0f, 70.0f, 80.0f};
    oc_attn_sink_append(&sink, key2, val2, 2, &slot);
    cr_assert_eq(slot, 2);

    const float *out_k, *out_v;
    cr_assert_eq(oc_attn_sink_get(&sink, 2, &out_k, &out_v), OC_OK);
    cr_assert_float_eq(out_k[0], 10.0f, 0.001f);
    cr_assert_float_eq(out_v[0], 50.0f, 0.001f);

    oc_attn_sink_free(&sink);
}

Test(sink, evict_oldest)
{
    OcAttentionSinkConfig cfg;
    oc_attn_sink_config_init(&cfg);
    cfg.sink_size = 2;
    cfg.window_size = 4;
    OcAttentionSink sink;
    cr_assert_eq(oc_attn_sink_init(&sink, &cfg, 4, 1), OC_OK);

    float key[4] = {1};
    float val[4] = {2};
    uint32_t slot;

    /* Fill sinks. */
    for (uint32_t i = 0; i < 2; i++)
        oc_attn_sink_append(&sink, key, val, i, &slot);

    /* Fill window. */
    for (uint32_t i = 2; i < 6; i++)
        oc_attn_sink_append(&sink, key, val, i, &slot);

    cr_assert_eq(oc_attn_sink_size(&sink), 6);

    /* Evict one. */
    cr_assert_eq(oc_attn_sink_evict_oldest(&sink), OC_OK);
    cr_assert_eq(oc_attn_sink_size(&sink), 5);

    oc_attn_sink_free(&sink);
}

Test(sink, reset_window)
{
    OcAttentionSinkConfig cfg;
    oc_attn_sink_config_init(&cfg);
    cfg.sink_size = 2;
    cfg.window_size = 8;
    OcAttentionSink sink;
    cr_assert_eq(oc_attn_sink_init(&sink, &cfg, 4, 1), OC_OK);

    float key[4] = {1};
    float val[4] = {2};
    uint32_t slot;

    oc_attn_sink_append(&sink, key, val, 0, &slot);
    oc_attn_sink_append(&sink, key, val, 1, &slot);
    oc_attn_sink_append(&sink, key, val, 2, &slot);
    cr_assert_eq(oc_attn_sink_size(&sink), 3);

    cr_assert_eq(oc_attn_sink_reset_window(&sink), OC_OK);
    cr_assert_eq(oc_attn_sink_size(&sink), 2); /* only sinks remain */

    oc_attn_sink_free(&sink);
}

Test(sink, size_empty)
{
    OcAttentionSinkConfig cfg;
    oc_attn_sink_config_init(&cfg);
    cfg.sink_size = 2;
    cfg.window_size = 8;
    OcAttentionSink sink;
    cr_assert_eq(oc_attn_sink_init(&sink, &cfg, 4, 1), OC_OK);
    cr_assert_eq(oc_attn_sink_size(&sink), 0);
    oc_attn_sink_free(&sink);
}

Test(sink, size_null)
{
    cr_assert_eq(oc_attn_sink_size(NULL), 0);
}

Test(sink, get_null)
{
    const float *k, *v;
    cr_assert_neq(oc_attn_sink_get(NULL, 0, &k, &v), OC_OK);
}

Test(sink, get_out_of_range)
{
    OcAttentionSinkConfig cfg;
    oc_attn_sink_config_init(&cfg);
    cfg.sink_size = 2;
    cfg.window_size = 4;
    OcAttentionSink sink;
    oc_attn_sink_init(&sink, &cfg, 4, 1);
    const float *k, *v;
    cr_assert_neq(oc_attn_sink_get(&sink, 100, &k, &v), OC_OK);
    oc_attn_sink_free(&sink);
}

Test(sink, append_null)
{
    cr_assert_neq(oc_attn_sink_append(NULL, NULL, NULL, 0, NULL), OC_OK);
}

Test(sink, evict_empty)
{
    OcAttentionSinkConfig cfg;
    oc_attn_sink_config_init(&cfg);
    cfg.sink_size = 2;
    cfg.window_size = 4;
    OcAttentionSink sink;
    oc_attn_sink_init(&sink, &cfg, 4, 1);
    cr_assert_eq(oc_attn_sink_evict_oldest(&sink), OC_OK);
    oc_attn_sink_free(&sink);
}

Test(sink, ring_buffer_wrap)
{
    OcAttentionSinkConfig cfg;
    oc_attn_sink_config_init(&cfg);
    cfg.sink_size = 1;
    cfg.window_size = 3;
    OcAttentionSink sink;
    cr_assert_eq(oc_attn_sink_init(&sink, &cfg, 4, 1), OC_OK);

    float key[4] = {1};
    float val[4] = {2};
    uint32_t slot;

    /* Fill sink. */
    oc_attn_sink_append(&sink, key, val, 0, &slot);

    /* Fill window. */
    for (uint32_t i = 1; i < 4; i++)
        oc_attn_sink_append(&sink, key, val, i, &slot);
    cr_assert_eq(oc_attn_sink_size(&sink), 4);

    /* Add one more (should evict oldest window entry). */
    oc_attn_sink_append(&sink, key, val, 4, &slot);
    cr_assert_eq(oc_attn_sink_size(&sink), 4); /* still 4 (1 sink + 3 window) */

    oc_attn_sink_free(&sink);
}
