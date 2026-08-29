/* test_rotorquant_cache.c — fused RotorQuant cache (3D rotor + int4). */
#include <criterion/criterion.h>
#include "oxidize/rotorquant_cache.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static float lcg(uint64_t *state)
{
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)(*state >> 40) / (float)(1u << 24) * 2.0f - 1.0f;
}

static void fill_lcg(float *x, size_t n, uint64_t *rng)
{
    size_t i;
    for (i = 0; i < n; i++) x[i] = lcg(rng);
}

Test(rotorquant_cache, rotate_then_unrotate_is_identity)
{
    OcRotorQuantCacheConfig cfg;
    OcRotorQuantCache cache;
    uint64_t rng = 7;
    float v[32], r[32], back[32];
    size_t i;
    float n0 = 0.0f, n1 = 0.0f;
    oc_rotorquant_cache_config_init(&cfg);
    cfg.head_dim = 32;
    cr_assert_eq(oc_rotorquant_cache_init(&cache, &cfg), OC_OK);
    fill_lcg(v, 32, &rng);
    cr_assert_eq(oc_rotorquant_cache_rotate(&cache, v, 32, r), OC_OK);
    cr_assert_eq(oc_rotorquant_cache_unrotate(&cache, r, 32, back), OC_OK);
    for (i = 0; i < 32; i++) {
        cr_assert(fabsf(back[i] - v[i]) < 1.0e-5f,
                  "rotor roundtrip should be identity at %zu", i);
        n0 += v[i] * v[i];
        n1 += r[i] * r[i];
    }
    cr_assert(fabsf(n0 - n1) < 1.0e-4f * n0, "rotor should preserve norm");
    oc_rotorquant_cache_free(&cache);
}

Test(rotorquant_cache, logits_track_f32_reference)
{
    OcRotorQuantCacheConfig cfg;
    OcRotorQuantCache cache;
    const size_t d = 64, tokens = 16;
    uint64_t rng = 42;
    float *keys, *values, *query, *logits;
    size_t t, i, n_out = 0;
    oc_rotorquant_cache_config_init(&cfg);
    cfg.head_dim = d;
    cr_assert_eq(oc_rotorquant_cache_init(&cache, &cfg), OC_OK);
    keys = malloc(tokens * d * sizeof(float));
    values = malloc(tokens * d * sizeof(float));
    query = malloc(d * sizeof(float));
    logits = malloc(tokens * sizeof(float));
    fill_lcg(keys, tokens * d, &rng);
    fill_lcg(values, tokens * d, &rng);
    fill_lcg(query, d, &rng);
    cr_assert_eq(oc_rotorquant_cache_store_page(&cache, 0, 0, keys, values,
                                                tokens, 0),
                 OC_OK);
    cr_assert_eq(oc_rotorquant_cache_logits(&cache, 0, 0, query, d, (size_t)-1,
                                            logits, tokens, &n_out),
                 OC_OK);
    cr_assert_eq(n_out, tokens);
    for (t = 0; t < tokens; t++) {
        float ref = 0.0f, knorm = 0.0f, tol;
        for (i = 0; i < d; i++) {
            ref += query[i] * keys[t * d + i];
            knorm += keys[t * d + i] * keys[t * d + i];
        }
        /* int4 with per-32 scale: relative error on the dot is small. */
        tol = 0.15f * sqrtf(knorm);
        cr_assert(fabsf(logits[t] - ref) <= tol,
                  "logit %zu: %f vs ref %f (tol %f)", t, logits[t], ref, tol);
    }
    free(keys); free(values); free(query); free(logits);
    oc_rotorquant_cache_free(&cache);
}

Test(rotorquant_cache, attention_tracks_f32_reference)
{
    OcRotorQuantCacheConfig cfg;
    OcRotorQuantCache cache;
    const size_t d = 64, tokens = 32;
    uint64_t rng = 99;
    float *keys, *values, *query, *out, *scores, *ref;
    size_t t, i;
    float scale, max_s, z;
    oc_rotorquant_cache_config_init(&cfg);
    cfg.head_dim = d;
    cr_assert_eq(oc_rotorquant_cache_init(&cache, &cfg), OC_OK);
    keys = malloc(tokens * d * sizeof(float));
    values = malloc(tokens * d * sizeof(float));
    query = malloc(d * sizeof(float));
    out = malloc(d * sizeof(float));
    scores = malloc(tokens * sizeof(float));
    ref = calloc(d, sizeof(float));
    fill_lcg(keys, tokens * d, &rng);
    fill_lcg(values, tokens * d, &rng);
    fill_lcg(query, d, &rng);
    cr_assert_eq(oc_rotorquant_cache_store_page(&cache, 0, 0, keys, values,
                                                tokens, 0),
                 OC_OK);
    scale = 1.0f / sqrtf((float)d);
    max_s = -1.0e30f;
    for (t = 0; t < tokens; t++) {
        float dot = 0.0f;
        for (i = 0; i < d; i++) dot += query[i] * keys[t * d + i];
        scores[t] = dot * scale;
        if (scores[t] > max_s) max_s = scores[t];
    }
    z = 0.0f;
    for (t = 0; t < tokens; t++) {
        scores[t] = expf(scores[t] - max_s);
        z += scores[t];
    }
    for (t = 0; t < tokens; t++) {
        for (i = 0; i < d; i++)
            ref[i] += scores[t] / z * values[t * d + i];
    }
    cr_assert_eq(oc_rotorquant_cache_attention(&cache, 0, 0, query, d,
                                                (size_t)-1, out),
                 OC_OK);
    for (i = 0; i < d; i++) {
        cr_assert(fabsf(out[i] - ref[i]) < 0.1f,
                  "attn dim %zu: %f vs ref %f", i, out[i], ref[i]);
    }
    free(keys); free(values); free(query); free(out); free(scores); free(ref);
    oc_rotorquant_cache_free(&cache);
}

Test(rotorquant_cache, compression_ratio_vs_f32)
{
    OcRotorQuantCacheConfig cfg;
    OcRotorQuantCache cache;
    OcRotorQuantCacheStats st;
    const size_t d = 128, tokens = 64;
    float *keys, *values;
    size_t i;
    OcRotorQuantPageView view;
    oc_rotorquant_cache_config_init(&cfg);
    cfg.head_dim = d;
    cr_assert_eq(oc_rotorquant_cache_init(&cache, &cfg), OC_OK);
    keys = malloc(tokens * d * sizeof(float));
    values = malloc(tokens * d * sizeof(float));
    for (i = 0; i < tokens * d; i++) {
        keys[i] = 0.5f;
        values[i] = -0.25f;
    }
    cr_assert_eq(oc_rotorquant_cache_store_page(&cache, 0, 0, keys, values,
                                                tokens, 0),
                 OC_OK);
    cr_assert_eq(oc_rotorquant_cache_stats(&cache, &st), OC_OK);
    cr_assert_eq(st.token_count, tokens);
    cr_assert(st.total_bits_per_coord > 4.0f && st.total_bits_per_coord < 6.0f,
              "int4 + scales should land near 5 bits/coord, got %f",
              st.total_bits_per_coord);
    cr_assert(oc_rotorquant_cache_compression_ratio(&st) > 6.0f,
              "rotorquant should compress > 6x vs f32, got %f",
              oc_rotorquant_cache_compression_ratio(&st));
    cr_assert_eq(oc_rotorquant_cache_page_count(&cache), (size_t)1);
    cr_assert(oc_rotorquant_cache_page_view(&cache, 0, &view));
    cr_assert_not_null(view.key_codes);
    cr_assert_not_null(view.key_scales);
    free(keys); free(values);
    oc_rotorquant_cache_free(&cache);
}

Test(rotorquant_cache, query_len_mismatch_is_invalid)
{
    OcRotorQuantCacheConfig cfg;
    OcRotorQuantCache cache;
    float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float out[32];
    oc_rotorquant_cache_config_init(&cfg);
    cfg.head_dim = 32;
    cr_assert_eq(oc_rotorquant_cache_init(&cache, &cfg), OC_OK);
    cr_assert_eq(oc_rotorquant_cache_attention(&cache, 0, 0, q, 4, 0, out),
                 OC_ERR_INVALID_ARG,
                 "short query must not walk past query_n");
    oc_rotorquant_cache_free(&cache);
}

Test(rotorquant_cache, restore_same_slot_is_upsert)
{
    OcRotorQuantCacheConfig cfg;
    OcRotorQuantCache cache;
    float keys[32], values[32];
    size_t i;
    oc_rotorquant_cache_config_init(&cfg);
    cfg.head_dim = 32;
    cr_assert_eq(oc_rotorquant_cache_init(&cache, &cfg), OC_OK);
    for (i = 0; i < 32; i++) {
        keys[i] = 0.5f;
        values[i] = 0.25f;
    }
    cr_assert_eq(oc_rotorquant_cache_store_page(&cache, 0, 0, keys, values, 1, 3),
                 OC_OK);
    cr_assert_eq(oc_rotorquant_cache_store_page(&cache, 0, 0, keys, values, 1, 3),
                 OC_OK);
    cr_assert_eq(oc_rotorquant_cache_page_count(&cache), (size_t)1,
                 "re-store of (layer,head,first_position) must replace");
    oc_rotorquant_cache_free(&cache);
}

Test(rotorquant_cache, causal_mask_drops_future_tokens)
{
    OcRotorQuantCacheConfig cfg;
    OcRotorQuantCache cache;
    const size_t d = 32, tokens = 4;
    float keys[128], values[128], query[32], logits[4];
    size_t n_out = 0, i;
    oc_rotorquant_cache_config_init(&cfg);
    cfg.head_dim = d;
    cr_assert_eq(oc_rotorquant_cache_init(&cache, &cfg), OC_OK);
    for (i = 0; i < tokens * d; i++) {
        keys[i] = 0.25f;
        values[i] = 1.0f;
    }
    for (i = 0; i < d; i++) query[i] = 0.5f;
    cr_assert_eq(oc_rotorquant_cache_store_page(&cache, 0, 0, keys, values,
                                                tokens, 0),
                 OC_OK);
    cr_assert_eq(oc_rotorquant_cache_logits(&cache, 0, 0, query, d, 1, logits, 4,
                                            &n_out),
                 OC_OK);
    cr_assert_eq(n_out, (size_t)2, "positions 0 and 1 are visible at query_pos=1");
    oc_rotorquant_cache_free(&cache);
}

Test(rotorquant_cache, attention_causal_mask_drops_future_tokens)
{
    OcRotorQuantCacheConfig cfg;
    OcRotorQuantCache cache;
    const size_t d = 32, tokens = 4;
    float keys[128], values[128], query[32], out_masked[32], out_all[32];
    size_t i, t, n_out = 0;
    float logits[4];
    int differed = 0;
    oc_rotorquant_cache_config_init(&cfg);
    cfg.head_dim = d;
    cr_assert_eq(oc_rotorquant_cache_init(&cache, &cfg), OC_OK);
    for (t = 0; t < tokens; t++) {
        for (i = 0; i < d; i++) {
            keys[t * d + i] = 0.25f + 0.1f * (float)t;
            values[t * d + i] = (t >= 2) ? 8.0f : 1.0f;
        }
    }
    for (i = 0; i < d; i++) query[i] = 0.5f;
    cr_assert_eq(oc_rotorquant_cache_store_page(&cache, 0, 0, keys, values,
                                                tokens, 0),
                 OC_OK);
    cr_assert_eq(oc_rotorquant_cache_logits(&cache, 0, 0, query, d, 1, logits, 4,
                                            &n_out),
                 OC_OK);
    cr_assert_eq(n_out, (size_t)2);
    cr_assert_eq(oc_rotorquant_cache_attention(&cache, 0, 0, query, d, 1,
                                                out_masked),
                 OC_OK);
    cr_assert_eq(oc_rotorquant_cache_attention(&cache, 0, 0, query, d,
                                                (size_t)-1, out_all),
                 OC_OK);
    for (i = 0; i < d; i++) {
        if (fabsf(out_masked[i] - out_all[i]) > 1.0e-3f) differed = 1;
    }
    cr_assert(differed,
              "future tokens with distinct values must change unmasked attn");
    oc_rotorquant_cache_free(&cache);
}

Test(rotorquant_cache, rewind_truncates_mid_page)
{
    OcRotorQuantCacheConfig cfg;
    OcRotorQuantCache cache;
    OcRotorQuantPageView view;
    float keys[32], values[32];
    size_t i;
    oc_rotorquant_cache_config_init(&cfg);
    cfg.head_dim = 8;
    cr_assert_eq(oc_rotorquant_cache_init(&cache, &cfg), OC_OK);
    for (i = 0; i < 32; i++) {
        keys[i] = 0.1f;
        values[i] = 0.2f;
    }
    cr_assert_eq(oc_rotorquant_cache_store_page(&cache, 0, 0, keys, values,
                                                4, 0),
                 OC_OK);
    cr_assert(oc_rotorquant_cache_page_view(&cache, 0, &view));
    {
        const uint8_t *kc = view.key_codes;
        const uint8_t *vc = view.value_codes;
        const float *ks = view.key_scales;
        const float *vs = view.value_scales;
        cr_assert_eq(oc_rotorquant_cache_rewind(&cache, 2), OC_OK);
        cr_assert_eq(oc_rotorquant_cache_page_count(&cache), (size_t)1);
        cr_assert_eq(oc_rotorquant_cache_n_logits(&cache, 0, 0), (size_t)2);
        cr_assert(oc_rotorquant_cache_page_view(&cache, 0, &view));
        cr_assert_eq(view.tokens, (size_t)2);
        cr_assert_eq(view.first_position, (size_t)0);
        cr_assert_eq(view.key_codes, kc);
        cr_assert_eq(view.value_codes, vc);
        cr_assert_eq(view.key_scales, ks);
        cr_assert_eq(view.value_scales, vs);
    }
    oc_rotorquant_cache_free(&cache);
}

Test(rotorquant_cache, store_rejects_position_range_overflow)
{
    OcRotorQuantCacheConfig cfg;
    OcRotorQuantCache cache;
    float keys[16], values[16];
    size_t i;
    oc_rotorquant_cache_config_init(&cfg);
    cfg.head_dim = 8;
    cr_assert_eq(oc_rotorquant_cache_init(&cache, &cfg), OC_OK);
    for (i = 0; i < 16; i++) {
        keys[i] = 0.1f;
        values[i] = 0.2f;
    }
    cr_assert_eq(oc_rotorquant_cache_store_page(&cache, 0, 0, keys, values,
                                                2, (size_t)-1),
                 OC_ERR_INVALID_ARG);
    oc_rotorquant_cache_free(&cache);
}

Test(rotorquant_cache, rejects_zero_dim)
{
    OcRotorQuantCacheConfig cfg;
    OcRotorQuantCache cache;
    oc_rotorquant_cache_config_init(&cfg);
    cfg.head_dim = 0;
    cr_assert_eq(oc_rotorquant_cache_init(&cache, &cfg), OC_ERR_INVALID_ARG);
}
