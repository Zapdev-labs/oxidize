/* test_helix_cache.c — polar 4-bit keys + Hadamard 3-bit values. */
#include <criterion/criterion.h>
#include "oxidize/helix_cache.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

Test(helix_cache, cold_page_attention_matches_rope_polar_reference)
{
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    OcHelixCacheStats stats;
    const float keys[] = {
        1.0f, 0.0f, 0.02f, 0.01f, 2.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.03f, 0.01f, 2.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.01f, 0.01f, 2.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.02f, 0.01f, 2.0f, 0.0f, 0.0f, 0.0f,
    };
    const float values[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
        2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f,
    };
    const size_t positions[] = {0, 1, 2, 3};
    const float query[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f};
    float logits[4], out[8];
    size_t n_out = 0, t;
    oc_helix_cache_config_init(&cfg);
    cfg.page_size = 4;
    cfg.head_dim = 8;
    cfg.inactive_threshold = 0.05f;
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_OK);
    cr_assert_eq(oc_helix_cache_store_cold_page(&cache, 0, 0, 0, keys, values,
                                                positions, 4),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_logits(&cache, 0, 0, query, 8, 3, 10000.0f,
                                       logits, 4, &n_out),
                 OC_OK);
    cr_assert_eq(n_out, (size_t)4);
    for (t = 0; t < 4; t++) {
        /* Pair 0: rho=1, q=1, freq=1. Pair 1 is below the inactive
         * threshold. Pair 2: rho=2, q=0.5, freq=theta^(-4/8)=0.01.
         * Cosine is even, so this case does not catch a RoPE sign error. */
        const float rel = (float)t - 3.0f;
        const float expected = cosf(rel) + cosf(0.01f * rel);
        cr_assert(fabsf(logits[t] - expected) < 0.001f,
                  "logit %zu: %f vs %f", t, logits[t], expected);
    }
    cr_assert_eq(oc_helix_cache_attention(&cache, 0, 0, query, 8, 3, 10000.0f,
                                          out),
                 OC_OK);
    cr_assert(fabsf(out[0] - 1.75f) < 0.35f,
              "Hadamard int3 value path, out[0]=%f", out[0]);
    cr_assert_eq(oc_helix_cache_stats(&cache, &stats), OC_OK);
    cr_assert_eq(stats.cold_pages, (size_t)1);
    cr_assert(stats.key_bits_per_coord > 0.0f);
    cr_assert(stats.value_bits_per_coord > 3.0f);
    /* d=8 with rho_lut metadata is not expected to beat f32. Compression
     * vs baseline is asserted at d=128 in compression_at_d128. */
    cr_assert(stats.f32_baseline_bytes > 0);
    cr_assert(stats.key_bytes + stats.value_bytes + stats.metadata_bytes > 0);
    oc_helix_cache_free(&cache);
}

Test(helix_cache, phase_shifted_logits_use_query_minus_key)
{
    /* Non-zero query/key phases: cos is no longer even, so the RoPE sign
     * is observable. q_phi=pi/2, k_phi=0, qpos=1, kpos=0, freq=1 →
     * correct logit = cos(pi/2 + 1) = -sin(1). The C++ PR's key-query
     * sign would yield +sin(1). */
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    const float keys[] = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };
    const float values[] = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };
    const size_t positions[] = {0};
    const float query[] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float logits[1];
    size_t n_out = 0;
    const float expected = -sinf(1.0f);
    oc_helix_cache_config_init(&cfg);
    cfg.page_size = 1;
    cfg.head_dim = 8;
    cfg.inactive_threshold = 0.0f;
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_OK);
    cr_assert_eq(oc_helix_cache_store_cold_page(&cache, 0, 0, 0, keys, values,
                                                positions, 1),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_logits(&cache, 0, 0, query, 8, 1, 10000.0f,
                                       logits, 1, &n_out),
                 OC_OK);
    cr_assert_eq(n_out, (size_t)1);
    cr_assert(fabsf(logits[0] - expected) < 0.05f,
              "phase-shifted logit %f vs expected %f (wrong sign would be %f)",
              logits[0], expected, sinf(1.0f));
    oc_helix_cache_free(&cache);
}

Test(helix_cache, causal_mask_drops_future_tokens)
{
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    const float keys[] = {
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    };
    const float values[32] = {0};
    const size_t positions[] = {0, 1, 2, 3};
    const float query[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float logits[4];
    size_t n_out = 0;
    oc_helix_cache_config_init(&cfg);
    cfg.page_size = 4;
    cfg.head_dim = 8;
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_OK);
    cr_assert_eq(oc_helix_cache_store_cold_page(&cache, 0, 0, 0, keys, values,
                                                positions, 4),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_logits(&cache, 0, 0, query, 8, 1, 10000.0f,
                                       logits, 4, &n_out),
                 OC_OK);
    cr_assert_eq(n_out, (size_t)2, "future keys after query_pos must be dropped");
    oc_helix_cache_free(&cache);
}

Test(helix_cache, rejects_malformed_dimensions)
{
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    oc_helix_cache_config_init(&cfg);
    cfg.page_size = 4;
    cfg.head_dim = 7;
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_ERR_INVALID_ARG);
}

Test(helix_cache, promotion_state_marks_uncertain_pages)
{
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    bool promote = true;
    const float keys[] = {
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    };
    const float values[16] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    };
    const size_t positions[] = {8, 9};
    oc_helix_cache_config_init(&cfg);
    cfg.page_size = 2;
    cfg.head_dim = 8;
    cfg.promotion_budget = 2;
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_OK);
    cr_assert_eq(oc_helix_cache_store_cold_page(&cache, 1, 2, 3, keys, values,
                                                positions, 2),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_bump_uncertainty(&cache, 1, 2, 3, 0.25f),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_should_promote(&cache, 1, 2, 3, &promote),
                 OC_OK);
    cr_assert(!promote, "first uncertainty hit should not promote");
    cr_assert_eq(oc_helix_cache_bump_uncertainty(&cache, 1, 2, 3, 0.25f),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_should_promote(&cache, 1, 2, 3, &promote),
                 OC_OK);
    cr_assert(promote, "budgeted uncertainty hits should request promotion");
    oc_helix_cache_free(&cache);
}

Test(helix_cache, restore_same_page_id_is_upsert)
{
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    const float keys[] = {
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    };
    const float values[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    const size_t positions[] = {0};
    oc_helix_cache_config_init(&cfg);
    cfg.page_size = 2;
    cfg.head_dim = 8;
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_OK);
    cr_assert_eq(oc_helix_cache_store_cold_page(&cache, 0, 0, 0, keys, values,
                                                positions, 1),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_store_cold_page(&cache, 0, 0, 0, keys, values,
                                                positions, 1),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_page_count(&cache), (size_t)1,
                 "re-store of (layer,head,page_id) must replace, not append");
    oc_helix_cache_free(&cache);
}

Test(helix_cache, append_amortizes_page_metadata)
{
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    OcHelixCacheStats st;
    const size_t d = 128, tokens = 64;
    float *keys, *values;
    size_t *pos;
    size_t i;
    oc_helix_cache_config_init(&cfg);
    cfg.head_dim = d;
    cfg.page_size = tokens;
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_OK);
    keys = malloc(d * sizeof(float));
    values = malloc(d * sizeof(float));
    pos = malloc(sizeof(size_t));
    cr_assert(keys && values && pos);
    for (i = 0; i < tokens; i++) {
        size_t j;
        for (j = 0; j < d; j++) {
            keys[j] = ((j % 7) + 1) * 0.1f;
            values[j] = ((int)(j % 5) - 2) * 0.25f;
        }
        pos[0] = i;
        cr_assert_eq(oc_helix_cache_append(&cache, 0, 0, keys, values, pos, 1),
                     OC_OK);
    }
    cr_assert_eq(oc_helix_cache_page_count(&cache), (size_t)1,
                 "64 singleton appends at page_size=64 must share one page");
    cr_assert_eq(oc_helix_cache_stats(&cache, &st), OC_OK);
    cr_assert(oc_helix_cache_compression_ratio(&st) > 4.0f,
              "appended helix pages should compress, got %f",
              oc_helix_cache_compression_ratio(&st));
    free(keys); free(values); free(pos);
    oc_helix_cache_free(&cache);
}

Test(helix_cache, compression_at_d128)
{
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    OcHelixCacheStats st;
    const size_t d = 128, tokens = 64;
    float *keys, *values;
    size_t *pos;
    size_t i;
    oc_helix_cache_config_init(&cfg);
    cfg.head_dim = d;
    cfg.page_size = tokens;
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_OK);
    keys = malloc(tokens * d * sizeof(float));
    values = malloc(tokens * d * sizeof(float));
    pos = malloc(tokens * sizeof(size_t));
    for (i = 0; i < tokens * d; i++) {
        keys[i] = ((i % 7) + 1) * 0.1f;
        values[i] = ((int)(i % 5) - 2) * 0.25f;
    }
    for (i = 0; i < tokens; i++) pos[i] = i;
    cr_assert_eq(oc_helix_cache_store_cold_page(&cache, 0, 0, 0, keys, values,
                                                pos, tokens),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_stats(&cache, &st), OC_OK);
    cr_assert(oc_helix_cache_compression_ratio(&st) > 4.0f,
              "helix should compress vs f32 after counting rho_lut, got %f",
              oc_helix_cache_compression_ratio(&st));
    free(keys); free(values); free(pos);
    oc_helix_cache_free(&cache);
}

Test(helix_cache, zero_rho_pairs_are_inactive)
{
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    OcHelixColdPageView view;
    const float keys[] = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };
    const float values[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    const size_t positions[] = {0};
    const float query[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float logits[1];
    size_t n_out = 0;
    oc_helix_cache_config_init(&cfg);
    cfg.page_size = 1;
    cfg.head_dim = 8;
    cfg.inactive_threshold = 0.0f;
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_OK);
    cr_assert_eq(oc_helix_cache_store_cold_page(&cache, 0, 0, 0, keys, values,
                                                positions, 1),
                 OC_OK);
    cr_assert(oc_helix_cache_cold_page_view(&cache, 0, &view));
    cr_assert((view.active_mask[0] & 1u) != 0, "nonzero pair stays active");
    cr_assert((view.active_mask[0] & 0x0Eu) == 0,
              "zero-magnitude pairs must not be marked active");
    cr_assert(isfinite(view.log_rho_min[0]));
    cr_assert(fabsf(view.log_rho_min[1]) < 1.0e-12f);
    cr_assert_eq(oc_helix_cache_logits(&cache, 0, 0, query, 8, 0, 10000.0f,
                                       logits, 1, &n_out),
                 OC_OK);
    cr_assert_eq(n_out, (size_t)1);
    cr_assert(isfinite(logits[0]), "zero-rho pairs must not poison logits");
    oc_helix_cache_free(&cache);
}

Test(helix_cache, rewind_trims_tokens_inside_page)
{
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    OcHelixColdPageView view;
    const float keys[] = {
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    };
    const float values[32] = {0};
    const size_t positions[] = {0, 1, 2, 3};
    oc_helix_cache_config_init(&cfg);
    cfg.page_size = 4;
    cfg.head_dim = 8;
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_OK);
    cr_assert_eq(oc_helix_cache_store_cold_page(&cache, 0, 0, 0, keys, values,
                                                positions, 4),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_rewind(&cache, 2), OC_OK);
    cr_assert_eq(oc_helix_cache_page_count(&cache), (size_t)1);
    cr_assert_eq(oc_helix_cache_n_logits(&cache, 0, 0), (size_t)2);
    cr_assert(oc_helix_cache_cold_page_view(&cache, 0, &view));
    cr_assert_eq(view.tokens, (size_t)2);
    cr_assert_eq(view.positions[0], (size_t)0);
    cr_assert_eq(view.positions[1], (size_t)1);
    oc_helix_cache_free(&cache);
}

Test(helix_cache, rewind_then_append_preserves_prefix)
{
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    const float keys[] = {
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    };
    const float values[32] = {0};
    const float extra_k[8] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    const float extra_v[8] = {0};
    const size_t positions[] = {0, 1, 2, 3};
    const size_t extra_pos[] = {2};
    oc_helix_cache_config_init(&cfg);
    cfg.page_size = 4;
    cfg.head_dim = 8;
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_OK);
    cr_assert_eq(oc_helix_cache_store_cold_page(&cache, 0, 0, 0, keys, values,
                                                positions, 4),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_rewind(&cache, 2), OC_OK);
    cr_assert_eq(oc_helix_cache_n_logits(&cache, 0, 0), (size_t)2);
    cr_assert_eq(oc_helix_cache_append(&cache, 0, 0, extra_k, extra_v,
                                       extra_pos, 1),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_n_logits(&cache, 0, 0), (size_t)3);
    oc_helix_cache_free(&cache);
}

Test(helix_cache, append_at_next_page_does_not_extend_page0)
{
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    float keys[8], values[8];
    size_t pos0 = 0, pos64 = 64;
    size_t i;
    oc_helix_cache_config_init(&cfg);
    cfg.page_size = 64;
    cfg.head_dim = 8;
    for (i = 0; i < 8; i++) {
        keys[i] = 1.0f;
        values[i] = 0.0f;
    }
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_OK);
    cr_assert_eq(oc_helix_cache_append(&cache, 0, 0, keys, values, &pos0, 1),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_append(&cache, 0, 0, keys, values, &pos64, 1),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_page_count(&cache), (size_t)2);
    cr_assert_eq(oc_helix_cache_n_logits(&cache, 0, 0), (size_t)2);
    oc_helix_cache_free(&cache);
}

Test(helix_cache, append_batch_splits_at_page_boundary)
{
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    float keys[24], values[24];
    size_t pos[] = {3, 4, 5};
    size_t i;
    oc_helix_cache_config_init(&cfg);
    cfg.page_size = 4;
    cfg.head_dim = 8;
    for (i = 0; i < 24; i++) {
        keys[i] = 1.0f;
        values[i] = 0.0f;
    }
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_OK);
    cr_assert_eq(oc_helix_cache_append(&cache, 0, 0, keys, values, pos, 3),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_page_count(&cache), (size_t)2);
    cr_assert_eq(oc_helix_cache_n_logits(&cache, 0, 0), (size_t)3);
    oc_helix_cache_free(&cache);
}

Test(helix_cache, append_to_full_page_is_rejected)
{
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    const float keys[] = {
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    };
    const float values[32] = {0};
    const float extra[8] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    const size_t positions[] = {0, 1, 2, 3};
    const size_t overflow = 3;
    oc_helix_cache_config_init(&cfg);
    cfg.page_size = 4;
    cfg.head_dim = 8;
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_OK);
    cr_assert_eq(oc_helix_cache_store_cold_page(&cache, 0, 0, 0, keys, values,
                                                positions, 4),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_append(&cache, 0, 0, extra, extra, &overflow, 1),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_helix_cache_page_count(&cache), (size_t)1);
    cr_assert_eq(oc_helix_cache_n_logits(&cache, 0, 0), (size_t)4);
    oc_helix_cache_free(&cache);
}

Test(helix_cache, non_positive_rope_theta_is_rejected)
{
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    const float keys[] = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };
    const float values[8] = {0};
    const size_t positions[] = {0};
    const float query[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    float logits[1], out[8];
    size_t n_out = 0;
    oc_helix_cache_config_init(&cfg);
    cfg.page_size = 1;
    cfg.head_dim = 8;
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_OK);
    cr_assert_eq(oc_helix_cache_store_cold_page(&cache, 0, 0, 0, keys, values,
                                                positions, 1),
                 OC_OK);
    cr_assert_eq(oc_helix_cache_logits(&cache, 0, 0, query, 8, 0, 0.0f,
                                       logits, 1, &n_out),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_helix_cache_attention(&cache, 0, 0, query, 8, 0, -1.0f,
                                          out),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_helix_cache_logits(&cache, 0, 0, query, 8, 0, nanf(""),
                                       logits, 1, &n_out),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_helix_cache_attention(&cache, 0, 0, query, 8, 0, INFINITY,
                                          out),
                 OC_ERR_INVALID_ARG);
    oc_helix_cache_free(&cache);
}

Test(helix_cache, odd_rope_dim_is_rejected)
{
    OcHelixCacheConfig cfg;
    OcHelixCache cache;
    oc_helix_cache_config_init(&cfg);
    cfg.head_dim = 8;
    cfg.rope_dim = 3;
    cr_assert_eq(oc_helix_cache_init(&cache, &cfg), OC_ERR_INVALID_ARG);
}
