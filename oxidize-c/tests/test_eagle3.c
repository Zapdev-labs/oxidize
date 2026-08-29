/* test_eagle3.c — Eagle-3 speculative decoding tests. */
#include "framework.h"
#include "oxidize/eagle3.h"
#include <string.h>

Test(eagle, config_init)
{
    OcEagleConfig cfg;
    cr_assert_eq(oc_eagle_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.max_draft_tokens, 4);
    cr_assert_eq(cfg.n_layers, 2);
    cr_assert_eq(cfg.hidden_dim, 1024);
    cr_assert(cfg.dynamic_draft);
}

OC_TEST_NULL_SAFE(eagle, config_init_null,
        cr_assert_neq(oc_eagle_config_init(NULL), OC_OK);)

Test(eagle, state_init)
{
    OcEagleConfig cfg;
    oc_eagle_config_init(&cfg);
    OcEagleState state;
    cr_assert_eq(oc_eagle_state_init(&state, &cfg), OC_OK);
    cr_assert(state.initialized);
    cr_assert_eq(state.n_draft, 0);
    oc_eagle_state_free(&state);
}

OC_TEST_NULL_SAFE(eagle, state_init_null,
        cr_assert_neq(oc_eagle_state_init(NULL, NULL), OC_OK);)

Test(eagle, state_init_default)
{
    OcEagleState state;
    cr_assert_eq(oc_eagle_state_init(&state, NULL), OC_OK);
    cr_assert(state.initialized);
    oc_eagle_state_free(&state);
}

Test(eagle, generate_draft)
{
    OcEagleState state;
    oc_eagle_state_init(&state, NULL);
    uint32_t ctx[] = {1, 2, 3};
    cr_assert_eq(oc_eagle_generate_draft(&state, ctx, 3, 4), OC_OK);
    cr_assert_eq(state.n_draft, 4);
    const uint32_t *tokens;
    uint32_t count;
    oc_eagle_get_draft_tokens(&state, &tokens, &count);
    cr_assert_eq(count, 4);
    for (uint32_t i = 0; i < count; i++)
        cr_assert(tokens[i] < OC_EAGLE_VOCAB_SIZE);
    oc_eagle_state_free(&state);
}

Test(eagle, generate_draft_null)
{
    cr_assert_neq(oc_eagle_generate_draft(NULL, NULL, 0, 0), OC_OK);
    OcEagleState state;
    oc_eagle_state_init(&state, NULL);
    cr_assert_neq(oc_eagle_generate_draft(&state, NULL, 0, 0), OC_OK);
    oc_eagle_state_free(&state);
}

Test(eagle, get_draft_tokens)
{
    OcEagleState state;
    oc_eagle_state_init(&state, NULL);
    uint32_t ctx[] = {42};
    oc_eagle_generate_draft(&state, ctx, 1, 3);
    const uint32_t *tokens;
    uint32_t count;
    cr_assert_eq(oc_eagle_get_draft_tokens(&state, &tokens, &count), OC_OK);
    cr_assert_eq(count, 3);
    oc_eagle_state_free(&state);
}

Test(eagle, get_draft_probs)
{
    OcEagleState state;
    oc_eagle_state_init(&state, NULL);
    uint32_t ctx[] = {42};
    oc_eagle_generate_draft(&state, ctx, 1, 2);
    const float *probs;
    uint32_t count;
    cr_assert_eq(oc_eagle_get_draft_probs(&state, &probs, &count), OC_OK);
    cr_assert_eq(count, 2);
    for (uint32_t i = 0; i < count; i++)
        cr_assert(probs[i] >= 0.0f && probs[i] <= 1.0f);
    oc_eagle_state_free(&state);
}

Test(eagle, n_draft)
{
    OcEagleState state;
    oc_eagle_state_init(&state, NULL);
    cr_assert_eq(oc_eagle_n_draft(&state), 0);
    uint32_t ctx[] = {1};
    oc_eagle_generate_draft(&state, ctx, 1, 4);
    cr_assert_eq(oc_eagle_n_draft(&state), 4);
    oc_eagle_state_free(&state);
}

OC_TEST_NULL_SAFE(eagle, n_draft_null,
        cr_assert_eq(oc_eagle_n_draft(NULL), 0);)

Test(eagle, update_acceptance)
{
    OcEagleState state;
    oc_eagle_state_init(&state, NULL);
    uint32_t ctx[] = {1};
    oc_eagle_generate_draft(&state, ctx, 1, 4);
    cr_assert_eq(oc_eagle_update_acceptance(&state, 2), OC_OK);
    float rate = oc_eagle_acceptance_rate(&state);
    cr_assert(rate > 0.0f);
    oc_eagle_state_free(&state);
}

Test(eagle, acceptance_rate_no_data)
{
    OcEagleState state;
    oc_eagle_state_init(&state, NULL);
    cr_assert_float_eq(oc_eagle_acceptance_rate(&state), 0.0f, 0.001f);
    oc_eagle_state_free(&state);
}

Test(eagle, generate_custom_max)
{
    OcEagleState state;
    oc_eagle_state_init(&state, NULL);
    uint32_t ctx[] = {100};
    oc_eagle_generate_draft(&state, ctx, 1, 2);
    cr_assert_eq(state.n_draft, 2);
    oc_eagle_state_free(&state);
}

OC_TEST_NULL_SAFE(eagle, free_null,
        oc_eagle_state_free(NULL);)

Test(eagle, multiple_generate)
{
    OcEagleState state;
    oc_eagle_state_init(&state, NULL);
    uint32_t ctx[] = {1, 2, 3};
    oc_eagle_generate_draft(&state, ctx, 3, 4);
    cr_assert_eq(state.n_draft, 4);
    oc_eagle_generate_draft(&state, ctx, 3, 2);
    cr_assert_eq(state.n_draft, 2);
    oc_eagle_state_free(&state);
}

/* ─── Real Eagle3 draft model tests ─────────────────────────────────── */

Test(eagle3, config_init)
{
    OcEagle3Config cfg;
    oc_eagle3_config_init(&cfg);
    cr_assert_eq(cfg.hidden_size, 4096);
    cr_assert_eq(cfg.num_hidden_layers, 1);
    cr_assert_eq(cfg.n_extract_layers, 3);
    cr_assert_eq(cfg.vocab_size, 128256);
}

Test(eagle3, head_dim)
{
    OcEagle3Config cfg;
    oc_eagle3_config_init(&cfg);
    cfg.hidden_size = 512;
    cfg.num_attention_heads = 8;
    cfg.head_dim = 0;
    cr_assert_eq(oc_eagle3_config_head_dim(&cfg), 64);

    cfg.head_dim = 128;
    cr_assert_eq(oc_eagle3_config_head_dim(&cfg), 128);
}

Test(eagle3, encoder_input_width)
{
    OcEagle3Config cfg;
    oc_eagle3_config_init(&cfg);
    cfg.target_hidden_size = 4096;
    cfg.n_extract_layers = 3;
    cr_assert_eq(oc_eagle3_config_encoder_input_width(&cfg), 3 * 4096);
}

Test(eagle3, model_init_free)
{
    OcEagle3Config cfg;
    oc_eagle3_config_init(&cfg);
    cfg.hidden_size = 8;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 16;
    cfg.vocab_size = 16;
    cfg.draft_vocab_size = 16;
    cfg.target_hidden_size = 8;

    OcEagle3DraftModel m;
    cr_assert_eq(oc_eagle3_model_init(&m, &cfg), OC_OK);
    cr_assert(m.loaded);
    cr_assert_not_null(m.g_embeddings);
    cr_assert_not_null(m.kv_caches);
    cr_assert_eq(m.position_offset, 0);

    oc_eagle3_model_free(&m);
    cr_assert(!m.loaded);
}

Test(eagle3, encode_features_no_fc)
{
    OcEagle3Config cfg;
    oc_eagle3_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.target_hidden_size = 4;
    cfg.n_extract_layers = 2;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 8;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 8;

    OcEagle3DraftModel m;
    oc_eagle3_model_init(&m, &cfg);

    /* Without fc weights, should copy first hidden_size elements. */
    float features[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    cr_assert_eq(oc_eagle3_encode_features(&m, features, 8), OC_OK);
    cr_assert_float_eq(m.g_embeddings[0], 1.0f, 1e-5f);
    cr_assert_float_eq(m.g_embeddings[3], 4.0f, 1e-5f);

    oc_eagle3_model_free(&m);
}

Test(eagle3, encode_features_wrong_len)
{
    OcEagle3Config cfg;
    oc_eagle3_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.target_hidden_size = 4;
    cfg.n_extract_layers = 2;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 8;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 8;

    OcEagle3DraftModel m;
    oc_eagle3_model_init(&m, &cfg);

    float features[4] = {0};
    cr_assert_neq(oc_eagle3_encode_features(&m, features, 4), OC_OK);

    oc_eagle3_model_free(&m);
}

Test(eagle3, forward_decoder_no_weights)
{
    /* Test that forward_decoder runs without crashing when no weights are loaded.
     * Output should be all zeros (or close to it). */
    OcEagle3Config cfg;
    oc_eagle3_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 8;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 8;
    cfg.target_hidden_size = 4;
    cfg.n_extract_layers = 1;

    OcEagle3DraftModel m;
    oc_eagle3_model_init(&m, &cfg);

    /* Set g_embeddings to something. */
    float features[4] = {1, 2, 3, 4};
    oc_eagle3_encode_features(&m, features, 4);

    float hidden[4] = {0};
    float logits[8] = {0};
    OcError e = oc_eagle3_forward_decoder(&m, 1, hidden, 4, logits, 8);
    cr_assert_eq(e, OC_OK);

    /* Position should have advanced. */
    cr_assert_eq(m.position_offset, 1);

    /* KV cache should have 1 entry. */
    cr_assert_eq(m.kv_caches[0].seq_len, 1);

    oc_eagle3_model_free(&m);
}

Test(eagle3, reset_cache)
{
    OcEagle3Config cfg;
    oc_eagle3_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 8;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 8;
    cfg.target_hidden_size = 4;
    cfg.n_extract_layers = 1;

    OcEagle3DraftModel m;
    oc_eagle3_model_init(&m, &cfg);

    /* Run a forward pass to populate cache. */
    float features[4] = {1, 2, 3, 4};
    oc_eagle3_encode_features(&m, features, 4);
    float hidden[4], logits[8];
    oc_eagle3_forward_decoder(&m, 1, hidden, 4, logits, 8);
    cr_assert_eq(m.kv_caches[0].seq_len, 1);
    cr_assert_eq(m.position_offset, 1);

    /* Reset. */
    oc_eagle3_reset_cache(&m);
    cr_assert_eq(m.kv_caches[0].seq_len, 0);
    cr_assert_eq(m.position_offset, 0);

    oc_eagle3_model_free(&m);
}

Test(eagle3, reserve_cache)
{
    OcEagle3Config cfg;
    oc_eagle3_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 8;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 8;
    cfg.target_hidden_size = 4;
    cfg.n_extract_layers = 1;

    OcEagle3DraftModel m;
    oc_eagle3_model_init(&m, &cfg);

    size_t orig_cap = m.kv_caches[0].capacity;
    cr_assert_eq(oc_eagle3_reserve_cache_tokens(&m, 512), OC_OK);
    cr_assert_geq(m.kv_caches[0].capacity, 512);

    oc_eagle3_model_free(&m);
    (void)orig_cap;
}

Test(eagle3, forward_multiple_tokens)
{
    OcEagle3Config cfg;
    oc_eagle3_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 8;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 8;
    cfg.target_hidden_size = 4;
    cfg.n_extract_layers = 1;

    OcEagle3DraftModel m;
    oc_eagle3_model_init(&m, &cfg);

    float features[4] = {0.5f, 0.3f, -0.2f, 0.1f};
    oc_eagle3_encode_features(&m, features, 4);

    float hidden[4], logits[8];
    for (int t = 0; t < 5; t++) {
        OcError e = oc_eagle3_forward_decoder(&m, (uint32_t)t, hidden, 4, logits, 8);
        cr_assert_eq(e, OC_OK);
        cr_assert_eq(m.position_offset, (size_t)(t + 1));
        cr_assert_eq(m.kv_caches[0].seq_len, (size_t)(t + 1));
    }

    oc_eagle3_model_free(&m);
}

Test(eagle3, null_safety)
{
    cr_assert_neq(oc_eagle3_model_init(NULL, NULL), OC_OK);
    cr_assert_neq(oc_eagle3_encode_features(NULL, NULL, 0), OC_OK);
    cr_assert_neq(oc_eagle3_forward_decoder(NULL, 0, NULL, 0, NULL, 0), OC_OK);
    cr_assert_neq(oc_eagle3_logits_from_hidden(NULL, NULL, 0, NULL, 0), OC_OK);
    cr_assert_neq(oc_eagle3_reserve_cache_tokens(NULL, 0), OC_OK);
    oc_eagle3_reset_cache(NULL);
    oc_eagle3_model_free(NULL);
}

Test(eagle3, logits_from_hidden)
{
    OcEagle3Config cfg;
    oc_eagle3_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 8;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 8;
    cfg.target_hidden_size = 4;
    cfg.n_extract_layers = 1;

    OcEagle3DraftModel m;
    oc_eagle3_model_init(&m, &cfg);

    float hidden[4] = {1.0f, 0.5f, -0.3f, 0.2f};
    float logits[8] = {0};
    /* Without output weights, logits should be zero. */
    OcError e = oc_eagle3_logits_from_hidden(&m, hidden, 4, logits, 8);
    cr_assert_eq(e, OC_OK);

    oc_eagle3_model_free(&m);
}
