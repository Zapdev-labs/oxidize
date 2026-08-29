/* test_dflash.c — DFlash speculative decoding tests. */
#include "framework.h"
#include "oxidize/dflash.h"
#include <math.h>
#include <string.h>

Test(dflash, config_init)
{
    OcDFlashConfig cfg;
    cr_assert_eq(oc_dflash_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.max_draft_tokens, 8);
    cr_assert_eq(cfg.verification_window, 16);
    cr_assert(cfg.adaptive);
}

Test(dflash, config_init_null)
{
    cr_assert_neq(oc_dflash_config_init(NULL), OC_OK);
}

Test(dflash, state_init)
{
    OcDFlashState state;
    cr_assert_eq(oc_dflash_state_init(&state, NULL), OC_OK);
    cr_assert_eq(state.n_draft, 0);
    cr_assert_eq(state.n_accepted, 0);
    oc_dflash_state_free(&state);
}

Test(dflash, state_init_null)
{
    cr_assert_neq(oc_dflash_state_init(NULL, NULL), OC_OK);
}

Test(dflash, set_draft)
{
    OcDFlashState state;
    oc_dflash_state_init(&state, NULL);
    uint32_t tokens[] = {1, 2, 3, 4};
    float logprobs[] = {-0.5f, -0.3f, -0.2f, -0.1f};
    cr_assert_eq(oc_dflash_set_draft(&state, tokens, logprobs, 4), OC_OK);
    cr_assert_eq(state.n_draft, 4);
    cr_assert_eq(state.draft_tokens[0], 1);
    oc_dflash_state_free(&state);
}

Test(dflash, set_draft_null)
{
    cr_assert_neq(oc_dflash_set_draft(NULL, NULL, NULL, 0), OC_OK);
}

Test(dflash, set_target)
{
    OcDFlashState state;
    oc_dflash_state_init(&state, NULL);
    uint32_t tokens[] = {1, 2, 3};
    float logprobs[] = {-0.4f, -0.2f, -0.1f};
    cr_assert_eq(oc_dflash_set_target(&state, tokens, logprobs, 3), OC_OK);
    cr_assert_eq(state.target_tokens[0], 1);
    oc_dflash_state_free(&state);
}

Test(dflash, verify_all_match)
{
    OcDFlashState state;
    oc_dflash_state_init(&state, NULL);
    uint32_t draft[] = {10, 20, 30};
    float dlp[] = {-0.5f, -0.5f, -0.5f};
    oc_dflash_set_draft(&state, draft, dlp, 3);
    oc_dflash_set_target(&state, draft, dlp, 3);
    uint32_t accepted[8];
    uint32_t n;
    cr_assert_eq(oc_dflash_verify(&state, accepted, &n), OC_OK);
    cr_assert_eq(n, 3, "should accept all 3 matching tokens");
    cr_assert_eq(state.n_accepted, 3);
    oc_dflash_state_free(&state);
}

Test(dflash, verify_partial_match)
{
    OcDFlashState state;
    oc_dflash_state_init(&state, NULL);
    uint32_t draft[] = {10, 20, 30};
    uint32_t target[] = {10, 25, 30};
    float lp[] = {-0.5f, -0.5f, -0.5f};
    oc_dflash_set_draft(&state, draft, lp, 3);
    oc_dflash_set_target(&state, target, lp, 3);
    uint32_t accepted[8];
    uint32_t n;
    cr_assert_eq(oc_dflash_verify(&state, accepted, &n), OC_OK);
    cr_assert_eq(n, 2, "should accept 1 match + 1 replacement = 2");
    cr_assert_eq(accepted[0], 10);
    cr_assert_eq(accepted[1], 25);
    oc_dflash_state_free(&state);
}

Test(dflash, verify_none_match)
{
    OcDFlashState state;
    oc_dflash_state_init(&state, NULL);
    uint32_t draft[] = {10, 20};
    uint32_t target[] = {15, 25};
    float lp[] = {-0.5f, -0.5f};
    oc_dflash_set_draft(&state, draft, lp, 2);
    oc_dflash_set_target(&state, target, lp, 2);
    uint32_t accepted[8];
    uint32_t n;
    cr_assert_eq(oc_dflash_verify(&state, accepted, &n), OC_OK);
    cr_assert_eq(n, 1, "should accept 1 replacement");
    cr_assert_eq(accepted[0], 15);
    oc_dflash_state_free(&state);
}

Test(dflash, acceptance_rate)
{
    OcDFlashState state;
    oc_dflash_state_init(&state, NULL);
    cr_assert_float_eq(oc_dflash_acceptance_rate(&state), 0.0f, 0.001f);
    /* Run a verify that accepts some. */
    uint32_t draft[] = {1, 2, 3};
    uint32_t target[] = {1, 2, 3};
    float lp[] = {-0.5f, -0.5f, -0.5f};
    oc_dflash_set_draft(&state, draft, lp, 3);
    oc_dflash_set_target(&state, target, lp, 3);
    uint32_t acc[8]; uint32_t n;
    oc_dflash_verify(&state, acc, &n);
    float rate = oc_dflash_acceptance_rate(&state);
    cr_assert_float_eq(rate, 1.0f, 0.001f);
    oc_dflash_state_free(&state);
}

Test(dflash, avg_acceptance)
{
    OcDFlashState state;
    oc_dflash_state_init(&state, NULL);
    cr_assert_eq(oc_dflash_avg_acceptance(&state), 0);
    oc_dflash_state_free(&state);
}

Test(dflash, get_accepted)
{
    OcDFlashState state;
    oc_dflash_state_init(&state, NULL);
    uint32_t draft[] = {1, 2, 3};
    float lp[] = {-0.5f, -0.5f, -0.5f};
    oc_dflash_set_draft(&state, draft, lp, 3);
    oc_dflash_set_target(&state, draft, lp, 3);
    uint32_t acc[8]; uint32_t n;
    oc_dflash_verify(&state, acc, &n);
    const uint32_t *out; uint32_t out_n;
    cr_assert_eq(oc_dflash_get_accepted(&state, &out, &out_n), OC_OK);
    cr_assert_eq(out_n, 3);
    oc_dflash_state_free(&state);
}

Test(dflash, verify_null)
{
    cr_assert_neq(oc_dflash_verify(NULL, NULL, NULL), OC_OK);
}

Test(dflash, set_draft_overflow)
{
    OcDFlashState state;
    OcDFlashConfig cfg;
    oc_dflash_config_init(&cfg);
    cfg.max_draft_tokens = 2;
    oc_dflash_state_init(&state, &cfg);
    uint32_t tokens[] = {1, 2, 3, 4};
    float lp[] = {-0.5f, -0.5f, -0.5f, -0.5f};
    oc_dflash_set_draft(&state, tokens, lp, 4);
    cr_assert_eq(state.n_draft, 2);
    oc_dflash_state_free(&state);
}

Test(dflash, free_null)
{
    oc_dflash_state_free(NULL);
}

Test(dflash, multiple_rounds)
{
    OcDFlashState state;
    oc_dflash_state_init(&state, NULL);
    uint32_t draft1[] = {1, 2, 3};
    float lp[] = {-0.5f, -0.5f, -0.5f};
    oc_dflash_set_draft(&state, draft1, lp, 3);
    oc_dflash_set_target(&state, draft1, lp, 3);
    uint32_t acc[8]; uint32_t n;
    oc_dflash_verify(&state, acc, &n);

    /* Second round with different tokens. */
    uint32_t draft2[] = {10, 20};
    oc_dflash_set_draft(&state, draft2, lp, 2);
    oc_dflash_set_target(&state, draft2, lp, 2);
    oc_dflash_verify(&state, acc, &n);
    cr_assert_eq(n, 2);
    cr_assert(state.total_proposed >= 5);
    oc_dflash_state_free(&state);
}

/* ─── Real DFlash draft model tests ─────────────────────────────────── */

Test(dflash_model, config_init)
{
    OcDFlashModelConfig cfg;
    oc_dflash_model_config_init(&cfg);
    cr_assert_eq(cfg.hidden_size, 2048);
    cr_assert_eq(cfg.num_hidden_layers, 8);
    cr_assert_eq(cfg.num_target_layers, 40);
    cr_assert_eq(cfg.vocab_size, 248320);
}

Test(dflash_model, head_dim)
{
    OcDFlashModelConfig cfg;
    oc_dflash_model_config_init(&cfg);
    /* 2048 / 32 = 64 */
    cr_assert_eq(oc_dflash_config_head_dim(&cfg), 64);
}

Test(dflash_model, target_hidden_width)
{
    OcDFlashModelConfig cfg;
    oc_dflash_model_config_init(&cfg);
    /* 2048 * 40 = 81920 */
    cr_assert_eq(oc_dflash_config_target_hidden_width(&cfg), 2048 * 40);
}

Test(dflash_model, init_free)
{
    OcDFlashModelConfig cfg;
    oc_dflash_model_config_init(&cfg);
    cfg.hidden_size = 8;
    cfg.num_hidden_layers = 2;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 16;
    cfg.vocab_size = 16;
    cfg.num_target_layers = 2;

    OcDFlashDraftModel m;
    cr_assert_eq(oc_dflash_model_init(&m, &cfg), OC_OK);
    cr_assert(m.loaded);
    cr_assert_not_null(m.layers);
    cr_assert_not_null(m.kv_cache);
    cr_assert_eq(m.position_offset, 0);

    oc_dflash_model_free(&m);
    cr_assert(!m.loaded);
}

Test(dflash_model, forward_token_no_weights)
{
    OcDFlashModelConfig cfg;
    oc_dflash_model_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.num_hidden_layers = 2;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 8;
    cfg.vocab_size = 8;
    cfg.num_target_layers = 1;

    OcDFlashDraftModel m;
    cr_assert_eq(oc_dflash_model_init(&m, &cfg), OC_OK);

    float hidden[4] = {0};
    OcError e = oc_dflash_forward_token(&m, 1, NULL, 0, hidden, 4);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(m.position_offset, 1);
    cr_assert_eq(m.kv_cache[0].seq_len, 1);
    cr_assert_eq(m.kv_cache[1].seq_len, 1);

    oc_dflash_model_free(&m);
}

Test(dflash_model, forward_multiple_tokens)
{
    OcDFlashModelConfig cfg;
    oc_dflash_model_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.num_hidden_layers = 1;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 8;
    cfg.vocab_size = 8;
    cfg.num_target_layers = 1;

    OcDFlashDraftModel m;
    cr_assert_eq(oc_dflash_model_init(&m, &cfg), OC_OK);

    float hidden[4];
    for (int t = 0; t < 5; t++) {
        OcError e = oc_dflash_forward_token(&m, (uint32_t)t, NULL, 0, hidden, 4);
        cr_assert_eq(e, OC_OK);
        cr_assert_eq(m.position_offset, (size_t)(t + 1));
        cr_assert_eq(m.kv_cache[0].seq_len, (size_t)(t + 1));
    }

    oc_dflash_model_free(&m);
}

Test(dflash_model, cache_target_hidden)
{
    OcDFlashModelConfig cfg;
    oc_dflash_model_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.num_hidden_layers = 1;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 8;
    cfg.vocab_size = 8;
    cfg.num_target_layers = 2;

    OcDFlashDraftModel m;
    oc_dflash_model_init(&m, &cfg);

    /* target_hidden_width = 4 * 2 = 8 */
    float hidden[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    cr_assert_eq(oc_dflash_cache_target_hidden(&m, hidden, 8), OC_OK);
    cr_assert_eq(m.target_hidden_cache_len, 8);
    cr_assert_not_null(m.target_hidden_cache);

    /* Wrong length should fail. */
    cr_assert_neq(oc_dflash_cache_target_hidden(&m, hidden, 4), OC_OK);

    oc_dflash_model_free(&m);
}

Test(dflash_model, clear_speculative_caches)
{
    OcDFlashModelConfig cfg;
    oc_dflash_model_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.num_hidden_layers = 1;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 8;
    cfg.vocab_size = 8;
    cfg.num_target_layers = 1;

    OcDFlashDraftModel m;
    oc_dflash_model_init(&m, &cfg);

    float hidden[4];
    oc_dflash_forward_token(&m, 0, NULL, 0, hidden, 4);
    cr_assert_eq(m.kv_cache[0].seq_len, 1);
    cr_assert_eq(m.position_offset, 1);

    oc_dflash_clear_speculative_caches(&m);
    cr_assert_eq(m.kv_cache[0].seq_len, 0);
    cr_assert_eq(m.position_offset, 0);
    cr_assert_null(m.target_hidden_cache);

    oc_dflash_model_free(&m);
}

Test(dflash_model, logits)
{
    OcDFlashModelConfig cfg;
    oc_dflash_model_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.num_hidden_layers = 1;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 8;
    cfg.vocab_size = 8;
    cfg.num_target_layers = 1;

    OcDFlashDraftModel m;
    oc_dflash_model_init(&m, &cfg);

    float hidden[4] = {1.0f, 0.5f, -0.3f, 0.2f};
    float logits[8] = {0};
    /* Without output weights, logits should be zero. */
    OcError e = oc_dflash_logits(&m, hidden, 4, logits, 8);
    cr_assert_eq(e, OC_OK);

    oc_dflash_model_free(&m);
}

Test(dflash_model, null_safety)
{
    cr_assert_neq(oc_dflash_model_init(NULL, NULL), OC_OK);
    cr_assert_neq(oc_dflash_cache_target_hidden(NULL, NULL, 0), OC_OK);
    cr_assert_neq(oc_dflash_forward_token(NULL, 0, NULL, 0, NULL, 0), OC_OK);
    cr_assert_neq(oc_dflash_logits(NULL, NULL, 0, NULL, 0), OC_OK);
    oc_dflash_clear_speculative_caches(NULL);
    oc_dflash_model_free(NULL);
}

Test(dflash_model, forward_with_target_hidden)
{
    OcDFlashModelConfig cfg;
    oc_dflash_model_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.num_hidden_layers = 1;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 8;
    cfg.vocab_size = 8;
    cfg.num_target_layers = 2;

    OcDFlashDraftModel m;
    oc_dflash_model_init(&m, &cfg);

    /* Without fc weights, target_hidden is ignored. */
    float target_hidden[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float hidden[4] = {0};
    OcError e = oc_dflash_forward_token(&m, 1, target_hidden, 8, hidden, 4);
    cr_assert_eq(e, OC_OK);

    oc_dflash_model_free(&m);
}
