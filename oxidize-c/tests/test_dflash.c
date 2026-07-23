/* test_dflash.c — DFlash speculative decoding tests. */
#include <criterion/criterion.h>
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
