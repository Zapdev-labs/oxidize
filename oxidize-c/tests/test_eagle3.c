/* test_eagle3.c — Eagle-3 speculative decoding tests. */
#include <criterion/criterion.h>
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

Test(eagle, config_init_null)
{
    cr_assert_neq(oc_eagle_config_init(NULL), OC_OK);
}

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

Test(eagle, state_init_null)
{
    cr_assert_neq(oc_eagle_state_init(NULL, NULL), OC_OK);
}

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

Test(eagle, n_draft_null)
{
    cr_assert_eq(oc_eagle_n_draft(NULL), 0);
}

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

Test(eagle, free_null)
{
    oc_eagle_state_free(NULL);
}

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
