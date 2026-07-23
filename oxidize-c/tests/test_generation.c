/* test_generation.c — Generation engine tests. */
#include <criterion/criterion.h>
#include "oxidize/generation.h"
#include <string.h>

Test(gen, config_init)
{
    OcGenConfig cfg;
    cr_assert_eq(oc_gen_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.max_tokens, 256);
    cr_assert_float_eq(cfg.temperature, 0.8f, 0.001f);
    cr_assert_float_eq(cfg.top_p, 0.9f, 0.001f);
    cr_assert_eq(cfg.top_k, 40);
    cr_assert(cfg.stream);
    cr_assert(cfg.stop_on_eos);
}

Test(gen, config_init_null)
{
    cr_assert_neq(oc_gen_config_init(NULL), OC_OK);
}

Test(gen, state_init_empty)
{
    OcGenState state;
    cr_assert_eq(oc_gen_state_init(&state, NULL, 0), OC_OK);
    cr_assert_eq(state.n_context, 0);
    cr_assert_eq(state.pos, 0);
}

Test(gen, state_init_with_context)
{
    OcGenState state;
    uint32_t ctx[] = {1, 2, 3, 4, 5};
    cr_assert_eq(oc_gen_state_init(&state, ctx, 5), OC_OK);
    cr_assert_eq(state.n_context, 5);
    cr_assert_eq(state.pos, 5);
    cr_assert_eq(state.n_recent, 5);
    cr_assert_eq(state.recent_tokens[0], 1);
    free(state.context);
}

Test(gen, state_init_large_context)
{
    OcGenState state;
    uint32_t ctx[5000];
    for (int i = 0; i < 5000; i++) ctx[i] = i;
    cr_assert_eq(oc_gen_state_init(&state, ctx, 5000), OC_OK);
    cr_assert_eq(state.n_context, 4096);
    cr_assert_eq(state.n_recent, 64);
    free(state.context);
}

Test(gen, state_add_token)
{
    OcGenState state;
    oc_gen_state_init(&state, NULL, 0);
    cr_assert_eq(oc_gen_state_add_token(&state, 42), OC_OK);
    cr_assert_eq(state.n_recent, 1);
    cr_assert_eq(state.recent_tokens[0], 42);
    cr_assert_eq(state.pos, 1);
}

Test(gen, state_add_many_tokens)
{
    OcGenState state;
    oc_gen_state_init(&state, NULL, 0);
    for (int i = 0; i < 100; i++)
        oc_gen_state_add_token(&state, (uint32_t)i);
    cr_assert_eq(state.n_recent, 64);
    cr_assert_eq(state.recent_tokens[63], 99);
    cr_assert_eq(state.pos, 100);
    free(state.context);
}

Test(gen, state_add_null)
{
    cr_assert_neq(oc_gen_state_add_token(NULL, 0), OC_OK);
}

Test(gen, result_init_free)
{
    OcGenResult result;
    cr_assert_eq(oc_gen_result_init(&result, 100), OC_OK);
    cr_assert_eq(result.n_tokens, 0);
    cr_assert_not_null(result.tokens);
    oc_gen_result_free(&result);
    cr_assert_null(result.tokens);
}

Test(gen, result_init_default_size)
{
    OcGenResult result;
    cr_assert_eq(oc_gen_result_init(&result, 0), OC_OK);
    cr_assert_not_null(result.tokens);
    oc_gen_result_free(&result);
}

Test(gen, result_init_null)
{
    cr_assert_neq(oc_gen_result_init(NULL, 0), OC_OK);
}

Test(gen, config_from_cli)
{
    OcGenConfig cfg;
    oc_gen_config_init(&cfg);
    cfg.top_k = 40;
    cfg.top_p = 0.8f;
    cfg.temperature = 0.7f;
    OcSamplerConfig scfg;
    cr_assert_eq(oc_gen_config_from_cli(&cfg, &scfg), OC_OK);
    cr_assert_eq(scfg.type, OC_SAMPLER_TOP_K);
    cr_assert_eq(scfg.top_k, 40);
    cr_assert_float_eq(scfg.top_p, 0.8f, 0.001f);
    cr_assert_float_eq(scfg.temperature, 0.7f, 0.001f);
}

Test(gen, config_from_cli_top_p_only)
{
    OcGenConfig cfg;
    oc_gen_config_init(&cfg);
    cfg.top_k = 0;
    cfg.top_p = 0.95f;
    OcSamplerConfig scfg;
    cr_assert_eq(oc_gen_config_from_cli(&cfg, &scfg), OC_OK);
    cr_assert_eq(scfg.type, OC_SAMPLER_TOP_P);
}

Test(gen, config_from_cli_null)
{
    cr_assert_neq(oc_gen_config_from_cli(NULL, NULL), OC_OK);
}

Test(gen, stop_reason)
{
    OcGenResult result = {0};
    result.stopped_on_eos = true;
    result.n_tokens = 5;
    cr_assert_str_eq(oc_gen_stop_reason(&result), "stop");

    result.stopped_on_eos = false;
    cr_assert_str_eq(oc_gen_stop_reason(&result), "length");

    result.n_tokens = 0;
    cr_assert_str_eq(oc_gen_stop_reason(&result), "error");
}

Test(gen, stop_reason_null)
{
    cr_assert_str_eq(oc_gen_stop_reason(NULL), "unknown");
}

Test(gen, total_tokens)
{
    OcGenResult result = {0};
    result.n_prompt_tokens = 100;
    result.n_tokens = 50;
    cr_assert_eq(oc_gen_total_tokens(&result), 150);
}

Test(gen, total_tokens_null)
{
    cr_assert_eq(oc_gen_total_tokens(NULL), 0);
}

Test(gen, result_free_null)
{
    oc_gen_result_free(NULL);
}
