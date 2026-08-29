/* test_gen_loop.c — Full generation loop tests. */
#include "framework.h"
#include "tiny_model.h"
#include "oxidize/generation.h"
#include "oxidize/inf_model.h"
#include "oxidize/weight_storage.h"
#include "oxidize/layer_weights.h"
#include "oxidize/inference.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int s_callback_count = 0;
static int token_cb(uint32_t token, const char *piece, void *user)
{
    (void)token; (void)piece; (void)user;
    s_callback_count++;
    return 0;
}

Test(gen_loop, basic_run)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 64);

    OcGenConfig cfg;
    oc_gen_config_init(&cfg);
    cfg.max_tokens = 5;
    cfg.temperature = 0.0f;  /* greedy */
    cfg.stop_on_eos = false;

    OcGenResult result;
    oc_gen_result_init(&result, 256);

    uint32_t prompt[] = {1, 2, 3};
    s_callback_count = 0;
    OcError e = oc_gen_run(&m, prompt, 3, &cfg, &result, token_cb, NULL);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(result.n_tokens, 5);
    cr_assert_eq(result.n_prompt_tokens, 3);
    cr_assert_eq(s_callback_count, 5);
    cr_assert_gt(result.tokens_per_sec, 0.0);

    oc_gen_result_free(&result);
    oc_inf_model_free(&m);
}

Test(gen_loop, stop_on_eos)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 64);

    /* Set token 0 as stop by using greedy with stop_on_eos. */
    OcGenConfig cfg;
    oc_gen_config_init(&cfg);
    cfg.max_tokens = 10;
    cfg.temperature = 0.0f;
    cfg.stop_on_eos = true;

    OcGenResult result;
    oc_gen_result_init(&result, 256);

    uint32_t prompt[] = {1, 2};
    OcError e = oc_gen_run(&m, prompt, 2, &cfg, &result, NULL, NULL);
    cr_assert_eq(e, OC_OK);
    /* Should stop when token 0 is generated (or at max_tokens). */
    cr_assert_leq(result.n_tokens, 10);
    cr_assert_geq(result.n_tokens, 1);

    oc_gen_result_free(&result);
    oc_inf_model_free(&m);
}

Test(gen_loop, no_prompt)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 64);

    OcGenConfig cfg;
    oc_gen_config_init(&cfg);
    cfg.max_tokens = 3;
    cfg.temperature = 0.0f;
    cfg.stop_on_eos = false;

    OcGenResult result;
    oc_gen_result_init(&result, 256);

    s_callback_count = 0;
    OcError e = oc_gen_run(&m, NULL, 0, &cfg, &result, token_cb, NULL);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(result.n_tokens, 3);
    cr_assert_eq(s_callback_count, 3);

    oc_gen_result_free(&result);
    oc_inf_model_free(&m);
}

Test(gen_loop, mtp_fallback)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 64);

    /* Model has no MTP -> should fall back to standard gen. */
    OcGenConfig cfg;
    oc_gen_config_init(&cfg);
    cfg.max_tokens = 4;
    cfg.temperature = 0.0f;
    cfg.stop_on_eos = false;

    OcGenResult result;
    oc_gen_result_init(&result, 256);

    uint32_t prompt[] = {5};
    OcError e = oc_gen_run_mtp(&m, prompt, 1, &cfg, 4, &result, NULL, NULL);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(result.n_tokens, 4);

    oc_gen_result_free(&result);
    oc_inf_model_free(&m);
}

Test(gen_loop, null_safety)
{
    OcGenConfig cfg;
    oc_gen_config_init(&cfg);
    OcGenResult result;
    oc_gen_result_init(&result, 256);

    cr_assert_neq(oc_gen_run(NULL, NULL, 0, &cfg, &result, NULL, NULL), OC_OK);
    cr_assert_neq(oc_gen_run(NULL, NULL, 0, NULL, &result, NULL, NULL), OC_OK);
    cr_assert_neq(oc_gen_run_mtp(NULL, NULL, 0, &cfg, 4, &result, NULL, NULL), OC_OK);

    oc_gen_result_free(&result);
}

Test(gen_loop, result_init_free)
{
    OcGenResult result;
    OcError e = oc_gen_result_init(&result, 128);
    cr_assert_eq(e, OC_OK);
    cr_assert_not_null(result.tokens);
    cr_assert_eq(result.n_tokens, 0);
    oc_gen_result_free(&result);
    cr_assert_null(result.tokens);
}

Test(gen_loop, config_init_defaults)
{
    OcGenConfig cfg;
    oc_gen_config_init(&cfg);
    cr_assert_eq(cfg.max_tokens, 256);
    cr_assert_float_eq(cfg.temperature, 0.8f, 0.001f);
    cr_assert_float_eq(cfg.top_p, 0.9f, 0.001f);
    cr_assert_eq(cfg.top_k, 40);
    cr_assert_float_eq(cfg.repeat_penalty, 1.1f, 0.001f);
    cr_assert(cfg.stream);
    cr_assert(cfg.stop_on_eos);
}

Test(gen_loop, stop_reason)
{
    OcGenResult result;
    memset(&result, 0, sizeof(result));
    cr_assert_str_eq(oc_gen_stop_reason(&result), "error");

    result.n_tokens = 5;
    result.stopped_on_eos = false;
    cr_assert_str_eq(oc_gen_stop_reason(&result), "length");

    result.stopped_on_eos = true;
    cr_assert_str_eq(oc_gen_stop_reason(&result), "stop");

    cr_assert_str_eq(oc_gen_stop_reason(NULL), "unknown");
}
