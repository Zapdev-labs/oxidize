/* test_strix.c — Structured output tests. */
#include "framework.h"
#include "oxidize/strix.h"
#include <string.h>
#include <math.h>

Test(strix, config_init)
{
    OcStrixConfig cfg;
    cr_assert_eq(oc_strix_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.mode, OC_STRIX_MODE_NONE);
    cr_assert(!cfg.strict);
}

OC_TEST_NULL_SAFE(strix, config_init_null,
        cr_assert_neq(oc_strix_config_init(NULL), OC_OK);)

Test(strix, config_json)
{
    OcStrixConfig cfg;
    cr_assert_eq(oc_strix_config_json(&cfg, "{}", true), OC_OK);
    cr_assert_eq(cfg.mode, OC_STRIX_MODE_JSON);
    cr_assert(cfg.strict);
    cr_assert_str_eq(cfg.schema, "{}");
}

Test(strix, config_json_no_schema)
{
    OcStrixConfig cfg;
    cr_assert_eq(oc_strix_config_json(&cfg, NULL, false), OC_OK);
    cr_assert_eq(cfg.mode, OC_STRIX_MODE_JSON);
    cr_assert_eq(cfg.schema[0], '\0');
}

Test(strix, config_regex)
{
    OcStrixConfig cfg;
    cr_assert_eq(oc_strix_config_regex(&cfg, "[0-9]+"), OC_OK);
    cr_assert_eq(cfg.mode, OC_STRIX_MODE_REGEX);
    cr_assert_str_eq(cfg.schema, "[0-9]+");
    cr_assert(cfg.strict);
}

Test(strix, state_init)
{
    OcStrixConfig cfg;
    oc_strix_config_json(&cfg, NULL, true);
    char output[256];
    OcStrixState state;
    cr_assert_eq(oc_strix_state_init(&state, &cfg, output, sizeof(output)), OC_OK);
    cr_assert_eq(state.output_len, 0);
    cr_assert(state.expecting_value);
    oc_strix_free(&state);
}

OC_TEST_NULL_SAFE(strix, state_init_null,
        cr_assert_neq(oc_strix_state_init(NULL, NULL, NULL, 0), OC_OK);)

Test(strix, accept_char)
{
    char output[256];
    OcStrixState state;
    oc_strix_state_init(&state, NULL, output, sizeof(output));
    cr_assert_eq(oc_strix_accept_char(&state, 'h'), OC_OK);
    cr_assert_eq(state.output_len, 1);
    cr_assert_str_eq(state.output, "h");
    oc_strix_free(&state);
}

Test(strix, accept_token)
{
    char output[256];
    OcStrixState state;
    oc_strix_state_init(&state, NULL, output, sizeof(output));
    cr_assert_eq(oc_strix_accept_token(&state, 0, "hello"), OC_OK);
    cr_assert_str_eq(state.output, "hello");
    oc_strix_free(&state);
}

OC_TEST_NULL_SAFE(strix, accept_token_null,
        cr_assert_neq(oc_strix_accept_token(NULL, 0, NULL), OC_OK);)

Test(strix, json_complete)
{
    char output[256];
    OcStrixConfig cfg;
    oc_strix_config_json(&cfg, NULL, true);
    OcStrixState state;
    oc_strix_state_init(&state, &cfg, output, sizeof(output));
    cr_assert(!oc_strix_is_complete(&state));
    oc_strix_accept_char(&state, '{');
    cr_assert(!oc_strix_is_complete(&state));
    oc_strix_accept_char(&state, '}');
    cr_assert(oc_strix_is_complete(&state));
    oc_strix_free(&state);
}

Test(strix, json_reject_after_complete)
{
    char output[256];
    OcStrixConfig cfg;
    oc_strix_config_json(&cfg, NULL, true);
    OcStrixState state;
    oc_strix_state_init(&state, &cfg, output, sizeof(output));
    oc_strix_accept_char(&state, '{');
    oc_strix_accept_char(&state, '}');
    cr_assert(oc_strix_is_complete(&state));
    cr_assert(!oc_strix_is_valid_token(&state, 0, "x"));
    oc_strix_free(&state);
}

Test(strix, none_mode_accepts_all)
{
    char output[256];
    OcStrixState state;
    oc_strix_state_init(&state, NULL, output, sizeof(output));
    cr_assert(oc_strix_is_valid_token(&state, 0, "anything"));
    cr_assert(oc_strix_is_valid_token(&state, 1, "{}[]"));
    oc_strix_free(&state);
}

Test(strix, mode_name)
{
    cr_assert_str_eq(oc_strix_mode_name(OC_STRIX_MODE_NONE), "none");
    cr_assert_str_eq(oc_strix_mode_name(OC_STRIX_MODE_JSON), "json");
    cr_assert_str_eq(oc_strix_mode_name(OC_STRIX_MODE_REGEX), "regex");
    cr_assert_str_eq(oc_strix_mode_name(OC_STRIX_MODE_GRAMMAR), "grammar");
}

Test(strix, json_type_name)
{
    cr_assert_str_eq(oc_strix_json_type_name(OC_STRIX_JSON_OBJECT), "object");
    cr_assert_str_eq(oc_strix_json_type_name(OC_STRIX_JSON_ARRAY), "array");
    cr_assert_str_eq(oc_strix_json_type_name(OC_STRIX_JSON_STRING), "string");
    cr_assert_str_eq(oc_strix_json_type_name(OC_STRIX_JSON_NUMBER), "number");
}

Test(strix, filter_logits)
{
    char output[256];
    OcStrixConfig cfg;
    oc_strix_config_json(&cfg, NULL, true);
    OcStrixState state;
    oc_strix_state_init(&state, &cfg, output, sizeof(output));
    oc_strix_accept_char(&state, '{');
    oc_strix_accept_char(&state, '}');
    /* Complete - all tokens should be filtered to -inf. */
    float logits[] = {1.0f, 2.0f, 3.0f};
    const char *texts[] = {"a", "b", "c"};
    cr_assert_eq(oc_strix_filter_logits(&state, logits, 3, texts, 3), OC_OK);
    /* All should be -inf since JSON is complete. */
    for (int i = 0; i < 3; i++)
        cr_assert(isinf(logits[i]) && logits[i] < 0);
    oc_strix_free(&state);
}

Test(strix, filter_logits_none_mode)
{
    char output[256];
    OcStrixState state;
    oc_strix_state_init(&state, NULL, output, sizeof(output));
    float logits[] = {1.0f, 2.0f};
    const char *texts[] = {"a", "b"};
    oc_strix_filter_logits(&state, logits, 2, texts, 2);
    /* None mode should not filter. */
    cr_assert_float_eq(logits[0], 1.0f, 0.001f);
    cr_assert_float_eq(logits[1], 2.0f, 0.001f);
    oc_strix_free(&state);
}

Test(strix, finalize)
{
    char output[256];
    OcStrixState state;
    oc_strix_state_init(&state, NULL, output, sizeof(output));
    cr_assert_eq(oc_strix_finalize(&state), OC_OK);
    cr_assert(!state.expecting_value);
    oc_strix_free(&state);
}

OC_TEST_NULL_SAFE(strix, free_null,
        oc_strix_free(NULL);)

Test(strix, accept_overflow)
{
    char output[4];
    OcStrixState state;
    oc_strix_state_init(&state, NULL, output, sizeof(output));
    cr_assert_eq(oc_strix_accept_char(&state, 'a'), OC_OK);
    cr_assert_eq(oc_strix_accept_char(&state, 'b'), OC_OK);
    cr_assert_eq(oc_strix_accept_char(&state, 'c'), OC_OK);
    /* Fourth char should fail - buffer full (need space for null). */
    cr_assert_neq(oc_strix_accept_char(&state, 'd'), OC_OK);
    oc_strix_free(&state);
}
