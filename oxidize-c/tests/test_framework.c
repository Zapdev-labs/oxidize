#include "framework.h"

Test(framework, skip_is_skip)
{
    cr_skip_test("runner must record this as SKIP, not PASS");
}

Test(framework, assert_eq_passes)
{
    cr_assert_eq(1, 1);
    cr_assert_neq(1, 2);
    cr_assert_leq(1, 1);
    cr_assert_geq(2, 1);
}

Test(framework, str_eq)
{
    cr_assert_str_eq("ok", "ok");
}

Test(framework, formatted_extra_args_bind_after_operands)
{
    int n = 7;
    cr_assert_eq(1, 1, "n=%d", n);
    cr_expect_neq(1, 2, "n=%d", n);
}

Test(framework, disabled_does_not_run, .disabled = true)
{
    cr_fail("disabled test must not execute");
}

Test(framework, filter_exact_case_skips_underscore_rewrite)
{
    OcTest exact = {"vpre", "config_init", NULL, 0, NULL, NULL};
    OcTest other = {"config", "init_defaults", NULL, 0, NULL, &exact};
    int rewrite = oc_filter_use_underscore_rewrite("config_init", &other);
    cr_assert_eq(rewrite, 0);
    cr_assert(oc_filter_selects("config_init", &exact, rewrite));
    cr_assert_not(oc_filter_selects("config_init", &other, rewrite));
}

Test(framework, filter_underscore_rewrite_when_no_exact_case)
{
    OcTest t = {"kv_cache", "init_free", NULL, 0, NULL, NULL};
    int rewrite = oc_filter_use_underscore_rewrite("kv_cache_init", &t);
    cr_assert_eq(rewrite, 1);
    cr_assert(oc_filter_selects("kv_cache_init", &t, rewrite));
}

Test(framework, filter_gguf_v3_header_prefix_glob)
{
    OcTest t = {"gguf", "v3_header_parses_correctly", NULL, 0, NULL, NULL};
    int rewrite = oc_filter_use_underscore_rewrite("gguf_v3_header", &t);
    cr_assert_eq(rewrite, 1);
    cr_assert(oc_filter_selects("gguf_v3_header", &t, rewrite));
}
