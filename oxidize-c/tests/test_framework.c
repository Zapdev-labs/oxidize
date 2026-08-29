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

Test(framework, disabled_does_not_run, .disabled = true)
{
    cr_fail("disabled test must not execute");
}
