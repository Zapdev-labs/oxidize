/* test_smoke.c — placeholder smoke test for the Criterion test harness.
 *
 * This file exists so the test-harness wiring can demonstrate that
 * the vendored Criterion (MIT) framework is wired up correctly:
 *
 *   - `make test` compiles every tests/test_*.c (including this one) against
 *     tests/framework_main.c and links them into a single
 *     `test_runner` binary.
 *   - Criterion auto-registers each `Test(suite, case)` and provides `main()`.
 *   - The smoke test exercises the canonical Criterion assertion API
 *     (cr_assert, cr_expect, cr_assert_eq, cr_assert_str_eq) so downstream
 *     workers can copy this file as a template when adding new test_<module>.c
 *     files.
 *
 * When real module tests are added, keep this file as a permanent harness
 * sanity check (it verifies the framework itself works).
 */
#include "framework.h"

#include <string.h>

Test(smoke, framework_loaded)
{
    /* If the framework header + runner link correctly, this test runs. */
    cr_expect(1, "Criterion is wired up");
}

Test(smoke, arithmetic)
{
    cr_assert_eq(2 + 2, 4, "basic arithmetic should work");
}

Test(smoke, string_compare)
{
    cr_assert_str_eq("oxidize-c", "oxidize-c", "literal compare");
}

Test(smoke, pointer_nullness)
{
    int x = 0;
    cr_assert_not_null(&x, "address-of is non-null");
    int *p = NULL;
    cr_assert_null(p, "NULL pointer is null");
}
