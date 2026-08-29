#include <criterion/criterion.h>

#include <string.h>

Test(smoke, framework_loaded)
{
    /* If Criterion's headers + libcriterion.a link correctly, this test runs. */
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
