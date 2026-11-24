/* test_unicode_case.c — Criterion tests for src/util/unicode_case.c. */
#include <criterion/criterion.h>

#include "oxidize/unicode.h"

Test(unicode_case, ascii)
{
    cr_assert(oc_unicode_is_upper('A') && !oc_unicode_is_lower('A'));
    cr_assert(oc_unicode_is_lower('z') && !oc_unicode_is_upper('z'));
    cr_assert(!oc_unicode_is_upper('1') && !oc_unicode_is_lower(' '));
}

Test(unicode_case, beyond_ascii)
{
    cr_assert(oc_unicode_is_upper(0x00C9));   /* É */
    cr_assert(oc_unicode_is_lower(0x00E9));   /* é */
    cr_assert(oc_unicode_is_upper(0x01C5));   /* Dž titlecase */
    cr_assert(!oc_unicode_is_upper(0x6771) && !oc_unicode_is_lower(0x6771));   /* 東: Lo */
    cr_assert(oc_unicode_is_lower(0x03B1) && oc_unicode_is_upper(0x0391));     /* α Α */
    cr_assert(!oc_unicode_is_lower(0x110000));
}
