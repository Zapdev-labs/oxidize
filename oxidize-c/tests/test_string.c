/* test_string.c — string util helpers tests. */
#include <criterion/criterion.h>
#include "oxidize/util/string.h"

#include <stdlib.h>
#include <string.h>

Test(string, strdup_and_strndup)
{
    char *s = oc_strdup("hello");
    cr_assert_str_eq(s, "hello", "");
    free(s);

    char *sn = oc_strndup("hello world", 5);
    cr_assert_str_eq(sn, "hello", "");
    free(sn);

    cr_assert_null(oc_strdup(NULL), "");
    cr_assert_null(oc_strndup(NULL, 5), "");
}

Test(string, strcmp_null_safe)
{
    cr_assert_eq(oc_strcmp(NULL, NULL), 0, "");
    cr_assert(oc_strcmp(NULL, "x") < 0, "");
    cr_assert(oc_strcmp("x", NULL) > 0, "");
    cr_assert_eq(oc_strcmp("abc", "abc"), 0, "");
    cr_assert(oc_strcmp("abc", "abd") < 0, "");
}

Test(string, starts_ends_with)
{
    cr_assert(oc_starts_with("hello world", "hello"), "");
    cr_assert(!oc_starts_with("hello", "world"), "");
    cr_assert(oc_ends_with("hello.txt", ".txt"), "");
    cr_assert(!oc_ends_with("hello.txt", ".exe"), "");
    cr_assert(!oc_starts_with(NULL, "x"), "");
    cr_assert(!oc_ends_with("x", NULL), "");
}

Test(string, split_once)
{
    char *left = NULL;
    const char *rest = oc_split_once("a,b,c", ',', &left);
    cr_assert_str_eq(left, "a", "");
    cr_assert_str_eq(rest, "b,c", "");
    free(left);

    left = NULL;
    rest = oc_split_once("nodelim", '-', &left);
    cr_assert_null(left, "no delim -> left NULL");
    cr_assert_str_eq(rest, "nodelim", "rest is whole string");
}

Test(string, trim)
{
    char buf[] = "   hello   ";
    char *t = oc_trim(buf);
    cr_assert_str_eq(t, "hello", "");

    char buf2[] = "\t\n  x  \n";
    cr_assert_str_eq(oc_trim(buf2), "x", "");
}

Test(string, parse_i64_f64)
{
    long long i;
    cr_assert(oc_parse_i64("12345", &i), "");
    cr_assert_eq(i, 12345, "");
    cr_assert(oc_parse_i64("-42", &i), "");
    cr_assert_eq(i, -42, "");
    cr_assert(!oc_parse_i64("12x", &i), "");
    cr_assert(!oc_parse_i64("", &i), "");

    double d;
    cr_assert(oc_parse_f64("3.14", &d), "");
    cr_assert(d > 3.13 && d < 3.15, "");
    cr_assert(!oc_parse_f64("not-a-num", &d), "");
}
