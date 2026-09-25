/* test_pretokenize.c — Criterion tests for src/format/pretokenize.c.
 *
 * Expected splits follow llama.cpp-k2 `unicode_regex_split_custom_k2_horizon`
 * for the regex
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|‌|‍)+
 *   |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 * The end-to-end id parity against llama.cpp on the real K2 vocab is covered
 * by scripts/tokenizer_parity.py. */

#include <criterion/criterion.h>

#include "oxidize/pretokenize.h"
#include "oxidize/unicode.h"

#include <stdint.h>
#include <string.h>

static void check_split(const char *text, const uint32_t *want, size_t n_want)
{
    uint32_t cpts[256], lens[256];
    size_t n = oc_unicode_cpts_from_utf8(text, strlen(text), cpts);
    size_t nw = oc_pretok_split(OC_PRETOK_K2_HORIZON, cpts, n, lens);
    cr_assert_eq(nw, n_want, "'%s': %zu words, want %zu", text, nw, n_want);
    for (size_t i = 0; i < nw; ++i)
        cr_assert_eq(lens[i], want[i], "'%s': word %zu len %u want %u",
                     text, i, lens[i], want[i]);
}

#define CHECK(text, ...) do {                                    \
        const uint32_t w_[] = { __VA_ARGS__ };                   \
        check_split((text), w_, sizeof(w_) / sizeof(w_[0]));     \
    } while (0)

Test(pretokenize, type_from_name)
{
    cr_assert_eq(oc_pretok_type_from_name("k2-horizon"), OC_PRETOK_K2_HORIZON);
    cr_assert_eq(oc_pretok_type_from_name("qwen2"), OC_PRETOK_NONE);
    cr_assert_eq(oc_pretok_type_from_name(NULL), OC_PRETOK_NONE);
}

Test(pretokenize, none_is_single_word)
{
    uint32_t cpts[3] = { 'a', ' ', '1' }, lens[3];
    cr_assert_eq(oc_pretok_split(OC_PRETOK_NONE, cpts, 3, lens), 1u);
    cr_assert_eq(lens[0], 3u);
    cr_assert_eq(oc_pretok_split(OC_PRETOK_K2_HORIZON, cpts, 0, lens), 0u);
}

Test(pretokenize, words_and_punctuation)
{
    CHECK("Hello, world!", 5, 1, 6, 1);
}

Test(pretokenize, digits_split_in_threes)
{
    CHECK("1234567", 3, 3, 1);
    CHECK("x1y22", 1, 1, 1, 2);
}

Test(pretokenize, contractions_case_insensitive)
{
    CHECK("I'M", 1, 2);
    CHECK("they'll", 4, 3);
}

Test(pretokenize, whitespace_rules)
{
    /* \s+(?!\S) leaves the last space to prefix the next word. */
    CHECK("a  b", 1, 1, 2);
    /* \s*[\r\n]+ then the space run. */
    CHECK("x\n\n  y", 1, 2, 1, 2);
    /* Trailing run at end of text is one \s+ word. */
    CHECK("trailing   ", 8, 3);
}

Test(pretokenize, marks_and_joiners_extend_letters)
{
    CHECK("e\xcc\x81", 2);                  /* e + U+0301 */
    CHECK("zero\xe2\x80\x8bwidth", 4, 6);   /* ZWSP prefixes the next word */
    CHECK("a\xe2\x80\x8d" "b", 3);          /* ZWJ inside a letter run */
}
