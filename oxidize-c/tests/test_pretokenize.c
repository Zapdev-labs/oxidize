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
    uint32_t cpts[256];
    size_t lens[256];
    size_t n = oc_unicode_cpts_from_utf8(text, strlen(text), cpts);
    size_t nw = oc_pretok_split(OC_PRETOK_K2_HORIZON, cpts, n, lens);
    cr_assert_eq(nw, n_want, "'%s': %zu words, want %zu", text, nw, n_want);
    for (size_t i = 0; i < nw; ++i)
        cr_assert_eq(lens[i], (size_t)want[i], "'%s': word %zu len %zu want %u",
                     text, i, lens[i], want[i]);
}

#define CHECK(text, ...) do {                                    \
        const uint32_t w_[] = { __VA_ARGS__ };                   \
        check_split((text), w_, sizeof(w_) / sizeof(w_[0]));     \
    } while (0)

Test(pretokenize, type_from_name)
{
    cr_assert_eq(oc_pretok_type_from_name("k2-horizon"), OC_PRETOK_K2_HORIZON);
    cr_assert_eq(oc_pretok_type_from_name("qwen2"), OC_PRETOK_QWEN2);
    cr_assert_eq(oc_pretok_type_from_name("gpt2"), OC_PRETOK_GPT2);
    cr_assert_eq(oc_pretok_type_from_name("llama-bpe"), OC_PRETOK_LLAMA3);
    cr_assert_eq(oc_pretok_type_from_name("llama3"), OC_PRETOK_LLAMA3);
    cr_assert_eq(oc_pretok_type_from_name("unknown"), OC_PRETOK_NONE);
    cr_assert_eq(oc_pretok_type_from_name(NULL), OC_PRETOK_NONE);
}

Test(pretokenize, none_is_single_word)
{
    uint32_t cpts[3] = { 'a', ' ', '1' };
    size_t lens[3];
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

/* Goldens independently generated with the official regexes (Python regex).
 * Explicit code points also cover embedded NUL, which a C string loses. */
static void check_standard(OcPretokType type, const uint32_t *text, size_t n,
                           const size_t *want, size_t nw)
{
    size_t got[256];
    cr_assert_eq(oc_pretok_split(type, text, n, got), nw);
    size_t total = 0;
    for (size_t i = 0; i < nw; i++) {
        cr_assert_eq(got[i], want[i], "type %d word %zu", type, i);
        total += got[i];
    }
    cr_assert_eq(total, n);
}

Test(pretokenize, standard_2_0)
{
    const uint32_t text[] = { 0x49, 0x27, 0x4d, 0x20, 0x74, 0x68, 0x65, 0x79, 0x27, 0x6c, 0x6c, 0x20, 0x49, 0x27, 0x53, 0x20, 0x27, 0x17f, 0x20, 0x27, 0x52, 0x45 };
    const size_t want[] = { 1, 1, 1, 5, 3, 2, 1, 1, 2, 1, 2, 2 };
    check_standard((OcPretokType)2, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_2_1)
{
    const uint32_t text[] = { 0x20, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37 };
    const size_t want[] = { 8 };
    check_standard((OcPretokType)2, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_2_2)
{
    const uint32_t text[] = { 0x65, 0x301, 0x20, 0x61, 0x200d, 0x62, 0x20, 0x301, 0x301, 0x61 };
    const size_t want[] = { 1, 1, 2, 1, 1, 3, 1 };
    check_standard((OcPretokType)2, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_2_3)
{
    const uint32_t text[] = { 0x78, 0xd, 0xa, 0x20, 0x20, 0x79, 0x21, 0xa, 0xa, 0x20, 0x20 };
    const size_t want[] = { 1, 3, 2, 1, 4 };
    check_standard((OcPretokType)2, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_2_4)
{
    const uint32_t text[] = { 0x61, 0x9, 0xa0, 0x62, 0x2028, 0x7a };
    const size_t want[] = { 1, 1, 1, 1, 1, 1 };
    check_standard((OcPretokType)2, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_2_5)
{
    const uint32_t text[] = { 0x0, 0x1c, 0x200b, 0x1f600, 0x4e2d, 0x6587, 0x661, 0xb2, 0x2167 };
    const size_t want[] = { 4, 2, 3 };
    check_standard((OcPretokType)2, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_3_0)
{
    const uint32_t text[] = { 0x49, 0x27, 0x4d, 0x20, 0x74, 0x68, 0x65, 0x79, 0x27, 0x6c, 0x6c, 0x20, 0x49, 0x27, 0x53, 0x20, 0x27, 0x17f, 0x20, 0x27, 0x52, 0x45 };
    const size_t want[] = { 1, 2, 5, 3, 2, 2, 2, 1, 2, 2 };
    check_standard((OcPretokType)3, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_3_1)
{
    const uint32_t text[] = { 0x20, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37 };
    const size_t want[] = { 1, 3, 3, 1 };
    check_standard((OcPretokType)3, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_3_2)
{
    const uint32_t text[] = { 0x65, 0x301, 0x20, 0x61, 0x200d, 0x62, 0x20, 0x301, 0x301, 0x61 };
    const size_t want[] = { 1, 1, 2, 2, 3, 1 };
    check_standard((OcPretokType)3, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_3_3)
{
    const uint32_t text[] = { 0x78, 0xd, 0xa, 0x20, 0x20, 0x79, 0x21, 0xa, 0xa, 0x20, 0x20 };
    const size_t want[] = { 1, 2, 1, 2, 3, 2 };
    check_standard((OcPretokType)3, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_3_4)
{
    const uint32_t text[] = { 0x61, 0x9, 0xa0, 0x62, 0x2028, 0x7a };
    const size_t want[] = { 1, 1, 2, 2 };
    check_standard((OcPretokType)3, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_3_5)
{
    const uint32_t text[] = { 0x0, 0x1c, 0x200b, 0x1f600, 0x4e2d, 0x6587, 0x661, 0xb2, 0x2167 };
    const size_t want[] = { 4, 2, 3 };
    check_standard((OcPretokType)3, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_4_0)
{
    const uint32_t text[] = { 0x49, 0x27, 0x4d, 0x20, 0x74, 0x68, 0x65, 0x79, 0x27, 0x6c, 0x6c, 0x20, 0x49, 0x27, 0x53, 0x20, 0x27, 0x17f, 0x20, 0x27, 0x52, 0x45 };
    const size_t want[] = { 1, 2, 5, 3, 2, 2, 2, 1, 2, 2 };
    check_standard((OcPretokType)4, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_4_1)
{
    const uint32_t text[] = { 0x20, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37 };
    const size_t want[] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    check_standard((OcPretokType)4, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_4_2)
{
    const uint32_t text[] = { 0x65, 0x301, 0x20, 0x61, 0x200d, 0x62, 0x20, 0x301, 0x301, 0x61 };
    const size_t want[] = { 1, 1, 2, 2, 3, 1 };
    check_standard((OcPretokType)4, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_4_3)
{
    const uint32_t text[] = { 0x78, 0xd, 0xa, 0x20, 0x20, 0x79, 0x21, 0xa, 0xa, 0x20, 0x20 };
    const size_t want[] = { 1, 2, 1, 2, 3, 2 };
    check_standard((OcPretokType)4, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_4_4)
{
    const uint32_t text[] = { 0x61, 0x9, 0xa0, 0x62, 0x2028, 0x7a };
    const size_t want[] = { 1, 1, 2, 2 };
    check_standard((OcPretokType)4, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}

Test(pretokenize, standard_4_5)
{
    const uint32_t text[] = { 0x0, 0x1c, 0x200b, 0x1f600, 0x4e2d, 0x6587, 0x661, 0xb2, 0x2167 };
    const size_t want[] = { 4, 2, 1, 1, 1 };
    check_standard((OcPretokType)4, text, sizeof(text) / sizeof(text[0]),
                   want, sizeof(want) / sizeof(want[0]));
}
