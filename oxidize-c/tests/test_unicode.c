/* test_unicode.c — Criterion tests for src/util/unicode.c (+ generated
 * src/util/unicode_data.c). The tables must classify code points exactly
 * like llama.cpp's unicode-data.cpp snapshot. */

#include <criterion/criterion.h>

#include "oxidize/unicode.h"

#include <stdint.h>
#include <string.h>

/* Range-table lookup without the ASCII fast path. */
static uint16_t table_flags(uint32_t cp)
{
    size_t lo = 0, hi = oc_unicode_n_ranges;
    while (hi - lo > 1) {
        size_t mid = lo + (hi - lo) / 2;
        if (oc_unicode_range_start[mid] <= cp) lo = mid;
        else hi = mid;
    }
    uint16_t f = oc_unicode_range_flags[lo];
    for (size_t i = 0; i < oc_unicode_n_whitespace; ++i) {
        if (oc_unicode_whitespace[i] == cp) f |= OC_UCPT_WHITESPACE;
    }
    return f;
}

Test(unicode, ascii_fast_path_matches_table)
{
    for (uint32_t cp = 0; cp < 128; ++cp) {
        cr_assert_eq(oc_unicode_cpt_flags(cp), table_flags(cp),
                     "cp %#x: fast %#x table %#x", cp,
                     oc_unicode_cpt_flags(cp), table_flags(cp));
    }
}

Test(unicode, tables_are_sorted_and_terminated)
{
    cr_assert_eq(oc_unicode_range_start[0], 0u);
    cr_assert_eq(oc_unicode_range_start[oc_unicode_n_ranges - 1], 0x110000u);
    for (size_t i = 1; i < oc_unicode_n_ranges; ++i)
        cr_assert_lt(oc_unicode_range_start[i - 1], oc_unicode_range_start[i]);
    for (size_t i = 1; i < oc_unicode_n_whitespace; ++i)
        cr_assert_lt(oc_unicode_whitespace[i - 1], oc_unicode_whitespace[i]);
    for (size_t i = 1; i < oc_unicode_n_lower; ++i)
        cr_assert_lt(oc_unicode_lower_from[i - 1], oc_unicode_lower_from[i]);
}

Test(unicode, categories)
{
    cr_assert(oc_unicode_cpt_flags('A') & OC_UCPT_LETTER);
    cr_assert(oc_unicode_cpt_flags('7') & OC_UCPT_NUMBER);
    cr_assert(oc_unicode_cpt_flags(0x00E9) & OC_UCPT_LETTER);       /* é */
    cr_assert(oc_unicode_cpt_flags(0x4E2D) & OC_UCPT_LETTER);       /* 中 */
    cr_assert(oc_unicode_cpt_flags(0x0301) & OC_UCPT_ACCENT_MARK);  /* ◌́ */
    cr_assert(oc_unicode_cpt_flags(0x094D) & OC_UCPT_ACCENT_MARK);  /* virama */
    cr_assert(oc_unicode_cpt_flags(0x0663) & OC_UCPT_NUMBER);       /* ٣ */
    cr_assert(oc_unicode_cpt_flags(0x2460) & OC_UCPT_NUMBER);       /* ① */
    cr_assert(oc_unicode_cpt_flags(0x3002) & OC_UCPT_PUNCTUATION);  /* 。 */
    cr_assert(oc_unicode_cpt_flags(0x20AC) & OC_UCPT_SYMBOL);       /* € */
    cr_assert(oc_unicode_cpt_flags(0x1F600) & OC_UCPT_SYMBOL);      /* 😀 */
    cr_assert(oc_unicode_cpt_flags(0x00A0) & OC_UCPT_WHITESPACE);
    cr_assert(oc_unicode_cpt_flags(0x3000) & OC_UCPT_WHITESPACE);
    cr_assert(oc_unicode_cpt_flags('\n') & OC_UCPT_WHITESPACE);
    cr_assert_not(oc_unicode_cpt_flags(0x200B) & OC_UCPT_WHITESPACE);
    cr_assert_eq(oc_unicode_cpt_flags(0x110000), OC_UCPT_UNDEFINED);
}

Test(unicode, tolower)
{
    cr_assert_eq(oc_unicode_tolower('A'), (uint32_t)'a');
    cr_assert_eq(oc_unicode_tolower('a'), (uint32_t)'a');
    cr_assert_eq(oc_unicode_tolower(0x0410), 0x0430u);  /* А → а */
    cr_assert_eq(oc_unicode_tolower(0x00C5), 0x00E5u);  /* Å → å */
    cr_assert_eq(oc_unicode_tolower(0x4E2D), 0x4E2Du);
}

Test(unicode, utf8_decode_matches_llama_rules)
{
    uint32_t out[16];
    /* "a", invalid 0xFF, "é", truncated 3-byte lead, "(" */
    const char s[] = "a\xff\xc3\xa9\xe2\x82(";
    size_t n = oc_unicode_cpts_from_utf8(s, strlen(s), out);
    uint32_t want[] = { 'a', 0xFFFD, 0xE9, 0xFFFD, 0xFFFD, '(' };
    cr_assert_eq(n, sizeof(want) / sizeof(want[0]));
    for (size_t i = 0; i < n; ++i) cr_assert_eq(out[i], want[i], "cpt %zu", i);

    const char emoji[] = "\xf0\x9f\x98\x80";
    n = oc_unicode_cpts_from_utf8(emoji, 4, out);
    cr_assert_eq(n, 1u);
    cr_assert_eq(out[0], 0x1F600u);
}

Test(unicode, utf8_encode_round_trip)
{
    const uint32_t cps[] = { 0x24, 0xE9, 0x20AC, 0x1F600, 0x10FFFF };
    for (size_t i = 0; i < sizeof(cps) / sizeof(cps[0]); ++i) {
        char buf[4];
        uint32_t back[4];
        size_t nb = oc_unicode_cpt_to_utf8(cps[i], buf);
        size_t n = oc_unicode_cpts_from_utf8(buf, nb, back);
        cr_assert_eq(n, 1u);
        cr_assert_eq(back[0], cps[i]);
    }
    char buf[4];
    cr_assert_eq(oc_unicode_cpt_to_utf8(0x110000, buf), 3u);  /* U+FFFD */
}
