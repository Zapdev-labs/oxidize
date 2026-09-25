/* unicode.c — code point classification + llama.cpp-compatible UTF-8 decode.
 *
 * Lookups are a binary search over the generated range table
 * (src/util/unicode_data.c), with an ASCII fast path because the vast
 * majority of prompt bytes are ASCII. */

#include "oxidize/unicode.h"

#include <stddef.h>
#include <stdint.h>

/* Flags (incl. whitespace) for code points 0..127. Equal to the table
 * lookup; tests/test_unicode.c checks that. */
static const uint16_t ascii_flags[128] = {
    /* 0x00-0x08 control */
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    /* \t \n \v \f \r : control + whitespace */
    0x180, 0x180, 0x180, 0x180, 0x180,
    /* 0x0E-0x1F control */
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    /* ' ' separator + whitespace */
    0x108,
    /* ! " # */ 0x20, 0x20, 0x20,
    /* $ */ 0x40,
    /* % & ' ( ) * */ 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    /* + */ 0x40,
    /* , - . / */ 0x20, 0x20, 0x20, 0x20,
    /* 0-9 */ 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    /* : ; */ 0x20, 0x20,
    /* < = > */ 0x40, 0x40, 0x40,
    /* ? @ */ 0x20, 0x20,
    /* A-Z */
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    /* [ \ ] */ 0x20, 0x20, 0x20,
    /* ^ */ 0x40,
    /* _ */ 0x20,
    /* ` */ 0x40,
    /* a-z */
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    /* { */ 0x20,
    /* | */ 0x40,
    /* } */ 0x20,
    /* ~ */ 0x40,
    /* DEL */ 0x80,
};

static bool is_whitespace(uint32_t cp)
{
    size_t lo = 0, hi = oc_unicode_n_whitespace;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (oc_unicode_whitespace[mid] < cp) lo = mid + 1;
        else hi = mid;
    }
    return lo < oc_unicode_n_whitespace && oc_unicode_whitespace[lo] == cp;
}

uint16_t oc_unicode_cpt_flags(uint32_t cp)
{
    if (cp < 128) return ascii_flags[cp];
    if (cp >= 0x110000u) return OC_UCPT_UNDEFINED;
    /* Last range whose start <= cp (the table starts at 0 and ends with the
     * 0x110000 sentinel, so the index is always valid). */
    size_t lo = 0, hi = oc_unicode_n_ranges;
    while (hi - lo > 1) {
        size_t mid = lo + (hi - lo) / 2;
        if (oc_unicode_range_start[mid] <= cp) lo = mid;
        else hi = mid;
    }
    uint16_t f = oc_unicode_range_flags[lo];
    if (is_whitespace(cp)) f |= OC_UCPT_WHITESPACE;
    return f;
}

uint32_t oc_unicode_tolower(uint32_t cp)
{
    if (cp < 128) return (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp;
    size_t lo = 0, hi = oc_unicode_n_lower;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (oc_unicode_lower_from[mid] < cp) lo = mid + 1;
        else hi = mid;
    }
    if (lo < oc_unicode_n_lower && oc_unicode_lower_from[lo] == cp)
        return oc_unicode_lower_to[lo];
    return cp;
}

size_t oc_unicode_cpts_from_utf8(const char *s, size_t len, uint32_t *out)
{
    const uint8_t *u = (const uint8_t *)s;
    size_t i = 0, n = 0;
    while (i < len) {
        uint8_t b = u[i];
        if (!(b & 0x80)) { out[n++] = b; i += 1; continue; }
        if (!(b & 0x40)) goto invalid;
        if (!(b & 0x20)) {
            if (i + 1 >= len || (u[i + 1] & 0xC0) != 0x80) goto invalid;
            out[n++] = ((uint32_t)(b & 0x1F) << 6) | (u[i + 1] & 0x3F);
            i += 2;
            continue;
        }
        if (!(b & 0x10)) {
            if (i + 2 >= len || (u[i + 1] & 0xC0) != 0x80
                || (u[i + 2] & 0xC0) != 0x80) goto invalid;
            out[n++] = ((uint32_t)(b & 0x0F) << 12)
                     | ((uint32_t)(u[i + 1] & 0x3F) << 6) | (u[i + 2] & 0x3F);
            i += 3;
            continue;
        }
        if (!(b & 0x08)) {
            if (i + 3 >= len || (u[i + 1] & 0xC0) != 0x80
                || (u[i + 2] & 0xC0) != 0x80 || (u[i + 3] & 0xC0) != 0x80)
                goto invalid;
            out[n++] = ((uint32_t)(b & 0x07) << 18)
                     | ((uint32_t)(u[i + 1] & 0x3F) << 12)
                     | ((uint32_t)(u[i + 2] & 0x3F) << 6) | (u[i + 3] & 0x3F);
            i += 4;
            continue;
        }
invalid:
        out[n++] = OC_UCPT_REPLACEMENT;
        i += 1;
    }
    return n;
}

size_t oc_unicode_cpt_to_utf8(uint32_t cp, char out[4])
{
    if (cp <= 0x7F) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp <= 0x7FF) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp <= 0xFFFF) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    if (cp <= 0x10FFFF) {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
    return oc_unicode_cpt_to_utf8(OC_UCPT_REPLACEMENT, out);
}
