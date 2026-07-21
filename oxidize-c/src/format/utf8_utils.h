/* utf8_utils.h — shared UTF-8 helpers for the tokenizer implementations.
 *
 * Internal header (src/format/ only). All functions are `static inline` so
 * each translation unit gets its own copy with no link-time dependency.
 *
 *   - oc_utf8_decode_cp:  decode one codepoint (permissive; used by the
 *     BPE / SentencePiece / WordPiece tokenizers, which treat invalid lead
 *     bytes as lone single-byte codepoints).
 *   - oc_utf8_encode_cp:  encode one codepoint as UTF-8.
 *   - oc_utf8_lossy:      strict lossy conversion matching Rust
 *     `String::from_utf8_lossy` (WHATWG: one U+FFFD per maximal invalid
 *     subsequence).
 */
#ifndef OXIDIZE_FORMAT_UTF8_UTILS_H
#define OXIDIZE_FORMAT_UTF8_UTILS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Decode 1..4 bytes of UTF-8 starting at `s` (up to `len` bytes available).
 * Writes the codepoint to `*cp` and returns the number of bytes consumed.
 * Returns 0 only when len == 0. Invalid UTF-8 (bad lead byte, truncated or
 * malformed continuation) yields the lead byte as a lone codepoint and
 * consumes 1 byte. */
static inline size_t oc_utf8_decode_cp(const char *s, size_t len, uint32_t *cp)
{
    if (len == 0) { *cp = 0; return 0; }
    unsigned char c0 = (unsigned char)s[0];
    if (c0 < 0x80) { *cp = c0; return 1; }
    if ((c0 & 0xE0) == 0xC0 && len >= 2) {
        unsigned char c1 = (unsigned char)s[1];
        if ((c1 & 0xC0) == 0x80) {
            *cp = ((uint32_t)(c0 & 0x1F) << 6) | (c1 & 0x3F);
            return 2;
        }
    }
    if ((c0 & 0xF0) == 0xE0 && len >= 3) {
        unsigned char c1 = (unsigned char)s[1];
        unsigned char c2 = (unsigned char)s[2];
        if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80) {
            *cp = ((uint32_t)(c0 & 0x0F) << 12)
                | ((uint32_t)(c1 & 0x3F) << 6)
                | (c2 & 0x3F);
            return 3;
        }
    }
    if ((c0 & 0xF8) == 0xF0 && len >= 4) {
        unsigned char c1 = (unsigned char)s[1];
        unsigned char c2 = (unsigned char)s[2];
        unsigned char c3 = (unsigned char)s[3];
        if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80 && (c3 & 0xC0) == 0x80) {
            *cp = ((uint32_t)(c0 & 0x07) << 18)
                | ((uint32_t)(c1 & 0x3F) << 12)
                | ((uint32_t)(c2 & 0x3F) << 6)
                | (c3 & 0x3F);
            return 4;
        }
    }
    *cp = c0;
    return 1;
}

/* Encode a Unicode codepoint as UTF-8 into `buf` (>= 5 bytes). Returns the
 * number of bytes written. Handles all codepoints up to U+10FFFF. */
static inline size_t oc_utf8_encode_cp(uint32_t cp, char *buf)
{
    if (cp <= 0x7F) {
        buf[0] = (char)cp;
        return 1;
    }
    if (cp <= 0x7FF) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp <= 0xFFFF) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    buf[0] = (char)(0xF0 | (cp >> 18));
    buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    buf[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* Lossy UTF-8 validation matching Rust `String::from_utf8_lossy` (WHATWG
 * "Encoding" spec): each maximal invalid subsequence (a bad lead byte, or a
 * lead byte plus its longest valid continuation prefix) is replaced by a
 * single U+FFFD (0xEF 0xBF 0xBD). Overlongs, surrogates, and codepoints
 * above U+10FFFF are rejected at the second byte via lead-specific ranges.
 * `out` must hold at least `len * 3 + 1` bytes (worst case: every byte
 * becomes a 3-byte replacement). Returns the output byte length (no NUL is
 * written). */
static inline size_t oc_utf8_lossy(const uint8_t *bytes, size_t len, uint8_t *out)
{
    static const uint8_t REPL[3] = { 0xEF, 0xBF, 0xBD };
    size_t i = 0;
    size_t j = 0;
    while (i < len) {
        uint8_t c0 = bytes[i];
        if (c0 < 0x80) {
            out[j++] = c0;
            i += 1;
            continue;
        }
        size_t need;
        uint8_t lo = 0x80, hi = 0xBF;  /* valid range for the 2nd byte */
        if (c0 >= 0xC2 && c0 <= 0xDF) {
            need = 2;
        } else if (c0 >= 0xE0 && c0 <= 0xEF) {
            need = 3;
            if (c0 == 0xE0) lo = 0xA0;        /* reject overlong */
            else if (c0 == 0xED) hi = 0x9F;   /* reject surrogates */
        } else if (c0 >= 0xF0 && c0 <= 0xF4) {
            need = 4;
            if (c0 == 0xF0) lo = 0x90;        /* reject overlong */
            else if (c0 == 0xF4) hi = 0x8F;   /* reject > U+10FFFF */
        } else {
            /* 0x80..0xC1 or 0xF5..0xFF: invalid lead byte. */
            out[j++] = REPL[0]; out[j++] = REPL[1]; out[j++] = REPL[2];
            i += 1;
            continue;
        }
        /* Consume the longest valid continuation prefix. */
        size_t k = 1;
        bool valid = true;
        for (; k < need; ++k) {
            if (i + k >= len) { valid = false; break; }
            uint8_t c = bytes[i + k];
            uint8_t l = (k == 1) ? lo : 0x80;
            uint8_t h = (k == 1) ? hi : 0xBF;
            if (c < l || c > h) { valid = false; break; }
        }
        if (valid) {
            memcpy(out + j, bytes + i, need);
            j += need;
            i += need;
        } else {
            /* One replacement for the whole invalid subsequence (lead byte
             * plus the k-1 valid continuation bytes consumed so far). */
            out[j++] = REPL[0]; out[j++] = REPL[1]; out[j++] = REPL[2];
            i += k;
        }
    }
    return j;
}

#endif /* OXIDIZE_FORMAT_UTF8_UTILS_H */
