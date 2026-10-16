/* pretokenize.c — hand-written equivalents of BPE pre-tokenizer regexes.
 *
 * OC_PRETOK_K2_HORIZON is a line-by-line port of llama.cpp-k2
 * `unicode_regex_split_custom_k2_horizon` (src/unicode.cpp), including its
 * quirks (e.g. `\s+(?!\S)` backing off one code point only when the run is
 * not at the end of the text). Keep it in lockstep with that function: any
 * divergence changes token ids. */

#include "oxidize/pretokenize.h"

#include "oxidize/unicode.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define OUT_OF_RANGE 0xFFFFFFFFu

OcPretokType oc_pretok_type_from_name(const char *pre)
{
    if (pre && strcmp(pre, "k2-horizon") == 0) return OC_PRETOK_K2_HORIZON;
    if (pre && strcmp(pre, "gpt2") == 0) return OC_PRETOK_GPT2;
    if (pre && (strcmp(pre, "llama-bpe") == 0 || strcmp(pre, "llama3") == 0))
        return OC_PRETOK_LLAMA3;
    if (pre && strcmp(pre, "qwen2") == 0) return OC_PRETOK_QWEN2;
    if (pre && strcmp(pre, "qwen_m") == 0) return OC_PRETOK_QWEN2_M;
    return OC_PRETOK_NONE;
}

typedef struct {
    const uint32_t *cpts;
    size_t          n;
    size_t         *lens;
    size_t          n_words;
    size_t          prev_end;
} Splitter;

static inline uint32_t get_cpt(const Splitter *s, size_t pos)
{
    return pos < s->n ? s->cpts[pos] : OUT_OF_RANGE;
}

/* Flags at `pos`, or 0 outside the text (llama's empty unicode_cpt_flags). */
static inline uint16_t get_flags(const Splitter *s, size_t pos)
{
    return pos < s->n ? oc_unicode_cpt_flags(s->cpts[pos]) : 0;
}

static inline size_t add_token(Splitter *s, size_t end)
{
    size_t len = end - s->prev_end;
    if (len > 0) s->lens[s->n_words++] = len;
    s->prev_end = end;
    return len;
}

static inline bool is_k2_letter(const Splitter *s, size_t pos)
{
    uint32_t c = get_cpt(s, pos);
    if (c == 0x200Cu || c == 0x200Du) return true;
    return (get_flags(s, pos) & (OC_UCPT_LETTER | OC_UCPT_ACCENT_MARK)) != 0;
}

static size_t split_k2_horizon(Splitter *s)
{
    const size_t end = s->n;
    const uint16_t WLN = OC_UCPT_WHITESPACE | OC_UCPT_LETTER | OC_UCPT_NUMBER;

    for (size_t pos = 0; pos < end;) {
        const uint32_t cpt = get_cpt(s, pos);
        const uint16_t flags = get_flags(s, pos);

        /* (?i:'s|'t|'re|'ve|'m|'ll|'d) */
        if (cpt == '\'' && pos + 1 < end) {
            uint32_t next = oc_unicode_tolower(get_cpt(s, pos + 1));
            if (next == 0x017Fu) next = 's';  /* long s case-folds to s */
            if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                pos += add_token(s, pos + 2);
                continue;
            }
            if (pos + 2 < end) {
                uint32_t nn = oc_unicode_tolower(get_cpt(s, pos + 2));
                if ((next == 'r' && nn == 'e') || (next == 'v' && nn == 'e')
                    || (next == 'l' && nn == 'l')) {
                    pos += add_token(s, pos + 3);
                    continue;
                }
            }
        }

        /* [^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|‌|‍)+ */
        if (!(cpt == '\r' || cpt == '\n' || (flags & OC_UCPT_NUMBER))) {
            if (is_k2_letter(s, pos) || is_k2_letter(s, pos + 1)) {
                pos++;
                while (is_k2_letter(s, pos)) pos++;
                add_token(s, pos);
                continue;
            }
        }

        /* \p{N}{1,3} */
        if (flags & OC_UCPT_NUMBER) {
            size_t ini = pos;
            while (get_flags(s, pos) & OC_UCPT_NUMBER) {
                if (++pos - ini >= 3) {
                    add_token(s, pos);
                    ini = pos;
                }
            }
            add_token(s, pos);
            continue;
        }

        /* <space>?[^\s\p{L}\p{N}]+[\r\n]* */
        uint16_t flags2 = (cpt == ' ') ? get_flags(s, pos + 1) : flags;
        if (!(flags2 & WLN) && flags) {
            pos += (cpt == ' ');
            while (!(flags2 & WLN) && flags2) flags2 = get_flags(s, ++pos);
            uint32_t cpt2 = get_cpt(s, pos);
            while (cpt2 == '\r' || cpt2 == '\n') cpt2 = get_cpt(s, ++pos);
            add_token(s, pos);
            continue;
        }

        size_t num_ws = 0;
        size_t last_end_r_or_n = 0;
        while (get_flags(s, pos + num_ws) & OC_UCPT_WHITESPACE) {
            uint32_t cpt2 = get_cpt(s, pos + num_ws);
            if (cpt2 == '\r' || cpt2 == '\n') last_end_r_or_n = pos + num_ws + 1;
            num_ws++;
        }

        /* \s*[\r\n]+ */
        if (last_end_r_or_n > 0) {
            pos = last_end_r_or_n;
            add_token(s, pos);
            continue;
        }

        /* \s+(?!\S) */
        if (num_ws > 1 && get_cpt(s, pos + num_ws) != OUT_OF_RANGE) {
            pos += num_ws - 1;
            add_token(s, pos);
            continue;
        }

        /* \s+ */
        if (num_ws > 0) {
            pos += num_ws;
            add_token(s, pos);
            continue;
        }

        /* no match */
        add_token(s, ++pos);
    }
    return s->n_words;
}

/* Standard patterns from openai/gpt-2 src/encoder.py, Meta llama3
 * llama/tokenizer.py, and transformers models/qwen2/tokenization_qwen2.py.
 * Unlike K2, marks and joiners are not letters in these patterns. */
static size_t split_standard(Splitter *s, OcPretokType type)
{
    const bool gpt2 = type == OC_PRETOK_GPT2;
    /* The qwen_m pattern keeps combining marks inside letter runs and
     * excludes them from the punctuation run. */
    const uint16_t LM = type == OC_PRETOK_QWEN2_M
                        ? OC_UCPT_LETTER | OC_UCPT_ACCENT_MARK
                        : OC_UCPT_LETTER;
    const uint16_t LN = LM | OC_UCPT_NUMBER;
    const uint16_t WLN = OC_UCPT_WHITESPACE | LN;
    for (size_t pos = 0; pos < s->n;) {
        uint32_t c = get_cpt(s, pos);
        uint16_t flags = get_flags(s, pos);
        if (c == '\'' && pos + 1 < s->n) {
            uint32_t a = get_cpt(s, pos + 1), b = get_cpt(s, pos + 2);
            if (!gpt2) {
                a = oc_unicode_tolower(a);
                b = oc_unicode_tolower(b);
                if (a == 0x017Fu) a = 's';
            }
            size_t count = 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') count = 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
                     (a == 'l' && b == 'l')) count = 3;
            if (count) {
                pos += count;
                add_token(s, pos);
                continue;
            }
        }

        size_t start = pos;
        bool prefix = gpt2 ? c == ' ' :
                      (c != '\r' && c != '\n' && !(flags & LN));
        if (prefix && (get_flags(s, pos + 1) & LM)) start++;
        if (get_flags(s, start) & LM) {
            pos = start + 1;
            while (get_flags(s, pos) & LM) pos++;
            add_token(s, pos);
            continue;
        }

        start = pos + (gpt2 && c == ' ');
        if (get_flags(s, start) & OC_UCPT_NUMBER) {
            size_t limit = gpt2 ? s->n - start : (type == OC_PRETOK_QWEN2 ? 1 : 3);
            pos = start;
            while (pos - start < limit && (get_flags(s, pos) & OC_UCPT_NUMBER)) pos++;
            add_token(s, pos);
            continue;
        }

        start = pos + (c == ' ');
        if (start < s->n && !(get_flags(s, start) & WLN)) {
            pos = start + 1;
            while (pos < s->n && !(get_flags(s, pos) & WLN)) pos++;
            if (!gpt2)
                while (get_cpt(s, pos) == '\r' || get_cpt(s, pos) == '\n') pos++;
            add_token(s, pos);
            continue;
        }

        size_t end = pos, newline_end = pos;
        while (get_flags(s, end) & OC_UCPT_WHITESPACE) {
            if (get_cpt(s, end) == '\r' || get_cpt(s, end) == '\n') newline_end = end + 1;
            end++;
        }
        if (!gpt2 && newline_end > pos) pos = newline_end;
        else if (end - pos > 1 && end < s->n) pos = end - 1;
        else if (end > pos) pos = end;
        else pos++; /* Defensive progress for an unclassified code point. */
        add_token(s, pos);
    }
    return s->n_words;
}

size_t oc_pretok_split(OcPretokType type, const uint32_t *cpts, size_t n,
                       size_t *out_lens)
{
    if (!cpts || !out_lens || n == 0) return 0;
    Splitter s = { cpts, n, out_lens, 0, 0 };
    switch (type) {
    case OC_PRETOK_K2_HORIZON:
        return split_k2_horizon(&s);
    case OC_PRETOK_GPT2:
    case OC_PRETOK_LLAMA3:
    case OC_PRETOK_QWEN2:
    case OC_PRETOK_QWEN2_M:
        return split_standard(&s, type);
    case OC_PRETOK_NONE:
    default:
        out_lens[0] = n;
        return 1;
    }
}
