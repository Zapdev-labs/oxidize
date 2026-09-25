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
    return OC_PRETOK_NONE;
}

typedef struct {
    const uint32_t *cpts;
    size_t          n;
    uint32_t       *lens;
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
    if (len > 0) s->lens[s->n_words++] = (uint32_t)len;
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

size_t oc_pretok_split(OcPretokType type, const uint32_t *cpts, size_t n,
                       uint32_t *out_lens)
{
    if (!cpts || !out_lens || n == 0) return 0;
    Splitter s = { cpts, n, out_lens, 0, 0 };
    switch (type) {
    case OC_PRETOK_K2_HORIZON:
        return split_k2_horizon(&s);
    case OC_PRETOK_NONE:
    default:
        out_lens[0] = (uint32_t)n;
        return 1;
    }
}
