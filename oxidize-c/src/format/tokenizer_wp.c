/* tokenizer_wp.c — WordPiece tokenizer with `##` continuation prefix,
 * for BERT model family.
 *
 * Faithful port of oxidize-core/src/format/tokenizer.rs::
 *   - `load_wordpiece`                          (load from GGUF metadata)
 *   - `WordPieceTokenizer::new`                  (test constructor)
 *   - `WordPieceTokenizer::with_unknown_token`
 *   - `WordPieceTokenizer::encode`               (greedy longest match)
 *   - `WordPieceTokenizer::decode`               (strip `##` on continuation)
 *   - `WordPieceTokenizer::encode_word_into`     (per-word greedy match)
 *
 * Algorithm (mirrors Rust `encode`):
 *   1. Iterate the input by Unicode codepoint. Whitespace codepoints flush
 *      the current word and emit the whitespace char's own id (or `<unk>`
 *      if absent). Non-whitespace codepoints accumulate into `current_word`.
 *   2. For each word, run greedy longest-match: starting at `start_idx = 0`,
 *      scan `end_idx` from the longest candidate down to `start_idx + 1`,
 *      prepend `##` to the candidate when `start_idx > 0` (continuation),
 *      and emit the first id found. If no candidate matches, emit `<unk>`
 *      and stop (mirrors Rust's early `return`).
 *
 * Whitespace detection uses the Unicode White_Space property to match
 * Rust's `char::is_whitespace` exactly (covers U+0009..U+000D, U+0020,
 * U+0085, U+00A0, U+1680, U+2000..U+200A, U+2028, U+2029, U+202F, U+205F,
 * U+3000). This matters for round-trip parity on multilingual corpora.
 *
 * `##` continuation (VAL-TOK-008): a token like `##ing` only matches at a
 * non-zero offset within a word. On decode, the `##` prefix is stripped and
 * the remainder is concatenated directly to the previous output (no
 * separator), so `["play", "##ing"]` decodes to `"playing"`.
 */

#define _POSIX_C_SOURCE 200809L  /* strdup */

#include "oxidize/tokenizer.h"

#include "oxidize/arena.h"
#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/hashtable.h"
#include "oxidize/log.h"
#include "oxidize/vector.h"

#include "utf8_utils.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ─── WordPiece tokenizer state ─────────────────────────────────────── */

struct OcWordPieceTokenizer {
    /* vocab: token string → id. Keys are arena-owned NUL-terminated strings. */
    OcHashtable *vocab;
    /* id → token string (dense array indexed by id). Arena-owned. */
    char  **id_to_token;
    size_t  vocab_size;
    /* Special-token ids (also mirrored in the OcTokenizer wrapper). */
    uint32_t  unknown_id;  bool has_unknown;
    uint32_t  bos_id;      bool has_bos;
    uint32_t  eos_id;      bool has_eos;
    uint32_t  pad_id;      bool has_pad;
    uint32_t  separator_id; bool has_separator;
    uint32_t  cls_id;      bool has_cls;
    uint32_t  mask_id;     bool has_mask;
};

/* ─── UTF-8 + Unicode helpers ────────────────────────────────────────── */

/* UTF-8 codepoint decode/encode are shared (utf8_utils.h). */

/* Whether a Unicode codepoint has the White_Space property (mirrors Rust's
 * `char::is_whitespace`). The set is drawn from the Unicode White_Space
 * property and is stable across Unicode revisions. */
static bool wp_is_whitespace(uint32_t cp)
{
    /* ASCII whitespace: U+0009..U+000D, U+0020 */
    if ((cp >= 0x09 && cp <= 0x0D) || cp == 0x20) return true;
    /* Latin-1 / next line */
    if (cp == 0x85 || cp == 0xA0) return true;
    /* Ogham space mark */
    if (cp == 0x1680) return true;
    /* En quad .. hair space (U+2000..U+200A) */
    if (cp >= 0x2000 && cp <= 0x200A) return true;
    /* Line separator, paragraph separator, narrow no-break space,
     * medium mathematical space, ideographic space */
    if (cp == 0x2028 || cp == 0x2029 || cp == 0x202F
        || cp == 0x205F || cp == 0x3000) return true;
    return false;
}

/* Compute UTF-8 char boundaries (byte offsets of each codepoint start,
 * followed by `text_len`). Mirrors Rust `char_boundaries`. `out` must be
 * large enough to hold `text_len + 1` entries. Returns the count written. */
static size_t wp_char_boundaries(const char *text, size_t text_len, size_t *out)
{
    size_t n = 0;
    size_t i = 0;
    while (i < text_len) {
        out[n++] = i;
        uint32_t cp;
        size_t adv = oc_utf8_decode_cp(text + i, text_len - i, &cp);
        if (adv == 0) adv = 1;
        i += adv;
    }
    out[n++] = text_len;
    return n;
}

/* ─── Constructor (mirrors Rust `WordPieceTokenizer::new`) ──────────── */

OcError oc_wp_new(const char *const *vocab_tokens, size_t n_tokens,
                  OcArena *arena, OcWordPieceTokenizer **out)
{
    if (!vocab_tokens || !arena || !out) return OC_ERR_INVALID_ARG;
    *out = NULL;

    OcWordPieceTokenizer *wp = (OcWordPieceTokenizer *)
        oc_arena_alloc(arena, sizeof(*wp), sizeof(void *));
    if (!wp) return OC_ERR_OOM;
    memset(wp, 0, sizeof(*wp));

    wp->vocab = oc_hashtable_new(n_tokens * 2 > 0 ? n_tokens * 2 : 16);
    wp->vocab_size = n_tokens;
    wp->id_to_token = oc_arena_alloc(arena,
                                      n_tokens * sizeof(char *), sizeof(void *));
    if (!wp->vocab || !wp->id_to_token) return OC_ERR_OOM;

    for (size_t id = 0; id < n_tokens; ++id) {
        char *dup = oc_arena_dup(arena, vocab_tokens[id]);
        if (!dup) return OC_ERR_OOM;
        wp->id_to_token[id] = dup;
        oc_hashtable_put(wp->vocab, dup, (void *)(uintptr_t)id, NULL);
    }

    *out = wp;
    return OC_OK;
}

OcError oc_wp_with_unknown_token(OcWordPieceTokenizer *wp, OcArena *arena,
                                 const char *token)
{
    if (!wp || !arena || !token) return OC_ERR_INVALID_ARG;

    void *vp;
    uint32_t id;
    if (oc_hashtable_get(wp->vocab, token, &vp)) {
        id = (uint32_t)(uintptr_t)vp;
    } else {
        id = (uint32_t)wp->vocab_size;
        char *dup = oc_arena_dup(arena, token);
        if (!dup) return OC_ERR_OOM;
        char **new_tokens = oc_arena_alloc(arena,
                                            (wp->vocab_size + 1) * sizeof(char *),
                                            sizeof(void *));
        if (!new_tokens) return OC_ERR_OOM;
        for (size_t i = 0; i < wp->vocab_size; ++i) {
            new_tokens[i] = wp->id_to_token[i];
        }
        new_tokens[id] = dup;
        OcError e = oc_hashtable_put(wp->vocab, dup, (void *)(uintptr_t)id, NULL);
        if (e != OC_OK) return e;
        wp->id_to_token = new_tokens;
        wp->vocab_size = id + 1;
    }
    wp->has_unknown = true;
    wp->unknown_id = id;
    return OC_OK;
}

/* ─── Encode ─────────────────────────────────────────────────────────── */

/* Encode one word via greedy longest-match. Mirrors Rust
 * `WordPieceTokenizer::encode_word_into`. Appends ids to `encoded`. */
static OcError wp_encode_word_into(const OcWordPieceTokenizer *wp,
                                   const char *word, size_t word_len,
                                   OcVector *encoded)
{
    if (word_len == 0) return OC_OK;

    /* Compute char boundaries of the word. */
    size_t *boundaries = (size_t *)malloc((word_len + 1) * sizeof(size_t));
    if (!boundaries) return OC_ERR_OOM;
    size_t n_bounds = wp_char_boundaries(word, word_len, boundaries);
    size_t token_count = (n_bounds == 0) ? 0 : (n_bounds - 1);

    size_t start_idx = 0;
    while (start_idx < token_count) {
        bool found = false;
        uint32_t found_id = 0;
        size_t next_idx = 0;
        /* Scan end_idx from the longest down to start_idx + 1. */
        for (size_t end_idx = token_count; end_idx > start_idx; --end_idx) {
            size_t start = boundaries[start_idx];
            size_t end = boundaries[end_idx];
            size_t len = end - start;
            /* Build the candidate. If start_idx > 0, prepend "##" (the
             * WordPiece continuation marker). */
            char stack_buf[256];
            char *key;
            size_t prefix = (start_idx > 0) ? 2 : 0;  /* "##" */
            size_t total = prefix + len;
            bool use_stack = (total < sizeof(stack_buf));
            if (use_stack) {
                if (prefix) { memcpy(stack_buf, "##", 2); }
                memcpy(stack_buf + prefix, word + start, len);
                stack_buf[total] = '\0';
                key = stack_buf;
            } else {
                key = (char *)malloc(total + 1);
                if (!key) { free(boundaries); return OC_ERR_OOM; }
                if (prefix) { memcpy(key, "##", 2); }
                memcpy(key + prefix, word + start, len);
                key[total] = '\0';
            }
            void *vp;
            bool hit = oc_hashtable_get(wp->vocab, key, &vp);
            if (!use_stack) free(key);
            if (hit) {
                found = true;
                found_id = (uint32_t)(uintptr_t)vp;
                next_idx = end_idx;
                break;
            }
        }
        if (found) {
            OcError e = oc_vector_push(encoded, &found_id);
            if (e != OC_OK) { free(boundaries); return e; }
            start_idx = next_idx;
        } else {
            /* No match: emit <unk> (if configured) and stop. Mirrors Rust's
             * `return;` after pushing unk. */
            if (wp->has_unknown) {
                uint32_t unk = wp->unknown_id;
                OcError e = oc_vector_push(encoded, &unk);
                if (e != OC_OK) { free(boundaries); return e; }
            }
            free(boundaries);
            return OC_OK;
        }
    }
    free(boundaries);
    return OC_OK;
}

OcError oc_wp_encode(const OcWordPieceTokenizer *wp, const char *text,
                     uint32_t **out_ids, size_t *out_count)
{
    if (!wp || !text || !out_ids || !out_count) return OC_ERR_INVALID_ARG;
    *out_ids = NULL;
    *out_count = 0;

    size_t text_len = strlen(text);
    if (text_len == 0) return OC_OK;

    OcVector result;
    OcError e = oc_vector_init(&result, sizeof(uint32_t));
    if (e != OC_OK) return e;

    /* Accumulate the current word's bytes. */
    OcVector current_word;
    e = oc_vector_init(&current_word, sizeof(char));
    if (e != OC_OK) { oc_vector_free(&result); return e; }

    const char *p = text;
    const char *end = text + text_len;
    while (p < end) {
        uint32_t cp;
        size_t adv = oc_utf8_decode_cp(p, (size_t)(end - p), &cp);
        if (adv == 0) adv = 1;

        if (wp_is_whitespace(cp)) {
            /* Flush the current word. */
            size_t word_len = oc_vector_len(&current_word);
            char *word_buf = NULL;
            if (word_len > 0) {
                word_buf = (char *)malloc(word_len + 1);
                if (!word_buf) {
                    oc_vector_free(&current_word); oc_vector_free(&result);
                    return OC_ERR_OOM;
                }
                for (size_t i = 0; i < word_len; ++i) {
                    word_buf[i] = *(const char *)oc_vector_get(&current_word, i);
                }
                word_buf[word_len] = '\0';
                e = wp_encode_word_into(wp, word_buf, word_len, &result);
                free(word_buf);
                if (e != OC_OK) {
                    oc_vector_free(&current_word); oc_vector_free(&result);
                    return e;
                }
            }
            oc_vector_clear(&current_word);

            /* Emit the whitespace char's id (or unk). */
            char ws_buf[5];
            size_t ws_len = oc_utf8_encode_cp(cp, ws_buf);
            ws_buf[ws_len] = '\0';
            void *vp;
            if (oc_hashtable_get(wp->vocab, ws_buf, &vp)) {
                uint32_t id = (uint32_t)(uintptr_t)vp;
                e = oc_vector_push(&result, &id);
                if (e != OC_OK) {
                    oc_vector_free(&current_word); oc_vector_free(&result);
                    return e;
                }
            } else if (wp->has_unknown) {
                uint32_t unk = wp->unknown_id;
                e = oc_vector_push(&result, &unk);
                if (e != OC_OK) {
                    oc_vector_free(&current_word); oc_vector_free(&result);
                    return e;
                }
            }
        } else {
            /* Append the UTF-8 bytes of this codepoint to the current word. */
            for (size_t i = 0; i < adv; ++i) {
                e = oc_vector_push(&current_word, &p[i]);
                if (e != OC_OK) {
                    oc_vector_free(&current_word); oc_vector_free(&result);
                    return e;
                }
            }
        }
        p += adv;
    }

    /* Flush the trailing word. */
    size_t word_len = oc_vector_len(&current_word);
    if (word_len > 0) {
        char *word_buf = (char *)malloc(word_len + 1);
        if (!word_buf) {
            oc_vector_free(&current_word); oc_vector_free(&result);
            return OC_ERR_OOM;
        }
        for (size_t i = 0; i < word_len; ++i) {
            word_buf[i] = *(const char *)oc_vector_get(&current_word, i);
        }
        word_buf[word_len] = '\0';
        e = wp_encode_word_into(wp, word_buf, word_len, &result);
        free(word_buf);
        if (e != OC_OK) {
            oc_vector_free(&current_word); oc_vector_free(&result);
            return e;
        }
    }
    oc_vector_free(&current_word);

    /* Materialize the result as a malloc'd array. */
    size_t count = oc_vector_len(&result);
    uint32_t *ids = (uint32_t *)malloc((count ? count : 1) * sizeof(uint32_t));
    if (!ids) { oc_vector_free(&result); return OC_ERR_OOM; }
    for (size_t i = 0; i < count; ++i) {
        ids[i] = *(const uint32_t *)oc_vector_get(&result, i);
    }
    oc_vector_free(&result);
    *out_ids = ids;
    *out_count = count;
    return OC_OK;
}

/* ─── Decode ─────────────────────────────────────────────────────────── */

OcError oc_wp_decode(const OcWordPieceTokenizer *wp, const uint32_t *ids,
                     size_t count, char **out_text)
{
    if (!wp || !out_text) return OC_ERR_INVALID_ARG;
    *out_text = NULL;
    if (count == 0 || !ids) {
        char *empty = (char *)malloc(1);
        if (!empty) return OC_ERR_OOM;
        empty[0] = '\0';
        *out_text = empty;
        return OC_OK;
    }

    /* First pass: total length. Strip "##" prefix on continuation tokens. */
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) {
        uint32_t id = ids[i];
        if (id >= wp->vocab_size || !wp->id_to_token[id]) {
            return OC_ERR_TOKENIZER;
        }
        const char *piece = wp->id_to_token[id];
        size_t len = strlen(piece);
        /* Rust: `piece.strip_prefix("##").unwrap_or(piece)`. */
        if (len >= 2 && piece[0] == '#' && piece[1] == '#') {
            len -= 2;
        }
        total += len;
    }

    char *out = (char *)malloc(total + 1);
    if (!out) return OC_ERR_OOM;
    size_t off = 0;
    for (size_t i = 0; i < count; ++i) {
        const char *piece = wp->id_to_token[ids[i]];
        size_t len = strlen(piece);
        if (len >= 2 && piece[0] == '#' && piece[1] == '#') {
            piece += 2;
            len -= 2;
        }
        memcpy(out + off, piece, len);
        off += len;
    }
    out[off] = '\0';
    *out_text = out;
    return OC_OK;
}

/* ─── Load from GGUF metadata ───────────────────────────────────────── */

static OcError wp_get_string_array(const OcGgufFile *gguf, const char *key,
                                   OcArena *arena, OcVector *out)
{
    const OcGgufMetadataValue *v = oc_gguf_metadata_get(gguf, key);
    if (!v || v->type != OC_GGUF_MT_ARRAY) {
        return OC_ERR_TOKENIZER;
    }
    OcError e = oc_vector_init(out, sizeof(char *));
    if (e != OC_OK) return e;
    for (size_t i = 0; i < v->v.arr.len; ++i) {
        const OcGgufMetadataValue *elem = &v->v.arr.values[i];
        if (elem->type != OC_GGUF_MT_STRING) {
            oc_vector_free(out);
            return OC_ERR_TOKENIZER;
        }
        char *dup = oc_arena_dup_n(arena, elem->v.str.data, elem->v.str.len);
        if (!dup) { oc_vector_free(out); return OC_ERR_OOM; }
        e = oc_vector_push(out, &dup);
        if (e != OC_OK) { oc_vector_free(out); return e; }
    }
    return OC_OK;
}

OcError oc_wp_load_from_gguf(const OcGgufFile *gguf, OcArena *arena,
                             OcWordPieceTokenizer **out)
{
    if (!gguf || !arena || !out) return OC_ERR_INVALID_ARG;
    *out = NULL;

    OcVector tokens;
    OcError e = wp_get_string_array(gguf, "tokenizer.ggml.tokens", arena, &tokens);
    if (e != OC_OK) return e;

    size_t vocab_size = oc_vector_len(&tokens);
    OcWordPieceTokenizer *wp = (OcWordPieceTokenizer *)
        oc_arena_alloc(arena, sizeof(*wp), sizeof(void *));
    if (!wp) { oc_vector_free(&tokens); return OC_ERR_OOM; }
    memset(wp, 0, sizeof(*wp));

    wp->vocab = oc_hashtable_new(vocab_size * 2 > 0 ? vocab_size * 2 : 16);
    wp->vocab_size = vocab_size;
    wp->id_to_token = oc_arena_alloc(arena, vocab_size * sizeof(char *), sizeof(void *));
    if (!wp->vocab || !wp->id_to_token) {
        oc_vector_free(&tokens);
        oc_wp_free(wp);
        return OC_ERR_OOM;
    }

    for (size_t id = 0; id < vocab_size; ++id) {
        char *tok = *(char *const *)oc_vector_get(&tokens, id);
        wp->id_to_token[id] = tok;
        oc_hashtable_put(wp->vocab, tok, (void *)(uintptr_t)id, NULL);
    }

    oc_vector_free(&tokens);

    /* Read special-token ids from metadata. */
    uint32_t v;
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.unknown_token_id", &v)) {
        wp->has_unknown = true; wp->unknown_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.bos_token_id", &v)) {
        wp->has_bos = true; wp->bos_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.eos_token_id", &v)) {
        wp->has_eos = true; wp->eos_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.padding_token_id", &v)
        || oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.pad_token_id", &v)) {
        wp->has_pad = true; wp->pad_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.separator_token_id", &v)
        || oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.sep_token_id", &v)) {
        wp->has_separator = true; wp->separator_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.cls_token_id", &v)) {
        wp->has_cls = true; wp->cls_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.mask_token_id", &v)) {
        wp->has_mask = true; wp->mask_id = v;
    }

    *out = wp;
    return OC_OK;
}

void oc_wp_fill_special_tokens(const OcWordPieceTokenizer *wp, OcTokenizer *out)
{
    if (!wp || !out) return;
    out->has_unknown = wp->has_unknown;     out->unknown_id = wp->unknown_id;
    out->has_bos = wp->has_bos;            out->bos_id = wp->bos_id;
    out->has_eos = wp->has_eos;            out->eos_id = wp->eos_id;
    out->has_pad = wp->has_pad;             out->pad_id = wp->pad_id;
    out->has_separator = wp->has_separator; out->separator_id = wp->separator_id;
    out->has_cls = wp->has_cls;             out->cls_id = wp->cls_id;
    out->has_mask = wp->has_mask;           out->mask_id = wp->mask_id;
}

void oc_wp_free(OcWordPieceTokenizer *wp)
{
    if (!wp) return;
    oc_hashtable_free(wp->vocab);
    wp->vocab = NULL;
}
