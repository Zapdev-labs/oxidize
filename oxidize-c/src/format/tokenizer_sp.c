/* tokenizer_sp.c — SentencePiece unigram tokenizer with Viterbi best-path
 * segmentation, for Llama / Gemma / Gemma4 model families.
 *
 * Faithful port of oxidize-core/src/format/tokenizer.rs::
 *   - `load_sentencepiece`        (load from GGUF metadata)
 *   - `SentencePieceUnigramTokenizer::new`          (test constructor)
 *   - `SentencePieceUnigramTokenizer::with_unknown_token`
 *   - `SentencePieceUnigramTokenizer::encode`      (Viterbi decode)
 *   - `SentencePieceUnigramTokenizer::decode`      (piece concatenation)
 *
 * Algorithm (mirrors Rust `encode` + `best_segmentation`):
 *   1. Compute UTF-8 char boundaries of the input.
 *   2. Starting at `boundary_idx = 0`, find the best-scoring segmentation of
 *      the suffix `text[boundaries[boundary_idx]..]` via a forward Viterbi
 *      pass: `best_scores[j] = max over i<j of (best_scores[i] +
 *      piece_score(text[i..j]))` for every piece present in the vocab.
 *      Backtracking from the largest reachable `end` reconstructs the token
 *      id sequence. If no segmentation reaches beyond `boundary_idx`, emit
 *      the `<unk>` id and advance one char.
 *   3. Repeat until the whole input is consumed.
 *
 * BOS handling: SentencePiece models add BOS by default (mirrors Rust
 * `add_bos_default()` returning `true` for SentencePiece). The BOS id is
 * prepended by `oc_tokenizer_encode(..., OC_TOK_ADD_BOS)` at the wrapper
 * layer, not here.
 *
 * `<unk>` fallback (VAL-TOK-006): when no piece starting at the current
 * position can be segmented, the configured unknown id is emitted for the
 * current codepoint and the search resumes on the next codepoint. This
 * matches Rust exactly (the `if let Some(unk) = ...` branch).
 */

#define _POSIX_C_SOURCE 200809L  /* strdup */

#include "oxidize/tokenizer.h"

#include "oxidize/arena.h"
#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/hashtable.h"
#include "oxidize/log.h"
#include "oxidize/vector.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ─── SentencePiece tokenizer state ─────────────────────────────────── */

struct OcSentencePieceTokenizer {
    /* vocab: token string → id. Keys are arena-owned NUL-terminated strings. */
    OcHashtable *vocab;
    /* id → token string (dense array indexed by id). Arena-owned. */
    char  **id_to_token;
    size_t  vocab_size;
    /* id → piece score (log-probability). Dense array indexed by id. */
    float  *piece_scores;
    /* Special-token ids (also mirrored in the OcTokenizer wrapper). */
    uint32_t  unknown_id;  bool has_unknown;
    uint32_t  bos_id;      bool has_bos;
    uint32_t  eos_id;      bool has_eos;
    uint32_t  pad_id;      bool has_pad;
    uint32_t  separator_id; bool has_separator;
    uint32_t  cls_id;      bool has_cls;
    uint32_t  mask_id;     bool has_mask;
};

/* ─── UTF-8 helpers ──────────────────────────────────────────────────── */

/* Decode 1..4 bytes of UTF-8 starting at `s` (up to `len` bytes available).
 * Writes the codepoint to `*cp` and returns the number of bytes consumed.
 * Returns 1 for invalid UTF-8 (treats the lead byte as a lone codepoint). */
static size_t sp_utf8_decode(const char *s, size_t len, uint32_t *cp)
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

/* Compute UTF-8 char boundaries (byte offsets of each codepoint start,
 * followed by `text_len`). Mirrors Rust `char_boundaries`. `out` must be
 * large enough to hold `strlen(text) + 1` entries (worst case: every byte
 * is a single-char boundary). Returns the count of boundaries written. */
static size_t sp_char_boundaries(const char *text, size_t text_len, size_t *out)
{
    size_t n = 0;
    size_t i = 0;
    while (i < text_len) {
        out[n++] = i;
        uint32_t cp;
        size_t adv = sp_utf8_decode(text + i, text_len - i, &cp);
        if (adv == 0) adv = 1;
        i += adv;
    }
    out[n++] = text_len;
    return n;
}

/* ─── Constructor (mirrors Rust `SentencePieceUnigramTokenizer::new`) ─── */

OcError oc_sp_new(const OcSpPiece *pieces, size_t n_pieces,
                  OcArena *arena, OcSentencePieceTokenizer **out)
{
    if (!pieces || !arena || !out) return OC_ERR_INVALID_ARG;
    *out = NULL;

    OcSentencePieceTokenizer *sp = (OcSentencePieceTokenizer *)
        oc_arena_alloc(arena, sizeof(*sp), sizeof(void *));
    if (!sp) return OC_ERR_OOM;
    memset(sp, 0, sizeof(*sp));

    sp->vocab = oc_hashtable_new(n_pieces * 2 > 0 ? n_pieces * 2 : 16);
    sp->vocab_size = n_pieces;
    sp->id_to_token = oc_arena_alloc(arena,
                                      n_pieces * sizeof(char *), sizeof(void *));
    sp->piece_scores = oc_arena_alloc(arena,
                                       n_pieces * sizeof(float), sizeof(float));
    if (!sp->vocab || !sp->id_to_token || !sp->piece_scores) {
        return OC_ERR_OOM;
    }

    for (size_t id = 0; id < n_pieces; ++id) {
        const char *piece = pieces[id].piece;
        /* Rust inserts the piece verbatim; the caller owns the lifetime.
         * We dup into the arena so the hashtable key outlives the caller's
         * stack array. */
        char *dup = oc_arena_dup(arena, piece);
        if (!dup) return OC_ERR_OOM;
        sp->id_to_token[id] = dup;
        sp->piece_scores[id] = pieces[id].score;
        oc_hashtable_put(sp->vocab, dup, (void *)(uintptr_t)id, NULL);
    }

    *out = sp;
    return OC_OK;
}

OcError oc_sp_with_unknown_token(OcSentencePieceTokenizer *sp, OcArena *arena,
                                 const char *token)
{
    if (!sp || !arena || !token) return OC_ERR_INVALID_ARG;

    void *vp;
    uint32_t id;
    if (oc_hashtable_get(sp->vocab, token, &vp)) {
        id = (uint32_t)(uintptr_t)vp;
    } else {
        /* Rust allocates a new id = (max existing id) + 1 and grows both
         * the vocab and id_to_token. We can't realloc arena memory, so we
         * allocate a new id_to_token + piece_scores array of size
         * (vocab_size + 1) and copy. */
        id = (uint32_t)sp->vocab_size;
        char *dup = oc_arena_dup(arena, token);
        if (!dup) return OC_ERR_OOM;
        oc_hashtable_put(sp->vocab, dup, (void *)(uintptr_t)id, NULL);

        char **new_tokens = oc_arena_alloc(arena,
                                            (sp->vocab_size + 1) * sizeof(char *),
                                            sizeof(void *));
        float *new_scores = oc_arena_alloc(arena,
                                            (sp->vocab_size + 1) * sizeof(float),
                                            sizeof(float));
        if (!new_tokens || !new_scores) return OC_ERR_OOM;
        for (size_t i = 0; i < sp->vocab_size; ++i) {
            new_tokens[i] = sp->id_to_token[i];
            new_scores[i] = sp->piece_scores[i];
        }
        new_tokens[id] = dup;
        new_scores[id] = -INFINITY;  /* Rust sets f32::NEG_INFINITY */
        sp->id_to_token = new_tokens;
        sp->piece_scores = new_scores;
        sp->vocab_size = id + 1;
    }
    sp->has_unknown = true;
    sp->unknown_id = id;
    return OC_OK;
}

/* ─── Encode (Viterbi best-path segmentation) ────────────────────────── */

/* Find the best-scoring segmentation of `text` (a UTF-8 string) and write
 * the resulting id sequence to `*out_ids` (malloc'd) and the number of
 * codepoints consumed to `*out_consumed`. Returns false if no piece
 * starting at index 0 can be segmented (caller should emit <unk>).
 *
 * Mirrors Rust `SentencePieceUnigramTokenizer::best_segmentation`.
 *
 * Algorithm: forward Viterbi. `best_scores[j]` = the best total score of
 * any segmentation reaching boundary `j`. `backtrack[j]` = the (prev, id)
 * pair that achieved `best_scores[j]`. After the forward pass, find the
 * largest `end` with a finite score; backtrack from there to emit ids.
 */
static bool sp_best_segmentation(const OcSentencePieceTokenizer *sp,
                                 const char *text, size_t text_len,
                                 uint32_t **out_ids, size_t *out_count,
                                 size_t *out_consumed)
{
    *out_ids = NULL;
    *out_count = 0;
    *out_consumed = 0;

    /* Compute char boundaries. Worst case: every byte is a boundary. */
    size_t *boundaries = (size_t *)malloc((text_len + 1) * sizeof(size_t));
    if (!boundaries) return false;
    size_t n_bounds = sp_char_boundaries(text, text_len, boundaries);
    /* token_count = number of codepoints = n_bounds - 1. */
    size_t token_count = (n_bounds == 0) ? 0 : (n_bounds - 1);
    if (token_count == 0) {
        free(boundaries);
        /* Empty input: no ids, consumed = 0. */
        return true;
    }

    /* best_scores[0..=token_count], init to -INFINITY except [0]=0. */
    float *best_scores = (float *)malloc((token_count + 1) * sizeof(float));
    /* backtrack[j] = (prev_idx, id). id is irrelevant when prev_idx is
     * SIZE_MAX (unreachable). */
    size_t  *back_prev = (size_t *)malloc((token_count + 1) * sizeof(size_t));
    uint32_t *back_id  = (uint32_t *)malloc((token_count + 1) * sizeof(uint32_t));
    if (!best_scores || !back_prev || !back_id) {
        free(boundaries); free(best_scores); free(back_prev); free(back_id);
        return false;
    }
    for (size_t i = 0; i <= token_count; ++i) {
        best_scores[i] = -INFINITY;
        back_prev[i] = SIZE_MAX;
    }
    best_scores[0] = 0.0f;

    /* Forward Viterbi pass. For each start i with a finite score, try every
     * end j in (i, token_count] and look up the piece text[i..j]. */
    for (size_t i = 0; i < token_count; ++i) {
        if (!isfinite(best_scores[i])) continue;
        for (size_t j = i + 1; j <= token_count; ++j) {
            size_t start = boundaries[i];
            size_t end = boundaries[j];
            size_t len = end - start;
            /* We need a NUL-terminated key for the hashtable. Use a stack
             * buffer for short pieces (common case); fall back to malloc for
             * very long pieces (rare; would indicate a malformed vocab). */
            char stack_buf[256];
            char *key;
            bool use_stack = (len < sizeof(stack_buf));
            if (use_stack) {
                memcpy(stack_buf, text + start, len);
                stack_buf[len] = '\0';
                key = stack_buf;
            } else {
                key = (char *)malloc(len + 1);
                if (!key) continue;
                memcpy(key, text + start, len);
                key[len] = '\0';
            }
            void *vp;
            bool found = oc_hashtable_get(sp->vocab, key, &vp);
            if (!use_stack) free(key);
            if (!found) continue;
            uint32_t id = (uint32_t)(uintptr_t)vp;
            float score = (id < sp->vocab_size) ? sp->piece_scores[id] : 0.0f;
            float candidate = best_scores[i] + score;
            if (candidate > best_scores[j]) {
                best_scores[j] = candidate;
                back_prev[j] = i;
                back_id[j] = id;
            }
        }
    }

    /* Find the largest `end` with a finite score (mirrors Rust's
     * `(1..=token_count).rev().find(...)`). */
    size_t end = 0;
    for (size_t idx = token_count; idx >= 1; --idx) {
        if (isfinite(best_scores[idx])) {
            end = idx;
            break;
        }
    }
    if (end == 0) {
        /* No reachable boundary beyond 0: no segmentation possible. */
        free(boundaries); free(best_scores); free(back_prev); free(back_id);
        return false;
    }

    /* Backtrack from `end` to 0, collecting ids. */
    /* Worst-case id count = end (one id per codepoint). */
    uint32_t *ids_rev = (uint32_t *)malloc(end * sizeof(uint32_t));
    if (!ids_rev) {
        free(boundaries); free(best_scores); free(back_prev); free(back_id);
        return false;
    }
    size_t n_ids = 0;
    size_t cursor = end;
    while (cursor > 0) {
        if (back_prev[cursor] == SIZE_MAX) {
            /* Shouldn't happen if best_scores[end] is finite, but guard. */
            free(ids_rev);
            free(boundaries); free(best_scores); free(back_prev); free(back_id);
            return false;
        }
        ids_rev[n_ids++] = back_id[cursor];
        cursor = back_prev[cursor];
    }
    /* Reverse to get forward order. */
    uint32_t *ids = (uint32_t *)malloc((n_ids ? n_ids : 1) * sizeof(uint32_t));
    if (!ids) {
        free(ids_rev);
        free(boundaries); free(best_scores); free(back_prev); free(back_id);
        return false;
    }
    for (size_t i = 0; i < n_ids; ++i) {
        ids[i] = ids_rev[n_ids - 1 - i];
    }
    free(ids_rev);

    free(boundaries); free(best_scores); free(back_prev); free(back_id);
    *out_ids = ids;
    *out_count = n_ids;
    *out_consumed = end;
    return true;
}

OcError oc_sp_encode(const OcSentencePieceTokenizer *sp, const char *text,
                     uint32_t **out_ids, size_t *out_count)
{
    if (!sp || !text || !out_ids || !out_count) return OC_ERR_INVALID_ARG;
    *out_ids = NULL;
    *out_count = 0;

    size_t text_len = strlen(text);
    if (text_len == 0) return OC_OK;

    /* Compute char boundaries of the full text (used to advance by char
     * when best_segmentation fails). */
    size_t *boundaries = (size_t *)malloc((text_len + 1) * sizeof(size_t));
    if (!boundaries) return OC_ERR_OOM;
    size_t n_bounds = sp_char_boundaries(text, text_len, boundaries);

    OcVector result;
    OcError e = oc_vector_init(&result, sizeof(uint32_t));
    if (e != OC_OK) { free(boundaries); return e; }

    size_t boundary_idx = 0;
    while (boundary_idx + 1 < n_bounds) {
        size_t start = boundaries[boundary_idx];
        const char *segment = text + start;
        size_t seg_len = text_len - start;
        uint32_t *seg_ids = NULL;
        size_t seg_count = 0;
        size_t consumed = 0;
        if (sp_best_segmentation(sp, segment, seg_len,
                                  &seg_ids, &seg_count, &consumed)) {
            if (seg_count > 0) {
                e = oc_vector_push_n(&result, seg_ids, seg_count);
                free(seg_ids);
                if (e != OC_OK) {
                    free(boundaries);
                    oc_vector_free(&result);
                    return e;
                }
            } else {
                /* consumed == 0: no segmentation and no ids. Don't advance
                 * infinitely — emit unk (if configured) and advance by 1. */
                if (sp->has_unknown) {
                    uint32_t unk = sp->unknown_id;
                    e = oc_vector_push(&result, &unk);
                    if (e != OC_OK) {
                        free(boundaries);
                        oc_vector_free(&result);
                        return e;
                    }
                }
                boundary_idx += 1;
                continue;
            }
            boundary_idx += consumed;
            continue;
        }
        /* best_segmentation returned false: emit <unk> and advance by 1
         * char (mirrors Rust's `boundary_idx += 1`). */
        if (sp->has_unknown) {
            uint32_t unk = sp->unknown_id;
            e = oc_vector_push(&result, &unk);
            if (e != OC_OK) {
                free(boundaries);
                oc_vector_free(&result);
                return e;
            }
        }
        boundary_idx += 1;
    }

    free(boundaries);

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

OcError oc_sp_decode(const OcSentencePieceTokenizer *sp, const uint32_t *ids,
                     size_t count, char **out_text)
{
    if (!sp || !out_text) return OC_ERR_INVALID_ARG;
    *out_text = NULL;
    if (count == 0 || !ids) {
        char *empty = (char *)malloc(1);
        if (!empty) return OC_ERR_OOM;
        empty[0] = '\0';
        *out_text = empty;
        return OC_OK;
    }

    /* First pass: total length. */
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) {
        uint32_t id = ids[i];
        if (id >= sp->vocab_size || !sp->id_to_token[id]) {
            return OC_ERR_TOKENIZER;
        }
        total += strlen(sp->id_to_token[id]);
    }

    char *out = (char *)malloc(total + 1);
    if (!out) return OC_ERR_OOM;
    size_t off = 0;
    for (size_t i = 0; i < count; ++i) {
        const char *tok = sp->id_to_token[ids[i]];
        size_t len = strlen(tok);
        memcpy(out + off, tok, len);
        off += len;
    }
    out[off] = '\0';
    *out_text = out;
    return OC_OK;
}

/* ─── Load from GGUF metadata ───────────────────────────────────────── */

/* Helper: get a metadata string array as a vector of arena-owned strings. */
static OcError sp_get_string_array(const OcGgufFile *gguf, const char *key,
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

/* Helper: get a metadata f32 array into a malloc'd buffer. */
static OcError sp_get_f32_array(const OcGgufFile *gguf, const char *key,
                                float **out, size_t *out_len)
{
    const OcGgufMetadataValue *v = oc_gguf_metadata_get(gguf, key);
    if (!v || v->type != OC_GGUF_MT_ARRAY) {
        return OC_ERR_TOKENIZER;
    }
    float *arr = (float *)malloc(v->v.arr.len * sizeof(float));
    if (!arr) return OC_ERR_OOM;
    for (size_t i = 0; i < v->v.arr.len; ++i) {
        const OcGgufMetadataValue *elem = &v->v.arr.values[i];
        if (elem->type == OC_GGUF_MT_FLOAT32) {
            arr[i] = elem->v.f32;
        } else if (elem->type == OC_GGUF_MT_FLOAT64) {
            arr[i] = (float)elem->v.f64;
        } else {
            free(arr);
            return OC_ERR_TOKENIZER;
        }
    }
    *out = arr;
    *out_len = v->v.arr.len;
    return OC_OK;
}

OcError oc_sp_load_from_gguf(const OcGgufFile *gguf, OcArena *arena,
                             OcSentencePieceTokenizer **out)
{
    if (!gguf || !arena || !out) return OC_ERR_INVALID_ARG;
    *out = NULL;

    OcVector tokens;
    OcError e = sp_get_string_array(gguf, "tokenizer.ggml.tokens", arena, &tokens);
    if (e != OC_OK) return e;

    float *scores = NULL;
    size_t n_scores = 0;
    e = sp_get_f32_array(gguf, "tokenizer.ggml.scores", &scores, &n_scores);
    if (e != OC_OK) { oc_vector_free(&tokens); return e; }

    if (oc_vector_len(&tokens) != n_scores) {
        oc_log_error("tokenizer_sp: tokens(%zu) != scores(%zu)",
                     oc_vector_len(&tokens), n_scores);
        oc_vector_free(&tokens);
        free(scores);
        return OC_ERR_TOKENIZER;
    }

    size_t vocab_size = oc_vector_len(&tokens);
    OcSentencePieceTokenizer *sp = (OcSentencePieceTokenizer *)
        oc_arena_alloc(arena, sizeof(*sp), sizeof(void *));
    if (!sp) { oc_vector_free(&tokens); free(scores); return OC_ERR_OOM; }
    memset(sp, 0, sizeof(*sp));

    sp->vocab = oc_hashtable_new(vocab_size * 2 > 0 ? vocab_size * 2 : 16);
    sp->vocab_size = vocab_size;
    sp->id_to_token = oc_arena_alloc(arena, vocab_size * sizeof(char *), sizeof(void *));
    sp->piece_scores = oc_arena_alloc(arena, vocab_size * sizeof(float), sizeof(float));
    if (!sp->vocab || !sp->id_to_token || !sp->piece_scores) {
        oc_vector_free(&tokens); free(scores);
        return OC_ERR_OOM;
    }

    for (size_t id = 0; id < vocab_size; ++id) {
        char *tok = *(char *const *)oc_vector_get(&tokens, id);
        sp->id_to_token[id] = tok;
        sp->piece_scores[id] = scores[id];
        oc_hashtable_put(sp->vocab, tok, (void *)(uintptr_t)id, NULL);
    }

    oc_vector_free(&tokens);
    free(scores);

    /* Read special-token ids from metadata. */
    uint32_t v;
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.unknown_token_id", &v)) {
        sp->has_unknown = true; sp->unknown_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.bos_token_id", &v)) {
        sp->has_bos = true; sp->bos_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.eos_token_id", &v)) {
        sp->has_eos = true; sp->eos_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.padding_token_id", &v)
        || oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.pad_token_id", &v)) {
        sp->has_pad = true; sp->pad_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.separator_token_id", &v)
        || oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.sep_token_id", &v)) {
        sp->has_separator = true; sp->separator_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.cls_token_id", &v)) {
        sp->has_cls = true; sp->cls_id = v;
    }
    if (oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.mask_token_id", &v)) {
        sp->has_mask = true; sp->mask_id = v;
    }

    *out = sp;
    return OC_OK;
}

void oc_sp_fill_special_tokens(const OcSentencePieceTokenizer *sp, OcTokenizer *out)
{
    if (!sp || !out) return;
    out->has_unknown = sp->has_unknown;     out->unknown_id = sp->unknown_id;
    out->has_bos = sp->has_bos;            out->bos_id = sp->bos_id;
    out->has_eos = sp->has_eos;            out->eos_id = sp->eos_id;
    out->has_pad = sp->has_pad;             out->pad_id = sp->pad_id;
    out->has_separator = sp->has_separator; out->separator_id = sp->separator_id;
    out->has_cls = sp->has_cls;             out->cls_id = sp->cls_id;
    out->has_mask = sp->has_mask;           out->mask_id = sp->mask_id;
}

void oc_sp_free(OcSentencePieceTokenizer *sp)
{
    if (!sp) return;
    /* Free the malloc'd vocab hashtable. id_to_token and piece_scores are
     * arena-owned. */
    oc_hashtable_free(sp->vocab);
    sp->vocab = NULL;
}
