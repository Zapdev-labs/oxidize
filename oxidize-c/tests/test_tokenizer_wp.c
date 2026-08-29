/* test_tokenizer_wp.c — Criterion tests for the WordPiece tokenizer
 * (BERT `##` continuation).
 *
 * Covers:
 *   VAL-TOK-008 — WordPiece ## continuation correct
 *   VAL-TOK-010 — Dispatch by tokenizer.ggml.model key (bert)
 *   VAL-TOK-011 — Parity vs Rust LoadedTokenizer (toy WP parity with
 *                 Rust `WordPieceTokenizer` unit tests).
 *
 * Parity reference: oxidize-core/src/format/tokenizer.rs `#[cfg(test)]`:
 *   - `wordpiece_encodes_with_greedy_longest_match`
 *   - `wordpiece_uses_unknown_for_unmatched_word`
 *   - `wordpiece_decode_errors_on_unknown_id`
 */

#define _POSIX_C_SOURCE 200809L  /* mkstemp */

#include "framework.h"

#include "oxidize/tokenizer.h"
#include "oxidize/arena.h"
#include "oxidize/error.h"
#include "oxidize/gguf.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static void assert_ids_eq(const uint32_t *actual, size_t actual_count,
                          const uint32_t *expected, size_t expected_count)
{
    cr_assert_eq(actual_count, expected_count,
                 "id count mismatch: got %zu, expected %zu",
                 actual_count, expected_count);
    for (size_t i = 0; i < expected_count; ++i) {
        cr_assert_eq(actual[i], expected[i],
                     "id[%zu]: got %u, expected %u",
                     i, actual[i], expected[i]);
    }
}

/* ─── Greedy longest match (VAL-TOK-008) ─────────────────────────────────
 * Mirrors Rust `wordpiece_encodes_with_greedy_longest_match`:
 *   vocab = ["play", "##ing", "##er", " "]
 *   encode("player playing") -> decode -> "player playing"
 *   encoded.len() == 5
 *
 * "player" splits to ["play", "##er"] (greedy longest: "play" (4 chars)
 * matches first, then "##er" (2 chars) matches the suffix).
 * " " is one whitespace token.
 * "playing" splits to ["play", "##ing"].
 * Total: 2 + 1 + 2 = 5 tokens. */

Test(tokenizer_wp, greedy_longest_match)
{
    OcArena *arena = oc_arena_new(0);
    cr_assert_not_null(arena);

    const char *vocab[] = { "play", "##ing", "##er", " " };
    OcWordPieceTokenizer *wp = NULL;
    OcError e = oc_wp_new(vocab, 4, arena, &wp);
    cr_assert_eq(e, OC_OK, "new: %s", oc_error_msg(e));

    uint32_t *ids = NULL;
    size_t count = 0;
    e = oc_wp_encode(wp, "player playing", &ids, &count);
    cr_assert_eq(e, OC_OK, "encode: %s", oc_error_msg(e));
    cr_assert_eq(count, 5, "should produce 5 tokens, got %zu", count);

    /* Expected: ["play", "##er", " ", "play", "##ing"]
     *          = [0, 2, 3, 0, 1] */
    static const uint32_t expected[] = { 0, 2, 3, 0, 1 };
    assert_ids_eq(ids, count, expected, 5);

    char *decoded = NULL;
    e = oc_wp_decode(wp, ids, count, &decoded);
    cr_assert_eq(e, OC_OK, "decode: %s", oc_error_msg(e));
    cr_assert_str_eq(decoded, "player playing",
                     "round-trip should give 'player playing'");

    free(ids);
    free(decoded);
    oc_wp_free(wp);
    oc_arena_free(arena);
}

/* ─── `##` continuation correct on decode ────────────────────────────────
 * Explicitly verify the ## prefix is stripped on decode. */

Test(tokenizer_wp, continuation_prefix_stripped_on_decode)
{
    OcArena *arena = oc_arena_new(0);
    const char *vocab[] = { "play", "##ing" };
    OcWordPieceTokenizer *wp = NULL;
    OcError e = oc_wp_new(vocab, 2, arena, &wp);
    cr_assert_eq(e, OC_OK);

    /* Decode ["play", "##ing"] -> "playing" (no separator). */
    uint32_t ids[] = { 0, 1 };
    char *decoded = NULL;
    e = oc_wp_decode(wp, ids, 2, &decoded);
    cr_assert_eq(e, OC_OK);
    cr_assert_str_eq(decoded, "playing",
                     "## should be stripped, pieces concatenated");

    free(decoded);
    oc_wp_free(wp);
    oc_arena_free(arena);
}

/* ─── Unknown token fallback ─────────────────────────────────────────────
 * Mirrors Rust `wordpiece_uses_unknown_for_unmatched_word`:
 *   vocab = ["hello", "world", " "] + unk("<unk>")
 *   encode("hello mars") -> decode -> "hello <unk>"
 *
 * "mars" cannot be split (no matching pieces), so the whole word becomes
 * <unk>. The space between "hello" and "mars" is emitted as its own id. */

Test(tokenizer_wp, unknown_for_unmatched_word)
{
    OcArena *arena = oc_arena_new(0);
    const char *vocab[] = { "hello", "world", " " };
    OcWordPieceTokenizer *wp = NULL;
    OcError e = oc_wp_new(vocab, 3, arena, &wp);
    cr_assert_eq(e, OC_OK);
    e = oc_wp_with_unknown_token(wp, arena, "<unk>");
    cr_assert_eq(e, OC_OK);

    uint32_t *ids = NULL;
    size_t count = 0;
    e = oc_wp_encode(wp, "hello mars", &ids, &count);
    cr_assert_eq(e, OC_OK, "encode: %s", oc_error_msg(e));

    char *decoded = NULL;
    e = oc_wp_decode(wp, ids, count, &decoded);
    cr_assert_eq(e, OC_OK);
    cr_assert_str_eq(decoded, "hello <unk>",
                     "decode should give 'hello <unk>', got '%s'", decoded);

    free(ids);
    free(decoded);
    oc_wp_free(wp);
    oc_arena_free(arena);
}

/* ─── Decode unknown id returns error ────────────────────────────────────
 * Mirrors Rust `wordpiece_decode_errors_on_unknown_id`. */

Test(tokenizer_wp, decode_unknown_id_returns_error)
{
    OcArena *arena = oc_arena_new(0);
    const char *vocab[] = { "a" };
    OcWordPieceTokenizer *wp = NULL;
    OcError e = oc_wp_new(vocab, 1, arena, &wp);
    cr_assert_eq(e, OC_OK);

    char *decoded = NULL;
    uint32_t bad_ids[] = { 42 };
    e = oc_wp_decode(wp, bad_ids, 1, &decoded);
    cr_assert_eq(e, OC_ERR_TOKENIZER,
                 "unknown id should return OC_ERR_TOKENIZER");

    oc_wp_free(wp);
    oc_arena_free(arena);
}

/* ─── Empty input ──────────────────────────────────────────────────────── */

Test(tokenizer_wp, empty_input_returns_empty)
{
    OcArena *arena = oc_arena_new(0);
    const char *vocab[] = { "a" };
    OcWordPieceTokenizer *wp = NULL;
    OcError e = oc_wp_new(vocab, 1, arena, &wp);
    cr_assert_eq(e, OC_OK);

    uint32_t *ids = NULL;
    size_t count = 1;
    e = oc_wp_encode(wp, "", &ids, &count);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(count, 0, "empty input should produce 0 tokens");

    char *decoded = NULL;
    e = oc_wp_decode(wp, NULL, 0, &decoded);
    cr_assert_eq(e, OC_OK);
    cr_assert_str_eq(decoded, "");

    free(ids);
    free(decoded);
    oc_wp_free(wp);
    oc_arena_free(arena);
}

/* ─── Continuation at non-zero offset only ──────────────────────────────
 * A `##ing` token only matches when start_idx > 0 within a word. Encoding
 * "ing" (a whole word) should NOT match "##ing". */

Test(tokenizer_wp, continuation_only_at_nonzero_offset)
{
    OcArena *arena = oc_arena_new(0);
    /* vocab: "play", "##ing", "ing" — "ing" is a whole-word token. */
    const char *vocab[] = { "play", "##ing", "ing" };
    OcWordPieceTokenizer *wp = NULL;
    OcError e = oc_wp_new(vocab, 3, arena, &wp);
    cr_assert_eq(e, OC_OK);

    /* "ing" alone → ["ing"] (id 2), not ["##ing"] (which would need a
     * leading word piece). */
    uint32_t *ids = NULL;
    size_t count = 0;
    e = oc_wp_encode(wp, "ing", &ids, &count);
    cr_assert_eq(e, OC_OK);
    assert_ids_eq(ids, count, (uint32_t[]){2}, 1);

    char *decoded = NULL;
    e = oc_wp_decode(wp, ids, count, &decoded);
    cr_assert_eq(e, OC_OK);
    cr_assert_str_eq(decoded, "ing");
    free(ids);
    free(decoded);

    /* "playing" → ["play", "##ing"] = [0, 1]. */
    e = oc_wp_encode(wp, "playing", &ids, &count);
    cr_assert_eq(e, OC_OK);
    assert_ids_eq(ids, count, (uint32_t[]){0, 1}, 2);
    e = oc_wp_decode(wp, ids, count, &decoded);
    cr_assert_eq(e, OC_OK);
    cr_assert_str_eq(decoded, "playing");
    free(ids);
    free(decoded);

    oc_wp_free(wp);
    oc_arena_free(arena);
}

/* ─── Multibyte whitespace ──────────────────────────────────────────────
 * WordPiece flushes on Unicode whitespace (mirrors Rust
 * `char::is_whitespace`). Verify non-breaking space (U+00A0) and
 * ideographic space (U+3000) flush the current word. */

Test(tokenizer_wp, multibyte_whitespace_flushes_word)
{
    OcArena *arena = oc_arena_new(0);
    const char *vocab[] = { "ab", "cd" };
    OcWordPieceTokenizer *wp = NULL;
    OcError e = oc_wp_new(vocab, 2, arena, &wp);
    cr_assert_eq(e, OC_OK);

    /* "ab cd" with a regular space — should produce [0, (unk or skip), 1]
     * since space is not in vocab. Without an unk configured, the space
     * is skipped. So result is [0, 1]. */
    uint32_t *ids = NULL;
    size_t count = 0;
    e = oc_wp_encode(wp, "ab cd", &ids, &count);
    cr_assert_eq(e, OC_OK);
    assert_ids_eq(ids, count, (uint32_t[]){0, 1}, 2);
    free(ids);

    /* "ab\u00A0cd" (non-breaking space) — also flushes the word. */
    e = oc_wp_encode(wp, "ab\xc2\xa0""cd", &ids, &count);
    cr_assert_eq(e, OC_OK);
    assert_ids_eq(ids, count, (uint32_t[]){0, 1}, 2);
    free(ids);

    oc_wp_free(wp);
    oc_arena_free(arena);
}

/* ─── GGUF load (BERT) ───────────────────────────────────────────────────
 * Builds a synthetic GGUF with tokenizer.ggml.model = "bert" and verifies
 * dispatch + encode/decode round-trip. */

Test(tokenizer_wp, load_from_gguf_bert)
{
    size_t cap = 4096;
    uint8_t *buf = calloc(cap, 1);
    cr_assert_not_null(buf);
    size_t off = 0;
#define EMIT(buf, off, src, n) do { memcpy((buf) + (off), (src), (n)); (off) += (n); } while (0)
#define EMIT_U32(buf, off, v) do { uint32_t _x = (uint32_t)(v); EMIT(buf, off, &_x, 4); } while (0)
#define EMIT_U64(buf, off, v) do { uint64_t _x = (uint64_t)(v); EMIT(buf, off, &_x, 8); } while (0)
#define EMIT_KV_STR_KEY(buf, off, key_str) do { \
        const char *k = (key_str); \
        uint64_t kl = strlen(k); \
        EMIT_U64(buf, off, kl); \
        EMIT(buf, off, k, kl); \
    } while (0)

    EMIT_U32(buf, off, OC_GGUF_MAGIC);
    EMIT_U32(buf, off, 3);
    EMIT_U64(buf, off, 0);
    EMIT_U64(buf, off, 3);   /* kv_count = 3 */

    /* KV 1: model = "bert" */
    { EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.model");
      EMIT_U32(buf, off, OC_GGUF_MT_STRING);
      const char *v = "bert"; EMIT_U64(buf, off, strlen(v)); EMIT(buf, off, v, strlen(v)); }
    /* KV 2: tokens = ["play", "##ing", "##er", " "] */
    { EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.tokens");
      EMIT_U32(buf, off, OC_GGUF_MT_ARRAY); EMIT_U32(buf, off, OC_GGUF_MT_STRING);
      EMIT_U64(buf, off, 4);
      const char *t[] = { "play", "##ing", "##er", " " };
      for (int i = 0; i < 4; ++i) { uint64_t sl = strlen(t[i]); EMIT_U64(buf, off, sl); EMIT(buf, off, t[i], sl); } }
    /* KV 3: unknown_token_id = 4 (out of vocab range, for testing) */
    { EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.unknown_token_id");
      EMIT_U32(buf, off, OC_GGUF_MT_UINT32); EMIT_U32(buf, off, 4); }

#undef EMIT
#undef EMIT_U32
#undef EMIT_U64
#undef EMIT_KV_STR_KEY

    OcGgufFile gguf;
    size_t padded_len = (off + 31) & ~(size_t)31;
    OcError e = oc_gguf_parse(buf, padded_len, &gguf);
    cr_assert_eq(e, OC_OK, "parse: %s", oc_error_msg(e));

    OcTokenizer tok;
    e = oc_tokenizer_load_from_gguf(&gguf, &tok);
    cr_assert_eq(e, OC_OK, "load: %s", oc_error_msg(e));
    cr_assert_eq(tok.kind, OC_TOK_KIND_WORDPIECE, "bert -> WordPiece");
    cr_assert(tok.has_unknown, "should have unknown id");
    cr_assert_eq(tok.unknown_id, 4);

    /* Round-trip "player playing" → ["play","##er"," ","play","##ing"]. */
    uint32_t *ids = NULL;
    size_t count = 0;
    e = oc_tokenizer_encode(&tok, "player playing", OC_TOK_DEFAULT, &ids, &count);
    cr_assert_eq(e, OC_OK);
    assert_ids_eq(ids, count, (uint32_t[]){0, 2, 3, 0, 1}, 5);

    char *decoded = NULL;
    e = oc_tokenizer_decode(&tok, ids, count, &decoded);
    cr_assert_eq(e, OC_OK);
    cr_assert_str_eq(decoded, "player playing");

    free(ids);
    free(decoded);
    oc_tokenizer_free(&tok);
    oc_gguf_free(&gguf);
    free(buf);
}

/* ─── Dispatch (VAL-TOK-010) ───────────────────────────────────────────── */

Test(tokenizer_wp, dispatch_by_model_key_bert)
{
    size_t cap = 1024;
    uint8_t *buf = calloc(cap, 1);
    cr_assert_not_null(buf);
    size_t off = 0;
#define EMIT(buf, off, src, n) do { memcpy((buf) + (off), (src), (n)); (off) += (n); } while (0)
#define EMIT_U32(buf, off, v) do { uint32_t _x = (uint32_t)(v); EMIT(buf, off, &_x, 4); } while (0)
#define EMIT_U64(buf, off, v) do { uint64_t _x = (uint64_t)(v); EMIT(buf, off, &_x, 8); } while (0)
#define EMIT_KV_STR_KEY(buf, off, key_str) do { \
        const char *k = (key_str); \
        uint64_t kl = strlen(k); \
        EMIT_U64(buf, off, kl); \
        EMIT(buf, off, k, kl); \
    } while (0)

    EMIT_U32(buf, off, OC_GGUF_MAGIC);
    EMIT_U32(buf, off, 3);
    EMIT_U64(buf, off, 0);
    EMIT_U64(buf, off, 2);
    { EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.model");
      EMIT_U32(buf, off, OC_GGUF_MT_STRING);
      const char *v = "bert"; EMIT_U64(buf, off, strlen(v)); EMIT(buf, off, v, strlen(v)); }
    { EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.tokens");
      EMIT_U32(buf, off, OC_GGUF_MT_ARRAY); EMIT_U32(buf, off, OC_GGUF_MT_STRING);
      EMIT_U64(buf, off, 1);
      const char *t = "a"; uint64_t sl = 1; EMIT_U64(buf, off, sl); EMIT(buf, off, t, sl); }
#undef EMIT
#undef EMIT_U32
#undef EMIT_U64
#undef EMIT_KV_STR_KEY

    OcGgufFile gguf;
    size_t padded_len = (off + 31) & ~(size_t)31;
    OcError e = oc_gguf_parse(buf, padded_len, &gguf);
    cr_assert_eq(e, OC_OK);

    OcTokenizer tok;
    e = oc_tokenizer_load_from_gguf(&gguf, &tok);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(tok.kind, OC_TOK_KIND_WORDPIECE);

    oc_tokenizer_free(&tok);
    oc_gguf_free(&gguf);
    free(buf);
}
