/* test_tokenizer_tiktoken.c — Criterion tests for the raw Tiktoken
 * tokenizer (byte-level, no GPT-2 mapping).
 *
 * Covers:
 *   VAL-TOK-009 — Tiktoken format encode/decode (round-trip)
 *   VAL-TOK-010 — Dispatch by tokenizer.ggml.model key (tiktoken)
 *   VAL-TOK-011 — Parity vs Rust LoadedTokenizer (toy parity with Rust
 *                 `TiktokenTokenizer` unit tests).
 *   <unk> fallback for OOV bytes (VAL-TOK-009 OOV clause).
 *
 * Parity reference: oxidize-core/src/format/tokenizer.rs `#[cfg(test)]`:
 *   - `tiktoken_merges_by_rank_and_round_trips`
 *   - `tiktoken_supports_utf8_bytes`
 *   - `tiktoken_uses_unknown_token_for_missing_bytes`
 */

#define _POSIX_C_SOURCE 200809L  /* mkstemp */

#include <criterion/criterion.h>

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

/* Build a byte slice from a C string literal. */
static OcByteSlice bs(const uint8_t *data, size_t len)
{
    OcByteSlice s; s.data = data; s.len = len; return s;
}

/* ─── Merge-by-rank + round-trip (VAL-TOK-009) ──────────────────────────
 * Mirrors Rust `tiktoken_merges_by_rank_and_round_trips`:
 *   vocab = [b"h", b"e", b"l", b"o", b"he", b"ll", b"hell", b"hello"]
 *   merges = [(h,e), (l,l), (he,ll), (hell,o)]
 *   encode("hello") -> len 1, decode -> "hello" */

Test(tokenizer_tiktoken, merges_by_rank_and_round_trips)
{
    OcArena *arena = oc_arena_new(0);
    cr_assert_not_null(arena);

    /* Vocab tokens. Note: ids 0..7. */
    static const uint8_t h[]  = { 'h' };
    static const uint8_t e[]  = { 'e' };
    static const uint8_t l[]  = { 'l' };
    static const uint8_t o[]  = { 'o' };
    static const uint8_t he[] = { 'h', 'e' };
    static const uint8_t ll[] = { 'l', 'l' };
    static const uint8_t hell[] = { 'h', 'e', 'l', 'l' };
    static const uint8_t hello[] = { 'h', 'e', 'l', 'l', 'o' };
    OcByteSlice vocab[] = {
        bs(h, 1), bs(e, 1), bs(l, 1), bs(o, 1),
        bs(he, 2), bs(ll, 2), bs(hell, 4), bs(hello, 5),
    };
    OcByteSlicePair merges[] = {
        { bs(h, 1), bs(e, 1) },      /* rank 0: h+e -> he (id 4) */
        { bs(l, 1), bs(l, 1) },      /* rank 1: l+l -> ll (id 5) */
        { bs(he, 2), bs(ll, 2) },    /* rank 2: he+ll -> hell (id 6) */
        { bs(hell, 4), bs(o, 1) },   /* rank 3: hell+o -> hello (id 7) */
    };

    OcTiktokenTokenizer *t = NULL;
    OcError err = oc_tiktoken_new(vocab, 8, merges, 4, arena, &t);
    cr_assert_eq(err, OC_OK, "new: %s", oc_error_msg(err));

    uint32_t *ids = NULL;
    size_t count = 0;
    err = oc_tiktoken_encode(t, "hello", &ids, &count);
    cr_assert_eq(err, OC_OK, "encode: %s", oc_error_msg(err));
    /* The merge loop should fully reduce "hello" to id 7. */
    cr_assert_eq(count, 1, "should produce 1 token, got %zu", count);
    cr_assert_eq(ids[0], 7, "merged token should be id 7 (hello)");

    char *decoded = NULL;
    err = oc_tiktoken_decode(t, ids, count, &decoded);
    cr_assert_eq(err, OC_OK, "decode: %s", oc_error_msg(err));
    cr_assert_str_eq(decoded, "hello", "round-trip should give 'hello'");

    free(ids);
    free(decoded);
    oc_tiktoken_free(t);
    oc_arena_free(arena);
}

/* ─── UTF-8 bytes (VAL-TOK-009) ──────────────────────────────────────────
 * Mirrors Rust `tiktoken_supports_utf8_bytes`:
 *   vocab = [b"h", b"i", b" ", &[0xc3], &[0xa9], b"\xc3\xa9"]
 *   encode("hi é") -> decode -> "hi é"
 *
 * The multi-byte é (U+00E9) is stored as two single-byte tokens (0xc3, 0xa9)
 * plus the merged two-byte form. With no merges, each byte maps to its
 * single-byte token. */

Test(tokenizer_tiktoken, supports_utf8_bytes)
{
    OcArena *arena = oc_arena_new(0);
    static const uint8_t h[]  = { 'h' };
    static const uint8_t i[]  = { 'i' };
    static const uint8_t sp[] = { ' ' };
    static const uint8_t c3[] = { 0xc3 };
    static const uint8_t a9[] = { 0xa9 };
    static const uint8_t e_acute[] = { 0xc3, 0xa9 };
    OcByteSlice vocab[] = {
        bs(h, 1), bs(i, 1), bs(sp, 1),
        bs(c3, 1), bs(a9, 1), bs(e_acute, 2),
    };

    OcTiktokenTokenizer *t = NULL;
    OcError err = oc_tiktoken_new(vocab, 6, NULL, 0, arena, &t);
    cr_assert_eq(err, OC_OK);

    /* "hi é" = h(0) i(1) ' '(2) 0xc3(3) 0xa9(4) — no merges, so 5 ids. */
    uint32_t *ids = NULL;
    size_t count = 0;
    err = oc_tiktoken_encode(t, "hi \xc3\xa9", &ids, &count);
    cr_assert_eq(err, OC_OK);
    /* Expected: [0, 1, 2, 3, 4] */
    assert_ids_eq(ids, count, (uint32_t[]){0, 1, 2, 3, 4}, 5);

    char *decoded = NULL;
    err = oc_tiktoken_decode(t, ids, count, &decoded);
    cr_assert_eq(err, OC_OK);
    cr_assert_str_eq(decoded, "hi \xc3\xa9",
                     "round-trip should give 'hi é'");

    free(ids);
    free(decoded);
    oc_tiktoken_free(t);
    oc_arena_free(arena);
}

Test(tokenizer_tiktoken, streaming_preserves_split_and_long_tokens)
{
    OcArena *arena = oc_arena_new(0);
    static const uint8_t lead[] = { 0xc3 };
    static const uint8_t continuation[] = { 0xa9 };
    uint8_t *long_token = malloc(5000);
    cr_assert_not_null(long_token);
    memset(long_token, 'x', 5000);
    OcByteSlice vocab[] = {
        bs(lead, sizeof(lead)),
        bs(continuation, sizeof(continuation)),
        bs(long_token, 5000),
    };
    OcTiktokenTokenizer *t = NULL;
    cr_assert_eq(oc_tiktoken_new(vocab, 3, NULL, 0, arena, &t), OC_OK);
    free(long_token);
    OcTokenizer tokenizer = { .kind = OC_TOK_KIND_TIKTOKEN, .tiktoken = t };
    OcStreamingDetokenizer first, second;
    oc_streaming_detok_init(&first, &tokenizer);
    oc_streaming_detok_init(&second, &tokenizer);

    const char *delta = NULL;
    size_t len = 0;
    cr_assert_eq(oc_streaming_detok_push(&first, 0, &delta, &len), OC_OK);
    cr_assert_eq(len, 0);
    cr_assert_eq(oc_streaming_detok_push(&first, 1, &delta, &len), OC_OK);
    cr_assert_eq(len, 2);
    cr_assert_arr_eq(delta, "\xc3\xa9", 2);
    const char *first_delta = delta;

    cr_assert_eq(oc_streaming_detok_push(&second, 2, &delta, &len), OC_OK);
    cr_assert_eq(len, 5000);
    cr_assert_eq(delta[0], 'x');
    cr_assert_eq(delta[4999], 'x');
    cr_assert_arr_eq(first_delta, "\xc3\xa9", 2);

    oc_streaming_detok_free(&first);
    oc_streaming_detok_free(&second);
    oc_tiktoken_free(t);
    oc_arena_free(arena);
}

/* ─── Unknown token fallback for missing bytes (VAL-TOK-009 OOV) ────────
 * Mirrors Rust `tiktoken_uses_unknown_token_for_missing_bytes`:
 *   vocab = [b"a"] + unk(b"<unk>")
 *   encode("ab") -> decode -> "a<unk>"
 *
 * 'b' (0x62) is not in the single-byte vocab, so it maps to <unk>. */

Test(tokenizer_tiktoken, unknown_for_missing_bytes)
{
    OcArena *arena = oc_arena_new(0);
    static const uint8_t a[] = { 'a' };
    OcByteSlice vocab[] = { bs(a, 1) };

    OcTiktokenTokenizer *t = NULL;
    OcError err = oc_tiktoken_new(vocab, 1, NULL, 0, arena, &t);
    cr_assert_eq(err, OC_OK);
    err = oc_tiktoken_with_unknown_token(t, arena, "<unk>");
    cr_assert_eq(err, OC_OK);

    /* Encode "ab": 'a' → id 0, 'b' → unk id (1). */
    uint32_t *ids = NULL;
    size_t count = 0;
    err = oc_tiktoken_encode(t, "ab", &ids, &count);
    cr_assert_eq(err, OC_OK);
    cr_assert_eq(count, 2, "should produce 2 ids");
    cr_assert_eq(ids[0], 0, "first id should be 'a' (0)");
    cr_assert_eq(ids[1], 1, "second id should be <unk> (1)");

    char *decoded = NULL;
    err = oc_tiktoken_decode(t, ids, count, &decoded);
    cr_assert_eq(err, OC_OK);
    cr_assert_str_eq(decoded, "a<unk>",
                     "decode should give 'a<unk>'");

    free(ids);
    free(decoded);
    oc_tiktoken_free(t);
    oc_arena_free(arena);
}

/* ─── Decode unknown id returns error ──────────────────────────────────── */

Test(tokenizer_tiktoken, decode_unknown_id_returns_error)
{
    OcArena *arena = oc_arena_new(0);
    static const uint8_t a[] = { 'a' };
    OcByteSlice vocab[] = { bs(a, 1) };
    OcTiktokenTokenizer *t = NULL;
    OcError err = oc_tiktoken_new(vocab, 1, NULL, 0, arena, &t);
    cr_assert_eq(err, OC_OK);

    char *decoded = NULL;
    uint32_t bad_ids[] = { 999 };
    err = oc_tiktoken_decode(t, bad_ids, 1, &decoded);
    cr_assert_eq(err, OC_ERR_TOKENIZER,
                 "unknown id should return OC_ERR_TOKENIZER");

    oc_tiktoken_free(t);
    oc_arena_free(arena);
}

/* ─── Empty input ──────────────────────────────────────────────────────── */

Test(tokenizer_tiktoken, empty_input_returns_empty)
{
    OcArena *arena = oc_arena_new(0);
    static const uint8_t a[] = { 'a' };
    OcByteSlice vocab[] = { bs(a, 1) };
    OcTiktokenTokenizer *t = NULL;
    OcError err = oc_tiktoken_new(vocab, 1, NULL, 0, arena, &t);
    cr_assert_eq(err, OC_OK);

    uint32_t *ids = NULL;
    size_t count = 1;
    err = oc_tiktoken_encode(t, "", &ids, &count);
    cr_assert_eq(err, OC_OK);
    cr_assert_eq(count, 0, "empty input should produce 0 tokens");

    char *decoded = NULL;
    err = oc_tiktoken_decode(t, NULL, 0, &decoded);
    cr_assert_eq(err, OC_OK);
    cr_assert_str_eq(decoded, "");

    free(ids);
    free(decoded);
    oc_tiktoken_free(t);
    oc_arena_free(arena);
}

/* ─── Lossy UTF-8 decode ──────────────────────────────────────────────────
 * Mirrors Rust `String::from_utf8_lossy`: invalid UTF-8 byte sequences are
 * replaced with U+FFFD (0xEF 0xBF 0xBD). */

Test(tokenizer_tiktoken, lossy_utf8_decode)
{
    OcArena *arena = oc_arena_new(0);
    /* Vocab with single-byte tokens for 0x80 and 0x41 ('A'). */
    static const uint8_t b80[] = { 0x80 };
    static const uint8_t A[]   = { 'A' };
    OcByteSlice vocab[] = { bs(b80, 1), bs(A, 1) };
    OcTiktokenTokenizer *t = NULL;
    OcError err = oc_tiktoken_new(vocab, 2, NULL, 0, arena, &t);
    cr_assert_eq(err, OC_OK);

    /* Decode [1, 0, 1] = "A" + lone 0x80 + "A". The lone 0x80 is invalid
     * UTF-8 (continuation byte without lead), so it becomes U+FFFD. */
    uint32_t ids[] = { 1, 0, 1 };
    char *decoded = NULL;
    err = oc_tiktoken_decode(t, ids, 3, &decoded);
    cr_assert_eq(err, OC_OK);
    /* Expected: "A" + U+FFFD + "A" = bytes 41 EF BF BD 41.
     * Split the literal so the trailing 'A' is not consumed by the
     * \xBD hex escape (C parses \xBDA as one out-of-range escape). */
    static const char expected[] = "A" "\xEF\xBF\xBD" "A";
    cr_assert_str_eq(decoded, expected,
                     "invalid UTF-8 should be replaced with U+FFFD");

    free(decoded);
    oc_tiktoken_free(t);
    oc_arena_free(arena);
}

/* ─── GGUF load (VAL-TOK-009 raw tiktoken format) ────────────────────────
 * Builds a synthetic GGUF with tokenizer.ggml.model = "tiktoken" and
 * verifies dispatch + encode/decode round-trip. */

Test(tokenizer_tiktoken, load_from_gguf_tiktoken)
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

    /* KV 1: model = "tiktoken" */
    { EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.model");
      EMIT_U32(buf, off, OC_GGUF_MT_STRING);
      const char *v = "tiktoken"; EMIT_U64(buf, off, strlen(v)); EMIT(buf, off, v, strlen(v)); }
    /* KV 2: tokens = ["h", "e", "l", "o", "he", "ll", "hell", "hello"] */
    { EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.tokens");
      EMIT_U32(buf, off, OC_GGUF_MT_ARRAY); EMIT_U32(buf, off, OC_GGUF_MT_STRING);
      EMIT_U64(buf, off, 8);
      const char *t[] = { "h", "e", "l", "o", "he", "ll", "hell", "hello" };
      for (int i = 0; i < 8; ++i) { uint64_t sl = strlen(t[i]); EMIT_U64(buf, off, sl); EMIT(buf, off, t[i], sl); } }
    /* KV 3: merges = ["h e", "l l", "he ll", "hell o"] */
    { EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.merges");
      EMIT_U32(buf, off, OC_GGUF_MT_ARRAY); EMIT_U32(buf, off, OC_GGUF_MT_STRING);
      EMIT_U64(buf, off, 4);
      const char *m[] = { "h e", "l l", "he ll", "hell o" };
      for (int i = 0; i < 4; ++i) { uint64_t sl = strlen(m[i]); EMIT_U64(buf, off, sl); EMIT(buf, off, m[i], sl); } }

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
    cr_assert_eq(tok.kind, OC_TOK_KIND_TIKTOKEN, "tiktoken -> Tiktoken");

    /* encode("hello") should fully merge to [7]. */
    uint32_t *ids = NULL;
    size_t count = 0;
    e = oc_tokenizer_encode(&tok, "hello", OC_TOK_DEFAULT, &ids, &count);
    cr_assert_eq(e, OC_OK);
    assert_ids_eq(ids, count, (uint32_t[]){7}, 1);

    char *decoded = NULL;
    e = oc_tokenizer_decode(&tok, ids, count, &decoded);
    cr_assert_eq(e, OC_OK);
    cr_assert_str_eq(decoded, "hello");

    free(ids);
    free(decoded);
    oc_tokenizer_free(&tok);
    oc_gguf_free(&gguf);
    free(buf);
}

/* ─── Dispatch (VAL-TOK-010) ───────────────────────────────────────────── */

Test(tokenizer_tiktoken, dispatch_by_model_key_tiktoken)
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
      const char *v = "tiktoken"; EMIT_U64(buf, off, strlen(v)); EMIT(buf, off, v, strlen(v)); }
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
    cr_assert_eq(tok.kind, OC_TOK_KIND_TIKTOKEN);

    oc_tokenizer_free(&tok);
    oc_gguf_free(&gguf);
    free(buf);
}
