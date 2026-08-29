/* test_tokenizer_sp.c — Criterion tests for the SentencePiece unigram VAL-TOK-006 — SentencePiece unigram round-trip (Llama fixture) VAL-TOK-007 — BOS token handling (prepend BOS via OC_TOK_ADD_BOS) */

#define _POSIX_C_SOURCE 200809L  /* mkstemp */

#include <criterion/criterion.h>

#include "oxidize/tokenizer.h"
#include "oxidize/arena.h"
#include "oxidize/gguf.h"
#include "oxidize/error.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>   /* mkstemp, unlink */


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


Test(tokenizer_sp, prefers_higher_probability_path)
{
    /* Rust: */
    OcArena *arena = oc_arena_new(0);
    cr_assert_not_null(arena);

    OcSpPiece pieces[] = {
        { "a",   -3.0f },
        { "b",   -3.0f },
        { "ab",  -0.5f },
        { "aba", -0.1f },
    };
    OcSentencePieceTokenizer *sp = NULL;
    OcError e = oc_sp_new(pieces, 4, arena, &sp);
    cr_assert_eq(e, OC_OK, "new: %s", oc_error_msg(e));

    uint32_t *ids = NULL;
    size_t count = 0;
    e = oc_sp_encode(sp, "aba", &ids, &count);
    cr_assert_eq(e, OC_OK, "encode: %s", oc_error_msg(e));
    cr_assert_eq(count, 1, "should produce 1 token (the 'aba' piece)");

    char *decoded = NULL;
    e = oc_sp_decode(sp, ids, count, &decoded);
    cr_assert_eq(e, OC_OK, "decode: %s", oc_error_msg(e));
    cr_assert_str_eq(decoded, "aba", "decode should give 'aba'");

    free(ids);
    free(decoded);
    oc_sp_free(sp);
    oc_arena_free(arena);
}

/* ─── Unknown token fallback (VAL-TOK-006 OOV) ──────────────────────────
 * Mirrors Rust `sentencepiece_uses_unknown_for_unmatched_text`. */

Test(tokenizer_sp, unknown_for_unmatched_text)
{
    /* Rust: */
    OcArena *arena = oc_arena_new(0);
    OcSpPiece pieces[] = {
        { "a", -0.5f },
        { "b", -0.5f },
    };
    OcSentencePieceTokenizer *sp = NULL;
    OcError e = oc_sp_new(pieces, 2, arena, &sp);
    cr_assert_eq(e, OC_OK);
    e = oc_sp_with_unknown_token(sp, arena, "<unk>");
    cr_assert_eq(e, OC_OK);

    uint32_t *ids = NULL;
    size_t count = 0;
    e = oc_sp_encode(sp, "abz", &ids, &count);
    cr_assert_eq(e, OC_OK, "encode: %s", oc_error_msg(e));

    char *decoded = NULL;
    e = oc_sp_decode(sp, ids, count, &decoded);
    cr_assert_eq(e, OC_OK, "decode: %s", oc_error_msg(e));
    cr_assert_str_eq(decoded, "ab<unk>",
                     "decode should give 'ab<unk>', got '%s'", decoded);

    free(ids);
    free(decoded);
    oc_sp_free(sp);
    oc_arena_free(arena);
}

/* ─── Round-trip known pieces (VAL-TOK-006) ─────────────────────────────
 * Mirrors Rust `sentencepiece_round_trips_known_pieces`. */

Test(tokenizer_sp, round_trip_known_pieces)
{
    /* Rust: */
    OcArena *arena = oc_arena_new(0);
    OcSpPiece pieces[] = {
        { "hello", -0.2f },
        { " ",     -0.1f },
        { "world", -0.2f },
        { "hell",  -1.5f },
        { "o",     -1.0f },
    };
    OcSentencePieceTokenizer *sp = NULL;
    OcError e = oc_sp_new(pieces, 5, arena, &sp);
    cr_assert_eq(e, OC_OK);

    uint32_t *ids = NULL;
    size_t count = 0;
    e = oc_sp_encode(sp, "hello world", &ids, &count);
    cr_assert_eq(e, OC_OK);

    char *decoded = NULL;
    e = oc_sp_decode(sp, ids, count, &decoded);
    cr_assert_eq(e, OC_OK);
    cr_assert_str_eq(decoded, "hello world",
                     "round-trip should preserve 'hello world'");

    /* The best path is ["hello", " ", "world"] (3 tokens), not ["hell",
     * "o", " ", "world"] (4 tokens). */
    cr_assert_eq(count, 3, "best path should be 3 tokens, got %zu", count);
    cr_assert_eq(ids[0], 0, "first token should be 'hello' (id 0)");
    cr_assert_eq(ids[1], 1, "second token should be ' ' (id 1)");
    cr_assert_eq(ids[2], 2, "third token should be 'world' (id 2)");

    free(ids);
    free(decoded);
    oc_sp_free(sp);
    oc_arena_free(arena);
}

/* ─── Multilingual round-trip (VAL-TOK-006 Llama fixture) ───────────────
 * SentencePiece models are used by Llama/Gemma for multilingual text.
 * We build a small multilingual vocab and verify round-trip. */

Test(tokenizer_sp, round_trip_multilingual)
{
    OcArena *arena = oc_arena_new(0);
    /* A small multilingual vocab with English, French, Chinese, Arabic,
     * Russian pieces plus space and <unk>. */
    OcSpPiece pieces[] = {
        { "hello",   -0.2f },
        { "world",   -0.2f },
        { "bonjour", -0.3f },
        { "monde",   -0.3f },
        { "你好",    -0.3f },
        { "世界",    -0.3f },
        { "مرحبا",   -0.3f },
        { "привет",  -0.3f },
        { "мир",     -0.3f },
        { " ",       -0.1f },
        { "<unk>",   -99.0f },
    };
    OcSentencePieceTokenizer *sp = NULL;
    OcError e = oc_sp_new(pieces, 11, arena, &sp);
    cr_assert_eq(e, OC_OK);
    e = oc_sp_with_unknown_token(sp, arena, "<unk>");
    cr_assert_eq(e, OC_OK);

    const char *test_strings[] = {
        "hello world",
        "bonjour monde",
        "你好 世界",
        "привет мир",
    };
    for (size_t i = 0; i < sizeof(test_strings)/sizeof(test_strings[0]); ++i) {
        uint32_t *ids = NULL;
        size_t count = 0;
        e = oc_sp_encode(sp, test_strings[i], &ids, &count);
        cr_assert_eq(e, OC_OK, "encode[%zu]: %s", i, oc_error_msg(e));
        cr_assert_gt(count, 0, "should produce tokens for '%s'", test_strings[i]);

        char *decoded = NULL;
        e = oc_sp_decode(sp, ids, count, &decoded);
        cr_assert_eq(e, OC_OK, "decode[%zu]: %s", i, oc_error_msg(e));
        cr_assert_str_eq(decoded, test_strings[i],
                         "round-trip[%zu]: expected '%s', got '%s'",
                         i, test_strings[i], decoded);
        free(ids);
        free(decoded);
    }

    oc_sp_free(sp);
    oc_arena_free(arena);
}


Test(tokenizer_sp, decode_unknown_id_returns_error)
{
    OcArena *arena = oc_arena_new(0);
    OcSpPiece pieces[] = { { "a", -0.5f } };
    OcSentencePieceTokenizer *sp = NULL;
    OcError e = oc_sp_new(pieces, 1, arena, &sp);
    cr_assert_eq(e, OC_OK);

    char *decoded = NULL;
    uint32_t bad_ids[] = { 999 };
    e = oc_sp_decode(sp, bad_ids, 1, &decoded);
    cr_assert_eq(e, OC_ERR_TOKENIZER,
                 "unknown id should return OC_ERR_TOKENIZER");

    oc_sp_free(sp);
    oc_arena_free(arena);
}


Test(tokenizer_sp, empty_input_returns_empty)
{
    OcArena *arena = oc_arena_new(0);
    OcSpPiece pieces[] = { { "a", -0.5f } };
    OcSentencePieceTokenizer *sp = NULL;
    OcError e = oc_sp_new(pieces, 1, arena, &sp);
    cr_assert_eq(e, OC_OK);

    uint32_t *ids = NULL;
    size_t count = 1;
    e = oc_sp_encode(sp, "", &ids, &count);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(count, 0, "empty input should produce 0 tokens");

    char *decoded = NULL;
    e = oc_sp_decode(sp, NULL, 0, &decoded);
    cr_assert_eq(e, OC_OK);
    cr_assert_str_eq(decoded, "", "decode of empty ids should give empty");

    free(ids);
    free(decoded);
    oc_sp_free(sp);
    oc_arena_free(arena);
}

/* ─── BOS token handling (VAL-TOK-007) ─────────────────────────────────── */

Test(tokenizer_sp, add_bos_default_true_for_sp)
{
    /* Rust: `add_bos_default()` returns true for SentencePiece when
     * `add_bos_token` metadata is absent. */
    OcTokenizer tok;
    memset(&tok, 0, sizeof(tok));
    tok.kind = OC_TOK_KIND_SENTENCEPIECE;
    cr_assert(oc_tokenizer_add_bos_default(&tok),
              "SP should default to add_bos=true");

    tok.kind = OC_TOK_KIND_BPE;
    cr_assert_not(oc_tokenizer_add_bos_default(&tok),
                   "BPE should default to add_bos=false");

    /* When metadata explicitly sets add_bos_token, honor it. */
    tok.kind = OC_TOK_KIND_SENTENCEPIECE;
    tok.has_add_bos_token = true;
    tok.add_bos_token = false;
    cr_assert_not(oc_tokenizer_add_bos_default(&tok),
                   "explicit add_bos_token=false should override default");

    tok.add_bos_token = true;
    cr_assert(oc_tokenizer_add_bos_default(&tok),
              "explicit add_bos_token=true should be honored");
}


/* Build a synthetic GGUF buffer for a SentencePiece tokenizer. */
static uint8_t *build_sp_gguf(const char *model_str, size_t *out_len)
{
    /* Tokens: ["he", "llo", "hello", "<unk>"]
     * Scores: [-1.0, -1.0, -0.1, -99.0]
     * unknown_token_id = 3 */
    static const char *toks[] = { "he", "llo", "hello", "<unk>" };
    static const float scores[] = { -1.0f, -1.0f, -0.1f, -99.0f };

    size_t cap = 4096;
    uint8_t *buf = calloc(cap, 1);
    cr_assert_not_null(buf);
    size_t off = 0;

#define EMIT(buf, off, src, n) do { memcpy((buf) + (off), (src), (n)); (off) += (n); } while (0)
#define EMIT_U32(buf, off, v) do { uint32_t _x = (uint32_t)(v); EMIT(buf, off, &_x, 4); } while (0)
#define EMIT_U64(buf, off, v) do { uint64_t _x = (uint64_t)(v); EMIT(buf, off, &_x, 8); } while (0)
#define EMIT_F32(buf, off, v) do { float _x = (float)(v); EMIT(buf, off, &_x, 4); } while (0)
#define EMIT_KV_STR_KEY(buf, off, key_str) do { \
        const char *k = (key_str); \
        uint64_t kl = strlen(k); \
        EMIT_U64(buf, off, kl); \
        EMIT(buf, off, k, kl); \
    } while (0)

    EMIT_U32(buf, off, OC_GGUF_MAGIC);
    EMIT_U32(buf, off, 3);
    EMIT_U64(buf, off, 0);   /* tensor_count */
    EMIT_U64(buf, off, 4);   /* kv_count */

    /* KV 1: tokenizer.ggml.model = model_str */
    {
        EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.model");
        EMIT_U32(buf, off, OC_GGUF_MT_STRING);
        uint64_t sl = strlen(model_str);
        EMIT_U64(buf, off, sl);
        EMIT(buf, off, model_str, sl);
    }
    /* KV 2: tokenizer.ggml.tokens = ARRAY<STRING>[4] */
    {
        EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.tokens");
        EMIT_U32(buf, off, OC_GGUF_MT_ARRAY);
        EMIT_U32(buf, off, OC_GGUF_MT_STRING);
        EMIT_U64(buf, off, 4);
        for (int i = 0; i < 4; ++i) {
            uint64_t sl = strlen(toks[i]);
            EMIT_U64(buf, off, sl);
            EMIT(buf, off, toks[i], sl);
        }
    }
    /* KV 3: tokenizer.ggml.scores = ARRAY<FLOAT32>[4] */
    {
        EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.scores");
        EMIT_U32(buf, off, OC_GGUF_MT_ARRAY);
        EMIT_U32(buf, off, OC_GGUF_MT_FLOAT32);
        EMIT_U64(buf, off, 4);
        for (int i = 0; i < 4; ++i) {
            EMIT_F32(buf, off, scores[i]);
        }
    }
    /* KV 4: tokenizer.ggml.unknown_token_id = 3 (UINT32) */
    {
        EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.unknown_token_id");
        EMIT_U32(buf, off, OC_GGUF_MT_UINT32);
        EMIT_U32(buf, off, 3);
    }

#undef EMIT
#undef EMIT_U32
#undef EMIT_U64
#undef EMIT_F32
#undef EMIT_KV_STR_KEY

    size_t aligned = (off + 31) & ~(size_t)31;
    if (aligned > cap) aligned = cap;
    *out_len = aligned;
    return buf;
}

Test(tokenizer_sp, load_from_gguf_llama)
{
    /* Mirrors Rust `loads_sentencepiece_tokenizer_from_gguf_metadata`: */
    size_t len = 0;
    uint8_t *buf = build_sp_gguf("llama", &len);
    OcGgufFile gguf;
    OcError e = oc_gguf_parse(buf, len, &gguf);
    cr_assert_eq(e, OC_OK, "parse: %s", oc_error_msg(e));

    OcTokenizer tok;
    e = oc_tokenizer_load_from_gguf(&gguf, &tok);
    cr_assert_eq(e, OC_OK, "load: %s", oc_error_msg(e));
    cr_assert_eq(tok.kind, OC_TOK_KIND_SENTENCEPIECE, "llama -> SP");
    cr_assert(tok.has_unknown, "should have unknown id");
    cr_assert_eq(tok.unknown_id, 3, "unknown id should be 3");

    /* encode("hello") -> [2] (the "hello" piece, id 2, score -0.1) */
    uint32_t *ids = NULL;
    size_t count = 0;
    e = oc_tokenizer_encode(&tok, "hello", OC_TOK_DEFAULT, &ids, &count);
    cr_assert_eq(e, OC_OK);
    static const uint32_t expected[] = { 2 };
    assert_ids_eq(ids, count, expected, 1);

    char *decoded = NULL;
    e = oc_tokenizer_decode(&tok, ids, count, &decoded);
    cr_assert_eq(e, OC_OK);
    cr_assert_str_eq(decoded, "hello");
    free(ids);
    free(decoded);

    /* encode("x") -> [3] (the <unk> id, since 'x' is not in vocab) */
    e = oc_tokenizer_encode(&tok, "x", OC_TOK_DEFAULT, &ids, &count);
    cr_assert_eq(e, OC_OK);
    assert_ids_eq(ids, count, (uint32_t[]){3}, 1);
    e = oc_tokenizer_decode(&tok, ids, count, &decoded);
    cr_assert_eq(e, OC_OK);
    cr_assert_str_eq(decoded, "<unk>");
    free(ids);
    free(decoded);

    oc_tokenizer_free(&tok);
    oc_gguf_free(&gguf);
    free(buf);
}

Test(tokenizer_sp, load_from_gguf_gemma4)
{
    /* Mirrors Rust `loads_gemma4_sentencepiece_tokenizer_from_gguf_metadata`. */
    size_t len = 0;
    uint8_t *buf = build_sp_gguf("gemma4", &len);
    OcGgufFile gguf;
    OcError e = oc_gguf_parse(buf, len, &gguf);
    cr_assert_eq(e, OC_OK);

    OcTokenizer tok;
    e = oc_tokenizer_load_from_gguf(&gguf, &tok);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(tok.kind, OC_TOK_KIND_SENTENCEPIECE, "gemma4 -> SP");

    uint32_t *ids = NULL;
    size_t count = 0;
    e = oc_tokenizer_encode(&tok, "hello", OC_TOK_DEFAULT, &ids, &count);
    cr_assert_eq(e, OC_OK);

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

Test(tokenizer_sp, load_from_gguf_with_bos_metadata)
{
    /* Build a SP GGUF with bos_token_id metadata and verify
     * oc_tokenizer_add_bos_default honors the GGUF flag. */
    size_t cap = 4096;
    uint8_t *buf = calloc(cap, 1);
    cr_assert_not_null(buf);
    size_t off = 0;
#define EMIT(buf, off, src, n) do { memcpy((buf) + (off), (src), (n)); (off) += (n); } while (0)
#define EMIT_U32(buf, off, v) do { uint32_t _x = (uint32_t)(v); EMIT(buf, off, &_x, 4); } while (0)
#define EMIT_U64(buf, off, v) do { uint64_t _x = (uint64_t)(v); EMIT(buf, off, &_x, 8); } while (0)
#define EMIT_F32(buf, off, v) do { float _x = (float)(v); EMIT(buf, off, &_x, 4); } while (0)
#define EMIT_KV_STR_KEY(buf, off, key_str) do { \
        const char *k = (key_str); \
        uint64_t kl = strlen(k); \
        EMIT_U64(buf, off, kl); \
        EMIT(buf, off, k, kl); \
    } while (0)

    EMIT_U32(buf, off, OC_GGUF_MAGIC);
    EMIT_U32(buf, off, 3);
    EMIT_U64(buf, off, 0);
    EMIT_U64(buf, off, 6);   /* kv_count = 6 */

    /* KV 1: model = "llama" */
    { EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.model");
      EMIT_U32(buf, off, OC_GGUF_MT_STRING);
      const char *v = "llama"; EMIT_U64(buf, off, strlen(v)); EMIT(buf, off, v, strlen(v)); }
    /* KV 2: tokens = ["a", "b", "<unk>", "<s>", "</s>", "<pad>"] */
    { EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.tokens");
      EMIT_U32(buf, off, OC_GGUF_MT_ARRAY); EMIT_U32(buf, off, OC_GGUF_MT_STRING);
      EMIT_U64(buf, off, 6);
      const char *t[] = { "a", "b", "<unk>", "<s>", "</s>", "<pad>" };
      for (int i = 0; i < 6; ++i) { uint64_t sl = strlen(t[i]); EMIT_U64(buf, off, sl); EMIT(buf, off, t[i], sl); } }
    /* KV 3: scores */
    { EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.scores");
      EMIT_U32(buf, off, OC_GGUF_MT_ARRAY); EMIT_U32(buf, off, OC_GGUF_MT_FLOAT32);
      EMIT_U64(buf, off, 6);
      float s[] = { -1, -1, -99, -99, -99, -99 };
      for (int i = 0; i < 6; ++i) EMIT_F32(buf, off, s[i]); }
    /* KV 4-6: unknown/bos/eos/pad ids */
    { EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.unknown_token_id");
      EMIT_U32(buf, off, OC_GGUF_MT_UINT32); EMIT_U32(buf, off, 2); }
    { EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.bos_token_id");
      EMIT_U32(buf, off, OC_GGUF_MT_UINT32); EMIT_U32(buf, off, 3); }
    { EMIT_KV_STR_KEY(buf, off, "tokenizer.ggml.add_bos_token");
      EMIT_U32(buf, off, OC_GGUF_MT_BOOL); uint8_t b = 1; EMIT(buf, off, &b, 1); }

#undef EMIT
#undef EMIT_U32
#undef EMIT_U64
#undef EMIT_F32
#undef EMIT_KV_STR_KEY

    OcGgufFile gguf;
    size_t padded_len = (off + 31) & ~(size_t)31;
    OcError e = oc_gguf_parse(buf, padded_len, &gguf);
    cr_assert_eq(e, OC_OK);

    OcTokenizer tok;
    e = oc_tokenizer_load_from_gguf(&gguf, &tok);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(tok.kind, OC_TOK_KIND_SENTENCEPIECE);
    cr_assert(tok.has_bos, "should have bos");
    cr_assert_eq(tok.bos_id, 3, "bos id should be 3");
    cr_assert(tok.has_add_bos_token, "add_bos_token flag should be set");
    cr_assert(oc_tokenizer_add_bos_default(&tok),
              "explicit add_bos_token=true should be honored");

    /* Encode with ADD_BOS: first id should be 3. */
    uint32_t *ids = NULL;
    size_t count = 0;
    e = oc_tokenizer_encode(&tok, "ab", OC_TOK_ADD_BOS, &ids, &count);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(ids[0], 3, "BOS id (3) should be prepended");

    free(ids);
    oc_tokenizer_free(&tok);
    oc_gguf_free(&gguf);
    free(buf);
}

/* ─── Dispatch by tokenizer.ggml.model key (VAL-TOK-010) ──────────────── */

Test(tokenizer_sp, dispatch_by_model_key_llama)
{
    size_t len = 0;
    uint8_t *buf = build_sp_gguf("llama", &len);
    OcGgufFile gguf;
    OcError e = oc_gguf_parse(buf, len, &gguf);
    cr_assert_eq(e, OC_OK);

    OcTokenizer tok;
    e = oc_tokenizer_load_from_gguf(&gguf, &tok);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(tok.kind, OC_TOK_KIND_SENTENCEPIECE, "llama -> SP");

    oc_tokenizer_free(&tok);
    oc_gguf_free(&gguf);
    free(buf);
}

Test(tokenizer_sp, dispatch_by_model_key_gemma)
{
    size_t len = 0;
    uint8_t *buf = build_sp_gguf("gemma", &len);
    OcGgufFile gguf;
    OcError e = oc_gguf_parse(buf, len, &gguf);
    cr_assert_eq(e, OC_OK);

    OcTokenizer tok;
    e = oc_tokenizer_load_from_gguf(&gguf, &tok);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(tok.kind, OC_TOK_KIND_SENTENCEPIECE, "gemma -> SP");

    oc_tokenizer_free(&tok);
    oc_gguf_free(&gguf);
    free(buf);
}
