/* test_tokenizer_k2.c — K2-Horizon tokenizer path (tokenizer_bpe.c).
 *
 * A tiny synthetic GGUF (byte vocab + a few merges + K2 special tokens) is
 * built in memory. It checks that `tokenizer.ggml.pre = "k2-horizon"`
 * switches on the pre-tokenized, per-word merge path (digits group in
 * threes, so "1234" merges as 123|4 even when "3 4" is the best merge),
 * that special pieces are atomic, the K2 chat template, and the extra EOG
 * id. Full-vocab parity with llama.cpp-k2 lives in
 * scripts/tokenizer_parity.py. */

#include <criterion/criterion.h>

#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/tokenizer.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── In-memory GGUF builder ───────────────────────────────────────────── */

typedef struct {
    uint8_t *p;
    size_t   n;
    size_t   cap;
} Buf;

static void put(Buf *b, const void *src, size_t n)
{
    if (b->n + n > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->n + n) cap *= 2;
        b->p = realloc(b->p, cap);
        cr_assert_not_null(b->p);
        b->cap = cap;
    }
    memcpy(b->p + b->n, src, n);
    b->n += n;
}
static void put_u32(Buf *b, uint32_t v) { put(b, &v, 4); }
static void put_u64(Buf *b, uint64_t v) { put(b, &v, 8); }
static void put_str(Buf *b, const char *s)
{
    put_u64(b, strlen(s));
    put(b, s, strlen(s));
}
static void kv_str(Buf *b, const char *k, const char *v)
{
    put_str(b, k);
    put_u32(b, OC_GGUF_MT_STRING);
    put_str(b, v);
}
static void kv_u32(Buf *b, const char *k, uint32_t v)
{
    put_str(b, k);
    put_u32(b, OC_GGUF_MT_UINT32);
    put_u32(b, v);
}

/* GPT-2 byte → unicode code point. */
static uint32_t gpt2_cp(uint8_t c)
{
    if ((c >= 33 && c <= 126) || (c >= 161 && c <= 172) || c >= 174) return c;
    uint32_t n = 0;
    for (uint32_t b = 0; b < c; ++b)
        if (!((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174)) n++;
    return 256 + n;
}

/* GPT-2-encode an ASCII/UTF-8 string into `out`. */
static void gpt2_str(const char *s, char *out)
{
    size_t o = 0;
    for (const uint8_t *p = (const uint8_t *)s; *p; ++p) {
        uint32_t cp = gpt2_cp(*p);
        if (cp < 0x80) out[o++] = (char)cp;
        else {
            out[o++] = (char)(0xC0 | (cp >> 6));
            out[o++] = (char)(0x80 | (cp & 0x3F));
        }
    }
    out[o] = '\0';
}

/* Vocab: 0..4 specials, 5..260 byte tokens, then merged tokens. */
enum { ID_BOS = 0, ID_EOS = 1, ID_IM_START = 2, ID_IM_END = 3, ID_THINK = 4,
       ID_BYTE0 = 5 };
static const char *const k_specials[] = {
    "<|ifm|begin_of_text|>", "<|ifm|endoftext|>", "<|ifm|im_start|>",
    "<|ifm|im_end|>", "<ifm|think>",
};
static const int32_t k_special_types[] = { 3, 3, 3, 3, 4 };
/* Merges in rank order; each produces the next id after the byte tokens.
 * "3 4" outranks "12 3" so whole-string merging would give 12|34. */
static const char *const k_merges[][2] = {
    { "1", "2" }, { "3", "4" }, { "12", "3" }, { "H", "i" }, { "'", "s" },
    { " ", "a" },
};
#define N_MERGES (sizeof(k_merges) / sizeof(k_merges[0]))
#define ID_MERGE(i) (ID_BYTE0 + 256 + (uint32_t)(i))

static uint32_t byte_id(uint8_t c) { return ID_BYTE0 + c; }

static uint8_t *build_k2_gguf(bool with_pre, size_t *out_len)
{
    Buf b = { 0 };
    const uint64_t n_tok = 5 + 256 + N_MERGES;
    put_u32(&b, OC_GGUF_MAGIC);
    put_u32(&b, 3);
    put_u64(&b, 0);
    put_u64(&b, with_pre ? 8 : 7);

    kv_str(&b, "tokenizer.ggml.model", "gpt2");
    if (with_pre) kv_str(&b, "tokenizer.ggml.pre", "k2-horizon");
    kv_u32(&b, "tokenizer.ggml.bos_token_id", ID_BOS);
    kv_u32(&b, "tokenizer.ggml.eos_token_id", ID_EOS);
    kv_str(&b, "tokenizer.chat_template",
           "{{- bos_token }}{{ '<|ifm|im_start|>' + message.role }}");

    char tmp[64], l[16], r[16];
    put_str(&b, "tokenizer.ggml.tokens");
    put_u32(&b, OC_GGUF_MT_ARRAY);
    put_u32(&b, OC_GGUF_MT_STRING);
    put_u64(&b, n_tok);
    for (size_t i = 0; i < 5; ++i) put_str(&b, k_specials[i]);
    for (uint32_t c = 0; c < 256; ++c) {
        char one[2] = { (char)c, 0 };
        if (c == 0) {
            /* gpt2_str stops at NUL; encode byte 0 directly (U+0100). */
            strcpy(tmp, "\xC4\x80");
        } else {
            gpt2_str(one, tmp);
        }
        put_str(&b, tmp);
    }
    for (size_t i = 0; i < N_MERGES; ++i) {
        char both[16];
        snprintf(both, sizeof(both), "%s%s", k_merges[i][0], k_merges[i][1]);
        gpt2_str(both, tmp);
        put_str(&b, tmp);
    }

    put_str(&b, "tokenizer.ggml.token_type");
    put_u32(&b, OC_GGUF_MT_ARRAY);
    put_u32(&b, OC_GGUF_MT_INT32);
    put_u64(&b, n_tok);
    for (uint64_t i = 0; i < n_tok; ++i) {
        int32_t t = i < 5 ? k_special_types[i] : 1;
        put(&b, &t, 4);
    }

    put_str(&b, "tokenizer.ggml.merges");
    put_u32(&b, OC_GGUF_MT_ARRAY);
    put_u32(&b, OC_GGUF_MT_STRING);
    put_u64(&b, N_MERGES);
    for (size_t i = 0; i < N_MERGES; ++i) {
        gpt2_str(k_merges[i][0], l);
        gpt2_str(k_merges[i][1], r);
        snprintf(tmp, sizeof(tmp), "%s %s", l, r);
        put_str(&b, tmp);
    }

    size_t pad = ((b.n + 31) & ~(size_t)31) - b.n;
    uint8_t zeros[32] = { 0 };
    put(&b, zeros, pad);
    *out_len = b.n;
    return b.p;
}

typedef struct {
    uint8_t    *buf;
    OcGgufFile  gguf;
    OcTokenizer tok;
} Fixture;

static void fixture_open(Fixture *f, bool with_pre)
{
    size_t len;
    f->buf = build_k2_gguf(with_pre, &len);
    cr_assert_eq(oc_gguf_parse(f->buf, len, &f->gguf), OC_OK);
    OcError e = oc_tokenizer_load_from_gguf(&f->gguf, &f->tok);
    cr_assert_eq(e, OC_OK, "load: %s", oc_error_msg(e));
}

static void fixture_close(Fixture *f)
{
    oc_tokenizer_free(&f->tok);
    oc_gguf_free(&f->gguf);
    free(f->buf);
}

static void expect_ids(Fixture *f, const char *text, OcSpecialTokenPolicy pol,
                       const uint32_t *want, size_t n_want)
{
    uint32_t *ids = NULL;
    size_t n = 0;
    OcError e = oc_tokenizer_encode(&f->tok, text, pol, &ids, &n);
    cr_assert_eq(e, OC_OK, "encode '%s': %s", text, oc_error_msg(e));
    cr_assert_eq(n, n_want, "'%s': %zu ids, want %zu", text, n, n_want);
    for (size_t i = 0; i < n; ++i)
        cr_assert_eq(ids[i], want[i], "'%s' id[%zu] = %u, want %u", text, i,
                     ids[i], want[i]);
    free(ids);
}

#define EXPECT(f, text, pol, ...) do {                                  \
        const uint32_t w_[] = { __VA_ARGS__ };                          \
        expect_ids((f), (text), (pol), w_, sizeof(w_) / sizeof(w_[0])); \
    } while (0)

/* ─── Tests ────────────────────────────────────────────────────────────── */

Test(tokenizer_k2, digits_merge_per_pretoken)
{
    Fixture f;
    fixture_open(&f, true);
    /* Pre-tokens "123" | "4": 1 2 → 12, 12 3 → 123; "4" alone. */
    EXPECT(&f, "1234", OC_TOK_DEFAULT, ID_MERGE(2), byte_id('4'));
    fixture_close(&f);

    /* Without tokenizer.ggml.pre the legacy whole-string path is kept:
     * "3 4" (rank 1) beats "12 3" (rank 2) → 12 | 34. */
    fixture_open(&f, false);
    EXPECT(&f, "1234", OC_TOK_DEFAULT, ID_MERGE(0), ID_MERGE(1));
    fixture_close(&f);
}

Test(tokenizer_k2, contraction_and_space_prefix)
{
    Fixture f;
    fixture_open(&f, true);
    EXPECT(&f, "Hi's a", OC_TOK_DEFAULT, ID_MERGE(3), ID_MERGE(4), ID_MERGE(5));
    fixture_close(&f);
}

Test(tokenizer_k2, specials_are_atomic)
{
    Fixture f;
    fixture_open(&f, true);
    EXPECT(&f, "<|ifm|im_start|>Hi<|ifm|im_end|>", OC_TOK_DEFAULT,
           ID_IM_START, ID_MERGE(3), ID_IM_END);
    /* BOS policy prepends id 0. */
    EXPECT(&f, "Hi", OC_TOK_ADD_BOS, ID_BOS, ID_MERGE(3));
    /* USER_DEFINED pieces match even when CONTROL parsing is off. */
    EXPECT(&f, "<ifm|think>", OC_TOK_DISALLOW_SPECIAL, ID_THINK);

    /* CONTROL text is plain bytes with specials disallowed:
     * pre-tokens "<|" "ifm" "|im" "_end" "|>", none of which merge. */
    uint32_t *ids = NULL;
    size_t n = 0;
    cr_assert_eq(oc_tokenizer_encode(&f.tok, "<|ifm|im_end|>",
                                     OC_TOK_DISALLOW_SPECIAL, &ids, &n), OC_OK);
    cr_assert_eq(n, strlen("<|ifm|im_end|>"));
    for (size_t i = 0; i < n; ++i)
        cr_assert_eq(ids[i], byte_id((uint8_t)"<|ifm|im_end|>"[i]));
    free(ids);
    fixture_close(&f);
}

Test(tokenizer_k2, long_input_is_consistent)
{
    Fixture f;
    fixture_open(&f, true);
    /* "1234 " x N → per unit: 123 | 4 | " " (word cache hit after the
     * first unit). */
    const size_t units = 20000;
    char *text = malloc(units * 5 + 1);
    cr_assert_not_null(text);
    for (size_t i = 0; i < units; ++i) memcpy(text + i * 5, "1234 ", 5);
    text[units * 5] = '\0';
    uint32_t *ids = NULL;
    size_t n = 0;
    cr_assert_eq(oc_tokenizer_encode(&f.tok, text, OC_TOK_DEFAULT, &ids, &n), OC_OK);
    cr_assert_eq(n, units * 3);
    for (size_t i = 0; i < units; ++i) {
        cr_assert_eq(ids[3 * i], ID_MERGE(2));
        cr_assert_eq(ids[3 * i + 1], byte_id('4'));
        cr_assert_eq(ids[3 * i + 2], byte_id(' '));
    }
    free(ids);
    free(text);
    fixture_close(&f);
}

Test(tokenizer_k2, eog_includes_im_end)
{
    Fixture f;
    fixture_open(&f, true);
    cr_assert(oc_tokenizer_is_eog(&f.tok, ID_EOS));
    cr_assert(oc_tokenizer_is_eog(&f.tok, ID_IM_END));
    cr_assert_not(oc_tokenizer_is_eog(&f.tok, ID_IM_START));
    cr_assert_not(oc_tokenizer_is_eog(&f.tok, byte_id('a')));
    fixture_close(&f);
}

Test(tokenizer_k2, template_detect_and_render)
{
    Fixture f;
    fixture_open(&f, true);
    cr_assert_eq(oc_tokenizer_detect_template(&f.gguf), OC_TEMPLATE_K2);
    cr_assert_eq(oc_tokenizer_detect_template(NULL), OC_TEMPLATE_CHATML);
    fixture_close(&f);

    OcChatMessage msgs[] = {
        { "system", "Be brief." },
        { "user", "Hi" },
        { "assistant", "<ifm|think>\nplan\n</ifm|think>\nHello" },
        { "user", "Again" },
    };
    char *out = NULL;
    cr_assert_eq(oc_tokenizer_apply_chat_template(msgs, 4, OC_TEMPLATE_K2, true, &out),
                 OC_OK);
    cr_assert_str_eq(out,
        "<|ifm|im_start|>system\nBe brief.<|ifm|im_end|>"
        "<|ifm|im_start|>user\nHi<|ifm|im_end|>"
        "<|ifm|im_start|>assistant\n<ifm|think>\nplan\n</ifm|think>\nHello<|ifm|im_end|>"
        "<|ifm|im_start|>user\nAgain<|ifm|im_end|>"
        "<|ifm|im_start|>assistant\n<ifm|think>\n");
    free(out);

    cr_assert_eq(oc_tokenizer_apply_chat_template(msgs + 1, 1, OC_TEMPLATE_K2_NO_THINK,
                                                  true, &out), OC_OK);
    cr_assert_str_eq(out,
        "<|ifm|im_start|>user\nHi<|ifm|im_end|>"
        "<|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>\n");
    free(out);

    /* Plain assistant content gets an empty think block. */
    OcChatMessage plain = { "assistant", "Sure." };
    cr_assert_eq(oc_tokenizer_apply_chat_template(&plain, 1, OC_TEMPLATE_K2, false, &out),
                 OC_OK);
    cr_assert_str_eq(out,
        "<|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>\nSure.<|ifm|im_end|>");
    free(out);
}
