/* tokenizer_dispatch.c — dispatch tokenizer loading by `tokenizer.ggml.model`.
 *
 * Port of oxidize-core/src/format/tokenizer.rs::load_tokenizer_from_gguf_metadata.
 *
 *   "gpt2" | "lfm2" | "lfm2moe"  → BPE (byte-level, tiktoken-style)
 *   "llama" | "gemma" | "gemma4" → SentencePiece (unigram, Viterbi)
 *   "bert"                       → WordPiece (## continuation)
 *   "tiktoken"                   → Tiktoken (raw byte-level)
 *   <other>                      → OC_ERR_TOKENIZER
 *
 * After loading the kind-specific implementation, special-token ids are
 * copied into the `OcTokenizer` wrapper and the optional
 * `tokenizer.ggml.add_bos_token` bool is read (defaults to false when
 * absent; SentencePiece models default to adding BOS per Rust
 * `add_bos_default()`).
 */

#include "oxidize/tokenizer.h"

#include "oxidize/arena.h"
#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/log.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Case-sensitive string equals. The GGUF metadata values for
 * `tokenizer.ggml.model` are lowercase ASCII strings emitted by
 * convert.py / llama.cpp, so case-sensitive comparison matches Rust's
 * `match model.as_str()` exactly. */
static bool str_eq(const char *a, const char *b)
{
    return a && b && strcmp(a, b) == 0;
}

/* Resolve a `tokenizer.ggml.model` string to an OcTokenizerKind. Mirrors
 * Rust `load_tokenizer_from_gguf_metadata`. Returns OC_TOK_KIND_NONE for
 * unrecognized strings. */
static OcTokenizerKind resolve_kind(const char *model)
{
    if (str_eq(model, "gpt2") || str_eq(model, "lfm2") || str_eq(model, "lfm2moe")) {
        return OC_TOK_KIND_BPE;
    }
    if (str_eq(model, "llama") || str_eq(model, "gemma") || str_eq(model, "gemma4")) {
        return OC_TOK_KIND_SENTENCEPIECE;
    }
    if (str_eq(model, "bert")) {
        return OC_TOK_KIND_WORDPIECE;
    }
    if (str_eq(model, "tiktoken")) {
        return OC_TOK_KIND_TIKTOKEN;
    }
    return OC_TOK_KIND_NONE;
}

/* Read `tokenizer.ggml.add_bos_token` (optional bool). When absent, leaves
 * `has_add_bos_token = false` so `oc_tokenizer_add_bos_default()` falls back
 * to the kind-specific default (true for SentencePiece, false otherwise). */
static void load_add_bos_flag(const OcGgufFile *gguf, OcTokenizer *out)
{
    bool add_bos = false;
    if (oc_gguf_metadata_get_bool(gguf, "tokenizer.ggml.add_bos_token", &add_bos)) {
        out->has_add_bos_token = true;
        out->add_bos_token = add_bos;
    }
}

OcError oc_tokenizer_load_from_gguf(const OcGgufFile *gguf, OcTokenizer *out)
{
    if (!gguf || !out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    const char *model = NULL;
    size_t model_len = 0;
    if (!oc_gguf_metadata_get_str(gguf, "tokenizer.ggml.model", &model, &model_len)) {
        oc_log_error("tokenizer: missing required metadata key "
                     "\"tokenizer.ggml.model\"");
        return OC_ERR_TOKENIZER;
    }

    OcTokenizerKind kind = resolve_kind(model);
    if (kind == OC_TOK_KIND_NONE) {
        oc_log_error("tokenizer: unsupported tokenizer.ggml.model=\"%s\"", model);
        return OC_ERR_TOKENIZER;
    }

    /* Create an arena for this tokenizer's lifetime allocations. */
    OcArena *arena = oc_arena_new(0);
    if (!arena) return OC_ERR_OOM;
    out->arena = arena;

    if (kind == OC_TOK_KIND_BPE) {
        OcBpeTokenizer *bpe = NULL;
        OcError e = oc_bpe_load_from_gguf(gguf, arena, &bpe);
        if (e != OC_OK) {
            oc_arena_free(arena);
            out->arena = NULL;
            return e;
        }
        out->kind = OC_TOK_KIND_BPE;
        out->bpe = bpe;
        oc_bpe_fill_special_tokens(bpe, out);
        load_add_bos_flag(gguf, out);
        return OC_OK;
    }

    if (kind == OC_TOK_KIND_SENTENCEPIECE) {
        OcSentencePieceTokenizer *sp = NULL;
        OcError e = oc_sp_load_from_gguf(gguf, arena, &sp);
        if (e != OC_OK) {
            oc_arena_free(arena);
            out->arena = NULL;
            return e;
        }
        out->kind = OC_TOK_KIND_SENTENCEPIECE;
        out->sp = sp;
        oc_sp_fill_special_tokens(sp, out);
        load_add_bos_flag(gguf, out);
        return OC_OK;
    }

    if (kind == OC_TOK_KIND_WORDPIECE) {
        OcWordPieceTokenizer *wp = NULL;
        OcError e = oc_wp_load_from_gguf(gguf, arena, &wp);
        if (e != OC_OK) {
            oc_arena_free(arena);
            out->arena = NULL;
            return e;
        }
        out->kind = OC_TOK_KIND_WORDPIECE;
        out->wp = wp;
        oc_wp_fill_special_tokens(wp, out);
        load_add_bos_flag(gguf, out);
        return OC_OK;
    }

    if (kind == OC_TOK_KIND_TIKTOKEN) {
        OcTiktokenTokenizer *t = NULL;
        OcError e = oc_tiktoken_load_from_gguf(gguf, arena, &t);
        if (e != OC_OK) {
            oc_arena_free(arena);
            out->arena = NULL;
            return e;
        }
        out->kind = OC_TOK_KIND_TIKTOKEN;
        out->tiktoken = t;
        oc_tiktoken_fill_special_tokens(t, out);
        load_add_bos_flag(gguf, out);
        return OC_OK;
    }

    /* Unreachable: resolve_kind returns NONE for unknown strings and we
     * already returned OC_ERR_TOKENIZER above. */
    oc_log_error("tokenizer: kind %d (model=\"%s\") not handled",
                 (int)kind, model);
    oc_arena_free(arena);
    out->arena = NULL;
    return OC_ERR_TOKENIZER;
}

/* oc_tokenizer_free() is implemented in tokenizer_bpe.c (it needs access to
 * the per-kind free functions). */

/* ─── Streaming detokenizer ────────────────────────────────────────────── */

void oc_streaming_detok_init(OcStreamingDetokenizer *sd,
                             const OcTokenizer *tok)
{
    if (!sd) return;
    memset(sd, 0, sizeof(*sd));
    sd->tokenizer = tok;
}

/* Count UTF-8 continuation bytes needed after `first_byte`.
 * Returns the total expected sequence length (1-4), or 1 if not a lead byte. */
static int utf8_seq_len(uint8_t first_byte)
{
    if ((first_byte & 0x80) == 0) return 1;       /* 0xxxxxxx */
    if ((first_byte & 0xE0) == 0xC0) return 2;    /* 110xxxxx */
    if ((first_byte & 0xF0) == 0xE0) return 3;    /* 1110xxxx */
    if ((first_byte & 0xF8) == 0xF0) return 4;    /* 11110xxx */
    return 1; /* invalid lead byte, treat as single */
}

OcError oc_streaming_detok_push(OcStreamingDetokenizer *sd, uint32_t token,
                                const char **out_delta, size_t *out_len)
{
    if (!sd || !out_delta || !out_len) return OC_ERR_INVALID_ARG;
    *out_delta = NULL;
    *out_len = 0;

    /* Decode the token to text. */
    char *text = NULL;
    OcError e = oc_tokenizer_decode(sd->tokenizer, &token, 1, &text);
    if (e != OC_OK || text == NULL) return e;

    size_t text_len = strlen(text);

    /* Combine pending bytes with new text. */
    static _Thread_local uint8_t combined[4096];
    size_t combined_len = sd->pending_len;
    if (combined_len > 0) {
        memcpy(combined, sd->pending, combined_len);
    }
    size_t copy_len = text_len;
    if (combined_len + copy_len > sizeof(combined)) {
        copy_len = sizeof(combined) - combined_len;
    }
    memcpy(combined + combined_len, text, copy_len);
    combined_len += copy_len;
    free(text);

    /* Find the longest complete UTF-8 prefix. */
    size_t emit_len = 0;
    size_t pos = 0;
    while (pos < combined_len) {
        int seq_len = utf8_seq_len(combined[pos]);
        if (pos + (size_t)seq_len > combined_len) break;
        /* Validate continuation bytes. */
        bool valid = true;
        for (int i = 1; i < seq_len; i++) {
            if ((combined[pos + i] & 0xC0) != 0x80) { valid = false; break; }
        }
        if (!valid) {
            /* Invalid byte, skip it. */
            pos++;
            continue;
        }
        pos += (size_t)seq_len;
        emit_len = pos;
    }

    /* Store incomplete tail as pending. */
    sd->pending_len = combined_len - emit_len;
    if (sd->pending_len > 0 && sd->pending_len <= sizeof(sd->pending)) {
        memcpy(sd->pending, combined + emit_len, sd->pending_len);
    } else {
        sd->pending_len = 0;
    }

    /* Emit the complete prefix. */
    if (emit_len > 0) {
        /* Point into combined buffer (caller must use before next push). */
        *out_delta = (const char *)combined;
        *out_len = emit_len;
    }
    return OC_OK;
}

OcError oc_streaming_detok_flush(OcStreamingDetokenizer *sd,
                                 const char **out_delta, size_t *out_len)
{
    if (!sd || !out_delta || !out_len) return OC_ERR_INVALID_ARG;
    *out_delta = NULL;
    *out_len = 0;
    if (sd->pending_len == 0) return OC_OK;
    /* Emit whatever is left, even if incomplete. */
    static _Thread_local uint8_t flush_buf[8];
    memcpy(flush_buf, sd->pending, sd->pending_len);
    *out_delta = (const char *)flush_buf;
    *out_len = sd->pending_len;
    sd->pending_len = 0;
    return OC_OK;
}

void oc_streaming_detok_reset(OcStreamingDetokenizer *sd)
{
    if (!sd) return;
    sd->pending_len = 0;
}

