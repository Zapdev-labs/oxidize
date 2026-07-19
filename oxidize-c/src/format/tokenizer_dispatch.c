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

