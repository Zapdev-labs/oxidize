/* tokenizer_dispatch.c — dispatch tokenizer loading by `tokenizer.ggml.model`.
 *
 * Port of oxidize-core/src/format/tokenizer.rs::load_tokenizer_from_gguf_metadata.
 *
 *   "gpt2" | "lfm2" | "lfm2moe"  → BPE (byte-level, tiktoken-style)
 *   "llama" | "gemma" | "gemma4" → SentencePiece (not yet implemented)
 *   "bert"                       → WordPiece (not yet implemented)
 *   "tiktoken"                   → Tiktoken (not yet implemented)
 *   <other>                      → OC_ERR_TOKENIZER
 *
 * Only the BPE path is implemented by the `tokenizer-bpe-qwen` feature;
 * the other branches return OC_ERR_TOKENIZER with a clear message so callers
 * can detect the missing implementation without a crash.
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
        /* Copy special-token ids into the wrapper. */
        oc_bpe_fill_special_tokens(bpe, out);
        /* BPE models do not add BOS by default (Rust `add_bos_default()`
         * returns false for BPE unless `tokenizer.ggml.add_bos_token` is
         * explicitly set). */
        bool add_bos = false;
        if (oc_gguf_metadata_get_bool(gguf, "tokenizer.ggml.add_bos_token", &add_bos)) {
            out->has_add_bos_token = true;
            out->add_bos_token = add_bos;
        }
        return OC_OK;
    }

    /* SentencePiece / WordPiece / Tiktoken are not yet implemented. */
    oc_log_error("tokenizer: kind %d (model=\"%s\") not yet implemented",
                 (int)kind, model);
    oc_arena_free(arena);
    out->arena = NULL;
    return OC_ERR_TOKENIZER;
}

/* oc_tokenizer_free() is implemented in tokenizer_bpe.c (it needs access to
 * the BPE internals to free the u64 maps). */

