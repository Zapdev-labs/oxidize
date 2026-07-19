/*
 * tokenizer.h — LoadedTokenizer (BPE/SP/WP/Tiktoken) public API.
 *
 * Port of oxidize-core/src/format/tokenizer.rs to C11.
 *
 * Dispatch: `oc_tokenizer_load_from_gguf()` reads `tokenizer.ggml.model`
 * from GGUF metadata and dispatches to the format-specific loader:
 *   - "gpt2" | "lfm2" | "lfm2moe"  → BPE (byte-level, tiktoken-style)
 *   - "llama" | "gemma" | "gemma4" → SentencePiece (future feature)
 *   - "bert"                       → WordPiece (future feature)
 *   - "tiktoken"                   → Tiktoken (future feature)
 *   - <other>                      → OC_ERR_TOKENIZER
 *
 * The BPE path (the only one implemented by the `tokenizer-bpe-qwen`
 * feature) is a faithful port of the Rust `BpeTokenizer`: byte-level
 * encoding via the GPT-2 `bytes_to_unicode` table, merge-rank hash lookup,
 * special-piece pre-split (CONTROL/USER_DEFINED tokens like `<|im_start|>`),
 * and ChatML template rendering. No regex pre-tokenization is performed —
 * matching the Rust reference exactly (VAL-TOK-001, VAL-TOK-011 require
 * bit-exact parity with Rust `LoadedTokenizer::Bpe`).
 */
#ifndef OXIDIZE_TOKENIZER_H
#define OXIDIZE_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/arena.h"
#include "oxidize/error.h"
#include "oxidize/gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Kinds ────────────────────────────────────────────────────────────── */

typedef enum {
    OC_TOK_KIND_NONE = 0,
    OC_TOK_KIND_BPE,
    OC_TOK_KIND_SENTENCEPIECE,
    OC_TOK_KIND_WORDPIECE,
    OC_TOK_KIND_TIKTOKEN,
} OcTokenizerKind;

/* Policy for handling special-token strings in user input.
 *
 *   OC_TOK_DISALLOW_SPECIAL — every byte of the input (including any text
 *     that happens to look like a special-token marker such as `<|im_start|>`)
 *     is tokenized via the byte-level BPE path; no special-token id is ever
 *     emitted. This is the injection-prevention mode (VAL-TOK-004).
 *
 *   OC_TOK_ALLOW_SPECIAL    — control / user-defined pieces (e.g. `<|im_start|>`,
 *     `<|im_end|>`) are matched verbatim and emitted as their single id before
 *     byte-level BPE runs on the surrounding text (mirrors Rust
 *     `BpeTokenizer::encode` with non-empty `special_pieces`).
 *
 *   OC_TOK_DEFAULT          — alias for `OC_TOK_ALLOW_SPECIAL` (the Rust
 *     `encode` path always honors `special_pieces` when present). Kept as a
 *     distinct value so callers can request "the tokenizer's default
 *     behavior" without having to know which kind is loaded.
 */
typedef enum {
    OC_TOK_DEFAULT         = 0,
    OC_TOK_ALLOW_SPECIAL   = 1,
    OC_TOK_DISALLOW_SPECIAL = 2,
} OcSpecialTokenPolicy;

/* Chat template kinds. `OC_TEMPLATE_CHATML` is the fast-path ChatML renderer
 * (mirrors Rust `process_chat_template` when the template contains
 * `<|im_start|>`). The raw Jinja2 path is NOT implemented — callers that
 * need it should use the Rust reference. */
typedef enum {
    OC_TEMPLATE_CHATML = 0,
} OcTemplateKind;

/* A single chat message (role + content). Mirrors Rust `ChatMessage`. */
typedef struct OcChatMessage {
    const char *role;
    const char *content;
} OcChatMessage;

/* ─── Loaded tokenizer ─────────────────────────────────────────────────── */

/* Opaque BPE tokenizer state (defined in tokenizer_bpe.c). */
typedef struct OcBpeTokenizer OcBpeTokenizer;

/* A loaded tokenizer. `kind` selects the implementation; `bpe` points to the
 * BPE state when `kind == OC_TOK_KIND_BPE` (NULL otherwise). The struct owns
 * its arena; `oc_tokenizer_free()` releases all memory. */
typedef struct OcTokenizer {
    OcTokenizerKind  kind;
    OcBpeTokenizer  *bpe;     /* valid iff kind == OC_TOK_KIND_BPE        */
    OcArena          *arena;   /* owns all tokenizer-lifetime allocations  */
    /* Special-token ids loaded from `tokenizer.ggml.*_token_id` metadata. */
    uint32_t unknown_id;  bool has_unknown;
    uint32_t bos_id;      bool has_bos;
    uint32_t eos_id;      bool has_eos;
    uint32_t pad_id;      bool has_pad;
    uint32_t separator_id; bool has_separator;
    uint32_t cls_id;      bool has_cls;
    uint32_t mask_id;     bool has_mask;
    /* `tokenizer.ggml.add_bos_token` (false when metadata absent; BPE
     * models do not add BOS by default). */
    bool add_bos_token;
    bool has_add_bos_token;
} OcTokenizer;

/* ─── Lifecycle ────────────────────────────────────────────────────────── */

/* Load a tokenizer from parsed GGUF metadata. Reads `tokenizer.ggml.model`
 * and dispatches to the format-specific loader. On success, `*out` is
 * initialized with `kind`, the implementation pointer, and the special-token
 * ids pulled from `tokenizer.ggml.*_token_id` metadata keys.
 *
 * Returns OC_OK, OC_ERR_TOKENIZER (unknown model string, missing required
 * metadata, or invalid merge entry), OC_ERR_OOM, or OC_ERR_INVALID_ARG. */
OcError oc_tokenizer_load_from_gguf(const OcGgufFile *gguf, OcTokenizer *out);

/* Free a loaded tokenizer and all its allocations. Safe on NULL or zeroed
 * OcTokenizer. After this call, `*t` is zeroed. */
void oc_tokenizer_free(OcTokenizer *t);

/* ─── Encode / Decode ──────────────────────────────────────────────────── */

/* Encode `text` (UTF-8, NUL-terminated) into a sequence of token ids.
 * `policy` controls special-token handling (see OcSpecialTokenPolicy).
 *
 * On success, `*out_ids` points to a freshly malloc'd array of `*out_count`
 * u32 ids owned by the caller (free with `free()`). Returns OC_OK,
 * OC_ERR_TOKENIZER, OC_ERR_OOM, or OC_ERR_INVALID_ARG. */
OcError oc_tokenizer_encode(const OcTokenizer *t, const char *text,
                            OcSpecialTokenPolicy policy,
                            uint32_t **out_ids, size_t *out_count);

/* Decode `ids` (array of `count` token ids) back into text. On success,
 * `*out_text` points to a freshly malloc'd NUL-terminated UTF-8 string owned
 * by the caller (free with `free()`). Unknown ids return OC_ERR_TOKENIZER.
 * Mirrors Rust `BpeTokenizer::decode` (byte-level GPT-2 reversal when
 * `use_byte_fallback` is set). */
OcError oc_tokenizer_decode(const OcTokenizer *t, const uint32_t *ids,
                            size_t count, char **out_text);

/* ─── Chat templates ───────────────────────────────────────────────────── */

/* Render a chat template for `messages` (array of `n_messages` entries).
 * `OC_TEMPLATE_CHATML` produces the canonical ChatML formatting:
 *
 *   <|im_start|>{role}\n{content}<|im_end|>\n
 *   ...
 *   <|im_start|>assistant\n     (when add_generation_prompt is true)
 *
 * On success, `*out_text` points to a freshly malloc'd NUL-terminated string
 * owned by the caller. Returns OC_OK, OC_ERR_OOM, or OC_ERR_INVALID_ARG. */
OcError oc_tokenizer_apply_chat_template(const OcChatMessage *messages,
                                         size_t n_messages,
                                         OcTemplateKind kind,
                                         bool add_generation_prompt,
                                         char **out_text);

/* ─── BPE direct API (for testing / advanced callers) ─────────────────── */

/* Load a BPE tokenizer from parsed GGUF metadata. Reads
 * `tokenizer.ggml.tokens`, `tokenizer.ggml.merges` (optional), and
 * `tokenizer.ggml.token_type` (optional, for CONTROL/USER_DEFINED special
 * pieces). All allocations live in `arena`. On success, `*out` points to an
 * arena-owned `OcBpeTokenizer`. Mirrors Rust `load_bpe`. */
OcError oc_bpe_load_from_gguf(const OcGgufFile *gguf, OcArena *arena,
                              OcBpeTokenizer **out);

/* Copy the BPE tokenizer's special-token ids into an OcTokenizer wrapper.
 * Used by `oc_tokenizer_load_from_gguf()` after loading the BPE impl. */
void oc_bpe_fill_special_tokens(const OcBpeTokenizer *bpe, OcTokenizer *out);

/* Train a toy BPE tokenizer from a corpus (mirrors Rust
 * `BpeTokenizer::train`). Used by the test suite to construct a known
 * vocab + merge table without needing a real Qwen GGUF fixture. The
 * returned tokenizer is owned by `arena`. `use_byte_fallback` is set to
 * false (char-level, matching the Rust `train` constructor). */
OcError oc_bpe_train(const char *const *corpus, size_t n_corpus,
                     size_t merge_limit, OcArena *arena,
                     OcBpeTokenizer **out);

/* Set the unknown token on a trained BPE tokenizer (mirrors Rust
 * `BpeTokenizer::with_unknown_token`). */
OcError oc_bpe_with_unknown_token(OcBpeTokenizer *bpe, OcArena *arena,
                                  const char *token);

/* Encode via BPE without going through the OcTokenizer wrapper. Used by
 * tests. Returns a malloc'd id array. */
OcError oc_bpe_encode(const OcBpeTokenizer *bpe, const char *text,
                      uint32_t **out_ids, size_t *out_count);

/* Decode via BPE without going through the OcTokenizer wrapper. */
OcError oc_bpe_decode(const OcBpeTokenizer *bpe, const uint32_t *ids,
                      size_t count, char **out_text);

/* Free the malloc'd internals of a BPE tokenizer (vocab hashtable, u64
 * maps). Does NOT free the OcBpeTokenizer struct itself (arena-owned) nor
 * the arena. Call before oc_arena_free() when the tokenizer was created via
 * oc_bpe_train(). When the tokenizer was loaded via oc_tokenizer_load_from_gguf(),
 * oc_tokenizer_free() already calls this. */
void oc_bpe_free(OcBpeTokenizer *bpe);

/* Whether a token id is a special token (bos/eos/pad/unk/sep/cls/mask). */
bool oc_tokenizer_is_special(const OcTokenizer *t, uint32_t id);

/* Whether a BOS token should be prepended by default for this tokenizer.
 * Mirrors Rust `LoadedTokenizer::add_bos_default()`. */
bool oc_tokenizer_add_bos_default(const OcTokenizer *t);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_TOKENIZER_H */
