/* tokenizer.h — LoadedTokenizer (BPE/SP/WP/Tiktoken) public API. Round-trip parity with Rust is a hard invariant (VAL-TOK-006..011): */
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


typedef enum {
    OC_TOK_KIND_NONE = 0,
    OC_TOK_KIND_BPE,
    OC_TOK_KIND_SENTENCEPIECE,
    OC_TOK_KIND_WORDPIECE,
    OC_TOK_KIND_TIKTOKEN,
} OcTokenizerKind;

/* Policy for handling special-token strings in user input. the injection-prevention mode (VAL-TOK-004). For BPE, marker text such (VAL-TOK-007). Used by Llama/Gemma callers that need the leading BOS */
typedef enum {
    OC_TOK_DEFAULT          = 0,
    OC_TOK_ALLOW_SPECIAL    = 1,
    OC_TOK_DISALLOW_SPECIAL = 2,
    OC_TOK_ADD_BOS          = 3,
} OcSpecialTokenPolicy;

/* Chat template kinds. `OC_TEMPLATE_CHATML` is the fast-path ChatML renderer */
typedef enum {
    OC_TEMPLATE_CHATML = 0,
} OcTemplateKind;

/* A single chat message (role + content). Mirrors Rust `ChatMessage`. */
typedef struct OcChatMessage {
    const char *role;
    const char *content;
} OcChatMessage;


/* A (piece, score) pair for constructing a SentencePiece unigram tokenizer. */
typedef struct OcSpPiece {
    const char *piece;
    float       score;
} OcSpPiece;

/* A non-NUL-terminated byte slice (pointer + length). Used by the Tiktoken
 * constructor because raw tiktoken vocab tokens are arbitrary byte
 * sequences (they may contain NUL bytes or invalid UTF-8). */
typedef struct OcByteSlice {
    const uint8_t *data;
    size_t         len;
} OcByteSlice;

/* A (left, right) pair of byte slices, used to declare Tiktoken merge rules. */
typedef struct OcByteSlicePair {
    OcByteSlice left;
    OcByteSlice right;
} OcByteSlicePair;


/* Opaque per-kind tokenizer states (defined in their respective .c files). */
typedef struct OcBpeTokenizer           OcBpeTokenizer;
typedef struct OcSentencePieceTokenizer OcSentencePieceTokenizer;
typedef struct OcWordPieceTokenizer     OcWordPieceTokenizer;
typedef struct OcTiktokenTokenizer       OcTiktokenTokenizer;

/* A loaded tokenizer. `kind` selects the implementation; the corresponding
 * pointer field is set (`bpe`, `sp`, `wp`, or `tiktoken`). The struct owns
 * its arena; `oc_tokenizer_free()` releases all memory. */
typedef struct OcTokenizer {
    OcTokenizerKind  kind;
    OcBpeTokenizer           *bpe;       /* valid iff kind == OC_TOK_KIND_BPE            */
    OcSentencePieceTokenizer *sp;         /* valid iff kind == OC_TOK_KIND_SENTENCEPIECE  */
    OcWordPieceTokenizer     *wp;         /* valid iff kind == OC_TOK_KIND_WORDPIECE      */
    OcTiktokenTokenizer      *tiktoken;   /* valid iff kind == OC_TOK_KIND_TIKTOKEN       */
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
     * models do not add BOS by default; SP models do). */
    bool add_bos_token;
    bool has_add_bos_token;
} OcTokenizer;


/* Load a tokenizer from parsed GGUF metadata. Reads `tokenizer.ggml.model` */
OcError oc_tokenizer_load_from_gguf(const OcGgufFile *gguf, OcTokenizer *out);

/* Free a loaded tokenizer and all its allocations. Safe on NULL or zeroed
 * OcTokenizer. After this call, `*t` is zeroed. */
void oc_tokenizer_free(OcTokenizer *t);


/* Encode `text` (UTF-8, NUL-terminated) into a sequence of token ids. */
OcError oc_tokenizer_encode(const OcTokenizer *t, const char *text,
                            OcSpecialTokenPolicy policy,
                            uint32_t **out_ids, size_t *out_count);

/* Decode `ids` (array of `count` token ids) back into text. */
OcError oc_tokenizer_decode(const OcTokenizer *t, const uint32_t *ids,
                            size_t count, char **out_text);


/* Render a chat template for `messages` (array of `n_messages` entries). */
OcError oc_tokenizer_apply_chat_template(const OcChatMessage *messages,
                                         size_t n_messages,
                                         OcTemplateKind kind,
                                         bool add_generation_prompt,
                                         char **out_text);

typedef struct OcStreamingDetokenizer {
    const OcTokenizer *tokenizer;
    /* Pending bytes from previous tokens that form incomplete UTF-8. */
    uint8_t pending[8];
    size_t  pending_len;
    uint8_t *output;
    size_t output_cap;
} OcStreamingDetokenizer;

/* Initialize a streaming detokenizer bound to `tok`. */
void oc_streaming_detok_init(OcStreamingDetokenizer *sd,
                             const OcTokenizer *tok);

/* Push a single token id and get back the printable text delta. */
OcError oc_streaming_detok_push(OcStreamingDetokenizer *sd, uint32_t token,
                                const char **out_delta, size_t *out_len);

/* Flush any pending bytes as a final delta (call at end of stream). */
OcError oc_streaming_detok_flush(OcStreamingDetokenizer *sd,
                                 const char **out_delta, size_t *out_len);

/* Reset the streaming detokenizer (clear pending buffer). */
void oc_streaming_detok_reset(OcStreamingDetokenizer *sd);


/* Given a token sequence and the tokenizer, try to heal the last 1-3 tokens. */
OcError oc_tokenizer_heal_tokens(const OcTokenizer *tok,
                                 const uint32_t *ids, size_t n_ids,
                                 uint32_t **out_ids, size_t *out_count);
void oc_streaming_detok_free(OcStreamingDetokenizer *sd);


/* Load a BPE tokenizer from parsed GGUF metadata. Reads */
OcError oc_bpe_load_from_gguf(const OcGgufFile *gguf, OcArena *arena,
                              OcBpeTokenizer **out);

/* Copy the BPE tokenizer's special-token ids into an OcTokenizer wrapper.
 * Used by `oc_tokenizer_load_from_gguf()` after loading the BPE impl. */
void oc_bpe_fill_special_tokens(const OcBpeTokenizer *bpe, OcTokenizer *out);

/* Train a toy BPE tokenizer from a corpus (mirrors Rust */
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
OcError oc_bpe_decode_raw(const OcBpeTokenizer *bpe, const uint32_t *ids,
                          size_t count, uint8_t **out_bytes, size_t *out_len);

/* Free the malloc'd internals of a BPE tokenizer (vocab hashtable, u64 */
void oc_bpe_free(OcBpeTokenizer *bpe);


/* Load a SentencePiece unigram tokenizer from GGUF metadata. Reads */
OcError oc_sp_load_from_gguf(const OcGgufFile *gguf, OcArena *arena,
                             OcSentencePieceTokenizer **out);

/* Copy the SentencePiece tokenizer's special-token ids into the wrapper. */
void oc_sp_fill_special_tokens(const OcSentencePieceTokenizer *sp, OcTokenizer *out);

/* Construct a SentencePiece tokenizer from an explicit (piece, score) list. */
OcError oc_sp_new(const OcSpPiece *pieces, size_t n_pieces,
                  OcArena *arena, OcSentencePieceTokenizer **out);

/* Set the unknown token on a SentencePiece tokenizer (mirrors Rust
 * `SentencePieceUnigramTokenizer::with_unknown_token`). */
OcError oc_sp_with_unknown_token(OcSentencePieceTokenizer *sp, OcArena *arena,
                                 const char *token);

/* Encode via SentencePiece unigram Viterbi segmentation. Mirrors Rust
 * `SentencePieceUnigramTokenizer::encode`. Returns a malloc'd id array. */
OcError oc_sp_encode(const OcSentencePieceTokenizer *sp, const char *text,
                     uint32_t **out_ids, size_t *out_count);

/* Decode via SentencePiece: concatenate token strings. Mirrors Rust
 * `SentencePieceUnigramTokenizer::decode`. */
OcError oc_sp_decode(const OcSentencePieceTokenizer *sp, const uint32_t *ids,
                     size_t count, char **out_text);

/* Free the malloc'd internals of a SentencePiece tokenizer (vocab
 * hashtable, score array). Does NOT free the struct itself (arena-owned). */
void oc_sp_free(OcSentencePieceTokenizer *sp);


/* Load a WordPiece tokenizer from GGUF metadata. Reads
 * `tokenizer.ggml.tokens` (required). All allocations live in `arena`.
 * Mirrors Rust `load_wordpiece`. */
OcError oc_wp_load_from_gguf(const OcGgufFile *gguf, OcArena *arena,
                             OcWordPieceTokenizer **out);

/* Copy the WordPiece tokenizer's special-token ids into the wrapper. */
void oc_wp_fill_special_tokens(const OcWordPieceTokenizer *wp, OcTokenizer *out);

/* Construct a WordPiece tokenizer from an explicit vocab list. Mirrors
 * Rust `WordPieceTokenizer::new`. Used by the test suite. */
OcError oc_wp_new(const char *const *vocab_tokens, size_t n_tokens,
                  OcArena *arena, OcWordPieceTokenizer **out);

/* Set the unknown token on a WordPiece tokenizer (mirrors Rust
 * `WordPieceTokenizer::with_unknown_token`). */
OcError oc_wp_with_unknown_token(OcWordPieceTokenizer *wp, OcArena *arena,
                                 const char *token);

/* Encode via WordPiece greedy longest-match with `##` continuation.
 * Mirrors Rust `WordPieceTokenizer::encode`. */
OcError oc_wp_encode(const OcWordPieceTokenizer *wp, const char *text,
                     uint32_t **out_ids, size_t *out_count);

/* Decode via WordPiece: concatenate pieces, stripping `##` prefixes on
 * continuation tokens. Mirrors Rust `WordPieceTokenizer::decode`. */
OcError oc_wp_decode(const OcWordPieceTokenizer *wp, const uint32_t *ids,
                     size_t count, char **out_text);

/* Free the malloc'd internals of a WordPiece tokenizer. */
void oc_wp_free(OcWordPieceTokenizer *wp);


/* Load a raw Tiktoken tokenizer from GGUF metadata. Reads */
OcError oc_tiktoken_load_from_gguf(const OcGgufFile *gguf, OcArena *arena,
                                  OcTiktokenTokenizer **out);

/* Copy the Tiktoken tokenizer's special-token ids into the wrapper. */
void oc_tiktoken_fill_special_tokens(const OcTiktokenTokenizer *t, OcTokenizer *out);

/* Construct a Tiktoken tokenizer from explicit vocab + merge-pair lists. */
OcError oc_tiktoken_new(const OcByteSlice *vocab_tokens, size_t n_vocab,
                        const OcByteSlicePair *merge_pairs, size_t n_merges,
                        OcArena *arena, OcTiktokenTokenizer **out);

/* Set the unknown token on a Tiktoken tokenizer (mirrors Rust
 * `TiktokenTokenizer::with_unknown_token`). */
OcError oc_tiktoken_with_unknown_token(OcTiktokenTokenizer *t, OcArena *arena,
                                       const char *token);

/* Encode via Tiktoken byte-level BPE (no GPT-2 mapping). Mirrors Rust
 * `TiktokenTokenizer::encode`. */
OcError oc_tiktoken_encode(const OcTiktokenTokenizer *t, const char *text,
                           uint32_t **out_ids, size_t *out_count);

/* Decode via Tiktoken: concatenate byte sequences, lossy UTF-8. Mirrors
 * Rust `TiktokenTokenizer::decode`. */
OcError oc_tiktoken_decode(const OcTiktokenTokenizer *t, const uint32_t *ids,
                           size_t count, char **out_text);
OcError oc_tiktoken_decode_raw(const OcTiktokenTokenizer *t,
                               const uint32_t *ids, size_t count,
                               uint8_t **out_bytes, size_t *out_len);

/* Free the malloc'd internals of a Tiktoken tokenizer. */
void oc_tiktoken_free(OcTiktokenTokenizer *t);


/* Whether a token id is a special token (bos/eos/pad/unk/sep/cls/mask). */
bool oc_tokenizer_is_special(const OcTokenizer *t, uint32_t id);

/* Whether a BOS token should be prepended by default for this tokenizer.
 * Mirrors Rust `LoadedTokenizer::add_bos_default()`. */
bool oc_tokenizer_add_bos_default(const OcTokenizer *t);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_TOKENIZER_H */
