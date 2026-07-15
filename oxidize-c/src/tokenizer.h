/* SentencePiece-style tokenizer from GGUF metadata (tokenizer.ggml.tokens /
 * scores / token_type). Viterbi best-path segmentation with byte fallback.
 * Piece strings point into the GgufFile mmap: keep the file open. */
#ifndef OC_TOKENIZER_H
#define OC_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "gguf.h"

typedef struct {
  const char* ptr; /* into gguf mmap */
  size_t len;
} TokPiece;

/* One BPE merge rule keyed by the pair of token ids it joins. left < 0 marks an
 * empty hash slot. */
typedef struct {
  int32_t left, right; /* the adjacent pair, or left<0 for an empty slot */
  int32_t rank;        /* merge priority: lower = applied first */
  int32_t merged;      /* id of the joined piece */
} BpeMerge;

typedef struct {
  TokPiece* pieces;
  float* scores;
  int32_t* token_types;
  size_t n_vocab;
  size_t max_piece_len;
  /* open-addressing hash of piece -> id */
  int32_t* ht;
  size_t ht_size; /* power of two */
  int64_t bos_id, eos_id, unk_id, eot_id;
  bool add_bos;
  bool add_space_prefix; /* tokenizer.ggml.add_space_prefix (gemma4: false) */
  /* GPT-2 byte-level BPE (tokenizer.ggml.model == "gpt2"): input bytes are
   * remapped through the GPT-2 byte->unicode table before matching, and
   * pieces decode back through the inverse map. When tokenizer.ggml.merges is
   * present, segmentation applies those merges in RANK order (real BPE, ids
   * match llama.cpp); with no merges it falls back to greedy longest-match. */
  bool is_bpe;
  uint16_t byte_to_cp[256]; /* GPT-2 byte -> unicode codepoint */
  uint8_t cp_to_byte[512];  /* inverse (codepoints < 512) */
  /* Rank-based BPE merges (tokenizer.ggml.merges). Open-addressing hash keyed on
   * (left,right); empty slots have left < 0. size 0 => no merges, greedy path. */
  BpeMerge* merge_ht;
  size_t merge_ht_size; /* power of two, or 0 */
  /* CONTROL/USER_DEFINED token ids (chat markers), longest piece first: the
   * input is split on these before byte-level BPE so they emit their exact id. */
  int32_t* special_ids;
  size_t n_special;
} Tokenizer;

/* 0 on success, -1 on missing/invalid metadata (message on stderr). */
int tokenizer_init(Tokenizer* t, const GgufFile* g);
void tokenizer_free(Tokenizer* t);

int32_t tokenizer_piece_id(const Tokenizer* t, const char* piece, size_t len);

/* Encodes text; returns malloc'd token array, count in *n_out. add_bos
 * prepends bos when the vocab wants it. */
int32_t* tokenizer_encode(const Tokenizer* t, const char* text, bool add_bos,
                          size_t* n_out);

/* Appends the decoded text of one token to buf (caller-sized); returns bytes
 * written. Strips the SentencePiece "\xe2\x96\x81" to ' ' and expands <0xXX>. */
size_t tokenizer_decode_token(const Tokenizer* t, int32_t id, char* buf, size_t cap);

/* ---- chat templates -------------------------------------------------------
 * A fixed control-token table keyed on family (the llama.cpp approach), NOT a
 * Jinja expander. Detect the family, then format (system?, user) turns into the
 * exact control-token string; the string is tokenized through the normal
 * special-token-aware encode path so <|im_start|> etc. map to their real ids. */
typedef enum {
  CHAT_CHATML = 0, /* <|im_start|>role\n...<|im_end|>   (Qwen/Yi/most) */
  CHAT_LLAMA3,     /* <|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|> */
  CHAT_MISTRAL,    /* [INST] ... [/INST] ...</s> */
  CHAT_GEMMA,      /* <start_of_turn>role\n...<end_of_turn> */
  CHAT_PHI3,       /* <|user|>\n...<|end|>\n<|assistant|> */
  CHAT_GEMMA4,     /* <|turn>role\n...<turn|> with thought channel */
} ChatFamily;

/* Detect the chat family. `chat_template` is tokenizer.ggml.chat_template and
 * may be NULL; when it is missing or unrecognized the vocab's special tokens
 * are probed instead. Falls back to generic ChatML. */
ChatFamily chat_detect(const Tokenizer* t, const char* chat_template);

/* Human-readable family name (logging / --print-plan style output). */
const char* chat_family_name(ChatFamily fam);

/* The turn-terminating control token for a family ("<|im_end|>", "<|eot_id|>",
 * "</s>", ...): use it as an extra stop id during generation. */
const char* chat_stop_token(ChatFamily fam);

/* Format one conversation turn into control-token text. `system` may be NULL
 * (applied only on the first turn). `first_turn` selects the opening form (adds
 * the system turn, no leading assistant-close); otherwise the previous
 * assistant turn is closed first so the result can be appended to a live KV
 * cache. Writes a NUL-terminated string into buf; returns its length, or 0 if
 * it would not fit in cap. */
size_t chat_format_turn(ChatFamily fam, const char* system, const char* user,
                        bool first_turn, char* buf, size_t cap);

/* Self-check for chat-template detection + formatting: builds a GGUF whose
 * vocab carries several families' control tokens, then for each family asserts
 * chat_detect() picks it AND the formatted turn tokenizes to the expected
 * control-token IDS (not just matching text). Returns 0, aborts on mismatch. */
int chat_selftest(void);

/* Self-check for the rank-based BPE merge policy and special-token pre-split:
 * builds a tiny in-memory GGUF where greedy longest-match and rank-based merges
 * DIVERGE, asserts the rank-based (llama.cpp-correct) ids, and checks the
 * round-trip. Returns 0 on success, aborts on mismatch. Not wired into the main
 * test binary (that harness is owned elsewhere); link + call directly to run. */
int tokenizer_selftest(void);

#endif
