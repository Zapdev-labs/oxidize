#pragma once
// GGUF tokenizer: SentencePiece (llama) and byte-level BPE (gpt2/qwen) encode +
// decode, built from tokenizer.ggml.* metadata. Lets the CLI run from prompt
// text instead of pre-tokenized ids.
//
// SPM: scores-driven symbol-pair merge + byte fallback (llama.cpp llm_tokenizer_spm).
// BPE: GPT-2 byte-level mapping + rank-ordered merges with a Unicode-category
//      pretokenizer (llama.cpp llm_tokenizer_bpe).

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "oxidize/gguf.hpp"
#include "oxidize/model.hpp"  // Token

namespace oxidize {

class Tokenizer {
 public:
  enum class Kind { SPM, BPE };

  // Build from a loaded GGUF (reads tokenizer.ggml.*). Throws if unsupported.
  static Tokenizer from_gguf(const GgufModel& g);

  // Encode text to token ids. `add_bos` prepends the BOS token when one exists.
  std::vector<Token> encode(const std::string& text, bool add_bos) const;

  // Decode ids back to text (inverse of encode; byte-fallback aware).
  std::string decode(const std::vector<Token>& ids) const;
  // Decode a single token to its text fragment (for streaming output).
  std::string decode_token(Token id) const;

  Kind kind() const { return kind_; }
  size_t vocab_size() const { return id_to_piece_.size(); }
  bool has_bos() const { return bos_id_ >= 0; }
  Token bos_id() const { return static_cast<Token>(bos_id_); }
  Token eos_id() const { return static_cast<Token>(eos_id_); }
  // End-of-generation: EOS or EOT (chat) — stop decoding when produced.
  bool is_eog(Token id) const;

 private:
  Kind kind_ = Kind::SPM;
  std::vector<std::string> id_to_piece_;
  std::vector<float> scores_;
  std::vector<int32_t> token_types_;
  std::unordered_map<std::string, int32_t> piece_to_id_;
  std::unordered_map<std::string, int32_t> bpe_ranks_;  // "a b" -> rank
  int64_t bos_id_ = -1, eos_id_ = -1, unk_id_ = -1, eot_id_ = -1;
  bool add_space_prefix_ = true;
  bool wants_bos_ = true;  // model convention (SPM adds BOS; gpt2/BPE usually not)

  int32_t piece_id(const std::string& p) const;
  std::vector<Token> encode_spm(const std::string& text) const;
  std::vector<Token> encode_bpe(const std::string& text) const;
};

}  // namespace oxidize
