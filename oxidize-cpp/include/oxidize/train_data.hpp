#pragma once
// Data pipeline: JSONL parse + Qwen2.5 chat template + tokenization + loss mask.
//
// Format expected in JSONL file:
//   {"id": "...", "messages": [{"role": "user"|"assistant"|"system", "content": "..."}]}
//
// Chat template (Qwen2.5 im_start/im_end):
//   <|im_start|>system\n{content}<|im_end|>\n
//   <|im_start|>user\n{content}<|im_end|>\n
//   <|im_start|>assistant\n{content}<|im_end|>\n
//
// Loss mask: 1.0 on tokens that are INSIDE assistant content (after the \n
// following <|im_start|>assistant), 0.0 everywhere else.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <random>

#include "oxidize/model.hpp"    // Token
#include "oxidize/tokenizer.hpp"

namespace oxidize {

struct ChatMessage {
  std::string role;
  std::string content;
};

struct TrainSample {
  std::vector<Token> tokens;      // full tokenized chat
  std::vector<float> loss_mask;   // same length; 1.0 = compute loss here
};

// Parse all samples from a JSONL file. Throws on IO/parse error.
// max_seq_len: samples longer than this are truncated.
std::vector<TrainSample> load_jsonl_samples(const std::string& path,
                                             const Tokenizer& tok,
                                             size_t max_seq_len);

// Build a TrainSample from a list of chat messages using the Qwen2.5 template.
// Throws if tokenizer can't find <|im_start|> or <|im_end|>.
TrainSample build_chat_sample(const std::vector<ChatMessage>& messages,
                               const Tokenizer& tok,
                               size_t max_seq_len);

class DataLoader {
 public:
  // Pre-tokenizes all samples. seed for shuffling.
  DataLoader(std::vector<TrainSample> samples, size_t batch_size,
             uint64_t seed);

  // Returns the next batch (circular, re-shuffled each epoch). Never returns
  // an empty batch when there are samples. For grad-accum batch_size == 1.
  const TrainSample& next_sample();

  size_t size() const { return samples_.size(); }
  const TrainSample& sample(size_t i) const { return samples_[i]; }

 private:
  std::vector<TrainSample> samples_;
  std::vector<size_t> order_;
  size_t pos_ = 0;
  size_t batch_size_;
  std::mt19937_64 rng_;

  void reshuffle();
};

}  // namespace oxidize
