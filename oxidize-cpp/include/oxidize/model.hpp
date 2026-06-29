#pragma once
// Mirror of oxidize-core/src/model/model.rs: Model trait, Session, Token, Logits.

#include <cstdint>
#include <vector>
#include "oxidize/config.hpp"

namespace oxidize {

using Token = uint32_t;
using Logits = std::vector<float>;

// Tracks how many tokens the model has consumed (KV positions).
class Session {
 public:
  size_t consumed_tokens() const { return consumed_tokens_; }
  void record_tokens(size_t n) { consumed_tokens_ += n; }
  void rewind_to(size_t n) { consumed_tokens_ = n; }
 private:
  size_t consumed_tokens_ = 0;
};

// Abstract inference model. Implemented by LlamaModel (Phase 1).
class Model {
 public:
  virtual ~Model() = default;
  virtual Logits forward(const std::vector<Token>& tokens, Session& session) = 0;
  virtual size_t vocab_size() const = 0;
  virtual size_t context_size() const = 0;
  virtual size_t layer_count() const = 0;
  virtual void rewind_to(size_t /*consumed_tokens*/) {}

  // Default: one forward per token, returning logits after each.
  virtual std::vector<Logits> forward_many(const std::vector<Token>& tokens, Session& session) {
    std::vector<Logits> out;
    out.reserve(tokens.size());
    for (Token t : tokens) out.push_back(forward({t}, session));
    return out;
  }
};

}  // namespace oxidize
