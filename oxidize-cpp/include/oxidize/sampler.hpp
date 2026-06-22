#pragma once
// Ported from oxidize-core/src/model/sampling.rs
// Token sampling: greedy/argmax plus temperature, top-k, top-p, min-p,
// typical, tail-free, locally-typical filtering, repetition penalties,
// grammar constraints, mirostat, speculative decoding and beam search.
//
// Numerically faithful to the Rust implementation: same softmax (subtract
// max, divide by temperature), same sort/total_cmp ordering semantics, same
// cumulative-sum selection. Greedy is exact (argmax with total_cmp tie-break).
//
// Errors are reported by throwing std::runtime_error (the CLI catches).

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "oxidize/config.hpp"
#include "oxidize/model.hpp"

namespace oxidize {

// Mirror of sampling.rs::SamplingConfig. std::nullopt == Rust None.
struct SamplerConfig {
  float temperature = 1.0f;
  std::optional<size_t> top_k;
  std::optional<float> top_p;
  std::optional<float> min_p;
  std::optional<float> typical_p;
  std::optional<float> tail_free_z;
  std::optional<float> locally_typical_tau;
};

// Mirror of sampling.rs::NewlinePenalty.
struct NewlinePenalty {
  uint32_t token_id = 0;
  float penalty = 0.0f;
};

// Mirror of sampling.rs::RepetitionPenaltyConfig.
struct RepetitionPenaltyConfig {
  float frequency_penalty = 0.0f;
  float presence_penalty = 0.0f;
  std::optional<NewlinePenalty> newline_penalty;
};

// Mirror of sampling.rs::MirostatConfig.
struct MirostatConfig {
  float tau = 0.0f;
  float eta = 0.0f;
  float mu = 0.0f;
};

// Mirror of sampling.rs::GrammarSymbol (a tagged union of Terminal/NonTerminal).
struct GrammarSymbol {
  enum class Kind : uint8_t { Terminal, NonTerminal };
  Kind kind = Kind::Terminal;
  uint32_t terminal = 0;       // valid when kind == Terminal
  std::string non_terminal;    // valid when kind == NonTerminal

  static GrammarSymbol make_terminal(uint32_t token) {
    GrammarSymbol s;
    s.kind = Kind::Terminal;
    s.terminal = token;
    return s;
  }
  static GrammarSymbol make_non_terminal(std::string name) {
    GrammarSymbol s;
    s.kind = Kind::NonTerminal;
    s.non_terminal = std::move(name);
    return s;
  }

  bool operator==(const GrammarSymbol& o) const {
    if (kind != o.kind) return false;
    if (kind == Kind::Terminal) return terminal == o.terminal;
    return non_terminal == o.non_terminal;
  }
};

// Mirror of sampling.rs::GrammarConstraint. Validates start + symbol references
// at construction (throws std::runtime_error on invalid input).
class GrammarConstraint {
 public:
  using Production = std::vector<GrammarSymbol>;
  using Productions = std::unordered_map<std::string, std::vector<Production>>;

  GrammarConstraint(std::string start, Productions productions);

  // Whether appending `token` to `generated_tokens` keeps a valid prefix.
  bool allows_token(const std::vector<uint32_t>& generated_tokens,
                    uint32_t token) const;

 private:
  bool accepts_prefix(const std::vector<uint32_t>& prefix) const;

  std::string start_;
  Productions productions_;
};

// Mirror of sampling.rs::SpeculativeDecodeResult.
struct SpeculativeDecodeResult {
  std::vector<uint32_t> tokens;
  size_t accepted_draft_tokens = 0;
  bool used_residual_fallback = false;
};

// Mirror of sampling.rs::BeamSearchResult.
struct BeamSearchResult {
  std::vector<uint32_t> tokens;
  float score = 0.0f;
};

// argmax with total_cmp tie-break. Throws on empty logits.
Token greedy(const Logits& logits);

// Convenience: sample(logits, config, random) == sample_with_repetition(..)
// with empty recent tokens + default repetition config.
Token sample(const Logits& logits, const SamplerConfig& config, float random);

Token sample_with_repetition(const Logits& logits, const SamplerConfig& config,
                             float random,
                             const std::vector<uint32_t>& recent_tokens,
                             const RepetitionPenaltyConfig& repetition);

Token sample_with_repetition_and_grammar(
    const Logits& logits, const SamplerConfig& config, float random,
    const std::vector<uint32_t>& recent_tokens,
    const RepetitionPenaltyConfig& repetition,
    const std::vector<uint32_t>& generated_tokens,
    const GrammarConstraint* grammar);

// Returns (token, updated_mu).
std::pair<Token, float> sample_mirostat(const Logits& logits, float temperature,
                                        const MirostatConfig& config,
                                        float random);

SpeculativeDecodeResult speculative_decode(
    const std::vector<uint32_t>& draft_tokens,
    const std::vector<Logits>& draft_logits,
    const std::vector<Logits>& target_logits, const SamplerConfig& config,
    const std::vector<float>& randoms);

BeamSearchResult beam_search(const std::vector<Logits>& logits_per_step,
                             size_t beam_width,
                             std::optional<uint32_t> eos_token);

// Deterministic, seedable RNG producing f32 in [0, 1). Used to drive `sample`.
// Implemented as a SplitMix64-seeded xoshiro256** so a given seed yields a
// reproducible stream independent of platform.
class Rng {
 public:
  explicit Rng(uint64_t seed);
  // next f32 uniformly in [0, 1).
  float next_unit();
  uint64_t next_u64();

 private:
  uint64_t s_[4];
};

}  // namespace oxidize
