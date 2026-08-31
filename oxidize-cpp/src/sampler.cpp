// Ported from oxidize-core/src/model/sampling.rs
//
// Faithful C++20 port. Notes on numeric fidelity:
//  * Rust uses f32::total_cmp for all logit/prob comparisons and sorts. We
//    replicate total_cmp exactly (bit-pattern total ordering over IEEE-754
//    binary32 incl. NaN/signed-zero) so argmax/sort tie-breaks match.
//  * Rust `sort_unstable_by` is not stable; std::sort is also not stable, so we
//    use std::sort to mirror it. Rust `sort_by` (stable) -> std::stable_sort.
//  * Softmax: exp((logit - max)/temperature); sum; normalize. Identical order.
//  * Selection: cumulative sum with `target <= cumulative` (target = random*sum).

#include "oxidize/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace oxidize {

namespace {

// Replicates Rust f32::total_cmp: returns <0, 0, >0.
inline int total_cmp(float a, float b) {
  int32_t ai, bi;
  std::memcpy(&ai, &a, 4);
  std::memcpy(&bi, &b, 4);
  // Rust f32::total_cmp: left ^= ((left >> 31) as u32 >> 1) as i32.
  // (left >> 31) is an arithmetic shift -> 0xFFFFFFFF when negative, then
  // >> 1 unsigned -> 0x7FFFFFFF, giving a total bit ordering incl. NaN/-0.
  ai ^= static_cast<int32_t>(static_cast<uint32_t>(ai >> 31) >> 1);
  bi ^= static_cast<int32_t>(static_cast<uint32_t>(bi >> 31) >> 1);
  if (ai < bi) return -1;
  if (ai > bi) return 1;
  return 0;
}

inline bool total_lt(float a, float b) { return total_cmp(a, b) < 0; }

constexpr float kMinPositive = std::numeric_limits<float>::min();  // f32::MIN_POSITIVE

inline bool is_finite(float x) { return std::isfinite(x); }

// Greedy argmax over a raw float span, mirroring Rust max_by(total_cmp).
size_t argmax_total_cmp(const std::vector<float>& v) {
  // Rust iter().max_by keeps the LAST maximum on ties (max_by returns the last
  // element if several are equally maximum). We replicate that.
  size_t best = 0;
  for (size_t i = 1; i < v.size(); ++i) {
    if (total_cmp(v[i], v[best]) >= 0) best = i;
  }
  return best;
}

float max_total_cmp(const std::vector<float>& v) {
  return v[argmax_total_cmp(v)];
}

}  // namespace

// GrammarConstraint

GrammarConstraint::GrammarConstraint(std::string start, Productions productions)
    : start_(std::move(start)), productions_(std::move(productions)) {
  if (start_.empty() || productions_.find(start_) == productions_.end()) {
    throw std::runtime_error("InvalidGrammarConstraint");
  }
  for (const auto& [name, alternatives] : productions_) {
    (void)name;
    for (const auto& production : alternatives) {
      for (const auto& symbol : production) {
        if (symbol.kind == GrammarSymbol::Kind::NonTerminal &&
            productions_.find(symbol.non_terminal) == productions_.end()) {
          throw std::runtime_error("InvalidGrammarConstraint");
        }
      }
    }
  }
}

bool GrammarConstraint::allows_token(
    const std::vector<uint32_t>& generated_tokens, uint32_t token) const {
  std::vector<uint32_t> candidate;
  candidate.reserve(generated_tokens.size() + 1);
  candidate.insert(candidate.end(), generated_tokens.begin(),
                   generated_tokens.end());
  candidate.push_back(token);
  return accepts_prefix(candidate);
}

bool GrammarConstraint::accepts_prefix(
    const std::vector<uint32_t>& prefix) const {
  struct ParseState {
    std::vector<GrammarSymbol> stack;
    size_t consumed = 0;
    bool operator==(const ParseState& o) const {
      return consumed == o.consumed && stack == o.stack;
    }
  };
  struct StateHash {
    size_t operator()(const ParseState& s) const {
      size_t h = std::hash<size_t>{}(s.consumed);
      for (const auto& sym : s.stack) {
        size_t sh;
        if (sym.kind == GrammarSymbol::Kind::Terminal) {
          sh = std::hash<uint32_t>{}(sym.terminal) ^ 0x9e3779b9u;
        } else {
          sh = std::hash<std::string>{}(sym.non_terminal);
        }
        h ^= sh + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      }
      return h;
    }
  };

  constexpr size_t kMaxStates = 20000;
  constexpr size_t kMaxStackLen = 256;

  std::deque<ParseState> queue;
  std::unordered_set<ParseState, StateHash> seen;

  ParseState initial;
  initial.stack.push_back(GrammarSymbol::make_non_terminal(start_));
  initial.consumed = 0;
  seen.insert(initial);
  queue.push_back(initial);

  while (!queue.empty()) {
    ParseState state = std::move(queue.front());
    queue.pop_front();

    if (state.consumed == prefix.size()) {
      return true;
    }
    if (seen.size() >= kMaxStates || state.stack.empty()) {
      continue;
    }

    std::vector<GrammarSymbol> next_stack = std::move(state.stack);
    GrammarSymbol symbol = next_stack.back();
    next_stack.pop_back();

    if (symbol.kind == GrammarSymbol::Kind::Terminal) {
      if (prefix[state.consumed] == symbol.terminal) {
        ParseState next;
        next.stack = next_stack;
        next.consumed = state.consumed + 1;
        if (seen.insert(next).second) {
          queue.push_back(std::move(next));
        }
      }
    } else {
      auto it = productions_.find(symbol.non_terminal);
      if (it == productions_.end()) {
        continue;
      }
      for (const auto& production : it->second) {
        std::vector<GrammarSymbol> expanded = next_stack;
        for (auto rit = production.rbegin(); rit != production.rend(); ++rit) {
          expanded.push_back(*rit);
        }
        if (expanded.size() > kMaxStackLen) {
          continue;
        }
        ParseState next;
        next.stack = std::move(expanded);
        next.consumed = state.consumed;
        if (seen.insert(next).second) {
          queue.push_back(std::move(next));
        }
      }
    }
  }

  return false;
}

// greedy

Token greedy(const Logits& logits) {
  if (logits.empty()) throw std::runtime_error("EmptyLogits");
  return static_cast<Token>(argmax_total_cmp(logits));
}

// Helper filters (operate on sorted-descending indexed_probs)

namespace {

using IndexedProb = std::pair<size_t, float>;

void apply_repetition_penalties(std::vector<float>& logits,
                                const std::vector<uint32_t>& recent_tokens,
                                const RepetitionPenaltyConfig& repetition) {
  if (logits.empty()) return;
  if (repetition.frequency_penalty == 0.0f &&
      repetition.presence_penalty == 0.0f &&
      !repetition.newline_penalty.has_value()) {
    return;
  }

  std::vector<uint32_t> frequencies(logits.size(), 0);
  for (uint32_t token : recent_tokens) {
    size_t idx = static_cast<size_t>(token);
    if (idx < frequencies.size()) {
      if (frequencies[idx] != std::numeric_limits<uint32_t>::max()) {
        frequencies[idx] += 1;  // saturating_add
      }
    }
  }

  for (size_t idx = 0; idx < logits.size(); ++idx) {
    uint32_t freq = frequencies[idx];
    if (freq == 0) continue;
    logits[idx] -= repetition.frequency_penalty * static_cast<float>(freq);
    logits[idx] -= repetition.presence_penalty;
  }

  if (repetition.newline_penalty.has_value()) {
    size_t idx = static_cast<size_t>(repetition.newline_penalty->token_id);
    if (idx < logits.size()) {
      logits[idx] -= repetition.newline_penalty->penalty;
    }
  }
}

void apply_typical_sampling(std::vector<IndexedProb>& indexed_probs,
                            float typical_p) {
  if (indexed_probs.empty()) return;
  float entropy = 0.0f;
  for (const auto& [idx, p] : indexed_probs) {
    (void)idx;
    float pp = std::max(p, kMinPositive);
    entropy += -pp * std::log(pp);
  }
  struct Typ {
    size_t idx;
    float prob;
    float dev;
  };
  std::vector<Typ> by_typicality;
  by_typicality.reserve(indexed_probs.size());
  for (const auto& [idx, prob] : indexed_probs) {
    float surprise = -std::log(std::max(prob, kMinPositive));
    by_typicality.push_back({idx, prob, std::abs(surprise - entropy)});
  }
  std::sort(by_typicality.begin(), by_typicality.end(),
            [](const Typ& a, const Typ& b) { return total_lt(a.dev, b.dev); });

  float cumulative = 0.0f;
  std::vector<IndexedProb> keep;
  keep.reserve(by_typicality.size());
  for (const auto& t : by_typicality) {
    keep.emplace_back(t.idx, t.prob);
    cumulative += t.prob;
    if (cumulative >= typical_p) break;
  }
  std::sort(keep.begin(), keep.end(),
            [](const IndexedProb& a, const IndexedProb& b) {
              return total_lt(b.second, a.second);
            });
  indexed_probs = std::move(keep);
}

void apply_tail_free_sampling(std::vector<IndexedProb>& indexed_probs,
                              float tail_free_z) {
  if (indexed_probs.size() <= 2) return;
  std::vector<float> second_derivative;
  second_derivative.reserve(indexed_probs.size() - 2);
  for (size_t i = 0; i + 2 < indexed_probs.size(); ++i) {
    float d1 = indexed_probs[i].second - indexed_probs[i + 1].second;
    float d2 = indexed_probs[i + 1].second - indexed_probs[i + 2].second;
    second_derivative.push_back(std::abs(d1 - d2));
  }
  float sd_sum = 0.0f;
  for (float v : second_derivative) sd_sum += v;
  if (sd_sum <= 0.0f || !is_finite(sd_sum)) return;

  float cumulative = 0.0f;
  size_t cutoff = indexed_probs.size();
  for (size_t i = 0; i < second_derivative.size(); ++i) {
    cumulative += second_derivative[i] / sd_sum;
    if (cumulative >= tail_free_z) {
      cutoff = std::max<size_t>(i + 2, 1);
      break;
    }
  }
  if (cutoff < indexed_probs.size()) indexed_probs.resize(cutoff);
}

void apply_locally_typical_sampling(std::vector<IndexedProb>& indexed_probs,
                                    float locally_typical_tau) {
  if (indexed_probs.empty()) return;
  float entropy = 0.0f;
  for (const auto& [idx, p] : indexed_probs) {
    (void)idx;
    float pp = std::max(p, kMinPositive);
    entropy += -pp * std::log(pp);
  }
  float deviation_limit = entropy * locally_typical_tau;
  std::vector<IndexedProb> filtered;
  for (const auto& ip : indexed_probs) {
    float surprise = -std::log(std::max(ip.second, kMinPositive));
    if (std::abs(surprise - entropy) <= deviation_limit) {
      filtered.push_back(ip);
    }
  }
  if (filtered.empty()) filtered.push_back(indexed_probs[0]);
  std::sort(filtered.begin(), filtered.end(),
            [](const IndexedProb& a, const IndexedProb& b) {
              return total_lt(b.second, a.second);
            });
  indexed_probs = std::move(filtered);
}

std::vector<float> softmax_probs(const std::vector<float>& logits,
                                 float temperature) {
  if (logits.empty()) throw std::runtime_error("EmptyLogits");
  float max_logit = max_total_cmp(logits);
  std::vector<float> probs;
  probs.reserve(logits.size());
  for (float logit : logits) {
    probs.push_back(std::exp((logit - max_logit) / temperature));
  }
  float sum = 0.0f;
  for (float p : probs) sum += p;
  if (sum <= 0.0f || !is_finite(sum)) throw std::runtime_error("EmptyLogits");
  for (float& p : probs) p /= sum;
  return probs;
}

std::vector<IndexedProb> build_sorted_probs(const std::vector<float>& logits,
                                            float temperature) {
  if (logits.empty()) throw std::runtime_error("EmptyLogits");
  float max_logit = max_total_cmp(logits);
  std::vector<IndexedProb> indexed_probs;
  indexed_probs.reserve(logits.size());
  for (size_t idx = 0; idx < logits.size(); ++idx) {
    indexed_probs.emplace_back(
        idx, std::exp((logits[idx] - max_logit) / temperature));
  }
  float raw_sum = 0.0f;
  for (const auto& ip : indexed_probs) raw_sum += ip.second;
  if (raw_sum <= 0.0f || !is_finite(raw_sum))
    throw std::runtime_error("EmptyLogits");
  for (auto& ip : indexed_probs) ip.second /= raw_sum;
  std::sort(indexed_probs.begin(), indexed_probs.end(),
            [](const IndexedProb& a, const IndexedProb& b) {
              return total_lt(b.second, a.second);
            });
  return indexed_probs;
}

std::optional<IndexedProb> weighted_pick(
    const std::vector<IndexedProb>& indexed_probs, float random) {
  if (indexed_probs.empty()) return std::nullopt;
  float filtered_sum = 0.0f;
  for (const auto& ip : indexed_probs) filtered_sum += ip.second;
  if (filtered_sum <= 0.0f || !is_finite(filtered_sum)) return std::nullopt;

  float cumulative = 0.0f;
  float target = random * filtered_sum;
  for (const auto& ip : indexed_probs) {
    cumulative += ip.second;
    if (target <= cumulative) return ip;
  }
  return indexed_probs.back();
}

std::vector<float> residual_probs(const std::vector<float>& target_probs,
                                  const std::vector<float>& draft_probs) {
  std::vector<float> residual;
  residual.reserve(target_probs.size());
  size_t n = std::min(target_probs.size(), draft_probs.size());
  for (size_t i = 0; i < n; ++i) {
    residual.push_back(std::max(target_probs[i] - draft_probs[i], 0.0f));
  }
  float sum = 0.0f;
  for (float r : residual) sum += r;
  if (sum > 0.0f && is_finite(sum)) {
    for (float& r : residual) r /= sum;
  }
  return residual;
}

size_t sample_probabilities(const std::vector<float>& probs, float random) {
  if (!is_finite(random) || !(random >= 0.0f && random < 1.0f))
    throw std::runtime_error("InvalidRandom");
  if (probs.empty()) throw std::runtime_error("EmptyLogits");
  float sum = 0.0f;
  for (float p : probs) sum += p;
  if (sum <= 0.0f || !is_finite(sum)) {
    return argmax_total_cmp(probs);
  }
  float cumulative = 0.0f;
  float target = random * sum;
  for (size_t idx = 0; idx < probs.size(); ++idx) {
    cumulative += probs[idx];
    if (target <= cumulative) return idx;
  }
  return probs.size() - 1;
}

std::optional<Token> sample_unfiltered(const std::vector<float>& logits,
                                       float temperature, float random) {
  // Returns nullopt to signal "fall back to greedy(logits)".
  float max_logit = max_total_cmp(logits);
  float raw_sum = 0.0f;
  for (float logit : logits) {
    raw_sum += std::exp((logit - max_logit) / temperature);
  }
  if (raw_sum <= 0.0f || !is_finite(raw_sum)) return std::nullopt;

  float target = random * raw_sum;
  float cumulative = 0.0f;
  for (size_t idx = 0; idx < logits.size(); ++idx) {
    cumulative += std::exp((logits[idx] - max_logit) / temperature);
    if (target <= cumulative) return static_cast<Token>(idx);
  }
  return std::nullopt;
}

}  // namespace

// sample family

Token sample(const Logits& logits, const SamplerConfig& config, float random) {
  return sample_with_repetition(logits, config, random, {},
                                RepetitionPenaltyConfig{});
}

Token sample_with_repetition(const Logits& logits, const SamplerConfig& config,
                             float random,
                             const std::vector<uint32_t>& recent_tokens,
                             const RepetitionPenaltyConfig& repetition) {
  return sample_with_repetition_and_grammar(logits, config, random,
                                            recent_tokens, repetition, {},
                                            nullptr);
}

Token sample_with_repetition_and_grammar(
    const Logits& logits, const SamplerConfig& config, float random,
    const std::vector<uint32_t>& recent_tokens,
    const RepetitionPenaltyConfig& repetition,
    const std::vector<uint32_t>& generated_tokens,
    const GrammarConstraint* grammar) {
  if (logits.empty()) throw std::runtime_error("EmptyLogits");
  if (!is_finite(config.temperature))
    throw std::runtime_error("InvalidTemperature");
  if (config.top_k.has_value() && config.top_k.value() == 0)
    throw std::runtime_error("InvalidTopK");
  if (config.top_p.has_value()) {
    float v = config.top_p.value();
    if (!is_finite(v) || v <= 0.0f || v > 1.0f)
      throw std::runtime_error("InvalidTopP");
  }
  if (config.min_p.has_value()) {
    float v = config.min_p.value();
    if (!is_finite(v) || v <= 0.0f || v > 1.0f)
      throw std::runtime_error("InvalidMinP");
  }
  if (config.typical_p.has_value()) {
    float v = config.typical_p.value();
    if (!is_finite(v) || v <= 0.0f || v > 1.0f)
      throw std::runtime_error("InvalidTypicalP");
  }
  if (config.tail_free_z.has_value()) {
    float v = config.tail_free_z.value();
    if (!is_finite(v) || v <= 0.0f || v > 1.0f)
      throw std::runtime_error("InvalidTailFreeZ");
  }
  if (config.locally_typical_tau.has_value()) {
    float v = config.locally_typical_tau.value();
    if (!is_finite(v) || v <= 0.0f)
      throw std::runtime_error("InvalidLocallyTypicalTau");
  }
  if (!is_finite(repetition.frequency_penalty) ||
      repetition.frequency_penalty < 0.0f)
    throw std::runtime_error("InvalidFrequencyPenalty");
  if (!is_finite(repetition.presence_penalty) ||
      repetition.presence_penalty < 0.0f)
    throw std::runtime_error("InvalidPresencePenalty");
  if (repetition.newline_penalty.has_value()) {
    float p = repetition.newline_penalty->penalty;
    if (!is_finite(p) || p < 0.0f)
      throw std::runtime_error("InvalidNewlinePenalty");
  }
  if (!is_finite(random) || !(random >= 0.0f && random < 1.0f))
    throw std::runtime_error("InvalidRandom");

  bool has_repetition_penalty =
      repetition.frequency_penalty != 0.0f ||
      repetition.presence_penalty != 0.0f ||
      repetition.newline_penalty.has_value();

  bool top_k_is_one = config.top_k.has_value() && config.top_k.value() == 1;

  if ((config.temperature <= 0.0f || top_k_is_one) && !has_repetition_penalty &&
      grammar == nullptr) {
    return greedy(logits);
  }
  if (config.temperature <= 0.0f) throw std::runtime_error("InvalidTemperature");

  bool has_rank_filter =
      config.top_k.has_value() || config.top_p.has_value() ||
      config.min_p.has_value() || config.typical_p.has_value() ||
      config.tail_free_z.has_value() || config.locally_typical_tau.has_value();

  if (logits.size() >= 4096 && !has_repetition_penalty && !has_rank_filter &&
      grammar == nullptr) {
    auto r = sample_unfiltered(logits, config.temperature, random);
    if (r.has_value()) return r.value();
    return greedy(logits);
  }

  std::vector<float> adjusted_logits = logits;
  apply_repetition_penalties(adjusted_logits, recent_tokens, repetition);

  float max_logit = max_total_cmp(adjusted_logits);

  // top_k_limit = config.top_k.filter(|k| k < len)
  std::optional<size_t> top_k_limit;
  if (config.top_k.has_value() && config.top_k.value() < adjusted_logits.size())
    top_k_limit = config.top_k;

  std::vector<IndexedProb> indexed_probs;

  if (top_k_limit.has_value()) {
    size_t top_k = top_k_limit.value();
    float raw_sum = 0.0f;
    std::vector<IndexedProb> top_candidates;
    top_candidates.reserve(top_k);
    for (size_t idx = 0; idx < adjusted_logits.size(); ++idx) {
      float prob = std::exp((adjusted_logits[idx] - max_logit) /
                            config.temperature);
      raw_sum += prob;
      if (top_candidates.size() < top_k) {
        top_candidates.emplace_back(idx, prob);
      } else {
        // find min by total_cmp over prob; Rust min_by returns first minimum.
        size_t min_idx = 0;
        for (size_t j = 1; j < top_candidates.size(); ++j) {
          if (total_cmp(top_candidates[j].second,
                        top_candidates[min_idx].second) < 0) {
            min_idx = j;
          }
        }
        if (prob > top_candidates[min_idx].second) {
          top_candidates[min_idx] = {idx, prob};
        }
      }
    }
    if (raw_sum <= 0.0f || !is_finite(raw_sum)) return greedy(logits);
    for (auto& c : top_candidates) c.second /= raw_sum;
    std::sort(top_candidates.begin(), top_candidates.end(),
              [](const IndexedProb& a, const IndexedProb& b) {
                return total_lt(b.second, a.second);
              });
    indexed_probs = std::move(top_candidates);
  } else {
    indexed_probs.reserve(adjusted_logits.size());
    for (size_t idx = 0; idx < adjusted_logits.size(); ++idx) {
      indexed_probs.emplace_back(
          idx, std::exp((adjusted_logits[idx] - max_logit) /
                        config.temperature));
    }
    float raw_sum = 0.0f;
    for (const auto& ip : indexed_probs) raw_sum += ip.second;
    if (raw_sum <= 0.0f || !is_finite(raw_sum)) return greedy(logits);
    for (auto& ip : indexed_probs) ip.second /= raw_sum;
    std::sort(indexed_probs.begin(), indexed_probs.end(),
              [](const IndexedProb& a, const IndexedProb& b) {
                return total_lt(b.second, a.second);
              });
    if (config.top_k.has_value() &&
        indexed_probs.size() > config.top_k.value()) {
      indexed_probs.resize(config.top_k.value());
    }
  }

  if (config.top_p.has_value()) {
    float top_p = config.top_p.value();
    float cumulative = 0.0f;
    size_t cutoff = indexed_probs.size();
    for (size_t i = 0; i < indexed_probs.size(); ++i) {
      cumulative += indexed_probs[i].second;
      if (cumulative >= top_p) {
        cutoff = i + 1;
        break;
      }
    }
    if (cutoff < indexed_probs.size()) indexed_probs.resize(cutoff);
  }

  if (config.min_p.has_value()) {
    float max_prob = indexed_probs.empty() ? 0.0f : indexed_probs.front().second;
    float threshold = max_prob * config.min_p.value();
    indexed_probs.erase(
        std::remove_if(indexed_probs.begin(), indexed_probs.end(),
                       [&](const IndexedProb& ip) {
                         return !(ip.second >= threshold);
                       }),
        indexed_probs.end());
  }

  if (config.typical_p.has_value())
    apply_typical_sampling(indexed_probs, config.typical_p.value());

  if (config.tail_free_z.has_value())
    apply_tail_free_sampling(indexed_probs, config.tail_free_z.value());

  if (config.locally_typical_tau.has_value())
    apply_locally_typical_sampling(indexed_probs,
                                   config.locally_typical_tau.value());

  if (grammar != nullptr) {
    indexed_probs.erase(
        std::remove_if(indexed_probs.begin(), indexed_probs.end(),
                       [&](const IndexedProb& ip) {
                         return !grammar->allows_token(
                             generated_tokens, static_cast<uint32_t>(ip.first));
                       }),
        indexed_probs.end());
  }

  if (indexed_probs.empty()) {
    if (grammar != nullptr) throw std::runtime_error("NoValidGrammarToken");
    return greedy(logits);
  }

  float filtered_sum = 0.0f;
  for (const auto& ip : indexed_probs) filtered_sum += ip.second;
  if (filtered_sum <= 0.0f || !is_finite(filtered_sum)) return greedy(logits);

  float cumulative = 0.0f;
  float target = random * filtered_sum;
  for (const auto& ip : indexed_probs) {
    cumulative += ip.second;
    if (target <= cumulative) return static_cast<Token>(ip.first);
  }

  return greedy(logits);
}

// mirostat

std::pair<Token, float> sample_mirostat(const Logits& logits, float temperature,
                                        const MirostatConfig& config,
                                        float random) {
  if (logits.empty()) throw std::runtime_error("EmptyLogits");
  if (!is_finite(temperature) || temperature <= 0.0f)
    throw std::runtime_error("InvalidTemperature");
  if (!is_finite(config.tau) || config.tau <= 0.0f || !is_finite(config.eta) ||
      config.eta <= 0.0f || !is_finite(config.mu))
    throw std::runtime_error("InvalidMirostat");
  if (!is_finite(random) || !(random >= 0.0f && random < 1.0f))
    throw std::runtime_error("InvalidRandom");

  std::vector<IndexedProb> indexed_probs = build_sorted_probs(logits, temperature);
  float target_surprisal = config.mu;
  // Rust sort_by is stable.
  std::stable_sort(
      indexed_probs.begin(), indexed_probs.end(),
      [&](const IndexedProb& a, const IndexedProb& b) {
        float a_surprise = -std::log(std::max(a.second, kMinPositive));
        float b_surprise = -std::log(std::max(b.second, kMinPositive));
        return total_lt(std::abs(a_surprise - target_surprisal),
                        std::abs(b_surprise - target_surprisal));
      });

  auto chosen = weighted_pick(indexed_probs, random);
  if (!chosen.has_value()) throw std::runtime_error("EmptyLogits");
  float observed_surprisal = -std::log(std::max(chosen->second, kMinPositive));
  float updated_mu = config.mu - config.eta * (observed_surprisal - config.tau);
  return {static_cast<Token>(chosen->first), updated_mu};
}

// speculative_decode

SpeculativeDecodeResult speculative_decode(
    const std::vector<uint32_t>& draft_tokens,
    const std::vector<Logits>& draft_logits,
    const std::vector<Logits>& target_logits, const SamplerConfig& config,
    const std::vector<float>& randoms) {
  if (draft_tokens.empty() || draft_logits.size() != draft_tokens.size() ||
      target_logits.size() != draft_tokens.size() + 1 ||
      randoms.size() < draft_tokens.size() + 1) {
    throw std::runtime_error("InvalidSpeculativeInputs");
  }

  bool greedy_mode = config.temperature <= 0.0f ||
                     (config.top_k.has_value() && config.top_k.value() == 1);
  float verify_temperature = greedy_mode ? 1.0f : config.temperature;

  std::vector<uint32_t> emitted;
  emitted.reserve(draft_tokens.size() + 1);

  for (size_t step = 0; step < draft_tokens.size(); ++step) {
    uint32_t draft_token = draft_tokens[step];

    if (greedy_mode) {
      if (target_logits[step].empty())
        throw std::runtime_error("InvalidSpeculativeInputs");
      uint32_t target_argmax =
          static_cast<uint32_t>(argmax_total_cmp(target_logits[step]));
      if (draft_token == target_argmax) {
        emitted.push_back(draft_token);
        continue;
      }
      emitted.push_back(target_argmax);
      return {emitted, step, true};
    }

    std::vector<float> draft_probs =
        softmax_probs(draft_logits[step], verify_temperature);
    std::vector<float> target_probs =
        softmax_probs(target_logits[step], verify_temperature);
    if (draft_probs.size() != target_probs.size())
      throw std::runtime_error("InvalidSpeculativeInputs");
    size_t token_idx = static_cast<size_t>(draft_token);
    if (token_idx >= draft_probs.size())
      throw std::runtime_error("InvalidSpeculativeInputs");
    float q = std::max(draft_probs[token_idx], kMinPositive);
    float p = target_probs[token_idx];
    float accept_prob = std::min(p / q, 1.0f);

    if (randoms[step] <= accept_prob) {
      emitted.push_back(draft_token);
      continue;
    }

    std::vector<float> residual = residual_probs(target_probs, draft_probs);
    size_t sampled = sample_probabilities(residual, randoms[step]);
    emitted.push_back(static_cast<uint32_t>(sampled));
    return {emitted, step, true};
  }

  Token final_token = sample(target_logits[draft_tokens.size()], config,
                             randoms[draft_tokens.size()]);
  emitted.push_back(final_token);
  return {emitted, draft_tokens.size(), false};
}

// beam_search

BeamSearchResult beam_search(const std::vector<Logits>& logits_per_step,
                             size_t beam_width,
                             std::optional<uint32_t> eos_token) {
  if (beam_width == 0) throw std::runtime_error("InvalidBeamWidth");
  if (logits_per_step.empty())
    throw std::runtime_error("InvalidBeamSearchInputs");
  for (const auto& step : logits_per_step) {
    if (step.empty()) throw std::runtime_error("InvalidBeamSearchInputs");
  }

  struct Beam {
    std::vector<uint32_t> tokens;
    float score = 0.0f;
    bool finished = false;
  };

  std::vector<Beam> beams;
  beams.push_back(Beam{});

  for (const auto& step_logits : logits_per_step) {
    std::vector<float> probs = softmax_probs(step_logits, 1.0f);
    std::vector<Beam> candidates;

    for (const auto& beam : beams) {
      if (beam.finished) {
        candidates.push_back(beam);
        continue;
      }
      for (size_t token_idx = 0; token_idx < probs.size(); ++token_idx) {
        float prob = probs[token_idx];
        if (prob <= 0.0f || !is_finite(prob)) continue;
        Beam nb;
        nb.tokens = beam.tokens;
        nb.tokens.push_back(static_cast<uint32_t>(token_idx));
        nb.score = beam.score + std::log(prob);
        nb.finished =
            eos_token.has_value() &&
            eos_token.value() == static_cast<uint32_t>(token_idx);
        candidates.push_back(std::move(nb));
      }
    }

    if (candidates.empty()) throw std::runtime_error("EmptyLogits");

    std::sort(candidates.begin(), candidates.end(),
              [](const Beam& a, const Beam& b) {
                return total_lt(b.score, a.score);
              });
    if (candidates.size() > beam_width) candidates.resize(beam_width);
    beams = std::move(candidates);

    bool all_finished = true;
    for (const auto& b : beams) {
      if (!b.finished) {
        all_finished = false;
        break;
      }
    }
    if (all_finished) break;
  }

  // max_by(total_cmp) over beams -> last maximum on ties.
  size_t best = 0;
  for (size_t i = 1; i < beams.size(); ++i) {
    if (total_cmp(beams[i].score, beams[best].score) >= 0) best = i;
  }
  return {beams[best].tokens, beams[best].score};
}

// Rng: SplitMix64-seeded xoshiro256** -> f32 in [0,1)

namespace {
inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
}  // namespace

Rng::Rng(uint64_t seed) {
  uint64_t z = seed;
  for (int i = 0; i < 4; ++i) {
    z += 0x9e3779b97f4a7c15ULL;
    uint64_t r = z;
    r = (r ^ (r >> 30)) * 0xbf58476d1ce4e5b9ULL;
    r = (r ^ (r >> 27)) * 0x94d049bb133111ebULL;
    r = r ^ (r >> 31);
    s_[i] = r;
  }
}

uint64_t Rng::next_u64() {
  uint64_t result = rotl(s_[1] * 5, 7) * 9;
  uint64_t t = s_[1] << 17;
  s_[2] ^= s_[0];
  s_[3] ^= s_[1];
  s_[1] ^= s_[2];
  s_[0] ^= s_[3];
  s_[2] ^= t;
  s_[3] = rotl(s_[3], 45);
  return result;
}

float Rng::next_unit() {
  // 24 high bits -> [0,1) with single-precision resolution.
  uint32_t bits = static_cast<uint32_t>(next_u64() >> 40);  // 24 bits
  return static_cast<float>(bits) * (1.0f / 16777216.0f);   // /2^24
}

}  // namespace oxidize
