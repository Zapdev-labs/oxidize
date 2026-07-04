#include "oxidize/rotor_quant.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

void require(bool cond, const char* msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

float lcg(uint64_t& state) {
  state = state * 6364136223846793005ull + 1442695040888963407ull;
  return static_cast<float>(state >> 40) / static_cast<float>(1u << 24) * 2.0f -
         1.0f;
}

void test_rotate_roundtrip_is_identity() {
  oxidize::RotorQuantConfig cfg;
  cfg.head_dim = 32;
  oxidize::RotorQuantCache cache(cfg);
  uint64_t rng = 7;
  std::vector<float> v(cfg.head_dim);
  for (auto& x : v) x = lcg(rng);
  const auto back = cache.unrotate(cache.rotate(v));
  for (size_t i = 0; i < v.size(); ++i) {
    require(std::fabs(back[i] - v[i]) < 1.0e-5f,
            "rotor roundtrip should be identity");
  }
  // Rotation preserves norm (orthogonality).
  const auto r = cache.rotate(v);
  float n0 = 0.0f, n1 = 0.0f;
  for (size_t i = 0; i < v.size(); ++i) {
    n0 += v[i] * v[i];
    n1 += r[i] * r[i];
  }
  require(std::fabs(n0 - n1) < 1.0e-4f * n0, "rotor should preserve norm");
}

void test_logits_match_f32_reference() {
  oxidize::RotorQuantConfig cfg;
  cfg.head_dim = 64;
  oxidize::RotorQuantCache cache(cfg);
  const size_t tokens = 16;
  uint64_t rng = 42;
  std::vector<float> keys(tokens * cfg.head_dim), values(tokens * cfg.head_dim),
      query(cfg.head_dim);
  for (auto& x : keys) x = lcg(rng);
  for (auto& x : values) x = lcg(rng);
  for (auto& x : query) x = lcg(rng);
  cache.store_page(0, 0, keys, values, tokens);

  const auto logits = cache.logits(0, 0, query);
  require(logits.size() == tokens, "one logit per token");
  for (size_t t = 0; t < tokens; ++t) {
    float ref = 0.0f;
    float knorm = 0.0f;
    for (size_t i = 0; i < cfg.head_dim; ++i) {
      ref += query[i] * keys[t * cfg.head_dim + i];
      knorm += keys[t * cfg.head_dim + i] * keys[t * cfg.head_dim + i];
    }
    // int4 with per-32 scale: relative error on the dot is small.
    const float tol = 0.15f * std::sqrt(knorm);
    if (std::fabs(logits[t] - ref) > tol) {
      std::fprintf(stderr, "logit %zu: %f vs ref %f (tol %f)\n", t, logits[t],
                   ref, tol);
      throw std::runtime_error("quantized logits should track f32 reference");
    }
  }
}

void test_attention_matches_f32_reference() {
  oxidize::RotorQuantConfig cfg;
  cfg.head_dim = 64;
  oxidize::RotorQuantCache cache(cfg);
  const size_t tokens = 32;
  uint64_t rng = 99;
  std::vector<float> keys(tokens * cfg.head_dim), values(tokens * cfg.head_dim),
      query(cfg.head_dim);
  for (auto& x : keys) x = lcg(rng);
  for (auto& x : values) x = lcg(rng);
  for (auto& x : query) x = lcg(rng);
  cache.store_page(0, 0, keys, values, tokens);

  // f32 reference attention
  std::vector<float> scores(tokens);
  const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
  float max_s = -1.0e30f;
  for (size_t t = 0; t < tokens; ++t) {
    float dot = 0.0f;
    for (size_t i = 0; i < cfg.head_dim; ++i) {
      dot += query[i] * keys[t * cfg.head_dim + i];
    }
    scores[t] = dot * scale;
    max_s = std::max(max_s, scores[t]);
  }
  float z = 0.0f;
  for (auto& s : scores) {
    s = std::exp(s - max_s);
    z += s;
  }
  std::vector<float> ref(cfg.head_dim, 0.0f);
  for (size_t t = 0; t < tokens; ++t) {
    for (size_t i = 0; i < cfg.head_dim; ++i) {
      ref[i] += scores[t] / z * values[t * cfg.head_dim + i];
    }
  }

  const auto out = cache.attention(0, 0, query);
  require(out.size() == cfg.head_dim, "attention output should match head_dim");
  for (size_t i = 0; i < cfg.head_dim; ++i) {
    require(std::fabs(out[i] - ref[i]) < 0.1f,
            "quantized attention should track f32 reference");
  }
}

void test_stats_report_compression() {
  oxidize::RotorQuantConfig cfg;
  cfg.head_dim = 128;
  oxidize::RotorQuantCache cache(cfg);
  const size_t tokens = 64;
  std::vector<float> keys(tokens * cfg.head_dim, 0.5f),
      values(tokens * cfg.head_dim, -0.25f);
  cache.store_page(0, 0, keys, values, tokens);
  const auto st = cache.stats();
  require(st.token_count == tokens, "token count should be tracked");
  require(st.total_bits_per_coord > 4.0f && st.total_bits_per_coord < 6.0f,
          "int4 + scales should land near 5 bits per coordinate");
  require(st.compression_ratio_vs_f32() > 6.0f,
          "rotorquant should compress > 6x vs f32");
}

}

int main() {
  test_rotate_roundtrip_is_identity();
  test_logits_match_f32_reference();
  test_attention_matches_f32_reference();
  test_stats_report_compression();
  std::puts("rotor quant tests passed");
  return 0;
}
