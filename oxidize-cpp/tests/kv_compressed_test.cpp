#include "oxidize/kv_compressed.hpp"

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

std::vector<float> rope_ref(const std::vector<float>& row, size_t pos,
                            size_t head_dim, float theta) {
  std::vector<float> out(row.size());
  for (size_t p = 0; p < head_dim / 2; ++p) {
    const float freq = std::pow(
        theta, -2.0f * static_cast<float>(p) / static_cast<float>(head_dim));
    const float a = freq * static_cast<float>(pos);
    const float c = std::cos(a);
    const float s = std::sin(a);
    out[2 * p] = row[2 * p] * c - row[2 * p + 1] * s;
    out[2 * p + 1] = row[2 * p] * s + row[2 * p + 1] * c;
  }
  return out;
}

void test_default_scheme_is_rotorquant() {
  oxidize::CompressedKvCache cache(128);
  require(cache.scheme() == oxidize::KvScheme::RotorQuant,
          "default scheme should be RotorQuant");
  require(cache.rotor() != nullptr && cache.helix() == nullptr,
          "default cache should be rotor-backed");
}

void test_both_schemes_track_f32_reference() {
  const size_t head_dim = 128;
  const size_t tokens = 64;
  const float theta = 10000.0f;
  uint64_t rng = 31;
  std::vector<float> keys(tokens * head_dim), values(tokens * head_dim),
      query(head_dim);
  std::vector<size_t> positions;
  for (auto& x : keys) x = lcg(rng);
  for (auto& x : values) x = lcg(rng);
  for (auto& x : query) x = lcg(rng);
  for (size_t t = 0; t < tokens; ++t) positions.push_back(t);
  const size_t qpos = tokens;

  // f32 reference: standard RoPE attention.
  std::vector<float> scores(tokens);
  const std::vector<float> rq = rope_ref(query, qpos, head_dim, theta);
  float max_s = -1.0e30f;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (size_t t = 0; t < tokens; ++t) {
    std::vector<float> row(keys.begin() + t * head_dim,
                           keys.begin() + (t + 1) * head_dim);
    const std::vector<float> rk = rope_ref(row, t, head_dim, theta);
    float dot = 0.0f;
    for (size_t i = 0; i < head_dim; ++i) dot += rq[i] * rk[i];
    scores[t] = dot * scale;
    max_s = std::max(max_s, scores[t]);
  }
  float z = 0.0f;
  for (auto& s : scores) {
    s = std::exp(s - max_s);
    z += s;
  }
  std::vector<float> ref(head_dim, 0.0f);
  for (size_t t = 0; t < tokens; ++t) {
    for (size_t i = 0; i < head_dim; ++i) {
      ref[i] += scores[t] / z * values[t * head_dim + i];
    }
  }

  for (auto scheme :
       {oxidize::KvScheme::RotorQuant, oxidize::KvScheme::Helix}) {
    oxidize::CompressedKvCache cache(head_dim, scheme, tokens, theta);
    cache.store_page(0, 0, keys, values, positions);
    const auto out = cache.attention(0, 0, query, qpos);
    require(out.size() == head_dim, "attention output size");
    // Helix values are 3-bit (vs rotor's 4-bit), so its noise floor is higher.
    const float tol = scheme == oxidize::KvScheme::Helix ? 0.25f : 0.1f;
    for (size_t i = 0; i < head_dim; ++i) {
      if (std::fabs(out[i] - ref[i]) > tol) {
        std::fprintf(stderr, "scheme %d dim %zu: %f vs ref %f\n",
                     static_cast<int>(scheme), i, out[i], ref[i]);
        throw std::runtime_error("compressed attention should track f32");
      }
    }
    require(cache.compression_ratio_vs_f32() > 6.0f,
            "compression should exceed 6x");
  }
}

}

int main() {
  test_default_scheme_is_rotorquant();
  test_both_schemes_track_f32_reference();
  std::puts("kv compressed tests passed");
  return 0;
}
