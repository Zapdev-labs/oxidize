#include "oxidize/helix_cache.hpp"

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

void require_close(float actual, float expected, float tol, const char* msg) {
  if (std::fabs(actual - expected) > tol) {
    std::fprintf(stderr, "%s: actual=%f expected=%f tol=%f\n", msg, actual,
                 expected, tol);
    throw std::runtime_error(msg);
  }
}

void test_cold_page_attention_matches_rope_polar_reference() {
  oxidize::HelixCacheConfig cfg;
  cfg.page_size = 4;
  cfg.head_dim = 8;
  cfg.inactive_threshold = 0.05f;

  oxidize::HelixCache cache(cfg);
  const std::vector<float> keys = {
      1.0f, 0.0f, 0.02f, 0.01f, 2.0f, 0.0f, 0.0f, 0.0f,
      1.0f, 0.0f, 0.03f, 0.01f, 2.0f, 0.0f, 0.0f, 0.0f,
      1.0f, 0.0f, 0.01f, 0.01f, 2.0f, 0.0f, 0.0f, 0.0f,
      1.0f, 0.0f, 0.02f, 0.01f, 2.0f, 0.0f, 0.0f, 0.0f,
  };
  const std::vector<float> values = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
      1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f,
      2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
      2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f,
  };
  cache.store_cold_page(0, 0, 0, keys, values, {0, 1, 2, 3});

  const std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f,
                                    0.5f, 0.0f, 0.0f, 0.0f};
  const std::vector<float> logits = cache.logits(0, 0, query, 3, 10000.0f);
  require(logits.size() == 4, "cold logits should cover every token");
  for (size_t t = 0; t < logits.size(); ++t) {
    const float rel = static_cast<float>(t) - 3.0f;
    const float expected = std::cos(rel) + std::cos(0.01f * rel);
    require_close(logits[t], expected, 0.001f,
                  "quantized polar logits should preserve active pairs");
  }

  const std::vector<float> out = cache.attention(0, 0, query, 3, 10000.0f);
  require(out.size() == cfg.head_dim, "attention output should match head_dim");
  require_close(out[0], 1.75f, 0.35f,
                "Hadamard int3 value path should reconstruct weighted output");

  const oxidize::HelixCacheStats stats = cache.stats();
  require(stats.cold_pages == 1, "one cold page should be accounted");
  require(stats.key_bits_per_coord > 0.0f, "key bpc should include metadata");
  require(stats.value_bits_per_coord > 3.0f,
          "value bpc should include Hadamard scale metadata");
  require(stats.compression_ratio_vs_f32() > 1.0f,
          "Helix cold cache should compress versus FP32 KV");
}

void test_rejects_malformed_dimensions() {
  oxidize::HelixCacheConfig cfg;
  cfg.page_size = 4;
  cfg.head_dim = 7;
  bool rejected = false;
  try {
    oxidize::HelixCache cache(cfg);
    (void)cache;
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "odd head_dim should be rejected");
}

void test_promotion_state_marks_uncertain_pages() {
  oxidize::HelixCacheConfig cfg;
  cfg.page_size = 2;
  cfg.head_dim = 8;
  cfg.promotion_budget = 2;
  oxidize::HelixCache cache(cfg);
  const std::vector<float> keys = {
      1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
      1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
  };
  const std::vector<float> values(16, 1.0f);
  cache.store_cold_page(1, 2, 3, keys, values, {8, 9});
  cache.bump_uncertainty(1, 2, 3, 0.25f);
  require(!cache.should_promote(1, 2, 3),
          "first uncertainty hit should not promote");
  cache.bump_uncertainty(1, 2, 3, 0.25f);
  require(cache.should_promote(1, 2, 3),
          "budgeted uncertainty hits should request promotion");
}

}

int main() {
  test_cold_page_attention_matches_rope_polar_reference();
  test_rejects_malformed_dimensions();
  test_promotion_state_marks_uncertain_pages();
  std::puts("helix cache tests passed");
  return 0;
}
