#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace oxidize {

enum class HelixPageTier {
  Cold,
  Hot,
};

struct HelixCacheConfig {
  size_t page_size = 64;
  size_t head_dim = 128;
  uint8_t key_radius_bits = 4;
  uint8_t key_phase_bits = 4;
  uint8_t value_bits = 3;
  float inactive_threshold = 0.0f;
  float promotion_epsilon = 0.1f;
  uint32_t promotion_budget = 3;
};

struct HelixCacheStats {
  size_t cold_pages = 0;
  size_t hot_pages = 0;
  size_t token_count = 0;
  size_t key_bytes = 0;
  size_t value_bytes = 0;
  size_t hot_bytes = 0;
  size_t metadata_bytes = 0;
  size_t key_metadata_bytes = 0;
  size_t value_metadata_bytes = 0;
  size_t page_metadata_bytes = 0;
  size_t f32_baseline_bytes = 0;
  float key_bits_per_coord = 0.0f;
  float value_bits_per_coord = 0.0f;
  float total_bits_per_coord = 0.0f;

  float compression_ratio_vs_f32() const;
};

// Read-only view of a stored cold page, for GPU backends to upload without
// re-quantizing. Pointers are valid until the cache is destroyed.
struct HelixColdPageView {
  size_t layer = 0;
  size_t kv_head = 0;
  size_t tokens = 0;
  const size_t* positions = nullptr;
  const uint8_t* key_codes = nullptr;    // 1 byte/pair: rho<<4 | phi
  const uint8_t* active_mask = nullptr;  // 1 bit/pair
  const float* mu_phi = nullptr;         // per pair
  const float* log_rho_min = nullptr;
  const float* log_rho_step = nullptr;
  const uint8_t* value_codes = nullptr;  // 3-bit packed
  const float* value_scales = nullptr;   // per 8-group
};

class HelixCache {
 public:
  explicit HelixCache(HelixCacheConfig config);
  ~HelixCache();
  HelixCache(HelixCache&&) noexcept;
  HelixCache& operator=(HelixCache&&) noexcept;
  HelixCache(const HelixCache&) = delete;
  HelixCache& operator=(const HelixCache&) = delete;

  void store_cold_page(size_t layer, size_t kv_head, size_t page_id,
                       const std::vector<float>& pre_rope_keys,
                       const std::vector<float>& values,
                       const std::vector<size_t>& positions);
  void store_hot_page(size_t layer, size_t kv_head, size_t page_id,
                      const std::vector<float>& pre_rope_keys,
                      const std::vector<float>& values,
                      const std::vector<size_t>& positions);

  std::vector<float> logits(size_t layer, size_t kv_head,
                            const std::vector<float>& query_pre_rope,
                            size_t query_position, float rope_theta) const;
  std::vector<float> attention(size_t layer, size_t kv_head,
                               const std::vector<float>& query_pre_rope,
                               size_t query_position, float rope_theta);

  // Enumerate cold pages for GPU upload. Returns false when index is past the
  // last cold page or points at a hot page.
  size_t page_count() const;
  bool cold_page_view(size_t index, HelixColdPageView* view) const;

  void bump_uncertainty(size_t layer, size_t kv_head, size_t page_id,
                        float interval_overlap);
  bool should_promote(size_t layer, size_t kv_head, size_t page_id) const;
  HelixCacheStats stats() const;

 private:
  struct Impl;
  HelixCacheConfig config_;
  std::vector<std::unique_ptr<Impl>> pages_;
};

}
