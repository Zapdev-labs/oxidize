#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "oxidize/helix_cache.hpp"
#include "oxidize/rotor_quant.hpp"

namespace oxidize {

// Compressed KV-cache scheme selection. RotorQuant is the default: fastest
// decode on CPU (AVX-512) and GPU (Vulkan), 6.4x compression at int4.
// Helix trades a little decode speed for better compression (7.26x).
enum class KvScheme {
  RotorQuant,
  Helix,
};

constexpr KvScheme kDefaultKvScheme = KvScheme::RotorQuant;

// Unified front over HelixCache / RotorQuantCache. Callers always pass
// pre-RoPE keys/queries plus positions; the facade applies RoPE where the
// scheme needs it (RotorQuant stores post-RoPE keys, Helix rotates at decode).
class CompressedKvCache {
 public:
  explicit CompressedKvCache(size_t head_dim,
                             KvScheme scheme = kDefaultKvScheme,
                             size_t page_size = 64,
                             float rope_theta = 10000.0f);
  ~CompressedKvCache();
  CompressedKvCache(CompressedKvCache&&) noexcept;
  CompressedKvCache& operator=(CompressedKvCache&&) noexcept;

  KvScheme scheme() const { return scheme_; }

  void store_page(size_t layer, size_t kv_head,
                  const std::vector<float>& pre_rope_keys,
                  const std::vector<float>& values,
                  const std::vector<size_t>& positions);

  std::vector<float> attention(size_t layer, size_t kv_head,
                               const std::vector<float>& query_pre_rope,
                               size_t query_position) const;

  // Underlying caches, for GPU upload and stats. Null for the other scheme.
  const HelixCache* helix() const { return helix_.get(); }
  const RotorQuantCache* rotor() const { return rotor_.get(); }

  float compression_ratio_vs_f32() const;

 private:
  std::vector<float> rope(const std::vector<float>& row, size_t position) const;

  KvScheme scheme_;
  size_t head_dim_;
  size_t page_size_;
  float rope_theta_;
  size_t next_page_id_ = 0;
  std::unique_ptr<HelixCache> helix_;
  std::unique_ptr<RotorQuantCache> rotor_;
};

}
