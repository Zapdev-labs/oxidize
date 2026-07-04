#include "oxidize/kv_compressed.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace oxidize {

CompressedKvCache::CompressedKvCache(size_t head_dim, KvScheme scheme,
                                     size_t page_size, float rope_theta)
    : scheme_(scheme),
      head_dim_(head_dim),
      page_size_(page_size),
      rope_theta_(rope_theta) {
  if (scheme_ == KvScheme::Helix) {
    HelixCacheConfig cfg;
    cfg.head_dim = head_dim;
    cfg.page_size = page_size;
    helix_ = std::make_unique<HelixCache>(cfg);
  } else {
    RotorQuantConfig cfg;
    cfg.head_dim = head_dim;
    rotor_ = std::make_unique<RotorQuantCache>(cfg);
  }
}

CompressedKvCache::~CompressedKvCache() = default;
CompressedKvCache::CompressedKvCache(CompressedKvCache&&) noexcept = default;
CompressedKvCache& CompressedKvCache::operator=(CompressedKvCache&&) noexcept =
    default;

std::vector<float> CompressedKvCache::rope(const std::vector<float>& row,
                                           size_t position) const {
  std::vector<float> out(row.size());
  const size_t pairs = head_dim_ / 2;
  for (size_t p = 0; p < pairs; ++p) {
    const float freq =
        std::pow(rope_theta_, -2.0f * static_cast<float>(p) /
                                  static_cast<float>(head_dim_));
    const float angle = freq * static_cast<float>(position);
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    const float x = row[2 * p];
    const float y = row[2 * p + 1];
    out[2 * p] = x * c - y * s;
    out[2 * p + 1] = x * s + y * c;
  }
  return out;
}

void CompressedKvCache::store_page(size_t layer, size_t kv_head,
                                   const std::vector<float>& pre_rope_keys,
                                   const std::vector<float>& values,
                                   const std::vector<size_t>& positions) {
  if (helix_) {
    helix_->store_cold_page(layer, kv_head, next_page_id_++, pre_rope_keys,
                            values, positions);
    return;
  }
  // RotorQuant stores post-RoPE keys; rotate each row to its position.
  const size_t tokens = positions.size();
  if (pre_rope_keys.size() != tokens * head_dim_) {
    throw std::invalid_argument("CompressedKvCache: key shape mismatch");
  }
  std::vector<float> roped(tokens * head_dim_);
  std::vector<float> row(head_dim_);
  for (size_t t = 0; t < tokens; ++t) {
    row.assign(pre_rope_keys.begin() + t * head_dim_,
               pre_rope_keys.begin() + (t + 1) * head_dim_);
    const std::vector<float> r = rope(row, positions[t]);
    std::copy(r.begin(), r.end(), roped.begin() + t * head_dim_);
  }
  rotor_->store_page(layer, kv_head, roped, values, tokens);
}

std::vector<float> CompressedKvCache::attention(
    size_t layer, size_t kv_head, const std::vector<float>& query_pre_rope,
    size_t query_position) const {
  if (helix_) {
    // HelixCache rotates at decode time; const_cast only bumps access stats.
    return const_cast<HelixCache*>(helix_.get())
        ->attention(layer, kv_head, query_pre_rope, query_position,
                    rope_theta_);
  }
  return rotor_->attention(layer, kv_head, rope(query_pre_rope, query_position));
}

float CompressedKvCache::compression_ratio_vs_f32() const {
  return helix_ ? helix_->stats().compression_ratio_vs_f32()
                : rotor_->stats().compression_ratio_vs_f32();
}

}
