#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace oxidize {

// RotorQuant KV-cache compression: blockwise 3D rotor (rotation) decorrelation
// followed by per-block int4 scalar quantization. The inverse rotation is
// folded into the query/output, so decode is a plain quantized dot product.
// ponytail: uniform scalar quant instead of Lloyd-Max codebooks, no QJL
// residual stage; add those if quality at 4 bits proves insufficient.
struct RotorQuantConfig {
  size_t head_dim = 128;
  size_t block_size = 32;  // elements per quantization scale
  uint32_t seed = 0x5EED0F;
};

struct RotorQuantStats {
  size_t token_count = 0;
  size_t key_bytes = 0;
  size_t value_bytes = 0;
  size_t metadata_bytes = 0;
  size_t f32_baseline_bytes = 0;
  float total_bits_per_coord = 0.0f;

  float compression_ratio_vs_f32() const;
};

class RotorQuantCache {
 public:
  explicit RotorQuantCache(RotorQuantConfig config);
  ~RotorQuantCache();
  RotorQuantCache(RotorQuantCache&&) noexcept;
  RotorQuantCache& operator=(RotorQuantCache&&) noexcept;
  RotorQuantCache(const RotorQuantCache&) = delete;
  RotorQuantCache& operator=(const RotorQuantCache&) = delete;

  // Keys are post-RoPE (RotorQuant is a generic vector transform).
  void store_page(size_t layer, size_t kv_head,
                  const std::vector<float>& keys,
                  const std::vector<float>& values, size_t tokens);

  std::vector<float> logits(size_t layer, size_t kv_head,
                            const std::vector<float>& query) const;
  std::vector<float> attention(size_t layer, size_t kv_head,
                               const std::vector<float>& query) const;

  // Rotate/unrotate a vector with the cache's rotor set (exposed for tests).
  std::vector<float> rotate(const std::vector<float>& v) const;
  std::vector<float> unrotate(const std::vector<float>& v) const;

  // Read-only page view for GPU backends. Pointers valid until destruction.
  struct PageView {
    size_t layer = 0;
    size_t kv_head = 0;
    size_t tokens = 0;
    const uint8_t* key_codes = nullptr;   // packed int4 nibbles
    const float* key_scales = nullptr;    // per token x block
    const uint8_t* value_codes = nullptr;
    const float* value_scales = nullptr;
  };
  size_t page_count() const;
  bool page_view(size_t index, PageView* view) const;

  RotorQuantStats stats() const;

 private:
  struct Page;
  RotorQuantConfig config_;
  std::vector<float> rotors_;  // head_dim/3 row-major 3x3 rotation matrices
  std::vector<std::unique_ptr<Page>> pages_;
};

}
