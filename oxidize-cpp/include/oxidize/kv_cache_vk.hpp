#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "oxidize/helix_cache.hpp"
#include "oxidize/rotor_quant.hpp"

namespace oxidize {

// Vulkan compute backend for HelixCache / RotorQuant decode. One instance
// owns the device; pages are uploaded once after CPU-side quantization and
// decode runs as a single submit (logits -> softmax -> weighted values).
// Works on any Vulkan 1.1 compute queue: NVIDIA, AMD (all cards), Intel Arc,
// integrated GPUs.
class VulkanKv {
 public:
  static bool available();

  VulkanKv();  // throws std::runtime_error when no usable device exists
  ~VulkanKv();
  VulkanKv(VulkanKv&&) noexcept;
  VulkanKv& operator=(VulkanKv&&) noexcept;
  VulkanKv(const VulkanKv&) = delete;
  VulkanKv& operator=(const VulkanKv&) = delete;

  std::string device_name() const;

  // Upload all cold pages of the cache, grouped by (layer, kv_head).
  // Re-uploading replaces previous contents. Hot pages are not supported on
  // the GPU path.
  void upload(const HelixCache& cache, const HelixCacheConfig& config);
  void upload(const RotorQuantCache& cache, const RotorQuantConfig& config);

  // Full attention over the uploaded pages. Mirrors the CPU signatures.
  std::vector<float> helix_attention(size_t layer, size_t kv_head,
                                     const std::vector<float>& query_pre_rope,
                                     size_t query_position, float rope_theta);
  std::vector<float> rotor_attention(const RotorQuantCache& cache, size_t layer,
                                     size_t kv_head,
                                     const std::vector<float>& query);

  // All heads of a layer in one submit — queries[h] is the query for kv_head
  // h. Amortizes the ~1ms submit+fence latency across the whole layer.
  std::vector<std::vector<float>> helix_attention_batch(
      size_t layer, const std::vector<std::vector<float>>& queries,
      size_t query_position, float rope_theta);
  std::vector<std::vector<float>> rotor_attention_batch(
      const RotorQuantCache& cache, size_t layer,
      const std::vector<std::vector<float>>& queries);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}
