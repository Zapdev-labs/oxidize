#include "oxidize/kv_cache_vk.hpp"

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

void test_helix_gpu_matches_cpu() {
  oxidize::HelixCacheConfig cfg;
  cfg.head_dim = 128;
  cfg.page_size = 64;
  oxidize::HelixCache cache(cfg);
  uint64_t rng = 11;
  const size_t pages = 4;
  for (size_t p = 0; p < pages; ++p) {
    std::vector<float> keys(cfg.page_size * cfg.head_dim);
    std::vector<float> values(cfg.page_size * cfg.head_dim);
    std::vector<size_t> positions;
    for (auto& x : keys) x = lcg(rng);
    for (auto& x : values) x = lcg(rng);
    for (size_t t = 0; t < cfg.page_size; ++t) {
      positions.push_back(p * cfg.page_size + t);
    }
    cache.store_cold_page(0, 0, p, keys, values, positions);
  }
  std::vector<float> query(cfg.head_dim);
  for (auto& x : query) x = lcg(rng);

  const size_t qpos = pages * cfg.page_size;
  const float theta = 10000.0f;
  const std::vector<float> cpu = cache.attention(0, 0, query, qpos, theta);

  oxidize::VulkanKv vk;
  std::printf("vulkan device: %s\n", vk.device_name().c_str());
  vk.upload(cache, cfg);
  const std::vector<float> gpu = vk.helix_attention(0, 0, query, qpos, theta);
  require(gpu.size() == cpu.size(), "helix gpu output size");
  for (size_t i = 0; i < cpu.size(); ++i) {
    if (std::fabs(gpu[i] - cpu[i]) > 5.0e-3f) {
      std::fprintf(stderr, "helix dim %zu: gpu %f vs cpu %f\n", i, gpu[i],
                   cpu[i]);
      throw std::runtime_error("helix gpu attention should match cpu");
    }
  }
}

void test_rotor_gpu_matches_cpu() {
  oxidize::RotorQuantConfig cfg;
  cfg.head_dim = 128;
  oxidize::RotorQuantCache cache(cfg);
  uint64_t rng = 23;
  const size_t pages = 4;
  const size_t page_tokens = 64;
  for (size_t p = 0; p < pages; ++p) {
    std::vector<float> keys(page_tokens * cfg.head_dim);
    std::vector<float> values(page_tokens * cfg.head_dim);
    for (auto& x : keys) x = lcg(rng);
    for (auto& x : values) x = lcg(rng);
    cache.store_page(0, 0, keys, values, page_tokens);
  }
  std::vector<float> query(cfg.head_dim);
  for (auto& x : query) x = lcg(rng);

  const std::vector<float> cpu = cache.attention(0, 0, query);

  oxidize::VulkanKv vk;
  vk.upload(cache, cfg);
  const std::vector<float> gpu = vk.rotor_attention(cache, 0, 0, query);
  require(gpu.size() == cpu.size(), "rotor gpu output size");
  for (size_t i = 0; i < cpu.size(); ++i) {
    if (std::fabs(gpu[i] - cpu[i]) > 5.0e-3f) {
      std::fprintf(stderr, "rotor dim %zu: gpu %f vs cpu %f\n", i, gpu[i],
                   cpu[i]);
      throw std::runtime_error("rotor gpu attention should match cpu");
    }
  }
}

}

int main() {
  if (!oxidize::VulkanKv::available()) {
    std::puts("kv cache vk tests skipped (no Vulkan device)");
    return 0;
  }
  test_helix_gpu_matches_cpu();
  test_rotor_gpu_matches_cpu();
  std::puts("kv cache vk tests passed");
  return 0;
}
