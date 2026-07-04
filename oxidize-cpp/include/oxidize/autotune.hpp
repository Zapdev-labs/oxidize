#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace oxidize {

struct HardwareInventory {
  int physical_cores = 1;
  int logical_cores = 1;
  int numa_nodes = 1;
  uint64_t total_ram_bytes = 0;
  bool hugepages_2mib_avail = false;
  bool has_cuda = false;
  uint64_t gpu_vram_bytes = 0;
  std::string gpu_name;
};

struct ModelFingerprint {
  uint64_t file_size_bytes = 0;
  size_t layer_count = 0;
  size_t num_kv_heads = 0;
  size_t head_dim = 0;
};

struct TuningPlan {
  std::string numa_mode;  // "single" | "interleave" | "replicate"
  int threads = 0;
  bool mmap_hugepages = false;
  bool use_cuda = false;
  std::string pipeline = "sequential";
  std::string kv_cache_dtype = "f32";
  std::string kv_quantization = "none";
  std::string weight_plan = "native";
  std::string attention_kernel = "default";
  bool cuda_graphs = false;
  bool persistent_decode_kernels = false;
  int tensor_parallelism = 1;
  int pipeline_parallelism = 1;
  int chunked_prefill_tokens = 0;
  int max_decode_batch = 1;
  double expected_decode_tps = 0.0;
  std::vector<std::string> rationale;
};

HardwareInventory detect_hardware(int numa_nodes_discovered);
ModelFingerprint fingerprint_model_file(const std::string& path);
TuningPlan plan_cpu(const HardwareInventory& inv, const ModelFingerprint& model);

std::string plan_to_json(const TuningPlan& plan);
std::string plan_summary(const TuningPlan& plan);

}  // namespace oxidize
