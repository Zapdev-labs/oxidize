#pragma once

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
};

struct ModelFingerprint {
  uint64_t file_size_bytes = 0;
  bool exists = false;
};

struct TuningPlan {
  std::string numa_mode;  // "single" | "interleave" | "replicate"
  int threads = 0;
  bool mmap_hugepages = false;
  std::string mmap_advice = "sequential_prefetch";
  std::vector<std::string> rationale;
};

HardwareInventory detect_hardware(int numa_nodes_discovered);
ModelFingerprint fingerprint_model_file(const std::string& path);
TuningPlan plan_cpu(const HardwareInventory& inv, const ModelFingerprint& model);

std::string plan_to_json(const TuningPlan& plan);
std::string plan_summary(const TuningPlan& plan);

}  // namespace oxidize
