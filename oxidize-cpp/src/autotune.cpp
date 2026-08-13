#include "oxidize/autotune.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include "oxidize/gpu_config.hpp"

namespace oxidize {

namespace {

uint64_t read_memtotal_bytes() {
  std::ifstream f("/proc/meminfo");
  if (!f.is_open()) return 0;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("MemTotal:", 0) == 0) {
      uint64_t kb = 0;
      if (std::sscanf(line.c_str(), "MemTotal: %llu kB",
                      reinterpret_cast<unsigned long long*>(&kb)) == 1) {
        return kb * 1024ULL;
      }
    }
  }
  return 0;
}

bool detect_hugepages_2mib() {
  std::ifstream f(
      "/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages");
  if (!f.is_open()) return false;
  long free_pages = 0;
  f >> free_pages;
  return free_pages > 0;
}

int physical_core_count() {
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  if (n < 1) n = 1;
  return static_cast<int>(n);
}

std::string lower_ascii(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

bool is_h100_name(const std::string& name) {
  return lower_ascii(name).find("h100") != std::string::npos;
}

uint64_t mib_to_bytes(uint64_t mib) {
  return mib * 1024ULL * 1024ULL;
}

void probe_nvidia_smi(HardwareInventory& inv) {
  FILE* pipe = popen(
      "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null",
      "r");
  if (!pipe) return;
  std::array<char, 512> buf{};
  if (fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
    std::string line(buf.data());
    auto comma = line.find(',');
    if (comma != std::string::npos) {
      inv.gpu_name = line.substr(0, comma);
      std::string mem = line.substr(comma + 1);
      unsigned long long mib = 0;
      if (std::sscanf(mem.c_str(), "%llu", &mib) == 1) {
        inv.gpu_vram_bytes = mib_to_bytes(mib);
      }
      inv.has_cuda = true;
    }
  }
  pclose(pipe);
}

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      default: out += c; break;
    }
  }
  return out;
}

}  // namespace

HardwareInventory detect_hardware(int numa_nodes_discovered) {
  HardwareInventory inv;
  inv.logical_cores = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  inv.physical_cores = physical_core_count();
  if (inv.physical_cores > inv.logical_cores) {
    inv.physical_cores = inv.logical_cores;
  }
  inv.numa_nodes = std::max(1, numa_nodes_discovered);
  inv.total_ram_bytes = read_memtotal_bytes();
  if (inv.total_ram_bytes == 0) {
    inv.total_ram_bytes = 4ULL << 30;
  }
  inv.hugepages_2mib_avail = detect_hugepages_2mib();
#ifdef OXIDIZE_CUDA
  inv.has_cuda = true;
#endif
  probe_nvidia_smi(inv);
  return inv;
}

ModelFingerprint fingerprint_model_file(const std::string& path) {
  ModelFingerprint m;
  struct stat st {};
  if (stat(path.c_str(), &st) == 0 && st.st_size > 0) {
    m.file_size_bytes = static_cast<uint64_t>(st.st_size);
  }
  return m;
}

TuningPlan plan_cpu(const HardwareInventory& inv, const ModelFingerprint& model) {
  TuningPlan plan;
  plan.numa_mode = "single";

  constexpr uint64_t k192GiB = 192ULL << 30;
  constexpr uint64_t k200GiB = 200ULL << 30;
  const uint64_t ram = inv.total_ram_bytes;
  const uint64_t size = model.file_size_bytes;
  const bool exceeds_ram = size > (ram * 8 / 10);
  const bool huge_model = size > k192GiB || exceeds_ram;

  if (huge_model) {
    plan.numa_mode = "interleave";
    if (inv.logical_cores > 48) {
      plan.threads = 48;
      plan.rationale.push_back(
          "model > 192 GiB or > 80% RAM → NUMA interleave, threads=48 "
          "(avoids cross-node thrash vs all logical cores)");
    } else {
      plan.threads = inv.logical_cores;
      plan.rationale.push_back(
          "model exceeds single-node budget → NUMA interleave, "
          "threads=all logical cores");
    }
  } else if (inv.numa_nodes >= 2) {
    plan.numa_mode = "single";
    plan.threads = inv.logical_cores / inv.numa_nodes;
    if (plan.threads < 1) plan.threads = inv.physical_cores / inv.numa_nodes;
    if (plan.threads < 1) plan.threads = inv.physical_cores;
    plan.rationale.push_back(
        "dense model fits single NUMA node → --numa single, "
        "threads=logical cores on node 0");
  } else {
    plan.numa_mode = "single";
    plan.threads = inv.physical_cores;
    plan.rationale.push_back("single NUMA node → threads=physical cores");
  }

  if (size >= k200GiB && inv.hugepages_2mib_avail) {
    plan.mmap_hugepages = true;
    plan.rationale.push_back(
        "model >= 200 GiB and 2 MiB hugepages free → enable mmap hugepages");
  } else if (size >= k200GiB) {
    plan.rationale.push_back(
        "model >= 200 GiB but no free 2 MiB hugepages → "
        "consider sysctl vm.nr_hugepages");
  }

  if (inv.has_cuda && is_h100_name(inv.gpu_name) &&
      inv.gpu_vram_bytes >= (70ULL << 30)) {
    plan.use_cuda = true;
    plan.pipeline = "paged";
    plan.kv_cache_dtype = "q4";
    plan.kv_quantization = "turboquant";
    plan.weight_plan = "w4a16";
    plan.attention_kernel = "flash_attention_3";
    plan.cuda_graphs = true;
    plan.persistent_decode_kernels = true;
    plan.tensor_parallelism = 1;
    plan.pipeline_parallelism = 1;
    plan.chunked_prefill_tokens = 512;
    plan.max_decode_batch = 16;
    plan.expected_decode_tps = 1150.0;
    plan.rationale.push_back(
        "single H100 throughput profile -> CUDA, paged KV, Q4 TurboQuant KV, "
        "chunked prefill, CUDA graphs, persistent decode kernels");
  }

  {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "model=%.1f GiB ram=%.1f GiB numa=%d",
                  size / (1024.0 * 1024.0 * 1024.0),
                  ram / (1024.0 * 1024.0 * 1024.0), inv.numa_nodes);
    plan.rationale.insert(plan.rationale.begin(), buf);
  }

  return plan;
}

std::string plan_to_json(const TuningPlan& plan) {
  std::ostringstream os;
  os << "{\n";
  os << "  \"numa_mode\": \"" << json_escape(plan.numa_mode) << "\",\n";
  os << "  \"threads\": " << plan.threads << ",\n";
  os << "  \"mmap_hugepages\": " << (plan.mmap_hugepages ? "true" : "false")
     << ",\n";
  os << "  \"use_cuda\": " << (plan.use_cuda ? "true" : "false") << ",\n";
  os << "  \"pipeline\": \"" << json_escape(plan.pipeline) << "\",\n";
  os << "  \"kv_cache_dtype\": \"" << json_escape(plan.kv_cache_dtype) << "\",\n";
  os << "  \"kv_quantization\": \"" << json_escape(plan.kv_quantization)
     << "\",\n";
  os << "  \"weight_plan\": \"" << json_escape(plan.weight_plan) << "\",\n";
  os << "  \"attention_kernel\": \"" << json_escape(plan.attention_kernel)
     << "\",\n";
  os << "  \"cuda_graphs\": " << (plan.cuda_graphs ? "true" : "false")
     << ",\n";
  os << "  \"persistent_decode_kernels\": "
     << (plan.persistent_decode_kernels ? "true" : "false") << ",\n";
  os << "  \"tensor_parallelism\": " << plan.tensor_parallelism << ",\n";
  os << "  \"pipeline_parallelism\": " << plan.pipeline_parallelism << ",\n";
  os << "  \"chunked_prefill_tokens\": " << plan.chunked_prefill_tokens << ",\n";
  os << "  \"max_decode_batch\": " << plan.max_decode_batch << ",\n";
  os << "  \"expected_decode_tps\": " << plan.expected_decode_tps << ",\n";
  os << "  \"rationale\": [";
  for (size_t i = 0; i < plan.rationale.size(); ++i) {
    if (i) os << ", ";
    os << "\"" << json_escape(plan.rationale[i]) << "\"";
  }
  os << "]\n";
  os << "}";
  return os.str();
}

std::string plan_summary(const TuningPlan& plan) {
  std::ostringstream os;
  os << "numa_mode         : " << plan.numa_mode << "\n";
  os << "threads           : " << plan.threads << "\n";
  os << "mmap_hugepages    : " << (plan.mmap_hugepages ? "true" : "false")
     << "\n";
  os << "use_cuda          : " << (plan.use_cuda ? "true" : "false") << "\n";
  os << "pipeline          : " << plan.pipeline << "\n";
  os << "kv_cache_dtype    : " << plan.kv_cache_dtype << " ("
     << plan.kv_quantization << ")\n";
  os << "weight_plan       : " << plan.weight_plan << "\n";
  os << "attention_kernel  : " << plan.attention_kernel << "\n";
  os << "cuda_graphs       : " << (plan.cuda_graphs ? "true" : "false")
     << "\n";
  os << "persistent_decode : "
     << (plan.persistent_decode_kernels ? "true" : "false") << "\n";
  os << "parallelism       : tensor=" << plan.tensor_parallelism
     << " pipeline=" << plan.pipeline_parallelism << "\n";
  os << "chunked_prefill   : " << plan.chunked_prefill_tokens
     << " tokens  max_decode_batch=" << plan.max_decode_batch << "\n";
  os << "expected_decode_tps: " << plan.expected_decode_tps << "\n";
  if (!plan.rationale.empty()) {
    os << "\nRationale:\n";
    for (const auto& r : plan.rationale) {
      os << "  - " << r << "\n";
    }
  }
  return os.str();
}

}  // namespace oxidize
