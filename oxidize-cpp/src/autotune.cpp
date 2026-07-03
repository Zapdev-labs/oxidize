#include "oxidize/autotune.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

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
  const bool way_over_ram = size > (ram * 15 / 10);

  if (huge_model) {
    plan.numa_mode = "interleave";
    plan.mmap_policy = "demand";
    plan.prefetch_layers = way_over_ram ? 2 : 1;
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
    plan.rationale.push_back(
        "model exceeds RAM locality budget → demand mmap policy avoids "
        "whole-file SSD prefetch");
    {
      char buf[128];
      std::snprintf(buf, sizeof(buf),
                    "model > RAM working-set budget → async layer-ahead "
                    "prefetch depth=%d hides SSD latency",
                    plan.prefetch_layers);
      plan.rationale.push_back(buf);
    }
  } else if (inv.numa_nodes >= 2) {
    // Dense dual-socket models that are large enough to be memory-bandwidth
    // bound run faster when weights are interleaved across both sockets and
    // all logical cores are used. Tiny models stay single-node to avoid
    // cross-node overhead.
    constexpr uint64_t k4GiB = 4ULL << 30;
    if (size > k4GiB) {
      plan.numa_mode = "interleave";
      plan.mmap_policy = "prefetch";
      plan.threads = inv.logical_cores;
      plan.rationale.push_back(
          "dense dual-socket model benefits from aggregated memory bandwidth "
          "→ --numa interleave, threads=all logical cores");
    } else {
      plan.numa_mode = "single";
      plan.mmap_policy = "prefetch";
      plan.threads = inv.logical_cores / inv.numa_nodes;
      if (plan.threads < 1) plan.threads = inv.physical_cores / inv.numa_nodes;
      if (plan.threads < 1) plan.threads = inv.physical_cores;
      plan.rationale.push_back(
          "small dense model fits single NUMA node → --numa single, "
          "threads=logical cores on node 0");
    }
  } else {
    plan.numa_mode = "single";
    plan.mmap_policy = "prefetch";
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
  os << "  \"mmap_policy\": \"" << json_escape(plan.mmap_policy) << "\",\n";
  os << "  \"prefetch_layers\": " << plan.prefetch_layers << ",\n";
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
  os << "mmap_policy       : " << plan.mmap_policy << "\n";
  os << "prefetch_layers   : " << plan.prefetch_layers << "\n";
  if (!plan.rationale.empty()) {
    os << "\nRationale:\n";
    for (const auto& r : plan.rationale) {
      os << "  - " << r << "\n";
    }
  }
  return os.str();
}

}  // namespace oxidize
