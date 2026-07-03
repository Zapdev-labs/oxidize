#include "oxidize/autotune.hpp"

#include <algorithm>
#include <cctype>
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

bool file_size_bytes(const std::string& path, uint64_t& out) {
  struct stat st {};
  if (stat(path.c_str(), &st) != 0) return false;
  if (st.st_size <= 0) {
    out = 0;
    return true;
  }
  out = static_cast<uint64_t>(st.st_size);
  return true;
}

bool all_digits(const std::string& s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) {
    return std::isdigit(c) != 0;
  });
}

bool parse_split_path(const std::string& path, std::string& prefix,
                      unsigned& split_no, unsigned& split_count) {
  constexpr const char* kExt = ".gguf";
  constexpr size_t kExtLen = 5;
  if (path.size() <= kExtLen ||
      path.compare(path.size() - kExtLen, kExtLen, kExt) != 0) {
    return false;
  }

  const size_t of_pos = path.rfind("-of-", path.size() - kExtLen);
  if (of_pos == std::string::npos) return false;
  const size_t dash_pos = path.rfind('-', of_pos - 1);
  if (dash_pos == std::string::npos) return false;

  const std::string no_str = path.substr(dash_pos + 1, of_pos - dash_pos - 1);
  const std::string count_str =
      path.substr(of_pos + 4, path.size() - kExtLen - of_pos - 4);
  if (no_str.size() != 5 || count_str.size() != 5 ||
      !all_digits(no_str) || !all_digits(count_str)) {
    return false;
  }

  split_no = static_cast<unsigned>(std::stoul(no_str));
  split_count = static_cast<unsigned>(std::stoul(count_str));
  if (split_no == 0 || split_count <= 1 || split_no > split_count) {
    return false;
  }
  prefix = path.substr(0, dash_pos);
  return true;
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
  uint64_t first_size = 0;
  if (!file_size_bytes(path, first_size)) {
    return m;
  }
  m.exists = true;
  m.file_size_bytes = first_size;

  std::string prefix;
  unsigned split_no = 0;
  unsigned split_count = 0;
  if (!parse_split_path(path, prefix, split_no, split_count)) return m;

  uint64_t total = 0;
  for (unsigned i = 1; i <= split_count; ++i) {
    char shard_buf[1024];
    std::snprintf(shard_buf, sizeof(shard_buf), "%s-%05u-of-%05u.gguf",
                  prefix.c_str(), i, split_count);
    uint64_t shard_size = 0;
    if (!file_size_bytes(shard_buf, shard_size) || shard_size == 0) {
      return m;
    }
    if (UINT64_MAX - total < shard_size) {
      return m;
    }
    total += shard_size;
  }
  m.file_size_bytes = total;
  return m;
}

TuningPlan plan_cpu(const HardwareInventory& inv, const ModelFingerprint& model) {
  TuningPlan plan;
  plan.numa_mode = "single";
  plan.mmap_advice = "sequential_prefetch";

  constexpr uint64_t k192GiB = 192ULL << 30;
  constexpr uint64_t k200GiB = 200ULL << 30;
  const uint64_t ram = inv.total_ram_bytes;
  const uint64_t size = model.file_size_bytes;
  const bool exceeds_ram = size > (ram * 8 / 10);
  const bool huge_model = size > k192GiB || exceeds_ram;

  if (huge_model) {
    plan.numa_mode = "interleave";
    plan.mmap_advice = "random";
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
    if (plan.threads > 16) plan.threads = 16;
    plan.rationale.push_back(
        "dense model fits single NUMA node → --numa single, "
        "threads=min(16, logical cores on node 0)");
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
  os << "  \"mmap_advice\": \"" << json_escape(plan.mmap_advice) << "\",\n";
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
  os << "mmap_advice       : " << plan.mmap_advice << "\n";
  if (!plan.rationale.empty()) {
    os << "\nRationale:\n";
    for (const auto& r : plan.rationale) {
      os << "  - " << r << "\n";
    }
  }
  return os.str();
}

}  // namespace oxidize
