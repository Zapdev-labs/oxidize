// NUMA/CPU affinity utilities for oxidize-cpp.
//
// Goals:
//   1. Discover NUMA node -> physical-core CPU list at runtime via
//      /sys/devices/system/node/nodeN/cpulist (no libnuma headers required).
//   2. Pin OpenMP threads to the physical cores of a chosen NUMA node
//      using sched_setaffinity(2) (libc / POSIX).
//   3. Bind memory allocation to the same node via set_mempolicy(2) or
//      mbind(2) as raw syscalls (libnuma.so.1 is present at runtime but
//      the dev header is not installed).
//   4. Expose a clean enum + init function that main.cpp can call before
//      model loading so first-touch pages land on the bound node.
//
// Thread safety: init_numa() is intended to be called once from the main
// thread before any parallel regions start.  The omp parallel binding
// region inside init_numa() is itself the synchronization point.
//
// CUDA: all of this is CPU-only.  The CUDA backend ignores NUMA; nothing
// here touches CUDA paths.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace oxidize {

/// NUMA binding policy requested by the caller.
enum class NumaMode {
  /// Auto-detect: pick the node with the lowest id, pin threads + memory.
  /// Reproduces  numactl --cpunodebind=0 --membind=0  without the wrapper.
  Single,
  /// Interleave memory pages across all nodes; use all physical cores.
  /// Reproduces  numactl --interleave=all.
  Interleave,
  /// Use every physical core on every node; no memory binding (OS default).
  All,
  /// Use all physical cores of a single node (same as Single but spelled out
  /// as a CLI value when the user passes --numa <N> with a numeric node id).
  Node,   // numeric node; the actual node number is passed separately
  /// Experimental: replicate the model weights onto both nodes (caller must
  /// allocate two copies).  Reported here for completeness; current CLI
  /// surfaces it as --numa replicate.
  Replicate,
};

/// Per-node CPU info discovered from sysfs.
struct NumaNode {
  int id;
  std::vector<int> cpus;  // logical cpu ids in the node
};

/// Discover all NUMA nodes from /sys/devices/system/node/.
/// Returns an empty vector if sysfs is unavailable (non-Linux, container).
std::vector<NumaNode> discover_numa_nodes();

/// Config produced by parse + used by init_numa.
struct NumaConfig {
  NumaMode mode = NumaMode::Single;
  int node = 0;          // which node to bind to (Single / Node modes)
  int threads = 0;       // 0 = auto (physical cores of bound node)
};

/// Parse --numa argument string into a NumaConfig.
/// Accepted values: "single", "interleave", "all", "replicate", "0".."N".
/// Throws std::invalid_argument on unrecognised values.
NumaConfig parse_numa_arg(const std::string& s);

/// Apply NUMA binding and thread affinity.
///
/// Must be called BEFORE model loading so that mmap/malloc pages land on
/// the target node (first-touch policy).
///
/// Returns the number of threads actually pinned / configured.
///
/// When OpenMP is not compiled in, falls back to set_mempolicy only.
int init_numa(const NumaConfig& cfg,
              const std::vector<NumaNode>& nodes);

}  // namespace oxidize
