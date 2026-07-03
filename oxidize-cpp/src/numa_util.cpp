// NUMA/CPU affinity implementation for oxidize-cpp.
//
// Uses:
//   sched_setaffinity(2)   - from <sched.h> (libc, always present)
//   set_mempolicy(2)       - raw syscall (no libnuma-dev needed)
//   mbind(2)               - raw syscall (used for interleave on all pages)
//   /sys/devices/system/node/ - discovery (Linux-only)
//   /proc/self/status      - fallback thread id
//
// No libnuma headers are used; libnuma.so.1 is present at runtime but
// the -dev package is absent.  We call the same kernel interfaces via
// syscall() directly.

#include "oxidize/numa_util.hpp"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>

// POSIX / Linux headers
#include <dirent.h>
#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---- syscall numbers (x86-64 Linux) ----------------------------------------
#ifndef __NR_set_mempolicy
#define __NR_set_mempolicy 238
#endif
#ifndef __NR_mbind
#define __NR_mbind 237
#endif

// MPOL constants (from linux/mempolicy.h)
#define OXIDIZE_MPOL_DEFAULT    0
#define OXIDIZE_MPOL_BIND       2
#define OXIDIZE_MPOL_INTERLEAVE 3
#define OXIDIZE_MPOL_MF_MOVE    2

// MPOL_F_STATIC_NODES (bit 15) — nodemask is literal node ids, not relative.
// We don't need it; we always build a proper bitmask.

namespace {

#ifdef _OPENMP
std::vector<int> g_thread_numa_node;  // omp thread id -> node id (-1 = unknown)
#else
std::vector<int> g_thread_numa_node(1, -1);
#endif
bool g_replication_enabled = false;

}  // namespace

namespace oxidize {

// ---------------------------------------------------------------------------
// Sysfs helpers
// ---------------------------------------------------------------------------

/// Parse a Linux cpulist string like "0,2,4-6,8" into a sorted vector of
/// cpu ids.
static std::vector<int> parse_cpulist(const std::string& s) {
    std::vector<int> out;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // trim whitespace / newlines
        while (!token.empty() && (token.back() == '\n' || token.back() == '\r' ||
                                   token.back() == ' '))
            token.pop_back();
        if (token.empty()) continue;
        auto dash = token.find('-');
        if (dash == std::string::npos) {
            out.push_back(std::stoi(token));
        } else {
            int lo = std::stoi(token.substr(0, dash));
            int hi = std::stoi(token.substr(dash + 1));
            for (int c = lo; c <= hi; ++c) out.push_back(c);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<NumaNode> discover_numa_nodes() {
    std::vector<NumaNode> nodes;
    const char* base = "/sys/devices/system/node";
    DIR* d = opendir(base);
    if (!d) return nodes;

    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name(ent->d_name);
        if (name.rfind("node", 0) != 0) continue;
        // "node" followed by digits only
        std::string suffix = name.substr(4);
        if (suffix.empty() || suffix.find_first_not_of("0123456789") != std::string::npos)
            continue;
        int node_id = std::stoi(suffix);

        std::string cpulist_path = std::string(base) + "/" + name + "/cpulist";
        std::ifstream f(cpulist_path);
        if (!f.is_open()) continue;
        std::string content;
        std::getline(f, content);
        auto cpus = parse_cpulist(content);
        if (!cpus.empty()) {
            nodes.push_back({node_id, std::move(cpus)});
        }
    }
    closedir(d);
    std::sort(nodes.begin(), nodes.end(), [](const NumaNode& a, const NumaNode& b) {
        return a.id < b.id;
    });
    return nodes;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

NumaConfig parse_numa_arg(const std::string& s) {
    NumaConfig cfg;
    if (s == "single") {
        cfg.mode = NumaMode::Single;
        cfg.node = 0;
    } else if (s == "interleave") {
        cfg.mode = NumaMode::Interleave;
    } else if (s == "all") {
        cfg.mode = NumaMode::All;
    } else if (s == "replicate") {
        cfg.mode = NumaMode::Replicate;
    } else {
        // Numeric node id
        bool numeric = !s.empty() &&
            s.find_first_not_of("0123456789") == std::string::npos;
        if (!numeric)
            throw std::invalid_argument("--numa: unrecognised value '" + s +
                "'; use: single|interleave|all|replicate|<node-id>");
        cfg.mode = NumaMode::Node;
        cfg.node = std::stoi(s);
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Syscall wrappers
// ---------------------------------------------------------------------------

/// Build a nodemask unsigned long array for a single node id.
/// Kernel nodemask is an array of `unsigned long` words; each bit = one node.
static std::vector<unsigned long> make_nodemask_single(int node) {
    // Max nodes we ever need to cover: 64 (typical). One ulong = 64 bits.
    const int ulong_bits = sizeof(unsigned long) * 8;
    int words = (node / ulong_bits) + 1;
    std::vector<unsigned long> mask(static_cast<size_t>(words), 0UL);
    mask[static_cast<size_t>(node / ulong_bits)] |=
        (1UL << (static_cast<unsigned>(node) % static_cast<unsigned>(ulong_bits)));
    return mask;
}

/// Build a nodemask for all nodes in `nodes`.
static std::vector<unsigned long> make_nodemask_all(
    const std::vector<NumaNode>& nodes) {
    int max_node = 0;
    for (auto& n : nodes) max_node = std::max(max_node, n.id);
    const int ulong_bits = sizeof(unsigned long) * 8;
    int words = (max_node / ulong_bits) + 1;
    std::vector<unsigned long> mask(static_cast<size_t>(words), 0UL);
    for (auto& n : nodes) {
        mask[static_cast<size_t>(n.id / ulong_bits)] |=
            (1UL << (static_cast<unsigned>(n.id) %
                     static_cast<unsigned>(ulong_bits)));
    }
    return mask;
}

/// Call set_mempolicy for this thread (inherits to child threads in Linux).
static bool set_thread_mempolicy(int policy,
                                  const std::vector<unsigned long>& mask) {
    long rc = syscall(__NR_set_mempolicy, policy,
                      mask.empty() ? nullptr : mask.data(),
                      static_cast<unsigned long>(mask.size() * sizeof(unsigned long) * 8));
    return rc == 0;
}

/// Bind THIS thread to a cpu_set.
static bool pin_thread_to_cpuset(const cpu_set_t& cpuset) {
    pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
    return sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset) == 0;
}

// ---------------------------------------------------------------------------
// Core init
// ---------------------------------------------------------------------------

int init_numa(const NumaConfig& cfg, const std::vector<NumaNode>& nodes) {
    g_replication_enabled = false;
    if (nodes.empty()) {
        // No sysfs NUMA info — skip binding, just set threads.
#ifdef _OPENMP
        int logical = omp_get_max_threads();
        int want = cfg.threads > 0 ? cfg.threads : logical;
        omp_set_num_threads(want);
        g_thread_numa_node.assign(want, -1);
        return want;
#else
        return 1;
#endif
    }

    // ---- Determine which node(s) to bind ----
    const NumaNode* bound_node = nullptr;
    if (cfg.mode == NumaMode::Single || cfg.mode == NumaMode::Node ||
        cfg.mode == NumaMode::Replicate) {
        // Find the requested node.
        for (auto& n : nodes) {
            if (n.id == cfg.node) {
                bound_node = &n;
                break;
            }
        }
        if (!bound_node && !nodes.empty()) {
            bound_node = &nodes[0];  // fallback to first node
        }
    }

    // ---- Determine thread count ----
    int n_threads = cfg.threads;
    if (n_threads <= 0) {
        if (cfg.mode == NumaMode::Replicate) {
            // Use every logical core on both sockets; each socket works on its
            // local weight copy.
            int total_logical = 0;
            for (auto& n : nodes) total_logical += static_cast<int>(n.cpus.size());
            n_threads = total_logical;
        } else if (bound_node) {
            // For memory-bandwidth-bound LLM decode, ALL logical CPUs on one
            // NUMA node outperform half (physical-only) because more in-flight
            // memory requests saturate the node's bandwidth channels better.
            // Measured on Xeon Silver 4110: 16 logical/node -> 16 threads = 44
            // tok/s vs 8 threads = 27 tok/s. Use all logical CPUs on the node.
            n_threads = static_cast<int>(bound_node->cpus.size());
        } else {
            // All / Interleave mode: total logical cores across all nodes.
            int total_logical = 0;
            for (auto& n : nodes) total_logical += static_cast<int>(n.cpus.size());
            n_threads = total_logical;
            if (n_threads < 1) n_threads = 1;
        }
    }

#ifdef _OPENMP
    omp_set_num_threads(n_threads);
    g_thread_numa_node.assign(n_threads, -1);

    // ---- Memory policy: set on main thread, inherited by OMP workers ----
    if (cfg.mode == NumaMode::Single || cfg.mode == NumaMode::Node ||
        cfg.mode == NumaMode::Replicate) {
        if (bound_node) {
            auto mask = make_nodemask_single(bound_node->id);
            if (!set_thread_mempolicy(OXIDIZE_MPOL_BIND, mask)) {
                // Non-fatal; warn and continue.
                std::fprintf(stderr,
                    "oxidize: NUMA warning: set_mempolicy(BIND, node=%d) failed "
                    "(errno=%d); memory placement uncontrolled\n",
                    bound_node->id, errno);
            }
        }
    } else if (cfg.mode == NumaMode::Interleave) {
        auto mask = make_nodemask_all(nodes);
        if (!set_thread_mempolicy(OXIDIZE_MPOL_INTERLEAVE, mask)) {
            std::fprintf(stderr,
                "oxidize: NUMA warning: set_mempolicy(INTERLEAVE) failed "
                "(errno=%d); memory placement uncontrolled\n", errno);
        }
    }
    // All mode: leave memory policy at OS default (MPOL_DEFAULT).

    // ---- Build target cpu list and per-thread node map ----
    //
    // For Single/Node: cpus = bound_node->cpus (all logical cpus on that node).
    //   We pick one per physical core.  On Skylake-SP hyperthreaded, each
    //   physical core has two siblings; sysfs lists them interleaved.
    //   For node0 (cpus 0,2,4,...,30): stride-2 gives us 16 logical cpus,
    //   we'll use all 16 slots (the OS won't over-subscribe HT if we want 8).
    //   Actually we want: first N unique physical cores = first N cpus in the list.
    //   With n_threads=8 we take cpus[0..7] which are {0,2,4,6,8,10,12,14}.
    //   That is correct: one per physical core, all on node0.
    //
    // For Interleave/All/Replicate: combine cpus from all nodes.  For Replicate
    // we keep node-0 cpus first and node-1 cpus second so the static OpenMP
    // split in gemv_quantized assigns each half of the output rows to the
    // socket that owns the local weight copy.
    //
    std::vector<int> target_cpus;
    std::vector<int> cpu_node;  // parallel to target_cpus: which node each cpu belongs to
    if (cfg.mode == NumaMode::Single || cfg.mode == NumaMode::Node) {
        if (bound_node) {
            target_cpus = bound_node->cpus;
            cpu_node.assign(target_cpus.size(), bound_node->id);
        }
    } else if (cfg.mode == NumaMode::Replicate) {
        g_replication_enabled = true;
        for (auto& n : nodes) {
            for (int c : n.cpus) {
                target_cpus.push_back(c);
                cpu_node.push_back(n.id);
            }
        }
    } else {
        // All / Interleave: gather all cpus across nodes (already sorted by node id).
        for (auto& n : nodes) {
            for (int c : n.cpus) {
                target_cpus.push_back(c);
                cpu_node.push_back(n.id);
            }
        }
    }

    if (!target_cpus.empty()) {
        // Trim to n_threads if we have more cpus than threads.
        // This ensures we use the first physical cores in the node's cpu list.
        int avail = static_cast<int>(target_cpus.size());
        // For the binding, only pin if we have at least as many cpus as threads.
        bool can_pin = (avail >= n_threads);

        if (can_pin) {
            // Launch the OMP region that pins each thread.
            #pragma omp parallel num_threads(n_threads)
            {
                int tid = omp_get_thread_num();
                if (tid < static_cast<int>(target_cpus.size())) {
                    int cpu = target_cpus[static_cast<size_t>(tid)];
                    int node = cpu_node[static_cast<size_t>(tid)];
                    g_thread_numa_node[tid] = node;
                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    CPU_SET(cpu, &cpuset);
                    if (!pin_thread_to_cpuset(cpuset)) {
                        // Non-fatal — thread can still run, just not pinned.
                        // Avoid fprintf in parallel region (mixed output); skip.
                    }
                }
            }
        } else {
            // Not enough cpus to pin all threads; still record node mapping.
            for (int tid = 0; tid < n_threads && tid < avail; ++tid) {
                g_thread_numa_node[tid] = cpu_node[static_cast<size_t>(tid)];
            }
        }
    }

    return n_threads;
#else
    // No OpenMP: at least set memory policy for the main thread.
    if ((cfg.mode == NumaMode::Single || cfg.mode == NumaMode::Node) && bound_node) {
        auto mask = make_nodemask_single(bound_node->id);
        set_thread_mempolicy(OXIDIZE_MPOL_BIND, mask);
    } else if (cfg.mode == NumaMode::Interleave) {
        auto mask = make_nodemask_all(nodes);
        set_thread_mempolicy(OXIDIZE_MPOL_INTERLEAVE, mask);
    }
    return 1;
#endif
}

int current_thread_numa_node() {
#ifdef _OPENMP
    int tid = omp_get_thread_num();
    if (tid >= 0 && tid < static_cast<int>(g_thread_numa_node.size())) {
        return g_thread_numa_node[tid];
    }
#endif
    return -1;
}

bool numa_replication_enabled() { return g_replication_enabled; }

bool bind_memory_to_node(void* ptr, size_t len, int node) {
    if (!ptr || len == 0) return false;
    const long page_size_l = sysconf(_SC_PAGESIZE);
    if (page_size_l <= 0) return false;
    const size_t page_size = static_cast<size_t>(page_size_l);

    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_start = start & ~(page_size - 1);
    uintptr_t end = start + len;
    uintptr_t aligned_end = (end + page_size - 1) & ~(page_size - 1);
    void* addr = reinterpret_cast<void*>(aligned_start);
    size_t alen = aligned_end - aligned_start;
    if (alen == 0) return false;

    auto mask = make_nodemask_single(node);
    long rc = syscall(__NR_mbind, addr, alen, OXIDIZE_MPOL_BIND,
                      mask.empty() ? nullptr : mask.data(),
                      static_cast<unsigned long>(mask.size() * sizeof(unsigned long) * 8),
                      OXIDIZE_MPOL_MF_MOVE);
    if (rc != 0) return false;
    // Fault pages in on the target node.
    volatile char* p = static_cast<char*>(ptr);
    for (size_t off = 0; off < len; off += page_size) {
        p[off] = p[off];
    }
    return true;
}

}  // namespace oxidize
