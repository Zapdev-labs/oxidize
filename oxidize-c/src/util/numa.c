/*
 * numa.c — NUMA awareness implementation.
 *
 * Uses Linux sysfs for NUMA topology detection. On non-Linux systems
 * or systems without NUMA, returns OC_OK with a single-node topology.
 */
#define _GNU_SOURCE
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1  /* expose _SC_NPROCESSORS_ONLN on macOS */
#endif
#include "oxidize/numa.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif

/* ─── Kernel NUMA memory policy (raw syscalls, no libnuma) ──────────────
 *
 * set_mempolicy(2)/mbind(2) are not exposed by glibc, so call them through
 * syscall(). Keeping this dependency-free matters: the whole point of the C
 * port is that it builds with nothing but a libc.
 *
 * These MPOL_* values are the kernel ABI (include/uapi/linux/mempolicy.h) and
 * deliberately do NOT match the OcNumaMemPolicy enum — map explicitly.
 */
#if defined(__linux__)
#define OC_MPOL_DEFAULT    0
#define OC_MPOL_PREFERRED  1
#define OC_MPOL_BIND       2
#define OC_MPOL_INTERLEAVE 3

/* nodemask word count. The kernel wants `maxnode` = the number of BITS the
 * mask can hold, and rejects a value that would read past the buffer. */
#define OC_NODEMASK_WORDS \
    ((OC_NUMA_MAX_NODES + (8 * sizeof(unsigned long)) - 1) / (8 * sizeof(unsigned long)))
#define OC_NODEMASK_BITS  (OC_NODEMASK_WORDS * 8 * sizeof(unsigned long))

static void nodemask_set(unsigned long *mask, uint32_t node)
{
    mask[node / (8 * sizeof(unsigned long))] |=
        1UL << (node % (8 * sizeof(unsigned long)));
}

static long sys_set_mempolicy(int mode, const unsigned long *nmask,
                              unsigned long maxnode)
{
    return syscall(SYS_set_mempolicy, mode, nmask, maxnode);
}

static long sys_mbind(void *addr, unsigned long len, int mode,
                      const unsigned long *nmask, unsigned long maxnode,
                      unsigned int flags)
{
    return syscall(SYS_mbind, addr, len, mode, nmask, maxnode, flags);
}
#endif /* __linux__ */

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static bool read_file_to_buf(const char *path, char *buf, size_t cap)
{
    FILE *fp = fopen(path, "r");
    if (!fp) return false;
    size_t n = fread(buf, 1, cap - 1, fp);
    buf[n] = '\0';
    fclose(fp);
    return true;
}

static uint64_t parse_meminfo_line(const char *line)
{
    while (*line && !isdigit((unsigned char)*line)) line++;
    return (uint64_t)strtoull(line, NULL, 10) * 1024; /* kB to bytes */
}

/* ─── API ──────────────────────────────────────────────────────────────── */

bool oc_numa_available(void)
{
    OcNumaTopology topo;
    OcError e = oc_numa_detect(&topo);
    return (e == OC_OK && topo.available && topo.n_nodes > 1);
}

OcError oc_numa_detect(OcNumaTopology *out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    /* Check if /sys/devices/system/node exists. */
    DIR *dir = opendir("/sys/devices/system/node");
    if (!dir) {
        /* No NUMA — single node. */
        out->n_nodes = 1;
        out->nodes[0].node_id = 0;
        out->nodes[0].cpu_count = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);
        out->n_cpus_total = out->nodes[0].cpu_count;
        out->available = false;
        return OC_OK;
    }

    uint32_t node_idx = 0;
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL && node_idx < OC_NUMA_MAX_NODES) {
        if (strncmp(ent->d_name, "node", 4) != 0) continue;
        if (!isdigit((unsigned char)ent->d_name[4])) continue;

        OcNumaNode *node = &out->nodes[node_idx];
        node->node_id = (uint32_t)atoi(ent->d_name + 4);

        /* Read CPU list. */
        char path[512];
        char buf[8192];
        int n = snprintf(path, sizeof(path),
                 "/sys/devices/system/node/%s/cpulist", ent->d_name);
        if (n < 0 || (size_t)n >= sizeof(path)) continue;
        if (read_file_to_buf(path, buf, sizeof(buf))) {
            /* Parse comma-separated ranges (e.g., "0-15,32-47"). */
            const char *p = buf;
            uint32_t cpu = 0;
            while (*p) {
                if (isdigit((unsigned char)*p)) {
                    cpu = (uint32_t)strtoul(p, (char **)&p, 10);
                    if (node->cpu_count < OC_NUMA_MAX_CPUS)
                        node->cpus[node->cpu_count++] = cpu;
                    if (*p == '-') {
                        p++;
                        uint32_t end = (uint32_t)strtoul(p, (char **)&p, 10);
                        for (uint32_t c = cpu + 1; c <= end && node->cpu_count < OC_NUMA_MAX_CPUS; c++)
                            node->cpus[node->cpu_count++] = c;
                    }
                } else {
                    p++;
                }
            }
        }

        /* Read memory info. */
        n = snprintf(path, sizeof(path),
                 "/sys/devices/system/node/%s/meminfo", ent->d_name);
        if (n < 0 || (size_t)n >= sizeof(path)) continue;
        if (read_file_to_buf(path, buf, sizeof(buf))) {
            char *line = strtok(buf, "\n");
            while (line) {
                if (strstr(line, "MemTotal:"))
                    node->memory_total = parse_meminfo_line(line);
                if (strstr(line, "MemFree:"))
                    node->memory_free = parse_meminfo_line(line);
                line = strtok(NULL, "\n");
            }
        }

        out->n_cpus_total += node->cpu_count;
        node_idx++;
    }
    closedir(dir);

    out->n_nodes = node_idx;
    out->available = (node_idx > 1);
    return OC_OK;
}

OcError oc_numa_node_for_cpu(uint32_t cpu, uint32_t *out_node)
{
    if (!out_node) return OC_ERR_INVALID_ARG;
    char path[128];
    char buf[16];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%u/topology/physical_package_id", cpu);
    if (!read_file_to_buf(path, buf, sizeof(buf)))
        return OC_ERR_IO;
    *out_node = (uint32_t)atoi(buf);
    return OC_OK;
}

OcError oc_numa_current_node(uint32_t *out_node)
{
    if (!out_node) return OC_ERR_INVALID_ARG;
#if defined(__linux__)
    int cpu = sched_getcpu();
    if (cpu < 0) return OC_ERR_IO;
    return oc_numa_node_for_cpu((uint32_t)cpu, out_node);
#else
    /* No sched_getcpu / NUMA topology outside Linux — report node 0. */
    *out_node = 0;
    return OC_OK;
#endif
}

/* Set this thread's memory policy for SUBSEQUENT allocations and page faults.
 *
 * Call before faulting in the weights: set_mempolicy does not migrate pages
 * that already exist, so ordering is the whole game. In particular the kernel
 * default is MPOL_DEFAULT — first touch allocates node-LOCAL, not interleaved
 * — so a large model faulted in by threads that happen to sit on one socket
 * ends up with its pages on that socket and every read from the other socket
 * crosses the interconnect. Interleave has to be requested explicitly. */
OcError oc_numa_set_policy(OcNumaMemPolicy policy, uint32_t node)
{
#if defined(__linux__)
    OcNumaTopology topo;
    if (oc_numa_detect(&topo) != OC_OK || !topo.available || topo.n_nodes <= 1)
        return OC_OK;  /* single node: nothing to place */

    unsigned long mask[OC_NODEMASK_WORDS];
    memset(mask, 0, sizeof(mask));
    int mode;

    switch (policy) {
    case OC_NUMA_POLICY_INTERLEAVE:
        mode = OC_MPOL_INTERLEAVE;
        for (uint32_t i = 0; i < topo.n_nodes && i < OC_NUMA_MAX_NODES; i++)
            nodemask_set(mask, i);
        break;
    case OC_NUMA_POLICY_BIND:
    case OC_NUMA_POLICY_PREFERRED:
        if (node >= topo.n_nodes) return OC_ERR_INVALID_ARG;
        mode = (policy == OC_NUMA_POLICY_BIND) ? OC_MPOL_BIND
                                               : OC_MPOL_PREFERRED;
        nodemask_set(mask, node);
        break;
    case OC_NUMA_POLICY_DEFAULT:
    default:
        /* MPOL_DEFAULT requires an empty nodemask. */
        if (sys_set_mempolicy(OC_MPOL_DEFAULT, NULL, 0) != 0)
            return OC_ERR_IO;
        return OC_OK;
    }

    if (sys_set_mempolicy(mode, mask, OC_NODEMASK_BITS) != 0)
        return OC_ERR_IO;
    return OC_OK;
#else
    (void)policy; (void)node;  /* NUMA policy is Linux-only. */
    return OC_OK;
#endif
}

OcError oc_numa_bind_thread(uint32_t node)
{
    OcNumaTopology topo;
    if (oc_numa_detect(&topo) != OC_OK || !topo.available)
        return OC_OK;

    if (node >= topo.n_nodes) return OC_ERR_INVALID_ARG;

#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    OcNumaNode *n = &topo.nodes[node];
    for (uint32_t i = 0; i < n->cpu_count; i++)
        CPU_SET(n->cpus[i], &cpuset);

    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0)
        return OC_ERR_IO;
    return OC_OK;
#else
    /* CPU affinity (sched_setaffinity/cpu_set_t) is Linux-only; no-op. */
    return OC_OK;
#endif
}

OcError oc_numa_pin_cpu(uint32_t cpu)
{
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0)
        return OC_ERR_IO;
    return OC_OK;
#else
    (void)cpu;  /* CPU affinity is Linux-only; no-op elsewhere. */
    return OC_OK;
#endif
}

/* ─── SMT-aware worker pinning ──────────────────────────────────────────
 *
 * Physical-core pinning for pool workers. Detects one CPU per physical
 * core via /sys thread_siblings_list; when the pool's thread count equals
 * the machine's physical-core count, each worker tid maps deterministi-
 * cally to one core. The µop-bound DFlash2 phases measured 19% slower
 * with SMT siblings (two workers contending for one core's FMA/LSU
 * throughput) than one-thread-per-core, so the pool consults this before
 * workers start. */

/* Parse "0-3,8,10-11" (or a single "N") into ascending CPU ids. */
static size_t parse_cpu_list(const char *s, uint32_t *out, size_t cap)
{
    size_t n = 0;
    const char *p = s;
    while (*p && n < cap) {
        if (!isdigit((unsigned char)*p)) { p++; continue; }
        uint32_t a = (uint32_t)strtoul(p, (char **)&p, 10);
        uint32_t b = a;
        if (*p == '-') {
            p++;
            b = (uint32_t)strtoul(p, (char **)&p, 10);
            if (b < a) b = a;
        }
        for (uint32_t c = a; c <= b && n < cap; c++) out[n++] = c;
    }
    return n;
}

/* One CPU per physical core (the lowest sibling id of each core), built
 * once under pthread_once — see oc_numa_distinct_core_for_worker. */
static uint32_t g_core_cpus[OC_NUMA_MAX_CPUS];
static size_t g_n_cores = (size_t)-1;

static void build_core_list_once(void)
{
    size_t n_cores = 0;
    uint32_t seen_siblings[OC_NUMA_MAX_CPUS];
    size_t n_seen = 0;
    for (uint32_t cpu = 0; cpu < OC_NUMA_MAX_CPUS; cpu++) {
        char path[128], buf[256];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%u/"
                 "topology/thread_siblings_list", cpu);
        if (!read_file_to_buf(path, buf, sizeof(buf))) continue;
        uint32_t sib[64];
        size_t ns = parse_cpu_list(buf, sib, 64);
        if (ns == 0) continue;
        /* Core representative = lowest sibling id; skip cores already
         * seen (each core appears once per sibling CPU). */
        uint32_t first = sib[0];
        bool dup = false;
        for (size_t i = 0; i < n_seen; i++)
            if (seen_siblings[i] == first) { dup = true; break; }
        if (dup) continue;
        if (n_seen < OC_NUMA_MAX_CPUS) seen_siblings[n_seen++] = first;
        if (n_cores < OC_NUMA_MAX_CPUS) g_core_cpus[n_cores++] = first;
    }
    g_n_cores = n_cores;
}

bool oc_numa_distinct_core_for_worker(size_t tid, size_t n_threads,
                                      uint32_t *out_cpu)
{
#if defined(__linux__)
    if (!out_cpu || n_threads == 0) return false;
    /* The pool consults this from every worker thread at startup, so the
     * list build must be race-free: pthread_once. */
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, build_core_list_once);
    if (g_n_cores == 0) return false;
    if (n_threads != g_n_cores || tid >= n_threads) return false;
    *out_cpu = g_core_cpus[tid];
    return true;
#else
    (void)tid; (void)n_threads; (void)out_cpu;
    return false;
#endif
}

/* Large allocations go through mmap so mbind() can place them; small ones
 * fall back to malloc, where per-node placement is not worth a syscall.
 * `mode`/`node` are applied with mbind before any page is touched, so the
 * placement takes effect on first fault. */
static void *numa_alloc_bound(size_t size, int mode, const unsigned long *mask)
{
    if (size < (1u << 20)) return malloc(size);

    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) return NULL;
#if defined(__linux__)
    /* Best-effort: an mbind failure leaves the mapping usable with default
     * placement, which is slower but correct. */
    (void)sys_mbind(ptr, size, mode, mask, OC_NODEMASK_BITS, 0);
#else
    (void)mode; (void)mask;
#endif
    return ptr;
}

void *oc_numa_alloc(size_t size, uint32_t node)
{
#if defined(__linux__)
    unsigned long mask[OC_NODEMASK_WORDS];
    memset(mask, 0, sizeof(mask));
    if (node < OC_NUMA_MAX_NODES) nodemask_set(mask, node);
    return numa_alloc_bound(size, OC_MPOL_BIND, mask);
#else
    (void)node;
    return numa_alloc_bound(size, 0, NULL);
#endif
}

void *oc_numa_alloc_interleaved(size_t size)
{
#if defined(__linux__)
    OcNumaTopology topo;
    unsigned long mask[OC_NODEMASK_WORDS];
    memset(mask, 0, sizeof(mask));
    if (oc_numa_detect(&topo) == OC_OK && topo.available) {
        for (uint32_t i = 0; i < topo.n_nodes && i < OC_NUMA_MAX_NODES; i++)
            nodemask_set(mask, i);
    }
    return numa_alloc_bound(size, OC_MPOL_INTERLEAVE, mask);
#else
    return numa_alloc_bound(size, 0, NULL);
#endif
}

void oc_numa_free(void *ptr, size_t size)
{
    if (!ptr) return;
    if (size >= (1u << 20))
        munmap(ptr, size);
    else
        free(ptr);
}

OcError oc_numa_addr_node(const void *addr, uint32_t *out_node)
{
    if (!addr || !out_node) return OC_ERR_INVALID_ARG;
    /* On Linux, we could use get_mempolicy with MPOL_F_ADDR. */
    *out_node = 0;
    return OC_OK;
}

OcError oc_numa_describe(const OcNumaTopology *topo,
                         char *out, size_t cap)
{
    if (!topo || !out || cap == 0) return OC_ERR_INVALID_ARG;
    int n = 0;
    n += snprintf(out + n, cap - n, "NUMA: %s, %u nodes, %u CPUs",
                  topo->available ? "available" : "unavailable",
                  topo->n_nodes, topo->n_cpus_total);
    for (uint32_t i = 0; i < topo->n_nodes && n < (int)cap - 64; i++) {
        n += snprintf(out + n, cap - n, "\n  node%u: %u CPUs, %llu MB",
                      topo->nodes[i].node_id,
                      topo->nodes[i].cpu_count,
                      (unsigned long long)(topo->nodes[i].memory_total / (1024 * 1024)));
    }
    return OC_OK;
}

uint32_t oc_numa_recommended_threads(const OcNumaTopology *topo)
{
    if (!topo || topo->n_nodes == 0) return 1;
    /* For single-node: use all CPUs.
     * For multi-node: use CPUs from one node (avoid cross-node access). */
    if (topo->n_nodes == 1)
        return topo->n_cpus_total > 0 ? topo->n_cpus_total : 1;
    /* For multi-node, recommend using half (one socket). */
    return topo->n_cpus_total / topo->n_nodes;
}
