/*
 * numa.c — NUMA awareness implementation.
 *
 * Uses Linux sysfs for NUMA topology detection. On non-Linux systems
 * or systems without NUMA, returns OC_OK with a single-node topology.
 */
#define _GNU_SOURCE
#include "oxidize/numa.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

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
    int cpu = sched_getcpu();
    if (cpu < 0) return OC_ERR_IO;
    return oc_numa_node_for_cpu((uint32_t)cpu, out_node);
}

OcError oc_numa_set_policy(OcNumaPolicy policy, uint32_t node)
{
    (void)policy;
    (void)node;
    /* numa_set_* functions require libnuma. We use the syscall interface
     * via set_mempolicy for interleave/bind. */
    /* For now, this is a no-op on systems without libnuma. */
    return OC_OK;
}

OcError oc_numa_bind_thread(uint32_t node)
{
    OcNumaTopology topo;
    if (oc_numa_detect(&topo) != OC_OK || !topo.available)
        return OC_OK;

    if (node >= topo.n_nodes) return OC_ERR_INVALID_ARG;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    OcNumaNode *n = &topo.nodes[node];
    for (uint32_t i = 0; i < n->cpu_count; i++)
        CPU_SET(n->cpus[i], &cpuset);

    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0)
        return OC_ERR_IO;
    return OC_OK;
}

OcError oc_numa_pin_cpu(uint32_t cpu)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0)
        return OC_ERR_IO;
    return OC_OK;
}

void *oc_numa_alloc(size_t size, uint32_t node)
{
    (void)node;
    /* Use mmap for large allocations, malloc for small. */
    if (size >= (1u << 20)) {
        void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) return NULL;
        return ptr;
    }
    return malloc(size);
}

void *oc_numa_alloc_interleaved(size_t size)
{
    if (size >= (1u << 20)) {
        void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) return NULL;
        return ptr;
    }
    return malloc(size);
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
