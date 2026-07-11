#include "oc.h"

#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef __NR_set_mempolicy
#define __NR_set_mempolicy 238
#endif
#ifndef __NR_mbind
#define __NR_mbind 237
#endif
#define OC_MPOL_BIND 2
#define OC_MPOL_MF_MOVE 2

static int *g_cpu_node = NULL;
static int g_ncpu_map = 0;
static int g_replica_node = -1;

typedef struct {
  const uint8_t *base;
  size_t size;
  const uint8_t *replica;
} oc_replica_range;

static oc_replica_range *g_replicas = NULL;
static size_t g_nreplicas = 0;
static size_t g_replica_cap = 0;
static int g_replicate_active = 0;

void oc_numa_set_cpu_node_map(const int *cpu_to_node, int ncpus) {
  free(g_cpu_node);
  g_cpu_node = NULL;
  g_ncpu_map = 0;
  if (!cpu_to_node || ncpus <= 0) return;
  g_cpu_node = malloc((size_t)ncpus * sizeof(int));
  if (!g_cpu_node) return;
  memcpy(g_cpu_node, cpu_to_node, (size_t)ncpus * sizeof(int));
  g_ncpu_map = ncpus;
}

void *oc_numa_alloc_bound(size_t bytes, int node) {
  if (bytes == 0) return NULL;
  void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return NULL;
  unsigned long mask = 1ul << (unsigned)node;
  syscall(__NR_mbind, p, bytes, OC_MPOL_BIND, &mask, 64ul, OC_MPOL_MF_MOVE);
  return p;
}

void oc_numa_register_replica(const uint8_t *primary, size_t size,
                              const uint8_t *replica) {
  if (!primary || !replica || size == 0) return;
  if (g_nreplicas >= g_replica_cap) {
    size_t cap = g_replica_cap ? g_replica_cap * 2 : 64;
    oc_replica_range *nr = realloc(g_replicas, cap * sizeof(*nr));
    if (!nr) return;
    g_replicas = nr;
    g_replica_cap = cap;
  }
  g_replicas[g_nreplicas++] = (oc_replica_range){primary, size, replica};
  g_replicate_active = 1;
}

static int current_thread_node(void) {
  static __thread int cached = -2;
  if (cached == -2) {
    int cpu = sched_getcpu();
    cached = (cpu >= 0 && cpu < g_ncpu_map) ? g_cpu_node[cpu] : 0;
  }
  return cached;
}

const uint8_t *oc_numa_local_data(const uint8_t *p) {
  if (!g_replicate_active || !p || g_replica_node < 0) return p;
  if (current_thread_node() != g_replica_node) return p;
  for (size_t i = 0; i < g_nreplicas; ++i) {
    const oc_replica_range *r = &g_replicas[i];
    if (p >= r->base && p < r->base + r->size)
      return r->replica + (size_t)(p - r->base);
  }
  return p;
}

int oc_numa_replicate_active(void) { return g_replicate_active; }

void oc_numa_set_replica_node(int node) { g_replica_node = node; }

void oc_numa_bind_alloc_node(int node) {
  unsigned long mask = 1ul << (unsigned)node;
  syscall(__NR_set_mempolicy, OC_MPOL_BIND, &mask, 64);
}
