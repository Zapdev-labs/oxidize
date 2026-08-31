/* numa.h — NUMA awareness for thread affinity and memory allocation. */
#ifndef OXIDIZE_NUMA_H
#define OXIDIZE_NUMA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OC_NUMA_MAX_NODES 16
#define OC_NUMA_MAX_CPUS 1024


typedef struct {
    uint32_t node_id;
    uint32_t cpu_count;
    uint32_t cpus[OC_NUMA_MAX_CPUS];
    uint64_t memory_total;
    uint64_t memory_free;
} OcNumaNode;

typedef struct {
    uint32_t n_nodes;
    OcNumaNode nodes[OC_NUMA_MAX_NODES];
    uint32_t n_cpus_total;
    bool available;
} OcNumaTopology;

typedef enum {
    OC_NUMA_POLICY_DEFAULT = 0,    /* system default */
    OC_NUMA_POLICY_BIND = 1,      /* bind to specific node */
    OC_NUMA_POLICY_INTERLEAVE = 2, /* interleave across nodes */
    OC_NUMA_POLICY_PREFERRED = 3,  /* prefer a node */
} OcNumaMemPolicy;


/* Detect NUMA topology. Returns OC_OK if NUMA is available. */
OcError oc_numa_detect(OcNumaTopology *out);

/* Get the NUMA node for a given CPU. */
OcError oc_numa_node_for_cpu(uint32_t cpu, uint32_t *out_node);

/* Get the NUMA node for the calling thread. */
OcError oc_numa_current_node(uint32_t *out_node);

/* Set the NUMA memory policy for the calling thread. */
OcError oc_numa_set_policy(OcNumaMemPolicy policy, uint32_t node);

/* Bind the calling thread to a specific NUMA node's CPUs. */
OcError oc_numa_bind_thread(uint32_t node);

/* Pin a thread to specific CPU. */
OcError oc_numa_pin_cpu(uint32_t cpu);

/* Allocate memory on a specific NUMA node. */
void *oc_numa_alloc(size_t size, uint32_t node);

/* Allocate interleaved memory across all nodes. */
void *oc_numa_alloc_interleaved(size_t size);

/* Free NUMA-allocated memory. */
void oc_numa_free(void *ptr, size_t size);

/* Get the NUMA node of a memory address. */
OcError oc_numa_addr_node(const void *addr, uint32_t *out_node);

/* Get a string description of the topology. */
OcError oc_numa_describe(const OcNumaTopology *topo,
                         char *out, size_t cap);

/* Get recommended thread count for inference based on NUMA topology. */
uint32_t oc_numa_recommended_threads(const OcNumaTopology *topo);

/* Check if NUMA is available on this system. */
bool oc_numa_available(void);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_NUMA_H */
