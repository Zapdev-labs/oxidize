/*
 * expert_stream.h — SSD expert offload for sparse MoE models.
 *
 * Expert tensors stay file-backed. Decode only faults the routed experts,
 * prefetches a predicted set, and drops cold pages once the working set
 * exceeds a byte budget. Mirrors Edge0's streaming pool on GGUF mmap.
 */
#ifndef OXIDIZE_EXPERT_STREAM_H
#define OXIDIZE_EXPERT_STREAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

struct OcLlamaModel;

#define OC_EXPERT_STREAM_DEFAULT_CACHE (3ull << 30)

typedef struct OcExpertStreamConfig {
    uint64_t cache_bytes; /* 0 → OC_EXPERT_STREAM_DEFAULT_CACHE */
    bool     prefetch;
    bool     reclaim;
} OcExpertStreamConfig;

typedef struct OcExpertStreamPool OcExpertStreamPool;

void oc_expert_stream_config_init(OcExpertStreamConfig *cfg);

OcError oc_expert_stream_new(struct OcLlamaModel *model,
                             const OcExpertStreamConfig *cfg,
                             OcExpertStreamPool **out);

void oc_expert_stream_free(OcExpertStreamPool *pool);

/* Advise RANDOM on the mapping and pin non-expert tensors. */
OcError oc_expert_stream_prepare_mapping(OcExpertStreamPool *pool);

/* Fault / prefetch the listed experts of `layer`. */
OcError oc_expert_stream_touch(OcExpertStreamPool *pool, uint32_t layer,
                               const uint32_t *experts, uint32_t n,
                               bool prefetch);

/* Prefetch every expert of `layer` (prefill). */
OcError oc_expert_stream_touch_layer(OcExpertStreamPool *pool, uint32_t layer);

uint64_t oc_expert_stream_resident_bytes(const OcExpertStreamPool *pool);
uint64_t oc_expert_stream_prefetch_bytes(const OcExpertStreamPool *pool);
uint64_t oc_expert_stream_reclaim_bytes(const OcExpertStreamPool *pool);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_EXPERT_STREAM_H */
