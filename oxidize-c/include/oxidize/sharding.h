#ifndef OXIDIZE_SHARDING_H
#define OXIDIZE_SHARDING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OC_SHARD_MAX_TENSORS  64u
#define OC_SHARD_NAME_LEN    128u


/* Sharding strategy. Values are stable for serialization. */
typedef enum {
    OC_SHARD_ROW    = 0,  /* split along rows (output dim)             */
    OC_SHARD_COLUMN = 1,  /* split along columns (input dim)           */
    OC_SHARD_BLOCK  = 2,  /* split both dims into a grid              */
} OcShardStrategy;

typedef struct OcShardConfig {
    uint32_t          n_shards;  /* default 1                              */
    uint32_t          shard_id;  /* default 0 (this node's shard)         */
    OcShardStrategy   strategy;  /* default OC_SHARD_ROW                  */
} OcShardConfig;

typedef struct OcShardInfo {
    uint32_t shard_id;
    uint32_t start_row;
    uint32_t end_row;
    uint32_t start_col;
    uint32_t end_col;
    char     tensor_name[OC_SHARD_NAME_LEN];
    uint64_t size_bytes;
} OcShardInfo;

typedef struct OcShardManager {
    OcShardConfig config;
    OcShardInfo   shards[OC_SHARD_MAX_TENSORS];
    uint32_t      n_shards;     /* number of shard records currently stored */
} OcShardManager;


/* Initialize config with defaults. */
OcError oc_shard_config_init(OcShardConfig *cfg);

/* Human-readable strategy name (e.g. "ROW"). Never NULL. */
const char *oc_shard_strategy_name(OcShardStrategy strategy);


/* Allocate a shard manager. `config` may be NULL (defaults are used).
 * Free with oc_shard_free. */
OcError oc_shard_init(const OcShardConfig *config, OcShardManager **out);

/* Free all owned storage and reset state. Safe on NULL / already-freed. */
void oc_shard_free(OcShardManager *mgr);


/* Compute and store shard assignments for `tensor_name` of shape */
OcError oc_shard_assign(OcShardManager *mgr, const char *tensor_name,
                        uint32_t n_rows, uint32_t n_cols);

/* Copy the shard assignment for `tensor_name` at `shard_id` into `out_info`.
 * Returns OC_ERR_MODEL if not found. */
OcError oc_shard_get_assignment(const OcShardManager *mgr,
                                const char *tensor_name, uint32_t shard_id,
                                OcShardInfo *out_info);

/* Copy up to `max` shard assignments for `tensor_name` into `out_array`. */
OcError oc_shard_get_all_assignments(const OcShardManager *mgr,
                                     const char *tensor_name,
                                     OcShardInfo *out_array, uint32_t max,
                                     uint32_t *out_count);


/* Number of shard records currently stored. */
uint32_t oc_shard_n_shards(const OcShardManager *mgr);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SHARDING_H */
