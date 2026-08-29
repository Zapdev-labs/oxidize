#include "oxidize/sharding.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static void shard_set_name(OcShardInfo *info, const char *name)
{
    if (!name) {
        info->tensor_name[0] = '\0';
        return;
    }
    size_t len = strlen(name);
    if (len >= OC_SHARD_NAME_LEN) len = OC_SHARD_NAME_LEN - 1;
    memcpy(info->tensor_name, name, len);
    info->tensor_name[len] = '\0';
}

static uint64_t shard_size_bytes(uint32_t rows, uint32_t cols)
{
    return (uint64_t)rows * (uint64_t)cols * sizeof(float);
}


OcError oc_shard_config_init(OcShardConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    cfg->n_shards = 1;
    cfg->shard_id = 0;
    cfg->strategy = OC_SHARD_ROW;
    return OC_OK;
}

const char *oc_shard_strategy_name(OcShardStrategy strategy)
{
    switch (strategy) {
    case OC_SHARD_ROW:    return "ROW";
    case OC_SHARD_COLUMN: return "COLUMN";
    case OC_SHARD_BLOCK:  return "BLOCK";
    default:              return "UNKNOWN";
    }
}


OcError oc_shard_init(const OcShardConfig *config, OcShardManager **out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    *out = NULL;

    OcShardConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        oc_shard_config_init(&cfg);
    }
    if (cfg.n_shards == 0) return OC_ERR_INVALID_ARG;

    OcShardManager *m = malloc(sizeof(*m));
    if (!m) return OC_ERR_OOM;
    memset(m, 0, sizeof(*m));
    m->config   = cfg;
    m->n_shards = 0;
    *out = m;
    return OC_OK;
}

void oc_shard_free(OcShardManager *mgr)
{
    if (!mgr) return;
    memset(mgr, 0, sizeof(*mgr));
    free(mgr);
}


OcError oc_shard_assign(OcShardManager *mgr, const char *tensor_name,
                        uint32_t n_rows, uint32_t n_cols)
{
    if (!mgr || !tensor_name) return OC_ERR_INVALID_ARG;
    if (n_rows == 0 || n_cols == 0) return OC_ERR_INVALID_ARG;
    if (mgr->config.n_shards == 0) return OC_ERR_INVALID_ARG;

    uint32_t need = mgr->config.n_shards;
    if (mgr->n_shards + need > OC_SHARD_MAX_TENSORS) return OC_ERR_OOM;

    uint32_t n_shards = mgr->config.n_shards;
    OcShardStrategy strat = mgr->config.strategy;

    for (uint32_t s = 0; s < n_shards; s++) {
        OcShardInfo *info = &mgr->shards[mgr->n_shards + s];
        memset(info, 0, sizeof(*info));
        info->shard_id = s;
        shard_set_name(info, tensor_name);

        if (strat == OC_SHARD_ROW) {
            uint32_t base = n_rows / n_shards;
            uint32_t rem  = n_rows % n_shards;
            uint32_t start = s * base + (s < rem ? s : rem);
            uint32_t end   = start + base + (s < rem ? 1 : 0);
            info->start_row = start;
            info->end_row   = end;
            info->start_col = 0;
            info->end_col   = n_cols;
        } else if (strat == OC_SHARD_COLUMN) {
            uint32_t base = n_cols / n_shards;
            uint32_t rem  = n_cols % n_shards;
            uint32_t start = s * base + (s < rem ? s : rem);
            uint32_t end   = start + base + (s < rem ? 1 : 0);
            info->start_row = 0;
            info->end_row   = n_rows;
            info->start_col = start;
            info->end_col   = end;
        } else { /* OC_SHARD_BLOCK */
            /* Choose grid as close to a square as possible: rows split into
             * `nr` parts, cols into `nc` parts, nr*nc >= n_shards. We pick
             * nr = ceil(sqrt(n_shards)) and nc = ceil(n_shards / nr). */
            uint32_t nr = 1;
            while (nr * nr < n_shards) nr++;
            uint32_t nc = (n_shards + nr - 1) / nr;
            if (nc == 0) nc = 1;

            uint32_t grid_row = s / nc;
            uint32_t grid_col = s % nc;

            uint32_t rbase = n_rows / nr;
            uint32_t rrem  = n_rows % nr;
            info->start_row = grid_row * rbase + (grid_row < rrem ? grid_row : rrem);
            info->end_row   = info->start_row + rbase + (grid_row < rrem ? 1 : 0);

            uint32_t cbase = n_cols / nc;
            uint32_t crem  = n_cols % nc;
            info->start_col = grid_col * cbase + (grid_col < crem ? grid_col : crem);
            info->end_col   = info->start_col + cbase + (grid_col < crem ? 1 : 0);
        }

        info->size_bytes = shard_size_bytes(info->end_row - info->start_row,
                                            info->end_col - info->start_col);
    }

    mgr->n_shards += need;
    return OC_OK;
}

OcError oc_shard_get_assignment(const OcShardManager *mgr,
                                const char *tensor_name, uint32_t shard_id,
                                OcShardInfo *out_info)
{
    if (!mgr || !tensor_name || !out_info) return OC_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < mgr->n_shards; i++) {
        if (mgr->shards[i].shard_id != shard_id) continue;
        if (strcmp(mgr->shards[i].tensor_name, tensor_name) != 0) continue;
        *out_info = mgr->shards[i];
        return OC_OK;
    }
    return OC_ERR_MODEL;
}

OcError oc_shard_get_all_assignments(const OcShardManager *mgr,
                                     const char *tensor_name,
                                     OcShardInfo *out_array, uint32_t max,
                                     uint32_t *out_count)
{
    if (!mgr || !out_count) return OC_ERR_INVALID_ARG;
    uint32_t count = 0;
    for (uint32_t i = 0; i < mgr->n_shards; i++) {
        if (tensor_name &&
            strcmp(mgr->shards[i].tensor_name, tensor_name) != 0) {
            continue;
        }
        if (out_array && count < max) {
            out_array[count] = mgr->shards[i];
        }
        count++;
    }
    *out_count = count;
    return OC_OK;
}


uint32_t oc_shard_n_shards(const OcShardManager *mgr)
{
    return mgr ? mgr->n_shards : 0;
}
