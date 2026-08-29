/* test_sharding.c — sharding tests. */
#include "framework.h"
#include "oxidize/sharding.h"
#include <string.h>

Test(shard, config_init)
{
    OcShardConfig cfg;
    cr_assert_eq(oc_shard_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.n_shards, 1);
    cr_assert_eq(cfg.shard_id, 0);
    cr_assert_eq(cfg.strategy, OC_SHARD_ROW);
}

Test(shard, config_init_null)
{
    cr_assert_neq(oc_shard_config_init(NULL), OC_OK);
}

Test(shard, strategy_name)
{
    cr_assert_str_eq(oc_shard_strategy_name(OC_SHARD_ROW), "ROW");
    cr_assert_str_eq(oc_shard_strategy_name(OC_SHARD_COLUMN), "COLUMN");
    cr_assert_str_eq(oc_shard_strategy_name(OC_SHARD_BLOCK), "BLOCK");
}

Test(shard, init_free)
{
    OcShardManager *mgr = NULL;
    cr_assert_eq(oc_shard_init(NULL, &mgr), OC_OK);
    cr_assert_not_null(mgr);
    cr_assert_eq(mgr->config.n_shards, 1);
    cr_assert_eq(oc_shard_n_shards(mgr), 0);
    oc_shard_free(mgr);
}

Test(shard, init_null_out)
{
    cr_assert_neq(oc_shard_init(NULL, NULL), OC_OK);
}

Test(shard, init_zero_shards)
{
    OcShardConfig cfg;
    oc_shard_config_init(&cfg);
    cfg.n_shards = 0;
    OcShardManager *mgr = NULL;
    cr_assert_neq(oc_shard_init(&cfg, &mgr), OC_OK);
}

Test(shard, assign_row)
{
    OcShardConfig cfg;
    oc_shard_config_init(&cfg);
    cfg.n_shards = 4;
    cfg.strategy = OC_SHARD_ROW;
    OcShardManager *mgr = NULL;
    oc_shard_init(&cfg, &mgr);
    cr_assert_eq(oc_shard_assign(mgr, "wq", 100, 50), OC_OK);
    cr_assert_eq(oc_shard_n_shards(mgr), 4);
    /* Row strategy: start_col=0, end_col=50 for all shards. */
    for (uint32_t i = 0; i < 4; i++) {
        OcShardInfo info;
        cr_assert_eq(oc_shard_get_assignment(mgr, "wq", i, &info), OC_OK);
        cr_assert_eq(info.start_col, 0);
        cr_assert_eq(info.end_col, 50);
        cr_assert_eq(info.size_bytes, (info.end_row - info.start_row) * 50 * sizeof(float));
    }
    /* Total rows must cover 100. */
    uint32_t total = 0;
    for (uint32_t i = 0; i < 4; i++) {
        OcShardInfo info;
        oc_shard_get_assignment(mgr, "wq", i, &info);
        total += info.end_row - info.start_row;
    }
    cr_assert_eq(total, 100);
    oc_shard_free(mgr);
}

Test(shard, assign_column)
{
    OcShardConfig cfg;
    oc_shard_config_init(&cfg);
    cfg.n_shards = 4;
    cfg.strategy = OC_SHARD_COLUMN;
    OcShardManager *mgr = NULL;
    oc_shard_init(&cfg, &mgr);
    cr_assert_eq(oc_shard_assign(mgr, "wk", 50, 100), OC_OK);
    for (uint32_t i = 0; i < 4; i++) {
        OcShardInfo info;
        cr_assert_eq(oc_shard_get_assignment(mgr, "wk", i, &info), OC_OK);
        cr_assert_eq(info.start_row, 0);
        cr_assert_eq(info.end_row, 50);
    }
    /* Total cols must cover 100. */
    uint32_t total = 0;
    for (uint32_t i = 0; i < 4; i++) {
        OcShardInfo info;
        oc_shard_get_assignment(mgr, "wk", i, &info);
        total += info.end_col - info.start_col;
    }
    cr_assert_eq(total, 100);
    oc_shard_free(mgr);
}

Test(shard, assign_block)
{
    OcShardConfig cfg;
    oc_shard_config_init(&cfg);
    cfg.n_shards = 4;
    cfg.strategy = OC_SHARD_BLOCK;
    OcShardManager *mgr = NULL;
    oc_shard_init(&cfg, &mgr);
    cr_assert_eq(oc_shard_assign(mgr, "wb", 100, 100), OC_OK);
    cr_assert_eq(oc_shard_n_shards(mgr), 4);
    /* Every shard should have non-zero extent in both dims. */
    for (uint32_t i = 0; i < 4; i++) {
        OcShardInfo info;
        cr_assert_eq(oc_shard_get_assignment(mgr, "wb", i, &info), OC_OK);
        cr_assert(info.end_row > info.start_row);
        cr_assert(info.end_col > info.start_col);
    }
    oc_shard_free(mgr);
}

Test(shard, assign_single_shard)
{
    OcShardConfig cfg;
    oc_shard_config_init(&cfg);
    cfg.n_shards = 1;
    OcShardManager *mgr = NULL;
    oc_shard_init(&cfg, &mgr);
    cr_assert_eq(oc_shard_assign(mgr, "w", 10, 20), OC_OK);
    OcShardInfo info;
    cr_assert_eq(oc_shard_get_assignment(mgr, "w", 0, &info), OC_OK);
    cr_assert_eq(info.start_row, 0);
    cr_assert_eq(info.end_row, 10);
    cr_assert_eq(info.start_col, 0);
    cr_assert_eq(info.end_col, 20);
    cr_assert_eq(info.size_bytes, 10 * 20 * sizeof(float));
    oc_shard_free(mgr);
}

Test(shard, assign_uneven)
{
    OcShardConfig cfg;
    oc_shard_config_init(&cfg);
    cfg.n_shards = 3;
    cfg.strategy = OC_SHARD_ROW;
    OcShardManager *mgr = NULL;
    oc_shard_init(&cfg, &mgr);
    cr_assert_eq(oc_shard_assign(mgr, "w", 10, 5), OC_OK);
    /* 10 / 3 -> shards of sizes 4, 3, 3 (first gets remainder). */
    uint32_t sizes[3] = {0, 0, 0};
    for (uint32_t i = 0; i < 3; i++) {
        OcShardInfo info;
        oc_shard_get_assignment(mgr, "w", i, &info);
        sizes[i] = info.end_row - info.start_row;
    }
    cr_assert_eq(sizes[0] + sizes[1] + sizes[2], 10);
    cr_assert_eq(sizes[0], 4);
    cr_assert_eq(sizes[1], 3);
    cr_assert_eq(sizes[2], 3);
    oc_shard_free(mgr);
}

Test(shard, assign_invalid_dims)
{
    OcShardConfig cfg;
    oc_shard_config_init(&cfg);
    cfg.n_shards = 2;
    OcShardManager *mgr = NULL;
    oc_shard_init(&cfg, &mgr);
    cr_assert_neq(oc_shard_assign(mgr, "w", 0, 10), OC_OK);
    cr_assert_neq(oc_shard_assign(mgr, "w", 10, 0), OC_OK);
    oc_shard_free(mgr);
}

Test(shard, assign_null_name)
{
    OcShardManager *mgr = NULL;
    oc_shard_init(NULL, &mgr);
    cr_assert_neq(oc_shard_assign(mgr, NULL, 10, 10), OC_OK);
    oc_shard_free(mgr);
}

Test(shard, get_assignment_not_found)
{
    OcShardManager *mgr = NULL;
    oc_shard_init(NULL, &mgr);
    OcShardInfo info;
    cr_assert_neq(oc_shard_get_assignment(mgr, "missing", 0, &info), OC_OK);
    oc_shard_free(mgr);
}

Test(shard, get_all_assignments)
{
    OcShardConfig cfg;
    oc_shard_config_init(&cfg);
    cfg.n_shards = 2;
    OcShardManager *mgr = NULL;
    oc_shard_init(&cfg, &mgr);
    oc_shard_assign(mgr, "a", 10, 10);
    oc_shard_assign(mgr, "b", 10, 10);
    uint32_t count = 0;
    cr_assert_eq(oc_shard_get_all_assignments(mgr, NULL, NULL, 0, &count), OC_OK);
    cr_assert_eq(count, 4);
    oc_shard_free(mgr);
}

Test(shard, get_all_for_tensor)
{
    OcShardConfig cfg;
    oc_shard_config_init(&cfg);
    cfg.n_shards = 3;
    OcShardManager *mgr = NULL;
    oc_shard_init(&cfg, &mgr);
    oc_shard_assign(mgr, "a", 30, 30);
    oc_shard_assign(mgr, "b", 30, 30);
    OcShardInfo arr[8];
    uint32_t count = 0;
    cr_assert_eq(oc_shard_get_all_assignments(mgr, "a", arr, 8, &count), OC_OK);
    cr_assert_eq(count, 3);
    for (uint32_t i = 0; i < count; i++) {
        cr_assert_str_eq(arr[i].tensor_name, "a");
    }
    oc_shard_free(mgr);
}

Test(shard, n_shards_after_assign)
{
    OcShardConfig cfg;
    oc_shard_config_init(&cfg);
    cfg.n_shards = 2;
    OcShardManager *mgr = NULL;
    oc_shard_init(&cfg, &mgr);
    cr_assert_eq(oc_shard_n_shards(mgr), 0);
    oc_shard_assign(mgr, "w", 10, 10);
    cr_assert_eq(oc_shard_n_shards(mgr), 2);
    oc_shard_assign(mgr, "w2", 10, 10);
    cr_assert_eq(oc_shard_n_shards(mgr), 4);
    oc_shard_free(mgr);
}

Test(shard, free_null_safe)
{
    oc_shard_free(NULL);
}

Test(shard, assign_capacity)
{
    OcShardConfig cfg;
    oc_shard_config_init(&cfg);
    cfg.n_shards = OC_SHARD_MAX_TENSORS;
    OcShardManager *mgr = NULL;
    oc_shard_init(&cfg, &mgr);
    cr_assert_eq(oc_shard_assign(mgr, "w", 1000, 1000), OC_OK);
    cr_assert_eq(oc_shard_n_shards(mgr), OC_SHARD_MAX_TENSORS);
    /* Adding more should fail. */
    cr_assert_neq(oc_shard_assign(mgr, "w2", 10, 10), OC_OK);
    oc_shard_free(mgr);
}
