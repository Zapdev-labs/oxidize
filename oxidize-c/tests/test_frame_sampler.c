/* test_frame_sampler.c — Frame sampler tests. */
#include "framework.h"
#include "oxidize/frame_sampler.h"
#include <string.h>

Test(fs, config_init)
{
    OcFsConfig cfg;
    cr_assert_eq(oc_fs_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.strategy, OC_FS_STRATEGY_UNIFORM);
    cr_assert_eq(cfg.n_frames, 8);
    cr_assert_eq(cfg.fps, 2);
    cr_assert_eq(cfg.seed, 42);
}

Test(fs, config_init_null)
{
    cr_assert_neq(oc_fs_config_init(NULL), OC_OK);
}

Test(fs, uniform)
{
    OcFsResult r;
    cr_assert_eq(oc_fs_sample_uniform(100, 10, &r), OC_OK);
    cr_assert_eq(r.n_indices, 10);
    cr_assert_eq(r.indices[0], 0);
    cr_assert_eq(r.indices[9], 90);
    oc_fs_result_free(&r);
}

Test(fs, uniform_more_than_total)
{
    OcFsResult r;
    cr_assert_eq(oc_fs_sample_uniform(5, 10, &r), OC_OK);
    cr_assert_eq(r.n_indices, 5);
    oc_fs_result_free(&r);
}

Test(fs, uniform_single)
{
    OcFsResult r;
    cr_assert_eq(oc_fs_sample_uniform(1, 1, &r), OC_OK);
    cr_assert_eq(r.n_indices, 1);
    cr_assert_eq(r.indices[0], 0);
    oc_fs_result_free(&r);
}

Test(fs, first_n)
{
    OcFsResult r;
    cr_assert_eq(oc_fs_sample_first_n(100, 5, &r), OC_OK);
    cr_assert_eq(r.n_indices, 5);
    cr_assert_eq(r.indices[0], 0);
    cr_assert_eq(r.indices[4], 4);
    oc_fs_result_free(&r);
}

Test(fs, first_n_more)
{
    OcFsResult r;
    cr_assert_eq(oc_fs_sample_first_n(3, 10, &r), OC_OK);
    cr_assert_eq(r.n_indices, 3);
    oc_fs_result_free(&r);
}

Test(fs, last_n)
{
    OcFsResult r;
    cr_assert_eq(oc_fs_sample_last_n(100, 5, &r), OC_OK);
    cr_assert_eq(r.n_indices, 5);
    cr_assert_eq(r.indices[0], 95);
    cr_assert_eq(r.indices[4], 99);
    oc_fs_result_free(&r);
}

Test(fs, last_n_more)
{
    OcFsResult r;
    cr_assert_eq(oc_fs_sample_last_n(3, 10, &r), OC_OK);
    cr_assert_eq(r.n_indices, 3);
    cr_assert_eq(r.indices[0], 0);
    oc_fs_result_free(&r);
}

Test(fs, random)
{
    OcFsResult r;
    cr_assert_eq(oc_fs_sample_random(100, 10, 42, &r), OC_OK);
    cr_assert_eq(r.n_indices, 10);
    /* All indices should be < 100. */
    for (uint32_t i = 0; i < 10; i++)
        cr_assert_lt(r.indices[i], 100);
    oc_fs_result_free(&r);
}

Test(fs, random_reproducible)
{
    OcFsResult r1, r2;
    oc_fs_sample_random(100, 10, 123, &r1);
    oc_fs_sample_random(100, 10, 123, &r2);
    cr_assert_eq(r1.n_indices, r2.n_indices);
    for (uint32_t i = 0; i < r1.n_indices; i++)
        cr_assert_eq(r1.indices[i], r2.indices[i]);
    oc_fs_result_free(&r1);
    oc_fs_result_free(&r2);
}

Test(fs, sample_config)
{
    OcFsConfig cfg;
    oc_fs_config_init(&cfg);
    cfg.n_frames = 5;
    OcFsResult r;
    cr_assert_eq(oc_fs_sample(&cfg, 10000, 100, &r), OC_OK);
    cr_assert_eq(r.n_indices, 5);
    oc_fs_result_free(&r);
}

Test(fs, sample_null)
{
    cr_assert_neq(oc_fs_sample(NULL, 0, 0, NULL), OC_OK);
}

Test(fs, estimate_n_frames)
{
    cr_assert_eq(oc_fs_estimate_n_frames(10000, 2, 64), 20);
    cr_assert_eq(oc_fs_estimate_n_frames(10000, 2, 10), 10);
    cr_assert_eq(oc_fs_estimate_n_frames(0, 2, 10), 0);
    cr_assert_eq(oc_fs_estimate_n_frames(1000, 0, 10), 0);
}

Test(fs, strategy_name)
{
    cr_assert_str_eq(oc_fs_strategy_name(OC_FS_STRATEGY_UNIFORM), "uniform");
    cr_assert_str_eq(oc_fs_strategy_name(OC_FS_STRATEGY_FIRST_N), "first_n");
    cr_assert_str_eq(oc_fs_strategy_name(OC_FS_STRATEGY_LAST_N), "last_n");
    cr_assert_str_eq(oc_fs_strategy_name(OC_FS_STRATEGY_RANDOM), "random");
    cr_assert_str_eq(oc_fs_strategy_name(OC_FS_STRATEGY_KEYFRAME), "keyframe");
}

Test(fs, result_free_null)
{
    oc_fs_result_free(NULL);
}

Test(fs, uniform_zero)
{
    OcFsResult r;
    cr_assert_neq(oc_fs_sample_uniform(0, 10, &r), OC_OK);
    cr_assert_neq(oc_fs_sample_uniform(10, 0, &r), OC_OK);
}
