#include <criterion/criterion.h>
#include <string.h>

#include "oxidize/video_config.h"


Test(video_config, init_defaults)
{
    OcVideoConfig cfg;
    oc_video_config_init(&cfg);
    cr_assert_eq(cfg.target_frames, 8u, "default target_frames");
    cr_assert_eq(cfg.sampling, OC_VID_SAMPLE_UNIFORM, "default sampling");
    cr_assert_eq(cfg.dense_stride, 1u, "default dense_stride");
    cr_assert_eq(cfg.temporal_pool, OC_VID_POOL_MEAN, "default temporal_pool");
    cr_assert_eq(cfg.temporal_hidden, 768u, "default temporal_hidden");
    cr_assert_eq(cfg.llm_hidden, 4096u, "default llm_hidden");
    cr_assert_eq(cfg.max_video_tokens, 256u, "default max_video_tokens");
}

Test(video_config, init_null_is_noop)
{
    /* Must not crash. */
    oc_video_config_init(NULL);
    cr_assert(true, "init(NULL) did not crash");
}


Test(video_config, validate_good_default)
{
    OcVideoConfig cfg;
    oc_video_config_init(&cfg);
    cr_assert_eq(oc_video_config_validate(&cfg), OC_OK, "default validates");
}

Test(video_config, validate_null)
{
    cr_assert_eq(oc_video_config_validate(NULL), OC_ERR_INVALID_ARG, "");
}

Test(video_config, validate_zero_target_frames)
{
    OcVideoConfig cfg;
    oc_video_config_init(&cfg);
    cfg.target_frames = 0;
    cr_assert_eq(oc_video_config_validate(&cfg), OC_ERR_INVALID_ARG, "");
}

Test(video_config, validate_zero_llm_hidden)
{
    OcVideoConfig cfg;
    oc_video_config_init(&cfg);
    cfg.llm_hidden = 0;
    cr_assert_eq(oc_video_config_validate(&cfg), OC_ERR_INVALID_ARG, "");
}

Test(video_config, validate_dense_zero_stride)
{
    OcVideoConfig cfg;
    oc_video_config_init(&cfg);
    cfg.sampling     = OC_VID_SAMPLE_DENSE;
    cfg.dense_stride = 0;
    cr_assert_eq(oc_video_config_validate(&cfg), OC_ERR_FORMAT, "");
}

Test(video_config, validate_dense_good_stride)
{
    OcVideoConfig cfg;
    oc_video_config_init(&cfg);
    cfg.sampling     = OC_VID_SAMPLE_DENSE;
    cfg.dense_stride = 4;
    cr_assert_eq(oc_video_config_validate(&cfg), OC_OK, "");
}

Test(video_config, validate_bad_sampling_enum)
{
    OcVideoConfig cfg;
    oc_video_config_init(&cfg);
    cfg.sampling = (OcFrameSamplingStrategy)99;
    cr_assert_eq(oc_video_config_validate(&cfg), OC_ERR_INVALID_ARG, "");
}

Test(video_config, validate_bad_pool_enum)
{
    OcVideoConfig cfg;
    oc_video_config_init(&cfg);
    cfg.temporal_pool = (OcTemporalPool)99;
    cr_assert_eq(oc_video_config_validate(&cfg), OC_ERR_INVALID_ARG, "");
}

Test(video_config, validate_zero_temporal_hidden)
{
    OcVideoConfig cfg;
    oc_video_config_init(&cfg);
    cfg.temporal_hidden = 0;
    cr_assert_eq(oc_video_config_validate(&cfg), OC_ERR_INVALID_ARG, "");
}

Test(video_config, validate_zero_max_tokens)
{
    OcVideoConfig cfg;
    oc_video_config_init(&cfg);
    cfg.max_video_tokens = 0;
    cr_assert_eq(oc_video_config_validate(&cfg), OC_ERR_INVALID_ARG, "");
}


Test(video_config, sampling_names)
{
    cr_assert_str_eq(oc_video_sampling_name(OC_VID_SAMPLE_UNIFORM),  "uniform", "");
    cr_assert_str_eq(oc_video_sampling_name(OC_VID_SAMPLE_DENSE),    "dense", "");
    cr_assert_str_eq(oc_video_sampling_name(OC_VID_SAMPLE_ADAPTIVE), "adaptive", "");
    cr_assert_str_eq(oc_video_sampling_name((OcFrameSamplingStrategy)42), "unknown", "");
}


Test(video_config, pool_names)
{
    cr_assert_str_eq(oc_video_pool_name(OC_VID_POOL_MEAN),      "mean", "");
    cr_assert_str_eq(oc_video_pool_name(OC_VID_POOL_MAX),       "max", "");
    cr_assert_str_eq(oc_video_pool_name(OC_VID_POOL_LAST),      "last", "");
    cr_assert_str_eq(oc_video_pool_name(OC_VID_POOL_ATTENTION), "attention", "");
    cr_assert_str_eq(oc_video_pool_name(OC_VID_POOL_LSTM),       "lstm", "");
    cr_assert_str_eq(oc_video_pool_name((OcTemporalPool)42),    "unknown", "");
}


Test(video_config, n_tokens_default)
{
    OcVideoConfig cfg;
    oc_video_config_init(&cfg);
    cr_assert_eq(oc_video_config_n_tokens(&cfg), 8u, "");
}

Test(video_config, n_tokens_capped)
{
    OcVideoConfig cfg;
    oc_video_config_init(&cfg);
    cfg.target_frames    = 1000u;
    cfg.max_video_tokens = 64u;
    cr_assert_eq(oc_video_config_n_tokens(&cfg), 64u, "capped to max");
}

Test(video_config, n_tokens_null)
{
    cr_assert_eq(oc_video_config_n_tokens(NULL), 0u, "");
}
