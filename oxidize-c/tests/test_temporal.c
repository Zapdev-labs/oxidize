/* test_temporal.c — OcTemporal aggregation tests. */
#include <criterion/criterion.h>
#include "oxidize/temporal.h"

#include <math.h>
#include <string.h>

Test(temporal, config_init_defaults)
{
    OcTemporalConfig cfg;
    cr_assert_eq(oc_temporal_config_init(&cfg), OC_OK, "");
    cr_assert_eq(cfg.agg_type, OC_TEMPORAL_MEAN, "");
    cr_assert_eq(cfg.n_frames, 8, "");
    cr_assert_eq(cfg.feature_dim, 768, "");
    cr_assert_eq(cfg.hidden_dim, 0, "");
}

Test(temporal, config_init_null)
{
    cr_assert_eq(oc_temporal_config_init(NULL), OC_ERR_INVALID_ARG, "");
}

Test(temporal, agg_type_name)
{
    cr_assert_str_eq(oc_temporal_agg_type_name(OC_TEMPORAL_MEAN), "mean", "");
    cr_assert_str_eq(oc_temporal_agg_type_name(OC_TEMPORAL_MAX), "max", "");
    cr_assert_str_eq(oc_temporal_agg_type_name(OC_TEMPORAL_LAST), "last", "");
    cr_assert_str_eq(oc_temporal_agg_type_name(OC_TEMPORAL_ATTENTION), "attention", "");
    cr_assert_str_eq(oc_temporal_agg_type_name(OC_TEMPORAL_LSTM), "lstm", "");
    cr_assert_str_eq(oc_temporal_agg_type_name(99), "unknown", "");
}

Test(temporal, init_and_free)
{
    OcTemporalConfig cfg;
    oc_temporal_config_init(&cfg);
    cfg.feature_dim = 4;
    OcTemporalState st;
    cr_assert_eq(oc_temporal_init(&st, &cfg), OC_OK, "");
    cr_assert_not_null(st.output, "");
    cr_assert_eq(st.n_output, 4, "");
    oc_temporal_free(&st);
    cr_assert_null(st.output, "");
    /* double free safe */
    oc_temporal_free(&st);
}

Test(temporal, init_null)
{
    OcTemporalConfig cfg;
    oc_temporal_config_init(&cfg);
    cr_assert_eq(oc_temporal_init(NULL, &cfg), OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_temporal_init((OcTemporalState *)0x1, NULL), OC_ERR_INVALID_ARG, "");
}

Test(temporal, init_zero_dim_error)
{
    OcTemporalConfig cfg;
    oc_temporal_config_init(&cfg);
    cfg.feature_dim = 0;
    OcTemporalState st;
    cr_assert_eq(oc_temporal_init(&st, &cfg), OC_ERR_INVALID_ARG, "");
}

Test(temporal, mean_pooling)
{
    /* 2 frames, dim 3. */
    float features[] = { 1, 2, 3,  4, 5, 6 };
    float out[3];
    cr_assert_eq(oc_temporal_aggregate_mean(features, 2, 3, out), OC_OK, "");
    cr_assert_float_eq(out[0], 2.5f, 1e-6, "out[0]");
    cr_assert_float_eq(out[1], 3.5f, 1e-6, "out[1]");
    cr_assert_float_eq(out[2], 4.5f, 1e-6, "out[2]");
}

Test(temporal, max_pooling)
{
    float features[] = { 1, 5, 3,  4, 2, 6 };
    float out[3];
    cr_assert_eq(oc_temporal_aggregate_max(features, 2, 3, out), OC_OK, "");
    cr_assert_float_eq(out[0], 4.0f, 1e-6, "");
    cr_assert_float_eq(out[1], 5.0f, 1e-6, "");
    cr_assert_float_eq(out[2], 6.0f, 1e-6, "");
}

Test(temporal, last_frame)
{
    float features[] = { 1, 2, 3,  7, 8, 9,  4, 5, 6 };
    float out[3];
    cr_assert_eq(oc_temporal_aggregate_last(features, 3, 3, out), OC_OK, "");
    cr_assert_float_eq(out[0], 4.0f, 1e-6, "");
    cr_assert_float_eq(out[1], 5.0f, 1e-6, "");
    cr_assert_float_eq(out[2], 6.0f, 1e-6, "");
}

Test(temporal, aggregate_mean_stateful)
{
    OcTemporalConfig cfg;
    oc_temporal_config_init(&cfg);
    cfg.feature_dim = 2;
    cfg.n_frames    = 2;
    OcTemporalState st;
    cr_assert_eq(oc_temporal_init(&st, &cfg), OC_OK, "");

    float features[] = { 2, 4,  6, 8 };
    float out[2];
    cr_assert_eq(oc_temporal_aggregate(&st, features, 2, out), OC_OK, "");
    cr_assert_float_eq(out[0], 4.0f, 1e-6, "");
    cr_assert_float_eq(out[1], 6.0f, 1e-6, "");
    /* State output should match. */
    cr_assert_float_eq(st.output[0], 4.0f, 1e-6, "");
    cr_assert_float_eq(st.output[1], 6.0f, 1e-6, "");
    oc_temporal_free(&st);
}

Test(temporal, aggregate_max_stateful)
{
    OcTemporalConfig cfg;
    oc_temporal_config_init(&cfg);
    cfg.agg_type    = OC_TEMPORAL_MAX;
    cfg.feature_dim = 2;
    OcTemporalState st;
    oc_temporal_init(&st, &cfg);

    float features[] = { 1, 9,  3, 2 };
    float out[2];
    cr_assert_eq(oc_temporal_aggregate(&st, features, 2, out), OC_OK, "");
    cr_assert_float_eq(out[0], 3.0f, 1e-6, "");
    cr_assert_float_eq(out[1], 9.0f, 1e-6, "");
    oc_temporal_free(&st);
}

Test(temporal, aggregate_last_stateful)
{
    OcTemporalConfig cfg;
    oc_temporal_config_init(&cfg);
    cfg.agg_type    = OC_TEMPORAL_LAST;
    cfg.feature_dim = 2;
    OcTemporalState st;
    oc_temporal_init(&st, &cfg);

    float features[] = { 1, 2,  3, 4,  5, 6 };
    float out[2];
    cr_assert_eq(oc_temporal_aggregate(&st, features, 3, out), OC_OK, "");
    cr_assert_float_eq(out[0], 5.0f, 1e-6, "");
    cr_assert_float_eq(out[1], 6.0f, 1e-6, "");
    oc_temporal_free(&st);
}

Test(temporal, aggregate_attention_stub_falls_back_to_mean)
{
    OcTemporalConfig cfg;
    oc_temporal_config_init(&cfg);
    cfg.agg_type    = OC_TEMPORAL_ATTENTION;
    cfg.feature_dim = 2;
    OcTemporalState st;
    oc_temporal_init(&st, &cfg);

    float features[] = { 2, 4,  6, 8 };
    float out[2];
    cr_assert_eq(oc_temporal_aggregate(&st, features, 2, out), OC_OK, "");
    /* Attention stub falls back to mean: (2+6)/2=4, (4+8)/2=6 */
    cr_assert_float_eq(out[0], 4.0f, 1e-6, "");
    cr_assert_float_eq(out[1], 6.0f, 1e-6, "");
    oc_temporal_free(&st);
}

Test(temporal, aggregate_zero_frames_error)
{
    OcTemporalConfig cfg;
    oc_temporal_config_init(&cfg);
    cfg.feature_dim = 2;
    OcTemporalState st;
    oc_temporal_init(&st, &cfg);
    float out[2];
    cr_assert_eq(oc_temporal_aggregate(&st, NULL, 0, out), OC_ERR_INVALID_ARG, "");
    oc_temporal_free(&st);
}

Test(temporal, aggregate_null_args)
{
    OcTemporalConfig cfg;
    oc_temporal_config_init(&cfg);
    cfg.feature_dim = 2;
    OcTemporalState st;
    oc_temporal_init(&st, &cfg);
    float out[2];
    cr_assert_eq(oc_temporal_aggregate(NULL, NULL, 1, out), OC_ERR_INVALID_ARG, "");
    float features[] = { 1, 2 };
    cr_assert_eq(oc_temporal_aggregate(&st, features, 1, NULL), OC_ERR_INVALID_ARG, "");
    oc_temporal_free(&st);
}
