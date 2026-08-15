#include <criterion/criterion.h>
#include "oxidize/dspark.h"
#include <math.h>

Test(dspark, config_init)
{
    OcDsparkConfig cfg;
    oc_dspark_config_init(&cfg);
    cr_assert_eq(cfg.block_size, 4);
    cr_assert_float_eq(cfg.p_min, 0.10f, 1e-5f);
}

Test(dspark, pairwise_conf_peaked)
{
    float logits[4] = { 10.0f, 0.0f, -5.0f, -5.0f };
    float c = oc_dspark_pairwise_conf(logits, 4);
    cr_assert(c > 0.99f);
}

Test(dspark, pairwise_conf_tied)
{
    float logits[3] = { 1.0f, 1.0f, 0.0f };
    float c = oc_dspark_pairwise_conf(logits, 3);
    cr_assert_float_eq(c, 0.5f, 0.01f);
}

Test(dspark, advance_null)
{
    cr_assert_neq(oc_dspark_advance(NULL, NULL, NULL, NULL, 0, NULL, NULL), OC_OK);
}
