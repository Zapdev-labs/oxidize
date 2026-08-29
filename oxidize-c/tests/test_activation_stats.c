/* test_activation_stats.c — activation stats tests. */
#include "framework.h"
#include "oxidize/activation_stats.h"
#include <math.h>

Test(act_stats, init_free)
{
    OcActivationStats stats;
    cr_assert_eq(oc_activation_stats_init(&stats, 5), OC_OK);
    cr_assert_eq(stats.n_layers, 5);
    cr_assert_not_null(stats.layers);
    oc_activation_stats_free(&stats);
    cr_assert_null(stats.layers);
}

Test(act_stats, observe_and_get_norms)
{
    OcActivationStats stats;
    oc_activation_stats_init(&stats, 1);

    /* Observe a batch of 2 samples, 3 features each.
     * Sample 0: [1, 0, 0]
     * Sample 1: [0, 2, 0]
     * sum_sq = [1, 4, 0], n_samples = 2
     * L2 norms = [sqrt(0.5), sqrt(2), 0] = [0.707, 1.414, 0] */
    float activations[] = {1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f};
    cr_assert_eq(oc_activation_stats_observe(&stats, 0, activations, 2, 3), OC_OK);

    float norms[3] = {0};
    cr_assert_eq(oc_activation_stats_get_l2_norms(&stats, 0, norms, 3), OC_OK);
    cr_assert_float_eq(norms[0], sqrt(0.5), 1e-5f);
    cr_assert_float_eq(norms[1], sqrt(2.0), 1e-5f);
    cr_assert_float_eq(norms[2], 0.0f, 1e-6f);

    oc_activation_stats_free(&stats);
}

Test(act_stats, multiple_observations)
{
    OcActivationStats stats;
    oc_activation_stats_init(&stats, 2);

    /* Layer 0: observe twice. */
    float a1[] = {1.0f, 1.0f};
    cr_assert_eq(oc_activation_stats_observe(&stats, 0, a1, 1, 2), OC_OK);
    float a2[] = {1.0f, 1.0f};
    cr_assert_eq(oc_activation_stats_observe(&stats, 0, a2, 1, 2), OC_OK);
    /* sum_sq = [2, 2], n_samples = 2, L2 = [1, 1] */

    /* Layer 1: observe once. */
    float a3[] = {3.0f, 4.0f};
    cr_assert_eq(oc_activation_stats_observe(&stats, 1, a3, 1, 2), OC_OK);
    /* sum_sq = [9, 16], n_samples = 1, L2 = [3, 4] */

    float norms0[2], norms1[2];
    oc_activation_stats_get_l2_norms(&stats, 0, norms0, 2);
    oc_activation_stats_get_l2_norms(&stats, 1, norms1, 2);
    cr_assert_float_eq(norms0[0], 1.0f, 1e-5f);
    cr_assert_float_eq(norms1[0], 3.0f, 1e-5f);
    cr_assert_float_eq(norms1[1], 4.0f, 1e-5f);

    oc_activation_stats_free(&stats);
}

Test(act_stats, summary)
{
    OcActivationStats stats;
    oc_activation_stats_init(&stats, 3);
    float a[] = {1.0f};
    cr_assert_eq(oc_activation_stats_observe(&stats, 1, a, 1, 1), OC_OK);

    char buf[256];
    oc_activation_stats_summary(&stats, buf, sizeof(buf));
    cr_assert(strstr(buf, "1/3 layers") != NULL);
    cr_assert(strstr(buf, "1 total samples") != NULL);

    oc_activation_stats_free(&stats);
}
