/* test_layer_range.c — run_layer_range + speculative stats tests. */
#include "framework.h"
#include "tiny_model.h"
#include "oxidize/inf_model.h"
#include "oxidize/weight_storage.h"
#include "oxidize/layer_weights.h"
#include "oxidize/inference.h"
#include "oxidize/speculative.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

Test(layer_range, run_single_layer)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 64);

    /* Embed token 5. */
    oc_inf_model_embed_token(&m, 5);

    /* Run only layer 0. */
    OcError e = oc_inf_model_run_layer_range(&m, 0, 1, 0);
    cr_assert_eq(e, OC_OK);

    /* Hidden state should be non-zero. */
    const float *x = oc_inf_model_hidden_state(&m);
    bool nonzero = false;
    for (int i = 0; i < 4; i++)
        if (fabsf(x[i]) > 0.001f) nonzero = true;
    cr_assert(nonzero);

    /* KV cache should have 1 token. */
    cr_assert_eq(oc_kv_cache_n_tokens(&m.kv_cache), 1);

    oc_inf_model_free(&m);
}

Test(layer_range, run_all_layers_split)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 64);

    oc_inf_model_embed_token(&m, 3);

    /* Run layer 0, then layer 1 separately. */
    OcError e = oc_inf_model_run_layer_range(&m, 0, 1, 0);
    cr_assert_eq(e, OC_OK);
    e = oc_inf_model_run_layer_range(&m, 1, 2, 0);
    cr_assert_eq(e, OC_OK);

    /* KV cache should have entries from both layers. */
    cr_assert_geq(oc_kv_cache_n_tokens(&m.kv_cache), 1);

    oc_inf_model_free(&m);
}

Test(layer_range, empty_range)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 64);
    /* start >= end -> no-op. */
    OcError e = oc_inf_model_run_layer_range(&m, 1, 1, 0);
    cr_assert_eq(e, OC_OK);
    oc_inf_model_free(&m);
}

Test(layer_range, out_of_bounds)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 64);
    cr_assert_neq(oc_inf_model_run_layer_range(&m, 0, 99, 0), OC_OK);
    cr_assert_neq(oc_inf_model_run_layer_range(NULL, 0, 1, 0), OC_OK);
    oc_inf_model_free(&m);
}

/* ─── Speculative stats accessors ──────────────────────────────────────── */

Test(spec_stats, acceptance_rate)
{
    OcSpeculativeStats stats = {
        .total_draft_tokens = 100,
        .accepted_draft_tokens = 75,
        .target_forward_passes = 25,
        .draft_forward_passes = 25,
        .emitted_tokens = 100,
    };
    cr_assert_float_eq(oc_speculative_acceptance_rate(&stats), 0.75, 0.001);
}

Test(spec_stats, acceptance_rate_zero)
{
    OcSpeculativeStats stats = {0, 0, 0, 0, 0};
    cr_assert_float_eq(oc_speculative_acceptance_rate(&stats), 0.0, 0.001);
}

Test(spec_stats, tokens_per_forward)
{
    OcSpeculativeStats stats = {
        .total_draft_tokens = 100,
        .accepted_draft_tokens = 75,
        .target_forward_passes = 25,
        .draft_forward_passes = 25,
        .emitted_tokens = 100,
    };
    /* 100 / 25 = 4.0 tokens per target forward. */
    cr_assert_float_eq(oc_speculative_tokens_per_target_forward(&stats), 4.0, 0.001);
}

Test(spec_stats, estimated_speedup)
{
    OcSpeculativeStats stats = {
        .total_draft_tokens = 80,
        .accepted_draft_tokens = 60,
        .target_forward_passes = 20,
        .draft_forward_passes = 20,
        .emitted_tokens = 80,
    };
    /* 80 / 20 = 4.0x speedup. */
    cr_assert_float_eq(oc_speculative_estimated_speedup(&stats), 4.0, 0.001);
}

OC_TEST_NULL_SAFE(spec_stats, null_safety,
        cr_assert_float_eq(oc_speculative_acceptance_rate(NULL), 0.0, 0.001);
        cr_assert_float_eq(oc_speculative_tokens_per_target_forward(NULL), 0.0, 0.001);
        cr_assert_float_eq(oc_speculative_estimated_speedup(NULL), 0.0, 0.001);)

OC_TEST_NULL_SAFE(spec_stats, load_draft_null,
        cr_assert_neq(oc_speculative_load_draft(NULL, NULL, NULL), OC_OK);)
