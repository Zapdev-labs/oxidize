/* test_prune.c — pruning tests. */
#include "framework.h"
#include "oxidize/prune.h"

Test(prune, strategy_name)
{
    cr_assert_str_eq(oc_prune_strategy_name(OC_PRUNE_WANDA), "wanda");
    cr_assert_str_eq(oc_prune_strategy_name(OC_PRUNE_MAGNITUDE), "magnitude");
}

OC_TEST_NULL_SAFE(prune, null_config,
        cr_assert_neq(oc_prune_model(NULL), OC_OK);)

Test(prune, invalid_sparsity)
{
    cr_assert_neq(oc_prune_magnitude("in.gguf", "out.gguf", -0.1f), OC_OK);
    cr_assert_neq(oc_prune_magnitude("in.gguf", "out.gguf", 1.0f), OC_OK);
}

OC_TEST_NULL_SAFE(prune, null_paths,
        cr_assert_neq(oc_prune_magnitude(NULL, "out.gguf", 0.5f), OC_OK);
        cr_assert_neq(oc_prune_magnitude("in.gguf", NULL, 0.5f), OC_OK);)

OC_TEST_NULL_SAFE(prune, null_sparsity_check,
        cr_assert_neq(oc_prune_compute_sparsity(NULL, NULL), OC_OK);)
