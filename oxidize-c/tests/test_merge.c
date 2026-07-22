/* test_merge.c — checkpoint merge tests. */
#include <criterion/criterion.h>
#include "oxidize/merge.h"

Test(merge, strategy_name)
{
    cr_assert_str_eq(oc_merge_strategy_name(OC_MERGE_LINEAR), "linear");
    cr_assert_str_eq(oc_merge_strategy_name(OC_MERGE_SLERP), "slerp");
    cr_assert_str_eq(oc_merge_strategy_name(OC_MERGE_TIES), "ties");
    cr_assert_str_eq(oc_merge_strategy_name(OC_MERGE_DARE), "dare");
}

Test(merge, null_config)
{
    cr_assert_neq(oc_merge_models(NULL), OC_OK);
}

Test(merge, linear_too_few_inputs)
{
    OcMergeInput inputs[] = {{"a.gguf", 1.0f}};
    cr_assert_neq(oc_merge_linear(inputs, 1, "out.gguf"), OC_OK);
}

Test(merge, slerp_null_paths)
{
    cr_assert_neq(oc_merge_slerp(NULL, NULL, 0.5f, NULL), OC_OK);
}
