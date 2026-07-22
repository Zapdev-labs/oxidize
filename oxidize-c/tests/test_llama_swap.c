/* test_llama_swap.c — Model swap tests. */
#include <criterion/criterion.h>
#include "oxidize/llama_swap.h"
#include <string.h>

Test(swap, init_free)
{
    OcModelSwap sw;
    cr_assert_eq(oc_model_swap_init(&sw), OC_OK);
    cr_assert_eq(sw.active_idx, -1);
    cr_assert_eq(sw.n_models, 0);
    oc_model_swap_free(&sw);
}

Test(swap, init_null)
{
    cr_assert_neq(oc_model_swap_init(NULL), OC_OK);
}

Test(swap, register)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    cr_assert_eq(oc_model_swap_register(&sw, "/path/model.gguf", "q4km", &idx), OC_OK);
    cr_assert_eq(idx, 0);
    cr_assert_eq(sw.n_models, 1);
    cr_assert_str_eq(sw.models[0].path, "/path/model.gguf");
    cr_assert_str_eq(sw.models[0].name, "q4km");
    oc_model_swap_free(&sw);
}

Test(swap, register_multiple)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx1, idx2;
    oc_model_swap_register(&sw, "/path/m1.gguf", "model1", &idx1);
    oc_model_swap_register(&sw, "/path/m2.gguf", "model2", &idx2);
    cr_assert_eq(idx1, 0);
    cr_assert_eq(idx2, 1);
    cr_assert_eq(sw.n_models, 2);
    oc_model_swap_free(&sw);
}

Test(swap, register_null)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    cr_assert_neq(oc_model_swap_register(NULL, "p", "n", NULL), OC_OK);
    cr_assert_neq(oc_model_swap_register(&sw, NULL, "n", NULL), OC_OK);
    oc_model_swap_free(&sw);
}

Test(swap, load)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    oc_model_swap_register(&sw, "/path/model.gguf", "test", &idx);
    cr_assert_eq(oc_model_swap_load(&sw, idx), OC_OK);
    cr_assert(sw.models[0].loaded);
    cr_assert(sw.total_loaded_bytes > 0);
    oc_model_swap_free(&sw);
}

Test(swap, unload)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    oc_model_swap_register(&sw, "/path/model.gguf", "test", &idx);
    oc_model_swap_load(&sw, idx);
    cr_assert_eq(oc_model_swap_unload(&sw, idx), OC_OK);
    cr_assert(!sw.models[0].loaded);
    oc_model_swap_free(&sw);
}

Test(swap, activate)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    oc_model_swap_register(&sw, "/path/model.gguf", "test", &idx);
    cr_assert_eq(oc_model_swap_activate(&sw, idx), OC_OK);
    cr_assert_eq(sw.active_idx, idx);
    cr_assert(sw.models[idx].loaded);
    oc_model_swap_free(&sw);
}

Test(swap, active)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    cr_assert_eq(oc_model_swap_active(&sw), -1);
    int idx;
    oc_model_swap_register(&sw, "/path/m.gguf", "t", &idx);
    oc_model_swap_activate(&sw, idx);
    cr_assert_eq(oc_model_swap_active(&sw), idx);
    oc_model_swap_free(&sw);
}

Test(swap, info)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    oc_model_swap_register(&sw, "/path/m.gguf", "name", &idx);
    const OcSwapModelEntry *info;
    cr_assert_eq(oc_model_swap_info(&sw, idx, &info), OC_OK);
    cr_assert_str_eq(info->path, "/path/m.gguf");
    cr_assert_str_eq(info->name, "name");
    oc_model_swap_free(&sw);
}

Test(swap, info_null)
{
    cr_assert_neq(oc_model_swap_info(NULL, 0, NULL), OC_OK);
}

Test(swap, info_out_of_range)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    const OcSwapModelEntry *info;
    cr_assert_neq(oc_model_swap_info(&sw, 99, &info), OC_OK);
    oc_model_swap_free(&sw);
}

Test(swap, list)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    oc_model_swap_register(&sw, "/m1.gguf", "a", NULL);
    oc_model_swap_register(&sw, "/m2.gguf", "b", NULL);
    const OcSwapModelEntry *arr;
    size_t n = oc_model_swap_list(&sw, &arr);
    cr_assert_eq(n, 2);
    cr_assert_str_eq(arr[0].name, "a");
    cr_assert_str_eq(arr[1].name, "b");
    oc_model_swap_free(&sw);
}

Test(swap, loaded_bytes)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    oc_model_swap_register(&sw, "/m.gguf", "t", &idx);
    cr_assert_eq(oc_model_swap_loaded_bytes(&sw), 0);
    oc_model_swap_load(&sw, idx);
    cr_assert(oc_model_swap_loaded_bytes(&sw) > 0);
    oc_model_swap_unload(&sw, idx);
    cr_assert_eq(oc_model_swap_loaded_bytes(&sw), 0);
    oc_model_swap_free(&sw);
}

Test(swap, double_load)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    oc_model_swap_register(&sw, "/m.gguf", "t", &idx);
    oc_model_swap_load(&sw, idx);
    cr_assert_eq(oc_model_swap_load(&sw, idx), OC_OK); /* idempotent */
    oc_model_swap_free(&sw);
}

Test(swap, double_unload)
{
    OcModelSwap sw;
    oc_model_swap_init(&sw);
    int idx;
    oc_model_swap_register(&sw, "/m.gguf", "t", &idx);
    oc_model_swap_load(&sw, idx);
    oc_model_swap_unload(&sw, idx);
    cr_assert_eq(oc_model_swap_unload(&sw, idx), OC_OK); /* idempotent */
    oc_model_swap_free(&sw);
}
