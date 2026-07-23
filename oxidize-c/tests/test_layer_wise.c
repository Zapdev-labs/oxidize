/* test_layer_wise.c — Layer-wise inference tests. */
#include <criterion/criterion.h>
#include "oxidize/layer_wise.h"
#include <string.h>

Test(lw, config_init)
{
    OcLayerWiseConfig cfg;
    cr_assert_eq(oc_lw_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.n_layers, 32);
    cr_assert_eq(cfg.max_concurrent_layers, 4);
    cr_assert(cfg.available_memory > 0);
}

Test(lw, config_init_null)
{
    cr_assert_neq(oc_lw_config_init(NULL), OC_OK);
}

Test(lw, state_init)
{
    OcLayerWiseState state;
    cr_assert_eq(oc_lw_state_init(&state, NULL), OC_OK);
    cr_assert(state.initialized);
    cr_assert(state.n_layers > 0);
    oc_lw_state_free(&state);
}

Test(lw, state_init_null)
{
    cr_assert_neq(oc_lw_state_init(NULL, NULL), OC_OK);
}

Test(lw, state_init_custom)
{
    OcLayerWiseConfig cfg;
    oc_lw_config_init(&cfg);
    cfg.n_layers = 10;
    cfg.max_concurrent_layers = 2;
    OcLayerWiseState state;
    cr_assert_eq(oc_lw_state_init(&state, &cfg), OC_OK);
    cr_assert_eq(state.n_layers, 10);
    cr_assert_eq(state.max_concurrent_layers, 2);
    oc_lw_state_free(&state);
}

Test(lw, register_layer)
{
    OcLayerWiseState state;
    oc_lw_state_init(&state, NULL);
    cr_assert_eq(oc_lw_register_layer(&state, 0, 1024), OC_OK);
    cr_assert_eq(state.layers[0].weight_size, 1024);
    cr_assert(state.total_weight_size > 0);
    oc_lw_state_free(&state);
}

Test(lw, register_tensor)
{
    OcLayerWiseState state;
    oc_lw_state_init(&state, NULL);
    oc_lw_register_layer(&state, 0, 1024);
    cr_assert_eq(oc_lw_register_tensor(&state, 0, "weight"), OC_OK);
    cr_assert_eq(oc_lw_register_tensor(&state, 0, "bias"), OC_OK);
    cr_assert_eq(state.layers[0].n_tensors, 2);
    cr_assert_str_eq(state.layers[0].tensor_names[0], "weight");
    oc_lw_state_free(&state);
}

Test(lw, load_layer)
{
    OcLayerWiseState state;
    oc_lw_state_init(&state, NULL);
    oc_lw_register_layer(&state, 0, 1024);
    cr_assert_eq(oc_lw_load_layer(&state, 0), OC_OK);
    cr_assert(state.layers[0].loaded);
    oc_lw_state_free(&state);
}

Test(lw, unload_layer)
{
    OcLayerWiseState state;
    oc_lw_state_init(&state, NULL);
    oc_lw_register_layer(&state, 0, 1024);
    oc_lw_load_layer(&state, 0);
    cr_assert_eq(oc_lw_unload_layer(&state, 0), OC_OK);
    cr_assert(!state.layers[0].loaded);
    oc_lw_state_free(&state);
}

Test(lw, n_loaded)
{
    OcLayerWiseState state;
    oc_lw_state_init(&state, NULL);
    oc_lw_register_layer(&state, 0, 1024);
    oc_lw_register_layer(&state, 1, 1024);
    cr_assert_eq(oc_lw_n_loaded(&state), 0);
    oc_lw_load_layer(&state, 0);
    cr_assert_eq(oc_lw_n_loaded(&state), 1);
    oc_lw_load_layer(&state, 1);
    cr_assert_eq(oc_lw_n_loaded(&state), 2);
    oc_lw_state_free(&state);
}

Test(lw, loaded_bytes)
{
    OcLayerWiseState state;
    oc_lw_state_init(&state, NULL);
    oc_lw_register_layer(&state, 0, 2048);
    oc_lw_load_layer(&state, 0);
    cr_assert_eq(oc_lw_loaded_bytes(&state), 2048);
    oc_lw_state_free(&state);
}

Test(lw, n_layers)
{
    OcLayerWiseState state;
    oc_lw_state_init(&state, NULL);
    cr_assert_eq(oc_lw_n_layers(&state), state.n_layers);
    oc_lw_state_free(&state);
}

Test(lw, advance)
{
    OcLayerWiseState state;
    oc_lw_state_init(&state, NULL);
    for (uint32_t i = 0; i < state.n_layers; i++)
        oc_lw_register_layer(&state, i, 1024);
    oc_lw_load_layer(&state, 0);
    cr_assert_eq(state.current_layer, 0);
    cr_assert_eq(oc_lw_advance(&state), OC_OK);
    cr_assert_eq(state.current_layer, 1);
    cr_assert(state.layers[1].loaded);
    oc_lw_state_free(&state);
}

Test(lw, current_layer)
{
    OcLayerWiseState state;
    oc_lw_state_init(&state, NULL);
    cr_assert_eq(oc_lw_current_layer(&state), 0);
    oc_lw_state_free(&state);
}

Test(lw, get_layer)
{
    OcLayerWiseState state;
    oc_lw_state_init(&state, NULL);
    oc_lw_register_layer(&state, 0, 512);
    const OcLwLayerState *l;
    cr_assert_eq(oc_lw_get_layer(&state, 0, &l), OC_OK);
    cr_assert_eq(l->weight_size, 512);
    oc_lw_state_free(&state);
}

Test(lw, get_layer_oob)
{
    OcLayerWiseState state;
    oc_lw_state_init(&state, NULL);
    const OcLwLayerState *l;
    cr_assert_neq(oc_lw_get_layer(&state, 999, &l), OC_OK);
    oc_lw_state_free(&state);
}

Test(lw, eviction)
{
    OcLayerWiseConfig cfg;
    oc_lw_config_init(&cfg);
    cfg.n_layers = 10;
    cfg.max_concurrent_layers = 2;
    OcLayerWiseState state;
    oc_lw_state_init(&state, &cfg);
    for (uint32_t i = 0; i < 10; i++)
        oc_lw_register_layer(&state, i, 1024);
    oc_lw_load_layer(&state, 0);
    oc_lw_load_layer(&state, 1);
    cr_assert_eq(oc_lw_n_loaded(&state), 2);
    /* Loading layer 3 should evict one of the existing layers. */
    oc_lw_load_layer(&state, 3);
    cr_assert_eq(oc_lw_n_loaded(&state), 2);
    oc_lw_state_free(&state);
}

Test(lw, double_load)
{
    OcLayerWiseState state;
    oc_lw_state_init(&state, NULL);
    oc_lw_register_layer(&state, 0, 1024);
    oc_lw_load_layer(&state, 0);
    cr_assert_eq(oc_lw_load_layer(&state, 0), OC_OK);
    oc_lw_state_free(&state);
}

Test(lw, unload_not_loaded)
{
    OcLayerWiseState state;
    oc_lw_state_init(&state, NULL);
    oc_lw_register_layer(&state, 0, 1024);
    cr_assert_eq(oc_lw_unload_layer(&state, 0), OC_OK);
    oc_lw_state_free(&state);
}

Test(lw, free_null)
{
    oc_lw_state_free(NULL);
}
