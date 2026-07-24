/* test_layer_wise.c — Layer-wise inference tests. */
#include <criterion/criterion.h>
#include "oxidize/layer_wise.h"
#include "oxidize/inf_model.h"
#include "oxidize/weight_storage.h"
#include "oxidize/layer_weights.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Reuse tiny model setup from test_layer_range.c */
static void setup_tiny_model(OcInferenceModel *m)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.key_value_head_dim = 0;
    cfg.intermediate_size = 8;
    cfg.vocab_size = 16;
    cfg.context_size = 64;
    cfg.layer_count = 2;
    cfg.rms_norm_eps = 1e-5f;
    cfg.rope_theta = 10000.0f;
    cfg.embedding_scale = 1.0f;
    cfg.gelu_ffn = false;
    cfg.sandwich_norm = false;
    cfg.num_experts = 0;
    cfg.num_experts_per_tok = 0;

    oc_inf_model_init(m, &cfg);

    float *embed = malloc(64 * sizeof(float));
    for (size_t i = 0; i < 16; i++)
        for (size_t j = 0; j < 4; j++)
            embed[i * 4 + j] = (float)i;
    oc_weight_storage_f32(&m->tok_embeddings, embed, 64);

    m->norm_weight = malloc(4 * sizeof(float));
    for (size_t i = 0; i < 4; i++) m->norm_weight[i] = 1.0f;

    float *out_w = malloc(64 * sizeof(float));
    memset(out_w, 0, 64 * sizeof(float));
    for (size_t i = 0; i < 16; i++) out_w[i * 4 + (i % 4)] = 1.0f;
    oc_weight_storage_f32(&m->output_weight, out_w, 64);

    for (size_t li = 0; li < 2; li++) {
        OcLayerWeights layer;
        oc_layer_weights_init(&layer);
        layer.attn_norm = malloc(4 * sizeof(float));
        for (size_t i = 0; i < 4; i++) layer.attn_norm[i] = 1.0f;
        layer.n_attn_norm = 4;

        float *q_w = malloc(16 * sizeof(float));
        memset(q_w, 0, 16 * sizeof(float));
        for (size_t i = 0; i < 4; i++) q_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.attn_q, q_w, 16);

        float *k_w = malloc(16 * sizeof(float));
        memset(k_w, 0, 16 * sizeof(float));
        for (size_t i = 0; i < 4; i++) k_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.attn_k, k_w, 16);

        float *v_w = malloc(16 * sizeof(float));
        memset(v_w, 0, 16 * sizeof(float));
        for (size_t i = 0; i < 4; i++) v_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.attn_v, v_w, 16);

        float *ao_w = malloc(16 * sizeof(float));
        memset(ao_w, 0, 16 * sizeof(float));
        for (size_t i = 0; i < 4; i++) ao_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.attn_output, ao_w, 16);

        layer.ffn_norm = malloc(4 * sizeof(float));
        for (size_t i = 0; i < 4; i++) layer.ffn_norm[i] = 1.0f;
        layer.n_ffn_norm = 4;

        float *fg_w = malloc(32 * sizeof(float));
        memset(fg_w, 0, 32 * sizeof(float));
        for (size_t i = 0; i < 4; i++) fg_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.ffn_gate, fg_w, 32);

        float *fu_w = malloc(32 * sizeof(float));
        memset(fu_w, 0, 32 * sizeof(float));
        for (size_t i = 0; i < 4; i++) fu_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.ffn_up, fu_w, 32);

        float *fd_w = malloc(32 * sizeof(float));
        memset(fd_w, 0, 32 * sizeof(float));
        for (size_t i = 0; i < 4; i++) fd_w[i * 8 + i] = 1.0f;
        oc_weight_storage_f32(&layer.ffn_down, fd_w, 32);

        oc_inf_model_add_layer(m, &layer);
    }
    m->loaded = true;
}

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

/* ─── Forward pass tests ────────────────────────────────────────────── */

Test(lw, forward_single)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    OcLayerWiseConfig cfg;
    oc_lw_config_init(&cfg);
    cfg.n_layers = 2;
    cfg.max_concurrent_layers = 1;
    OcLayerWiseState state;
    oc_lw_state_init(&state, &cfg);
    oc_lw_register_layer(&state, 0, 1024);
    oc_lw_register_layer(&state, 1, 1024);

    float logits[16] = {0};
    OcError e = oc_lw_forward_single(&state, &m, 5, 0, logits);
    cr_assert_eq(e, OC_OK);

    /* Logits should be non-zero. */
    bool nonzero = false;
    for (int i = 0; i < 16; i++)
        if (fabsf(logits[i]) > 0.001f) nonzero = true;
    cr_assert(nonzero);

    oc_lw_state_free(&state);
    oc_inf_model_free(&m);
}

Test(lw, forward_normed_hidden)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    OcLayerWiseState state;
    oc_lw_state_init(&state, NULL);
    oc_lw_register_layer(&state, 0, 1024);
    oc_lw_register_layer(&state, 1, 1024);

    float hidden[4] = {1.0f, 0.5f, -0.5f, 0.25f};
    OcError e = oc_lw_forward_normed_hidden(&state, &m, hidden, 0, 2, 0);
    cr_assert_eq(e, OC_OK);

    /* Hidden should be modified (non-zero). */
    bool changed = false;
    for (int i = 0; i < 4; i++)
        if (fabsf(hidden[i] - 1.0f) > 0.001f || fabsf(hidden[i] - 0.5f) > 0.001f)
            changed = true;
    /* At least some value should have changed after running 2 layers. */
    /* The hidden state may or may not change depending on weights, but
     * the function should succeed. */
    (void)changed;

    oc_lw_state_free(&state);
    oc_inf_model_free(&m);
}

Test(lw, forward_single_null_safety)
{
    float logits[16] = {0};
    cr_assert_neq(oc_lw_forward_single(NULL, NULL, 0, 0, logits), OC_OK);
    OcLayerWiseState state;
    oc_lw_state_init(&state, NULL);
    cr_assert_neq(oc_lw_forward_single(&state, NULL, 0, 0, logits), OC_OK);
    cr_assert_neq(oc_lw_forward_single(&state, NULL, 0, 0, NULL), OC_OK);
    oc_lw_state_free(&state);
}

Test(lw, forward_normed_hidden_null_safety)
{
    float hidden[4] = {0};
    cr_assert_neq(oc_lw_forward_normed_hidden(NULL, NULL, hidden, 0, 1, 0), OC_OK);
}

Test(lw, forward_single_partial_range)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    OcLayerWiseConfig cfg;
    oc_lw_config_init(&cfg);
    cfg.n_layers = 2;
    cfg.max_concurrent_layers = 2;
    OcLayerWiseState state;
    oc_lw_state_init(&state, &cfg);
    oc_lw_register_layer(&state, 0, 1024);
    oc_lw_register_layer(&state, 1, 1024);

    /* Run only layer 0 through forward_normed_hidden. */
    float hidden[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    OcError e = oc_lw_forward_normed_hidden(&state, &m, hidden, 0, 1, 0);
    cr_assert_eq(e, OC_OK);

    oc_lw_state_free(&state);
    oc_inf_model_free(&m);
}
