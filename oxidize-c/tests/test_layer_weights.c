/* test_layer_weights.c — Layer weights tests. */
#include <criterion/criterion.h>
#include "oxidize/layer_weights.h"
#include <stdlib.h>

Test(lw, init)
{
    OcLayerWeights lw;
    oc_layer_weights_init(&lw);
    cr_assert(!oc_layer_weights_has_attention(&lw));
    cr_assert(!oc_layer_weights_has_dense_ffn(&lw));
    cr_assert(!oc_layer_weights_has_moe(&lw));
    cr_assert(!oc_layer_weights_has_ssm(&lw));
    cr_assert(!oc_layer_weights_has_mla(&lw));
    oc_layer_weights_free(&lw);
}

Test(lw, init_null)
{
    oc_layer_weights_init(NULL);
    oc_layer_weights_free(NULL);
}

Test(lw, has_attention)
{
    OcLayerWeights lw;
    oc_layer_weights_init(&lw);
    float *q = malloc(4 * sizeof(float));
    cr_assert_eq(oc_weight_storage_f32(&lw.attn_q, q, 4), OC_OK);
    cr_assert(oc_layer_weights_has_attention(&lw));
    oc_layer_weights_free(&lw);
}

Test(lw, has_dense_ffn)
{
    OcLayerWeights lw;
    oc_layer_weights_init(&lw);
    float *g = malloc(4 * sizeof(float));
    oc_weight_storage_f32(&lw.ffn_gate, g, 4);
    cr_assert(oc_layer_weights_has_dense_ffn(&lw));
    cr_assert(!oc_layer_weights_has_moe(&lw));
    oc_layer_weights_free(&lw);
}

Test(lw, has_moe)
{
    OcLayerWeights lw;
    oc_layer_weights_init(&lw);
    uint8_t *data = malloc(144);
    oc_weight_storage_quantized(&lw.ffn_gate_exps, OC_QUANT_Q4_K_M, data, 144);
    cr_assert(oc_layer_weights_has_moe(&lw));
    cr_assert(!oc_layer_weights_has_dense_ffn(&lw));
    oc_layer_weights_free(&lw);
}

Test(lw, has_ssm)
{
    OcLayerWeights lw;
    oc_layer_weights_init(&lw);
    lw.ssm_a = malloc(8 * sizeof(float));
    cr_assert(oc_layer_weights_has_ssm(&lw));
    oc_layer_weights_free(&lw);
}

Test(lw, has_mla)
{
    OcLayerWeights lw;
    oc_layer_weights_init(&lw);
    float *qa = malloc(4 * sizeof(float));
    oc_weight_storage_f32(&lw.mla_q_a, qa, 4);
    cr_assert(oc_layer_weights_has_mla(&lw));
    oc_layer_weights_free(&lw);
}

Test(lw, free_releases_weights)
{
    OcLayerWeights lw;
    oc_layer_weights_init(&lw);
    float *norm = malloc(32 * sizeof(float));
    lw.attn_norm = norm;
    lw.n_attn_norm = 32;
    float *q = malloc(128 * sizeof(float));
    oc_weight_storage_f32(&lw.attn_q, q, 128);
    oc_layer_weights_free(&lw);
    /* After free, all should be zeroed. */
    cr_assert_null(lw.attn_norm);
    cr_assert(oc_weight_storage_is_empty(&lw.attn_q));
}

Test(lw, multiple_layers)
{
    OcLayerWeights layers[4];
    for (int i = 0; i < 4; i++) {
        oc_layer_weights_init(&layers[i]);
        float *norm = malloc(64 * sizeof(float));
        memset(norm, i, 64 * sizeof(float));
        layers[i].attn_norm = norm;
        layers[i].n_attn_norm = 64;
    }
    cr_assert_not_null(layers[0].attn_norm);
    cr_assert_not_null(layers[3].attn_norm);
    for (int i = 0; i < 4; i++)
        oc_layer_weights_free(&layers[i]);
}

Test(lw, has_attention_via_qkv)
{
    OcLayerWeights lw;
    oc_layer_weights_init(&lw);
    float *qkv = malloc(12 * sizeof(float));
    oc_weight_storage_f32(&lw.attn_qkv, qkv, 12);
    cr_assert(oc_layer_weights_has_attention(&lw));
    oc_layer_weights_free(&lw);
}

Test(lw, ssm_via_out)
{
    OcLayerWeights lw;
    oc_layer_weights_init(&lw);
    float *out = malloc(8 * sizeof(float));
    oc_weight_storage_f32(&lw.ssm_out, out, 8);
    cr_assert(oc_layer_weights_has_ssm(&lw));
    oc_layer_weights_free(&lw);
}
