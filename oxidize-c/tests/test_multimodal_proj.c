/*
 * test_multimodal_proj.c — tests for the multimodal projection layer.
 */
#include "framework.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "oxidize/multimodal_proj.h"

Test(mm_proj, config_default)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cr_assert_eq(cfg.modality, OC_MODALITY_VISION);
    cr_assert_eq(cfg.n_layers, 2);
    cr_assert_eq(cfg.hidden_dim, 0);
    cr_assert_eq(cfg.activation, OC_MM_ACT_GELU);
}

Test(mm_proj, init_valid)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 64;
    cfg.output_dim = 128;
    cfg.hidden_dim = 64;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    cr_assert_not_null(proj);
    cr_assert(proj->initialized);
    cr_assert_eq(proj->config.input_dim, 64);
    cr_assert_eq(proj->config.output_dim, 128);
    cr_assert_eq(proj->n_weights, 2);
    /* layer 0: in=64, out=64 (hidden); layer 1: in=64, out=128 */
    cr_assert_eq(proj->in_dims[0], 64);
    cr_assert_eq(proj->out_dims[0], 64);
    cr_assert_eq(proj->in_dims[1], 64);
    cr_assert_eq(proj->out_dims[1], 128);
    oc_mm_proj_free(proj);
}

Test(mm_proj, init_single_layer)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 32;
    cfg.output_dim = 64;
    cfg.n_layers = 1;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    cr_assert_not_null(proj);
    cr_assert_eq(proj->n_weights, 1);
    cr_assert_eq(proj->in_dims[0], 32);
    cr_assert_eq(proj->out_dims[0], 64);
    oc_mm_proj_free(proj);
}

OC_TEST_NULL_SAFE(mm_proj, init_null,
        cr_assert_null(oc_mm_proj_init(NULL));)

Test(mm_proj, init_bad_dims)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 0;
    cfg.output_dim = 64;
    cr_assert_null(oc_mm_proj_init(&cfg));

    cfg.input_dim = 32;
    cfg.output_dim = 0;
    cr_assert_null(oc_mm_proj_init(&cfg));
}

Test(mm_proj, init_bad_n_layers)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 32;
    cfg.output_dim = 64;
    cfg.n_layers = 0;
    cr_assert_null(oc_mm_proj_init(&cfg));
}

Test(mm_proj, init_hidden_dim_defaults_to_input)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 48;
    cfg.output_dim = 96;
    cfg.hidden_dim = 0; /* should default to input_dim */
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    cr_assert_not_null(proj);
    cr_assert_eq(proj->config.hidden_dim, 48);
    /* layer 0 out = hidden_dim = 48 */
    cr_assert_eq(proj->out_dims[0], 48);
    oc_mm_proj_free(proj);
}

OC_TEST_NULL_SAFE(mm_proj, free_null,
        oc_mm_proj_free(NULL);)

Test(mm_proj, set_layer_weight)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 4;
    cfg.output_dim = 8;
    cfg.n_layers = 1;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    cr_assert_not_null(proj);

    float w[32]; /* 8 * 4 */
    for (int i = 0; i < 32; i++) w[i] = (float)i;
    cr_assert_eq(oc_mm_proj_set_layer_weight(proj, 0, w, 32), OC_OK);
    cr_assert_not_null(proj->weights[0]);
    /* Verify copy. */
    cr_assert_float_eq(proj->weights[0][0], 0.0f, 1e-5f);
    cr_assert_float_eq(proj->weights[0][31], 31.0f, 1e-5f);

    oc_mm_proj_free(proj);
}

Test(mm_proj, set_layer_weight_bad_idx)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 4;
    cfg.output_dim = 8;
    cfg.n_layers = 1;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    float w[1] = {0};
    cr_assert_neq(oc_mm_proj_set_layer_weight(proj, 5, w, 1), OC_OK);
    oc_mm_proj_free(proj);
}

Test(mm_proj, set_layer_weight_bad_count)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 4;
    cfg.output_dim = 8;
    cfg.n_layers = 1;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    float w[1] = {0};
    /* Expected 8*4=32, not 1. */
    cr_assert_neq(oc_mm_proj_set_layer_weight(proj, 0, w, 1), OC_OK);
    oc_mm_proj_free(proj);
}

Test(mm_proj, set_layer_bias)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 4;
    cfg.output_dim = 8;
    cfg.n_layers = 1;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    float b[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    cr_assert_eq(oc_mm_proj_set_layer_bias(proj, 0, b, 8), OC_OK);
    cr_assert_not_null(proj->biases[0]);
    cr_assert_float_eq(proj->biases[0][0], 1.0f, 1e-5f);
    cr_assert_float_eq(proj->biases[0][7], 8.0f, 1e-5f);
    oc_mm_proj_free(proj);
}

Test(mm_proj, set_layer_bias_bad_count)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 4;
    cfg.output_dim = 8;
    cfg.n_layers = 1;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    float b[1] = {0};
    cr_assert_neq(oc_mm_proj_set_layer_bias(proj, 0, b, 1), OC_OK);
    oc_mm_proj_free(proj);
}

Test(mm_proj, load_weights)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 4;
    cfg.output_dim = 8;
    cfg.n_layers = 1;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    /* Buffer: 8*4 weights + 8 biases = 40 floats. */
    float buf[40];
    for (int i = 0; i < 40; i++) buf[i] = (float)i;
    cr_assert_eq(oc_mm_proj_load_weights(proj, buf, 40 * sizeof(float)), OC_OK);
    cr_assert_not_null(proj->weights[0]);
    cr_assert_not_null(proj->biases[0]);
    /* Weight[0][0] = 0, bias[0] = 32 (after 32 weight elements). */
    cr_assert_float_eq(proj->weights[0][0], 0.0f, 1e-5f);
    cr_assert_float_eq(proj->biases[0][0], 32.0f, 1e-5f);
    oc_mm_proj_free(proj);
}

Test(mm_proj, load_weights_too_small)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 4;
    cfg.output_dim = 8;
    cfg.n_layers = 1;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    float buf[10]; /* too small */
    cr_assert_neq(oc_mm_proj_load_weights(proj, buf, 10 * sizeof(float)), OC_OK);
    oc_mm_proj_free(proj);
}

Test(mm_proj, forward_identity)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 4;
    cfg.output_dim = 4;
    cfg.n_layers = 1;
    cfg.activation = OC_MM_ACT_RELU;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    cr_assert_not_null(proj);

    /* Identity weight matrix (4x4), zero bias. */
    float w[16] = {0};
    for (int i = 0; i < 4; i++) w[i * 4 + i] = 1.0f; /* diag */
    float b[4] = {0, 0, 0, 0};
    cr_assert_eq(oc_mm_proj_set_layer_weight(proj, 0, w, 16), OC_OK);
    cr_assert_eq(oc_mm_proj_set_layer_bias(proj, 0, b, 4), OC_OK);

    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float *out = oc_mm_proj_forward(proj, input, 1);
    cr_assert_not_null(out);
    /* Identity: output should equal input. */
    for (int i = 0; i < 4; i++) {
        cr_assert_float_eq(out[i], input[i], 1e-5f, "idx %d", i);
    }
    free(out);
    oc_mm_proj_free(proj);
}

Test(mm_proj, forward_linear_bias)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 4;
    cfg.output_dim = 4;
    cfg.n_layers = 1;
    cfg.activation = OC_MM_ACT_RELU;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    cr_assert_not_null(proj);

    /* Identity weight, bias = [1, 1, 1, 1] */
    float w[16] = {0};
    for (int i = 0; i < 4; i++) w[i * 4 + i] = 1.0f;
    float b[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    oc_mm_proj_set_layer_weight(proj, 0, w, 16);
    oc_mm_proj_set_layer_bias(proj, 0, b, 4);

    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float *out = oc_mm_proj_forward(proj, input, 1);
    cr_assert_not_null(out);
    for (int i = 0; i < 4; i++) {
        cr_assert_float_eq(out[i], input[i] + 1.0f, 1e-5f, "idx %d", i);
    }
    free(out);
    oc_mm_proj_free(proj);
}

Test(mm_proj, forward_multi_token)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 4;
    cfg.output_dim = 4;
    cfg.n_layers = 1;
    cfg.activation = OC_MM_ACT_RELU;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    cr_assert_not_null(proj);

    float w[16] = {0};
    for (int i = 0; i < 4; i++) w[i * 4 + i] = 1.0f;
    float b[4] = {0};
    oc_mm_proj_set_layer_weight(proj, 0, w, 16);
    oc_mm_proj_set_layer_bias(proj, 0, b, 4);

    /* 3 tokens of 4 dims each. */
    float input[12] = {1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12};
    float *out = oc_mm_proj_forward(proj, input, 3);
    cr_assert_not_null(out);
    for (int t = 0; t < 3; t++) {
        for (int i = 0; i < 4; i++) {
            cr_assert_float_eq(out[t * 4 + i], input[t * 4 + i], 1e-5f,
                                "token %d idx %d", t, i);
        }
    }
    free(out);
    oc_mm_proj_free(proj);
}

Test(mm_proj, forward_two_layers)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 4;
    cfg.output_dim = 4;
    cfg.hidden_dim = 4;
    cfg.n_layers = 2;
    cfg.activation = OC_MM_ACT_RELU;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    cr_assert_not_null(proj);

    /* Both layers: identity weight, zero bias. */
    float w[16] = {0};
    for (int i = 0; i < 4; i++) w[i * 4 + i] = 1.0f;
    float b[4] = {0};
    oc_mm_proj_set_layer_weight(proj, 0, w, 16);
    oc_mm_proj_set_layer_bias(proj, 0, b, 4);
    oc_mm_proj_set_layer_weight(proj, 1, w, 16);
    oc_mm_proj_set_layer_bias(proj, 1, b, 4);

    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float *out = oc_mm_proj_forward(proj, input, 1);
    cr_assert_not_null(out);
    /* Identity through both layers (ReLU on positive values is identity). */
    for (int i = 0; i < 4; i++) {
        cr_assert_float_eq(out[i], input[i], 1e-5f, "idx %d", i);
    }
    free(out);
    oc_mm_proj_free(proj);
}

Test(mm_proj, forward_no_weights)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 4;
    cfg.output_dim = 4;
    cfg.n_layers = 1;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    float input[4] = {1, 2, 3, 4};
    /* No weights set -> should return NULL. */
    cr_assert_null(oc_mm_proj_forward(proj, input, 1));
    oc_mm_proj_free(proj);
}

Test(mm_proj, forward_null_args)
{
    cr_assert_null(oc_mm_proj_forward(NULL, (float[4]){0}, 1));
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 4;
    cfg.output_dim = 4;
    cfg.n_layers = 1;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    cr_assert_null(oc_mm_proj_forward(proj, NULL, 1));
    cr_assert_null(oc_mm_proj_forward(proj, (float[4]){0}, 0));
    oc_mm_proj_free(proj);
}

Test(mm_proj, concat_prompt)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 4;
    cfg.output_dim = 4;
    cfg.n_layers = 1;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    cr_assert_not_null(proj);

    float text[8] = {1, 2, 3, 4,  5, 6, 7, 8}; /* 2 tokens */
    float mm[8] = {9, 10, 11, 12,  13, 14, 15, 16}; /* 2 tokens */
    float *out = oc_mm_proj_concat_prompt(proj, text, 2, mm, 2);
    cr_assert_not_null(out);
    /* Should be: [text, mm] = 4 tokens of 4 dims. */
    cr_assert_float_eq(out[0], 1.0f, 1e-5f);
    cr_assert_float_eq(out[3], 4.0f, 1e-5f);
    cr_assert_float_eq(out[4], 5.0f, 1e-5f);
    cr_assert_float_eq(out[8], 9.0f, 1e-5f);
    cr_assert_float_eq(out[15], 16.0f, 1e-5f);
    free(out);
    oc_mm_proj_free(proj);
}

Test(mm_proj, concat_prompt_text_only)
{
    OcMultimodalProjectionConfig cfg = OC_MM_PROJ_CONFIG_DEFAULT;
    cfg.input_dim = 4;
    cfg.output_dim = 4;
    cfg.n_layers = 1;
    OcMultimodalProjection *proj = oc_mm_proj_init(&cfg);
    float text[4] = {1, 2, 3, 4};
    float *out = oc_mm_proj_concat_prompt(proj, text, 1, NULL, 0);
    cr_assert_not_null(out);
    cr_assert_float_eq(out[0], 1.0f, 1e-5f);
    free(out);
    oc_mm_proj_free(proj);
}

OC_TEST_NULL_SAFE(mm_proj, concat_prompt_null_proj,
        cr_assert_null(oc_mm_proj_concat_prompt(NULL, NULL, 0, NULL, 0));)
