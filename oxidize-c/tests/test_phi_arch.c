/* test_phi_arch.c — Phi-2/3 architecture forward pass tests. */
#include "framework.h"
#include "oxidize/phi_arch.h"
#include <stdlib.h>
#include <string.h>

Test(phi, config_init_defaults)
{
    OcPhiConfig cfg;
    cr_assert_eq(oc_phi_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.n_layers, 24);
    cr_assert_eq(cfg.n_heads, 32);
    cr_assert_eq(cfg.head_dim, 80);
    cr_assert_eq(cfg.hidden_dim, 2560);
    cr_assert_eq(cfg.intermediate_dim, 10240);
    cr_assert_eq(cfg.vocab_size, 51200);
    cr_assert_float_eq(cfg.rope_theta, 10000.0f, 0.001f);
}

OC_TEST_REJECTS_NULL(phi, config_init_null, oc_phi_config_init(NULL))

Test(phi, model_init_default)
{
    OcPhiModel model;
    cr_assert_eq(oc_phi_model_init(&model, NULL), OC_OK);
    cr_assert(model.initialized);
    cr_assert_eq(model.config.n_layers, 24);
    cr_assert_eq(model.config.vocab_size, 51200);
    cr_assert_neq(model.layers, NULL);
    cr_assert_neq(model.tok_emb, NULL);
    cr_assert_neq(model.output_norm, NULL);
    cr_assert_neq(model.output, NULL);
    oc_phi_free(&model);
}

Test(phi, model_init_custom_config)
{
    OcPhiConfig cfg;
    oc_phi_config_init(&cfg);
    cfg.n_layers = 2;
    cfg.vocab_size = 100;
    cfg.hidden_dim = 16;
    OcPhiModel model;
    cr_assert_eq(oc_phi_model_init(&model, &cfg), OC_OK);
    cr_assert(model.initialized);
    cr_assert_eq(model.config.n_layers, 2);
    cr_assert_eq(model.config.vocab_size, 100);
    oc_phi_free(&model);
}

OC_TEST_REJECTS_NULL(phi, model_init_null, oc_phi_model_init(NULL, NULL))

Test(phi, model_init_bad_config)
{
    OcPhiConfig cfg;
    oc_phi_config_init(&cfg);
    cfg.n_layers = 0;
    OcPhiModel model;
    cr_assert_eq(oc_phi_model_init(&model, &cfg), OC_ERR_INVALID_ARG);

    oc_phi_config_init(&cfg);
    cfg.vocab_size = 0;
    cr_assert_eq(oc_phi_model_init(&model, &cfg), OC_ERR_INVALID_ARG);

    oc_phi_config_init(&cfg);
    cfg.hidden_dim = 0;
    cr_assert_eq(oc_phi_model_init(&model, &cfg), OC_ERR_INVALID_ARG);
}

Test(phi, forward_zeros_logits)
{
    OcPhiConfig cfg;
    oc_phi_config_init(&cfg);
    cfg.n_layers = 1;
    cfg.vocab_size = 32;
    cfg.hidden_dim = 8;
    OcPhiModel model;
    cr_assert_eq(oc_phi_model_init(&model, &cfg), OC_OK);
    float *logits = malloc(32 * sizeof(float));
    for (uint32_t i = 0; i < 32; i++) logits[i] = 1.0f;
    cr_assert_eq(oc_phi_forward(&model, 7, logits), OC_OK);
    for (uint32_t i = 0; i < 32; i++)
        cr_assert_float_eq(logits[i], 0.0f, 0.0001f);
    free(logits);
    oc_phi_free(&model);
}

Test(phi, forward_null_args)
{
    cr_assert_neq(oc_phi_forward(NULL, 0, NULL), OC_OK);
    OcPhiModel model;
    oc_phi_model_init(&model, NULL);
    cr_assert_neq(oc_phi_forward(&model, 0, NULL), OC_OK);
    oc_phi_free(&model);
}

Test(phi, forward_uninitialized)
{
    OcPhiModel model;
    memset(&model, 0, sizeof(model));
    float logits[8] = {0};
    cr_assert_neq(oc_phi_forward(&model, 0, logits), OC_OK);
}

OC_TEST_NULL_SAFE(phi, free_null,
        oc_phi_free(NULL);)

Test(phi, free_idempotent_safe)
{
    OcPhiModel model;
    oc_phi_model_init(&model, NULL);
    oc_phi_free(&model);
    oc_phi_free(&model);
}

Test(phi, dense_attention_no_gqa)
{
    OcPhiConfig cfg;
    oc_phi_config_init(&cfg);
    /* Phi-2 uses dense multi-head attention. There is no separate
     * n_kv_heads knob in the Phi config (unlike Mistral/Gemma). */
    cr_assert_eq(cfg.n_heads, 32);
    cr_assert(cfg.head_dim * cfg.n_heads == cfg.hidden_dim);
}

Test(phi, forward_custom_vocab)
{
    OcPhiConfig cfg;
    oc_phi_config_init(&cfg);
    cfg.n_layers = 1;
    cfg.vocab_size = 4;
    cfg.hidden_dim = 8;
    OcPhiModel model;
    cr_assert_eq(oc_phi_model_init(&model, &cfg), OC_OK);
    float logits[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
    cr_assert_eq(oc_phi_forward(&model, 42, logits), OC_OK);
    for (uint32_t i = 0; i < 4; i++)
        cr_assert_float_eq(logits[i], 0.0f, 0.0001f);
    oc_phi_free(&model);
}

Test(phi, model_init_zeroed_layers)
{
    OcPhiModel model;
    cr_assert_eq(oc_phi_model_init(&model, NULL), OC_OK);
    /* After init the per-layer pointers should be NULL until weights
     * are attached. */
    for (uint32_t i = 0; i < model.config.n_layers; i++) {
        cr_assert_eq(model.layers[i].input_norm, NULL);
        cr_assert_eq(model.layers[i].post_attn_norm, NULL);
        cr_assert_eq(model.layers[i].wq, NULL);
        cr_assert_eq(model.layers[i].wk, NULL);
        cr_assert_eq(model.layers[i].wv, NULL);
        cr_assert_eq(model.layers[i].wo, NULL);
        cr_assert_eq(model.layers[i].w_gate, NULL);
        cr_assert_eq(model.layers[i].w_up, NULL);
        cr_assert_eq(model.layers[i].w_down, NULL);
    }
    oc_phi_free(&model);
}
