/* test_mistral_arch.c — Mistral architecture forward pass tests. */
#include "framework.h"
#include "oxidize/mistral_arch.h"
#include <stdlib.h>
#include <string.h>

Test(mistral, config_init_defaults)
{
    OcMistralConfig cfg;
    cr_assert_eq(oc_mistral_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.n_layers, 32);
    cr_assert_eq(cfg.n_heads, 32);
    cr_assert_eq(cfg.n_kv_heads, 8);
    cr_assert_eq(cfg.head_dim, 128);
    cr_assert_eq(cfg.hidden_dim, 4096);
    cr_assert_eq(cfg.intermediate_dim, 14336);
    cr_assert_eq(cfg.vocab_size, 32000);
    cr_assert_eq(cfg.sliding_window, 4096);
    cr_assert_float_eq(cfg.rope_theta, 10000.0f, 0.001f);
    cr_assert_eq(cfg.max_position, 32768);
}

OC_TEST_REJECTS_NULL(mistral, config_init_null, oc_mistral_config_init(NULL))

Test(mistral, model_init_default)
{
    OcMistralModel model;
    cr_assert_eq(oc_mistral_model_init(&model, NULL), OC_OK);
    cr_assert(model.initialized);
    cr_assert_eq(model.config.n_layers, 32);
    cr_assert_eq(model.config.vocab_size, 32000);
    cr_assert_neq(model.layers, NULL);
    cr_assert_neq(model.tok_emb, NULL);
    cr_assert_neq(model.output_norm, NULL);
    cr_assert_neq(model.output, NULL);
    oc_mistral_free(&model);
}

Test(mistral, model_init_custom_config)
{
    OcMistralConfig cfg;
    oc_mistral_config_init(&cfg);
    cfg.n_layers = 2;
    cfg.vocab_size = 100;
    cfg.hidden_dim = 16;
    OcMistralModel model;
    cr_assert_eq(oc_mistral_model_init(&model, &cfg), OC_OK);
    cr_assert(model.initialized);
    cr_assert_eq(model.config.n_layers, 2);
    cr_assert_eq(model.config.vocab_size, 100);
    oc_mistral_free(&model);
}

OC_TEST_REJECTS_NULL(mistral, model_init_null, oc_mistral_model_init(NULL, NULL))

Test(mistral, model_init_bad_config)
{
    OcMistralConfig cfg;
    oc_mistral_config_init(&cfg);
    cfg.n_layers = 0;
    OcMistralModel model;
    cr_assert_eq(oc_mistral_model_init(&model, &cfg), OC_ERR_INVALID_ARG);

    oc_mistral_config_init(&cfg);
    cfg.vocab_size = 0;
    cr_assert_eq(oc_mistral_model_init(&model, &cfg), OC_ERR_INVALID_ARG);

    oc_mistral_config_init(&cfg);
    cfg.hidden_dim = 0;
    cr_assert_eq(oc_mistral_model_init(&model, &cfg), OC_ERR_INVALID_ARG);
}

Test(mistral, forward_zeros_logits)
{
    OcMistralModel model;
    cr_assert_eq(oc_mistral_model_init(&model, NULL), OC_OK);
    float *logits = malloc(model.config.vocab_size * sizeof(float));
    /* Fill with garbage so we know the stub actually clears it. */
    for (uint32_t i = 0; i < model.config.vocab_size; i++) logits[i] = 1.0f;
    cr_assert_eq(oc_mistral_forward(&model, 5, logits), OC_OK);
    for (uint32_t i = 0; i < model.config.vocab_size; i++) {
        cr_assert_float_eq(logits[i], 0.0f, 0.0001f,
            "logits[%u] not zeroed", i);
    }
    free(logits);
    oc_mistral_free(&model);
}

Test(mistral, forward_null_args)
{
    cr_assert_neq(oc_mistral_forward(NULL, 0, NULL), OC_OK);
    OcMistralModel model;
    oc_mistral_model_init(&model, NULL);
    cr_assert_neq(oc_mistral_forward(&model, 0, NULL), OC_OK);
    oc_mistral_free(&model);
}

Test(mistral, forward_uninitialized)
{
    OcMistralModel model;
    memset(&model, 0, sizeof(model));
    float logits[8] = {0};
    cr_assert_neq(oc_mistral_forward(&model, 0, logits), OC_OK);
}

Test(mistral, forward_custom_vocab)
{
    OcMistralConfig cfg;
    oc_mistral_config_init(&cfg);
    cfg.n_layers = 1;
    cfg.vocab_size = 4;
    cfg.hidden_dim = 8;
    OcMistralModel model;
    cr_assert_eq(oc_mistral_model_init(&model, &cfg), OC_OK);
    float logits[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
    cr_assert_eq(oc_mistral_forward(&model, 42, logits), OC_OK);
    for (uint32_t i = 0; i < 4; i++)
        cr_assert_float_eq(logits[i], 0.0f, 0.0001f);
    oc_mistral_free(&model);
}

OC_TEST_NULL_SAFE(mistral, free_null,
        oc_mistral_free(NULL);)

Test(mistral, free_idempotent_safe)
{
    OcMistralModel model;
    oc_mistral_model_init(&model, NULL);
    oc_mistral_free(&model);
    /* Double free must be safe (struct zeroed). */
    oc_mistral_free(&model);
}

Test(mistral, model_init_zeroed_layers)
{
    OcMistralModel model;
    cr_assert_eq(oc_mistral_model_init(&model, NULL), OC_OK);
    /* After init the per-layer pointers should be NULL until weights are
     * attached; the layers array itself is allocated. */
    for (uint32_t i = 0; i < model.config.n_layers; i++) {
        cr_assert_eq(model.layers[i].attention_norm, NULL);
        cr_assert_eq(model.layers[i].wq, NULL);
        cr_assert_eq(model.layers[i].wk, NULL);
        cr_assert_eq(model.layers[i].wv, NULL);
        cr_assert_eq(model.layers[i].wo, NULL);
        cr_assert_eq(model.layers[i].w_gate, NULL);
        cr_assert_eq(model.layers[i].w_up, NULL);
        cr_assert_eq(model.layers[i].w_down, NULL);
    }
    oc_mistral_free(&model);
}

Test(mistral, swa_default_is_4096)
{
    OcMistralConfig cfg;
    oc_mistral_config_init(&cfg);
    /* Sliding-window attention window for Mistral-7B is 4096 tokens. */
    cr_assert_eq(cfg.sliding_window, 4096);
}

Test(mistral, gqa_n_kv_heads_lt_n_heads)
{
    OcMistralConfig cfg;
    oc_mistral_config_init(&cfg);
    /* Mistral-7B uses GQA with 8 KV heads and 32 query heads. */
    cr_assert(cfg.n_kv_heads < cfg.n_heads);
    cr_assert_eq(cfg.n_heads / cfg.n_kv_heads, 4);
}
