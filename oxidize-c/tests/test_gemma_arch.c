/* test_gemma_arch.c — Gemma architecture forward pass tests. */
#include "framework.h"
#include "oxidize/gemma_arch.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

Test(gemma, config_init_defaults)
{
    OcGemmaConfig cfg;
    cr_assert_eq(oc_gemma_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.n_layers, 18);
    cr_assert_eq(cfg.n_heads, 8);
    cr_assert_eq(cfg.n_kv_heads, 1);
    cr_assert_eq(cfg.head_dim, 256);
    cr_assert_eq(cfg.hidden_dim, 2048);
    cr_assert_eq(cfg.intermediate_dim, 16384);
    cr_assert_eq(cfg.vocab_size, 256000);
    /* embedding_scale must default to sqrt(hidden_dim). */
    cr_assert_float_eq(cfg.embedding_scale, sqrtf((float)cfg.hidden_dim),
                       0.001f);
    cr_assert_float_eq(cfg.rope_theta, 10000.0f, 0.001f);
}

Test(gemma, config_init_null)
{
    cr_assert_eq(oc_gemma_config_init(NULL), OC_ERR_INVALID_ARG);
}

Test(gemma, embedding_scale_is_sqrt_hidden)
{
    OcGemmaConfig cfg;
    oc_gemma_config_init(&cfg);
    /* Gemma's signature quirk: embeddings are scaled by sqrt(hidden_dim). */
    cr_assert_float_eq(cfg.embedding_scale, sqrtf(2048.0f), 0.001f);
    cr_assert(cfg.embedding_scale > 1.0f);
}

Test(gemma, model_init_default)
{
    OcGemmaModel model;
    cr_assert_eq(oc_gemma_model_init(&model, NULL), OC_OK);
    cr_assert(model.initialized);
    cr_assert_eq(model.config.n_layers, 18);
    cr_assert_eq(model.config.vocab_size, 256000);
    cr_assert_neq(model.layers, NULL);
    cr_assert_neq(model.tok_emb, NULL);
    cr_assert_neq(model.output_norm, NULL);
    cr_assert_neq(model.output, NULL);
    oc_gemma_free(&model);
}

Test(gemma, model_init_custom_config)
{
    OcGemmaConfig cfg;
    oc_gemma_config_init(&cfg);
    cfg.n_layers = 2;
    cfg.vocab_size = 100;
    cfg.hidden_dim = 16;
    cfg.embedding_scale = sqrtf(16.0f);
    OcGemmaModel model;
    cr_assert_eq(oc_gemma_model_init(&model, &cfg), OC_OK);
    cr_assert(model.initialized);
    cr_assert_eq(model.config.n_layers, 2);
    cr_assert_eq(model.config.vocab_size, 100);
    oc_gemma_free(&model);
}

Test(gemma, model_init_null)
{
    cr_assert_eq(oc_gemma_model_init(NULL, NULL), OC_ERR_INVALID_ARG);
}

Test(gemma, model_init_bad_config)
{
    OcGemmaConfig cfg;
    oc_gemma_config_init(&cfg);
    cfg.n_layers = 0;
    OcGemmaModel model;
    cr_assert_eq(oc_gemma_model_init(&model, &cfg), OC_ERR_INVALID_ARG);

    oc_gemma_config_init(&cfg);
    cfg.vocab_size = 0;
    cr_assert_eq(oc_gemma_model_init(&model, &cfg), OC_ERR_INVALID_ARG);

    oc_gemma_config_init(&cfg);
    cfg.hidden_dim = 0;
    cr_assert_eq(oc_gemma_model_init(&model, &cfg), OC_ERR_INVALID_ARG);
}

Test(gemma, model_init_defaults_embedding_scale)
{
    /* Caller passes a config with embedding_scale = 0; the model should
     * re-derive it as sqrt(hidden_dim). */
    OcGemmaConfig cfg;
    oc_gemma_config_init(&cfg);
    cfg.n_layers = 1;
    cfg.vocab_size = 4;
    cfg.hidden_dim = 25;
    cfg.embedding_scale = 0.0f;
    OcGemmaModel model;
    cr_assert_eq(oc_gemma_model_init(&model, &cfg), OC_OK);
    cr_assert_float_eq(model.config.embedding_scale, sqrtf(25.0f), 0.001f);
    oc_gemma_free(&model);
}

Test(gemma, forward_zeros_logits)
{
    OcGemmaConfig cfg;
    oc_gemma_config_init(&cfg);
    cfg.n_layers = 1;
    cfg.vocab_size = 32;
    cfg.hidden_dim = 8;
    OcGemmaModel model;
    cr_assert_eq(oc_gemma_model_init(&model, &cfg), OC_OK);
    float *logits = malloc(32 * sizeof(float));
    for (uint32_t i = 0; i < 32; i++) logits[i] = 1.0f;
    cr_assert_eq(oc_gemma_forward(&model, 7, logits), OC_OK);
    for (uint32_t i = 0; i < 32; i++)
        cr_assert_float_eq(logits[i], 0.0f, 0.0001f);
    free(logits);
    oc_gemma_free(&model);
}

Test(gemma, forward_null_args)
{
    cr_assert_neq(oc_gemma_forward(NULL, 0, NULL), OC_OK);
    OcGemmaModel model;
    oc_gemma_model_init(&model, NULL);
    cr_assert_neq(oc_gemma_forward(&model, 0, NULL), OC_OK);
    oc_gemma_free(&model);
}

Test(gemma, forward_uninitialized)
{
    OcGemmaModel model;
    memset(&model, 0, sizeof(model));
    float logits[8] = {0};
    cr_assert_neq(oc_gemma_forward(&model, 0, logits), OC_OK);
}

Test(gemma, free_null)
{
    oc_gemma_free(NULL);
}

Test(gemma, free_idempotent_safe)
{
    OcGemmaModel model;
    oc_gemma_model_init(&model, NULL);
    oc_gemma_free(&model);
    oc_gemma_free(&model);
}

Test(gemma, mqa_single_kv_head)
{
    OcGemmaConfig cfg;
    oc_gemma_config_init(&cfg);
    /* Gemma-2B uses multi-query attention: a single KV head. */
    cr_assert_eq(cfg.n_kv_heads, 1);
    cr_assert(cfg.n_kv_heads < cfg.n_heads);
}
