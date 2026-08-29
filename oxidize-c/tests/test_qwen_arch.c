/* test_qwen_arch.c — Qwen architecture tests. */
#include "framework.h"
#include "oxidize/qwen_arch.h"
#include <string.h>

Test(qwen, config_init)
{
    OcQwenConfig cfg;
    cr_assert_eq(oc_qwen_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.n_layers, 28);
    cr_assert_eq(cfg.n_heads, 28);
    cr_assert_eq(cfg.n_kv_heads, 4);
    cr_assert_eq(cfg.head_dim, 128);
    cr_assert_eq(cfg.hidden_dim, 3584);
    cr_assert_eq(cfg.intermediate_dim, 18944);
    cr_assert_eq(cfg.vocab_size, 152064);
    cr_assert_float_eq(cfg.rope_theta, 1000000.0f, 0.1f);
    cr_assert(!cfg.use_qk_norm);
    cr_assert(!cfg.tie_word_embeddings);
}

OC_TEST_NULL_SAFE(qwen, config_init_null,
        cr_assert_neq(oc_qwen_config_init(NULL), OC_OK);)

Test(qwen, config_qwen3_06b)
{
    OcQwenConfig cfg;
    cr_assert_eq(oc_qwen_config_qwen3_06b(&cfg), OC_OK);
    cr_assert_eq(cfg.n_layers, 28);
    cr_assert_eq(cfg.n_heads, 16);
    cr_assert_eq(cfg.n_kv_heads, 8);
    cr_assert_eq(cfg.hidden_dim, 1024);
    cr_assert_eq(cfg.intermediate_dim, 3072);
    cr_assert_eq(cfg.vocab_size, 151936);
    cr_assert(cfg.use_qk_norm);
    cr_assert(cfg.tie_word_embeddings);
}

Test(qwen, config_qwen25_7b)
{
    OcQwenConfig cfg;
    cr_assert_eq(oc_qwen_config_qwen25_7b(&cfg), OC_OK);
    cr_assert_eq(cfg.n_layers, 28);
    cr_assert_eq(cfg.hidden_dim, 3584);
    cr_assert_eq(cfg.vocab_size, 152064);
}

Test(qwen, model_init)
{
    OcQwenModel model;
    OcQwenConfig cfg;
    oc_qwen_config_qwen3_06b(&cfg);
    cr_assert_eq(oc_qwen_model_init(&model, &cfg), OC_OK);
    cr_assert(model.initialized);
    cr_assert_not_null(model.layers);
    cr_assert_not_null(model.tok_emb);
    cr_assert_not_null(model.output_norm);
    cr_assert_null(model.output); /* tied embeddings */
    oc_qwen_free(&model);
}

Test(qwen, model_init_default)
{
    OcQwenModel model;
    cr_assert_eq(oc_qwen_model_init(&model, NULL), OC_OK);
    cr_assert(model.initialized);
    cr_assert_not_null(model.output); /* not tied */
    oc_qwen_free(&model);
}

OC_TEST_NULL_SAFE(qwen, model_init_null,
        cr_assert_neq(oc_qwen_model_init(NULL, NULL), OC_OK);)

Test(qwen, model_init_bad_config)
{
    OcQwenModel model;
    OcQwenConfig cfg = {0};
    cr_assert_neq(oc_qwen_model_init(&model, &cfg), OC_OK);
}

Test(qwen, model_init_bad_heads)
{
    OcQwenModel model;
    OcQwenConfig cfg;
    oc_qwen_config_init(&cfg);
    cfg.n_heads = 7;
    cfg.hidden_dim = 3583; /* not divisible by 7 */
    cr_assert_neq(oc_qwen_model_init(&model, &cfg), OC_OK);
}

Test(qwen, forward_zeros)
{
    OcQwenModel model;
    cr_assert_eq(oc_qwen_model_init(&model, NULL), OC_OK);
    float *logits = calloc(model.config.vocab_size, sizeof(float));
    /* Fill with garbage. */
    memset(logits, 0x42, model.config.vocab_size * sizeof(float));
    cr_assert_eq(oc_qwen_forward(&model, 0, logits), OC_OK);
    /* Check first few are zero. */
    for (int i = 0; i < 10; i++)
        cr_assert_float_eq(logits[i], 0.0f, 0.001f);
    free(logits);
    oc_qwen_free(&model);
}

Test(qwen, forward_uninit)
{
    OcQwenModel model = {0};
    float logits[10];
    cr_assert_neq(oc_qwen_forward(&model, 0, logits), OC_OK);
}

OC_TEST_NULL_SAFE(qwen, forward_null,
        cr_assert_neq(oc_qwen_forward(NULL, 0, NULL), OC_OK);)

OC_TEST_NULL_SAFE(qwen, free_null,
        oc_qwen_free(NULL);)

Test(qwen, free_reuse)
{
    OcQwenModel model;
    oc_qwen_model_init(&model, NULL);
    oc_qwen_free(&model);
    oc_qwen_free(&model); /* should not crash */
}

Test(qwen, gqa_ratio)
{
    OcQwenConfig cfg;
    oc_qwen_config_init(&cfg);
    /* Qwen2.5-7B: 28 heads, 4 KV heads = 7:1 GQA ratio. */
    cr_assert_eq(cfg.n_heads / cfg.n_kv_heads, 7);
}

Test(qwen, qwen3_gqa_ratio)
{
    OcQwenConfig cfg;
    oc_qwen_config_qwen3_06b(&cfg);
    /* Qwen3-0.6B: 16 heads, 8 KV heads = 2:1 GQA ratio. */
    cr_assert_eq(cfg.n_heads / cfg.n_kv_heads, 2);
}

Test(qwen, model_init_tied)
{
    OcQwenModel model;
    OcQwenConfig cfg;
    oc_qwen_config_qwen3_06b(&cfg);
    cr_assert_eq(oc_qwen_model_init(&model, &cfg), OC_OK);
    cr_assert(model.config.tie_word_embeddings);
    cr_assert_null(model.output);
    oc_qwen_free(&model);
}

Test(qwen, model_init_not_tied)
{
    OcQwenModel model;
    cr_assert_eq(oc_qwen_model_init(&model, NULL), OC_OK);
    cr_assert(!model.config.tie_word_embeddings);
    cr_assert_not_null(model.output);
    oc_qwen_free(&model);
}
