/* test_diffusion_gemma.c — Gemma diffusion model tests. */
#include "framework.h"
#include "oxidize/diffusion_gemma.h"
#include <string.h>

Test(dg, config_init)
{
    OcDiffGemmaConfig cfg;
    cr_assert_eq(oc_diff_gemma_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.n_layers, 18);
    cr_assert_eq(cfg.hidden_dim, 2048);
    cr_assert_eq(cfg.vocab_size, 256000);
    cr_assert_eq(cfg.n_diffusion_steps, 50);
    cr_assert_float_eq(cfg.sigma_min, 0.002f, 0.0001f);
    cr_assert_float_eq(cfg.sigma_max, 80.0f, 0.1f);
    cr_assert_eq(cfg.sampler, OC_DIFF_GEMMA_FLOW_MATCH);
}

OC_TEST_NULL_SAFE(dg, config_init_null,
        cr_assert_neq(oc_diff_gemma_config_init(NULL), OC_OK);)

Test(dg, model_init)
{
    OcDiffGemmaModel model;
    cr_assert_eq(oc_diff_gemma_model_init(&model, NULL), OC_OK);
    cr_assert(model.initialized);
    cr_assert_not_null(model.embedding);
    cr_assert_not_null(model.output);
    oc_diff_gemma_free(&model);
}

Test(dg, model_init_custom)
{
    OcDiffGemmaModel model;
    OcDiffGemmaConfig cfg;
    oc_diff_gemma_config_init(&cfg);
    cfg.n_layers = 10;
    cfg.hidden_dim = 512;
    cfg.vocab_size = 1000;
    cr_assert_eq(oc_diff_gemma_model_init(&model, &cfg), OC_OK);
    cr_assert_eq(model.config.n_layers, 10);
    oc_diff_gemma_free(&model);
}

OC_TEST_NULL_SAFE(dg, model_init_null,
        cr_assert_neq(oc_diff_gemma_model_init(NULL, NULL), OC_OK);)

Test(dg, model_init_bad)
{
    OcDiffGemmaModel model;
    OcDiffGemmaConfig cfg = {0};
    cr_assert_neq(oc_diff_gemma_model_init(&model, &cfg), OC_OK);
}

Test(dg, forward)
{
    OcDiffGemmaModel model;
    oc_diff_gemma_model_init(&model, NULL);
    float *logits = calloc(model.config.vocab_size, sizeof(float));
    memset(logits, 0x42, model.config.vocab_size * sizeof(float));
    cr_assert_eq(oc_diff_gemma_forward(&model, 0, 1.0f, logits), OC_OK);
    for (int i = 0; i < 10; i++)
        cr_assert_float_eq(logits[i], 0.0f, 0.001f);
    free(logits);
    oc_diff_gemma_free(&model);
}

Test(dg, forward_uninit)
{
    OcDiffGemmaModel model = {0};
    float logits[10];
    cr_assert_neq(oc_diff_gemma_forward(&model, 0, 0, logits), OC_OK);
}

OC_TEST_NULL_SAFE(dg, forward_null,
        cr_assert_neq(oc_diff_gemma_forward(NULL, 0, 0, NULL), OC_OK);)

Test(dg, sample)
{
    OcDiffGemmaModel model;
    oc_diff_gemma_model_init(&model, NULL);
    float *logits = calloc(model.config.vocab_size, sizeof(float));
    cr_assert_eq(oc_diff_gemma_sample(&model, logits, model.config.vocab_size, 0), OC_OK);
    free(logits);
    oc_diff_gemma_free(&model);
}

Test(dg, denoise)
{
    OcDiffGemmaModel model;
    oc_diff_gemma_model_init(&model, NULL);
    float tokens[] = {1.0f, 2.0f, 3.0f};
    cr_assert_eq(oc_diff_gemma_denoise(&model, tokens, 3, 80.0f, 0.002f), OC_OK);
    oc_diff_gemma_free(&model);
}

OC_TEST_NULL_SAFE(dg, denoise_null,
        cr_assert_neq(oc_diff_gemma_denoise(NULL, NULL, 0, 0, 0), OC_OK);)

Test(dg, sampler_name)
{
    cr_assert_str_eq(oc_diff_gemma_sampler_name(OC_DIFF_GEMMA_DDIM), "ddim");
    cr_assert_str_eq(oc_diff_gemma_sampler_name(OC_DIFF_GEMMA_DPM2M), "dpm2m");
    cr_assert_str_eq(oc_diff_gemma_sampler_name(OC_DIFF_GEMMA_EULER), "euler");
    cr_assert_str_eq(oc_diff_gemma_sampler_name(OC_DIFF_GEMMA_FLOW_MATCH), "flow_match");
}

OC_TEST_NULL_SAFE(dg, free_null,
        oc_diff_gemma_free(NULL);)

Test(dg, free_reuse)
{
    OcDiffGemmaModel model;
    oc_diff_gemma_model_init(&model, NULL);
    oc_diff_gemma_free(&model);
    oc_diff_gemma_free(&model);
}
