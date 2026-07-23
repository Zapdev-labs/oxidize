/* test_inference.c — Inference engine tests. */
#include <criterion/criterion.h>
#include "oxidize/inference.h"
#include <string.h>

Test(inf, config_init)
{
    OcInfConfig cfg;
    cr_assert_eq(oc_inf_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.model_type, OC_INF_MODEL_LLAMA);
    cr_assert_eq(cfg.n_ctx, 4096);
    cr_assert_eq(cfg.n_batch, 512);
    cr_assert(!cfg.use_gpu);
}

Test(inf, config_init_null)
{
    cr_assert_neq(oc_inf_config_init(NULL), OC_OK);
}

Test(inf, engine_init)
{
    OcInfEngine engine;
    cr_assert_eq(oc_inf_engine_init(&engine, NULL), OC_OK);
    cr_assert(!engine.loaded);
    cr_assert_eq(engine.n_loaded_layers, 0);
    oc_inf_engine_free(&engine);
}

Test(inf, engine_init_with_config)
{
    OcInfEngine engine;
    OcInfConfig cfg;
    oc_inf_config_init(&cfg);
    cfg.n_ctx = 8192;
    cfg.use_gpu = true;
    cr_assert_eq(oc_inf_engine_init(&engine, &cfg), OC_OK);
    cr_assert_eq(engine.config.n_ctx, 8192);
    cr_assert(engine.config.use_gpu);
    oc_inf_engine_free(&engine);
}

Test(inf, engine_init_null)
{
    cr_assert_neq(oc_inf_engine_init(NULL, NULL), OC_OK);
}

Test(inf, engine_load)
{
    OcInfEngine engine;
    oc_inf_engine_init(&engine, NULL);
    cr_assert_eq(oc_inf_engine_load(&engine, "/tmp/model.gguf"), OC_OK);
    cr_assert(engine.loaded);
    cr_assert_eq(engine.n_loaded_layers, 32);
    oc_inf_engine_free(&engine);
}

Test(inf, engine_load_null)
{
    cr_assert_neq(oc_inf_engine_load(NULL, NULL), OC_OK);
}

Test(inf, engine_generate_unloaded)
{
    OcInfEngine engine;
    oc_inf_engine_init(&engine, NULL);
    char out[256];
    cr_assert_neq(oc_inf_engine_generate(&engine, "hi", NULL, out, sizeof(out), NULL), OC_OK);
    oc_inf_engine_free(&engine);
}

Test(inf, engine_generate_loaded)
{
    OcInfEngine engine;
    oc_inf_engine_init(&engine, NULL);
    oc_inf_engine_load(&engine, "/tmp/model.gguf");
    char out[256];
    OcGenResult result;
    cr_assert_eq(oc_inf_engine_generate(&engine, "hello", NULL, out, sizeof(out), &result), OC_OK);
    cr_assert_eq(result.n_prompt_tokens, 5);
    cr_assert(result.stopped_on_eos);
    oc_inf_engine_free(&engine);
}

Test(inf, engine_encode)
{
    OcInfEngine engine;
    oc_inf_engine_init(&engine, NULL);
    oc_inf_engine_load(&engine, "/tmp/model.gguf");
    uint32_t *tokens;
    size_t n;
    cr_assert_eq(oc_inf_engine_encode(&engine, "hello", &tokens, &n), OC_OK);
    cr_assert_eq(n, 5);
    cr_assert_eq(tokens[0], (uint32_t)'h');
    free(tokens);
    oc_inf_engine_free(&engine);
}

Test(inf, engine_decode)
{
    OcInfEngine engine;
    oc_inf_engine_init(&engine, NULL);
    oc_inf_engine_load(&engine, "/tmp/model.gguf");
    uint32_t tokens[] = {(uint32_t)'h', (uint32_t)'i'};
    char out[16];
    cr_assert_eq(oc_inf_engine_decode(&engine, tokens, 2, out, sizeof(out)), OC_OK);
    cr_assert_str_eq(out, "hi");
    oc_inf_engine_free(&engine);
}

Test(inf, engine_stats)
{
    OcInfEngine engine;
    oc_inf_engine_init(&engine, NULL);
    oc_inf_engine_load(&engine, "/tmp/model.gguf");
    char stats[1024];
    cr_assert_eq(oc_inf_engine_stats(&engine, stats, sizeof(stats)), OC_OK);
    cr_assert(strstr(stats, "Inference Engine Stats") != NULL);
    cr_assert(strstr(stats, "Loaded: yes") != NULL);
    oc_inf_engine_free(&engine);
}

Test(inf, engine_stats_null)
{
    cr_assert_neq(oc_inf_engine_stats(NULL, NULL, 0), OC_OK);
}

Test(inf, is_loaded)
{
    OcInfEngine engine;
    oc_inf_engine_init(&engine, NULL);
    cr_assert(!oc_inf_engine_is_loaded(&engine));
    oc_inf_engine_load(&engine, "/tmp/model.gguf");
    cr_assert(oc_inf_engine_is_loaded(&engine));
    oc_inf_engine_free(&engine);
}

Test(inf, is_loaded_null)
{
    cr_assert(!oc_inf_engine_is_loaded(NULL));
}

Test(inf, model_type_from_arch)
{
    cr_assert_eq(oc_inf_model_type_from_arch("llama"), OC_INF_MODEL_LLAMA);
    cr_assert_eq(oc_inf_model_type_from_arch("mistral"), OC_INF_MODEL_MISTRAL);
    cr_assert_eq(oc_inf_model_type_from_arch("gemma"), OC_INF_MODEL_GEMMA);
    cr_assert_eq(oc_inf_model_type_from_arch("phi3"), OC_INF_MODEL_PHI);
    cr_assert_eq(oc_inf_model_type_from_arch("chatglm"), OC_INF_MODEL_GLM);
    cr_assert_eq(oc_inf_model_type_from_arch("qwen2"), OC_INF_MODEL_QWEN);
    cr_assert_eq(oc_inf_model_type_from_arch(NULL), OC_INF_MODEL_LLAMA);
    cr_assert_eq(oc_inf_model_type_from_arch("unknown"), OC_INF_MODEL_LLAMA);
}

Test(inf, model_type_name)
{
    cr_assert_str_eq(oc_inf_model_type_name(OC_INF_MODEL_LLAMA), "llama");
    cr_assert_str_eq(oc_inf_model_type_name(OC_INF_MODEL_MISTRAL), "mistral");
    cr_assert_str_eq(oc_inf_model_type_name(OC_INF_MODEL_GEMMA), "gemma");
    cr_assert_str_eq(oc_inf_model_type_name(OC_INF_MODEL_PHI), "phi");
    cr_assert_str_eq(oc_inf_model_type_name(OC_INF_MODEL_GLM), "glm");
    cr_assert_str_eq(oc_inf_model_type_name(OC_INF_MODEL_QWEN), "qwen");
}

Test(inf, model_type_arch)
{
    cr_assert_str_eq(oc_inf_model_type_arch(OC_INF_MODEL_LLAMA), "llama");
    cr_assert_str_eq(oc_inf_model_type_arch(OC_INF_MODEL_MISTRAL), "mistral");
    cr_assert_str_eq(oc_inf_model_type_arch(OC_INF_MODEL_GLM), "chatglm");
    cr_assert_str_eq(oc_inf_model_type_arch(OC_INF_MODEL_QWEN), "qwen2");
}

Test(inf, free_null)
{
    oc_inf_engine_free(NULL);
}

Test(inf, encode_decode_roundtrip)
{
    OcInfEngine engine;
    oc_inf_engine_init(&engine, NULL);
    oc_inf_engine_load(&engine, "/tmp/model.gguf");
    const char *text = "hello world";
    uint32_t *tokens;
    size_t n;
    oc_inf_engine_encode(&engine, text, &tokens, &n);
    char out[32];
    oc_inf_engine_decode(&engine, tokens, n, out, sizeof(out));
    cr_assert_str_eq(out, text);
    free(tokens);
    oc_inf_engine_free(&engine);
}
