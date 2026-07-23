/* test_inference.c — Inference engine tests. */
#include <criterion/criterion.h>
#include "oxidize/inference.h"
#include "oxidize/model.h"
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

/* ─── OcInferenceConfig tests ─────────────────────────────────────────── */

Test(inf_cfg, init_defaults)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cr_assert_eq(cfg.vocab_size, 32000);
    cr_assert_eq(cfg.context_size, 4096);
    cr_assert_eq(cfg.layer_count, 32);
    cr_assert_eq(cfg.hidden_size, 4096);
    cr_assert_eq(cfg.intermediate_size, 11008);
    cr_assert_eq(cfg.num_attention_heads, 32);
    cr_assert_eq(cfg.num_key_value_heads, 32);
    cr_assert_eq(cfg.key_value_head_dim, 0);
    cr_assert_eq(cfg.kv_cache_dtype, 0);
    cr_assert_float_eq(cfg.rms_norm_eps, 1e-5f, 1e-7f);
    cr_assert_float_eq(cfg.rope_theta, 10000.0f, 0.01f);
    cr_assert_eq(cfg.model_type, OC_INF_MODEL_LLAMA);
    cr_assert_eq(cfg.sliding_window, 0);
    cr_assert_eq(cfg.num_experts, 0);
    cr_assert_eq(cfg.expert_weights_scale, 1.0f);
    cr_assert_eq(cfg.embedding_scale, 1.0f);
    cr_assert(!cfg.gelu_ffn);
    cr_assert(!cfg.sandwich_norm);
    cr_assert(!cfg.rms_norm_weight_plus_one);
    cr_assert_eq(cfg.nextn_predict_layers, 0);
    cr_assert_eq(cfg.yarn_factor, 0.0f);
}

Test(inf_cfg, head_dim)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    /* hidden=4096, heads=32 -> head_dim=128 */
    cr_assert_eq(oc_inference_config_head_dim(&cfg), 128);
}

Test(inf_cfg, head_dim_custom)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.hidden_size = 2048;
    cfg.num_attention_heads = 16;
    cr_assert_eq(oc_inference_config_head_dim(&cfg), 128);
}

Test(inf_cfg, head_dim_null)
{
    cr_assert_eq(oc_inference_config_head_dim(NULL), 0);
}

Test(inf_cfg, effective_rope_dim_default)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    /* rope_dim=0 -> falls back to head_dim=128 */
    cr_assert_eq(oc_inference_config_effective_rope_dim(&cfg), 128);
}

Test(inf_cfg, effective_rope_dim_partial)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.rope_dim = 64;  /* MiniMax-style partial RoPE */
    cr_assert_eq(oc_inference_config_effective_rope_dim(&cfg), 64);
}

Test(inf_cfg, kv_head_dim_default)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    /* key_value_head_dim=0 -> falls back to head_dim=128 */
    cr_assert_eq(oc_inference_config_kv_head_dim(&cfg), 128);
}

Test(inf_cfg, kv_head_dim_custom)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.key_value_head_dim = 96;
    cr_assert_eq(oc_inference_config_kv_head_dim(&cfg), 96);
}

Test(inf_cfg, validate_ok)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cr_assert_eq(oc_inference_config_validate(&cfg), OC_OK);
}

Test(inf_cfg, validate_null)
{
    cr_assert_neq(oc_inference_config_validate(NULL), OC_OK);
}

Test(inf_cfg, validate_zero_hidden)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.hidden_size = 0;
    cr_assert_neq(oc_inference_config_validate(&cfg), OC_OK);
}

Test(inf_cfg, validate_zero_heads)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.num_attention_heads = 0;
    cr_assert_neq(oc_inference_config_validate(&cfg), OC_OK);
}

Test(inf_cfg, validate_mismatch_heads)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.hidden_size = 100;
    cfg.num_attention_heads = 32;  /* 100 % 32 != 0 */
    cr_assert_neq(oc_inference_config_validate(&cfg), OC_OK);
}

Test(inf_cfg, gqa_config)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.num_key_value_heads = 8;  /* GQA: 32 query heads, 8 KV heads */
    cr_assert_eq(oc_inference_config_kv_head_dim(&cfg), 128);
}

Test(inf_cfg, gemma_config)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.model_type = OC_INF_MODEL_GEMMA;
    cfg.gelu_ffn = true;
    cfg.sandwich_norm = true;
    cfg.embedding_scale = 64.0f;  /* sqrt(4096) */
    cfg.rms_norm_weight_plus_one = true;
    cr_assert(cfg.gelu_ffn);
    cr_assert(cfg.sandwich_norm);
    cr_assert(cfg.rms_norm_weight_plus_one);
    cr_assert_float_eq(cfg.embedding_scale, 64.0f, 0.01f);
}

Test(inf_cfg, moe_config)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.num_experts = 8;
    cfg.num_experts_per_tok = 2;
    cfg.expert_intermediate_size = 1792;
    cfg.expert_weights_scale = 2.827f;
    cfg.expert_gating_sigmoid = true;
    cfg.expert_group_count = 4;
    cfg.expert_group_used_count = 1;
    cr_assert_eq(cfg.num_experts, 8);
    cr_assert_eq(cfg.num_experts_per_tok, 2);
    cr_assert_eq(cfg.expert_intermediate_size, 1792);
    cr_assert(cfg.expert_gating_sigmoid);
}

Test(inf_cfg, yarn_config)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.yarn_factor = 1.5f;
    cfg.yarn_orig_ctx = 8192.0f;
    cr_assert_float_eq(cfg.yarn_factor, 1.5f, 0.001f);
    cr_assert_float_eq(cfg.yarn_orig_ctx, 8192.0f, 0.01f);
}

Test(inf_cfg, layer_is_global_no_swa)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    /* No SWA -> all layers are global */
    cr_assert(oc_inference_config_layer_is_global(&cfg, 0));
    cr_assert(oc_inference_config_layer_is_global(&cfg, 5));
    cr_assert(oc_inference_config_layer_is_global(&cfg, 31));
}

Test(inf_cfg, layer_is_global_uniform_swa)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.sliding_window = 4096;
    cfg.sliding_window_pattern = 0;  /* uniform SWA: no global layers */
    cr_assert(!oc_inference_config_layer_is_global(&cfg, 0));
    cr_assert(!oc_inference_config_layer_is_global(&cfg, 5));
}

Test(inf_cfg, layer_is_global_interleaved)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.sliding_window = 1024;
    cfg.sliding_window_pattern = 6;  /* Gemma-style: every 6th layer is global */
    /* Layer 5 (0-indexed) -> (5+1)%6 == 0 -> global */
    cr_assert(oc_inference_config_layer_is_global(&cfg, 5));
    cr_assert(oc_inference_config_layer_is_global(&cfg, 11));
    cr_assert(oc_inference_config_layer_is_global(&cfg, 17));
    /* Layer 0 -> (0+1)%6 == 1 -> local */
    cr_assert(!oc_inference_config_layer_is_global(&cfg, 0));
    cr_assert(!oc_inference_config_layer_is_global(&cfg, 3));
}

Test(inf_cfg, layer_is_global_null)
{
    cr_assert(oc_inference_config_layer_is_global(NULL, 0));
}

Test(inf_cfg, layer_rope_theta_default)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    /* No SWA theta set -> all layers use rope_theta */
    cr_assert_float_eq(oc_inference_config_layer_rope_theta(&cfg, 0), 10000.0f, 0.01f);
    cr_assert_float_eq(oc_inference_config_layer_rope_theta(&cfg, 5), 10000.0f, 0.01f);
}

Test(inf_cfg, layer_rope_theta_swa)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.sliding_window = 1024;
    cfg.sliding_window_pattern = 6;
    cfg.rope_theta = 1000000.0f;       /* global layers */
    cfg.rope_theta_swa = 10000.0f;     /* local layers */
    /* Layer 5 is global -> uses rope_theta */
    cr_assert_float_eq(oc_inference_config_layer_rope_theta(&cfg, 5), 1000000.0f, 0.01f);
    /* Layer 0 is local -> uses rope_theta_swa */
    cr_assert_float_eq(oc_inference_config_layer_rope_theta(&cfg, 0), 10000.0f, 0.01f);
}

Test(inf_cfg, layer_rope_theta_null)
{
    cr_assert_float_eq(oc_inference_config_layer_rope_theta(NULL, 0), 10000.0f, 0.01f);
}

Test(inf_cfg, layer_sliding_window_none)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    /* No SWA configured -> all layers return 0 (full attention) */
    cr_assert_eq(oc_inference_config_layer_sliding_window(&cfg, 0), 0);
    cr_assert_eq(oc_inference_config_layer_sliding_window(&cfg, 5), 0);
}

Test(inf_cfg, layer_sliding_window_uniform)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.sliding_window = 4096;
    cfg.sliding_window_pattern = 0;  /* uniform: all layers are SWA */
    cr_assert_eq(oc_inference_config_layer_sliding_window(&cfg, 0), 4096);
    cr_assert_eq(oc_inference_config_layer_sliding_window(&cfg, 10), 4096);
}

Test(inf_cfg, layer_sliding_window_interleaved)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.sliding_window = 1024;
    cfg.sliding_window_pattern = 6;
    /* Layer 5 is global -> SW = 0 */
    cr_assert_eq(oc_inference_config_layer_sliding_window(&cfg, 5), 0);
    /* Layer 0 is local -> SW = 1024 */
    cr_assert_eq(oc_inference_config_layer_sliding_window(&cfg, 0), 1024);
}

Test(inf_cfg, layer_sliding_window_null)
{
    cr_assert_eq(oc_inference_config_layer_sliding_window(NULL, 0), 0);
}

/* ─── Model arch trait method tests ───────────────────────────────────── */

Test(arch_traits, sliding_window)
{
    cr_assert(oc_model_arch_uses_sliding_window(OC_ARCH_QWEN));
    cr_assert(oc_model_arch_uses_sliding_window(OC_ARCH_MISTRAL));
    cr_assert(!oc_model_arch_uses_sliding_window(OC_ARCH_LLAMA));
    cr_assert(!oc_model_arch_uses_sliding_window(OC_ARCH_GEMMA));
}

Test(arch_traits, shortconv)
{
    cr_assert(oc_model_arch_uses_shortconv(OC_ARCH_LFM2));
    cr_assert(oc_model_arch_uses_shortconv(OC_ARCH_LFM2_MOE));
    cr_assert(!oc_model_arch_uses_shortconv(OC_ARCH_LLAMA));
    cr_assert(!oc_model_arch_uses_shortconv(OC_ARCH_QWEN));
}

Test(arch_traits, parallel_attn_ffn)
{
    cr_assert(oc_model_arch_uses_parallel_attn_ffn(OC_ARCH_GEMMA));
    cr_assert(oc_model_arch_uses_parallel_attn_ffn(OC_ARCH_PHI));
    cr_assert(!oc_model_arch_uses_parallel_attn_ffn(OC_ARCH_LLAMA));
    cr_assert(!oc_model_arch_uses_parallel_attn_ffn(OC_ARCH_MISTRAL));
}

Test(arch_traits, moe)
{
    cr_assert(oc_model_arch_uses_moe(OC_ARCH_MIXTRAL));
    cr_assert(oc_model_arch_uses_moe(OC_ARCH_DEEPSEEK));
    cr_assert(oc_model_arch_uses_moe(OC_ARCH_GLM_MOE_DSA));
    cr_assert(oc_model_arch_uses_moe(OC_ARCH_HUNYUAN_MOE));
    cr_assert(!oc_model_arch_uses_moe(OC_ARCH_LLAMA));
    cr_assert(!oc_model_arch_uses_moe(OC_ARCH_QWEN));
}

Test(arch_traits, mla)
{
    cr_assert(oc_model_arch_uses_mla(OC_ARCH_DEEPSEEK));
    cr_assert(oc_model_arch_uses_mla(OC_ARCH_GLM_MOE_DSA));
    cr_assert(!oc_model_arch_uses_mla(OC_ARCH_LLAMA));
    cr_assert(!oc_model_arch_uses_mla(OC_ARCH_MIXTRAL));
}

Test(arch_traits, alibi)
{
    cr_assert(oc_model_arch_uses_alibi(OC_ARCH_FALCON));
    cr_assert(oc_model_arch_uses_alibi(OC_ARCH_GPT2));
    cr_assert(oc_model_arch_uses_alibi(OC_ARCH_GPTJ));
    cr_assert(oc_model_arch_uses_alibi(OC_ARCH_GPTNEOX));
    cr_assert(!oc_model_arch_uses_alibi(OC_ARCH_LLAMA));
    cr_assert(!oc_model_arch_uses_alibi(OC_ARCH_QWEN));
}
