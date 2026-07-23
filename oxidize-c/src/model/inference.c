/*
 * inference.c — High-level inference engine implementation.
 */
#include "oxidize/inference.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

__attribute__((unused))
static void copy_str(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0 || !src) { if (dst && cap > 0) dst[0] = '\0'; return; }
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

OcError oc_inf_config_init(OcInfConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->model_type = OC_INF_MODEL_LLAMA;
    cfg->model_path = NULL;
    cfg->n_threads = 0; /* 0 = auto */
    cfg->n_ctx = 4096;
    cfg->n_batch = 512;
    cfg->use_gpu = false;
    cfg->use_numa = false;
    cfg->verbose = false;
    return OC_OK;
}

/* ─── OcInferenceConfig (full model config) ────────────────────────────── */

void oc_inference_config_init(OcInferenceConfig *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->vocab_size          = 32000;
    cfg->context_size        = 4096;
    cfg->layer_count         = 32;
    cfg->hidden_size         = 4096;
    cfg->intermediate_size   = 11008;
    cfg->num_attention_heads  = 32;
    cfg->num_key_value_heads  = 32;
    cfg->key_value_head_dim   = 0;  /* 0 = hidden/num_heads */
    cfg->kv_cache_dtype       = 0;  /* f32 */
    cfg->rms_norm_eps         = 1e-5f;
    cfg->rope_theta           = 10000.0f;
    cfg->model_type           = OC_INF_MODEL_LLAMA;
    cfg->sliding_window       = 0;
    cfg->alibi_num_heads      = 0;
    cfg->rope_dim             = 0;
    cfg->rope_theta_swa       = 0.0f;
    cfg->sliding_window_pattern = 0;
    cfg->num_experts          = 0;
    cfg->num_experts_per_tok  = 0;
    cfg->expert_intermediate_size = 0;
    cfg->expert_gating_sigmoid = false;
    cfg->expert_weights_scale = 1.0f;
    cfg->expert_group_count   = 0;
    cfg->expert_group_used_count = 0;
    cfg->leading_dense_layers = 0;
    cfg->shortconv_l_cache    = 0;
    cfg->embedding_scale      = 1.0f;
    cfg->gelu_ffn             = false;
    cfg->sandwich_norm        = false;
    cfg->rms_norm_weight_plus_one = false;
    cfg->nextn_predict_layers = 0;
    cfg->yarn_factor          = 0.0f;
    cfg->yarn_orig_ctx        = 0.0f;
}

uint32_t oc_inference_config_head_dim(const OcInferenceConfig *cfg)
{
    if (!cfg || cfg->num_attention_heads == 0) return 0;
    return cfg->hidden_size / cfg->num_attention_heads;
}

uint32_t oc_inference_config_effective_rope_dim(const OcInferenceConfig *cfg)
{
    if (!cfg) return 0;
    uint32_t hd = oc_inference_config_head_dim(cfg);
    return cfg->rope_dim > 0 ? cfg->rope_dim : hd;
}

uint32_t oc_inference_config_kv_head_dim(const OcInferenceConfig *cfg)
{
    if (!cfg) return 0;
    if (cfg->key_value_head_dim > 0) return cfg->key_value_head_dim;
    return oc_inference_config_head_dim(cfg);
}

OcError oc_inference_config_validate(const OcInferenceConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    if (cfg->hidden_size == 0) return OC_ERR_INVALID_ARG;
    if (cfg->num_attention_heads == 0) return OC_ERR_INVALID_ARG;
    if (cfg->vocab_size == 0) return OC_ERR_INVALID_ARG;
    if (cfg->layer_count == 0) return OC_ERR_INVALID_ARG;
    if (cfg->hidden_size % cfg->num_attention_heads != 0)
        return OC_ERR_INVALID_ARG;
    return OC_OK;
}

bool oc_inference_config_layer_is_global(const OcInferenceConfig *cfg, uint32_t layer_idx)
{
    if (!cfg) return true;
    if (cfg->sliding_window == 0) return true;
    if (cfg->sliding_window_pattern == 0) return false;
    return ((layer_idx + 1) % cfg->sliding_window_pattern) == 0;
}

float oc_inference_config_layer_rope_theta(const OcInferenceConfig *cfg, uint32_t layer_idx)
{
    if (!cfg) return 10000.0f;
    if (cfg->rope_theta_swa > 0.0f && !oc_inference_config_layer_is_global(cfg, layer_idx))
        return cfg->rope_theta_swa;
    return cfg->rope_theta;
}

uint32_t oc_inference_config_layer_sliding_window(const OcInferenceConfig *cfg, uint32_t layer_idx)
{
    if (!cfg) return 0;
    if (cfg->sliding_window > 0 && !oc_inference_config_layer_is_global(cfg, layer_idx))
        return cfg->sliding_window;
    return 0;
}

void oc_inference_config_apply_rope_head(const OcInferenceConfig *cfg,
                                          const float *input, float *output,
                                          size_t head_dim, size_t rope_len,
                                          int64_t position, float theta)
{
    if (!cfg || !input || !output) return;
    oc_apply_rope_yarn_f32(input, output, head_dim, rope_len,
                           position, theta,
                           cfg->yarn_factor, (uint32_t)cfg->yarn_orig_ctx);
}

OcError oc_inf_engine_init(OcInfEngine *engine, const OcInfConfig *cfg)
{
    if (!engine) return OC_ERR_INVALID_ARG;
    memset(engine, 0, sizeof(*engine));
    if (cfg) {
        engine->config = *cfg;
    } else {
        oc_inf_config_init(&engine->config);
    }
    engine->loaded = false;
    engine->n_loaded_layers = 0;
    engine->model_size_bytes = 0;
    return OC_OK;
}

OcError oc_inf_engine_load(OcInfEngine *engine, const char *model_path)
{
    if (!engine || !model_path) return OC_ERR_INVALID_ARG;
    /* Stub: in real implementation, load GGUF model. */
    engine->config.model_path = model_path;
    engine->loaded = true;
    engine->n_loaded_layers = 32;
    engine->model_size_bytes = (size_t)4096 * 1024 * 1024; /* 4 GB stub */
    return OC_OK;
}

OcError oc_inf_engine_generate(OcInfEngine *engine, const char *prompt,
                              const OcGenConfig *gen_cfg,
                              char *out_text, size_t out_size,
                              OcGenResult *result)
{
    (void)gen_cfg;
    if (!engine || !prompt || !out_text) return OC_ERR_INVALID_ARG;
    if (!engine->loaded) return OC_ERR_MODEL;

    if (out_size > 0) out_text[0] = '\0';
    if (result) {
        memset(result, 0, sizeof(*result));
        result->n_prompt_tokens = strlen(prompt);
        result->stopped_on_eos = true;
    }
    return OC_OK;
}

OcError oc_inf_engine_encode(OcInfEngine *engine, const char *text,
                            uint32_t **out_tokens, size_t *out_n)
{
    if (!engine || !text || !out_tokens || !out_n) return OC_ERR_INVALID_ARG;
    if (!engine->loaded) return OC_ERR_MODEL;

    size_t len = strlen(text);
    if (len > OC_INF_MAX_CONTEXT) len = OC_INF_MAX_CONTEXT;

    uint32_t *tokens = malloc(len * sizeof(uint32_t));
    if (!tokens) return OC_ERR_OOM;
    for (size_t i = 0; i < len; i++)
        tokens[i] = (uint32_t)(unsigned char)text[i];

    *out_tokens = tokens;
    *out_n = len;
    return OC_OK;
}

OcError oc_inf_engine_decode(OcInfEngine *engine, const uint32_t *tokens,
                           size_t n, char *out, size_t out_size)
{
    if (!engine || !tokens || !out) return OC_ERR_INVALID_ARG;
    if (!engine->loaded) return OC_ERR_MODEL;

    size_t copy_n = n < out_size - 1 ? n : out_size - 1;
    for (size_t i = 0; i < copy_n; i++)
        out[i] = (char)(tokens[i] & 0xFF);
    out[copy_n] = '\0';
    return OC_OK;
}

OcError oc_inf_engine_stats(const OcInfEngine *engine,
                           char *out, size_t out_size)
{
    if (!engine || !out || out_size == 0) return OC_ERR_INVALID_ARG;
    snprintf(out, out_size,
        "=== Inference Engine Stats ===\n"
        "Model type: %s\n"
        "Loaded: %s\n"
        "Layers: %u\n"
        "Model size: %zu MB\n"
        "Context: %u\n"
        "Threads: %u\n"
        "GPU: %s\n"
        "NUMA: %s\n",
        oc_inf_model_type_name(engine->config.model_type),
        engine->loaded ? "yes" : "no",
        engine->n_loaded_layers,
        engine->model_size_bytes / (1024 * 1024),
        engine->config.n_ctx,
        engine->config.n_threads,
        engine->config.use_gpu ? "yes" : "no",
        engine->config.use_numa ? "yes" : "no");
    return OC_OK;
}

bool oc_inf_engine_is_loaded(const OcInfEngine *engine)
{
    return engine ? engine->loaded : false;
}

OcInfModelType oc_inf_model_type_from_arch(const char *arch_name)
{
    if (!arch_name) return OC_INF_MODEL_LLAMA;
    if (strcmp(arch_name, "llama") == 0) return OC_INF_MODEL_LLAMA;
    if (strcmp(arch_name, "mistral") == 0) return OC_INF_MODEL_MISTRAL;
    if (strcmp(arch_name, "gemma") == 0) return OC_INF_MODEL_GEMMA;
    if (strcmp(arch_name, "phi") == 0 || strcmp(arch_name, "phi2") == 0 ||
        strcmp(arch_name, "phi3") == 0) return OC_INF_MODEL_PHI;
    if (strcmp(arch_name, "glm") == 0 || strcmp(arch_name, "chatglm") == 0)
        return OC_INF_MODEL_GLM;
    if (strcmp(arch_name, "qwen") == 0 || strcmp(arch_name, "qwen2") == 0)
        return OC_INF_MODEL_QWEN;
    return OC_INF_MODEL_LLAMA;
}

const char *oc_inf_model_type_name(OcInfModelType type)
{
    switch (type) {
    case OC_INF_MODEL_LLAMA:   return "llama";
    case OC_INF_MODEL_MISTRAL: return "mistral";
    case OC_INF_MODEL_GEMMA:   return "gemma";
    case OC_INF_MODEL_PHI:     return "phi";
    case OC_INF_MODEL_GLM:     return "glm";
    case OC_INF_MODEL_QWEN:    return "qwen";
    default: return "unknown";
    }
}

const char *oc_inf_model_type_arch(OcInfModelType type)
{
    switch (type) {
    case OC_INF_MODEL_LLAMA:   return "llama";
    case OC_INF_MODEL_MISTRAL: return "mistral";
    case OC_INF_MODEL_GEMMA:   return "gemma";
    case OC_INF_MODEL_PHI:     return "phi3";
    case OC_INF_MODEL_GLM:     return "chatglm";
    case OC_INF_MODEL_QWEN:    return "qwen2";
    default: return "llama";
    }
}

void oc_inf_engine_free(OcInfEngine *engine)
{
    if (!engine) return;
    memset(engine, 0, sizeof(*engine));
}
