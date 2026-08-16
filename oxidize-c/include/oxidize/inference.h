/*
 * inference.h — High-level inference engine.
 *
 * Wraps model loading, tokenization, and generation into a clean API.
 * Port from oxidize-core/src/model/inference.rs.
 */
#ifndef OXIDIZE_INFERENCE_H
#define OXIDIZE_INFERENCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/generation.h"
#include "oxidize/activation.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_INF_MAX_CONTEXT 32768

typedef enum {
    OC_INF_MODEL_LLAMA = 0,
    OC_INF_MODEL_MISTRAL = 1,
    OC_INF_MODEL_GEMMA = 2,
    OC_INF_MODEL_PHI = 3,
    OC_INF_MODEL_GLM = 4,
    OC_INF_MODEL_QWEN = 5,
} OcInfModelType;

/* Full inference configuration mirroring Rust InferenceConfig.
 * Used by the engine to configure model architecture parameters. */
typedef struct {
    /* Basic dimensions. */
    uint32_t vocab_size;
    uint32_t context_size;
    uint32_t layer_count;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t key_value_head_dim;       /* 0 = hidden/num_heads */
    uint32_t kv_cache_dtype;           /* 0=f32, 1=f16 */
    float    rms_norm_eps;
    float    rope_theta;
    OcInfModelType model_type;

    /* Attention / positional. */
    uint32_t sliding_window;           /* 0 = full attention */
    uint32_t alibi_num_heads;           /* 0 = not used */
    uint32_t rope_dim;                  /* 0 = full head_dim */
    float    rope_theta_swa;            /* 0 = use rope_theta */
    uint32_t sliding_window_pattern;    /* 0 = no interleaving */

    /* MoE. */
    uint32_t num_experts;
    uint32_t num_experts_per_tok;
    uint32_t expert_intermediate_size;   /* 0 = intermediate_size */
    bool     expert_gating_sigmoid;
    float    expert_weights_scale;
    uint32_t expert_group_count;
    uint32_t expert_group_used_count;
    uint32_t leading_dense_layers;

    /* Architecture-specific. */
    uint32_t shortconv_l_cache;         /* LFM2 conv kernel width */
    float    embedding_scale;           /* Gemma: sqrt(hidden) */
    bool     gelu_ffn;                  /* Gemma: GeGLU vs SwiGLU */
    bool     sandwich_norm;             /* Gemma: pre-residual norms */
    bool     rms_norm_weight_plus_one;  /* Qwen: (1+w) norm */
    uint32_t nextn_predict_layers;      /* MTP/nextn draft layers */

    /* YaRN rope extension. */
    float    yarn_factor;               /* 0 = disabled */
    float    yarn_orig_ctx;             /* original context length */
    /* deepseek_yarn only: the two mscale knobs. Left at 0 by every other
     * architecture, which keeps the plain 1/sqrt(head_dim) attention scale
     * and an unscaled RoPE. See oc_rope_deepseek_yarn_scales(). */
    float    yarn_mscale;
    float    yarn_mscale_all_dim;

    /* LongCat: identity experts occupying router slots
     * [num_experts, num_experts + zero_expert_count). They hold no weights
     * and contribute their weighted normalized input to the FFN output.
     * Appended to preserve the layout of the pre-existing configuration fields. */
    uint32_t zero_expert_count;
} OcInferenceConfig;

/* Initialize with Rust Default values. */
void oc_inference_config_init(OcInferenceConfig *cfg);

/* Compute head_dim = hidden_size / num_attention_heads. */
uint32_t oc_inference_config_head_dim(const OcInferenceConfig *cfg);

/* Compute effective RoPE dimension (rope_dim or head_dim). */
uint32_t oc_inference_config_effective_rope_dim(const OcInferenceConfig *cfg);

/* Compute KV head dim (key_value_head_dim or head_dim). */
uint32_t oc_inference_config_kv_head_dim(const OcInferenceConfig *cfg);

/* Validate config: check hidden_size > 0, num_heads > 0, etc. */
OcError oc_inference_config_validate(const OcInferenceConfig *cfg);

/* Whether the given layer index uses global (full) attention vs sliding-window.
 * - No SWA (sliding_window == 0): every layer is global.
 * - Uniform SWA (pattern == 0): no layer is global.
 * - Interleaved (pattern > 0): every pattern-th layer (1-indexed) is global. */
bool oc_inference_config_layer_is_global(const OcInferenceConfig *cfg, uint32_t layer_idx);

/* RoPE theta for the given layer: global layers use rope_theta;
 * SWA layers use rope_theta_swa when set, otherwise rope_theta. */
float oc_inference_config_layer_rope_theta(const OcInferenceConfig *cfg, uint32_t layer_idx);

/* Effective sliding-window size for the given layer (0 = full attention). */
uint32_t oc_inference_config_layer_sliding_window(const OcInferenceConfig *cfg, uint32_t layer_idx);

/* Apply RoPE to one head using the config's YaRN parameters.
 * Wraps oc_apply_rope_yarn_f32 with cfg->yarn_factor and cfg->yarn_orig_ctx. */
void oc_inference_config_apply_rope_head(const OcInferenceConfig *cfg,
                                          const float *input, float *output,
                                          size_t head_dim, size_t rope_len,
                                          int64_t position, float theta);

/* ─── Legacy CLI config (still used by the CLI) ────────────────────────── */

typedef struct {
    OcInfModelType model_type;
    const char *model_path;
    uint32_t n_threads;
    uint32_t n_ctx;
    uint32_t n_batch;
    bool use_gpu;
    bool use_numa;
    bool verbose;
} OcInfConfig;

typedef struct {
    void *model;
    void *tokenizer;
    OcInfConfig config;
    bool loaded;
    uint32_t n_loaded_layers;
    size_t model_size_bytes;
} OcInfEngine;

OcError oc_inf_config_init(OcInfConfig *cfg);
OcError oc_inf_engine_init(OcInfEngine *engine, const OcInfConfig *cfg);
OcError oc_inf_engine_load(OcInfEngine *engine, const char *model_path);
OcError oc_inf_engine_generate(OcInfEngine *engine, const char *prompt,
                              const OcGenConfig *gen_cfg,
                              char *out_text, size_t out_size,
                              OcGenResult *result);
OcError oc_inf_engine_encode(OcInfEngine *engine, const char *text,
                            uint32_t **out_tokens, size_t *out_n);
OcError oc_inf_engine_decode(OcInfEngine *engine, const uint32_t *tokens,
                           size_t n, char *out, size_t out_size);
OcError oc_inf_engine_stats(const OcInfEngine *engine,
                           char *out, size_t out_size);
bool oc_inf_engine_is_loaded(const OcInfEngine *engine);
OcInfModelType oc_inf_model_type_from_arch(const char *arch_name);
const char *oc_inf_model_type_name(OcInfModelType type);
const char *oc_inf_model_type_arch(OcInfModelType type);
void oc_inf_engine_free(OcInfEngine *engine);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_INFERENCE_H */
