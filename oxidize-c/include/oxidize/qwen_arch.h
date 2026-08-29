/* qwen_arch.h — Qwen architecture forward pass. */
#ifndef OXIDIZE_QWEN_ARCH_H
#define OXIDIZE_QWEN_ARCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Qwen model configuration. Defaults match Qwen2.5-7B: */
typedef struct OcQwenConfig {
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t vocab_size;
    float    rope_theta;
    uint32_t max_position;
    bool     tie_word_embeddings;
    bool     use_qk_norm;       /* Qwen3 adds RMSNorm on Q/K */
    float    norm_eps;
} OcQwenConfig;

typedef struct OcQwenLayer {
    float *attn_norm;     /* RMSNorm before attention  ([hidden_dim]) */
    float *attn_q;        /* query projection           */
    float *attn_q_norm;   /* QK-norm on Q  ([head_dim]) */
    float *attn_k;        /* key projection             */
    float *attn_k_norm;   /* QK-norm on K  ([head_dim]) */
    float *attn_v;        /* value projection           */
    float *attn_output;   /* output projection          */
    float *ffn_norm;      /* RMSNorm before FFN        ([hidden_dim]) */
    float *ffn_gate;      /* SwiGLU gate                */
    float *ffn_up;        /* SwiGLU up                  */
    float *ffn_down;      /* SwiGLU down                */
} OcQwenLayer;

typedef struct OcQwenModel {
    OcQwenConfig  config;
    OcQwenLayer   *layers;
    float         *tok_emb;       /* [vocab_size, hidden_dim] */
    float         *output_norm;   /* [hidden_dim] */
    float         *output;        /* [vocab_size, hidden_dim] (NULL if tied) */
    bool           initialized;
} OcQwenModel;

OcError oc_qwen_config_init(OcQwenConfig *cfg);
OcError oc_qwen_config_qwen25_7b(OcQwenConfig *cfg);
OcError oc_qwen_config_qwen3_06b(OcQwenConfig *cfg);
OcError oc_qwen_model_init(OcQwenModel *model, const OcQwenConfig *cfg);
OcError oc_qwen_forward(OcQwenModel *model, uint32_t token, float *logits);
void oc_qwen_free(OcQwenModel *model);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_QWEN_ARCH_H */
