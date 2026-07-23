/*
 * qwen_arch.h — Qwen architecture forward pass.
 *
 * Qwen2/Qwen3 models use SwiGLU FFN, RoPE, GQA with Q/K-norm,
 * and tie word embeddings. Qwen3 adds QK-norm (RMSNorm on Q and K
 * before RoPE). Key difference from Llama: no tied embeddings by default,
 * separate QK-norm weights per layer.
 *
 * Port of oxidize-core ModelArchitecture::Qwen forward path.
 * The forward function is a stub that zero-fills the logits buffer.
 *
 * Weight tensor names (GGUF):
 *   token_embd.weight
 *   output_norm.weight, output.weight (or tied to token_embd)
 *   blk.N.attn_norm.weight
 *   blk.N.attn_q.weight, blk.N.attn_q_norm.weight
 *   blk.N.attn_k.weight, blk.N.attn_k_norm.weight
 *   blk.N.attn_v.weight, blk.N.attn_output.weight
 *   blk.N.ffn_norm.weight
 *   blk.N.ffn_gate.weight, blk.N.ffn_up.weight, blk.N.ffn_down.weight
 */
#ifndef OXIDIZE_QWEN_ARCH_H
#define OXIDIZE_QWEN_ARCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Qwen model configuration. Defaults match Qwen2.5-7B:
 *   n_layers=28, n_heads=28, n_kv_heads=4, head_dim=128,
 *   hidden_dim=3584, intermediate_dim=18944, vocab_size=152064,
 *   rope_theta=1000000.0, max_position=32768.
 * For Qwen3-0.6B: n_layers=28, n_heads=16, n_kv_heads=8,
 *   head_dim=128, hidden_dim=1024, intermediate_dim=3072,
 *   vocab_size=151936, rope_theta=1000000.0 */
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
