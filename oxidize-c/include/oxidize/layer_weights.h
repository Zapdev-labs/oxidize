/*
 * layer_weights.h — Per-layer weight bundle for inference.
 *
 * Port of oxidize-core/src/model/inference.rs::LayerWeights.
 *
 * Contains all weight tensors for a single transformer layer:
 * attention norms, Q/K/V projections, output projection, FFN weights
 * (dense or MoE), SSM/Mamba tensors, short-conv, MLA, and per-head norms.
 */
#ifndef OXIDIZE_LAYER_WEIGHTS_H
#define OXIDIZE_LAYER_WEIGHTS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/weight_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Attention norms + projections. */
    float            *attn_norm;        /* [hidden_size] or NULL */
    OcWeightStorage   attn_q;           /* Q projection */
    float            *attn_q_bias;      /* [n_heads * head_dim] or NULL */
    OcWeightStorage   attn_k;           /* K projection */
    float            *attn_k_bias;
    OcWeightStorage   attn_v;           /* V projection */
    float            *attn_v_bias;
    OcWeightStorage   attn_output;      /* output projection */
    float            *attn_output_bias;

    /* FFN norms. */
    float            *ffn_norm;          /* pre-FFN norm */
    float            *post_attention_norm; /* Gemma sandwich: post-attn */
    float            *post_ffn_norm;     /* Gemma sandwich: post-FFN */

    /* Dense FFN weights (num_experts == 0). */
    OcWeightStorage   ffn_gate;         /* gate_proj */
    OcWeightStorage   ffn_up;           /* up_proj */
    OcWeightStorage   ffn_down;         /* down_proj */
    float            *ffn_down_bias;

    /* MoE expert weights (num_experts > 0). */
    OcWeightStorage   ffn_gate_exps;    /* [n_experts, inter, hidden] */
    OcWeightStorage   ffn_up_exps;
    OcWeightStorage   ffn_down_exps;
    OcWeightStorage   ffn_gate_inp;     /* router: [hidden, n_experts] */

    /* Fused QKV (for architectures that store QKV as one tensor). */
    OcWeightStorage   attn_qkv;

    /* SSM / Mamba tensors. */
    OcWeightStorage   attn_gate;
    float            *ssm_a;
    float            *ssm_alpha;
    float            *ssm_beta;
    float            *ssm_conv1d;
    float            *ssm_dt_bias;
    float            *ssm_norm;
    OcWeightStorage   ssm_out;

    /* Per-head Q/K norms (Qwen, Gemma). */
    float            *attn_q_norm;
    float            *attn_k_norm;

    /* LFM2 short-convolution. */
    OcWeightStorage   shortconv_in_proj;
    float            *shortconv_conv;
    OcWeightStorage   shortconv_out_proj;

    /* LFM2MoE per-layer expert routing bias. */
    float            *ffn_exp_probs_b;

    /* DeepSeek2 MLA compressed attention. */
    OcWeightStorage   mla_q_a;
    float            *mla_q_a_norm;
    OcWeightStorage   mla_q_b;
    OcWeightStorage   mla_kv_a_mqa;
    float            *mla_kv_a_norm;
    OcWeightStorage   mla_k_b;
    OcWeightStorage   mla_v_b;

    /* Count of allocated float* fields (for memory management). */
    size_t            n_attn_norm;
    size_t            n_ffn_norm;
    size_t            n_post_attn_norm;
    size_t            n_post_ffn_norm;
    size_t            n_q_bias;
    size_t            n_k_bias;
    size_t            n_v_bias;
    size_t            n_out_bias;
    size_t            n_down_bias;
    size_t            n_ssm_a;
    size_t            n_ssm_alpha;
    size_t            n_ssm_beta;
    size_t            n_ssm_conv1d;
    size_t            n_ssm_dt_bias;
    size_t            n_ssm_norm;
    size_t            n_q_norm;
    size_t            n_k_norm;
    size_t            n_shortconv_conv;
    size_t            n_exp_probs_b;
    size_t            n_mla_q_a_norm;
    size_t            n_mla_kv_a_norm;
} OcLayerWeights;

/* Initialize a layer weights struct to empty. */
void oc_layer_weights_init(OcLayerWeights *lw);

/* Free all owned memory in the layer weights. Safe on NULL. */
void oc_layer_weights_free(OcLayerWeights *lw);

/* Check if the layer has attention weights loaded. */
bool oc_layer_weights_has_attention(const OcLayerWeights *lw);

/* Check if the layer has dense FFN weights loaded. */
bool oc_layer_weights_has_dense_ffn(const OcLayerWeights *lw);

/* Check if the layer has MoE weights loaded. */
bool oc_layer_weights_has_moe(const OcLayerWeights *lw);

/* Check if the layer has SSM/Mamba weights loaded. */
bool oc_layer_weights_has_ssm(const OcLayerWeights *lw);

/* Check if the layer has MLA weights loaded. */
bool oc_layer_weights_has_mla(const OcLayerWeights *lw);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_LAYER_WEIGHTS_H */
