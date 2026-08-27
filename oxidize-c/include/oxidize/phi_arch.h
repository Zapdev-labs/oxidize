/*
 * phi_arch.h — Phi-2 / Phi-3 architecture forward pass.
 *
 * Phi-2 and Phi-3 (Microsoft) are dense transformer LLMs using:
 *   - GeGLU activation (tanh-approximated GELU gated FFN).
 *   - Rotary Positional Embeddings (RoPE) on Q/K.
 *   - Dense (full multi-head) attention — no GQA/MQA; n_kv_heads is not
 *     a separate config knob (it equals n_heads).
 *
 * Phi-2 uses LayerNorm (not RMSNorm) and tied embeddings. Phi-3 uses
 * RMSNorm and may untie embeddings. The config here covers both variants
 * since the forward stub is structural.
 *
 * Port of oxidize-core/src/model/inference.rs::ModelArchitecture::Phi
 * forward path to the C11 port. The forward function is a stub that
 * allocates a logits buffer sized by the model config, fills it with
 * zeros, and returns OC_OK.
 *
 * Weight tensor canonical names (after HF → oxidize mapping):
 *   tok_embeddings.weight
 *   output.weight (tied with embeddings in Phi-2)
 *   norm.weight
 *   blk.N.attn_q.weight, blk.N.attn_k.weight, blk.N.attn_v.weight,
 *   blk.N.attn_output.weight
 *   blk.N.input_layernorm.weight, blk.N.post_attention_layernorm.weight
 *   blk.N.ffn_gate.weight, blk.N.ffn_up.weight, blk.N.ffn_down.weight
 */
#ifndef OXIDIZE_PHI_ARCH_H
#define OXIDIZE_PHI_ARCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Phi model configuration. Defaults populated by oc_phi_config_init()
 * match the Phi-2 reference:
 *   n_layers=24, n_heads=32, head_dim=80, hidden_dim=2560,
 *   intermediate_dim=10240, vocab_size=51200, rope_theta=10000.0. */
typedef struct OcPhiConfig {
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t vocab_size;
    float    rope_theta;        /* RoPE base frequency */
} OcPhiConfig;

/* Per-layer weight pointers. The owning OcPhiModel allocates these as
 * contiguous float arrays; here they are non-owning views into that
 * storage. NULL pointers are permitted until weights are loaded.
 *
 * Note: Phi uses dense attention so there is one K/V projection per
 * head (no separate n_kv_heads knob); wq/wk/wv all use
 * [n_heads * head_dim, hidden_dim]. */
typedef struct OcPhiLayer {
    float *input_norm;       /* LayerNorm (Phi-2) / RMSNorm (Phi-3) before attn */
    float *post_attn_norm;   /* LayerNorm/RMSNorm before FFN                   */
    float *wq;               /* query projection                              */
    float *wk;               /* key projection (dense: n_heads heads)         */
    float *wv;               /* value projection (dense: n_heads heads)       */
    float *wo;               /* output projection                            */
    float *w_gate;          /* GeGLU gate                                   */
    float *w_up;            /* GeGLU up                                     */
    float *w_down;          /* GeGLU down                                   */
} OcPhiLayer;

/* Owning model struct. `layers` is heap-allocated (n_layers entries) and
 * freed by oc_phi_free(). The weight buffers are owned by the model. */
typedef struct OcPhiModel {
    OcPhiConfig  config;
    OcPhiLayer  *layers;       /* heap array, length config.n_layers */
    float        *tok_emb;     /* [vocab_size, hidden_dim]            */
    float        *output_norm;  /* [hidden_dim]                       */
    float        *output;      /* [vocab_size, hidden_dim]            */
    bool          initialized;
} OcPhiModel;

/* Initialize `cfg` with Phi-2 defaults. Returns OC_OK on success,
 * OC_ERR_INVALID_ARG if `cfg` is NULL. */
OcError oc_phi_config_init(OcPhiConfig *cfg);

/* Allocate a stub Phi model with `cfg` (or defaults if NULL). Weight
 * buffers are allocated and zeroed. Returns OC_OK on success. */
OcError oc_phi_model_init(OcPhiModel *model, const OcPhiConfig *cfg);

/* Stub forward pass: writes zeros into `logits` (length config.vocab_size).
 * Real path: embed(token) → layers (GeGLU FFN, RoPE on Q/K, dense MHA)
 * → final norm → output projection. */
OcError oc_phi_forward(OcPhiModel *model, uint32_t token, float *logits);

/* Free all heap storage owned by `model` and zero the struct. Safe on
 * NULL and on uninitialized models. */
void oc_phi_free(OcPhiModel *model);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_PHI_ARCH_H */
