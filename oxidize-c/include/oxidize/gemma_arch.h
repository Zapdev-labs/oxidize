/*
 * gemma_arch.h — Gemma architecture forward pass.
 *
 * Gemma (Google) is structurally close to Llama/Mistral but with two
 * distinguishing features:
 *   - GeGLU activation (tanh-approximated GELU gated FFN) instead of
 *     Llama's SwiGLU (SiLU-gated FFN).
 *   - Embedding scaling factor: the token-embedding lookup is multiplied
 *     by sqrt(hidden_dim) before the first layer. This is the most
 *     commonly-forgotten Gemma-specific quirk and is captured here as
 *     `embedding_scale` (default sqrt(hidden_dim)).
 *
 * Gemma also uses RoPE (theta default 10000.0) and Grouped-Query
 * Attention (GQA). Gemma-1 uses a single KV head; Gemma-2/3/4 add
 * interleaved local/global sliding-window attention layers and
 * sandwich normalization (handled by the model loader when present).
 *
 * Port of oxidize-core/src/model/inference.rs::ModelArchitecture::Gemma
 * forward path to the C11 port. The forward function is a stub that
 * allocates a logits buffer sized by the model config, fills it with
 * zeros, and returns OC_OK.
 *
 * Weight tensor canonical names (after HF → oxidize mapping):
 *   tok_embeddings.weight
 *   output.weight (tied with embeddings in some Gemma variants)
 *   norm.weight
 *   blk.N.attn_q.weight, blk.N.attn_k.weight, blk.N.attn_v.weight,
 *   blk.N.attn_output.weight
 *   blk.N.attn_norm.weight, blk.N.ffn_norm.weight,
 *   blk.N.post_attention_norm.weight, blk.N.post_ffw_norm.weight (Gemma 2+)
 *   blk.N.ffn_gate.weight, blk.N.ffn_up.weight, blk.N.ffn_down.weight
 */
#ifndef OXIDIZE_GEMMA_ARCH_H
#define OXIDIZE_GEMMA_ARCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Gemma model configuration. Defaults populated by oc_gemma_config_init()
 * match the Gemma-2B reference:
 *   n_layers=18, n_heads=8, n_kv_heads=1, head_dim=256, hidden_dim=2048,
 *   intermediate_dim=16384, vocab_size=256000,
 *   embedding_scale=sqrt(hidden_dim), rope_theta=10000.0. */
typedef struct OcGemmaConfig {
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t vocab_size;
    float    embedding_scale;   /* multiply token embedding by this */
    float    rope_theta;        /* RoPE base frequency */
} OcGemmaConfig;

/* Per-layer weight pointers. The owning OcGemmaModel allocates these as
 * contiguous float arrays; here they are non-owning views into that
 * storage. NULL pointers are permitted until weights are loaded. */
typedef struct OcGemmaLayer {
    float *attention_norm;  /* RMSNorm before attention  ([hidden_dim])    */
    float *ffn_norm;        /* RMSNorm before FFN        ([hidden_dim])    */
    float *wq;              /* query projection          ([n_heads*head_dim, hidden_dim]) */
    float *wk;              /* key projection             ([n_kv_heads*head_dim, hidden_dim]) */
    float *wv;              /* value projection          ([n_kv_heads*head_dim, hidden_dim]) */
    float *wo;              /* output projection         ([n_heads*head_dim, hidden_dim]) */
    float *w_gate;         /* GeGLU gate                ([intermediate_dim, hidden_dim]) */
    float *w_up;           /* GeGLU up                  ([intermediate_dim, hidden_dim]) */
    float *w_down;         /* GeGLU down                ([hidden_dim, intermediate_dim]) */
} OcGemmaLayer;

/* Owning model struct. `layers` is heap-allocated (n_layers entries) and
 * freed by oc_gemma_free(). The weight buffers are owned by the model. */
typedef struct OcGemmaModel {
    OcGemmaConfig  config;
    OcGemmaLayer  *layers;       /* heap array, length config.n_layers */
    float          *tok_emb;     /* [vocab_size, hidden_dim]            */
    float          *output_norm;  /* [hidden_dim]                       */
    float          *output;      /* [vocab_size, hidden_dim]            */
    bool            initialized;
} OcGemmaModel;

/* Initialize `cfg` with Gemma-2B defaults, including
 * embedding_scale = sqrt(hidden_dim). Returns OC_OK on success,
 * OC_ERR_INVALID_ARG if `cfg` is NULL. */
OcError oc_gemma_config_init(OcGemmaConfig *cfg);

/* Allocate a stub Gemma model with `cfg` (or defaults if NULL). Weight
 * buffers are allocated and zeroed. Returns OC_OK on success. */
OcError oc_gemma_model_init(OcGemmaModel *model, const OcGemmaConfig *cfg);

/* Stub forward pass: writes zeros into `logits` (length config.vocab_size).
 * Real path: embed(token) * embedding_scale → layers (GeGLU FFN, RoPE on
 * Q/K, GQA) → RMSNorm → output projection. */
OcError oc_gemma_forward(OcGemmaModel *model, uint32_t token, float *logits);

/* Free all heap storage owned by `model` and zero the struct. Safe on
 * NULL and on uninitialized models. */
void oc_gemma_free(OcGemmaModel *model);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_GEMMA_ARCH_H */
