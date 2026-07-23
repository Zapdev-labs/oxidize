/*
 * mistral_arch.h — Mistral architecture forward pass.
 *
 * Mistral uses SwiGLU activation (SiLU-gated FFN, same as Llama), Rotary
 * Positional Embeddings (RoPE) on Q/K, Grouped-Query Attention (GQA, where
 * n_kv_heads may be < n_heads), and Sliding Window Attention (SWA) with a
 * configurable sliding_window (default 4096). The architecture is
 * structurally close to Llama/Mistral dense in oxidize-core; the dedicated
 * module exists to surface the SWA + GQA configuration knobs and to mirror
 * the per-architecture module layout used for GLM, Gemma, and Phi.
 *
 * Port of oxidize-core/src/model/inference.rs::ModelArchitecture::Mistral
 * forward path to the C11 port. The forward function is a stub that
 * allocates a logits buffer sized by the model config, fills it with zeros,
 * and returns OC_OK. Real weight loading + token generation is wired up
 * later via oc_model_arch_from_str + the GGUF loader.
 *
 * Weight tensor canonical names (after HF → oxidize mapping, same as
 * Llama):
 *   tok_embeddings.weight
 *   output.weight
 *   norm.weight
 *   blk.N.attn_q.weight, blk.N.attn_k.weight, blk.N.attn_v.weight,
 *   blk.N.attn_output.weight
 *   blk.N.attn_norm.weight, blk.N.ffn_norm.weight
 *   blk.N.ffn_gate.weight, blk.N.ffn_up.weight, blk.N.ffn_down.weight
 */
#ifndef OXIDIZE_MISTRAL_ARCH_H
#define OXIDIZE_MISTRAL_ARCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mistral model configuration. Default values populated by
 * oc_mistral_config_init() match the Mistral-7B-v0.1 reference:
 *   n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, hidden_dim=4096,
 *   intermediate_dim=14336, vocab_size=32000, sliding_window=4096,
 *   rope_theta=10000.0, max_position=32768. */
typedef struct OcMistralConfig {
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t vocab_size;
    uint32_t sliding_window;   /* SWA window (0 = full attention) */
    float    rope_theta;       /* RoPE base frequency */
    uint32_t max_position;     /* n_ctx */
} OcMistralConfig;

/* Per-layer weight pointers. The owning OcMistralModel allocates these as
 * contiguous float arrays; here they are non-owning views into that
 * storage. NULL pointers are permitted until weights are loaded. */
typedef struct OcMistralLayer {
    float *attention_norm;  /* RMSNorm before attention  ([hidden_dim])   */
    float *ffn_norm;        /* RMSNorm before FFN        ([hidden_dim])   */
    float *wq;              /* query projection          ([n_heads*head_dim, hidden_dim]) */
    float *wk;              /* key projection             ([n_kv_heads*head_dim, hidden_dim]) */
    float *wv;              /* value projection          ([n_kv_heads*head_dim, hidden_dim]) */
    float *wo;              /* output projection         ([n_heads*head_dim, hidden_dim]) */
    float *w_gate;         /* SwiGLU gate                ([intermediate_dim, hidden_dim]) */
    float *w_up;           /* SwiGLU up                  ([intermediate_dim, hidden_dim]) */
    float *w_down;         /* SwiGLU down                ([hidden_dim, intermediate_dim]) */
} OcMistralLayer;

/* Owning model struct. `layers` is heap-allocated (n_layers entries) and
 * freed by oc_mistral_free(). The weight buffers (tok_emb, output_norm,
 * output, and per-layer weights) are owned by the model and freed together
 * with the struct. */
typedef struct OcMistralModel {
    OcMistralConfig  config;
    OcMistralLayer  *layers;     /* heap array, length config.n_layers */
    float           *tok_emb;    /* [vocab_size, hidden_dim]            */
    float           *output_norm; /* [hidden_dim]                       */
    float           *output;     /* [vocab_size, hidden_dim]            */
    bool             initialized;
} OcMistralModel;

/* Initialize `cfg` with Mistral-7B-v0.1 defaults. Returns OC_OK on
 * success, OC_ERR_INVALID_ARG if `cfg` is NULL. */
OcError oc_mistral_config_init(OcMistralConfig *cfg);

/* Allocate a stub Mistral model with `cfg` (or defaults if NULL).
 * Weight buffers are allocated and zeroed; layer pointers are wired
 * into the allocated storage so the struct is immediately usable as
 * a scaffold for the real forward pass. Returns OC_OK on success,
 * OC_ERR_OOM on allocation failure, OC_ERR_INVALID_ARG on bad args. */
OcError oc_mistral_model_init(OcMistralModel *model, const OcMistralConfig *cfg);

/* Stub forward pass: writes zeros into `logits` (length config.vocab_size).
 * `logits` must be a caller-allocated buffer of at least
 * `model->config.vocab_size` floats. Returns OC_OK on success. */
OcError oc_mistral_forward(OcMistralModel *model, uint32_t token, float *logits);

/* Free all heap storage owned by `model` and zero the struct. Safe on
 * NULL and on uninitialized models. */
void oc_mistral_free(OcMistralModel *model);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MISTRAL_ARCH_H */
