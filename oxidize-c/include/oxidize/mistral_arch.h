/* mistral_arch.h — Mistral architecture forward pass. */
#ifndef OXIDIZE_MISTRAL_ARCH_H
#define OXIDIZE_MISTRAL_ARCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mistral model configuration. */
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

/* Owning model struct. */
typedef struct OcMistralModel {
    OcMistralConfig  config;
    OcMistralLayer  *layers;     /* heap array, length config.n_layers */
    float           *tok_emb;    /* [vocab_size, hidden_dim]            */
    float           *output_norm; /* [hidden_dim]                       */
    float           *output;     /* [vocab_size, hidden_dim]            */
    /* Session-scoped KV cache (per-layer, replaces static globals). */
    float           **kv_cache_k; /* [n_layers][cap * kv_len] */
    float           **kv_cache_v; /* [n_layers][cap * kv_len] */
    size_t           kv_cache_cap;
    size_t           kv_seq_len;
    bool             initialized;
} OcMistralModel;

/* Initialize `cfg` with Mistral-7B-v0.1 defaults. Returns OC_OK on
 * success, OC_ERR_INVALID_ARG if `cfg` is NULL. */
OcError oc_mistral_config_init(OcMistralConfig *cfg);

/* Allocate a stub Mistral model with `cfg` (or defaults if NULL). */
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
