/* gemma_arch.h — Gemma architecture forward pass. */
#ifndef OXIDIZE_GEMMA_ARCH_H
#define OXIDIZE_GEMMA_ARCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Gemma model configuration. Defaults populated by oc_gemma_config_init() */
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
