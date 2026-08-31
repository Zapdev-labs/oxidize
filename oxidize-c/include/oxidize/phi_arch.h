/* phi_arch.h — Phi-2 / Phi-3 architecture forward pass. */
#ifndef OXIDIZE_PHI_ARCH_H
#define OXIDIZE_PHI_ARCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Phi model configuration. Defaults populated by oc_phi_config_init() */
typedef struct OcPhiConfig {
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t vocab_size;
    float    rope_theta;        /* RoPE base frequency */
} OcPhiConfig;

/* Per-layer weight pointers. */
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
