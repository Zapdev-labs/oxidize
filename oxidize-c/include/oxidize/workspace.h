#ifndef OXIDIZE_WORKSPACE_H
#define OXIDIZE_WORKSPACE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/inference.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Persistent activation across layers. */
    float    *x;                /* [hidden_size] */
    /* Hidden-size scratch. */
    float    *hidden_a;         /* [hidden_size] */
    float    *hidden_b;         /* [hidden_size] */
    /* Intermediate-size scratch (gate, up, SwiGLU). */
    float    *intermediate_a;   /* [max(inter, expert_inter)] */
    float    *intermediate_b;
    float    *intermediate_c;
    /* Q/K/V projection scratch. */
    float    *q_full;           /* [max_qkv] */
    float    *k_vec;            /* [kv_scratch] */
    float    *v_vec;
    /* Attention result scratch. */
    float    *attn_result;      /* [max_qkv] */
    /* Flash attention Q scratch. */
    float    *flash_q;          /* [max(max_qkv, 64)] */
    /* Per-head scratch. */
    float    *head_scratch;     /* [head_dim] */
    /* KV cache copy fallback buffers. */
    float    *kv_keys_copy;     /* [ctx * max_kv_len] */
    float    *kv_values_copy;
    /* Final logits buffer. */
    float    *logits;           /* [vocab_size] */
    /* MoE router scratch. */
    float    *moe_router_logits;  /* [n_experts] -- spans zero-expert slots */
    /* [n_experts] routing scratch. Previously this aliased moe_gate_all,
     * which is sized for the FFN and can be far SMALLER than the router on
     * narrow-expert models. Owned separately so the two cannot collide. */
    struct OcExpertScore *moe_expert_scores;
    /* Mamba/SSM scratch. */
    float    *mamba_scratch;   /* [max(hidden, 576)] */
    float    *conv_out;         /* [max_qkv] */
    /* LFM2 short-convolution scratch. */
    float    *shortconv_bcx;    /* [hidden * 3] */
    float    *shortconv_bx;     /* [hidden] */
    /* Batched MoE expert outputs. */
    float    *moe_gate_all;     /* [n_experts_per_tok * expert_inter] */
    float    *moe_up_all;
    float    *moe_down_all;     /* [n_experts_per_tok * hidden] */

    /* Sizes (for deallocation and bounds checking). */
    size_t    hidden_size;
    size_t    intermediate_size;
    size_t    max_qkv;
    size_t    kv_scratch;
    size_t    head_dim;
    size_t    kv_copy_size;
    size_t    vocab_size;
    size_t    n_experts;
    size_t    n_experts_per_tok;
    size_t    expert_inter;
} OcWorkspace;

/* Allocate a workspace sized for the given config. */
OcError oc_workspace_for_config(OcWorkspace *ws, const OcInferenceConfig *cfg);

/* Free all workspace buffers. Safe on NULL. */
void oc_workspace_free(OcWorkspace *ws);

/* Zero all workspace buffers (called between forward passes). */
void oc_workspace_zero(OcWorkspace *ws);

/* Total allocated bytes. */
size_t oc_workspace_size_bytes(const OcWorkspace *ws);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_WORKSPACE_H */
