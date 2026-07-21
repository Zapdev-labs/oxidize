/*
 * llama.h — Llama-family dense forward pass (CPU, scalar + SIMD dequant).
 *
 * Port of oxidize-core/src/model/inference.rs (the Llama/Mistral path of
 * `InferenceModel::forward_single`). Implements: token embedding lookup,
 * per-layer RMSNorm → GQA attention (RoPE, online softmax) → SwiGLU FFN,
 * final RMSNorm + lm_head. Weight matrices are read zero-copy from the
 * mmap'd GGUF; matvec dequantizes one row at a time via the SIMD-accelerated
 * `oc_quant_dequant_row`.
 *
 * Scope of the `cpu-llama-forward` feature: the standard Llama/Mistral
 * dense path. MoE (Qwen3-MoE), MLA (DeepSeek), Gemma sandwich-norm, and
 * YaRN scaling are added by later features. Tied embeddings (output.weight
 * absent) are supported.
 */
#ifndef OXIDIZE_LLAMA_H
#define OXIDIZE_LLAMA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/quant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Config (port of InferenceConfig, Llama-relevant subset) ──────────── */
typedef struct OcLlamaConfig {
    uint32_t vocab_size;
    uint32_t n_embd;        /* hidden_size                          */
    uint32_t n_layer;      /* layer_count                          */
    uint32_t n_head;       /* num_attention_heads                  */
    uint32_t n_head_kv;    /* num_key_value_heads (GQA)            */
    uint32_t n_ff;         /* intermediate_size (dense FFN)        */
    uint32_t n_ctx;        /* context_size                         */
    uint32_t head_dim;     /* n_embd / n_head (or attention.key_length) */
    uint32_t kv_head_dim;  /* usually == head_dim                  */
    uint32_t rope_dim;     /* effective rope dims (0 → full kv_head_dim) */
    float    rope_theta;
    float    rms_norm_eps;
    bool     tied_embeddings;  /* output.weight absent → tok_embeddings */
    /* MoE (Qwen3-MoE / Mixtral). When num_experts > 0, the FFN branch uses
     * the MoE path (router + top-k experts + optional shared expert). */
    uint32_t num_experts;          /* 0 = dense FFN                       */
    uint32_t num_experts_per_tok;  /* top-k (0 → 1 when num_experts>0)    */
    uint32_t expert_intermediate_size;  /* 0 → falls back to n_ff          */
    bool     expert_gating_sigmoid;     /* false = softmax (Qwen3-MoE)     */
    float    expert_weights_scale;      /* 1.0 = no routed scaling          */
    bool     uses_geglu;                /* true = Gemma GeGLU FFN (vs SwiGLU) */
    float    norm_scale;               /* RMSNorm scale factor (Gemma: sqrt(n_embd)) */
    /* YaRN long-context RoPE scaling. */
    float    yarn_factor;               /* scaling factor (0 = no YaRN)            */
    uint32_t yarn_orig_ctx;             /* original context length (for YaRN)      */
    /* DeepSeek MLA (Multi-head Latent Attention). */
    bool     uses_mla;                  /* true = DeepSeek-V2/V3 MLA attention      */
    uint32_t mla_q_lora_dim;           /* q_a_proj output dim (latent dim)         */
    uint32_t mla_kv_lora_dim;          /* kv_a_proj latent dim (without kv_pe)     */
    uint32_t mla_q_rope_dim;           /* RoPE dim for q_pe                        */
    uint32_t mla_kv_nope_head_dim;     /* per-head nope dim (k_b output / n_heads) */
    uint32_t mla_v_head_dim;           /* per-head v dim (v_b output / n_heads)    */
} OcLlamaConfig;

/* Non-owning view over a mmap'd GGUF tensor. */
typedef struct OcWeightView {
    const uint8_t *data;
    OcGgufQuantizationType qtype;
    size_t rows;
    size_t cols;
    size_t row_bytes;       /* bytes per row (quantized or f32 stride) */
} OcWeightView;

typedef struct OcLlamaLayer {
    OcWeightView attn_q, attn_k, attn_v, attn_output;
    OcWeightView ffn_gate, ffn_up, ffn_down;     /* dense FFN (when num_experts==0) */
    /* MoE (Qwen3-MoE / Mixtral). Stacked expert tensors: expert i occupies
     * bytes [i * per_expert_row_bytes, (i+1) * per_expert_row_bytes) per row.
     * `ffn_gate_exps.rows` = num_experts * expert_intermediate_size. */
    OcWeightView ffn_gate_inp;            /* router: [num_experts, n_embd]        */
    OcWeightView ffn_gate_exps;           /* [num_experts*exp_i_size, n_embd]    */
    OcWeightView ffn_up_exps;             /* [num_experts*exp_i_size, n_embd]    */
    OcWeightView ffn_down_exps;           /* [num_experts*n_embd, exp_i_size]    */
    /* Shared expert (always active, added with weight 1.0). */
    OcWeightView ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
    OcWeightView ffn_gate_inp_shexp;     /* optional sigmoid gate for shared    */
    /* DeepSeek MLA weights (used when cfg.uses_mla is true). */
    OcWeightView mla_q_a;               /* q_a_proj: [q_lora_dim, n_embd]        */
    OcWeightView mla_q_b;               /* q_b_proj: [n_heads*q_head_dim, q_lora_dim] */
    OcWeightView mla_kv_a_mqa;         /* kv_a_proj_with_mqa: [kv_lora+kv_pe, n_embd] */
    OcWeightView mla_k_b;               /* k_b_proj: [n_heads*k_nope_dim, kv_lora_dim] */
    OcWeightView mla_v_b;               /* v_b_proj: [n_heads*v_head_dim, kv_lora_dim] */
    float *mla_q_a_norm;                /* RMSNorm weight for q_a (len q_lora_dim) */
    float *mla_kv_a_norm;               /* RMSNorm weight for kv_a (len kv_lora_dim) */
    float *attn_norm;       /* owned f32, length n_embd              */
    float *ffn_norm;       /* owned f32, length n_embd              */
} OcLlamaLayer;

typedef struct OcLlamaModel {
    OcLlamaConfig    cfg;
    OcGgufMmappedFile gguf;   /* owns the mmap + per-shard arenas    */
    OcWeightView     tok_embeddings;
    OcWeightView     output;        /* aliases tok_embeddings if tied */
    float           *final_norm;    /* owned f32, length n_embd       */
    OcLlamaLayer    *layers;        /* n_layer entries                */
    OcModelArchitecture arch;
} OcLlamaModel;

/* Per-sequence KV cache + scratch workspace. One session = one sequence. */
typedef struct OcLlamaSession {
    OcLlamaModel *model;
    /* KV cache: [n_layer][n_ctx][n_head_kv * kv_head_dim], f32. */
    float *kv_k;
    float *kv_v;
    size_t kv_row_floats;     /* n_head_kv * kv_head_dim             */
    int64_t pos;              /* next position to fill               */
    /* Scratch workspace (sized to the model, owned). */
    float *x;                /* n_embd                              */
    float *normed;           /* n_embd                              */
    float *q;                /* n_head * head_dim                   */
    float *k;                /* n_head_kv * kv_head_dim              */
    float *v;                /* n_head_kv * kv_head_dim             */
    float *attn_out;         /* n_head * head_dim                   */
    float *ffn_gate;         /* n_ff (dense) or exp_i_size (MoE per-expert) */
    float *ffn_up;           /* n_ff or exp_i_size                          */
    float *dequant_temp;     /* max(n_embd, n_ff, exp_i_size)               */
    float *logits;           /* vocab_size                          */
    /* MoE temporaries. */
    float *router_logits;    /* num_experts                         */
    float *expert_gate;      /* expert_intermediate_size            */
    float *expert_up;        /* expert_intermediate_size            */
    float *expert_out;       /* n_embd                              */
    float *shexp_gate;       /* expert_intermediate_size (shared)   */
    float *shexp_up;         /* expert_intermediate_size (shared)   */
    float *shexp_out;        /* n_embd                             */
    /* MLA temporaries. */
    float *mla_c_q;          /* q_lora_dim                          */
    float *mla_c_kv;         /* kv_lora_dim                          */
    float *mla_q_full;       /* n_heads * q_head_dim                */
    float *mla_kv_compressed; /* kv_lora + kv_pe                     */
} OcLlamaSession;

/* Load a Llama-family GGUF (mmap, zero-copy weights). Returns OC_OK,
 * OC_ERR_IO, OC_ERR_FORMAT, OC_ERR_MODEL (unsupported arch / missing
 * tensors), or OC_ERR_OOM. */
OcError oc_llama_load(const char *path, OcLlamaModel *out);

/* Initialize a session with a fresh KV cache for `model`. */
OcError oc_llama_session_init(OcLlamaModel *model, OcLlamaSession *out);

/* Run one forward step: embed `token`, advance position, write logits_out
 * (length model->cfg.vocab_size). Returns OC_OK or OC_ERR_INVALID_ARG.
 * `logits_out` may be NULL to skip the lm_head projection (useful for
 * prompt prefill where only the KV cache matters). */
OcError oc_llama_forward(OcLlamaSession *sess, uint32_t token, float *logits_out);

/* Reset position to 0 (start a new sequence; KV cache is overwritten on
 * subsequent forwards). Does NOT zero the cache. */
void oc_llama_session_reset(OcLlamaSession *sess);

/* Rewind position to `pos` (for speculative decoding cache rollback).
 * KV cache entries at positions >= pos will be overwritten on subsequent
 * forwards. Does NOT zero the cache. */
void oc_llama_session_rewind(OcLlamaSession *sess, uint32_t pos);

void oc_llama_session_free(OcLlamaSession *sess);
void oc_llama_free(OcLlamaModel *model);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_LLAMA_H */
