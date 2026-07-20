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
    uint32_t n_ff;         /* intermediate_size                    */
    uint32_t n_ctx;        /* context_size                         */
    uint32_t head_dim;     /* n_embd / n_head (or attention.key_length) */
    uint32_t kv_head_dim;  /* usually == head_dim                  */
    uint32_t rope_dim;     /* effective rope dims (0 → full kv_head_dim) */
    float    rope_theta;
    float    rms_norm_eps;
    bool     tied_embeddings;  /* output.weight absent → tok_embeddings */
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
    OcWeightView ffn_gate, ffn_up, ffn_down;
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
    float *ffn_gate;         /* n_ff                                */
    float *ffn_up;           /* n_ff                                */
    float *dequant_temp;     /* max(n_embd, n_ff)                    */
    float *logits;           /* vocab_size                          */
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

void oc_llama_session_free(OcLlamaSession *sess);
void oc_llama_free(OcLlamaModel *model);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_LLAMA_H */
