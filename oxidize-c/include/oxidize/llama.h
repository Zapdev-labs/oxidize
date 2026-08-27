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
#include "oxidize/qwen35_delta.h"

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
    uint32_t moe_layer_start;       /* first MoE layer (0 = all layers MoE) */
    bool     expert_gating_sigmoid;     /* false = softmax (Qwen3-MoE)     */
    float    expert_weights_scale;      /* 1.0 = no routed scaling          */
    bool     uses_geglu;                /* true = Gemma GeGLU FFN (vs SwiGLU) */
    float    norm_scale;               /* RMSNorm scale factor (Gemma: sqrt(n_embd)) */
    /* GPT-2/Falcon specific flags. */
    bool     uses_gpt2;                /* true = GPT-2 architecture            */
    bool     uses_par_attn;           /* true = parallel QKV (GPT-2/J/NeoX)   */
    /* Sliding window attention (Gemma2/Phi). Layers alternate between
     * global (full) and sliding-window attention. `sliding_window` is the
     * window size in tokens (0 = no sliding window). */
    uint32_t sliding_window;            /* 0 = disabled                         */
    uint32_t sliding_window_pattern;   /* 1 = all global, 2 = alternating     */
    /* ── Gemma 4 dual-geometry attention ──────────────────────────────────
     *
     * Gemma 4 does not merely alternate *window size* between layers the way
     * Gemma 2 does — sliding and global layers have genuinely different
     * attention shapes, so a single head_dim/n_head_kv cannot describe the
     * model. In gemma-4-31B-it, 50 sliding layers use head_dim 256 with 16 KV
     * heads and rope theta 1e4, while 10 global layers use head_dim 512 with
     * 4 KV heads and rope theta 1e6. The GGUF reflects this by storing
     * attention.head_count_kv as a 60-element ARRAY rather than a scalar.
     *
     * The non-suffixed fields above (head_dim, n_head_kv, rope_dim,
     * rope_theta) carry the GLOBAL-layer geometry; the _swa fields here carry
     * the sliding-layer geometry. `layer_is_swa` gives the per-layer pattern
     * directly from metadata rather than inferring it from a modulus, because
     * the pattern is data, not arithmetic.
     *
     * When uses_gemma4 is false every _swa field mirrors its base field, so
     * geometry lookups are uniform for all other architectures. */
    bool     uses_gemma4;
    uint32_t head_dim_swa;             /* sliding-layer head dim (256)         */
    uint32_t n_head_kv_swa;            /* sliding-layer KV heads (16)          */
    uint32_t rope_dim_swa;             /* sliding-layer rotary dims (256)      */
    float    rope_theta_swa;           /* sliding-layer rope base (1e4)        */
    /* Per-layer attention kind, n_layer entries: 1 = sliding, 0 = global.
     * Owned by the model (freed in oc_llama_free); NULL when uses_gemma4 is
     * false. */
    uint8_t *layer_is_swa;
    /* Softcap applied to the final logits: logits = tanh(l/c)*c. 0 = off.
     * Gemma 4 uses 30.0; skipping it measurably distorts the sampling
     * distribution, so it is correctness, not a tuning knob. */
    float    logit_softcap;
    /* Global layers ship no attn_v tensor at all — V is the same projection
     * as K (config.json `attention_k_eq_v`). The loader aliases attn_v onto
     * attn_k for those layers, and the KV cache stores one buffer instead of
     * two, halving global-layer cache footprint. */
    bool     k_eq_v;
    /* Pre-attention score scale. Gemma 4 sets this to 1.0 — it applies NO
     * 1/sqrt(head_dim) factor (llama.cpp: hparams.f_attention_scale = 1.0f,
     * "Gemma4 uses self.scaling = 1.0"). 0 means "use the usual
     * 1/sqrt(head_dim)", which is every other architecture. Getting this wrong
     * is not subtle: dividing scores by sqrt(512) flattens the softmax toward
     * uniform, so attention returns roughly the mean of V. */
    float    attn_scale;
    /* Gemma 4 RMS-normalizes V after projection with NO weight tensor —
     * a plain normalization, not a learned one. */
    bool     v_rms_norm;
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
    /* LongCat (ScMoE + MLA + n-gram over-embedding).
     *
     * LongCat packs TWO attention+FFN sub-blocks into each GGUF `blk.N`, so
     * `n_layer` is 2 * longcat.block_count and per-sub-block tensors carry a
     * `_0`/`_1` suffix. Router, expert pool and indexer appear once per
     * `blk.N` and attach to the even sub-layer 2N. */
    bool     is_longcat;
    uint32_t zero_expert_count;        /* identity experts appended after routed  */
    /* deepseek_yarn parameters. Absent from GGUF metadata — LongCat's
     * config.json sets both mscale terms to 1, which makes
     * the RoPE attn factor exactly 1.0 and folds a 1.4787^2 term into the
     * softmax scale instead. See oc_rope_deepseek_yarn_scales(). */
    float    yarn_mscale;
    float    yarn_mscale_all_dim;
    /* n-gram over-embedding: (neighbor_num - 1) * split_num tables. */
    uint32_t ngram_n_grams;
    uint32_t ngram_split_num;
    /* Qwen3.5 hybrid attention. block_count includes optional trailing MTP
     * blocks; n_layer is the executable transformer-layer count. */
    bool     is_qwen35;
    uint32_t nextn_predict_layers;
    uint32_t full_attention_interval;
    uint32_t n_full_attention_layers;
    uint32_t n_recurrent_layers;
    uint32_t ssm_conv_kernel;
    uint32_t ssm_state_size;
    uint32_t ssm_group_count;
    uint32_t ssm_value_heads;
    uint32_t ssm_inner_size;
    uint32_t shared_expert_intermediate_size;
    /* ── Muse Glimmer ─────────────────────────────────────────────────────
     *
     * Meta's Muse-Glimmer-30B (GGUF arch "muse-glimmer") is a dense
     * transformer that reuses pieces already present here — Gemma's sandwich
     * norms, Gemma 4's per-layer sliding-window pattern, Gemma's per-head
     * QK-norm, Gemma's final-logit softcap — plus four things nothing else
     * needed. They are expressed as behaviour flags rather than an
     * `is_muse` switch so each one reads as what it does. */
    /* The token embedding is RMS-normalized (no weight) before layer 0.
     * llama.cpp: build_norm(inpL, nullptr, nullptr, LLM_NORM_RMS, -1). */
    bool     embd_rms_norm;
    /* sigmoid(W_gate · attn_norm(x)) gates the attention output elementwise
     * before the output projection. The gate is n_head*head_dim wide and is
     * computed from the PRE-attention normed hidden state, not from the
     * attention output. */
    bool     attn_out_gate;
    /* RoPE runs only on sliding-window layers; global layers are NoPE.
     * Resolved per layer into OcLlamaLayer.use_rope. */
    bool     rope_swa_only;
    /* The sandwich (post-attention / post-FFN) norms use their own epsilon,
     * 1e-8, not attention.layer_norm_rms_epsilon. 0 → use rms_norm_eps,
     * which is what every other architecture wants. */
    float    post_norm_eps;
    /* Final logits are multiplied by this before the softcap. 0 = 1.0. */
    float    logit_scale;
    /* RoPE pairs dimensions (2i, 2i+1) instead of (i, i + rope_dim/2) —
     * llama.cpp's LLAMA_ROPE_TYPE_NORM. The two conventions assign different
     * frequencies to the same dimension, so this is correctness, not style:
     * with the wrong one the model stays locally fluent and drifts within a
     * couple of dozen tokens. */
    bool     rope_norm_pairs;
} OcLlamaConfig;

/* Upper bound on LongCat n-gram tables. LongCat-2.0 has
 * (neighbor_num - 1) * split_num = (5-1)*4 = 16. */
#define OC_LONGCAT_MAX_NGRAM 32

/* Non-owning view over a mmap'd GGUF tensor. */
typedef struct OcWeightView {
    const uint8_t *data;
    OcGgufQuantizationType qtype;
    size_t rows;
    size_t cols;
    size_t row_bytes;       /* bytes per row (quantized or f32 stride) */
} OcWeightView;

typedef enum OcLlamaLayerKind {
    OC_LLAMA_LAYER_FULL_ATTENTION = 0,
    OC_LLAMA_LAYER_QWEN35_RECURRENT = 1,
} OcLlamaLayerKind;

typedef struct OcLlamaLayer {
    OcLlamaLayerKind kind;
    uint32_t state_index;
    uint32_t kv_cache_index;
    OcWeightView attn_q, attn_k, attn_v, attn_output;
    OcWeightView attn_gate, attn_qkv;
    OcWeightView ssm_a, ssm_alpha, ssm_beta, ssm_conv1d;
    OcWeightView ssm_dt_bias, ssm_norm, ssm_out;
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
    float *mla_q_a_norm;                /* owned f32, length q_lora_dim */
    float *mla_kv_a_norm;               /* owned f32, length kv_lora_dim */
    /* LongCat router bias (`blk.N.exp_probs_b.bias`), owned f32 of length
     * num_experts + zero_expert_count. Added to the router logits for
     * SELECTION only — the gate weight applied to an expert's output is the
     * unbiased probability. NULL when absent. */
    float *exp_probs_b;
    float *attn_norm;       /* owned f32, length n_embd              */
    float *ffn_norm;       /* owned f32, length n_embd              */
    /* LayerNorm biases (beta) for GPT-2/NeoX/Falcon; NULL for RMSNorm
     * architectures (Llama-family has no norm bias). Owned f32, n_embd. */
    float *attn_norm_bias;
    float *ffn_norm_bias;
    /* Attention Q/K/V projection biases. Present in Qwen2-family models
     * (and any arch whose GGUF carries attn_{q,k,v}.bias); NULL otherwise.
     * Owned f32, lengths n_head*head_dim / n_head_kv*kv_head_dim. */
    float *attn_q_bias;
    float *attn_k_bias;
    float *attn_v_bias;
    /* ── Gemma-family extra norms (all owned f32, NULL when absent) ────────
     *
     * Gemma applies RMSNorm to Q and K per head *after* projection and before
     * RoPE (lengths are the layer's own head_dim — 256 on sliding layers, 512
     * on global ones, so these are not interchangeable between layers), and
     * wraps each residual branch in a second "sandwich" norm applied to the
     * branch output before it is added back. */
    float *attn_q_norm;         /* length = this layer's head_dim   */
    float *attn_k_norm;         /* length = this layer's head_dim   */
    float *post_attention_norm; /* length n_embd, applied to attn branch out */
    float *post_ffw_norm;       /* length n_embd, applied to FFN branch out  */
    /* Gemma 4 per-layer output scale (blk.N.layer_output_scale.weight, a
     * single f32). Multiplies the layer's contribution; 1.0 when absent. */
    float  layer_output_scale;
    /* Resolved geometry for this layer, filled by the loader so the forward
     * passes never re-derive it from the layer index. For non-Gemma-4 models
     * every layer gets the model-wide values. */
    uint32_t head_dim;      /* per-head query/key width               */
    uint32_t n_head_kv;     /* KV heads in this layer                 */
    uint32_t rope_dim;      /* rotary dims in this layer              */
    float    rope_theta;    /* rope base in this layer                */
    uint32_t sliding_window;/* 0 = global (full) attention            */
    /* Whether RoPE applies in this layer. False only on Muse Glimmer's
     * global (NoPE) layers; true everywhere else. */
    bool     use_rope;
} OcLlamaLayer;

#define OC_MTP_DEFAULT_DRAFT 3u

typedef struct OcLlamaMtp {
    bool present;
    OcLlamaLayer layer;
    OcWeightView eh_proj;
    OcWeightView embed_tokens;
    OcWeightView shared_head_head;
    float *enorm;
    float *hnorm;
    float *shared_head_norm;
} OcLlamaMtp;

typedef struct OcLlamaModel {
    OcLlamaConfig    cfg;
    OcGgufMmappedFile gguf;   /* owns the mmap + per-shard arenas    */
    OcWeightView     tok_embeddings;
    OcWeightView     output;        /* aliases tok_embeddings if tied */
    float           *final_norm;    /* owned f32, length n_embd       */
    float           *final_norm_bias; /* LayerNorm beta (GPT-2/NeoX/Falcon),
                                       * NULL for RMSNorm archs. Owned.  */
    OcLlamaLayer    *layers;        /* n_layer entries                */
    OcLlamaMtp       mtp;
    OcModelArchitecture arch;
    /* GPT-2 learned positional embeddings (wpe), resolved lazily on first
     * forward. Per-model (views into this model's mmap). */
    OcWeightView     gpt2_pos_embed;
    bool             gpt2_pos_resolved;
    /* LongCat n-gram over-embedding. `cfg.ngram_n_grams` tables (16 for
     * LongCat-2.0), each a hashed embedding of width mla_kv_lora_dim (512)
     * projected back up to n_embd. Row counts differ per table and are read
     * from each tensor's own shape, never recomputed. Unused views are
     * zeroed. */
    OcWeightView     ngram_embd[OC_LONGCAT_MAX_NGRAM];
    OcWeightView     ngram_proj[OC_LONGCAT_MAX_NGRAM];
} OcLlamaModel;

/* KV cache element type.
 *
 * The cache is allocated for the full n_ctx up front, so on long-context
 * models it can dwarf the weights — a 128k-context model can want tens of GB
 * of f32 KV. Q8 stores one int8 per element plus one f32 scale per
 * (layer, position, kv head), cutting that ~4x for a quantization error far
 * below the model's own weight quantization.
 *
 * F32 is the default so existing behaviour is bit-for-bit unchanged; select
 * Q8 explicitly with oc_llama_session_init_kv(), or set OX_KV_TYPE=q8. */
typedef enum {
    OC_KV_F32 = 0,
    OC_KV_Q8  = 1,
} OcKvCacheType;

/* Per-sequence KV cache + scratch workspace. One session = one sequence. */
typedef struct OcLlamaSession {
    OcLlamaModel *model;
    /* KV cache: [n_layer][n_ctx][n_head_kv * kv_head_dim].
     * kv_type == OC_KV_F32: kv_k/kv_v hold f32 values; the quantized code
     *   and scale pointers are NULL.
     * kv_type == OC_KV_Q8:  kv_k_q/kv_v_q hold int8 codes with the same
     *   indexing, kv_k_scale/kv_v_scale hold one f32 per
     *   (layer, pos, kv_head), and kv_k/kv_v are NULL. */
    float *kv_k;
    float *kv_v;
    OcKvCacheType kv_type;
    int8_t *kv_k_q;
    int8_t *kv_v_q;
    float  *kv_k_scale;      /* [n_layer][n_ctx][n_head_kv]         */
    float  *kv_v_scale;
    size_t kv_group;         /* elements per scale (kv_head_dim)    */
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
    float *expert_gate_all;
    float *expert_up_all;
    float *expert_down_all;
    uint32_t *selected_experts;
    float *shexp_gate;       /* expert_intermediate_size (shared)   */
    float *shexp_up;         /* expert_intermediate_size (shared)   */
    float *shexp_out;        /* n_embd                             */
    /* MLA temporaries. */
    float *mla_c_q;          /* q_lora_dim                          */
    float *mla_c_kv;         /* kv_lora_dim                          */
    float *mla_q_full;       /* n_heads * q_head_dim                */
    float *mla_kv_compressed; /* kv_lora + kv_pe                     */
    /* Per-head, held concurrently so the cache can be swept once for all
     * heads rather than once per head. n_head * kv_lora_dim each. */
    float *mla_q_absorbed;   /* k_b[h]^T @ q_nope, per head          */
    float *mla_ctx_latent;   /* attention-weighted c_kv, per head    */
    float *mla_run_max;      /* n_head online-softmax running max    */
    float *mla_run_sum;      /* n_head online-softmax running sum    */
    /* Qwen3.5 persistent state. Entries correspond to model layers; only
     * recurrent entries are bound to the contiguous backing stores. */
    OcQwen35DeltaState *qwen35_delta;
    float *qwen35_conv_state;
    float *qwen35_recurrent_state;
    float *qwen35_qkv;
    float *qwen35_gate;
    float *qwen35_beta;
    float *qwen35_alpha;
    float *qwen35_conv_output;
    float *qwen35_delta_output;
    /* Muse Glimmer attention-output gate, n_head * head_dim. Allocated only
     * when cfg.attn_out_gate is set. */
    float *muse_gate;
    float *last_hidden;
    float *mtp_hidden;
    float *mtp_concat;
    uint32_t last_token;
    uint32_t mtp_pos;
} OcLlamaSession;

/* ─── Batched decode ─────────────────────────────────────────────────────
 *
 * Processes multiple sequences in a single forward pass. Each sequence has
 * its own KV cache view and position, but they share the weight matrices
 * (better cache locality than sequential processing).
 */

#define OC_MAX_BATCH_SEQ 16

typedef struct OcBatchSeq {
    uint32_t token;       /* input token for this step             */
    int64_t  pos;        /* current position in sequence          */
    float   *logits;     /* output logits (vocab_size per seq)    */
    bool     active;     /* true if this sequence needs forward   */
} OcBatchSeq;

typedef struct OcBatchSession {
    OcLlamaModel *model;
    /* KV cache: [n_layer][n_ctx][n_head_kv * kv_head_dim * OC_MAX_BATCH_SEQ]. */
    float *kv_k;
    float *kv_v;
    size_t kv_row_floats;
    size_t max_seqs;
    /* Per-sequence positions. */
    int64_t pos[OC_MAX_BATCH_SEQ];
    bool active[OC_MAX_BATCH_SEQ];
    /* Shared workspace (reused across sequences in the batch). */
    float *x;
    float *normed;
    float *q;
    float *k;
    float *v;
    float *attn_out;
    float *ffn_gate;
    float *ffn_up;
    float *dequant_temp;
    /* MoE temporaries (shared). */
    float *router_logits;
    float *expert_gate;
    float *expert_up;
    float *expert_out;
    float *shexp_gate;
    float *shexp_up;
    float *shexp_out;
    /* MLA temporaries (shared; NULL unless model->cfg.uses_mla). */
    float *mla_c_q;
    float *mla_c_kv;
    float *mla_q_full;
    float *mla_kv_compressed;
    float *mla_q_absorbed;
    float *mla_ctx_latent;
    float *mla_run_max;
    float *mla_run_sum;
} OcBatchSession;

OcError oc_batch_session_init(OcLlamaModel *model, size_t max_seqs,
                               OcBatchSession *out);

/* Run one forward step for each active sequence in the batch.
 * `seqs` is an array of `max_seqs` entries. Only sequences with
 * `active=true` are processed; their `logits` will be filled. */
OcError oc_batch_forward(OcBatchSession *bs, OcBatchSeq *seqs);

void oc_batch_session_free(OcBatchSession *bs);

/* Load a Llama-family GGUF (mmap, zero-copy weights). Returns OC_OK,
 * OC_ERR_IO, OC_ERR_FORMAT, OC_ERR_MODEL (unsupported arch / missing
 * tensors), or OC_ERR_OOM. */
OcError oc_llama_load(const char *path, OcLlamaModel *out);

/* Initialize a session with a fresh KV cache for `model`. */
/* Initialize a session with an f32 KV cache, unless OX_KV_TYPE=q8 is set in
 * the environment. Equivalent to oc_llama_session_init_kv() with the
 * environment-derived default. */
OcError oc_llama_session_init(OcLlamaModel *model, OcLlamaSession *out);

/* Initialize a session with an explicit KV cache type. Q8 is refused (falls
 * back to f32) when head_dim != kv_head_dim, since the quantization group and
 * the attention read stride must agree. */
OcError oc_llama_session_init_kv(OcLlamaModel *model, OcLlamaSession *out,
                                 OcKvCacheType kv_type);

/* Resolve KV type from `--kv` / OX_KV_TYPE / context length.
 * explicit: "q8", "f32", or NULL. Contexts >= 8192 default to Q8. */
OcKvCacheType oc_llama_select_kv_type(uint32_t n_ctx,
                                      const char *explicit_value);

/* Bytes the KV cache occupies for `model` under `kv_type`. Useful for
 * reporting and for deciding whether a context length is affordable. */
size_t oc_llama_kv_cache_bytes(const OcLlamaModel *model, OcKvCacheType kv_type);

/* Run one forward step: embed `token`, advance position, write logits_out
 * (length model->cfg.vocab_size). Returns OC_OK or OC_ERR_INVALID_ARG.
 * `logits_out` may be NULL to skip the lm_head projection (useful for
 * prompt prefill where only the KV cache matters). */
OcError oc_llama_forward(OcLlamaSession *sess, uint32_t token, float *logits_out);

bool oc_llama_mtp_present(const OcLlamaModel *model);

/* Greedy MTP: emit 1 + accepted drafts. Updates session and logits. */
OcError oc_llama_mtp_greedy_advance(OcLlamaSession *sess, float *logits,
                                    uint32_t *out_tokens, size_t max_out,
                                    size_t *n_out);

/* Draft up to `k` MTP tokens from last_token/last_hidden. Does not touch the
 * backbone KV. Writes pairwise confidence per draft into out_conf (nullable). */
OcError oc_llama_mtp_draft_tokens(OcLlamaSession *sess, uint32_t k,
                                  uint32_t *out_tokens, float *out_conf,
                                  uint32_t *n_out);

/* Prefill a whole prompt, processing `chunk` tokens per pass so they share
 * one sweep over the weights instead of one sweep each.
 *
 * Equivalent to calling oc_llama_forward() for tokens[0..n_tokens), with
 * `logits_out` receiving the LAST token's logits (NULL to skip the lm_head).
 * KV cache entries are written for positions sess->pos .. sess->pos+n-1 and
 * the position advances by n_tokens.
 *
 * `chunk` of 0 selects the default. Models the batched path does not cover
 * (MLA, LongCat, Gemma 4, and the LayerNorm architectures with their own
 * forward passes) fall back to the per-token loop transparently, as does a
 * scratch allocation failure — the result is the same either way.
 *
 * Numerics: every matmul quantizes each activation exactly as the
 * single-vector path does (tiling changes only how many are in flight), so
 * dense models come out bit-identical. MoE models differ at float-rounding
 * level and only there: grouping tokens by expert means a token's expert
 * contributions are summed in expert-index order rather than in descending
 * gate order, and float addition is not associative. */
OcError oc_llama_prefill(OcLlamaSession *sess, const uint32_t *tokens,
                         size_t n_tokens, size_t chunk, float *logits_out);

/* Copy the already-prefilled prefix state from `src` into `dst`. Both
 * sessions must belong to the same model and use the same KV type. */
OcError oc_llama_session_copy_prefix(OcLlamaSession *dst,
                                     const OcLlamaSession *src);

/* Reset position to 0 (start a new sequence; KV cache is overwritten on
 * subsequent forwards). Does NOT zero the cache. */
void oc_llama_session_reset(OcLlamaSession *sess);

/* Rewind position to `pos` (for speculative decoding cache rollback).
 * KV cache entries at positions >= pos will be overwritten on subsequent
 * forwards. Does NOT zero the cache.
 *
 * Only the KV position is restored. Recurrent state (Qwen3.5 DeltaNet conv
 * and recurrent matrices) has already absorbed every token stepped through,
 * and no snapshot of it is kept, so a rewind cannot undo it. That is why
 * oc_speculative_generate() refuses is_qwen35 models outright. */
void oc_llama_session_rewind(OcLlamaSession *sess, uint32_t pos);

void oc_llama_session_free(OcLlamaSession *sess);
void oc_llama_free(OcLlamaModel *model);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_LLAMA_H */
