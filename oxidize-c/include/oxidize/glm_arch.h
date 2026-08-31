/*
 * glm_arch.h — GLM (ChatGLM/Zhipu) and Hunyuan architecture support.
 *
 * Configuration parsing and registry entries for the GLM-4 / ChatGLM and
 * Hunyuan-MoE families in the C11 port. Both share structural elements
 * with Llama/Mistral (RMSNorm, RoPE, SwiGLU, GQA); inference runs
 * through the shared llama.c session paths, so this header only carries
 * the family-specific configuration surface:
 *
 *   GLM-4 / ChatGLM:
 *     - RMSNorm (not LayerNorm) pre-attention + pre-FFN.
 *     - RoPE on Q and K (GLM-4 uses the standard split-halves NeoX layout;
 *       ChatGLM-6B uses the interleaved GPT-J layout, selected via
 *       `glm_version`).
 *     - SwiGLU FFN (silu-gated, same as Llama).
 *     - Grouped-Query Attention (GQA): n_head_kv may be < n_head.
 *     - GLM-4 uses a unified RoPE with a larger theta (500000.0 for GLM-4-9B)
 *       and supports partial RoPE on a subset of head dims.
 *     - Tied embeddings are common in ChatGLM-6B / GLM-4.
 *     - GLM-4 adds a "qk_norm" step: RMSNorm is applied to Q and K after
 *       the QKV projection (before RoPE) to stabilize training. This is
 *       controlled by `apply_qk_norm`.
 *
 *   Hunyuan (Hunyuan-MoE / Hunyuan-Large):
 *     - MoE-based architecture with top-k expert routing (same as
 *       Mixtral/DeepSeek-MoE).
 *     - RMSNorm, SwiGLU FFN per expert, GQA.
 *     - MLA (Multi-head Latent Attention) support: when `uses_mla` is true,
 *       the attention uses the DeepSeek-V2/V3 latent compression scheme
 *       (q_a → q_b, kv_a → k_b/v_b with a shared compressed KV cache).
 *     - Shared expert (always active) similar to Qwen3-MoE.
 *     - RoPE with configurable theta; interleaved layout is NOT used
 *       (standard NeoX split-halves).
 *
 * The config structs below are populated from GGUF metadata keys (the
 * `glm.` and `hunyuan.` prefixes used by the GGUF converter) and feed
 * the arch registry / inspect tooling.
 *
 * Weight tensor canonical names (after HF → oxidize mapping):
 *   GLM-4:
 *     blk.N.attn_q.weight, blk.N.attn_k.weight, blk.N.attn_v.weight,
 *     blk.N.attn_output.weight,
 *     blk.N.attn_q_norm.weight (qk_norm, GLM-4),
 *     blk.N.attn_k_norm.weight (qk_norm, GLM-4),
 *     blk.N.ffn_gate.weight, blk.N.ffn_up.weight, blk.N.ffn_down.weight,
 *     blk.N.attn_norm.weight, blk.N.ffn_norm.weight
 *   Hunyuan:
 *     (same attention tensors) +
 *     blk.N.ffn_gate_inp.weight (router: [n_routed_experts, n_embd]),
 *     blk.N.ffn_gate.<M>.weight, blk.N.ffn_up.<M>.weight,
 *     blk.N.ffn_down.<M>.weight (per expert M),
 *     blk.N.ffn_gate_shexp.weight, blk.N.ffn_up_shexp.weight,
 *     blk.N.ffn_down_shexp.weight (shared expert)
 *     (MLA tensors when uses_mla: blk.N.attn_q_a.weight, etc.)
 */
#ifndef OXIDIZE_GLM_ARCH_H
#define OXIDIZE_GLM_ARCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/llama.h"   /* OcLlamaSession, OcWeightView, OcLlamaModel */
#include "oxidize/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── GLM version ────────────────────────────────────────────────────────
 *
 * Differentiates the ChatGLM-6B (interleaved GPT-J RoPE, no qk_norm,
 * 2-layer RMSNorm epsilon) from the GLM-4 (NeoX split-halves RoPE, qk_norm,
 * larger rope_theta, partial RoPE). The version is detected from the GGUF
 * architecture string: "glm" → ChatGLM (v1); "glm4" → GLM-4 (v4). The
 * MoE/DSA variants are routed to the Hunyuan or DeepSeek forward paths. */
typedef enum {
    OC_GLM_VERSION_UNKNOWN = 0,   /* not a GLM model                         */
    OC_GLM_VERSION_1       = 1,   /* ChatGLM-6B / GLM-130B                   */
    OC_GLM_VERSION_2       = 2,   /* ChatGLM2-6B                              */
    OC_GLM_VERSION_3       = 3,   /* ChatGLM3-6B                              */
    OC_GLM_VERSION_4       = 4,   /* GLM-4 (incl. GLM-4-9B, GLM-4-32B)        */
} OcGlmVersion;

/* ─── GLM config ──────────────────────────────────────────────────────────
 *
 * Populated from GGUF metadata keys under the `glm.` (or `chatglm.`)
 * namespace. Fields mirror the HuggingFace GLM config.json.
 *
 * GGUF key → struct field mapping:
 *   glm.vocab_size                  → vocab_size
 *   glm.hidden_size                 → hidden_size (n_embd)
 *   glm.num_layers                   → n_layer
 *   glm.num_attention_heads          → num_attention_heads (n_head)
 *   glm.num_key_value_heads          → num_kv_heads (n_head_kv)
 *   glm.intermediate_size            → intermediate_size (n_ff)
 *   glm.max_position_embeddings      → max_position_embeddings (n_ctx)
 *   glm.rope_theta                   → rope_theta
 *   glm.rms_norm_eps                 → rms_norm_eps
 *   glm.attention.key_length         → head_dim
 *   glm.rope.dimension_count         → rope_dim
 *   glm.attention.qkv_bias           → uses_qkv_bias (not stored; structural)
 *   glm.apply_qk_norm                 → apply_qk_norm (GLM-4)
 *   glm.tied_word_embeddings          → tied_embeddings
 */
typedef struct OcGlmConfig {
    uint32_t    vocab_size;              /* vocabulary size                    */
    uint32_t    hidden_size;             /* n_embd                              */
    uint32_t    n_layer;                 /* number of transformer layers       */
    uint32_t    num_attention_heads;     /* n_head                              */
    uint32_t    num_kv_heads;            /* n_head_kv (GQA); == n_head if MQA   */
    uint32_t    intermediate_size;       /* n_ff (SwiGLU FFN intermediate)      */
    uint32_t    max_position_embeddings; /* n_ctx                               */
    uint32_t    head_dim;                /* per-head dim (n_embd / n_head)      */
    uint32_t    kv_head_dim;             /* per kv-head dim (usually == head_dim)*/
    uint32_t    rope_dim;                /* RoPE dims (0 → full kv_head_dim)     */
    float       rope_theta;              /* RoPE base frequency                 */
    float       rms_norm_eps;            /* RMSNorm epsilon                     */
    bool        uses_mla;                /* true = MLA (Multi-head Latent Attn) */
    bool        apply_qk_norm;           /* true = RMSNorm on Q/K after proj    */
    bool        uses_interleaved_rope;   /* true = GPT-J interleaved (ChatGLM-6B)*/
    bool        tied_embeddings;         /* true = output.weight == tok_emb     */
    OcGlmVersion glm_version;            /* detected GLM version                */
} OcGlmConfig;

/* ─── Hunyuan config ──────────────────────────────────────────────────────
 *
 * Populated from GGUF metadata keys under the `hunyuan.` namespace.
 *
 * GGUF key → struct field mapping:
 *   hunyuan.vocab_size               → vocab_size
 *   hunyuan.hidden_size              → hidden_size
 *   hunyuan.num_layers                → n_layer
 *   hunyuan.num_attention_heads       → num_attention_heads
 *   hunyuan.num_key_value_heads       → num_kv_heads
 *   hunyuan.intermediate_size         → intermediate_size (dense FFN, layer 0)
 *   hunyuan.max_position_embeddings   → max_position_embeddings
 *   hunyuan.rope_theta                → rope_theta
 *   hunyuan.rms_norm_eps              → rms_norm_eps
 *   hunyuan.num_experts               → n_routed_experts
 *   hunyuan.num_experts_per_tok       → n_active_experts
 *   hunyuan.expert_intermediate_size  → expert_intermediate_size
 *   hunyuan.moe_layer_start           → moe_layer_start (0-indexed)
 *   hunyuan.attention.q_lora_rank     → mla_q_lora_dim
 *   hunyuan.attention.kv_lora_rank    → mla_kv_lora_dim
 *   hunyuan.attention.key_length      → mla_q_head_dim
 *   hunyuan.attention.key_length_rope → mla_q_rope_dim
 *   hunyuan.attention.value_length    → mla_v_head_dim
 *   hunyuan.use_mla                   → uses_mla
 */
typedef struct OcHunyuanConfig {
    uint32_t    vocab_size;              /* vocabulary size                    */
    uint32_t    hidden_size;             /* n_embd                              */
    uint32_t    n_layer;                 /* number of transformer layers       */
    uint32_t    num_attention_heads;     /* n_head                              */
    uint32_t    num_kv_heads;            /* n_head_kv (GQA)                     */
    uint32_t    intermediate_size;       /* dense FFN size (pre-MoE layers)     */
    uint32_t    max_position_embeddings; /* n_ctx                               */
    uint32_t    head_dim;                /* per-head dim                         */
    uint32_t    kv_head_dim;             /* per kv-head dim                      */
    uint32_t    rope_dim;                /* RoPE dims                            */
    float       rope_theta;              /* RoPE base frequency                  */
    float       rms_norm_eps;            /* RMSNorm epsilon                      */
    /* MoE fields. */
    uint32_t    n_routed_experts;        /* total number of experts              */
    uint32_t    n_active_experts;        /* top-k experts per token              */
    uint32_t    expert_intermediate_size;/* per-expert FFN intermediate size     */
    uint32_t    moe_layer_start;         /* first MoE layer index (0-based)      */
    bool        has_shared_expert;        /* true = shared expert present         */
    uint32_t    shared_expert_intermediate_size; /* shared expert FFN size     */
    /* MLA fields (Hunyuan-Large). */
    bool        uses_mla;                 /* true = MLA attention                */
    uint32_t    mla_q_lora_dim;          /* q_a_proj output dim (latent)        */
    uint32_t    mla_kv_lora_dim;         /* kv_a_proj latent dim (without kv_pe) */
    uint32_t    mla_q_head_dim;          /* per-head q dim (nope + rope)         */
    uint32_t    mla_q_rope_dim;          /* RoPE dim for q_pe                    */
    uint32_t    mla_q_nope_dim;          /* per-head nope dim (mla_q_head_dim - rope_dim) */
    uint32_t    mla_v_head_dim;          /* per-head v dim                       */
    bool        tied_embeddings;         /* true = output.weight == tok_emb      */
} OcHunyuanConfig;

/* ─── Config parsing ──────────────────────────────────────────────────────
 *
 * Parse GLM/Hunyuan config from GGUF metadata. The `arch_str` is the raw
 * `general.architecture` value from the GGUF (e.g. "glm", "glm4",
 * "hunyuan", "hunyuan_moe"). The config structs are populated with
 * sensible defaults for any missing keys.
 *
 * Returns OC_OK on success, OC_ERR_INVALID_ARG (NULL args), or OC_ERR_MODEL
 * (invalid dimensions). */
OcError oc_glm_config_parse(const OcGgufFile *f, const char *arch_str,
                             OcGlmConfig *cfg);

OcError oc_hunyuan_config_parse(const OcGgufFile *f, const char *arch_str,
                                 OcHunyuanConfig *cfg);

/* Detect the GLM version from the architecture string.
 *   "glm"   → OC_GLM_VERSION_1 (ChatGLM-6B)
 *   "glm2"  → OC_GLM_VERSION_2 (ChatGLM2-6B)
 *   "glm3"  → OC_GLM_VERSION_3 (ChatGLM3-6B)
 *   "glm4"  → OC_GLM_VERSION_4 (GLM-4)
 *   others → OC_GLM_VERSION_UNKNOWN */
OcGlmVersion oc_glm_version_from_str(const char *s);

/* Initialize a GLM config to sensible defaults (all zeros + default eps,
 * rope_theta, version). Used by tests and as a starting point for parsing. */
void oc_glm_config_defaults(OcGlmConfig *cfg);

/* Initialize a Hunyuan config to sensible defaults. */
void oc_hunyuan_config_defaults(OcHunyuanConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_GLM_ARCH_H */
