/*
 * model.h — OcModelArchitecture enum + detection + tensor-name mapping.
 *
 * Port of oxidize-core/src/model/inference.rs::ModelArchitecture and
 * oxidize-core/src/format/gguf.rs::map_tensor_name to C11.
 *
 * `oc_model_arch_from_str()` recognizes the 17 architecture variants
 * enumerated in inference.rs::ModelArchitecture::from_gguf:
 *   llama, mistral, mixtral, deepseek (+deepseek2/v2/v3/moe),
 *   qwen (+qwen2/2moe/3/3moe/35/3_5/3_5_text/35_text/3_5_moe/3_5_moe_text/35moe),
 *   gemma (+2/3/4), phi (+3), falcon, gpt2, gptj, gptneox,
 *   minimax (+minimax-m2/text-01), lfm2, lfm2moe,
 *   glm (+glm4/moe/moe_dsa/dsa/glmmoe/glmmoedsa),
 *   hunyuan (+moe/hunyuanmoe/hy_v3/hyv3/hunyuan_v3),
 *   plus OC_ARCH_UNKNOWN for unrecognized strings (16 recognized + 1 unknown
 *   = 17 enum values). LongCat is currently implemented by the C port only.
 *
 * `oc_gguf_map_tensor_name()` maps HuggingFace tensor names to the oxidize
 * canonical form (e.g. "model.layers.3.self_attn.q_proj.weight" →
 * "blk.3.attn_q.weight"). Per-architecture mapping tables mirror Rust.
 */
#ifndef OXIDIZE_MODEL_H
#define OXIDIZE_MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/arena.h"
#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Detected model architecture from GGUF metadata. The first 17 values mirror
 * oxidize-core, including OC_ARCH_UNKNOWN at its historical numeric value.
 * LongCat and Muse Glimmer are C-port-only extensions. Values MUST stay stable. */
typedef enum {
    OC_ARCH_LLAMA         = 0,
    OC_ARCH_MISTRAL       = 1,
    OC_ARCH_MIXTRAL       = 2,
    OC_ARCH_DEEPSEEK      = 3,
    OC_ARCH_QWEN          = 4,
    OC_ARCH_GEMMA         = 5,
    OC_ARCH_PHI           = 6,
    OC_ARCH_FALCON        = 7,
    OC_ARCH_GPT2          = 8,
    OC_ARCH_GPTJ          = 9,
    OC_ARCH_GPTNEOX       = 10,
    OC_ARCH_MINIMAX       = 11,
    OC_ARCH_LFM2          = 12,
    OC_ARCH_LFM2_MOE      = 13,
    OC_ARCH_GLM_MOE_DSA   = 14,
    OC_ARCH_HUNYUAN_MOE   = 15,
    OC_ARCH_UNKNOWN       = 16,
    OC_ARCH_LONGCAT       = 17,
    OC_ARCH_MUSE_GLIMMER  = 18,
    OC_ARCH__COUNT,
} OcModelArchitecture;

/* Map a `general.architecture` string to OcModelArchitecture. The string is
 * matched case-insensitively after '-' → '_' normalization (mirrors Rust
 * `arch.replace('-', "_")`). Unknown strings return OC_ARCH_UNKNOWN.
 * `oc_error_msg(OC_ERR_MODEL)` is the canonical error code for callers that
 * need to propagate "unsupported architecture" — they should check for
 * OC_ARCH_UNKNOWN and return OC_ERR_MODEL. */
OcModelArchitecture oc_model_arch_from_str(const char *s);

/* Inverse of oc_model_arch_from_str: returns a stable canonical name
 * ("llama", "qwen2", "deepseek2", ...) for the variant. Returns "unknown"
 * for OC_ARCH_UNKNOWN. Never returns NULL. */
const char *oc_model_arch_name(OcModelArchitecture arch);

/* Whether this architecture uses a Mixture-of-Experts FFN. Mirrors Rust
 * `ModelArchitecture::uses_moe()`. */
bool oc_model_arch_uses_moe(OcModelArchitecture arch);

/* Whether this architecture uses Multi-head Latent Attention (MLA) with a
 * compressed KV cache. Mirrors Rust `ModelArchitecture::uses_mla()`. */
bool oc_model_arch_uses_mla(OcModelArchitecture arch);

/* Whether this architecture uses Alibi positional encoding (no RoPE). */
bool oc_model_arch_uses_alibi(OcModelArchitecture arch);

/* Whether this architecture uses sliding window attention.
 * Qwen and Mistral use SWA. */
bool oc_model_arch_uses_sliding_window(OcModelArchitecture arch);

/* Whether this architecture uses LFM2 short-convolution token mixing on
 * non-attention layers (in addition to interleaved GQA attention layers). */
bool oc_model_arch_uses_shortconv(OcModelArchitecture arch);

/* Whether this architecture uses parallel attention + FFN (fused residual).
 * Gemma and Phi use this pattern. */
bool oc_model_arch_uses_parallel_attn_ffn(OcModelArchitecture arch);

/* Map a HuggingFace tensor name to the oxidize canonical form for the given
 * architecture. Returns an arena-owned, NUL-terminated string (so the caller
 * must keep `arena` alive for the lifetime of the result). If the name does
 * not match any known pattern, returns an arena-owned copy of `name`
 * (mirrors Rust's `mapped.unwrap_or_else(|| name.to_owned())`). Returns NULL
 * only on OOM.
 *
 * Examples (Llama/Qwen2 dense):
 *   "model.embed_tokens.weight"        → "tok_embeddings.weight"
 *   "lm_head.weight"                   → "output.weight"
 *   "model.norm.weight"                → "norm.weight"
 *   "model.layers.3.self_attn.q_proj.weight" → "blk.3.attn_q.weight"
 *   "model.layers.3.input_layernorm.weight"  → "blk.3.attn_norm.weight"
 *
 * MoE (Qwen2-MoE / Mixtral — block_sparse_moe.experts.M.w1.weight):
 *   "model.layers.4.block_sparse_moe.experts.2.w1.weight"
 *     → "blk.4.ffn_gate.2.weight"
 *
 * Qwen3-MoE shared expert (mlp.shared_expert.*):
 *   "model.layers.1.mlp.shared_expert.gate_proj.weight"
 *     → "blk.1.ffn_gate_shexp.weight"
 *
 * DeepSeek MLA (self_attn.q_a_proj.weight etc.):
 *   "model.layers.1.self_attn.q_a_proj.weight"            → "blk.1.attn_q_a.weight"
 *   "model.layers.1.self_attn.kv_a_proj_with_mqa.weight"  → "blk.1.attn_kv_a_mqa.weight"
 */
const char *oc_gguf_map_tensor_name(OcModelArchitecture arch, const char *name,
                                    OcArena *arena);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MODEL_H */
