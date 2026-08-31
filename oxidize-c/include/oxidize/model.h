/* model.h — OcModelArchitecture enum + detection + tensor-name mapping. */
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

/* Map a `general.architecture` string to OcModelArchitecture. */
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

/* Map a HuggingFace tensor name to the oxidize canonical form for the given architecture. Returns an arena-owned, NUL-terminated string (keep `arena` alive); unmatched names return an arena copy of `name`. Returns NULL only on OOM. */
const char *oc_gguf_map_tensor_name(OcModelArchitecture arch, const char *name,
                                    OcArena *arena);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MODEL_H */
