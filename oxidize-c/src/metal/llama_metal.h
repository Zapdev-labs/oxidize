/* ============================================================================
 * UNVERIFIED — this file has NEVER been compiled or run. It was written BLIND
 * against the verified CUDA reference (src/cuda/llama_cuda.h) and the Rust
 * Metal backend (oxidize-core/src/backends/metal.rs). It requires macOS + Xcode
 * Metal and an Apple GPU to compile and validate. It MAY NOT COMPILE. No
 * equivalence gate has ever been run against it.
 * ============================================================================
 *
 * Metal backend for the generic llama-family dense engine. Reuses llama_load()
 * for GGUF parsing/geometry; uploads still-quantized weights and runs a
 * GPU-resident decode with one commit+wait per token when results are needed.
 *
 * Features on the GPU: GQA, BOTH RoPE modes (NeoX + ggml NORMAL), optional
 * q/k/v/o biases, optional per-head q/k RMSNorm, SwiGLU FFN, tied/untied
 * logits. Same dqv<T> types as CUDA (via metal_dequant.h).
 *
 * NOT on the GPU: MoE layers (refused if in offload range). Single GPU only.
 * KV cache is f16. */
#ifndef OC_LLAMA_METAL_H
#define OC_LLAMA_METAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../model_llama.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LlamaMetal LlamaMetal;

/* m must stay alive for the handle's lifetime. n_gpus must be 1. n_gpu_layers
 * is `-ngl`; 0 is refused. Returns 0 on success, -1 with err. */
int llama_metal_init(LlamaMetal** out, const LlamaModel* m, int n_gpus,
                     int n_gpu_layers, char* err, size_t errlen);
void llama_metal_free(LlamaMetal* c);

/* Hybrid step: all-GPU or GPU [0,ngl) then llama_forward_from(). */
float* llama_metal_step(LlamaMetal* c, LlamaModel* m, int32_t token, size_t pos,
                        bool need_logits, int* failed);

#ifdef __cplusplus
}
#endif

#endif
