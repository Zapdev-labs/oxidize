/* ======================================================================
 * UNVERIFIED — this file has NEVER been compiled or run.
 * Written BLIND against src/cuda/llama_cuda.h/.cu + the shaders in
 * src/vulkan/shaders/. Requires a Vulkan 1.1+ driver, libvulkan, and
 * SPIR-V built by src/vulkan/Makefile. IT MAY NOT COMPILE and MAY BE
 * WRONG. No verification was performed.
 * ======================================================================
 *
 * Vulkan-compute backend for the generic llama-family dense engine (arch
 * keys "llama", "mistral", "qwen2", "qwen3", "yi", "phi3", ...). Reuses
 * llama_load() for GGUF parsing/geometry; uploads still-quantized weights
 * into DEVICE_LOCAL buffers and runs a GPU-resident decode step.
 *
 * On the GPU: GQA, BOTH RoPE modes (NeoX via rope_neox pipe; ggml NORMAL
 * via rope_normal — honoring m->rope_norm), optional q/k/v/o biases,
 * optional per-head q/k RMSNorm (qwen3), SwiGLU FFN, tied/untied logits.
 *
 * Refusals:
 *   - multi-GPU (--gpus > 1)
 *   - --ngl 0
 *   - any MoE layer inside the offload range [0, ngl)
 *   - unsupported weight quants
 *   - head_dim > 256 (attn.comp shared q cache)
 *
 * KV cache is FP32 in these shaders (CUDA llama uses f16; see NOTES.md).
 */
#ifndef OC_LLAMA_VK_H
#define OC_LLAMA_VK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../model_llama.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LlamaVk LlamaVk;

/* m must stay alive (and its GGUF mmap open) for the lifetime of the handle.
 * n_gpus must be 1. n_gpu_layers is `-ngl`. 0 is refused. Returns 0 on
 * success, -1 with a message in err. */
int llama_vk_init(LlamaVk** out, const LlamaModel* m, int n_gpus,
                  int n_gpu_layers, char* err, size_t errlen);
void llama_vk_free(LlamaVk* c);

/* Hybrid step: all-GPU, or GPU layers [0, ngl) + llama_forward_from() on CPU.
 * Returns m->logits (need_logits) or NULL; *failed is set on error. */
float* llama_vk_step(LlamaVk* c, LlamaModel* m, int32_t token, size_t pos,
                     bool need_logits, int* failed);

#ifdef __cplusplus
}
#endif

#endif
