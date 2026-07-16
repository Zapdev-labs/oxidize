/* ======================================================================
 * UNVERIFIED — NEVER COMPILED OR RUN. Written BLIND against
 * src/cuda/llama_cuda.h and the WebGPU host in webgpu_common.h.
 * Requires Dawn/emdawn + a GPU. MAY NOT COMPILE.
 * ======================================================================
 *
 * WebGPU backend for the generic llama-family dense engine. Reuses llama_load()
 * for GGUF parsing/geometry; uploads still-quantized weights; f16 KV cache.
 *
 * REFUSED at init:
 *   - n_gpus != 1 (multi-GPU is gemma4-CUDA-only; WebGPU has no peer copy)
 *   - MoE layer inside the offload range [0, n_gpu_layers)
 *   - unsupported weight quant types
 *   - head_dim > 256 (attn.wgsl workgroup q cache)
 */
#ifndef OC_LLAMA_WEBGPU_H
#define OC_LLAMA_WEBGPU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../model_llama.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LlamaWebGpu LlamaWebGpu;

/* m must stay alive (GGUF mmap open) for the handle's lifetime.
 * n_gpus must be 1. n_gpu_layers is `-ngl`. 0 refused. Returns 0 / -1+err. */
int llama_webgpu_init(LlamaWebGpu** out, const LlamaModel* m, int n_gpus,
                      int n_gpu_layers, char* err, size_t errlen);
void llama_webgpu_free(LlamaWebGpu* c);

/* Hybrid step: all-GPU or GPU [0,ngl) + llama_forward_from(). */
float* llama_webgpu_step(LlamaWebGpu* c, LlamaModel* m, int32_t token,
                         size_t pos, bool need_logits, int* failed);

#ifdef __cplusplus
}
#endif

#endif /* OC_LLAMA_WEBGPU_H */
