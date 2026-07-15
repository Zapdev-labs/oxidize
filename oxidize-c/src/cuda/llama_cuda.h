/* CUDA backend for the generic llama-family dense engine (arch keys "llama",
 * "mistral", "qwen2", "qwen3", "yi", "phi3", ...). Reuses llama_load() (CPU
 * loader) for all GGUF parsing/geometry; this module uploads the still-quantized
 * weights and runs a GPU-resident decode step with a single device->host copy
 * per token, mirroring src/cuda/gemma4_cuda.cu.
 *
 * Features on the GPU: GQA (n_head vs n_kv_heads), BOTH RoPE modes (NeoX
 * split-half for qwen2/qwen3/phi3; ggml NORMAL adjacent-pair for llama/mistral/
 * yi — honoring m->rope_norm so the stored (permuted-for-normal) q/k rotate the
 * right pairs), optional q/k/v/o biases, optional per-head q/k RMSNorm (qwen3),
 * SwiGLU FFN, and tied vs untied (output.weight) logits. The device dequant is
 * the SAME dqv<T> the gemma4 backend uses (cuda_dequant.cuh), so F32/F16/Q4_0/
 * Q8_0/Q4_K/Q5_K/Q6_K/AL5_XS all work and are held to tests/cuda_equiv.c.
 *
 * NOT on the GPU: Mixture-of-Experts layers. A model whose GPU-offloaded range
 * [0, n_gpu_layers) contains a MoE layer is refused at init with a clear message
 * (set --ngl to keep the MoE tail on the CPU, or run pure-CPU). The KV cache is
 * f16 (llama's rotoquant int4 is gemma4-only). Single GPU only: --gpus > 1 is
 * refused (the layer-split pipeline buys memory capacity, not tok/s, and is a
 * gemma4-only feature for the big MoE weights). */
#ifndef OC_LLAMA_CUDA_H
#define OC_LLAMA_CUDA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../model_llama.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LlamaCuda LlamaCuda;

/* m must stay alive (and its GGUF mmap open) for the lifetime of the handle.
 * n_gpus must be 1. n_gpu_layers is `-ngl`: layers [0, n_gpu_layers) are
 * uploaded and run on the GPU and the CPU runs the rest (and the head) from
 * llama_forward_from(); -1 or >= n_layers means the whole stack. 0 is refused
 * (that is the pure-CPU binary). Returns 0 on success, -1 with a message in
 * err (unsupported weight type, a MoE layer in the offload range, etc.). */
int llama_cuda_init(LlamaCuda** out, const LlamaModel* m, int n_gpus,
                    int n_gpu_layers, char* err, size_t errlen);
void llama_cuda_free(LlamaCuda* c);

/* One decode step, whatever the split: all-GPU (final norm + tied/untied logits
 * on the GPU), or GPU layers [0, ngl) followed by llama_forward_from() on the
 * CPU. Returns m->logits (need_logits) or NULL; *failed is set on error. This is
 * the one place the hybrid step is spelled out — the CLI and the equivalence
 * test both go through it. */
float* llama_cuda_step(LlamaCuda* c, LlamaModel* m, int32_t token, size_t pos,
                       bool need_logits, int* failed);

#ifdef __cplusplus
}
#endif

#endif
