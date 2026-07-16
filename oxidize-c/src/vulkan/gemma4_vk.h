/* ======================================================================
 * UNVERIFIED — this file has NEVER been compiled or run.
 * Written BLIND against src/cuda/gemma4_cuda.h/.cu + the shaders in
 * src/vulkan/shaders/. Requires a Vulkan 1.1+ driver, libvulkan, and
 * SPIR-V built by src/vulkan/Makefile. IT MAY NOT COMPILE and MAY BE
 * WRONG. No verification was performed.
 * ======================================================================
 *
 * Vulkan-compute backend for the Gemma 4 engine. Reuses gemma4_load()
 * (CPU loader) for GGUF parsing/geometry; this module uploads still-
 * quantized weights into DEVICE_LOCAL buffers and runs a GPU-resident
 * decode step with one fence wait + optional D2H copy per token.
 *
 * Refusals (honest, loud):
 *   - multi-GPU (--gpus > 1): CUDA's layer-split pipeline is NOT ported
 *   - --ngl 0: that is the pure-CPU path
 *   - unsupported weight quants (only F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/AL5_XS)
 *   - rotoquant KV (m->kv_quant): shaders keep FP32 KV caches (see NOTES.md);
 *     the int4 Hadamard path has no SPIR-V counterpart
 *   - head_dim / v_head_dim > 256 (attn.comp shared q cache is fixed)
 */
#ifndef OC_GEMMA4_VK_H
#define OC_GEMMA4_VK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../model_gemma4.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Gemma4Vk Gemma4Vk;

/* m must stay alive (and its GGUF mmap open) for the lifetime of the handle.
 * n_gpus must be 1. n_gpu_layers is `-ngl`: layers [0, n_gpu_layers) are
 * uploaded and run on the GPU and the CPU runs the rest (and the head) from
 * gemma4_forward_from(); -1 or >= n_layers means the whole stack. 0 is refused.
 * Returns 0 on success, -1 with a message in err. */
int gemma4_vk_init(Gemma4Vk** out, const Gemma4Model* m, int n_gpus,
                   int n_gpu_layers, char* err, size_t errlen);
void gemma4_vk_free(Gemma4Vk* c);

/* One decode step at `pos` over the GPU's layers. Exactly one fence wait
 * happens per call. Copy-back rules mirror gemma4_cuda_forward:
 *   hidden_out != NULL: residual after GPU layers (partial offload). Takes
 *     precedence over the two below.
 *   argmax_out != NULL: greedy argmax on device (4-byte copy). Softcap skipped.
 *   logits_out != NULL: full softcapped logits (vocab floats).
 *   all NULL (prefill): still submits+waits (Vulkan has no stream to order
 *     against the next token — see NOTES.md).
 * Returns 0 on success, -1 on Vulkan error (printed to stderr). */
int gemma4_vk_forward(Gemma4Vk* c, int32_t token, size_t pos, float* logits_out,
                      int32_t* argmax_out, float* hidden_out);

/* Hybrid step: all-GPU, or GPU layers [0, ngl) + gemma4_forward_from() on CPU.
 * Returns m->logits (need_logits) or NULL; *failed is set on error. */
float* gemma4_vk_step(Gemma4Vk* c, Gemma4Model* m, int32_t token, size_t pos,
                      bool need_logits, int* failed);

#ifdef __cplusplus
}
#endif

#endif
