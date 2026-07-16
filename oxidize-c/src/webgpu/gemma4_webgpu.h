/* ======================================================================
 * UNVERIFIED — NEVER COMPILED OR RUN. Written BLIND against
 * src/cuda/gemma4_cuda.h and the WebGPU host in webgpu_common.h.
 * Requires Dawn/emdawn + a GPU. MAY NOT COMPILE.
 * ======================================================================
 *
 * WebGPU backend for the Gemma 4 engine. Reuses gemma4_load() (CPU loader) for
 * all GGUF parsing/geometry; this module uploads still-quantized weights and
 * runs a GPU-resident decode step with a single device->host copy per token.
 *
 * Differences from CUDA (stated loudly):
 *   - Multi-GPU (`--gpus N`) is REFUSED. WebGPU has no layer-split peer copy.
 *   - Rotoquant KV (`m->kv_quant`) is REFUSED. Only the f16 KV path is ported.
 *   - Same quant gate as CUDA (F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/AL5_XS).
 */
#ifndef OC_GEMMA4_WEBGPU_H
#define OC_GEMMA4_WEBGPU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../model_gemma4.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Gemma4WebGpu Gemma4WebGpu;

/* m must stay alive (GGUF mmap open) for the handle's lifetime.
 * n_gpus must be 1 (multi-GPU refused). n_gpu_layers is `-ngl`: layers
 * [0, n_gpu_layers) on GPU, CPU finishes via gemma4_forward_from(); -1 or
 * >= n_layers means the whole stack. 0 is refused (pure-CPU path).
 * Returns 0 on success, -1 with a message in err. */
int gemma4_webgpu_init(Gemma4WebGpu** out, const Gemma4Model* m, int n_gpus,
                       int n_gpu_layers, char* err, size_t errlen);
void gemma4_webgpu_free(Gemma4WebGpu* c);

/* One decode step. Prefer gemma4_webgpu_step for the hybrid path.
 *   hidden_out != NULL: residual after GPU layers (partial offload).
 *   argmax_out != NULL: greedy argmax on device (4-byte copy); softcap skipped.
 *   logits_out != NULL: full softcapped logits.
 *   all NULL (prefill): submit without download.
 * Returns 0 on success, -1 on error. */
int gemma4_webgpu_forward(Gemma4WebGpu* c, int32_t token, size_t pos,
                          float* logits_out, int32_t* argmax_out,
                          float* hidden_out);

/* Hybrid step: all-GPU or GPU [0,ngl) + gemma4_forward_from(). Returns
 * m->logits (need_logits) or NULL; *failed set on error. */
float* gemma4_webgpu_step(Gemma4WebGpu* c, Gemma4Model* m, int32_t token,
                          size_t pos, bool need_logits, int* failed);

#ifdef __cplusplus
}
#endif

#endif /* OC_GEMMA4_WEBGPU_H */
