/* ============================================================================
 * UNVERIFIED — this file has NEVER been compiled or run. It was written BLIND
 * against the verified CUDA reference (src/cuda/gemma4_cuda.h) and the Rust
 * Metal backend (oxidize-core/src/backends/metal.rs). It requires macOS + Xcode
 * Metal and an Apple GPU to compile and validate. It MAY NOT COMPILE. No
 * equivalence gate has ever been run against it.
 * ============================================================================
 *
 * Metal backend for the Gemma 4 engine. Reuses gemma4_load() (CPU loader) for
 * all GGUF parsing/geometry; this module uploads the still-quantized weights
 * (AL5_XS stays AL5_XS in unified memory) and runs a GPU-resident decode step
 * with a single commit+wait per token when results are requested.
 *
 * Apple Silicon is a single unified-memory device: --gpus > 1 is refused (the
 * CUDA layer-split pipeline is not ported). Rotoquant int4 KV is refused (MSL
 * has no k_fht / k_attn_q4 port); f16 KV only. ngl==0 is refused (pure-CPU). */
#ifndef OC_GEMMA4_METAL_H
#define OC_GEMMA4_METAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../model_gemma4.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Gemma4Metal Gemma4Metal;

/* m must stay alive (and its GGUF mmap open) for the lifetime of the handle.
 * n_gpus must be 1. n_gpu_layers is `-ngl`: layers [0, n_gpu_layers) are
 * uploaded and run on the GPU and the CPU runs the rest (and the head) from
 * gemma4_forward_from(); -1 or >= n_layers means the whole stack. 0 is refused.
 * Returns 0 on success, -1 with a message in err. */
int gemma4_metal_init(Gemma4Metal** out, const Gemma4Model* m, int n_gpus,
                      int n_gpu_layers, char* err, size_t errlen);
void gemma4_metal_free(Gemma4Metal* c);

/* One decode step at `pos` over the GPU's layers. Exactly one commit+wait
 * happens per call when results are requested:
 *   hidden_out != NULL: residual stream after the GPU's layers (partial offload).
 *     Takes precedence over the two below.
 *   argmax_out != NULL: greedy argmax on device (4-byte copy). Softcap skipped
 *     for argmax (tanh is monotonic).
 *   logits_out != NULL: full softcapped logits (vocab floats).
 *   all NULL (prefill): commit without wait.
 * Returns 0 on success, -1 on Metal error (printed to stderr). */
int gemma4_metal_forward(Gemma4Metal* c, int32_t token, size_t pos,
                         float* logits_out, int32_t* argmax_out,
                         float* hidden_out);

/* Hybrid step: all-GPU, or GPU layers [0, ngl) then gemma4_forward_from().
 * Returns m->logits (need_logits) or NULL; *failed is set on error. */
float* gemma4_metal_step(Gemma4Metal* c, Gemma4Model* m, int32_t token,
                         size_t pos, bool need_logits, int* failed);

#ifdef __cplusplus
}
#endif

#endif
