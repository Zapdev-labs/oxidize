/* CUDA backend for the Gemma 4 engine. Reuses gemma4_load() (CPU loader) for
 * all GGUF parsing/geometry; this module uploads the still-quantized weights
 * (AL5_XS stays AL5_XS in VRAM) and runs a GPU-resident decode step with a
 * single device->host copy per token.
 *
 * Multi-GPU (`--gpus N`) is a LAYER-SPLIT PIPELINE: contiguous layer ranges
 * per GPU, one 21KB hidden-state peer copy per boundary per token. Chosen
 * over row-split tensor parallel because it is trivially correct unseen (no
 * per-matmul allreduce); tradeoff: little single-stream speedup, mainly
 * memory capacity. One 80GB H100 fits the 13.45GB model, so --gpus 1 is the
 * expected perf path (see CUDA.md). */
#ifndef OC_GEMMA4_CUDA_H
#define OC_GEMMA4_CUDA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../model_gemma4.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Gemma4Cuda Gemma4Cuda;

/* m must stay alive (and its GGUF mmap open) for the lifetime of the handle.
 * n_gpus in [1, 8]. n_gpu_layers is `-ngl`: layers [0, n_gpu_layers) are
 * uploaded and run on the GPU and the CPU runs the rest (and the head) from
 * gemma4_forward_from(); -1 or >= n_layers means the whole stack. 0 is not a
 * valid handle (that is the pure-CPU binary) and is refused.
 * Returns 0 on success, -1 with a message in err. */
int gemma4_cuda_init(Gemma4Cuda** out, const Gemma4Model* m, int n_gpus,
                     int n_gpu_layers, char* err, size_t errlen);
void gemma4_cuda_free(Gemma4Cuda* c);

/* One decode step at `pos` over the GPU's layers. The whole step is enqueued
 * asynchronously; exactly one sync + D2H copy happens per call, and only if
 * requested:
 *   hidden_out != NULL: the residual stream after the GPU's layers is copied
 *     back (partial offload; no final norm/logits are computed here). Takes
 *     precedence over the two below.
 *   argmax_out != NULL: greedy argmax computed on device (4-byte copy).
 *     Softcap is skipped for argmax (tanh is monotonic; argmax unchanged).
 *   logits_out != NULL: full softcapped logits copied (vocab floats).
 *   all NULL (prefill): no sync at all.
 * Returns 0 on success, -1 on CUDA error (printed to stderr). */
int gemma4_cuda_forward(Gemma4Cuda* c, int32_t token, size_t pos,
                        float* logits_out, int32_t* argmax_out,
                        float* hidden_out);

/* One decode step, whatever the split: all-GPU, or GPU layers [0, ngl) followed
 * by gemma4_forward_from() on the CPU. Returns m->logits (need_logits) or NULL;
 * *failed is set on error. This is the ONE place the hybrid step is spelled
 * out — the CLI and the equivalence test both go through it. */
float* gemma4_cuda_step(Gemma4Cuda* c, Gemma4Model* m, int32_t token, size_t pos,
                        bool need_logits, int* failed);

#ifdef __cplusplus
}
#endif

#endif
