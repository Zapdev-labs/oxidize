/* ============================================================================
 * UNVERIFIED — THIS FILE HAS NEVER BEEN COMPILED OR RUN.
 * ----------------------------------------------------------------------------
 * Written BLIND against:
 *   - the VERIFIED CUDA backend  src/cuda/{gemma4_cuda.cu,llama_cuda.cu,
 *     gemma4_cuda.h,llama_cuda.h,cuda_dequant.cuh}  (the resident-forward shape
 *     this backend mirrors), and
 *   - the Rust MLX backend  oxidize-core/src/backends/mlx.rs  (which mlx-c /
 *     mlx_rs primitives to lean on).
 * Requires:  a Mac with Apple Silicon + a working MLX C API (mlx-c) install.
 * It CANNOT be built or run in the authoring environment and MAY NOT COMPILE.
 * The mlx-c function names/signatures below are ASSUMED from the mlx_rs Rust
 * bindings and the public MLX C++ headers; a validator on real hardware must
 * confirm each one against the installed <mlx/c/*.h>. Do not trust any of it
 * until tests/mlx_equiv (see README) is green against the CPU forward.
 * ============================================================================
 *
 * MLX backend for the Gemma 4 and generic Llama-family engines. Same contract
 * as the CUDA backend: reuse the CPU loaders (gemma4_load / llama_load) for ALL
 * GGUF parsing and geometry, then run a resident forward where the model weights
 * live in MLX unified memory and exactly ONE host<->device synchronization
 * (mlx_eval) happens per token.
 *
 * KEY DIFFERENCE FROM CUDA, STATED LOUDLY (see README "Design"):
 *   The CUDA backend keeps weights in their GGUF-quantized form in VRAM and a
 *   custom kernel fuses ggml dequant with the dot product (dqv<T> in
 *   cuda_dequant.cuh). MLX has NO way to run a ggml k-quant / AL5_XS decode on
 *   device: mlx_quantize is MLX's own affine group-quant, a DIFFERENT bit
 *   layout, so an MLX quantized_matmul would NOT be bit-equal to the CPU
 *   forward. Because the acceptance bar is "every logit agrees with the CPU
 *   forward" (tests/cuda_equiv.c for CUDA), this port DEQUANTIZES every weight
 *   to F32 on load via the reference decoder oc_dequant_row() and uploads F32
 *   arrays to unified memory. It therefore SUPPORTS THE SAME TYPES CUDA does
 *   (F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/AL5_XS) — it just decodes them on the
 *   host instead of on the GPU. Refuses any other type, loudly.
 *   MLX-native requant (mlx.rs from_gguf_tensor_quantized) is a memory
 *   optimization left STUBBED; see README.
 */
#ifndef OC_MLX_BACKEND_H
#define OC_MLX_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../model_gemma4.h"
#include "../model_llama.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Gemma 4 --------------------------------------------------------------
 * Mirrors gemma4_cuda.h. `m` must stay alive (GGUF mmap open) for the handle's
 * lifetime. n_gpu_layers is `-ngl`: layers [0, n_gpu_layers) run on the GPU and
 * the CPU runs the rest (and the head) from gemma4_forward_from(); -1 or
 * >= n_layers means the whole stack. 0 is refused (that is the pure-CPU path).
 * Apple Silicon is single-GPU, so there is no multi-GPU layer-split pipeline
 * (the gemma4-only CUDA feature) — n_gpus is accepted for signature parity but
 * only 1 is honored. Returns 0 on success, -1 with a message in err. */
typedef struct Gemma4Mlx Gemma4Mlx;
int gemma4_mlx_init(Gemma4Mlx** out, const Gemma4Model* m, int n_gpus,
                    int n_gpu_layers, char* err, size_t errlen);
void gemma4_mlx_free(Gemma4Mlx* c);

/* One decode step, whatever the split: all-GPU (final norm + tied logits +
 * softcap on the GPU), or GPU layers [0, ngl) then gemma4_forward_from() on the
 * CPU. Returns m->logits (need_logits) or NULL; *failed is set on error.
 * Mirrors gemma4_cuda_step. */
float* gemma4_mlx_step(Gemma4Mlx* c, Gemma4Model* m, int32_t token, size_t pos,
                       bool need_logits, int* failed);

/* ---- Llama family ---------------------------------------------------------
 * Mirrors llama_cuda.h. Same offload semantics. MoE layers are NOT offloaded:
 * a MoE layer inside [0, ngl) is refused at init (use --ngl to keep the MoE
 * tail on the CPU). Single GPU only. Returns 0 / -1 with err. */
typedef struct LlamaMlx LlamaMlx;
int llama_mlx_init(LlamaMlx** out, const LlamaModel* m, int n_gpus,
                   int n_gpu_layers, char* err, size_t errlen);
void llama_mlx_free(LlamaMlx* c);

float* llama_mlx_step(LlamaMlx* c, LlamaModel* m, int32_t token, size_t pos,
                      bool need_logits, int* failed);

#ifdef __cplusplus
}
#endif

#endif /* OC_MLX_BACKEND_H */
