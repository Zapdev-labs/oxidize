/*
 * cuda.h — CUDA backend for GPU-accelerated inference.
 *
 * When OC_CUDA is defined, the forward path dispatches to CUDA kernels
 * running on an NVIDIA GPU (e.g., L40S). Weights are dequantized to f32
 * on load and uploaded to device memory. KV cache resides on GPU.
 *
 * When OC_CUDA is not defined, these functions are no-ops / stubs and
 * the CPU forward path is used.
 */
#ifndef OXIDIZE_CUDA_H
#define OXIDIZE_CUDA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/llama.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcCudaContext {
    void *stream;               /* cudaStream_t (opaque)                  */
    /* Device pointers for weight tensors (f32, dequantized on upload). */
    float *d_tok_embeddings;     /* [vocab_size, n_embd]                    */
    float *d_final_norm;        /* [n_embd]                                */
    float *d_output;            /* [vocab_size, n_embd] (may alias tok_emb) */
    /* Per-layer device pointers. */
    float **d_attn_q;           /* [n_layer] [n_head*head_dim, n_embd]     */
    float **d_attn_k;
    float **d_attn_v;
    float **d_attn_output;
    float **d_ffn_gate;
    float **d_ffn_up;
    float **d_ffn_down;
    float **d_attn_norm;
    float **d_ffn_norm;
    /* KV cache on GPU: [n_layer][n_ctx][n_head_kv*head_dim] for K and V. */
    float *d_kv_k;
    float *d_kv_v;
    /* Workspace for activations. */
    float *d_x, *d_normed, *d_q, *d_k, *d_v, *d_attn_out;
    float *d_ffn_gate_buf, *d_ffn_up_buf, *d_logits;
    /* Model dimensions. */
    uint32_t n_embd, n_head, n_head_kv, n_ff, head_dim, n_layer, vocab_size;
    uint32_t n_ctx, rope_dim;
    float rope_theta, rms_norm_eps, norm_scale;
    bool uses_geglu;
    bool initialized;
} OcCudaContext;

/* Initialize the CUDA context: allocate device memory, upload + dequantize
 * weights from the loaded OcLlamaModel. Returns OC_ERR_UNSUPPORTED if CUDA
 * is not available (compiled without OC_CUDA or no GPU). */
OcError oc_cuda_init(OcCudaContext *ctx, const OcLlamaModel *model);

/* Run one forward step on GPU: embed token, run all layers, output logits.
 * `logits_out` receives vocab_size floats (copied from device).
 * If `logits_out` is NULL, skips the lm_head projection. */
OcError oc_cuda_forward(OcCudaContext *ctx, uint32_t token, uint32_t pos,
                        float *logits_out);

/* Reset position counter (start a new sequence). */
void oc_cuda_reset(OcCudaContext *ctx);

/* Free all device memory. */
void oc_cuda_free(OcCudaContext *ctx);

/* Check if CUDA is available (compiled with OC_CUDA and a GPU is present). */
bool oc_cuda_available(void);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_CUDA_H */
