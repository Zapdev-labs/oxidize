/*
 * cuda.h — CUDA backend for GPU-accelerated inference.
 *
 * When OC_CUDA is defined, the forward path dispatches to CUDA kernels
 * running on an NVIDIA GPU (e.g., L40S). Weights stay in their packed GGUF
 * form in device memory when a device kernel exists for the type (see
 * cuda_mmq.h — Q4_K/Q6_K/Q8_0), and are dequantized to f32 on upload only as
 * a fallback. The KV cache resides on the GPU in f16.
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

/* A weight tensor resident in device memory.
 *
 * `packed` distinguishes the two residency modes: when true, `data` holds the
 * raw GGUF blocks for `qtype` and rows are `row_bytes` apart, and the matvec
 * dispatches to a cuda_mmq.h kernel. When false, `data` is a plain f32
 * [rows, cols] buffer produced by host-side dequantization — the fallback for
 * types without a device kernel, and for shapes whose row length is not a
 * whole number of blocks (K-quants need cols % 256 == 0). */
typedef struct OcCudaWeight {
    void *data;                 /* device buffer: packed blocks or f32      */
    size_t row_bytes;           /* device row stride in bytes               */
    uint32_t qtype;             /* OcGgufQuantizationType of `data`         */
    uint32_t rows, cols;
    bool packed;
} OcCudaWeight;

typedef struct OcCudaContext {
    void *stream;               /* cudaStream_t (opaque)                  */
    /* Weight tensors resident on the device (packed where possible). */
    OcCudaWeight d_tok_embeddings; /* [vocab_size, n_embd]                 */
    float *d_final_norm;        /* [n_embd]                                */
    OcCudaWeight d_output;      /* [vocab_size, n_embd] (may alias tok_emb) */
    /* Per-layer device weights. */
    OcCudaWeight *d_attn_q;     /* [n_layer] [n_head*head_dim, n_embd]     */
    OcCudaWeight *d_attn_k;
    OcCudaWeight *d_attn_v;
    /* Optional Q/K/V projection biases (Qwen2-family). Entry is NULL for
     * layers/models without them. */
    float **d_attn_q_bias;
    float **d_attn_k_bias;
    float **d_attn_v_bias;
    OcCudaWeight *d_attn_output;
    OcCudaWeight *d_ffn_gate;
    OcCudaWeight *d_ffn_up;
    OcCudaWeight *d_ffn_down;
    float **d_attn_norm;
    float **d_ffn_norm;
    /* MoE (Qwen3-MoE / Mixtral / MiniMax-style). Present only when
     * num_experts > 0; the dense d_ffn_* above are then unused. Expert
     * tensors are stacked — expert i occupies rows [i*i_size, (i+1)*i_size)
     * in gate/up and [i*n_embd, (i+1)*n_embd) in down. */
    OcCudaWeight *d_ffn_gate_inp;    /* router: [num_experts, n_embd]      */
    OcCudaWeight *d_ffn_gate_exps;
    OcCudaWeight *d_ffn_up_exps;
    OcCudaWeight *d_ffn_down_exps;
    /* Shared expert (always active, weight 1.0), optional per layer. */
    OcCudaWeight *d_ffn_gate_shexp;
    OcCudaWeight *d_ffn_up_shexp;
    OcCudaWeight *d_ffn_down_shexp;
    OcCudaWeight *d_ffn_gate_inp_shexp;  /* optional sigmoid gate          */
    /* KV cache on GPU: [n_layer][n_ctx][n_head_kv*head_dim] for K and V,
     * stored as __half (opaque here to keep this header C11-clean). f16
     * halves the cache footprint versus f32 at no measurable quality cost —
     * it is what the Rust CUDA backend stores as well. */
    void *d_kv_k;
    void *d_kv_v;
    /* Workspace for activations. */
    float *d_x, *d_normed, *d_q, *d_k, *d_v, *d_attn_out;
    float *d_ffn_gate_buf, *d_ffn_up_buf, *d_logits;
    /* Per-head attention score scratch, [n_head, n_ctx]. Kept in device
     * memory rather than shared: a full context of scores exceeds the 48 KB
     * shared-memory budget well before n_ctx reaches its typical 32768. */
    float *d_attn_scores;
    /* MoE scratch (allocated only when num_experts > 0). The routing result
     * lives on the device so no per-layer host sync is needed. */
    float    *d_router_logits;   /* [num_experts]                          */
    uint32_t *d_expert_sel;      /* [k] chosen expert ids                  */
    float    *d_expert_w;        /* [k] renormalized, scaled gate weights  */
    float    *d_expert_gate;     /* [expert_intermediate_size]             */
    float    *d_expert_up;       /* [expert_intermediate_size]             */
    float    *d_expert_tmp;      /* [n_embd] one expert's down output      */
    float    *d_expert_out;      /* [n_embd] accumulator                   */
    float    *d_shexp_gate_logit;/* [1] shared-expert sigmoid gate logit   */
    /* Model dimensions. */
    uint32_t n_embd, n_head, n_head_kv, n_ff, head_dim, n_layer, vocab_size;
    uint32_t n_ctx, rope_dim;
    float rope_theta, rms_norm_eps, norm_scale;
    bool uses_geglu;
    /* MoE config mirror (0 experts = dense FFN). */
    uint32_t num_experts, num_experts_per_tok, expert_intermediate_size;
    bool  expert_gating_sigmoid;
    float expert_weights_scale;
    bool initialized;
    /* Device memory accounting, filled during init (for --verbose reporting). */
    size_t vram_weight_bytes;   /* all weight tensors                       */
    size_t vram_kv_bytes;       /* KV cache                                 */
    uint32_t n_packed_tensors;  /* tensors kept quantized on the device     */
    uint32_t n_f32_tensors;     /* tensors that fell back to f32            */
} OcCudaContext;

/* Initialize the CUDA context: allocate device memory and upload weights from
 * the loaded OcLlamaModel, keeping them packed where a device kernel exists.
 * Returns OC_ERR_UNSUPPORTED if CUDA is not available (compiled without
 * OC_CUDA or no GPU). */
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
