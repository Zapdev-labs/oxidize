/* cuda.h — CUDA backend for GPU-accelerated inference. */
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

/* A weight tensor resident in device memory. */
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
    /* Gemma-family extra norms, per layer; entry is NULL when the model has no such tensor. attn_q/k_norm are per-head and are the LAYER's head_dim long (256 on Gemma 4 sliding layers, 512 on global), so they are not interchangeable between layers. */
    float **d_attn_q_norm;
    float **d_attn_k_norm;
    float **d_post_attn_norm;
    float **d_post_ffw_norm;
    /* MoE (Qwen3-MoE / Mixtral / MiniMax-style). Present only when num_experts > 0; the dense d_ffn_* above are then unused. Expert tensors are stacked — expert i occupies rows [i*i_size, (i+1)*i_size) in gate/up and [i*n_embd, (i+1)*n_embd) in down. */
    OcCudaWeight *d_ffn_gate_inp;    /* router: [num_experts, n_embd]      */
    OcCudaWeight *d_ffn_gate_exps;
    OcCudaWeight *d_ffn_up_exps;
    OcCudaWeight *d_ffn_down_exps;
    /* Shared expert (always active, weight 1.0), optional per layer. */
    OcCudaWeight *d_ffn_gate_shexp;
    OcCudaWeight *d_ffn_up_shexp;
    OcCudaWeight *d_ffn_down_shexp;
    OcCudaWeight *d_ffn_gate_inp_shexp;  /* optional sigmoid gate          */
    /* KV cache on GPU: [n_layer][n_ctx][n_head_kv*head_dim] for K and V, stored as __half (opaque here to keep this header C11-clean). */
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
    float yarn_factor;
    uint32_t yarn_orig_ctx;
    bool uses_geglu;
    bool      uses_gemma4;
    uint32_t *l_head_dim;
    uint32_t *l_n_head_kv;
    uint32_t *l_rope_dim;
    float    *l_rope_theta;
    uint32_t *l_sliding;        /* window size, 0 = global attention        */
    /* KV cache stride in elements per position per layer. */
    size_t   kv_row;
    /* Final logit softcap: logits = tanh(l/c)*c. 0 = disabled (Gemma 4: 30). */
    float    logit_softcap;
    /* Pre-attention score scale; 0 = the usual 1/sqrt(head_dim). Gemma 4
     * uses 1.0 (no scaling at all). See OcLlamaConfig::attn_scale. */
    float    attn_scale;
    /* Gemma 4 RMS-normalizes V after projection, with no weight tensor. */
    bool     v_rms_norm;
    /* Per-layer output scale (blk.N.layer_output_scale.weight), host-side,
     * n_layer entries; NULL when the model has none. */
    float   *l_out_scale;
    /* MoE config mirror (0 experts = dense FFN). */
    uint32_t num_experts, num_experts_per_tok, expert_intermediate_size;
    uint32_t shared_expert_intermediate_size;
    bool  expert_gating_sigmoid;
    float expert_weights_scale;
    /* Qwen3.5 / Qwen 3.8 hybrid: Gated DeltaNet layers mixed with full GQA.
     * l_kind[l] is OcLlamaLayerKind. Full-attention layers index the KV cache
     * by l_kv_index[l], not by the transformer layer id. */
    bool      is_qwen35;
    uint8_t  *l_kind;
    uint32_t *l_kv_index;
    uint32_t  n_full_attention_layers;
    uint32_t  n_recurrent_layers;
    uint32_t  ssm_conv_kernel;
    uint32_t  ssm_group_count;
    uint32_t  ssm_state_size;
    uint32_t  ssm_value_heads;
    uint32_t  ssm_inner_size;
    OcCudaWeight *d_attn_qkv;
    OcCudaWeight *d_attn_gate;
    OcCudaWeight *d_ssm_alpha;
    OcCudaWeight *d_ssm_beta;
    OcCudaWeight *d_ssm_out;
    float **d_ssm_conv1d;
    float **d_ssm_a;
    float **d_ssm_dt_bias;
    float **d_ssm_norm;
    float *d_conv_state;
    float *d_recurrent_state;
    float *d_qwen35_qkv;
    float *d_qwen35_gate;
    float *d_qwen35_beta;
    float *d_qwen35_alpha;
    float *d_qwen35_conv_out;
    float *d_qwen35_delta_out;
    size_t conv_state_per_layer;
    size_t recurrent_state_per_layer;
    bool initialized;
    /* Non-blocking streams for independent GEMVs (Q/K/V or FFN gate/up). */
    void *compute_streams[3];
    /* Device memory accounting, filled during init (for --verbose reporting). */
    size_t vram_weight_bytes;   /* all weight tensors                       */
    size_t vram_kv_bytes;       /* KV cache                                 */
    uint32_t n_packed_tensors;  /* tensors kept quantized on the device     */
    uint32_t n_f32_tensors;     /* tensors that fell back to f32            */
} OcCudaContext;

/* Initialize the CUDA context: allocate device memory and upload weights from the loaded OcLlamaModel, keeping them packed where a device kernel exists. Returns OC_ERR_UNSUPPORTED if CUDA is not available (compiled without OC_CUDA or no GPU). */
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

/* Device-side kernel self-test: Qwen3.5 unpack/QK-norm/RoPE/DeltaNet and
 * packed Q4_0 GEMV. No GGUF required. Returns OC_OK on pass. */
OcError oc_cuda_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_CUDA_H */
