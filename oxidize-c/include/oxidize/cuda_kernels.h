/* cuda_kernels.h — CUDA fused kernels for GPU-accelerated LLM inference. */
#ifndef OXIDIZE_CUDA_KERNELS_H
#define OXIDIZE_CUDA_KERNELS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool oc_cuda_rmsnorm_rope_fused(
    const float *d_x, const float *d_weight, float *d_out,
    uint32_t hidden_dim, uint32_t n_heads, uint32_t head_dim,
    uint32_t rope_dim, int64_t pos, float theta,
    float eps, float norm_scale);

bool oc_cuda_swiglu(float *d_gate, const float *d_up, size_t n);

bool oc_cuda_attention_softmax(
    const float *d_q, const float *d_k_cache, const float *d_v_cache,
    float *d_out, uint32_t head_dim, size_t n_past);

bool oc_cuda_q4k_matvec(
    const void *d_weights, const float *d_x, float *d_out,
    size_t rows, size_t cols);

bool oc_cuda_q4k_dequantize(
    const void *d_src, float *d_dst, size_t n_blocks);

bool oc_cuda_embedding_lookup(
    const float *d_embeddings, const uint32_t *d_tokens, float *d_out,
    uint32_t vocab_size, uint32_t embd_dim, size_t n_tokens);

bool oc_cuda_argmax(
    const float *d_logits, uint32_t *d_out_idx, uint32_t vocab_size);

bool oc_cuda_topk(
    const float *d_logits, uint32_t *d_out_idx, float *d_out_val,
    uint32_t vocab_size, uint32_t k);

bool oc_cuda_qk_norm_rope(
    float *d_x, const float *d_weight,
    uint32_t n_heads, uint32_t head_dim, uint32_t rope_dim,
    int64_t pos, float theta, float eps,
    float yarn_factor, uint32_t yarn_orig_ctx, void *stream);

/* Packed Qwen3.5 Q projection is [n_head, 2, head_dim] = concat(Q, gate). */
bool oc_cuda_qwen35_unpack_qgate(
    const float *d_packed, float *d_q, float *d_gate,
    uint32_t n_heads, uint32_t head_dim, void *stream);

bool oc_cuda_sigmoid_gate(float *d_x, const float *d_gate, size_t n,
                          void *stream);

/* One Gated-DeltaNet token step. Persistent conv/recurrent state lives on
 * the device; scratch `d_conv_out` / `d_out` are overwritten. */
bool oc_cuda_qwen35_delta_step(
    float *d_conv_state, float *d_recurrent,
    const float *d_qkv, const float *d_gate,
    const float *d_beta, const float *d_alpha,
    const float *d_conv_w, const float *d_ssm_a,
    const float *d_dt_bias, const float *d_norm_w,
    float *d_conv_out, float *d_out,
    uint32_t n_key_heads, uint32_t n_value_heads,
    uint32_t key_head_dim, uint32_t value_head_dim,
    uint32_t conv_kernel, float eps, void *stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OXIDIZE_CUDA_KERNELS_H */
