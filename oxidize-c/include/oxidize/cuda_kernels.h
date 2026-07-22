/*
 * cuda_kernels.h — CUDA fused kernels for GPU-accelerated LLM inference.
 *
 * Companion header for cuda_kernels.cu. The host-side wrappers have
 * `extern "C"` linkage so the C11 forward path in oxidize-c can call
 * them directly without requiring a C++ compiler on the call site.
 *
 * Kernels (see cuda_kernels.cu for implementation):
 *   - RMSNorm + RoPE fused
 *   - SwiGLU activation
 *   - Attention online softmax
 *   - Q4_K quantized matvec
 *   - Q4_K → F32 dequantize
 *   - Embedding lookup
 *   - Argmax (greedy sampling)
 *   - Top-k partial sort
 *
 * Build: nvcc -O2 -c src/backends/cuda_kernels.cu -o cuda_kernels.cu.o
 */
#ifndef OXIDIZE_CUDA_KERNELS_H
#define OXIDIZE_CUDA_KERNELS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── RMSNorm + RoPE fused ───────────────────────────────────────────────
 *
 * Fuses RMSNorm and rotary position embedding into a single kernel launch
 * for the Q (or K) projection of a single head. The input hidden states
 * are first RMS-normalized with `weight`, then RoPE-rotated.
 *
 * `hidden_dim` must equal `n_heads * head_dim`. `rope_dim` is the number
 * of dimensions that receive rotation (usually == head_dim).
 *
 * Parameters:
 *   d_x        — device pointer to hidden states [hidden_dim]
 *   d_weight   — RMSNorm weight [hidden_dim]
 *   d_out      — device output [hidden_dim]
 *   hidden_dim — total hidden size
 *   n_heads    — number of query heads in this tensor
 *   head_dim   — per-head dimension
 *   rope_dim   — rotary dimensions (≤ head_dim)
 *   pos        — token position
 *   theta      — RoPE base frequency
 *   eps        — RMSNorm epsilon
 *   norm_scale — Gemma norm scale (1.0 for Llama)
 */
bool oc_cuda_rmsnorm_rope_fused(
    const float *d_x, const float *d_weight, float *d_out,
    uint32_t hidden_dim, uint32_t n_heads, uint32_t head_dim,
    uint32_t rope_dim, int64_t pos, float theta,
    float eps, float norm_scale);

/* ─── SwiGLU activation ──────────────────────────────────────────────────
 *
 * silu(gate) * up, where silu(x) = x * sigmoid(x). One element per thread.
 */
bool oc_cuda_swiglu(float *d_gate, const float *d_up, size_t n);

/* ─── Attention online softmax ───────────────────────────────────────────
 *
 * Flash-attention-style online softmax for a single query head against
 * a key/value cache. Computes softmax(QK^T / sqrt(d)) * V.
 *
 * Parameters:
 *   d_q        — query [head_dim]
 *   d_k_cache  — keys [n_past, head_dim] for this head
 *   d_v_cache  — values [n_past, head_dim] for this head
 *   d_out      — output [head_dim]
 *   head_dim   — per-head dimension
 *   n_past     — number of cached KV pairs
 */
bool oc_cuda_attention_softmax(
    const float *d_q, const float *d_k_cache, const float *d_v_cache,
    float *d_out, uint32_t head_dim, size_t n_past);

/* ─── Q4_K quantized matvec ───────────────────────────────────────────────
 *
 * Matrix-vector multiply for Q4_K weights × F32 activation vector.
 * One block per output row; threads cooperatively dequantize and
 * accumulate over Q4_K super-blocks (256 elements each).
 *
 * Parameters:
 *   d_weights — Q4_K packed weights [rows, cols] (cols must be multiple of 256)
 *   d_x       — F32 activation vector [cols]
 *   d_out     — F32 output [rows]
 *   rows      — number of output rows
 *   cols      — number of input columns (must be a multiple of OC_QK_K=256)
 */
bool oc_cuda_q4k_matvec(
    const void *d_weights, const float *d_x, float *d_out,
    size_t rows, size_t cols);

/* ─── Q4_K → F32 dequantize ───────────────────────────────────────────────
 *
 * Dequantizes an entire Q4_K weight tensor to F32. Used for GPU offload
 * when weights are uploaded to the device. Each thread handles one
 * super-block of 256 elements.
 *
 * Parameters:
 *   d_src   — Q4_K packed weights (length src_bytes bytes)
 *   d_dst   — F32 output (length n_blocks * 256 floats)
 *   n_blocks — number of Q4_K super-blocks
 */
bool oc_cuda_q4k_dequantize(
    const void *d_src, float *d_dst, size_t n_blocks);

/* ─── Embedding lookup ────────────────────────────────────────────────────
 *
 * Gathers `n_tokens` embedding rows from a [vocab_size, embd_dim] table.
 *
 * Parameters:
 *   d_embeddings — F32 embedding table [vocab_size, embd_dim]
 *   d_tokens     — uint32 token IDs [n_tokens]
 *   d_out        — F32 output [n_tokens, embd_dim]
 *   vocab_size   — size of vocab axis (for bounds checking)
 *   embd_dim     — embedding dimension
 *   n_tokens     — number of tokens to look up
 */
bool oc_cuda_embedding_lookup(
    const float *d_embeddings, const uint32_t *d_tokens, float *d_out,
    uint32_t vocab_size, uint32_t embd_dim, size_t n_tokens);

/* ─── Argmax (greedy sampling) ────────────────────────────────────────────
 *
 * Finds the index of the maximum logit. Uses a single-block reduction
 * (suitable for vocab sizes up to ~32k; for larger vocabs, multiple
 * blocks + a second-pass reduction would be needed).
 *
 * Parameters:
 *   d_logits   — F32 logits [vocab_size]
 *   d_out_idx  — uint32 output index [1]
 *   vocab_size — number of logits
 */
bool oc_cuda_argmax(
    const float *d_logits, uint32_t *d_out_idx, uint32_t vocab_size);

/* ─── Top-k partial sort ──────────────────────────────────────────────────
 *
 * Returns the indices of the top-k logits in descending order of logit
 * value. Uses a per-thread bitonic selection pass followed by a block
 * reduction; k must be ≤ 1024 (one block). Suitable for typical
 * top-k sampling values (k = 40, 50, 100).
 *
 * Parameters:
 *   d_logits   — F32 logits [vocab_size]
 *   d_out_idx  — uint32 output indices [k]
 *   d_out_val  — F32 output logits [k]
 *   vocab_size — number of logits
 *   k          — number of top elements to return (≤ 1024)
 */
bool oc_cuda_topk(
    const float *d_logits, uint32_t *d_out_idx, float *d_out_val,
    uint32_t vocab_size, uint32_t k);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OXIDIZE_CUDA_KERNELS_H */
