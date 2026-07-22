/*
 * tensor_ops.h — High-level tensor operations for the C port.
 *
 * Port of oxidize-core/src/compute/tensor.rs operations that are not
 * covered by the existing matvec/quantization modules. Provides:
 *   - Element-wise ops (add, mul, scale, exp, tanh)
 *   - Reductions (sum, max, argmax, norm)
 *   - Copy/transpose/concat
 *   - Softmax (online + naive)
 *   - Layer norm / RMS norm (f32 + quantized)
 *   - Rotary position embedding (RoPE) variants
 *   - GEMM (f32 × f32, quantized × f32)
 *
 * All operations work on flat f32 buffers (row-major) unless otherwise
 * noted. No tensor descriptor struct — just raw pointers + dims, keeping
 * with the C port's "no abstraction" convention.
 */
#ifndef OXIDIZE_TENSOR_OPS_H
#define OXIDIZE_TENSOR_OPS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Element-wise operations ──────────────────────────────────────────── */

/* out = a + b (element-wise). n elements. */
void oc_tensor_add_f32(const float *a, const float *b, float *out, size_t n);

/* out = a * b (element-wise). */
void oc_tensor_mul_f32(const float *a, const float *b, float *out, size_t n);

/* out = a * scale. */
void oc_tensor_scale_f32(const float *a, float scale, float *out, size_t n);

/* out = a + scale. */
void oc_tensor_add_scalar_f32(const float *a, float scalar, float *out, size_t n);

/* out = exp(a). */
void oc_tensor_exp_f32(const float *a, float *out, size_t n);

/* out = tanh(a). */
void oc_tensor_tanh_f32(const float *a, float *out, size_t n);

/* out = sigmoid(a) = 1 / (1 + exp(-a)). */
void oc_tensor_sigmoid_f32(const float *a, float *out, size_t n);

/* out = silu(a) = a * sigmoid(a). */
void oc_tensor_silu_f32(const float *a, float *out, size_t n);

/* out = gelu(a) (tanh approximation). */
void oc_tensor_gelu_f32(const float *a, float *out, size_t n);

/* out = relu(a) = max(a, 0). */
void oc_tensor_relu_f32(const float *a, float *out, size_t n);

/* In-place: a[i] = a[i] + b[i]. */
void oc_tensor_iadd_f32(float *a, const float *b, size_t n);

/* In-place: a[i] = a[i] * b[i]. */
void oc_tensor_imul_f32(float *a, const float *b, size_t n);

/* In-place: a[i] = a[i] * scale. */
void oc_tensor_iscale_f32(float *a, float scale, size_t n);

/* ─── Reductions ────────────────────────────────────────────────────────── */

/* Sum of n elements. */
float oc_tensor_sum_f32(const float *a, size_t n);

/* Max of n elements. */
float oc_tensor_max_f32(const float *a, size_t n);

/* Argmax: index of max element. Returns 0 for n=0. */
size_t oc_tensor_argmax_f32(const float *a, size_t n);

/* L2 norm: sqrt(sum(a[i]^2)). */
float oc_tensor_l2_norm_f32(const float *a, size_t n);

/* Mean: sum / n. */
float oc_tensor_mean_f32(const float *a, size_t n);

/* Variance: sum((a[i] - mean)^2) / n. */
float oc_tensor_variance_f32(const float *a, size_t n);

/* ─── Copy / transpose / concat ─────────────────────────────────────────── */

/* Copy n elements. */
void oc_tensor_copy_f32(const float *src, float *dst, size_t n);

/* Transpose a 2D matrix [rows × cols] → [cols × rows]. */
void oc_tensor_transpose_f32(const float *src, float *dst,
                              size_t rows, size_t cols);

/* Concatenate two arrays along the last dimension.
 * a: [m × n_a], b: [m × n_b], out: [m × (n_a + n_b)] */
void oc_tensor_concat_f32(const float *a, const float *b, float *out,
                           size_t m, size_t n_a, size_t n_b);

/* Repeat a row vector [n] into [m × n] (row-major). */
void oc_tensor_repeat_row_f32(const float *row, float *out,
                                size_t m, size_t n);

/* ─── Softmax ───────────────────────────────────────────────────────────── */

/* Naive softmax: out = softmax(a). n elements. */
void oc_tensor_softmax_f32(const float *a, float *out, size_t n);

/* Online softmax (flash attention style): single pass, O(1) extra memory. */
void oc_tensor_softmax_online_f32(const float *a, float *out, size_t n);

/* Softmax with temperature: out = softmax(a / temperature). */
void oc_tensor_softmax_temp_f32(const float *a, float *out, size_t n, float temp);

/* Log-softmax: out[i] = a[i] - logsumexp(a). */
void oc_tensor_log_softmax_f32(const float *a, float *out, size_t n);

/* ─── Normalization ──────────────────────────────────────────────────────── */

/* LayerNorm: out = (a - mean) / sqrt(var + eps) * weight + bias.
 * a, out: [n], weight: [n], bias: [n] (or NULL). */
void oc_tensor_layer_norm_f32(const float *a, const float *weight,
                               const float *bias, float *out,
                               size_t n, float eps);

/* RMSNorm: out = a / sqrt(mean(a^2) + eps) * weight. */
void oc_tensor_rms_norm_f32(const float *a, const float *weight,
                              float *out, size_t n, float eps);

/* ─── Rotary position embedding (RoPE) ───────────────────────────────────── */

/* Apply RoPE (GPT-NeoX style, split halves) to a single head.
 * x: [head_dim], position: token position, freq_base: theta. */
void oc_tensor_rope_neox_f32(float *x, size_t head_dim,
                              uint32_t position, float freq_base);

/* Apply RoPE (GPT-J style, interleaved) to a single head. */
void oc_tensor_rope_gptj_f32(float *x, size_t head_dim,
                              uint32_t position, float freq_base);

/* Apply RoPE to all heads in a row.
 * x: [n_head * head_dim], position, freq_base. */
void oc_tensor_rope_neox_row_f32(float *x, size_t n_head, size_t head_dim,
                                   uint32_t position, float freq_base);

/* ─── GEMM ──────────────────────────────────────────────────────────────── */

/* C = A @ B where A: [M×K], B: [K×N], C: [M×N] (row-major). */
void oc_tensor_gemm_f32(const float *A, const float *B, float *C,
                         size_t M, size_t K, size_t N);

/* C = A^T @ B where A: [K×M], B: [K×N], C: [M×N]. */
void oc_tensor_gemm_at_f32(const float *A, const float *B, float *C,
                            size_t M, size_t K, size_t N);

/* Batched GEMM: batch × [M×K] @ [K×N] → batch × [M×N]. */
void oc_tensor_gemm_batch_f32(const float *A, const float *B, float *C,
                                size_t batch, size_t M, size_t K, size_t N);

/* ─── Attention helpers ────────────────────────────────────────────────── */

/* Scaled dot-product attention for a single head.
 * Q: [d_head], K: [seq_len × d_head], V: [seq_len × d_head]
 * out: [d_head], scale: 1/sqrt(d_head) */
void oc_tensor_attention_head_f32(const float *Q, const float *K,
                                    const float *V, float *out,
                                    size_t seq_len, size_t d_head, float scale);

/* Multi-head attention with GQA.
 * Q: [n_head_q × d_head], K: [n_head_kv × seq_len × d_head], V: same.
 * out: [n_head_q × d_head] */
void oc_tensor_attention_mha_f32(const float *Q, const float *K,
                                   const float *V, float *out,
                                   size_t n_head_q, size_t n_head_kv,
                                   size_t seq_len, size_t d_head);

/* ─── Utility ───────────────────────────────────────────────────────────── */

/* Fill with constant. */
void oc_tensor_fill_f32(float *a, float val, size_t n);

/* Fill with zeros. */
void oc_tensor_zero_f32(float *a, size_t n);

/* Fill with random uniform [0, 1). Uses a simple LCG (no libm dependency). */
void oc_tensor_random_f32(float *a, size_t n, uint32_t seed);

/* Print a 1D tensor to stderr (first n elements). */
void oc_tensor_print_f32(const float *a, size_t n, const char *name);

/* Print a 2D tensor [rows × cols] to stderr. */
void oc_tensor_print_2d_f32(const float *a, size_t rows, size_t cols,
                             const char *name);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_TENSOR_OPS_H */
