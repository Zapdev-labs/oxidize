/*
 * activation.h — RMSNorm, RoPE, SwiGLU, and related activation kernels.
 *
 * Port of oxidize-core/src/compute/tensor/kernels/activation.rs:
 *   - rms_norm_f32          (RMSNorm with eps)
 *   - apply_rope_f32_yarn   (split-halves NeoX-style RoPE, YaRN-capable)
 *   - apply_swiglu_inplace_f32 (SwiGLU: silu(x)*y)
 *
 * Bit-exactness with the Rust reference is the hard invariant (VAL-FWD-001..
 * 004). These are the scalar reference paths; SIMD-accelerated variants are
 * layered on later by the quant-simd-dispatch feature.
 */
#ifndef OXIDIZE_ACTIVATION_H
#define OXIDIZE_ACTIVATION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* RMSNorm: out[i] = x[i] * inv_rms * weight[i], where
 *   inv_rms = 1 / sqrt(mean(x^2) + eps), mean over `n` elements.
 * `x`, `weight`, `out` are length `n`. Mirrors Rust `rms_norm_f32`. */
void oc_rms_norm_f32(const float *x, const float *weight, float *out,
                     size_t n, float eps);

/* SwiGLU in-place: gate[i] = silu(gate[i]) * up[i], where silu(x)=x*sigmoid(x).
 * `gate` is modified in place; `up` is read-only. Length `n`. Mirrors Rust
 * `apply_swiglu_inplace_f32`. Uses the Schraudolph-tolerant exp from <math.h>
 * (exact sigmoid; no fast-approx here — parity-first). */
void oc_swiglu_inplace_f32(float *gate, const float *up, size_t n);

/* GeGLU in-place: gate[i] = gelu(gate[i]) * up[i].
 * Uses the exact (erf-based) GeLU, matching the Rust reference.
 * Used by Gemma's FFN (instead of SwiGLU). */
void oc_geglu_inplace_f32(float *gate, const float *up, size_t n);

/* Approximate GeLU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
 * Used by Gemma2/Gemma3 attention output and FFN gate when configured. */
float oc_gelu_approx_f32(float x);

/* Exact GeLU: 0.5 * x * (1 + erf(x / sqrt(2))). */
float oc_gelu_exact_f32(float x);

/* Apply RoPE (Rotary Positional Embedding) to one head of length `head_dim`
 * at absolute `position`. Split-halves (NeoX-style) layout:
 *   half = head_dim / 2
 *   freq_i = theta ^ (-2*i / head_dim),  i = 0..half-1
 *   angle_i = position * freq_i
 *   out[i]          = x[i]        * cos(angle_i) - x[half+i] * sin(angle_i)
 *   out[half+i]     = x[i]        * sin(angle_i) + x[half+i] * cos(angle_i)
 * Only the first `rope_len` elements are rotated when `rope_len < head_dim`
 * (partial RoPE, MiniMax/Qwen3.5); the remainder pass through unchanged.
 * `in` and `out` may alias. `rope_len` is capped to head_dim internally.
 *
 * Position 0 fast path: copies input unchanged (no rotation needed).
 *
 * YaRN scaling is NOT yet ported (yarn_factor/yarn_orig_ctx paths); this is
 * the no-scaling path only. TODO: add YaRN when porting DeepSeek/long-context. */
void oc_apply_rope_f32(const float *in, float *out, size_t head_dim,
                      size_t rope_len, int64_t position, float theta);

/* Apply RoPE with YaRN long-context scaling.
 * YaRN: scale = yarn_factor * (orig_ctx / position) when position > orig_ctx,
 * with a smooth interpolation between [orig_ctx * 0.8, orig_ctx * 1.2].
 * When yarn_factor == 0 or position <= yarn_orig_ctx, behaves as normal RoPE. */
void oc_apply_rope_yarn_f32(const float *in, float *out, size_t head_dim,
                             size_t rope_len, int64_t position, float theta,
                             float yarn_factor, uint32_t yarn_orig_ctx);

/* Softmax: out[i] = exp(x[i] - max) / sum(exp(x[j] - max)).
 * Numerically stable (subtracts max before exp). Uses f64 accumulator. */
void oc_softmax_f32(const float *input, float *output, size_t n);

/* LayerNorm: out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i],
 * where mean = sum(x)/n, var = sum((x-mean)^2)/n, inv_std = 1/sqrt(var+eps). */
void oc_layer_norm_f32(const float *input, const float *weight, const float *bias,
                        float *output, size_t n, float eps);

/* SwiGLU (non-inplace): out[i] = silu(gate[i]) * up[i].
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x)). */
void oc_swiglu_f32(const float *gate, const float *up, float *output, size_t n);

/* Scaled dot-product attention for a single query against seq_len key/value pairs.
 * query: [dim], key: [seq_len * dim], value: [seq_len * dim], output: [dim].
 * scale = 1/sqrt(dim). Uses online softmax for numerical stability. */
void oc_scaled_dot_product_attention_f32(const float *query,
                                          const float *key,
                                          const float *value,
                                          size_t seq_len, size_t dim,
                                          float *output);

/* RMSNorm variant for Qwen: out[i] = x[i] * inv_rms * (1 + weight[i]).
 * When weight_plus_one is false, behaves as standard rms_norm. */
void oc_rms_norm_f32_qwen(const float *x, const float *weight, float *out,
                           size_t n, float eps, bool weight_plus_one);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ACTIVATION_H */
