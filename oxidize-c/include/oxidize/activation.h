/* activation.h — RMSNorm, RoPE, SwiGLU. Bit-exact with Rust (VAL-FWD-001). */
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

/* SwiGLU in-place: gate[i] = silu(gate[i]) * up[i], where silu(x)=x*sigmoid(x). */
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

/* Apply RoPE (Rotary Positional Embedding) to one head of length `head_dim` at absolute `position`. */
void oc_apply_rope_f32(const float *in, float *out, size_t head_dim,
                      size_t rope_len, int64_t position, float theta);

/* Interleaved ("NORM") RoPE — rotates (2i, 2i+1) instead of (i, i+half). Same signature and aliasing rules as oc_apply_rope_f32; the two are not interchangeable. */
void oc_apply_rope_norm_f32(const float *in, float *out, size_t head_dim,
                            size_t rope_len, int64_t position, float theta);

/* Apply RoPE with YaRN long-context scaling. */
/* As oc_apply_rope_yarn_f32, but with an explicit cos/sin amplitude. */
void oc_apply_rope_yarn_scaled_f32(const float *in, float *out, size_t head_dim,
                                    size_t rope_len, int64_t position,
                                    float theta, float yarn_factor,
                                    uint32_t yarn_orig_ctx, float attn_factor);

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
