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

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ACTIVATION_H */
