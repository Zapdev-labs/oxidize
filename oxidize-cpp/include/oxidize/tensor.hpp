#pragma once
// CPU tensor kernels for Llama/Mistral/Qwen dense decode.
//
// Ported from:
//   oxidize-core/src/compute/tensor/kernels.rs
//     - rms_norm_f32, apply_rope_f32, apply_swiglu_inplace_f32,
//       apply_geglu_inplace_f32, softmax_f32, gemv_f32, gemv_quantized_f32
//   oxidize-core/src/compute/flash_attention.rs
//     - flash_attention_decode_impl / _heads (causal, GQA-aware, online softmax)
//   oxidize-core/src/model/layer_wise.rs (partial-RoPE per-head application)
//
// Math is numerically faithful to the Rust scalar paths (the Rust SIMD fast
// paths are bit-different from each other by design; we port the scalar
// reference math, which the Rust SIMD paths are validated against). Quantized
// GEMV decodes each weight row via quant.hpp::dequantize_row and dots it with
// the input vector, matching `dequant(W) * x`.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "oxidize/config.hpp"

namespace oxidize {

// out[i] = x[i] * inv_rms * scale_i
//   inv_rms = 1 / sqrt(mean(x^2) + eps)
//   scale_i = weight_plus_one ? (weight[i] + 1) : weight[i]
// All buffers length n. Throws std::runtime_error on length/zero-dim errors.
void rms_norm(float* out, const float* x, const float* weight, size_t n,
              float eps, bool weight_plus_one);

// In-place rotary position embedding over a [num_heads * head_dim] vector.
// Partial RoPE: only the first `rope_dim` dims of each head are rotated (split
// into two halves of rope_dim/2), the remaining dims pass through unchanged.
// `rope_dim == 0` => rotate the full head_dim. Matches apply_rope_f32 +
// layer_wise.rs per-head application. Throws on odd effective rope width.
void apply_rope(float* vec, size_t head_dim, size_t num_heads, size_t pos,
                float theta, size_t rope_dim);

// In-place adjacent-pair rotary position embedding (LLAMA_ROPE_TYPE_NORM).
// Used by GLM-DSA / DeepSeek-V2 MLA. Rotates pairs (h[2i], h[2i+1]) with
// angle = pos * theta^(-2i/rope_len). Same partial-RoPE semantics as
// apply_rope (rope_dim==0 => full head_dim; dims >= rope_len pass through).
// Do NOT use for Qwen/GPT-NeoX models — they use apply_rope (split-half).
void apply_rope_norm(float* vec, size_t head_dim, size_t num_heads, size_t pos,
                     float theta, size_t rope_dim);

// out[i] = silu(gate[i]) * up[i],  silu(x) = x * sigmoid(x).
// May write in place (out == gate is allowed). Length n each.
void swiglu_inplace(float* gate, const float* up, float* out, size_t n);

// out[i] = gelu_tanh(gate[i]) * up[i] (Gemma gelu_pytorch_tanh). Length n each.
void geglu_inplace(float* gate, const float* up, float* out, size_t n);

// In-place numerically-stable softmax over n elements.
void softmax_inplace(float* x, size_t n);

// y = W * x, W row-major [rows x cols], x length cols, y length rows.
void matvec(float* y, const float* W, const float* x, size_t rows, size_t cols);

// y = dequant(W) * x. W is a tightly packed block sequence of `quant` rows,
// each row holding `cols` scalar values. y length rows, x length cols.
void gemv_quantized(float* y, QuantType quant, const uint8_t* W, size_t rows,
                    size_t cols, const float* x);

// Batched GEMM: outputs[b*rows + r] = dot(W[r,:], inputs[b*cols + :]) for
// b in [0,batch). W is row-major [rows x cols]; inputs/outputs are
// row-major [batch x cols] / [batch x rows]. For quantized W each weight row is
// decoded once and dotted against every batch position (llama.cpp-style prefill).
void gemm_quantized(float* outputs, QuantType quant, const uint8_t* W,
                    size_t rows, size_t cols, const float* inputs, size_t batch);

// Single decode-step attention over a KV cache (causal, GQA-aware).
//   q          : [num_heads * head_dim] query for the current position
//   k_cache    : [seq_len * kv_heads * head_dim] keys   (row per position)
//   v_cache    : [seq_len * kv_heads * head_dim] values (row per position)
//   out        : [num_heads * head_dim] attention output
//   seq_len    : number of valid KV positions (= pos + 1)
// kv_heads must divide num_heads (grouped-query attention). Online softmax with
// scale 1/sqrt(head_dim). Matches flash_attention_decode_impl.
void attention_decode(float* out, const float* q, const float* k_cache,
                      const float* v_cache, size_t seq_len, size_t num_heads,
                      size_t kv_heads, size_t head_dim);

}  // namespace oxidize
