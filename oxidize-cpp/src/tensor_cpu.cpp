// CPU tensor kernels for Llama/Mistral/Qwen dense decode.
//
// Ported from:
//   oxidize-core/src/compute/tensor/kernels.rs
//     rms_norm_f32 (L5704), apply_rope_f32 (L5491), apply_swiglu_inplace_f32
//     (L5603), apply_geglu_inplace_f32 (L5665), softmax_f32 (L5938),
//     gemv_f32_cpu (L5216), gemv_quantized_f32 (L1611, here via dequant rows).
//   oxidize-core/src/compute/flash_attention.rs
//     flash_attention_decode_impl (L460), flash_attention_decode_heads_impl
//     (L623) — causal, GQA-aware online softmax.
//   oxidize-core/src/model/layer_wise.rs (L2270-2309) partial per-head RoPE.
//
// Numerically faithful to the Rust scalar reference paths. Heavy loops use
// OpenMP. Throws std::runtime_error on hard errors (the CLI catches).

#include "oxidize/tensor.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "oxidize/quant.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace oxidize {

namespace {

// Scalar f32 dot product (matches the non-SIMD fallback in kernels.rs and the
// reference math the SIMD dot kernels are validated against).
inline float dot_f32(const float* a, const float* b, size_t n) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

}  // namespace

void rms_norm(float* out, const float* x, const float* weight, size_t n,
              float eps, bool weight_plus_one) {
  if (n == 0) {
    throw std::runtime_error("rms_norm: zero dimension");
  }
  // mean of squares -> inverse RMS (kernels.rs::rms_norm_f32 scalar path).
  float sum_sq = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    sum_sq += x[i] * x[i];
  }
  const float mean_sq = sum_sq / static_cast<float>(n);
  const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

  for (size_t i = 0; i < n; ++i) {
    // Gemma applies (weight + 1); standard Llama uses weight directly.
    const float scale = weight_plus_one ? (weight[i] + 1.0f) : weight[i];
    out[i] = x[i] * inv_rms * scale;
  }
}

void apply_rope(float* vec, size_t head_dim, size_t num_heads, size_t pos,
                float theta, size_t rope_dim) {
  if (head_dim == 0) {
    throw std::runtime_error("apply_rope: zero head_dim");
  }
  // Effective rotary width per head: full head_dim unless a smaller rope_dim is
  // requested (partial RoPE, e.g. MiniMax / some Qwen variants).
  size_t rope_len = (rope_dim == 0) ? head_dim : rope_dim;
  if (rope_len > head_dim) {
    rope_len = head_dim;
  }
  if (rope_len % 2 != 0) {
    throw std::runtime_error("apply_rope: odd rotary dimension " +
                             std::to_string(rope_len));
  }
  if (pos == 0) {
    // apply_rope_f32 returns the input unchanged at position 0.
    return;
  }
  if (rope_len == 0) {
    return;
  }

  const float position_f = static_cast<float>(pos);
  const size_t half_dim = rope_len / 2;
  const float inv_rope_len = 1.0f / static_cast<float>(rope_len);
  // freq_{i+1} = freq_i * theta^(-2/rope_len) geometric recurrence (matches the
  // powf recurrence in apply_rope_f32 to stay numerically identical).
  const float freq_multiplier = std::pow(theta, -2.0f * inv_rope_len);

  for (size_t head = 0; head < num_heads; ++head) {
    float* h = vec + head * head_dim;
    float freq = 1.0f;
    for (size_t i = 0; i < half_dim; ++i) {
      const float x0 = h[i];
      const float x1 = h[half_dim + i];
      const float angle = position_f * freq;
      const float cos_a = std::cos(angle);
      const float sin_a = std::sin(angle);
      h[i] = x0 * cos_a - x1 * sin_a;
      h[half_dim + i] = x0 * sin_a + x1 * cos_a;
      freq *= freq_multiplier;
    }
    // Dims [rope_len, head_dim) are left untouched (partial RoPE pass-through).
  }
}

void swiglu_inplace(float* gate, const float* up, float* out, size_t n) {
  // silu(g) * up = g * sigmoid(g) * up (kernels.rs::apply_swiglu_inplace_f32
  // scalar path). out may alias gate.
  for (size_t i = 0; i < n; ++i) {
    const float g = gate[i];
    const float sigmoid = 1.0f / (1.0f + std::exp(-g));
    out[i] = g * sigmoid * up[i];
  }
}

void geglu_inplace(float* gate, const float* up, float* out, size_t n) {
  // gelu_tanh(g) * up (kernels.rs::apply_geglu_inplace_f32). K = sqrt(2/pi).
  constexpr float K = 0.797884560f;
  for (size_t i = 0; i < n; ++i) {
    const float g = gate[i];
    const float gelu =
        0.5f * g * (1.0f + std::tanh(K * (g + 0.044715f * g * g * g)));
    out[i] = gelu * up[i];
  }
}

void softmax_inplace(float* x, size_t n) {
  if (n == 0) {
    return;
  }
  // Stable softmax with f64 sum accumulation (kernels.rs::softmax_f32).
  float max_input = x[0];
  for (size_t i = 1; i < n; ++i) {
    if (x[i] > max_input) {
      max_input = x[i];
    }
  }
  double sum_exp = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const float e = std::exp(x[i] - max_input);
    x[i] = e;
    sum_exp += static_cast<double>(e);
  }
  const float inv_sum = static_cast<float>(1.0 / sum_exp);
  for (size_t i = 0; i < n; ++i) {
    x[i] *= inv_sum;
  }
}

void matvec(float* y, const float* W, const float* x, size_t rows,
            size_t cols) {
  // y[r] = dot(W[r, :], x) (kernels.rs::gemv_f32_cpu).
#pragma omp parallel for schedule(static)
  for (long long r = 0; r < static_cast<long long>(rows); ++r) {
    const float* row = W + static_cast<size_t>(r) * cols;
    y[r] = dot_f32(row, x, cols);
  }
}

void gemv_quantized(float* y, QuantType quant, const uint8_t* W, size_t rows,
                    size_t cols, const float* x) {
  // Per-row dequant then dot: y[r] = dot(dequant(W_row_r), x). This matches the
  // semantics of gemv_quantized_f32 (the Rust fused kernels are a perf
  // specialization of exactly this). dequantize_row enforces block layout.
  const size_t row_bytes = quantized_size(quant, cols);

#pragma omp parallel
  {
    std::vector<float> row(cols);
#pragma omp for schedule(static)
    for (long long r = 0; r < static_cast<long long>(rows); ++r) {
      const uint8_t* src = W + static_cast<size_t>(r) * row_bytes;
      dequantize_row(quant, src, row.data(), cols);
      y[r] = dot_f32(row.data(), x, cols);
    }
  }
}

void attention_decode(float* out, const float* q, const float* k_cache,
                      const float* v_cache, size_t seq_len, size_t num_heads,
                      size_t kv_heads, size_t head_dim) {
  if (head_dim == 0) {
    throw std::runtime_error("attention_decode: zero head_dim");
  }
  if (kv_heads == 0 || num_heads % kv_heads != 0) {
    throw std::runtime_error(
        "attention_decode: num_heads not divisible by kv_heads (" +
        std::to_string(num_heads) + " / " + std::to_string(kv_heads) + ")");
  }
  const size_t group_size = num_heads / kv_heads;
  // Each cache row holds all kv heads contiguously: stride = kv_heads*head_dim.
  const size_t kv_len = kv_heads * head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  if (seq_len == 0) {
    for (size_t i = 0; i < num_heads * head_dim; ++i) {
      out[i] = 0.0f;
    }
    return;
  }

#pragma omp parallel for schedule(static)
  for (long long h = 0; h < static_cast<long long>(num_heads); ++h) {
    const size_t head = static_cast<size_t>(h);
    const size_t kv_head = head / group_size;
    const size_t kv_offset = kv_head * head_dim;
    const float* q_head = q + head * head_dim;
    float* out_head = out + head * head_dim;

    // Online softmax with running max/sum (flash_attention_decode_impl).
    float running_max = -std::numeric_limits<float>::infinity();
    float running_sum = 0.0f;
    for (size_t d = 0; d < head_dim; ++d) {
      out_head[d] = 0.0f;
    }

    for (size_t t = 0; t < seq_len; ++t) {
      const float* key_row = k_cache + t * kv_len + kv_offset;
      float score = dot_f32(q_head, key_row, head_dim) * scale;

      const float new_max = running_max > score ? running_max : score;
      const float exp_factor = std::exp(running_max - new_max);
      const float exp_score = std::exp(score - new_max);

      if (exp_factor != 1.0f) {
        for (size_t d = 0; d < head_dim; ++d) {
          out_head[d] *= exp_factor;
        }
      }

      const float* value_row = v_cache + t * kv_len + kv_offset;
      for (size_t d = 0; d < head_dim; ++d) {
        out_head[d] += exp_score * value_row[d];
      }

      running_sum = running_sum * exp_factor + exp_score;
      running_max = new_max;
    }

    if (running_sum > 0.0f) {
      const float inv_sum = 1.0f / running_sum;
      for (size_t d = 0; d < head_dim; ++d) {
        out_head[d] *= inv_sum;
      }
    }
  }
}

}  // namespace oxidize
