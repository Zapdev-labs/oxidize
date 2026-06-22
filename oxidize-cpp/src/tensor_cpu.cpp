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

#if defined(__F16C__) && defined(__AVX2__)
#include <immintrin.h>
#define OXIDIZE_HAVE_F16C 1
#endif

namespace oxidize {

namespace {

// f32 dot product with 8 independent accumulators. A single-accumulator float
// reduction is NOT auto-vectorizable (float add is non-associative, so the
// compiler must keep it scalar without -ffast-math). Eight parallel reduction
// chains let -O3 -march=native emit AVX/FMA, ~4-8x faster than the scalar loop,
// while staying deterministic (fixed reduction tree -> identical across runs).
inline float dot_f32(const float* __restrict a, const float* __restrict b,
                     size_t n) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    s0 += a[i + 0] * b[i + 0];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
    s4 += a[i + 4] * b[i + 4];
    s5 += a[i + 5] * b[i + 5];
    s6 += a[i + 6] * b[i + 6];
    s7 += a[i + 7] * b[i + 7];
  }
  float sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
  for (; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

// Fused f16(weight) . f32(activation) dot. Keeping weights as f16 halves the
// memory traffic of a memory-bound decode; F16C converts 8 halves -> f32 in one
// instruction so there is no separate (scalar) dequant pass. `w` is a row of
// little-endian IEEE half precision values; `x` is f32.
inline float dot_f16(const uint16_t* __restrict w, const float* __restrict x,
                     size_t n) {
#ifdef OXIDIZE_HAVE_F16C
  // Four independent accumulators hide the ~4-cycle FMA latency: a 2-wide chain
  // bottlenecks on the dependency before memory bandwidth (measured ~1.5x on the
  // F16 decode hot path).
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256 w0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i)));
    __m256 w1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + 8)));
    __m256 w2 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + 16)));
    __m256 w3 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + 24)));
    acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + i), acc0);
    acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + i + 8), acc1);
    acc2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x + i + 16), acc2);
    acc3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x + i + 24), acc3);
  }
  for (; i + 8 <= n; i += 8) {
    __m256 w0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i)));
    acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + i), acc0);
  }
  __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  float sum = _mm_cvtss_f32(s);
  for (; i < n; ++i) sum += f16_le_to_f32(reinterpret_cast<const uint8_t*>(w + i)) * x[i];
  return sum;
#else
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) sum += f16_le_to_f32(reinterpret_cast<const uint8_t*>(w + i)) * x[i];
  return sum;
#endif
}

#ifdef OXIDIZE_HAVE_F16C
// Horizontal sum of a __m256.
inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}
#endif

// Fused Q4_0 row . f32 dot (block: f16 d + 16B nibbles, ggml split-halves:
// low nibble of byte j -> out j, high -> out j+16; value = (nibble-8)*d).
inline float dot_q4_0(const uint8_t* __restrict row, const float* __restrict x,
                      size_t cols) {
  const size_t nb = cols / 32;
#ifdef OXIDIZE_HAVE_F16C
  __m256 acc = _mm256_setzero_ps();
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* blk = row + b * 18;
    const float* xb = x + b * 32;
    __m256 vd = _mm256_set1_ps(f16_le_to_f32(blk));
    const uint8_t* qs = blk + 2;
    for (int g = 0; g < 4; ++g) {
      int half = g / 2, sub = g % 2, base = half * 16 + sub * 8;
      __m256i n32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
          reinterpret_cast<const __m128i*>(qs + sub * 8)));
      __m256i nib = half == 0 ? _mm256_and_si256(n32, _mm256_set1_epi32(0x0F))
                              : _mm256_srli_epi32(n32, 4);
      __m256i q = _mm256_sub_epi32(nib, _mm256_set1_epi32(8));
      __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(q), vd);
      acc = _mm256_fmadd_ps(f, _mm256_loadu_ps(xb + base), acc);
    }
  }
  return hsum256(acc);
#else
  float sum = 0.0f;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* blk = row + b * 18;
    float d = f16_le_to_f32(blk);
    const uint8_t* qs = blk + 2;
    for (int j = 0; j < 16; ++j) {
      sum += (static_cast<int>(qs[j] & 0x0F) - 8) * d * x[b * 32 + j];
      sum += (static_cast<int>(qs[j] >> 4) - 8) * d * x[b * 32 + j + 16];
    }
  }
  return sum;
#endif
}

// Fused Q5_0 row . f32 dot (block: f16 d + u32 qh + 16B nibbles; value =
// ((nibble | (bit_of_qh<<4)) - 16) * d, ggml split-halves with bit j<->out j,
// bit j+16<->out j+16).
inline float dot_q5_0(const uint8_t* __restrict row, const float* __restrict x,
                      size_t cols) {
  const size_t nb = cols / 32;
#ifdef OXIDIZE_HAVE_F16C
  __m256 acc = _mm256_setzero_ps();
  const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* blk = row + b * 22;
    const float* xb = x + b * 32;
    __m256 vd = _mm256_set1_ps(f16_le_to_f32(blk));
    uint32_t qh = static_cast<uint32_t>(blk[2]) | (static_cast<uint32_t>(blk[3]) << 8) |
                  (static_cast<uint32_t>(blk[4]) << 16) | (static_cast<uint32_t>(blk[5]) << 24);
    const uint8_t* qs = blk + 6;
    __m256i qhv = _mm256_set1_epi32(static_cast<int>(qh));
    for (int g = 0; g < 4; ++g) {
      int half = g / 2, sub = g % 2, base = half * 16 + sub * 8;
      __m256i n32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
          reinterpret_cast<const __m128i*>(qs + sub * 8)));
      __m256i nib = half == 0 ? _mm256_and_si256(n32, _mm256_set1_epi32(0x0F))
                              : _mm256_srli_epi32(n32, 4);
      __m256i bitidx = _mm256_add_epi32(_mm256_set1_epi32(base), lane);
      __m256i mask = _mm256_sllv_epi32(_mm256_set1_epi32(1), bitidx);
      __m256i isset = _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(qhv, mask), _mm256_setzero_si256()),
          _mm256_set1_epi32(-1));
      __m256i fifth = _mm256_and_si256(isset, _mm256_set1_epi32(16));
      __m256i q = _mm256_sub_epi32(_mm256_add_epi32(nib, fifth), _mm256_set1_epi32(16));
      __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(q), vd);
      acc = _mm256_fmadd_ps(f, _mm256_loadu_ps(xb + base), acc);
    }
  }
  return hsum256(acc);
#else
  float sum = 0.0f;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* blk = row + b * 22;
    float d = f16_le_to_f32(blk);
    uint32_t qh = static_cast<uint32_t>(blk[2]) | (static_cast<uint32_t>(blk[3]) << 8) |
                  (static_cast<uint32_t>(blk[4]) << 16) | (static_cast<uint32_t>(blk[5]) << 24);
    const uint8_t* qs = blk + 6;
    for (int j = 0; j < 16; ++j) {
      int lo = (qs[j] & 0x0F) | (((qh >> j) & 1) << 4);
      int hi = (qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
      sum += (lo - 16) * d * x[b * 32 + j];
      sum += (hi - 16) * d * x[b * 32 + j + 16];
    }
  }
  return sum;
#endif
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
  // Fast path: native F16 weights. Fused F16C convert+FMA dot, no dequant pass.
  if (quant == QuantType::F16) {
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(rows); ++r) {
      const uint16_t* row = reinterpret_cast<const uint16_t*>(
          W + static_cast<size_t>(r) * cols * 2);
      y[r] = dot_f16(row, x, cols);
    }
    return;
  }
  // Fast paths: native Q4_0 / Q5_0 fused unpack+dot (SIMD), no scalar dequant pass.
  if (quant == QuantType::Q4_0) {
    const size_t rb = (cols / 32) * 18;
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(rows); ++r)
      y[r] = dot_q4_0(W + static_cast<size_t>(r) * rb, x, cols);
    return;
  }
  if (quant == QuantType::Q5_0) {
    const size_t rb = (cols / 32) * 22;
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(rows); ++r)
      y[r] = dot_q5_0(W + static_cast<size_t>(r) * rb, x, cols);
    return;
  }

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
