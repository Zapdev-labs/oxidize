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
#include <cstring>
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

#ifdef OXIDIZE_HAVE_F16C
inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}
#endif

// f32 dot product with 8 independent accumulators.
// reduction is NOT auto-vectorizable (float add is non-associative, so the
// compiler must keep it scalar without -ffast-math). Eight parallel reduction
// chains let -O3 -march=native emit AVX/FMA, ~4-8x faster than the scalar loop,
// while staying deterministic (fixed reduction tree -> identical across runs).
inline float dot_f32(const float* __restrict a, const float* __restrict b,
                     size_t n) {
#ifdef OXIDIZE_HAVE_F16C
  if (n >= 8 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
      acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
      acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
      acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
      acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float sum = hsum256(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
  }
#endif
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

inline void get_scale_min_k4(size_t j, const uint8_t* scales, uint8_t& sc,
                             uint8_t& mn) {
  if (j < 4) {
    sc = scales[j] & 63;
    mn = scales[j + 4] & 63;
  } else {
    sc = (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4);
    mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
  }
}

inline float q4_k_dot_scalar(const uint8_t* block, const float* vector) {
  float d = f16_le_to_f32(block);
  float min = f16_le_to_f32(block + 2);
  const uint8_t* scales = block + 4;
  const uint8_t* qs = block + 16;
  float sum = 0.0f;
  for (int group_pair = 0; group_pair < 4; ++group_pair) {
    const int group1 = group_pair * 2;
    const int group2 = group1 + 1;
    uint8_t scale1, min_scale1, scale2, min_scale2;
    get_scale_min_k4(static_cast<size_t>(group1), scales, scale1, min_scale1);
    get_scale_min_k4(static_cast<size_t>(group2), scales, scale2, min_scale2);
    const float d1 = d * static_cast<float>(scale1);
    const float min1 = min * static_cast<float>(min_scale1);
    const float d2 = d * static_cast<float>(scale2);
    const float min2 = min * static_cast<float>(min_scale2);
    const size_t q_base = static_cast<size_t>(group_pair) * 32;
    const size_t v_base = static_cast<size_t>(group_pair) * 64;
    for (size_t l = 0; l < 32; ++l) {
      const uint8_t packed = qs[q_base + l];
      sum += (d1 * static_cast<float>(packed & 0x0f) - min1) * vector[v_base + l];
      sum += (d2 * static_cast<float>(packed >> 4) - min2) * vector[v_base + 32 + l];
    }
  }
  return sum;
}

inline float dot_q4_k(const uint8_t* row, const float* x, size_t cols) {
  const size_t blocks_per_row = cols / QK_K;
#ifdef OXIDIZE_HAVE_F16C
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
    const __m128i mask = _mm_set1_epi8(0x0f);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
      const uint8_t* block = row + block_idx * BLOCK_Q4_K_SIZE;
      const float* v_ptr = x + block_idx * QK_K;
      const float d = f16_le_to_f32(block);
      const float min = f16_le_to_f32(block + 2);
      const uint8_t* scales = block + 4;
      const uint8_t* qs = block + 16;
      for (int group_pair = 0; group_pair < 4; ++group_pair) {
        const int group1 = group_pair * 2;
        const int group2 = group1 + 1;
        uint8_t scale1, min_scale1, scale2, min_scale2;
        get_scale_min_k4(static_cast<size_t>(group1), scales, scale1, min_scale1);
        get_scale_min_k4(static_cast<size_t>(group2), scales, scale2, min_scale2);
        const __m256 d1 = _mm256_set1_ps(d * static_cast<float>(scale1));
        const __m256 min1 = _mm256_set1_ps(min * static_cast<float>(min_scale1));
        const __m256 d2 = _mm256_set1_ps(d * static_cast<float>(scale2));
        const __m256 min2 = _mm256_set1_ps(min * static_cast<float>(min_scale2));
        const int q_base = group_pair * 32;
        const int v_base = group_pair * 64;
        const __m128i p0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + q_base));
        const __m128i p1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + q_base + 8));
        const __m128i p2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + q_base + 16));
        const __m128i p3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + q_base + 24));
        const __m256 l0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(p0, mask)));
        const __m256 h0 = _mm256_cvtepi32_ps(
            _mm256_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(p0, 4), mask)));
        const __m256 l1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(p1, mask)));
        const __m256 h1 = _mm256_cvtepi32_ps(
            _mm256_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(p1, 4), mask)));
        const __m256 l2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(p2, mask)));
        const __m256 h2 = _mm256_cvtepi32_ps(
            _mm256_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(p2, 4), mask)));
        const __m256 l3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(p3, mask)));
        const __m256 h3 = _mm256_cvtepi32_ps(
            _mm256_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(p3, 4), mask)));
        const __m256 t_l0 = _mm256_fmsub_ps(l0, d1, min1);
        const __m256 t_l1 = _mm256_fmsub_ps(l1, d1, min1);
        const __m256 t_l2 = _mm256_fmsub_ps(l2, d1, min1);
        const __m256 t_l3 = _mm256_fmsub_ps(l3, d1, min1);
        const __m256 t_h0 = _mm256_fmsub_ps(h0, d2, min2);
        const __m256 t_h1 = _mm256_fmsub_ps(h1, d2, min2);
        const __m256 t_h2 = _mm256_fmsub_ps(h2, d2, min2);
        const __m256 t_h3 = _mm256_fmsub_ps(h3, d2, min2);
        acc0 = _mm256_fmadd_ps(t_l0, _mm256_loadu_ps(v_ptr + v_base), acc0);
        acc1 = _mm256_fmadd_ps(t_l1, _mm256_loadu_ps(v_ptr + v_base + 8), acc1);
        acc2 = _mm256_fmadd_ps(t_l2, _mm256_loadu_ps(v_ptr + v_base + 16), acc2);
        acc3 = _mm256_fmadd_ps(t_l3, _mm256_loadu_ps(v_ptr + v_base + 24), acc3);
        acc0 = _mm256_fmadd_ps(t_h0, _mm256_loadu_ps(v_ptr + v_base + 32), acc0);
        acc1 = _mm256_fmadd_ps(t_h1, _mm256_loadu_ps(v_ptr + v_base + 40), acc1);
        acc2 = _mm256_fmadd_ps(t_h2, _mm256_loadu_ps(v_ptr + v_base + 48), acc2);
        acc3 = _mm256_fmadd_ps(t_h3, _mm256_loadu_ps(v_ptr + v_base + 56), acc3);
      }
    }
    const __m256 acc01 = _mm256_add_ps(acc0, acc1);
    const __m256 acc23 = _mm256_add_ps(acc2, acc3);
    const __m256 acc = _mm256_add_ps(acc01, acc23);
    const __m128 lo = _mm256_castps256_ps128(acc);
    const __m128 hi = _mm256_extractf128_ps(acc, 1);
    const __m128 sum128 = _mm_add_ps(lo, hi);
    const __m128 shuf = _mm_movehdup_ps(sum128);
    const __m128 sums = _mm_add_ps(sum128, shuf);
    const __m128 shuf2 = _mm_movehl_ps(shuf, sums);
    return _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
  }
#endif
  float sum = 0.0f;
  for (size_t bi = 0; bi < blocks_per_row; ++bi) {
    sum += q4_k_dot_scalar(row + bi * BLOCK_Q4_K_SIZE, x + bi * QK_K);
  }
  return sum;
}

inline void decode_q4_k_block(const uint8_t* block, float* out) {
  float d = f16_le_to_f32(block);
  float min = f16_le_to_f32(block + 2);
  const uint8_t* scales = block + 4;
  const uint8_t* qs = block + 16;
  for (int g = 0; g < 8; ++g) {
    uint8_t sc, m;
    get_scale_min_k4(static_cast<size_t>(g), scales, sc, m);
    const float dl = d * static_cast<float>(sc);
    const float ml = min * static_cast<float>(m);
    const int pair = g / 2;
    const size_t qs_off = static_cast<size_t>(pair) * 32;
    const size_t out_off = static_cast<size_t>(g) * 32;
    if ((g & 1) == 0) {
      for (size_t l = 0; l < 32; ++l) {
        out[out_off + l] = dl * static_cast<float>(qs[qs_off + l] & 0x0f) - ml;
      }
    } else {
      for (size_t l = 0; l < 32; ++l) {
        out[out_off + l] = dl * static_cast<float>(qs[qs_off + l] >> 4) - ml;
      }
    }
  }
}

inline float q8_0_dot_scalar(const uint8_t* block, const float* vector) {
  const float scale = f16_le_to_f32(block);
#ifdef OXIDIZE_HAVE_F16C
  __m256 acc = _mm256_setzero_ps();
  __m256 vs = _mm256_set1_ps(scale);
  for (int i = 0; i < 32; i += 8) {
    __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(block + 2 + i));
    __m256i q32 = _mm256_cvtepi8_epi32(q8);
    __m256 w = _mm256_mul_ps(_mm256_cvtepi32_ps(q32), vs);
    acc = _mm256_fmadd_ps(w, _mm256_loadu_ps(vector + i), acc);
  }
  return hsum256(acc);
#else
  for (size_t l = 0; l < QK8_0; ++l) {
    sum += static_cast<float>(static_cast<int8_t>(block[2 + l])) * scale * vector[l];
  }
  return sum;
#endif
}

inline float dot_q8_0(const uint8_t* row, const float* x, size_t cols) {
  const size_t nb = cols / QK8_0;
  float sum = 0.0f;
  for (size_t b = 0; b < nb; ++b) {
    sum += q8_0_dot_scalar(row + b * BLOCK_Q8_0_SIZE, x + b * QK8_0);
  }
  return sum;
}

inline void decode_q8_0_block(const uint8_t* block, float* out) {
  const float scale = f16_le_to_f32(block);
  for (size_t l = 0; l < QK8_0; ++l) {
    out[l] = static_cast<float>(static_cast<int8_t>(block[2 + l])) * scale;
  }
}

inline bool is_q4_k(QuantType q) {
  return q == QuantType::Q4_K_S || q == QuantType::Q4_K_M;
}

// ---- int8 fused-dot path (ported back from oxidize-c) ----------------------
// The activation vector is quantized to int8 blocks of 32 ONCE per matvec,
// then Q4_0/Q8_0/Q4_K/Q6_K weight rows are dotted with AVX2 maddubs integer
// ops instead of nibble->float->FMA. ~2.4x measured on Q4_0 decode. Trade-off:
// ~1e-3 relative rounding from activation quantization (llama.cpp-standard).
// s = d * sum(q) lets K-quant mins fold in without a second pass.
struct Q8Act {
  float d, s;
  int8_t q[32];
};

inline void quantize_act(const float* x, Q8Act* out, size_t n) {
  for (size_t b = 0; b < n / 32; ++b) {
    const float* xb = x + b * 32;
    float amax = 0.0f;
    for (size_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(xb[i]));
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    int sum = 0;
    for (size_t i = 0; i < 32; ++i) {
      int q = static_cast<int>(std::lrintf(xb[i] * id));
      q = q > 127 ? 127 : (q < -127 ? -127 : q);
      out[b].q[i] = static_cast<int8_t>(q);
      sum += q;
    }
    out[b].d = d;
    out[b].s = d * static_cast<float>(sum);
  }
}

#ifdef OXIDIZE_HAVE_F16C
// f16 scale load without the cross-TU f16_le_to_f32 call (hot: one per block).
inline float f16s(const uint8_t* p) {
  uint16_t bits;
  std::memcpy(&bits, p, 2);
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

inline __m256i i8dot(__m256i x, __m256i y) {  // signed x * signed y -> 8 x i32
  __m256i ax = _mm256_sign_epi8(x, x);
  __m256i sy = _mm256_sign_epi8(y, x);
  return _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sy), _mm256_set1_epi16(1));
}
inline __m256i u8dot(__m256i x, __m256i y) {  // unsigned nibbles * signed y
  return _mm256_madd_epi16(_mm256_maddubs_epi16(x, y), _mm256_set1_epi16(1));
}
#else
inline float f16s(const uint8_t* p) { return f16_le_to_f32(p); }
#endif

inline float dot_q4_0_i8(const uint8_t* row, const Q8Act* xq, size_t cols) {
  const size_t nb = cols / 32;
#ifdef OXIDIZE_HAVE_F16C
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  size_t b = 0;
  for (; b + 2 <= nb; b += 2) {
    const uint8_t* blk0 = row + b * 18;
    const uint8_t* blk1 = blk0 + 18;
    __m128i q40 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk0 + 2));
    __m128i q41 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk1 + 2));
    __m256i w0 = _mm256_sub_epi8(
        _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q40, 4), _mm_set1_epi8(0x0F)),
                         _mm_and_si128(q40, _mm_set1_epi8(0x0F))),
        _mm256_set1_epi8(8));
    __m256i w1 = _mm256_sub_epi8(
        _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q41, 4), _mm_set1_epi8(0x0F)),
                         _mm_and_si128(q41, _mm_set1_epi8(0x0F))),
        _mm256_set1_epi8(8));
    __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[b].q));
    __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[b + 1].q));
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i8dot(w0, y0)),
                           _mm256_set1_ps(f16s(blk0) * xq[b].d), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i8dot(w1, y1)),
                           _mm256_set1_ps(f16s(blk1) * xq[b + 1].d), acc1);
  }
  for (; b < nb; ++b) {
    const uint8_t* blk = row + b * 18;
    __m128i q4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
    __m256i w = _mm256_sub_epi8(
        _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q4, 4), _mm_set1_epi8(0x0F)),
                         _mm_and_si128(q4, _mm_set1_epi8(0x0F))),
        _mm256_set1_epi8(8));
    __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[b].q));
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i8dot(w, y)),
                           _mm256_set1_ps(f16s(blk) * xq[b].d), acc0);
  }
  return hsum256(_mm256_add_ps(acc0, acc1));
#else
  float sum = 0.0f;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* blk = row + b * 18;
    int isum = 0;
    for (int j = 0; j < 16; ++j) {
      isum += ((blk[2 + j] & 0x0F) - 8) * xq[b].q[j];
      isum += ((blk[2 + j] >> 4) - 8) * xq[b].q[j + 16];
    }
    sum += f16s(blk) * xq[b].d * static_cast<float>(isum);
  }
  return sum;
#endif
}

inline float dot_iq4_nl_i8(const uint8_t* row, const Q8Act* xq, size_t cols) {
  const size_t nb = cols / 32;
  static constexpr int8_t kIq4Nl[16] = {
      -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
#ifdef OXIDIZE_HAVE_F16C
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kIq4Nl));
  size_t b = 0;
  for (; b + 2 <= nb; b += 2) {
    const uint8_t* blk0 = row + b * BLOCK_IQ4_NL_SIZE;
    const uint8_t* blk1 = blk0 + BLOCK_IQ4_NL_SIZE;
    __m128i q40 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk0 + 2));
    __m128i q41 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk1 + 2));
    __m128i lo0 = _mm_and_si128(q40, _mm_set1_epi8(0x0F));
    __m128i hi0 = _mm_and_si128(_mm_srli_epi16(q40, 4), _mm_set1_epi8(0x0F));
    __m128i lo1 = _mm_and_si128(q41, _mm_set1_epi8(0x0F));
    __m128i hi1 = _mm_and_si128(_mm_srli_epi16(q41, 4), _mm_set1_epi8(0x0F));
    __m256i w0 = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi0),
                                  _mm_shuffle_epi8(lut, lo0));
    __m256i w1 = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi1),
                                  _mm_shuffle_epi8(lut, lo1));
    __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[b].q));
    __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[b + 1].q));
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i8dot(w0, y0)),
                           _mm256_set1_ps(f16s(blk0) * xq[b].d), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i8dot(w1, y1)),
                           _mm256_set1_ps(f16s(blk1) * xq[b + 1].d), acc1);
  }
  for (; b < nb; ++b) {
    const uint8_t* blk = row + b * BLOCK_IQ4_NL_SIZE;
    __m128i q4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
    __m128i lo = _mm_and_si128(q4, _mm_set1_epi8(0x0F));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(q4, 4), _mm_set1_epi8(0x0F));
    __m256i w = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi),
                                 _mm_shuffle_epi8(lut, lo));
    __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[b].q));
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i8dot(w, y)),
                           _mm256_set1_ps(f16s(blk) * xq[b].d), acc0);
  }
  return hsum256(_mm256_add_ps(acc0, acc1));
#else
  float sum = 0.0f;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* blk = row + b * BLOCK_IQ4_NL_SIZE;
    int isum = 0;
    for (int j = 0; j < 16; ++j) {
      isum += kIq4Nl[blk[2 + j] & 0x0F] * xq[b].q[j];
      isum += kIq4Nl[blk[2 + j] >> 4] * xq[b].q[j + 16];
    }
    sum += f16s(blk) * xq[b].d * static_cast<float>(isum);
  }
  return sum;
#endif
}

inline float dot_q8_0_i8(const uint8_t* row, const Q8Act* xq, size_t cols) {
  const size_t nb = cols / 32;
#ifdef OXIDIZE_HAVE_F16C
  __m256 acc = _mm256_setzero_ps();
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* blk = row + b * 34;
    __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 2));
    __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[b].q));
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i8dot(w, y)),
                          _mm256_set1_ps(f16s(blk) * xq[b].d), acc);
  }
  return hsum256(acc);
#else
  float sum = 0.0f;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* blk = row + b * 34;
    int isum = 0;
    for (int j = 0; j < 32; ++j) isum += static_cast<int8_t>(blk[2 + j]) * xq[b].q[j];
    sum += f16s(blk) * xq[b].d * static_cast<float>(isum);
  }
  return sum;
#endif
}

// Q4_K: per 32-value group g, value = d*sc_g*nib - min*m_g, so
// group dot = d*sc*dx*idot(nib, q8) - min*m * (dx * sum(q8) == xq.s).
inline float dot_q4_k_i8(const uint8_t* row, const Q8Act* xq, size_t cols) {
  const size_t nsb = cols / QK_K;
  float sum = 0.0f;
  for (size_t sb = 0; sb < nsb; ++sb) {
    const uint8_t* blk = row + sb * BLOCK_Q4_K_SIZE;
    float d = f16s(blk);
    float min = f16s(blk + 2);
    const uint8_t* scales = blk + 4;
    const uint8_t* qs = blk + 16;
    const Q8Act* x8 = xq + sb * (QK_K / 32);
#ifdef OXIDIZE_HAVE_F16C
    __m256 acc = _mm256_setzero_ps();
    float msum = 0.0f;
    for (int p = 0; p < 4; ++p) {
      uint8_t sc1, m1, sc2, m2;
      get_scale_min_k4(static_cast<size_t>(p) * 2, scales, sc1, m1);
      get_scale_min_k4(static_cast<size_t>(p) * 2 + 1, scales, sc2, m2);
      __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + p * 32));
      __m256i lo = _mm256_and_si256(q, _mm256_set1_epi8(0x0F));
      __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), _mm256_set1_epi8(0x0F));
      const Q8Act* b1 = &x8[p * 2];
      const Q8Act* b2 = &x8[p * 2 + 1];
      __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b1->q));
      __m256i y2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b2->q));
      acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(u8dot(lo, y1)),
                            _mm256_set1_ps(d * static_cast<float>(sc1) * b1->d), acc);
      acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(u8dot(hi, y2)),
                            _mm256_set1_ps(d * static_cast<float>(sc2) * b2->d), acc);
      msum += min * static_cast<float>(m1) * b1->s + min * static_cast<float>(m2) * b2->s;
    }
    sum += hsum256(acc) - msum;
#else
    for (int p = 0; p < 4; ++p) {
      uint8_t sc1, m1, sc2, m2;
      get_scale_min_k4(static_cast<size_t>(p) * 2, scales, sc1, m1);
      get_scale_min_k4(static_cast<size_t>(p) * 2 + 1, scales, sc2, m2);
      const Q8Act* b1 = &x8[p * 2];
      const Q8Act* b2 = &x8[p * 2 + 1];
      int i1 = 0, i2 = 0;
      for (int l = 0; l < 32; ++l) {
        i1 += (qs[p * 32 + l] & 0x0F) * b1->q[l];
        i2 += (qs[p * 32 + l] >> 4) * b2->q[l];
      }
      sum += d * static_cast<float>(sc1) * b1->d * static_cast<float>(i1) -
             min * static_cast<float>(m1) * b1->s;
      sum += d * static_cast<float>(sc2) * b2->d * static_cast<float>(i2) -
             min * static_cast<float>(m2) * b2->s;
    }
#endif
  }
  return sum;
}

// Q6_K int dot (scalar int accumulation; still beats dequant+f32-dot by
// skipping the float row expansion). Layout mirrors dequant_q6_k.
inline float dot_q6_k_i8(const uint8_t* row, const Q8Act* xq, size_t cols) {
  const size_t nsb = cols / QK_K;
  float sum = 0.0f;
  for (size_t sb = 0; sb < nsb; ++sb) {
    const uint8_t* blk = row + sb * BLOCK_Q6_K_SIZE;
    float d = f16s(blk + 208);
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
    const Q8Act* x8 = xq + sb * (QK_K / 32);
    for (int g = 0; g < 2; ++g) {
      size_t qlo = static_cast<size_t>(g) * 64, qho = static_cast<size_t>(g) * 32,
             sco = static_cast<size_t>(g) * 8;
      for (int half = 0; half < 4; ++half) {
        const Q8Act* xb = &x8[g * 4 + half];
        int isum0 = 0, isum1 = 0;
        for (int l = 0; l < 32; ++l) {
          int q;
          switch (half) {
            case 0: q = ((ql[qlo + l] & 0x0F) | ((qh[qho + l] & 3) << 4)) - 32; break;
            case 1: q = ((ql[qlo + l + 32] & 0x0F) | (((qh[qho + l] >> 2) & 3) << 4)) - 32; break;
            case 2: q = ((ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) - 32; break;
            default: q = ((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) - 32; break;
          }
          if (l < 16) isum0 += q * xb->q[l];
          else isum1 += q * xb->q[l];
        }
        sum += d * xb->d *
               (static_cast<float>(sc[sco + static_cast<size_t>(half) * 2]) * isum0 +
                static_cast<float>(sc[sco + static_cast<size_t>(half) * 2 + 1]) * isum1);
      }
    }
  }
  return sum;
}

inline bool is_q4_sym(QuantType q) {
  return q == QuantType::Q4_0 || q == QuantType::Q4_O;
}

inline bool int8_gemv_ok(QuantType q, size_t cols) {
  if (cols % 32 != 0) return false;
  if (is_q4_sym(q) || q == QuantType::Q8_0 || q == QuantType::IQ4_NL) return true;
  if ((is_q4_k(q) || q == QuantType::Q6_K) && cols % QK_K == 0) return true;
  return false;
}

inline float int8_row_dot(QuantType q, const uint8_t* row, const Q8Act* xq,
                          size_t cols) {
  switch (q) {
    case QuantType::Q4_0:
    case QuantType::Q4_O:
      return dot_q4_0_i8(row, xq, cols);
    case QuantType::Q8_0: return dot_q8_0_i8(row, xq, cols);
    case QuantType::IQ4_NL: return dot_iq4_nl_i8(row, xq, cols);
    case QuantType::Q6_K: return dot_q6_k_i8(row, xq, cols);
    default: return dot_q4_k_i8(row, xq, cols);  // Q4_K_S / Q4_K_M
  }
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

void apply_rope_norm(float* vec, size_t head_dim, size_t num_heads, size_t pos,
                     float theta, size_t rope_dim) {
  // Adjacent-pair (LLAMA_ROPE_TYPE_NORM / GLM-DSA) rotation.
  // Pairs h[2i] with h[2i+1] for i in [0, rope_len/2).
  // angle_i = pos * theta^(-2i / rope_len)
  // This is distinct from apply_rope (GPT-NeoX / split-half) which pairs
  // h[i] with h[i + rope_len/2]. Do NOT modify apply_rope — Qwen uses it.
  if (head_dim == 0) {
    throw std::runtime_error("apply_rope_norm: zero head_dim");
  }
  size_t rope_len = (rope_dim == 0) ? head_dim : rope_dim;
  if (rope_len > head_dim) {
    rope_len = head_dim;
  }
  if (rope_len % 2 != 0) {
    throw std::runtime_error("apply_rope_norm: odd rotary dimension " +
                             std::to_string(rope_len));
  }
  if (pos == 0) {
    return;
  }
  if (rope_len == 0) {
    return;
  }

  const float position_f = static_cast<float>(pos);
  const size_t half_dim = rope_len / 2;
  const float inv_rope_len = 1.0f / static_cast<float>(rope_len);
  // freq multiplier per pair: theta^(-2/rope_len), same geometric recurrence.
  const float freq_multiplier = std::pow(theta, -2.0f * inv_rope_len);

  for (size_t head = 0; head < num_heads; ++head) {
    float* h = vec + head * head_dim;
    float freq = 1.0f;
    for (size_t i = 0; i < half_dim; ++i) {
      const float x0 = h[2 * i];
      const float x1 = h[2 * i + 1];
      const float angle = position_f * freq;
      const float cos_a = std::cos(angle);
      const float sin_a = std::sin(angle);
      h[2 * i]     = x0 * cos_a - x1 * sin_a;
      h[2 * i + 1] = x0 * sin_a + x1 * cos_a;
      freq *= freq_multiplier;
    }
    // Dims [rope_len, head_dim) pass through unchanged (partial RoPE).
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
  static const bool no_fused = std::getenv("OXK_NO_FUSED") != nullptr;
  if (no_fused) {
    const size_t rb = quantized_size(quant, cols);
#pragma omp parallel
    {
      std::vector<float> row(cols);
#pragma omp for schedule(static)
      for (long long r = 0; r < static_cast<long long>(rows); ++r) {
        dequantize_row(quant, W + static_cast<size_t>(r) * rb, row.data(), cols);
        y[r] = dot_f32(row.data(), x, cols);
      }
    }
    return;
  }
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
  // Fastest path: quantize the activation to int8 once, then integer maddubs
  // row dots (Q4_0/Q8_0/Q4_K/Q6_K). ~2.4x over the float fused kernels below.
  if (int8_gemv_ok(quant, cols)) {
    std::vector<Q8Act> xq(cols / 32);
    quantize_act(x, xq.data(), cols);
    const size_t rb = quantized_size(quant, cols);
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(rows); ++r)
      y[r] = int8_row_dot(quant, W + static_cast<size_t>(r) * rb, xq.data(), cols);
    return;
  }
  // Fast paths: native Q4_0 / Q5_0 fused unpack+dot (SIMD), no scalar dequant pass.
  if (is_q4_sym(quant)) {
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
  if (is_q4_k(quant) && cols % QK_K == 0) {
    const size_t rb = (cols / QK_K) * BLOCK_Q4_K_SIZE;
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(rows); ++r)
      y[r] = dot_q4_k(W + static_cast<size_t>(r) * rb, x, cols);
    return;
  }
  if (quant == QuantType::Q8_0 && cols % QK8_0 == 0) {
    const size_t rb = (cols / QK8_0) * BLOCK_Q8_0_SIZE;
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(rows); ++r)
      y[r] = dot_q8_0(W + static_cast<size_t>(r) * rb, x, cols);
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

void gemm_quantized(float* outputs, QuantType quant, const uint8_t* W,
                    size_t rows, size_t cols, const float* inputs, size_t batch) {
  if (batch == 0) return;
  if (batch == 1) {
    gemv_quantized(outputs, quant, W, rows, cols, inputs);
    return;
  }

  const size_t row_bytes = quantized_size(quant, cols);

  if (quant == QuantType::F16) {
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(rows); ++r) {
      const uint16_t* row = reinterpret_cast<const uint16_t*>(
          W + static_cast<size_t>(r) * cols * 2);
      for (size_t b = 0; b < batch; ++b) {
        outputs[b * rows + static_cast<size_t>(r)] =
            dot_f16(row, inputs + b * cols, cols);
      }
    }
    return;
  }
  // Int8 fused path: quantize every batch row once, integer row dots.
  if (int8_gemv_ok(quant, cols)) {
    const size_t xstride = cols / 32;
    std::vector<Q8Act> xq(batch * xstride);
    for (size_t b = 0; b < batch; ++b)
      quantize_act(inputs + b * cols, xq.data() + b * xstride, cols);
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(rows); ++r) {
      const uint8_t* row = W + static_cast<size_t>(r) * row_bytes;
      for (size_t b = 0; b < batch; ++b) {
        outputs[b * rows + static_cast<size_t>(r)] =
            int8_row_dot(quant, row, xq.data() + b * xstride, cols);
      }
    }
    return;
  }
  if (quant == QuantType::BF16) {
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(rows); ++r) {
      const uint8_t* row = W + static_cast<size_t>(r) * cols * 2;
      for (size_t b = 0; b < batch; ++b) {
        float acc = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
          acc += f16_le_to_f32(row + c * 2) * inputs[b * cols + c];
        }
        outputs[b * rows + static_cast<size_t>(r)] = acc;
      }
    }
    return;
  }
  if (is_q4_sym(quant)) {
    const size_t rb = (cols / 32) * 18;
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(rows); ++r) {
      const uint8_t* row = W + static_cast<size_t>(r) * rb;
      for (size_t b = 0; b < batch; ++b) {
        outputs[b * rows + static_cast<size_t>(r)] =
            dot_q4_0(row, inputs + b * cols, cols);
      }
    }
    return;
  }
  if (quant == QuantType::Q5_0) {
    const size_t rb = (cols / 32) * 22;
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(rows); ++r) {
      const uint8_t* row = W + static_cast<size_t>(r) * rb;
      for (size_t b = 0; b < batch; ++b) {
        outputs[b * rows + static_cast<size_t>(r)] =
            dot_q5_0(row, inputs + b * cols, cols);
      }
    }
    return;
  }
  if (is_q4_k(quant) && cols % QK_K == 0) {
    const size_t blocks_per_row = cols / QK_K;
    const size_t rb = blocks_per_row * BLOCK_Q4_K_SIZE;
#pragma omp parallel
    {
      float scratch[QK_K];
#pragma omp for schedule(static)
      for (long long r = 0; r < static_cast<long long>(rows); ++r) {
        const uint8_t* row = W + static_cast<size_t>(r) * rb;
        for (size_t b = 0; b < batch; ++b) {
          outputs[b * rows + static_cast<size_t>(r)] = 0.0f;
        }
        for (size_t bi = 0; bi < blocks_per_row; ++bi) {
          decode_q4_k_block(row + bi * BLOCK_Q4_K_SIZE, scratch);
          const size_t in_off = bi * QK_K;
          for (size_t b = 0; b < batch; ++b) {
            outputs[b * rows + static_cast<size_t>(r)] +=
                dot_f32(scratch, inputs + b * cols + in_off, QK_K);
          }
        }
      }
    }
    return;
  }
  if (quant == QuantType::Q8_0 && cols % QK8_0 == 0) {
    const size_t blocks_per_row = cols / QK8_0;
    const size_t rb = blocks_per_row * BLOCK_Q8_0_SIZE;
#pragma omp parallel
    {
      float scratch[QK8_0];
#pragma omp for schedule(static)
      for (long long r = 0; r < static_cast<long long>(rows); ++r) {
        const uint8_t* row = W + static_cast<size_t>(r) * rb;
        for (size_t b = 0; b < batch; ++b) {
          outputs[b * rows + static_cast<size_t>(r)] = 0.0f;
        }
        for (size_t bi = 0; bi < blocks_per_row; ++bi) {
          decode_q8_0_block(row + bi * BLOCK_Q8_0_SIZE, scratch);
          const size_t in_off = bi * QK8_0;
          for (size_t b = 0; b < batch; ++b) {
            outputs[b * rows + static_cast<size_t>(r)] +=
                dot_f32(scratch, inputs + b * cols + in_off, QK8_0);
          }
        }
      }
    }
    return;
  }

  // Decode-once path: dequantize each row once, dot all batch positions.
#pragma omp parallel
  {
    std::vector<float> row(cols);
#pragma omp for schedule(static)
    for (long long r = 0; r < static_cast<long long>(rows); ++r) {
      const uint8_t* src = W + static_cast<size_t>(r) * row_bytes;
      dequantize_row(quant, src, row.data(), cols);
      for (size_t b = 0; b < batch; ++b) {
        outputs[b * rows + static_cast<size_t>(r)] =
            dot_f32(row.data(), inputs + b * cols, cols);
      }
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