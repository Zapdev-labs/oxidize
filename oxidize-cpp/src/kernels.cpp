// OXK CPU kernels — C++ port of the Rust `oxidize-kernels` crate.
// See include/oxidize/kernels.hpp for the source-file mapping. The Q4_K x Q8_K
// row dot is bit-identical to oxidize-kernels/src/q4k_scalar.rs.

#include "oxidize/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace oxidize {
namespace kernels {

namespace {

// f16 (little-endian bytes) -> f32, no dependency (oxidize-kernels f16_le_to_f32).
inline float f16_le_to_f32(const uint8_t* b) {
  uint16_t bits = static_cast<uint16_t>(b[0]) | (static_cast<uint16_t>(b[1]) << 8);
  uint32_t sign = (bits >> 15) & 1u;
  uint32_t exp = (bits >> 10) & 0x1fu;
  uint32_t frac = bits & 0x03ffu;
  uint32_t out;
  if (exp == 0) {
    if (frac == 0) {
      out = sign << 31;
    } else {
      uint32_t fn = frac;
      int32_t e = -14;
      while ((fn & 0x0400u) == 0) {
        fn <<= 1;
        --e;
      }
      fn &= 0x03ffu;
      out = (sign << 31) | (static_cast<uint32_t>(e + 127) << 23) | (fn << 13);
    }
  } else if (exp == 0x1f) {
    out = (sign << 31) | (0xffu << 23) | (frac << 13);
  } else {
    out = (sign << 31) | ((exp + 112u) << 23) | (frac << 13);
  }
  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

// Q4_K 12-byte scale field -> (scale, min) for sub-group j (llama.cpp get_scale_min_k4).
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

inline int16_t read_q8_k_bsum(const uint8_t* bsums, size_t index) {
  const uint8_t* p = bsums + index * 2;
  return static_cast<int16_t>(static_cast<uint16_t>(p[0]) |
                              (static_cast<uint16_t>(p[1]) << 8));
}

// IEEE-754 total order over f32 (matches Rust f32::total_cmp), so NaN scores
// can't corrupt the selection partition. Returns true if a < b in total order.
inline bool total_less(float a, float b) {
  int32_t ai, bi;
  std::memcpy(&ai, &a, 4);
  std::memcpy(&bi, &b, 4);
  // Standard total_cmp transform: invert negatives, flip sign bit for positives.
  ai = (ai >> 31) ? ~ai : (ai ^ static_cast<int32_t>(0x80000000));
  bi = (bi >> 31) ? ~bi : (bi ^ static_cast<int32_t>(0x80000000));
  return ai < bi;
}

void mask_row_by_scores(const std::vector<float>& scores,
                        std::vector<size_t>& indices, size_t drop,
                        uint8_t* row_mask) {
  std::iota(indices.begin(), indices.end(), size_t{0});
  // Partition so the `drop` smallest-scoring indices land in [0, drop).
  std::nth_element(indices.begin(), indices.begin() + (drop - 1), indices.end(),
                   [&](size_t a, size_t b) {
                     return total_less(scores[a], scores[b]);
                   });
  for (size_t k = 0; k < drop; ++k) row_mask[indices[k]] = 0;
}

std::vector<uint8_t> mask_impl(const float* weights, const float* norms,
                               size_t rows, size_t cols, float sparsity) {
  size_t keep_per_row =
      static_cast<size_t>(std::lround((1.0f - sparsity) * static_cast<float>(cols)));
  size_t drop = cols > keep_per_row ? cols - keep_per_row : 0;
  std::vector<uint8_t> mask(rows * cols, 1);
  if (drop == 0) return mask;
  std::vector<float> scratch(cols);
  std::vector<size_t> indices(cols);
  for (size_t r = 0; r < rows; ++r) {
    const float* row = weights + r * cols;
    for (size_t i = 0; i < cols; ++i) {
      float s = std::fabs(row[i]);
      scratch[i] = norms ? s * norms[i] : s;
    }
    mask_row_by_scores(scratch, indices, drop, mask.data() + r * cols);
  }
  return mask;
}

}  // namespace

bool avx2_available() {
#if defined(__x86_64__) || defined(__i386__)
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

bool avx512_available() {
#if defined(__x86_64__) || defined(__i386__)
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
#else
  return false;
#endif
}

bool avx512vnni_available() {
#if defined(__x86_64__) || defined(__i386__)
  return avx512_available() && __builtin_cpu_supports("avx512vnni");
#else
  return false;
#endif
}

std::string cpu_summary() {
  std::string s = "oxk cpu:";
  s += avx2_available() ? " avx2+fma" : " no-avx2";
  if (avx512_available()) s += " avx512f+bw";
  if (avx512vnni_available()) s += " avx512vnni";
  return s;
}

void quantize_q8_k_into(const float* vector, size_t n_blocks, uint8_t* out) {
  for (size_t b = 0; b < n_blocks; ++b) {
    const float* in = vector + b * QK_K;
    uint8_t* blk = out + b * BLOCK_Q8_K_BYTES;
    float amax = 0.0f, maxv = 0.0f;
    for (size_t i = 0; i < QK_K; ++i) {
      float av = std::fabs(in[i]);
      if (av > amax) {
        amax = av;
        maxv = in[i];
      }
    }
    if (amax == 0.0f) {
      std::memset(blk, 0, BLOCK_Q8_K_BYTES);
      continue;
    }
    float iscale = -128.0f / maxv;
    float d = 1.0f / iscale;
    std::memcpy(blk, &d, 4);
    const size_t qs_off = 4;
    for (size_t i = 0; i < QK_K; ++i) {
      int32_t q = static_cast<int32_t>(std::lround(iscale * in[i]));
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      blk[qs_off + i] = static_cast<uint8_t>(static_cast<int8_t>(q));
    }
    const size_t bsums_off = qs_off + QK_K;
    for (size_t g = 0; g < QK_K / 16; ++g) {
      int32_t sum = 0;
      for (size_t i = 0; i < 16; ++i)
        sum += static_cast<int8_t>(blk[qs_off + g * 16 + i]);
      if (sum < INT16_MIN) sum = INT16_MIN;
      if (sum > INT16_MAX) sum = INT16_MAX;
      int16_t s16 = static_cast<int16_t>(sum);
      std::memcpy(blk + bsums_off + g * 2, &s16, 2);
    }
  }
}

float q4k_q8k_row_dot(const uint8_t* row, size_t blocks_per_row,
                      const uint8_t* q8k) {
  if (avx512vnni_available()) {
    return q4k_q8k_row_dot_avx512vnni(row, blocks_per_row, q8k);
  }
  if (avx512_available()) {
    return q4k_q8k_row_dot_avx512(row, blocks_per_row, q8k);
  }
  float acc = 0.0f;
  for (size_t bi = 0; bi < blocks_per_row; ++bi) {
    const uint8_t* w = row + bi * BLOCK_Q4_K_SIZE;
    const uint8_t* q8b = q8k + bi * BLOCK_Q8_K_BYTES;
    float d_w = f16_le_to_f32(w);
    float dmin_w = f16_le_to_f32(w + 2);
    float d_q8;
    std::memcpy(&d_q8, q8b, 4);
    const uint8_t* scales = w + 4;
    const uint8_t* qs = w + 16;
    const uint8_t* q8 = q8b + 4;
    const uint8_t* bsums = q8b + 4 + QK_K;

    int32_t pos = 0, min_acc = 0;
    for (size_t gp = 0; gp < 4; ++gp) {
      size_t g1 = gp * 2, g2 = g1 + 1;
      uint8_t s1, ms1, s2, ms2;
      get_scale_min_k4(g1, scales, s1, ms1);
      get_scale_min_k4(g2, scales, s2, ms2);
      int32_t sum1 = 0, sum2 = 0;
      for (size_t i = 0; i < 32; ++i) {
        uint8_t byte = qs[gp * 32 + i];
        sum1 += static_cast<int32_t>(byte & 0x0f) *
                static_cast<int8_t>(q8[g1 * 32 + i]);
        sum2 += static_cast<int32_t>(byte >> 4) *
                static_cast<int8_t>(q8[g2 * 32 + i]);
      }
      pos += static_cast<int32_t>(s1) * sum1 + static_cast<int32_t>(s2) * sum2;
      int32_t bs1 = read_q8_k_bsum(bsums, g1 * 2) + read_q8_k_bsum(bsums, g1 * 2 + 1);
      int32_t bs2 = read_q8_k_bsum(bsums, g2 * 2) + read_q8_k_bsum(bsums, g2 * 2 + 1);
      min_acc += static_cast<int32_t>(ms1) * bs1;
      min_acc += static_cast<int32_t>(ms2) * bs2;
    }
    acc += d_w * d_q8 * static_cast<float>(pos) -
           dmin_w * d_q8 * static_cast<float>(min_acc);
  }
  return acc;
}

void gemv_q4k_range(const uint8_t* rows, size_t blocks_per_row,
                    const uint8_t* q8k, float* out, size_t n_rows) {
  if (avx512vnni_available()) {
    gemv_q4k_range_avx512vnni(rows, blocks_per_row, q8k, out, n_rows);
    return;
  }
  if (avx512_available()) {
    gemv_q4k_range_avx512(rows, blocks_per_row, q8k, out, n_rows);
    return;
  }
  const size_t row_bytes = blocks_per_row * BLOCK_Q4_K_SIZE;
#pragma omp parallel for schedule(static)
  for (long long r = 0; r < static_cast<long long>(n_rows); ++r) {
    out[r] = q4k_q8k_row_dot(rows + static_cast<size_t>(r) * row_bytes,
                             blocks_per_row, q8k);
  }
}

std::vector<uint8_t> magnitude_mask(const float* weights, size_t rows,
                                    size_t cols, float sparsity) {
  return mask_impl(weights, nullptr, rows, cols, sparsity);
}

std::vector<uint8_t> wanda_mask(const float* weights, const float* act_norms,
                                size_t rows, size_t cols, float sparsity) {
  return mask_impl(weights, act_norms, rows, cols, sparsity);
}

void apply_mask_inplace(float* weights, const uint8_t* mask, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (!mask[i]) weights[i] = 0.0f;
}

}  // namespace kernels
}  // namespace oxidize
