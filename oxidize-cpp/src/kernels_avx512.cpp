// AVX-512 Q4_K x Q8_K row-dot kernels, ported from oxidize-kernels/src/q4k_avx512.rs.
// Functions are marked with target attributes so the file can be compiled with a
// conservative default -march while still emitting AVX-512 instructions for these
// specific routines. Runtime dispatch in src/kernels.cpp ensures the functions are
// only called on CPUs that advertise the required ISA.

#include "oxidize/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>

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

// Load one Q8_K block: d_q8, four 64-byte lanes of int8 values, and the 8 grouped
// bsum values needed by the Q4_K dot product.
struct Q8Block512 {
  float d_q8;
  __m512i q8v[4];
  int32_t bs[8];
};

__attribute__((target("avx512f,avx512bw")))
inline Q8Block512 load_q8_block_512(const uint8_t* q8b) {
  Q8Block512 block;
  std::memcpy(&block.d_q8, q8b, 4);
  const uint8_t* q8 = q8b + 4;
  const uint8_t* bsums = q8b + 4 + QK_K;
  block.q8v[0] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q8));
  block.q8v[1] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q8 + 64));
  block.q8v[2] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q8 + 128));
  block.q8v[3] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q8 + 192));
  for (int g = 0; g < 8; ++g) {
    block.bs[g] = static_cast<int32_t>(read_q8_k_bsum(bsums, g * 2)) +
                  static_cast<int32_t>(read_q8_k_bsum(bsums, g * 2 + 1));
  }
  return block;
}

// Decoded Q4_K block state for the AVX-512F/BW (non-VNNI) path.
struct Q4Block512 {
  float d_w;
  float dmin_w;
  __m512i q4_512[4];
  __m512i scale_v[4];
  int32_t mins[8];
};

__attribute__((target("avx512f,avx512bw")))
inline Q4Block512 decode_q4_block_512(const uint8_t* w) {
  const __m256i mask = _mm256_set1_epi8(0x0f);
  Q4Block512 b;
  b.d_w = f16_le_to_f32(w);
  b.dmin_w = f16_le_to_f32(w + 2);
  const uint8_t* scales = w + 4;
  const uint8_t* qs = w + 16;

  for (int gp = 0; gp < 4; ++gp) {
    const int g1 = gp * 2;
    const int g2 = g1 + 1;
    uint8_t s1, ms1, s2, ms2;
    get_scale_min_k4(static_cast<size_t>(g1), scales, s1, ms1);
    get_scale_min_k4(static_cast<size_t>(g2), scales, s2, ms2);
    b.mins[g1] = static_cast<int32_t>(ms1);
    b.mins[g2] = static_cast<int32_t>(ms2);

    const __m256i packed =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + gp * 32));
    const __m256i q4_low = _mm256_and_si256(packed, mask);
    const __m256i q4_high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
    b.q4_512[gp] =
        _mm512_inserti64x4(_mm512_castsi256_si512(q4_low), q4_high, 1);

    const __m256i s_low = _mm256_set1_epi16(static_cast<short>(s1));
    const __m256i s_high = _mm256_set1_epi16(static_cast<short>(s2));
    b.scale_v[gp] =
        _mm512_inserti64x4(_mm512_castsi256_si512(s_low), s_high, 1);
  }
  return b;
}

__attribute__((target("avx512f,avx512bw")))
inline float row_dot_decoded_512(const Q4Block512& b, float d_q8,
                                 const Q8Block512& q8) {
  __m512i vec_pos = _mm512_setzero_si512();
  int32_t min_acc = 0;
  for (int gp = 0; gp < 4; ++gp) {
    const int g1 = gp * 2;
    const int g2 = g1 + 1;
    const __m512i p16 = _mm512_maddubs_epi16(b.q4_512[gp], q8.q8v[gp]);
    const __m512i p32 = _mm512_madd_epi16(p16, b.scale_v[gp]);
    vec_pos = _mm512_add_epi32(vec_pos, p32);
    min_acc += b.mins[g1] * q8.bs[g1];
    min_acc += b.mins[g2] * q8.bs[g2];
  }
  const int32_t pos_acc = _mm512_reduce_add_epi32(vec_pos);
  return b.d_w * d_q8 * static_cast<float>(pos_acc) -
         b.dmin_w * d_q8 * static_cast<float>(min_acc);
}

// Decoded Q4_K block state for the AVX-512 VNNI path.
struct Q4BlockVnni512 {
  float d_w;
  float dmin_w;
  __m512i q4_512[4];
  __m512i scale_v[4];
  int32_t mins[8];
};

__attribute__((target("avx512f,avx512bw,avx512vnni")))
inline Q4BlockVnni512 decode_q4_block_vnni512(const uint8_t* w) {
  const __m256i mask = _mm256_set1_epi8(0x0f);
  Q4BlockVnni512 b;
  b.d_w = f16_le_to_f32(w);
  b.dmin_w = f16_le_to_f32(w + 2);
  const uint8_t* scales = w + 4;
  const uint8_t* qs = w + 16;

  for (int gp = 0; gp < 4; ++gp) {
    const int g1 = gp * 2;
    const int g2 = g1 + 1;
    uint8_t s1, ms1, s2, ms2;
    get_scale_min_k4(static_cast<size_t>(g1), scales, s1, ms1);
    get_scale_min_k4(static_cast<size_t>(g2), scales, s2, ms2);
    b.mins[g1] = static_cast<int32_t>(ms1);
    b.mins[g2] = static_cast<int32_t>(ms2);

    const __m256i packed =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + gp * 32));
    const __m256i q4_low = _mm256_and_si256(packed, mask);
    const __m256i q4_high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
    b.q4_512[gp] =
        _mm512_inserti64x4(_mm512_castsi256_si512(q4_low), q4_high, 1);

    const __m256i s_low = _mm256_set1_epi32(static_cast<int>(s1));
    const __m256i s_high = _mm256_set1_epi32(static_cast<int>(s2));
    b.scale_v[gp] =
        _mm512_inserti64x4(_mm512_castsi256_si512(s_low), s_high, 1);
  }
  return b;
}

__attribute__((target("avx512f,avx512bw,avx512vnni")))
inline float row_dot_decoded_vnni512(const Q4BlockVnni512& b, float d_q8,
                                     const Q8Block512& q8) {
  __m512i vec_pos = _mm512_setzero_si512();
  int32_t min_acc = 0;
  for (int gp = 0; gp < 4; ++gp) {
    const int g1 = gp * 2;
    const int g2 = g1 + 1;
    const __m512i prod =
        _mm512_dpbusd_epi32(_mm512_setzero_si512(), b.q4_512[gp], q8.q8v[gp]);
    const __m512i scaled = _mm512_mullo_epi32(prod, b.scale_v[gp]);
    vec_pos = _mm512_add_epi32(vec_pos, scaled);
    min_acc += b.mins[g1] * q8.bs[g1];
    min_acc += b.mins[g2] * q8.bs[g2];
  }
  const int32_t pos_acc = _mm512_reduce_add_epi32(vec_pos);
  return b.d_w * d_q8 * static_cast<float>(pos_acc) -
         b.dmin_w * d_q8 * static_cast<float>(min_acc);
}

}  // namespace

__attribute__((target("avx512f,avx512bw")))
float q4k_q8k_row_dot_avx512(const uint8_t* row, size_t blocks_per_row,
                             const uint8_t* q8k) {
  float acc = 0.0f;
  for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
    const uint8_t* w = row + block_idx * BLOCK_Q4_K_SIZE;
    const uint8_t* q8b = q8k + block_idx * BLOCK_Q8_K_BYTES;
    const Q4Block512 b = decode_q4_block_512(w);
    const Q8Block512 q8 = load_q8_block_512(q8b);
    acc += row_dot_decoded_512(b, q8.d_q8, q8);
  }
  return acc;
}

__attribute__((target("avx512f,avx512bw,avx512vnni")))
float q4k_q8k_row_dot_avx512vnni(const uint8_t* row, size_t blocks_per_row,
                                 const uint8_t* q8k) {
  float acc = 0.0f;
  for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
    const uint8_t* w = row + block_idx * BLOCK_Q4_K_SIZE;
    const uint8_t* q8b = q8k + block_idx * BLOCK_Q8_K_BYTES;
    const Q4BlockVnni512 b = decode_q4_block_vnni512(w);
    const Q8Block512 q8 = load_q8_block_512(q8b);
    acc += row_dot_decoded_vnni512(b, q8.d_q8, q8);
  }
  return acc;
}

__attribute__((target("avx512f,avx512bw")))
void gemv_q4k_range_avx512(const uint8_t* rows, size_t blocks_per_row,
                           const uint8_t* q8k, float* out, size_t n_rows) {
  const size_t row_bytes = blocks_per_row * BLOCK_Q4_K_SIZE;
#pragma omp parallel for schedule(static)
  for (long long r = 0; r < static_cast<long long>(n_rows); ++r) {
    out[r] = q4k_q8k_row_dot_avx512(rows + static_cast<size_t>(r) * row_bytes,
                                    blocks_per_row, q8k);
  }
}


__attribute__((target("avx512f,avx512bw,avx512vnni")))
void q4k_q8k_row_dot_x4_avx512vnni(const uint8_t* rows_base,
                                    size_t row_bytes,
                                    size_t blocks_per_row,
                                    const uint8_t* q8k,
                                    float* out) {
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
    const uint8_t* q8b = q8k + block_idx * BLOCK_Q8_K_BYTES;
    const Q8Block512 q8 = load_q8_block_512(q8b);
    for (size_t r = 0; r < 4; ++r) {
      const uint8_t* w = rows_base + r * row_bytes + block_idx * BLOCK_Q4_K_SIZE;
      const Q4BlockVnni512 b = decode_q4_block_vnni512(w);
      acc[r] += row_dot_decoded_vnni512(b, q8.d_q8, q8);
    }
  }
  for (size_t r = 0; r < 4; ++r) out[r] = acc[r];
}

__attribute__((target("avx512f,avx512bw,avx512vnni")))
void gemv_q4k_range_avx512vnni(const uint8_t* rows, size_t blocks_per_row,
                               const uint8_t* q8k, float* out,
                               size_t n_rows) {
  const size_t row_bytes = blocks_per_row * BLOCK_Q4_K_SIZE;
  const size_t n4 = n_rows & ~size_t(3);
#pragma omp parallel for schedule(static)
  for (long long r = 0; r < static_cast<long long>(n4); r += 4) {
    q4k_q8k_row_dot_x4_avx512vnni(rows + static_cast<size_t>(r) * row_bytes,
                                   row_bytes, blocks_per_row, q8k,
                                   out + static_cast<size_t>(r));
  }
  if (n4 < n_rows) {
#pragma omp parallel for schedule(static)
    for (long long r = static_cast<long long>(n4);
         r < static_cast<long long>(n_rows); ++r) {
      out[r] = q4k_q8k_row_dot_avx512vnni(
          rows + static_cast<size_t>(r) * row_bytes, blocks_per_row, q8k);
    }
  }
}

}  // namespace kernels
}  // namespace oxidize
