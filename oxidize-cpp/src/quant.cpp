// Quantization / dequantization implementation.
// Ported from oxidize-core/src/compute/quantization.rs (dequant kernels, block
// layouts, f16 widening) and oxidize-core/src/format/gguf.rs::from_ggml_type
// (ggml type-id mapping). Block layouts are byte/bit faithful to llama.cpp/ggml;
// kernels are numerically faithful to the Rust scalar implementations.

#include "oxidize/quant.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace oxidize {

namespace {

// IQ4_NL nonlinear codebook (shared by IQ4_NL and IQ4_XS).
constexpr std::array<int8_t, 16> KVALUES_IQ4NL = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// sign mask used by IQ2/IQ3 dequant (kmask_iq2xs).
constexpr std::array<uint8_t, 8> KMASK_IQ2XS = {1, 2, 4, 8, 16, 32, 64, 128};

constexpr std::array<float, 16> E2M1_DOUBLED_VALUES = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f,
};

constexpr float IQ1S_DELTA = 0.125f;

// Block size for the IQ types Rust supports dequantizing.
constexpr size_t BLOCK_IQ1_S_SIZE = 2 + QK_K / 8 + QK_K / 16;                    // 50
constexpr size_t BLOCK_IQ1_M_SIZE = QK_K / 8 + QK_K / 16 + QK_K / 32;            // 56
constexpr size_t BLOCK_IQ4_XS_SIZE = 2 + 2 + QK_K / 64 + QK_K / 2;              // 136
constexpr size_t BLOCK_IQ3_S_SIZE =
    2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64;                            // 110

// iq3s_grid: 512 packed u32 entries (4 nonlinear int8 grid values each, LE).
// Verbatim from ggml-common.h.
const std::array<uint32_t, 512> IQ3S_GRID = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

[[noreturn]] void fail_input(QuantType /*q*/, size_t expected_multiple, size_t actual) {
  throw std::runtime_error("quant: invalid input length: expected multiple of " +
                           std::to_string(expected_multiple) + ", got " + std::to_string(actual));
}

// Validate block layout: src must be a whole number of blocks, dst must hold the
// matching number of values. (mirror validate_layout in quantization.rs)
void validate_layout(QuantType q, size_t input_bytes, size_t output_values,
                     size_t input_block_size, size_t values_per_block) {
  if (input_bytes % input_block_size != 0) {
    fail_input(q, input_block_size, input_bytes);
  }
  size_t expected_output = (input_bytes / input_block_size) * values_per_block;
  if (output_values != expected_output) {
    throw std::runtime_error("quant: invalid output length: expected " +
                             std::to_string(expected_output) + ", got " +
                             std::to_string(output_values));
  }
}

inline uint16_t read_u16_le(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

inline uint32_t read_u32_le(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

// get_scale_min_k4 (quantization.rs) — extract 6-bit scale/min pair for K-quants.
inline void get_scale_min_k4(size_t j, const uint8_t* scales, uint8_t& sc, uint8_t& m) {
  if (j < 4) {
    sc = scales[j] & 63;
    m = scales[j + 4] & 63;
  } else {
    sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
    m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
  }
}

// ue4m3_to_f32 (quantization.rs)
inline float ue4m3_to_f32(uint8_t byte) {
  uint8_t exp = (byte >> 3) & 0x0f;
  uint8_t mant = byte & 0x07;
  if (exp == 0) {
    return static_cast<float>(mant) * 0.001953125f;  // 2^-9
  }
  return (1.0f + static_cast<float>(mant) / 8.0f) *
         std::ldexp(1.0f, static_cast<int>(exp) - 7);
}

// iq1s_grid_decode (quantization.rs) — simplified ternary reconstruction.
inline void iq1s_grid_decode(uint16_t index, int8_t* out) {
  uint16_t idx = index;
  for (int i = 0; i < 8; ++i) {
    int bits = idx & 3;
    out[i] = (bits == 0) ? -1 : (bits == 1) ? 0 : 1;
    idx >>= 2;
    if (i == 3) {
      idx = static_cast<uint16_t>(index >> 8);
    }
  }
}

// ── Per-type dequant kernels ──────────────────────────────────────────────

void dequant_f32(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::F32, n * 4, n, 4, 1);
  for (size_t i = 0; i < n; ++i) {
    uint32_t bits = read_u32_le(in + 4 * i);
    float v;
    std::memcpy(&v, &bits, 4);
    out[i] = v;
  }
}

void dequant_f16(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::F16, n * 2, n, 2, 1);
  for (size_t i = 0; i < n; ++i) {
    out[i] = f16_le_to_f32(in + 2 * i);
  }
}

void dequant_bf16(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::BF16, n * 2, n, 2, 1);
  for (size_t i = 0; i < n; ++i) {
    uint32_t bits = static_cast<uint32_t>(read_u16_le(in + 2 * i)) << 16;
    float v;
    std::memcpy(&v, &bits, 4);
    out[i] = v;
  }
}

void dequant_q4_0(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::Q4_0, n / QK4_0 * BLOCK_Q4_0_SIZE, n, BLOCK_Q4_0_SIZE, QK4_0);
  size_t nb = n / QK4_0;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* block = in + b * BLOCK_Q4_0_SIZE;
    float* o = out + b * QK4_0;
    float d = f16_le_to_f32(block);
    for (size_t i = 0; i < 16; ++i) {
      uint8_t packed = block[2 + i];
      o[i] = static_cast<float>(static_cast<int>(packed & 0x0F) - 8) * d;
      o[i + 16] = static_cast<float>(static_cast<int>(packed >> 4) - 8) * d;
    }
  }
}

void dequant_q4_1(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::Q4_1, n / QK4_1 * BLOCK_Q4_1_SIZE, n, BLOCK_Q4_1_SIZE, QK4_1);
  size_t nb = n / QK4_1;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* block = in + b * BLOCK_Q4_1_SIZE;
    float* o = out + b * QK4_1;
    float d = f16_le_to_f32(block);
    float m = f16_le_to_f32(block + 2);
    for (size_t i = 0; i < 16; ++i) {
      uint8_t packed = block[4 + i];
      o[i] = static_cast<float>(packed & 0x0F) * d + m;
      o[i + 16] = static_cast<float>(packed >> 4) * d + m;
    }
  }
}

void dequant_q5_0(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::Q5_0, n / QK5_0 * BLOCK_Q5_0_SIZE, n, BLOCK_Q5_0_SIZE, QK5_0);
  size_t nb = n / QK5_0;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* block = in + b * BLOCK_Q5_0_SIZE;
    float* o = out + b * QK5_0;
    float d = f16_le_to_f32(block);
    uint32_t qh;
    std::memcpy(&qh, block + 2, 4);  // 4 bytes little-endian
    const uint8_t* qs = block + 6;
    for (size_t j = 0; j < QK5_0 / 2; ++j) {  // j = 0..15
      uint8_t xh_0 = ((qh >> j) & 0x01) << 4;
      uint8_t xh_1 = ((qh >> (j + 16)) & 0x01) << 4;
      int q0 = static_cast<int>((qs[j] & 0x0F) | xh_0) - 16;
      int q1 = static_cast<int>((qs[j] >> 4) | xh_1) - 16;
      o[j] = static_cast<float>(q0) * d;
      o[j + QK5_0 / 2] = static_cast<float>(q1) * d;
    }
  }
}

void dequant_q5_1(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::Q5_1, n / QK5_1 * BLOCK_Q5_1_SIZE, n, BLOCK_Q5_1_SIZE, QK5_1);
  size_t nb = n / QK5_1;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* block = in + b * BLOCK_Q5_1_SIZE;
    float* o = out + b * QK5_1;
    float d = f16_le_to_f32(block);
    float m = f16_le_to_f32(block + 2);
    uint32_t qh;
    std::memcpy(&qh, block + 4, 4);  // 4 bytes little-endian
    const uint8_t* qs = block + 8;
    for (size_t j = 0; j < QK5_1 / 2; ++j) {  // j = 0..15
      uint8_t xh_0 = ((qh >> j) & 0x01) << 4;
      uint8_t xh_1 = ((qh >> (j + 16)) & 0x01) << 4;
      uint8_t q0 = (qs[j] & 0x0F) | xh_0;
      uint8_t q1 = (qs[j] >> 4) | xh_1;
      o[j] = static_cast<float>(q0) * d + m;
      o[j + QK5_1 / 2] = static_cast<float>(q1) * d + m;
    }
  }
}

void dequant_q8_0(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::Q8_0, n / QK8_0 * BLOCK_Q8_0_SIZE, n, BLOCK_Q8_0_SIZE, QK8_0);
  size_t nb = n / QK8_0;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* block = in + b * BLOCK_Q8_0_SIZE;
    float* o = out + b * QK8_0;
    float d = f16_le_to_f32(block);
    for (size_t i = 0; i < QK8_0; ++i) {
      o[i] = static_cast<float>(static_cast<int8_t>(block[2 + i])) * d;
    }
  }
}

namespace {
// f32 -> IEEE half-precision bits, round-to-nearest-even.
uint16_t f32_to_f16_bits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  uint32_t e = (x >> 23) & 0xff;
  uint32_t mant = x & 0x7fffffu;
  if (e == 0xff) return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
  int32_t exp = static_cast<int32_t>(e) - 127 + 15;
  if (exp >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00u);  // overflow -> inf
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);  // underflow -> 0
    mant |= 0x800000u;
    int shift = 14 - exp;
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1u);
    uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (half & 1u))) half++;
    return static_cast<uint16_t>(sign | half);
  }
  uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
  uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;  // round-to-even (carry ok)
  return h;
}
}  // namespace

void dequant_q2_k(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::Q2_K, n / QK_K * BLOCK_Q2_K_SIZE, n, BLOCK_Q2_K_SIZE, QK_K);
  size_t nb = n / QK_K;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* block = in + b * BLOCK_Q2_K_SIZE;
    float* o = out + b * QK_K;
    float d = f16_le_to_f32(block + 80);
    float min = f16_le_to_f32(block + 82);
    const uint8_t* scales = block;       // [0..16]
    const uint8_t* qs = block + 16;      // [16..80]
    size_t q_ptr = 0;
    size_t is = 0;
    for (int outer = 0; outer < 2; ++outer) {
      size_t qs_base = static_cast<size_t>(outer) * 32;
      for (int k = 0; k < 4; ++k) {
        uint8_t sc1 = scales[is];
        float dl1 = d * static_cast<float>(sc1 & 0xF);
        float ml1 = min * static_cast<float>(sc1 >> 4);
        ++is;
        uint8_t sc2 = scales[is];
        float dl2 = d * static_cast<float>(sc2 & 0xF);
        float ml2 = min * static_cast<float>(sc2 >> 4);
        ++is;
        size_t shift = ((is / 2 - 1) % 4) * 2;
        for (size_t l = 0; l < 16; ++l) {
          o[q_ptr + l] = dl1 * static_cast<float>((qs[qs_base + l] >> shift) & 3) - ml1;
        }
        for (size_t l = 0; l < 16; ++l) {
          o[q_ptr + 16 + l] = dl2 * static_cast<float>((qs[qs_base + 16 + l] >> shift) & 3) - ml2;
        }
        q_ptr += 32;
      }
    }
  }
}

void dequant_q3_k(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::Q3_K_S, n / QK_K * BLOCK_Q3_K_SIZE, n, BLOCK_Q3_K_SIZE, QK_K);
  size_t nb = n / QK_K;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* block = in + b * BLOCK_Q3_K_SIZE;
    float* o = out + b * QK_K;
    float d_all = f16_le_to_f32(block + 108);
    const uint8_t* hmask = block;        // [0..32]
    const uint8_t* qs = block + 32;      // [32..96]
    uint32_t scales_raw[4];
    scales_raw[0] = read_u32_le(block + 96);
    scales_raw[1] = read_u32_le(block + 100);
    scales_raw[2] = read_u32_le(block + 104);
    uint32_t tmp = scales_raw[2];
    scales_raw[2] = ((scales_raw[0] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 4) & 0x03030303u) << 4);
    scales_raw[3] = ((scales_raw[1] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 6) & 0x03030303u) << 4);
    scales_raw[0] = (scales_raw[0] & 0x0F0F0F0Fu) | ((tmp & 0x03030303u) << 4);
    scales_raw[1] = (scales_raw[1] & 0x0F0F0F0Fu) | (((tmp >> 2) & 0x03030303u) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(scales_raw);

    size_t q_ptr = 0;
    size_t is = 0;
    uint8_t m = 1;
    for (int g = 0; g < 2; ++g) {
      for (int k = 0; k < 4; ++k) {
        float dl = d_all * static_cast<float>(static_cast<int>(scales[is]) - 32);
        ++is;
        size_t shift = ((is - 1) % 4) * 2;
        for (size_t l = 0; l < 16; ++l) {
          int qv = (qs[l] >> shift) & 3;
          int hbit = (hmask[l] & m) != 0 ? 0 : 4;
          o[q_ptr + l] = dl * static_cast<float>(qv - hbit);
        }
        float dl2 = d_all * static_cast<float>(static_cast<int>(scales[is]) - 32);
        ++is;
        for (size_t l = 0; l < 16; ++l) {
          int qv = (qs[l + 16] >> shift) & 3;
          int hbit = (hmask[l + 16] & m) != 0 ? 0 : 4;
          o[q_ptr + 16 + l] = dl2 * static_cast<float>(qv - hbit);
        }
        q_ptr += 32;
        m <<= 1;
      }
    }
  }
}

void dequant_q4_k(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::Q4_K_M, n / QK_K * BLOCK_Q4_K_SIZE, n, BLOCK_Q4_K_SIZE, QK_K);
  size_t nb = n / QK_K;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* block = in + b * BLOCK_Q4_K_SIZE;
    float* o = out + b * QK_K;
    float d = f16_le_to_f32(block);
    float min = f16_le_to_f32(block + 2);
    const uint8_t* scales = block + 4;   // [4..16]
    const uint8_t* qs = block + 16;      // [16..144]
    size_t out_ptr = 0;
    size_t is = 0;
    for (int gp = 0; gp < 4; ++gp) {
      size_t q_base = static_cast<size_t>(gp) * 32;
      uint8_t sc1, m1, sc2, m2;
      get_scale_min_k4(is, scales, sc1, m1);
      get_scale_min_k4(is + 1, scales, sc2, m2);
      float d1 = d * static_cast<float>(sc1);
      float min1 = min * static_cast<float>(m1);
      float d2 = d * static_cast<float>(sc2);
      float min2 = min * static_cast<float>(m2);
      for (size_t l = 0; l < 32; ++l) {
        o[out_ptr + l] = d1 * static_cast<float>(qs[q_base + l] & 0xF) - min1;
      }
      for (size_t l = 0; l < 32; ++l) {
        o[out_ptr + 32 + l] = d2 * static_cast<float>(qs[q_base + l] >> 4) - min2;
      }
      out_ptr += 64;
      is += 2;
    }
  }
}

void dequant_q5_k(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::Q5_K_M, n / QK_K * BLOCK_Q5_K_SIZE, n, BLOCK_Q5_K_SIZE, QK_K);
  size_t nb = n / QK_K;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* block = in + b * BLOCK_Q5_K_SIZE;
    float* o = out + b * QK_K;
    float d = f16_le_to_f32(block);
    float min = f16_le_to_f32(block + 2);
    const uint8_t* scales = block + 4;   // [4..16]
    const uint8_t* qh = block + 16;      // [16..48]
    const uint8_t* qs = block + 48;      // [48..176]
    size_t q_ptr = 0;
    size_t is = 0;
    uint8_t u1 = 1;
    uint8_t u2 = 2;
    for (int g = 0; g < 4; ++g) {
      uint8_t sc1, m1, sc2, m2;
      get_scale_min_k4(is, scales, sc1, m1);
      get_scale_min_k4(is + 1, scales, sc2, m2);
      float d1 = d * static_cast<float>(sc1);
      float min1 = min * static_cast<float>(m1);
      float d2 = d * static_cast<float>(sc2);
      float min2 = min * static_cast<float>(m2);
      for (size_t l = 0; l < 32; ++l) {
        uint32_t qv1 = static_cast<uint32_t>(qs[l] & 0xF) + ((qh[l] & u1) != 0 ? 16u : 0u);
        o[q_ptr + l] = d1 * static_cast<float>(qv1) - min1;
      }
      for (size_t l = 0; l < 32; ++l) {
        uint32_t qv2 = static_cast<uint32_t>(qs[l] >> 4) + ((qh[l] & u2) != 0 ? 16u : 0u);
        o[q_ptr + 32 + l] = d2 * static_cast<float>(qv2) - min2;
      }
      q_ptr += 64;
      is += 2;
      u1 <<= 2;
      u2 <<= 2;
    }
  }
}

void dequant_q6_k(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::Q6_K, n / QK_K * BLOCK_Q6_K_SIZE, n, BLOCK_Q6_K_SIZE, QK_K);
  size_t nb = n / QK_K;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* block = in + b * BLOCK_Q6_K_SIZE;
    float* o = out + b * QK_K;
    float d = f16_le_to_f32(block + 208);
    const uint8_t* ql = block;           // [0..128]
    const uint8_t* qh = block + 128;     // [128..192]
    const int8_t* sc = reinterpret_cast<const int8_t*>(block + 192);  // [192..208]
    size_t q_ptr = 0;
    for (int group = 0; group < 2; ++group) {
      size_t ql_off = static_cast<size_t>(group) * 64;
      size_t qh_off = static_cast<size_t>(group) * 32;
      size_t sc_off = static_cast<size_t>(group) * 8;
      for (size_t l = 0; l < 32; ++l) {
        size_t is = l / 16;
        int q1 = (static_cast<int>(ql[ql_off + l] & 0xF) |
                  (static_cast<int>(qh[qh_off + l] & 3) << 4)) - 32;
        int q2 = (static_cast<int>(ql[ql_off + l + 32] & 0xF) |
                  (static_cast<int>((qh[qh_off + l] >> 2) & 3) << 4)) - 32;
        int q3 = (static_cast<int>(ql[ql_off + l] >> 4) |
                  (static_cast<int>((qh[qh_off + l] >> 4) & 3) << 4)) - 32;
        int q4 = (static_cast<int>(ql[ql_off + l + 32] >> 4) |
                  (static_cast<int>((qh[qh_off + l] >> 6) & 3) << 4)) - 32;
        o[q_ptr + l] = d * static_cast<float>(sc[sc_off + is]) * static_cast<float>(q1);
        o[q_ptr + 32 + l] = d * static_cast<float>(sc[sc_off + is + 2]) * static_cast<float>(q2);
        o[q_ptr + 64 + l] = d * static_cast<float>(sc[sc_off + is + 4]) * static_cast<float>(q3);
        o[q_ptr + 96 + l] = d * static_cast<float>(sc[sc_off + is + 6]) * static_cast<float>(q4);
      }
      q_ptr += 128;
    }
  }
}

void dequant_iq4_xs(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::IQ4_XS, n / QK_K * BLOCK_IQ4_XS_SIZE, n, BLOCK_IQ4_XS_SIZE, QK_K);
  size_t nb = n / QK_K;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* block = in + b * BLOCK_IQ4_XS_SIZE;
    float* o = out + b * QK_K;
    float d = f16_le_to_f32(block);
    uint16_t scales_h = read_u16_le(block + 2);
    const uint8_t* scales_l = block + 4;  // [4..8]
    const uint8_t* qs = block + 8;        // [8..136]
    for (size_t ib = 0; ib < QK_K / 32; ++ib) {
      int ls_l = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf;
      int ls_h = (static_cast<int>((scales_h >> (2 * ib)) & 3)) << 4;
      float dl = d * static_cast<float>((ls_l | ls_h) - 32);
      size_t qoff = ib * 16;
      size_t ooff = ib * 32;
      for (size_t j = 0; j < 16; ++j) {
        uint8_t bb = qs[qoff + j];
        o[ooff + j] = dl * static_cast<float>(KVALUES_IQ4NL[bb & 0xf]);
        o[ooff + j + 16] = dl * static_cast<float>(KVALUES_IQ4NL[bb >> 4]);
      }
    }
  }
}

void dequant_iq3_s(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::IQ3_S, n / QK_K * BLOCK_IQ3_S_SIZE, n, BLOCK_IQ3_S_SIZE, QK_K);
  auto grid = [](size_t idx, size_t j) -> float {
    return static_cast<float>((IQ3S_GRID[idx] >> (8 * j)) & 0xff);
  };
  size_t nb = n / QK_K;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* block = in + b * BLOCK_IQ3_S_SIZE;
    float* o = out + b * QK_K;
    float d = f16_le_to_f32(block);
    const uint8_t* qs = block + 2;        // 64 bytes
    const uint8_t* qh = block + 66;       // 8 bytes
    const uint8_t* signs = block + 74;    // 32 bytes
    const uint8_t* scales = block + 106;  // 4 bytes
    size_t qs_o = 0, qh_o = 0, sg_o = 0, y = 0, ib32 = 0;
    while (ib32 < QK_K / 32) {
      float db1 = d * static_cast<float>(1 + 2 * (scales[ib32 / 2] & 0xf));
      float db2 = d * static_cast<float>(1 + 2 * (scales[ib32 / 2] >> 4));
      for (size_t l = 0; l < 4; ++l) {
        size_t h = qh[qh_o];
        size_t i1 = static_cast<size_t>(qs[qs_o + 2 * l]) | ((h << (8 - 2 * l)) & 256);
        size_t i2 = static_cast<size_t>(qs[qs_o + 2 * l + 1]) | ((h << (7 - 2 * l)) & 256);
        uint8_t s = signs[sg_o + l];
        for (size_t j = 0; j < 4; ++j) {
          float f1 = (s & KMASK_IQ2XS[j]) != 0 ? -1.0f : 1.0f;
          float f2 = (s & KMASK_IQ2XS[j + 4]) != 0 ? -1.0f : 1.0f;
          o[y + j] = db1 * grid(i1, j) * f1;
          o[y + j + 4] = db1 * grid(i2, j) * f2;
        }
        y += 8;
      }
      qs_o += 8;
      sg_o += 4;
      for (size_t l = 0; l < 4; ++l) {
        size_t h = qh[qh_o + 1];
        size_t i1 = static_cast<size_t>(qs[qs_o + 2 * l]) | ((h << (8 - 2 * l)) & 256);
        size_t i2 = static_cast<size_t>(qs[qs_o + 2 * l + 1]) | ((h << (7 - 2 * l)) & 256);
        uint8_t s = signs[sg_o + l];
        for (size_t j = 0; j < 4; ++j) {
          float f1 = (s & KMASK_IQ2XS[j]) != 0 ? -1.0f : 1.0f;
          float f2 = (s & KMASK_IQ2XS[j + 4]) != 0 ? -1.0f : 1.0f;
          o[y + j] = db2 * grid(i1, j) * f1;
          o[y + j + 4] = db2 * grid(i2, j) * f2;
        }
        y += 8;
      }
      qh_o += 2;
      qs_o += 8;
      sg_o += 4;
      ib32 += 2;
    }
  }
}

void dequant_iq1_s(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::IQ1_S, n / QK_K * BLOCK_IQ1_S_SIZE, n, BLOCK_IQ1_S_SIZE, QK_K);
  size_t nb = n / QK_K;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* block = in + b * BLOCK_IQ1_S_SIZE;
    float* o = out + b * QK_K;
    float d = f16_le_to_f32(block);
    const uint8_t* qs = block + 2;   // 32 bytes
    const uint8_t* qh = block + 34;  // 16 bytes = 8 u16
    size_t out_ptr = 0;
    int8_t grid_vals[8];
    for (size_t ib = 0; ib < QK_K / 32; ++ib) {
      uint16_t qh_v = read_u16_le(qh + 2 * ib);
      float dl = d * (2.0f * static_cast<float>((qh_v >> 12) & 7) + 1.0f);
      float delta = (qh_v & 0x8000) != 0 ? -IQ1S_DELTA : IQ1S_DELTA;
      for (size_t l = 0; l < 4; ++l) {
        uint16_t grid_idx =
            static_cast<uint16_t>(qs[l + ib * 4]) |
            static_cast<uint16_t>(((qh_v >> (3 * l)) & 7) << 8);
        iq1s_grid_decode(grid_idx, grid_vals);
        for (size_t j = 0; j < 8; ++j) {
          o[out_ptr + j] = dl * (static_cast<float>(grid_vals[j]) + delta);
        }
        out_ptr += 8;
      }
    }
  }
}

void dequant_iq1_m(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::IQ1_M, n / QK_K * BLOCK_IQ1_M_SIZE, n, BLOCK_IQ1_M_SIZE, QK_K);
  size_t nb = n / QK_K;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* block = in + b * BLOCK_IQ1_M_SIZE;
    float* o = out + b * QK_K;
    const uint8_t* qs = block;         // 32 bytes
    const uint8_t* qh = block + 32;    // 16 bytes
    const uint8_t* scales = block + 48;  // 8 bytes = 4 u16

    uint16_t sc0 = read_u16_le(scales);
    uint16_t sc1v = read_u16_le(scales + 2);
    uint16_t sc2v = read_u16_le(scales + 4);
    uint16_t sc3v = read_u16_le(scales + 6);
    uint16_t scale_u16 = (sc0 >> 12) | ((sc1v >> 8) & 0x00f0) |
                         ((sc2v >> 4) & 0x0f00) | (sc3v & 0xf000);
    uint8_t scale_bytes[2] = {static_cast<uint8_t>(scale_u16 & 0xff),
                              static_cast<uint8_t>(scale_u16 >> 8)};
    float d = f16_le_to_f32(scale_bytes);

    size_t out_ptr = 0;
    int8_t grid_vals[8];
    for (size_t ib = 0; ib < QK_K / 32; ++ib) {
      uint8_t sc_ib = scales[ib / 2];
      float dl1 = d * (2.0f * static_cast<float>((sc_ib >> (6 * (ib % 2))) & 0x7) + 1.0f);
      float dl2 = d * (2.0f * static_cast<float>((sc_ib >> (6 * (ib % 2) + 3)) & 0x7) + 1.0f);

      uint16_t idx0 = static_cast<uint16_t>(qs[ib * 4]) |
                      ((static_cast<uint16_t>(qh[ib * 2]) << 8) & 0x700);
      uint16_t idx1 = static_cast<uint16_t>(qs[ib * 4 + 1]) |
                      ((static_cast<uint16_t>(qh[ib * 2]) << 4) & 0x700);
      uint16_t idx2 = static_cast<uint16_t>(qs[ib * 4 + 2]) |
                      ((static_cast<uint16_t>(qh[ib * 2 + 1]) << 8) & 0x700);
      uint16_t idx3 = static_cast<uint16_t>(qs[ib * 4 + 3]) |
                      ((static_cast<uint16_t>(qh[ib * 2 + 1]) << 4) & 0x700);

      float deltas[4] = {
          (qh[ib * 2] & 0x08) != 0 ? -IQ1S_DELTA : IQ1S_DELTA,
          (qh[ib * 2] & 0x80) != 0 ? -IQ1S_DELTA : IQ1S_DELTA,
          (qh[ib * 2 + 1] & 0x08) != 0 ? -IQ1S_DELTA : IQ1S_DELTA,
          (qh[ib * 2 + 1] & 0x80) != 0 ? -IQ1S_DELTA : IQ1S_DELTA,
      };

      iq1s_grid_decode(idx0, grid_vals);
      for (size_t j = 0; j < 8; ++j) o[out_ptr + j] = dl1 * (static_cast<float>(grid_vals[j]) + deltas[0]);
      out_ptr += 8;
      iq1s_grid_decode(idx1, grid_vals);
      for (size_t j = 0; j < 8; ++j) o[out_ptr + j] = dl1 * (static_cast<float>(grid_vals[j]) + deltas[1]);
      out_ptr += 8;
      iq1s_grid_decode(idx2, grid_vals);
      for (size_t j = 0; j < 8; ++j) o[out_ptr + j] = dl2 * (static_cast<float>(grid_vals[j]) + deltas[2]);
      out_ptr += 8;
      iq1s_grid_decode(idx3, grid_vals);
      for (size_t j = 0; j < 8; ++j) o[out_ptr + j] = dl2 * (static_cast<float>(grid_vals[j]) + deltas[3]);
      out_ptr += 8;
    }
  }
}

void dequant_nvfp4(const uint8_t* in, float* out, size_t n) {
  validate_layout(QuantType::NVFP4, n / QK_NVFP4 * BLOCK_NVFP4_SIZE, n, BLOCK_NVFP4_SIZE, QK_NVFP4);
  size_t nb = n / QK_NVFP4;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* block = in + b * BLOCK_NVFP4_SIZE;
    float* o = out + b * QK_NVFP4;
    const uint8_t* scales = block;                          // QK_NVFP4/QK_NVFP4_SUB bytes
    const uint8_t* qs = block + QK_NVFP4 / QK_NVFP4_SUB;
    for (size_t sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
      float scale = ue4m3_to_f32(scales[sub]);
      size_t base_q = sub * (QK_NVFP4_SUB / 2);
      size_t base_out = sub * QK_NVFP4_SUB;
      for (size_t j = 0; j < QK_NVFP4_SUB / 2; ++j) {
        uint8_t packed = qs[base_q + j];
        o[base_out + j] = scale * E2M1_DOUBLED_VALUES[packed & 0x0f];
        o[base_out + j + QK_NVFP4_SUB / 2] = scale * E2M1_DOUBLED_VALUES[packed >> 4];
      }
    }
  }
}

}  // namespace

void quantize_row_q8_0(const float* x, uint8_t* out, size_t n) {
  size_t nb = n / QK8_0;
  for (size_t b = 0; b < nb; ++b) {
    const float* xb = x + b * QK8_0;
    uint8_t* o = out + b * BLOCK_Q8_0_SIZE;
    float amax = 0.0f;
    for (size_t i = 0; i < QK8_0; ++i) {
      float a = std::fabs(xb[i]);
      if (a > amax) amax = a;
    }
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    uint16_t dh = f32_to_f16_bits(d);
    o[0] = static_cast<uint8_t>(dh & 0xff);
    o[1] = static_cast<uint8_t>(dh >> 8);
    for (size_t i = 0; i < QK8_0; ++i) {
      int q = static_cast<int>(std::lround(xb[i] * id));
      if (q > 127) q = 127;
      if (q < -127) q = -127;
      o[2 + i] = static_cast<uint8_t>(static_cast<int8_t>(q));
    }
  }
}

float f16_le_to_f32(const uint8_t* bytes) {
  uint16_t bits = read_u16_le(bytes);
  uint32_t sign = (static_cast<uint32_t>(bits) >> 15) & 1;
  uint32_t exp = (static_cast<uint32_t>(bits) >> 10) & 0x1F;
  uint32_t frac = static_cast<uint32_t>(bits) & 0x03FF;

  uint32_t f32_bits;
  if (exp == 0) {
    if (frac == 0) {
      f32_bits = sign << 31;
    } else {
      uint32_t frac_norm = frac;
      int e = -14;
      while ((frac_norm & 0x0400) == 0) {
        frac_norm <<= 1;
        e -= 1;
      }
      frac_norm &= 0x03FF;
      f32_bits = (sign << 31) | (static_cast<uint32_t>(e + 127) << 23) | (frac_norm << 13);
    }
  } else if (exp == 0x1F) {
    f32_bits = (sign << 31) | 0x7F800000u | (frac << 13);
  } else {
    int e = static_cast<int>(exp) - 15 + 127;
    f32_bits = (sign << 31) | (static_cast<uint32_t>(e) << 23) | (frac << 13);
  }

  float v;
  std::memcpy(&v, &f32_bits, 4);
  return v;
}

size_t quant_block_values(QuantType q) {
  switch (q) {
    case QuantType::F32:
    case QuantType::F16:
    case QuantType::BF16:
    case QuantType::I8:
    case QuantType::I16:
    case QuantType::I32:
    case QuantType::I64:
    case QuantType::F64:
      return 1;
    case QuantType::Q4_0:
      return QK4_0;
    case QuantType::Q4_1:
      return QK4_1;
    case QuantType::Q5_0:
      return QK5_0;
    case QuantType::Q5_1:
      return QK5_1;
    case QuantType::Q8_0:
      return QK8_0;
    case QuantType::Q2_K:
    case QuantType::Q3_K_S:
    case QuantType::Q3_K_M:
    case QuantType::Q3_K_L:
    case QuantType::Q4_K_S:
    case QuantType::Q4_K_M:
    case QuantType::Q5_K_S:
    case QuantType::Q5_K_M:
    case QuantType::Q6_K:
    case QuantType::IQ1_S:
    case QuantType::IQ1_M:
    case QuantType::IQ4_XS:
    case QuantType::IQ3_S:
      return QK_K;
    case QuantType::NVFP4:
      return QK_NVFP4;
    default:
      throw std::runtime_error("quant: no block layout for this type");
  }
}

size_t quant_block_bytes(QuantType q) {
  switch (q) {
    case QuantType::F32:
      return 4;
    case QuantType::F16:
    case QuantType::BF16:
      return 2;
    case QuantType::I8:
      return 1;
    case QuantType::I16:
      return 2;
    case QuantType::I32:
      return 4;
    case QuantType::I64:
    case QuantType::F64:
      return 8;
    case QuantType::Q4_0:
      return BLOCK_Q4_0_SIZE;
    case QuantType::Q4_1:
      return BLOCK_Q4_1_SIZE;
    case QuantType::Q5_0:
      return BLOCK_Q5_0_SIZE;
    case QuantType::Q5_1:
      return BLOCK_Q5_1_SIZE;
    case QuantType::Q8_0:
      return BLOCK_Q8_0_SIZE;
    case QuantType::Q2_K:
      return BLOCK_Q2_K_SIZE;
    case QuantType::Q3_K_S:
    case QuantType::Q3_K_M:
    case QuantType::Q3_K_L:
      return BLOCK_Q3_K_SIZE;
    case QuantType::Q4_K_S:
    case QuantType::Q4_K_M:
      return BLOCK_Q4_K_SIZE;
    case QuantType::Q5_K_S:
    case QuantType::Q5_K_M:
      return BLOCK_Q5_K_SIZE;
    case QuantType::Q6_K:
      return BLOCK_Q6_K_SIZE;
    case QuantType::IQ1_S:
      return BLOCK_IQ1_S_SIZE;
    case QuantType::IQ1_M:
      return BLOCK_IQ1_M_SIZE;
    case QuantType::IQ4_XS:
      return BLOCK_IQ4_XS_SIZE;
    case QuantType::IQ3_S:
      return BLOCK_IQ3_S_SIZE;
    case QuantType::NVFP4:
      return BLOCK_NVFP4_SIZE;
    default:
      throw std::runtime_error("quant: no block layout for this type");
  }
}

size_t quantized_size(QuantType q, size_t value_count) {
  size_t vpb = quant_block_values(q);
  size_t bpb = quant_block_bytes(q);
  if (value_count % vpb != 0) {
    fail_input(q, vpb, value_count);
  }
  return (value_count / vpb) * bpb;
}

// Mirror gguf.rs::GgufQuantizationType::from_ggml_type.
QuantType from_ggml_type(uint32_t ggml_type) {
  switch (ggml_type) {
    case 0: return QuantType::F32;
    case 1: return QuantType::F16;
    case 2: return QuantType::Q4_0;
    case 3: return QuantType::Q4_1;
    case 6: return QuantType::Q5_0;
    case 7: return QuantType::Q5_1;
    case 8: return QuantType::Q8_0;
    case 10: return QuantType::Q2_K;
    case 11: return QuantType::Q3_K_S;  // ggml Q3_K
    case 12: return QuantType::Q4_K_M;  // ggml Q4_K (S/M is metadata-level)
    case 13: return QuantType::Q5_K_M;  // ggml Q5_K
    case 14: return QuantType::Q6_K;    // ggml Q6_K
    case 15: return QuantType::Q4_K_M;  // ggml Q8_K — closest supported type
    case 19: return QuantType::IQ1_S;
    case 20: return QuantType::IQ4_NL;
    case 21: return QuantType::IQ3_S;
    case 22: return QuantType::IQ2_S;
    case 23: return QuantType::IQ4_XS;
    case 24: return QuantType::I8;
    case 25: return QuantType::I16;
    case 26: return QuantType::I32;
    case 27: return QuantType::I64;
    case 28: return QuantType::F64;
    case 29: return QuantType::IQ1_M;
    case 30: return QuantType::BF16;
    case 31: return QuantType::Q4_0;
    case 32: return QuantType::Q8_0;
    case 33: return QuantType::IQ2_XXS;
    case 34: return QuantType::IQ2_XS;
    case 35: return QuantType::IQ3_XXS;
    case 40: return QuantType::NVFP4;
    default: return QuantType::Unknown;
  }
}

void dequantize_row(QuantType q, const uint8_t* src, float* dst, size_t n) {
  switch (q) {
    case QuantType::F32: dequant_f32(src, dst, n); return;
    case QuantType::F16: dequant_f16(src, dst, n); return;
    case QuantType::BF16: dequant_bf16(src, dst, n); return;
    case QuantType::Q4_0: dequant_q4_0(src, dst, n); return;
    case QuantType::Q4_1: dequant_q4_1(src, dst, n); return;
    case QuantType::Q5_0: dequant_q5_0(src, dst, n); return;
    case QuantType::Q5_1: dequant_q5_1(src, dst, n); return;
    case QuantType::Q8_0: dequant_q8_0(src, dst, n); return;
    case QuantType::Q2_K: dequant_q2_k(src, dst, n); return;
    case QuantType::Q3_K_S:
    case QuantType::Q3_K_M:
    case QuantType::Q3_K_L: dequant_q3_k(src, dst, n); return;
    case QuantType::Q4_K_S:
    case QuantType::Q4_K_M: dequant_q4_k(src, dst, n); return;
    case QuantType::Q5_K_S:
    case QuantType::Q5_K_M: dequant_q5_k(src, dst, n); return;
    case QuantType::Q6_K: dequant_q6_k(src, dst, n); return;
    case QuantType::IQ1_S: dequant_iq1_s(src, dst, n); return;
    case QuantType::IQ1_M: dequant_iq1_m(src, dst, n); return;
    case QuantType::IQ4_XS: dequant_iq4_xs(src, dst, n); return;
    case QuantType::IQ3_S: dequant_iq3_s(src, dst, n); return;
    case QuantType::NVFP4: dequant_nvfp4(src, dst, n); return;
    default:
      throw std::runtime_error("quant: unsupported dequantization type");
  }
}

}  // namespace oxidize
