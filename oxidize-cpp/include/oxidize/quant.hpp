#pragma once
// Quantization / dequantization block layouts and dequant kernels.
// Ported from oxidize-core/src/compute/quantization.rs (and ggml type-id
// mapping from oxidize-core/src/format/gguf.rs::from_ggml_type).
//
// Block layouts are byte/bit faithful to llama.cpp/ggml. Dequant kernels are
// numerically faithful to the Rust scalar implementations.

#include <cstddef>
#include <cstdint>

#include "oxidize/config.hpp"

namespace oxidize {

// Block element counts (mirror QK* constants in quantization.rs).
constexpr size_t QK4_0 = 32;
constexpr size_t QK4_1 = 32;
constexpr size_t QK5_0 = 32;
constexpr size_t QK5_1 = 32;
constexpr size_t QK8_0 = 32;
constexpr size_t QK4_NL = 32;
constexpr size_t QK_K = 256;
constexpr size_t QK_NVFP4 = 64;
constexpr size_t QK_NVFP4_SUB = 16;

// Block byte sizes (mirror BLOCK_*_SIZE constants in quantization.rs).
constexpr size_t BLOCK_Q4_0_SIZE = 2 + 16;
constexpr size_t BLOCK_Q4_1_SIZE = 2 + 2 + 16;
constexpr size_t BLOCK_Q5_0_SIZE = 2 + 4 + 16;
constexpr size_t BLOCK_Q5_1_SIZE = 2 + 2 + 4 + 16;
constexpr size_t BLOCK_Q8_0_SIZE = 2 + 32;

constexpr size_t BLOCK_Q2_K_SIZE = 2 * 2 + QK_K / 16 + QK_K / 4;       // 84
constexpr size_t BLOCK_Q3_K_SIZE = 2 + QK_K / 4 + QK_K / 8 + 12;       // 110
constexpr size_t BLOCK_Q4_K_SIZE = 2 * 2 + 12 + QK_K / 2;             // 144
constexpr size_t BLOCK_Q5_K_SIZE = 2 * 2 + 12 + QK_K / 2 + QK_K / 8;  // 176
constexpr size_t BLOCK_Q6_K_SIZE = 2 + QK_K / 16 + 3 * QK_K / 4;       // 210
constexpr size_t BLOCK_Q8_K_SIZE = 4 + QK_K + QK_K / 16 * 2;          // 292

constexpr size_t BLOCK_NVFP4_SIZE = QK_NVFP4 / QK_NVFP4_SUB + QK_NVFP4 / 2;  // 36

constexpr size_t BLOCK_IQ2_XS_SIZE = 2 + QK_K / 8 * 2 + QK_K / 32;               // 74
constexpr size_t BLOCK_IQ2_S_SIZE = 2 + QK_K / 4 + QK_K / 32 + QK_K / 32;      // 82
constexpr size_t BLOCK_IQ4_NL_SIZE = 2 + QK4_NL / 2;                             // 18

constexpr int8_t KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// Number of scalar values represented by one block of the given type.
// Returns 1 for the per-element (non-blocked) types.
size_t quant_block_values(QuantType q);

// Number of bytes occupied by one block of the given type.
size_t quant_block_bytes(QuantType q);

// Total byte size for `value_count` elements of `q`.
// Throws std::runtime_error if value_count is not a multiple of the block size,
// or if the type has no defined block layout.
size_t quantized_size(QuantType q, size_t value_count);

// Map a GGUF per-tensor ggml_type id to QuantType
// (mirror gguf.rs::GgufQuantizationType::from_ggml_type). Unknown ids map to
// QuantType::Unknown.
QuantType from_ggml_type(uint32_t ggml_type);

// Dequantize `n` scalar values from `src` (a tightly packed sequence of blocks
// of type `q`) into `dst` (must hold `n` floats). `n` must be a multiple of the
// block size for blocked types. Throws std::runtime_error on layout violations
// or unsupported types.
void dequantize_row(QuantType q, const uint8_t* src, float* dst, size_t n);

// IEEE half-precision (f16) little-endian -> f32. Bit-exact port of
// quantization.rs::f16_le_to_f32.
float f16_le_to_f32(const uint8_t* bytes);

// Quantize `n` f32 values (n % 32 == 0) into Q8_0 blocks (d:f16 + 32×int8).
// `out` must hold quantized_size(Q8_0, n) bytes. Used for on-the-fly F16->Q8_0
// weight quantization at load (halves weight bytes, ~1.3x decode, near-lossless).
void quantize_row_q8_0(const float* x, uint8_t* out, size_t n);

// Quantize `n` f32 values (n % 32 == 0) into Q4_0 blocks (d:f16 + 16×nibbles).
void quantize_row_q4_0(const float* x, uint8_t* out, size_t n);

// AL5: same 18-byte layout as Q4_0, MSE-optimal per-block scale (ggml type 240).
// Decode uses dequantize_row(AL5) == Q4_0 bitstream.
void quantize_row_al5(const float* x, uint8_t* out, size_t n);

}  // namespace oxidize
