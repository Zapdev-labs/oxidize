#pragma once
// OXK: Oxidize CPU kernels, ported from the Rust `oxidize-kernels` crate.
//
//   oxidize-kernels/src/q8k.rs          -> quantize_q8_k_into
//   oxidize-kernels/src/q4k_scalar.rs   -> q4k_q8k_row_dot  (bit-identical scalar ref)
//   oxidize-kernels/src/lib.rs          -> gemv_q4k_range, get_scale_min_k4, constants
//   oxidize-kernels/src/prune.rs        -> magnitude_mask / wanda_mask / apply_mask_inplace
//   oxidize-kernels/src/cpu.rs          -> ISA feature detection + summary
//
// The Q4_K x Q8_K row dot reproduces the Rust scalar reference's integer op
// sequence and per-block f32 accumulation order, so results match bit-for-bit.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace oxidize {
namespace kernels {

// K-quant geometry (matches GGUF / llama.cpp).
constexpr size_t QK_K = 256;                  // values per super-block
constexpr size_t BLOCK_Q4_K_SIZE = 144;       // f16 d + f16 dmin + 12 scales + 128 nibbles
constexpr size_t BLOCK_Q8_K_BYTES = 4 + 256 + 32;  // f32 d + 256 int8 + 16 i16 bsums

// --- ISA detection (oxidize-kernels/src/lib.rs + cpu.rs) -------------------
bool avx2_available();        // AVX2 + FMA
bool avx512_available();      // AVX-512F + BW
bool avx512vnni_available();  // + VNNI
std::string cpu_summary();    // one-line human summary of detected ISA

// --- Q8_K activation quantization (oxidize-kernels/src/q8k.rs) -------------
// `vector` has n_blocks*QK_K f32 values; `out` receives n_blocks*BLOCK_Q8_K_BYTES.
void quantize_q8_k_into(const float* vector, size_t n_blocks, uint8_t* out);

// --- Q4_K x Q8_K dot (oxidize-kernels/src/q4k_scalar.rs) -------------------
// Dot one Q4_K row (blocks_per_row blocks) against a Q8_K vector.
float q4k_q8k_row_dot(const uint8_t* row, size_t blocks_per_row,
                      const uint8_t* q8k);

// Dot `n_rows` contiguous Q4_K rows against one Q8_K vector into `out`.
void gemv_q4k_range(const uint8_t* rows, size_t blocks_per_row,
                    const uint8_t* q8k, float* out, size_t n_rows);

// AVX-512 dispatch targets (compiled with target attributes; safe to call
// only after the corresponding ISA check above).
float q4k_q8k_row_dot_avx512(const uint8_t* row, size_t blocks_per_row,
                             const uint8_t* q8k);
float q4k_q8k_row_dot_avx512vnni(const uint8_t* row, size_t blocks_per_row,
                                 const uint8_t* q8k);
void gemv_q4k_range_avx512(const uint8_t* rows, size_t blocks_per_row,
                           const uint8_t* q8k, float* out, size_t n_rows);
void gemv_q4k_range_avx512vnni(const uint8_t* rows, size_t blocks_per_row,
                               const uint8_t* q8k, float* out,
                               size_t n_rows);

// --- Pruning (oxidize-kernels/src/prune.rs) -------------------------------
// Per-output-row masks; 1 = keep, 0 = prune. Row-major weights[rows*cols].
std::vector<uint8_t> magnitude_mask(const float* weights, size_t rows,
                                    size_t cols, float sparsity);
std::vector<uint8_t> wanda_mask(const float* weights, const float* act_norms,
                                size_t rows, size_t cols, float sparsity);
// Zero entries where mask[i]==0. `n` must equal the weight count.
void apply_mask_inplace(float* weights, const uint8_t* mask, size_t n);

}  // namespace kernels
}  // namespace oxidize
