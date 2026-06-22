// Tests for the OXK C++ kernels, mirroring oxidize-kernels Rust unit tests
// (prune.rs, q8k.rs/q4k_scalar.rs round-trip). Plain asserts; exit code = status.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "oxidize/kernels.hpp"

using namespace oxidize::kernels;

static int checks = 0;
#define CHECK(c)                                                  \
  do {                                                            \
    if (!(c)) {                                                   \
      std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); \
      return 1;                                                   \
    }                                                             \
    ++checks;                                                     \
  } while (0)

// Deterministic xorshift byte stream (matches the Rust fixture generator).
static void fill_pseudo(uint8_t* b, size_t n, uint64_t state) {
  for (size_t i = 0; i < n; ++i) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    b[i] = static_cast<uint8_t>(state);
  }
}

int main() {
  // --- magnitude_mask: keeps top-(1-sparsity) per row ---
  {
    std::vector<float> w(16);
    for (int i = 0; i < 16; ++i) w[i] = static_cast<float>(i);
    auto mask = magnitude_mask(w.data(), 2, 8, 0.5f);
    for (int r = 0; r < 2; ++r) {
      int kept = 0;
      for (int c = 0; c < 8; ++c) kept += mask[r * 8 + c];
      CHECK(kept == 4);
    }
    for (int c = 0; c < 4; ++c) CHECK(mask[c] == 0);   // smallest pruned
    for (int c = 4; c < 8; ++c) CHECK(mask[c] == 1);   // largest kept
  }

  // --- wanda_mask: prefers high-activation columns ---
  {
    std::vector<float> w = {10, 10, 10, 1, 1, 1};
    std::vector<float> norms = {0, 0, 0, 10, 10, 10};
    auto mask = wanda_mask(w.data(), norms.data(), 1, 6, 0.5f);
    for (int c = 0; c < 3; ++c) CHECK(mask[c] == 0);
    for (int c = 3; c < 6; ++c) CHECK(mask[c] == 1);
  }

  // --- apply_mask_inplace zeros pruned entries ---
  {
    std::vector<float> w = {1, 2, 3, 4};
    std::vector<uint8_t> mask = {1, 0, 1, 0};
    apply_mask_inplace(w.data(), mask.data(), 4);
    CHECK(w[0] == 1.0f && w[1] == 0.0f && w[2] == 3.0f && w[3] == 0.0f);
  }

  // --- Q8_K round-trip: quantize then reconstruct ~= input ---
  {
    const size_t nb = 2;
    std::vector<float> vec(nb * QK_K);
    std::vector<uint8_t> raw(nb * QK_K);
    fill_pseudo(raw.data(), raw.size(), 0x1234);
    for (size_t i = 0; i < vec.size(); ++i)
      vec[i] = (static_cast<float>(raw[i]) - 127.5f) / 32.0f;
    std::vector<uint8_t> q8(nb * BLOCK_Q8_K_BYTES);
    quantize_q8_k_into(vec.data(), nb, q8.data());
    // Reconstruct: d * q[i] should approximate vec[i].
    double max_err = 0.0;
    for (size_t b = 0; b < nb; ++b) {
      float d;
      std::memcpy(&d, q8.data() + b * BLOCK_Q8_K_BYTES, 4);
      for (size_t i = 0; i < QK_K; ++i) {
        int8_t q = static_cast<int8_t>(q8[b * BLOCK_Q8_K_BYTES + 4 + i]);
        float recon = d * static_cast<float>(q);
        max_err = std::max(max_err,
                           static_cast<double>(std::fabs(recon - vec[b * QK_K + i])));
      }
    }
    CHECK(max_err < 0.1);  // 8-bit block quant of values in ~[-4,4]
  }

  // --- gemv_q4k_range matches per-row scalar dot ---
  {
    const size_t rows = 5, bpr = 3;
    std::vector<uint8_t> weights(rows * bpr * BLOCK_Q4_K_SIZE);
    fill_pseudo(weights.data(), weights.size(), 0xBEEF);
    // Tame the f16 d/dmin headers so they're finite small normals.
    for (size_t blk = 0; blk < rows * bpr; ++blk) {
      for (int half = 0; half < 2; ++half) {
        size_t off = blk * BLOCK_Q4_K_SIZE + half * 2;
        uint16_t r = weights[off] | (weights[off + 1] << 8);
        uint16_t tamed = (r & 0x83ff) | (0x3000 + ((r >> 10) & 0x7) * 0x400);
        weights[off] = tamed & 0xff;
        weights[off + 1] = tamed >> 8;
      }
    }
    std::vector<float> vec(bpr * QK_K);
    for (size_t i = 0; i < vec.size(); ++i)
      vec[i] = std::sin(0.01f * static_cast<float>(i));
    std::vector<uint8_t> q8(bpr * BLOCK_Q8_K_BYTES);
    quantize_q8_k_into(vec.data(), bpr, q8.data());
    std::vector<float> out(rows);
    gemv_q4k_range(weights.data(), bpr, q8.data(), out.data(), rows);
    const size_t row_bytes = bpr * BLOCK_Q4_K_SIZE;
    for (size_t r = 0; r < rows; ++r) {
      float want = q4k_q8k_row_dot(weights.data() + r * row_bytes, bpr, q8.data());
      CHECK(out[r] == want);  // exact: same code path
    }
  }

  std::printf("OXK KERNELS OK: %d checks passed. (%s)\n", checks,
              cpu_summary().c_str());
  return 0;
}
