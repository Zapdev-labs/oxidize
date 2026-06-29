#pragma once
// LoRA adapter: low-rank delta W = scaling * B * A.
//
// A: [rank x cols]  init ~ N(0, 1/sqrt(rank))
// B: [rows x rank]  init = 0   (so delta W = 0 at start)
// scaling = alpha / rank
//
// Forward:  y += scaling * (B * (A * x))
// Backward: dB += scaling * dy outer (A*x)
//           dA += scaling * (B^T * dy) outer x
//           dx += W_frozen^T * dy  + scaling * A^T * (B^T * dy)
//   (W_frozen backward is handled by the caller via matmul_backward on frozen W;
//    only the LoRA delta contribution to dx is added here.)

#include <cstddef>
#include <vector>
#include <random>

namespace oxidize {

struct LoraAdapter {
  size_t rows;    // output dimension
  size_t cols;    // input dimension
  size_t rank;
  float  scaling; // alpha / rank

  std::vector<float> A;   // [rank x cols]
  std::vector<float> B;   // [rows x rank]

  // Adam state for A and B.
  std::vector<float> mA, vA;
  std::vector<float> mB, vB;

  // Initialize: A ~ N(0, 1/sqrt(rank)), B = 0.
  void init(size_t rows_, size_t cols_, size_t rank_, float alpha,
            std::mt19937_64& rng);

  // Forward contribution: y += scaling * B * (A * x).
  // ax_out must be pre-allocated [rank] scratch (reused across layers).
  void forward(const float* x, float* y, float* ax_scratch) const;

  // Backward. Assumes ax = A*x was saved by the caller in ax_saved [rank].
  // Accumulates: dB += scaling * dy outer ax_saved
  //              dA += scaling * (B^T dy) outer x
  //              dx_lora += scaling * A^T * (B^T dy)
  // dx_lora is the LoRA contribution to input gradient (caller adds to full dx).
  std::vector<float> dA;  // gradient buffers
  std::vector<float> dB;

  void backward(const float* x, const float* dy, const float* ax_saved,
                float* dx_lora_out, size_t rank_scratch_len);

  // Zero gradient buffers.
  void zero_grads();

  // AdamW step. t = global step (1-based). no_wd_B: skip weight decay on B.
  void adamw_step(float lr, float beta1, float beta2, float eps,
                  float weight_decay, int t);

  // Merge delta into a float weight matrix W [rows x cols].
  void merge_into(float* W) const;
};

}  // namespace oxidize
