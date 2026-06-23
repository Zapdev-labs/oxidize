#pragma once
// Training type definitions: TrainConfig, TrainMode, LossRecord.
// No inference-path dependencies.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace oxidize {

enum class TrainMode { LoRA, FullFT };

struct LoRAConfig {
  size_t rank   = 16;
  float  alpha  = 32.0f;  // scaling = alpha / rank
};

struct AdamWConfig {
  float lr        = 1e-4f;
  float beta1     = 0.9f;
  float beta2     = 0.999f;
  float eps       = 1e-8f;
  float weight_decay = 0.01f;
  int   warmup_steps = 100;
  int   total_steps  = 1000;
};

struct TrainConfig {
  TrainMode    mode        = TrainMode::LoRA;
  LoRAConfig   lora;
  AdamWConfig  adamw;
  size_t       seq_len     = 512;
  size_t       grad_accum  = 1;
  size_t       max_steps   = 200;
  uint64_t     seed        = 42;
  bool         overfit_one_batch = false;
  float        grad_clip   = 1.0f;  // global gradient norm clipping (<=0 disables)
  std::string  model_path;
  std::string  data_path;
};

struct LossRecord {
  int    step;
  float  loss;
};

}  // namespace oxidize

namespace oxidize {
// Batched dense matmul for training. All in oxidize:: namespace.
// Y[T x rows] = X[T x cols] * W^T  (W row-major [rows x cols]).
void train_batch_matvec(float* Y, const float* W, const float* X,
                         size_t T, size_t rows, size_t cols);
// Accumulate weight gradient: dW[rows x cols] += dY^T * X  (sum over T batch).
void train_outer_accum(float* dW, const float* dY, const float* X,
                        size_t T, size_t rows, size_t cols);
// Accumulate input gradient: dX[T x cols] += dY[T x rows] * W[rows x cols].
void train_wt_dy(float* dX, const float* W, const float* dY,
                  size_t T, size_t rows, size_t cols);
}  // namespace oxidize
