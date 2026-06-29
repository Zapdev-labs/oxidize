// LoRA adapter: init, forward, backward, optimizer step, merge.

#include "oxidize/lora.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace oxidize {

void LoraAdapter::init(size_t rows_, size_t cols_, size_t rank_, float alpha,
                       std::mt19937_64& rng) {
  rows = rows_;
  cols = cols_;
  rank = rank_;
  scaling = alpha / static_cast<float>(rank_);

  A.assign(rank_ * cols_, 0.0f);
  B.assign(rows_ * rank_, 0.0f);  // B = 0 at init
  dA.assign(rank_ * cols_, 0.0f);
  dB.assign(rows_ * rank_, 0.0f);
  mA.assign(rank_ * cols_, 0.0f);
  vA.assign(rank_ * cols_, 0.0f);
  mB.assign(rows_ * rank_, 0.0f);
  vB.assign(rows_ * rank_, 0.0f);

  // A ~ N(0, 1/sqrt(rank))
  float std_dev = 1.0f / std::sqrt(static_cast<float>(rank_));
  std::normal_distribution<float> dist(0.0f, std_dev);
  for (float& v : A) v = dist(rng);
}

void LoraAdapter::forward(const float* x, float* y, float* ax_scratch) const {
  // ax = A * x,  A: [rank x cols], x: [cols], ax: [rank]
  for (size_t r = 0; r < rank; ++r) {
    float s = 0.0f;
    const float* Ar = A.data() + r * cols;
    for (size_t c = 0; c < cols; ++c) s += Ar[c] * x[c];
    ax_scratch[r] = s;
  }
  // y += scaling * B * ax,  B: [rows x rank], ax: [rank], y: [rows]
  for (size_t row = 0; row < rows; ++row) {
    float s = 0.0f;
    const float* Br = B.data() + row * rank;
    for (size_t r = 0; r < rank; ++r) s += Br[r] * ax_scratch[r];
    y[row] += scaling * s;
  }
}

void LoraAdapter::backward(const float* x, const float* dy, const float* ax_saved,
                            float* dx_lora_out, size_t /*rank_scratch_len*/) {
  // B^T * dy -> [rank]
  std::vector<float> Bt_dy(rank, 0.0f);
  for (size_t r = 0; r < rank; ++r) {
    float s = 0.0f;
    for (size_t row = 0; row < rows; ++row) s += B[row * rank + r] * dy[row];
    Bt_dy[r] = s;
  }

  // dB += scaling * dy outer ax_saved;  dB[row*rank + r] += scaling * dy[row] * ax_saved[r]
  for (size_t row = 0; row < rows; ++row) {
    for (size_t r = 0; r < rank; ++r) {
      dB[row * rank + r] += scaling * dy[row] * ax_saved[r];
    }
  }

  // dA += scaling * Bt_dy outer x;  dA[r*cols + c] += scaling * Bt_dy[r] * x[c]
  for (size_t r = 0; r < rank; ++r) {
    float s = scaling * Bt_dy[r];
    for (size_t c = 0; c < cols; ++c) {
      dA[r * cols + c] += s * x[c];
    }
  }

  // dx_lora += scaling * A^T * Bt_dy;  dx_lora[c] += scaling * A[r*cols+c] * Bt_dy[r]
  for (size_t r = 0; r < rank; ++r) {
    float s = scaling * Bt_dy[r];
    const float* Ar = A.data() + r * cols;
    for (size_t c = 0; c < cols; ++c) {
      dx_lora_out[c] += s * Ar[c];
    }
  }
}

void LoraAdapter::zero_grads() {
  std::fill(dA.begin(), dA.end(), 0.0f);
  std::fill(dB.begin(), dB.end(), 0.0f);
}

void LoraAdapter::adamw_step(float lr, float beta1, float beta2, float eps,
                              float weight_decay, int t) {
  float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t));
  float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t));

  // Update A with weight decay.
  size_t nA = rank * cols;
  for (size_t i = 0; i < nA; ++i) {
    float g = dA[i] + weight_decay * A[i];
    mA[i] = beta1 * mA[i] + (1.0f - beta1) * g;
    vA[i] = beta2 * vA[i] + (1.0f - beta2) * g * g;
    A[i] -= lr * (mA[i] / bc1) / (std::sqrt(vA[i] / bc2) + eps);
  }

  // Update B WITHOUT weight decay (LoRA-B starts at zero; WD would fight init).
  size_t nB = rows * rank;
  for (size_t i = 0; i < nB; ++i) {
    float g = dB[i];  // no WD on B
    mB[i] = beta1 * mB[i] + (1.0f - beta1) * g;
    vB[i] = beta2 * vB[i] + (1.0f - beta2) * g * g;
    B[i] -= lr * (mB[i] / bc1) / (std::sqrt(vB[i] / bc2) + eps);
  }
}

void LoraAdapter::merge_into(float* W) const {
  // W[row*cols+col] += scaling * sum_r(B[row*rank+r] * A[r*cols+col])
  for (size_t row = 0; row < rows; ++row) {
    for (size_t r = 0; r < rank; ++r) {
      float br = scaling * B[row * rank + r];
      const float* Ar = A.data() + r * cols;
      float* Wrow = W + row * cols;
      for (size_t c = 0; c < cols; ++c) Wrow[c] += br * Ar[c];
    }
  }
}

}  // namespace oxidize
