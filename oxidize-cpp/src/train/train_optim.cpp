// AdamW optimizer state implementation.

#include "oxidize/train_optim.hpp"

#include <cmath>
#include <cstring>

namespace oxidize {

void AdamWState::init(size_t n) {
  m.assign(n, 0.0f);
  v.assign(n, 0.0f);
  t = 0;
}

void AdamWState::step(float* param, const float* grad, size_t n,
                      float lr, float beta1, float beta2, float eps,
                      float weight_decay, bool skip_wd) {
  ++t;
  float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t));
  float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t));

  for (size_t i = 0; i < n; ++i) {
    float g = grad[i] + (skip_wd ? 0.0f : weight_decay * param[i]);
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
    param[i] -= lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps);
  }
}

}  // namespace oxidize
