#pragma once
// AdamW optimizer with grad accumulation and warmup/cosine LR schedule.

#include <cstddef>
#include <cmath>
#include <vector>

namespace oxidize {

// Compute LR for step t (1-based). warmup_steps linear ramp, then cosine decay.
inline float compute_lr(float base_lr, int t, int warmup_steps, int total_steps) {
  if (t <= 0) return 0.0f;
  if (warmup_steps > 0 && t <= warmup_steps) {
    return base_lr * static_cast<float>(t) / static_cast<float>(warmup_steps);
  }
  int decay_steps = total_steps - warmup_steps;
  if (decay_steps <= 0) return base_lr;
  int step_in_decay = t - warmup_steps;
  float progress = static_cast<float>(step_in_decay) / static_cast<float>(decay_steps);
  if (progress > 1.0f) progress = 1.0f;
  return base_lr * 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * progress));
}

// AdamW state for a single parameter tensor.
struct AdamWState {
  std::vector<float> m;  // first moment
  std::vector<float> v;  // second moment
  int t = 0;             // step counter (1-based for bias correction)

  void init(size_t n);

  // In-place update of param[0..n-1] using grad[0..n-1].
  // skip_wd: set true for norms, biases, LoRA-B (per AdamW convention).
  void step(float* param, const float* grad, size_t n,
            float lr, float beta1, float beta2, float eps,
            float weight_decay, bool skip_wd);
};

}  // namespace oxidize
