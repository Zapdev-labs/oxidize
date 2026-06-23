// Cross-entropy loss helpers.

#include "oxidize/train_loss.hpp"
#include "oxidize/autograd.hpp"

#include <cmath>

namespace oxidize {

float sequence_cross_entropy(const float* logits, const Token* targets,
                              const float* mask, size_t seq_len, size_t vocab) {
  float total = 0.0f;
  float active = 0.0f;
  for (size_t t = 0; t < seq_len; ++t) {
    if (mask[t] > 0.0f) {
      total += cross_entropy_forward(logits + t * vocab, targets[t], mask[t], vocab);
      active += mask[t];
    }
  }
  return active > 0.0f ? total / active : 0.0f;
}

}  // namespace oxidize
