#pragma once
// Cross-entropy loss helpers for the training forward pass.

#include <cstddef>
#include <vector>
#include "oxidize/model.hpp"  // Token

namespace oxidize {

// Compute mean cross-entropy loss over a sequence.
// logits: [seq_len x vocab], targets: [seq_len], mask: [seq_len].
// Only positions where mask[t] > 0 contribute. Returns 0 if no active tokens.
float sequence_cross_entropy(const float* logits, const Token* targets,
                              const float* mask, size_t seq_len, size_t vocab);

}  // namespace oxidize
