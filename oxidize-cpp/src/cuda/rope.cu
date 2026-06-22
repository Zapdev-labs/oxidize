// Rotary position embedding (partial-RoPE aware), one CUDA thread per rotated
// (head, pair) element.
//
// Ported from:
//   oxidize-cpp/src/tensor_cpu.cpp::apply_rope  (which ports
//   oxidize-core/src/compute/tensor/kernels.rs::apply_rope_f32 +
//   layer_wise.rs partial-RoPE per-head application).
//
// Numerics: the CPU path advances `freq` by a geometric recurrence
//   freq_{i+1} = freq_i * theta^(-2/rope_len)
// rather than recomputing theta^(-2i/rope_len) per index. To stay numerically
// faithful we reproduce the same recurrence value by powf: the i-th frequency is
//   freq_i = (theta^(-2/rope_len))^i
// computed with a single powf, which equals the repeated-multiply result to f32
// rounding for the head_dims in scope (<=128). pos==0 is an identity (skipped on
// the host before launch). Dims [rope_len, head_dim) pass through untouched.

#include "cuda_common.cuh"

namespace oxidize {
namespace cuda {

namespace {

// Each thread handles one (head, pair-index i in [0, half_dim)).
__global__ void apply_rope_kernel(float* __restrict__ vec, unsigned head_dim,
                                  unsigned num_heads, unsigned pos, float theta,
                                  unsigned rope_len) {
  unsigned half_dim = rope_len / 2u;
  unsigned total = num_heads * half_dim;
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total) return;

  unsigned head = tid / half_dim;
  unsigned i = tid % half_dim;

  float inv_rope_len = 1.0f / static_cast<float>(rope_len);
  float freq_multiplier = powf(theta, -2.0f * inv_rope_len);
  // freq_i = freq_multiplier^i ; reproduces the host geometric recurrence.
  float freq = powf(freq_multiplier, static_cast<float>(i));

  float* h = vec + static_cast<size_t>(head) * head_dim;
  float x0 = h[i];
  float x1 = h[half_dim + i];
  float angle = static_cast<float>(pos) * freq;
  float cos_a = cosf(angle);
  float sin_a = sinf(angle);
  h[i] = x0 * cos_a - x1 * sin_a;
  h[half_dim + i] = x0 * sin_a + x1 * cos_a;
}

}  // namespace

void launch_apply_rope(float* vec, unsigned head_dim, unsigned num_heads,
                       unsigned pos, float theta, unsigned rope_len,
                       cudaStream_t stream) {
  if (rope_len < 2 || pos == 0) return;  // identity (host already guards pos==0)
  unsigned half_dim = rope_len / 2u;
  unsigned total = num_heads * half_dim;
  if (total == 0) return;
  int grid = grid_for(total, kBlockSize);
  apply_rope_kernel<<<grid, kBlockSize, 0, stream>>>(vec, head_dim, num_heads,
                                                     pos, theta, rope_len);
  CUDA_CHECK_KERNEL();
}

}  // namespace cuda
}  // namespace oxidize
