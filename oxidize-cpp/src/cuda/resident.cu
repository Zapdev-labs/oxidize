// Elementwise kernels for the GPU-resident decode path (residual add, repeating
// bias add). Kept tiny and self-contained; the heavy ops reuse the existing
// gemv / rms_norm / rope / flash-decode launchers.
//
// WIP / UNVERIFIED: compiled only on the Modal build (nvcc, sm_80/sm_90); not
// yet validated on a GPU (no nvcc locally, Modal credits exhausted).

#include "cuda_common.cuh"

namespace oxidize {
namespace cuda {

namespace {

__global__ void k_residual_add(float* __restrict y, const float* __restrict x,
                               unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += x[i];
}

// y[i] += bias[i % bias_len]  (matches the CPU add_repeating_bias).
__global__ void k_add_bias_mod(float* __restrict y, const float* __restrict bias,
                               unsigned n, unsigned bias_len) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += bias[i % bias_len];
}

}  // namespace

void launch_residual_add(float* y, const float* x, unsigned n,
                         cudaStream_t stream) {
  if (n == 0) return;
  int grid = grid_for(n, kBlockSize);
  k_residual_add<<<grid, kBlockSize, 0, stream>>>(y, x, n);
  CUDA_CHECK_KERNEL();
}

void launch_add_bias_mod(float* y, const float* bias, unsigned n,
                         unsigned bias_len, cudaStream_t stream) {
  if (n == 0 || bias_len == 0) return;
  int grid = grid_for(n, kBlockSize);
  k_add_bias_mod<<<grid, kBlockSize, 0, stream>>>(y, bias, n, bias_len);
  CUDA_CHECK_KERNEL();
}

}  // namespace cuda
}  // namespace oxidize
