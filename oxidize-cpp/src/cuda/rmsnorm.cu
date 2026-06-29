// RMSNorm: out[i] = x[i] * inv_rms * scale_i, with inv_rms = 1/sqrt(mean(x^2)+eps)
// and scale_i = weight_plus_one ? (weight[i]+1) : weight[i].
//
// Ported from:
//   oxidize-cpp/src/tensor_cpu.cpp::rms_norm (which ports
//   oxidize-core/src/compute/tensor/kernels.rs::rms_norm_f32 scalar path).
//
// One thread block reduces the sum of squares for a single n-length vector. The
// reduction order differs from the strictly-sequential CPU sum, so the result is
// within f32 rounding of the CPU reference rather than bit-identical (the Rust
// SIMD paths likewise differ from the scalar order by design).

#include "cuda_common.cuh"

namespace oxidize {
namespace cuda {

namespace {

__global__ void rms_norm_kernel(float* __restrict__ out,
                                const float* __restrict__ x,
                                const float* __restrict__ weight, unsigned n,
                                float eps, int weight_plus_one) {
  extern __shared__ float sdata[];
  unsigned tid = threadIdx.x;

  float local = 0.0f;
  for (unsigned i = tid; i < n; i += blockDim.x) {
    float v = x[i];
    local += v * v;
  }
  sdata[tid] = local;
  __syncthreads();

  for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  __shared__ float inv_rms;
  if (tid == 0) {
    float mean_sq = sdata[0] / static_cast<float>(n);
    inv_rms = 1.0f / sqrtf(mean_sq + eps);
  }
  __syncthreads();

  for (unsigned i = tid; i < n; i += blockDim.x) {
    float scale = weight_plus_one ? (weight[i] + 1.0f) : weight[i];
    out[i] = x[i] * inv_rms * scale;
  }
}

}  // namespace

void launch_rms_norm(float* out, const float* x, const float* weight,
                     unsigned n, float eps, bool weight_plus_one,
                     cudaStream_t stream) {
  if (n == 0) return;
  int threads = kBlockSize;
  size_t shmem = static_cast<size_t>(threads) * sizeof(float);
  rms_norm_kernel<<<1, threads, shmem, stream>>>(
      out, x, weight, n, eps, weight_plus_one ? 1 : 0);
  CUDA_CHECK_KERNEL();
}

}  // namespace cuda
}  // namespace oxidize
