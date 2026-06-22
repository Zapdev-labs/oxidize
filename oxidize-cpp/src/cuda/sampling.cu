// GPU sampling helpers: numerically-stable softmax in place + argmax.
//
// Ported from:
//   oxidize-cpp/src/sampler.cpp (greedy: argmax with total_cmp tie-break; the
//     first index attaining the maximum wins) and the softmax used by `sample`
//     (subtract max, exponentiate, normalize).
//   oxidize-core/src/model/sampling.rs (same semantics).
//
// argmax returns the lowest index attaining the maximum value, matching the
// CPU greedy() which scans left-to-right keeping the first strict max. For the
// logit ranges in practice this equals total_cmp's ordering (NaNs are not
// produced by the forward pass; if present they sort below normals here, which
// matches total_cmp placing NaN above +inf only for negative-sign NaN — the
// forward pass does not emit NaN logits so this edge is unreachable).

#include "cuda_common.cuh"

#include <cfloat>

namespace oxidize {
namespace cuda {

namespace {

__global__ void softmax_kernel(float* __restrict__ x, unsigned n) {
  extern __shared__ float sdata[];
  unsigned tid = threadIdx.x;

  // Phase 1: max.
  float local_max = -FLT_MAX;
  for (unsigned i = tid; i < n; i += blockDim.x) local_max = fmaxf(local_max, x[i]);
  sdata[tid] = local_max;
  __syncthreads();
  for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  __shared__ float max_val;
  if (tid == 0) max_val = sdata[0];
  __syncthreads();

  // Phase 2: exponentiate + sum.
  float local_sum = 0.0f;
  for (unsigned i = tid; i < n; i += blockDim.x) {
    float e = expf(x[i] - max_val);
    x[i] = e;
    local_sum += e;
  }
  sdata[tid] = local_sum;
  __syncthreads();
  for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  __shared__ float sum_val;
  if (tid == 0) sum_val = sdata[0];
  __syncthreads();

  // Phase 3: normalize.
  float inv = (sum_val > 0.0f) ? (1.0f / sum_val) : 0.0f;
  for (unsigned i = tid; i < n; i += blockDim.x) x[i] *= inv;
}

// argmax with first-index-wins on ties (matches CPU greedy left-to-right scan).
__global__ void argmax_kernel(const float* __restrict__ logits, unsigned n,
                              uint32_t* __restrict__ out_index) {
  extern __shared__ float fdata[];
  unsigned* idata = reinterpret_cast<unsigned*>(fdata + blockDim.x);
  unsigned tid = threadIdx.x;

  float best = -FLT_MAX;
  unsigned best_i = 0;
  for (unsigned i = tid; i < n; i += blockDim.x) {
    float v = logits[i];
    if (v > best) {
      best = v;
      best_i = i;
    }
  }
  fdata[tid] = best;
  idata[tid] = best_i;
  __syncthreads();

  for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1) {
    if (tid < s) {
      float other = fdata[tid + s];
      unsigned other_i = idata[tid + s];
      // Strictly greater wins; on equal value the lower index wins.
      if (other > fdata[tid] ||
          (other == fdata[tid] && other_i < idata[tid])) {
        fdata[tid] = other;
        idata[tid] = other_i;
      }
    }
    __syncthreads();
  }
  if (tid == 0) *out_index = idata[0];
}

}  // namespace

void launch_softmax_inplace(float* x, unsigned n, cudaStream_t stream) {
  if (n == 0) return;
  int threads = kBlockSize;
  size_t shmem = static_cast<size_t>(threads) * sizeof(float);
  softmax_kernel<<<1, threads, shmem, stream>>>(x, n);
  CUDA_CHECK_KERNEL();
}

void launch_argmax(const float* logits, unsigned n, uint32_t* out_index,
                   cudaStream_t stream) {
  if (n == 0) return;
  int threads = kBlockSize;
  size_t shmem = static_cast<size_t>(threads) * (sizeof(float) + sizeof(unsigned));
  argmax_kernel<<<1, threads, shmem, stream>>>(logits, n, out_index);
  CUDA_CHECK_KERNEL();
}

}  // namespace cuda
}  // namespace oxidize
