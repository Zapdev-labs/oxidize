// Fused decode-phase flash attention (single query, full KV prefix) with GQA.
//
// Ported from:
//   oxidize-core/src/compute/flash_attention.rs
//     - flash_attention_decode_impl: online softmax over the sequence with
//       running max/sum, scale 1/sqrt(head_dim).
//     - flash_attention_decode_heads_impl: per-head dispatch, kv_head =
//       head / group_size, group_size = num_heads / kv_heads.
//   oxidize-cpp tensor.hpp::attention_decode (KV layout
//     [seq_len * kv_heads * head_dim], row per position; the query head's kv
//     head reads offset kv_head * head_dim within each position row).
//
// Numerics: the CPU online-softmax updates the running max per-key and rescales
// the accumulator; we reproduce the same recurrence but reduce the per-key dot
// product across the block's threads (parallel reduction), so the result is
// within f32 rounding of the strictly-sequential CPU path. One block per query
// head; head_dim lanes (rounded to kBlockSize) cooperate.

#include "cuda_common.cuh"

namespace oxidize {
namespace cuda {

namespace {

__global__ void flash_decode_kernel(float* __restrict__ out,
                                    const float* __restrict__ q,
                                    const float* __restrict__ k_cache,
                                    const float* __restrict__ v_cache,
                                    unsigned seq_len, unsigned num_heads,
                                    unsigned kv_heads, unsigned head_dim,
                                    unsigned group_size, float scale) {
  unsigned head = blockIdx.x;
  if (head >= num_heads) return;
  unsigned kv_head = head / group_size;
  unsigned tid = threadIdx.x;

  const float* q_head = q + static_cast<size_t>(head) * head_dim;
  float* out_head = out + static_cast<size_t>(head) * head_dim;

  unsigned kv_len = kv_heads * head_dim;
  unsigned kv_offset = kv_head * head_dim;

  extern __shared__ float shmem[];
  float* red = shmem;             // [blockDim.x] reduction scratch
  float* acc = shmem + blockDim.x;  // [head_dim] running (unnormalized) output

  for (unsigned d = tid; d < head_dim; d += blockDim.x) acc[d] = 0.0f;
  __syncthreads();

  __shared__ float running_max;
  __shared__ float running_sum;
  __shared__ float s_exp_factor;
  __shared__ float s_exp_score;
  if (tid == 0) {
    running_max = -3.4028235e38f;  // f32::NEG_INFINITY
    running_sum = 0.0f;
  }
  __syncthreads();

  for (unsigned t = 0; t < seq_len; ++t) {
    const float* key_row = k_cache + static_cast<size_t>(t) * kv_len + kv_offset;
    // Dot(q_head, key_row) reduced across threads.
    float partial = 0.0f;
    for (unsigned d = tid; d < head_dim; d += blockDim.x) {
      partial += q_head[d] * key_row[d];
    }
    red[tid] = partial;
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }

    if (tid == 0) {
      float score = red[0] * scale;
      float new_max = fmaxf(running_max, score);
      s_exp_factor = expf(running_max - new_max);
      s_exp_score = expf(score - new_max);
      running_sum = running_sum * s_exp_factor + s_exp_score;
      running_max = new_max;
    }
    __syncthreads();

    float exp_factor = s_exp_factor;
    float exp_score = s_exp_score;
    const float* val_row = v_cache + static_cast<size_t>(t) * kv_len + kv_offset;
    // Rescale accumulator + accumulate weighted value.
    for (unsigned d = tid; d < head_dim; d += blockDim.x) {
      float a = acc[d];
      if (exp_factor != 1.0f) a *= exp_factor;
      acc[d] = a + exp_score * val_row[d];
    }
    __syncthreads();
  }

  float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
  for (unsigned d = tid; d < head_dim; d += blockDim.x) {
    out_head[d] = (running_sum > 0.0f) ? acc[d] * inv_sum : 0.0f;
  }
}

}  // namespace

void launch_flash_decode(float* out, const float* q, const float* k_cache,
                         const float* v_cache, unsigned seq_len,
                         unsigned num_heads, unsigned kv_heads,
                         unsigned head_dim, cudaStream_t stream) {
  if (num_heads == 0 || head_dim == 0) return;
  if (seq_len == 0) {
    CUDA_CHECK(cudaMemsetAsync(
        out, 0, static_cast<size_t>(num_heads) * head_dim * sizeof(float),
        stream));
    return;
  }
  unsigned group_size = num_heads / kv_heads;
  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  // Threads per block: power-of-two <= 256 covering head_dim.
  unsigned threads = 1;
  while (threads < head_dim && threads < static_cast<unsigned>(kBlockSize))
    threads <<= 1;
  if (threads < 32u) threads = 32u;

  size_t shmem = (static_cast<size_t>(threads) + head_dim) * sizeof(float);
  flash_decode_kernel<<<num_heads, threads, shmem, stream>>>(
      out, q, k_cache, v_cache, seq_len, num_heads, kv_heads, head_dim,
      group_size, scale);
  CUDA_CHECK_KERNEL();
}

}  // namespace cuda
}  // namespace oxidize
