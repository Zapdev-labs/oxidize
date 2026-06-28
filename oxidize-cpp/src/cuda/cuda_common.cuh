#pragma once
// Shared GPU backend internals (CUDA or ROCm-HIP): error-checking macros and
// device kernel launch declarations used across the .cu translation units.
//
// Ported from: oxidize-core/src/backends/cuda.rs (kernel geometry: 256 threads
// per block, rows*32 threads for the per-row GEMV kernels) and
// oxidize-core/src/compute/flash_attention.rs (online-softmax decode).

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "gpu_common.cuh"

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define GPU_CHECK(expr)                                                      \
  do {                                                                       \
    cudaError_t _ox_err = (expr);                                            \
    if (_ox_err != cudaSuccess) {                                            \
      std::fprintf(stderr, "GPU error %s at %s:%d: %s\n", #expr, __FILE__,   \
                   __LINE__, cudaGetErrorString(_ox_err));                 \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

#define CUDA_CHECK GPU_CHECK

// Check after a kernel launch (captures launch + async errors).
#define GPU_CHECK_KERNEL()                                                   \
  do {                                                                       \
    cudaError_t _ox_err = cudaGetLastError();                                \
    if (_ox_err != cudaSuccess) {                                            \
      std::fprintf(stderr, "GPU kernel launch error at %s:%d: %s\n",         \
                   __FILE__, __LINE__, cudaGetErrorString(_ox_err));       \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

#define CUDA_CHECK_KERNEL GPU_CHECK_KERNEL

namespace oxidize {
namespace cuda {

// Threads per block used by all the per-row GEMV / dequant kernels (matches the
// `block_size = 256` constant in backends/cuda.rs).
constexpr int kBlockSize = 256;
// One warp (32 lanes) cooperates per output row in the GEMV kernels, again
// mirroring the `rows * 32` thread count in backends/cuda.rs.
constexpr int kRowWarp = 32;

inline int grid_for(unsigned total, int block) {
  return static_cast<int>((total + static_cast<unsigned>(block) - 1u) /
                          static_cast<unsigned>(block));
}

// ---------------------------------------------------------------------------
// Dequant kernel launchers (dequant.cu). `src` is the raw packed quantized
// matrix; `dst` receives `n_blocks * vals_per_block` __half values. n_blocks is
// the total block count across all rows.
// ---------------------------------------------------------------------------
void launch_dequant_q8_0(const uint8_t* src, __half* dst, unsigned n_blocks,
                         cudaStream_t stream);
void launch_dequant_q4_k(const uint8_t* src, __half* dst, unsigned n_blocks,
                         cudaStream_t stream);
void launch_dequant_q6_k(const uint8_t* src, __half* dst, unsigned n_blocks,
                         cudaStream_t stream);
void launch_dequant_q2_k(const uint8_t* src, __half* dst, unsigned n_blocks,
                         cudaStream_t stream);

// ---------------------------------------------------------------------------
// GEMV f16 kernel (gemm.cu): y[row] = sum_c W_f16[row*cols + c] * x[c],
// accumulated in f32. Matches GEMV_F16_KERNEL_NAME semantics in cuda.rs.
// ---------------------------------------------------------------------------
void launch_gemv_f16(const __half* W, const float* x, float* y, unsigned rows,
                     unsigned cols, cudaStream_t stream);

// GEMV f32 kernel (gemm.cu): y = W_f32 * x (row-major). Used as the non-cuBLAS
// path / fallback and to keep numerics identical to the CPU matvec.
void launch_gemv_f32(const float* W, const float* x, float* y, unsigned rows,
                     unsigned cols, cudaStream_t stream);

// Row-major C[m x n] = A[m x k] * B[k x n] on device pointers via cuBLASLt
// (FP8 e4m3 on sm_90, F16 on sm_80). `scratch` must hold the cast operands:
//   FP8: (m*k + k*n) bytes ; F16: (m*k + k*n) * sizeof(__half) bytes.
void cuda_gemm_device(const float* dA, const float* dB, float* dC, int m, int k,
                      int n, void* scratch, size_t scratch_bytes,
                      cudaStream_t stream);

// ---------------------------------------------------------------------------
// Elementwise / norm kernels.
// ---------------------------------------------------------------------------
// rmsnorm.cu
void launch_rms_norm(float* out, const float* x, const float* weight,
                     unsigned n, float eps, bool weight_plus_one,
                     cudaStream_t stream);
// rope.cu
void launch_apply_rope(float* vec, unsigned head_dim, unsigned num_heads,
                       unsigned pos, float theta, unsigned rope_len,
                       cudaStream_t stream);
// Same as above but `d_pos` is a device pointer (for CUDA graph replay).
void launch_apply_rope_dpos(float* vec, unsigned head_dim, unsigned num_heads,
                            const unsigned* d_pos, float theta,
                            unsigned rope_len, cudaStream_t stream);
// gemm.cu (small elementwise helpers kept with the GEMV kernels)
void launch_swiglu(float* gate, const float* up, float* out, unsigned n,
                   cudaStream_t stream);
void launch_geglu(float* gate, const float* up, float* out, unsigned n,
                  cudaStream_t stream);

// ---------------------------------------------------------------------------
// Flash decode attention (flash_attn.cu): fused GQA online-softmax decode.
//   q       : [num_heads * head_dim]
//   k_cache : [seq_len * kv_heads * head_dim]   (row per position)
//   v_cache : [seq_len * kv_heads * head_dim]
//   out     : [num_heads * head_dim]
// One block per query head; group_size = num_heads / kv_heads.
// ---------------------------------------------------------------------------
void launch_flash_decode(float* out, const float* q, const float* k_cache,
                         const float* v_cache, unsigned seq_len,
                         unsigned num_heads, unsigned kv_heads,
                         unsigned head_dim, cudaStream_t stream);
// seq_len = *d_pos + 1 (device-side position for CUDA graph replay).
void launch_flash_decode_dpos(float* out, const float* q, const float* k_cache,
                              const float* v_cache, const unsigned* d_pos,
                              unsigned num_heads, unsigned kv_heads,
                              unsigned head_dim, cudaStream_t stream);

// ---------------------------------------------------------------------------
// resident.cu: elementwise helpers for the GPU-resident decode path.
void launch_residual_add(float* y, const float* x, unsigned n,
                         cudaStream_t stream);
void launch_add_bias_mod(float* y, const float* bias, unsigned n,
                         unsigned bias_len, cudaStream_t stream);
// Append K/V vectors to layer `layer` slot at position *d_pos % ctx.
void launch_kv_append(float* kvk, float* kvv, const float* k, const float* v,
                      unsigned kv_tok, unsigned ctx, unsigned layer,
                      const unsigned* d_pos, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Sampling (sampling.cu): softmax in place + argmax.
// ---------------------------------------------------------------------------
void launch_softmax_inplace(float* x, unsigned n, cudaStream_t stream);
// Writes the argmax index into *out_index (device pointer, 1 uint32).
void launch_argmax(const float* logits, unsigned n, uint32_t* out_index,
                   cudaStream_t stream);

}  // namespace cuda
}  // namespace oxidize
