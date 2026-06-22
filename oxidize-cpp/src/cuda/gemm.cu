// GEMV / GEMM kernels and the cuBLASLt F16 (+ FP8 e4m3 on sm_90) GEMM path,
// plus the SwiGLU / GeGLU elementwise activations.
//
// Ported from:
//   oxidize-core/src/backends/cuda.rs
//     - GEMV_F16_KERNEL_NAME: f16 weight GEMV accumulating dot products in f32
//       (cuBLAS Hgemm's f16 accumulation drifts unacceptably for LLMs).
//     - gemv_f32_cuda / gemm_f32_cuda (cuBLAS row-major-via-transpose trick).
//   oxidize-cpp tensor_cpu.cpp  (swiglu_inplace, geglu_inplace scalar math).
//
// The GEMV kernels use one warp (32 lanes) per output row and a warp-shuffle
// reduction, matching the `rows * 32` thread launch geometry in cuda.rs.
//
// FP8 path: cuBLASLt's e4m3 matmul (CUDA_R_8F_E4M3) requires sm_90. We probe the
// device compute capability once; on sm_90 with both operands in range we use
// e4m3 inputs with f32 accumulation, otherwise we fall back to an F16 (CUDA_R_16F
// input, CUDA_R_32F compute) matmul which is correct on sm_80 and sm_90.

#include "cuda_common.cuh"

#include <cublasLt.h>
#include <cuda_fp8.h>

namespace oxidize {
namespace cuda {

namespace {

// Warp reduction (sum) over 32 lanes.
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

// y[row] = sum_c W[row*cols + c] * x[c], W is f16. One warp per row, f32 accum.
__global__ void gemv_f16_kernel(const __half* __restrict__ W,
                                const float* __restrict__ x,
                                float* __restrict__ y, unsigned rows,
                                unsigned cols) {
  unsigned global_warp =
      (blockIdx.x * blockDim.x + threadIdx.x) / kRowWarp;
  unsigned lane = threadIdx.x % kRowWarp;
  if (global_warp >= rows) return;
  const __half* row = W + static_cast<size_t>(global_warp) * cols;
  float acc = 0.0f;
  for (unsigned c = lane; c < cols; c += kRowWarp) {
    acc += __half2float(row[c]) * x[c];
  }
  acc = warp_reduce_sum(acc);
  if (lane == 0) y[global_warp] = acc;
}

// y[row] = sum_c W[row*cols + c] * x[c], W is f32. One warp per row.
__global__ void gemv_f32_kernel(const float* __restrict__ W,
                                const float* __restrict__ x,
                                float* __restrict__ y, unsigned rows,
                                unsigned cols) {
  unsigned global_warp =
      (blockIdx.x * blockDim.x + threadIdx.x) / kRowWarp;
  unsigned lane = threadIdx.x % kRowWarp;
  if (global_warp >= rows) return;
  const float* row = W + static_cast<size_t>(global_warp) * cols;
  float acc = 0.0f;
  for (unsigned c = lane; c < cols; c += kRowWarp) {
    acc += row[c] * x[c];
  }
  acc = warp_reduce_sum(acc);
  if (lane == 0) y[global_warp] = acc;
}

// silu(g) * up ; out may alias gate.
__global__ void swiglu_kernel(float* __restrict__ gate,
                              const float* __restrict__ up,
                              float* __restrict__ out, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = gate[i];
  float sigmoid = 1.0f / (1.0f + __expf(-g));
  out[i] = g * sigmoid * up[i];
}

// gelu_tanh(g) * up (Gemma gelu_pytorch_tanh). Matches the CPU scalar formula:
//   0.5 * g * (1 + tanh( sqrt(2/pi) * (g + 0.044715 * g^3) )).
__global__ void geglu_kernel(float* __restrict__ gate,
                             const float* __restrict__ up,
                             float* __restrict__ out, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = gate[i];
  const float kSqrt2OverPi = 0.7978845608028654f;
  float inner = kSqrt2OverPi * (g + 0.044715f * g * g * g);
  float gelu = 0.5f * g * (1.0f + tanhf(inner));
  out[i] = gelu * up[i];
}

// --- cuBLASLt GEMM machinery ----------------------------------------------

cublasLtHandle_t g_lt_handle = nullptr;
bool g_lt_inited = false;
int g_sm_major = 0;

void ensure_lt() {
  if (!g_lt_inited) {
    if (cublasLtCreate(&g_lt_handle) != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "cublasLtCreate failed\n");
      std::abort();
    }
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    g_sm_major = prop.major;
    g_lt_inited = true;
  }
}

__global__ void f32_to_f16_kernel(const float* __restrict__ in,
                                  __half* __restrict__ out, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

__global__ void f32_to_e4m3_kernel(const float* __restrict__ in,
                                   __nv_fp8_storage_t* __restrict__ out,
                                   unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = __nv_cvt_float_to_fp8(in[i], __NV_SATFINITE, __NV_E4M3);
  }
}

// Run a single Lt matmul: C = alpha*op(A)*op(B) + beta*C.
void lt_matmul(cudaDataType_t in_type, cublasComputeType_t compute_type,
               const void* A, const void* B, float* C, int m, int n, int k,
               int lda, int ldb, int ldc, cublasOperation_t opA,
               cublasOperation_t opB, void* workspace, size_t ws_bytes,
               cudaStream_t stream) {
  cublasLtMatmulDesc_t op_desc = nullptr;
  if (cublasLtMatmulDescCreate(&op_desc, compute_type, CUDA_R_32F) !=
      CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasLtMatmulDescCreate failed\n");
    std::abort();
  }
  cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA,
                                 sizeof(opA));
  cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB,
                                 sizeof(opB));

  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  // op(A) is m x k, op(B) is k x n, C is m x n, all column-major to Lt.
  unsigned a_rows = (opA == CUBLAS_OP_N) ? m : k;
  unsigned a_cols = (opA == CUBLAS_OP_N) ? k : m;
  unsigned b_rows = (opB == CUBLAS_OP_N) ? k : n;
  unsigned b_cols = (opB == CUBLAS_OP_N) ? n : k;
  cublasLtMatrixLayoutCreate(&a_layout, in_type, a_rows, a_cols, lda);
  cublasLtMatrixLayoutCreate(&b_layout, in_type, b_rows, b_cols, ldb);
  cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_32F, m, n, ldc);

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
      sizeof(ws_bytes));

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  cublasLtMatmulAlgoGetHeuristic(g_lt_handle, op_desc, a_layout, b_layout,
                                 c_layout, c_layout, pref, 1, &heuristic,
                                 &returned);
  if (returned == 0) {
    std::fprintf(stderr, "cublasLt: no matmul algo for GEMM\n");
    std::abort();
  }

  cublasStatus_t st =
      cublasLtMatmul(g_lt_handle, op_desc, &alpha, A, a_layout, B, b_layout,
                     &beta, C, c_layout, C, c_layout, &heuristic.algo,
                     workspace, ws_bytes, stream);
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasLtMatmul failed: %d\n", static_cast<int>(st));
    std::abort();
  }

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c_layout);
  cublasLtMatrixLayoutDestroy(b_layout);
  cublasLtMatrixLayoutDestroy(a_layout);
  cublasLtMatmulDescDestroy(op_desc);
}

}  // namespace

void launch_gemv_f16(const __half* W, const float* x, float* y, unsigned rows,
                     unsigned cols, cudaStream_t stream) {
  unsigned threads = rows * static_cast<unsigned>(kRowWarp);
  int grid = grid_for(threads, kBlockSize);
  gemv_f16_kernel<<<grid, kBlockSize, 0, stream>>>(W, x, y, rows, cols);
  CUDA_CHECK_KERNEL();
}

void launch_gemv_f32(const float* W, const float* x, float* y, unsigned rows,
                     unsigned cols, cudaStream_t stream) {
  unsigned threads = rows * static_cast<unsigned>(kRowWarp);
  int grid = grid_for(threads, kBlockSize);
  gemv_f32_kernel<<<grid, kBlockSize, 0, stream>>>(W, x, y, rows, cols);
  CUDA_CHECK_KERNEL();
}

void launch_swiglu(float* gate, const float* up, float* out, unsigned n,
                   cudaStream_t stream) {
  int grid = grid_for(n, kBlockSize);
  swiglu_kernel<<<grid, kBlockSize, 0, stream>>>(gate, up, out, n);
  CUDA_CHECK_KERNEL();
}

void launch_geglu(float* gate, const float* up, float* out, unsigned n,
                  cudaStream_t stream) {
  int grid = grid_for(n, kBlockSize);
  geglu_kernel<<<grid, kBlockSize, 0, stream>>>(gate, up, out, n);
  CUDA_CHECK_KERNEL();
}

// ---------------------------------------------------------------------------
// Row-major C[m x n] = A[m x k] * B[k x n], device f32 inputs/output.
//
// cuBLASLt is column-major. We compute C^T (n x m, column-major) = B^T * A^T,
// which in column-major terms with our row-major buffers is exactly: treat B
// (row-major k x n) as a column-major n x k matrix and A (row-major m x k) as a
// column-major k x m matrix, multiply (n x k)*(k x m) = (n x m) col-major == C
// row-major (m x n). So: op = N for both, leading dims = n, k, n.
//
// FP8 path (sm_90): cast A,B to e4m3 and matmul with f32 accumulation. Falls
// back to F16 inputs on sm_80.
// ---------------------------------------------------------------------------
void cuda_gemm_device(const float* dA, const float* dB, float* dC, int m, int k,
                      int n, void* scratch, size_t scratch_bytes,
                      cudaStream_t stream) {
  ensure_lt();

  const size_t a_elems = static_cast<size_t>(m) * k;
  const size_t b_elems = static_cast<size_t>(k) * n;
  const bool use_fp8 = (g_sm_major >= 9);

  // Cast inputs into the scratch arena.
  if (use_fp8) {
    __nv_fp8_storage_t* a8 = reinterpret_cast<__nv_fp8_storage_t*>(scratch);
    __nv_fp8_storage_t* b8 = a8 + a_elems;
    (void)scratch_bytes;
    f32_to_e4m3_kernel<<<grid_for(a_elems, kBlockSize), kBlockSize, 0,
                         stream>>>(dA, a8, static_cast<unsigned>(a_elems));
    CUDA_CHECK_KERNEL();
    f32_to_e4m3_kernel<<<grid_for(b_elems, kBlockSize), kBlockSize, 0,
                         stream>>>(dB, b8, static_cast<unsigned>(b_elems));
    CUDA_CHECK_KERNEL();
    // C^T(n x m) = B(as n x k col-major) * A(as k x m col-major).
    lt_matmul(CUDA_R_8F_E4M3, CUBLAS_COMPUTE_32F, b8, a8, dC, n, m, k, n, k, n,
              CUBLAS_OP_N, CUBLAS_OP_N, nullptr, 0, stream);
  } else {
    __half* a16 = reinterpret_cast<__half*>(scratch);
    __half* b16 = a16 + a_elems;
    f32_to_f16_kernel<<<grid_for(a_elems, kBlockSize), kBlockSize, 0, stream>>>(
        dA, a16, static_cast<unsigned>(a_elems));
    CUDA_CHECK_KERNEL();
    f32_to_f16_kernel<<<grid_for(b_elems, kBlockSize), kBlockSize, 0, stream>>>(
        dB, b16, static_cast<unsigned>(b_elems));
    CUDA_CHECK_KERNEL();
    lt_matmul(CUDA_R_16F, CUBLAS_COMPUTE_32F, b16, a16, dC, n, m, k, n, k, n,
              CUBLAS_OP_N, CUBLAS_OP_N, nullptr, 0, stream);
  }
}

}  // namespace cuda
}  // namespace oxidize
