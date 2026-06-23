// Batched dense fp32 matrix multiply for training forward/backward.
// Y[t,r] = sum_c W[r,c] * X[t,c]   (Y=[T x rows], W=[rows x cols], X=[T x cols])
// Parallelized via OpenMP over rows; uses the same AVX2/FMA dot_f32 as tensor_cpu.

#include "oxidize/train_types.hpp"  // just for include guard
#include <cstddef>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__F16C__) && defined(__AVX2__)
#include <immintrin.h>
#define TRAIN_HAVE_AVX2 1
#endif

namespace oxidize {

namespace {

inline float dot_f32_train(const float* __restrict a, const float* __restrict b, size_t n) {
#ifdef TRAIN_HAVE_AVX2
  if (n >= 8 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
      acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i),    acc0);
      acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8),  acc1);
      acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), acc2);
      acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), acc3);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0,acc1), _mm256_add_ps(acc2,acc3));
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc,1);
    __m128 s = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s,s); s = _mm_hadd_ps(s,s);
    float sum = _mm_cvtss_f32(s);
    for (; i < n; ++i) sum += a[i]*b[i];
    return sum;
  }
#endif
  float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
  size_t i = 0;
  for (; i+8 <= n; i+=8) {
    s0+=a[i]*b[i]; s1+=a[i+1]*b[i+1]; s2+=a[i+2]*b[i+2]; s3+=a[i+3]*b[i+3];
    s4+=a[i+4]*b[i+4]; s5+=a[i+5]*b[i+5]; s6+=a[i+6]*b[i+6]; s7+=a[i+7]*b[i+7];
  }
  float sum = ((s0+s1)+(s2+s3))+((s4+s5)+(s6+s7));
  for (; i < n; ++i) sum += a[i]*b[i];
  return sum;
}

}  // namespace

// Dense batched GEMM: Y[T x rows] = X[T x cols] * W^T[cols x rows]
// W is row-major [rows x cols]; each Y[t,r] = dot(W[r,:], X[t,:])
void train_batch_matvec(float* Y, const float* W, const float* X,
                         size_t T, size_t rows, size_t cols) {
  // Parallelize over rows (output features). For each row r, iterate over T.
  // This is cache-friendly: W[r,:] is loaded once per row, dot against each X[t,:].
  // Better: parallelize over T*rows combinations.
#pragma omp parallel for schedule(static)
  for (long long idx = 0; idx < (long long)(T * rows); ++idx) {
    size_t t = static_cast<size_t>(idx) / rows;
    size_t r = static_cast<size_t>(idx) % rows;
    Y[t * rows + r] = dot_f32_train(W + r * cols, X + t * cols, cols);
  }
}

// Accumulate outer product: dW[rows x cols] += dY[T x rows]^T * X[T x cols]
// dW[r,c] += sum_t dY[t,r] * X[t,c]
void train_outer_accum(float* dW, const float* dY, const float* X,
                        size_t T, size_t rows, size_t cols) {
#pragma omp parallel for schedule(static)
  for (long long r = 0; r < (long long)rows; ++r) {
    float* dW_r = dW + r * cols;
    for (size_t t = 0; t < T; ++t) {
      float dy = dY[t * rows + r];
      const float* x_t = X + t * cols;
      for (size_t c = 0; c < cols; ++c) dW_r[c] += dy * x_t[c];
    }
  }
}

// Accumulate W^T * dY: dX[T x cols] += dY[T x rows] * W[rows x cols]
// dX[t,c] += sum_r dY[t,r] * W[r,c]
//
// Row-major parallel implementation: each thread handles a block of rows,
// accumulates into a thread-private dX buffer (T x cols), then merges into global dX.
// This reads W sequentially (cache-friendly) vs column-major strided reads.
void train_wt_dy(float* dX, const float* W, const float* dY,
                  size_t T, size_t rows, size_t cols) {
  const size_t NTHREADS = static_cast<size_t>(omp_get_max_threads());
  const size_t ROW_BLOCK = (rows + NTHREADS - 1) / NTHREADS;
  const size_t dX_size = T * cols;

#pragma omp parallel
  {
    const size_t tid = static_cast<size_t>(omp_get_thread_num());
    const size_t r0 = tid * ROW_BLOCK;
    if (r0 < rows) {
      const size_t r1 = (r0 + ROW_BLOCK < rows) ? (r0 + ROW_BLOCK) : rows;

      std::vector<float> loc(dX_size, 0.0f);
      for (size_t r = r0; r < r1; ++r) {
        const float* W_r = W + r * cols;
        for (size_t t = 0; t < T; ++t) {
          float dy_tr = dY[t * rows + r];
          if (dy_tr == 0.0f) continue;
          float* lo = loc.data() + t * cols;
          size_t c = 0;
#ifdef TRAIN_HAVE_AVX2
          if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            __m256 vdy = _mm256_set1_ps(dy_tr);
            for (; c + 8 <= cols; c += 8) {
              _mm256_storeu_ps(lo + c,
                _mm256_fmadd_ps(vdy, _mm256_loadu_ps(W_r + c), _mm256_loadu_ps(lo + c)));
            }
          }
#endif
          for (; c < cols; ++c) lo[c] += dy_tr * W_r[c];
        }
      }
#pragma omp critical
      {
        for (size_t i = 0; i < dX_size; ++i) dX[i] += loc[i];
      }
    }
  }
}

}  // namespace oxidize
