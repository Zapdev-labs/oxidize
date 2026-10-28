/* Standalone microbenchmark for the fp16 mma GEMM kernels: times
 * k_mmq_gemm_xh_q4k / k_mmq_gemm_xh_q6k at realistic model shapes.
 * Not part of the build — compiled ad hoc on the bench machine:
 *   nvcc -O2 -gencode arch=compute_75,code=sm_75 \
 *     src/backends/bench_mma.cu src/backends/cuda_mmq.cuo -o /tmp/bench_mma */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern __global__ void
k_mmq_gemm_xh_q4k(const uint8_t *w, const __half *xh,
                  float *out, uint32_t n_tok, size_t out_stride,
                  size_t rows, uint32_t n_groups, size_t row_bytes);
extern __global__ void
k_mmq_gemm_xh_q6k(const uint8_t *w, const __half *xh,
                  float *out, uint32_t n_tok, size_t out_stride,
                  size_t rows, uint32_t n_groups, size_t row_bytes);

#define MG_THREADS_B 256
#define MG_SMEM_B    (16384 + 16384)

static void bench_shape(const char *tag, bool q6, size_t rows, size_t cols,
                        uint32_t n_tok, int iters)
{
    /* device block sizes: Q6_K is padded 210 -> 224 (see cuda.cu packer) */
    const size_t blk = q6 ? 224 : 144;
    const size_t row_bytes = (cols / 256) * blk;
    const uint32_t n_groups = (uint32_t)(cols / 32);
    uint8_t *dw; __half *dx; float *dout;
    cudaMalloc(&dw, rows * row_bytes);
    cudaMalloc(&dx, (size_t)n_tok * n_groups * 32 * sizeof(__half));
    cudaMalloc(&dout, (size_t)n_tok * rows * sizeof(float));
    cudaMemset(dw, 0x3c, rows * row_bytes);
    cudaMemset(dx, 0, (size_t)n_tok * n_groups * 32 * sizeof(__half));
    cudaMemset(dout, 0, (size_t)n_tok * rows * sizeof(float));

    const uint32_t tt = (n_tok < 32u) ? n_tok : 32u;
    dim3 grid((unsigned)((rows + 31) / 32), (unsigned)((n_tok + tt - 1) / tt));
    for (int i = 0; i < 3; i++) {
        if (q6)
            k_mmq_gemm_xh_q6k<<<grid, MG_THREADS_B, MG_SMEM_B>>>(
                dw, dx, dout, n_tok, rows, rows, n_groups, row_bytes);
        else
            k_mmq_gemm_xh_q4k<<<grid, MG_THREADS_B, MG_SMEM_B>>>(
                dw, dx, dout, n_tok, rows, rows, n_groups, row_bytes);
    }
    cudaDeviceSynchronize();
    {
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("%s %zux%zux%u: LAUNCH FAIL %s\n", tag, rows, cols, n_tok,
                   cudaGetErrorString(err));
            return;
        }
    }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) {
        if (q6)
            k_mmq_gemm_xh_q6k<<<grid, MG_THREADS_B, MG_SMEM_B>>>(
                dw, dx, dout, n_tok, rows, rows, n_groups, row_bytes);
        else
            k_mmq_gemm_xh_q4k<<<grid, MG_THREADS_B, MG_SMEM_B>>>(
                dw, dx, dout, n_tok, rows, rows, n_groups, row_bytes);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    const double us = ms * 1000.0 / iters;
    const double wbytes = (double)rows * row_bytes;
    const double flops = 2.0 * rows * cols * n_tok;
    printf("%s %5zux%5zux%3u: %7.1f us  w=%6.1f GB/s  %6.1f GFLOP/s  blocks=%ux%u\n",
           tag, rows, cols, n_tok, us, wbytes / (us * 1e3),
           flops / (us * 1e3), grid.x, grid.y);
    cudaFree(dw); cudaFree(dx); cudaFree(dout);
}

int main(void)
{
    int dev = 0; cudaDeviceProp p;
    cudaGetDevice(&dev); cudaGetDeviceProperties(&p, dev);
    printf("%s sm_%d%d, %d SMs\n", p.name, p.major, p.minor,
           p.multiProcessorCount);
    for (uint32_t nt = 16; nt <= 64; nt *= 2) {
        bench_shape("q4k", false, 1536, 1536, nt, 200);
        bench_shape("q4k", false,  256, 1536, nt, 200);
        bench_shape("q4k", false, 8960, 1536, nt, 100);
        bench_shape("q6k", true,  1536, 8960, nt, 100);
        bench_shape("q6k", true,  8960, 8960, nt, 50);
    }
    return 0;
}
