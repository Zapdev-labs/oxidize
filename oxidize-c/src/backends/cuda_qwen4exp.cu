/*
 * cuda_qwen4exp.cu — device kernels for qwen4exp (CYBER-FROST) decode.
 *
 * Hyper-connections (hc_mix / hc_combine around every sublayer) and the
 * single PLE n-gram layer. Each kernel mirrors the matching q4e_* function
 * in src/model/llama.c; reductions run in f32 tree order, so results agree
 * with the CPU to f32 tolerance rather than bit-exactly. The projections
 * between these kernels (hc down/up/inject, PLE key/value) are ordinary
 * matvecs dispatched by cuda.cu.
 */
#include "oxidize/cuda_kernels.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define Q4E_BLOCK 256u

__device__ __forceinline__ float q4e_sigmoid(float x)
{
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

/* Block-wide sum; blockDim.x must be a power of two <= Q4E_BLOCK. */
__device__ float q4e_block_sum(float v, float *red)
{
    const uint32_t tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2u; s > 0u; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

/* One block per stream. */
__global__ void k_q4e_norm_streams(const float *res, const float *gamma,
                                   float *xn, uint32_t d, float eps)
{
    __shared__ float red[Q4E_BLOCK];
    const uint32_t c = blockIdx.x;
    const float *rc = res + (size_t)c * d;
    const float *g = gamma + (size_t)c * d;
    float *xo = xn + (size_t)c * d;
    float ss = 0.0f;
    for (uint32_t e = threadIdx.x; e < d; e += blockDim.x) ss += rc[e] * rc[e];
    const float inv = rsqrtf(q4e_block_sum(ss, red) / (float)d + eps);
    for (uint32_t e = threadIdx.x; e < d; e += blockDim.x)
        xo[e] = rc[e] * inv * g[e];
}

__global__ void k_q4e_silu_scale(float *lo, uint32_t n, float scale)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = lo[i] * scale;
    lo[i] = v * q4e_sigmoid(v);
}

__global__ void k_q4e_hc_fold(const float *xn, const float *gate, float *out,
                              uint32_t n_streams, uint32_t d)
{
    const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= d) return;
    float acc = 0.0f;
    for (uint32_t c = 0; c < n_streams; c++) {
        const size_t i = (size_t)c * d + e;
        acc += xn[i] * q4e_sigmoid(gate[i]);
    }
    out[e] = acc * (1.0f / (float)n_streams);
}

__global__ void k_q4e_hc_combine(float *res, const float *block,
                                 const float *inject, uint32_t n_streams,
                                 uint32_t d)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n_streams * d) return;
    const uint32_t c = (uint32_t)(i / d);
    const uint32_t e = (uint32_t)(i % d);
    const float w = inject
        ? 2.0f * q4e_sigmoid(inject[c] / (float)n_streams)
        : 1.0f;     /* absent inject: CPU zero-fills, 2·sigmoid(0) = 1 */
    res[i] += block[e] * w;
}

/* One block per stream. */
__global__ void k_q4e_ple_gate(const float *kn, const float *qn,
                               const float *val, float *gated, uint32_t d)
{
    __shared__ float red[Q4E_BLOCK];
    const uint32_t c = blockIdx.x;
    const float *k = kn + (size_t)c * d;
    const float *q = qn + (size_t)c * d;
    float acc = 0.0f;
    for (uint32_t e = threadIdx.x; e < d; e += blockDim.x) acc += k[e] * q[e];
    const float sc = q4e_block_sum(acc, red) * rsqrtf((float)d);
    const float mag = sqrtf(fmaxf(fabsf(sc), 1e-6f));
    const float g = q4e_sigmoid(sc < 0.0f ? -mag : mag);
    float *o = gated + (size_t)c * d;
    for (uint32_t e = threadIdx.x; e < d; e += blockDim.x) o[e] = val[e] * g;
}

/* One thread per channel. Tap k (< kern-1) reads ring row k*dil, which is
 * (kern-1-k)*dil positions back; the last tap reads this token's input.
 * Each thread owns its ring column, so the shift needs no barrier. */
__global__ void k_q4e_ple_conv(const float *conv_w, float *ring,
                               const float *conv_in, const float *gated,
                               float *resid, uint32_t n_ch, uint32_t kern,
                               uint32_t dil)
{
    const uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= n_ch) return;
    const size_t hist = (size_t)(kern - 1u) * dil;
    const float *w = conv_w + (size_t)ch * kern;
    float acc = 0.0f;
    for (uint32_t k = 0; k + 1u < kern; k++)
        acc += w[k] * ring[(size_t)k * dil * n_ch + ch];
    const float x = conv_in[ch];
    acc += w[kern - 1u] * x;
    resid[ch] += gated[ch] + acc * q4e_sigmoid(acc);
    for (size_t t = 0; t + 1u < hist; t++)
        ring[t * n_ch + ch] = ring[(t + 1u) * n_ch + ch];
    ring[(hist - 1u) * n_ch + ch] = x;
}

static uint32_t q4e_grid(size_t n)
{
    return (uint32_t)((n + Q4E_BLOCK - 1u) / Q4E_BLOCK);
}

extern "C" {

bool oc_cuda_q4e_norm_streams(const float *d_res, const float *d_gamma,
                              float *d_xn, uint32_t n_streams, uint32_t d,
                              float eps, void *stream)
{
    if (!d_res || !d_gamma || !d_xn || n_streams == 0u || d == 0u)
        return false;
    k_q4e_norm_streams<<<n_streams, Q4E_BLOCK, 0, (cudaStream_t)stream>>>(
        d_res, d_gamma, d_xn, d, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool oc_cuda_q4e_silu_scale(float *d_lo, uint32_t n, float scale,
                            void *stream)
{
    if (!d_lo || n == 0u) return false;
    k_q4e_silu_scale<<<q4e_grid(n), Q4E_BLOCK, 0, (cudaStream_t)stream>>>(
        d_lo, n, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool oc_cuda_q4e_hc_fold(const float *d_xn, const float *d_gate,
                         float *d_out, uint32_t n_streams, uint32_t d,
                         void *stream)
{
    if (!d_xn || !d_gate || !d_out || n_streams == 0u || d == 0u)
        return false;
    k_q4e_hc_fold<<<q4e_grid(d), Q4E_BLOCK, 0, (cudaStream_t)stream>>>(
        d_xn, d_gate, d_out, n_streams, d);
    return cudaGetLastError() == cudaSuccess;
}

bool oc_cuda_q4e_hc_combine(float *d_res, const float *d_block,
                            const float *d_inject, uint32_t n_streams,
                            uint32_t d, void *stream)
{
    if (!d_res || !d_block || n_streams == 0u || d == 0u) return false;
    k_q4e_hc_combine<<<q4e_grid((size_t)n_streams * d), Q4E_BLOCK, 0,
                       (cudaStream_t)stream>>>(d_res, d_block, d_inject,
                                               n_streams, d);
    return cudaGetLastError() == cudaSuccess;
}

bool oc_cuda_q4e_ple_gate(const float *d_kn, const float *d_qn,
                          const float *d_val, float *d_gated,
                          uint32_t n_streams, uint32_t d, void *stream)
{
    if (!d_kn || !d_qn || !d_val || !d_gated || n_streams == 0u || d == 0u)
        return false;
    k_q4e_ple_gate<<<n_streams, Q4E_BLOCK, 0, (cudaStream_t)stream>>>(
        d_kn, d_qn, d_val, d_gated, d);
    return cudaGetLastError() == cudaSuccess;
}

bool oc_cuda_q4e_ple_conv(const float *d_conv_w, float *d_ring,
                          const float *d_conv_in, const float *d_gated,
                          float *d_resid, uint32_t n_ch, uint32_t kern,
                          uint32_t dil, void *stream)
{
    if (!d_conv_w || !d_ring || !d_conv_in || !d_gated || !d_resid ||
        n_ch == 0u || kern < 2u || dil == 0u)
        return false;
    k_q4e_ple_conv<<<q4e_grid(n_ch), Q4E_BLOCK, 0, (cudaStream_t)stream>>>(
        d_conv_w, d_ring, d_conv_in, d_gated, d_resid, n_ch, kern, dil);
    return cudaGetLastError() == cudaSuccess;
}

} /* extern "C" */
