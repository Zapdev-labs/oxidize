/*
 * cuda_qwen35.cu — Ampere kernels for Qwen3.5 / Qwen 3.8 hybrid decode.
 *
 * Full-attention layers: packed Q/gate unpack, per-head QK-norm + RoPE,
 * sigmoid attention gate. Recurrent layers: causal conv1d, L2-normalized
 * Q/K, Gated DeltaNet state update, RMSNorm ⊕ SiLU output gate.
 *
 * Geometry matches src/model/qwen35_delta.c bit for bit; the reduction
 * order differs so results agree to f32 tolerance, not bit-exactly.
 */
#include "oxidize/cuda_kernels.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define Q35_CHECK(call)                                                      \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) return false;                                 \
    } while (0)

__device__ __forceinline__ float q35_sigmoid(float x)
{
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

__device__ __forceinline__ float q35_silu(float x)
{
    if (x >= 0.0f) return x / (1.0f + expf(-x));
    const float e = expf(x);
    return x * e / (1.0f + e);
}

__device__ __forceinline__ float q35_softplus(float x)
{
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

/* ─── QK-norm + RoPE: one block per head ──────────────────────────────── */

__global__ void k_qk_norm_rope(float *x, const float *weight,
                               uint32_t n_heads, uint32_t head_dim,
                               uint32_t rope_dim, int64_t pos,
                               float theta, float eps,
                               float yarn_factor, uint32_t yarn_orig_ctx)
{
    const uint32_t head = blockIdx.x;
    if (head >= n_heads) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    float *h = x + (size_t)head * head_dim;

    extern __shared__ float red[];
    float sum = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += nthreads) sum += h[i] * h[i];
    red[tid] = sum;
    __syncthreads();
    for (uint32_t s = nthreads / 2u; s > 0u; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)head_dim + eps);

    if (weight != NULL) {
        for (uint32_t i = tid; i < head_dim; i += nthreads)
            h[i] = h[i] * inv * weight[i];
    } else {
        for (uint32_t i = tid; i < head_dim; i += nthreads) h[i] *= inv;
    }
    __syncthreads();

    if (rope_dim < 2u) return;

    const uint32_t half = rope_dim / 2u;
    const float freq_mul = powf(theta, -2.0f / (float)rope_dim);
    const bool yarn = yarn_factor > 1.0f && yarn_orig_ctx > 0u;
    const float mscale = yarn ? (1.0f + 0.1f * logf(yarn_factor)) : 1.0f;
    float corr_lo = 0.0f, corr_hi = 0.0f, denom = 0.001f;
    if (yarn) {
        const float base_log = 2.0f * logf(theta);
        const float ratio_fast =
            (float)yarn_orig_ctx / (32.0f * 2.0f * 3.14159265f);
        const float ratio_slow =
            (float)yarn_orig_ctx / (1.0f * 2.0f * 3.14159265f);
        corr_lo = floorf((float)rope_dim * logf(ratio_fast) / base_log);
        corr_hi = ceilf((float)rope_dim * logf(ratio_slow) / base_log);
        if (corr_lo < 0.0f) corr_lo = 0.0f;
        if (corr_hi > (float)rope_dim - 1.0f) corr_hi = (float)rope_dim - 1.0f;
        denom = corr_hi - corr_lo;
        if (denom < 0.001f) denom = 0.001f;
    }

    if (!yarn && pos == 0) return;

    for (uint32_t i = tid; i < half; i += nthreads) {
        float freq = powf(freq_mul, (float)i);
        float angle = (float)pos * freq;
        if (yarn) {
            if (pos == 0) {
                h[i] *= mscale;
                h[half + i] *= mscale;
                continue;
            }
            float theta_extrap = (float)pos * freq;
            float theta_interp = theta_extrap / yarn_factor;
            float ramp = 1.0f - ((float)i - corr_lo) / denom;
            if (ramp < 0.0f) ramp = 0.0f;
            if (ramp > 1.0f) ramp = 1.0f;
            angle = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
        }
        float c = cosf(angle) * mscale;
        float s = sinf(angle) * mscale;
        float x0 = h[i];
        float x1 = h[half + i];
        h[i] = x0 * c - x1 * s;
        h[half + i] = x0 * s + x1 * c;
    }
}

/* ─── Packed Q / gate unpack ──────────────────────────────────────────── */

__global__ void k_qwen35_unpack_qgate(const float *packed, float *q,
                                      float *gate, uint32_t n_heads,
                                      uint32_t head_dim)
{
    const size_t n = (size_t)n_heads * head_dim;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t head = (uint32_t)(i / head_dim);
    const uint32_t d = (uint32_t)(i % head_dim);
    const float *src = packed + (size_t)head * 2u * head_dim;
    q[i] = src[d];
    gate[i] = src[head_dim + d];
}

__global__ void k_sigmoid_gate(float *x, const float *gate, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= q35_sigmoid(gate[i]);
}

/* ─── Causal conv1d + SiLU, one thread per channel ────────────────────── */

__global__ void k_qwen35_conv1d(float *state, const float *weight,
                                const float *input, float *output,
                                uint32_t conv_dim, uint32_t kernel)
{
    const uint32_t ch = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;
    const uint32_t history = kernel - 1u;
    float *past = state + (size_t)ch * history;
    const float *w = weight + (size_t)ch * kernel;
    float sum = w[history] * input[ch];
    for (uint32_t i = 0; i < history; i++) sum += w[i] * past[i];
    for (uint32_t i = 0; i + 1u < history; i++) past[i] = past[i + 1u];
    if (history > 0u) past[history - 1u] = input[ch];
    output[ch] = q35_silu(sum);
}

/* L2-normalize Q and K key-heads in the conv output. One block per key head. */
__global__ void k_qwen35_norm_keys(float *conv, uint32_t n_key_heads,
                                   uint32_t dk, float eps)
{
    const uint32_t head = blockIdx.x;
    if (head >= n_key_heads) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const size_t key_total = (size_t)n_key_heads * dk;
    float *q = conv + (size_t)head * dk;
    float *k = conv + key_total + (size_t)head * dk;
    const float q_scale = rsqrtf((float)dk);

    extern __shared__ float red[];
    float qsum = 0.0f, ksum = 0.0f;
    for (uint32_t i = tid; i < dk; i += nthreads) {
        qsum += q[i] * q[i];
        ksum += k[i] * k[i];
    }
    red[tid] = qsum;
    red[nthreads + tid] = ksum;
    __syncthreads();
    for (uint32_t s = nthreads / 2u; s > 0u; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
            red[nthreads + tid] += red[nthreads + tid + s];
        }
        __syncthreads();
    }
    const float qinv = rsqrtf(fmaxf(red[0], eps));
    const float kinv = rsqrtf(fmaxf(red[nthreads], eps));
    for (uint32_t i = tid; i < dk; i += nthreads) {
        q[i] *= qinv * q_scale;
        k[i] *= kinv;
    }
}

/* Gated DeltaNet: one block per value head. Threads split the value dim. */
__global__ void k_qwen35_delta_heads(float *recurrent, const float *conv,
                                     const float *beta_in, const float *alpha,
                                     const float *ssm_a, const float *dt_bias,
                                     float *out,
                                     uint32_t n_key_heads, uint32_t n_value_heads,
                                     uint32_t dk, uint32_t dv)
{
    const uint32_t head = blockIdx.x;
    if (head >= n_value_heads) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const uint32_t v_per_k = n_value_heads / n_key_heads;
    const uint32_t key_head = head / v_per_k;
    const size_t key_total = (size_t)n_key_heads * dk;
    const float *q = conv + (size_t)key_head * dk;
    const float *k = conv + key_total + (size_t)key_head * dk;
    const float *v = conv + 2u * key_total + (size_t)head * dv;
    float *matrix = recurrent + (size_t)head * dv * dk;
    float *hout = out + (size_t)head * dv;

    extern __shared__ float sm[];
    float *sq = sm;
    float *sk = sm + dk;
    for (uint32_t j = tid; j < dk; j += nthreads) {
        sq[j] = q[j];
        sk[j] = k[j];
    }
    __syncthreads();

    const float beta = q35_sigmoid(beta_in[head]);
    const float decay = expf(ssm_a[head] * q35_softplus(alpha[head] + dt_bias[head]));
    const bool vec4 = (dk % 4u) == 0u;

    for (uint32_t i = tid; i < dv; i += nthreads) {
        float *row = matrix + (size_t)i * dk;
        float state_k = 0.0f;
        if (vec4) {
            for (uint32_t j = 0; j < dk; j += 4u) {
                float4 r = *reinterpret_cast<float4 *>(row + j);
                const float4 kv = *reinterpret_cast<const float4 *>(sk + j);
                r.x *= decay; r.y *= decay; r.z *= decay; r.w *= decay;
                state_k += r.x * kv.x + r.y * kv.y + r.z * kv.z + r.w * kv.w;
                *reinterpret_cast<float4 *>(row + j) = r;
            }
        } else {
            for (uint32_t j = 0; j < dk; j++) {
                row[j] *= decay;
                state_k += row[j] * sk[j];
            }
        }
        const float delta = (v[i] - state_k) * beta;
        float o = 0.0f;
        if (vec4) {
            for (uint32_t j = 0; j < dk; j += 4u) {
                float4 r = *reinterpret_cast<float4 *>(row + j);
                const float4 kv = *reinterpret_cast<const float4 *>(sk + j);
                const float4 qv = *reinterpret_cast<const float4 *>(sq + j);
                r.x += delta * kv.x;
                r.y += delta * kv.y;
                r.z += delta * kv.z;
                r.w += delta * kv.w;
                *reinterpret_cast<float4 *>(row + j) = r;
                o += r.x * qv.x + r.y * qv.y + r.z * qv.z + r.w * qv.w;
            }
        } else {
            for (uint32_t j = 0; j < dk; j++) {
                row[j] += delta * sk[j];
                o += row[j] * sq[j];
            }
        }
        hout[i] = o;
    }
}

/* RMSNorm over each value head, then SiLU(gate). One block per head. */
__global__ void k_qwen35_gate_heads(float *out, const float *gate,
                                    const float *norm_w, uint32_t n_heads,
                                    uint32_t dv, float eps)
{
    const uint32_t head = blockIdx.x;
    if (head >= n_heads) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    float *h = out + (size_t)head * dv;
    const float *g = gate + (size_t)head * dv;

    extern __shared__ float red[];
    float sum = 0.0f;
    for (uint32_t i = tid; i < dv; i += nthreads) sum += h[i] * h[i];
    red[tid] = sum;
    __syncthreads();
    for (uint32_t s = nthreads / 2u; s > 0u; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)dv + eps);
    for (uint32_t i = tid; i < dv; i += nthreads)
        h[i] = h[i] * inv * norm_w[i] * q35_silu(g[i]);
}

static uint32_t q35_block(uint32_t n, uint32_t cap)
{
    uint32_t b = n < cap ? n : cap;
    if (b < 32u) b = 32u;
    uint32_t p = 32u;
    while (p < b) p <<= 1;
    return p > cap ? cap : p;
}

#ifdef __cplusplus
extern "C" {
#endif

bool oc_cuda_qk_norm_rope(float *d_x, const float *d_weight,
                          uint32_t n_heads, uint32_t head_dim,
                          uint32_t rope_dim, int64_t pos, float theta,
                          float eps, float yarn_factor, uint32_t yarn_orig_ctx,
                          void *stream)
{
    if (!d_x || n_heads == 0u || head_dim == 0u || rope_dim > head_dim)
        return false;
    const uint32_t block = q35_block(head_dim, 256u);
    cudaStream_t s = (cudaStream_t)stream;
    k_qk_norm_rope<<<n_heads, block, block * sizeof(float), s>>>(
        d_x, d_weight, n_heads, head_dim, rope_dim, pos, theta, eps,
        yarn_factor, yarn_orig_ctx);
    return cudaGetLastError() == cudaSuccess;
}

bool oc_cuda_qwen35_unpack_qgate(const float *d_packed, float *d_q,
                                 float *d_gate, uint32_t n_heads,
                                 uint32_t head_dim, void *stream)
{
    if (!d_packed || !d_q || !d_gate || n_heads == 0u || head_dim == 0u)
        return false;
    const size_t n = (size_t)n_heads * head_dim;
    const uint32_t block = 256u;
    const uint32_t grid = (uint32_t)((n + block - 1u) / block);
    cudaStream_t s = (cudaStream_t)stream;
    k_qwen35_unpack_qgate<<<grid, block, 0, s>>>(
        d_packed, d_q, d_gate, n_heads, head_dim);
    return cudaGetLastError() == cudaSuccess;
}

bool oc_cuda_sigmoid_gate(float *d_x, const float *d_gate, size_t n,
                          void *stream)
{
    if (!d_x || !d_gate || n == 0) return false;
    const uint32_t block = 256u;
    const uint32_t grid = (uint32_t)((n + block - 1u) / block);
    cudaStream_t s = (cudaStream_t)stream;
    k_sigmoid_gate<<<grid, block, 0, s>>>(d_x, d_gate, n);
    return cudaGetLastError() == cudaSuccess;
}

bool oc_cuda_qwen35_delta_step(float *d_conv_state, float *d_recurrent,
                               const float *d_qkv, const float *d_gate,
                               const float *d_beta, const float *d_alpha,
                               const float *d_conv_w, const float *d_ssm_a,
                               const float *d_dt_bias, const float *d_norm_w,
                               float *d_conv_out, float *d_out,
                               uint32_t n_key_heads, uint32_t n_value_heads,
                               uint32_t key_head_dim, uint32_t value_head_dim,
                               uint32_t conv_kernel, float eps, void *stream)
{
    if (!d_conv_state || !d_recurrent || !d_qkv || !d_gate || !d_beta ||
        !d_alpha || !d_conv_w || !d_ssm_a || !d_dt_bias || !d_norm_w ||
        !d_conv_out || !d_out)
        return false;
    if (n_key_heads == 0u || n_value_heads == 0u || key_head_dim == 0u ||
        value_head_dim == 0u || conv_kernel < 2u ||
        n_value_heads % n_key_heads != 0u)
        return false;

    const uint32_t conv_dim = 2u * n_key_heads * key_head_dim +
                              n_value_heads * value_head_dim;
    cudaStream_t s = (cudaStream_t)stream;

    {
        const uint32_t block = 256u;
        const uint32_t grid = (conv_dim + block - 1u) / block;
        k_qwen35_conv1d<<<grid, block, 0, s>>>(
            d_conv_state, d_conv_w, d_qkv, d_conv_out, conv_dim, conv_kernel);
        Q35_CHECK(cudaGetLastError());
    }

    {
        const uint32_t block = q35_block(key_head_dim, 256u);
        k_qwen35_norm_keys<<<n_key_heads, block, 2u * block * sizeof(float), s>>>(
            d_conv_out, n_key_heads, key_head_dim, eps);
        Q35_CHECK(cudaGetLastError());
    }

    {
        const uint32_t block = q35_block(value_head_dim, 256u);
        const size_t smem = 2u * (size_t)key_head_dim * sizeof(float);
        k_qwen35_delta_heads<<<n_value_heads, block, smem, s>>>(
            d_recurrent, d_conv_out, d_beta, d_alpha, d_ssm_a, d_dt_bias,
            d_out, n_key_heads, n_value_heads, key_head_dim, value_head_dim);
        Q35_CHECK(cudaGetLastError());
    }

    {
        const uint32_t block = q35_block(value_head_dim, 256u);
        k_qwen35_gate_heads<<<n_value_heads, block, block * sizeof(float), s>>>(
            d_out, d_gate, d_norm_w, n_value_heads, value_head_dim, eps);
        Q35_CHECK(cudaGetLastError());
    }
    return true;
}

#ifdef __cplusplus
}
#endif
