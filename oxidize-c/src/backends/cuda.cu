/*
 * cuda.cu — CUDA kernels for GPU-accelerated LLM inference.
 *
 * Weight residency: tensors whose quant type has a device kernel stay packed
 * in device memory and go through cuda_mmq.h; everything else is dequantized
 * on the host and uploaded as f32. For a Q4_K_M model that keeps device VRAM
 * at roughly the on-disk size instead of ~8x it.
 *
 * Kernels:
 *   - matvec_f32: matrix-vector multiply for f32-resident tensors
 *   - rms_norm: RMSNorm with optional Gemma scaling
 *   - apply_rope: rotary positional embedding (split-halves)
 *   - swiglu: SwiGLU / GeGLU activation
 *   - attention_head: per-head attention with online softmax
 *   - embed_lookup: token embedding lookup
 *
 * Build: make cuda (uses nvcc, defines OC_CUDA)
 *
 * The CUDA forward path replaces the CPU forward loop in oc_llama_forward
 * when OC_CUDA is defined and a GPU is available.
 */
#include "oxidize/cuda.h"
#include "oxidize/activation.h"
#include "oxidize/cuda_kernels.h"
#include "oxidize/cuda_mmq.h"
#include "oxidize/llama.h"
#include "oxidize/log.h"
#include "oxidize/quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── CUDA error checking ──────────────────────────────────────────────── */

/* Threads per attention block. Power of two (the kernel tree-reduces) and a
 * good match for the head_dim = 128 that Llama-family models use. */
#define OC_CUDA_ATTN_THREADS 128u

#define OC_CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            return OC_ERR_BACKEND; \
        } \
    } while (0)

__device__ __forceinline__ float oc_warp_sum(float v)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float oc_warp_max(float v)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ float oc_block_sum(float v, float *red)
{
    v = oc_warp_sum(v);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    if (lane == 0u) red[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0u) {
        float total = 0.0f;
        const uint32_t nwarps = blockDim.x >> 5;
        for (uint32_t w = 0u; w < nwarps; w++) total += red[w];
        red[0] = total;
    }
    __syncthreads();
    return red[0];
}

__device__ __forceinline__ float oc_block_max(float v, float *red)
{
    v = oc_warp_max(v);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    if (lane == 0u) red[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0u) {
        float m = red[0];
        const uint32_t nwarps = blockDim.x >> 5;
        for (uint32_t w = 1u; w < nwarps; w++) m = fmaxf(m, red[w]);
        red[0] = m;
    }
    __syncthreads();
    return red[0];
}

/* ─── CUDA kernels ──────────────────────────────────────────────────────── */

/* Embedding lookup: copy row `token` from [vocab, n_embd] embedding matrix. */
__global__ void k_embed_lookup(const float *embeddings, uint32_t token,
                                size_t n_embd, float *out)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = embeddings[(size_t)token * n_embd + i];
    }
}

/* RMSNorm: out[i] = x[i] * inv_rms * weight[i] * norm_scale. */
__global__ void k_rms_norm(const float *x, const float *weight, float *out,
                           size_t n, float eps, float norm_scale)
{
    extern __shared__ float red[];
    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    float acc = 0.0f;
    const bool vec4 = (n & 3u) == 0u;
    if (vec4) {
        const float4 *x4 = reinterpret_cast<const float4 *>(x);
        const size_t n4 = n / 4u;
        for (size_t i = tid; i < n4; i += nthreads) {
            const float4 v = x4[i];
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (size_t i = tid; i < n; i += nthreads) acc += x[i] * x[i];
    }
    const float inv = rsqrtf(oc_block_sum(acc, red) / (float)n + eps) * norm_scale;
    if (vec4) {
        const float4 *x4 = reinterpret_cast<const float4 *>(x);
        const float4 *w4 = reinterpret_cast<const float4 *>(weight);
        float4 *o4 = reinterpret_cast<float4 *>(out);
        const size_t n4 = n / 4u;
        for (size_t i = tid; i < n4; i += nthreads) {
            const float4 v = x4[i];
            const float4 w = w4[i];
            float4 o;
            o.x = v.x * inv * w.x;
            o.y = v.y * inv * w.y;
            o.z = v.z * inv * w.z;
            o.w = v.w * inv * w.w;
            o4[i] = o;
        }
    } else {
        for (size_t i = tid; i < n; i += nthreads)
            out[i] = x[i] * inv * weight[i];
    }
}

/* Gemma per-head RMSNorm over Q or K, one block per head.
 *
 * Applied after projection and before RoPE. `weight` is head_dim long and is
 * shared by every head of the layer. Mirrors the CPU path in llama.c, which
 * calls oc_rms_norm_f32 per head with the layer's own head_dim. */
__global__ void k_rms_norm_heads(float *x, const float *weight,
                                 uint32_t head_dim, float eps)
{
    extern __shared__ float red[];
    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    float *h = x + (size_t)blockIdx.x * head_dim;

    float sum = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += nthreads) sum += h[i] * h[i];
    const float inv = rsqrtf(oc_block_sum(sum, red) / (float)head_dim + eps);
    /* weight == NULL is a plain (weightless) normalization — Gemma 4
     * normalizes V that way, with no learned scale. */
    if (weight == NULL) {
        for (uint32_t i = tid; i < head_dim; i += nthreads) h[i] *= inv;
    } else {
        for (uint32_t i = tid; i < head_dim; i += nthreads)
            h[i] = h[i] * inv * weight[i];
    }
}

/* Final logit softcap: l = tanh(l/c) * c. */
__global__ void k_softcap(float *x, float cap, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] / cap) * cap;
}

/* Elementwise bias add: out[i] += bias[i]. Used for the Qwen2-family QKV
 * projection biases. */
__global__ void k_add_bias(float *out, const float *bias, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += bias[i];
}

/* Matvec: out[row] = sum(W[row, col] * x[col]). One warp per row, 8 rows per
 * block. Ampere issues float4 loads when cols is a multiple of 4 (Qwen3.5-27B
 * n_embd=5120). Warp shuffles replace the old shared-memory tree. */
#define OC_CUDA_MV_WARPS 8u

__global__ void k_matvec_f32(const float *W, size_t rows, size_t cols,
                              const float *x, float *out)
{
    const uint32_t lane = threadIdx.x & 31u;
    const size_t row = (size_t)blockIdx.x * OC_CUDA_MV_WARPS + (threadIdx.x >> 5);
    if (row >= rows) return;

    const float *wrow = W + row * cols;
    float sum = 0.0f;
    if ((cols & 3u) == 0u) {
        const float4 *w4 = reinterpret_cast<const float4 *>(wrow);
        const float4 *x4 = reinterpret_cast<const float4 *>(x);
        const size_t n4 = cols / 4u;
        for (size_t c = lane; c < n4; c += 32u) {
            const float4 a = w4[c];
            const float4 b = x4[c];
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (size_t c = lane; c < cols; c += 32u)
            sum += wrow[c] * x[c];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0u) out[row] = sum;
}

__device__ __forceinline__ float oc_bf16_to_f32(uint16_t h)
{
    return __uint_as_float((uint32_t)h << 16);
}

__global__ void k_matvec_bf16(const uint16_t *W, size_t rows, size_t cols,
                              const float *x, float *out)
{
    const uint32_t lane = threadIdx.x & 31u;
    const size_t row = (size_t)blockIdx.x * OC_CUDA_MV_WARPS + (threadIdx.x >> 5);
    if (row >= rows) return;

    const uint16_t *wrow = W + row * cols;
    float sum = 0.0f;
    for (size_t c = lane; c < cols; c += 32u)
        sum += oc_bf16_to_f32(wrow[c]) * x[c];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0u) out[row] = sum;
}

/* RoPE (split-halves NeoX-style), with YaRN when yarn_factor > 1. One block
 * per head. Frequency ladder uses rope_dim, matching oc_apply_rope_f32. */
__global__ void k_apply_rope(float *q, size_t head_dim, size_t rope_dim,
                              int64_t pos, float theta, uint32_t n_heads,
                              float yarn_factor, uint32_t yarn_orig_ctx)
{
    size_t head = blockIdx.x;
    if (head >= n_heads) return;

    float *hq = q + head * head_dim;
    size_t half = rope_dim / 2;
    if (rope_dim < 2) return;
    float freq_mul = powf(theta, -2.0f / (float)rope_dim);
    const bool yarn = yarn_factor > 1.0f && yarn_orig_ctx > 0u;
    const float mscale = yarn ? (1.0f + 0.1f * logf(yarn_factor)) : 1.0f;
    float corr_lo = 0.0f, corr_hi = 0.0f, denom = 0.001f;
    if (yarn) {
        float base_log = 2.0f * logf(theta);
        float ratio_fast = (float)yarn_orig_ctx / (32.0f * 2.0f * 3.14159265f);
        float ratio_slow = (float)yarn_orig_ctx / (1.0f * 2.0f * 3.14159265f);
        corr_lo = floorf((float)rope_dim * logf(ratio_fast) / base_log);
        corr_hi = ceilf((float)rope_dim * logf(ratio_slow) / base_log);
        if (corr_lo < 0.0f) corr_lo = 0.0f;
        if (corr_hi > (float)rope_dim - 1.0f) corr_hi = (float)rope_dim - 1.0f;
        denom = corr_hi - corr_lo;
        if (denom < 0.001f) denom = 0.001f;
    }

    for (size_t i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = powf(freq_mul, (float)i);
        float angle = (float)pos * freq;
        if (yarn) {
            if (pos == 0) {
                hq[i] *= mscale;
                hq[half + i] *= mscale;
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
        float x0 = hq[i];
        float x1 = hq[half + i];
        hq[i] = x0 * c - x1 * s;
        hq[half + i] = x0 * s + x1 * c;
    }
}

/* SwiGLU in-place: gate[i] = silu(gate[i]) * up[i]. */
__global__ void k_swiglu(float *gate, const float *up, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if ((n & 3u) == 0u) {
        i *= 4u;
        if (i >= n) return;
        float4 g = *reinterpret_cast<float4 *>(gate + i);
        const float4 u = *reinterpret_cast<const float4 *>(up + i);
        float s;
        s = 1.0f / (1.0f + expf(-g.x)); g.x = g.x * s * u.x;
        s = 1.0f / (1.0f + expf(-g.y)); g.y = g.y * s * u.y;
        s = 1.0f / (1.0f + expf(-g.z)); g.z = g.z * s * u.z;
        s = 1.0f / (1.0f + expf(-g.w)); g.w = g.w * s * u.w;
        *reinterpret_cast<float4 *>(gate + i) = g;
        return;
    }
    if (i < n) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        gate[i] = g * sig * up[i];
    }
}

/* GeGLU in-place: gate[i] = gelu(gate[i]) * up[i]. */
__global__ void k_geglu(float *gate, const float *up, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float gelu = 0.5f * g * (1.0f + erff(g / 1.41421356f));
        gate[i] = gelu * up[i];
    }
}

/* Attention for every head of a layer, one block per head.
 *
 * Replaces an earlier formulation that launched one <<<1, block>>> kernel per
 * head — 336 single-block launches per token for a 28-layer/12-head model, each
 * occupying a single SM — and in which every thread redundantly recomputed the
 * full q·k scan over all past positions, twice, with a barrier inside the loop.
 * Decode cost therefore grew steeply with context. Here the work is properly
 * parallel: scores across positions, the softmax reductions across threads, and
 * the V accumulation across head_dim.
 *
 * `scores` is per-head scratch with stride `score_stride` (>= n_ctx).
 * Shared memory: (head_dim + blockDim.x) floats.
 * blockDim.x must be a power of two (tree reductions). */
__global__ void k_attention_all_heads(
    const float *q,           /* [n_head, head_dim]                      */
    const __half *kv_k,       /* [n_ctx, n_head_kv, head_dim]            */
    const __half *kv_v,       /* [n_ctx, n_head_kv, head_dim]            */
    float *out,               /* [n_head, head_dim]                      */
    float *scores,            /* [n_head, score_stride] scratch          */
    size_t score_stride,
    uint32_t n_head,
    uint32_t n_head_kv,
    uint32_t head_dim,
    uint32_t n_past,
    /* Pre-attention score scale. Passed in rather than computed as
     * rsqrtf(head_dim): Gemma 4 uses 1.0. */
    float attn_scale,
    /* Element stride between cached positions. NOT n_head_kv*head_dim: the
     * cache is indexed at one uniform stride across layers, which on Gemma 4
     * is the max over its two geometries. */
    size_t kv_row,
    /* First position to attend to. Non-zero on a sliding-window layer, where
     * only the last `sliding_window` positions are in scope. */
    uint32_t kv_start)
{
    const uint32_t h = blockIdx.x;
    if (h >= n_head) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const uint32_t kv_head = h / (n_head / n_head_kv);
    const size_t kv_off = (size_t)kv_head * head_dim;

    extern __shared__ float smem[];
    float *qs  = smem;                 /* cached query, head_dim floats  */
    float *red = smem + head_dim;      /* reduction buffer, nthreads     */

    float *sc = scores + (size_t)h * score_stride;

    /* Cache this head's query vector. */
    for (uint32_t d = tid; d < head_dim; d += nthreads)
        qs[d] = q[(size_t)h * head_dim + d];
    __syncthreads();

    /* 1. Scores, parallel over past positions. Ampere: float4 Q × half2 K. */
    const float inv_sqrt_d = attn_scale;
    const bool pair = ((head_dim & 1u) == 0u) && ((kv_off & 1u) == 0u);
    const bool vec4 = pair && ((head_dim & 3u) == 0u);
    for (uint32_t p = kv_start + tid; p < n_past; p += nthreads) {
        const __half *k = kv_k + (size_t)p * kv_row + kv_off;
        float s = 0.0f;
        if (vec4) {
            const float4 *q4 = reinterpret_cast<const float4 *>(qs);
            const __half2 *k2 = reinterpret_cast<const __half2 *>(k);
            const uint32_t n4 = head_dim / 4u;
            for (uint32_t d = 0; d < n4; d++) {
                const float4 qv = q4[d];
                const float2 a = __half22float2(__ldg(&k2[2u * d]));
                const float2 b = __half22float2(__ldg(&k2[2u * d + 1u]));
                s += qv.x * a.x + qv.y * a.y + qv.z * b.x + qv.w * b.y;
            }
        } else if (pair) {
            const __half2 *k2 = reinterpret_cast<const __half2 *>(k);
            for (uint32_t d = 0; d < head_dim / 2u; d++) {
                const float2 kv = __half22float2(__ldg(&k2[d]));
                s += qs[2u * d] * kv.x + qs[2u * d + 1u] * kv.y;
            }
        } else {
            for (uint32_t d = 0; d < head_dim; d++)
                s += qs[d] * __half2float(k[d]);
        }
        sc[p] = s * inv_sqrt_d;
    }
    __syncthreads();

    /* 2. Max, warp-reduced across the block. */
    float m = -INFINITY;
    for (uint32_t p = kv_start + tid; p < n_past; p += nthreads) m = fmaxf(m, sc[p]);
    const float max_score = oc_block_max(m, red);
    __syncthreads();

    /* 3. exp in place, sum warp-reduced. */
    float local_sum = 0.0f;
    for (uint32_t p = kv_start + tid; p < n_past; p += nthreads) {
        const float e = __expf(sc[p] - max_score);
        sc[p] = e;
        local_sum += e;
    }
    const float denom = oc_block_sum(local_sum, red);
    const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

    /* 4. Weighted V sum. Pair consecutive dims so Ampere issues half2 loads. */
    if (pair) {
        for (uint32_t d2 = tid; d2 < head_dim / 2u; d2 += nthreads) {
            float acc0 = 0.0f, acc1 = 0.0f;
            for (uint32_t p = kv_start; p < n_past; p++) {
                const __half2 v2 = __ldg(reinterpret_cast<const __half2 *>(
                    kv_v + (size_t)p * kv_row + kv_off + (size_t)d2 * 2u));
                const float2 fv = __half22float2(v2);
                acc0 += sc[p] * fv.x;
                acc1 += sc[p] * fv.y;
            }
            out[(size_t)h * head_dim + (size_t)d2 * 2u]     = acc0 * inv_denom;
            out[(size_t)h * head_dim + (size_t)d2 * 2u + 1u] = acc1 * inv_denom;
        }
    } else {
        for (uint32_t d = tid; d < head_dim; d += nthreads) {
            float acc = 0.0f;
            for (uint32_t p = kv_start; p < n_past; p++)
                acc += sc[p] * __half2float(kv_v[(size_t)p * kv_row + kv_off + d]);
            out[(size_t)h * head_dim + d] = acc * inv_denom;
        }
    }
}

/* Copy KV to cache: write K/V at position `pos`, narrowing to f16. */
__global__ void k_kv_cache_write(__half *kv_k, __half *kv_v,
                                  const float *k, const float *v,
                                  size_t pos, size_t n_head_kv,
                                  uint32_t head_dim, size_t n_ctx,
                                  /* Uniform element stride between positions;
                                   * see k_attention_all_heads. Only the first
                                   * n_head_kv*head_dim of each row is used —
                                   * a global Gemma 4 layer leaves the rest
                                   * untouched, and never reads it. */
                                  size_t kv_row)
{
    (void)n_ctx;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)n_head_kv * head_dim;
    if (i < total) {
        size_t offset = pos * kv_row + i;
        kv_k[offset] = __float2half(k[i]);
        kv_v[offset] = __float2half(v[i]);
    }
}

/* MoE routing: gate the router logits, pick top-k, renormalize.
 *
 * A deliberate single-thread kernel. It is ~128 exp() calls plus a k-pass
 * selection sort over 128 candidates — microseconds — and running it serially
 * reproduces llama.c::forward_moe_ffn exactly, including its `double`
 * accumulators and its strict-greater tie-breaking (first index wins). Getting
 * that bit-for-bit matters more than speed here: a different tie-break picks a
 * different expert, which is a far larger output deviation than a rounding
 * difference. Results stay on the device so routing never syncs to the host.
 */
__global__ void k_moe_route(const float *logits_in, float *logits,
                            uint32_t *sel, float *weights,
                            uint32_t n_exp, uint32_t k,
                            bool sigmoid_gating, float routed_scale)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if (!sigmoid_gating) {
        float mx = logits_in[0];
        for (uint32_t i = 1; i < n_exp; i++)
            if (logits_in[i] > mx) mx = logits_in[i];
        double sum = 0.0;
        for (uint32_t i = 0; i < n_exp; i++) {
            logits[i] = expf(logits_in[i] - mx);
            sum += (double)logits[i];
        }
        if (sum > 0.0) {
            const float inv = (float)(1.0 / sum);
            for (uint32_t i = 0; i < n_exp; i++) logits[i] *= inv;
        }
    } else {
        for (uint32_t i = 0; i < n_exp; i++)
            logits[i] = 1.0f / (1.0f + expf(-logits_in[i]));
    }

    /* Partial selection sort over an identity permutation, as on the CPU. */
    for (uint32_t i = 0; i < n_exp; i++) sel[i] = i;
    for (uint32_t i = 0; i < k; i++) {
        uint32_t best = i;
        for (uint32_t j = i + 1; j < n_exp; j++)
            if (logits[sel[j]] > logits[sel[best]]) best = j;
        const uint32_t t = sel[i]; sel[i] = sel[best]; sel[best] = t;
    }

    double weight_norm = 0.0;
    for (uint32_t i = 0; i < k; i++) weight_norm += (double)logits[sel[i]];
    if (weight_norm <= 0.0) weight_norm = 1.0;
    for (uint32_t i = 0; i < k; i++)
        weights[i] = routed_scale * (float)((double)logits[sel[i]] / weight_norm);
}

/* out[i] += weights[slot] * src[i] — accumulate one expert's contribution. */
__global__ void k_moe_accum(float *out, const float *src,
                            const float *weights, uint32_t slot, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += weights[slot] * src[i];
}

/* Scale by sigmoid(gate_logit) — the Qwen2-MoE shared_expert_gate. */
__global__ void k_sigmoid_scale(float *x, const float *gate_logit, size_t n)
{
    const float s = 1.0f / (1.0f + expf(-gate_logit[0]));
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

/* Residual add: x[i] += y[i]. */
__global__ void k_residual_add(float *x, const float *y, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

/* Fused decode: x += residual, then RMSNorm(x) -> out. Safe when residual
 * aliases out: the add is consumed into registers before out is written. */
__global__ void k_residual_rms(float *x, const float *add, const float *weight,
                               float *out, size_t n, float eps, float norm_scale)
{
    extern __shared__ float red[];
    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    float acc = 0.0f;
    const bool vec4 = (n & 3u) == 0u;
    if (vec4) {
        float4 *x4 = reinterpret_cast<float4 *>(x);
        const float4 *a4 = reinterpret_cast<const float4 *>(add);
        const size_t n4 = n / 4u;
        for (size_t i = tid; i < n4; i += nthreads) {
            const float4 xv = x4[i];
            const float4 av = a4[i];
            float4 s;
            s.x = xv.x + av.x;
            s.y = xv.y + av.y;
            s.z = xv.z + av.z;
            s.w = xv.w + av.w;
            x4[i] = s;
            acc += s.x * s.x + s.y * s.y + s.z * s.z + s.w * s.w;
        }
    } else {
        for (size_t i = tid; i < n; i += nthreads) {
            float v = x[i] + add[i];
            x[i] = v;
            acc += v * v;
        }
    }
    const float inv = rsqrtf(oc_block_sum(acc, red) / (float)n + eps) * norm_scale;
    if (vec4) {
        const float4 *x4 = reinterpret_cast<const float4 *>(x);
        const float4 *w4 = reinterpret_cast<const float4 *>(weight);
        float4 *o4 = reinterpret_cast<float4 *>(out);
        const size_t n4 = n / 4u;
        for (size_t i = tid; i < n4; i += nthreads) {
            const float4 v = x4[i];
            const float4 w = w4[i];
            float4 o;
            o.x = v.x * inv * w.x;
            o.y = v.y * inv * w.y;
            o.z = v.z * inv * w.z;
            o.w = v.w * inv * w.w;
            o4[i] = o;
        }
    } else {
        for (size_t i = tid; i < n; i += nthreads)
            out[i] = x[i] * inv * weight[i];
    }
}

/* Scale: x[i] *= scale. */
__global__ void k_scale(float *x, float scale, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

/* ─── Host-side helper: upload a weight tensor ───────────────────────────── */

/* Upload `view` to the device.
 *
 * Preferred path: the quant type has a cuda_mmq.h kernel and the row length is
 * a whole number of blocks, so the packed bytes are copied verbatim and the
 * matvec dequantizes in registers. This is what keeps a Q4_K_M model near its
 * on-disk size in VRAM.
 *
 * Fallback: dequantize row by row on the host and upload f32. Correct for
 * every type, but costs 8x the memory of Q4_K — used only where no device
 * kernel applies (e.g. n_embd = 896 is not a multiple of the 256-element
 * K-quant super-block). */
static OcError upload_weight_rows(const OcWeightView *view, OcCudaWeight *out,
                                  float *host_temp, OcCudaContext *ctx,
                                  size_t rows_override)
{
    /* Stacked MoE tensors are 3-D: view_from_info() reports rows = dims[1]
     * (one expert's rows) and drops dims[2], the expert count. Callers pass
     * num_experts * view->rows so the whole stack is uploaded. */
    const size_t rows = rows_override ? rows_override : view->rows;
    const size_t cols = view->cols;

    memset(out, 0, sizeof(*out));
    out->qtype = (uint32_t)view->qtype;
    out->rows = (uint32_t)rows;
    out->cols = (uint32_t)cols;

    if ((view->qtype == OC_QUANT_BF16 || view->qtype == OC_QUANT_F16) &&
        view->row_bytes == cols * 2u) {
        const size_t total = rows * cols * 2u;
        OC_CUDA_CHECK(cudaMalloc(&out->data, total));
        OC_CUDA_CHECK(cudaMemcpy(out->data, view->data, total,
                                 cudaMemcpyHostToDevice));
        out->row_bytes = cols * 2u;
        out->packed = true;
        if (ctx) { ctx->vram_weight_bytes += total; ctx->n_packed_tensors++; }
        return OC_OK;
    }

    size_t src_block = 0, dev_block = 0, n_blocks = 0;
    if (view->qtype != OC_QUANT_F32 &&
        oc_cuda_mmq_block_layout((uint32_t)view->qtype, cols,
                                 &src_block, &dev_block, &n_blocks) &&
        /* The on-disk stride must be exactly blocks-back-to-back; if the
         * loader laid rows out differently, fall through to f32. */
        n_blocks * src_block == view->row_bytes) {
        const size_t dev_row = n_blocks * dev_block;
        const size_t total = rows * dev_row;
        OC_CUDA_CHECK(cudaMalloc(&out->data, total));
        if (dev_block == src_block) {
            OC_CUDA_CHECK(cudaMemcpy(out->data, view->data, total,
                                     cudaMemcpyHostToDevice));
        } else {
            /* Padded device layout (Q6_K: 210 -> 224). Blocks are contiguous
             * on the host across the whole tensor, so one strided copy places
             * every block at its aligned device offset; the pad bytes are
             * never read. */
            OC_CUDA_CHECK(cudaMemcpy2D(out->data, dev_block,
                                       view->data, src_block,
                                       src_block, rows * n_blocks,
                                       cudaMemcpyHostToDevice));
        }
        out->row_bytes = dev_row;
        out->packed = true;
        if (ctx) { ctx->vram_weight_bytes += total; ctx->n_packed_tensors++; }
        return OC_OK;
    }

    /* f32 fallback. */
    const size_t total = rows * cols * sizeof(float);
    OC_CUDA_CHECK(cudaMalloc(&out->data, total));
    float *d_f32 = (float *)out->data;
    for (size_t r = 0; r < rows; r++) {
        if (view->qtype == OC_QUANT_F32) {
            const float *src = (const float *)view->data + r * cols;
            OC_CUDA_CHECK(cudaMemcpyAsync(d_f32 + r * cols, src,
                cols * sizeof(float), cudaMemcpyHostToDevice, 0));
        } else {
            const uint8_t *src = view->data + r * view->row_bytes;
            OcError e = oc_quant_dequant_row(view->qtype, src, view->row_bytes,
                                             host_temp, cols);
            if (e != OC_OK) return e;
            OC_CUDA_CHECK(cudaMemcpyAsync(d_f32 + r * cols, host_temp,
                cols * sizeof(float), cudaMemcpyHostToDevice, 0));
        }
    }
    OC_CUDA_CHECK(cudaDeviceSynchronize());
    out->row_bytes = cols * sizeof(float);
    out->qtype = OC_QUANT_F32;
    out->packed = false;
    if (ctx) { ctx->vram_weight_bytes += total; ctx->n_f32_tensors++; }
    return OC_OK;
}

static OcError upload_weight(const OcWeightView *view, OcCudaWeight *out,
                             float *host_temp, OcCudaContext *ctx)
{
    return upload_weight_rows(view, out, host_temp, ctx, 0);
}

/* ─── Matvec dispatch ────────────────────────────────────────────────────── */

/* out[0..rows) = W . x, picking the packed-quant kernel or the f32 kernel
 * based on how the tensor was uploaded. Asynchronous on `stream` (NULL = default). */
static OcError cuda_matvec_on(const OcCudaWeight *w, const float *d_x,
                              float *d_out, uint32_t block, void *stream)
{
    const size_t rows = w->rows;
    const size_t cols = w->cols;
    (void)block;
    cudaStream_t s = (cudaStream_t)stream;

    if (w->packed) {
        if (w->qtype == OC_QUANT_BF16) {
            const uint32_t grid =
                (uint32_t)((rows + OC_CUDA_MV_WARPS - 1u) / OC_CUDA_MV_WARPS);
            k_matvec_bf16<<<grid, OC_CUDA_MV_WARPS * 32u, 0, s>>>(
                (const uint16_t *)w->data, rows, cols, d_x, d_out);
            OC_CUDA_CHECK(cudaGetLastError());
            return OC_OK;
        }
        if (!oc_cuda_mmq_matvec(w->qtype, w->data, d_x, d_out, rows, cols, stream))
            return OC_ERR_BACKEND;
        return OC_OK;
    }
    {
        const uint32_t grid =
            (uint32_t)((rows + OC_CUDA_MV_WARPS - 1u) / OC_CUDA_MV_WARPS);
        k_matvec_f32<<<grid, OC_CUDA_MV_WARPS * 32u, 0, s>>>(
            (const float *)w->data, rows, cols, d_x, d_out);
    }
    OC_CUDA_CHECK(cudaGetLastError());
    return OC_OK;
}

static OcError cuda_matvec(const OcCudaWeight *w, const float *d_x,
                           float *d_out, uint32_t block)
{
    return cuda_matvec_on(w, d_x, d_out, block, NULL);
}

static OcError cuda_wait_compute(OcCudaContext *ctx)
{
    for (int i = 0; i < 3; i++) {
        if (!ctx->compute_streams[i]) continue;
        OC_CUDA_CHECK(cudaStreamSynchronize(
            (cudaStream_t)ctx->compute_streams[i]));
    }
    return OC_OK;
}

/* Expert-indexed matvec: same dispatch as cuda_matvec, but the stacked expert
 * tensor is offset on the device by the routed expert id in slot `slot`.
 * `rows` is the per-expert row count, not the stacked total. */
static OcError cuda_matvec_expert(const OcCudaWeight *w, const float *d_x,
                                  float *d_out, size_t rows,
                                  const uint32_t *d_sel, uint32_t slot,
                                  uint32_t block)
{
    if (!w->packed) {
        /* f32 fallback: slice the stacked buffer directly. Only reachable for
         * shapes/types without a device kernel, which for MoE means a very
         * large f32 upload — correct, but the packed path is the real one. */
        (void)block;
        return OC_ERR_BACKEND;
    }
    if (!oc_cuda_mmq_matvec_expert(w->qtype, w->data, d_x, d_out, rows,
                                   w->cols, d_sel, slot, NULL))
        return OC_ERR_BACKEND;
    return OC_OK;
}

/* MoE FFN for one layer. Mirrors llama.c::forward_moe_ffn step for step:
 * router matvec, gating + top-k + renormalize, per-expert SwiGLU accumulated
 * with the routed weight, then the optional shared expert. Everything is
 * enqueued asynchronously — the routing decision is consumed on the device. */
static OcError cuda_moe_ffn(OcCudaContext *ctx, uint32_t l, uint32_t block)
{
    const uint32_t embd = ctx->n_embd;
    const uint32_t n_exp = ctx->num_experts;
    const uint32_t k = ctx->num_experts_per_tok;
    const uint32_t i_size = ctx->expert_intermediate_size;
    const uint32_t egrid = (i_size + block - 1) / block;
    const uint32_t xgrid = (embd + block - 1) / block;

    /* 1. Router logits, then gating + top-k on device. */
    OcError e = cuda_matvec(&ctx->d_ffn_gate_inp[l], ctx->d_normed,
                            ctx->d_router_logits, block);
    if (e != OC_OK) return e;
    k_moe_route<<<1, 1>>>(ctx->d_router_logits, ctx->d_router_logits,
                          ctx->d_expert_sel, ctx->d_expert_w,
                          n_exp, k, ctx->expert_gating_sigmoid,
                          ctx->expert_weights_scale);
    OC_CUDA_CHECK(cudaGetLastError());

    /* 2. Per-expert SwiGLU FFN, weighted into the accumulator. */
    OC_CUDA_CHECK(cudaMemsetAsync(ctx->d_expert_out, 0,
                                  (size_t)embd * sizeof(float)));
    for (uint32_t slot = 0; slot < k; slot++) {
        e = cuda_matvec_expert(&ctx->d_ffn_gate_exps[l], ctx->d_normed,
                               ctx->d_expert_gate, i_size,
                               ctx->d_expert_sel, slot, block);
        if (e != OC_OK) return e;
        e = cuda_matvec_expert(&ctx->d_ffn_up_exps[l], ctx->d_normed,
                               ctx->d_expert_up, i_size,
                               ctx->d_expert_sel, slot, block);
        if (e != OC_OK) return e;
        k_swiglu<<<egrid, block>>>(ctx->d_expert_gate, ctx->d_expert_up, i_size);
        OC_CUDA_CHECK(cudaGetLastError());
        e = cuda_matvec_expert(&ctx->d_ffn_down_exps[l], ctx->d_expert_gate,
                               ctx->d_expert_tmp, embd,
                               ctx->d_expert_sel, slot, block);
        if (e != OC_OK) return e;
        k_moe_accum<<<xgrid, block>>>(ctx->d_expert_out, ctx->d_expert_tmp,
                                      ctx->d_expert_w, slot, embd);
        OC_CUDA_CHECK(cudaGetLastError());
    }

    /* 3. Shared expert, always active with weight 1.0. */
    if (ctx->d_ffn_gate_shexp[l].data && ctx->d_ffn_up_shexp[l].data &&
        ctx->d_ffn_down_shexp[l].data) {
        const uint32_t shexp = ctx->shared_expert_intermediate_size
                             ? ctx->shared_expert_intermediate_size : i_size;
        const uint32_t sgrid = (shexp + block - 1) / block;
        e = cuda_matvec(&ctx->d_ffn_gate_shexp[l], ctx->d_normed,
                        ctx->d_expert_gate, block);
        if (e != OC_OK) return e;
        e = cuda_matvec(&ctx->d_ffn_up_shexp[l], ctx->d_normed,
                        ctx->d_expert_up, block);
        if (e != OC_OK) return e;
        k_swiglu<<<sgrid, block>>>(ctx->d_expert_gate, ctx->d_expert_up, shexp);
        OC_CUDA_CHECK(cudaGetLastError());
        e = cuda_matvec(&ctx->d_ffn_down_shexp[l], ctx->d_expert_gate,
                        ctx->d_expert_tmp, block);
        if (e != OC_OK) return e;
        if (ctx->d_ffn_gate_inp_shexp[l].data) {
            e = cuda_matvec(&ctx->d_ffn_gate_inp_shexp[l], ctx->d_normed,
                            ctx->d_shexp_gate_logit, block);
            if (e != OC_OK) return e;
            k_sigmoid_scale<<<xgrid, block>>>(ctx->d_expert_tmp,
                                              ctx->d_shexp_gate_logit, embd);
            OC_CUDA_CHECK(cudaGetLastError());
        }
        k_residual_add<<<xgrid, block>>>(ctx->d_expert_out,
                                         ctx->d_expert_tmp, embd);
        OC_CUDA_CHECK(cudaGetLastError());
    }

    /* 4. Residual. */
    k_residual_add<<<xgrid, block>>>(ctx->d_x, ctx->d_expert_out, embd);
    OC_CUDA_CHECK(cudaGetLastError());
    return OC_OK;
}

/* Embedding lookup that understands both residency modes. */
static OcError cuda_embed(const OcCudaWeight *w, uint32_t token,
                          float *d_out, uint32_t n_embd, uint32_t block)
{
    if (w->packed) {
        if (!oc_cuda_mmq_get_row(w->qtype, w->data, token, d_out, w->cols, NULL))
            return OC_ERR_BACKEND;
        return OC_OK;
    }
    k_embed_lookup<<<(n_embd + block - 1) / block, block>>>(
        (const float *)w->data, token, n_embd, d_out);
    OC_CUDA_CHECK(cudaGetLastError());
    return OC_OK;
}

static OcError cuda_qwen35_recurrent(OcCudaContext *ctx, uint32_t l,
                                     uint32_t block)
{
    OcError e = cuda_matvec_on(&ctx->d_attn_qkv[l], ctx->d_normed,
                               ctx->d_qwen35_qkv, block,
                               ctx->compute_streams[0]);
    if (e != OC_OK) return e;
    e = cuda_matvec_on(&ctx->d_attn_gate[l], ctx->d_normed,
                       ctx->d_qwen35_gate, block, ctx->compute_streams[1]);
    if (e != OC_OK) return e;
    e = cuda_matvec_on(&ctx->d_ssm_beta[l], ctx->d_normed,
                       ctx->d_qwen35_beta, block, ctx->compute_streams[2]);
    if (e != OC_OK) return e;
    e = cuda_matvec_on(&ctx->d_ssm_alpha[l], ctx->d_normed,
                       ctx->d_qwen35_alpha, block, ctx->compute_streams[0]);
    if (e != OC_OK) return e;
    e = cuda_wait_compute(ctx);
    if (e != OC_OK) return e;

    uint32_t rec_i = 0;
    for (uint32_t i = 0; i < l; i++) {
        if (ctx->l_kind && ctx->l_kind[i] == (uint8_t)OC_LLAMA_LAYER_QWEN35_RECURRENT)
            rec_i++;
    }
    float *conv_state = ctx->d_conv_state +
        (size_t)rec_i * ctx->conv_state_per_layer;
    float *recurrent = ctx->d_recurrent_state +
        (size_t)rec_i * ctx->recurrent_state_per_layer;
    const uint32_t dk = ctx->ssm_state_size;
    const uint32_t dv = ctx->ssm_value_heads
        ? ctx->ssm_inner_size / ctx->ssm_value_heads : 0;
    if (!oc_cuda_qwen35_delta_step(
            conv_state, recurrent,
            ctx->d_qwen35_qkv, ctx->d_qwen35_gate,
            ctx->d_qwen35_beta, ctx->d_qwen35_alpha,
            ctx->d_ssm_conv1d[l], ctx->d_ssm_a[l],
            ctx->d_ssm_dt_bias[l], ctx->d_ssm_norm[l],
            ctx->d_qwen35_conv_out, ctx->d_qwen35_delta_out,
            ctx->ssm_group_count, ctx->ssm_value_heads,
            dk, dv, ctx->ssm_conv_kernel, ctx->rms_norm_eps, NULL))
        return OC_ERR_BACKEND;

    e = cuda_matvec(&ctx->d_ssm_out[l], ctx->d_qwen35_delta_out,
                    ctx->d_normed, block);
    if (e != OC_OK) return e;
    return OC_OK;
}

static OcError cuda_qwen35_full_attn(OcCudaContext *ctx, uint32_t l,
                                     uint32_t pos, uint32_t block,
                                     uint32_t hd_l, uint32_t nkv_l,
                                     uint32_t rd_l, float rth_l,
                                     uint32_t win_l)
{
    const uint32_t n_head = ctx->n_head;
    const float eps = ctx->rms_norm_eps;
    OcError e = cuda_matvec_on(&ctx->d_attn_q[l], ctx->d_normed,
                               ctx->d_qwen35_qkv, block,
                               ctx->compute_streams[0]);
    if (e != OC_OK) return e;
    e = cuda_matvec_on(&ctx->d_attn_k[l], ctx->d_normed, ctx->d_k, block,
                       ctx->compute_streams[1]);
    if (e != OC_OK) return e;
    e = cuda_matvec_on(&ctx->d_attn_v[l], ctx->d_normed, ctx->d_v, block,
                       ctx->compute_streams[2]);
    if (e != OC_OK) return e;
    e = cuda_wait_compute(ctx);
    if (e != OC_OK) return e;

    if (!oc_cuda_qwen35_unpack_qgate(ctx->d_qwen35_qkv, ctx->d_q,
                                     ctx->d_qwen35_gate, n_head, hd_l, NULL))
        return OC_ERR_BACKEND;

    if (!oc_cuda_qk_norm_rope(ctx->d_q, ctx->d_attn_q_norm[l],
                              n_head, hd_l, rd_l, (int64_t)pos, rth_l,
                              eps, 0.0f, 0u, NULL))
        return OC_ERR_BACKEND;
    if (!oc_cuda_qk_norm_rope(ctx->d_k, ctx->d_attn_k_norm[l],
                              nkv_l, hd_l, rd_l, (int64_t)pos, rth_l,
                              eps, 0.0f, 0u, NULL))
        return OC_ERR_BACKEND;

    const uint32_t kv_slot = ctx->l_kv_index ? ctx->l_kv_index[l] : l;
    size_t kv_live = (size_t)nkv_l * hd_l;
    __half *kv_k_layer = (__half *)ctx->d_kv_k
                       + (size_t)kv_slot * ctx->n_ctx * ctx->kv_row;
    __half *kv_v_layer = (__half *)ctx->d_kv_v
                       + (size_t)kv_slot * ctx->n_ctx * ctx->kv_row;
    k_kv_cache_write<<<(kv_live + block - 1) / block, block>>>(
        kv_k_layer, kv_v_layer,
        ctx->d_k, ctx->d_v, pos, nkv_l, hd_l, ctx->n_ctx, ctx->kv_row);
    OC_CUDA_CHECK(cudaGetLastError());

    {
        const uint32_t athreads = OC_CUDA_ATTN_THREADS;
        const size_t asmem = ((size_t)hd_l + athreads) * sizeof(float);
        const uint32_t n_past = pos + 1;
        const uint32_t kv_start =
            (win_l > 0 && n_past > win_l) ? (n_past - win_l) : 0u;
        k_attention_all_heads<<<n_head, athreads, asmem>>>(
            ctx->d_q, kv_k_layer, kv_v_layer, ctx->d_attn_out,
            ctx->d_attn_scores, ctx->n_ctx,
            n_head, nkv_l, hd_l, n_past,
            ctx->attn_scale > 0.0f ? ctx->attn_scale : rsqrtf((float)hd_l),
            ctx->kv_row, kv_start);
        OC_CUDA_CHECK(cudaGetLastError());
    }

    if (!oc_cuda_sigmoid_gate(ctx->d_attn_out, ctx->d_qwen35_gate,
                              (size_t)n_head * hd_l, NULL))
        return OC_ERR_BACKEND;

    e = cuda_matvec(&ctx->d_attn_output[l], ctx->d_attn_out,
                    ctx->d_normed, block);
    if (e != OC_OK) return e;
    return OC_OK;
}

/* ─── Public API ────────────────────────────────────────────────────────── */

bool oc_cuda_available(void)
{
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    return (e == cudaSuccess && n > 0);
}

OcError oc_cuda_init(OcCudaContext *ctx, const OcLlamaModel *model)
{
    if (!ctx || !model) return OC_ERR_INVALID_ARG;
    memset(ctx, 0, sizeof(*ctx));

    if (!oc_cuda_available()) return OC_ERR_BACKEND;

    for (int i = 0; i < 3; i++) {
        cudaStream_t s = NULL;
        if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess)
            ctx->compute_streams[i] = (void *)s;
    }

    const OcLlamaConfig *c = &model->cfg;
    ctx->n_embd = c->n_embd;
    ctx->n_head = c->n_head;
    ctx->n_head_kv = c->n_head_kv;
    ctx->n_ff = c->n_ff;
    ctx->head_dim = c->head_dim;
    ctx->n_layer = c->n_layer;
    ctx->vocab_size = c->vocab_size;
    ctx->n_ctx = c->n_ctx;
    ctx->rope_dim = c->rope_dim;
    ctx->rope_theta = c->rope_theta;
    ctx->rms_norm_eps = c->rms_norm_eps;
    ctx->norm_scale = c->norm_scale;
    ctx->yarn_factor = c->yarn_factor;
    ctx->yarn_orig_ctx = c->yarn_orig_ctx;
    ctx->uses_geglu = c->uses_geglu;
    ctx->num_experts = c->num_experts;
    ctx->num_experts_per_tok = c->num_experts_per_tok;
    ctx->expert_intermediate_size = c->expert_intermediate_size
                                  ? c->expert_intermediate_size : c->n_ff;
    ctx->shared_expert_intermediate_size = c->shared_expert_intermediate_size
                                         ? c->shared_expert_intermediate_size
                                         : ctx->expert_intermediate_size;
    ctx->expert_gating_sigmoid = c->expert_gating_sigmoid;
    ctx->expert_weights_scale = c->expert_weights_scale;
    ctx->uses_gemma4 = c->uses_gemma4;
    ctx->logit_softcap = c->logit_softcap;
    ctx->attn_scale = c->attn_scale;
    ctx->v_rms_norm = c->v_rms_norm;
    ctx->is_qwen35 = c->is_qwen35;
    ctx->n_full_attention_layers = c->is_qwen35 ? c->n_full_attention_layers
                                                : c->n_layer;
    ctx->n_recurrent_layers = c->n_recurrent_layers;
    ctx->ssm_conv_kernel = c->ssm_conv_kernel;
    ctx->ssm_group_count = c->ssm_group_count;
    ctx->ssm_state_size = c->ssm_state_size;
    ctx->ssm_value_heads = c->ssm_value_heads;
    ctx->ssm_inner_size = c->ssm_inner_size;

    /* Mirror the loader's resolved per-layer geometry onto the host side of
     * the context, so the forward loop reads it instead of re-deriving it.
     * These are small host arrays, not device memory. */
    if (c->uses_gemma4) {
        ctx->l_head_dim   = (uint32_t *)calloc(c->n_layer, sizeof(uint32_t));
        ctx->l_n_head_kv  = (uint32_t *)calloc(c->n_layer, sizeof(uint32_t));
        ctx->l_rope_dim   = (uint32_t *)calloc(c->n_layer, sizeof(uint32_t));
        ctx->l_rope_theta = (float *)calloc(c->n_layer, sizeof(float));
        ctx->l_sliding    = (uint32_t *)calloc(c->n_layer, sizeof(uint32_t));
        if (!ctx->l_head_dim || !ctx->l_n_head_kv || !ctx->l_rope_dim ||
            !ctx->l_rope_theta || !ctx->l_sliding) {
            oc_cuda_free(ctx);
            return OC_ERR_OOM;
        }
        ctx->l_out_scale = (float *)calloc(c->n_layer, sizeof(float));
        if (!ctx->l_out_scale) { oc_cuda_free(ctx); return OC_ERR_OOM; }
        for (uint32_t l = 0; l < c->n_layer; l++) {
            const OcLlamaLayer *L = &model->layers[l];
            ctx->l_out_scale[l]  = L->layer_output_scale;
            ctx->l_head_dim[l]   = L->head_dim ? L->head_dim : c->head_dim;
            ctx->l_n_head_kv[l]  = L->n_head_kv ? L->n_head_kv : c->n_head_kv;
            ctx->l_rope_dim[l]   = L->head_dim ? L->rope_dim : c->rope_dim;
            ctx->l_rope_theta[l] = L->head_dim ? L->rope_theta : c->rope_theta;
            ctx->l_sliding[l]    = L->sliding_window;
        }
    }

    if (c->is_qwen35) {
        ctx->l_kind = (uint8_t *)calloc(c->n_layer, sizeof(uint8_t));
        ctx->l_kv_index = (uint32_t *)calloc(c->n_layer, sizeof(uint32_t));
        if (!ctx->l_kind || !ctx->l_kv_index) {
            oc_cuda_free(ctx);
            return OC_ERR_OOM;
        }
        for (uint32_t l = 0; l < c->n_layer; l++) {
            ctx->l_kind[l] = (uint8_t)model->layers[l].kind;
            ctx->l_kv_index[l] = model->layers[l].kv_cache_index;
        }
    }

    /* Allocate workspace. Every per-token buffer is sized for the LARGEST
     * layer geometry, since one allocation serves all layers. */
    size_t embd = c->n_embd;
    size_t max_head_dim = c->head_dim;
    size_t kv_row = (size_t)c->n_head_kv * c->head_dim;
    if (c->uses_gemma4) {
        if (c->head_dim_swa > max_head_dim) max_head_dim = c->head_dim_swa;
        const size_t swa_row = (size_t)c->n_head_kv_swa * c->head_dim_swa;
        if (swa_row > kv_row) kv_row = swa_row;
    }
    size_t q_size = (size_t)c->n_head * max_head_dim;
    ctx->kv_row = kv_row;
    const uint32_t kv_layers = ctx->n_full_attention_layers
                             ? ctx->n_full_attention_layers : c->n_layer;
    size_t kv_total = (size_t)kv_layers * c->n_ctx * kv_row;
    size_t ff_scratch = c->n_ff;
    if (ctx->expert_intermediate_size > ff_scratch)
        ff_scratch = ctx->expert_intermediate_size;
    if (ctx->shared_expert_intermediate_size > ff_scratch)
        ff_scratch = ctx->shared_expert_intermediate_size;
    if (ff_scratch == 0) ff_scratch = 1;

#define CUDA_INIT_ALLOC(ptr, bytes) \
    do { \
        if (cudaMalloc((void **)(ptr), (bytes)) != cudaSuccess) { \
            oc_cuda_free(ctx); \
            return OC_ERR_BACKEND; \
        } \
    } while (0)
    CUDA_INIT_ALLOC(&ctx->d_x, embd * sizeof(float));
    CUDA_INIT_ALLOC(&ctx->d_normed, embd * sizeof(float));
    CUDA_INIT_ALLOC(&ctx->d_q, q_size * sizeof(float));
    CUDA_INIT_ALLOC(&ctx->d_k, kv_row * sizeof(float));
    CUDA_INIT_ALLOC(&ctx->d_v, kv_row * sizeof(float));
    CUDA_INIT_ALLOC(&ctx->d_attn_out, q_size * sizeof(float));
    CUDA_INIT_ALLOC(&ctx->d_ffn_gate_buf, ff_scratch * sizeof(float));
    CUDA_INIT_ALLOC(&ctx->d_ffn_up_buf, ff_scratch * sizeof(float));
    CUDA_INIT_ALLOC(&ctx->d_logits, c->vocab_size * sizeof(float));
    CUDA_INIT_ALLOC(&ctx->d_attn_scores,
                    (size_t)c->n_head * c->n_ctx * sizeof(float));
    if (ctx->num_experts > 0) {
        const size_t isz = ff_scratch;
        CUDA_INIT_ALLOC(&ctx->d_router_logits, ctx->num_experts * sizeof(float));
        CUDA_INIT_ALLOC(&ctx->d_expert_sel, ctx->num_experts * sizeof(uint32_t));
        CUDA_INIT_ALLOC(&ctx->d_expert_w, ctx->num_experts_per_tok * sizeof(float));
        CUDA_INIT_ALLOC(&ctx->d_expert_gate, isz * sizeof(float));
        CUDA_INIT_ALLOC(&ctx->d_expert_up, isz * sizeof(float));
        CUDA_INIT_ALLOC(&ctx->d_expert_tmp, embd * sizeof(float));
        CUDA_INIT_ALLOC(&ctx->d_expert_out, embd * sizeof(float));
        CUDA_INIT_ALLOC(&ctx->d_shexp_gate_logit, sizeof(float));
    }
    if (c->is_qwen35) {
        const size_t key_dim = (size_t)c->ssm_group_count * c->ssm_state_size;
        const size_t conv_dim = 2u * key_dim + c->ssm_inner_size;
        const size_t packed_q = 2u * (size_t)c->n_head * max_head_dim;
        const size_t qkv_bytes = (conv_dim > packed_q ? conv_dim : packed_q)
                                 * sizeof(float);
        const size_t gate_n = c->ssm_inner_size > q_size ? c->ssm_inner_size
                                                         : q_size;
        CUDA_INIT_ALLOC(&ctx->d_qwen35_qkv, qkv_bytes ? qkv_bytes : sizeof(float));
        CUDA_INIT_ALLOC(&ctx->d_qwen35_gate, gate_n * sizeof(float));
        if (c->ssm_value_heads > 0) {
            CUDA_INIT_ALLOC(&ctx->d_qwen35_beta,
                            c->ssm_value_heads * sizeof(float));
            CUDA_INIT_ALLOC(&ctx->d_qwen35_alpha,
                            c->ssm_value_heads * sizeof(float));
        }
        if (conv_dim > 0)
            CUDA_INIT_ALLOC(&ctx->d_qwen35_conv_out, conv_dim * sizeof(float));
        if (c->ssm_inner_size > 0)
            CUDA_INIT_ALLOC(&ctx->d_qwen35_delta_out,
                            c->ssm_inner_size * sizeof(float));
        if (c->n_recurrent_layers > 0 && conv_dim > 0 &&
            c->ssm_conv_kernel >= 2) {
            ctx->conv_state_per_layer = conv_dim * (c->ssm_conv_kernel - 1u);
            ctx->recurrent_state_per_layer =
                (size_t)c->ssm_inner_size * c->ssm_state_size;
            CUDA_INIT_ALLOC(&ctx->d_conv_state,
                            (size_t)c->n_recurrent_layers *
                            ctx->conv_state_per_layer * sizeof(float));
            CUDA_INIT_ALLOC(&ctx->d_recurrent_state,
                            (size_t)c->n_recurrent_layers *
                            ctx->recurrent_state_per_layer * sizeof(float));
            cudaMemset(ctx->d_conv_state, 0,
                       (size_t)c->n_recurrent_layers *
                       ctx->conv_state_per_layer * sizeof(float));
            cudaMemset(ctx->d_recurrent_state, 0,
                       (size_t)c->n_recurrent_layers *
                       ctx->recurrent_state_per_layer * sizeof(float));
        }
    }
    /* f16 KV cache: half the footprint of f32 for the same context. */
    CUDA_INIT_ALLOC(&ctx->d_kv_k, kv_total * sizeof(__half));
    CUDA_INIT_ALLOC(&ctx->d_kv_v, kv_total * sizeof(__half));
    ctx->vram_kv_bytes = 2 * kv_total * sizeof(__half);
    oc_log(OC_LOG_INFO, "cuda: KV cache f16, %.1f MB total",
           (double)ctx->vram_kv_bytes / 1e6);

    /* Upload embeddings. */
    /* Allocate host temp large enough for the largest weight row. */
    size_t max_cols = embd;
    if (c->n_ff > max_cols) max_cols = c->n_ff;
    if (c->vocab_size > max_cols) max_cols = c->vocab_size;
    if (ctx->expert_intermediate_size > max_cols)
        max_cols = ctx->expert_intermediate_size;
    if (ctx->shared_expert_intermediate_size > max_cols)
        max_cols = ctx->shared_expert_intermediate_size;
    if (c->ssm_inner_size > max_cols) max_cols = c->ssm_inner_size;
    float *host_temp = (float *)malloc(max_cols * sizeof(float));
    if (!host_temp) { oc_cuda_free(ctx); return OC_ERR_OOM; }

    OcError e = upload_weight(&model->tok_embeddings, &ctx->d_tok_embeddings,
                              host_temp, ctx);
    if (e != OC_OK) { free(host_temp); oc_cuda_free(ctx); return e; }
    oc_log(OC_LOG_DEBUG, "cuda: embeddings uploaded (%s)",
           ctx->d_tok_embeddings.packed ? "packed" : "f32");

    /* Upload final norm. */
    if (cudaMalloc((void **)&ctx->d_final_norm, embd * sizeof(float)) != cudaSuccess) {
        free(host_temp); oc_cuda_free(ctx); return OC_ERR_BACKEND;
    }
    cudaMemcpy(ctx->d_final_norm, model->final_norm, embd * sizeof(float),
              cudaMemcpyHostToDevice);

    /* Upload output (or alias tok_embeddings if tied). */
    if (model->cfg.tied_embeddings) {
        ctx->d_output = ctx->d_tok_embeddings;
    } else {
        e = upload_weight(&model->output, &ctx->d_output, host_temp, ctx);
        if (e != OC_OK) { free(host_temp); oc_cuda_free(ctx); return e; }
    }

    /* Upload per-layer weights. */
    ctx->d_attn_q_bias = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_attn_k_bias = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_attn_v_bias = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_attn_q = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
    ctx->d_attn_k = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
    ctx->d_attn_v = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
    ctx->d_attn_output = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
    ctx->d_ffn_gate = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
    ctx->d_ffn_up = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
    ctx->d_ffn_down = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
    ctx->d_attn_norm = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_ffn_norm = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_attn_q_norm = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_attn_k_norm = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_post_attn_norm = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_post_ffw_norm = (float **)calloc(c->n_layer, sizeof(float *));
    if (c->is_qwen35) {
        ctx->d_attn_qkv = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
        ctx->d_attn_gate = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
        ctx->d_ssm_alpha = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
        ctx->d_ssm_beta = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
        ctx->d_ssm_out = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
        ctx->d_ssm_conv1d = (float **)calloc(c->n_layer, sizeof(float *));
        ctx->d_ssm_a = (float **)calloc(c->n_layer, sizeof(float *));
        ctx->d_ssm_dt_bias = (float **)calloc(c->n_layer, sizeof(float *));
        ctx->d_ssm_norm = (float **)calloc(c->n_layer, sizeof(float *));
        if (!ctx->d_attn_qkv || !ctx->d_attn_gate || !ctx->d_ssm_alpha ||
            !ctx->d_ssm_beta || !ctx->d_ssm_out || !ctx->d_ssm_conv1d ||
            !ctx->d_ssm_a || !ctx->d_ssm_dt_bias || !ctx->d_ssm_norm) {
            free(host_temp); oc_cuda_free(ctx); return OC_ERR_OOM;
        }
    }
    if (ctx->num_experts > 0) {
        ctx->d_ffn_gate_inp = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
        ctx->d_ffn_gate_exps = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
        ctx->d_ffn_up_exps = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
        ctx->d_ffn_down_exps = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
        ctx->d_ffn_gate_shexp = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
        ctx->d_ffn_up_shexp = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
        ctx->d_ffn_down_shexp = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
        ctx->d_ffn_gate_inp_shexp = (OcCudaWeight *)calloc(c->n_layer, sizeof(OcCudaWeight));
        if (!ctx->d_ffn_gate_inp || !ctx->d_ffn_gate_exps || !ctx->d_ffn_up_exps ||
            !ctx->d_ffn_down_exps || !ctx->d_ffn_gate_shexp ||
            !ctx->d_ffn_up_shexp || !ctx->d_ffn_down_shexp ||
            !ctx->d_ffn_gate_inp_shexp) {
            free(host_temp); oc_cuda_free(ctx); return OC_ERR_OOM;
        }
    }
    if (!ctx->d_attn_q_bias || !ctx->d_attn_k_bias || !ctx->d_attn_v_bias ||
        !ctx->d_attn_q || !ctx->d_attn_k || !ctx->d_attn_v ||
        !ctx->d_attn_output || !ctx->d_ffn_gate || !ctx->d_ffn_up ||
        !ctx->d_ffn_down || !ctx->d_attn_norm || !ctx->d_ffn_norm ||
        !ctx->d_attn_q_norm || !ctx->d_attn_k_norm ||
        !ctx->d_post_attn_norm || !ctx->d_post_ffw_norm) {
        free(host_temp);
        oc_cuda_free(ctx);
        return OC_ERR_OOM;
    }

    for (uint32_t l = 0; l < c->n_layer; l++) {
        const OcLlamaLayer *L = &model->layers[l];
        oc_log(OC_LOG_DEBUG,
               "cuda: uploading layer %u/%u (q rows=%zu cols=%zu qtype=%s)",
               l, c->n_layer, L->attn_q.rows, L->attn_q.cols,
               oc_quant_type_name(L->attn_q.qtype));
        /* `stacked` marks the 3-D MoE expert tensors whose row count must be
         * multiplied by num_experts (see upload_weight_rows). */
        const uint32_t nx = ctx->num_experts;
        struct { const OcWeightView *view; OcCudaWeight *dst; bool stacked; } ws[] = {
            { &L->attn_q,      &ctx->d_attn_q[l],      false },
            { &L->attn_k,      &ctx->d_attn_k[l],      false },
            { &L->attn_v,      &ctx->d_attn_v[l],      false },
            { &L->attn_output, &ctx->d_attn_output[l], false },
            /* Dense FFN; empty views on a MoE layer, skipped below. */
            { &L->ffn_gate,    &ctx->d_ffn_gate[l],    false },
            { &L->ffn_up,      &ctx->d_ffn_up[l],      false },
            { &L->ffn_down,    &ctx->d_ffn_down[l],    false },
            /* MoE; all NULL on a dense layer, skipped below. */
            { &L->ffn_gate_inp,       ctx->d_ffn_gate_inp       ? &ctx->d_ffn_gate_inp[l]       : NULL, false },
            { &L->ffn_gate_exps,      ctx->d_ffn_gate_exps      ? &ctx->d_ffn_gate_exps[l]      : NULL, true  },
            { &L->ffn_up_exps,        ctx->d_ffn_up_exps        ? &ctx->d_ffn_up_exps[l]        : NULL, true  },
            { &L->ffn_down_exps,      ctx->d_ffn_down_exps      ? &ctx->d_ffn_down_exps[l]      : NULL, true  },
            { &L->ffn_gate_shexp,     ctx->d_ffn_gate_shexp     ? &ctx->d_ffn_gate_shexp[l]     : NULL, false },
            { &L->ffn_up_shexp,       ctx->d_ffn_up_shexp       ? &ctx->d_ffn_up_shexp[l]       : NULL, false },
            { &L->ffn_down_shexp,     ctx->d_ffn_down_shexp     ? &ctx->d_ffn_down_shexp[l]     : NULL, false },
            { &L->ffn_gate_inp_shexp, ctx->d_ffn_gate_inp_shexp ? &ctx->d_ffn_gate_inp_shexp[l] : NULL, false },
            { &L->attn_qkv,    ctx->d_attn_qkv    ? &ctx->d_attn_qkv[l]    : NULL, false },
            { &L->attn_gate,   ctx->d_attn_gate   ? &ctx->d_attn_gate[l]   : NULL, false },
            { &L->ssm_alpha,   ctx->d_ssm_alpha   ? &ctx->d_ssm_alpha[l]   : NULL, false },
            { &L->ssm_beta,    ctx->d_ssm_beta    ? &ctx->d_ssm_beta[l]    : NULL, false },
            { &L->ssm_out,     ctx->d_ssm_out     ? &ctx->d_ssm_out[l]     : NULL, false },
        };
        for (size_t wi = 0; wi < sizeof(ws) / sizeof(ws[0]); wi++) {
            /* A tensor absent from this model (dense FFN on a MoE layer, or
             * the optional shared expert) has a NULL data pointer. */
            if (ws[wi].dst == NULL || ws[wi].view->data == NULL) continue;
            const size_t ro = ws[wi].stacked ? (size_t)nx * ws[wi].view->rows : 0;
            e = upload_weight_rows(ws[wi].view, ws[wi].dst, host_temp, ctx, ro);
            if (e != OC_OK) { free(host_temp); oc_cuda_free(ctx); return e; }
        }
        if (cudaMalloc((void **)&ctx->d_attn_norm[l], embd * sizeof(float)) != cudaSuccess) {
            free(host_temp); oc_cuda_free(ctx); return OC_ERR_BACKEND;
        }
        cudaMemcpy(ctx->d_attn_norm[l], L->attn_norm, embd * sizeof(float),
                  cudaMemcpyHostToDevice);

        /* Optional QKV projection biases (Qwen2-family). Uploaded per layer;
         * a NULL host pointer leaves the device pointer NULL and the forward
         * pass skips the add. */
        {
            struct { const float *host; float **dev; size_t n; } bs[] = {
                { L->attn_q_bias, &ctx->d_attn_q_bias[l],
                  (size_t)c->n_head * c->head_dim },
                { L->attn_k_bias, &ctx->d_attn_k_bias[l],
                  (size_t)c->n_head_kv * c->head_dim },
                { L->attn_v_bias, &ctx->d_attn_v_bias[l],
                  (size_t)c->n_head_kv * c->head_dim },
            };
            for (size_t bi = 0; bi < 3; bi++) {
                if (bs[bi].host == NULL) continue;
                if (cudaMalloc((void **)bs[bi].dev,
                               bs[bi].n * sizeof(float)) != cudaSuccess) {
                    free(host_temp); oc_cuda_free(ctx); return OC_ERR_BACKEND;
                }
                cudaMemcpy(*bs[bi].dev, bs[bi].host, bs[bi].n * sizeof(float),
                           cudaMemcpyHostToDevice);
            }
        }
        if (L->ffn_norm != NULL) {
            if (cudaMalloc((void **)&ctx->d_ffn_norm[l], embd * sizeof(float)) != cudaSuccess) {
                free(host_temp); oc_cuda_free(ctx); return OC_ERR_BACKEND;
            }
            cudaMemcpy(ctx->d_ffn_norm[l], L->ffn_norm, embd * sizeof(float),
                      cudaMemcpyHostToDevice);
        }
        /* Gemma extra norms. Q/K norms are the LAYER's head_dim long; the
         * post norms are n_embd. Absent tensors leave a NULL entry, which the
         * forward pass treats as "skip". */
        {
            const size_t hd_l = ctx->l_head_dim ? ctx->l_head_dim[l]
                                                : ctx->head_dim;
            const struct { float **dev; const float *host; size_t n; } ns[] = {
                { &ctx->d_attn_q_norm[l],    L->attn_q_norm,        hd_l },
                { &ctx->d_attn_k_norm[l],    L->attn_k_norm,        hd_l },
                { &ctx->d_post_attn_norm[l], L->post_attention_norm, embd },
                { &ctx->d_post_ffw_norm[l],  L->post_ffw_norm,       embd },
            };
            for (size_t ni = 0; ni < 4; ni++) {
                if (ns[ni].host == NULL) continue;
                if (cudaMalloc((void **)ns[ni].dev,
                               ns[ni].n * sizeof(float)) != cudaSuccess) {
                    free(host_temp);
                    oc_cuda_free(ctx);
                    return OC_ERR_BACKEND;
                }
                cudaMemcpy(*ns[ni].dev, ns[ni].host, ns[ni].n * sizeof(float),
                           cudaMemcpyHostToDevice);
            }
        }
        if (ctx->is_qwen35 && L->kind == OC_LLAMA_LAYER_QWEN35_RECURRENT) {
            struct { float **dev; const OcWeightView *view; } fs[] = {
                { &ctx->d_ssm_conv1d[l],  &L->ssm_conv1d },
                { &ctx->d_ssm_a[l],       &L->ssm_a },
                { &ctx->d_ssm_dt_bias[l], &L->ssm_dt_bias },
                { &ctx->d_ssm_norm[l],    &L->ssm_norm },
            };
            for (size_t fi = 0; fi < 4; fi++) {
                if (fs[fi].view->data == NULL) continue;
                const size_t n = fs[fi].view->rows * fs[fi].view->cols;
                if (cudaMalloc((void **)fs[fi].dev, n * sizeof(float)) != cudaSuccess) {
                    free(host_temp); oc_cuda_free(ctx); return OC_ERR_BACKEND;
                }
                cudaMemcpy(*fs[fi].dev, fs[fi].view->data, n * sizeof(float),
                           cudaMemcpyHostToDevice);
                ctx->vram_weight_bytes += n * sizeof(float);
                ctx->n_f32_tensors++;
            }
        }
    }
    oc_log(OC_LOG_INFO,
           "cuda: %u tensors packed on device, %u dequantized to f32; "
           "weights %.1f MB + KV %.1f MB = %.1f MB VRAM",
           ctx->n_packed_tensors, ctx->n_f32_tensors,
           (double)ctx->vram_weight_bytes / 1e6,
           (double)ctx->vram_kv_bytes / 1e6,
           (double)(ctx->vram_weight_bytes + ctx->vram_kv_bytes) / 1e6);

    free(host_temp);
    ctx->initialized = true;
#undef CUDA_INIT_ALLOC
    return OC_OK;
}

OcError oc_cuda_forward(OcCudaContext *ctx, uint32_t token, uint32_t pos,
                        float *logits_out)
{
    if (!ctx || !ctx->initialized || pos >= ctx->n_ctx ||
        token >= ctx->vocab_size)
        return OC_ERR_INVALID_ARG;

    const uint32_t embd = ctx->n_embd;
    const uint32_t head_dim = ctx->head_dim;
    const uint32_t n_head = ctx->n_head;
    const uint32_t n_head_kv = ctx->n_head_kv;
    const uint32_t n_ff = ctx->n_ff;
    const float eps = ctx->rms_norm_eps;

    int block = 256;

    /* 1. Embedding lookup. */
    {
        OcError e = cuda_embed(&ctx->d_tok_embeddings, token, ctx->d_x, embd,
                               block);
        if (e != OC_OK) return e;
    }

    /* 2. Per-layer forward. */
    for (uint32_t l = 0; l < ctx->n_layer; l++) {
        /* Geometry is per-layer on Gemma 4 (sliding: head_dim 256 / 16 KV
         * heads / theta 1e4; global: 512 / 4 / 1e6). For every other model
         * these resolve to the model-wide scalars. */
        const uint32_t hd_l   = ctx->l_head_dim ? ctx->l_head_dim[l] : head_dim;
        const uint32_t nkv_l  = ctx->l_n_head_kv ? ctx->l_n_head_kv[l]
                                                 : n_head_kv;
        const uint32_t rd_l   = ctx->l_rope_dim ? ctx->l_rope_dim[l]
                                                : ctx->rope_dim;
        const float    rth_l  = ctx->l_rope_theta ? ctx->l_rope_theta[l]
                                                  : ctx->rope_theta;
        const uint32_t win_l  = ctx->l_sliding ? ctx->l_sliding[l] : 0u;

        /* Pre-attention RMSNorm. Gemma 4 folds its sqrt(n_embd) factor into
         * the embedding instead, so norm_scale must not be reapplied here —
         * matching embed_token in llama.c. */
        k_rms_norm<<<1, block, block * sizeof(float)>>>(
            ctx->d_x, ctx->d_attn_norm[l], ctx->d_normed, embd,
            eps, ctx->uses_gemma4 ? 1.0f : ctx->norm_scale);
        OC_CUDA_CHECK(cudaGetLastError());

        if (ctx->is_qwen35 && ctx->l_kind &&
            ctx->l_kind[l] == (uint8_t)OC_LLAMA_LAYER_QWEN35_RECURRENT) {
            OcError e = cuda_qwen35_recurrent(ctx, l, (uint32_t)block);
            if (e != OC_OK) return e;
        } else if (ctx->is_qwen35) {
            OcError e = cuda_qwen35_full_attn(ctx, l, pos, (uint32_t)block,
                                              hd_l, nkv_l, rd_l, rth_l, win_l);
            if (e != OC_OK) return e;
        } else {
        /* Q/K/V projections. */
        {
            OcError e = cuda_matvec_on(&ctx->d_attn_q[l], ctx->d_normed,
                                       ctx->d_q, block,
                                       ctx->compute_streams[0]);
            if (e != OC_OK) return e;
            e = cuda_matvec_on(&ctx->d_attn_k[l], ctx->d_normed, ctx->d_k,
                               block, ctx->compute_streams[1]);
            if (e != OC_OK) return e;
            e = cuda_matvec_on(&ctx->d_attn_v[l], ctx->d_normed, ctx->d_v,
                               block, ctx->compute_streams[2]);
            if (e != OC_OK) return e;
            e = cuda_wait_compute(ctx);
            if (e != OC_OK) return e;
        }

        /* Qwen2-family QKV projection biases, added before RoPE to match the
         * CPU forward in llama.c (and llama.cpp's build_qkv). */
        {
            size_t nq = (size_t)n_head * head_dim;
            size_t nkv = (size_t)n_head_kv * head_dim;
            if (ctx->d_attn_q_bias[l]) {
                k_add_bias<<<(unsigned)((nq + block - 1) / block), block>>>(
                    ctx->d_q, ctx->d_attn_q_bias[l], nq);
                OC_CUDA_CHECK(cudaGetLastError());
            }
            if (ctx->d_attn_k_bias[l]) {
                k_add_bias<<<(unsigned)((nkv + block - 1) / block), block>>>(
                    ctx->d_k, ctx->d_attn_k_bias[l], nkv);
                OC_CUDA_CHECK(cudaGetLastError());
            }
            if (ctx->d_attn_v_bias[l]) {
                k_add_bias<<<(unsigned)((nkv + block - 1) / block), block>>>(
                    ctx->d_v, ctx->d_attn_v_bias[l], nkv);
                OC_CUDA_CHECK(cudaGetLastError());
            }
        }

        /* Per-head Q/K RMSNorm fused with RoPE (Qwen3 / Gemma). */
        if (ctx->d_attn_q_norm[l] != NULL) {
            if (!oc_cuda_qk_norm_rope(ctx->d_q, ctx->d_attn_q_norm[l],
                                      n_head, hd_l, rd_l, (int64_t)pos,
                                      rth_l, eps, ctx->yarn_factor,
                                      ctx->yarn_orig_ctx, NULL))
                return OC_ERR_BACKEND;
        } else {
            k_apply_rope<<<n_head, block>>>(
                ctx->d_q, hd_l, rd_l, pos, rth_l, n_head,
                ctx->yarn_factor, ctx->yarn_orig_ctx);
            OC_CUDA_CHECK(cudaGetLastError());
        }
        if (ctx->d_attn_k_norm[l] != NULL) {
            if (!oc_cuda_qk_norm_rope(ctx->d_k, ctx->d_attn_k_norm[l],
                                      nkv_l, hd_l, rd_l, (int64_t)pos,
                                      rth_l, eps, ctx->yarn_factor,
                                      ctx->yarn_orig_ctx, NULL))
                return OC_ERR_BACKEND;
        } else {
            k_apply_rope<<<nkv_l, block>>>(
                ctx->d_k, hd_l, rd_l, pos, rth_l, nkv_l,
                ctx->yarn_factor, ctx->yarn_orig_ctx);
            OC_CUDA_CHECK(cudaGetLastError());
        }

        /* Gemma 4 also RMS-normalizes V (weightless), before the cache
         * write. V never gets RoPE, so this is the last touch. */
        if (ctx->v_rms_norm) {
            k_rms_norm_heads<<<nkv_l, block, block * sizeof(float)>>>(
                ctx->d_v, NULL, hd_l, eps);
            OC_CUDA_CHECK(cudaGetLastError());
        }

        /* KV cache write. */
        /* Layer base uses the UNIFORM stride, not this layer's own row size,
         * so layers stay at fixed offsets regardless of geometry. */
        size_t kv_live = (size_t)nkv_l * hd_l;
        __half *kv_k_layer = (__half *)ctx->d_kv_k
                           + (size_t)l * ctx->n_ctx * ctx->kv_row;
        __half *kv_v_layer = (__half *)ctx->d_kv_v
                           + (size_t)l * ctx->n_ctx * ctx->kv_row;
        k_kv_cache_write<<<(kv_live + block - 1) / block, block>>>(
            kv_k_layer, kv_v_layer,
            ctx->d_k, ctx->d_v, pos, nkv_l, hd_l, ctx->n_ctx, ctx->kv_row);
        OC_CUDA_CHECK(cudaGetLastError());

        /* Attention: all heads in one launch. */
        {
            const uint32_t athreads = OC_CUDA_ATTN_THREADS;
            const size_t asmem = ((size_t)hd_l + athreads) * sizeof(float);
            /* Sliding layers attend only to the last `win_l` positions. This
             * is what keeps 50 of Gemma 4's 60 layers nearly independent of
             * context length. */
            const uint32_t n_past = pos + 1;
            const uint32_t kv_start =
                (win_l > 0 && n_past > win_l) ? (n_past - win_l) : 0u;
            k_attention_all_heads<<<n_head, athreads, asmem>>>(
                ctx->d_q, kv_k_layer, kv_v_layer, ctx->d_attn_out,
                ctx->d_attn_scores, ctx->n_ctx,
                n_head, nkv_l, hd_l, n_past,
                ctx->attn_scale > 0.0f ? ctx->attn_scale
                                       : rsqrtf((float)hd_l),
                ctx->kv_row, kv_start);
            OC_CUDA_CHECK(cudaGetLastError());
        }

        /* Output projection. */
        {
            OcError e = cuda_matvec(&ctx->d_attn_output[l], ctx->d_attn_out,
                                    ctx->d_normed, block);
            if (e != OC_OK) return e;
        }
        /* Gemma sandwich norm: the attention branch output is normed again
         * before rejoining the residual stream. */
        if (ctx->d_post_attn_norm[l] != NULL) {
            k_rms_norm<<<1, block, block * sizeof(float)>>>(
                ctx->d_normed, ctx->d_post_attn_norm[l], ctx->d_attn_out,
                embd, eps, 1.0f);
            OC_CUDA_CHECK(cudaGetLastError());
            OC_CUDA_CHECK(cudaMemcpyAsync(ctx->d_normed, ctx->d_attn_out,
                embd * sizeof(float), cudaMemcpyDeviceToDevice, 0));
        }
        }

        /* Residual + pre-FFN RMSNorm. Qwen3.5 stores the FFN pre-norm as
         * post_attention_norm (not a Gemma sandwich). */
        {
            const float *ffn_n = (ctx->is_qwen35 && ctx->d_post_attn_norm[l])
                ? ctx->d_post_attn_norm[l] : ctx->d_ffn_norm[l];
            if (ffn_n == NULL) return OC_ERR_MODEL;
            k_residual_rms<<<1, block, block * sizeof(float)>>>(
                ctx->d_x, ctx->d_normed, ffn_n, ctx->d_normed, embd,
                eps, ctx->uses_gemma4 ? 1.0f : ctx->norm_scale);
            OC_CUDA_CHECK(cudaGetLastError());
        }

        /* FFN: MoE (router + top-k experts + shared) or dense. */
        if (ctx->num_experts > 0) {
            OcError e = cuda_moe_ffn(ctx, l, block);
            if (e != OC_OK) return e;
            continue;
        }

        /* FFN: gate, up, SwiGLU, down. */
        {
            OcError e = cuda_matvec_on(&ctx->d_ffn_gate[l], ctx->d_normed,
                                       ctx->d_ffn_gate_buf, block,
                                       ctx->compute_streams[0]);
            if (e != OC_OK) return e;
            e = cuda_matvec_on(&ctx->d_ffn_up[l], ctx->d_normed,
                               ctx->d_ffn_up_buf, block,
                               ctx->compute_streams[1]);
            if (e != OC_OK) return e;
            e = cuda_wait_compute(ctx);
            if (e != OC_OK) return e;
        }
        if (ctx->uses_geglu) {
            k_geglu<<<(n_ff + block - 1) / block, block>>>(
                ctx->d_ffn_gate_buf, ctx->d_ffn_up_buf, n_ff);
        } else {
            k_swiglu<<<(n_ff + block - 1) / block, block>>>(
                ctx->d_ffn_gate_buf, ctx->d_ffn_up_buf, n_ff);
        }
        OC_CUDA_CHECK(cudaGetLastError());
        {
            OcError e = cuda_matvec(&ctx->d_ffn_down[l], ctx->d_ffn_gate_buf,
                                    ctx->d_normed, block);
            if (e != OC_OK) return e;
        }
        if (ctx->d_post_ffw_norm[l] != NULL) {
            k_rms_norm<<<1, block, block * sizeof(float)>>>(
                ctx->d_normed, ctx->d_post_ffw_norm[l], ctx->d_attn_out,
                embd, eps, 1.0f);
            OC_CUDA_CHECK(cudaGetLastError());
            OC_CUDA_CHECK(cudaMemcpyAsync(ctx->d_normed, ctx->d_attn_out,
                embd * sizeof(float), cudaMemcpyDeviceToDevice, 0));
        }
        k_residual_add<<<(embd + block - 1) / block, block>>>(
            ctx->d_x, ctx->d_normed, embd);
        OC_CUDA_CHECK(cudaGetLastError());

        /* Gemma 4 per-layer output scale: multiplies the whole running
         * residual stream, not one branch. 0 means "unset" rather than
         * "scale by zero" — see the matching guard in llama.c. */
        if (ctx->l_out_scale && ctx->l_out_scale[l] != 0.0f &&
            ctx->l_out_scale[l] != 1.0f) {
            k_scale<<<(embd + block - 1) / block, block>>>(
                ctx->d_x, ctx->l_out_scale[l], embd);
            OC_CUDA_CHECK(cudaGetLastError());
        }
    }

    /* 3. Final norm + lm_head. */
    k_rms_norm<<<1, block, block * sizeof(float)>>>(
        ctx->d_x, ctx->d_final_norm, ctx->d_normed, embd,
        eps, ctx->uses_gemma4 ? 1.0f : ctx->norm_scale);
    OC_CUDA_CHECK(cudaGetLastError());

    if (logits_out != NULL) {
        OcError e = cuda_matvec(&ctx->d_output, ctx->d_normed, ctx->d_logits,
                                block);
        if (e != OC_OK) return e;
        /* Gemma 4 softcaps the final logits; it reshapes the sampling
         * distribution, so it is part of the model. */
        if (ctx->logit_softcap > 0.0f) {
            k_softcap<<<(ctx->vocab_size + block - 1) / block, block>>>(
                ctx->d_logits, ctx->logit_softcap, ctx->vocab_size);
            OC_CUDA_CHECK(cudaGetLastError());
        }
        OC_CUDA_CHECK(cudaDeviceSynchronize());
        OC_CUDA_CHECK(cudaMemcpy(logits_out, ctx->d_logits,
            ctx->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        OC_CUDA_CHECK(cudaDeviceSynchronize());
    }

    return OC_OK;
}

void oc_cuda_reset(OcCudaContext *ctx)
{
    if (!ctx) return;
    if (ctx->d_conv_state && ctx->n_recurrent_layers && ctx->conv_state_per_layer)
        cudaMemset(ctx->d_conv_state, 0,
                   (size_t)ctx->n_recurrent_layers *
                   ctx->conv_state_per_layer * sizeof(float));
    if (ctx->d_recurrent_state && ctx->n_recurrent_layers &&
        ctx->recurrent_state_per_layer)
        cudaMemset(ctx->d_recurrent_state, 0,
                   (size_t)ctx->n_recurrent_layers *
                   ctx->recurrent_state_per_layer * sizeof(float));
}

void oc_cuda_free(OcCudaContext *ctx)
{
    if (!ctx) return;

    cudaFree(ctx->d_tok_embeddings.data);
    /* d_output aliases d_tok_embeddings when embeddings are tied. */
    if (ctx->d_output.data && ctx->d_output.data != ctx->d_tok_embeddings.data)
        cudaFree(ctx->d_output.data);
    cudaFree(ctx->d_final_norm);
    cudaFree(ctx->d_x); cudaFree(ctx->d_normed);
    cudaFree(ctx->d_q); cudaFree(ctx->d_k); cudaFree(ctx->d_v);
    cudaFree(ctx->d_attn_out);
    cudaFree(ctx->d_ffn_gate_buf); cudaFree(ctx->d_ffn_up_buf);
    cudaFree(ctx->d_logits); cudaFree(ctx->d_attn_scores);
    cudaFree(ctx->d_kv_k); cudaFree(ctx->d_kv_v);
    cudaFree(ctx->d_router_logits); cudaFree(ctx->d_expert_sel);
    cudaFree(ctx->d_expert_w); cudaFree(ctx->d_expert_gate);
    cudaFree(ctx->d_expert_up); cudaFree(ctx->d_expert_tmp);
    cudaFree(ctx->d_expert_out); cudaFree(ctx->d_shexp_gate_logit);

    for (uint32_t l = 0; l < ctx->n_layer; l++) {
        if (ctx->d_attn_q_bias && ctx->d_attn_q_bias[l]) cudaFree(ctx->d_attn_q_bias[l]);
        if (ctx->d_attn_k_bias && ctx->d_attn_k_bias[l]) cudaFree(ctx->d_attn_k_bias[l]);
        if (ctx->d_attn_v_bias && ctx->d_attn_v_bias[l]) cudaFree(ctx->d_attn_v_bias[l]);
        if (ctx->d_attn_q) cudaFree(ctx->d_attn_q[l].data);
        if (ctx->d_attn_k) cudaFree(ctx->d_attn_k[l].data);
        if (ctx->d_attn_v) cudaFree(ctx->d_attn_v[l].data);
        if (ctx->d_attn_output) cudaFree(ctx->d_attn_output[l].data);
        if (ctx->d_ffn_gate) cudaFree(ctx->d_ffn_gate[l].data);
        if (ctx->d_ffn_up) cudaFree(ctx->d_ffn_up[l].data);
        if (ctx->d_ffn_down) cudaFree(ctx->d_ffn_down[l].data);
        if (ctx->d_ffn_gate_inp) cudaFree(ctx->d_ffn_gate_inp[l].data);
        if (ctx->d_ffn_gate_exps) cudaFree(ctx->d_ffn_gate_exps[l].data);
        if (ctx->d_ffn_up_exps) cudaFree(ctx->d_ffn_up_exps[l].data);
        if (ctx->d_ffn_down_exps) cudaFree(ctx->d_ffn_down_exps[l].data);
        if (ctx->d_ffn_gate_shexp) cudaFree(ctx->d_ffn_gate_shexp[l].data);
        if (ctx->d_ffn_up_shexp) cudaFree(ctx->d_ffn_up_shexp[l].data);
        if (ctx->d_ffn_down_shexp) cudaFree(ctx->d_ffn_down_shexp[l].data);
        if (ctx->d_ffn_gate_inp_shexp) cudaFree(ctx->d_ffn_gate_inp_shexp[l].data);
        if (ctx->d_attn_norm) cudaFree(ctx->d_attn_norm[l]);
        if (ctx->d_ffn_norm) cudaFree(ctx->d_ffn_norm[l]);
        if (ctx->d_attn_q_norm) cudaFree(ctx->d_attn_q_norm[l]);
        if (ctx->d_attn_k_norm) cudaFree(ctx->d_attn_k_norm[l]);
        if (ctx->d_post_attn_norm) cudaFree(ctx->d_post_attn_norm[l]);
        if (ctx->d_post_ffw_norm) cudaFree(ctx->d_post_ffw_norm[l]);
        if (ctx->d_attn_qkv) cudaFree(ctx->d_attn_qkv[l].data);
        if (ctx->d_attn_gate) cudaFree(ctx->d_attn_gate[l].data);
        if (ctx->d_ssm_alpha) cudaFree(ctx->d_ssm_alpha[l].data);
        if (ctx->d_ssm_beta) cudaFree(ctx->d_ssm_beta[l].data);
        if (ctx->d_ssm_out) cudaFree(ctx->d_ssm_out[l].data);
        if (ctx->d_ssm_conv1d) cudaFree(ctx->d_ssm_conv1d[l]);
        if (ctx->d_ssm_a) cudaFree(ctx->d_ssm_a[l]);
        if (ctx->d_ssm_dt_bias) cudaFree(ctx->d_ssm_dt_bias[l]);
        if (ctx->d_ssm_norm) cudaFree(ctx->d_ssm_norm[l]);
    }
    free(ctx->d_attn_q_bias); free(ctx->d_attn_k_bias); free(ctx->d_attn_v_bias);
    free(ctx->d_attn_q); free(ctx->d_attn_k); free(ctx->d_attn_v);
    free(ctx->d_attn_output);
    free(ctx->d_ffn_gate); free(ctx->d_ffn_up); free(ctx->d_ffn_down);
    free(ctx->d_attn_norm); free(ctx->d_ffn_norm);
    free(ctx->d_attn_q_norm); free(ctx->d_attn_k_norm);
    free(ctx->d_post_attn_norm); free(ctx->d_post_ffw_norm);
    free(ctx->l_head_dim); free(ctx->l_n_head_kv); free(ctx->l_rope_dim);
    free(ctx->l_rope_theta); free(ctx->l_sliding); free(ctx->l_out_scale);
    free(ctx->l_kind); free(ctx->l_kv_index);
    free(ctx->d_ffn_gate_inp); free(ctx->d_ffn_gate_exps);
    free(ctx->d_ffn_up_exps); free(ctx->d_ffn_down_exps);
    free(ctx->d_ffn_gate_shexp); free(ctx->d_ffn_up_shexp);
    free(ctx->d_ffn_down_shexp); free(ctx->d_ffn_gate_inp_shexp);
    free(ctx->d_attn_qkv); free(ctx->d_attn_gate);
    free(ctx->d_ssm_alpha); free(ctx->d_ssm_beta); free(ctx->d_ssm_out);
    free(ctx->d_ssm_conv1d); free(ctx->d_ssm_a);
    free(ctx->d_ssm_dt_bias); free(ctx->d_ssm_norm);
    cudaFree(ctx->d_conv_state); cudaFree(ctx->d_recurrent_state);
    cudaFree(ctx->d_qwen35_qkv); cudaFree(ctx->d_qwen35_gate);
    cudaFree(ctx->d_qwen35_beta); cudaFree(ctx->d_qwen35_alpha);
    cudaFree(ctx->d_qwen35_conv_out); cudaFree(ctx->d_qwen35_delta_out);
    for (int i = 0; i < 3; i++) {
        if (ctx->compute_streams[i])
            cudaStreamDestroy((cudaStream_t)ctx->compute_streams[i]);
    }

    memset(ctx, 0, sizeof(*ctx));
}
