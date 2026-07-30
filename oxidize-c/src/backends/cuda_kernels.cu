/*
 * cuda_kernels.cu — Fused CUDA kernels for GPU-accelerated LLM inference.
 *
 * This file is the standalone companion to cuda.cu. It provides eight
 * high-throughput fused kernels that can be linked into the C port via
 * the host-side wrappers declared in include/oxidize/cuda_kernels.h.
 *
 * Kernels:
 *   1. RMSNorm + RoPE fused              (k_rmsnorm_rope_fused)
 *   2. SwiGLU activation                 (k_swiglu)
 *   3. Attention online softmax          (k_attention_softmax)
 *   4. Q4_K quantized matvec             (k_q4k_matvec)
 *   5. Q4_K → F32 dequantize             (k_q4k_dequantize)
 *   6. Embedding lookup                  (k_embed_lookup)
 *   7. Argmax (greedy sampling)          (k_argmax)
 *   8. Top-k partial sort                (k_topk)
 *
 * Build:
 *   nvcc -O2 -c src/backends/cuda_kernels.cu -o src/backends/cuda_kernels.cu.o
 *
 * The Q4_K block layout matches oxidize-c/src/compute/quantization.c
 * (OC_BLOCK_Q4_K_SIZE = 144 bytes, OC_QK_K = 256 elements per super-block):
 *   block[0..1]   : d   (f16 LE)
 *   block[2..3]   : min (f16 LE)
 *   block[4..15]  : 12-byte scale/min table (get_scale_min_k4)
 *   block[16..143]: 128 bytes = 256 4-bit nibbles
 *
 * Four groups of 64 elements each; group `gp` (0..3) uses scale indices
 * (2*gp, 2*gp+1). Within a group, the first 32 elements read the low
 * nibble (qs[i] & 0x0F), the next 32 read the high nibble (qs[i] >> 4).
 */
#include "oxidize/cuda_kernels.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

/* ─── Compile-time constants (must match oxidize-c/include/oxidize/quant.h) ── */
#define CK_QK_K             256u
#define CK_BLOCK_Q4_K_SIZE  144u
/* Top-k limit: the block staging area needs blockDim.x * k * 8 bytes of
 * shared memory; with the minimum 32-thread block that caps k at
 * 48 KiB / (32 * 8) = 192. */
#define CK_MAX_TOPK         192u
#define CK_TOPK_SMEM_MAX    (48u * 1024u)

/* ─── CUDA error checking macro ───────────────────────────────────────────── */
#define CK_CUDA_CHECK(call)                                              \
    do {                                                                 \
        cudaError_t _ck_e = (call);                                      \
        if (_ck_e != cudaSuccess) {                                       \
            return false;                                                  \
        }                                                                 \
    } while (0)

/* ─── Device-side helpers ──────────────────────────────────────────────────── */

/* f16 (little-endian 2 bytes) → f32. Mirrors quantization.c::f16_le_to_f32. */
__device__ __forceinline__ float ck_f16_to_f32(uint16_t bits)
{
    uint32_t sign = (uint32_t)((bits >> 15) & 1u);
    uint32_t exp  = (uint32_t)((bits >> 10) & 0x1Fu);
    uint32_t frac = (uint32_t)(bits & 0x03FFu);
    uint32_t f;

    if (exp == 0u) {
        if (frac == 0u) {
            f = sign << 31;
        } else {
            uint32_t fn = frac;
            int32_t e = -14;
            while ((fn & 0x0400u) == 0u) { fn <<= 1; e -= 1; }
            fn &= 0x03FFu;
            f = (sign << 31) | (((uint32_t)(e + 127)) << 23) | (fn << 13);
        }
    } else if (exp == 0x1Fu) {
        f = (sign << 31) | 0x7F800000u | (frac << 13);
    } else {
        int32_t e = (int32_t)exp - 15 + 127;
        f = (sign << 31) | (((uint32_t)e) << 23) | (frac << 13);
    }
    return __uint_as_float(f);
}

/* Assemble a 16-bit LE value from two bytes. */
__device__ __forceinline__ uint16_t ck_f16_from_bytes(uint8_t b0, uint8_t b1)
{
    return (uint16_t)((uint16_t)b0 | ((uint16_t)b1 << 8));
}

/* Q4_K scale/min unpack. Mirrors quantization.c::get_scale_min_k4. */
__device__ __forceinline__ void ck_get_scale_min_k4(uint32_t j,
                                                     const uint8_t *scales,
                                                     uint8_t *out_sc,
                                                     uint8_t *out_m)
{
    if (j < 4u) {
        *out_sc = (uint8_t)(scales[j] & 63u);
        *out_m  = (uint8_t)(scales[j + 4] & 63u);
    } else {
        *out_sc = (uint8_t)((scales[j + 4] & 0x0Fu)
                            | ((scales[j - 4] >> 6) << 4));
        *out_m  = (uint8_t)(((scales[j + 4] >> 4) & 0x0Fu)
                            | ((scales[j] >> 6) << 4));
    }
}

/* Dequantize one Q4_K super-block (256 elements, 144 bytes) → f32.
 * `block` points to the start of the 144-byte block. */
__device__ __forceinline__ void ck_dequant_q4k_block(const uint8_t *block,
                                                      float *out)
{
    float d   = ck_f16_to_f32(ck_f16_from_bytes(block[0], block[1]));
    float min = ck_f16_to_f32(ck_f16_from_bytes(block[2], block[3]));
    const uint8_t *scales = block + 4;
    const uint8_t *qs     = block + 16;

    uint32_t is = 0u;
    uint32_t out_ptr = 0u;
    for (uint32_t gp = 0u; gp < 4u; gp++) {
        uint32_t q_base = gp * 32u;
        uint8_t sc1, m1, sc2, m2;
        ck_get_scale_min_k4(is,     scales, &sc1, &m1);
        ck_get_scale_min_k4(is + 1, scales, &sc2, &m2);
        float d1 = d * (float)sc1;
        float min1 = min * (float)m1;
        float d2 = d * (float)sc2;
        float min2 = min * (float)m2;
        for (uint32_t l = 0u; l < 32u; l++) {
            out[out_ptr + l] = d1 * (float)(qs[q_base + l] & 0x0Fu) - min1;
        }
        for (uint32_t l = 0u; l < 32u; l++) {
            out[out_ptr + 32u + l] =
                d2 * (float)(qs[q_base + l] >> 4) - min2;
        }
        out_ptr += 64u;
        is += 2u;
    }
}

/* ─── 1. RMSNorm + RoPE fused kernel ──────────────────────────────────────────
 *
 * Fuses RMSNorm and rotary position embedding for a Q (or K) tensor with
 * `n_heads` heads. One block handles one head; threads cooperatively
 * compute the RMS sum, then each thread applies norm + rotation.
 *
 * Layout: the input `x` of length `hidden_dim = n_heads * head_dim` is
 * first RMS-normalized (with `weight`), then each head's slice of
 * `rope_dim` elements is rotated by `pos` using the NeoX split-halves
 * convention (rotate [i] with [i + rope_dim/2]).
 */
__global__ void k_rmsnorm_rope_fused(const float *x,
                                      const float *weight,
                                      float *out,
                                      uint32_t hidden_dim,
                                      uint32_t n_heads,
                                      uint32_t head_dim,
                                      uint32_t rope_dim,
                                      int64_t pos,
                                      float theta,
                                      float eps,
                                      float norm_scale)
{
    uint32_t head = blockIdx.x;
    if (head >= n_heads) return;

    uint32_t tid = threadIdx.x;
    uint32_t blockSize = blockDim.x;
    const float *xh = x + head * head_dim;
    float *oh = out + head * head_dim;
    const float *wh = weight + head * head_dim;

    /* Step 1: full-vector RMS reduction in shared memory. RMSNorm is
     * defined over the whole hidden vector, so every block redundantly
     * computes the same sum over all hidden_dim elements. */
    extern __shared__ float sdata[];
    float v = 0.0f;
    for (uint32_t i = tid; i < hidden_dim; i += blockSize)
        v += x[i] * x[i];
    sdata[tid] = v;
    __syncthreads();

    for (uint32_t s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_rms = norm_scale / sqrtf(sdata[0] / (float)hidden_dim + eps);

    /* Step 2: normalize + apply RoPE (split-halves). */
    uint32_t half = rope_dim / 2u;
    /* Precompute base frequency multiplier once. */
    for (uint32_t i = tid; i < half; i += blockSize) {
        float xn = xh[i] * inv_rms * wh[i];
        float xn2 = xh[i + half] * inv_rms * wh[i + half];
        /* Match activation.c::oc_apply_rope_f32: theta^(-2i/head_dim). */
        float freq = powf(theta, -2.0f * (float)i / (float)head_dim);
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        oh[i] = xn * c - xn2 * s;
        oh[i + half] = xn * s + xn2 * c;
    }
    /* Pass-through (normalized only) for any remaining dims beyond rope_dim. */
    for (uint32_t i = half * 2u + tid; i < head_dim; i += blockSize) {
        oh[i] = xh[i] * inv_rms * wh[i];
    }
}

/* ─── 2. SwiGLU activation kernel ────────────────────────────────────────────
 *
 * silu(gate) * up, where silu(x) = x * sigmoid(x). One element per thread.
 */
/* Named k_swiglu_fused to avoid a duplicate symbol with cuda.cu's k_swiglu. */
__global__ void k_swiglu_fused(float *gate, const float *up, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        gate[i] = g * sig * up[i];
    }
}

/* ─── 3. Attention online softmax kernel ─────────────────────────────────────
 *
 * Flash-attention-style online softmax for a single query head against a
 * cached KV. One block, threads cooperate.
 *
 * Pass 1: compute all QK scores, track running max + weighted V sum
 *         (the online-softmax trick: rescale the running sum by
 *         exp(old_max - new_max) when a new max is observed).
 * Pass 2: normalize by the running sum of weights.
 *
 * `d_k_cache` / `d_v_cache` are laid out as [n_past, head_dim].
 */
__global__ void k_attention_softmax(const float *q,
                                    const float *kv_k,
                                    const float *kv_v,
                                    float *out,
                                    uint32_t head_dim,
                                    size_t n_past)
{
    uint32_t tid = threadIdx.x;
    uint32_t blockSize = blockDim.x;

    extern __shared__ float smem[];
    float *acc = smem;                 /* head_dim accumulators */
    /* The caller reserves a second head_dim span after `acc` for a future
     * two-pass softmax; the online formulation below does not need it. */

    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    /* Zero accumulators. */
    for (uint32_t d = tid; d < head_dim; d += blockSize) acc[d] = 0.0f;
    __syncthreads();

    /* Online softmax: maintain running max m and running weighted sum. */
    float m = -INFINITY;   /* running max */
    float denom = 0.0f;    /* running sum of exp(score - m) */

    for (size_t p = 0; p < n_past; p++) {
        const float *k = kv_k + p * head_dim;
        const float *v = kv_v + p * head_dim;

        /* Dot product Q·K. */
        float score = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++)
            score += q[d] * k[d];
        score *= inv_sqrt_d;

        /* Update running max; rescale accumulator if max changed. */
        float m_new = fmaxf(m, score);
        float rescale = expf(m - m_new);
        for (uint32_t d = tid; d < head_dim; d += blockSize)
            acc[d] *= rescale;
        __syncthreads();

        float w = expf(score - m_new);
        denom = denom * rescale + w;
        m = m_new;

        for (uint32_t d = tid; d < head_dim; d += blockSize)
            acc[d] += w * v[d];
        __syncthreads();
    }

    /* Normalize and write output. */
    float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    for (uint32_t d = tid; d < head_dim; d += blockSize)
        out[d] = acc[d] * inv_denom;
}

/* ─── 4. Q4_K quantized matvec kernel ─────────────────────────────────────────
 *
 * Q4_K weight × F32 activation matrix-vector multiply.
 * One block per output row. Each thread accumulates a partial sum over
 * the super-blocks assigned to it; a tree reduction combines partials.
 *
 * `d_weights` points to packed Q4_K data; `cols` must be a multiple of
 * CK_QK_K (256). Row stride in bytes = (cols / 256) * 144.
 */
__global__ void k_q4k_matvec(const uint8_t *weights,
                              const float *x,
                              float *out,
                              size_t rows,
                              size_t cols)
{
    size_t row = blockIdx.x;
    if (row >= rows) return;

    size_t n_blocks_per_row = cols / CK_QK_K;     /* Q4_K super-blocks per row */
    size_t row_bytes = n_blocks_per_row * CK_BLOCK_Q4_K_SIZE;
    const uint8_t *row_w = weights + row * row_bytes;

    uint32_t tid = threadIdx.x;
    uint32_t blockSize = blockDim.x;

    extern __shared__ float sdata[];
    float partial = 0.0f;

    /* Each thread strides over super-blocks. */
    for (size_t b = tid; b < n_blocks_per_row; b += blockSize) {
        const uint8_t *blk = row_w + b * CK_BLOCK_Q4_K_SIZE;
        /* Dequantize this 256-element block into registers. */
        float vals[CK_QK_K];
        ck_dequant_q4k_block(blk, vals);
        const float *xb = x + b * CK_QK_K;
        #pragma unroll
        for (uint32_t i = 0; i < CK_QK_K; i++)
            partial += vals[i] * xb[i];
    }
    sdata[tid] = partial;
    __syncthreads();

    /* Tree reduction. */
    for (uint32_t s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[row] = sdata[0];
}

/* ─── 5. Q4_K → F32 dequantize kernel ────────────────────────────────────────
 *
 * Dequantizes an entire Q4_K tensor to F32. One thread per super-block
 * (256 elements). Used during GPU offload (weight upload).
 */
__global__ void k_q4k_dequantize(const uint8_t *src, float *dst,
                                  size_t n_blocks)
{
    size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    ck_dequant_q4k_block(src + b * CK_BLOCK_Q4_K_SIZE,
                         dst + b * CK_QK_K);
}

/* ─── 6. Embedding lookup kernel ─────────────────────────────────────────────
 *
 * Gathers embedding rows from a [vocab_size, embd_dim] table.
 * One block per (token, chunk) pair; each thread copies one element.
 */
__global__ void k_embed_lookup(const float *embeddings,
                               const uint32_t *tokens,
                               float *out,
                               uint32_t vocab_size,
                               uint32_t embd_dim,
                               size_t n_tokens)
{
    size_t tok_idx = blockIdx.x;
    if (tok_idx >= n_tokens) return;

    uint32_t token = tokens[tok_idx];
    if (token >= vocab_size) token = 0u;  /* clamp to <unk>-ish */

    size_t base_in  = (size_t)token * embd_dim;
    size_t base_out = tok_idx * embd_dim;

    uint32_t tid = threadIdx.x;
    uint32_t blockSize = blockDim.x;
    for (uint32_t i = tid; i < embd_dim; i += blockSize)
        out[base_out + i] = embeddings[base_in + i];
}

/* ─── 7. Argmax kernel (greedy sampling) ─────────────────────────────────────
 *
 * Single-block reduction argmax. Suitable for vocab sizes up to ~32k
 * when launched with blockDim.x = 1024 (each thread handles ~32 logits).
 * For larger vocabs, a two-pass approach would be needed.
 */
__global__ void k_argmax(const float *logits, uint32_t *out_idx,
                         uint32_t vocab_size)
{
    extern __shared__ float sdata[];      /* shared value buffer */
    uint32_t *sidx = (uint32_t *)(sdata + blockDim.x);  /* shared idx buffer */

    uint32_t tid = threadIdx.x;
    uint32_t blockSize = blockDim.x;

    float best_val = -INFINITY;
    uint32_t best_i = 0u;
    for (uint32_t i = tid; i < vocab_size; i += blockSize) {
        float v = logits[i];
        if (v > best_val) { best_val = v; best_i = i; }
    }
    sdata[tid] = best_val;
    sidx[tid]  = best_i;
    __syncthreads();

    /* Tree reduction tracking (val, idx) pairs. */
    for (uint32_t s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = sdata[tid];
            float b = sdata[tid + s];
            if (b > a) {
                sdata[tid] = b;
                sidx[tid]  = sidx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) *out_idx = sidx[0];
}

/* ─── 8. Top-k partial sort kernel ────────────────────────────────────────────
 *
 * Returns the indices of the top-k logits in descending order.
 * Strategy: each thread maintains a local top-k list (of size k) by
 * scanning a strided slice of the vocab. A shared-memory bitonic merge
 * then produces the final top-k.
 *
 * Constraints: k ≤ CK_MAX_TOPK (192), k ≤ vocab_size, and blockDim.x must
 * be a power of two ≤ 1024 with blockDim.x * k * 8 bytes of shared memory.
 */
__global__ void k_topk(const float *logits,
                       uint32_t *out_idx,
                       float *out_val,
                       uint32_t vocab_size,
                       uint32_t k)
{
    extern __shared__ float smem[];
    float  *s_val = smem;
    /* Index buffer starts after the full value buffer (blockDim.x * k). */
    uint32_t *s_idx = (uint32_t *)(s_val + (size_t)blockDim.x * k);

    uint32_t tid = threadIdx.x;
    uint32_t blockSize = blockDim.x;

    /* Each thread's local top-k arrays (live in registers / local mem). */
    float  local_val[CK_MAX_TOPK];
    uint32_t local_idx[CK_MAX_TOPK];

    /* Initialize local list to -inf. */
    for (uint32_t j = 0; j < k; j++) {
        local_val[j] = -INFINITY;
        local_idx[j] = 0u;
    }

    /* Scan strided slice of the vocab, insertion-sort into local top-k. */
    for (uint32_t i = tid; i < vocab_size; i += blockSize) {
        float v = logits[i];
        /* If v beats the smallest element in the local list, insert. */
        if (v > local_val[k - 1u]) {
            /* Find insertion position (descending order). */
            uint32_t pos = k - 1u;
            while (pos > 0u && v > local_val[pos - 1u]) {
                local_val[pos] = local_val[pos - 1u];
                local_idx[pos] = local_idx[pos - 1u];
                pos--;
            }
            local_val[pos] = v;
            local_idx[pos] = i;
        }
    }

    /* Stage thread-local top-k into shared memory (block-strided layout). */
    for (uint32_t j = 0; j < k; j++) {
        s_val[tid * k + j] = local_val[j];
        s_idx[tid * k + j] = local_idx[j];
    }
    __syncthreads();

    /* Block-level merge: take the top-k across all threads via successive
     * pairwise merges (bitonic-style). We do log2(blockSize) merge passes. */
    for (uint32_t step = 1; step < blockSize; step <<= 1) {
        uint32_t partner = tid ^ step;
        if (tid < partner) {
            /* Merge two sorted k-lists into one top-k. */
            float  merged_val[CK_MAX_TOPK];
            uint32_t merged_idx[CK_MAX_TOPK];
            uint32_t a = 0u, b = 0u, m = 0u;
            while (m < k && a < k && b < k) {
                float va = s_val[tid * k + a];
                float vb = s_val[partner * k + b];
                if (va >= vb) {
                    merged_val[m] = va;
                    merged_idx[m] = s_idx[tid * k + a];
                    a++;
                } else {
                    merged_val[m] = vb;
                    merged_idx[m] = s_idx[partner * k + b];
                    b++;
                }
                m++;
            }
            /* Drain remaining (only one of these loops runs). */
            while (m < k && a < k) {
                merged_val[m] = s_val[tid * k + a];
                merged_idx[m] = s_idx[tid * k + a];
                a++; m++;
            }
            while (m < k && b < k) {
                merged_val[m] = s_val[partner * k + b];
                merged_idx[m] = s_idx[partner * k + b];
                b++; m++;
            }
            for (uint32_t j = 0; j < k; j++) {
                s_val[tid * k + j] = merged_val[j];
                s_idx[tid * k + j] = merged_idx[j];
            }
        }
        __syncthreads();
    }

    /* Thread 0 writes the final top-k. */
    if (tid == 0) {
        for (uint32_t j = 0; j < k; j++) {
            out_val[j] = s_val[j];
            out_idx[j] = s_idx[j];
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Host-side wrappers (extern "C" — callable from the C11 forward path)
 * ════════════════════════════════════════════════════════════════════════════ */

#ifdef __cplusplus
extern "C" {
#endif

/* ─── 1. RMSNorm + RoPE fused wrapper ──────────────────────────────────────── */
bool oc_cuda_rmsnorm_rope_fused(const float *d_x, const float *d_weight,
                                float *d_out,
                                uint32_t hidden_dim, uint32_t n_heads,
                                uint32_t head_dim, uint32_t rope_dim,
                                int64_t pos, float theta,
                                float eps, float norm_scale)
{
    if (!d_x || !d_weight || !d_out || n_heads == 0u || head_dim == 0u)
        return false;
    if (hidden_dim != n_heads * head_dim) return false;
    if (rope_dim > head_dim) return false;

    /* One block per head, up to 256 threads. */
    uint32_t block = (head_dim < 256u) ? head_dim : 256u;
    /* Round up to the nearest warp (32). */
    block = ((block + 31u) / 32u) * 32u;
    size_t smem = block * sizeof(float);

    k_rmsnorm_rope_fused<<<n_heads, block, smem>>>(
        d_x, d_weight, d_out, hidden_dim, n_heads, head_dim,
        rope_dim, pos, theta, eps, norm_scale);
    CK_CUDA_CHECK(cudaGetLastError());
    CK_CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

/* ─── 2. SwiGLU activation wrapper ─────────────────────────────────────────── */
bool oc_cuda_swiglu(float *d_gate, const float *d_up, size_t n)
{
    if (!d_gate || !d_up || n == 0u) return false;

    uint32_t block = 256u;
    size_t grid = (n + block - 1u) / block;
    k_swiglu_fused<<<grid, block>>>(d_gate, d_up, n);
    CK_CUDA_CHECK(cudaGetLastError());
    CK_CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

/* ─── 3. Attention online softmax wrapper ──────────────────────────────────── */
bool oc_cuda_attention_softmax(const float *d_q, const float *d_k_cache,
                               const float *d_v_cache, float *d_out,
                               uint32_t head_dim, size_t n_past)
{
    if (!d_q || !d_k_cache || !d_v_cache || !d_out) return false;
    if (head_dim == 0u) return false;

    uint32_t block = 256u;
    if (head_dim > block) block = 1024u;
    /* Round to warp. */
    block = ((block + 31u) / 32u) * 32u;
    /* Shared mem: head_dim accumulators (score_buf reserved). */
    size_t smem = (size_t)head_dim * sizeof(float) * 2u;

    k_attention_softmax<<<1, block, smem>>>(
        d_q, d_k_cache, d_v_cache, d_out, head_dim, n_past);
    CK_CUDA_CHECK(cudaGetLastError());
    CK_CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

/* ─── 4. Q4_K quantized matvec wrapper ──────────────────────────────────────── */
bool oc_cuda_q4k_matvec(const void *d_weights, const float *d_x,
                        float *d_out, size_t rows, size_t cols)
{
    if (!d_weights || !d_x || !d_out || rows == 0u || cols == 0u)
        return false;
    if ((cols % CK_QK_K) != 0u) return false;  /* cols must be a multiple of 256 */

    uint32_t block = 256u;
    size_t smem = block * sizeof(float);

    k_q4k_matvec<<<rows, block, smem>>>(
        (const uint8_t *)d_weights, d_x, d_out, rows, cols);
    CK_CUDA_CHECK(cudaGetLastError());
    CK_CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

/* ─── 5. Q4_K → F32 dequantize wrapper ─────────────────────────────────────── */
bool oc_cuda_q4k_dequantize(const void *d_src, float *d_dst,
                            size_t n_blocks)
{
    if (!d_src || !d_dst || n_blocks == 0u) return false;

    uint32_t block = 256u;
    size_t grid = (n_blocks + block - 1u) / block;

    k_q4k_dequantize<<<grid, block>>>((const uint8_t *)d_src, d_dst, n_blocks);
    CK_CUDA_CHECK(cudaGetLastError());
    CK_CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

/* ─── 6. Embedding lookup wrapper ──────────────────────────────────────────── */
bool oc_cuda_embedding_lookup(const float *d_embeddings,
                              const uint32_t *d_tokens, float *d_out,
                              uint32_t vocab_size, uint32_t embd_dim,
                              size_t n_tokens)
{
    if (!d_embeddings || !d_tokens || !d_out || n_tokens == 0u)
        return false;
    if (vocab_size == 0u || embd_dim == 0u) return false;

    uint32_t block = 256u;
    if (embd_dim > block) block = 1024u;
    block = ((block + 31u) / 32u) * 32u;

    k_embed_lookup<<<n_tokens, block>>>(
        d_embeddings, d_tokens, d_out, vocab_size, embd_dim, n_tokens);
    CK_CUDA_CHECK(cudaGetLastError());
    CK_CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

/* ─── 7. Argmax wrapper ─────────────────────────────────────────────────────── */
bool oc_cuda_argmax(const float *d_logits, uint32_t *d_out_idx,
                    uint32_t vocab_size)
{
    if (!d_logits || !d_out_idx || vocab_size == 0u) return false;

    /* Use up to 1024 threads, one block. */
    uint32_t block = 1024u;
    if (vocab_size < block) {
        /* Round up to warp. */
        block = ((vocab_size + 31u) / 32u) * 32u;
        if (block == 0u) block = 32u;
    }
    /* Shared mem: block floats (values) + block uint32 (indices). */
    size_t smem = (size_t)block * sizeof(float) + (size_t)block * sizeof(uint32_t);

    k_argmax<<<1, block, smem>>>(d_logits, d_out_idx, vocab_size);
    CK_CUDA_CHECK(cudaGetLastError());
    CK_CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

/* ─── 8. Top-k partial sort wrapper ─────────────────────────────────────────── */
bool oc_cuda_topk(const float *d_logits, uint32_t *d_out_idx,
                  float *d_out_val, uint32_t vocab_size, uint32_t k)
{
    if (!d_logits || !d_out_idx || !d_out_val) return false;
    if (vocab_size == 0u || k == 0u) return false;
    if (k > CK_MAX_TOPK) return false;
    if (k > vocab_size) return false;  /* would fabricate entries */

    /* Pick the largest power-of-two block whose staging area
     * (blockSize * k * 8 bytes) fits in shared memory. */
    size_t per_thread = (size_t)k * (sizeof(float) + sizeof(uint32_t));
    uint32_t block = 1024u;
    while (block > 32u && (size_t)block * per_thread > CK_TOPK_SMEM_MAX)
        block >>= 1u;
    if ((size_t)block * per_thread > CK_TOPK_SMEM_MAX) return false;
    size_t smem = (size_t)block * per_thread;

    k_topk<<<1, block, smem>>>(d_logits, d_out_idx, d_out_val,
                                vocab_size, k);
    CK_CUDA_CHECK(cudaGetLastError());
    CK_CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
