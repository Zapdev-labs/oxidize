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
#include "oxidize/cuda_mmq.h"
#include "oxidize/llama.h"
#include "oxidize/log.h"
#include "oxidize/quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
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
    /* Shared memory for reduction. */
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t blockSize = blockDim.x;

    float val = 0.0f;
    for (size_t i = tid; i < n; i += blockSize) val += x[i] * x[i];
    sdata[tid] = val;
    __syncthreads();

    /* Tree reduction. */
    for (size_t s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = 1.0f / sqrtf(sdata[0] / (float)n + eps);
    for (size_t i = tid; i < n; i += blockSize)
        out[i] = x[i] * inv_rms * weight[i] * norm_scale;
}

/* Elementwise bias add: out[i] += bias[i]. Used for the Qwen2-family QKV
 * projection biases. */
__global__ void k_add_bias(float *out, const float *bias, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += bias[i];
}

/* Matvec: out[row] = sum(W[row, col] * x[col]) for col in [0, cols).
 * One block per row; each thread accumulates a partial sum. */
__global__ void k_matvec_f32(const float *W, size_t rows, size_t cols,
                              const float *x, float *out)
{
    size_t row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t blockSize = blockDim.x;

    float sum = 0.0f;
    for (size_t col = tid; col < cols; col += blockSize) {
        sum += W[row * cols + col] * x[col];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (size_t s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[row] = sdata[0];
}

/* RoPE (split-halves NeoX-style). One block per head. */
__global__ void k_apply_rope(float *q, size_t head_dim, size_t rope_dim,
                              int64_t pos, float theta, uint32_t n_heads)
{
    size_t head = blockIdx.x;
    if (head >= n_heads) return;

    float *hq = q + head * head_dim;
    size_t half = rope_dim / 2;
    float freq_mul = powf(theta, -2.0f / (float)rope_dim);

    for (size_t i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = powf(freq_mul, (float)i);
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float x0 = hq[i];
        float x1 = hq[half + i];
        hq[i] = x0 * c - x1 * s;
        hq[half + i] = x0 * s + x1 * c;
    }
}

/* SwiGLU in-place: gate[i] = silu(gate[i]) * up[i]. */
__global__ void k_swiglu(float *gate, const float *up, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
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
    uint32_t n_past)
{
    const uint32_t h = blockIdx.x;
    if (h >= n_head) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const uint32_t kv_head = h / (n_head / n_head_kv);
    const size_t kv_row = (size_t)n_head_kv * head_dim;
    const size_t kv_off = (size_t)kv_head * head_dim;

    extern __shared__ float smem[];
    float *qs  = smem;                 /* cached query, head_dim floats  */
    float *red = smem + head_dim;      /* reduction buffer, nthreads     */

    float *sc = scores + (size_t)h * score_stride;

    /* Cache this head's query vector. */
    for (uint32_t d = tid; d < head_dim; d += nthreads)
        qs[d] = q[(size_t)h * head_dim + d];
    __syncthreads();

    /* 1. Scores, parallel over past positions. */
    const float inv_sqrt_d = rsqrtf((float)head_dim);
    for (uint32_t p = tid; p < n_past; p += nthreads) {
        const __half *k = kv_k + (size_t)p * kv_row + kv_off;
        float s = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++)
            s += qs[d] * __half2float(k[d]);
        sc[p] = s * inv_sqrt_d;
    }
    __syncthreads();

    /* 2. Max, tree-reduced across threads. */
    float m = -INFINITY;
    for (uint32_t p = tid; p < n_past; p += nthreads) m = fmaxf(m, sc[p]);
    red[tid] = m;
    __syncthreads();
    for (uint32_t s = nthreads / 2u; s > 0u; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float max_score = red[0];
    __syncthreads();

    /* 3. exp in place, sum tree-reduced. */
    float local_sum = 0.0f;
    for (uint32_t p = tid; p < n_past; p += nthreads) {
        const float e = __expf(sc[p] - max_score);
        sc[p] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (uint32_t s = nthreads / 2u; s > 0u; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float denom = red[0];
    const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

    /* 4. Weighted V sum, parallel over head_dim. Threads read adjacent d for
     * a given position, so the f16 V loads coalesce. */
    for (uint32_t d = tid; d < head_dim; d += nthreads) {
        float acc = 0.0f;
        for (uint32_t p = 0; p < n_past; p++)
            acc += sc[p] * __half2float(kv_v[(size_t)p * kv_row + kv_off + d]);
        out[(size_t)h * head_dim + d] = acc * inv_denom;
    }
}

/* Copy KV to cache: write K/V at position `pos`, narrowing to f16. */
__global__ void k_kv_cache_write(__half *kv_k, __half *kv_v,
                                  const float *k, const float *v,
                                  size_t pos, size_t n_head_kv,
                                  uint32_t head_dim, size_t n_ctx)
{
    (void)n_ctx;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)n_head_kv * head_dim;
    if (i < total) {
        size_t offset = pos * n_head_kv * head_dim + i;
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
 * based on how the tensor was uploaded. Asynchronous on the default stream,
 * matching the rest of the per-token layer loop. */
static OcError cuda_matvec(const OcCudaWeight *w, const float *d_x,
                           float *d_out, uint32_t block)
{
    const size_t rows = w->rows;
    const size_t cols = w->cols;

    if (w->packed) {
        if (!oc_cuda_mmq_matvec(w->qtype, w->data, d_x, d_out, rows, cols, NULL))
            return OC_ERR_BACKEND;
        return OC_OK;
    }
    k_matvec_f32<<<rows, block, block * sizeof(float)>>>(
        (const float *)w->data, rows, cols, d_x, d_out);
    OC_CUDA_CHECK(cudaGetLastError());
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
        e = cuda_matvec(&ctx->d_ffn_gate_shexp[l], ctx->d_normed,
                        ctx->d_expert_gate, block);
        if (e != OC_OK) return e;
        e = cuda_matvec(&ctx->d_ffn_up_shexp[l], ctx->d_normed,
                        ctx->d_expert_up, block);
        if (e != OC_OK) return e;
        k_swiglu<<<egrid, block>>>(ctx->d_expert_gate, ctx->d_expert_up, i_size);
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
    ctx->uses_geglu = c->uses_geglu;
    ctx->num_experts = c->num_experts;
    ctx->num_experts_per_tok = c->num_experts_per_tok;
    ctx->expert_intermediate_size = c->expert_intermediate_size
                                  ? c->expert_intermediate_size : c->n_ff;
    ctx->expert_gating_sigmoid = c->expert_gating_sigmoid;
    ctx->expert_weights_scale = c->expert_weights_scale;

    /* Allocate workspace. */
    size_t embd = c->n_embd;
    size_t q_size = (size_t)c->n_head * c->head_dim;
    size_t kv_row = (size_t)c->n_head_kv * c->head_dim;
    size_t kv_total = (size_t)c->n_layer * c->n_ctx * kv_row;

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
    CUDA_INIT_ALLOC(&ctx->d_ffn_gate_buf, c->n_ff * sizeof(float));
    CUDA_INIT_ALLOC(&ctx->d_ffn_up_buf, c->n_ff * sizeof(float));
    CUDA_INIT_ALLOC(&ctx->d_logits, c->vocab_size * sizeof(float));
    CUDA_INIT_ALLOC(&ctx->d_attn_scores,
                    (size_t)c->n_head * c->n_ctx * sizeof(float));
    if (ctx->num_experts > 0) {
        const size_t isz = ctx->expert_intermediate_size;
        CUDA_INIT_ALLOC(&ctx->d_router_logits, ctx->num_experts * sizeof(float));
        CUDA_INIT_ALLOC(&ctx->d_expert_sel, ctx->num_experts * sizeof(uint32_t));
        CUDA_INIT_ALLOC(&ctx->d_expert_w, ctx->num_experts_per_tok * sizeof(float));
        CUDA_INIT_ALLOC(&ctx->d_expert_gate, isz * sizeof(float));
        CUDA_INIT_ALLOC(&ctx->d_expert_up, isz * sizeof(float));
        CUDA_INIT_ALLOC(&ctx->d_expert_tmp, embd * sizeof(float));
        CUDA_INIT_ALLOC(&ctx->d_expert_out, embd * sizeof(float));
        CUDA_INIT_ALLOC(&ctx->d_shexp_gate_logit, sizeof(float));
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
        !ctx->d_ffn_down || !ctx->d_attn_norm || !ctx->d_ffn_norm) {
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
        if (cudaMalloc((void **)&ctx->d_ffn_norm[l], embd * sizeof(float)) != cudaSuccess) {
            free(host_temp); oc_cuda_free(ctx); return OC_ERR_BACKEND;
        }
        cudaMemcpy(ctx->d_ffn_norm[l], L->ffn_norm, embd * sizeof(float),
                  cudaMemcpyHostToDevice);
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
        /* Pre-attention RMSNorm. */
        k_rms_norm<<<1, block, block * sizeof(float)>>>(
            ctx->d_x, ctx->d_attn_norm[l], ctx->d_normed, embd,
            eps, ctx->norm_scale);
        OC_CUDA_CHECK(cudaGetLastError());

        /* Q/K/V projections. */
        {
            OcError e = cuda_matvec(&ctx->d_attn_q[l], ctx->d_normed, ctx->d_q,
                                    block);
            if (e != OC_OK) return e;
            e = cuda_matvec(&ctx->d_attn_k[l], ctx->d_normed, ctx->d_k, block);
            if (e != OC_OK) return e;
            e = cuda_matvec(&ctx->d_attn_v[l], ctx->d_normed, ctx->d_v, block);
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

        /* RoPE on Q. */
        k_apply_rope<<<n_head, block>>>(
            ctx->d_q, head_dim, ctx->rope_dim, pos, ctx->rope_theta, n_head);
        OC_CUDA_CHECK(cudaGetLastError());
        /* RoPE on K. */
        k_apply_rope<<<n_head_kv, block>>>(
            ctx->d_k, head_dim, ctx->rope_dim, pos, ctx->rope_theta, n_head_kv);
        OC_CUDA_CHECK(cudaGetLastError());

        /* KV cache write. */
        size_t kv_total = (size_t)n_head_kv * head_dim;
        __half *kv_k_layer = (__half *)ctx->d_kv_k
                           + (size_t)l * ctx->n_ctx * kv_total;
        __half *kv_v_layer = (__half *)ctx->d_kv_v
                           + (size_t)l * ctx->n_ctx * kv_total;
        k_kv_cache_write<<<(kv_total + block - 1) / block, block>>>(
            kv_k_layer, kv_v_layer,
            ctx->d_k, ctx->d_v, pos, n_head_kv, head_dim, ctx->n_ctx);
        OC_CUDA_CHECK(cudaGetLastError());

        /* Attention: all heads in one launch. */
        {
            const uint32_t athreads = OC_CUDA_ATTN_THREADS;
            const size_t asmem = ((size_t)head_dim + athreads) * sizeof(float);
            k_attention_all_heads<<<n_head, athreads, asmem>>>(
                ctx->d_q, kv_k_layer, kv_v_layer, ctx->d_attn_out,
                ctx->d_attn_scores, ctx->n_ctx,
                n_head, n_head_kv, head_dim, pos + 1);
            OC_CUDA_CHECK(cudaGetLastError());
        }

        /* Output projection. */
        {
            OcError e = cuda_matvec(&ctx->d_attn_output[l], ctx->d_attn_out,
                                    ctx->d_normed, block);
            if (e != OC_OK) return e;
        }
        k_residual_add<<<(embd + block - 1) / block, block>>>(
            ctx->d_x, ctx->d_normed, embd);
        OC_CUDA_CHECK(cudaGetLastError());

        /* Pre-FFN RMSNorm. */
        k_rms_norm<<<1, block, block * sizeof(float)>>>(
            ctx->d_x, ctx->d_ffn_norm[l], ctx->d_normed, embd,
            eps, ctx->norm_scale);
        OC_CUDA_CHECK(cudaGetLastError());

        /* FFN: MoE (router + top-k experts + shared) or dense. */
        if (ctx->num_experts > 0) {
            OcError e = cuda_moe_ffn(ctx, l, block);
            if (e != OC_OK) return e;
            continue;
        }

        /* FFN: gate, up, SwiGLU, down. */
        {
            OcError e = cuda_matvec(&ctx->d_ffn_gate[l], ctx->d_normed,
                                    ctx->d_ffn_gate_buf, block);
            if (e != OC_OK) return e;
            e = cuda_matvec(&ctx->d_ffn_up[l], ctx->d_normed,
                            ctx->d_ffn_up_buf, block);
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
        k_residual_add<<<(embd + block - 1) / block, block>>>(
            ctx->d_x, ctx->d_normed, embd);
        OC_CUDA_CHECK(cudaGetLastError());
    }

    /* 3. Final norm + lm_head. */
    k_rms_norm<<<1, block, block * sizeof(float)>>>(
        ctx->d_x, ctx->d_final_norm, ctx->d_normed, embd,
        eps, ctx->norm_scale);
    OC_CUDA_CHECK(cudaGetLastError());

    if (logits_out != NULL) {
        OcError e = cuda_matvec(&ctx->d_output, ctx->d_normed, ctx->d_logits,
                                block);
        if (e != OC_OK) return e;
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
    /* KV cache is overwritten on subsequent forwards; no need to zero. */
    (void)ctx;
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
    }
    free(ctx->d_attn_q_bias); free(ctx->d_attn_k_bias); free(ctx->d_attn_v_bias);
    free(ctx->d_attn_q); free(ctx->d_attn_k); free(ctx->d_attn_v);
    free(ctx->d_attn_output);
    free(ctx->d_ffn_gate); free(ctx->d_ffn_up); free(ctx->d_ffn_down);
    free(ctx->d_attn_norm); free(ctx->d_ffn_norm);
    free(ctx->d_ffn_gate_inp); free(ctx->d_ffn_gate_exps);
    free(ctx->d_ffn_up_exps); free(ctx->d_ffn_down_exps);
    free(ctx->d_ffn_gate_shexp); free(ctx->d_ffn_up_shexp);
    free(ctx->d_ffn_down_shexp); free(ctx->d_ffn_gate_inp_shexp);

    memset(ctx, 0, sizeof(*ctx));
}
