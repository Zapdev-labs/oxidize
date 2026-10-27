/*
 * cuda_fast.cu — fully-asynchronous forward path for dense llama-family
 * models.
 *
 * The legacy per-token forward (cuda.cu::oc_cuda_forward) submits ~400
 * kernels through the synchronizing default stream and drains three compute
 * streams twice per layer — on the order of sixty host synchronizations plus
 * a vocab-sized logits copy per token. On a virtualized GPU the launch and
 * sync latency dominates the actual bandwidth-bound work by nearly an order
 * of magnitude.
 *
 * This file removes the host from the loop entirely:
 *
 *   - A 16-byte state {token, pos, step} lives in device memory. Every
 *     dynamic value a kernel needs — the token to embed, the absolute
 *     position for RoPE / KV-cache addressing / attention length, the ring
 *     slot for the emitted token — is read from it at execution time.
 *   - One non-blocking stream carries the whole sequence. A decode step is
 *     a fixed kernel chain (embed -> 28 layers -> final norm -> lm_head ->
 *     argmax), and the argmax kernel itself writes the winning token back
 *     into the state and bumps pos/step, so step N+1 needs nothing from the
 *     host. Greedy decode becomes a self-driving device program.
 *   - Because every kernel's parameters are static pointers, the step can be
 *     captured once into a CUDA graph and replayed per token, which cuts
 *     host launch overhead to a single cudaGraphLaunch and shrinks the
 *     GPU-side inter-kernel gaps to sub-microsecond graph-node latency.
 *   - The prompt is batched: oc_cuda_mmq_matvec_batch dots each weight row
 *     against up to FAST_BATCH activation vectors per fetch, so prefill
 *     pays ~one weight sweep per chunk instead of per token.
 *
 * The emitted tokens land in a device ring buffer the host copies back in
 * chunks — no logits ever round-trip for greedy sampling.
 *
 * Gate (oc_cuda_fast_supported): dense decoder only — no MoE, Gemma-4,
 * Qwen3.5, GeGLU, logit softcap, per-layer geometry, or post-norms — and
 * every tensor used must be a packed MMQ weight on the device.
 */
#include "oxidize/cuda.h"
#include "oxidize/cuda_mmq.h"
#include "oxidize/log.h"
#include "oxidize/quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define FAST_BLOCK         256u
#define FAST_ATTN_THREADS  128u
#define FAST_ATTN_SMEM_SCORE_CAP 4096u
#define FAST_RING_CAP      8192u
#define FAST_BATCH         64u

/* Device-resident step state. token/pos are consumed by embed/rope/KV/
 * attention; step indexes the emitted-token ring. Written only by the
 * argmax kernel (decode), k_f_set (prefill advance / sampled feed), or a
 * host memcpy at seed time. */
typedef struct {
    uint32_t token;
    uint32_t pos;
    uint32_t step;
    uint32_t pad;
} OcFastState;

/* ─── Reductions ─────────────────────────────────────────────────────────── */

__device__ __forceinline__ float f_warp_sum(float v)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float f_warp_max(float v)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ float f_block_sum(float v, float *red)
{
    v = f_warp_sum(v);
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

__device__ __forceinline__ float f_block_max(float v, float *red)
{
    v = f_warp_max(v);
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

/* ─── State kernels ──────────────────────────────────────────────────────── */

/* Prefill advance / sampled feed: adopt `token`, step the position. */
__global__ void k_f_set(OcFastState *st, uint32_t token)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        st->token = token;
        st->pos++;
    }
}

/* Rewind the ring cursor at the head of each decode submission so emitted
 * tokens always land at ring[0..n). */
__global__ void k_f_reset_step(OcFastState *st)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u) st->step = 0u;
}

/* ─── Embed (f32 table fallback; packed goes through mmq getrow) ─────────── */

__global__ void k_f_embed_f32(const float *emb, const OcFastState *st,
                              float *out, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = emb[(size_t)st->token * n + i];
}

__global__ void k_f_embed_f32_batch(const float *emb, const uint32_t *ids,
                                    float *out, size_t n, size_t stride)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[(size_t)blockIdx.y * stride + i] =
        emb[(size_t)ids[blockIdx.y] * n + i];
}

/* ─── Elementwise / norm kernels (grid.y = token index, stride = row) ────── */

/* out[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i] * scale, per token row. */
__global__ void k_f_rmsnorm(const float *x, const float *w, float *out,
                            size_t n, float eps, float scale, size_t stride)
{
    const float *xr = x + (size_t)blockIdx.y * stride;
    float *orow = out + (size_t)blockIdx.y * stride;
    extern __shared__ float red[];
    const uint32_t tid = threadIdx.x;
    float acc = 0.0f;
    for (size_t i = tid; i < n; i += blockDim.x) acc += xr[i] * xr[i];
    const float inv = rsqrtf(f_block_sum(acc, red) / (float)n + eps) * scale;
    for (size_t i = tid; i < n; i += blockDim.x)
        orow[i] = xr[i] * inv * w[i];
}

/* x += add, then out = rmsnorm(x)*w. Aliased out==x is safe: the sum is
 * consumed into the reduction before out is written. */
__global__ void k_f_residual_rms(float *x, const float *add, const float *w,
                                 float *out, size_t n, float eps,
                                 float scale, size_t stride)
{
    float *xr = x + (size_t)blockIdx.y * stride;
    const float *ar = add + (size_t)blockIdx.y * stride;
    float *orow = out + (size_t)blockIdx.y * stride;
    extern __shared__ float red[];
    const uint32_t tid = threadIdx.x;
    float acc = 0.0f;
    for (size_t i = tid; i < n; i += blockDim.x) {
        const float v = xr[i] + ar[i];
        xr[i] = v;
        acc += v * v;
    }
    const float inv = rsqrtf(f_block_sum(acc, red) / (float)n + eps) * scale;
    for (size_t i = tid; i < n; i += blockDim.x)
        orow[i] = xr[i] * inv * w[i];
}

/* x[i] += y[i] over the whole [n_tok, n] tile. */
__global__ void k_f_resadd(float *x, const float *y, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

/* ─── Fused per-head [bias] -> [RMSNorm] -> RoPE over Q and K ─────────────
 *
 * One launch covers every Q and KV head of every token in the batch:
 * blockIdx.x in [0, n_head + n_kv) picks the head, blockIdx.y the token row.
 * The Qwen2-family q/k projection biases fold in here (they precede the norm
 * and rotation anyway); the v bias rides along in k_f_kvwrite below.
 *
 * pos comes from device state in decode mode (st != NULL, blockIdx.y == 0)
 * or pos0 + blockIdx.y in batched-prefill mode (st == NULL). */
__global__ void k_f_rope_qk(float *q, float *k,
                            const float *qb, const float *kb,
                            const float *qn, const float *kn,
                            uint32_t n_head, uint32_t n_kv,
                            uint32_t head_dim, uint32_t rope_dim,
                            float theta, float eps,
                            float yarn_factor, uint32_t yarn_orig_ctx,
                            uint32_t pos0, const OcFastState *st,
                            size_t tq, size_t tk)
{
    uint32_t b = blockIdx.x;
    const uint32_t t = blockIdx.y;
    const int64_t pos = st ? (int64_t)st->pos : (int64_t)(pos0 + t);

    float *h;
    const float *bias, *norm;
    if (b < n_head) {
        h = q + t * tq + (size_t)b * head_dim;
        bias = qb ? qb + (size_t)b * head_dim : NULL;
        norm = qn;
    } else {
        b -= n_head;
        h = k + t * tk + (size_t)b * head_dim;
        bias = kb ? kb + (size_t)b * head_dim : NULL;
        norm = kn;
    }

    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    extern __shared__ float red[];

    if (bias)
        for (uint32_t i = tid; i < head_dim; i += nthreads) h[i] += bias[i];
    __syncthreads();

    if (norm) {
        float sum = 0.0f;
        for (uint32_t i = tid; i < head_dim; i += nthreads)
            sum += h[i] * h[i];
        const float inv = rsqrtf(f_block_sum(sum, red) / (float)head_dim + eps);
        for (uint32_t i = tid; i < head_dim; i += nthreads)
            h[i] = h[i] * inv * norm[i];
        __syncthreads();
    }

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
        const float freq = powf(freq_mul, (float)i);
        float angle = (float)pos * freq;
        if (yarn) {
            if (pos == 0) {
                h[i] *= mscale;
                h[half + i] *= mscale;
                continue;
            }
            const float theta_extrap = (float)pos * freq;
            const float theta_interp = theta_extrap / yarn_factor;
            float ramp = 1.0f - ((float)i - corr_lo) / denom;
            if (ramp < 0.0f) ramp = 0.0f;
            if (ramp > 1.0f) ramp = 1.0f;
            angle = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
        }
        const float c = cosf(angle) * mscale;
        const float s = sinf(angle) * mscale;
        const float x0 = h[i];
        const float x1 = h[half + i];
        h[i] = x0 * c - x1 * s;
        h[half + i] = x0 * s + x1 * c;
    }
}

/* K/V cache write for a whole token row, folding in the optional v bias.
 * grid.x covers n_kv*head_dim elements, grid.y the token index. */
__global__ void k_f_kvwrite(__half *kk, __half *vv,
                            const float *k, const float *v,
                            const float *vbias,
                            size_t kv_row, uint32_t n,
                            uint32_t pos0, const OcFastState *st,
                            size_t stride)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t pos = st ? st->pos : pos0 + blockIdx.y;
    const size_t off = (size_t)pos * kv_row + i;
    kk[off] = __float2half(k[(size_t)blockIdx.y * stride + i]);
    vv[off] = __float2half(v[(size_t)blockIdx.y * stride + i]
                           + (vbias ? vbias[i] : 0.0f));
}

/* ─── Attention, all heads × all tokens, one launch ───────────────────────
 *
 * Same structure as cuda.cu's k_attention_all_heads: per-head scratch row in
 * `scores` (stride score_stride), shared-memory query + reductions, f16 KV
 * cache. n_past is state.pos + 1 in decode or pos0 + t + 1 in prefill. */
__global__ void k_f_attention(const float *q,
                              const __half *kv_k, const __half *kv_v,
                              float *out, float *scores,
                              size_t score_stride,
                              uint32_t n_head, uint32_t n_kv,
                              uint32_t head_dim, float scale,
                              size_t kv_row,
                              uint32_t pos0, const OcFastState *st,
                              size_t tq, size_t to)
{
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    if (h >= n_head) return;

    const uint32_t n_past = st ? (st->pos + 1u) : (pos0 + t + 1u);
    const uint32_t kv_start = 0u;

    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const uint32_t kv_head = h / (n_head / n_kv);
    const size_t kv_off = (size_t)kv_head * head_dim;

    extern __shared__ float smem[];
    float *qs = smem;
    float *red = smem + head_dim;

    float *sc = scores + ((size_t)t * n_head + h) * score_stride;
    float *ss = (score_stride <= FAST_ATTN_SMEM_SCORE_CAP)
              ? (red + nthreads) : sc;
    float *orow = out + t * to + (size_t)h * head_dim;
    const float *qh = q + t * tq + (size_t)h * head_dim;

    for (uint32_t d = tid; d < head_dim; d += nthreads) qs[d] = qh[d];
    __syncthreads();

    const bool pair = ((head_dim & 1u) == 0u) && ((kv_off & 1u) == 0u);
    /* scores: one warp per position so lanes read each K row coalesced,
     * then a warp reduction produces the dot product. */
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t nw = nthreads >> 5;
    for (uint32_t p = kv_start + warp; p < n_past; p += nw) {
        const __half *k = kv_k + (size_t)p * kv_row + kv_off;
        float s = 0.0f;
        if (pair) {
            const __half2 *k2 = reinterpret_cast<const __half2 *>(k);
            for (uint32_t d = lane; d < head_dim / 2u; d += 32u) {
                const float2 kv = __half22float2(__ldg(&k2[d]));
                s += qs[2u * d] * kv.x + qs[2u * d + 1u] * kv.y;
            }
        } else {
            for (uint32_t d = lane; d < head_dim; d += 32u)
                s += qs[d] * __half2float(__ldg(&k[d]));
        }
        s = f_warp_sum(s);
        if (lane == 0u) {
            const float sv = s * scale;
            sc[p] = sv;
            ss[p] = sv;
        }
    }
    __syncthreads();

    float m = -INFINITY;
    for (uint32_t p = kv_start + tid; p < n_past; p += nthreads)
        m = fmaxf(m, ss[p]);
    const float max_score = f_block_max(m, red);
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t p = kv_start + tid; p < n_past; p += nthreads) {
        const float e = __expf(ss[p] - max_score);
        ss[p] = e;
        local_sum += e;
    }
    const float denom = f_block_sum(local_sum, red);
    const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

    if (pair) {
        const size_t vs = (size_t)kv_row >> 1;
        for (uint32_t d2 = tid; d2 < head_dim / 2u; d2 += nthreads) {
            float acc0 = 0.0f, acc1 = 0.0f;
            const __half2 *vb = reinterpret_cast<const __half2 *>(
                kv_v + kv_off) + (size_t)d2;
            uint32_t p = kv_start;
            for (; p + 4u <= n_past; p += 4u) {
                const __half2 v0 = __ldg(vb + (size_t)p * vs);
                const __half2 v1 = __ldg(vb + (size_t)(p + 1u) * vs);
                const __half2 v2 = __ldg(vb + (size_t)(p + 2u) * vs);
                const __half2 v3 = __ldg(vb + (size_t)(p + 3u) * vs);
                const float2 f0 = __half22float2(v0);
                const float2 f1 = __half22float2(v1);
                const float2 f2 = __half22float2(v2);
                const float2 f3 = __half22float2(v3);
                acc0 += ss[p] * f0.x + ss[p + 1u] * f1.x
                      + ss[p + 2u] * f2.x + ss[p + 3u] * f3.x;
                acc1 += ss[p] * f0.y + ss[p + 1u] * f1.y
                      + ss[p + 2u] * f2.y + ss[p + 3u] * f3.y;
            }
            for (; p < n_past; p++) {
                const float2 fv = __half22float2(
                    __ldg(vb + (size_t)p * vs));
                acc0 += ss[p] * fv.x;
                acc1 += ss[p] * fv.y;
            }
            orow[(size_t)d2 * 2u]     = acc0 * inv_denom;
            orow[(size_t)d2 * 2u + 1u] = acc1 * inv_denom;
        }
    } else {
        for (uint32_t d = tid; d < head_dim; d += nthreads) {
            float acc = 0.0f;
            const __half *vb = kv_v + kv_off + d;
            uint32_t p = kv_start;
            for (; p + 4u <= n_past; p += 4u) {
                const float v0 = __half2float(__ldg(vb + (size_t)p * kv_row));
                const float v1 = __half2float(
                    __ldg(vb + (size_t)(p + 1u) * kv_row));
                const float v2 = __half2float(
                    __ldg(vb + (size_t)(p + 2u) * kv_row));
                const float v3 = __half2float(
                    __ldg(vb + (size_t)(p + 3u) * kv_row));
                acc += ss[p] * v0 + ss[p + 1u] * v1
                     + ss[p + 2u] * v2 + ss[p + 3u] * v3;
            }
            for (; p < n_past; p++)
                acc += ss[p] * __half2float(
                    __ldg(vb + (size_t)p * kv_row));
            orow[d] = acc * inv_denom;
        }
    }
}

/* ─── SwiGLU over the whole [n_tok, n_ff] tile ───────────────────────────── */

__global__ void k_f_swiglu(float *gate, const float *up, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        const float s = 1.0f / (1.0f + expf(-g));
        gate[i] = g * s * up[i];
    }
}

/* ─── Argmax -> ring + state advance ──────────────────────────────────────
 *
 * Single block, warp-shuffle argmax over the vocab. Strictly-greater
 * comparisons keep the first maximum's index, matching oc_argmax. The
 * winning token is appended to the emitted-token ring AND written back into
 * the state — the next decode step embeds it without any host involvement.
 * pos and step advance as a side effect, which is what makes the whole
 * decode step a static kernel sequence. */
__global__ void k_f_argmax(const float *logits, uint32_t n,
                           uint32_t *ring, OcFastState *st)
{
    extern __shared__ float sred[];
    uint32_t *ired = (uint32_t *)(sred + blockDim.x);

    const uint32_t tid = threadIdx.x;
    float bv = -INFINITY;
    uint32_t bi = 0u;
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        const float v = logits[i];
        if (v > bv) { bv = v; bi = i; }
    }
    /* Warp reduce: larger value wins, smaller index wins ties. */
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const uint32_t oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    if (lane == 0u) { sred[warp] = bv; ired[warp] = bi; }
    __syncthreads();
    if (tid == 0u) {
        float m = sred[0];
        uint32_t idx = ired[0];
        const uint32_t nwarps = blockDim.x >> 5;
        for (uint32_t w = 1u; w < nwarps; w++) {
            if (sred[w] > m || (sred[w] == m && ired[w] < idx)) {
                m = sred[w];
                idx = ired[w];
            }
        }
        const uint32_t slot = st->step;
        ring[slot] = idx;
        st->token = idx;
        st->pos++;
        st->step = slot + 1u;
    }
}

/* ─── Launch helpers ─────────────────────────────────────────────────────── */

static cudaStream_t fast_stream(OcCudaContext *ctx)
{
    return (cudaStream_t)ctx->fast_stream;
}

/* Quantize an activation tile into the shared xq scratch. Every matvec that
 * consumes this tile must be submitted before the next fast_qx — the
 * scratch is single-buffered, which is safe because stream order is also
 * program order on the fast stream. */
static void fast_qx(OcCudaContext *ctx, const float *d_x, size_t cols,
                    uint32_t n_tok, cudaStream_t s)
{
    oc_cuda_mmq_quantize_x(d_x, ctx->d_xq, ctx->d_xs,
                           ctx->d_xqsum, ctx->d_xh, n_tok, cols, cols, s);
}

/* One matvec through the dp4a path when the weight type supports it (the
 * activation must already be in the xq scratch), else the f32 mmq path —
 * single-token or batch. d_x is only read by the f32 fallback. */
static void fast_mv(OcCudaContext *ctx, const OcCudaWeight *w,
                    const float *d_x, size_t x_stride, float *d_out,
                    size_t out_stride, uint32_t n_tok, cudaStream_t s)
{
    if (oc_cuda_mmq_xq_supported(w->qtype, w->cols)) {
        oc_cuda_mmq_matvec_xq(w->qtype, w->data, ctx->d_xq, ctx->d_xs,
                              ctx->d_xqsum, ctx->d_xh, d_out, n_tok,
                              out_stride, w->rows, w->cols, s);
    } else if (n_tok == 1u) {
        oc_cuda_mmq_matvec(w->qtype, w->data, d_x, d_out,
                           w->rows, w->cols, s);
    } else {
        oc_cuda_mmq_matvec_batch(w->qtype, w->data, d_x, d_out, n_tok,
                                 x_stride, out_stride, w->rows, w->cols, s);
    }
}

/* Submit one decode forward for the single-token pipeline: embed the token
 * in state, run all layers. lm_head/argmax come separately so the same body
 * works for prefill's no-logits steps and the sampled path's feed->step. */
static void fast_submit_body(OcCudaContext *ctx, cudaStream_t s)
{
    const OcFastState *st = (const OcFastState *)ctx->d_fast_state;
    const uint32_t embd = ctx->n_embd;
    const uint32_t hd = ctx->head_dim;
    const uint32_t nh = ctx->n_head;
    const uint32_t nkv = ctx->n_head_kv;
    const uint32_t n_ff = ctx->n_ff;
    const uint32_t qd = nh * hd;
    const float eps = ctx->rms_norm_eps;
    const uint32_t block = FAST_BLOCK;

    /* Embed. */
    const OcCudaWeight *emb = &ctx->d_tok_embeddings;
    if (emb->packed) {
        oc_cuda_mmq_get_row_dev(emb->qtype, emb->data,
                                (const uint32_t *)ctx->d_fast_state,
                                ctx->d_x, emb->cols, s);
    } else {
        k_f_embed_f32<<<(embd + block - 1) / block, block, 0, s>>>(
            (const float *)emb->data, st, ctx->d_x, embd);
    }

    for (uint32_t l = 0; l < ctx->n_layer; l++) {
        /* attn_norm -> normed. */
        k_f_rmsnorm<<<dim3(1, 1), block, block * sizeof(float), s>>>(
            ctx->d_x, ctx->d_attn_norm[l], ctx->d_normed, embd, eps,
            ctx->norm_scale, 0);

        /* Q/K/V projections: one x-quantize feeds all three matvecs. */
        fast_qx(ctx, ctx->d_normed, embd, 1u, s);
        fast_mv(ctx, &ctx->d_attn_q[l], ctx->d_normed, embd,
                ctx->d_q, 0, 1u, s);
        fast_mv(ctx, &ctx->d_attn_k[l], ctx->d_normed, embd,
                ctx->d_k, 0, 1u, s);
        fast_mv(ctx, &ctx->d_attn_v[l], ctx->d_normed, embd,
                ctx->d_v, 0, 1u, s);

        /* Per-head [bias ->] [norm ->] RoPE on Q and K in one launch. */
        {
            dim3 grid(nh + nkv, 1);
            k_f_rope_qk<<<grid, block, block * sizeof(float), s>>>(
                ctx->d_q, ctx->d_k,
                ctx->d_attn_q_bias ? ctx->d_attn_q_bias[l] : NULL,
                ctx->d_attn_k_bias ? ctx->d_attn_k_bias[l] : NULL,
                ctx->d_attn_q_norm ? ctx->d_attn_q_norm[l] : NULL,
                ctx->d_attn_k_norm ? ctx->d_attn_k_norm[l] : NULL,
                nh, nkv, hd, ctx->rope_dim, ctx->rope_theta, eps,
                ctx->yarn_factor, ctx->yarn_orig_ctx,
                0u, st, 0, 0);
        }

        /* KV cache write (folds the v bias). */
        {
            const uint32_t kv_n = nkv * hd;
            __half *kkl = (__half *)ctx->d_kv_k
                        + (size_t)l * ctx->n_ctx * ctx->kv_row;
            __half *vvl = (__half *)ctx->d_kv_v
                        + (size_t)l * ctx->n_ctx * ctx->kv_row;
            dim3 grid((kv_n + block - 1) / block, 1);
            k_f_kvwrite<<<grid, block, 0, s>>>(
                kkl, vvl, ctx->d_k, ctx->d_v,
                ctx->d_attn_v_bias ? ctx->d_attn_v_bias[l] : NULL,
                ctx->kv_row, kv_n, 0u, st, 0);
        }

        /* Attention, all heads, n_past from state. */
        {
            __half *kkl = (__half *)ctx->d_kv_k
                        + (size_t)l * ctx->n_ctx * ctx->kv_row;
            __half *vvl = (__half *)ctx->d_kv_v
                        + (size_t)l * ctx->n_ctx * ctx->kv_row;
            const size_t nsc = ctx->n_ctx <= FAST_ATTN_SMEM_SCORE_CAP
                             ? ctx->n_ctx : 0u;
            const size_t asmem = ((size_t)hd + FAST_ATTN_THREADS + nsc)
                               * sizeof(float);
            dim3 grid(nh, 1);
            k_f_attention<<<grid, FAST_ATTN_THREADS, asmem, s>>>(
                ctx->d_q, kkl, vvl, ctx->d_attn_out, ctx->d_attn_scores,
                ctx->n_ctx, nh, nkv, hd,
                ctx->attn_scale > 0.0f ? ctx->attn_scale
                                       : rsqrtf((float)hd),
                ctx->kv_row, 0u, st, 0, 0);
        }

        /* O projection (input is the qd-wide attention output), then
         * residual + pre-FFN norm. */
        fast_qx(ctx, ctx->d_attn_out, qd, 1u, s);
        fast_mv(ctx, &ctx->d_attn_output[l], ctx->d_attn_out, qd,
                ctx->d_normed, 0, 1u, s);
        k_f_residual_rms<<<dim3(1, 1), block, block * sizeof(float), s>>>(
            ctx->d_x, ctx->d_normed, ctx->d_ffn_norm[l], ctx->d_normed,
            embd, eps, ctx->norm_scale, 0);

        /* FFN: gate/up share the normed input, down reads swiglu out. */
        fast_qx(ctx, ctx->d_normed, embd, 1u, s);
        fast_mv(ctx, &ctx->d_ffn_gate[l], ctx->d_normed, embd,
                ctx->d_ffn_gate_buf, 0, 1u, s);
        fast_mv(ctx, &ctx->d_ffn_up[l], ctx->d_normed, embd,
                ctx->d_ffn_up_buf, 0, 1u, s);
        k_f_swiglu<<<(n_ff + block - 1) / block, block, 0, s>>>(
            ctx->d_ffn_gate_buf, ctx->d_ffn_up_buf, n_ff);
        fast_qx(ctx, ctx->d_ffn_gate_buf, n_ff, 1u, s);
        fast_mv(ctx, &ctx->d_ffn_down[l], ctx->d_ffn_gate_buf, n_ff,
                ctx->d_normed, 0, 1u, s);
        k_f_resadd<<<(embd + block - 1) / block, block, 0, s>>>(
            ctx->d_x, ctx->d_normed, embd);
    }
}

/* Final norm -> lm_head -> optional argmax. */
static void fast_submit_head(OcCudaContext *ctx, cudaStream_t s, bool argmax)
{
    const uint32_t embd = ctx->n_embd;
    const uint32_t block = FAST_BLOCK;
    k_f_rmsnorm<<<dim3(1, 1), block, block * sizeof(float), s>>>(
        ctx->d_x, ctx->d_final_norm, ctx->d_normed, embd,
        ctx->rms_norm_eps, ctx->norm_scale, 0);
    fast_qx(ctx, ctx->d_normed, embd, 1u, s);
    fast_mv(ctx, &ctx->d_output, ctx->d_normed, embd,
            ctx->d_logits, 0, 1u, s);
    if (argmax) {
        const uint32_t ab = 1024u;
        const size_t smem = ab * (sizeof(float) + sizeof(uint32_t));
        k_f_argmax<<<1, ab, smem, s>>>(ctx->d_logits, ctx->vocab_size,
                                       ctx->d_fast_ring,
                                       (OcFastState *)ctx->d_fast_state);
    }
}

/* ─── Batched prefill submission ─────────────────────────────────────────── */

/* One layer over a [n_tok, embd] activation tile. */
static void fast_submit_batch_layer(OcCudaContext *ctx, uint32_t l,
                                    uint32_t n_tok, uint32_t pos0,
                                    cudaStream_t s)
{
    const uint32_t embd = ctx->n_embd;
    const uint32_t hd = ctx->head_dim;
    const uint32_t nh = ctx->n_head;
    const uint32_t nkv = ctx->n_head_kv;
    const uint32_t n_ff = ctx->n_ff;
    const float eps = ctx->rms_norm_eps;
    const uint32_t block = FAST_BLOCK;
    const uint32_t qd = nh * hd, kd = nkv * hd;

    k_f_rmsnorm<<<dim3(1, n_tok), block, block * sizeof(float), s>>>(
        ctx->d_bx, ctx->d_attn_norm[l], ctx->d_bnormed, embd, eps,
        ctx->norm_scale, embd);

    fast_qx(ctx, ctx->d_bnormed, embd, n_tok, s);
    fast_mv(ctx, &ctx->d_attn_q[l], ctx->d_bnormed, embd,
            ctx->d_bq, qd, n_tok, s);
    fast_mv(ctx, &ctx->d_attn_k[l], ctx->d_bnormed, embd,
            ctx->d_bk, kd, n_tok, s);
    fast_mv(ctx, &ctx->d_attn_v[l], ctx->d_bnormed, embd,
            ctx->d_bv, kd, n_tok, s);

    {
        dim3 grid(nh + nkv, n_tok);
        k_f_rope_qk<<<grid, block, block * sizeof(float), s>>>(
            ctx->d_bq, ctx->d_bk,
            ctx->d_attn_q_bias ? ctx->d_attn_q_bias[l] : NULL,
            ctx->d_attn_k_bias ? ctx->d_attn_k_bias[l] : NULL,
            ctx->d_attn_q_norm ? ctx->d_attn_q_norm[l] : NULL,
            ctx->d_attn_k_norm ? ctx->d_attn_k_norm[l] : NULL,
            nh, nkv, hd, ctx->rope_dim, ctx->rope_theta, eps,
            ctx->yarn_factor, ctx->yarn_orig_ctx,
            pos0, NULL, qd, kd);
    }

    {
        const uint32_t kv_n = nkv * hd;
        __half *kkl = (__half *)ctx->d_kv_k
                    + (size_t)l * ctx->n_ctx * ctx->kv_row;
        __half *vvl = (__half *)ctx->d_kv_v
                    + (size_t)l * ctx->n_ctx * ctx->kv_row;
        dim3 grid((kv_n + block - 1) / block, n_tok);
        k_f_kvwrite<<<grid, block, 0, s>>>(
            kkl, vvl, ctx->d_bk, ctx->d_bv,
            ctx->d_attn_v_bias ? ctx->d_attn_v_bias[l] : NULL,
            ctx->kv_row, kv_n, pos0, NULL, kd);
    }

    {
        __half *kkl = (__half *)ctx->d_kv_k
                    + (size_t)l * ctx->n_ctx * ctx->kv_row;
        __half *vvl = (__half *)ctx->d_kv_v
                    + (size_t)l * ctx->n_ctx * ctx->kv_row;
        const size_t nsc = ctx->n_ctx <= FAST_ATTN_SMEM_SCORE_CAP
                         ? ctx->n_ctx : 0u;
        const size_t asmem = ((size_t)hd + FAST_ATTN_THREADS + nsc)
                           * sizeof(float);
        dim3 grid(nh, n_tok);
        k_f_attention<<<grid, FAST_ATTN_THREADS, asmem, s>>>(
            ctx->d_bq, kkl, vvl, ctx->d_battn, ctx->d_bscores,
            ctx->n_ctx, nh, nkv, hd,
            ctx->attn_scale > 0.0f ? ctx->attn_scale : rsqrtf((float)hd),
            ctx->kv_row, pos0, NULL, qd, qd);
    }

    fast_qx(ctx, ctx->d_battn, qd, n_tok, s);
    fast_mv(ctx, &ctx->d_attn_output[l], ctx->d_battn, qd,
            ctx->d_bnormed, embd, n_tok, s);
    k_f_residual_rms<<<dim3(1, n_tok), block, block * sizeof(float), s>>>(
        ctx->d_bx, ctx->d_bnormed, ctx->d_ffn_norm[l], ctx->d_bnormed,
        embd, eps, ctx->norm_scale, embd);

    fast_qx(ctx, ctx->d_bnormed, embd, n_tok, s);
    fast_mv(ctx, &ctx->d_ffn_gate[l], ctx->d_bnormed, embd,
            ctx->d_bg, n_ff, n_tok, s);
    fast_mv(ctx, &ctx->d_ffn_up[l], ctx->d_bnormed, embd,
            ctx->d_bu, n_ff, n_tok, s);
    k_f_swiglu<<<(n_tok * n_ff + block - 1) / block, block, 0, s>>>(
        ctx->d_bg, ctx->d_bu, (size_t)n_tok * n_ff);
    fast_qx(ctx, ctx->d_bg, n_ff, n_tok, s);
    fast_mv(ctx, &ctx->d_ffn_down[l], ctx->d_bg, n_ff,
            ctx->d_bnormed, embd, n_tok, s);
    k_f_resadd<<<(n_tok * embd + block - 1) / block, block, 0, s>>>(
        ctx->d_bx, ctx->d_bnormed, (size_t)n_tok * embd);
}

/* ─── Host API ───────────────────────────────────────────────────────────── */

extern "C" bool oc_cuda_fast_supported(const OcCudaContext *ctx)
{
    return ctx && ctx->initialized && ctx->fast_ok;
}

extern "C" OcError oc_cuda_fast_wait(OcCudaContext *ctx)
{
    if (!ctx || !ctx->fast_stream) return OC_ERR_BACKEND;
    return cudaStreamSynchronize(fast_stream(ctx)) == cudaSuccess
         ? OC_OK : OC_ERR_BACKEND;
}

extern "C" OcError oc_cuda_fast_take(OcCudaContext *ctx, uint32_t n,
                                     uint32_t *out)
{
    if (!ctx || !ctx->fast_stream || !out) return OC_ERR_INVALID_ARG;
    if (n > ctx->fast_ring_cap) return OC_ERR_INVALID_ARG;
    if (cudaStreamSynchronize(fast_stream(ctx)) != cudaSuccess)
        return OC_ERR_BACKEND;
    if (n > 0 && cudaMemcpy(out, ctx->d_fast_ring, n * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost) != cudaSuccess)
        return OC_ERR_BACKEND;
    return OC_OK;
}

extern "C" OcError oc_cuda_fast_logits(OcCudaContext *ctx, float *out)
{
    if (!ctx || !ctx->fast_stream || !out) return OC_ERR_INVALID_ARG;
    if (cudaStreamSynchronize(fast_stream(ctx)) != cudaSuccess)
        return OC_ERR_BACKEND;
    if (cudaMemcpy(out, ctx->d_logits,
                   ctx->vocab_size * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return OC_ERR_BACKEND;
    return OC_OK;
}

extern "C" OcError oc_cuda_fast_feed(OcCudaContext *ctx, uint32_t token)
{
    if (!ctx || !ctx->fast_stream) return OC_ERR_INVALID_ARG;
    k_f_set<<<1, 1, 0, fast_stream(ctx)>>>(
        (OcFastState *)ctx->d_fast_state, token);
    return cudaGetLastError() == cudaSuccess ? OC_OK : OC_ERR_BACKEND;
}

extern "C" OcError oc_cuda_fast_step(OcCudaContext *ctx)
{
    if (!ctx || !ctx->fast_stream) return OC_ERR_INVALID_ARG;
    cudaStream_t s = fast_stream(ctx);
    fast_submit_body(ctx, s);
    fast_submit_head(ctx, s, false);
    return cudaGetLastError() == cudaSuccess ? OC_OK : OC_ERR_BACKEND;
}

/* Capture the single-token decode step (body + head + argmax) into a CUDA
 * graph. Everything the step touches is device-resident, so the captured
 * sequence is identical every replay — pos/token/step advance via the state
 * struct and the ring cursor, never via kernel arguments. */
static bool fast_build_graph(OcCudaContext *ctx)
{
    cudaStream_t s = fast_stream(ctx);
    cudaGraph_t g = NULL;
    cudaGraphExec_t exec = NULL;
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal) != cudaSuccess)
        return false;
    fast_submit_body(ctx, s);
    fast_submit_head(ctx, s, true);
    if (cudaStreamEndCapture(s, &g) != cudaSuccess || g == NULL)
        return false;
    if (cudaGraphInstantiateWithFlags(&exec, g, 0ull) != cudaSuccess) {
        cudaGraphDestroy(g);
        return false;
    }
    ctx->fast_graph = (void *)g;
    ctx->fast_graph_exec = (void *)exec;
    return true;
}

extern "C" OcError oc_cuda_fast_decode(OcCudaContext *ctx, uint32_t n)
{
    if (!ctx || !ctx->fast_stream) return OC_ERR_INVALID_ARG;
    if (n == 0 || n > ctx->fast_ring_cap) return OC_ERR_INVALID_ARG;
    cudaStream_t s = fast_stream(ctx);

    k_f_reset_step<<<1, 1, 0, s>>>((OcFastState *)ctx->d_fast_state);

    if (!ctx->fast_graph_exec) {
        static bool logged = false;
        const bool ok = fast_build_graph(ctx);
        if (!logged) {
            oc_log(ok ? OC_LOG_INFO : OC_LOG_WARN,
                   "cuda: fast decode CUDA graph capture %s",
                   ok ? "enabled" : "failed; unrolled submission");
            logged = true;
        }
    }
    if (ctx->fast_graph_exec) {
        for (uint32_t i = 0; i < n; i++)
            if (cudaGraphLaunch((cudaGraphExec_t)ctx->fast_graph_exec, s)
                != cudaSuccess)
                return OC_ERR_BACKEND;
    } else {
        for (uint32_t i = 0; i < n; i++) {
            fast_submit_body(ctx, s);
            fast_submit_head(ctx, s, true);
        }
    }
    return cudaGetLastError() == cudaSuccess ? OC_OK : OC_ERR_BACKEND;
}

extern "C" OcError oc_cuda_fast_prefill(OcCudaContext *ctx,
                                        const uint32_t *ids, size_t n_ids,
                                        uint32_t pos0, bool emit_first)
{
    if (!ctx || !ctx->fast_stream || !ids || n_ids == 0)
        return OC_ERR_INVALID_ARG;
    if (pos0 + n_ids > ctx->n_ctx) return OC_ERR_INVALID_ARG;
    cudaStream_t s = fast_stream(ctx);
    const uint32_t embd = ctx->n_embd;
    const uint32_t block = FAST_BLOCK;
    const uint32_t T = ctx->fast_batch;

    for (size_t base = 0; base < n_ids; base += T) {
        const uint32_t tn = (uint32_t)((n_ids - base < T)
                                       ? (n_ids - base) : T);
        const uint32_t p0 = pos0 + (uint32_t)base;

        if (cudaMemcpyAsync(ctx->d_bprompt, ids + base,
                            tn * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, s) != cudaSuccess)
            return OC_ERR_BACKEND;

        /* Embed the chunk. */
        const OcCudaWeight *emb = &ctx->d_tok_embeddings;
        if (emb->packed) {
            if (!oc_cuda_mmq_get_row_batch(emb->qtype, emb->data,
                                           ctx->d_bprompt, ctx->d_bx,
                                           embd, embd, tn, s))
                return OC_ERR_BACKEND;
        } else {
            dim3 grid((embd + block - 1) / block, tn);
            k_f_embed_f32_batch<<<grid, block, 0, s>>>(
                (const float *)emb->data, ctx->d_bprompt, ctx->d_bx,
                embd, embd);
        }

        for (uint32_t l = 0; l < ctx->n_layer; l++)
            fast_submit_batch_layer(ctx, l, tn, p0, s);
    }

    /* Seed the decode state: pos = pos0 + n_ids - 1, so the argmax's pos++
     * (or a feed's) leaves the next generated token at pos0 + n_ids. */
    {
        OcFastState seed = { 0u, pos0 + (uint32_t)n_ids - 1u, 0u, 0u };
        if (cudaMemcpyAsync(ctx->d_fast_state, &seed, sizeof(seed),
                            cudaMemcpyHostToDevice, s) != cudaSuccess)
            return OC_ERR_BACKEND;
    }

    /* Final norm + lm_head on the last token's residual row. */
    {
        float *x_last = ctx->d_bx
                      + (size_t)((n_ids - 1u) % FAST_BATCH) * embd;
        k_f_rmsnorm<<<dim3(1, 1), block, block * sizeof(float), s>>>(
            x_last, ctx->d_final_norm, ctx->d_normed, embd,
            ctx->rms_norm_eps, ctx->norm_scale, 0);
        fast_qx(ctx, ctx->d_normed, embd, 1u, s);
        fast_mv(ctx, &ctx->d_output, ctx->d_normed, embd,
                ctx->d_logits, 0, 1u, s);
        if (emit_first) {
            const uint32_t ab = 1024u;
            const size_t smem = ab * (sizeof(float) + sizeof(uint32_t));
            k_f_argmax<<<1, ab, smem, s>>>(ctx->d_logits, ctx->vocab_size,
                                           ctx->d_fast_ring,
                                           (OcFastState *)ctx->d_fast_state);
        }
    }
    return cudaGetLastError() == cudaSuccess ? OC_OK : OC_ERR_BACKEND;
}

/* ─── Attach / detach ────────────────────────────────────────────────────── */

static bool fast_packed(const OcCudaWeight *w)
{
    return w->data != NULL && w->packed && w->qtype != OC_QUANT_BF16;
}

static bool fast_model_ok(const OcCudaContext *ctx)
{
    /* Architectures the fast forward does not implement. */
    if (ctx->num_experts > 0 || ctx->uses_gemma4 || ctx->is_qwen35 ||
        ctx->uses_geglu || ctx->logit_softcap > 0.0f || ctx->v_rms_norm)
        return false;
    /* Per-layer geometry (Gemma-4 sliding/global split) unsupported. */
    if (ctx->l_head_dim || ctx->l_n_head_kv || ctx->l_rope_dim ||
        ctx->l_rope_theta || ctx->l_sliding)
        return false;
    if (ctx->kv_row != (size_t)ctx->n_head_kv * ctx->head_dim)
        return false;
    if (ctx->rope_dim > ctx->head_dim || ctx->rope_dim < 2u)
        return false;
    for (uint32_t l = 0; l < ctx->n_layer; l++) {
        if (ctx->l_out_scale && ctx->l_out_scale[l] != 0.0f &&
            ctx->l_out_scale[l] != 1.0f)
            return false;
        if (ctx->d_post_attn_norm && ctx->d_post_attn_norm[l])
            return false;
        if (ctx->d_post_ffw_norm && ctx->d_post_ffw_norm[l])
            return false;
        if (!ctx->d_attn_norm[l] || !ctx->d_ffn_norm[l]) return false;
        if (!fast_packed(&ctx->d_attn_q[l]) ||
            !fast_packed(&ctx->d_attn_k[l]) ||
            !fast_packed(&ctx->d_attn_v[l]) ||
            !fast_packed(&ctx->d_attn_output[l]) ||
            !fast_packed(&ctx->d_ffn_gate[l]) ||
            !fast_packed(&ctx->d_ffn_up[l]) ||
            !fast_packed(&ctx->d_ffn_down[l]))
            return false;
    }
    if (!fast_packed(&ctx->d_output)) return false;
    if (!ctx->d_final_norm) return false;
    /* The embedding table may stay f32 (k_f_embed_f32 covers it), but a
     * packed table must be an mmq type get_row_dev understands. */
    if (ctx->d_tok_embeddings.packed &&
        !oc_cuda_mmq_supported(ctx->d_tok_embeddings.qtype,
                               ctx->d_tok_embeddings.cols))
        return false;
    if (!ctx->d_tok_embeddings.packed &&
        ctx->d_tok_embeddings.qtype != OC_QUANT_F32)
        return false;
    return true;
}

extern "C" void oc_cuda_fast_attach(OcCudaContext *ctx)
{
    if (!ctx || !ctx->initialized || ctx->fast_ok) return;
    if (!fast_model_ok(ctx)) return;

    const uint32_t T = FAST_BATCH;
    const uint32_t embd = ctx->n_embd;
    const uint32_t qd = ctx->n_head * ctx->head_dim;
    const uint32_t kd = ctx->n_head_kv * ctx->head_dim;
    const uint32_t n_ff = ctx->n_ff;

    cudaStream_t s = NULL;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess)
        return;

#define FAST_ALLOC(ptr, bytes)                                              \
    do {                                                                    \
        if (cudaMalloc((void **)&(ptr), (bytes)) != cudaSuccess)            \
            goto fail;                                                      \
    } while (0)

    FAST_ALLOC(ctx->d_fast_state, sizeof(OcFastState));
    FAST_ALLOC(ctx->d_fast_ring, FAST_RING_CAP * sizeof(uint32_t));
    FAST_ALLOC(ctx->d_bx,      (size_t)T * embd * sizeof(float));
    FAST_ALLOC(ctx->d_bnormed, (size_t)T * embd * sizeof(float));
    FAST_ALLOC(ctx->d_bq,      (size_t)T * qd * sizeof(float));
    FAST_ALLOC(ctx->d_bk,      (size_t)T * kd * sizeof(float));
    FAST_ALLOC(ctx->d_bv,      (size_t)T * kd * sizeof(float));
    FAST_ALLOC(ctx->d_battn,   (size_t)T * qd * sizeof(float));
    FAST_ALLOC(ctx->d_bg,      (size_t)T * n_ff * sizeof(float));
    FAST_ALLOC(ctx->d_bu,      (size_t)T * n_ff * sizeof(float));
    FAST_ALLOC(ctx->d_bscores, (size_t)T * ctx->n_head * ctx->n_ctx
                               * sizeof(float));
    FAST_ALLOC(ctx->d_bprompt, T * sizeof(uint32_t));

    /* int8-activation scratch for the dp4a matvecs. Sized for the widest
     * activation any matvec consumes: n_embd (normed), n_head*head_dim
     * (attn out), n_ff (swiglu out). */
    {
        uint32_t max_cols = embd > n_ff ? embd : n_ff;
        if (qd > max_cols) max_cols = qd;
        const uint32_t mg = max_cols / 32u;
        ctx->xq_max_groups = mg;
        FAST_ALLOC(ctx->d_xq,    (size_t)T * mg * 32u);
        FAST_ALLOC(ctx->d_xs,    (size_t)T * mg * 2u * sizeof(float));
        FAST_ALLOC(ctx->d_xqsum, (size_t)T * mg * 2u * sizeof(int32_t));
        FAST_ALLOC(ctx->d_xh,    (size_t)T * mg * 64u);
    }
#undef FAST_ALLOC

    ctx->fast_stream = (void *)s;
    ctx->fast_ring_cap = FAST_RING_CAP;
    ctx->fast_batch = T;
    ctx->fast_ok = true;
    return;

fail:
    cudaFree(ctx->d_fast_state); ctx->d_fast_state = NULL;
    cudaFree(ctx->d_fast_ring);  ctx->d_fast_ring = NULL;
    cudaFree(ctx->d_bx);      ctx->d_bx = NULL;
    cudaFree(ctx->d_bnormed); ctx->d_bnormed = NULL;
    cudaFree(ctx->d_bq);      ctx->d_bq = NULL;
    cudaFree(ctx->d_bk);      ctx->d_bk = NULL;
    cudaFree(ctx->d_bv);      ctx->d_bv = NULL;
    cudaFree(ctx->d_battn);   ctx->d_battn = NULL;
    cudaFree(ctx->d_bg);      ctx->d_bg = NULL;
    cudaFree(ctx->d_bu);      ctx->d_bu = NULL;
    cudaFree(ctx->d_bscores); ctx->d_bscores = NULL;
    cudaFree(ctx->d_bprompt); ctx->d_bprompt = NULL;
    cudaFree(ctx->d_xq);     ctx->d_xq = NULL;
    cudaFree(ctx->d_xs);     ctx->d_xs = NULL;
    cudaFree(ctx->d_xqsum);  ctx->d_xqsum = NULL;
    cudaFree(ctx->d_xh);     ctx->d_xh = NULL;
    cudaStreamDestroy(s);
}

extern "C" void oc_cuda_fast_detach(OcCudaContext *ctx)
{
    if (!ctx) return;
    if (ctx->fast_graph_exec) {
        cudaGraphExecDestroy((cudaGraphExec_t)ctx->fast_graph_exec);
        ctx->fast_graph_exec = NULL;
    }
    if (ctx->fast_graph) {
        cudaGraphDestroy((cudaGraph_t)ctx->fast_graph);
        ctx->fast_graph = NULL;
    }
    cudaFree(ctx->d_fast_state); ctx->d_fast_state = NULL;
    cudaFree(ctx->d_fast_ring);  ctx->d_fast_ring = NULL;
    cudaFree(ctx->d_bx);      ctx->d_bx = NULL;
    cudaFree(ctx->d_bnormed); ctx->d_bnormed = NULL;
    cudaFree(ctx->d_bq);      ctx->d_bq = NULL;
    cudaFree(ctx->d_bk);      ctx->d_bk = NULL;
    cudaFree(ctx->d_bv);      ctx->d_bv = NULL;
    cudaFree(ctx->d_battn);   ctx->d_battn = NULL;
    cudaFree(ctx->d_bg);      ctx->d_bg = NULL;
    cudaFree(ctx->d_bu);      ctx->d_bu = NULL;
    cudaFree(ctx->d_bscores); ctx->d_bscores = NULL;
    cudaFree(ctx->d_bprompt); ctx->d_bprompt = NULL;
    cudaFree(ctx->d_xq);     ctx->d_xq = NULL;
    cudaFree(ctx->d_xs);     ctx->d_xs = NULL;
    cudaFree(ctx->d_xqsum);  ctx->d_xqsum = NULL;
    cudaFree(ctx->d_xh);     ctx->d_xh = NULL;
    if (ctx->fast_stream) {
        cudaStreamDestroy((cudaStream_t)ctx->fast_stream);
        ctx->fast_stream = NULL;
    }
    ctx->fast_ok = false;
    ctx->fast_ring_cap = 0;
    ctx->fast_batch = 0;
    ctx->xq_max_groups = 0;
}
