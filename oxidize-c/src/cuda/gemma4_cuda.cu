/* GPU-resident Gemma 4 decode. Kernels + host orchestration in one file.
 *
 * Everything for one token is enqueued on per-GPU streams with no
 * intermediate syncs; the only device->host traffic per token is the sampled
 * token id (greedy) or the softcapped logits (sampling).
 *
 * Weights stay in their GGUF quantized form; the matvec kernels fuse dequant
 * with the dot product (one warp per output row). The supported weight types
 * are exactly the ones dqv<T>() decodes, and every one of them is held to
 * tests/cuda_equiv.c: the SAME model run through gemma4_forward (CPU) and
 * through here must agree on every logit. A type that has not passed that gate
 * is refused by check_type(), loudly, rather than computing garbage.
 *
 * The graph below mirrors model_gemma4.c EXACTLY, including the four places
 * that are easy to get subtly wrong (all four WERE wrong here before the gate
 * existed, and nothing caught them): V is the raw K projection copied BEFORE
 * attn_k_norm/rope; V gets a scale-less RMSNorm; global (non-SWA) layers divide
 * the rope angle by rope_freqs; and the residual is
 * `x = (ffn + (post_attn_norm(proj) + x)) * output_scale`, not two
 * independently scaled adds.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "../gguf.h"
#include "../model_gemma4.h"
#include "../quant.h"
#include "gemma4_cuda.h"
}

#define MAX_GPUS 8
#define MAX_BATCH 8 /* batched-matvec ceiling (future speculative verify) */

#define CUDA_TRY(expr)                                                     \
  do {                                                                     \
    cudaError_t e_ = (expr);                                               \
    if (e_ != cudaSuccess) {                                               \
      if (err && errlen)                                                   \
        snprintf(err, errlen, "cuda: %s at %s:%d", cudaGetErrorString(e_), \
                 __FILE__, __LINE__);                                      \
      return -1;                                                           \
    }                                                                      \
  } while (0)

/* Device-side per-value dequant (dqv<T>, dh, ksm) — extracted to a shared header
 * so the llama backend decodes the SAME bytes the same way. */
#include "cuda_dequant.cuh"

/* ======================== kernels ======================== */

/* ---- fused dequant matvec, one warp per row ----
 * y[b*rows + r] = dot(dequant(W row r), x + b*cols). For decode nb == 1. */
template <int T>
__global__ void k_matvec(float* __restrict__ y, const uint8_t* __restrict__ W,
                         int rows, int cols, const float* __restrict__ x,
                         int nb, size_t rowbytes) {
  int row = (int)(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
  if (row >= rows) return;
  int lane = threadIdx.x & 31;
  const uint8_t* rp = W + (size_t)row * rowbytes;
  for (int b = 0; b < nb; ++b) {
    float acc = 0.0f;
    for (int i = lane; i < cols; i += 32) acc += dqv<T>(rp, i) * x[b * cols + i];
    for (int o = 16; o > 0; o >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[(size_t)b * rows + row] = acc;
  }
}

/* AL5_XS keeps its hand-fused variant (3-bit codes unpack 32 at a time and this
 * is the resident-decode hot path). Same result as k_matvec<OC_AL5_XS>; both are
 * held to the equivalence gate. */
__global__ void k_matvec_al5xs(float* __restrict__ y,
                               const uint8_t* __restrict__ W, int rows,
                               int cols, const float* __restrict__ x, int nb) {
  int row = (int)(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
  if (row >= rows) return;
  int lane = threadIdx.x & 31;
  int nblk = cols / 32;
  const uint8_t* rp = W + (size_t)row * (size_t)nblk * OC_BLK_AL5_XS;
  float acc[MAX_BATCH];
#pragma unroll
  for (int b = 0; b < MAX_BATCH; ++b) acc[b] = 0.0f;
  for (int blk = lane; blk < nblk; blk += 32) {
    const uint8_t* bp = rp + (size_t)blk * OC_BLK_AL5_XS;
    float d = dh(bp);
    const uint8_t* q = bp + 2;
    uint32_t ww[4];
    ww[0] = (uint32_t)q[0] | ((uint32_t)q[1] << 8) | ((uint32_t)q[2] << 16) |
            ((uint32_t)q[3] << 24);
    ww[1] = (uint32_t)q[4] | ((uint32_t)q[5] << 8) | ((uint32_t)q[6] << 16) |
            ((uint32_t)q[7] << 24);
    ww[2] = (uint32_t)q[8] | ((uint32_t)q[9] << 8) | ((uint32_t)q[10] << 16) |
            ((uint32_t)q[11] << 24);
    ww[3] = 0;
    const float* xb = x + blk * 32;
    float pa[MAX_BATCH];
#pragma unroll
    for (int b = 0; b < MAX_BATCH; ++b) pa[b] = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      int bit = 3 * i;
      int word = bit >> 5, off = bit & 31;
      uint32_t v = ww[word] >> off;
      if (off > 29) v |= ww[word + 1] << (32 - off);
      float qq = (float)(int)(v & 7u) - 4.0f;
      for (int b = 0; b < nb; ++b) pa[b] += qq * xb[b * cols + i];
    }
    for (int b = 0; b < nb; ++b) acc[b] += d * pa[b];
  }
#pragma unroll
  for (int b = 0; b < MAX_BATCH; ++b)
    for (int o = 16; o > 0; o >>= 1)
      acc[b] += __shfl_down_sync(0xffffffffu, acc[b], o);
  if (lane == 0)
    for (int b = 0; b < nb; ++b) y[(size_t)b * rows + row] = acc[b];
}

/* ---- embedding row dequant * emb_scale into x. */
template <int T>
__global__ void k_embed(float* __restrict__ x, const uint8_t* __restrict__ row,
                        int n, float scale) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) x[i] = dqv<T>(row, i) * scale;
}

/* ---- RMSNorm: grid.x independent vectors of length `per`, w shared.
 * out may alias x. Mirrors oc_rms_norm (Gemma weights already carry +1).
 * w == NULL is the scale-less form the V heads use. */
__global__ void k_rmsnorm(float* __restrict__ out, const float* __restrict__ x,
                          const float* __restrict__ w, int per, float eps) {
  x += (size_t)blockIdx.x * per;
  out += (size_t)blockIdx.x * per;
  __shared__ float red[256];
  float s = 0.0f;
  for (int i = threadIdx.x; i < per; i += blockDim.x) s += x[i] * x[i];
  red[threadIdx.x] = s;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if ((int)threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)per + eps);
  for (int i = threadIdx.x; i < per; i += blockDim.x)
    out[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

/* ---- NeoX split-half RoPE, grid.x = heads. Mirrors oc_rope:
 * freq_i = theta^(-2i/rope_len), angle = pos*freq_i, divided by freqs[i] when
 * present; pairs (p[i], p[half+i]); dims >= rope_len pass through. `freqs` is
 * gemma4's rope_freqs.weight, used on GLOBAL (non-SWA) layers only. Host skips
 * the launch when pos == 0 (identity). */
__global__ void k_rope(float* __restrict__ vec, int head_dim, int pos,
                       float theta, int rope_len,
                       const float* __restrict__ freqs) {
  float* p = vec + (size_t)blockIdx.x * head_dim;
  int half = rope_len / 2;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    float freq = powf(theta, -2.0f * (float)i / (float)rope_len);
    float angle = (float)pos * freq;
    if (freqs) angle /= freqs[i];
    float c = cosf(angle), s = sinf(angle);
    float x0 = p[i], x1 = p[half + i];
    p[i] = x0 * c - x1 * s;
    p[half + i] = x0 * s + x1 * c;
  }
}

/* ---- store K/V rows (f32) into the f16 ring caches at `slot`. */
__global__ void k_kv_store(__half* __restrict__ kc, __half* __restrict__ vc,
                           const float* __restrict__ k,
                           const float* __restrict__ v, int k_len, int v_len,
                           size_t slot) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < k_len) kc[slot * k_len + i] = __float2half(k[i]);
  if (i < v_len) vc[slot * v_len + i] = __float2half(v[i]);
}

/* ---- fused decode attention: one block per q head. Scores for positions
 * [t0, t1) are materialized in shared memory (count <= cache_cap; smem
 * attribute raised at init when ctx is large), then softmax + weighted V.
 * Mirrors attn_heads() in model_gemma4.c (same ring indexing t % cache_cap). */
__global__ void k_attn(float* __restrict__ out, const float* __restrict__ q,
                       const __half* __restrict__ kc,
                       const __half* __restrict__ vc, int hd, int vd,
                       int group, int cache_cap, int t0, int t1, float scale) {
  extern __shared__ float sm[]; /* [hd] q | [count] scores | [256] reduce */
  int h = blockIdx.x;
  int kvh = h / group;
  int n_kv = gridDim.x / group;
  int k_row = n_kv * hd, v_row = n_kv * vd;
  int count = t1 - t0;
  float* sq = sm;
  float* sp = sm + hd;
  float* red = sp + count;

  for (int i = threadIdx.x; i < hd; i += blockDim.x) sq[i] = q[h * hd + i];
  __syncthreads();

  /* scores */
  for (int t = t0 + (int)threadIdx.x; t < t1; t += blockDim.x) {
    const __half* krow = kc + (size_t)(t % cache_cap) * k_row + kvh * hd;
    float dot = 0.0f;
    for (int d = 0; d < hd; ++d) dot += sq[d] * __half2float(krow[d]);
    sp[t - t0] = dot * scale;
  }
  __syncthreads();

  /* max */
  float mx = -INFINITY;
  for (int i = threadIdx.x; i < count; i += blockDim.x)
    mx = fmaxf(mx, sp[i]);
  red[threadIdx.x] = mx;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if ((int)threadIdx.x < o)
      red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
    __syncthreads();
  }
  mx = red[0];
  __syncthreads();

  /* exp + sum */
  float s = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    float e = expf(sp[i] - mx);
    sp[i] = e;
    s += e;
  }
  red[threadIdx.x] = s;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if ((int)threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  float inv = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;

  /* weighted V */
  for (int d = threadIdx.x; d < vd; d += blockDim.x) {
    float acc = 0.0f;
    for (int t = t0; t < t1; ++t) {
      const __half* vrow = vc + (size_t)(t % cache_cap) * v_row + kvh * vd;
      acc += sp[t - t0] * __half2float(vrow[d]);
    }
    out[(size_t)h * vd + d] = acc * inv;
  }
}

/* ---- rotoquant KV cache (mirrors oc_fht/oc_kvq_* in quant.c) ---- */

/* Normalized fast Walsh-Hadamard transform, one head per block. Self-inverse
 * and orthonormal (dot products invariant). d = power of two <= 512. */
__global__ void k_fht(float* __restrict__ v, int d) {
  extern __shared__ float sb[];
  float* p = v + (size_t)blockIdx.x * d;
  for (int i = threadIdx.x; i < d; i += blockDim.x) sb[i] = p[i];
  __syncthreads();
  for (int len = 1; len < d; len <<= 1) {
    for (int j = threadIdx.x; j < d / 2; j += blockDim.x) {
      int i = (j / len) * (len << 1) + (j % len);
      float a = sb[i], b = sb[i + len];
      sb[i] = a + b;
      sb[i + len] = a - b;
    }
    __syncthreads();
  }
  float s = rsqrtf((float)d);
  for (int i = threadIdx.x; i < d; i += blockDim.x) p[i] = sb[i] * s;
}

/* Asymmetric int4 row quantizer: n floats -> n/2 bytes (low nibble first) +
 * meta {scale, min}. Whole block cooperates; call twice per k_kv_store_q4. */
__device__ static void q4_row(const float* __restrict__ x, int n,
                              uint8_t* __restrict__ out,
                              float* __restrict__ meta) {
  __shared__ float rmin[128], rmax[128];
  float mn = INFINITY, mx = -INFINITY;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    mn = fminf(mn, x[i]);
    mx = fmaxf(mx, x[i]);
  }
  rmin[threadIdx.x] = mn;
  rmax[threadIdx.x] = mx;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if ((int)threadIdx.x < o) {
      rmin[threadIdx.x] = fminf(rmin[threadIdx.x], rmin[threadIdx.x + o]);
      rmax[threadIdx.x] = fmaxf(rmax[threadIdx.x], rmax[threadIdx.x + o]);
    }
    __syncthreads();
  }
  float lo = rmin[0];
  float s = (rmax[0] - lo) / 15.0f;
  float inv = s > 0.0f ? 1.0f / s : 0.0f;
  if (threadIdx.x == 0) { meta[0] = s; meta[1] = lo; }
  for (int i = threadIdx.x * 2; i < n; i += blockDim.x * 2) {
    int q0 = min(15, max(0, __float2int_rn((x[i] - lo) * inv)));
    int q1 = min(15, max(0, __float2int_rn((x[i + 1] - lo) * inv)));
    out[i >> 1] = (uint8_t)(q0 | (q1 << 4));
  }
  __syncthreads(); /* shared arrays are reused by the next q4_row call */
}

/* Store one token's (already rotated) K/V heads as int4, one kv head/block. */
__global__ void k_kv_store_q4(uint8_t* __restrict__ kq, float* __restrict__ km,
                              uint8_t* __restrict__ vq, float* __restrict__ vm,
                              const float* __restrict__ k,
                              const float* __restrict__ v, int hd, int vd,
                              int n_kv, size_t slot) {
  int h = blockIdx.x;
  size_t r = slot * n_kv + h;
  q4_row(k + (size_t)h * hd, hd, kq + (slot * (size_t)n_kv * hd + (size_t)h * hd) / 2,
         km + r * 2);
  q4_row(v + (size_t)h * vd, vd, vq + (slot * (size_t)n_kv * vd + (size_t)h * vd) / 2,
         vm + r * 2);
}

/* k_attn over the rotated int4 cache. Same shape/smem contract as k_attn.
 * q must already be rotated; out comes back rotated (host runs k_fht on it).
 * dot(q, dequant(k)) = scale*dot(q, qk) + min*sum(q). */
__global__ void k_attn_q4(float* __restrict__ out, const float* __restrict__ q,
                          const uint8_t* __restrict__ kq,
                          const float* __restrict__ km,
                          const uint8_t* __restrict__ vq,
                          const float* __restrict__ vm, int hd, int vd,
                          int group, int cache_cap, int t0, int t1,
                          float scale) {
  extern __shared__ float sm[]; /* [hd] q | [count] scores | [256] reduce */
  int h = blockIdx.x;
  int kvh = h / group;
  int n_kv = gridDim.x / group;
  int count = t1 - t0;
  float* sq = sm;
  float* sp = sm + hd;
  float* red = sp + count;

  for (int i = threadIdx.x; i < hd; i += blockDim.x) sq[i] = q[h * hd + i];
  __syncthreads();

  /* sum(q) for the asymmetric zero-point term */
  float qs = 0.0f;
  for (int i = threadIdx.x; i < hd; i += blockDim.x) qs += sq[i];
  red[threadIdx.x] = qs;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if ((int)threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  float qsum = red[0];
  __syncthreads();

  /* scores */
  size_t k_half = (size_t)n_kv * hd / 2;
  for (int t = t0 + (int)threadIdx.x; t < t1; t += blockDim.x) {
    size_t slot = (size_t)(t % cache_cap);
    const uint8_t* kr = kq + slot * k_half + (size_t)kvh * hd / 2;
    float dot = 0.0f;
    for (int d = 0; d < hd; d += 2) {
      uint8_t b = kr[d >> 1];
      dot += sq[d] * (float)(b & 15u) + sq[d + 1] * (float)(b >> 4);
    }
    const float* meta = km + (slot * n_kv + kvh) * 2;
    sp[t - t0] = (meta[0] * dot + meta[1] * qsum) * scale;
  }
  __syncthreads();

  /* max */
  float mx = -INFINITY;
  for (int i = threadIdx.x; i < count; i += blockDim.x) mx = fmaxf(mx, sp[i]);
  red[threadIdx.x] = mx;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if ((int)threadIdx.x < o)
      red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
    __syncthreads();
  }
  mx = red[0];
  __syncthreads();

  /* exp + sum */
  float s = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    float e = expf(sp[i] - mx);
    sp[i] = e;
    s += e;
  }
  red[threadIdx.x] = s;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if ((int)threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  float inv = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;

  /* weighted V (still rotated) */
  size_t v_half = (size_t)n_kv * vd / 2;
  for (int d = threadIdx.x; d < vd; d += blockDim.x) {
    float acc = 0.0f;
    for (int t = t0; t < t1; ++t) {
      size_t slot = (size_t)(t % cache_cap);
      uint8_t b = vq[slot * v_half + ((size_t)kvh * vd + d) / 2];
      float qv = (float)((d & 1) ? (b >> 4) : (b & 15u));
      const float* meta = vm + (slot * n_kv + kvh) * 2;
      acc += sp[t - t0] * (meta[0] * qv + meta[1]);
    }
    out[(size_t)h * vd + d] = acc * inv;
  }
}

/* ---- GeGLU: gate = gelu_tanh(gate) * up (mirrors oc_geglu). */
__global__ void k_geglu(float* __restrict__ gate, const float* __restrict__ up,
                        int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  const float K = 0.797884560f; /* sqrt(2/pi) */
  float g = gate[i];
  float gelu = 0.5f * g * (1.0f + tanhf(K * (g + 0.044715f * g * g * g)));
  gate[i] = gelu * up[i];
}

/* ---- c[i] += x[i]: the attention residual (attn_out = post_attn_norm + x). */
__global__ void k_add(float* __restrict__ c, const float* __restrict__ x, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) c[i] += x[i];
}

/* ---- x[i] = (ffn[i] + attn[i]) * s   (blk.N.layer_output_scale). */
__global__ void k_resid_out(float* __restrict__ x, const float* __restrict__ ffn,
                            const float* __restrict__ attn, float s, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) x[i] = (ffn[i] + attn[i]) * s;
}

/* ---- final softcap: l = c * tanh(l / c). */
__global__ void k_softcap(float* __restrict__ l, float c, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) l[i] = c * tanhf(l[i] / c);
}

/* ---- two-stage argmax over n floats. */
__global__ void k_argmax_stage1(const float* __restrict__ v, int n,
                                float* __restrict__ bmax,
                                int* __restrict__ bidx) {
  __shared__ float rmax[256];
  __shared__ int ridx[256];
  float mx = -INFINITY;
  int mi = 0;
  for (int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); i < n;
       i += (int)(gridDim.x * blockDim.x))
    if (v[i] > mx) { mx = v[i]; mi = i; }
  rmax[threadIdx.x] = mx;
  ridx[threadIdx.x] = mi;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if ((int)threadIdx.x < o && rmax[threadIdx.x + o] > rmax[threadIdx.x]) {
      rmax[threadIdx.x] = rmax[threadIdx.x + o];
      ridx[threadIdx.x] = ridx[threadIdx.x + o];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { bmax[blockIdx.x] = rmax[0]; bidx[blockIdx.x] = ridx[0]; }
}

__global__ void k_argmax_stage2(const float* __restrict__ bmax,
                                const int* __restrict__ bidx, int n,
                                int* __restrict__ out) {
  __shared__ float rmax[256];
  __shared__ int ridx[256];
  float mx = -INFINITY;
  int mi = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    if (bmax[i] > mx) { mx = bmax[i]; mi = bidx[i]; }
  rmax[threadIdx.x] = mx;
  ridx[threadIdx.x] = mi;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if ((int)threadIdx.x < o && rmax[threadIdx.x + o] > rmax[threadIdx.x]) {
      rmax[threadIdx.x] = rmax[threadIdx.x + o];
      ridx[threadIdx.x] = ridx[threadIdx.x + o];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = ridx[0];
}

/* ======================== host side ======================== */

#define ARGMAX_BLOCKS 256

typedef struct {
  int dev;
  uint8_t *q_w, *k_w, *v_w, *o_w, *gate_w, *up_w, *down_w; /* device blobs */
  float *attn_norm, *q_norm, *k_norm, *post_attn_norm, *ffn_norm,
      *post_ffn_norm; /* device f32, may be NULL (matches host) */
  __half *k_cache, *v_cache;      /* f16 path */
  uint8_t *k_qcache, *v_qcache;   /* rotoquant path (m->kv_quant) */
  float *k_qmeta, *v_qmeta;
} CudaLayer;

typedef struct {
  float *x, *normed, *q, *k, *v, *attn_res, *attn_proj, *gate, *up, *ffn_out;
  uint8_t* tok_embd; /* uploaded only on the first and last GPU */
  float* rope_freqs; /* global-layer rope divisors */
} GpuScratch;

struct Gemma4Cuda {
  const Gemma4Model* m;
  int n_gpus;
  size_t n_gpu_layers; /* layers [0, n_gpu_layers) here; the rest on the CPU */
  cudaStream_t stream[MAX_GPUS];
  cudaEvent_t ev[MAX_GPUS];      /* hand-off copy complete */
  cudaEvent_t ev_done[MAX_GPUS]; /* GPU finished its layer segment (per token) */
  CudaLayer* layers;
  GpuScratch gpu[MAX_GPUS];
  /* on the last GPU (only when the GPU owns the whole stack): */
  float* out_norm;
  float* logits;
  float* red_max;
  int* red_idx;
  int* argmax_d;
  int last_dev;
};

static int layer_dev(const Gemma4Cuda* c, size_t l) {
  return (int)(l * (size_t)c->n_gpus / c->n_gpu_layers);
}

/* Upload `n` bytes to the current device. Returns NULL on failure. */
static void* dupload(const void* src, size_t n) {
  void* d = NULL;
  if (cudaMalloc(&d, n) != cudaSuccess) return NULL;
  if (cudaMemcpy(d, src, n, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d);
    return NULL;
  }
  return d;
}

static size_t tensor_bytes(const GgufTensorInfo* t) {
  size_t cols = (size_t)t->dims[0], rows = 1;
  for (uint32_t d = 1; d < t->n_dims; ++d) rows *= (size_t)t->dims[d];
  return rows * oc_row_bytes(t->ggml_type, cols);
}

/* The weight types dqv<T>() decodes AND tests/cuda_equiv.c has proven equal to
 * the CPU forward. Anything else is refused: a wrong kernel is worse than none. */
static int check_type(uint32_t t, const char* what, char* err, size_t errlen) {
  switch (t) {
    case OC_F32:
    case OC_F16:
    case OC_Q4_0:
    case OC_Q8_0:
    case OC_Q4_K:
    case OC_Q5_K:
    case OC_Q6_K:
    case OC_AL5_XS:
      return 0;
    default:
      break;
  }
  if (err && errlen)
    snprintf(err, errlen,
             "cuda: %s has quant type %u; GPU kernels exist for "
             "F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/AL5_XS only",
             what, t);
  return -1;
}

/* Launch the right fused matvec for a weight's ggml type. */
#define MV(T)                                                 \
  k_matvec<T><<<grid, BLOCK, 0, s>>>(y, W, rows, cols, x, nb, \
                                     oc_row_bytes(T, (size_t)cols))
static void matvec(cudaStream_t s, uint32_t type, const uint8_t* W, int rows,
                   int cols, const float* x, float* y, int nb) {
  const int WARPS = 8, BLOCK = WARPS * 32;
  int grid = (rows + WARPS - 1) / WARPS;
  switch (type) {
    case OC_AL5_XS:
      k_matvec_al5xs<<<grid, BLOCK, 0, s>>>(y, W, rows, cols, x, nb);
      break;
    case OC_F32: MV(OC_F32); break;
    case OC_F16: MV(OC_F16); break;
    case OC_Q4_0: MV(OC_Q4_0); break;
    case OC_Q8_0: MV(OC_Q8_0); break;
    case OC_Q4_K: MV(OC_Q4_K); break;
    case OC_Q5_K: MV(OC_Q5_K); break;
    case OC_Q6_K: MV(OC_Q6_K); break;
    default: break; /* unreachable: check_type() gates every upload */
  }
}
#undef MV

#define EMB(T) k_embed<T><<<(n + 255) / 256, 256, 0, s>>>(x, row, n, scale)
static void embed(cudaStream_t s, uint32_t type, const uint8_t* row, int n,
                  float scale, float* x) {
  switch (type) {
    case OC_F32: EMB(OC_F32); break;
    case OC_F16: EMB(OC_F16); break;
    case OC_Q4_0: EMB(OC_Q4_0); break;
    case OC_Q8_0: EMB(OC_Q8_0); break;
    case OC_Q4_K: EMB(OC_Q4_K); break;
    case OC_Q5_K: EMB(OC_Q5_K); break;
    case OC_Q6_K: EMB(OC_Q6_K); break;
    case OC_AL5_XS: EMB(OC_AL5_XS); break;
    default: break;
  }
}
#undef EMB

int gemma4_cuda_init(Gemma4Cuda** out, const Gemma4Model* m, int n_gpus,
                     int n_gpu_layers, char* err, size_t errlen) {
  *out = NULL;
  int dev_count = 0;
  CUDA_TRY(cudaGetDeviceCount(&dev_count));
  if (n_gpus < 1) n_gpus = 1;
  if (n_gpus > MAX_GPUS) n_gpus = MAX_GPUS;
  if (n_gpus > dev_count) {
    if (err && errlen)
      snprintf(err, errlen, "cuda: %d GPUs requested, %d available", n_gpus,
               dev_count);
    return -1;
  }
  size_t ngl = (n_gpu_layers < 0 || (size_t)n_gpu_layers > m->n_layers)
                   ? m->n_layers
                   : (size_t)n_gpu_layers;
  if (ngl == 0) {
    if (err && errlen)
      snprintf(err, errlen, "cuda: --ngl 0 is the pure-CPU path (no GPU init)");
    return -1;
  }
  bool partial = ngl < m->n_layers;

  /* scratch sizes (the same maxima the CPU loader computes), over the GPU's
   * layers only — before the first goto (C++ forbids jumping over inits). */
  size_t max_q = m->hidden, max_k = 0, max_hd = 0;
  for (size_t l = 0; l < ngl; ++l) {
    const Gemma4Layer* L = &m->layers[l];
    size_t q_rows = m->n_head * L->head_dim;
    size_t k_rows = L->n_kv_heads * L->head_dim;
    size_t v_rows = L->n_kv_heads * L->v_head_dim;
    size_t attn_in = m->n_head * L->v_head_dim;
    if (q_rows > max_q) max_q = q_rows;
    if (attn_in > max_q) max_q = attn_in;
    if (k_rows > max_k) max_k = k_rows;
    if (v_rows > max_k) max_k = v_rows;
    if (L->head_dim > max_hd) max_hd = L->head_dim;
    if (L->v_head_dim > max_hd) max_hd = L->v_head_dim;
  }

  Gemma4Cuda* c = (Gemma4Cuda*)calloc(1, sizeof(Gemma4Cuda));
  if (!c) return -1;
  c->m = m;
  c->n_gpus = n_gpus;
  c->n_gpu_layers = ngl;
  c->last_dev = layer_dev(c, ngl - 1);
  c->layers = (CudaLayer*)calloc(ngl, sizeof(CudaLayer));
  if (!c->layers) { free(c); return -1; }

  if (check_type(m->tok_embd->ggml_type, "token_embd", err, errlen) != 0)
    goto fail;

  for (int g = 0; g < n_gpus; ++g) {
    if (cudaSetDevice(g) != cudaSuccess ||
        cudaStreamCreate(&c->stream[g]) != cudaSuccess ||
        cudaEventCreateWithFlags(&c->ev[g], cudaEventDisableTiming) !=
            cudaSuccess ||
        cudaEventCreateWithFlags(&c->ev_done[g], cudaEventDisableTiming) !=
            cudaSuccess)
      goto fail_msg;
    GpuScratch* S = &c->gpu[g];
    if (cudaMalloc(&S->x, m->hidden * 4) != cudaSuccess ||
        cudaMalloc(&S->normed, m->hidden * 4) != cudaSuccess ||
        cudaMalloc(&S->q, max_q * 4) != cudaSuccess ||
        cudaMalloc(&S->k, max_k * 4) != cudaSuccess ||
        cudaMalloc(&S->v, max_k * 4) != cudaSuccess ||
        cudaMalloc(&S->attn_res, max_q * 4) != cudaSuccess ||
        cudaMalloc(&S->attn_proj, m->hidden * 4) != cudaSuccess ||
        cudaMalloc(&S->gate, m->inter * 4) != cudaSuccess ||
        cudaMalloc(&S->up, m->inter * 4) != cudaSuccess ||
        cudaMalloc(&S->ffn_out, m->hidden * 4) != cudaSuccess)
      goto fail_msg;
    /* tok_embd on the first GPU (embedding) and — unless the CPU owns the tail
     * and therefore the tied-embedding head — on the last GPU too. */
    if (g == 0 || (!partial && g == c->last_dev)) {
      S->tok_embd = (uint8_t*)dupload(m->tok_embd->data, tensor_bytes(m->tok_embd));
      if (!S->tok_embd) goto fail_msg;
    }
    if (m->rope_freqs && max_hd >= 2) {
      S->rope_freqs = (float*)dupload(m->rope_freqs, (max_hd / 2) * 4);
      if (!S->rope_freqs) goto fail_msg;
    }
    /* best-effort peer access for the pipeline hand-off copies */
    for (int p = 0; p < n_gpus; ++p)
      if (p != g) cudaDeviceEnablePeerAccess(p, 0); /* ok if unsupported */
    cudaGetLastError(); /* clear any peer-access "already enabled" error */
  }

  if (!partial) { /* last-GPU head buffers */
    if (cudaSetDevice(c->last_dev) != cudaSuccess) goto fail_msg;
    c->out_norm = (float*)dupload(m->out_norm, m->hidden * 4);
    if (!c->out_norm || cudaMalloc(&c->logits, m->vocab * 4) != cudaSuccess ||
        cudaMalloc(&c->red_max, ARGMAX_BLOCKS * 4) != cudaSuccess ||
        cudaMalloc(&c->red_idx, ARGMAX_BLOCKS * 4) != cudaSuccess ||
        cudaMalloc(&c->argmax_d, 4) != cudaSuccess)
      goto fail_msg;
  }

  /* per-layer weights, norms, KV caches */
  for (size_t l = 0; l < ngl; ++l) {
    const Gemma4Layer* L = &m->layers[l];
    CudaLayer* D = &c->layers[l];
    D->dev = layer_dev(c, l);
    if (cudaSetDevice(D->dev) != cudaSuccess) goto fail_msg;

#define UP_MAT(dst, src, what)                                          \
  do {                                                                  \
    if (check_type((src)->ggml_type, what, err, errlen) != 0) goto fail; \
    D->dst = (uint8_t*)dupload((src)->data, tensor_bytes(src));         \
    if (!D->dst) goto fail_msg;                                         \
  } while (0)
#define UP_VEC(dst, src, n)                      \
  do {                                           \
    if (L->src) {                                \
      D->dst = (float*)dupload(L->src, (n) * 4); \
      if (!D->dst) goto fail_msg;                \
    }                                            \
  } while (0)

    UP_MAT(q_w, L->attn_q, "attn_q");
    UP_MAT(k_w, L->attn_k, "attn_k");
    if (L->attn_v) UP_MAT(v_w, L->attn_v, "attn_v");
    UP_MAT(o_w, L->attn_out, "attn_output");
    UP_MAT(gate_w, L->ffn_gate, "ffn_gate");
    UP_MAT(up_w, L->ffn_up, "ffn_up");
    UP_MAT(down_w, L->ffn_down, "ffn_down");
    UP_VEC(attn_norm, attn_norm, m->hidden);
    UP_VEC(q_norm, attn_q_norm, L->head_dim);
    UP_VEC(k_norm, attn_k_norm, L->head_dim);
    UP_VEC(post_attn_norm, post_attn_norm, m->hidden);
    UP_VEC(ffn_norm, ffn_norm, m->hidden);
    UP_VEC(post_ffn_norm, post_ffn_norm, m->hidden);
#undef UP_MAT
#undef UP_VEC

    size_t kn = L->cache_cap * L->n_kv_heads * L->head_dim;
    size_t vn = L->cache_cap * L->n_kv_heads * L->v_head_dim;
    size_t mn = L->cache_cap * L->n_kv_heads * 2 * sizeof(float);
    if (m->kv_quant) {
      if (cudaMalloc(&D->k_qcache, kn / 2) != cudaSuccess ||
          cudaMalloc(&D->v_qcache, vn / 2) != cudaSuccess ||
          cudaMalloc(&D->k_qmeta, mn) != cudaSuccess ||
          cudaMalloc(&D->v_qmeta, mn) != cudaSuccess ||
          cudaMemset(D->k_qcache, 0, kn / 2) != cudaSuccess ||
          cudaMemset(D->v_qcache, 0, vn / 2) != cudaSuccess ||
          cudaMemset(D->k_qmeta, 0, mn) != cudaSuccess ||
          cudaMemset(D->v_qmeta, 0, mn) != cudaSuccess)
        goto fail_msg;
    } else if (cudaMalloc(&D->k_cache, kn * 2) != cudaSuccess ||
               cudaMalloc(&D->v_cache, vn * 2) != cudaSuccess ||
               cudaMemset(D->k_cache, 0, kn * 2) != cudaSuccess ||
               cudaMemset(D->v_cache, 0, vn * 2) != cudaSuccess)
      goto fail_msg;
  }
  if (m->kv_quant)
    fprintf(stderr, "cuda: rotoquant KV cache (rotated int4, 4x vs f16)\n");

  /* Raise k_attn's dynamic smem cap if a large --ctx needs it. */
  {
    size_t mhd = 0, max_cap = 0;
    for (size_t l = 0; l < ngl; ++l) {
      if (m->layers[l].head_dim > mhd) mhd = m->layers[l].head_dim;
      if (m->layers[l].cache_cap > max_cap) max_cap = m->layers[l].cache_cap;
    }
    size_t need = (mhd + max_cap + 256) * 4;
    if (need > 48 * 1024) {
      for (int g = 0; g < n_gpus; ++g) {
        cudaSetDevice(g);
        if (cudaFuncSetAttribute(k_attn,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)need) != cudaSuccess ||
            cudaFuncSetAttribute(k_attn_q4,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)need) != cudaSuccess) {
          if (err && errlen)
            snprintf(err, errlen,
                     "cuda: ctx too large for k_attn shared memory (%zu bytes)",
                     need);
          goto fail;
        }
      }
    }
  }

  fprintf(stderr, "cuda: %zu/%zu layers on %d GPU(s)%s\n", ngl, m->n_layers,
          n_gpus, partial ? "; remaining layers + head on CPU" : "");
  *out = c;
  return 0;

fail_msg:
  if (err && errlen && !err[0])
    snprintf(err, errlen, "cuda: allocation/upload failed: %s",
             cudaGetErrorString(cudaGetLastError()));
fail:
  gemma4_cuda_free(c);
  return -1;
}

void gemma4_cuda_free(Gemma4Cuda* c) {
  if (!c) return;
  for (size_t l = 0; c->layers && l < c->n_gpu_layers; ++l) {
    CudaLayer* D = &c->layers[l];
    cudaFree(D->q_w); cudaFree(D->k_w); cudaFree(D->v_w); cudaFree(D->o_w);
    cudaFree(D->gate_w); cudaFree(D->up_w); cudaFree(D->down_w);
    cudaFree(D->attn_norm); cudaFree(D->q_norm); cudaFree(D->k_norm);
    cudaFree(D->post_attn_norm); cudaFree(D->ffn_norm); cudaFree(D->post_ffn_norm);
    cudaFree(D->k_cache); cudaFree(D->v_cache);
    cudaFree(D->k_qcache); cudaFree(D->v_qcache);
    cudaFree(D->k_qmeta); cudaFree(D->v_qmeta);
  }
  free(c->layers);
  for (int g = 0; g < c->n_gpus; ++g) {
    GpuScratch* S = &c->gpu[g];
    cudaFree(S->x); cudaFree(S->normed); cudaFree(S->q); cudaFree(S->k);
    cudaFree(S->v); cudaFree(S->attn_res); cudaFree(S->attn_proj);
    cudaFree(S->gate); cudaFree(S->up); cudaFree(S->ffn_out);
    cudaFree(S->tok_embd); cudaFree(S->rope_freqs);
    if (c->stream[g]) cudaStreamDestroy(c->stream[g]);
    if (c->ev[g]) cudaEventDestroy(c->ev[g]);
    if (c->ev_done[g]) cudaEventDestroy(c->ev_done[g]);
  }
  cudaFree(c->out_norm); cudaFree(c->logits);
  cudaFree(c->red_max); cudaFree(c->red_idx); cudaFree(c->argmax_d);
  free(c);
}

int gemma4_cuda_forward(Gemma4Cuda* c, int32_t token, size_t pos,
                        float* logits_out, int32_t* argmax_out,
                        float* hidden_out) {
  const Gemma4Model* m = c->m;
  const int h = (int)m->hidden;
  char* err = NULL; size_t errlen = 0; /* CUDA_TRY prints via the return path */
  if (pos >= m->ctx) {
    fprintf(stderr, "cuda: position %zu exceeds context %zu\n", pos, m->ctx);
    return -1;
  }

  int cur = 0;
  cudaSetDevice(0);
  cudaStream_t s = c->stream[0];

  /* embedding lookup * sqrt(hidden) */
  size_t tk = (size_t)token < m->vocab ? (size_t)token : m->vocab - 1;
  size_t emb_row = oc_row_bytes(m->tok_embd->ggml_type, m->hidden);
  embed(s, m->tok_embd->ggml_type, c->gpu[0].tok_embd + tk * emb_row, h,
        m->emb_scale, c->gpu[0].x);

  for (size_t l = 0; l < c->n_gpu_layers; ++l) {
    const Gemma4Layer* L = &m->layers[l];
    const CudaLayer* D = &c->layers[l];
    if (D->dev != cur) {
      /* pipeline hand-off: copy the hidden state to the next GPU once */
      int prev = cur;
      /* don't clobber dst's x while it is still finishing the previous
       * token's segment (matters during fully-async prefill) */
      cudaStreamWaitEvent(s, c->ev_done[D->dev], 0);
      cudaMemcpyPeerAsync(c->gpu[D->dev].x, D->dev, c->gpu[prev].x, prev,
                          (size_t)h * 4, s);
      cudaEventRecord(c->ev[prev], s);
      cudaEventRecord(c->ev_done[prev], s); /* prev's segment is over */
      cur = D->dev;
      cudaSetDevice(cur);
      s = c->stream[cur];
      cudaStreamWaitEvent(s, c->ev[prev], 0);
    }
    GpuScratch* S = &c->gpu[cur];
    int hd = (int)L->head_dim, vd = (int)L->v_head_dim;
    int n_head = (int)m->n_head, n_kv = (int)L->n_kv_heads;
    int q_len = n_head * hd, k_len = n_kv * hd, v_len = n_kv * vd;
    int group = n_head / n_kv;

    /* ---- attention ---- */
    k_rmsnorm<<<1, 256, 0, s>>>(S->normed, S->x, D->attn_norm, h, m->eps);
    matvec(s, L->attn_q->ggml_type, D->q_w, q_len, h, S->normed, S->q, 1);
    matvec(s, L->attn_k->ggml_type, D->k_w, k_len, h, S->normed, S->k, 1);
    if (D->v_w)
      matvec(s, L->attn_v->ggml_type, D->v_w, v_len, h, S->normed, S->v, 1);
    else /* K=V layers: V is the RAW K projection — copied BEFORE k_norm/rope */
      cudaMemcpyAsync(S->v, S->k, (size_t)k_len * 4, cudaMemcpyDeviceToDevice, s);

    /* per-head Q/K RMSNorm then rope; V gets a SCALE-LESS RMSNorm and no rope */
    if (D->q_norm) k_rmsnorm<<<n_head, 256, 0, s>>>(S->q, S->q, D->q_norm, hd, m->eps);
    if (D->k_norm) k_rmsnorm<<<n_kv, 256, 0, s>>>(S->k, S->k, D->k_norm, hd, m->eps);

    int rope_len = L->rope.rope_dim ? (int)L->rope.rope_dim : hd;
    const float* freqs = L->is_swa ? NULL : S->rope_freqs; /* global layers only */
    if (pos > 0 && rope_len > 0) {
      k_rope<<<n_head, 128, 0, s>>>(S->q, hd, (int)pos, L->rope.theta, rope_len,
                                    freqs);
      k_rope<<<n_kv, 128, 0, s>>>(S->k, hd, (int)pos, L->rope.theta, rope_len,
                                  freqs);
    }
    k_rmsnorm<<<n_kv, 256, 0, s>>>(S->v, S->v, NULL, vd, m->eps);

    size_t slot = pos % L->cache_cap;
    size_t seq = pos + 1;
    size_t t0 = seq > L->cache_cap ? seq - L->cache_cap : 0;
    float scale = m->attn_scale > 0.0f ? m->attn_scale : 1.0f / sqrtf((float)hd);
    size_t smem = ((size_t)hd + (seq - t0) + 256) * 4;
    if (m->kv_quant) {
      /* rotoquant: rotate q/k/v heads, store int4, attend over int4; the
       * attention output stays rotated until the final k_fht undoes it. */
      k_fht<<<n_head, 128, (size_t)hd * 4, s>>>(S->q, hd);
      k_fht<<<n_kv, 128, (size_t)hd * 4, s>>>(S->k, hd);
      k_fht<<<n_kv, 128, (size_t)vd * 4, s>>>(S->v, vd);
      k_kv_store_q4<<<n_kv, 128, 0, s>>>(D->k_qcache, D->k_qmeta, D->v_qcache,
                                         D->v_qmeta, S->k, S->v, hd, vd, n_kv,
                                         slot);
      k_attn_q4<<<n_head, 128, smem, s>>>(S->attn_res, S->q, D->k_qcache,
                                          D->k_qmeta, D->v_qcache, D->v_qmeta,
                                          hd, vd, group, (int)L->cache_cap,
                                          (int)t0, (int)seq, scale);
      k_fht<<<n_head, 128, (size_t)vd * 4, s>>>(S->attn_res, vd);
    } else {
      int kv_max = k_len > v_len ? k_len : v_len;
      k_kv_store<<<(kv_max + 255) / 256, 256, 0, s>>>(D->k_cache, D->v_cache,
                                                      S->k, S->v, k_len, v_len, slot);
      k_attn<<<n_head, 128, smem, s>>>(S->attn_res, S->q, D->k_cache, D->v_cache,
                                       hd, vd, group, (int)L->cache_cap, (int)t0,
                                       (int)seq, scale);
    }

    matvec(s, L->attn_out->ggml_type, D->o_w, h, n_head * vd, S->attn_res,
           S->attn_proj, 1);
    if (D->post_attn_norm)
      k_rmsnorm<<<1, 256, 0, s>>>(S->attn_proj, S->attn_proj, D->post_attn_norm,
                                  h, m->eps);
    k_add<<<(h + 255) / 256, 256, 0, s>>>(S->attn_proj, S->x, h); /* + residual */

    /* ---- FFN (GeGLU) over attn_out ---- */
    k_rmsnorm<<<1, 256, 0, s>>>(S->normed, S->attn_proj, D->ffn_norm, h, m->eps);
    matvec(s, L->ffn_gate->ggml_type, D->gate_w, (int)m->inter, h, S->normed,
           S->gate, 1);
    matvec(s, L->ffn_up->ggml_type, D->up_w, (int)m->inter, h, S->normed, S->up, 1);
    k_geglu<<<((int)m->inter + 255) / 256, 256, 0, s>>>(S->gate, S->up,
                                                        (int)m->inter);
    matvec(s, L->ffn_down->ggml_type, D->down_w, h, (int)m->inter, S->gate,
           S->ffn_out, 1);
    if (D->post_ffn_norm)
      k_rmsnorm<<<1, 256, 0, s>>>(S->ffn_out, S->ffn_out, D->post_ffn_norm, h,
                                  m->eps);
    /* x = (post_ffw_norm(ffn_out) + attn_out) * layer_output_scale */
    k_resid_out<<<(h + 255) / 256, 256, 0, s>>>(S->x, S->ffn_out, S->attn_proj,
                                                L->output_scale, h);
  }

  cudaEventRecord(c->ev_done[cur], s); /* last GPU's segment is over */
  GpuScratch* S = &c->gpu[cur];

  if (hidden_out) { /* partial offload: hand the residual stream to the CPU */
    CUDA_TRY(cudaMemcpyAsync(hidden_out, S->x, (size_t)h * 4,
                             cudaMemcpyDeviceToHost, s));
    cudaError_t e = cudaStreamSynchronize(s);
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e != cudaSuccess) {
      fprintf(stderr, "cuda: forward failed: %s\n", cudaGetErrorString(e));
      return -1;
    }
    return 0;
  }

  if (!logits_out && !argmax_out) {
    /* prefill: fully async, no sync. Catch sticky launch errors cheaply. */
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
      fprintf(stderr, "cuda: launch error: %s\n", cudaGetErrorString(e));
      return -1;
    }
    return 0;
  }

  /* final norm + tied-embedding logits on the last GPU */
  k_rmsnorm<<<1, 256, 0, s>>>(S->normed, S->x, c->out_norm, h, m->eps);
  matvec(s, m->tok_embd->ggml_type, S->tok_embd, (int)m->vocab, h, S->normed,
         c->logits, 1);

  if (argmax_out) {
    /* tanh softcap is monotonic: argmax over raw logits is identical */
    k_argmax_stage1<<<ARGMAX_BLOCKS, 256, 0, s>>>(c->logits, (int)m->vocab,
                                                  c->red_max, c->red_idx);
    k_argmax_stage2<<<1, 256, 0, s>>>(c->red_max, c->red_idx, ARGMAX_BLOCKS,
                                      c->argmax_d);
    CUDA_TRY(cudaMemcpyAsync(argmax_out, c->argmax_d, 4, cudaMemcpyDeviceToHost, s));
  }
  if (logits_out) {
    if (m->final_softcap > 0.0f)
      k_softcap<<<((int)m->vocab + 255) / 256, 256, 0, s>>>(
          c->logits, m->final_softcap, (int)m->vocab);
    CUDA_TRY(cudaMemcpyAsync(logits_out, c->logits, m->vocab * 4,
                             cudaMemcpyDeviceToHost, s));
  }
  cudaError_t e = cudaStreamSynchronize(s); /* the ONE sync per token */
  if (e == cudaSuccess) e = cudaGetLastError();
  if (e != cudaSuccess) {
    fprintf(stderr, "cuda: forward failed: %s\n", cudaGetErrorString(e));
    return -1;
  }
  return 0;
}

float* gemma4_cuda_step(Gemma4Cuda* c, Gemma4Model* m, int32_t token, size_t pos,
                        bool need_logits, int* failed) {
  *failed = 0;
  if (c->n_gpu_layers == m->n_layers) { /* whole stack on the GPU */
    float* lg = need_logits ? m->logits : NULL;
    if (gemma4_cuda_forward(c, token, pos, lg, NULL, NULL) != 0) {
      *failed = 1;
      return NULL;
    }
    m->kv_len = pos + 1;
    return lg;
  }
  /* partial offload: GPU layers [0, ngl), then the CPU finishes from m->x */
  if (gemma4_cuda_forward(c, token, pos, NULL, NULL, m->x) != 0) {
    *failed = 1;
    return NULL;
  }
  return gemma4_forward_from(m, pos, c->n_gpu_layers, need_logits);
}
