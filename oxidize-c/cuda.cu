/* Resident CUDA forward for oxidize-c: the WHOLE per-token forward runs on the
 * GPU. Only the token embedding is copied in (H2D) and the logits copied out
 * (D2H) — one transfer each per token, vs ~450 in the old per-op offload. Every
 * op (RMSNorm, RoPE+YaRN, GQA attention, gated-DeltaNet recurrence, SwiGLU,
 * gated norms, residual adds) is a CUDA kernel; matmuls use cuBLAS with FP16
 * weights (tensor cores) and FP32 accumulation. The residual stream, KV cache
 * and SSM state all stay device-resident.
 *
 * Numerics: kernels port the exact CPU math in model.c/kernels.c. Weights are
 * FP16 (vs the CPU's Q4_K), so logits differ by ~1% — fine for greedy argmax.
 * Build: `make cuda` (nvcc + -DOC_CUDA). */
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "oc.h"
}

#define CONV_K 4
#define CDIE(call)                                                       \
  do {                                                                   \
    cudaError_t e_ = (call);                                            \
    if (e_ != cudaSuccess) {                                            \
      fprintf(stderr, "cuda fatal: %s at %s:%d\n", cudaGetErrorString(e_), \
              __FILE__, __LINE__);                                       \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

namespace {

struct GpuW {           /* weight matrix [rows x cols] row-major on device:
                         * either FP16 dense (d) or raw quant blocks (qd) with a
                         * fused dequant-GEMV kernel (IQ4_XS / Q4_K) */
  __half *d = nullptr;
  uint8_t *qd = nullptr;
  int quant = -1;
  int rows = 0, cols = 0;
};

struct GpuLayer {
  bool is_gdn = false;
  float *attn_norm = nullptr, *ffn_norm = nullptr;
  /* attention */
  GpuW wq, wk, wv, wo;
  float *q_norm = nullptr, *k_norm = nullptr;  /* per-head, may be null */
  int kv_slot = -1;
  /* per-layer geometry (gemma4 dual attention) */
  int hd = 0, n_kv = 0, n_rot = 0;
  float theta = 0.0f, attn_scale = 0.0f;      /* attn_scale 0 = 1/sqrt(hd) */
  const float *rope_ff = nullptr;             /* device freq divisors or null */
  bool v_from_k = false, v_rms = false;
  float *attn_post_norm = nullptr, *ffn_post_norm = nullptr;
  float out_scale = 1.0f;
  float *my_kv_k = nullptr, *my_kv_v = nullptr; /* private cache (gemma) */
  int my_kv_cap = 0;
  /* gdn */
  GpuW qkv, ssm_alpha, ssm_beta, ssm_out, gdn_gate;
  float *ssm_a = nullptr, *ssm_dt_bias = nullptr, *ssm_conv1d = nullptr,
        *ssm_norm = nullptr;
  float *state = nullptr, *conv_ring = nullptr;
  int ring_head = 0, ring_len = 0;
  int qkv_out = 0, value_dim = 0, key_dim = 0;
  int n_v_heads = 0, n_k_heads = 0, head_k = 0, head_v = 0;
  bool a_baked = true;
  /* shared FFN */
  GpuW gate, up, down;
};

struct GpuCtx {
  cublasHandle_t cublas;
  int h, n_heads, kv_heads, head_dim, rope_dim, inter, vocab, n_layers;
  float rms_eps, rope_theta, yarn_factor, yarn_orig_ctx;
  bool gemma = false;
  float logit_softcap = 0.0f;
  float *rope_freqs = nullptr;               /* device, shared by full layers */
  int kv_ctx, kv_stride;
  GpuLayer *layers;
  GpuW lm_head;
  float *final_norm;
  float *kv_k, *kv_v;                 /* device [n_kv_layers][kv_ctx][kv_stride] */
  /* scratch (device) */
  float *d_x, *d_norm, *d_qg, *d_q, *d_gate, *d_k, *d_v, *d_attn, *d_tmp;
  float *d_gf, *d_uf;                 /* ffn gate/up */
  float *d_mixed, *d_conv, *d_a, *d_b, *d_z, *d_core, *d_logits;
  __half *d_xh;                       /* fp16 matmul input staging (max width) */
  float *aq_d, *aq_s;                 /* int8 activation blocks (per 32 values) */
  int *aq_q;                          /* packed q8, 8 ints per block */
  cudaStream_t stream;
  int *d_pos;                         /* current absolute position (device) */
  cudaGraphExec_t graph[2] = {nullptr, nullptr};   /* [want_logits] */
  float *h_pin;                       /* pinned host staging (embedding / logits) */
  size_t h_pin_cap;
};

GpuCtx *g_ctx = nullptr;

/* ---------- kernels ---------- */

__global__ void k_f32_to_f16(__half *dst, const float *src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void k_rms_norm(float *out, const float *x, const float *w, int n,
                           float eps) {
  __shared__ float red[256];
  float s = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
  red[threadIdx.x] = s;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * inv * w[i];
}

/* per-head RMSNorm (Qwen3 q/k norm): one block per head, over head_dim */
__global__ void k_head_rms_norm(float *v, const float *w, int hd, float eps) {
  __shared__ float red[256];
  float *hp = v + blockIdx.x * hd;
  float s = 0.0f;
  for (int i = threadIdx.x; i < hd; i += blockDim.x) s += hp[i] * hp[i];
  red[threadIdx.x] = s;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)hd + eps);
  for (int i = threadIdx.x; i < hd; i += blockDim.x)
    hp[i] = hp[i] * inv * (w ? w[i] : 1.0f);
}

__global__ void k_add(float *x, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += b[i];
}

__global__ void k_swiglu(float *gate, const float *up, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = gate[i];
    gate[i] = g / (1.0f + expf(-g)) * up[i];
  }
}

__global__ void k_geglu(float *gate, const float *up, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = gate[i];
    const float kk = 0.7978845608028654f;
    gate[i] = 0.5f * g * (1.0f + tanhf(kk * (g + 0.044715f * g * g * g))) * up[i];
  }
}

__global__ void k_scale(float *x, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= s;
}

__global__ void k_softcap(float *x, float cap, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

__global__ void k_copy(float *dst, const float *srcv, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = srcv[i];
}

__global__ void k_sigmoid_gate(float *x, const float *gate, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

/* de-interleave per-head [q(hd)|gate(hd)] -> q[heads*hd], gate[heads*hd] */
__global__ void k_deinterleave(const float *qg, float *q, float *gate, int heads,
                               int hd) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= heads * hd) return;
  int hh = idx / hd, d = idx % hd;
  q[idx] = qg[hh * 2 * hd + d];
  gate[idx] = qg[hh * 2 * hd + hd + d];
}

/* RoPE (NeoX split-half, partial) with optional YaRN. One thread per
 * (head, i<half). Ports oc_rope. */
__global__ void k_rope(float *vec, int head_dim, int n_heads,
                       const int *__restrict__ posp, float theta, int rope_dim,
                       float yf, float yorig, const float *ff) {
  int pos = *posp;
  int rl = (rope_dim == 0 || rope_dim > head_dim) ? head_dim : rope_dim;
  int half = rl / 2;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_heads * half) return;
  int hh = idx / half, i = idx % half;
  bool yarn = yf > 1.0f && yorig > 0.0f;
  if (!yarn && pos == 0 && !ff) return;
  float freq = powf(theta, -2.0f * (float)i / (float)rl);
  if (ff) freq /= ff[i];
  float ang = (float)pos * freq, c, s;
  if (yarn) {
    float fs = 1.0f / yf, msc = 1.0f + 0.1f * logf(yf);
    float clo = floorf((float)rl * logf(yorig / (32.0f * 2.0f * (float)M_PI)) /
                       (2.0f * logf(theta)));
    float chi = ceilf((float)rl * logf(yorig / (1.0f * 2.0f * (float)M_PI)) /
                      (2.0f * logf(theta)));
    if (clo < 0) clo = 0;
    if (chi > (float)rl - 1) chi = (float)rl - 1;
    float te = (float)pos * freq, ti = te * fs;
    float denom = chi - clo;
    float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, ((float)i - clo) /
                                          (denom > 0.001f ? denom : 0.001f)));
    ang = ti * (1.0f - ramp) + te * ramp;
    c = cosf(ang) * msc;
    s = sinf(ang) * msc;
  } else {
    c = cosf(ang);
    s = sinf(ang);
  }
  float *hp = vec + hh * head_dim;
  float x0 = hp[i], x1 = hp[half + i];
  hp[i] = x0 * c - x1 * s;
  hp[half + i] = x0 * s + x1 * c;
}

/* append current k/v (kv_heads*head_dim) to the device KV cache at slot */
__global__ void k_kv_append(float *kv_k, float *kv_v, const float *k,
                            const float *v, const int *__restrict__ posp,
                            int kv_ctx, int kvn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kvn) return;
  int base = (*posp % kv_ctx) * kvn;
  kv_k[base + i] = k[i];
  kv_v[base + i] = v[i];
}

/* GQA attention decode, online softmax. One block per query head (head_dim
 * threads). Ports oc_attention. kv_k/kv_v point at this layer's slice. */
/* GQA attention decode: one block (4 warps) per query head. Warps take
 * interleaved position chunks with private online-softmax state, merged at
 * the end (flash-decoding style). Lanes own head_dim/32 dims. */
__global__ void k_attention(float *out, const float *q, const float *kv_k,
                            const float *kv_v, const int *__restrict__ posp,
                            int cap, int n_heads, int kv_heads, int head_dim,
                            float scale) {
  int seq_len = *posp + 1;
  if (seq_len > cap) seq_len = cap;
  int head = blockIdx.x;
  if (head >= n_heads) return;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int group = n_heads / kv_heads;
  int kv_off = (head / group) * head_dim;
  int kvn = kv_heads * head_dim;
  int per = head_dim / 32;             /* dims per lane (8 or 16) */
  const float *qh = q + head * head_dim + lane * per;
  if (scale == 0.0f) scale = rsqrtf((float)head_dim);
  float qreg[16], acc[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
  for (int j = 0; j < per; ++j) qreg[j] = qh[j];
  float rmax = -1e30f, rsum = 0.0f;
  for (int t = warp; t < seq_len; t += 4) {
    const float *kr = kv_k + (size_t)t * kvn + kv_off + lane * per;
    float sc = 0.0f;
    for (int j = 0; j < per; ++j) sc += qreg[j] * kr[j];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      sc += __shfl_xor_sync(0xFFFFFFFF, sc, off);
    sc *= scale;
    float nm = rmax > sc ? rmax : sc;
    float f = __expf(rmax - nm), es = __expf(sc - nm);
    const float *vr = kv_v + (size_t)t * kvn + kv_off + lane * per;
    for (int j = 0; j < per; ++j) acc[j] = acc[j] * f + es * vr[j];
    rsum = rsum * f + es;
    rmax = nm;
  }
  /* merge the 4 warp-partials */
  __shared__ float sm[4], ss[4];
  __shared__ float sacc[4][512];
  sm[warp] = rmax;
  ss[warp] = rsum;
  for (int j = 0; j < per; ++j) sacc[warp][lane * per + j] = acc[j];
  __syncthreads();
  if (warp == 0) {
    float m = sm[0];
#pragma unroll
    for (int w = 1; w < 4; ++w) m = fmaxf(m, sm[w]);
    float fw[4], sum = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      fw[w] = __expf(sm[w] - m);
      sum += ss[w] * fw[w];
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    float *oh = out + head * head_dim + lane * per;
    for (int j = 0; j < per; ++j) {
      float o = 0.0f;
#pragma unroll
      for (int w = 0; w < 4; ++w) o += sacc[w][lane * per + j] * fw[w];
      oh[j] = o * inv;
    }
  }
}

/* ---- gated-DeltaNet kernels ---- */

/* causal conv (ring) + SiLU over qkv_out channels, single token. One thread per
 * channel. Ring frames are device memory; ring_head/len passed as args. */
__global__ void k_conv_silu(const float *mixed, float *conv, const float *w,
                            const float *ring, int qo, int ring_head,
                            int ring_len) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= qo) return;
  float sum = w[(CONV_K - 1) * qo + ch] * mixed[ch];
  for (int back = 1; back < CONV_K; ++back) {
    if (back > ring_len) break;
    int idx = (ring_head + CONV_K - back) % CONV_K;
    sum += w[(CONV_K - 1 - back) * qo + ch] * ring[(size_t)idx * qo + ch];
  }
  conv[ch] = sum / (1.0f + expf(-sum));  /* silu */
}

__global__ void k_ring_push(float *ring, const float *mixed, int qo,
                            int ring_head) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch < qo) ring[(size_t)ring_head * qo + ch] = mixed[ch];
}

/* delta-rule recurrence, single token. One block per v_head, head_v threads
 * (each owns state column j). Ports gdn_layer inner loop. */
__global__ void k_delta_rule(const float *conv, float *state, float *core,
                             const float *a_all, const float *b_all,
                             const float *ssm_a, const float *ssm_dt,
                             int qo, int kd, int vd, int nkh, int hk, int hv,
                             bool a_baked, float out_scale) {
  int vh = blockIdx.x;
  int j = threadIdx.x;                 /* head_v element */
  if (j >= hv) return;
  int k_head = vh % nkh;
  int q_off = k_head * hk, k_off = kd + k_head * hk, v_off = kd * 2 + vh * hv;
  float *st = state + (size_t)vh * hk * hv;
  extern __shared__ float sh[];        /* [hk] q, [hk] k */
  float *sq = sh, *sk = sh + hk;
  /* load q,k (head_k dims); threads j cover hk if hk<=blockDim (hk==hv here) */
  if (j < hk) { sq[j] = conv[q_off + j]; sk[j] = conv[k_off + j]; }
  __syncthreads();
  /* l2norm q,k (reduction over hk) using shared scratch */
  __shared__ float rq, rk;
  __shared__ float red[256];
  red[j] = j < hk ? sq[j] * sq[j] : 0.0f;
  __syncthreads();
  for (int st_ = blockDim.x / 2; st_ > 0; st_ >>= 1) {
    if (j < st_) red[j] += red[j + st_];
    __syncthreads();
  }
  if (j == 0) rq = 1.0f / fmaxf(sqrtf(red[0]), 1e-6f);
  __syncthreads();
  red[j] = j < hk ? sk[j] * sk[j] : 0.0f;
  __syncthreads();
  for (int st_ = blockDim.x / 2; st_ > 0; st_ >>= 1) {
    if (j < st_) red[j] += red[j + st_];
    __syncthreads();
  }
  if (j == 0) rk = 1.0f / fmaxf(sqrtf(red[0]), 1e-6f);
  __syncthreads();
  if (j < hk) { sq[j] *= rq; sk[j] *= rk; }
  __syncthreads();

  float beta = 1.0f / (1.0f + expf(-b_all[vh]));
  float a_val = a_all[vh];
  float dtb = ssm_dt ? ssm_dt[vh] : 0.0f;
  float x = a_val + dtb;
  float dt = x > 20.0f ? x : logf(1.0f + expf(x));
  float g = a_baked ? ssm_a[vh] * dt : -expf(ssm_a[vh]) * dt;
  float decay = expf(g);

  /* thread j owns column j: state[i*hv + j], i in [0,hk) */
  float vj = conv[v_off + j];
  /* decay + kv_mem[j] = sum_i state[i*hv+j]*k[i] */
  float kv_mem = 0.0f;
  for (int i = 0; i < hk; ++i) {
    float sij = st[i * hv + j] * decay;
    st[i * hv + j] = sij;
    kv_mem += sij * sk[i];
  }
  float delta = (vj - kv_mem) * beta;
  /* state[i*hv+j] += k[i]*delta ; out[j] = sum_i state[i*hv+j]*q[i] */
  float outj = 0.0f;
  for (int i = 0; i < hk; ++i) {
    float sij = st[i * hv + j] + sk[i] * delta;
    st[i * hv + j] = sij;
    outj += sij * sq[i];
  }
  core[vh * hv + j] = outj * out_scale;
}

__global__ void k_gated_rms_norm(float *core, const float *w, const float *z,
                                 int hv, float eps, int gate_after) {
  __shared__ float red[256];
  float *cp = core + blockIdx.x * hv;
  const float *zp = z + blockIdx.x * hv;
  int j = threadIdx.x;
  float s = 0.0f;
  for (int i = j; i < hv; i += blockDim.x) {
    float value = cp[i];
    if (!gate_after) {
      float zg = zp[i];
      value *= zg / (1.0f + expf(-zg));
    }
    s += value * value;
  }
  red[j] = s;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (j < st) red[j] += red[j + st];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)hv + eps);
  for (int i = j; i < hv; i += blockDim.x) {
    float zg = zp[i];
    float gate = zg / (1.0f + expf(-zg));
    cp[i] = cp[i] * (gate_after ? gate : 1.0f) * inv * w[i];
  }
}

/* ---- fused dequant-GEMV kernels: weights stay 4-bit resident on device.
 * One block per output row; each thread accumulates 32-value sub-blocks,
 * then a shared-memory tree reduction. Bandwidth-bound by design. ---- */


/* quantize fp32 activations to int8 blocks of 32 (mirrors oc_quantize_act):
 * one thread per block-of-32. d = amax/127, s = d*sum(q). q packed 4/int. */
__global__ void k_quant_act(const float *__restrict__ x, float *__restrict__ qd,
                            float *__restrict__ qs, int *__restrict__ qq, int nb) {
  int bidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bidx >= nb) return;
  const float *xb = x + bidx * 32;
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(xb[i]));
  float d = amax / 127.0f;
  float id = d != 0.0f ? 1.0f / d : 0.0f;
  int sum = 0;
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    unsigned packed = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int q = __float2int_rn(xb[w * 4 + j] * id);
      q = max(-127, min(127, q));
      sum += q;
      packed |= (unsigned)(q & 0xFF) << (8 * j);
    }
    qq[bidx * 8 + w] = (int)packed;
  }
  qd[bidx] = d;
  qs[bidx] = d * (float)sum;
}

/* warp-per-row int8 GEMV, dp4a integer dots. 4 warps/block = 4 rows. */
__constant__ float c_kv_iq4nl[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f,
                                     -22.f,  -10.f,  1.f,   13.f,  25.f,  38.f,
                                     53.f,   69.f,   89.f,  113.f};

__global__ void k_gemv_iq4xs(const uint8_t *__restrict__ W,
                             const float *__restrict__ qd,
                             const float *__restrict__ qs,
                             const int *__restrict__ qq, float *__restrict__ y,
                             int rows, int cols) {
  __shared__ int lut[16];             /* kvalues as int8 in low byte */
  if (threadIdx.x < 16) lut[threadIdx.x] = (int)c_kv_iq4nl[threadIdx.x] & 0xFF;
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int r = blockIdx.x * 8 + warp;
  if (r >= rows) return;
  const uint8_t *row = W + (size_t)r * (cols / 256) * 136;
  int nsub = cols / 32;
  float acc = 0.0f;
  /* two consecutive sub-blocks per lane: 32B contiguous weight reads and two
   * independent dp4a chains to hide DRAM latency */
  for (int i0 = lane * 2; i0 < nsub; i0 += 64) {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      int i = i0 + k;
      if (i >= nsub) break;
      int sb = i >> 3, ib = i & 7;
      const uint8_t *blk = row + (size_t)sb * 136;
      float d = __half2float(*(const __half *)blk);
      uint16_t shl = *(const uint16_t *)(blk + 2);
      int ls = ((blk[4 + ib / 2] >> (4 * (ib & 1))) & 0xF) |
               (((shl >> (2 * ib)) & 3) << 4);
      const uint2 qa = *(const uint2 *)(blk + 8 + ib * 16);
      const uint2 qb = *(const uint2 *)(blk + 8 + ib * 16 + 8);
      unsigned wds[4] = {qa.x, qa.y, qb.x, qb.y};
      const int *xq = qq + i * 8;
      int isum = 0;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        unsigned wv = wds[w];
        int lo = lut[wv & 0xF] | (lut[(wv >> 8) & 0xF] << 8) |
                 (lut[(wv >> 16) & 0xF] << 16) | (lut[(wv >> 24) & 0xF] << 24);
        int hi = lut[(wv >> 4) & 0xF] | (lut[(wv >> 12) & 0xF] << 8) |
                 (lut[(wv >> 20) & 0xF] << 16) | (lut[(wv >> 28) & 0xF] << 24);
        isum = __dp4a(lo, xq[w], isum);
        isum = __dp4a(hi, xq[4 + w], isum);
      }
      acc += d * (float)(ls - 32) * qd[i] * (float)isum;
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
  if (lane == 0) y[r] = acc;
}

__device__ inline void d_scale_min_k4(int j, const uint8_t *s, int *sc, int *m) {
  if (j < 4) {
    *sc = s[j] & 63;
    *m = s[j + 4] & 63;
  } else {
    *sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
    *m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
  }
}

__global__ void k_gemv_q4k(const uint8_t *__restrict__ W,
                           const float *__restrict__ qd,
                           const float *__restrict__ qs,
                           const int *__restrict__ qq, float *__restrict__ y,
                           int rows, int cols) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int r = blockIdx.x * 8 + warp;
  if (r >= rows) return;
  const uint8_t *row = W + (size_t)r * (cols / 256) * 144;
  int nsub = cols / 32;
  float acc = 0.0f;
  for (int i = lane; i < nsub; i += 32) {
    int sb = i >> 3, g = i & 7;
    const uint8_t *blk = row + (size_t)sb * 144;
    float d = __half2float(*(const __half *)blk);
    float mn = __half2float(*(const __half *)(blk + 2));
    int sc, m;
    d_scale_min_k4(g, blk + 4, &sc, &m);
    const uint8_t *qsrc = blk + 16 + (g / 2) * 32;
    const int *xq = qq + i * 8;
    int shift = (g & 1) * 4;
    int isum = 0;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      uint2 qv = *(const uint2 *)(qsrc + w * 8);
      unsigned lo0 = (qv.x >> shift) & 0x0F0F0F0F;
      unsigned lo1 = (qv.y >> shift) & 0x0F0F0F0F;
      isum = __dp4a((int)lo0, xq[w * 2], isum);
      isum = __dp4a((int)lo1, xq[w * 2 + 1], isum);
    }
    acc += d * (float)sc * qd[i] * (float)isum - mn * (float)m * qs[i];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
  if (lane == 0) y[r] = acc;
}

/* ---------- matmul helper: y[rows] = W[rows x cols] @ x[cols], device fp32 ---- */
double g_prof[8];          /* 0 qkvo 1 ffn 2 head 3 attn 4 other */
bool g_prof_on = false;
cudaStream_t g_prof_stream = 0;

struct ProfScope {
  int slot;
  cudaEvent_t e0, e1;
  ProfScope(int s) : slot(s) {
    if (!g_prof_on) return;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, g_prof_stream);
  }
  ~ProfScope() {
    if (!g_prof_on) return;
    cudaEventRecord(e1, g_prof_stream);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    g_prof[slot] += ms;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
  }
};

void gemv(GpuCtx *c, const GpuW &w, const float *d_in, float *d_out,
          bool requant = true) {
  if (w.qd) {
    int slot = w.rows >= 200000 ? 2 : (w.cols > 16000 || w.rows > 16000) ? 1 : 0;
    ProfScope ps(slot);
    int nb = w.cols / 32;
    if (requant)
      k_quant_act<<<(nb + 255) / 256, 256, 0, c->stream>>>(d_in, c->aq_d, c->aq_s, c->aq_q, nb);
    int grid = (w.rows + 7) / 8;
    if (w.quant == OC_IQ4_XS)
      k_gemv_iq4xs<<<grid, 256, 0, c->stream>>>(w.qd, c->aq_d, c->aq_s, c->aq_q, d_out,
                                  w.rows, w.cols);
    else
      k_gemv_q4k<<<grid, 256, 0, c->stream>>>(w.qd, c->aq_d, c->aq_s, c->aq_q, d_out,
                                w.rows, w.cols);
    return;
  }
  int n = w.cols;
  int blk = 256, grid = (n + blk - 1) / blk;
  k_f32_to_f16<<<grid, blk, 0, c->stream>>>(c->d_xh, d_in, n);
  const float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st = cublasGemmEx(
      c->cublas, CUBLAS_OP_T, CUBLAS_OP_N, w.rows, 1, w.cols, &alpha, w.d,
      CUDA_R_16F, w.cols, c->d_xh, CUDA_R_16F, w.cols, &beta, d_out, CUDA_R_32F,
      w.rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (st != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublas gemv failed: %d (rows=%d cols=%d)\n", (int)st,
            w.rows, w.cols);
    exit(1);
  }
}

/* ---------- upload helpers ---------- */

/* dequantize a host oc_weight to fp32 in chunks, convert to fp16, upload */
__half *upload_weight_fp16(const oc_weight *w) {
  if (!w->quantized && !w->f32) return nullptr;
  size_t elems = w->rows * w->cols;
  __half *d = nullptr;
  CDIE(cudaMalloc(&d, elems * sizeof(__half)));
  size_t chunk_rows = ((size_t)16 << 20) / (w->cols ? w->cols : 1) + 1;
  float *tmp = (float *)malloc(chunk_rows * w->cols * sizeof(float));
  __half *stage = (__half *)malloc(chunk_rows * w->cols * sizeof(__half));
  size_t rb = w->quantized ? oc_row_bytes(w->quant, w->cols) : 0;
  for (size_t r0 = 0; r0 < w->rows; r0 += chunk_rows) {
    size_t nr = w->rows - r0 < chunk_rows ? w->rows - r0 : chunk_rows;
    size_t ne = nr * w->cols;
    if (w->quantized)
      oc_dequant_row(w->quant, w->data + r0 * rb, tmp, ne);
    else
      memcpy(tmp, w->f32 + r0 * w->cols, ne * sizeof(float));
    for (size_t i = 0; i < ne; ++i) stage[i] = __float2half(tmp[i]);
    CDIE(cudaMemcpy(d + r0 * w->cols, stage, ne * sizeof(__half),
                    cudaMemcpyHostToDevice));
  }
  free(tmp);
  free(stage);
  return d;
}

float *upload_fp32(const float *host, size_t n) {
  if (!host || n == 0) return nullptr;
  float *d = nullptr;
  CDIE(cudaMalloc(&d, n * sizeof(float)));
  CDIE(cudaMemcpy(d, host, n * sizeof(float), cudaMemcpyHostToDevice));
  return d;
}

GpuW mkw(const oc_weight *w) {
  GpuW g;
  if (!w->quantized && !w->f32) return g;
  g.rows = (int)w->rows;
  g.cols = (int)w->cols;
  if (w->quantized && (w->quant == OC_IQ4_XS || w->quant == OC_Q4_K) &&
      w->cols % 256 == 0) {
    size_t bytes = w->rows * oc_row_bytes(w->quant, w->cols);
    CDIE(cudaMalloc(&g.qd, bytes));
    CDIE(cudaMemcpy(g.qd, w->data, bytes, cudaMemcpyHostToDevice));
    g.quant = (int)w->quant;
    return g;
  }
  g.d = upload_weight_fp16(w);
  return g;
}

}  // namespace

/* ================= public C API ================= */
extern "C" {

int oc_cuda_build(oc_model *m) {
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return -1;
  if (getenv("OC_NO_GPU")) return -1;

  GpuCtx *c = (GpuCtx *)calloc(1, sizeof(GpuCtx));
  if (cublasCreate(&c->cublas) != CUBLAS_STATUS_SUCCESS) { free(c); return -1; }
  cublasSetMathMode(c->cublas, CUBLAS_TF32_TENSOR_OP_MATH);
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  fprintf(stderr, "cuda: resident forward on %s (%.1f GB)\n", p.name,
          (double)p.totalGlobalMem / 1e9);

  const oc_config *cf = &m->cfg;
  c->h = (int)cf->hidden_size;
  c->n_heads = (int)cf->n_heads;
  c->kv_heads = (int)cf->kv_heads;
  c->head_dim = (int)cf->head_dim;
  c->rope_dim = (int)cf->rope_dim;
  c->inter = (int)cf->intermediate_size;
  c->vocab = (int)cf->vocab_size;
  c->n_layers = (int)cf->layer_count;
  c->rms_eps = cf->rms_eps;
  c->rope_theta = cf->rope_theta;
  c->yarn_factor = cf->yarn_factor;
  c->yarn_orig_ctx = cf->yarn_orig_ctx;
  c->kv_ctx = (int)m->kv_ctx;
  c->kv_stride = (int)m->kv_stride;
  c->gemma = m->gemma;
  c->logit_softcap = m->logit_softcap;
  if (m->rope_freqs && m->gemma) {
    /* full-attn freq divisors sized n_rot/2 (largest head_dim / 2 is enough) */
    c->rope_freqs = upload_fp32(m->rope_freqs, (size_t)cf->head_dim / 2);
  }

  c->layers = (GpuLayer *)calloc(c->n_layers, sizeof(GpuLayer));
  size_t wbytes = 0;
  for (int l = 0; l < c->n_layers; ++l) {
    oc_layer *L = &m->layers[l];
    GpuLayer *g = &c->layers[l];
    g->is_gdn = L->is_gdn;
    g->attn_norm = upload_fp32(L->attn_norm, c->h);
    g->ffn_norm = upload_fp32(L->ffn_norm, c->h);
    g->gate = mkw(&L->gate);
    g->up = mkw(&L->up);
    g->down = mkw(&L->down);
    if (L->is_gdn) {
      g->kv_slot = -1;
      g->qkv = mkw(&L->qkv);
      g->ssm_alpha = mkw(&L->ssm_alpha);
      g->ssm_beta = mkw(&L->ssm_beta);
      g->gdn_gate = mkw(&L->gdn_gate);
      g->ssm_out = mkw(&L->ssm_out);
      g->ssm_a = upload_fp32(L->ssm_a, L->n_v_heads);
      g->ssm_dt_bias = upload_fp32(L->ssm_dt_bias, L->n_v_heads);
      g->ssm_conv1d = upload_fp32(L->ssm_conv1d, CONV_K * L->qkv_out);
      g->ssm_norm = upload_fp32(L->ssm_norm, L->head_v);
      CDIE(cudaMalloc(&g->state,
                      (size_t)L->n_v_heads * L->head_k * L->head_v * sizeof(float)));
      CDIE(cudaMemset(g->state, 0,
                      (size_t)L->n_v_heads * L->head_k * L->head_v * sizeof(float)));
      CDIE(cudaMalloc(&g->conv_ring, (size_t)CONV_K * L->qkv_out * sizeof(float)));
      CDIE(cudaMemset(g->conv_ring, 0, (size_t)CONV_K * L->qkv_out * sizeof(float)));
      g->qkv_out = (int)L->qkv_out;
      g->value_dim = (int)L->value_dim;
      g->key_dim = (int)L->key_dim;
      g->n_v_heads = (int)L->n_v_heads;
      g->n_k_heads = (int)L->n_k_heads;
      g->head_k = (int)L->head_k;
      g->head_v = (int)L->head_v;
      g->a_baked = true;
      for (size_t i = 0; i < L->n_v_heads; ++i)
        if (L->ssm_a[i] > 0) { g->a_baked = false; break; }
    } else {
      g->kv_slot = L->kv_slot;
      g->wq = mkw(&L->wq);
      g->wk = mkw(&L->wk);
      g->wv = mkw(&L->wv);
      g->wo = mkw(&L->wo);
      g->hd = L->hd ? (int)L->hd : c->head_dim;
      g->n_kv = L->n_kv ? (int)L->n_kv : c->kv_heads;
      g->n_rot = L->n_rot ? (int)L->n_rot : c->rope_dim;
      g->theta = L->theta != 0.0f ? L->theta : c->rope_theta;
      g->attn_scale = L->attn_scale;
      g->v_from_k = L->v_from_k;
      g->v_rms = L->v_rms;
      g->out_scale = L->out_scale_v != 0.0f ? L->out_scale_v : 1.0f;
      g->rope_ff = L->rope_ff ? c->rope_freqs : nullptr;
      g->q_norm = upload_fp32(L->q_norm, g->hd);
      g->k_norm = upload_fp32(L->k_norm, g->hd);
      g->attn_post_norm = upload_fp32(L->attn_post_norm, c->h);
      g->ffn_post_norm = upload_fp32(L->ffn_post_norm, c->h);
      if (L->kv_ck) {   /* gemma: private per-layer cache on device */
        g->my_kv_cap = (int)L->kv_cap;
        size_t elems = (size_t)g->my_kv_cap * g->n_kv * g->hd;
        CDIE(cudaMalloc(&g->my_kv_k, elems * sizeof(float)));
        CDIE(cudaMalloc(&g->my_kv_v, elems * sizeof(float)));
      }
    }
  }
  c->final_norm = upload_fp32(m->final_norm, c->h);
  c->lm_head = mkw(&m->lm_head);

  /* KV cache (device); gemma uses per-layer private caches instead */
  if (!m->gemma) {
    size_t kv_elems = (size_t)m->n_kv_layers * c->kv_ctx * c->kv_stride;
    CDIE(cudaMalloc(&c->kv_k, kv_elems * sizeof(float)));
    CDIE(cudaMalloc(&c->kv_v, kv_elems * sizeof(float)));
  }

  /* scratch: size the widest matmul output */
  int wide = c->inter;
  if (c->vocab > wide) wide = c->vocab;
  int qwide = c->n_heads * c->head_dim * 2;  /* qg */
  int maxcol = c->h > c->inter ? c->h : c->inter;
  auto A = [&](float **p, size_t n) { CDIE(cudaMalloc(p, n * sizeof(float))); };
  A(&c->d_x, c->h);
  A(&c->d_norm, c->h);
  A(&c->d_qg, qwide);
  A(&c->d_q, c->n_heads * c->head_dim);
  A(&c->d_gate, c->n_heads * c->head_dim);
  A(&c->d_k, c->kv_stride);
  A(&c->d_v, c->kv_stride);
  A(&c->d_attn, c->n_heads * c->head_dim);
  A(&c->d_tmp, c->h);
  A(&c->d_gf, c->inter);
  A(&c->d_uf, c->inter);
  A(&c->d_mixed, 8192 * 2);
  A(&c->d_conv, 8192 * 2);
  A(&c->d_a, 256);
  A(&c->d_b, 256);
  A(&c->d_z, 8192);
  A(&c->d_core, 8192);
  A(&c->d_logits, c->vocab);
  CDIE(cudaMalloc(&c->d_xh, (size_t)(maxcol) * sizeof(__half)));
  {
    int nbmax = maxcol / 32 + 8;
    CDIE(cudaMalloc(&c->aq_d, nbmax * sizeof(float)));
    CDIE(cudaMalloc(&c->aq_s, nbmax * sizeof(float)));
    CDIE(cudaMalloc(&c->aq_q, nbmax * 8 * sizeof(int)));
  }
  CDIE(cudaStreamCreate(&c->stream));
  CDIE(cudaMalloc(&c->d_pos, sizeof(int)));
  g_prof_stream = c->stream;
  c->h_pin_cap = (size_t)c->h + (size_t)c->vocab;
  CDIE(cudaMallocHost(&c->h_pin, c->h_pin_cap * sizeof(float)));
  (void)wide;

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    fprintf(stderr, "cuda: build error %s\n", cudaGetErrorString(e));
    return -1;
  }
  double gb = 0;
  size_t freeb, totb;
  cudaMemGetInfo(&freeb, &totb);
  fprintf(stderr, "cuda: resident, %.1f GB free of %.1f GB\n",
          (double)freeb / 1e9, (double)totb / 1e9);
  (void)wbytes; (void)gb;
  g_ctx = c;
  m->gpu_active = true;
  return 0;
}

void oc_cuda_reset(oc_model *m) {
  (void)m;
  GpuCtx *c = g_ctx;
  if (!c) return;
  for (int l = 0; l < c->n_layers; ++l) {
    GpuLayer *g = &c->layers[l];
    if (!g->is_gdn) continue;
    cudaMemset(g->state, 0,
               (size_t)g->n_v_heads * g->head_k * g->head_v * sizeof(float));
    cudaMemset(g->conv_ring, 0, (size_t)CONV_K * g->qkv_out * sizeof(float));
    g->ring_head = g->ring_len = 0;
  }
}

/* one resident decode step at absolute position `pos`. Embedding provided as a
 * host fp32 row (already dequantized + scaled by the caller). Writes logits to
 * host buf when want_logits. */
void oc_cuda_forward(oc_model *m, const float *embed_host, size_t pos,
                     int want_logits, float *logits_host, float *normed_host) {
  GpuCtx *c = g_ctx;
  g_prof_on = getenv("OC_PROF") != NULL;
  if (g_prof_on) memset(g_prof, 0, sizeof(g_prof));
  double prof_t0 = 0;
  if (g_prof_on) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    prof_t0 = ts.tv_sec + ts.tv_nsec * 1e-9;
  }
  const int h = c->h, hd = c->head_dim, nh = c->n_heads, kvh = c->kv_heads;
  const int B = 256;
  auto G = [&](int n) { return (n + B - 1) / B; };

  memcpy(c->h_pin, embed_host, h * sizeof(float));
  int posi = (int)pos;
  /* pos travels via device memory so a captured graph replays correctly */
  CDIE(cudaMemcpyAsync(c->d_pos, &posi, sizeof(int), cudaMemcpyHostToDevice,
                       c->stream));
  CDIE(cudaStreamSynchronize(c->stream)); /* &posi is stack memory */
  bool use_graph = c->gemma && !g_prof_on && !getenv("OC_TRACE") &&
                   !getenv("OC_NO_GRAPH") && !normed_host;
  const int gate_after = getenv("OC_GDN_GATE_AFTER") != NULL;
  int gi = want_logits ? 1 : 0;
  if (use_graph && c->graph[gi]) {
    CDIE(cudaGraphLaunch(c->graph[gi], c->stream));
    CDIE(cudaStreamSynchronize(c->stream));
    if (want_logits)
      memcpy(logits_host, c->h_pin, c->vocab * sizeof(float));
    return;
  }
  if (use_graph)
    CDIE(cudaStreamBeginCapture(c->stream, cudaStreamCaptureModeGlobal));
  CDIE(cudaMemcpyAsync(c->d_x, c->h_pin, h * sizeof(float),
                       cudaMemcpyHostToDevice, c->stream));

  for (int l = 0; l < c->n_layers; ++l) {
    GpuLayer *g = &c->layers[l];
    k_rms_norm<<<1, 256, 0, c->stream>>>(c->d_norm, c->d_x, g->attn_norm, h, c->rms_eps);

    if (!g->is_gdn) {
      int lhd = g->hd ? g->hd : hd;
      int lkv = g->n_kv ? g->n_kv : kvh;
      int qg_len = g->wq.rows, kvn = lkv * lhd, q_len = g->wo.cols;
      int q_heads = q_len / lhd;
      float ltheta = g->theta != 0.0f ? g->theta : c->rope_theta;
      int lrot = g->n_rot ? g->n_rot : c->rope_dim;
      gemv(c, g->wq, c->d_norm, c->d_qg);
      gemv(c, g->wk, c->d_norm, c->d_k, false);
      if (g->v_from_k)
        k_copy<<<G(kvn), B, 0, c->stream>>>(c->d_v, c->d_k, kvn);
      else
        gemv(c, g->wv, c->d_norm, c->d_v, false);
      float *qptr = c->d_qg, *gate = nullptr;
      if (qg_len >= 2 * q_len) {
        k_deinterleave<<<G(q_heads * lhd), B, 0, c->stream>>>(c->d_qg, c->d_q, c->d_gate,
                                                q_heads, lhd);
        qptr = c->d_q;
        gate = c->d_gate;
      }
      if (g->q_norm)
        k_head_rms_norm<<<q_heads, 256, 0, c->stream>>>(qptr, g->q_norm, lhd, c->rms_eps);
      if (g->k_norm)
        k_head_rms_norm<<<lkv, 256, 0, c->stream>>>(c->d_k, g->k_norm, lhd, c->rms_eps);
      if (g->v_rms)
        k_head_rms_norm<<<lkv, 256, 0, c->stream>>>(c->d_v, nullptr, lhd, c->rms_eps);
      k_rope<<<G(q_heads * (lhd / 2)), B, 0, c->stream>>>(
          qptr, lhd, q_heads, c->d_pos, ltheta, lrot, c->yarn_factor,
          c->yarn_orig_ctx, g->rope_ff);
      k_rope<<<G(lkv * (lhd / 2)), B, 0, c->stream>>>(
          c->d_k, lhd, lkv, c->d_pos, ltheta, lrot, c->yarn_factor,
          c->yarn_orig_ctx, g->rope_ff);
      float *kv_k, *kv_v;
      int cap;
      if (g->my_kv_k) {
        kv_k = g->my_kv_k; kv_v = g->my_kv_v; cap = g->my_kv_cap;
      } else {
        kv_k = c->kv_k + (size_t)g->kv_slot * c->kv_ctx * c->kv_stride;
        kv_v = c->kv_v + (size_t)g->kv_slot * c->kv_ctx * c->kv_stride;
        cap = c->kv_ctx;
      }
      k_kv_append<<<G(kvn), B, 0, c->stream>>>(kv_k, kv_v, c->d_k, c->d_v,
                                               c->d_pos, cap, kvn);
      {
        ProfScope ps(3);
        k_attention<<<q_heads, 128, 0, c->stream>>>(
            c->d_attn, qptr, kv_k, kv_v, c->d_pos, cap, q_heads, lkv, lhd,
            g->attn_scale);
      }
      if (gate) k_sigmoid_gate<<<G(q_len), B, 0, c->stream>>>(c->d_attn, gate, q_len);
      gemv(c, g->wo, c->d_attn, c->d_tmp);
      if (g->attn_post_norm)
        k_rms_norm<<<1, 256, 0, c->stream>>>(c->d_tmp, c->d_tmp, g->attn_post_norm, h,
                               c->rms_eps);
      k_add<<<G(h), B, 0, c->stream>>>(c->d_x, c->d_tmp, h);
    } else {
      int qo = g->qkv_out, vd = g->value_dim, kd = g->key_dim;
      int nvh = g->n_v_heads, hk = g->head_k, hv = g->head_v;
      gemv(c, g->qkv, c->d_norm, c->d_mixed);
      gemv(c, g->ssm_alpha, c->d_norm, c->d_a, false);
      if (g->ssm_beta.d) gemv(c, g->ssm_beta, c->d_norm, c->d_b, false);
      else cudaMemset(c->d_b, 0, nvh * sizeof(float));
      gemv(c, g->gdn_gate, c->d_norm, c->d_z, false);
      k_conv_silu<<<G(qo), B, 0, c->stream>>>(c->d_mixed, c->d_conv, g->ssm_conv1d,
                                g->conv_ring, qo, g->ring_head, g->ring_len);
      k_ring_push<<<G(qo), B, 0, c->stream>>>(g->conv_ring, c->d_mixed, qo, g->ring_head);
      g->ring_head = (g->ring_head + 1) % CONV_K;
      if (g->ring_len < CONV_K) g->ring_len++;
      float out_scale = 1.0f / sqrtf((float)hv);
      size_t shmem = 2 * hk * sizeof(float);
      k_delta_rule<<<nvh, hv, shmem, c->stream>>>(c->d_conv, g->state, c->d_core, c->d_a,
                                       c->d_b, g->ssm_a, g->ssm_dt_bias, qo, kd,
                                       vd, g->n_k_heads, hk, hv, g->a_baked,
                                       out_scale);
      k_gated_rms_norm<<<nvh, 128, 0, c->stream>>>(c->d_core, g->ssm_norm, c->d_z, hv,
                                     c->rms_eps, gate_after);
      gemv(c, g->ssm_out, c->d_core, c->d_tmp);
      k_add<<<G(h), B, 0, c->stream>>>(c->d_x, c->d_tmp, h);
    }

    /* FFN */
    k_rms_norm<<<1, 256, 0, c->stream>>>(c->d_norm, c->d_x, g->ffn_norm, h, c->rms_eps);
    gemv(c, g->gate, c->d_norm, c->d_gf);
    gemv(c, g->up, c->d_norm, c->d_uf, false);
    if (c->gemma)
      k_geglu<<<G(c->inter), B, 0, c->stream>>>(c->d_gf, c->d_uf, c->inter);
    else
      k_swiglu<<<G(c->inter), B, 0, c->stream>>>(c->d_gf, c->d_uf, c->inter);
    gemv(c, g->down, c->d_gf, c->d_tmp);
    if (g->ffn_post_norm)
      k_rms_norm<<<1, 256, 0, c->stream>>>(c->d_tmp, c->d_tmp, g->ffn_post_norm, h, c->rms_eps);
    k_add<<<G(h), B, 0, c->stream>>>(c->d_x, c->d_tmp, h);
    if (g->out_scale != 1.0f && g->out_scale != 0.0f)
      k_scale<<<G(h), B, 0, c->stream>>>(c->d_x, g->out_scale, h);
    if (pos == 0 && getenv("OC_TRACE")) {
      cudaStreamSynchronize(c->stream);
      float *dbg = (float *)malloc(c->h * sizeof(float));
      cudaMemcpy(dbg, c->d_x, c->h * sizeof(float), cudaMemcpyDeviceToHost);
      double sum = 0, asum = 0;
      for (int i = 0; i < c->h; ++i) { sum += dbg[i]; asum += fabsf(dbg[i]); }
      fprintf(stderr, "TRACE L%d gpu  t0 sum=%.6e |sum|=%.6e x[0..4]=%.4f %.4f %.4f %.4f\n",
              l, sum, asum, dbg[0], dbg[1], dbg[2], dbg[3]);
      free(dbg);
    }
  }

  if (g_prof_on) {
    cudaStreamSynchronize(c->stream);
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double total = (ts.tv_sec + ts.tv_nsec * 1e-9 - prof_t0) * 1e3;
    fprintf(stderr, "prof pos=%zu total=%.1fms qkvo=%.1f ffn=%.1f head=%.1f attn=%.1f\n",
            pos, total, g_prof[0], g_prof[1], g_prof[2], g_prof[3]);
  }
  if (want_logits || normed_host) {
    k_rms_norm<<<1, 256, 0, c->stream>>>(c->d_norm, c->d_x, c->final_norm, h,
                                         c->rms_eps);
    if (normed_host) {
      CDIE(cudaMemcpyAsync(c->h_pin, c->d_norm, h * sizeof(float),
                           cudaMemcpyDeviceToHost, c->stream));
    }
    if (want_logits) {
      gemv(c, c->lm_head, c->d_norm, c->d_logits);
      if (c->logit_softcap > 0.0f)
        k_softcap<<<G(c->vocab), B, 0, c->stream>>>(c->d_logits, c->logit_softcap,
                                                    c->vocab);
      CDIE(cudaMemcpyAsync(c->h_pin + (normed_host ? h : 0), c->d_logits,
                           c->vocab * sizeof(float), cudaMemcpyDeviceToHost,
                           c->stream));
    }
  }
  if (use_graph) {
    cudaGraph_t gr;
    CDIE(cudaStreamEndCapture(c->stream, &gr));
    CDIE(cudaGraphInstantiate(&c->graph[gi], gr, nullptr, nullptr, 0));
    cudaGraphDestroy(gr);
    CDIE(cudaGraphLaunch(c->graph[gi], c->stream));
  }
  CDIE(cudaStreamSynchronize(c->stream));
  if (normed_host)
    memcpy(normed_host, c->h_pin, h * sizeof(float));
  if (want_logits)
    memcpy(logits_host, c->h_pin + (normed_host ? h : 0), c->vocab * sizeof(float));
  (void)m;
}

}  // extern "C"
