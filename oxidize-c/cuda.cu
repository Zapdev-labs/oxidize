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

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

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

struct oc_cuda_context {
  void *gpu;
  int device_id;
  size_t max_slots;
  size_t max_ctx;
  int *d_slot_positions;
  uint8_t *d_slot_active;
  pthread_mutex_t decode_mutex;
  bool decode_mutex_ready;
};

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
  GpuW tok_emb;
  GpuW lm_head;
  bool lm_head_tied = false;
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
  size_t max_slots;
  int batch_cap = 64;
  uint32_t *d_tokens = nullptr, *d_lane_slots = nullptr, *d_ids = nullptr;
  int *d_lane_pos = nullptr;
  uint8_t *d_lane_active = nullptr;
};

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

__global__ void k_embed_gather(float *out, const __half *table,
                               const uint32_t *tokens, const uint8_t *active,
                               int h, float scale) {
  int lane = blockIdx.x;
  for (int i = threadIdx.x; i < h; i += blockDim.x)
    out[(size_t)lane * h + i] = active[lane] ?
        __half2float(table[(size_t)tokens[lane] * h + i]) * scale : 0.0f;
}

__global__ void k_rms_norm_batch(float *out, const float *x, const float *w,
                                 int h, float eps) {
  __shared__ float red[256];
  int lane = blockIdx.x;
  const float *src = x + (size_t)lane * h;
  float *dst = out + (size_t)lane * h;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < h; i += blockDim.x) sum += src[i] * src[i];
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)h + eps);
  for (int i = threadIdx.x; i < h; i += blockDim.x) dst[i] = src[i] * inv * w[i];
}

__global__ void k_head_rms_norm_batch(float *v, const float *w, int hd,
                                      float eps) {
  __shared__ float red[256];
  float *hp = v + (size_t)blockIdx.x * hd;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < hd; i += blockDim.x) sum += hp[i] * hp[i];
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)hd + eps);
  for (int i = threadIdx.x; i < hd; i += blockDim.x)
    hp[i] = hp[i] * inv * (w ? w[i] : 1.0f);
}

__global__ void k_deinterleave_batch(const float *qg, float *q, float *gate,
                                     int heads, int hd) {
  int lane = blockIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= heads * hd) return;
  int hh = idx / hd, d = idx % hd;
  q[(size_t)lane * heads * hd + idx] = qg[(size_t)lane * heads * 2 * hd + hh * 2 * hd + d];
  gate[(size_t)lane * heads * hd + idx] = qg[(size_t)lane * heads * 2 * hd + hh * 2 * hd + hd + d];
}

__global__ void k_rope_batch(float *vec, int head_dim, int n_heads,
                             const int *positions, float theta, int rope_dim,
                             float yf, float yorig, const float *ff) {
  int lane = blockIdx.y;
  int rl = (rope_dim == 0 || rope_dim > head_dim) ? head_dim : rope_dim;
  int half = rl / 2;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_heads * half) return;
  int hh = idx / half, i = idx % half;
  int pos = positions[lane];
  bool yarn = yf > 1.0f && yorig > 0.0f;
  if (!yarn && pos == 0 && !ff) return;
  float freq = powf(theta, -2.0f * (float)i / (float)rl);
  if (ff) freq /= ff[i];
  float angle = (float)pos * freq;
  float c, s;
  if (yarn) {
    float fs = 1.0f / yf, msc = 1.0f + 0.1f * logf(yf);
    float clo = floorf((float)rl * logf(yorig / (32.0f * 2.0f * (float)M_PI)) /
                      (2.0f * logf(theta)));
    float chi = ceilf((float)rl * logf(yorig / (1.0f * 2.0f * (float)M_PI)) /
                     (2.0f * logf(theta)));
    clo = fmaxf(0.0f, clo); chi = fminf((float)rl - 1.0f, chi);
    float denom = fmaxf(chi - clo, 0.001f);
    float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, ((float)i - clo) / denom));
    angle = angle * fs * (1.0f - ramp) + angle * ramp;
    c = cosf(angle) * msc; s = sinf(angle) * msc;
  } else { c = cosf(angle); s = sinf(angle); }
  float *hp = vec + ((size_t)lane * n_heads + hh) * head_dim;
  float x0 = hp[i], x1 = hp[half + i];
  hp[i] = x0 * c - x1 * s;
  hp[half + i] = x0 * s + x1 * c;
}

__global__ void k_kv_append_batch(float *kv_k, float *kv_v, const float *k,
                                  const float *v, const int *positions,
                                  const uint32_t *slots, const uint8_t *active,
                                  int cap, int kvn) {
  int lane = blockIdx.y;
  if (!active[lane]) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kvn) return;
  size_t base = ((size_t)slots[lane] * cap + positions[lane] % cap) * kvn;
  kv_k[base + i] = k[(size_t)lane * kvn + i];
  kv_v[base + i] = v[(size_t)lane * kvn + i];
}

__global__ void k_attention_batch(float *out, const float *q, const float *kv_k,
                                  const float *kv_v, const int *positions,
                                  const uint32_t *slots, const uint8_t *active,
                                  int cap, int n_heads, int kv_heads, int head_dim,
                                  float scale) {
  int lane_id = blockIdx.x / n_heads, head = blockIdx.x % n_heads;
  if (!active[lane_id]) return;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int group = n_heads / kv_heads, kv_off = (head / group) * head_dim;
  int kvn = kv_heads * head_dim, per = head_dim / 32;
  const float *qh = q + ((size_t)lane_id * n_heads + head) * head_dim + lane * per;
  float qreg[16], acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
  for (int i = 0; i < per; ++i) qreg[i] = qh[i];
  int seq_len = min(positions[lane_id] + 1, cap);
  int first_pos = positions[lane_id] + 1 - seq_len;
  float maxv = -INFINITY, sum = 0.0f;
  size_t slot_base = (size_t)slots[lane_id] * cap * kvn;
  for (int t = warp; t < seq_len; t += 4) {
    int physical_pos = (first_pos + t) % cap;
    const float *kr = kv_k + slot_base + (size_t)physical_pos * kvn + kv_off + lane * per;
    float score = 0.0f;
    for (int i = 0; i < per; ++i) score += qreg[i] * kr[i];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) score += __shfl_xor_sync(0xFFFFFFFF, score, off);
    score *= scale == 0.0f ? rsqrtf((float)head_dim) : scale;
    float next_max = fmaxf(maxv, score), old = __expf(maxv - next_max), e = __expf(score - next_max);
    const float *vr = kv_v + slot_base + (size_t)physical_pos * kvn + kv_off + lane * per;
    for (int i = 0; i < per; ++i) acc[i] = acc[i] * old + e * vr[i];
    sum = sum * old + e; maxv = next_max;
  }
  __shared__ float sm[4], ss[4], sacc[4][512];
  sm[warp] = maxv; ss[warp] = sum;
  for (int i = 0; i < per; ++i) sacc[warp][lane * per + i] = acc[i];
  __syncthreads();
  if (warp == 0) {
    float max_all = sm[0];
#pragma unroll
    for (int w = 1; w < 4; ++w) max_all = fmaxf(max_all, sm[w]);
    float fw[4], total = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) { fw[w] = __expf(sm[w] - max_all); total += ss[w] * fw[w]; }
    float *dst = out + ((size_t)lane_id * n_heads + head) * head_dim + lane * per;
    for (int i = 0; i < per; ++i) {
      float value = 0.0f;
#pragma unroll
      for (int w = 0; w < 4; ++w) value += sacc[w][lane * per + i] * fw[w];
      dst[i] = value / total;
    }
  }
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

/* gated RMS norm per v_head (gate-after): core = xhat * w * silu(z) */
__global__ void k_gated_rms_norm(float *core, const float *w, const float *z,
                                 int hv, float eps) {
  __shared__ float red[256];
  float *cp = core + blockIdx.x * hv;
  const float *zp = z + blockIdx.x * hv;
  int j = threadIdx.x;
  float s = 0.0f;
  for (int i = j; i < hv; i += blockDim.x) s += cp[i] * cp[i];
  red[j] = s;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (j < st) red[j] += red[j + st];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)hv + eps);
  for (int i = j; i < hv; i += blockDim.x) {
    float zg = zp[i];
    cp[i] = cp[i] * inv * w[i] * (zg / (1.0f + expf(-zg)));
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

bool gemm_batch(GpuCtx *c, const GpuW &w, const float *d_in, float *d_out,
                int batch) {
  if (!w.d || batch < 1 || batch > c->batch_cap) return false;
  int total = w.cols * batch;
  k_f32_to_f16<<<(total + 255) / 256, 256, 0, c->stream>>>(c->d_xh, d_in, total);
  const float alpha = 1.0f, beta = 0.0f;
  return cudaGetLastError() == cudaSuccess &&
      cublasGemmEx(c->cublas, CUBLAS_OP_T, CUBLAS_OP_N, w.rows, batch, w.cols,
                   &alpha, w.d, CUDA_R_16F, w.cols, c->d_xh, CUDA_R_16F,
                   w.cols, &beta, d_out, CUDA_R_32F, w.rows,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) ==
          CUBLAS_STATUS_SUCCESS;
}

__global__ void k_batch_argmax(const float *logits, const uint8_t *active,
                                uint32_t *ids, int rows) {
  const int slot = blockIdx.x;
  if (!active[slot]) {
    if (threadIdx.x == 0) ids[slot] = UINT32_MAX;
    return;
  }
  __shared__ float values[256];
  __shared__ int indices[256];
  float best = -INFINITY;
  int best_index = 0;
  const float *row = logits + (size_t)slot * rows;
  for (int index = threadIdx.x; index < rows; index += blockDim.x) {
    const float value = row[index];
    if (value > best || (value == best && index < best_index)) {
      best = value;
      best_index = index;
    }
  }
  values[threadIdx.x] = best;
  indices[threadIdx.x] = best_index;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other = values[threadIdx.x + stride];
      const int other_index = indices[threadIdx.x + stride];
      if (other > values[threadIdx.x] ||
          (other == values[threadIdx.x] && other_index < indices[threadIdx.x])) {
        values[threadIdx.x] = other;
        indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) ids[slot] = (uint32_t)indices[0];
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

GpuW mkw(const oc_weight *w, bool force_fp16 = false) {
  GpuW g;
  if (!w->quantized && !w->f32) return g;
  g.rows = (int)w->rows;
  g.cols = (int)w->cols;
  if (!force_fp16 && w->quantized && (w->quant == OC_IQ4_XS || w->quant == OC_Q4_K) &&
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

void free_gpuw(GpuW *w) {
  if (!w) return;
  if (w->d) cudaFree(w->d);
  if (w->qd) cudaFree(w->qd);
  w->d = nullptr;
  w->qd = nullptr;
}

void free_gpu_layer(GpuLayer *g) {
  if (!g) return;
  free_gpuw(&g->wq); free_gpuw(&g->wk); free_gpuw(&g->wv); free_gpuw(&g->wo);
  free_gpuw(&g->qkv); free_gpuw(&g->ssm_alpha); free_gpuw(&g->ssm_beta);
  free_gpuw(&g->ssm_out); free_gpuw(&g->gdn_gate);
  free_gpuw(&g->gate); free_gpuw(&g->up); free_gpuw(&g->down);
  cudaFree(g->attn_norm); cudaFree(g->ffn_norm);
  cudaFree(g->q_norm); cudaFree(g->k_norm);
  cudaFree(g->attn_post_norm); cudaFree(g->ffn_post_norm);
  cudaFree(g->my_kv_k); cudaFree(g->my_kv_v);
  cudaFree(g->ssm_a); cudaFree(g->ssm_dt_bias); cudaFree(g->ssm_conv1d);
  cudaFree(g->ssm_norm); cudaFree(g->state); cudaFree(g->conv_ring);
}

void free_gpu_context(GpuCtx *c) {
  if (!c) return;
  for (int i = 0; i < c->n_layers; ++i) free_gpu_layer(&c->layers[i]);
  free(c->layers);
  free_gpuw(&c->tok_emb);
  if (!c->lm_head_tied) free_gpuw(&c->lm_head);
  cudaFree(c->rope_freqs); cudaFree(c->final_norm);
  cudaFree(c->kv_k); cudaFree(c->kv_v);
  cudaFree(c->d_x); cudaFree(c->d_norm); cudaFree(c->d_qg); cudaFree(c->d_q);
  cudaFree(c->d_gate); cudaFree(c->d_k); cudaFree(c->d_v); cudaFree(c->d_attn);
  cudaFree(c->d_tmp); cudaFree(c->d_gf); cudaFree(c->d_uf); cudaFree(c->d_mixed);
  cudaFree(c->d_conv); cudaFree(c->d_a); cudaFree(c->d_b); cudaFree(c->d_z);
  cudaFree(c->d_core); cudaFree(c->d_logits); cudaFree(c->d_xh);
  cudaFree(c->aq_d); cudaFree(c->aq_s); cudaFree(c->aq_q); cudaFree(c->d_pos);
  cudaFree(c->d_tokens); cudaFree(c->d_lane_slots); cudaFree(c->d_lane_pos);
  cudaFree(c->d_lane_active); cudaFree(c->d_ids);
  if (c->graph[0]) cudaGraphExecDestroy(c->graph[0]);
  if (c->graph[1]) cudaGraphExecDestroy(c->graph[1]);
  if (c->h_pin) cudaFreeHost(c->h_pin);
  if (c->cublas) cublasDestroy(c->cublas);
  if (c->stream) cudaStreamDestroy(c->stream);
  free(c);
}

struct DecodeLock {
  oc_cuda_context *owner;
  bool locked;

  explicit DecodeLock(oc_cuda_context *context) : owner(context), locked(false) {
    if (owner && owner->decode_mutex_ready)
      locked = pthread_mutex_lock(&owner->decode_mutex) == 0;
  }

  ~DecodeLock() {
    if (locked) pthread_mutex_unlock(&owner->decode_mutex);
  }
};

bool checked_add(size_t *total, size_t value) {
  if (!total || value > SIZE_MAX - *total) return false;
  *total += value;
  return true;
}

bool checked_mul(size_t left, size_t right, size_t *product) {
  if (!product || (left != 0 && right > SIZE_MAX / left)) return false;
  *product = left * right;
  return true;
}

bool add_product(size_t *total, size_t left, size_t right) {
  size_t product = 0;
  return checked_mul(left, right, &product) && checked_add(total, product);
}

bool add_product3(size_t *total, size_t first, size_t second, size_t third) {
  size_t product = 0;
  return checked_mul(first, second, &product) && add_product(total, product, third);
}

bool has_weight(const oc_weight *weight) {
  return weight && weight->rows != 0 && weight->cols != 0 &&
         (weight->f32 != nullptr || weight->data != nullptr);
}

bool add_fp16_weight(size_t *total, const oc_weight *weight) {
  if (!has_weight(weight)) return true;
  size_t elems = 0;
  return checked_mul(weight->rows, weight->cols, &elems) && add_product(total, elems, sizeof(__half));
}

bool add_fp32_vector(size_t *total, const float *data, size_t count) {
  return !data || count == 0 || add_product(total, count, sizeof(float));
}

/* The batch kernels assume one concrete, warp-aligned Gemma attention shape
 * for every layer.  Do not fill missing per-layer fields from the global
 * config here: doing so turns a malformed model into a launchable one. */
bool batch_contract_valid(const oc_model *m, size_t max_ctx, bool require_batch_contract) {
  if (!m || !m->layers || max_ctx == 0 || max_ctx > m->kv_ctx ||
      m->cfg.hidden_size == 0 || m->cfg.n_heads == 0 || m->cfg.kv_heads == 0 ||
      m->cfg.head_dim == 0 || m->cfg.intermediate_size == 0 ||
      m->cfg.vocab_size == 0 || m->cfg.layer_count == 0 || m->kv_stride == 0 ||
      m->cfg.head_dim < 32 || m->cfg.head_dim > 512 ||
      m->cfg.head_dim % 32 != 0 || m->cfg.n_heads % m->cfg.kv_heads != 0)
    return false;
  if (!require_batch_contract) return true;
  if (!m->gemma || m->cfg.kv_int8) return false;
  for (size_t l = 0; l < m->cfg.layer_count; ++l) {
    const oc_layer *layer = &m->layers[l];
    if (layer->is_gdn || layer->q_bias || layer->k_bias || layer->v_bias ||
        !layer->kv_ck || !layer->kv_cv || layer->kv_cap == 0 ||
        layer->kv_cap > m->kv_ctx ||
        layer->hd == 0 || layer->n_kv == 0)
      return false;
    const size_t hd = layer->hd;
    const size_t kvh = layer->n_kv;
    const size_t q_len = layer->wo.cols;
    if (hd < 32 || hd > 512 || hd % 32 != 0 || kvh == 0 || q_len == 0 ||
        q_len % hd != 0 || q_len / hd % kvh != 0 ||
        hd > SIZE_MAX / kvh || hd * kvh > m->kv_stride ||
        !has_weight(&layer->wq) || !has_weight(&layer->wk) ||
        !has_weight(&layer->wo) || !has_weight(&layer->gate) ||
        !has_weight(&layer->up) || !has_weight(&layer->down) ||
        layer->wq.cols != m->cfg.hidden_size ||
        (layer->wq.rows != q_len && layer->wq.rows != 2 * q_len) ||
        layer->wk.cols != m->cfg.hidden_size || layer->wk.rows != hd * kvh ||
        (!layer->v_from_k && (!has_weight(&layer->wv) ||
                              layer->wv.cols != m->cfg.hidden_size ||
                              layer->wv.rows != hd * kvh)) ||
        layer->wo.rows != m->cfg.hidden_size || layer->wo.cols != q_len ||
        layer->gate.rows != m->cfg.intermediate_size ||
        layer->gate.cols != m->cfg.hidden_size ||
        layer->up.rows != m->cfg.intermediate_size ||
        layer->up.cols != m->cfg.hidden_size ||
        layer->down.rows != m->cfg.hidden_size ||
        layer->down.cols != m->cfg.intermediate_size)
      return false;
  }
  return true;
}

bool estimate_batch_memory(const oc_model *m, size_t max_slots, size_t max_ctx,
                           bool require_batch_contract, oc_cuda_memory_report *report) {
  if (!report || !m || max_slots == 0 || !batch_contract_valid(m, max_ctx, require_batch_contract))
    return false;
  memset(report, 0, sizeof(*report));
  size_t weights = 0, auxiliary = 0, slot_kv = 0, scratch = 0;
  if (!add_fp16_weight(&weights, &m->tok_emb) ||
      (!m->tied && !add_fp16_weight(&weights, &m->lm_head)))
    return false;
  if (!add_fp32_vector(&auxiliary, m->final_norm, m->cfg.hidden_size) ||
      !add_fp32_vector(&auxiliary, m->rope_freqs,
                       m->gemma ? m->cfg.head_dim / 2 : 0))
    return false;

  size_t max_qg = 0, max_attn = 0, max_kv = 0, max_inter = m->cfg.intermediate_size;
  for (size_t l = 0; l < m->cfg.layer_count; ++l) {
    const oc_layer *layer = &m->layers[l];
    const oc_weight *weights_to_upload[] = {
        &layer->wq, &layer->wk, &layer->wv, &layer->wo, &layer->gate,
        &layer->up, &layer->down, &layer->qkv, &layer->ssm_alpha,
        &layer->ssm_beta, &layer->ssm_out, &layer->gdn_gate};
    for (const oc_weight *weight : weights_to_upload)
      if (!add_fp16_weight(&weights, weight)) return false;
    if (!add_fp32_vector(&auxiliary, layer->attn_norm, m->cfg.hidden_size) ||
        !add_fp32_vector(&auxiliary, layer->ffn_norm, m->cfg.hidden_size) ||
        !add_fp32_vector(&auxiliary, layer->q_norm, layer->hd ? layer->hd : m->cfg.head_dim) ||
        !add_fp32_vector(&auxiliary, layer->k_norm, layer->hd ? layer->hd : m->cfg.head_dim) ||
        !add_fp32_vector(&auxiliary, layer->attn_post_norm, m->cfg.hidden_size) ||
        !add_fp32_vector(&auxiliary, layer->ffn_post_norm, m->cfg.hidden_size) ||
        !add_fp32_vector(&auxiliary, layer->ssm_a, layer->n_v_heads) ||
        !add_fp32_vector(&auxiliary, layer->ssm_dt_bias, layer->n_v_heads) ||
        !add_fp32_vector(&auxiliary, layer->ssm_conv1d, CONV_K * layer->qkv_out) ||
        !add_fp32_vector(&auxiliary, layer->ssm_norm, layer->head_v))
      return false;
    if (!layer->is_gdn) {
      const size_t hd = layer->hd ? layer->hd : m->cfg.head_dim;
      const size_t kvh = layer->n_kv ? layer->n_kv : m->cfg.kv_heads;
      const size_t cap = layer->kv_cap < max_ctx ? layer->kv_cap : max_ctx;
      size_t kvn = 0, elems = 0;
      if (!checked_mul(hd, kvh, &kvn) || !checked_mul(max_slots, cap, &elems) ||
          !checked_mul(elems, kvn, &elems) || !add_product(&slot_kv, elems, 2 * sizeof(float)))
        return false;
      if (layer->wq.rows > max_qg) max_qg = layer->wq.rows;
      if (layer->wo.cols > max_attn) max_attn = layer->wo.cols;
      if (kvn > max_kv) max_kv = kvn;
    } else {
      size_t state = 0, ring = 0;
      if (!checked_mul(layer->n_v_heads, layer->head_k, &state) ||
          !checked_mul(state, layer->head_v, &state) ||
          !checked_mul(max_slots, state, &state) ||
          !checked_mul(max_slots, CONV_K * layer->qkv_out, &ring) ||
          !checked_add(&state, ring) || !add_product(&slot_kv, state, sizeof(float)))
        return false;
    }
    if (layer->gate.rows > max_inter) max_inter = layer->gate.rows;
    if (layer->up.rows > max_inter) max_inter = layer->up.rows;
  }
  if (!m->gemma) {
    size_t elems = 0;
    if (!checked_mul(m->n_kv_layers, max_ctx, &elems) ||
        !checked_mul(elems, m->kv_stride, &elems) ||
        !checked_mul(elems, max_slots, &elems) ||
        !add_product(&slot_kv, elems, 2 * sizeof(float)))
      return false;
  }
  const size_t bucket = 64;
  const size_t h = m->cfg.hidden_size, vocab = m->cfg.vocab_size;
  const size_t q = max_qg ? max_qg : 2 * m->cfg.n_heads * m->cfg.head_dim;
  const size_t attn = max_attn ? max_attn : m->cfg.n_heads * m->cfg.head_dim;
  const size_t kv = max_kv ? max_kv : m->kv_stride;
  const size_t inter = max_inter;
  const size_t half_width = h > inter ? h : inter;
  size_t float_rows = 0;
  const size_t scratch_widths[] = {h, h, q, attn, attn, kv, kv, h, h, inter, inter, vocab};
  for (size_t width : scratch_widths)
    if (!checked_add(&float_rows, width)) return false;
  if (!add_product3(&scratch, bucket, float_rows, sizeof(float)) ||
      !add_product3(&scratch, bucket, half_width, sizeof(__half)) ||
      !add_product(&scratch, 8192 * 6 + 256 * 2, sizeof(float)) ||
      !add_product(&scratch, 8 * (half_width / 32 + 8), sizeof(int)) ||
      !add_product(&scratch, 2 * (half_width / 32 + 8), sizeof(float)) ||
      !add_product(&scratch, bucket, sizeof(uint32_t) * 3 + sizeof(int) + sizeof(uint8_t)) ||
      !add_product(&auxiliary, max_slots, sizeof(int) + sizeof(uint8_t)))
    return false;
  if (!checked_add(&auxiliary, sizeof(int)) || !checked_add(&scratch, sizeof(uint32_t) * bucket))
    return false;
  size_t required = 0;
  if (!checked_add(&required, weights) || !checked_add(&required, auxiliary) ||
      !checked_add(&required, scratch) || !checked_add(&required, slot_kv))
    return false;
  report->fp16_weight_bytes = weights;
  report->auxiliary_bytes = auxiliary;
  report->scratch_bytes = scratch;
  report->slot_kv_bytes = slot_kv;
  report->required_bytes = required;
  return true;
}

}  // namespace

/* ================= public C API ================= */
extern "C" {

void oc_cuda_release(oc_model *m) {
  if (!m || !m->cuda_ctx) return;
  oc_cuda_context *owner = m->cuda_ctx;
  cudaSetDevice(owner->device_id);
  free_gpu_context(static_cast<GpuCtx *>(owner->gpu));
  if (owner->d_slot_positions) cudaFree(owner->d_slot_positions);
  if (owner->d_slot_active) cudaFree(owner->d_slot_active);
  if (owner->decode_mutex_ready) pthread_mutex_destroy(&owner->decode_mutex);
  free(owner);
  m->cuda_ctx = nullptr;
  m->gpu_active = false;
}

size_t oc_cuda_slot_count(const oc_model *m) {
  return (m && m->cuda_ctx) ? m->cuda_ctx->max_slots : 0;
}

size_t oc_cuda_slot_max_ctx(const oc_model *m) {
  return (m && m->cuda_ctx) ? m->cuda_ctx->max_ctx : 0;
}

int cuda_batch_memory_preflight(const oc_model *m, int device_id, size_t max_slots,
                                size_t max_ctx, bool require_batch_contract,
                                oc_cuda_memory_report *report) {
  if (!m || device_id < 0 || max_slots == 0 || max_ctx == 0 ||
      max_slots > INT_MAX || max_ctx > INT_MAX ||
      max_slots > SIZE_MAX / sizeof(int) ||
      !estimate_batch_memory(m, max_slots, max_ctx, require_batch_contract, report))
    return -1;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || device_id >= devices ||
      cudaSetDevice(device_id) != cudaSuccess)
    return -1;
  size_t free_bytes = 0, total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      report->required_bytes > free_bytes)
    return -1;
  report->free_bytes = free_bytes;
  return 0;
}

int oc_cuda_batch_memory_preflight(const oc_model *m, int device_id, size_t max_slots,
                                   size_t max_ctx, oc_cuda_memory_report *report) {
  return cuda_batch_memory_preflight(m, device_id, max_slots, max_ctx, true, report);
}

int configure_cuda_context(oc_model *m, int device_id, size_t max_slots,
                           size_t max_ctx, bool require_batch_contract) {
  oc_cuda_memory_report report = {};
  if (cuda_batch_memory_preflight(m, device_id, max_slots, max_ctx,
                                  require_batch_contract, &report) != 0)
    return -1;

  /* Preflight completed before releasing a usable context or allocating any
   * resident state, so an oversized request has no partial upload side effect. */
  oc_cuda_release(m);
  if (cudaSetDevice(device_id) != cudaSuccess) return -1;
  oc_cuda_context *owner = static_cast<oc_cuda_context *>(calloc(1, sizeof(*owner)));
  GpuCtx *c = static_cast<GpuCtx *>(calloc(1, sizeof(*c)));
  if (!owner || !c) {
    free(owner);
    free(c);
    return -1;
  }
  owner->gpu = c;
  owner->device_id = device_id;
  owner->max_slots = max_slots;
  owner->max_ctx = max_ctx;
  c->max_slots = max_slots;
  c->batch_cap = 64;
  const int mutex_status = pthread_mutex_init(&owner->decode_mutex, nullptr);
  owner->decode_mutex_ready = mutex_status == 0;
  if (mutex_status != 0 ||
      cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking) != cudaSuccess ||
      cublasCreate(&c->cublas) != CUBLAS_STATUS_SUCCESS ||
      cublasSetStream(c->cublas, c->stream) != CUBLAS_STATUS_SUCCESS ||
      cublasSetMathMode(c->cublas, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS ||
      cudaMalloc(&owner->d_slot_positions, max_slots * sizeof(int)) != cudaSuccess ||
      cudaMalloc(&owner->d_slot_active, max_slots * sizeof(uint8_t)) != cudaSuccess ||
      cudaMemsetAsync(owner->d_slot_positions, 0, max_slots * sizeof(int), c->stream) != cudaSuccess ||
      cudaMemsetAsync(owner->d_slot_active, 0, max_slots * sizeof(uint8_t), c->stream) != cudaSuccess ||
      cudaStreamSynchronize(c->stream) != cudaSuccess) {
    free_gpu_context(c);
    if (owner->d_slot_positions) cudaFree(owner->d_slot_positions);
    if (owner->d_slot_active) cudaFree(owner->d_slot_active);
    if (owner->decode_mutex_ready) pthread_mutex_destroy(&owner->decode_mutex);
    free(owner);
    return -1;
  }
  m->cuda_ctx = owner;
  return 0;
}

int oc_cuda_configure_batch(oc_model *m, int device_id, size_t max_slots,
                            size_t max_ctx) {
  return configure_cuda_context(m, device_id, max_slots, max_ctx, true);
}

int oc_cuda_batch_gemm_argmax(const uint16_t *weights_f16, size_t rows,
                              size_t cols, const float *inputs_f32,
                              size_t batch, const uint8_t *active,
                              uint32_t *token_ids) {
  if (!weights_f16 || !inputs_f32 || !active || !token_ids || !rows || !cols ||
      !batch || rows > INT_MAX || cols > INT_MAX || batch > INT_MAX)
    return -1;

  cublasHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
      cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS ||
      cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
    if (handle) cublasDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    return -1;
  }

  __half *d_weights = nullptr, *d_inputs = nullptr;
  float *d_logits = nullptr;
  uint8_t *d_active = nullptr;
  uint32_t *d_ids = nullptr;
  int status = -1;
  const size_t weight_count = rows * cols;
  const size_t input_count = batch * cols;
  const size_t logit_count = batch * rows;
  if (weight_count / rows != cols || input_count / batch != cols ||
      logit_count / batch != rows ||
      input_count > INT_MAX ||
      cudaMalloc(&d_weights, weight_count * sizeof(__half)) != cudaSuccess ||
      cudaMalloc(&d_inputs, input_count * sizeof(__half)) != cudaSuccess ||
      cudaMalloc(&d_logits, logit_count * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&d_active, batch * sizeof(uint8_t)) != cudaSuccess ||
      cudaMalloc(&d_ids, batch * sizeof(uint32_t)) != cudaSuccess)
    goto done;

  if (cudaMemcpyAsync(d_weights, weights_f16, weight_count * sizeof(__half),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(d_active, active, batch * sizeof(uint8_t),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess)
    goto done;
  k_f32_to_f16<<<(input_count + 255) / 256, 256, 0, stream>>>(
      d_inputs, inputs_f32, (int)input_count);
  if (cudaGetLastError() != cudaSuccess || cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS)
    goto done;
  {
    const float alpha = 1.0f, beta = 0.0f;
    if (cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)rows, (int)batch,
                     (int)cols, &alpha, d_weights, CUDA_R_16F, (int)cols,
                     d_inputs, CUDA_R_16F, (int)cols, &beta, d_logits,
                     CUDA_R_32F, (int)rows, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
      goto done;
  }
  k_batch_argmax<<<(int)batch, 256, 0, stream>>>(d_logits, d_active, d_ids, (int)rows);
  if (cudaGetLastError() != cudaSuccess ||
      cudaMemcpyAsync(token_ids, d_ids, batch * sizeof(uint32_t),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess)
    goto done;
  status = 0;

done:
  cudaFree(d_ids);
  cudaFree(d_active);
  cudaFree(d_logits);
  cudaFree(d_inputs);
  cudaFree(d_weights);
  cublasDestroy(handle);
  cudaStreamDestroy(stream);
  return status;
}

int oc_cuda_decode_batch(oc_model *m, const oc_cuda_decode_lane *lanes,
                         size_t count, uint32_t *next_ids) {
  if (!m || !lanes || !next_ids || count == 0 || count > 64 || !m->gemma ||
      m->cfg.kv_int8 || !m->cuda_ctx || !m->gpu_active)
    return -1;
  oc_cuda_context *owner = m->cuda_ctx;
  DecodeLock decode_lock(owner);
  if (!decode_lock.locked ||
      !batch_contract_valid(m, owner->max_ctx, true))
    return -1;
  GpuCtx *c = static_cast<GpuCtx *>(owner->gpu);
  if (!c || !c->tok_emb.d || !c->lm_head.d || count > (size_t)c->batch_cap ||
      cudaSetDevice(owner->device_id) != cudaSuccess)
    return -1;
  for (int l = 0; l < c->n_layers; ++l)
    if (c->layers[l].is_gdn || !c->layers[l].my_kv_k || !c->layers[l].my_kv_v)
      return -1;

  std::vector<uint32_t> tokens(c->batch_cap, 0), slots(c->batch_cap, 0);
  std::vector<int> positions(c->batch_cap, 0);
  std::vector<uint8_t> active(c->batch_cap, 0);
  std::vector<size_t> original;
  original.reserve(count);
  for (size_t i = 0; i < count; ++i) next_ids[i] = UINT32_MAX;
  for (size_t i = 0; i < count; ++i) {
    if (!lanes[i].want_token) continue;
    if (lanes[i].token >= m->cfg.vocab_size || lanes[i].slot >= owner->max_slots ||
        lanes[i].position >= owner->max_ctx)
      return -1;
    for (size_t prior = 0; prior < original.size(); ++prior)
      if (slots[prior] == lanes[i].slot) return -1;
    size_t lane = original.size();
    tokens[lane] = lanes[i].token;
    slots[lane] = lanes[i].slot;
    positions[lane] = (int)lanes[i].position;
    active[lane] = 1;
    original.push_back(i);
  }
  if (original.empty()) return -1;
  int batch = 1;
  while (batch < (int)original.size()) batch <<= 1;
  if (batch > c->batch_cap) return -1;
  const int B = 256;
  auto G = [&](int n) { return (n + B - 1) / B; };
  if (cudaMemcpyAsync(c->d_tokens, tokens.data(), batch * sizeof(uint32_t),
                      cudaMemcpyHostToDevice, c->stream) != cudaSuccess ||
      cudaMemcpyAsync(c->d_lane_slots, slots.data(), batch * sizeof(uint32_t),
                      cudaMemcpyHostToDevice, c->stream) != cudaSuccess ||
      cudaMemcpyAsync(c->d_lane_pos, positions.data(), batch * sizeof(int),
                      cudaMemcpyHostToDevice, c->stream) != cudaSuccess ||
      cudaMemcpyAsync(c->d_lane_active, active.data(), batch * sizeof(uint8_t),
                      cudaMemcpyHostToDevice, c->stream) != cudaSuccess)
    return -1;
  k_embed_gather<<<batch, B, 0, c->stream>>>(c->d_x, c->tok_emb.d, c->d_tokens,
                                               c->d_lane_active, c->h, m->emb_scale);
  for (int l = 0; l < c->n_layers; ++l) {
    GpuLayer *g = &c->layers[l];
    const int hd = g->hd, kvh = g->n_kv, kvn = hd * kvh;
    const int q_len = g->wo.cols, q_heads = q_len / hd;
    k_rms_norm_batch<<<batch, B, 0, c->stream>>>(c->d_norm, c->d_x, g->attn_norm,
                                                  c->h, c->rms_eps);
    if (!gemm_batch(c, g->wq, c->d_norm, c->d_qg, batch) ||
        !gemm_batch(c, g->wk, c->d_norm, c->d_k, batch)) return -1;
    if (g->v_from_k) {
      for (int lane = 0; lane < batch; ++lane)
        k_copy<<<G(kvn), B, 0, c->stream>>>(c->d_v + (size_t)lane * kvn,
                                             c->d_k + (size_t)lane * kvn, kvn);
    } else if (!gemm_batch(c, g->wv, c->d_norm, c->d_v, batch)) return -1;
    float *q = c->d_qg, *gate = nullptr;
    if (g->wq.rows >= 2 * q_len) {
      k_deinterleave_batch<<<dim3(G(q_len), batch), B, 0, c->stream>>>(
          c->d_qg, c->d_q, c->d_gate, q_heads, hd);
      q = c->d_q;
      gate = c->d_gate;
    }
    if (g->q_norm)
      k_head_rms_norm_batch<<<batch * q_heads, B, 0, c->stream>>>(q, g->q_norm, hd,
                                                                    c->rms_eps);
    if (g->k_norm)
      k_head_rms_norm_batch<<<batch * kvh, B, 0, c->stream>>>(c->d_k, g->k_norm, hd,
                                                                c->rms_eps);
    if (g->v_rms)
      k_head_rms_norm_batch<<<batch * kvh, B, 0, c->stream>>>(c->d_v, nullptr, hd,
                                                                c->rms_eps);
    k_rope_batch<<<dim3(G(q_heads * (hd / 2)), batch), B, 0, c->stream>>>(
        q, hd, q_heads, c->d_lane_pos, g->theta, g->n_rot, c->yarn_factor,
        c->yarn_orig_ctx, g->rope_ff);
    k_rope_batch<<<dim3(G(kvh * (hd / 2)), batch), B, 0, c->stream>>>(
        c->d_k, hd, kvh, c->d_lane_pos, g->theta, g->n_rot, c->yarn_factor,
        c->yarn_orig_ctx, g->rope_ff);
    k_kv_append_batch<<<dim3(G(kvn), batch), B, 0, c->stream>>>(
        g->my_kv_k, g->my_kv_v, c->d_k, c->d_v, c->d_lane_pos, c->d_lane_slots,
        c->d_lane_active, g->my_kv_cap, kvn);
    k_attention_batch<<<batch * q_heads, 128, 0, c->stream>>>(
        c->d_attn, q, g->my_kv_k, g->my_kv_v, c->d_lane_pos, c->d_lane_slots,
        c->d_lane_active, g->my_kv_cap, q_heads, kvh, hd, g->attn_scale);
    if (gate) k_sigmoid_gate<<<G(batch * q_len), B, 0, c->stream>>>(c->d_attn, gate,
                                                                      batch * q_len);
    if (!gemm_batch(c, g->wo, c->d_attn, c->d_tmp, batch)) return -1;
    if (g->attn_post_norm)
      k_rms_norm_batch<<<batch, B, 0, c->stream>>>(c->d_tmp, c->d_tmp,
                                                     g->attn_post_norm, c->h,
                                                     c->rms_eps);
    k_add<<<G(batch * c->h), B, 0, c->stream>>>(c->d_x, c->d_tmp, batch * c->h);
    k_rms_norm_batch<<<batch, B, 0, c->stream>>>(c->d_norm, c->d_x, g->ffn_norm,
                                                  c->h, c->rms_eps);
    if (!gemm_batch(c, g->gate, c->d_norm, c->d_gf, batch) ||
        !gemm_batch(c, g->up, c->d_norm, c->d_uf, batch)) return -1;
    k_geglu<<<G(batch * c->inter), B, 0, c->stream>>>(c->d_gf, c->d_uf,
                                                       batch * c->inter);
    if (!gemm_batch(c, g->down, c->d_gf, c->d_tmp, batch)) return -1;
    if (g->ffn_post_norm)
      k_rms_norm_batch<<<batch, B, 0, c->stream>>>(c->d_tmp, c->d_tmp,
                                                     g->ffn_post_norm, c->h,
                                                     c->rms_eps);
    k_add<<<G(batch * c->h), B, 0, c->stream>>>(c->d_x, c->d_tmp, batch * c->h);
    if (g->out_scale != 1.0f && g->out_scale != 0.0f)
      k_scale<<<G(batch * c->h), B, 0, c->stream>>>(c->d_x, g->out_scale,
                                                     batch * c->h);
  }
  k_rms_norm_batch<<<batch, B, 0, c->stream>>>(c->d_norm, c->d_x, c->final_norm,
                                                c->h, c->rms_eps);
  if (!gemm_batch(c, c->lm_head, c->d_norm, c->d_logits, batch)) return -1;
  if (c->logit_softcap > 0.0f)
    k_softcap<<<G(batch * c->vocab), B, 0, c->stream>>>(c->d_logits,
                                                         c->logit_softcap,
                                                         batch * c->vocab);
  k_batch_argmax<<<batch, B, 0, c->stream>>>(c->d_logits, c->d_lane_active,
                                               c->d_ids, c->vocab);
  std::vector<uint32_t> ids(batch);
  if (cudaGetLastError() != cudaSuccess ||
      cudaMemcpyAsync(ids.data(), c->d_ids, batch * sizeof(uint32_t),
                      cudaMemcpyDeviceToHost, c->stream) != cudaSuccess ||
      cudaStreamSynchronize(c->stream) != cudaSuccess)
    return -1;
  for (size_t lane = 0; lane < original.size(); ++lane) next_ids[original[lane]] = ids[lane];
  return 0;
}

int oc_cuda_build(oc_model *m) {
  if (!m || getenv("OC_NO_GPU")) return -1;
  if (!m->cuda_ctx) {
    if (m->kv_ctx == 0 || configure_cuda_context(m, 0, 1, m->kv_ctx,
                                                   m->gemma) != 0)
      return -1;
  }
  oc_cuda_context *owner = m->cuda_ctx;
  if (!owner || !batch_contract_valid(m, owner->max_ctx, m->gemma) ||
      cudaSetDevice(owner->device_id) != cudaSuccess)
    return -1;
  if (m->gpu_active) return 0;
  const oc_config *cf = &m->cfg;
  if (cf->hidden_size > INT_MAX ||
      cf->n_heads > INT_MAX || cf->kv_heads > INT_MAX ||
      cf->head_dim > INT_MAX || cf->intermediate_size > INT_MAX ||
      cf->vocab_size > INT_MAX || cf->layer_count > INT_MAX ||
      m->kv_stride > INT_MAX || owner->max_ctx > INT_MAX)
    return -1;
  GpuCtx *c = static_cast<GpuCtx *>(owner->gpu);
  if (!c || !c->cublas || !c->stream) return -1;
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, owner->device_id) != cudaSuccess) return -1;
  fprintf(stderr, "cuda: resident forward on %s (%.1f GB)\n", p.name,
          (double)p.totalGlobalMem / 1e9);

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
  c->kv_ctx = (int)owner->max_ctx;
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
    g->gate = mkw(&L->gate, c->gemma);
    g->up = mkw(&L->up, c->gemma);
    g->down = mkw(&L->down, c->gemma);
    if (L->is_gdn) {
      g->kv_slot = -1;
      g->qkv = mkw(&L->qkv, c->gemma);
      g->ssm_alpha = mkw(&L->ssm_alpha, c->gemma);
      g->ssm_beta = mkw(&L->ssm_beta, c->gemma);
      g->gdn_gate = mkw(&L->gdn_gate, c->gemma);
      g->ssm_out = mkw(&L->ssm_out, c->gemma);
      g->ssm_a = upload_fp32(L->ssm_a, L->n_v_heads);
      g->ssm_dt_bias = upload_fp32(L->ssm_dt_bias, L->n_v_heads);
      g->ssm_conv1d = upload_fp32(L->ssm_conv1d, CONV_K * L->qkv_out);
      g->ssm_norm = upload_fp32(L->ssm_norm, L->head_v);
      if (c->max_slots > SIZE_MAX /
              ((size_t)L->n_v_heads * L->head_k * L->head_v) ||
          c->max_slots > SIZE_MAX / ((size_t)CONV_K * L->qkv_out))
        return -1;
      CDIE(cudaMalloc(&g->state,
                      c->max_slots * (size_t)L->n_v_heads * L->head_k * L->head_v * sizeof(float)));
      CDIE(cudaMemset(g->state, 0,
                      c->max_slots * (size_t)L->n_v_heads * L->head_k * L->head_v * sizeof(float)));
      CDIE(cudaMalloc(&g->conv_ring, c->max_slots * (size_t)CONV_K * L->qkv_out * sizeof(float)));
      CDIE(cudaMemset(g->conv_ring, 0, c->max_slots * (size_t)CONV_K * L->qkv_out * sizeof(float)));
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
      g->wq = mkw(&L->wq, c->gemma);
      g->wk = mkw(&L->wk, c->gemma);
      g->wv = mkw(&L->wv, c->gemma);
      g->wo = mkw(&L->wo, c->gemma);
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
        g->my_kv_cap = (int)(L->kv_cap < (size_t)c->kv_ctx ? L->kv_cap : (size_t)c->kv_ctx);
        size_t elems = (size_t)g->my_kv_cap * g->n_kv * g->hd;
        if (elems == 0 || c->max_slots > SIZE_MAX / elems) return -1;
        elems *= c->max_slots;
        CDIE(cudaMalloc(&g->my_kv_k, elems * sizeof(float)));
        CDIE(cudaMalloc(&g->my_kv_v, elems * sizeof(float)));
        CDIE(cudaMemset(g->my_kv_k, 0, elems * sizeof(float)));
        CDIE(cudaMemset(g->my_kv_v, 0, elems * sizeof(float)));
      }
    }
  }
  c->final_norm = upload_fp32(m->final_norm, c->h);
  c->tok_emb = mkw(&m->tok_emb, c->gemma);
  c->lm_head_tied = m->tied;
  c->lm_head = m->tied ? c->tok_emb : mkw(&m->lm_head, c->gemma);

  /* KV cache (device); gemma uses per-layer private caches instead */
  if (!m->gemma) {
    size_t kv_elems = (size_t)m->n_kv_layers * c->kv_ctx * c->kv_stride;
    if (kv_elems == 0 || c->max_slots > SIZE_MAX / kv_elems) return -1;
    kv_elems *= c->max_slots;
    CDIE(cudaMalloc(&c->kv_k, kv_elems * sizeof(float)));
    CDIE(cudaMalloc(&c->kv_v, kv_elems * sizeof(float)));
    CDIE(cudaMemset(c->kv_k, 0, kv_elems * sizeof(float)));
    CDIE(cudaMemset(c->kv_v, 0, kv_elems * sizeof(float)));
  }

  /* scratch: size the widest matmul output */
  int wide = c->inter;
  if (c->vocab > wide) wide = c->vocab;
  int qwide = c->n_heads * c->head_dim * 2;
  int attnwide = c->n_heads * c->head_dim;
  int kvwide = c->kv_stride;
  for (int l = 0; l < c->n_layers; ++l) {
    const GpuLayer *g = &c->layers[l];
    if (g->is_gdn) continue;
    if (g->wq.rows > qwide) qwide = g->wq.rows;
    if (g->wo.cols > attnwide) attnwide = g->wo.cols;
    if (g->hd > 0 && g->n_kv > 0 && g->hd <= INT_MAX / g->n_kv &&
        g->hd * g->n_kv > kvwide)
      kvwide = g->hd * g->n_kv;
  }
  int maxcol = c->h > c->inter ? c->h : c->inter;
  if (attnwide > maxcol) maxcol = attnwide;
  auto A = [&](float **p, size_t n) { CDIE(cudaMalloc(p, n * sizeof(float))); };
  A(&c->d_x, (size_t)c->batch_cap * c->h);
  A(&c->d_norm, (size_t)c->batch_cap * c->h);
  A(&c->d_qg, (size_t)c->batch_cap * qwide);
  A(&c->d_q, (size_t)c->batch_cap * attnwide);
  A(&c->d_gate, (size_t)c->batch_cap * attnwide);
  A(&c->d_k, (size_t)c->batch_cap * kvwide);
  A(&c->d_v, (size_t)c->batch_cap * kvwide);
  A(&c->d_attn, (size_t)c->batch_cap * attnwide);
  A(&c->d_tmp, (size_t)c->batch_cap * c->h);
  A(&c->d_gf, (size_t)c->batch_cap * c->inter);
  A(&c->d_uf, (size_t)c->batch_cap * c->inter);
  A(&c->d_mixed, 8192 * 2);
  A(&c->d_conv, 8192 * 2);
  A(&c->d_a, 256);
  A(&c->d_b, 256);
  A(&c->d_z, 8192);
  A(&c->d_core, 8192);
  A(&c->d_logits, (size_t)c->batch_cap * c->vocab);
  CDIE(cudaMalloc(&c->d_xh, (size_t)c->batch_cap * maxcol * sizeof(__half)));
  {
    int nbmax = maxcol / 32 + 8;
    CDIE(cudaMalloc(&c->aq_d, nbmax * sizeof(float)));
    CDIE(cudaMalloc(&c->aq_s, nbmax * sizeof(float)));
    CDIE(cudaMalloc(&c->aq_q, nbmax * 8 * sizeof(int)));
  }
  CDIE(cudaMalloc(&c->d_pos, sizeof(int)));
  CDIE(cudaMalloc(&c->d_tokens, c->batch_cap * sizeof(uint32_t)));
  CDIE(cudaMalloc(&c->d_lane_slots, c->batch_cap * sizeof(uint32_t)));
  CDIE(cudaMalloc(&c->d_lane_pos, c->batch_cap * sizeof(int)));
  CDIE(cudaMalloc(&c->d_lane_active, c->batch_cap * sizeof(uint8_t)));
  CDIE(cudaMalloc(&c->d_ids, c->batch_cap * sizeof(uint32_t)));
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
  m->gpu_active = true;
  return 0;
}

void oc_cuda_reset(oc_model *m) {
  GpuCtx *c = m && m->cuda_ctx ? static_cast<GpuCtx *>(m->cuda_ctx->gpu) : nullptr;
  if (!c) return;
  cudaSetDevice(m->cuda_ctx->device_id);
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
  GpuCtx *c = m && m->cuda_ctx ? static_cast<GpuCtx *>(m->cuda_ctx->gpu) : nullptr;
  if (!c || !embed_host || (want_logits && !logits_host)) return;
  cudaSetDevice(m->cuda_ctx->device_id);
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
                                     c->rms_eps);
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
