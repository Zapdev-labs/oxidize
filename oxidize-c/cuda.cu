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

struct GpuW {           /* FP16 weight matrix [rows x cols] row-major on device */
  __half *d = nullptr;
  int rows = 0, cols = 0;
};

struct GpuLayer {
  bool is_gdn = false;
  float *attn_norm = nullptr, *ffn_norm = nullptr;
  /* attention */
  GpuW wq, wk, wv, wo;
  float *q_norm = nullptr, *k_norm = nullptr;  /* per-head, may be null */
  int kv_slot = -1;
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
  for (int i = threadIdx.x; i < hd; i += blockDim.x) hp[i] = hp[i] * inv * w[i];
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
__global__ void k_rope(float *vec, int head_dim, int n_heads, int pos,
                       float theta, int rope_dim, float yf, float yorig) {
  int rl = (rope_dim == 0 || rope_dim > head_dim) ? head_dim : rope_dim;
  int half = rl / 2;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_heads * half) return;
  int hh = idx / half, i = idx % half;
  bool yarn = yf > 1.0f && yorig > 0.0f;
  if (!yarn && pos == 0) return;
  float freq = powf(theta, -2.0f * (float)i / (float)rl);
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
                            const float *v, int slot, int kv_ctx, int kvn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kvn) return;
  int base = (slot % kv_ctx) * kvn;
  kv_k[base + i] = k[i];
  kv_v[base + i] = v[i];
}

/* GQA attention decode, online softmax. One block per query head (head_dim
 * threads). Ports oc_attention. kv_k/kv_v point at this layer's slice. */
__global__ void k_attention(float *out, const float *q, const float *kv_k,
                            const float *kv_v, int seq_len, int n_heads,
                            int kv_heads, int head_dim) {
  int head = blockIdx.x;
  int d = threadIdx.x;                 /* one thread per head_dim element */
  if (head >= n_heads || d >= head_dim) return;
  int group = n_heads / kv_heads;
  int kv_off = (head / group) * head_dim;
  int kvn = kv_heads * head_dim;
  const float *qh = q + head * head_dim;
  __shared__ float sh_q[256];
  __shared__ float sh_red[256];
  __shared__ float s_max, s_sum, s_score, s_factor, s_es;
  sh_q[d] = qh[d];
  float acc = 0.0f;                    /* this thread's out[d] accumulator */
  if (d == 0) { s_max = -1e30f; s_sum = 0.0f; }
  __syncthreads();
  float scale = rsqrtf((float)head_dim);
  for (int t = 0; t < seq_len; ++t) {
    const float *kr = kv_k + t * kvn + kv_off;
    /* score = dot(q, k) * scale  via block reduction */
    sh_red[d] = sh_q[d] * kr[d];
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
      if (d < st) sh_red[d] += sh_red[d + st];
      __syncthreads();
    }
    if (d == 0) {
      float score = sh_red[0] * scale;
      float nm = s_max > score ? s_max : score;
      s_factor = expf(s_max - nm);
      s_es = expf(score - nm);
      s_sum = s_sum * s_factor + s_es;
      s_max = nm;
    }
    __syncthreads();
    const float *vr = kv_v + t * kvn + kv_off;
    acc = acc * s_factor + s_es * vr[d];
    __syncthreads();
  }
  out[head * head_dim + d] = s_sum > 0.0f ? acc / s_sum : 0.0f;
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

/* ---------- matmul helper: y[rows] = W[rows x cols] @ x[cols], device fp32 ---- */
void gemv(GpuCtx *c, const GpuW &w, const float *d_in, float *d_out) {
  int n = w.cols;
  int blk = 256, grid = (n + blk - 1) / blk;
  k_f32_to_f16<<<grid, blk>>>(c->d_xh, d_in, n);
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
  if (w->quantized || w->f32) {
    g.d = upload_weight_fp16(w);
    g.rows = (int)w->rows;
    g.cols = (int)w->cols;
  }
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
      g->q_norm = upload_fp32(L->q_norm, c->head_dim);
      g->k_norm = upload_fp32(L->k_norm, c->head_dim);
    }
  }
  c->final_norm = upload_fp32(m->final_norm, c->h);
  c->lm_head = mkw(&m->lm_head);

  /* KV cache (device) */
  size_t kv_elems = (size_t)m->n_kv_layers * c->kv_ctx * c->kv_stride;
  CDIE(cudaMalloc(&c->kv_k, kv_elems * sizeof(float)));
  CDIE(cudaMalloc(&c->kv_v, kv_elems * sizeof(float)));

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
  c->h_pin_cap = (size_t)(c->vocab > c->h ? c->vocab : c->h);
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
                     int want_logits, float *logits_host) {
  GpuCtx *c = g_ctx;
  const int h = c->h, hd = c->head_dim, nh = c->n_heads, kvh = c->kv_heads;
  const int B = 256;
  auto G = [&](int n) { return (n + B - 1) / B; };

  memcpy(c->h_pin, embed_host, h * sizeof(float));
  CDIE(cudaMemcpy(c->d_x, c->h_pin, h * sizeof(float), cudaMemcpyHostToDevice));

  for (int l = 0; l < c->n_layers; ++l) {
    GpuLayer *g = &c->layers[l];
    k_rms_norm<<<1, 256>>>(c->d_norm, c->d_x, g->attn_norm, h, c->rms_eps);

    if (!g->is_gdn) {
      int qg_len = g->wq.rows, kvn = c->kv_stride, q_len = g->wo.cols;
      int q_heads = q_len / hd;
      gemv(c, g->wq, c->d_norm, c->d_qg);
      gemv(c, g->wk, c->d_norm, c->d_k);
      gemv(c, g->wv, c->d_norm, c->d_v);
      float *qptr = c->d_qg, *gate = nullptr;
      if (qg_len >= 2 * q_len) {
        k_deinterleave<<<G(q_heads * hd), B>>>(c->d_qg, c->d_q, c->d_gate,
                                               q_heads, hd);
        qptr = c->d_q;
        gate = c->d_gate;
      }
      if (g->q_norm)
        k_head_rms_norm<<<q_heads, 256>>>(qptr, g->q_norm, hd, c->rms_eps);
      if (g->k_norm)
        k_head_rms_norm<<<kvh, 256>>>(c->d_k, g->k_norm, hd, c->rms_eps);
      k_rope<<<G(q_heads * (hd / 2)), B>>>(qptr, hd, q_heads, (int)pos,
                                           c->rope_theta, c->rope_dim,
                                           c->yarn_factor, c->yarn_orig_ctx);
      k_rope<<<G(kvh * (hd / 2)), B>>>(c->d_k, hd, kvh, (int)pos, c->rope_theta,
                                       c->rope_dim, c->yarn_factor,
                                       c->yarn_orig_ctx);
      float *kv_k = c->kv_k + (size_t)g->kv_slot * c->kv_ctx * c->kv_stride;
      float *kv_v = c->kv_v + (size_t)g->kv_slot * c->kv_ctx * c->kv_stride;
      k_kv_append<<<G(kvn), B>>>(kv_k, kv_v, c->d_k, c->d_v, (int)pos, c->kv_ctx,
                                 kvn);
      int seq_len = (int)pos + 1;
      if (seq_len > c->kv_ctx) seq_len = c->kv_ctx;
      k_attention<<<nh, hd>>>(c->d_attn, qptr, kv_k, kv_v, seq_len, nh, kvh, hd);
      if (gate) k_sigmoid_gate<<<G(q_len), B>>>(c->d_attn, gate, q_len);
      gemv(c, g->wo, c->d_attn, c->d_tmp);
      k_add<<<G(h), B>>>(c->d_x, c->d_tmp, h);
    } else {
      int qo = g->qkv_out, vd = g->value_dim, kd = g->key_dim;
      int nvh = g->n_v_heads, hk = g->head_k, hv = g->head_v;
      gemv(c, g->qkv, c->d_norm, c->d_mixed);
      gemv(c, g->ssm_alpha, c->d_norm, c->d_a);
      if (g->ssm_beta.d) gemv(c, g->ssm_beta, c->d_norm, c->d_b);
      else cudaMemset(c->d_b, 0, nvh * sizeof(float));
      gemv(c, g->gdn_gate, c->d_norm, c->d_z);
      k_conv_silu<<<G(qo), B>>>(c->d_mixed, c->d_conv, g->ssm_conv1d,
                                g->conv_ring, qo, g->ring_head, g->ring_len);
      k_ring_push<<<G(qo), B>>>(g->conv_ring, c->d_mixed, qo, g->ring_head);
      g->ring_head = (g->ring_head + 1) % CONV_K;
      if (g->ring_len < CONV_K) g->ring_len++;
      float out_scale = 1.0f / sqrtf((float)hv);
      size_t shmem = 2 * hk * sizeof(float);
      k_delta_rule<<<nvh, hv, shmem>>>(c->d_conv, g->state, c->d_core, c->d_a,
                                       c->d_b, g->ssm_a, g->ssm_dt_bias, qo, kd,
                                       vd, g->n_k_heads, hk, hv, g->a_baked,
                                       out_scale);
      k_gated_rms_norm<<<nvh, 128>>>(c->d_core, g->ssm_norm, c->d_z, hv,
                                     c->rms_eps);
      gemv(c, g->ssm_out, c->d_core, c->d_tmp);
      k_add<<<G(h), B>>>(c->d_x, c->d_tmp, h);
    }

    /* FFN */
    k_rms_norm<<<1, 256>>>(c->d_norm, c->d_x, g->ffn_norm, h, c->rms_eps);
    gemv(c, g->gate, c->d_norm, c->d_gf);
    gemv(c, g->up, c->d_norm, c->d_uf);
    k_swiglu<<<G(c->inter), B>>>(c->d_gf, c->d_uf, c->inter);
    gemv(c, g->down, c->d_gf, c->d_tmp);
    k_add<<<G(h), B>>>(c->d_x, c->d_tmp, h);
  }

  if (!want_logits) {
    cudaDeviceSynchronize();
    return;
  }
  k_rms_norm<<<1, 256>>>(c->d_norm, c->d_x, c->final_norm, h, c->rms_eps);
  gemv(c, c->lm_head, c->d_norm, c->d_logits);
  CDIE(cudaMemcpy(logits_host, c->d_logits, c->vocab * sizeof(float),
                  cudaMemcpyDeviceToHost));
  (void)m;
}

}  // extern "C"
