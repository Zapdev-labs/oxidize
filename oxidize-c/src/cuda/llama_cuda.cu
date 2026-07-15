/* GPU-resident llama-family dense decode. Kernels + host orchestration in one
 * file, mirroring src/cuda/gemma4_cuda.cu.
 *
 * Everything for one token is enqueued on one stream with no intermediate syncs;
 * the only device->host traffic per token is the vocab logits (or nothing, on
 * prefill). Weights stay in their GGUF quantized form; the matvec kernel fuses
 * dequant with the dot product (one warp per output row) over the SHARED dqv<T>
 * (cuda_dequant.cuh) — the exact decoders the gemma4 backend and the CPU forward
 * use, so F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/AL5_XS all work and are held to
 * tests/cuda_equiv.c.
 *
 * The graph mirrors llama_forward()/llama_forward_from() in model_llama.c: input
 * RMSNorm -> Q/K/V projections (+ optional q/k/v bias) -> optional per-head q/k
 * RMSNorm -> RoPE (NeoX split-half OR ggml NORMAL adjacent-pair, per
 * m->rope_norm — llama/mistral/yi ship q/k permuted for NORMAL, so the mode must
 * match or they rotate the wrong pairs) -> GQA full-causal attention -> O
 * projection (+ optional o bias) -> residual -> post-attn RMSNorm -> SwiGLU FFN
 * (SiLU, not GeGLU) -> residual; final RMSNorm -> tied or untied logits. No
 * embedding scale, no logit softcap (both gemma4-only). The KV cache is f16.
 *
 * MoE layers stay on the CPU: a model with a MoE layer inside the GPU-offloaded
 * range is refused at init (use --ngl to keep the MoE tail on the CPU). */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "../gguf.h"
#include "../model_llama.h"
#include "../quant.h"
#include "llama_cuda.h"
}

#include "cuda_dequant.cuh" /* dh, ksm, dqv<T> — shared with gemma4_cuda.cu */

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

/* ======================== kernels ======================== */

/* ---- fused dequant matvec, one warp per row: y[r] = dot(dequant(W row r), x).
 * Decode is nb == 1, so no batch axis (the gemma4 backend keeps a batched
 * variant for a future speculative verify; llama does not need it yet). AL5_XS
 * routes through dqv<OC_AL5_XS> like every other type — the hand-fused unpack is
 * a gemma4 decode-speed optimization, not a correctness need, and dqv<AL5_XS> is
 * already proven by the gate. */
template <int T>
__global__ void lk_matvec(float* __restrict__ y, const uint8_t* __restrict__ W,
                          int rows, int cols, const float* __restrict__ x,
                          size_t rowbytes) {
  int row = (int)(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
  if (row >= rows) return;
  int lane = threadIdx.x & 31;
  const uint8_t* rp = W + (size_t)row * rowbytes;
  float acc = 0.0f;
  for (int i = lane; i < cols; i += 32) acc += dqv<T>(rp, i) * x[i];
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) y[row] = acc;
}

/* ---- embedding row dequant into x (llama does NOT scale embeddings). */
template <int T>
__global__ void lk_embed(float* __restrict__ x, const uint8_t* __restrict__ row,
                         int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) x[i] = dqv<T>(row, i);
}

/* ---- RMSNorm: grid.x independent vectors of length `per`, weight w shared.
 * out may alias x. Mirrors oc_rms_norm (the +1 is baked into the GGUF norm
 * weight at conversion for both llama and gemma). w == NULL => scale 1. */
__global__ void lk_rmsnorm(float* __restrict__ out, const float* __restrict__ x,
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

/* ---- NeoX split-half RoPE, grid.x = heads. Pairs (p[i], p[half+i]); dims
 * [rope_len, head_dim) pass through (partial rotary). freq = theta^(-2i/rope_len),
 * matching oc_rope. Host skips the launch at pos 0 (identity). */
__global__ void lk_rope_neox(float* __restrict__ vec, int head_dim, int pos,
                             float theta, int rope_len) {
  float* p = vec + (size_t)blockIdx.x * head_dim;
  int half = rope_len / 2;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    float freq = powf(theta, -2.0f * (float)i / (float)rope_len);
    float angle = (float)pos * freq;
    float c = cosf(angle), s = sinf(angle);
    float x0 = p[i], x1 = p[half + i];
    p[i] = x0 * c - x1 * s;
    p[half + i] = x0 * s + x1 * c;
  }
}

/* ---- ggml NORMAL RoPE: rotates ADJACENT pairs (p[2i], p[2i+1]). llama.cpp
 * permutes q/k for this on llama/mistral/yi, so the stored layout rotates
 * correctly here. Mirrors oc_rope_normal. */
__global__ void lk_rope_normal(float* __restrict__ vec, int head_dim, int pos,
                               float theta, int rope_len) {
  float* p = vec + (size_t)blockIdx.x * head_dim;
  int half = rope_len / 2;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    float freq = powf(theta, -2.0f * (float)i / (float)rope_len);
    float angle = (float)pos * freq;
    float c = cosf(angle), s = sinf(angle);
    float x0 = p[2 * i], x1 = p[2 * i + 1];
    p[2 * i] = x0 * c - x1 * s;
    p[2 * i + 1] = x0 * s + x1 * c;
  }
}

/* ---- store one token's K/V rows (f32) into the f16 caches at `slot`. The
 * llama cache is LINEAR (cap = ctx, no ring), so slot == pos. */
__global__ void lk_kv_store(__half* __restrict__ kc, __half* __restrict__ vc,
                            const float* __restrict__ k,
                            const float* __restrict__ v, int k_len, int v_len,
                            size_t slot) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < k_len) kc[slot * k_len + i] = __float2half(k[i]);
  if (i < v_len) vc[slot * v_len + i] = __float2half(v[i]);
}

/* ---- fused decode attention: one block per q head, full causal over [t0, t1).
 * Scores materialized in shared memory, then softmax + weighted f16 V. Mirrors
 * attn_heads() in model_llama.c (hd == vd for llama). cache_cap == ctx, so
 * t % cache_cap == t; the modulo is kept so the kernel matches k_attn exactly. */
__global__ void lk_attn(float* __restrict__ out, const float* __restrict__ q,
                        const __half* __restrict__ kc,
                        const __half* __restrict__ vc, int hd, int vd, int group,
                        int cache_cap, int t0, int t1, float scale) {
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

  for (int t = t0 + (int)threadIdx.x; t < t1; t += blockDim.x) {
    const __half* krow = kc + (size_t)(t % cache_cap) * k_row + kvh * hd;
    float dot = 0.0f;
    for (int d = 0; d < hd; ++d) dot += sq[d] * __half2float(krow[d]);
    sp[t - t0] = dot * scale;
  }
  __syncthreads();

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

  for (int d = threadIdx.x; d < vd; d += blockDim.x) {
    float acc = 0.0f;
    for (int t = t0; t < t1; ++t) {
      const __half* vrow = vc + (size_t)(t % cache_cap) * v_row + kvh * vd;
      acc += sp[t - t0] * __half2float(vrow[d]);
    }
    out[(size_t)h * vd + d] = acc * inv;
  }
}

/* ---- c[i] += x[i]: residual adds and the optional bias adds. */
__global__ void lk_add(float* __restrict__ c, const float* __restrict__ x,
                       int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) c[i] += x[i];
}

/* ---- SwiGLU: gate = silu(gate) * up  (SiLU = x*sigmoid(x)). */
__global__ void lk_silu_mul(float* __restrict__ gate,
                            const float* __restrict__ up, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  float g = gate[i];
  gate[i] = (g / (1.0f + expf(-g))) * up[i];
}

/* ======================== host side ======================== */

typedef struct {
  bool is_moe; /* refused at init; kept so free() is uniform */
  uint8_t *q_w, *k_w, *v_w, *o_w, *gate_w, *up_w, *down_w; /* device blobs */
  float *attn_norm, *ffn_norm, *q_norm, *k_norm;           /* device f32, may be NULL */
  float *bias_q, *bias_k, *bias_v, *bias_o;                /* device f32, may be NULL */
  __half *k_cache, *v_cache;
} CudaLayer;

struct LlamaCuda {
  const LlamaModel* m;
  size_t n_gpu_layers; /* layers [0, n_gpu_layers) here; the rest on the CPU */
  cudaStream_t stream;
  CudaLayer* layers;
  /* scratch */
  float *x, *normed, *q, *k, *v, *attn_res, *attn_proj, *gate, *up, *ffn_out;
  uint8_t* tok_embd;  /* embedding rows (always) */
  uint8_t* out_w;     /* untied head; NULL when tied (reuse tok_embd) */
  float* out_norm;    /* only when the GPU owns the head */
  float* logits;
};

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
    case OC_F32: case OC_F16: case OC_Q4_0: case OC_Q8_0:
    case OC_Q4_K: case OC_Q5_K: case OC_Q6_K: case OC_AL5_XS:
      return 0;
    default: break;
  }
  if (err && errlen)
    snprintf(err, errlen,
             "cuda: %s has quant type %u; GPU kernels exist for "
             "F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/AL5_XS only",
             what, t);
  return -1;
}

/* Launch the right fused matvec for a weight's ggml type. */
#define MV(T) lk_matvec<T><<<grid, BLOCK, 0, s>>>(y, W, rows, cols, x, \
                                                  oc_row_bytes(T, (size_t)cols))
static void matvec(cudaStream_t s, uint32_t type, const uint8_t* W, int rows,
                   int cols, const float* x, float* y) {
  const int WARPS = 8, BLOCK = WARPS * 32;
  int grid = (rows + WARPS - 1) / WARPS;
  switch (type) {
    case OC_F32: MV(OC_F32); break;
    case OC_F16: MV(OC_F16); break;
    case OC_Q4_0: MV(OC_Q4_0); break;
    case OC_Q8_0: MV(OC_Q8_0); break;
    case OC_Q4_K: MV(OC_Q4_K); break;
    case OC_Q5_K: MV(OC_Q5_K); break;
    case OC_Q6_K: MV(OC_Q6_K); break;
    case OC_AL5_XS: MV(OC_AL5_XS); break;
    default: break; /* unreachable: check_type() gates every upload */
  }
}
#undef MV

#define EMB(T) lk_embed<T><<<(n + 255) / 256, 256, 0, s>>>(x, row, n)
static void embed(cudaStream_t s, uint32_t type, const uint8_t* row, int n,
                  float* x) {
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

int llama_cuda_init(LlamaCuda** out, const LlamaModel* m, int n_gpus,
                    int n_gpu_layers, char* err, size_t errlen) {
  *out = NULL;
  int dev_count = 0;
  CUDA_TRY(cudaGetDeviceCount(&dev_count));
  if (dev_count < 1) {
    if (err && errlen) snprintf(err, errlen, "cuda: no GPU available");
    return -1;
  }
  if (n_gpus > 1) {
    if (err && errlen)
      snprintf(err, errlen,
               "cuda: llama backend is single-GPU (--gpus 1); the layer-split "
               "pipeline is gemma4-only");
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

  /* MoE layers stay on the CPU. Refuse if the offload range contains one. */
  for (size_t l = 0; l < ngl; ++l)
    if (m->layers[l].is_moe) {
      if (err && errlen)
        snprintf(err, errlen,
                 "cuda: layer %zu is Mixture-of-Experts; MoE is not offloaded. "
                 "Use --ngl %zu to keep the MoE layers on the CPU, or run on CPU",
                 l, l);
      return -1;
    }

  LlamaCuda* c = (LlamaCuda*)calloc(1, sizeof(LlamaCuda));
  if (!c) return -1;
  c->m = m;
  c->n_gpu_layers = ngl;
  c->layers = (CudaLayer*)calloc(ngl, sizeof(CudaLayer));
  if (!c->layers) { free(c); return -1; }

  if (check_type(m->tok_embd->ggml_type, "token_embd", err, errlen) != 0) goto fail;
  if (!partial && m->out_w != m->tok_embd &&
      check_type(m->out_w->ggml_type, "output.weight", err, errlen) != 0)
    goto fail;

  if (cudaSetDevice(0) != cudaSuccess ||
      cudaStreamCreate(&c->stream) != cudaSuccess)
    goto fail_msg;

  {
    size_t hd = m->head_dim;
    size_t q_len = m->n_head * hd, kv_row = m->n_kv_heads * hd;
    if (cudaMalloc(&c->x, m->hidden * 4) != cudaSuccess ||
        cudaMalloc(&c->normed, m->hidden * 4) != cudaSuccess ||
        cudaMalloc(&c->q, q_len * 4) != cudaSuccess ||
        cudaMalloc(&c->k, kv_row * 4) != cudaSuccess ||
        cudaMalloc(&c->v, kv_row * 4) != cudaSuccess ||
        cudaMalloc(&c->attn_res, q_len * 4) != cudaSuccess ||
        cudaMalloc(&c->attn_proj, m->hidden * 4) != cudaSuccess ||
        cudaMalloc(&c->gate, m->inter * 4) != cudaSuccess ||
        cudaMalloc(&c->up, m->inter * 4) != cudaSuccess ||
        cudaMalloc(&c->ffn_out, m->hidden * 4) != cudaSuccess)
      goto fail_msg;
  }

  c->tok_embd = (uint8_t*)dupload(m->tok_embd->data, tensor_bytes(m->tok_embd));
  if (!c->tok_embd) goto fail_msg;
  if (!partial) { /* GPU owns the head */
    c->out_norm = (float*)dupload(m->out_norm, m->hidden * 4);
    if (!c->out_norm || cudaMalloc(&c->logits, m->vocab * 4) != cudaSuccess)
      goto fail_msg;
    if (m->out_w != m->tok_embd) { /* untied */
      c->out_w = (uint8_t*)dupload(m->out_w->data, tensor_bytes(m->out_w));
      if (!c->out_w) goto fail_msg;
    }
  }

  {
    size_t hd = m->head_dim, kv_row = m->n_kv_heads * hd;
    for (size_t l = 0; l < ngl; ++l) {
      const LlamaLayer* L = &m->layers[l];
      CudaLayer* D = &c->layers[l];

#define UP_MAT(dst, src, what)                                          \
  do {                                                                  \
    if (check_type((src)->ggml_type, what, err, errlen) != 0) goto fail; \
    D->dst = (uint8_t*)dupload((src)->data, tensor_bytes(src));         \
    if (!D->dst) goto fail_msg;                                         \
  } while (0)
#define UP_VEC(dst, src, n)                       \
  do {                                            \
    if (L->src) {                                 \
      D->dst = (float*)dupload(L->src, (n) * 4);  \
      if (!D->dst) goto fail_msg;                 \
    }                                             \
  } while (0)

      UP_MAT(q_w, L->attn_q, "attn_q");
      UP_MAT(k_w, L->attn_k, "attn_k");
      UP_MAT(v_w, L->attn_v, "attn_v");
      UP_MAT(o_w, L->attn_out, "attn_output");
      UP_MAT(gate_w, L->ffn_gate, "ffn_gate");
      UP_MAT(up_w, L->ffn_up, "ffn_up");
      UP_MAT(down_w, L->ffn_down, "ffn_down");
      UP_VEC(attn_norm, attn_norm, m->hidden);
      UP_VEC(ffn_norm, ffn_norm, m->hidden);
      UP_VEC(q_norm, attn_q_norm, hd);
      UP_VEC(k_norm, attn_k_norm, hd);
      UP_VEC(bias_q, bias_q, m->n_head * hd);
      UP_VEC(bias_k, bias_k, kv_row);
      UP_VEC(bias_v, bias_v, kv_row);
      UP_VEC(bias_o, bias_o, m->hidden);
#undef UP_MAT
#undef UP_VEC

      size_t kn = m->ctx * kv_row;
      if (cudaMalloc(&D->k_cache, kn * 2) != cudaSuccess ||
          cudaMalloc(&D->v_cache, kn * 2) != cudaSuccess ||
          cudaMemset(D->k_cache, 0, kn * 2) != cudaSuccess ||
          cudaMemset(D->v_cache, 0, kn * 2) != cudaSuccess)
        goto fail_msg;
    }
  }

  /* Raise lk_attn's dynamic smem cap if a large --ctx needs it. */
  {
    size_t need = (m->head_dim + m->ctx + 256) * 4;
    if (need > 48 * 1024) {
      if (cudaFuncSetAttribute(lk_attn,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               (int)need) != cudaSuccess) {
        if (err && errlen)
          snprintf(err, errlen,
                   "cuda: ctx too large for lk_attn shared memory (%zu bytes)",
                   need);
        goto fail;
      }
    }
  }

  fprintf(stderr, "cuda: llama %zu/%zu layers on 1 GPU%s (f16 KV)\n", ngl,
          m->n_layers, partial ? "; remaining layers + head on CPU" : "");
  *out = c;
  return 0;

fail_msg:
  if (err && errlen && !err[0])
    snprintf(err, errlen, "cuda: allocation/upload failed: %s",
             cudaGetErrorString(cudaGetLastError()));
fail:
  llama_cuda_free(c);
  return -1;
}

void llama_cuda_free(LlamaCuda* c) {
  if (!c) return;
  for (size_t l = 0; c->layers && l < c->n_gpu_layers; ++l) {
    CudaLayer* D = &c->layers[l];
    cudaFree(D->q_w); cudaFree(D->k_w); cudaFree(D->v_w); cudaFree(D->o_w);
    cudaFree(D->gate_w); cudaFree(D->up_w); cudaFree(D->down_w);
    cudaFree(D->attn_norm); cudaFree(D->ffn_norm);
    cudaFree(D->q_norm); cudaFree(D->k_norm);
    cudaFree(D->bias_q); cudaFree(D->bias_k); cudaFree(D->bias_v); cudaFree(D->bias_o);
    cudaFree(D->k_cache); cudaFree(D->v_cache);
  }
  free(c->layers);
  cudaFree(c->x); cudaFree(c->normed); cudaFree(c->q); cudaFree(c->k);
  cudaFree(c->v); cudaFree(c->attn_res); cudaFree(c->attn_proj);
  cudaFree(c->gate); cudaFree(c->up); cudaFree(c->ffn_out);
  cudaFree(c->tok_embd); cudaFree(c->out_w);
  cudaFree(c->out_norm); cudaFree(c->logits);
  if (c->stream) cudaStreamDestroy(c->stream);
  free(c);
}

/* One decode step over the GPU's layers.
 *   hidden_out != NULL: the residual stream after the GPU's layers is copied
 *     back (partial offload; no final norm/logits). Takes precedence.
 *   logits_out != NULL: full logits copied (vocab floats) — GPU owns the head.
 *   both NULL (prefill): no sync at all.
 * Returns 0 on success, -1 on CUDA error (printed to stderr). */
static int llama_cuda_forward(LlamaCuda* c, int32_t token, size_t pos,
                              float* logits_out, float* hidden_out) {
  const LlamaModel* m = c->m;
  const int h = (int)m->hidden;
  char* err = NULL; size_t errlen = 0; /* CUDA_TRY prints via the return path */
  if (pos >= m->ctx) {
    fprintf(stderr, "cuda: position %zu exceeds context %zu\n", pos, m->ctx);
    return -1;
  }
  cudaSetDevice(0);
  cudaStream_t s = c->stream;

  /* embedding lookup (no scale) */
  size_t tk = (size_t)token < m->vocab ? (size_t)token : m->vocab - 1;
  size_t emb_row = oc_row_bytes(m->tok_embd->ggml_type, m->hidden);
  embed(s, m->tok_embd->ggml_type, c->tok_embd + tk * emb_row, h, c->x);

  const int hd = (int)m->head_dim;
  const int n_head = (int)m->n_head, n_kv = (int)m->n_kv_heads;
  const int q_len = n_head * hd, kv_len = n_kv * hd;
  const int group = n_head / n_kv;
  const int cap = (int)m->ctx;
  const int rope_len = m->rope_dim ? (int)m->rope_dim : hd;
  const float scale = 1.0f / sqrtf((float)hd);

  for (size_t l = 0; l < c->n_gpu_layers; ++l) {
    const CudaLayer* D = &c->layers[l];

    /* ---- attention ---- */
    lk_rmsnorm<<<1, 256, 0, s>>>(c->normed, c->x, D->attn_norm, h, m->eps);
    matvec(s, m->layers[l].attn_q->ggml_type, D->q_w, q_len, h, c->normed, c->q);
    matvec(s, m->layers[l].attn_k->ggml_type, D->k_w, kv_len, h, c->normed, c->k);
    matvec(s, m->layers[l].attn_v->ggml_type, D->v_w, kv_len, h, c->normed, c->v);
    if (D->bias_q) lk_add<<<(q_len + 255) / 256, 256, 0, s>>>(c->q, D->bias_q, q_len);
    if (D->bias_k) lk_add<<<(kv_len + 255) / 256, 256, 0, s>>>(c->k, D->bias_k, kv_len);
    if (D->bias_v) lk_add<<<(kv_len + 255) / 256, 256, 0, s>>>(c->v, D->bias_v, kv_len);

    /* per-head optional q/k RMSNorm, then rope (V gets neither) */
    if (D->q_norm) lk_rmsnorm<<<n_head, 256, 0, s>>>(c->q, c->q, D->q_norm, hd, m->eps);
    if (D->k_norm) lk_rmsnorm<<<n_kv, 256, 0, s>>>(c->k, c->k, D->k_norm, hd, m->eps);
    if (pos > 0 && rope_len > 0) {
      if (m->rope_norm) {
        lk_rope_normal<<<n_head, 128, 0, s>>>(c->q, hd, (int)pos, m->rope_theta, rope_len);
        lk_rope_normal<<<n_kv, 128, 0, s>>>(c->k, hd, (int)pos, m->rope_theta, rope_len);
      } else {
        lk_rope_neox<<<n_head, 128, 0, s>>>(c->q, hd, (int)pos, m->rope_theta, rope_len);
        lk_rope_neox<<<n_kv, 128, 0, s>>>(c->k, hd, (int)pos, m->rope_theta, rope_len);
      }
    }

    /* linear cache: slot == pos, attend over [0, pos+1) */
    lk_kv_store<<<(kv_len + 255) / 256, 256, 0, s>>>(D->k_cache, D->v_cache, c->k,
                                                     c->v, kv_len, kv_len, pos);
    int seq = (int)pos + 1;
    size_t smem = ((size_t)hd + (size_t)seq + 256) * 4;
    lk_attn<<<n_head, 128, smem, s>>>(c->attn_res, c->q, D->k_cache, D->v_cache,
                                      hd, hd, group, cap, 0, seq, scale);

    matvec(s, m->layers[l].attn_out->ggml_type, D->o_w, h, q_len, c->attn_res,
           c->attn_proj);
    if (D->bias_o) lk_add<<<(h + 255) / 256, 256, 0, s>>>(c->attn_proj, D->bias_o, h);
    lk_add<<<(h + 255) / 256, 256, 0, s>>>(c->x, c->attn_proj, h); /* residual */

    /* ---- FFN (SwiGLU) ---- */
    lk_rmsnorm<<<1, 256, 0, s>>>(c->normed, c->x, D->ffn_norm, h, m->eps);
    matvec(s, m->layers[l].ffn_gate->ggml_type, D->gate_w, (int)m->inter, h,
           c->normed, c->gate);
    matvec(s, m->layers[l].ffn_up->ggml_type, D->up_w, (int)m->inter, h, c->normed,
           c->up);
    lk_silu_mul<<<((int)m->inter + 255) / 256, 256, 0, s>>>(c->gate, c->up,
                                                            (int)m->inter);
    matvec(s, m->layers[l].ffn_down->ggml_type, D->down_w, h, (int)m->inter,
           c->gate, c->ffn_out);
    lk_add<<<(h + 255) / 256, 256, 0, s>>>(c->x, c->ffn_out, h); /* residual */
  }

  if (hidden_out) { /* partial offload: hand the residual stream to the CPU */
    CUDA_TRY(cudaMemcpyAsync(hidden_out, c->x, (size_t)h * 4,
                             cudaMemcpyDeviceToHost, s));
    cudaError_t e = cudaStreamSynchronize(s);
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e != cudaSuccess) {
      fprintf(stderr, "cuda: forward failed: %s\n", cudaGetErrorString(e));
      return -1;
    }
    return 0;
  }

  if (!logits_out) { /* prefill: fully async, no sync */
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
      fprintf(stderr, "cuda: launch error: %s\n", cudaGetErrorString(e));
      return -1;
    }
    return 0;
  }

  /* final norm + tied/untied logits on the GPU (no softcap for llama) */
  const uint8_t* head_w = c->out_w ? c->out_w : c->tok_embd;
  lk_rmsnorm<<<1, 256, 0, s>>>(c->normed, c->x, c->out_norm, h, m->eps);
  matvec(s, m->out_w->ggml_type, head_w, (int)m->vocab, h, c->normed, c->logits);
  CUDA_TRY(cudaMemcpyAsync(logits_out, c->logits, m->vocab * 4,
                           cudaMemcpyDeviceToHost, s));
  cudaError_t e = cudaStreamSynchronize(s); /* the ONE sync per token */
  if (e == cudaSuccess) e = cudaGetLastError();
  if (e != cudaSuccess) {
    fprintf(stderr, "cuda: forward failed: %s\n", cudaGetErrorString(e));
    return -1;
  }
  return 0;
}

float* llama_cuda_step(LlamaCuda* c, LlamaModel* m, int32_t token, size_t pos,
                       bool need_logits, int* failed) {
  *failed = 0;
  if (c->n_gpu_layers == m->n_layers) { /* whole stack on the GPU */
    float* lg = need_logits ? m->logits : NULL;
    if (llama_cuda_forward(c, token, pos, lg, NULL) != 0) {
      *failed = 1;
      return NULL;
    }
    m->kv_len = pos + 1;
    return lg;
  }
  /* partial offload: GPU layers [0, ngl), then the CPU finishes from m->x */
  if (llama_cuda_forward(c, token, pos, NULL, m->x) != 0) {
    *failed = 1;
    return NULL;
  }
  return llama_forward_from(m, pos, c->n_gpu_layers, need_logits);
}
