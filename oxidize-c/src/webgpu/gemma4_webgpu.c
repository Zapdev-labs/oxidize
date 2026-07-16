/* ======================================================================
 * UNVERIFIED — NEVER COMPILED OR RUN. Written BLIND against
 * src/cuda/gemma4_cuda.cu + webgpu_common.*. Requires Dawn/emdawn.
 * MAY NOT COMPILE. MAY BE WRONG. No equivalence test has been run.
 * ====================================================================== */
#include "gemma4_webgpu.h"
#include "webgpu_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../quant.h"

#define UBO_POOL 1024
#define ARGMAX_BLOCKS 256
#define UBO_SIZE 64

typedef struct {
  WgpuBuf q_w, k_w, v_w, o_w, gate_w, up_w, down_w;
  WgpuBuf attn_norm, q_norm, k_norm, post_attn_norm, ffn_norm, post_ffn_norm;
  WgpuBuf k_cache, v_cache; /* packed f16 */
  int has_v;
  int has_q_norm, has_k_norm, has_post_attn, has_post_ffn;
} WgpuG4Layer;

struct Gemma4WebGpu {
  const Gemma4Model* m;
  size_t n_gpu_layers;
  WgpuCtx ctx;
  WgpuKernels k;
  WgpuG4Layer* layers;
  /* scratch */
  WgpuBuf x, normed, q, k, v, attn_res, attn_proj, gate, up, ffn_out;
  WgpuBuf tok_embd, out_norm, logits, rope_freqs, dummy, attn_scratch;
  WgpuBuf red_max, red_idx, argmax_d, bidx_dummy;
  /* uniform pool: one UBO per dispatch in a token (avoids overwrite races) */
  WgpuBuf ubos[UBO_POOL];
  int ubo_i;
  int owns_head;
  size_t max_q, max_k, max_hd, max_cap;
};

static size_t tensor_bytes(const GgufTensorInfo* t) {
  size_t cols = (size_t)t->dims[0], rows = 1;
  for (uint32_t d = 1; d < t->n_dims; ++d) rows *= (size_t)t->dims[d];
  return rows * oc_row_bytes(t->ggml_type, cols);
}

static size_t even_up(size_t n) { return (n + 1u) & ~1ull; }

static WgpuBuf* ubo_next(Gemma4WebGpu* c) {
  if (c->ubo_i >= UBO_POOL) return NULL;
  return &c->ubos[c->ubo_i++];
}

static int ubo_set(Gemma4WebGpu* c, WgpuBuf** out, const void* src, size_t n) {
  WgpuBuf* u = ubo_next(c);
  if (!u) return -1;
  if (wgpu_upload_uniform(&c->ctx, u, src, n) != 0) return -1;
  *out = u;
  return 0;
}

static int upload_bytes(WgpuCtx* ctx, WgpuBuf* b, const void* src, size_t n) {
  if (wgpu_buf_device(ctx, n, b) != 0) return -1;
  return wgpu_upload(ctx, b, src, n);
}

/* ---- POD uniforms matching .wgsl (16-byte aligned fields) ---- */
typedef struct { uint32_t rows, cols, qtype, rowbytes; } MvU;
typedef struct { uint32_t n, qtype, rowbase; float scale; } EmbU;
typedef struct { uint32_t per, has_w; float eps; uint32_t pad; } RmsU;
typedef struct {
  uint32_t head_dim, rope_len, pos, mode;
  float theta;
  uint32_t has_freqs, pad0, pad1;
} RopeU;
typedef struct { uint32_t k_len, v_len, slot, pad; } KvU;
typedef struct {
  uint32_t hd, vd, group, cache_cap, t0, t1, n_heads, pad;
  float scale;
  uint32_t p1, p2, p3;
} AttnU;
typedef struct { uint32_t n, p0, p1, p2; } N4U;
typedef struct { uint32_t n, pad; float s; uint32_t p1; } ResidU;
typedef struct { uint32_t n, pad; float c; uint32_t p1; } SoftU;
typedef struct { uint32_t n, mode, p0, p1; } ArgU;

static void dispatch_mv(Gemma4WebGpu* c, WgpuRec* r, const WgpuBuf* W,
                        uint32_t qtype, int rows, int cols, const WgpuBuf* x,
                        WgpuBuf* y) {
  MvU u = {(uint32_t)rows, (uint32_t)cols, qtype,
           (uint32_t)oc_row_bytes(qtype, (size_t)cols)};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* bufs[] = {(WgpuBuf*)W, ubo, (WgpuBuf*)x, y};
  wgpu_dispatch(r, &c->k.matvec, bufs, (uint32_t)rows, 1, 1);
}

static void dispatch_embed(Gemma4WebGpu* c, WgpuRec* r, uint32_t qtype,
                           uint32_t rowbase, int n, float scale, WgpuBuf* x) {
  EmbU u = {(uint32_t)n, qtype, rowbase, scale};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* bufs[] = {&c->tok_embd, ubo, x};
  wgpu_dispatch(r, &c->k.embed, bufs, ((uint32_t)n + 255) / 256, 1, 1);
}

static void dispatch_rms(Gemma4WebGpu* c, WgpuRec* r, WgpuBuf* outv,
                         const WgpuBuf* xin, const WgpuBuf* w, int per,
                         int has_w, float eps, uint32_t n_vecs) {
  RmsU u = {(uint32_t)per, (uint32_t)has_w, eps, 0};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* ww = has_w ? (WgpuBuf*)w : &c->dummy;
  WgpuBuf* bufs[] = {ubo, (WgpuBuf*)xin, ww, outv};
  wgpu_dispatch(r, &c->k.rmsnorm, bufs, n_vecs, 1, 1);
}

static void dispatch_rope(Gemma4WebGpu* c, WgpuRec* r, WgpuBuf* vec, int hd,
                          int rope_len, size_t pos, float theta, int mode,
                          int has_freqs, uint32_t n_heads) {
  RopeU u = {(uint32_t)hd, (uint32_t)rope_len, (uint32_t)pos, (uint32_t)mode,
             theta, (uint32_t)has_freqs, 0, 0};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* freqs = has_freqs ? &c->rope_freqs : &c->dummy;
  WgpuBuf* bufs[] = {ubo, vec, freqs};
  wgpu_dispatch(r, &c->k.rope, bufs, n_heads, 1, 1);
}

static void dispatch_add(Gemma4WebGpu* c, WgpuRec* r, WgpuBuf* dst,
                         const WgpuBuf* src, int n) {
  N4U u = {(uint32_t)n, 0, 0, 0};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* bufs[] = {ubo, dst, (WgpuBuf*)src};
  wgpu_dispatch(r, &c->k.add, bufs, ((uint32_t)n + 255) / 256, 1, 1);
}

static void dispatch_geglu(Gemma4WebGpu* c, WgpuRec* r, WgpuBuf* gate,
                           const WgpuBuf* up, int n) {
  N4U u = {(uint32_t)n, 0, 0, 0};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* bufs[] = {ubo, gate, (WgpuBuf*)up};
  wgpu_dispatch(r, &c->k.geglu, bufs, ((uint32_t)n + 255) / 256, 1, 1);
}

static void dispatch_resid(Gemma4WebGpu* c, WgpuRec* r, WgpuBuf* x,
                           const WgpuBuf* ffn, const WgpuBuf* attn, float s,
                           int n) {
  ResidU u = {(uint32_t)n, 0, s, 0};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* bufs[] = {ubo, x, (WgpuBuf*)ffn, (WgpuBuf*)attn};
  wgpu_dispatch(r, &c->k.resid_out, bufs, ((uint32_t)n + 255) / 256, 1, 1);
}

static void dispatch_kv(Gemma4WebGpu* c, WgpuRec* r, WgpuG4Layer* D, int k_len,
                        int v_len, size_t slot) {
  /* Pad lengths to even for packed f16 word ownership in kv_store.wgsl. */
  uint32_t kl = (uint32_t)even_up((size_t)k_len);
  uint32_t vl = (uint32_t)even_up((size_t)v_len);
  KvU u = {kl, vl, (uint32_t)slot, 0};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* bufs[] = {ubo, &D->k_cache, &D->v_cache, &c->k, &c->v};
  uint32_t words = ((kl > vl ? kl : vl) + 1u) >> 1u;
  wgpu_dispatch(r, &c->k.kv_store, bufs, (words + 255u) / 256u, 1, 1);
}

static void dispatch_attn(Gemma4WebGpu* c, WgpuRec* r, WgpuG4Layer* D, int hd,
                          int vd, int group, int cache_cap, int t0, int t1,
                          int n_head, float scale) {
  AttnU u = {(uint32_t)hd,         (uint32_t)vd,        (uint32_t)group,
             (uint32_t)cache_cap,  (uint32_t)t0,        (uint32_t)t1,
             (uint32_t)n_head,     0,                   scale,
             0,                    0,                   0};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* bufs[] = {ubo, &c->attn_res, &c->q, &D->k_cache, &D->v_cache,
                     &c->attn_scratch};
  wgpu_dispatch(r, &c->k.attn, bufs, (uint32_t)n_head, 1, 1);
}

int gemma4_webgpu_init(Gemma4WebGpu** out, const Gemma4Model* m, int n_gpus,
                       int n_gpu_layers, char* err, size_t errlen) {
  *out = NULL;
  if (n_gpus != 1) {
    if (err && errlen)
      snprintf(err, errlen,
               "webgpu: multi-GPU is not supported (got --gpus %d); use 1",
               n_gpus);
    return -1;
  }
  if (m->kv_quant) {
    if (err && errlen)
      snprintf(err, errlen,
               "webgpu: rotoquant KV (kv_quant) is not ported; load with "
               "kv_quant=false for the f16 KV path");
    return -1;
  }

  size_t ngl = (n_gpu_layers < 0 || (size_t)n_gpu_layers > m->n_layers)
                   ? m->n_layers
                   : (size_t)n_gpu_layers;
  if (ngl == 0) {
    if (err && errlen)
      snprintf(err, errlen, "webgpu: --ngl 0 is the pure-CPU path (no GPU init)");
    return -1;
  }
  int partial = ngl < m->n_layers;

  size_t max_q = m->hidden, max_k = 0, max_hd = 0, max_cap = 0;
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
    if (L->cache_cap > max_cap) max_cap = L->cache_cap;
    if (L->head_dim > 256 || L->v_head_dim > 256) {
      if (err && errlen)
        snprintf(err, errlen,
                 "webgpu: head_dim/v_head_dim > 256 refused (attn.wgsl limit)");
      return -1;
    }
    if ((L->n_kv_heads * L->head_dim) & 1u ||
        (L->n_kv_heads * L->v_head_dim) & 1u) {
      if (err && errlen)
        snprintf(err, errlen,
                 "webgpu: odd KV row on layer %zu; packed f16 requires even",
                 l);
      return -1;
    }
  }

  if (wgpu_check_type(m->tok_embd->ggml_type, "token_embd", err, errlen) != 0)
    return -1;

  Gemma4WebGpu* c = (Gemma4WebGpu*)calloc(1, sizeof(*c));
  if (!c) return -1;
  c->m = m;
  c->n_gpu_layers = ngl;
  c->owns_head = !partial;
  c->max_q = max_q;
  c->max_k = max_k;
  c->max_hd = max_hd;
  c->max_cap = max_cap;

  if (wgpu_ctx_init(&c->ctx, err, errlen) != 0) goto fail;
  {
    const char* dir = getenv("OXIDIZE_WEBGPU_WGSL");
    if (wgpu_kernels_init(&c->ctx, &c->k, dir, err, errlen) != 0) goto fail;
  }

  for (int i = 0; i < UBO_POOL; ++i)
    if (wgpu_buf_uniform(&c->ctx, UBO_SIZE, &c->ubos[i]) != 0) goto fail_msg;

  c->layers = (WgpuG4Layer*)calloc(ngl, sizeof(WgpuG4Layer));
  if (!c->layers) goto fail_msg;

  if (wgpu_buf_device(&c->ctx, m->hidden * 4, &c->x) != 0 ||
      wgpu_buf_device(&c->ctx, m->hidden * 4, &c->normed) != 0 ||
      wgpu_buf_device(&c->ctx, max_q * 4, &c->q) != 0 ||
      wgpu_buf_device(&c->ctx, max_k * 4, &c->k) != 0 ||
      wgpu_buf_device(&c->ctx, max_k * 4, &c->v) != 0 ||
      wgpu_buf_device(&c->ctx, max_q * 4, &c->attn_res) != 0 ||
      wgpu_buf_device(&c->ctx, m->hidden * 4, &c->attn_proj) != 0 ||
      wgpu_buf_device(&c->ctx, m->inter * 4, &c->gate) != 0 ||
      wgpu_buf_device(&c->ctx, m->inter * 4, &c->up) != 0 ||
      wgpu_buf_device(&c->ctx, m->hidden * 4, &c->ffn_out) != 0 ||
      wgpu_buf_device(&c->ctx, 4, &c->dummy) != 0 ||
      wgpu_buf_device(&c->ctx, m->n_head * max_cap * 4, &c->attn_scratch) != 0)
    goto fail_msg;

  if (upload_bytes(&c->ctx, &c->tok_embd, m->tok_embd->data,
                   tensor_bytes(m->tok_embd)) != 0)
    goto fail_msg;
  if (m->rope_freqs && max_hd >= 2) {
    if (upload_bytes(&c->ctx, &c->rope_freqs, m->rope_freqs,
                     (max_hd / 2) * 4) != 0)
      goto fail_msg;
  } else {
    if (wgpu_buf_device(&c->ctx, 4, &c->rope_freqs) != 0) goto fail_msg;
  }

  if (c->owns_head) {
    if (upload_bytes(&c->ctx, &c->out_norm, m->out_norm, m->hidden * 4) != 0 ||
        wgpu_buf_device(&c->ctx, m->vocab * 4, &c->logits) != 0 ||
        wgpu_buf_device(&c->ctx, ARGMAX_BLOCKS * 4, &c->red_max) != 0 ||
        wgpu_buf_device(&c->ctx, ARGMAX_BLOCKS * 4, &c->red_idx) != 0 ||
        wgpu_buf_device(&c->ctx, 4, &c->argmax_d) != 0 ||
        wgpu_buf_device(&c->ctx, 4, &c->bidx_dummy) != 0)
      goto fail_msg;
  }

  for (size_t l = 0; l < ngl; ++l) {
    const Gemma4Layer* L = &m->layers[l];
    WgpuG4Layer* D = &c->layers[l];
#define UP_MAT(dst, src, what)                                                 \
  do {                                                                         \
    if (wgpu_check_type((src)->ggml_type, what, err, errlen) != 0) goto fail;  \
    if (upload_bytes(&c->ctx, &D->dst, (src)->data, tensor_bytes(src)) != 0)   \
      goto fail_msg;                                                          \
  } while (0)
#define UP_VEC(dst, src, n, flag)                                              \
  do {                                                                         \
    if (L->src) {                                                              \
      if (upload_bytes(&c->ctx, &D->dst, L->src, (n)*4) != 0) goto fail_msg;   \
      D->flag = 1;                                                             \
    }                                                                          \
  } while (0)

    UP_MAT(q_w, L->attn_q, "attn_q");
    UP_MAT(k_w, L->attn_k, "attn_k");
    if (L->attn_v) {
      UP_MAT(v_w, L->attn_v, "attn_v");
      D->has_v = 1;
    }
    UP_MAT(o_w, L->attn_out, "attn_output");
    UP_MAT(gate_w, L->ffn_gate, "ffn_gate");
    UP_MAT(up_w, L->ffn_up, "ffn_up");
    UP_MAT(down_w, L->ffn_down, "ffn_down");
    if (L->attn_norm) {
      if (upload_bytes(&c->ctx, &D->attn_norm, L->attn_norm, m->hidden * 4) != 0)
        goto fail_msg;
    }
    UP_VEC(q_norm, attn_q_norm, L->head_dim, has_q_norm);
    UP_VEC(k_norm, attn_k_norm, L->head_dim, has_k_norm);
    UP_VEC(post_attn_norm, post_attn_norm, m->hidden, has_post_attn);
    if (L->ffn_norm) {
      if (upload_bytes(&c->ctx, &D->ffn_norm, L->ffn_norm, m->hidden * 4) != 0)
        goto fail_msg;
    }
    UP_VEC(post_ffn_norm, post_ffn_norm, m->hidden, has_post_ffn);
#undef UP_MAT
#undef UP_VEC

    size_t k_row = even_up(L->n_kv_heads * L->head_dim);
    size_t v_row = even_up(L->n_kv_heads * L->v_head_dim);
    size_t kn = L->cache_cap * k_row * 2; /* packed f16 bytes */
    size_t vn = L->cache_cap * v_row * 2;
    if (wgpu_buf_device(&c->ctx, kn, &D->k_cache) != 0 ||
        wgpu_buf_device(&c->ctx, vn, &D->v_cache) != 0 ||
        wgpu_zero(&c->ctx, &D->k_cache, kn) != 0 ||
        wgpu_zero(&c->ctx, &D->v_cache, vn) != 0)
      goto fail_msg;
  }

  fprintf(stderr, "webgpu: gemma4 %zu/%zu layers on 1 GPU%s (f16 KV, UNVERIFIED)\n",
          ngl, m->n_layers, partial ? "; remaining layers + head on CPU" : "");
  *out = c;
  return 0;

fail_msg:
  if (err && errlen && err[0] == 0)
    snprintf(err, errlen, "webgpu: allocation/upload failed");
fail:
  gemma4_webgpu_free(c);
  return -1;
}

void gemma4_webgpu_free(Gemma4WebGpu* c) {
  if (!c) return;
  for (size_t l = 0; c->layers && l < c->n_gpu_layers; ++l) {
    WgpuG4Layer* D = &c->layers[l];
    wgpu_buf_free(&c->ctx, &D->q_w);
    wgpu_buf_free(&c->ctx, &D->k_w);
    wgpu_buf_free(&c->ctx, &D->v_w);
    wgpu_buf_free(&c->ctx, &D->o_w);
    wgpu_buf_free(&c->ctx, &D->gate_w);
    wgpu_buf_free(&c->ctx, &D->up_w);
    wgpu_buf_free(&c->ctx, &D->down_w);
    wgpu_buf_free(&c->ctx, &D->attn_norm);
    wgpu_buf_free(&c->ctx, &D->q_norm);
    wgpu_buf_free(&c->ctx, &D->k_norm);
    wgpu_buf_free(&c->ctx, &D->post_attn_norm);
    wgpu_buf_free(&c->ctx, &D->ffn_norm);
    wgpu_buf_free(&c->ctx, &D->post_ffn_norm);
    wgpu_buf_free(&c->ctx, &D->k_cache);
    wgpu_buf_free(&c->ctx, &D->v_cache);
  }
  free(c->layers);
  wgpu_buf_free(&c->ctx, &c->x);
  wgpu_buf_free(&c->ctx, &c->normed);
  wgpu_buf_free(&c->ctx, &c->q);
  wgpu_buf_free(&c->ctx, &c->k);
  wgpu_buf_free(&c->ctx, &c->v);
  wgpu_buf_free(&c->ctx, &c->attn_res);
  wgpu_buf_free(&c->ctx, &c->attn_proj);
  wgpu_buf_free(&c->ctx, &c->gate);
  wgpu_buf_free(&c->ctx, &c->up);
  wgpu_buf_free(&c->ctx, &c->ffn_out);
  wgpu_buf_free(&c->ctx, &c->tok_embd);
  wgpu_buf_free(&c->ctx, &c->out_norm);
  wgpu_buf_free(&c->ctx, &c->logits);
  wgpu_buf_free(&c->ctx, &c->rope_freqs);
  wgpu_buf_free(&c->ctx, &c->dummy);
  wgpu_buf_free(&c->ctx, &c->attn_scratch);
  wgpu_buf_free(&c->ctx, &c->red_max);
  wgpu_buf_free(&c->ctx, &c->red_idx);
  wgpu_buf_free(&c->ctx, &c->argmax_d);
  wgpu_buf_free(&c->ctx, &c->bidx_dummy);
  for (int i = 0; i < UBO_POOL; ++i) wgpu_buf_free(&c->ctx, &c->ubos[i]);
  wgpu_kernels_free(&c->ctx, &c->k);
  wgpu_ctx_free(&c->ctx);
  free(c);
}

int gemma4_webgpu_forward(Gemma4WebGpu* c, int32_t token, size_t pos,
                          float* logits_out, int32_t* argmax_out,
                          float* hidden_out) {
  const Gemma4Model* m = c->m;
  const int h = (int)m->hidden;
  if (pos >= m->ctx) {
    fprintf(stderr, "webgpu: position %zu exceeds context %zu\n", pos, m->ctx);
    return -1;
  }
  c->ubo_i = 0;

  WgpuRec rec;
  if (wgpu_rec_begin(&c->ctx, &rec) != 0) return -1;

  size_t tk = (size_t)token < m->vocab ? (size_t)token : m->vocab - 1;
  size_t emb_row = oc_row_bytes(m->tok_embd->ggml_type, m->hidden);
  dispatch_embed(c, &rec, m->tok_embd->ggml_type, (uint32_t)(tk * emb_row), h,
                 m->emb_scale, &c->x);

  for (size_t l = 0; l < c->n_gpu_layers; ++l) {
    const Gemma4Layer* L = &m->layers[l];
    WgpuG4Layer* D = &c->layers[l];
    int hd = (int)L->head_dim, vd = (int)L->v_head_dim;
    int n_head = (int)m->n_head, n_kv = (int)L->n_kv_heads;
    int q_len = n_head * hd, k_len = n_kv * hd, v_len = n_kv * vd;
    int group = n_head / n_kv;

    dispatch_rms(c, &rec, &c->normed, &c->x, &D->attn_norm, h, 1, m->eps, 1);
    dispatch_mv(c, &rec, &D->q_w, L->attn_q->ggml_type, q_len, h, &c->normed,
                &c->q);
    dispatch_mv(c, &rec, &D->k_w, L->attn_k->ggml_type, k_len, h, &c->normed,
                &c->k);
    if (D->has_v)
      dispatch_mv(c, &rec, &D->v_w, L->attn_v->ggml_type, v_len, h, &c->normed,
                  &c->v);
    else
      /* K=V layers: V is the RAW K projection (before k_norm/rope). Recompute
       * into v with the same k_w — equivalent to CUDA's D2D copy of k->v. */
      dispatch_mv(c, &rec, &D->k_w, L->attn_k->ggml_type, k_len, h, &c->normed,
                  &c->v);

    if (D->has_q_norm)
      dispatch_rms(c, &rec, &c->q, &c->q, &D->q_norm, hd, 1, m->eps,
                   (uint32_t)n_head);
    if (D->has_k_norm)
      dispatch_rms(c, &rec, &c->k, &c->k, &D->k_norm, hd, 1, m->eps,
                   (uint32_t)n_kv);

    int rope_len = L->rope.rope_dim ? (int)L->rope.rope_dim : hd;
    int has_freqs = (!L->is_swa && c->rope_freqs.buf) ? 1 : 0;
    if (pos > 0 && rope_len > 0) {
      dispatch_rope(c, &rec, &c->q, hd, rope_len, pos, L->rope.theta, 0,
                    has_freqs, (uint32_t)n_head);
      dispatch_rope(c, &rec, &c->k, hd, rope_len, pos, L->rope.theta, 0,
                    has_freqs, (uint32_t)n_kv);
    }
    dispatch_rms(c, &rec, &c->v, &c->v, &c->dummy, vd, 0, m->eps,
                 (uint32_t)n_kv);

    size_t slot = pos % L->cache_cap;
    size_t seq = pos + 1;
    size_t t0 = seq > L->cache_cap ? seq - L->cache_cap : 0;
    float scale =
        m->attn_scale > 0.0f ? m->attn_scale : 1.0f / sqrtf((float)hd);
    dispatch_kv(c, &rec, D, k_len, v_len, slot);
    dispatch_attn(c, &rec, D, hd, vd, group, (int)L->cache_cap, (int)t0,
                  (int)seq, n_head, scale);

    dispatch_mv(c, &rec, &D->o_w, L->attn_out->ggml_type, h, n_head * vd,
                &c->attn_res, &c->attn_proj);
    if (D->has_post_attn)
      dispatch_rms(c, &rec, &c->attn_proj, &c->attn_proj, &D->post_attn_norm, h,
                   1, m->eps, 1);
    dispatch_add(c, &rec, &c->attn_proj, &c->x, h);

    dispatch_rms(c, &rec, &c->normed, &c->attn_proj, &D->ffn_norm, h, 1, m->eps,
                 1);
    dispatch_mv(c, &rec, &D->gate_w, L->ffn_gate->ggml_type, (int)m->inter, h,
                &c->normed, &c->gate);
    dispatch_mv(c, &rec, &D->up_w, L->ffn_up->ggml_type, (int)m->inter, h,
                &c->normed, &c->up);
    dispatch_geglu(c, &rec, &c->gate, &c->up, (int)m->inter);
    dispatch_mv(c, &rec, &D->down_w, L->ffn_down->ggml_type, h, (int)m->inter,
                &c->gate, &c->ffn_out);
    if (D->has_post_ffn)
      dispatch_rms(c, &rec, &c->ffn_out, &c->ffn_out, &D->post_ffn_norm, h, 1,
                   m->eps, 1);
    dispatch_resid(c, &rec, &c->x, &c->ffn_out, &c->attn_proj, L->output_scale,
                   h);
  }

  if (hidden_out) {
    if (wgpu_rec_submit_wait(&rec) != 0) return -1;
    return wgpu_download(&c->ctx, &c->x, hidden_out, (uint64_t)h * 4);
  }

  if (!logits_out && !argmax_out) {
    return wgpu_rec_submit_wait(&rec);
  }

  /* final norm + tied logits */
  dispatch_rms(c, &rec, &c->normed, &c->x, &c->out_norm, h, 1, m->eps, 1);
  dispatch_mv(c, &rec, &c->tok_embd, m->tok_embd->ggml_type, (int)m->vocab, h,
              &c->normed, &c->logits);

  if (argmax_out) {
    ArgU a1 = {(uint32_t)m->vocab, 0, 0, 0};
    WgpuBuf* ubo = NULL;
    if (ubo_set(c, &ubo, &a1, sizeof(a1)) != 0) return -1;
    WgpuBuf* b1[] = {ubo, &c->logits, &c->bidx_dummy, &c->red_max, &c->red_idx};
    wgpu_dispatch(&rec, &c->k.argmax, b1, ARGMAX_BLOCKS, 1, 1);
    ArgU a2 = {ARGMAX_BLOCKS, 1, 0, 0};
    if (ubo_set(c, &ubo, &a2, sizeof(a2)) != 0) return -1;
    WgpuBuf* b2[] = {ubo, &c->red_max, &c->red_idx, &c->red_max, &c->argmax_d};
    wgpu_dispatch(&rec, &c->k.argmax, b2, 1, 1, 1);
  }
  if (logits_out && m->final_softcap > 0.0f) {
    SoftU s = {(uint32_t)m->vocab, 0, m->final_softcap, 0};
    WgpuBuf* ubo = NULL;
    if (ubo_set(c, &ubo, &s, sizeof(s)) != 0) return -1;
    WgpuBuf* bufs[] = {ubo, &c->logits};
    wgpu_dispatch(&rec, &c->k.softcap, bufs, ((uint32_t)m->vocab + 255) / 256, 1,
                  1);
  }

  if (wgpu_rec_submit_wait(&rec) != 0) return -1;
  if (argmax_out) {
    if (wgpu_download(&c->ctx, &c->argmax_d, argmax_out, 4) != 0) return -1;
  }
  if (logits_out) {
    if (wgpu_download(&c->ctx, &c->logits, logits_out, m->vocab * 4) != 0)
      return -1;
  }
  return 0;
}

float* gemma4_webgpu_step(Gemma4WebGpu* c, Gemma4Model* m, int32_t token,
                          size_t pos, bool need_logits, int* failed) {
  *failed = 0;
  if (c->n_gpu_layers == m->n_layers) {
    float* lg = need_logits ? m->logits : NULL;
    if (gemma4_webgpu_forward(c, token, pos, lg, NULL, NULL) != 0) {
      *failed = 1;
      return NULL;
    }
    m->kv_len = pos + 1;
    return lg;
  }
  if (gemma4_webgpu_forward(c, token, pos, NULL, NULL, m->x) != 0) {
    *failed = 1;
    return NULL;
  }
  return gemma4_forward_from(m, pos, c->n_gpu_layers, need_logits);
}
