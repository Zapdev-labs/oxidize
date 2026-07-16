/* ======================================================================
 * UNVERIFIED — NEVER COMPILED OR RUN. Written BLIND against
 * src/cuda/llama_cuda.cu + webgpu_common.*. Requires Dawn/emdawn.
 * MAY NOT COMPILE. MAY BE WRONG. No equivalence test has been run.
 * ====================================================================== */
#include "llama_webgpu.h"
#include "webgpu_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../quant.h"

#define UBO_POOL 1024
#define UBO_SIZE 64

typedef struct {
  WgpuBuf q_w, k_w, v_w, o_w, gate_w, up_w, down_w;
  WgpuBuf attn_norm, ffn_norm, q_norm, k_norm;
  WgpuBuf bias_q, bias_k, bias_v, bias_o;
  WgpuBuf k_cache, v_cache; /* packed f16 */
  int has_q_norm, has_k_norm;
  int has_bias_q, has_bias_k, has_bias_v, has_bias_o;
} WgpuLlamaLayer;

struct LlamaWebGpu {
  const LlamaModel* m;
  size_t n_gpu_layers;
  WgpuCtx ctx;
  WgpuKernels k;
  WgpuLlamaLayer* layers;
  WgpuBuf x, normed, q, k, v, attn_res, attn_proj, gate, up, ffn_out;
  WgpuBuf tok_embd, out_w, out_norm, logits, dummy, attn_scratch, rope_dummy;
  WgpuBuf ubos[UBO_POOL];
  int ubo_i;
  int owns_head;
  int tied_head; /* out_w aliases tok_embd */
};

static size_t tensor_bytes(const GgufTensorInfo* t) {
  size_t cols = (size_t)t->dims[0], rows = 1;
  for (uint32_t d = 1; d < t->n_dims; ++d) rows *= (size_t)t->dims[d];
  return rows * oc_row_bytes(t->ggml_type, cols);
}

static size_t even_up(size_t n) { return (n + 1u) & ~1ull; }

static WgpuBuf* ubo_next(LlamaWebGpu* c) {
  if (c->ubo_i >= UBO_POOL) return NULL;
  return &c->ubos[c->ubo_i++];
}

static int ubo_set(LlamaWebGpu* c, WgpuBuf** out, const void* src, size_t n) {
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

static void dispatch_mv(LlamaWebGpu* c, WgpuRec* r, const WgpuBuf* W,
                        uint32_t qtype, int rows, int cols, const WgpuBuf* x,
                        WgpuBuf* y) {
  MvU u = {(uint32_t)rows, (uint32_t)cols, qtype,
           (uint32_t)oc_row_bytes(qtype, (size_t)cols)};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* bufs[] = {(WgpuBuf*)W, ubo, (WgpuBuf*)x, y};
  wgpu_dispatch(r, &c->k.matvec, bufs, (uint32_t)rows, 1, 1);
}

static void dispatch_embed(LlamaWebGpu* c, WgpuRec* r, uint32_t qtype,
                           uint32_t rowbase, int n, WgpuBuf* x) {
  EmbU u = {(uint32_t)n, qtype, rowbase, 1.0f};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* bufs[] = {&c->tok_embd, ubo, x};
  wgpu_dispatch(r, &c->k.embed, bufs, ((uint32_t)n + 255) / 256, 1, 1);
}

static void dispatch_rms(LlamaWebGpu* c, WgpuRec* r, WgpuBuf* outv,
                         const WgpuBuf* xin, const WgpuBuf* w, int per,
                         int has_w, float eps, uint32_t n_vecs) {
  RmsU u = {(uint32_t)per, (uint32_t)has_w, eps, 0};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* ww = has_w ? (WgpuBuf*)w : &c->dummy;
  WgpuBuf* bufs[] = {ubo, (WgpuBuf*)xin, ww, outv};
  wgpu_dispatch(r, &c->k.rmsnorm, bufs, n_vecs, 1, 1);
}

static void dispatch_rope(LlamaWebGpu* c, WgpuRec* r, WgpuBuf* vec, int hd,
                          int rope_len, size_t pos, float theta, int mode,
                          uint32_t n_heads) {
  RopeU u = {(uint32_t)hd, (uint32_t)rope_len, (uint32_t)pos, (uint32_t)mode,
             theta, 0, 0, 0};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* bufs[] = {ubo, vec, &c->rope_dummy};
  wgpu_dispatch(r, &c->k.rope, bufs, n_heads, 1, 1);
}

static void dispatch_add(LlamaWebGpu* c, WgpuRec* r, WgpuBuf* dst,
                         const WgpuBuf* src, int n) {
  N4U u = {(uint32_t)n, 0, 0, 0};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* bufs[] = {ubo, dst, (WgpuBuf*)src};
  wgpu_dispatch(r, &c->k.add, bufs, ((uint32_t)n + 255) / 256, 1, 1);
}

static void dispatch_silu(LlamaWebGpu* c, WgpuRec* r, WgpuBuf* gate,
                          const WgpuBuf* up, int n) {
  N4U u = {(uint32_t)n, 0, 0, 0};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* bufs[] = {ubo, gate, (WgpuBuf*)up};
  wgpu_dispatch(r, &c->k.silu_mul, bufs, ((uint32_t)n + 255) / 256, 1, 1);
}

static void dispatch_kv(LlamaWebGpu* c, WgpuRec* r, WgpuLlamaLayer* D, int kv_len,
                        size_t slot) {
  uint32_t kl = (uint32_t)even_up((size_t)kv_len);
  KvU u = {kl, kl, (uint32_t)slot, 0};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* bufs[] = {ubo, &D->k_cache, &D->v_cache, &c->k, &c->v};
  uint32_t words = (kl + 1u) >> 1u;
  wgpu_dispatch(r, &c->k.kv_store, bufs, (words + 255u) / 256u, 1, 1);
}

static void dispatch_attn(LlamaWebGpu* c, WgpuRec* r, WgpuLlamaLayer* D, int hd,
                          int group, int cap, int t1, int n_head, float scale) {
  AttnU u = {(uint32_t)hd, (uint32_t)hd, (uint32_t)group, (uint32_t)cap,
             0,            (uint32_t)t1, (uint32_t)n_head, 0,
             scale,        0,            0,               0};
  WgpuBuf* ubo = NULL;
  if (ubo_set(c, &ubo, &u, sizeof(u)) != 0) return;
  WgpuBuf* bufs[] = {ubo, &c->attn_res, &c->q, &D->k_cache, &D->v_cache,
                     &c->attn_scratch};
  wgpu_dispatch(r, &c->k.attn, bufs, (uint32_t)n_head, 1, 1);
}

static int llama_webgpu_forward(LlamaWebGpu* c, int32_t token, size_t pos,
                                float* logits_out, float* hidden_out);

int llama_webgpu_init(LlamaWebGpu** out, const LlamaModel* m, int n_gpus,
                      int n_gpu_layers, char* err, size_t errlen) {
  *out = NULL;
  if (n_gpus != 1) {
    if (err && errlen)
      snprintf(err, errlen,
               "webgpu: llama backend is single-GPU (--gpus 1); got %d",
               n_gpus);
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

  for (size_t l = 0; l < ngl; ++l)
    if (m->layers[l].is_moe) {
      if (err && errlen)
        snprintf(err, errlen,
                 "webgpu: layer %zu is Mixture-of-Experts; MoE is not "
                 "offloaded. Use --ngl %zu to keep MoE on the CPU, or run CPU",
                 l, l);
      return -1;
    }

  if (m->head_dim > 256) {
    if (err && errlen)
      snprintf(err, errlen, "webgpu: head_dim > 256 refused (attn.wgsl limit)");
    return -1;
  }

  size_t kv_row = m->n_kv_heads * m->head_dim;
  if (kv_row & 1u) {
    if (err && errlen)
      snprintf(err, errlen,
               "webgpu: odd KV row length %zu; packed f16 store requires even",
               kv_row);
    return -1;
  }

  if (wgpu_check_type(m->tok_embd->ggml_type, "token_embd", err, errlen) != 0)
    return -1;
  if (!partial && m->out_w != m->tok_embd &&
      wgpu_check_type(m->out_w->ggml_type, "output.weight", err, errlen) != 0)
    return -1;

  LlamaWebGpu* c = (LlamaWebGpu*)calloc(1, sizeof(*c));
  if (!c) return -1;
  c->m = m;
  c->n_gpu_layers = ngl;
  c->owns_head = !partial;
  c->tied_head = (m->out_w == m->tok_embd);

  if (wgpu_ctx_init(&c->ctx, err, errlen) != 0) goto fail;
  {
    const char* dir = getenv("OXIDIZE_WEBGPU_WGSL");
    if (wgpu_kernels_init(&c->ctx, &c->k, dir, err, errlen) != 0) goto fail;
  }
  for (int i = 0; i < UBO_POOL; ++i)
    if (wgpu_buf_uniform(&c->ctx, UBO_SIZE, &c->ubos[i]) != 0) goto fail_msg;

  c->layers = (WgpuLlamaLayer*)calloc(ngl, sizeof(WgpuLlamaLayer));
  if (!c->layers) goto fail_msg;

  size_t hd = m->head_dim;
  size_t q_len = m->n_head * hd;
  if (wgpu_buf_device(&c->ctx, m->hidden * 4, &c->x) != 0 ||
      wgpu_buf_device(&c->ctx, m->hidden * 4, &c->normed) != 0 ||
      wgpu_buf_device(&c->ctx, q_len * 4, &c->q) != 0 ||
      wgpu_buf_device(&c->ctx, kv_row * 4, &c->k) != 0 ||
      wgpu_buf_device(&c->ctx, kv_row * 4, &c->v) != 0 ||
      wgpu_buf_device(&c->ctx, q_len * 4, &c->attn_res) != 0 ||
      wgpu_buf_device(&c->ctx, m->hidden * 4, &c->attn_proj) != 0 ||
      wgpu_buf_device(&c->ctx, m->inter * 4, &c->gate) != 0 ||
      wgpu_buf_device(&c->ctx, m->inter * 4, &c->up) != 0 ||
      wgpu_buf_device(&c->ctx, m->hidden * 4, &c->ffn_out) != 0 ||
      wgpu_buf_device(&c->ctx, 4, &c->dummy) != 0 ||
      wgpu_buf_device(&c->ctx, 4, &c->rope_dummy) != 0 ||
      wgpu_buf_device(&c->ctx, m->n_head * m->ctx * 4, &c->attn_scratch) != 0)
    goto fail_msg;

  if (upload_bytes(&c->ctx, &c->tok_embd, m->tok_embd->data,
                   tensor_bytes(m->tok_embd)) != 0)
    goto fail_msg;

  if (c->owns_head) {
    if (upload_bytes(&c->ctx, &c->out_norm, m->out_norm, m->hidden * 4) != 0 ||
        wgpu_buf_device(&c->ctx, m->vocab * 4, &c->logits) != 0)
      goto fail_msg;
    if (!c->tied_head) {
      if (upload_bytes(&c->ctx, &c->out_w, m->out_w->data,
                       tensor_bytes(m->out_w)) != 0)
        goto fail_msg;
    }
  }

  for (size_t l = 0; l < ngl; ++l) {
    const LlamaLayer* L = &m->layers[l];
    WgpuLlamaLayer* D = &c->layers[l];
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
    UP_MAT(v_w, L->attn_v, "attn_v");
    UP_MAT(o_w, L->attn_out, "attn_output");
    UP_MAT(gate_w, L->ffn_gate, "ffn_gate");
    UP_MAT(up_w, L->ffn_up, "ffn_up");
    UP_MAT(down_w, L->ffn_down, "ffn_down");
    if (L->attn_norm) {
      if (upload_bytes(&c->ctx, &D->attn_norm, L->attn_norm, m->hidden * 4) != 0)
        goto fail_msg;
    }
    if (L->ffn_norm) {
      if (upload_bytes(&c->ctx, &D->ffn_norm, L->ffn_norm, m->hidden * 4) != 0)
        goto fail_msg;
    }
    UP_VEC(q_norm, attn_q_norm, hd, has_q_norm);
    UP_VEC(k_norm, attn_k_norm, hd, has_k_norm);
    UP_VEC(bias_q, bias_q, m->n_head * hd, has_bias_q);
    UP_VEC(bias_k, bias_k, kv_row, has_bias_k);
    UP_VEC(bias_v, bias_v, kv_row, has_bias_v);
    UP_VEC(bias_o, bias_o, m->hidden, has_bias_o);
#undef UP_MAT
#undef UP_VEC

    size_t kn = m->ctx * even_up(kv_row) * 2;
    if (wgpu_buf_device(&c->ctx, kn, &D->k_cache) != 0 ||
        wgpu_buf_device(&c->ctx, kn, &D->v_cache) != 0 ||
        wgpu_zero(&c->ctx, &D->k_cache, kn) != 0 ||
        wgpu_zero(&c->ctx, &D->v_cache, kn) != 0)
      goto fail_msg;
  }

  fprintf(stderr, "webgpu: llama %zu/%zu layers on 1 GPU%s (f16 KV, UNVERIFIED)\n",
          ngl, m->n_layers, partial ? "; remaining layers + head on CPU" : "");
  *out = c;
  return 0;

fail_msg:
  if (err && errlen && err[0] == 0)
    snprintf(err, errlen, "webgpu: allocation/upload failed");
fail:
  llama_webgpu_free(c);
  return -1;
}

void llama_webgpu_free(LlamaWebGpu* c) {
  if (!c) return;
  for (size_t l = 0; c->layers && l < c->n_gpu_layers; ++l) {
    WgpuLlamaLayer* D = &c->layers[l];
    wgpu_buf_free(&c->ctx, &D->q_w);
    wgpu_buf_free(&c->ctx, &D->k_w);
    wgpu_buf_free(&c->ctx, &D->v_w);
    wgpu_buf_free(&c->ctx, &D->o_w);
    wgpu_buf_free(&c->ctx, &D->gate_w);
    wgpu_buf_free(&c->ctx, &D->up_w);
    wgpu_buf_free(&c->ctx, &D->down_w);
    wgpu_buf_free(&c->ctx, &D->attn_norm);
    wgpu_buf_free(&c->ctx, &D->ffn_norm);
    wgpu_buf_free(&c->ctx, &D->q_norm);
    wgpu_buf_free(&c->ctx, &D->k_norm);
    wgpu_buf_free(&c->ctx, &D->bias_q);
    wgpu_buf_free(&c->ctx, &D->bias_k);
    wgpu_buf_free(&c->ctx, &D->bias_v);
    wgpu_buf_free(&c->ctx, &D->bias_o);
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
  wgpu_buf_free(&c->ctx, &c->out_w);
  wgpu_buf_free(&c->ctx, &c->out_norm);
  wgpu_buf_free(&c->ctx, &c->logits);
  wgpu_buf_free(&c->ctx, &c->dummy);
  wgpu_buf_free(&c->ctx, &c->rope_dummy);
  wgpu_buf_free(&c->ctx, &c->attn_scratch);
  for (int i = 0; i < UBO_POOL; ++i) wgpu_buf_free(&c->ctx, &c->ubos[i]);
  wgpu_kernels_free(&c->ctx, &c->k);
  wgpu_ctx_free(&c->ctx);
  free(c);
}

static int llama_webgpu_forward(LlamaWebGpu* c, int32_t token, size_t pos,
                                float* logits_out, float* hidden_out) {
  const LlamaModel* m = c->m;
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
                 &c->x);

  const int hd = (int)m->head_dim;
  const int n_head = (int)m->n_head, n_kv = (int)m->n_kv_heads;
  const int q_len = n_head * hd, kv_len = n_kv * hd;
  const int group = n_head / n_kv;
  const int cap = (int)m->ctx;
  const int rope_len = m->rope_dim ? (int)m->rope_dim : hd;
  const float scale = 1.0f / sqrtf((float)hd);
  const int rope_mode = m->rope_norm ? 1 : 0; /* 1=normal, 0=neox */

  for (size_t l = 0; l < c->n_gpu_layers; ++l) {
    WgpuLlamaLayer* D = &c->layers[l];
    const LlamaLayer* L = &m->layers[l];

    dispatch_rms(c, &rec, &c->normed, &c->x, &D->attn_norm, h, 1, m->eps, 1);
    dispatch_mv(c, &rec, &D->q_w, L->attn_q->ggml_type, q_len, h, &c->normed,
                &c->q);
    dispatch_mv(c, &rec, &D->k_w, L->attn_k->ggml_type, kv_len, h, &c->normed,
                &c->k);
    dispatch_mv(c, &rec, &D->v_w, L->attn_v->ggml_type, kv_len, h, &c->normed,
                &c->v);
    if (D->has_bias_q) dispatch_add(c, &rec, &c->q, &D->bias_q, q_len);
    if (D->has_bias_k) dispatch_add(c, &rec, &c->k, &D->bias_k, kv_len);
    if (D->has_bias_v) dispatch_add(c, &rec, &c->v, &D->bias_v, kv_len);

    if (D->has_q_norm)
      dispatch_rms(c, &rec, &c->q, &c->q, &D->q_norm, hd, 1, m->eps,
                   (uint32_t)n_head);
    if (D->has_k_norm)
      dispatch_rms(c, &rec, &c->k, &c->k, &D->k_norm, hd, 1, m->eps,
                   (uint32_t)n_kv);
    if (pos > 0 && rope_len > 0) {
      dispatch_rope(c, &rec, &c->q, hd, rope_len, pos, m->rope_theta, rope_mode,
                    (uint32_t)n_head);
      dispatch_rope(c, &rec, &c->k, hd, rope_len, pos, m->rope_theta, rope_mode,
                    (uint32_t)n_kv);
    }

    dispatch_kv(c, &rec, D, kv_len, pos);
    dispatch_attn(c, &rec, D, hd, group, cap, (int)pos + 1, n_head, scale);

    dispatch_mv(c, &rec, &D->o_w, L->attn_out->ggml_type, h, q_len, &c->attn_res,
                &c->attn_proj);
    if (D->has_bias_o) dispatch_add(c, &rec, &c->attn_proj, &D->bias_o, h);
    dispatch_add(c, &rec, &c->x, &c->attn_proj, h);

    dispatch_rms(c, &rec, &c->normed, &c->x, &D->ffn_norm, h, 1, m->eps, 1);
    dispatch_mv(c, &rec, &D->gate_w, L->ffn_gate->ggml_type, (int)m->inter, h,
                &c->normed, &c->gate);
    dispatch_mv(c, &rec, &D->up_w, L->ffn_up->ggml_type, (int)m->inter, h,
                &c->normed, &c->up);
    dispatch_silu(c, &rec, &c->gate, &c->up, (int)m->inter);
    dispatch_mv(c, &rec, &D->down_w, L->ffn_down->ggml_type, h, (int)m->inter,
                &c->gate, &c->ffn_out);
    dispatch_add(c, &rec, &c->x, &c->ffn_out, h);
  }

  if (hidden_out) {
    if (wgpu_rec_submit_wait(&rec) != 0) return -1;
    return wgpu_download(&c->ctx, &c->x, hidden_out, (uint64_t)h * 4);
  }
  if (!logits_out) return wgpu_rec_submit_wait(&rec);

  WgpuBuf* head = c->tied_head ? &c->tok_embd : &c->out_w;
  dispatch_rms(c, &rec, &c->normed, &c->x, &c->out_norm, h, 1, m->eps, 1);
  dispatch_mv(c, &rec, head, m->out_w->ggml_type, (int)m->vocab, h, &c->normed,
              &c->logits);
  if (wgpu_rec_submit_wait(&rec) != 0) return -1;
  return wgpu_download(&c->ctx, &c->logits, logits_out, m->vocab * 4);
}

float* llama_webgpu_step(LlamaWebGpu* c, LlamaModel* m, int32_t token,
                         size_t pos, bool need_logits, int* failed) {
  *failed = 0;
  if (c->n_gpu_layers == m->n_layers) {
    float* lg = need_logits ? m->logits : NULL;
    if (llama_webgpu_forward(c, token, pos, lg, NULL) != 0) {
      *failed = 1;
      return NULL;
    }
    m->kv_len = pos + 1;
    return lg;
  }
  if (llama_webgpu_forward(c, token, pos, NULL, m->x) != 0) {
    *failed = 1;
    return NULL;
  }
  return llama_forward_from(m, pos, c->n_gpu_layers, need_logits);
}
