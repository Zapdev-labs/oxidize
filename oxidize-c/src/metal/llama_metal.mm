/* ============================================================================
 * UNVERIFIED — this file has NEVER been compiled or run. It was written BLIND
 * against the verified CUDA reference (src/cuda/llama_cuda.cu) and the Rust
 * Metal backend (oxidize-core/src/backends/metal.rs). It requires macOS + Xcode
 * Metal and an Apple GPU to compile and validate. It MAY NOT COMPILE. No
 * equivalence gate has ever been run against it. Treat every logit it would
 * produce as unproven.
 * ============================================================================
 *
 * GPU-resident llama-family dense decode host orchestration. Kernels live in
 * llama.metal (lk_*); this file mirrors the host side of llama_cuda.cu.
 *
 * Graph: input RMSNorm -> Q/K/V (+ optional bias) -> optional per-head q/k
 * RMSNorm -> RoPE (NeoX OR ggml NORMAL per m->rope_norm) -> GQA attention ->
 * O (+ optional bias) -> residual -> post-attn RMSNorm -> SwiGLU -> residual;
 * final RMSNorm -> tied/untied logits. f16 KV. MoE refused in offload range.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CoreFoundation/CoreFoundation.h>

#include "llama_metal.h"
#include "metal_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "../gguf.h"
#include "../quant.h"
}

typedef struct {
  MtBuf q_w, k_w, v_w, o_w, gate_w, up_w, down_w;
  MtBuf attn_norm, ffn_norm, q_norm, k_norm;
  MtBuf bias_q, bias_k, bias_v, bias_o;
  MtBuf k_cache, v_cache;
  int has_q_norm, has_k_norm;
  int has_bias_q, has_bias_k, has_bias_v, has_bias_o;
} MetalLayer;

struct LlamaMetal {
  const LlamaModel* m;
  size_t n_gpu_layers;
  MtCtx ctx;
  MetalLayer* layers;
  MtBuf x, normed, q, k, v, attn_res, attn_proj, gate, up, ffn_out;
  MtBuf tok_embd, out_w, out_norm, logits;
  int has_out_w; /* untied head */
  int owns_head;
};

static size_t tensor_bytes(const GgufTensorInfo* t) {
  size_t cols = (size_t)t->dims[0], rows = 1;
  for (uint32_t d = 1; d < t->n_dims; ++d) rows *= (size_t)t->dims[d];
  return rows * oc_row_bytes(t->ggml_type, cols);
}

static int check_type(uint32_t t, const char* what, char* err, size_t errlen) {
  if (mt_qidx(t) >= 0) return 0;
  if (err && errlen)
    snprintf(err, errlen,
             "metal: %s has quant type %u; GPU kernels exist for "
             "F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/AL5_XS only",
             what, t);
  return -1;
}

static int upload_mat(MtCtx* ctx, MtBuf* dst, const GgufTensorInfo* src,
                      const char* what, char* err, size_t errlen) {
  if (check_type(src->ggml_type, what, err, errlen) != 0) return -1;
  size_t n = tensor_bytes(src);
  if (mt_buf_alloc(ctx, n, dst) != 0) return -1;
  return mt_buf_upload(ctx, dst, src->data, n);
}

static int upload_vec(MtCtx* ctx, MtBuf* dst, const float* src, size_t n) {
  if (!src) return 0;
  if (mt_buf_alloc(ctx, n * 4, dst) != 0) return -1;
  return mt_buf_upload(ctx, dst, src, n * 4);
}

static int enc_matvec(MtRec* r, uint32_t type, const MtBuf* W, int rows,
                      int cols, const MtBuf* x, MtBuf* y) {
  const char* name = mt_matvec_name("lk", type);
  if (!name) return -1;
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, name, &p, err, sizeof(err)) != 0) {
    fprintf(stderr, "%s\n", err);
    return -1;
  }
  int rows_i = rows, cols_i = cols;
  uint32_t rowbytes = (uint32_t)oc_row_bytes(type, (size_t)cols);
  MtBind b[] = {MT_BUF(y),
                MT_BUF(W),
                MT_BYTES(&rows_i, sizeof(rows_i)),
                MT_BYTES(&cols_i, sizeof(cols_i)),
                MT_BUF(x),
                MT_BYTES(&rowbytes, sizeof(rowbytes))};
  mt_dispatch(r, &p, b, 6, 0, mt_matvec_grid(rows), mt_matvec_tgs());
  return 0;
}

static int enc_embed(MtRec* r, uint32_t type, const MtBuf* table, size_t row_off,
                     int n, MtBuf* x) {
  const char* name = mt_embed_name("lk", type);
  if (!name) return -1;
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, name, &p, err, sizeof(err)) != 0) {
    fprintf(stderr, "%s\n", err);
    return -1;
  }
  int n_i = n;
  MtBind b[] = {MT_BUF(x), MT_BUF_OFF(table, row_off),
                MT_BYTES(&n_i, sizeof(n_i))};
  mt_dispatch(r, &p, b, 3, 0, mt_grid_1d((size_t)n, 256), mt_tgs_1d(256));
  return 0;
}

static int enc_rmsnorm(MtRec* r, MtBuf* out, const MtBuf* x, const MtBuf* w,
                       int per, float eps, int nvec, int has_w) {
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, "lk_rmsnorm", &p, err, sizeof(err)) != 0) {
    fprintf(stderr, "%s\n", err);
    return -1;
  }
  int per_i = per, has = has_w;
  MtBind b[] = {MT_BUF(out), MT_BUF(x), MT_BUF(w), MT_BYTES(&per_i, sizeof(per_i)),
                MT_BYTES(&eps, sizeof(eps)), MT_BYTES(&has, sizeof(has))};
  MtSize grid = {(size_t)nvec, 1, 1};
  mt_dispatch(r, &p, b, 6, 0, grid, mt_tgs_1d(256));
  return 0;
}

static int enc_rope(MtRec* r, int rope_norm, MtBuf* vec, int head_dim, int pos,
                    float theta, int rope_len, int nheads) {
  const char* name = rope_norm ? "lk_rope_normal" : "lk_rope_neox";
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, name, &p, err, sizeof(err)) != 0) {
    fprintf(stderr, "%s\n", err);
    return -1;
  }
  int hd = head_dim, ps = pos, rl = rope_len;
  MtBind b[] = {MT_BUF(vec), MT_BYTES(&hd, sizeof(hd)), MT_BYTES(&ps, sizeof(ps)),
                MT_BYTES(&theta, sizeof(theta)), MT_BYTES(&rl, sizeof(rl))};
  MtSize grid = {(size_t)nheads, 1, 1};
  mt_dispatch(r, &p, b, 5, 0, grid, mt_tgs_1d(128));
  return 0;
}

static int enc_kv_store(MtRec* r, const MtBuf* kc, const MtBuf* vc,
                        const MtBuf* k, const MtBuf* v, int k_len, int v_len,
                        uint32_t slot) {
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, "lk_kv_store", &p, err, sizeof(err)) != 0) return -1;
  int kl = k_len, vl = v_len;
  MtBind b[] = {MT_BUF(kc), MT_BUF(vc), MT_BUF(k), MT_BUF(v),
                MT_BYTES(&kl, sizeof(kl)), MT_BYTES(&vl, sizeof(vl)),
                MT_BYTES(&slot, sizeof(slot))};
  mt_dispatch(r, &p, b, 7, 0, mt_grid_1d((size_t)k_len, 256), mt_tgs_1d(256));
  return 0;
}

static int enc_attn(MtRec* r, MtBuf* out, const MtBuf* q, const MtBuf* kc,
                    const MtBuf* vc, int hd, int vd, int group, int cache_cap,
                    int t0, int t1, float scale, int n_head) {
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, "lk_attn", &p, err, sizeof(err)) != 0) return -1;
  int count = t1 - t0;
  size_t smem = ((size_t)hd + (size_t)count + 256) * sizeof(float);
  MtBind b[] = {MT_BUF(out),
                MT_BUF(q),
                MT_BUF(kc),
                MT_BUF(vc),
                MT_BYTES(&hd, sizeof(hd)),
                MT_BYTES(&vd, sizeof(vd)),
                MT_BYTES(&group, sizeof(group)),
                MT_BYTES(&cache_cap, sizeof(cache_cap)),
                MT_BYTES(&t0, sizeof(t0)),
                MT_BYTES(&t1, sizeof(t1)),
                MT_BYTES(&scale, sizeof(scale))};
  MtSize grid = {(size_t)n_head, 1, 1};
  mt_dispatch(r, &p, b, 11, smem, grid, mt_tgs_1d(128));
  return 0;
}

static int enc_add(MtRec* r, MtBuf* c, const MtBuf* x, int n) {
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, "lk_add", &p, err, sizeof(err)) != 0) return -1;
  MtBind b[] = {MT_BUF(c), MT_BUF(x), MT_BYTES(&n, sizeof(n))};
  mt_dispatch(r, &p, b, 3, 0, mt_grid_1d((size_t)n, 256), mt_tgs_1d(256));
  return 0;
}

static int enc_silu_mul(MtRec* r, MtBuf* gate, const MtBuf* up, int n) {
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, "lk_silu_mul", &p, err, sizeof(err)) != 0) return -1;
  MtBind b[] = {MT_BUF(gate), MT_BUF(up), MT_BYTES(&n, sizeof(n))};
  mt_dispatch(r, &p, b, 3, 0, mt_grid_1d((size_t)n, 256), mt_tgs_1d(256));
  return 0;
}

static int llama_metal_forward(LlamaMetal* c, int32_t token, size_t pos,
                               float* logits_out, float* hidden_out);

int llama_metal_init(LlamaMetal** out, const LlamaModel* m, int n_gpus,
                     int n_gpu_layers, char* err, size_t errlen) {
  *out = NULL;
  if (n_gpus > 1) {
    if (err && errlen)
      snprintf(err, errlen,
               "metal: llama backend is single-GPU (--gpus 1); the layer-split "
               "pipeline is gemma4-CUDA-only");
    return -1;
  }
  size_t ngl = (n_gpu_layers < 0 || (size_t)n_gpu_layers > m->n_layers)
                   ? m->n_layers
                   : (size_t)n_gpu_layers;
  if (ngl == 0) {
    if (err && errlen)
      snprintf(err, errlen, "metal: --ngl 0 is the pure-CPU path (no GPU init)");
    return -1;
  }
  bool partial = ngl < m->n_layers;

  for (size_t l = 0; l < ngl; ++l)
    if (m->layers[l].is_moe) {
      if (err && errlen)
        snprintf(err, errlen,
                 "metal: layer %zu is Mixture-of-Experts; MoE is not offloaded. "
                 "Use --ngl %zu to keep the MoE layers on the CPU, or run on CPU",
                 l, l);
      return -1;
    }

  LlamaMetal* c = (LlamaMetal*)calloc(1, sizeof(LlamaMetal));
  if (!c) return -1;
  c->m = m;
  c->n_gpu_layers = ngl;
  c->owns_head = !partial;
  c->layers = (MetalLayer*)calloc(ngl, sizeof(MetalLayer));
  if (!c->layers) {
    free(c);
    return -1;
  }

  if (mt_ctx_init(&c->ctx, NULL, err, errlen) != 0) goto fail;
  if (check_type(m->tok_embd->ggml_type, "token_embd", err, errlen) != 0)
    goto fail;
  if (!partial && m->out_w != m->tok_embd &&
      check_type(m->out_w->ggml_type, "output.weight", err, errlen) != 0)
    goto fail;

  {
    size_t hd = m->head_dim;
    size_t q_len = m->n_head * hd, kv_row = m->n_kv_heads * hd;
    if (mt_buf_alloc(&c->ctx, m->hidden * 4, &c->x) != 0 ||
        mt_buf_alloc(&c->ctx, m->hidden * 4, &c->normed) != 0 ||
        mt_buf_alloc(&c->ctx, q_len * 4, &c->q) != 0 ||
        mt_buf_alloc(&c->ctx, kv_row * 4, &c->k) != 0 ||
        mt_buf_alloc(&c->ctx, kv_row * 4, &c->v) != 0 ||
        mt_buf_alloc(&c->ctx, q_len * 4, &c->attn_res) != 0 ||
        mt_buf_alloc(&c->ctx, m->hidden * 4, &c->attn_proj) != 0 ||
        mt_buf_alloc(&c->ctx, m->inter * 4, &c->gate) != 0 ||
        mt_buf_alloc(&c->ctx, m->inter * 4, &c->up) != 0 ||
        mt_buf_alloc(&c->ctx, m->hidden * 4, &c->ffn_out) != 0)
      goto fail_msg;
  }

  if (upload_mat(&c->ctx, &c->tok_embd, m->tok_embd, "token_embd", err,
                 errlen) != 0)
    goto fail;
  if (!partial) {
    if (upload_vec(&c->ctx, &c->out_norm, m->out_norm, m->hidden) != 0 ||
        mt_buf_alloc(&c->ctx, m->vocab * 4, &c->logits) != 0)
      goto fail_msg;
    if (m->out_w != m->tok_embd) {
      if (upload_mat(&c->ctx, &c->out_w, m->out_w, "output.weight", err,
                     errlen) != 0)
        goto fail;
      c->has_out_w = 1;
    }
  }

  {
    size_t hd = m->head_dim, kv_row = m->n_kv_heads * hd;
    for (size_t l = 0; l < ngl; ++l) {
      const LlamaLayer* L = &m->layers[l];
      MetalLayer* D = &c->layers[l];
      if (upload_mat(&c->ctx, &D->q_w, L->attn_q, "attn_q", err, errlen) != 0 ||
          upload_mat(&c->ctx, &D->k_w, L->attn_k, "attn_k", err, errlen) != 0 ||
          upload_mat(&c->ctx, &D->v_w, L->attn_v, "attn_v", err, errlen) != 0 ||
          upload_mat(&c->ctx, &D->o_w, L->attn_out, "attn_output", err,
                     errlen) != 0 ||
          upload_mat(&c->ctx, &D->gate_w, L->ffn_gate, "ffn_gate", err,
                     errlen) != 0 ||
          upload_mat(&c->ctx, &D->up_w, L->ffn_up, "ffn_up", err, errlen) != 0 ||
          upload_mat(&c->ctx, &D->down_w, L->ffn_down, "ffn_down", err,
                     errlen) != 0)
        goto fail;
      if (upload_vec(&c->ctx, &D->attn_norm, L->attn_norm, m->hidden) != 0 ||
          upload_vec(&c->ctx, &D->ffn_norm, L->ffn_norm, m->hidden) != 0 ||
          upload_vec(&c->ctx, &D->q_norm, L->attn_q_norm, hd) != 0 ||
          upload_vec(&c->ctx, &D->k_norm, L->attn_k_norm, hd) != 0 ||
          upload_vec(&c->ctx, &D->bias_q, L->bias_q, m->n_head * hd) != 0 ||
          upload_vec(&c->ctx, &D->bias_k, L->bias_k, kv_row) != 0 ||
          upload_vec(&c->ctx, &D->bias_v, L->bias_v, kv_row) != 0 ||
          upload_vec(&c->ctx, &D->bias_o, L->bias_o, m->hidden) != 0)
        goto fail_msg;
      D->has_q_norm = L->attn_q_norm != NULL;
      D->has_k_norm = L->attn_k_norm != NULL;
      D->has_bias_q = L->bias_q != NULL;
      D->has_bias_k = L->bias_k != NULL;
      D->has_bias_v = L->bias_v != NULL;
      D->has_bias_o = L->bias_o != NULL;

      size_t kn = m->ctx * kv_row;
      if (mt_buf_alloc(&c->ctx, kn * 2, &D->k_cache) != 0 ||
          mt_buf_alloc(&c->ctx, kn * 2, &D->v_cache) != 0 ||
          mt_buf_zero(&c->ctx, &D->k_cache, kn * 2) != 0 ||
          mt_buf_zero(&c->ctx, &D->v_cache, kn * 2) != 0)
        goto fail_msg;
    }
  }

  fprintf(stderr,
          "metal: llama %zu/%zu layers on 1 GPU%s (f16 KV) [UNVERIFIED]\n", ngl,
          m->n_layers, partial ? "; remaining layers + head on CPU" : "");
  *out = c;
  return 0;

fail_msg:
  if (err && errlen && !err[0])
    snprintf(err, errlen, "metal: allocation/upload failed");
fail:
  llama_metal_free(c);
  return -1;
}

void llama_metal_free(LlamaMetal* c) {
  if (!c) return;
  for (size_t l = 0; c->layers && l < c->n_gpu_layers; ++l) {
    MetalLayer* D = &c->layers[l];
    mt_buf_free(&c->ctx, &D->q_w);
    mt_buf_free(&c->ctx, &D->k_w);
    mt_buf_free(&c->ctx, &D->v_w);
    mt_buf_free(&c->ctx, &D->o_w);
    mt_buf_free(&c->ctx, &D->gate_w);
    mt_buf_free(&c->ctx, &D->up_w);
    mt_buf_free(&c->ctx, &D->down_w);
    mt_buf_free(&c->ctx, &D->attn_norm);
    mt_buf_free(&c->ctx, &D->ffn_norm);
    mt_buf_free(&c->ctx, &D->q_norm);
    mt_buf_free(&c->ctx, &D->k_norm);
    mt_buf_free(&c->ctx, &D->bias_q);
    mt_buf_free(&c->ctx, &D->bias_k);
    mt_buf_free(&c->ctx, &D->bias_v);
    mt_buf_free(&c->ctx, &D->bias_o);
    mt_buf_free(&c->ctx, &D->k_cache);
    mt_buf_free(&c->ctx, &D->v_cache);
  }
  free(c->layers);
  mt_buf_free(&c->ctx, &c->x);
  mt_buf_free(&c->ctx, &c->normed);
  mt_buf_free(&c->ctx, &c->q);
  mt_buf_free(&c->ctx, &c->k);
  mt_buf_free(&c->ctx, &c->v);
  mt_buf_free(&c->ctx, &c->attn_res);
  mt_buf_free(&c->ctx, &c->attn_proj);
  mt_buf_free(&c->ctx, &c->gate);
  mt_buf_free(&c->ctx, &c->up);
  mt_buf_free(&c->ctx, &c->ffn_out);
  mt_buf_free(&c->ctx, &c->tok_embd);
  mt_buf_free(&c->ctx, &c->out_w);
  mt_buf_free(&c->ctx, &c->out_norm);
  mt_buf_free(&c->ctx, &c->logits);
  mt_ctx_free(&c->ctx);
  free(c);
}

static int llama_metal_forward(LlamaMetal* c, int32_t token, size_t pos,
                               float* logits_out, float* hidden_out) {
  const LlamaModel* m = c->m;
  const int h = (int)m->hidden;
  if (pos >= m->ctx) {
    fprintf(stderr, "metal: position %zu exceeds context %zu\n", pos, m->ctx);
    return -1;
  }

  MtRec rec;
  if (mt_rec_begin(&c->ctx, &rec) != 0) {
    fprintf(stderr, "metal: failed to begin command buffer\n");
    return -1;
  }

  size_t tk = (size_t)token < m->vocab ? (size_t)token : m->vocab - 1;
  size_t emb_row = oc_row_bytes(m->tok_embd->ggml_type, m->hidden);
  if (enc_embed(&rec, m->tok_embd->ggml_type, &c->tok_embd, tk * emb_row, h,
                &c->x) != 0)
    goto fail_enc;

  const int hd = (int)m->head_dim;
  const int n_head = (int)m->n_head, n_kv = (int)m->n_kv_heads;
  const int q_len = n_head * hd, kv_len = n_kv * hd;
  const int group = n_head / n_kv;
  const int cap = (int)m->ctx;
  const int rope_len = m->rope_dim ? (int)m->rope_dim : hd;
  const float scale = 1.0f / sqrtf((float)hd);

  for (size_t l = 0; l < c->n_gpu_layers; ++l) {
    const MetalLayer* D = &c->layers[l];

    if (enc_rmsnorm(&rec, &c->normed, &c->x, &D->attn_norm, h, m->eps, 1, 1) !=
            0 ||
        enc_matvec(&rec, m->layers[l].attn_q->ggml_type, &D->q_w, q_len, h,
                   &c->normed, &c->q) != 0 ||
        enc_matvec(&rec, m->layers[l].attn_k->ggml_type, &D->k_w, kv_len, h,
                   &c->normed, &c->k) != 0 ||
        enc_matvec(&rec, m->layers[l].attn_v->ggml_type, &D->v_w, kv_len, h,
                   &c->normed, &c->v) != 0)
      goto fail_enc;
    if (D->has_bias_q && enc_add(&rec, &c->q, &D->bias_q, q_len) != 0)
      goto fail_enc;
    if (D->has_bias_k && enc_add(&rec, &c->k, &D->bias_k, kv_len) != 0)
      goto fail_enc;
    if (D->has_bias_v && enc_add(&rec, &c->v, &D->bias_v, kv_len) != 0)
      goto fail_enc;

    if (D->has_q_norm &&
        enc_rmsnorm(&rec, &c->q, &c->q, &D->q_norm, hd, m->eps, n_head, 1) != 0)
      goto fail_enc;
    if (D->has_k_norm &&
        enc_rmsnorm(&rec, &c->k, &c->k, &D->k_norm, hd, m->eps, n_kv, 1) != 0)
      goto fail_enc;
    if (pos > 0 && rope_len > 0) {
      if (enc_rope(&rec, m->rope_norm ? 1 : 0, &c->q, hd, (int)pos, m->rope_theta,
                   rope_len, n_head) != 0 ||
          enc_rope(&rec, m->rope_norm ? 1 : 0, &c->k, hd, (int)pos, m->rope_theta,
                   rope_len, n_kv) != 0)
        goto fail_enc;
    }

    if (enc_kv_store(&rec, &D->k_cache, &D->v_cache, &c->k, &c->v, kv_len,
                     kv_len, (uint32_t)pos) != 0)
      goto fail_enc;
    int seq = (int)pos + 1;
    if (enc_attn(&rec, &c->attn_res, &c->q, &D->k_cache, &D->v_cache, hd, hd,
                 group, cap, 0, seq, scale, n_head) != 0)
      goto fail_enc;

    if (enc_matvec(&rec, m->layers[l].attn_out->ggml_type, &D->o_w, h, q_len,
                   &c->attn_res, &c->attn_proj) != 0)
      goto fail_enc;
    if (D->has_bias_o && enc_add(&rec, &c->attn_proj, &D->bias_o, h) != 0)
      goto fail_enc;
    if (enc_add(&rec, &c->x, &c->attn_proj, h) != 0) goto fail_enc;

    if (enc_rmsnorm(&rec, &c->normed, &c->x, &D->ffn_norm, h, m->eps, 1, 1) !=
            0 ||
        enc_matvec(&rec, m->layers[l].ffn_gate->ggml_type, &D->gate_w,
                   (int)m->inter, h, &c->normed, &c->gate) != 0 ||
        enc_matvec(&rec, m->layers[l].ffn_up->ggml_type, &D->up_w,
                   (int)m->inter, h, &c->normed, &c->up) != 0 ||
        enc_silu_mul(&rec, &c->gate, &c->up, (int)m->inter) != 0 ||
        enc_matvec(&rec, m->layers[l].ffn_down->ggml_type, &D->down_w, h,
                   (int)m->inter, &c->gate, &c->ffn_out) != 0 ||
        enc_add(&rec, &c->x, &c->ffn_out, h) != 0)
      goto fail_enc;
  }

  if (hidden_out) {
    if (mt_rec_commit_wait(&rec) != 0) return -1;
    if (mt_buf_download(&c->ctx, &c->x, hidden_out, (size_t)h * 4) != 0)
      return -1;
    return 0;
  }

  if (!logits_out) {
    if (mt_rec_commit(&rec) != 0) return -1;
    return 0;
  }

  const MtBuf* head_w = c->has_out_w ? &c->out_w : &c->tok_embd;
  if (enc_rmsnorm(&rec, &c->normed, &c->x, &c->out_norm, h, m->eps, 1, 1) != 0 ||
      enc_matvec(&rec, m->out_w->ggml_type, head_w, (int)m->vocab, h, &c->normed,
                 &c->logits) != 0)
    goto fail_enc;

  if (mt_rec_commit_wait(&rec) != 0) return -1;
  if (mt_buf_download(&c->ctx, &c->logits, logits_out, m->vocab * 4) != 0)
    return -1;
  return 0;

fail_enc:
  fprintf(stderr, "metal: encode failed\n");
  @autoreleasepool {
    if (rec.enc) {
      CFRelease(rec.enc);
      rec.enc = NULL;
    }
    if (rec.cmd) {
      CFRelease(rec.cmd);
      rec.cmd = NULL;
    }
  }
  return -1;
}

float* llama_metal_step(LlamaMetal* c, LlamaModel* m, int32_t token, size_t pos,
                        bool need_logits, int* failed) {
  *failed = 0;
  if (c->n_gpu_layers == m->n_layers) {
    float* lg = need_logits ? m->logits : NULL;
    if (llama_metal_forward(c, token, pos, lg, NULL) != 0) {
      *failed = 1;
      return NULL;
    }
    m->kv_len = pos + 1;
    return lg;
  }
  if (llama_metal_forward(c, token, pos, NULL, m->x) != 0) {
    *failed = 1;
    return NULL;
  }
  return llama_forward_from(m, pos, c->n_gpu_layers, need_logits);
}
