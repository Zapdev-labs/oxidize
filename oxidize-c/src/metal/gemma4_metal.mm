/* ============================================================================
 * UNVERIFIED — this file has NEVER been compiled or run. It was written BLIND
 * against the verified CUDA reference (src/cuda/gemma4_cuda.cu) and the Rust
 * Metal backend (oxidize-core/src/backends/metal.rs). It requires macOS + Xcode
 * Metal and an Apple GPU to compile and validate. It MAY NOT COMPILE. No
 * equivalence gate has ever been run against it. Treat every logit it would
 * produce as unproven.
 * ============================================================================
 *
 * GPU-resident Gemma 4 decode host orchestration. Kernels live in gemma4.metal
 * (gk_*); this file mirrors the host side of gemma4_cuda.cu: upload still-
 * quantized weights, encode the layer graph, one commit+wait per token when
 * results are requested.
 *
 * The graph mirrors model_gemma4.c / gemma4_cuda.cu EXACTLY on the f16-KV path:
 * V is the raw K projection copied BEFORE attn_k_norm/rope; V gets a scale-less
 * RMSNorm; global (non-SWA) layers divide the rope angle by rope_freqs; residual
 * is `x = (ffn + (post_attn_norm(proj) + x)) * output_scale`.
 *
 * REFUSED at init: --gpus > 1, rotoquant int4 KV (m->kv_quant), ngl==0,
 * unsupported weight quants.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CoreFoundation/CoreFoundation.h>

#include "gemma4_metal.h"
#include "metal_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "../gguf.h"
#include "../quant.h"
}

#define ARGMAX_BLOCKS 256

typedef struct {
  MtBuf q_w, k_w, v_w, o_w, gate_w, up_w, down_w;
  MtBuf attn_norm, q_norm, k_norm, post_attn_norm, ffn_norm, post_ffn_norm;
  MtBuf k_cache, v_cache; /* f16 */
  int has_v_w;
  int has_q_norm, has_k_norm, has_post_attn, has_post_ffn;
} MetalLayer;

struct Gemma4Metal {
  const Gemma4Model* m;
  size_t n_gpu_layers;
  MtCtx ctx;
  MetalLayer* layers;
  /* scratch */
  MtBuf x, normed, q, k, v, attn_res, attn_proj, gate, up, ffn_out;
  MtBuf tok_embd;
  MtBuf rope_freqs;
  int has_rope_freqs;
  /* head (only when GPU owns the whole stack) */
  MtBuf out_norm, logits, red_max, red_idx, argmax_d;
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

/* ---- encode helpers (one command buffer) ---- */

static int enc_matvec(MtRec* r, uint32_t type, const MtBuf* W, int rows,
                      int cols, const MtBuf* x, MtBuf* y) {
  const char* name = mt_matvec_name("gk", type);
  if (!name) return -1;
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, name, &p, err, sizeof(err)) != 0) {
    fprintf(stderr, "%s\n", err);
    return -1;
  }
  int rows_i = rows, cols_i = cols;
  if (type == OC_AL5_XS) {
    MtBind b[] = {MT_BUF(y), MT_BUF(W), MT_BYTES(&rows_i, sizeof(rows_i)),
                  MT_BYTES(&cols_i, sizeof(cols_i)), MT_BUF(x)};
    mt_dispatch(r, &p, b, 5, 0, mt_matvec_grid(rows), mt_matvec_tgs());
  } else {
    uint32_t rowbytes = (uint32_t)oc_row_bytes(type, (size_t)cols);
    MtBind b[] = {MT_BUF(y),
                  MT_BUF(W),
                  MT_BYTES(&rows_i, sizeof(rows_i)),
                  MT_BYTES(&cols_i, sizeof(cols_i)),
                  MT_BUF(x),
                  MT_BYTES(&rowbytes, sizeof(rowbytes))};
    mt_dispatch(r, &p, b, 6, 0, mt_matvec_grid(rows), mt_matvec_tgs());
  }
  return 0;
}

static int enc_embed(MtRec* r, uint32_t type, const MtBuf* table, size_t row_off,
                     int n, float scale, MtBuf* x) {
  const char* name = mt_embed_name("gk", type);
  if (!name) return -1;
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, name, &p, err, sizeof(err)) != 0) {
    fprintf(stderr, "%s\n", err);
    return -1;
  }
  int n_i = n;
  MtBind b[] = {MT_BUF(x), MT_BUF_OFF(table, row_off),
                MT_BYTES(&n_i, sizeof(n_i)), MT_BYTES(&scale, sizeof(scale))};
  mt_dispatch(r, &p, b, 4, 0, mt_grid_1d((size_t)n, 256), mt_tgs_1d(256));
  return 0;
}

static int enc_rmsnorm(MtRec* r, MtBuf* out, const MtBuf* x, const MtBuf* w,
                       int per, float eps, int nvec, int has_w) {
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, "gk_rmsnorm", &p, err, sizeof(err)) != 0) {
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

static int enc_rope(MtRec* r, MtBuf* vec, int head_dim, int pos, float theta,
                    int rope_len, const MtBuf* freqs, int has_freqs, int nheads) {
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, "gk_rope", &p, err, sizeof(err)) != 0) {
    fprintf(stderr, "%s\n", err);
    return -1;
  }
  int hd = head_dim, ps = pos, rl = rope_len, hf = has_freqs;
  MtBind b[] = {MT_BUF(vec),
                MT_BYTES(&hd, sizeof(hd)),
                MT_BYTES(&ps, sizeof(ps)),
                MT_BYTES(&theta, sizeof(theta)),
                MT_BYTES(&rl, sizeof(rl)),
                MT_BUF(freqs),
                MT_BYTES(&hf, sizeof(hf))};
  MtSize grid = {(size_t)nheads, 1, 1};
  mt_dispatch(r, &p, b, 7, 0, grid, mt_tgs_1d(128));
  return 0;
}

static int enc_kv_store(MtRec* r, const MtBuf* kc, const MtBuf* vc,
                        const MtBuf* k, const MtBuf* v, int k_len, int v_len,
                        uint32_t slot) {
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, "gk_kv_store", &p, err, sizeof(err)) != 0) {
    fprintf(stderr, "%s\n", err);
    return -1;
  }
  int kl = k_len, vl = v_len;
  int kv_max = k_len > v_len ? k_len : v_len;
  MtBind b[] = {MT_BUF(kc), MT_BUF(vc), MT_BUF(k), MT_BUF(v),
                MT_BYTES(&kl, sizeof(kl)), MT_BYTES(&vl, sizeof(vl)),
                MT_BYTES(&slot, sizeof(slot))};
  mt_dispatch(r, &p, b, 7, 0, mt_grid_1d((size_t)kv_max, 256), mt_tgs_1d(256));
  return 0;
}

static int enc_attn(MtRec* r, MtBuf* out, const MtBuf* q, const MtBuf* kc,
                    const MtBuf* vc, int hd, int vd, int group, int cache_cap,
                    int t0, int t1, float scale, int n_head) {
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, "gk_attn", &p, err, sizeof(err)) != 0) {
    fprintf(stderr, "%s\n", err);
    return -1;
  }
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

static int enc_geglu(MtRec* r, MtBuf* gate, const MtBuf* up, int n) {
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, "gk_geglu", &p, err, sizeof(err)) != 0) return -1;
  MtBind b[] = {MT_BUF(gate), MT_BUF(up), MT_BYTES(&n, sizeof(n))};
  mt_dispatch(r, &p, b, 3, 0, mt_grid_1d((size_t)n, 256), mt_tgs_1d(256));
  return 0;
}

static int enc_add(MtRec* r, MtBuf* c, const MtBuf* x, int n) {
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, "gk_add", &p, err, sizeof(err)) != 0) return -1;
  MtBind b[] = {MT_BUF(c), MT_BUF(x), MT_BYTES(&n, sizeof(n))};
  mt_dispatch(r, &p, b, 3, 0, mt_grid_1d((size_t)n, 256), mt_tgs_1d(256));
  return 0;
}

static int enc_resid_out(MtRec* r, MtBuf* x, const MtBuf* ffn, const MtBuf* attn,
                         float s, int n) {
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, "gk_resid_out", &p, err, sizeof(err)) != 0) return -1;
  MtBind b[] = {MT_BUF(x), MT_BUF(ffn), MT_BUF(attn), MT_BYTES(&s, sizeof(s)),
                MT_BYTES(&n, sizeof(n))};
  mt_dispatch(r, &p, b, 5, 0, mt_grid_1d((size_t)n, 256), mt_tgs_1d(256));
  return 0;
}

static int enc_softcap(MtRec* r, MtBuf* l, float cap, int n) {
  MtPipe p;
  char err[256];
  if (mt_pipe_get(r->ctx, "gk_softcap", &p, err, sizeof(err)) != 0) return -1;
  MtBind b[] = {MT_BUF(l), MT_BYTES(&cap, sizeof(cap)), MT_BYTES(&n, sizeof(n))};
  mt_dispatch(r, &p, b, 3, 0, mt_grid_1d((size_t)n, 256), mt_tgs_1d(256));
  return 0;
}

static int enc_argmax(MtRec* r, const MtBuf* logits, int n, MtBuf* red_max,
                      MtBuf* red_idx, MtBuf* argmax_d) {
  MtPipe p1, p2;
  char err[256];
  if (mt_pipe_get(r->ctx, "gk_argmax_stage1", &p1, err, sizeof(err)) != 0)
    return -1;
  if (mt_pipe_get(r->ctx, "gk_argmax_stage2", &p2, err, sizeof(err)) != 0)
    return -1;
  MtBind b1[] = {MT_BUF(logits), MT_BYTES(&n, sizeof(n)), MT_BUF(red_max),
                 MT_BUF(red_idx)};
  MtSize g1 = {ARGMAX_BLOCKS, 1, 1};
  mt_dispatch(r, &p1, b1, 4, 0, g1, mt_tgs_1d(256));
  int nb = ARGMAX_BLOCKS;
  MtBind b2[] = {MT_BUF(red_max), MT_BUF(red_idx), MT_BYTES(&nb, sizeof(nb)),
                 MT_BUF(argmax_d)};
  MtSize g2 = {1, 1, 1};
  mt_dispatch(r, &p2, b2, 4, 0, g2, mt_tgs_1d(256));
  return 0;
}

/* Copy device->device via a tiny host round-trip on shared memory (Apple
 * unified). Used for K=V layers: V = raw K before norm/rope. */
static int enc_copy_f32(MtRec* r, const MtBuf* dst, const MtBuf* src, int n) {
  /* No dedicated copy kernel; reuse add onto a zeroed dst would need a clear.
   * Shared-memory path: record nothing — caller does host memcpy between
   * commits. For in-encoder ordering we dispatch a 1:1 "add to zero" by first
   * zeroing via a softcap-free path is messy. Simplest correct approach on
   * unified memory: the host memcpy AFTER a wait would break the no-mid-sync
   * contract. Use gk_add onto a buffer we zero with setBytes... unavailable.
   *
   * Instead: bind src as both sides of a dedicated pattern — upload is shared,
   * so we memcpy host-side contents of src into dst NOW (CPU) before later
   * kernels that write src. That races with GPU. BAD.
   *
   * Correct: a trivial "copy" using resid_out with s=1 and attn=zeros, or
   * encode add after zeroing dst. Zero dst via mt_buf_zero only safe between
   * tokens. During encode, use gk_resid_out(x=dst, ffn=src, attn=dummy_zero, s=1)
   * where dummy is the ctx zero dummy — but dummy is 64 bytes.
   *
   * Allocate a persistent zeros buffer at init sized to max_k. Done below as
   * copy via matvec identity is overkill. We'll use a host-side note: on
   * unified memory, blit via MTLBlitCommandEncoder. */
  @autoreleasepool {
    id<MTLComputeCommandEncoder> enc =
        (__bridge id<MTLComputeCommandEncoder>)r->enc;
    (void)enc;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)r->cmd;
    /* End compute, blit, resume compute — Metal allows multiple encoders. */
    [enc endEncoding];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:(__bridge id<MTLBuffer>)src->buf
            sourceOffset:0
                toBuffer:(__bridge id<MTLBuffer>)dst->buf
       destinationOffset:0
                    size:(NSUInteger)n * 4];
    [blit endEncoding];
    id<MTLComputeCommandEncoder> enc2 = [cmd computeCommandEncoder];
    CFRelease(r->enc);
    r->enc = (__bridge_retained void*)enc2;
  }
  return 0;
}

int gemma4_metal_init(Gemma4Metal** out, const Gemma4Model* m, int n_gpus,
                      int n_gpu_layers, char* err, size_t errlen) {
  *out = NULL;
  if (n_gpus > 1) {
    if (err && errlen)
      snprintf(err, errlen,
               "metal: gemma4 backend is single-GPU (--gpus 1); Apple Silicon "
               "has no layer-split multi-GPU path");
    return -1;
  }
  if (m->kv_quant) {
    if (err && errlen)
      snprintf(err, errlen,
               "metal: rotoquant int4 KV is not ported (no gk_fht / gk_attn_q4). "
               "Load with kv_quant=false for f16 KV");
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

  Gemma4Metal* c = (Gemma4Metal*)calloc(1, sizeof(Gemma4Metal));
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

  if (mt_buf_alloc(&c->ctx, m->hidden * 4, &c->x) != 0 ||
      mt_buf_alloc(&c->ctx, m->hidden * 4, &c->normed) != 0 ||
      mt_buf_alloc(&c->ctx, max_q * 4, &c->q) != 0 ||
      mt_buf_alloc(&c->ctx, max_k * 4, &c->k) != 0 ||
      mt_buf_alloc(&c->ctx, max_k * 4, &c->v) != 0 ||
      mt_buf_alloc(&c->ctx, max_q * 4, &c->attn_res) != 0 ||
      mt_buf_alloc(&c->ctx, m->hidden * 4, &c->attn_proj) != 0 ||
      mt_buf_alloc(&c->ctx, m->inter * 4, &c->gate) != 0 ||
      mt_buf_alloc(&c->ctx, m->inter * 4, &c->up) != 0 ||
      mt_buf_alloc(&c->ctx, m->hidden * 4, &c->ffn_out) != 0)
    goto fail_msg;

  if (upload_mat(&c->ctx, &c->tok_embd, m->tok_embd, "token_embd", err,
                 errlen) != 0)
    goto fail;
  if (m->rope_freqs && max_hd >= 2) {
    if (upload_vec(&c->ctx, &c->rope_freqs, m->rope_freqs, max_hd / 2) != 0)
      goto fail_msg;
    c->has_rope_freqs = 1;
  }

  if (!partial) {
    if (upload_vec(&c->ctx, &c->out_norm, m->out_norm, m->hidden) != 0 ||
        mt_buf_alloc(&c->ctx, m->vocab * 4, &c->logits) != 0 ||
        mt_buf_alloc(&c->ctx, ARGMAX_BLOCKS * 4, &c->red_max) != 0 ||
        mt_buf_alloc(&c->ctx, ARGMAX_BLOCKS * 4, &c->red_idx) != 0 ||
        mt_buf_alloc(&c->ctx, 4, &c->argmax_d) != 0)
      goto fail_msg;
  }

  for (size_t l = 0; l < ngl; ++l) {
    const Gemma4Layer* L = &m->layers[l];
    MetalLayer* D = &c->layers[l];
    if (upload_mat(&c->ctx, &D->q_w, L->attn_q, "attn_q", err, errlen) != 0 ||
        upload_mat(&c->ctx, &D->k_w, L->attn_k, "attn_k", err, errlen) != 0)
      goto fail;
    if (L->attn_v) {
      if (upload_mat(&c->ctx, &D->v_w, L->attn_v, "attn_v", err, errlen) != 0)
        goto fail;
      D->has_v_w = 1;
    }
    if (upload_mat(&c->ctx, &D->o_w, L->attn_out, "attn_output", err, errlen) !=
            0 ||
        upload_mat(&c->ctx, &D->gate_w, L->ffn_gate, "ffn_gate", err, errlen) !=
            0 ||
        upload_mat(&c->ctx, &D->up_w, L->ffn_up, "ffn_up", err, errlen) != 0 ||
        upload_mat(&c->ctx, &D->down_w, L->ffn_down, "ffn_down", err, errlen) !=
            0)
      goto fail;
    if (upload_vec(&c->ctx, &D->attn_norm, L->attn_norm, m->hidden) != 0 ||
        upload_vec(&c->ctx, &D->q_norm, L->attn_q_norm, L->head_dim) != 0 ||
        upload_vec(&c->ctx, &D->k_norm, L->attn_k_norm, L->head_dim) != 0 ||
        upload_vec(&c->ctx, &D->post_attn_norm, L->post_attn_norm, m->hidden) !=
            0 ||
        upload_vec(&c->ctx, &D->ffn_norm, L->ffn_norm, m->hidden) != 0 ||
        upload_vec(&c->ctx, &D->post_ffn_norm, L->post_ffn_norm, m->hidden) != 0)
      goto fail_msg;
    D->has_q_norm = L->attn_q_norm != NULL;
    D->has_k_norm = L->attn_k_norm != NULL;
    D->has_post_attn = L->post_attn_norm != NULL;
    D->has_post_ffn = L->post_ffn_norm != NULL;

    size_t kn = L->cache_cap * L->n_kv_heads * L->head_dim;
    size_t vn = L->cache_cap * L->n_kv_heads * L->v_head_dim;
    if (mt_buf_alloc(&c->ctx, kn * 2, &D->k_cache) != 0 ||
        mt_buf_alloc(&c->ctx, vn * 2, &D->v_cache) != 0 ||
        mt_buf_zero(&c->ctx, &D->k_cache, kn * 2) != 0 ||
        mt_buf_zero(&c->ctx, &D->v_cache, vn * 2) != 0)
      goto fail_msg;
  }

  fprintf(stderr, "metal: gemma4 %zu/%zu layers on 1 GPU%s (f16 KV) [UNVERIFIED]\n",
          ngl, m->n_layers, partial ? "; remaining layers + head on CPU" : "");
  *out = c;
  return 0;

fail_msg:
  if (err && errlen && !err[0])
    snprintf(err, errlen, "metal: allocation/upload failed");
fail:
  gemma4_metal_free(c);
  return -1;
}

void gemma4_metal_free(Gemma4Metal* c) {
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
    mt_buf_free(&c->ctx, &D->q_norm);
    mt_buf_free(&c->ctx, &D->k_norm);
    mt_buf_free(&c->ctx, &D->post_attn_norm);
    mt_buf_free(&c->ctx, &D->ffn_norm);
    mt_buf_free(&c->ctx, &D->post_ffn_norm);
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
  mt_buf_free(&c->ctx, &c->rope_freqs);
  mt_buf_free(&c->ctx, &c->out_norm);
  mt_buf_free(&c->ctx, &c->logits);
  mt_buf_free(&c->ctx, &c->red_max);
  mt_buf_free(&c->ctx, &c->red_idx);
  mt_buf_free(&c->ctx, &c->argmax_d);
  mt_ctx_free(&c->ctx);
  free(c);
}

int gemma4_metal_forward(Gemma4Metal* c, int32_t token, size_t pos,
                         float* logits_out, int32_t* argmax_out,
                         float* hidden_out) {
  const Gemma4Model* m = c->m;
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
                m->emb_scale, &c->x) != 0)
    goto fail_enc;

  for (size_t l = 0; l < c->n_gpu_layers; ++l) {
    const Gemma4Layer* L = &m->layers[l];
    const MetalLayer* D = &c->layers[l];
    int hd = (int)L->head_dim, vd = (int)L->v_head_dim;
    int n_head = (int)m->n_head, n_kv = (int)L->n_kv_heads;
    int q_len = n_head * hd, k_len = n_kv * hd, v_len = n_kv * vd;
    int group = n_head / n_kv;

    if (enc_rmsnorm(&rec, &c->normed, &c->x, &D->attn_norm, h, m->eps, 1, 1) !=
        0)
      goto fail_enc;
    if (enc_matvec(&rec, L->attn_q->ggml_type, &D->q_w, q_len, h, &c->normed,
                   &c->q) != 0 ||
        enc_matvec(&rec, L->attn_k->ggml_type, &D->k_w, k_len, h, &c->normed,
                   &c->k) != 0)
      goto fail_enc;
    if (D->has_v_w) {
      if (enc_matvec(&rec, L->attn_v->ggml_type, &D->v_w, v_len, h, &c->normed,
                     &c->v) != 0)
        goto fail_enc;
    } else {
      /* K=V: V is the RAW K projection — before k_norm/rope */
      if (enc_copy_f32(&rec, &c->v, &c->k, k_len) != 0) goto fail_enc;
    }

    if (D->has_q_norm &&
        enc_rmsnorm(&rec, &c->q, &c->q, &D->q_norm, hd, m->eps, n_head, 1) != 0)
      goto fail_enc;
    if (D->has_k_norm &&
        enc_rmsnorm(&rec, &c->k, &c->k, &D->k_norm, hd, m->eps, n_kv, 1) != 0)
      goto fail_enc;

    int rope_len = L->rope.rope_dim ? (int)L->rope.rope_dim : hd;
    int use_freqs = (!L->is_swa && c->has_rope_freqs) ? 1 : 0;
    if (pos > 0 && rope_len > 0) {
      if (enc_rope(&rec, &c->q, hd, (int)pos, L->rope.theta, rope_len,
                   &c->rope_freqs, use_freqs, n_head) != 0 ||
          enc_rope(&rec, &c->k, hd, (int)pos, L->rope.theta, rope_len,
                   &c->rope_freqs, use_freqs, n_kv) != 0)
        goto fail_enc;
    }
    /* scale-less V RMSNorm */
    if (enc_rmsnorm(&rec, &c->v, &c->v, NULL, vd, m->eps, n_kv, 0) != 0)
      goto fail_enc;

    uint32_t slot = (uint32_t)(pos % L->cache_cap);
    size_t seq = pos + 1;
    size_t t0 = seq > L->cache_cap ? seq - L->cache_cap : 0;
    float scale =
        m->attn_scale > 0.0f ? m->attn_scale : 1.0f / sqrtf((float)hd);
    if (enc_kv_store(&rec, &D->k_cache, &D->v_cache, &c->k, &c->v, k_len, v_len,
                     slot) != 0 ||
        enc_attn(&rec, &c->attn_res, &c->q, &D->k_cache, &D->v_cache, hd, vd,
                 group, (int)L->cache_cap, (int)t0, (int)seq, scale, n_head) !=
            0)
      goto fail_enc;

    if (enc_matvec(&rec, L->attn_out->ggml_type, &D->o_w, h, n_head * vd,
                   &c->attn_res, &c->attn_proj) != 0)
      goto fail_enc;
    if (D->has_post_attn &&
        enc_rmsnorm(&rec, &c->attn_proj, &c->attn_proj, &D->post_attn_norm, h,
                    m->eps, 1, 1) != 0)
      goto fail_enc;
    if (enc_add(&rec, &c->attn_proj, &c->x, h) != 0) goto fail_enc;

    if (enc_rmsnorm(&rec, &c->normed, &c->attn_proj, &D->ffn_norm, h, m->eps, 1,
                    1) != 0 ||
        enc_matvec(&rec, L->ffn_gate->ggml_type, &D->gate_w, (int)m->inter, h,
                   &c->normed, &c->gate) != 0 ||
        enc_matvec(&rec, L->ffn_up->ggml_type, &D->up_w, (int)m->inter, h,
                   &c->normed, &c->up) != 0 ||
        enc_geglu(&rec, &c->gate, &c->up, (int)m->inter) != 0 ||
        enc_matvec(&rec, L->ffn_down->ggml_type, &D->down_w, h, (int)m->inter,
                   &c->gate, &c->ffn_out) != 0)
      goto fail_enc;
    if (D->has_post_ffn &&
        enc_rmsnorm(&rec, &c->ffn_out, &c->ffn_out, &D->post_ffn_norm, h, m->eps,
                    1, 1) != 0)
      goto fail_enc;
    if (enc_resid_out(&rec, &c->x, &c->ffn_out, &c->attn_proj, L->output_scale,
                      h) != 0)
      goto fail_enc;
  }

  if (hidden_out) {
    if (mt_rec_commit_wait(&rec) != 0) return -1;
    if (mt_buf_download(&c->ctx, &c->x, hidden_out, (size_t)h * 4) != 0)
      return -1;
    return 0;
  }

  if (!logits_out && !argmax_out) {
    if (mt_rec_commit(&rec) != 0) return -1;
    return 0;
  }

  /* final norm + tied logits */
  if (enc_rmsnorm(&rec, &c->normed, &c->x, &c->out_norm, h, m->eps, 1, 1) != 0 ||
      enc_matvec(&rec, m->tok_embd->ggml_type, &c->tok_embd, (int)m->vocab, h,
                 &c->normed, &c->logits) != 0)
    goto fail_enc;

  if (argmax_out) {
    if (enc_argmax(&rec, &c->logits, (int)m->vocab, &c->red_max, &c->red_idx,
                   &c->argmax_d) != 0)
      goto fail_enc;
  }
  if (logits_out && m->final_softcap > 0.0f) {
    if (enc_softcap(&rec, &c->logits, m->final_softcap, (int)m->vocab) != 0)
      goto fail_enc;
  }

  if (mt_rec_commit_wait(&rec) != 0) return -1;
  if (argmax_out) {
    if (mt_buf_download(&c->ctx, &c->argmax_d, argmax_out, 4) != 0) return -1;
  }
  if (logits_out) {
    if (mt_buf_download(&c->ctx, &c->logits, logits_out, m->vocab * 4) != 0)
      return -1;
  }
  return 0;

fail_enc:
  fprintf(stderr, "metal: encode failed\n");
  /* drop the half-built command buffer */
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

float* gemma4_metal_step(Gemma4Metal* c, Gemma4Model* m, int32_t token,
                         size_t pos, bool need_logits, int* failed) {
  *failed = 0;
  if (c->n_gpu_layers == m->n_layers) {
    float* lg = need_logits ? m->logits : NULL;
    if (gemma4_metal_forward(c, token, pos, lg, NULL, NULL) != 0) {
      *failed = 1;
      return NULL;
    }
    m->kv_len = pos + 1;
    return lg;
  }
  if (gemma4_metal_forward(c, token, pos, NULL, NULL, m->x) != 0) {
    *failed = 1;
    return NULL;
  }
  return gemma4_forward_from(m, pos, c->n_gpu_layers, need_logits);
}
