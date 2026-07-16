/* ======================================================================
 * UNVERIFIED — this file has NEVER been compiled or run.
 * Written BLIND against src/cuda/gemma4_cuda.cu + src/vulkan/shaders/*.
 * Requires Vulkan 1.1+, libvulkan, SPIR-V from src/vulkan/Makefile.
 * IT MAY NOT COMPILE and MAY BE WRONG. No verification was performed.
 * ====================================================================== */
#include "gemma4_vk.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../gguf.h"
#include "../quant.h"
#include "vk_common.h"

typedef struct {
  VkBuf q_w, k_w, v_w, o_w, gate_w, up_w, down_w;
  int has_v;
  VkBuf attn_norm, q_norm, k_norm, post_attn_norm, ffn_norm, post_ffn_norm;
  int has_q_norm, has_k_norm, has_post_attn, has_post_ffn;
  VkBuf k_cache, v_cache;
} VkG4Layer;

struct Gemma4Vk {
  const Gemma4Model* m;
  size_t n_gpu_layers;
  int owns_head; /* whole stack on GPU */
  VkCtx ctx;
  VkKernels ker;
  VkG4Layer* layers;
  VkBuf x, normed, q, k, v, attn_res, attn_proj, gate, up, ffn_out;
  VkBuf tok_embd, rope_freqs, dummy, attn_scratch;
  int has_rope_freqs;
  VkBuf out_norm, logits, argmax_d;
};

static size_t tensor_bytes(const GgufTensorInfo* t) {
  size_t cols = (size_t)t->dims[0], rows = 1;
  for (uint32_t d = 1; d < t->n_dims; ++d) rows *= (size_t)t->dims[d];
  return rows * oc_row_bytes(t->ggml_type, cols);
}

static int check_type(uint32_t t, const char* what, char* err, size_t errlen) {
  if (vk_qidx(t) >= 0) return 0;
  if (err && errlen)
    snprintf(err, errlen,
             "vulkan: %s has quant type %u; GPU kernels exist for "
             "F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/AL5_XS only",
             what, t);
  return -1;
}

static int up_bytes(VkCtx* c, VkBuf* b, const void* src, size_t n) {
  if (vk_buf_device(c, n, b) != 0) return -1;
  return vk_upload(c, b, src, n);
}

static int up_mat(VkCtx* c, const GgufTensorInfo* t, VkBuf* b, const char* what,
                  char* err, size_t errlen) {
  if (check_type(t->ggml_type, what, err, errlen) != 0) return -1;
  return up_bytes(c, b, t->data, tensor_bytes(t));
}

static int up_vec(VkCtx* c, const float* src, size_t n, VkBuf* b) {
  if (!src) return 0;
  return up_bytes(c, b, src, n * sizeof(float));
}

static const char* shader_dir(void) {
  const char* e = getenv("OXIDIZE_VK_SHADERS");
  return e && e[0] ? e : "src/vulkan/shaders";
}

/* ---- push-constant packs (must match .comp layouts) ---- */
typedef struct {
  int rows, cols;
  uint32_t rowbytes;
} PCMatvec;
typedef struct {
  int n;
  uint32_t row_off;
  float scale;
} PCEmbed;
typedef struct {
  int per;
  float eps;
  int has_w;
} PCRms;
typedef struct {
  int head_dim, pos, rope_len;
  float theta;
  int has_freqs;
} PCRope;
typedef struct {
  int k_len, v_len;
  uint32_t slot;
} PCKv;
typedef struct {
  int hd, vd, group, cache_cap, t0, t1;
  float scale;
} PCAttn;
typedef struct {
  int n;
} PCInt;
typedef struct {
  int n;
  float s;
} PCResid;
typedef struct {
  int n;
  float c;
} PCSoft;

static void d_matvec(VkRec* r, const VkKernels* ker, uint32_t type,
                     const VkBuf* W, const VkBuf* x, const VkBuf* y, int rows,
                     int cols) {
  int qi = vk_qidx(type);
  if (qi < 0) return;
  PCMatvec pc = {rows, cols, (uint32_t)oc_row_bytes(type, (size_t)cols)};
  const VkBuf* bufs[3] = {W, x, y};
  vk_dispatch(r, &ker->matvec[qi], bufs, &pc, sizeof(pc), (uint32_t)rows, 1, 1);
}

static void d_embed(VkRec* r, const VkKernels* ker, uint32_t type,
                    const VkBuf* W, const VkBuf* x, int n, uint32_t row_off,
                    float scale) {
  int qi = vk_qidx(type);
  if (qi < 0) return;
  PCEmbed pc = {n, row_off, scale};
  const VkBuf* bufs[2] = {W, x};
  vk_dispatch(r, &ker->embed[qi], bufs, &pc, sizeof(pc),
              (uint32_t)((n + 255) / 256), 1, 1);
}

static void d_rms(VkRec* r, const VkKernels* ker, const VkBuf* out,
                  const VkBuf* x, const VkBuf* w, int has_w, int per, float eps,
                  uint32_t nvec) {
  PCRms pc = {per, eps, has_w};
  const VkBuf* bufs[3] = {out, x, w};
  vk_dispatch(r, &ker->rmsnorm, bufs, &pc, sizeof(pc), nvec, 1, 1);
}

static void d_rope(VkRec* r, const VkPipe* pipe, const VkBuf* vec,
                   const VkBuf* freqs, int hd, int pos, int rope_len,
                   float theta, int has_freqs, uint32_t nhead) {
  PCRope pc = {hd, pos, rope_len, theta, has_freqs};
  const VkBuf* bufs[2] = {vec, freqs};
  vk_dispatch(r, pipe, bufs, &pc, sizeof(pc), nhead, 1, 1);
}

static void d_add(VkRec* r, const VkKernels* ker, const VkBuf* c,
                  const VkBuf* x, int n) {
  PCInt pc = {n};
  const VkBuf* bufs[2] = {c, x};
  vk_dispatch(r, &ker->add, bufs, &pc, sizeof(pc), (uint32_t)((n + 255) / 256),
              1, 1);
}

int gemma4_vk_init(Gemma4Vk** out, const Gemma4Model* m, int n_gpus,
                   int n_gpu_layers, char* err, size_t errlen) {
  *out = NULL;
  if (n_gpus > 1) {
    if (err && errlen)
      snprintf(err, errlen,
               "vulkan: gemma4 backend is single-GPU only; multi-GPU "
               "layer-split is not ported");
    return -1;
  }
  if (m->kv_quant) {
    if (err && errlen)
      snprintf(err, errlen,
               "vulkan: rotoquant KV (kv_quant) is not ported; load with "
               "kv_quant=false (FP32 KV caches in shaders)");
    return -1;
  }
  size_t ngl = (n_gpu_layers < 0 || (size_t)n_gpu_layers > m->n_layers)
                   ? m->n_layers
                   : (size_t)n_gpu_layers;
  if (ngl == 0) {
    if (err && errlen)
      snprintf(err, errlen, "vulkan: --ngl 0 is the pure-CPU path (no GPU init)");
    return -1;
  }

  size_t max_q = m->hidden, max_k = 0, max_hd = 0, max_cap = 0;
  for (size_t l = 0; l < ngl; ++l) {
    const Gemma4Layer* L = &m->layers[l];
    if (L->head_dim > 256 || L->v_head_dim > 256) {
      if (err && errlen)
        snprintf(err, errlen,
                 "vulkan: head_dim/v_head_dim > 256 refused (attn.comp shared "
                 "q cache is fixed at 256)");
      return -1;
    }
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
  }

  Gemma4Vk* c = calloc(1, sizeof(*c));
  if (!c) return -1;
  c->m = m;
  c->n_gpu_layers = ngl;
  c->owns_head = (ngl == m->n_layers);
  c->layers = calloc(ngl, sizeof(VkG4Layer));
  if (!c->layers) {
    free(c);
    return -1;
  }

  if (vk_ctx_init(&c->ctx, err, errlen) != 0) goto fail;
  if (vk_kernels_init(&c->ctx, &c->ker, shader_dir(), err, errlen) != 0)
    goto fail;

  if (check_type(m->tok_embd->ggml_type, "token_embd", err, errlen) != 0)
    goto fail;

  VkCtx* vx = &c->ctx;
  if (vk_buf_device(vx, m->hidden * 4, &c->x) != 0 ||
      vk_buf_device(vx, m->hidden * 4, &c->normed) != 0 ||
      vk_buf_device(vx, max_q * 4, &c->q) != 0 ||
      vk_buf_device(vx, max_k * 4, &c->k) != 0 ||
      vk_buf_device(vx, max_k * 4, &c->v) != 0 ||
      vk_buf_device(vx, max_q * 4, &c->attn_res) != 0 ||
      vk_buf_device(vx, m->hidden * 4, &c->attn_proj) != 0 ||
      vk_buf_device(vx, m->inter * 4, &c->gate) != 0 ||
      vk_buf_device(vx, m->inter * 4, &c->up) != 0 ||
      vk_buf_device(vx, m->hidden * 4, &c->ffn_out) != 0 ||
      vk_buf_device(vx, m->n_head * max_cap * 4, &c->attn_scratch) != 0 ||
      up_bytes(vx, &c->dummy, (float[1]){1.0f}, sizeof(float)) != 0)
    goto fail_msg;

  if (up_bytes(vx, &c->tok_embd, m->tok_embd->data, tensor_bytes(m->tok_embd)) !=
      0)
    goto fail_msg;

  if (m->rope_freqs && max_hd >= 2) {
    if (up_bytes(vx, &c->rope_freqs, m->rope_freqs, (max_hd / 2) * 4) != 0)
      goto fail_msg;
    c->has_rope_freqs = 1;
  } else {
    c->rope_freqs = c->dummy; /* descriptor must be valid; has_freqs=0 */
  }

  if (c->owns_head) {
    if (up_vec(vx, m->out_norm, m->hidden, &c->out_norm) != 0 ||
        vk_buf_device(vx, m->vocab * 4, &c->logits) != 0 ||
        vk_buf_device(vx, 4, &c->argmax_d) != 0)
      goto fail_msg;
  }

  for (size_t l = 0; l < ngl; ++l) {
    const Gemma4Layer* L = &m->layers[l];
    VkG4Layer* D = &c->layers[l];
    if (up_mat(vx, L->attn_q, &D->q_w, "attn_q", err, errlen) != 0 ||
        up_mat(vx, L->attn_k, &D->k_w, "attn_k", err, errlen) != 0 ||
        up_mat(vx, L->attn_out, &D->o_w, "attn_output", err, errlen) != 0 ||
        up_mat(vx, L->ffn_gate, &D->gate_w, "ffn_gate", err, errlen) != 0 ||
        up_mat(vx, L->ffn_up, &D->up_w, "ffn_up", err, errlen) != 0 ||
        up_mat(vx, L->ffn_down, &D->down_w, "ffn_down", err, errlen) != 0)
      goto fail;
    if (L->attn_v) {
      if (up_mat(vx, L->attn_v, &D->v_w, "attn_v", err, errlen) != 0) goto fail;
      D->has_v = 1;
    }
    if (up_vec(vx, L->attn_norm, m->hidden, &D->attn_norm) != 0 ||
        up_vec(vx, L->attn_q_norm, L->head_dim, &D->q_norm) != 0 ||
        up_vec(vx, L->attn_k_norm, L->head_dim, &D->k_norm) != 0 ||
        up_vec(vx, L->post_attn_norm, m->hidden, &D->post_attn_norm) != 0 ||
        up_vec(vx, L->ffn_norm, m->hidden, &D->ffn_norm) != 0 ||
        up_vec(vx, L->post_ffn_norm, m->hidden, &D->post_ffn_norm) != 0)
      goto fail_msg;
    D->has_q_norm = L->attn_q_norm != NULL;
    D->has_k_norm = L->attn_k_norm != NULL;
    D->has_post_attn = L->post_attn_norm != NULL;
    D->has_post_ffn = L->post_ffn_norm != NULL;

    size_t kn = L->cache_cap * L->n_kv_heads * L->head_dim * 4;
    size_t vn = L->cache_cap * L->n_kv_heads * L->v_head_dim * 4;
    if (vk_buf_device(vx, kn, &D->k_cache) != 0 ||
        vk_buf_device(vx, vn, &D->v_cache) != 0 ||
        vk_zero(vx, &D->k_cache, kn) != 0 || vk_zero(vx, &D->v_cache, vn) != 0)
      goto fail_msg;
  }

  fprintf(stderr, "vulkan: gemma4 %zu/%zu layers on 1 GPU%s (FP32 KV)\n", ngl,
          m->n_layers, c->owns_head ? "" : "; remaining layers + head on CPU");
  *out = c;
  return 0;

fail_msg:
  if (err && errlen && !err[0])
    snprintf(err, errlen, "vulkan: allocation/upload failed");
fail:
  gemma4_vk_free(c);
  return -1;
}

void gemma4_vk_free(Gemma4Vk* c) {
  if (!c) return;
  VkCtx* vx = &c->ctx;
  for (size_t l = 0; c->layers && l < c->n_gpu_layers; ++l) {
    VkG4Layer* D = &c->layers[l];
    vk_buf_free(vx, &D->q_w);
    vk_buf_free(vx, &D->k_w);
    vk_buf_free(vx, &D->v_w);
    vk_buf_free(vx, &D->o_w);
    vk_buf_free(vx, &D->gate_w);
    vk_buf_free(vx, &D->up_w);
    vk_buf_free(vx, &D->down_w);
    vk_buf_free(vx, &D->attn_norm);
    vk_buf_free(vx, &D->q_norm);
    vk_buf_free(vx, &D->k_norm);
    vk_buf_free(vx, &D->post_attn_norm);
    vk_buf_free(vx, &D->ffn_norm);
    vk_buf_free(vx, &D->post_ffn_norm);
    vk_buf_free(vx, &D->k_cache);
    vk_buf_free(vx, &D->v_cache);
  }
  free(c->layers);
  vk_buf_free(vx, &c->x);
  vk_buf_free(vx, &c->normed);
  vk_buf_free(vx, &c->q);
  vk_buf_free(vx, &c->k);
  vk_buf_free(vx, &c->v);
  vk_buf_free(vx, &c->attn_res);
  vk_buf_free(vx, &c->attn_proj);
  vk_buf_free(vx, &c->gate);
  vk_buf_free(vx, &c->up);
  vk_buf_free(vx, &c->ffn_out);
  vk_buf_free(vx, &c->tok_embd);
  vk_buf_free(vx, &c->attn_scratch);
  vk_buf_free(vx, &c->dummy);
  if (c->has_rope_freqs) vk_buf_free(vx, &c->rope_freqs);
  vk_buf_free(vx, &c->out_norm);
  vk_buf_free(vx, &c->logits);
  vk_buf_free(vx, &c->argmax_d);
  vk_kernels_free(vx, &c->ker);
  vk_ctx_free(vx);
  free(c);
}

int gemma4_vk_forward(Gemma4Vk* c, int32_t token, size_t pos, float* logits_out,
                      int32_t* argmax_out, float* hidden_out) {
  const Gemma4Model* m = c->m;
  const int h = (int)m->hidden;
  if (pos >= m->ctx) {
    fprintf(stderr, "vulkan: position %zu exceeds context %zu\n", pos, m->ctx);
    return -1;
  }

  VkRec rec;
  if (vk_rec_begin(&c->ctx, &rec) != 0) {
    fprintf(stderr, "vulkan: rec_begin failed\n");
    return -1;
  }

  size_t tk = (size_t)token < m->vocab ? (size_t)token : m->vocab - 1;
  size_t emb_row = oc_row_bytes(m->tok_embd->ggml_type, m->hidden);
  d_embed(&rec, &c->ker, m->tok_embd->ggml_type, &c->tok_embd, &c->x, h,
          (uint32_t)(tk * emb_row), m->emb_scale);

  for (size_t l = 0; l < c->n_gpu_layers; ++l) {
    const Gemma4Layer* L = &m->layers[l];
    const VkG4Layer* D = &c->layers[l];
    int hd = (int)L->head_dim, vd = (int)L->v_head_dim;
    int n_head = (int)m->n_head, n_kv = (int)L->n_kv_heads;
    int q_len = n_head * hd, k_len = n_kv * hd, v_len = n_kv * vd;
    int group = n_head / n_kv;
    const VkBuf* dummy = &c->dummy;

    d_rms(&rec, &c->ker, &c->normed, &c->x,
          D->attn_norm.buf ? &D->attn_norm : dummy, D->attn_norm.buf != 0, h,
          m->eps, 1);
    d_matvec(&rec, &c->ker, L->attn_q->ggml_type, &D->q_w, &c->normed, &c->q,
             q_len, h);
    d_matvec(&rec, &c->ker, L->attn_k->ggml_type, &D->k_w, &c->normed, &c->k,
             k_len, h);
    if (D->has_v)
      d_matvec(&rec, &c->ker, L->attn_v->ggml_type, &D->v_w, &c->normed, &c->v,
               v_len, h);
    else
      vk_rec_copy(&rec, &c->v, &c->k, (VkDeviceSize)k_len * 4);

    if (D->has_q_norm)
      d_rms(&rec, &c->ker, &c->q, &c->q, &D->q_norm, 1, hd, m->eps,
            (uint32_t)n_head);
    if (D->has_k_norm)
      d_rms(&rec, &c->ker, &c->k, &c->k, &D->k_norm, 1, hd, m->eps,
            (uint32_t)n_kv);

    int rope_len = L->rope.rope_dim ? (int)L->rope.rope_dim : hd;
    int has_freqs = (!L->is_swa && c->has_rope_freqs) ? 1 : 0;
    if (pos > 0 && rope_len > 0) {
      d_rope(&rec, &c->ker.rope_neox, &c->q, &c->rope_freqs, hd, (int)pos,
             rope_len, L->rope.theta, has_freqs, (uint32_t)n_head);
      d_rope(&rec, &c->ker.rope_neox, &c->k, &c->rope_freqs, hd, (int)pos,
             rope_len, L->rope.theta, has_freqs, (uint32_t)n_kv);
    }
    d_rms(&rec, &c->ker, &c->v, &c->v, dummy, 0, vd, m->eps, (uint32_t)n_kv);

    size_t slot = pos % L->cache_cap;
    size_t seq = pos + 1;
    size_t t0 = seq > L->cache_cap ? seq - L->cache_cap : 0;
    float scale =
        m->attn_scale > 0.0f ? m->attn_scale : 1.0f / sqrtf((float)hd);

    {
      PCKv pc = {k_len, v_len, (uint32_t)slot};
      const VkBuf* bufs[4] = {&D->k_cache, &D->v_cache, &c->k, &c->v};
      int kv_max = k_len > v_len ? k_len : v_len;
      vk_dispatch(&rec, &c->ker.kv_store, bufs, &pc, sizeof(pc),
                  (uint32_t)((kv_max + 255) / 256), 1, 1);
    }
    {
      PCAttn pc = {hd, vd, group, (int)L->cache_cap, (int)t0, (int)seq, scale};
      const VkBuf* bufs[5] = {&c->attn_res, &c->q, &D->k_cache, &D->v_cache,
                              &c->attn_scratch};
      vk_dispatch(&rec, &c->ker.attn, bufs, &pc, sizeof(pc), (uint32_t)n_head,
                  1, 1);
    }

    d_matvec(&rec, &c->ker, L->attn_out->ggml_type, &D->o_w, &c->attn_res,
             &c->attn_proj, h, n_head * vd);
    if (D->has_post_attn)
      d_rms(&rec, &c->ker, &c->attn_proj, &c->attn_proj, &D->post_attn_norm, 1,
            h, m->eps, 1);
    d_add(&rec, &c->ker, &c->attn_proj, &c->x, h);

    d_rms(&rec, &c->ker, &c->normed, &c->attn_proj,
          D->ffn_norm.buf ? &D->ffn_norm : dummy, D->ffn_norm.buf != 0, h,
          m->eps, 1);
    d_matvec(&rec, &c->ker, L->ffn_gate->ggml_type, &D->gate_w, &c->normed,
             &c->gate, (int)m->inter, h);
    d_matvec(&rec, &c->ker, L->ffn_up->ggml_type, &D->up_w, &c->normed, &c->up,
             (int)m->inter, h);
    {
      PCInt pc = {(int)m->inter};
      const VkBuf* bufs[2] = {&c->gate, &c->up};
      vk_dispatch(&rec, &c->ker.geglu, bufs, &pc, sizeof(pc),
                  (uint32_t)(((int)m->inter + 255) / 256), 1, 1);
    }
    d_matvec(&rec, &c->ker, L->ffn_down->ggml_type, &D->down_w, &c->gate,
             &c->ffn_out, h, (int)m->inter);
    if (D->has_post_ffn)
      d_rms(&rec, &c->ker, &c->ffn_out, &c->ffn_out, &D->post_ffn_norm, 1, h,
            m->eps, 1);
    {
      PCResid pc = {h, L->output_scale};
      const VkBuf* bufs[3] = {&c->x, &c->ffn_out, &c->attn_proj};
      vk_dispatch(&rec, &c->ker.resid_out, bufs, &pc, sizeof(pc),
                  (uint32_t)((h + 255) / 256), 1, 1);
    }
  }

  if (hidden_out) {
    if (vk_rec_submit_wait(&rec) != 0) {
      fprintf(stderr, "vulkan: submit_wait failed\n");
      return -1;
    }
    if (vk_download(&c->ctx, &c->x, hidden_out, (VkDeviceSize)h * 4) != 0) {
      fprintf(stderr, "vulkan: hidden download failed\n");
      return -1;
    }
    return 0;
  }

  if (!logits_out && !argmax_out) {
    if (vk_rec_submit_wait(&rec) != 0) {
      fprintf(stderr, "vulkan: submit_wait failed\n");
      return -1;
    }
    return 0;
  }

  /* final norm + tied-embedding logits */
  d_rms(&rec, &c->ker, &c->normed, &c->x, &c->out_norm, 1, h, m->eps, 1);
  d_matvec(&rec, &c->ker, m->tok_embd->ggml_type, &c->tok_embd, &c->normed,
           &c->logits, (int)m->vocab, h);

  if (argmax_out) {
    PCInt pc = {(int)m->vocab};
    const VkBuf* bufs[2] = {&c->logits, &c->argmax_d};
    vk_dispatch(&rec, &c->ker.argmax, bufs, &pc, sizeof(pc), 1, 1, 1);
  }
  if (logits_out && m->final_softcap > 0.0f) {
    PCSoft pc = {(int)m->vocab, m->final_softcap};
    const VkBuf* bufs[1] = {&c->logits};
    vk_dispatch(&rec, &c->ker.softcap, bufs, &pc, sizeof(pc),
                (uint32_t)(((int)m->vocab + 255) / 256), 1, 1);
  }

  if (vk_rec_submit_wait(&rec) != 0) {
    fprintf(stderr, "vulkan: submit_wait failed\n");
    return -1;
  }
  if (argmax_out &&
      vk_download(&c->ctx, &c->argmax_d, argmax_out, 4) != 0) {
    fprintf(stderr, "vulkan: argmax download failed\n");
    return -1;
  }
  if (logits_out &&
      vk_download(&c->ctx, &c->logits, logits_out, m->vocab * 4) != 0) {
    fprintf(stderr, "vulkan: logits download failed\n");
    return -1;
  }
  return 0;
}

float* gemma4_vk_step(Gemma4Vk* c, Gemma4Model* m, int32_t token, size_t pos,
                      bool need_logits, int* failed) {
  *failed = 0;
  if (c->n_gpu_layers == m->n_layers) {
    float* lg = need_logits ? m->logits : NULL;
    if (gemma4_vk_forward(c, token, pos, lg, NULL, NULL) != 0) {
      *failed = 1;
      return NULL;
    }
    m->kv_len = pos + 1;
    return lg;
  }
  if (gemma4_vk_forward(c, token, pos, NULL, NULL, m->x) != 0) {
    *failed = 1;
    return NULL;
  }
  return gemma4_forward_from(m, pos, c->n_gpu_layers, need_logits);
}
