/* ============================================================================
 * UNVERIFIED — THIS FILE HAS NEVER BEEN COMPILED OR RUN.
 * ----------------------------------------------------------------------------
 * Written BLIND against:
 *   - the VERIFIED CUDA backend  src/cuda/gemma4_cuda.cu  (resident-forward
 *     shape this file mirrors), and
 *   - the Rust MLX backend  oxidize-core/src/backends/mlx.rs.
 * Requires: a Mac with Apple Silicon + a working mlx-c install.
 * It CANNOT be built or run in the authoring environment and MAY NOT COMPILE.
 * Do not trust any of it until a real-hardware validator confirms every mlx-c
 * call and logit-equivalence against gemma4_forward (CPU).
 * ============================================================================
 *
 * Host orchestration for Gemma 4 on MLX. Same layer graph as gemma4_cuda.cu /
 * model_gemma4.c, KEY DIFFERENCE: every weight is host-dequantized to f32 via
 * mlx_upload_weight() (no on-device ggml decode). Single GPU only. */
#include "mlx_backend.h"
#include "mlx_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  MlxMat q, k, v, o, gate, up, down;
  int has_v; /* 0 => K=V: V is raw K projection copied BEFORE k_norm/rope */
  mlx_array attn_norm, q_norm, k_norm, post_attn_norm, ffn_norm, post_ffn_norm;
  mlx_array k_cache, v_cache; /* growing [1,n_kv,seq,dim]; empty until first tok */
} MlxGemmaLayer;

struct Gemma4Mlx {
  const Gemma4Model* m;
  size_t n_gpu_layers;
  mlx_stream stream;
  MlxGemmaLayer* layers;
  MlxMat tok_embd;   /* f32 [vocab, hidden]; head only when full stack */
  mlx_array out_norm;
  mlx_array rope_freqs; /* global layers; empty handle unused on SWA */
  mlx_array ones;       /* scale-less V rmsnorm weight (max_hd) */
  mlx_array empty_freqs;
  int max_hd;
  int owns_head; /* 1 => final norm + tied logits + softcap on MLX */
};

#define FAIL_EMPTY(a)                     \
  do {                                    \
    if (mlx_array_empty_p(a)) return -1;  \
  } while (0)

static void free_mat(MlxMat* m) {
  mlx_release(m->a);
  m->a = mlx_array_new();
  m->rows = m->cols = 0;
}

static int up_mat(MlxMat* dst, const GgufTensorInfo* t, const char* what,
                  char* err, size_t errlen) {
  if (mlx_check_type(t->ggml_type, what, err, errlen) != 0) return -1;
  *dst = mlx_upload_weight(t, err, errlen);
  return mlx_array_empty_p(dst->a) ? -1 : 0;
}

static mlx_array up_vec_or_empty(const float* v, int n) {
  if (!v || n <= 0) return mlx_array_new();
  return mlx_upload_vec(v, n);
}

int gemma4_mlx_init(Gemma4Mlx** out, const Gemma4Model* m, int n_gpus,
                    int n_gpu_layers, char* err, size_t errlen) {
  *out = NULL;
  if (n_gpus != 1) {
    if (err && errlen)
      snprintf(err, errlen,
               "mlx: Apple Silicon is single-GPU; n_gpus=%d refused "
               "(pass 1; the CUDA layer-split pipeline is gemma4-CUDA-only)",
               n_gpus);
    return -1;
  }
  size_t ngl = (n_gpu_layers < 0 || (size_t)n_gpu_layers > m->n_layers)
                   ? m->n_layers
                   : (size_t)n_gpu_layers;
  if (ngl == 0) {
    if (err && errlen)
      snprintf(err, errlen, "mlx: --ngl 0 is the pure-CPU path (no MLX init)");
    return -1;
  }
  bool partial = ngl < m->n_layers;

  int max_hd = 0;
  for (size_t l = 0; l < ngl; ++l) {
    const Gemma4Layer* L = &m->layers[l];
    if ((int)L->head_dim > max_hd) max_hd = (int)L->head_dim;
    if ((int)L->v_head_dim > max_hd) max_hd = (int)L->v_head_dim;
  }

  Gemma4Mlx* c = (Gemma4Mlx*)calloc(1, sizeof(Gemma4Mlx));
  if (!c) return -1;
  c->m = m;
  c->n_gpu_layers = ngl;
  c->owns_head = !partial;
  c->max_hd = max_hd;
  c->stream = mlx_default_gpu_stream_new();
  c->empty_freqs = mlx_array_new();
  c->layers = (MlxGemmaLayer*)calloc(ngl, sizeof(MlxGemmaLayer));
  if (!c->layers) {
    free(c);
    return -1;
  }
  for (size_t l = 0; l < ngl; ++l) {
    c->layers[l].k_cache = mlx_array_new();
    c->layers[l].v_cache = mlx_array_new();
    c->layers[l].attn_norm = mlx_array_new();
    c->layers[l].q_norm = mlx_array_new();
    c->layers[l].k_norm = mlx_array_new();
    c->layers[l].post_attn_norm = mlx_array_new();
    c->layers[l].ffn_norm = mlx_array_new();
    c->layers[l].post_ffn_norm = mlx_array_new();
    c->layers[l].q.a = mlx_array_new();
    c->layers[l].k.a = mlx_array_new();
    c->layers[l].v.a = mlx_array_new();
    c->layers[l].o.a = mlx_array_new();
    c->layers[l].gate.a = mlx_array_new();
    c->layers[l].up.a = mlx_array_new();
    c->layers[l].down.a = mlx_array_new();
  }
  c->tok_embd.a = mlx_array_new();
  c->out_norm = mlx_array_new();
  c->rope_freqs = mlx_array_new();
  c->ones = mlx_array_new();

  if (mlx_check_type(m->tok_embd->ggml_type, "token_embd", err, errlen) != 0)
    goto fail;

  if (m->rope_freqs && max_hd >= 2) {
    c->rope_freqs = mlx_upload_vec(m->rope_freqs, max_hd / 2);
    if (mlx_array_empty_p(c->rope_freqs)) goto fail_msg;
  }
  if (m->ones && max_hd > 0) {
    c->ones = mlx_upload_vec(m->ones, max_hd);
    if (mlx_array_empty_p(c->ones)) goto fail_msg;
  }

  if (c->owns_head) {
    if (up_mat(&c->tok_embd, m->tok_embd, "token_embd", err, errlen) != 0)
      goto fail;
    c->out_norm = mlx_upload_vec(m->out_norm, (int)m->hidden);
    if (mlx_array_empty_p(c->out_norm)) goto fail_msg;
  }

  for (size_t l = 0; l < ngl; ++l) {
    const Gemma4Layer* L = &m->layers[l];
    MlxGemmaLayer* D = &c->layers[l];
    if (up_mat(&D->q, L->attn_q, "attn_q", err, errlen) != 0) goto fail;
    if (up_mat(&D->k, L->attn_k, "attn_k", err, errlen) != 0) goto fail;
    if (L->attn_v) {
      if (up_mat(&D->v, L->attn_v, "attn_v", err, errlen) != 0) goto fail;
      D->has_v = 1;
    } else {
      D->has_v = 0;
    }
    if (up_mat(&D->o, L->attn_out, "attn_output", err, errlen) != 0) goto fail;
    if (up_mat(&D->gate, L->ffn_gate, "ffn_gate", err, errlen) != 0) goto fail;
    if (up_mat(&D->up, L->ffn_up, "ffn_up", err, errlen) != 0) goto fail;
    if (up_mat(&D->down, L->ffn_down, "ffn_down", err, errlen) != 0) goto fail;

    D->attn_norm = up_vec_or_empty(L->attn_norm, (int)m->hidden);
    D->q_norm = up_vec_or_empty(L->attn_q_norm, (int)L->head_dim);
    D->k_norm = up_vec_or_empty(L->attn_k_norm, (int)L->head_dim);
    D->post_attn_norm = up_vec_or_empty(L->post_attn_norm, (int)m->hidden);
    D->ffn_norm = up_vec_or_empty(L->ffn_norm, (int)m->hidden);
    D->post_ffn_norm = up_vec_or_empty(L->post_ffn_norm, (int)m->hidden);
    if (mlx_array_empty_p(D->attn_norm) || mlx_array_empty_p(D->ffn_norm))
      goto fail_msg;
  }

  if (m->kv_quant)
    fprintf(stderr,
            "mlx: warning: model requested rotoquant KV; MLX path uses f32 "
            "growing caches (not bit-equal to CUDA kv_quant)\n");

  fprintf(stderr, "mlx: gemma4 %zu/%zu layers on MLX%s (f32 weights+KV)\n", ngl,
          m->n_layers,
          partial ? "; remaining layers + head on CPU" : "");
  *out = c;
  return 0;

fail_msg:
  if (err && errlen && !err[0])
    snprintf(err, errlen, "mlx: gemma4 weight upload / allocation failed");
fail:
  gemma4_mlx_free(c);
  return -1;
}

void gemma4_mlx_free(Gemma4Mlx* c) {
  if (!c) return;
  for (size_t l = 0; c->layers && l < c->n_gpu_layers; ++l) {
    MlxGemmaLayer* D = &c->layers[l];
    free_mat(&D->q);
    free_mat(&D->k);
    free_mat(&D->v);
    free_mat(&D->o);
    free_mat(&D->gate);
    free_mat(&D->up);
    free_mat(&D->down);
    mlx_release(D->attn_norm);
    mlx_release(D->q_norm);
    mlx_release(D->k_norm);
    mlx_release(D->post_attn_norm);
    mlx_release(D->ffn_norm);
    mlx_release(D->post_ffn_norm);
    mlx_release(D->k_cache);
    mlx_release(D->v_cache);
  }
  free(c->layers);
  free_mat(&c->tok_embd);
  mlx_release(c->out_norm);
  mlx_release(c->rope_freqs);
  mlx_release(c->ones);
  mlx_release(c->empty_freqs);
  /* stream free: ASSUMED mlx_stream_free; validator confirms. */
  mlx_stream_free(c->stream);
  free(c);
}

/* Ones weight of length `vd` for scale-less V rmsnorm. Always a fresh upload
 * so the caller can mlx_release without worrying about borrowing c->ones. */
static mlx_array ones_vd(Gemma4Mlx* c, int vd) {
  if (vd <= 0 || !c->m->ones) return mlx_array_new();
  return mlx_upload_vec(c->m->ones, vd);
}

static int gemma4_mlx_forward(Gemma4Mlx* c, int32_t token, size_t pos,
                              float* logits_out, float* hidden_out) {
  const Gemma4Model* m = c->m;
  const int h = (int)m->hidden;
  mlx_stream s = c->stream;
  char ebuf[128];

  if (pos >= m->ctx) {
    fprintf(stderr, "mlx: position %zu exceeds context %zu\n", pos, m->ctx);
    return -1;
  }

  mlx_array x =
      mlx_embed_row(m->tok_embd, token, m->hidden, m->emb_scale, ebuf, sizeof(ebuf));
  if (mlx_array_empty_p(x)) {
    fprintf(stderr, "mlx: embed failed: %s\n", ebuf);
    return -1;
  }

  for (size_t l = 0; l < c->n_gpu_layers; ++l) {
    const Gemma4Layer* L = &m->layers[l];
    MlxGemmaLayer* D = &c->layers[l];
    int hd = (int)L->head_dim, vd = (int)L->v_head_dim;
    int n_head = (int)m->n_head, n_kv = (int)L->n_kv_heads;
    int q_len = n_head * hd, k_len = n_kv * hd;
    int rope_len = L->rope.rope_dim ? (int)L->rope.rope_dim : hd;
    float scale =
        m->attn_scale > 0.0f ? m->attn_scale : 1.0f / sqrtf((float)hd);
    int window = L->is_swa ? (int)L->cache_cap : 0;
    mlx_array freqs = L->is_swa ? c->empty_freqs : c->rope_freqs;

    /* ---- attention ---- */
    mlx_array normed = mlx_rmsnorm(s, x, D->attn_norm, h, m->eps);
    FAIL_EMPTY(normed);
    mlx_array q = mlx_matvec(s, D->q, normed);
    mlx_array k = mlx_matvec(s, D->k, normed);
    FAIL_EMPTY(q);
    FAIL_EMPTY(k);
    mlx_array v;
    if (D->has_v) {
      v = mlx_matvec(s, D->v, normed);
    } else {
      /* K=V: V is the RAW K projection — BEFORE k_norm/rope */
      v = mlx_ew_scale(s, k, 1.0f);
    }
    mlx_release(normed);
    FAIL_EMPTY(v);

    /* per-head Q/K RMSNorm */
    if (!mlx_array_empty_p(D->q_norm)) {
      int qsh[2] = {n_head, hd};
      mlx_array q2 = mlx_reshape_to(s, q, qsh, 2);
      mlx_release(q);
      q = mlx_rmsnorm(s, q2, D->q_norm, hd, m->eps);
      mlx_release(q2);
      FAIL_EMPTY(q);
      int qflat[1] = {q_len};
      mlx_array qf = mlx_reshape_to(s, q, qflat, 1);
      mlx_release(q);
      q = qf;
      FAIL_EMPTY(q);
    }
    if (!mlx_array_empty_p(D->k_norm)) {
      int ksh[2] = {n_kv, hd};
      mlx_array k2 = mlx_reshape_to(s, k, ksh, 2);
      mlx_release(k);
      k = mlx_rmsnorm(s, k2, D->k_norm, hd, m->eps);
      mlx_release(k2);
      FAIL_EMPTY(k);
      int kflat[1] = {k_len};
      mlx_array kf = mlx_reshape_to(s, k, kflat, 1);
      mlx_release(k);
      k = kf;
      FAIL_EMPTY(k);
    }

    /* RoPE on [heads, 1, head_dim]; NeoX (traditional=0). Global layers get
     * rope_freqs; SWA passes empty. Host skips at pos==0 (identity). */
    {
      int q3[3] = {n_head, 1, hd};
      int k3[3] = {n_kv, 1, hd};
      mlx_array qh = mlx_reshape_to(s, q, q3, 3);
      mlx_array kh = mlx_reshape_to(s, k, k3, 3);
      mlx_release(q);
      mlx_release(k);
      if (pos > 0 && rope_len > 0) {
        q = mlx_rope(s, qh, n_head, hd, rope_len, pos, L->rope.theta, 0, freqs);
        k = mlx_rope(s, kh, n_kv, hd, rope_len, pos, L->rope.theta, 0, freqs);
        mlx_release(qh);
        mlx_release(kh);
      } else {
        q = qh;
        k = kh;
      }
      FAIL_EMPTY(q);
      FAIL_EMPTY(k);
    }

    /* V: scale-less RMSNorm, no rope */
    {
      int vsh[2] = {n_kv, vd};
      mlx_array v2 = mlx_reshape_to(s, v, vsh, 2);
      mlx_release(v);
      mlx_array ov = ones_vd(c, vd);
      v = mlx_rmsnorm(s, v2, ov, vd, m->eps);
      mlx_release(v2);
      mlx_release(ov);
      FAIL_EMPTY(v);
    }

    /* flatten q/k for cache append shapes [n_kv, dim] */
    {
      int k2[2] = {n_kv, hd};
      int v2s[2] = {n_kv, vd};
      mlx_array krow = mlx_reshape_to(s, k, k2, 2);
      mlx_array vrow = mlx_reshape_to(s, v, v2s, 2);
      mlx_release(k);
      mlx_release(v);
      D->k_cache = mlx_kv_append(s, D->k_cache, krow, n_kv, hd);
      D->v_cache = mlx_kv_append(s, D->v_cache, vrow, n_kv, vd);
      mlx_release(krow);
      mlx_release(vrow);
      FAIL_EMPTY(D->k_cache);
      FAIL_EMPTY(D->v_cache);
    }

    /* q -> [1, n_head, 1, hd] for SDPA */
    {
      int q4[4] = {1, n_head, 1, hd};
      mlx_array qsdpa = mlx_reshape_to(s, q, q4, 4);
      mlx_release(q);
      mlx_array attn =
          mlx_attention(s, qsdpa, D->k_cache, D->v_cache, n_head, hd, vd, scale,
                        window);
      mlx_release(qsdpa);
      FAIL_EMPTY(attn);
      int aflat[1] = {n_head * vd};
      mlx_array aflat_a = mlx_reshape_to(s, attn, aflat, 1);
      mlx_release(attn);
      attn = aflat_a;
      FAIL_EMPTY(attn);

      mlx_array attn_proj = mlx_matvec(s, D->o, attn);
      mlx_release(attn);
      FAIL_EMPTY(attn_proj);
      if (!mlx_array_empty_p(D->post_attn_norm)) {
        mlx_array pan =
            mlx_rmsnorm(s, attn_proj, D->post_attn_norm, h, m->eps);
        mlx_release(attn_proj);
        attn_proj = pan;
        FAIL_EMPTY(attn_proj);
      }
      /* attn_out = post_attn_norm(proj) + residual */
      mlx_array attn_out = mlx_ew_add(s, attn_proj, x);
      mlx_release(attn_proj);
      mlx_release(x);
      FAIL_EMPTY(attn_out);

      /* ---- FFN (GeGLU) over attn_out ---- */
      mlx_array fn = mlx_rmsnorm(s, attn_out, D->ffn_norm, h, m->eps);
      FAIL_EMPTY(fn);
      mlx_array gate = mlx_matvec(s, D->gate, fn);
      mlx_array up = mlx_matvec(s, D->up, fn);
      mlx_release(fn);
      FAIL_EMPTY(gate);
      FAIL_EMPTY(up);
      mlx_array gg = mlx_geglu(s, gate, up);
      mlx_release(gate);
      mlx_release(up);
      FAIL_EMPTY(gg);
      mlx_array ffn_out = mlx_matvec(s, D->down, gg);
      mlx_release(gg);
      FAIL_EMPTY(ffn_out);
      if (!mlx_array_empty_p(D->post_ffn_norm)) {
        mlx_array pfn = mlx_rmsnorm(s, ffn_out, D->post_ffn_norm, h, m->eps);
        mlx_release(ffn_out);
        ffn_out = pfn;
        FAIL_EMPTY(ffn_out);
      }
      /* x = (ffn + attn_out) * output_scale */
      x = mlx_resid_out(s, ffn_out, attn_out, L->output_scale);
      mlx_release(ffn_out);
      mlx_release(attn_out);
      FAIL_EMPTY(x);
    }
  }

  if (hidden_out) {
    if (mlx_eval_to_host(s, x, hidden_out, h) != 0) {
      mlx_release(x);
      fprintf(stderr, "mlx: hidden D2H failed\n");
      return -1;
    }
    mlx_release(x);
    return 0;
  }

  if (!logits_out) {
    if (mlx_array_eval(x) != 0) {
      mlx_release(x);
      fprintf(stderr, "mlx: eval failed\n");
      return -1;
    }
    mlx_release(x);
    return 0;
  }

  /* final norm + tied logits + softcap */
  mlx_array normed = mlx_rmsnorm(s, x, c->out_norm, h, m->eps);
  mlx_release(x);
  FAIL_EMPTY(normed);
  mlx_array logits = mlx_matvec(s, c->tok_embd, normed);
  mlx_release(normed);
  FAIL_EMPTY(logits);
  if (m->final_softcap > 0.0f) {
    mlx_array sc = mlx_softcap(s, logits, m->final_softcap);
    mlx_release(logits);
    logits = sc;
    FAIL_EMPTY(logits);
  }
  if (mlx_eval_to_host(s, logits, logits_out, (int)m->vocab) != 0) {
    mlx_release(logits);
    fprintf(stderr, "mlx: logits D2H failed\n");
    return -1;
  }
  mlx_release(logits);
  return 0;
}

float* gemma4_mlx_step(Gemma4Mlx* c, Gemma4Model* m, int32_t token, size_t pos,
                       bool need_logits, int* failed) {
  *failed = 0;
  if (c->n_gpu_layers == m->n_layers) {
    float* lg = need_logits ? m->logits : NULL;
    if (gemma4_mlx_forward(c, token, pos, lg, NULL) != 0) {
      *failed = 1;
      return NULL;
    }
    m->kv_len = pos + 1;
    return lg;
  }
  if (gemma4_mlx_forward(c, token, pos, NULL, m->x) != 0) {
    *failed = 1;
    return NULL;
  }
  return gemma4_forward_from(m, pos, c->n_gpu_layers, need_logits);
}
