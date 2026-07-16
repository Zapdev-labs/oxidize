/* ============================================================================
 * UNVERIFIED — THIS FILE HAS NEVER BEEN COMPILED OR RUN.
 * ----------------------------------------------------------------------------
 * Written BLIND against:
 *   - the VERIFIED CUDA backend  src/cuda/llama_cuda.cu  (resident-forward
 *     shape this file mirrors), and
 *   - the Rust MLX backend  oxidize-core/src/backends/mlx.rs.
 * Requires: a Mac with Apple Silicon + a working mlx-c install.
 * It CANNOT be built or run in the authoring environment and MAY NOT COMPILE.
 * Do not trust any of it until a real-hardware validator confirms every mlx-c
 * call and logit-equivalence against llama_forward (CPU).
 * ============================================================================
 *
 * Host orchestration for the generic llama-family dense engine on MLX.
 * Mirrors llama_cuda.cu: GQA, both RoPE modes, optional biases, optional q/k
 * norms, SwiGLU, tied/untied head. KEY DIFFERENCE: host-dequant to f32 via
 * mlx_upload_weight(). MoE in the offload range is refused. Single GPU only. */
#include "mlx_backend.h"
#include "mlx_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  MlxMat q, k, v, o, gate, up, down;
  mlx_array attn_norm, ffn_norm, q_norm, k_norm;
  mlx_array bias_q, bias_k, bias_v, bias_o;
  mlx_array k_cache, v_cache; /* growing [1,n_kv,seq,hd] */
} MlxLlamaLayer;

struct LlamaMlx {
  const LlamaModel* m;
  size_t n_gpu_layers;
  mlx_stream stream;
  MlxLlamaLayer* layers;
  MlxMat tok_embd; /* uploaded for untied reuse / tied head */
  MlxMat out_w;    /* untied head; empty when tied */
  int tied;        /* 1 => logits use tok_embd */
  mlx_array out_norm;
  mlx_array empty_freqs; /* passed to mlx_rope when no custom freqs */
  int owns_head;
};

#define FAIL_EMPTY(a)                    \
  do {                                   \
    if (mlx_array_empty_p(a)) return -1; \
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

int llama_mlx_init(LlamaMlx** out, const LlamaModel* m, int n_gpus,
                   int n_gpu_layers, char* err, size_t errlen) {
  *out = NULL;
  if (n_gpus != 1) {
    if (err && errlen)
      snprintf(err, errlen,
               "mlx: llama backend is single-GPU (n_gpus=%d refused); "
               "Apple Silicon has one GPU and the CUDA layer-split pipeline "
               "is gemma4-only",
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

  for (size_t l = 0; l < ngl; ++l)
    if (m->layers[l].is_moe) {
      if (err && errlen)
        snprintf(err, errlen,
                 "mlx: layer %zu is Mixture-of-Experts; MoE is not offloaded. "
                 "Use --ngl %zu to keep the MoE layers on the CPU, or run on CPU",
                 l, l);
      return -1;
    }

  LlamaMlx* c = (LlamaMlx*)calloc(1, sizeof(LlamaMlx));
  if (!c) return -1;
  c->m = m;
  c->n_gpu_layers = ngl;
  c->owns_head = !partial;
  c->tied = (m->out_w == m->tok_embd);
  c->stream = mlx_default_gpu_stream_new();
  c->empty_freqs = mlx_array_new();
  c->layers = (MlxLlamaLayer*)calloc(ngl, sizeof(MlxLlamaLayer));
  if (!c->layers) {
    free(c);
    return -1;
  }

  for (size_t l = 0; l < ngl; ++l) {
    MlxLlamaLayer* D = &c->layers[l];
    D->q.a = mlx_array_new();
    D->k.a = mlx_array_new();
    D->v.a = mlx_array_new();
    D->o.a = mlx_array_new();
    D->gate.a = mlx_array_new();
    D->up.a = mlx_array_new();
    D->down.a = mlx_array_new();
    D->attn_norm = mlx_array_new();
    D->ffn_norm = mlx_array_new();
    D->q_norm = mlx_array_new();
    D->k_norm = mlx_array_new();
    D->bias_q = mlx_array_new();
    D->bias_k = mlx_array_new();
    D->bias_v = mlx_array_new();
    D->bias_o = mlx_array_new();
    D->k_cache = mlx_array_new();
    D->v_cache = mlx_array_new();
  }
  c->tok_embd.a = mlx_array_new();
  c->out_w.a = mlx_array_new();
  c->out_norm = mlx_array_new();

  if (mlx_check_type(m->tok_embd->ggml_type, "token_embd", err, errlen) != 0)
    goto fail;
  if (!partial && !c->tied &&
      mlx_check_type(m->out_w->ggml_type, "output.weight", err, errlen) != 0)
    goto fail;

  if (c->owns_head) {
    if (up_mat(&c->tok_embd, m->tok_embd, "token_embd", err, errlen) != 0)
      goto fail;
    c->out_norm = mlx_upload_vec(m->out_norm, (int)m->hidden);
    if (mlx_array_empty_p(c->out_norm)) goto fail_msg;
    if (!c->tied) {
      if (up_mat(&c->out_w, m->out_w, "output.weight", err, errlen) != 0)
        goto fail;
    }
  }

  {
    size_t hd = m->head_dim, kv_row = m->n_kv_heads * hd;
    for (size_t l = 0; l < ngl; ++l) {
      const LlamaLayer* L = &m->layers[l];
      MlxLlamaLayer* D = &c->layers[l];
      if (up_mat(&D->q, L->attn_q, "attn_q", err, errlen) != 0) goto fail;
      if (up_mat(&D->k, L->attn_k, "attn_k", err, errlen) != 0) goto fail;
      if (up_mat(&D->v, L->attn_v, "attn_v", err, errlen) != 0) goto fail;
      if (up_mat(&D->o, L->attn_out, "attn_output", err, errlen) != 0) goto fail;
      if (up_mat(&D->gate, L->ffn_gate, "ffn_gate", err, errlen) != 0) goto fail;
      if (up_mat(&D->up, L->ffn_up, "ffn_up", err, errlen) != 0) goto fail;
      if (up_mat(&D->down, L->ffn_down, "ffn_down", err, errlen) != 0) goto fail;

      D->attn_norm = up_vec_or_empty(L->attn_norm, (int)m->hidden);
      D->ffn_norm = up_vec_or_empty(L->ffn_norm, (int)m->hidden);
      D->q_norm = up_vec_or_empty(L->attn_q_norm, (int)hd);
      D->k_norm = up_vec_or_empty(L->attn_k_norm, (int)hd);
      D->bias_q = up_vec_or_empty(L->bias_q, (int)(m->n_head * hd));
      D->bias_k = up_vec_or_empty(L->bias_k, (int)kv_row);
      D->bias_v = up_vec_or_empty(L->bias_v, (int)kv_row);
      D->bias_o = up_vec_or_empty(L->bias_o, (int)m->hidden);
      if (mlx_array_empty_p(D->attn_norm) || mlx_array_empty_p(D->ffn_norm))
        goto fail_msg;
    }
  }

  fprintf(stderr, "mlx: llama %zu/%zu layers on MLX%s (f32 weights+KV)\n", ngl,
          m->n_layers,
          partial ? "; remaining layers + head on CPU" : "");
  *out = c;
  return 0;

fail_msg:
  if (err && errlen && !err[0])
    snprintf(err, errlen, "mlx: llama weight upload / allocation failed");
fail:
  llama_mlx_free(c);
  return -1;
}

void llama_mlx_free(LlamaMlx* c) {
  if (!c) return;
  for (size_t l = 0; c->layers && l < c->n_gpu_layers; ++l) {
    MlxLlamaLayer* D = &c->layers[l];
    free_mat(&D->q);
    free_mat(&D->k);
    free_mat(&D->v);
    free_mat(&D->o);
    free_mat(&D->gate);
    free_mat(&D->up);
    free_mat(&D->down);
    mlx_release(D->attn_norm);
    mlx_release(D->ffn_norm);
    mlx_release(D->q_norm);
    mlx_release(D->k_norm);
    mlx_release(D->bias_q);
    mlx_release(D->bias_k);
    mlx_release(D->bias_v);
    mlx_release(D->bias_o);
    mlx_release(D->k_cache);
    mlx_release(D->v_cache);
  }
  free(c->layers);
  free_mat(&c->tok_embd);
  free_mat(&c->out_w);
  mlx_release(c->out_norm);
  mlx_release(c->empty_freqs);
  mlx_stream_free(c->stream);
  free(c);
}

static int llama_mlx_forward(LlamaMlx* c, int32_t token, size_t pos,
                             float* logits_out, float* hidden_out) {
  const LlamaModel* m = c->m;
  const int h = (int)m->hidden;
  mlx_stream s = c->stream;
  char ebuf[128];

  if (pos >= m->ctx) {
    fprintf(stderr, "mlx: position %zu exceeds context %zu\n", pos, m->ctx);
    return -1;
  }

  /* embedding lookup (no scale) */
  mlx_array x =
      mlx_embed_row(m->tok_embd, token, m->hidden, 1.0f, ebuf, sizeof(ebuf));
  if (mlx_array_empty_p(x)) {
    fprintf(stderr, "mlx: embed failed: %s\n", ebuf);
    return -1;
  }

  const int hd = (int)m->head_dim;
  const int n_head = (int)m->n_head, n_kv = (int)m->n_kv_heads;
  const int q_len = n_head * hd, kv_len = n_kv * hd;
  const int rope_len = m->rope_dim ? (int)m->rope_dim : hd;
  const float scale = 1.0f / sqrtf((float)hd);
  const int traditional = m->rope_norm ? 1 : 0; /* NORMAL vs NeoX */

  for (size_t l = 0; l < c->n_gpu_layers; ++l) {
    MlxLlamaLayer* D = &c->layers[l];

    mlx_array normed = mlx_rmsnorm(s, x, D->attn_norm, h, m->eps);
    FAIL_EMPTY(normed);
    mlx_array q = mlx_matvec(s, D->q, normed);
    mlx_array k = mlx_matvec(s, D->k, normed);
    mlx_array v = mlx_matvec(s, D->v, normed);
    mlx_release(normed);
    FAIL_EMPTY(q);
    FAIL_EMPTY(k);
    FAIL_EMPTY(v);

    if (!mlx_array_empty_p(D->bias_q)) {
      mlx_array t = mlx_ew_add(s, q, D->bias_q);
      mlx_release(q);
      q = t;
      FAIL_EMPTY(q);
    }
    if (!mlx_array_empty_p(D->bias_k)) {
      mlx_array t = mlx_ew_add(s, k, D->bias_k);
      mlx_release(k);
      k = t;
      FAIL_EMPTY(k);
    }
    if (!mlx_array_empty_p(D->bias_v)) {
      mlx_array t = mlx_ew_add(s, v, D->bias_v);
      mlx_release(v);
      v = t;
      FAIL_EMPTY(v);
    }

    /* optional per-head q/k RMSNorm (qwen3); V gets neither */
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
      int kflat[1] = {kv_len};
      mlx_array kf = mlx_reshape_to(s, k, kflat, 1);
      mlx_release(k);
      k = kf;
      FAIL_EMPTY(k);
    }

    /* RoPE: traditional=1 => ggml NORMAL; 0 => NeoX. Skip at pos 0. */
    {
      int q3[3] = {n_head, 1, hd};
      int k3[3] = {n_kv, 1, hd};
      mlx_array qh = mlx_reshape_to(s, q, q3, 3);
      mlx_array kh = mlx_reshape_to(s, k, k3, 3);
      mlx_release(q);
      mlx_release(k);
      if (pos > 0 && rope_len > 0) {
        q = mlx_rope(s, qh, n_head, hd, rope_len, pos, m->rope_theta,
                     traditional, c->empty_freqs);
        k = mlx_rope(s, kh, n_kv, hd, rope_len, pos, m->rope_theta,
                     traditional, c->empty_freqs);
        mlx_release(qh);
        mlx_release(kh);
      } else {
        q = qh;
        k = kh;
      }
      FAIL_EMPTY(q);
      FAIL_EMPTY(k);
    }

    /* V: reshape for cache; no rope / no norm beyond projection */
    {
      int k2[2] = {n_kv, hd};
      int v2[2] = {n_kv, hd};
      mlx_array krow = mlx_reshape_to(s, k, k2, 2);
      mlx_array vrow = mlx_reshape_to(s, v, v2, 2);
      mlx_release(k);
      mlx_release(v);
      D->k_cache = mlx_kv_append(s, D->k_cache, krow, n_kv, hd);
      D->v_cache = mlx_kv_append(s, D->v_cache, vrow, n_kv, hd);
      mlx_release(krow);
      mlx_release(vrow);
      FAIL_EMPTY(D->k_cache);
      FAIL_EMPTY(D->v_cache);
    }

    {
      int q4[4] = {1, n_head, 1, hd};
      mlx_array qsdpa = mlx_reshape_to(s, q, q4, 4);
      mlx_release(q);
      /* full causal: window=0 */
      mlx_array attn = mlx_attention(s, qsdpa, D->k_cache, D->v_cache, n_head,
                                     hd, hd, scale, 0);
      mlx_release(qsdpa);
      FAIL_EMPTY(attn);
      int aflat[1] = {q_len};
      mlx_array flat = mlx_reshape_to(s, attn, aflat, 1);
      mlx_release(attn);
      FAIL_EMPTY(flat);

      mlx_array attn_proj = mlx_matvec(s, D->o, flat);
      mlx_release(flat);
      FAIL_EMPTY(attn_proj);
      if (!mlx_array_empty_p(D->bias_o)) {
        mlx_array t = mlx_ew_add(s, attn_proj, D->bias_o);
        mlx_release(attn_proj);
        attn_proj = t;
        FAIL_EMPTY(attn_proj);
      }
      mlx_array x2 = mlx_ew_add(s, x, attn_proj);
      mlx_release(x);
      mlx_release(attn_proj);
      x = x2;
      FAIL_EMPTY(x);
    }

    /* ---- FFN (SwiGLU) ---- */
    {
      mlx_array fn = mlx_rmsnorm(s, x, D->ffn_norm, h, m->eps);
      FAIL_EMPTY(fn);
      mlx_array gate = mlx_matvec(s, D->gate, fn);
      mlx_array up = mlx_matvec(s, D->up, fn);
      mlx_release(fn);
      FAIL_EMPTY(gate);
      FAIL_EMPTY(up);
      mlx_array sm = mlx_silu_mul(s, gate, up);
      mlx_release(gate);
      mlx_release(up);
      FAIL_EMPTY(sm);
      mlx_array ffn_out = mlx_matvec(s, D->down, sm);
      mlx_release(sm);
      FAIL_EMPTY(ffn_out);
      mlx_array x2 = mlx_ew_add(s, x, ffn_out);
      mlx_release(x);
      mlx_release(ffn_out);
      x = x2;
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

  /* final norm + tied/untied logits (no softcap) */
  mlx_array normed = mlx_rmsnorm(s, x, c->out_norm, h, m->eps);
  mlx_release(x);
  FAIL_EMPTY(normed);
  MlxMat head = c->tied ? c->tok_embd : c->out_w;
  mlx_array logits = mlx_matvec(s, head, normed);
  mlx_release(normed);
  FAIL_EMPTY(logits);
  if (mlx_eval_to_host(s, logits, logits_out, (int)m->vocab) != 0) {
    mlx_release(logits);
    fprintf(stderr, "mlx: logits D2H failed\n");
    return -1;
  }
  mlx_release(logits);
  return 0;
}

float* llama_mlx_step(LlamaMlx* c, LlamaModel* m, int32_t token, size_t pos,
                      bool need_logits, int* failed) {
  *failed = 0;
  if (c->n_gpu_layers == m->n_layers) {
    float* lg = need_logits ? m->logits : NULL;
    if (llama_mlx_forward(c, token, pos, lg, NULL) != 0) {
      *failed = 1;
      return NULL;
    }
    m->kv_len = pos + 1;
    return lg;
  }
  if (llama_mlx_forward(c, token, pos, NULL, m->x) != 0) {
    *failed = 1;
    return NULL;
  }
  return llama_forward_from(m, pos, c->n_gpu_layers, need_logits);
}
