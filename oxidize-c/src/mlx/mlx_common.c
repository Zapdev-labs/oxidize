/* ============================================================================
 * UNVERIFIED — NEVER COMPILED OR RUN. Written blind against src/cuda/ (verified
 * resident-forward reference) + oxidize-core/src/backends/mlx.rs. Requires a Mac
 * with Apple Silicon + mlx-c. MAY NOT COMPILE. Every mlx-c signature here is
 * ASSUMED from mlx_rs / the MLX C++ headers and must be checked against the
 * installed <mlx/c/*.h>; this file is deliberately the ONE place they all live.
 * ----------------------------------------------------------------------------
 * mlx-c calling convention ASSUMED throughout (matches mlx-c HEAD circa 2024):
 *   - handles are `mlx_array`, ref-counted; `mlx_array_new()` makes an empty
 *     out-handle, `mlx_array_free()` drops a ref.
 *   - ops write into an out-handle and return int status (0 ok):
 *        int mlx_matmul(mlx_array* res, mlx_array a, mlx_array b, mlx_stream s);
 *   - `mlx_array mlx_array_new_data(const void*, const int* shape, int ndim,
 *        mlx_dtype)` copies host bytes into a unified-memory array.
 *   - `mlx_array_eval(mlx_array)` forces evaluation; `mlx_array_data_float32()`
 *     returns the resident f32 pointer afterwards.
 * If the real API differs (e.g. status-返回 vs handle-return for _new_data), fix
 * it HERE. The forward passes never touch mlx-c directly.
 * ============================================================================ */
#include "mlx_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- tiny status helper: abort-to-empty on any nonzero mlx status. Callers
 * treat mlx_array_empty() as failure. We do not thread rich errors through the
 * hot path (mirrors the CUDA kernels, which just print + return -1). ---- */
#define MTRY(expr)                          \
  do {                                      \
    if ((expr) != 0) return mlx_array_new(); \
  } while (0)

int mlx_check_type(uint32_t t, const char* what, char* err, size_t errlen) {
  switch (t) {
    case OC_F32: case OC_F16: case OC_Q4_0: case OC_Q8_0:
    case OC_Q4_K: case OC_Q5_K: case OC_Q6_K: case OC_AL5_XS:
      return 0;
    default: break;
  }
  if (err && errlen)
    snprintf(err, errlen,
             "mlx: %s has quant type %u; this backend decodes "
             "F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/AL5_XS only", what, t);
  return -1;
}

/* rows = product of dims[1..], cols = dims[0] — same as cuda tensor_bytes(). */
static void mat_shape(const GgufTensorInfo* t, size_t* rows, size_t* cols) {
  *cols = (size_t)t->dims[0];
  *rows = 1;
  for (uint32_t d = 1; d < t->n_dims; ++d) *rows *= (size_t)t->dims[d];
}

MlxMat mlx_upload_weight(const GgufTensorInfo* t, char* err, size_t errlen) {
  MlxMat m = {mlx_array_new(), 0, 0};
  if (mlx_check_type(t->ggml_type, "weight", err, errlen) != 0) return m;
  size_t rows, cols;
  mat_shape(t, &rows, &cols);
  size_t rowb = oc_row_bytes(t->ggml_type, cols);
  float* f = (float*)malloc(rows * cols * sizeof(float));
  if (!f) {
    if (err && errlen) snprintf(err, errlen, "mlx: OOM dequantizing weight");
    return m;
  }
  const uint8_t* src = (const uint8_t*)t->data;
  for (size_t r = 0; r < rows; ++r)
    /* oc_dequant_row: the exact reference decoder (ggml semantics) the CPU
     * forward uses. This is where F32..AL5_XS become plain f32. */
    if (oc_dequant_row(t->ggml_type, src + r * rowb, f + r * cols, cols) != 0) {
      free(f);
      if (err && errlen) snprintf(err, errlen, "mlx: oc_dequant_row failed");
      return m;
    }
  int shape[2] = {(int)rows, (int)cols};
  m.a = mlx_array_new_data(f, shape, 2, MLX_FLOAT32);
  m.rows = (int)rows;
  m.cols = (int)cols;
  free(f); /* mlx_array_new_data copies into unified memory (mlx.rs comment) */
  return m;
}

mlx_array mlx_upload_vec(const float* v, int n) {
  int shape[1] = {n};
  return mlx_array_new_data(v, shape, 1, MLX_FLOAT32);
}

mlx_array mlx_embed_row(const GgufTensorInfo* tok_embd, int32_t token,
                        size_t hidden, float scale, char* err, size_t errlen) {
  if (mlx_check_type(tok_embd->ggml_type, "token_embd", err, errlen) != 0)
    return mlx_array_new();
  size_t rows, cols;
  mat_shape(tok_embd, &rows, &cols); /* cols == hidden */
  size_t tk = (size_t)token < rows ? (size_t)token : rows - 1;
  size_t rowb = oc_row_bytes(tok_embd->ggml_type, cols);
  float* f = (float*)malloc(hidden * sizeof(float));
  if (!f) return mlx_array_new();
  if (oc_dequant_row(tok_embd->ggml_type,
                     (const uint8_t*)tok_embd->data + tk * rowb, f, hidden) != 0) {
    free(f);
    return mlx_array_new();
  }
  if (scale != 1.0f)
    for (size_t i = 0; i < hidden; ++i) f[i] *= scale;
  int shape[1] = {(int)hidden};
  mlx_array a = mlx_array_new_data(f, shape, 1, MLX_FLOAT32);
  free(f);
  return a;
}

mlx_array mlx_matvec(mlx_stream s, MlxMat W, mlx_array x) {
  /* x [cols] -> [cols,1]; y = W[rows,cols] @ x[cols,1] -> [rows,1] -> [rows]. */
  mlx_array xc = mlx_array_new(), y = mlx_array_new(), yr = mlx_array_new();
  int cshape[2] = {W.cols, 1};
  MTRY(mlx_reshape(&xc, x, cshape, 2, s));
  MTRY(mlx_matmul(&y, W.a, xc, s));
  int rshape[1] = {W.rows};
  MTRY(mlx_reshape(&yr, y, rshape, 1, s));
  mlx_array_free(xc);
  mlx_array_free(y);
  return yr;
}

mlx_array mlx_rmsnorm(mlx_stream s, mlx_array x, mlx_array w, int n, float eps) {
  (void)n;
  mlx_array out = mlx_array_new();
  /* mlx_fast_rms_norm(out, x, weight, eps, stream). Gemma/llama bake +1 into the
   * stored weight, matching oc_rms_norm, so no +1 here. */
  MTRY(mlx_fast_rms_norm(&out, x, w, eps, s));
  return out;
}

mlx_array mlx_rope(mlx_stream s, mlx_array x, int heads, int head_dim,
                   int rope_len, size_t pos, float theta, int traditional,
                   mlx_array freqs) {
  (void)heads; (void)head_dim;
  mlx_array out = mlx_array_new();
  /* mlx_fast_rope(out, x, dims, traditional, base, scale, offset, freqs, s).
   * `dims` = rope_len (partial rotary; the tail passes through). `offset` = pos.
   * `freqs` non-empty overrides `base` with gemma's per-dim custom frequencies
   * (see README RISK #2). scale = 1.0. */
  MTRY(mlx_fast_rope(&out, x, rope_len, traditional, theta, 1.0f, (int)pos,
                     freqs, s));
  return out;
}

mlx_array mlx_kv_append(mlx_stream s, mlx_array cache, mlx_array row, int n_kv,
                        int dim) {
  /* row [n_kv,dim] -> [1,n_kv,1,dim]; concat onto cache [1,n_kv,seq,dim] axis 2. */
  mlx_array r4 = mlx_array_new();
  int rshape[4] = {1, n_kv, 1, dim};
  MTRY(mlx_reshape(&r4, row, rshape, 4, s));
  if (mlx_array_empty_p(cache)) return r4; /* first token: cache IS the row */
  mlx_array out = mlx_array_new();
  mlx_vector_array v = mlx_vector_array_new();
  mlx_vector_array_append_value(v, cache);
  mlx_vector_array_append_value(v, r4);
  int rc = mlx_concatenate(&out, v, /*axis=*/2, s);
  mlx_vector_array_free(v);
  mlx_array_free(r4);
  mlx_array_free(cache);
  if (rc != 0) return mlx_array_new();
  return out;
}

mlx_array mlx_attention(mlx_stream s, mlx_array q, mlx_array k, mlx_array v,
                        int n_head, int hd, int vd, float scale, int window) {
  (void)n_head; (void)hd; (void)vd;
  mlx_array kw = k, vw = v; /* sliced views for SWA; NOT freed (borrowed) */
  mlx_array ks = mlx_array_new(), vs = mlx_array_new();
  if (window > 0) {
    /* attend only the last `window` positions: slice axis 2 to [seq-window, seq).
     * mlx_slice(res, a, start[], stop[], strides[], ndim, s); negative-index
     * semantics assumed like numpy — a validator must confirm. Here we pass
     * start along axis 2 = -window (last window rows), stop = end. */
    int start[4] = {0, 0, -window, 0};
    int stop[4]  = {0, 0, 0, 0};      /* 0 == to-end per axis (ASSUMED) */
    int strd[4]  = {1, 1, 1, 1};
    if (mlx_slice(&ks, k, start, stop, strd, 4, s) == 0) kw = ks;
    if (mlx_slice(&vs, v, start, stop, strd, 4, s) == 0) vw = vs;
  }
  mlx_array out = mlx_array_new(), out2 = mlx_array_new();
  /* mlx_fast_scaled_dot_product_attention(out, q, k, v, scale, mask, s). q is
   * [1,n_head,1,hd]; GQA broadcast handled by MLX when n_head != n_kv. mask
   * empty: a single decode query legitimately sees every cached key. */
  int rc = mlx_fast_scaled_dot_product_attention(&out, q, kw, vw, scale,
                                                  mlx_array_new(), s);
  mlx_array_free(ks);
  mlx_array_free(vs);
  if (rc != 0) { mlx_array_free(out); return mlx_array_new(); }
  /* out [1,n_head,1,vd] -> [n_head,vd] for the O projection matvec loop. */
  int rshape[2] = {n_head, vd};
  if (mlx_reshape(&out2, out, rshape, 2, s) != 0) { mlx_array_free(out); return mlx_array_new(); }
  mlx_array_free(out);
  return out2;
}

mlx_array mlx_ew_add(mlx_stream s, mlx_array a, mlx_array b) {
  mlx_array out = mlx_array_new();
  MTRY(mlx_add(&out, a, b, s));
  return out;
}

mlx_array mlx_ew_scale(mlx_stream s, mlx_array a, float k) {
  mlx_array out = mlx_array_new(), sc = mlx_array_new_float(k);
  int rc = mlx_multiply(&out, a, sc, s);
  mlx_array_free(sc);
  if (rc != 0) return mlx_array_new();
  return out;
}

mlx_array mlx_geglu(mlx_stream s, mlx_array gate, mlx_array up) {
  /* gelu_tanh(gate) * up, matching k_geglu (0.5 g (1+tanh(K(g+0.044715 g^3)))).
   * MLX ships mlx_gelu_approx (tanh approximation) == exactly this constant K. */
  mlx_array g = mlx_array_new(), out = mlx_array_new();
  MTRY(mlx_gelu_approx(&g, gate, s));
  int rc = mlx_multiply(&out, g, up, s);
  mlx_array_free(g);
  if (rc != 0) return mlx_array_new();
  return out;
}

mlx_array mlx_silu_mul(mlx_stream s, mlx_array gate, mlx_array up) {
  mlx_array g = mlx_array_new(), out = mlx_array_new();
  MTRY(mlx_silu(&g, gate, s)); /* x*sigmoid(x) */
  int rc = mlx_multiply(&out, g, up, s);
  mlx_array_free(g);
  if (rc != 0) return mlx_array_new();
  return out;
}

mlx_array mlx_softcap(mlx_stream s, mlx_array l, float cap) {
  /* cap * tanh(l / cap). */
  mlx_array d = mlx_ew_scale(s, l, 1.0f / cap);
  mlx_array t = mlx_array_new();
  if (mlx_tanh(&t, d, s) != 0) { mlx_array_free(d); return mlx_array_new(); }
  mlx_array_free(d);
  mlx_array out = mlx_ew_scale(s, t, cap);
  mlx_array_free(t);
  return out;
}

mlx_array mlx_resid_out(mlx_stream s, mlx_array ffn, mlx_array attn, float k) {
  mlx_array sum = mlx_ew_add(s, ffn, attn);
  mlx_array out = mlx_ew_scale(s, sum, k);
  mlx_array_free(sum);
  return out;
}

mlx_array mlx_reshape_to(mlx_stream s, mlx_array a, const int* shape, int ndim) {
  mlx_array out = mlx_array_new();
  MTRY(mlx_reshape(&out, a, shape, ndim, s));
  return out;
}

mlx_array mlx_row(mlx_stream s, MlxMat m, int r) {
  /* slice [r:r+1, :] then flatten to [cols]. */
  mlx_array sl = mlx_array_new(), out = mlx_array_new();
  int start[2] = {r, 0};
  int stop[2] = {r + 1, m.cols};
  int strd[2] = {1, 1};
  if (mlx_slice(&sl, m.a, start, stop, strd, 2, s) != 0) return mlx_array_new();
  int shape[1] = {m.cols};
  int rc = mlx_reshape(&out, sl, shape, 1, s);
  mlx_array_free(sl);
  if (rc != 0) return mlx_array_new();
  return out;
}

int mlx_eval_to_host(mlx_stream s, mlx_array a, float* out, int n) {
  (void)s;
  if (mlx_array_empty_p(a)) return -1;
  if (mlx_array_eval(a) != 0) return -1; /* THE per-token sync */
  const float* p = mlx_array_data_float32(a);
  if (!p) return -1;
  memcpy(out, p, (size_t)n * sizeof(float));
  return 0;
}

int mlx_argmax_to_host(mlx_stream s, mlx_array a, int n, int32_t* out) {
  mlx_array idx = mlx_array_new();
  if (mlx_argmax(&idx, a, /*axis=*/0, /*keepdims=*/false, s) != 0) return -1;
  if (mlx_array_eval(idx) != 0) { mlx_array_free(idx); return -1; }
  const int32_t* p = mlx_array_data_int32(idx);
  if (!p) { mlx_array_free(idx); return -1; }
  *out = *p;
  (void)n;
  mlx_array_free(idx);
  return 0;
}

void mlx_release(mlx_array a) {
  if (!mlx_array_empty_p(a)) mlx_array_free(a);
}
