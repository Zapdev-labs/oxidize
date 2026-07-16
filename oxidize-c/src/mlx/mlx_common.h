/* ============================================================================
 * UNVERIFIED — NEVER COMPILED OR RUN. Written blind against src/cuda/ (the
 * verified resident-forward reference) and oxidize-core/src/backends/mlx.rs.
 * Requires a Mac with Apple Silicon + mlx-c. MAY NOT COMPILE.
 * ----------------------------------------------------------------------------
 * Internal seam shared by gemma4_mlx.c and llama_mlx.c. EVERY mlx-c call in the
 * whole backend is funnelled through the small wrappers declared here and
 * defined in mlx_common.c, so a validator who finds an mlx-c signature is wrong
 * fixes it in ONE place instead of across two 500-line forward passes. The
 * wrappers take/return `mlx_array` by value (mlx-c's ref-counted handle) exactly
 * like the Rust `Array` in mlx.rs.
 *
 * The mlx-c symbols used (mlx_matmul, mlx_fast_rms_norm, mlx_fast_rope,
 * mlx_fast_scaled_dot_product_attention, mlx_add/mlx_multiply/mlx_tanh/…,
 * mlx_eval, mlx_array_new_data, mlx_concatenate, mlx_slice, mlx_reshape) are
 * ASSUMED from mlx_rs and the MLX C++ headers. Confirm against <mlx/c/*.h>.
 * ============================================================================ */
#ifndef OC_MLX_COMMON_H
#define OC_MLX_COMMON_H

#include <mlx/c/mlx.h> /* mlx_array, mlx_stream, mlx_eval, ... (ASSUMED path) */

#include <stddef.h>
#include <stdint.h>

#include "../gguf.h"
#include "../quant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A resident weight matrix in unified memory: [rows, cols] f32, dequantized on
 * the host from its GGUF bytes via oc_dequant_row(). Mirrors a device weight
 * blob in the CUDA backend, minus the fused on-device decode (see mlx_backend.h
 * "KEY DIFFERENCE"). */
typedef struct {
  mlx_array a; /* shape [rows, cols], f32 */
  int rows, cols;
} MlxMat;

/* The eight weight types the CUDA backend proves against the CPU forward. This
 * is the SAME gate as cuda check_type(); here it decides what oc_dequant_row()
 * is allowed to decode on load. Returns 0 ok, -1 with err set. */
int mlx_check_type(uint32_t ggml_type, const char* what, char* err, size_t errlen);

/* Host-dequantize a whole GGUF tensor (row by row, oc_dequant_row) to f32 and
 * upload it as a resident [rows, cols] mlx_array. cols = dims[0], rows = product
 * of the higher dims (same convention as cuda tensor_bytes()). On failure
 * returns a MlxMat with a==mlx_array_empty() and writes err. */
MlxMat mlx_upload_weight(const GgufTensorInfo* t, char* err, size_t errlen);

/* Upload an already-f32 host vector as a resident [n] array (norm weights,
 * biases, gemma rope_freqs, the all-ones scale-less-norm weight). */
mlx_array mlx_upload_vec(const float* v, int n);

/* One embedding row -> f32 [hidden], optionally * scale (gemma emb_scale; pass
 * 1.0f for llama). Decodes on the host from the still-quantized tok_embd row. */
mlx_array mlx_embed_row(const GgufTensorInfo* tok_embd, int32_t token,
                        size_t hidden, float scale, char* err, size_t errlen);

/* y[rows] = W[rows,cols] . x[cols]   (mlx_matmul with x as a column). */
mlx_array mlx_matvec(mlx_stream s, MlxMat W, mlx_array x);

/* RMSNorm over the last axis. `w` is the resident scale ([n]); for the gemma V
 * heads' scale-less norm pass an all-ones array (the model already carries
 * m->ones). +1 is baked into the GGUF norm weights, matching oc_rms_norm. */
mlx_array mlx_rmsnorm(mlx_stream s, mlx_array x, mlx_array w, int n, float eps);

/* RoPE on a [heads, 1, head_dim] q or k tensor at absolute position `pos`.
 *   traditional=true  -> ggml NORMAL adjacent-pair (llama/mistral/yi).
 *   traditional=false -> NeoX split-half (qwen2/qwen3/phi3, gemma).
 * rope_len dims are rotated; the tail passes through (partial rotary).
 * `freqs` (or mlx_array_empty()) supplies gemma's per-dim custom frequencies:
 * freqs_i = theta^(-2i/rope_len) / rope_freqs_i, applied on GLOBAL layers only.
 * When empty, MLX derives freqs from `theta`. See README RISK #2 — this is the
 * least-certain mapping in the backend. */
mlx_array mlx_rope(mlx_stream s, mlx_array x, int heads, int head_dim,
                   int rope_len, size_t pos, float theta, int traditional,
                   mlx_array freqs);

/* Append one token's k/v row to a growing cache along the sequence axis and
 * return the new cache. `cache` shape [1, n_kv, seq, dim]; `row` shape
 * [n_kv, dim] (reshaped internally to [1,n_kv,1,dim]). The old cache handle is
 * released. On the first call pass mlx_array_empty() as cache. */
mlx_array mlx_kv_append(mlx_stream s, mlx_array cache, mlx_array row, int n_kv,
                        int dim);

/* Single-query GQA attention via MLX fast SDPA. q [1,n_head,1,hd],
 * k [1,n_kv,seq,hd], v [1,n_kv,seq,vd]; returns [n_head, vd]. For a gemma SWA
 * layer pass window>0 to attend only the last `window` cached positions (the
 * cache is sliced on axis 2); pass 0 for full-causal. scale is m->attn_scale or
 * 1/sqrt(hd). No explicit mask: a single decode query attends every cached key. */
mlx_array mlx_attention(mlx_stream s, mlx_array q, mlx_array k, mlx_array v,
                        int n_head, int hd, int vd, float scale, int window);

/* Elementwise, all returning a fresh array (inputs released where noted in .c). */
mlx_array mlx_ew_add(mlx_stream s, mlx_array a, mlx_array b);        /* a+b */
mlx_array mlx_ew_scale(mlx_stream s, mlx_array a, float k);          /* a*k */
mlx_array mlx_geglu(mlx_stream s, mlx_array gate, mlx_array up);     /* gelu_tanh(gate)*up */
mlx_array mlx_silu_mul(mlx_stream s, mlx_array gate, mlx_array up);  /* silu(gate)*up */
mlx_array mlx_softcap(mlx_stream s, mlx_array l, float cap);         /* cap*tanh(l/cap) */
/* x = (ffn + attn) * s  — gemma's fused layer_output_scale residual. */
mlx_array mlx_resid_out(mlx_stream s, mlx_array ffn, mlx_array attn, float k);

/* Reshape without freeing `a` (returns a fresh view/handle). Lets the arch
 * forwards flip a flat [n_head*hd] projection to [n_head, hd] for per-head norm /
 * rope / attention without touching mlx-c themselves. */
mlx_array mlx_reshape_to(mlx_stream s, mlx_array a, const int* shape, int ndim);

/* Copy row `r` of a resident weight matrix to a fresh [cols] array. Used for the
 * gemma tied-embedding lookup from the same array that serves the output head. */
mlx_array mlx_row(mlx_stream s, MlxMat m, int r);

/* Force evaluation of `a` (the ONE per-token sync) and copy n floats to host. */
int mlx_eval_to_host(mlx_stream s, mlx_array a, float* out, int n);

/* argmax of a 1-D array to a host int (gemma greedy fast path; softcap-free is
 * fine, tanh is monotonic). */
int mlx_argmax_to_host(mlx_stream s, mlx_array a, int n, int32_t* out);

void mlx_release(mlx_array a); /* mlx_array_free wrapper, empty-safe */

#ifdef __cplusplus
}
#endif

#endif /* OC_MLX_COMMON_H */
