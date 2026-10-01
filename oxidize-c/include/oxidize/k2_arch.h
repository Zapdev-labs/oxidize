/*
 * k2_arch.h — K2-Horizon (GGUF arch "k2-horizon") building blocks.
 *
 * K2-Horizon-MoVA is a GQA transformer with four things the generic
 * Llama-family path in llama.c does not otherwise do:
 *
 *   1. Grouped RMSNorm. Every norm (attn_norm, ffn_norm, output_norm) splits
 *      the hidden vector into `groups` contiguous slices, RMS-normalizes each
 *      slice on its own, then multiplies by the full-width weight.
 *   2. MoVA routed values. On the MoE layers V is not one projection but a
 *      mixture of value experts:
 *          p   = sigmoid(W_router · x)               (no bias in the logits)
 *          sel = top-k(p + bias)                     (bias steers selection)
 *          w   = p[sel] / max(sum p[sel], 6.1035e-5) * scale
 *          V   = sum_e w_e * SiLU(W_v[e] · x)
 *   3. A per-element softplus(beta = ln 2) attention-output gate computed
 *      from the pre-attention normed input and applied before W_o:
 *          out *= log(1 + exp(ln2 * g)) / ln2
 *   4. NEOX RoPE over the full head at base 1e7. Angles are evaluated in
 *      double once per position so they stay accurate at 262k positions.
 *
 * The FFN side (sigmoid router + selection bias + weight norm + x2.5 +
 * ungated shared expert, leading dense blocks) is served by the existing MoE
 * path in llama.c.
 *
 * The forward pass itself lives in llama.c (it owns the session, KV cache
 * and prefill machinery); these are the arch-specific pieces it calls.
 */
#ifndef OXIDIZE_K2_ARCH_H
#define OXIDIZE_K2_ARCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/llama.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Minimum denominator when normalizing selected expert weights (llama.cpp
 * clamps the weight sum to 6.1035e-5, the smallest normal f16). */
#define OC_K2_WEIGHT_SUM_MIN 6.103515625e-5f

/* Upper bound on attention.value_expert_used_count (K2 ships 4). Keeps the
 * decode path's per-token selection arrays on the stack at a fixed size. */
#define OC_K2_MAX_VALUE_USED 64u

/* True for "k2-horizon" / "k2_horizon". */
bool oc_k2_arch_match(const char *arch_str);

/* Peek at a GGUF file's general.architecture without mapping it. Reads only
 * the header. Returns true when the file exists and is a K2-Horizon GGUF. */
bool oc_k2_gguf_path_is_k2(const char *path);

/* Fill the K2-specific config fields from metadata. `prefix` is
 * "<arch>." (e.g. "k2-horizon."). */
OcError oc_k2_parse_config(const OcGgufFile *f, const char *prefix,
                           OcLlamaConfig *cfg);

/* Grouped RMSNorm over n = groups * (n / groups) values. groups <= 1, or a
 * group count that does not divide n, falls back to a plain RMSNorm.
 * `out` must not alias `x`. */
void oc_k2_grouped_rms_norm(const float *x, const float *weight, float *out,
                            size_t n, uint32_t groups, float eps);

/* out[i] *= softplus_{beta=ln2}(gate[i]) = log2(1 + 2^gate[i]). Switches to
 * the identity when ln2*g > 20, matching torch.nn.functional.softplus. */
void oc_k2_softplus_gate_apply(float *out, const float *gate, size_t n);

/* Scalar form of the gate function, for tests and references. */
float oc_k2_softplus_ln2(float g);

/* cos/sin of pos * theta^(-2j/rope_dim) for j < rope_dim/2, evaluated in
 * double and stored as float. */
void oc_k2_rope_table(int64_t pos, float theta, uint32_t rope_dim,
                      float *cos_out, float *sin_out);

/* NEOX (split-half) rotation of x[0..rope_dim) using a table from
 * oc_k2_rope_table(); dims past rope_dim are untouched. */
void oc_k2_rope_apply(float *x, uint32_t rope_dim, const float *cos_t,
                      const float *sin_t);

/* Sigmoid-routed top-k selection shared by MoVA.
 *   probs  in: router logits (n); out: sigmoid(logits)
 *   bias   selection-only bias (n), may be NULL
 *   sel    out: k chosen indices, highest biased score first
 *   w      out: k applied weights (unbiased probs, optionally normalized with
 *          the OC_K2_WEIGHT_SUM_MIN clamp, times `scale`)
 *   idx    scratch of n uint32
 * Ties go to the lower index. */
void oc_k2_route(float *probs, const float *bias, uint32_t n, uint32_t k,
                 bool normalize, float scale, uint32_t *sel, float *w,
                 uint32_t *idx);

/* Decode-time MoVA value path for one token: writes V (kv_row floats).
 * Scratch: probs/idx of value_expert_count, w of value_expert_used,
 * out_all of value_expert_used * kv_row, temp of n_embd. The selected
 * experts' matvecs run as one fused parallel region when their weights are
 * quantized. */
void oc_k2_mova_value(const OcLlamaConfig *c, const OcLlamaLayer *L,
                      const float *normed, float *v_out, float *probs,
                      uint32_t *idx, float *w, float *out_all, float *temp);

/* Batched (prefill) MoVA scratch. */
typedef struct OcK2MovaBatch {
    size_t    cap;         /* tokens per chunk                          */
    uint32_t  n_exp;       /* value_expert_count                        */
    uint32_t  k;           /* value_expert_used                         */
    size_t    n_embd;
    size_t    kv_row;
    float    *probs;       /* [cap][n_exp]                               */
    uint32_t *idx;         /* [n_exp] selection scratch                  */
    uint32_t *sel;         /* [cap][k]                                   */
    float    *w;           /* [cap][k]                                   */
    uint32_t *ex_tok;      /* [cap*k] token ids grouped by expert        */
    float    *ex_w;        /* [cap*k] matching weights                   */
    uint32_t *ex_off;      /* [n_exp+1]                                  */
    uint32_t *ex_fill;     /* [n_exp]                                    */
    float    *gath;        /* [cap][n_embd] gathered inputs              */
    float    *eout;        /* [cap][kv_row] expert outputs               */
} OcK2MovaBatch;

OcError oc_k2_mova_batch_init(OcK2MovaBatch *mb, const OcLlamaConfig *c,
                              size_t cap);
void oc_k2_mova_batch_free(OcK2MovaBatch *mb);

/* MoVA for n tokens: normed is [n][in_stride], v_out is [n][out_stride]
 * (kv_row values written per token). Each routed expert runs one batched
 * matmul over all of its tokens. `temp` holds n_embd floats; act/act_bytes is
 * the oc_matvec_quantized_batch activation scratch (may be NULL/0). */
void oc_k2_mova_value_batch(OcK2MovaBatch *mb, const OcLlamaConfig *c,
                            const OcLlamaLayer *L, const float *normed,
                            size_t in_stride, float *v_out, size_t out_stride,
                            size_t n, float *temp, uint8_t *act,
                            size_t act_bytes);

/* Load-time checks for a K2 model: dense layers need attn_v, MoVA layers
 * need a coherent attn_v_exps / attn_v_gate / bias triple. */
OcError oc_k2_validate_layers(const OcLlamaModel *m);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_K2_ARCH_H */
