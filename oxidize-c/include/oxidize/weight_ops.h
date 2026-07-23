/*
 * weight_ops.h — Weight matrix operations for inference.
 *
 * Port of oxidize-core/src/model/inference.rs GEMV/GEMM functions:
 *   gemv_weight, gemv_expert_weight, gemv_weight_fused, gemm_weight,
 *   add_repeating_bias
 *
 * These operate on OcWeightStorage to perform matrix-vector products
 * against f32 or quantized weight matrices in GGUF natural row-major layout.
 */
#ifndef OXIDIZE_WEIGHT_OPS_H
#define OXIDIZE_WEIGHT_OPS_H

#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/weight_storage.h"
#include "oxidize/inference.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GEMV: y = W @ x, where W is [rows, cols] in natural row-major layout.
 * For F32 storage: direct dot product per row.
 * For Quantized/Mmap: dequantizes blocks on the fly.
 * input must have cols floats, output must have rows floats. */
OcError oc_gemv_weight(const OcWeightStorage *ws,
                       size_t rows, size_t cols,
                       const float *input, float *output);

/* GEMV for a single expert in a MoE weight matrix.
 * The expert's weights start at expert_idx * (rows * cols) into the storage. */
OcError oc_gemv_expert_weight(const OcWeightStorage *ws,
                               size_t expert_idx, size_t n_experts,
                               size_t rows, size_t cols,
                               const float *input, float *output);

/* Fused GEMV: run multiple projections (q/k/v, gate/up) sharing the same
 * input vector. Each part specifies (storage, rows, output_buffer).
 * n_parts is the number of entries. cols and input are shared. */
typedef struct {
    const OcWeightStorage *storage;
    size_t                  rows;
    float                  *output;
} OcGemvPart;

OcError oc_gemv_weight_fused(OcGemvPart *parts, size_t n_parts,
                              size_t cols, const float *input);

/* Batched GEMM: Y = W @ X, where W is [rows, cols] and X is [batch, cols].
 * output is [batch, rows] row-major. */
OcError oc_gemm_weight(const OcWeightStorage *ws,
                        size_t rows, size_t cols,
                        const float *inputs, float *outputs, size_t batch);

/* Add a per-row bias, repeating modulo bias_len when shorter than a row.
 * buf is [batch * row_len] or [len]. bias is [bias_len]. */
void oc_add_repeating_bias(float *buf, size_t buf_len,
                            const float *bias, size_t bias_len);

/* Simple f32 GEMV: y[j] = sum_i W[j*cols + i] * x[i] for j in 0..rows. */
OcError oc_gemv_f32(const float *weights, size_t rows, size_t cols,
                     const float *input, float *output);

/* MoE routing result: expert index + weight. */
typedef struct {
    size_t idx;
    float  weight;
} OcExpertScore;

/* MoE FFN forward: routes input through top-k experts and combines results.
 *
 * gate_inp:   router weight [n_experts, hidden]
 * gate_exps:  expert gate weights [n_experts, i_size, hidden]
 * up_exps:    expert up weights   [n_experts, i_size, hidden]
 * down_exps:  expert down weights  [n_experts, hidden, i_size]
 * exp_probs_b: per-layer expert bias (may be NULL, n_experts elements)
 * normed:     input vector [hidden]
 * ffn_out:    output vector [hidden] (zeroed before writing)
 * gate_scratch: [i_size] scratch
 * up_scratch:   [i_size] scratch
 * expert_out:   [hidden] scratch
 * router_logits: [n_experts] (filled with router logits/weights)
 * expert_scores: [n_experts] (filled with (idx, score) pairs)
 *
 * Uses softmax (Mixtral) or sigmoid (LFM2MoE) gating based on cfg.
 * Handles DeepSeek group-limited routing when expert_group_count > 1. */
OcError oc_moe_ffn_forward(const OcWeightStorage *gate_inp,
                             const OcWeightStorage *gate_exps,
                             const OcWeightStorage *up_exps,
                             const OcWeightStorage *down_exps,
                             const float *exp_probs_b,
                             const OcInferenceConfig *cfg,
                             const float *normed, float *ffn_out,
                             float *gate_scratch, float *up_scratch,
                             float *expert_out,
                             float *router_logits,
                             OcExpertScore *expert_scores);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_WEIGHT_OPS_H */
