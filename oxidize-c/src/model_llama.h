/* Generic dense decoder-only transformer (arch keys "llama", "mistral",
 * "qwen2", "qwen3", "yi", "phi3", ...). The standard Llama-family stack:
 * input RMSNorm -> QKV projections (optional q/k/v/o bias; optional per-head
 * q/k RMSNorm) -> RoPE (NeoX split-half for qwen2/qwen3/phi3; ggml NORMAL
 * adjacent-pair for llama/mistral/yi, whose q/k llama.cpp permutes for it) ->
 * GQA/MQA causal attention -> O projection -> residual -> post-attn RMSNorm ->
 * FFN -> residual; final RMSNorm -> output head (tied to token_embd, or an
 * untied output.weight). The FFN is either the dense SwiGLU (gate/up/down, SiLU)
 * or, when the layer carries a router + expert-stacked tensors, a Mixture-of-
 * Experts block: softmax router, top-k experts, weighted SwiGLU sum, plus an
 * optional always-on shared expert (Mixtral / Qwen2-3-MoE / DeepSeek / OLMoE /
 * gpt-oss). Reference: oxidize-core/src/model/inference/moe.rs.
 *
 * Geometry is read from GGUF KVs under the model's OWN arch prefix and from the
 * tensor shapes (shapes win, like gemma4). Dispatch is by which tensors a layer
 * carries — a bias is applied iff attn_q.bias exists, a q-norm iff
 * attn_q_norm.weight exists — not by an arch enum.
 * Reference: oxidize-core/src/model/inference/forward.rs + layers.rs. */
#ifndef OC_MODEL_LLAMA_H
#define OC_MODEL_LLAMA_H

#include <stdbool.h>

#include "gguf.h"
#include "tensor.h"

typedef struct {
  /* dequantized f32 vectors */
  float *attn_norm, *ffn_norm;
  float *attn_q_norm, *attn_k_norm;         /* optional (Qwen3); NULL if absent */
  float *bias_q, *bias_k, *bias_v, *bias_o; /* optional (Qwen2); NULL if absent */
  /* quantized weight views (mmap) */
  const GgufTensorInfo *attn_q, *attn_k, *attn_v, *attn_out;
  const GgufTensorInfo *ffn_gate, *ffn_up, *ffn_down; /* dense SwiGLU; NULL when is_moe */
  /* Mixture-of-Experts FFN (Mixtral/Qwen2-3-MoE/DeepSeek/OLMoE/gpt-oss). A layer
   * is MoE when the router (ffn_gate_inp) and all three expert-stacked tensors
   * are present; then the dense ffn_* above are NULL. Detection is per layer, so
   * a model with leading dense blocks then MoE (DeepSeek) loads correctly.
   * gate/up_exps are 3D [n_expert][expert_inter][hidden]; down_exps is
   * [n_expert][hidden][expert_inter] — expert e's matrix starts at
   * e*rows*oc_row_bytes(type, cols) in the mmap. */
  bool is_moe;
  const GgufTensorInfo *ffn_gate_inp;                                /* router [n_expert, hidden] */
  const GgufTensorInfo *ffn_gate_exps, *ffn_up_exps, *ffn_down_exps; /* 3D expert stacks */
  const GgufTensorInfo *ffn_gate_shexp, *ffn_up_shexp, *ffn_down_shexp; /* shared expert (opt) */
  float* ffn_gate_inp_shexp; /* opt Qwen2-MoE shared-expert sigmoid gate [hidden]; NULL if absent */
  /* KV cache: ctx positions, row = n_kv_heads*head_dim VALUES, stored at
   * oc_kv_elem_bytes(kv_type) bytes each in a byte buffer (f32/f16/q8). Q8
   * carries one scale per (position, kv-head) in k_scale/v_scale (NULL for
   * f32/f16). Full causal attention: slots are unique (no ring) and K/V may be
   * committed before attention. */
  uint8_t *k_cache, *v_cache;
  float *k_scale, *v_scale;
} LlamaLayer;

typedef struct {
  GgufFile g; /* owned */
  size_t hidden, n_layers, n_head, n_kv_heads, head_dim, inter, vocab, ctx, rope_dim;
  float eps, rope_theta;
  bool rope_norm; /* true => ggml NORMAL (adjacent-pair) rope, for llama/mistral/
                     yi whose llama.cpp GGUFs ship q/k permuted for it; false =>
                     NeoX split-half (qwen2/qwen3/phi3, unpermuted q/k). */
  const GgufTensorInfo* tok_embd;
  const GgufTensorInfo* out_w; /* == tok_embd when tied */
  float* out_norm;
  /* MoE config (0 / false on a dense model). n_experts_used in [1, n_experts];
   * expert_inter is the experts' SwiGLU width (may differ from `inter`);
   * shexp_inter the shared expert's. norm_topk_prob renormalizes the selected
   * experts' weights; expert_weights_scale is DeepSeek's routed scaling. */
  bool has_moe, norm_topk_prob;
  size_t n_experts, n_experts_used, n_experts_shared, expert_inter, shexp_inter;
  float expert_weights_scale;
  /* MoE scratch: one token at a time, shared by decode and the batched-prefill
   * token loop (the pool serializes forward passes anyway). */
  float *me_logits, *me_w, *me_gate, *me_up, *me_eout;
  int* me_sel;
  OcKvType kv_type; /* KV cache element precision (from oc_kv_get_type at load) */
  size_t kv_len;    /* highest cached position + 1; advanced by forward, moved
                       back by llama_kv_rewind */
  LlamaLayer* layers;
  float* x;      /* [hidden] residual stream */
  float* logits; /* [vocab] */
  /* scratch (one forward pass at a time) */
  float *normed, *q, *k, *v, *attn_res, *attn_proj, *gate, *up, *ffn_out, *head_tmp;
  /* batched-prefill state: the same scratch, one row per token (see gemma4.h). */
  OcCtx* octx;
  size_t batch;     /* prefill chunk size; may be lowered by a caller */
  size_t batch_cap; /* rows actually allocated at load; chunks clamp to this */
  float *bx, *bnormed, *bq, *bk, *bv, *battn, *bproj, *bgate, *bup, *bffn;
} LlamaModel;

/* Takes ownership of *g on success (0). On failure writes err and leaves *g for
 * the caller to close. max_ctx caps the KV cache (0 = model context). Rejects a
 * model whose standard llama tensors are missing or unsupported (loud, never a
 * silent wrong number). */
int llama_load(LlamaModel* m, GgufFile* g, size_t max_ctx, char* err, size_t errlen);
void llama_free(LlamaModel* m);

/* One decode step. Returns m->logits when need_logits, else NULL. */
float* llama_forward(LlamaModel* m, int32_t token, size_t pos, bool need_logits);

/* llama_forward's tail: run layers [l0, n_layers) with m->x ALREADY holding the
 * residual stream for `pos` (no embedding lookup), then the final norm and
 * tied/untied logits. l0 == 0 is exactly what llama_forward runs after the
 * embedding lookup. Exists for the CUDA backend's -ngl partial offload: the GPU
 * runs layers [0, l0) and hands the residual stream back, the CPU finishes here.
 * Only the CPU layers [l0, n) touch the CPU KV cache. */
float* llama_forward_from(LlamaModel* m, size_t pos, size_t l0, bool need_logits);

/* Drop cached KV for positions >= pos (speculative decoding / chat editing).
 * The full causal cache is positionally addressed, so this is bookkeeping: the
 * next forward at `pos` overwrites its own slot and attention only ever reads
 * [0, pos]. Sets kv_len = pos (clamped to the current length; rewinding forward
 * is a no-op). Regenerating the dropped positions with the same tokens
 * reproduces the original logits bit-for-bit. */
void llama_kv_rewind(LlamaModel* m, size_t pos);

/* MoE router: softmax(logits[0..n]) into probs[n] (caller scratch, may alias
 * logits), pick the top-k by probability into sel[0..k] (descending), and write
 * each selected expert's routing weight into w[0..k] — renormalized over the k
 * when norm_topk, then multiplied by scale. Pure; the one routing path both
 * llama_moe_ffn and the router-correctness test drive. k must be <= n. */
void llama_moe_route(const float* logits, size_t n, size_t k, bool norm_topk,
                     float scale, float* probs, int* sel, float* w);

/* MoE FFN for one token — the routed + optional shared expert block the forward
 * passes run per token: router over ffn_gate_inp -> softmax top-k experts ->
 * weighted SwiGLU sum over the 3D expert stacks, plus an optional always-on
 * (optionally sigmoid-gated) shared expert, written into out[hidden]. x is the
 * post-ffn_norm hidden vector. Exposed so the routing, the top-k weighting and
 * the per-expert 3D stride can be checked against an independent reference: the
 * batched==sequential test cannot see a routing/stride bug because BOTH forward
 * paths call this same function, so a shared error cancels. */
void llama_moe_ffn(LlamaModel* m, const LlamaLayer* L, const float* x, float* out);

/* Prefill: n_tokens at positions pos0.. in ONE pass, numerically equivalent to
 * that many sequential llama_forward calls (the batched==sequential test is the
 * acceptance criterion). Splits internally at m->batch. need_logits returns the
 * LAST token's logits; NULL otherwise or on error. */
float* llama_forward_batch(LlamaModel* m, const int32_t* tokens, size_t n_tokens,
                           size_t pos0, bool need_logits);

#endif
