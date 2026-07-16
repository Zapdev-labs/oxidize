/* DeepSeek-V2/V3 (arch "deepseek2") — Multi-head Latent Attention (MLA) + a
 * fine-grained, group-limited Mixture-of-Experts. This is a distinct attention
 * family from the dense llama path, so it lives in its own file.
 *
 * ---- MLA ------------------------------------------------------------------
 * Instead of storing a full K and V per head per position, MLA stores a small
 * COMPRESSED latent (kv_lora_rank values) plus one decoupled RoPE key
 * (qk_rope_head_dim values, shared across heads — MQA-style) per position, and
 * reconstructs the per-head K/V on the fly at attention time. That latent is
 * the whole point: the persistent KV cache is kv_lora_rank + qk_rope_head_dim
 * values per position per layer, versus n_head*(qk_head_dim + v_head_dim) for a
 * dense cache — a large saving (see the load-time log).
 *
 * Per position:
 *   c_q   = q_a_norm(q_a_proj(x))         (only if q_lora_rank > 0; else q = q_proj(x))
 *   q     = q_b_proj(c_q)                 -> [n_head, qk_head_dim]; split per head into
 *                                            q_nope[qk_nope_head_dim] | q_pe[qk_rope_head_dim]
 *   kv_a  = kv_a_proj_with_mqa(x)         -> [kv_lora_rank | qk_rope_head_dim]
 *   c_kv  = kv_a_norm(kv_a[:kv_lora])     (the cached latent)
 *   k_pe  = RoPE(kv_a[kv_lora:])          (the cached decoupled key; q_pe is RoPE'd too)
 * At attention time, per cached position t:
 *   kv    = kv_b_proj(c_kv[t])            -> [n_head, qk_nope_head_dim | v_head_dim]
 *   k[h]  = concat(kv.k_nope[h], k_pe[t]) -> qk_head_dim ; v[h] = kv.v[h] -> v_head_dim
 *   score = (q_nope[h].k_nope[h] + q_pe[h].k_pe[t]) * softmax_scale
 * softmax_scale = mscale^2 / sqrt(qk_head_dim); mscale is the YaRN log-multiplier
 * correction when the rope.scaling.yarn_log_multiplier KV is present, else 1.
 *
 * ---- RoPE mode: ADJACENT-PAIR, not split-half (the subtle one) -------------
 * q_pe/k_pe use oc_rope_normal (rotates p[2i],p[2i+1]), NOT oc_rope (rotates
 * p[i],p[i+half]). DeepSeek's HF apply_rotary_pos_emb de-interleaves before it
 * rotates:
 *     q = q.view(b,h,s,d//2,2).transpose(4,3).reshape(b,h,s,d)   # [evens|odds]
 *     q_embed = q*cos + rotate_half(q)*sin                       # split-half rope
 * Element 2i lands at index i and 2i+1 at index d/2+i, so the split-half rotation
 * pairs (q[2i], q[2i+1]) with angle_i — an adjacent-pair rotation on the layout
 * the weights are actually stored in. The de-interleave permutation is shared by
 * q_pe and k_pe, so it cancels inside the q.k dot product and never needs doing
 * here. That is exactly why llama.cpp lists LLM_ARCH_DEEPSEEK2 under
 * LLAMA_ROPE_TYPE_NORM. Getting this wrong costs nothing observable at load time
 * and yields fluent, confidently wrong text; batched==sequential cannot see it
 * either (both paths call mla_project_pos), so deepseek_mla_e2e_ref in
 * tests/test_model.c pins it with an independent adjacent-pair reference.
 * Reference: oxidize-core/src/model/inference/layers.rs deepseek_mla_layer (note:
 * that path caches DENSE per-head K/V and drops mscale, so it is a shape
 * reference only) and llama.cpp build_deepseek2 (the combined attn_kv_b
 * reconstruction path this file follows).
 *
 * ---- DeepSeek MoE (group-limited routing) ---------------------------------
 * n_expert routed experts, n_expert_used per token, plus n_expert_shared always-
 * on shared experts (a single merged SwiGLU). Routing (llama.cpp build_moe_ffn):
 *   prob      = expert_gating_func==sigmoid ? sigmoid(gate.x) : softmax(gate.x)
 *   selscore  = prob + exp_probs_b          (V3 bias correction; else prob)
 *   groups    : split selscore into n_group groups; group_score = sum of the
 *               top-2 selscore in the group; keep the top topk_group groups,
 *               mask the rest out.
 *   sel       = top-(n_expert_used) experts by (masked) selscore
 *   weight[j] = prob[sel[j]]                 (ORIGINAL prob, not the biased score)
 *   if norm_topk_prob && used>1: weight /= sum(weight); weight *= routed_scale
 * Leading dense layers (leading_dense_block_count) carry a plain SwiGLU instead
 * of the router+experts; that is detected per layer by tensor presence, exactly
 * like the llama path, so a wrong leading_dense count cannot mis-route a layer.
 * Reference: oxidize-core/src/model/inference/moe.rs + llama.cpp build_moe_ffn. */
#ifndef OC_MODEL_DEEPSEEK_H
#define OC_MODEL_DEEPSEEK_H

#include <stdbool.h>

#include "gguf.h"
#include "tensor.h"

typedef struct {
  /* dequantized f32 vectors */
  float *attn_norm, *ffn_norm;
  float *mla_q_a_norm;  /* [q_lora_rank]; NULL when q is not compressed (V2-Lite) */
  float *mla_kv_a_norm; /* [kv_lora_rank]; always present */
  /* quantized weight views (mmap) */
  const GgufTensorInfo* mla_q_a;      /* [hidden, q_lora]; NULL when uncompressed */
  const GgufTensorInfo* mla_q_b;      /* [q_lora, n_head*qk_head_dim] (or attn_q when uncompressed) */
  const GgufTensorInfo* mla_kv_a_mqa; /* [hidden, kv_lora + qk_rope] */
  const GgufTensorInfo* mla_kv_b;     /* [kv_lora, n_head*(qk_nope + v_head_dim)] */
  const GgufTensorInfo* attn_out;     /* [n_head*v_head_dim, hidden] */
  /* dense FFN (leading layers); NULL when is_moe */
  const GgufTensorInfo *ffn_gate, *ffn_up, *ffn_down;
  /* MoE FFN */
  bool is_moe;
  const GgufTensorInfo* ffn_gate_inp;                                /* router [hidden, n_expert] */
  float* ffn_exp_probs_b;                                            /* [n_expert] V3 bias; NULL if absent */
  const GgufTensorInfo *ffn_gate_exps, *ffn_up_exps, *ffn_down_exps; /* 3D expert stacks */
  const GgufTensorInfo *ffn_gate_shexp, *ffn_up_shexp, *ffn_down_shexp; /* shared expert (opt) */
  /* Latent KV cache: ctx positions, kept f32 (already the compressed form).
   * kv_lat = kv_lora_rank latent; k_pe = qk_rope_head_dim decoupled RoPE key. */
  float* kv_lat_cache; /* [ctx][kv_lora] */
  float* k_pe_cache;   /* [ctx][qk_rope] */
} DeepseekLayer;

typedef struct {
  GgufFile g; /* owned */
  size_t hidden, n_layers, n_head, inter, vocab, ctx;
  /* MLA geometry */
  size_t kv_lora, q_lora, qk_nope, qk_rope, qk_head_dim, v_head_dim;
  bool q_compressed; /* q_lora_rank > 0 */
  float eps, rope_theta, softmax_scale;
  const GgufTensorInfo* tok_embd;
  const GgufTensorInfo* out_w; /* == tok_embd when tied */
  float* out_norm;
  /* MoE config (0 / false when the model has no MoE layer). */
  bool has_moe, norm_topk_prob, gating_sigmoid;
  size_t n_experts, n_experts_used, n_experts_shared, expert_inter, shexp_inter;
  size_t n_group, topk_group, leading_dense;
  float routed_scale;
  /* MoE scratch (one token at a time; decode and the batched loop share it). */
  float *me_prob, *me_selscore, *me_grp, *me_w, *me_gate, *me_up, *me_eout;
  int *me_sel, *me_grpsel;
  size_t kv_len; /* highest cached position + 1 */
  DeepseekLayer* layers;
  float* x;      /* [hidden] residual */
  float* logits; /* [vocab] */
  /* scratch (one forward pass at a time) */
  float *normed, *c_q, *c_q_normed, *q, *kv_a, *c_kv, *k_pe, *attn_res, *attn_proj;
  float *gate, *up, *ffn_out;
  /* kv_b reconstruction over all cached positions, shared decode+batch:
   * [ctx][n_head*(qk_nope + v_head_dim)]. Transient (one layer at a time), so
   * it never becomes a per-layer full-KV cache — the persistent cache stays the
   * small latent. */
  float* kv_b_recon;
  OcCtx* octx;
  size_t batch, batch_cap;
  /* batched-prefill scratch: one row per token. (The per-position MLA down-
   * projection reuses the decode scratch m->kv_a/c_q — it is consumed into the
   * caches before the next token, so no per-token copy is needed.) */
  float *bx, *bnormed, *bcq, *bq, *battn, *bproj, *bgate, *bup, *bffn;
} DeepseekModel;

/* Takes ownership of *g on success (0). On failure writes err and leaves *g for
 * the caller to close. max_ctx caps the KV cache (0 = model context). Rejects a
 * model whose MLA/MoE tensors are missing or geometrically inconsistent (loud,
 * never a silent wrong number). */
int deepseek_load(DeepseekModel* m, GgufFile* g, size_t max_ctx, char* err, size_t errlen);
void deepseek_free(DeepseekModel* m);

/* One decode step. Returns m->logits when need_logits, else NULL. */
float* deepseek_forward(DeepseekModel* m, int32_t token, size_t pos, bool need_logits);

/* Prefill n_tokens at positions pos0.. in one pass, numerically equal to that
 * many sequential deepseek_forward calls (the batched==sequential test is the
 * acceptance bar). Splits internally at m->batch. need_logits returns the LAST
 * token's logits; NULL otherwise or on error. */
float* deepseek_forward_batch(DeepseekModel* m, const int32_t* tokens, size_t n_tokens,
                              size_t pos0, bool need_logits);

/* Drop cached latent for positions >= pos (speculative decoding / chat editing).
 * The latent cache is positionally addressed, so this is bookkeeping: sets
 * kv_len = pos (clamped). */
void deepseek_kv_rewind(DeepseekModel* m, size_t pos);

/* Per-session MLA caches (same sizing as deepseek_load): kv_lat_cache and
 * k_pe_cache per layer, plus kv_len. Session-owned; never freed by
 * deepseek_free. install swaps layer pointers onto kv (stashing the
 * previous/primary ones); release restores them and copies m->kv_len back. */
typedef struct DeepseekKv DeepseekKv;
DeepseekKv* deepseek_kv_new(const DeepseekModel* m);
void deepseek_kv_free(DeepseekKv* kv);
void deepseek_kv_clear(DeepseekKv* kv);
void deepseek_kv_install(DeepseekModel* m, DeepseekKv* kv);
void deepseek_kv_release(DeepseekModel* m, DeepseekKv* kv);

/* DeepSeek group-limited router — exposed so the routing can be checked against
 * an independent hand reference (the batched==sequential test cannot see a
 * routing bug because both forward paths call this same code). Given raw router
 * logits[n_expert], writes the selected expert indices into sel[0..used] and
 * their routing weights into w[0..used]. bias (len n_expert) is the V3
 * correction, NULL if absent. Scratch prob/selscore (len n_expert) and grp/
 * grpsel (len n_group) are caller-owned. Pure. */
void deepseek_moe_route(const DeepseekModel* m, const float* logits, const float* bias,
                        float* prob, float* selscore, float* grp, int* grpsel,
                        int* sel, float* w);

#endif
