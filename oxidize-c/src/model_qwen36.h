/* Qwen3.5/3.6 hybrid forward pass (arch key "qwen35"). 3-of-4 layers are
 * gated-DeltaNet linear attention (conv4 + recurrent state, no KV growth);
 * every 4th layer is gated full attention (GQA, per-head q/k RMSNorm, partial
 * NeoX RoPE). SiLU FFN, untied output head. The trailing NextN/MTP block is
 * loaded and USED as a speculative-decode draft head (see qwen36_mtp_draft /
 * qwen36_spec_step below); it is skipped only for plain single-token decode.
 * Reference: llama.cpp src/models/qwen35.cpp + delta-net-base.cpp;
 * oxidize-core/src/model/inference/mtp.rs for the native-nextn draft. */
#ifndef OC_MODEL_QWEN36_H
#define OC_MODEL_QWEN36_H

#include <stdbool.h>

#include "gguf.h"
#include "tensor.h"

typedef struct {
  bool is_linear;
  /* shared */
  float *attn_norm, *post_attn_norm;
  const GgufTensorInfo *ffn_gate, *ffn_up, *ffn_down;
  /* full attention */
  const GgufTensorInfo *attn_q, *attn_k, *attn_v, *attn_out;
  float *attn_q_norm, *attn_k_norm;
  float *k_cache, *v_cache; /* [ctx * n_kv_heads * head_dim] */
  /* linear attention (gated delta net) */
  const GgufTensorInfo *wqkv, *wgate, *ssm_out;
  const GgufTensorInfo *beta_w, *alpha_w; /* [hidden -> n_v_heads] */
  float *dt_bias, *ssm_a;                 /* [n_v_heads] */
  float *conv_w;                          /* [conv_dim][d_conv] f32 */
  float *ssm_norm;                        /* [head_v_dim] */
  float *conv_state;                      /* [(d_conv-1) * conv_dim] */
  float *S;                               /* [n_v_heads][head_v_dim][head_k_dim] */
  /* Speculative-decode snapshots of the two recurrent tensors, allocated only
   * when the model carries an MTP head. The DeltaNet state has no per-position
   * history, so a speculative step saves it here before the throwaway draft-
   * verification batch and restores it before committing the accepted tokens. */
  float *conv_snap, *S_snap;
} Qwen36Layer;

/* Native NextN / MTP (multi-token-prediction) draft block. One gated full-
 * attention transformer layer (same shape as the qwen35 full layers) plus the
 * enorm/hnorm/eh_proj fusion and an optional own embedding + output head. It
 * proposes the next token from (last committed token, last committed hidden)
 * cheaply; the target then verifies. GGUF tensors: blk.N.nextn.* (fusion/head)
 * and blk.N.{attn_*,ffn_*} for N == n_layers. */
typedef struct {
  /* fusion + head */
  const GgufTensorInfo *eh_proj;          /* [2*hidden -> hidden] */
  float *enorm, *hnorm;                   /* [hidden] */
  const GgufTensorInfo *embed_tokens;     /* [vocab,hidden] or NULL -> tok_embd */
  float *shared_head_norm;                /* [hidden] or NULL -> out_norm */
  const GgufTensorInfo *shared_head_head; /* [vocab,hidden] or NULL -> out_w */
  /* one gated full-attention layer */
  float *attn_norm, *post_attn_norm, *attn_q_norm, *attn_k_norm;
  const GgufTensorInfo *attn_q, *attn_k, *attn_v, *attn_out;
  const GgufTensorInfo *ffn_gate, *ffn_up, *ffn_down;
  float *k_cache, *v_cache;               /* [cap * n_kv_heads * head_dim] mini-KV */
  size_t cap;                             /* draft KV slots (== batch_cap) */
} Qwen36Mtp;

typedef struct {
  GgufFile g; /* owned */
  size_t hidden, n_layers, inter, vocab, ctx;
  /* full attention geometry */
  size_t n_head, n_kv_heads, head_dim, rope_dim;
  float rope_theta;
  /* linear attention geometry */
  size_t d_conv, d_state, n_k_heads, n_v_heads, head_v_dim, conv_dim, key_dim, value_dim;
  float eps;
  const GgufTensorInfo *tok_embd, *out_w;
  float* out_norm;
  Qwen36Layer* layers;
  float* x;      /* [hidden] residual stream */
  float* logits; /* [vocab] */
  /* scratch */
  float *normed, *q, *k, *v, *attn_res, *attn_proj, *gate, *up, *ffn_out;
  float* qpack; /* [n_head * head_dim] q with the interleaved gates stripped */
  float *qkv, *z, *conv_out, *o_lin; /* linear-attention scratch */
  /* batched-prefill state: the same scratch, one row per token. */
  OcCtx* octx;
  size_t batch;     /* prefill chunk size; may be lowered by a caller */
  size_t batch_cap; /* rows actually allocated at load; chunks clamp to this */
  float *bx, *bnormed, *bq, *bqpack, *bk, *bv, *battn, *bproj;
  float *bgate, *bup, *bffn;
  float *bqkv, *bz, *bconv, *bolin, *bbeta, *bgdec;
  /* NextN/MTP draft head (P16). has_mtp is false when the GGUF ships no usable
   * nextn block, in which case the spec path is unavailable and callers fall
   * back to plain decode. */
  bool has_mtp;
  Qwen36Mtp mtp;
  float *mtp_concat; /* [2*hidden] enorm|hnorm fusion input */
  float *mtp_prev;   /* [hidden] chained draft hidden across a burst */
  float *mtp_logits; /* [vocab] draft head output (kept off m->logits) */
  float *spec_logits; /* [spec_cap * vocab] per-row verification logits */
  size_t spec_cap;    /* rows spec_logits can hold */
} Qwen36Model;

int qwen36_load(Qwen36Model* m, GgufFile* g, size_t max_ctx, char* err, size_t errlen);
void qwen36_free(Qwen36Model* m);
float* qwen36_forward(Qwen36Model* m, int32_t token, size_t pos, bool need_logits);

/* Prefill: n_tokens at positions pos0.. in one pass. Numerically equivalent to
 * that many sequential qwen36_forward calls (the test asserts it). The
 * projections and the FFN batch into GEMMs; the gated-DeltaNet conv and delta
 * state are recurrent and are SCANNED over the batch (threaded across channels
 * / v-heads, which are independent — the token axis is not). */
float* qwen36_forward_batch(Qwen36Model* m, const int32_t* tokens, size_t n_tokens,
                            size_t pos0, bool need_logits);

/* ---- Speculative decoding with the native MTP/nextn draft head (P16) --------
 *
 * The loop, per step (matches oxidize-core/src/model/generation.rs run_mtp_step):
 *   draft[0..k)   = qwen36_mtp_draft(m, last_committed_token, k);
 *   len           = qwen36_spec_step(m, draft, k, pos, out, &accepted);
 *   // emit out[0..len); pos += len; last_committed_token = out[len-1];
 * On entry to each step m->logits must hold the target distribution for the
 * next position (left there by the previous forward/commit) and m->normed the
 * matching out-norm hidden — both are produced by qwen36_forward_batch(.,true).
 *
 * qwen36_spec_step is numerically EXACTLY equivalent, for greedy (argmax)
 * acceptance, to feeding the emitted tokens through qwen36_forward one at a
 * time: every emitted token is the target's own argmax at a position the target
 * actually forwarded with the correct prefix. */

/* Draft k tokens from the MTP head. seed_token is the last committed token; the
 * seed hidden is read from m->normed (the target's out-norm hidden at that
 * position). Writes k ids to draft[0..k). k should be <= m->batch_cap-1; a larger
 * k is clamped to the draft cache (batch_cap rows) and the surplus draft[] slots
 * are set to -1. No-op (draft filled with -1) when !m->has_mtp. Uses only
 * MTP-private scratch, so it
 * leaves m->logits / m->normed untouched for the verify step. */
void qwen36_mtp_draft(Qwen36Model* m, int32_t seed_token, int32_t* draft, size_t k);

/* One greedy speculative verify+commit step. Given k proposed tokens and the
 * pending target distribution in m->logits, snapshots the DeltaNet state, runs
 * ONE batched target forward over the drafts, accepts the longest argmax-
 * matching prefix, restores the state and re-forwards only the committed tokens
 * (accepted prefix + one target correction/continuation). Writes the emitted
 * ids to out[0..], which must hold k+1; returns the count (1..k+1). *accepted
 * (may be NULL) receives how many DRAFT tokens matched (0..k). Afterwards
 * m->logits/m->normed describe the position after the last emitted token, so
 * the caller advances pos by the return value and continues. */
size_t qwen36_spec_step(Qwen36Model* m, const int32_t* draft, size_t k, size_t pos,
                        int32_t* out, size_t* accepted);

/* Snapshot / restore the two recurrent DeltaNet tensors (conv window + state S)
 * of every linear layer. qwen36_spec_step calls these around its verification
 * batch; exposed for direct testing of the rollback (the full-attention KV is
 * positional and needs no snapshot — the commit re-forward overwrites it). */
void qwen36_state_snapshot(Qwen36Model* m);
void qwen36_state_restore(Qwen36Model* m);

#endif
