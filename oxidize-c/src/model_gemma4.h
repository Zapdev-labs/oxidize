/* Gemma 4 forward pass (arch key "gemma4"). Interleaved SWA/global attention,
 * per-layer KV head counts and head dims, sandwich norms, GeGLU FFN, tied
 * embeddings, final logit softcapping. All geometry parsed from GGUF KVs and
 * tensor shapes (shapes win; geometry is logged at load). */
#ifndef OC_MODEL_GEMMA4_H
#define OC_MODEL_GEMMA4_H

#include <stdbool.h>

#include "gguf.h"
#include "tensor.h"

typedef struct {
  float theta;
  size_t rope_dim; /* dims rotated per head (0 = full head_dim) */
} Gemma4RopeConfig; /* isolated per layer kind so it is easy to fix */

typedef struct {
  bool is_swa;
  size_t n_kv_heads;
  size_t head_dim;   /* per-head q/k width (from tensor shapes) */
  size_t v_head_dim; /* per-head v width */
  Gemma4RopeConfig rope;
  float output_scale; /* blk.N.layer_output_scale.weight (scalar), default 1 */
  /* quantized weight views (mmap) */
  const GgufTensorInfo *attn_q, *attn_k, *attn_v, *attn_out;
  const GgufTensorInfo *ffn_gate, *ffn_up, *ffn_down;
  /* dequantized f32 norm vectors */
  float *attn_norm, *attn_q_norm, *attn_k_norm;
  float *post_attn_norm, *ffn_norm, *post_ffn_norm;
  /* KV cache: ring of cache_cap positions, row = n_kv_heads*head_dim (k) /
   * n_kv_heads*v_head_dim (v). f32 lives in k_cache/v_cache; every other
   * precision packs into the k_qcache/v_qcache byte buffers (kv_type picks the
   * codec): F16 at 2 B/value (no meta), Q8 at 1 B/value with one scale per
   * (slot, kv_head) in k_qmeta[r*2]/v_qmeta[r*2], or the Hadamard-rotated int4
   * rotoquant at 0.5 B/value with a scale+min pair in k_qmeta/v_qmeta. */
  float *k_cache, *v_cache;
  uint8_t *k_qcache, *v_qcache;
  float *k_qmeta, *v_qmeta;
  size_t cache_cap;
} Gemma4Layer;

typedef struct {
  GgufFile g; /* owned */
  size_t hidden, n_layers, n_head, inter, vocab, ctx, window;
  float eps, final_softcap, emb_scale, attn_scale; /* attn_scale 0 => 1/sqrt(head_dim) */
  OcKvType kv_type; /* KV cache precision (f32/f16/q8/q4); see Gemma4Layer */
  bool kv_quant;    /* == (kv_type == OC_KV_Q4): rotated int4 rotoquant active.
                       Kept as a bool because the CUDA backend gates on it. */
  size_t kv_len;    /* highest cached position + 1; moved back by gemma4_kv_rewind */
  const GgufTensorInfo* tok_embd;
  float* out_norm;
  float* rope_freqs; /* [max_head_dim/2] proportional divisors (global layers) */
  float* ones;       /* [max_head_dim] all-ones weight for scale-less V norm */
  Gemma4Layer* layers;
  float* x;      /* [hidden] residual stream */
  float* logits; /* [vocab] */
  /* scratch */
  float *normed, *q, *k, *v, *attn_res, *attn_proj, *gate, *up, *ffn_out, *head_tmp;
  /* Batched-prefill state: the same scratch, one row per token. `batch` is the
   * prefill chunk size (OC_BATCH, default 32) — enough tokens that a weight row
   * is reused into being compute-bound, while the GEMM's k-panel slice
   * (256 x batch floats) still fits L1. Tune it with `make gemm-bench` on the
   * target box; 32 is the measured optimum on a Zen3+ and the shape of the
   * curve is flat enough either side that it is not worth auto-detecting. */
  OcCtx* octx;
  size_t batch;     /* prefill chunk size; may be lowered by a caller */
  size_t batch_cap; /* rows actually allocated at load; chunks clamp to this */
  float *bx, *bnormed, *bq, *bk, *bv, *battn, *bproj, *bgate, *bup, *bffn;
} Gemma4Model;

/* Takes ownership of *g on success (0). On failure writes err and leaves *g
 * for the caller to close. max_ctx caps the KV cache (0 = model context).
 * kv_quant requests the rotated 4-bit KV cache (falls back to f32 with a
 * warning when head dims are not powers of two). */
int gemma4_load(Gemma4Model* m, GgufFile* g, size_t max_ctx, bool kv_quant,
                char* err, size_t errlen);
void gemma4_free(Gemma4Model* m);

/* One decode step. Returns m->logits when need_logits, else NULL. */
float* gemma4_forward(Gemma4Model* m, int32_t token, size_t pos, bool need_logits);

/* gemma4_forward's tail: run layers [l0, n_layers) with m->x ALREADY holding
 * the residual stream for `pos` (no embedding lookup), then the final norm and
 * tied logits. l0 == 0 is what gemma4_forward calls. Exists for the CUDA
 * backend's -ngl partial offload: the GPU runs layers [0, l0) and copies x
 * back, the CPU finishes. Only the CPU layers touch the CPU KV cache. */
float* gemma4_forward_from(Gemma4Model* m, size_t pos, size_t l0, bool need_logits);

/* Drop cached KV for positions >= pos (speculative decoding / chat editing) and
 * return true when the rewind is exactly reproducible. The ring is positionally
 * addressed, so re-running the dropped positions with the same tokens reproduces
 * the original logits — but only while every position the regenerated window
 * re-reads is still resident. An SWA layer keeps just the last cache_cap
 * positions, so once generation has passed the window (kv_len > cache_cap) the
 * ONLY exactly-reproducible rewind is dropping the last token (pos == kv_len-1);
 * any deeper rewind would re-read a slot already overwritten by a later position
 * (distance < cache_cap does NOT make it safe — the tail is gone regardless).
 * Such a rewind is refused: kv_len is left unchanged and false is returned, so
 * the caller re-encodes from scratch rather than silently reading stale KV.
 * While nothing has been evicted yet (kv_len <= cache_cap) any rewind succeeds,
 * matching the full-cache llama behaviour. Rewinding forward is a no-op (true). */
bool gemma4_kv_rewind(Gemma4Model* m, size_t pos);

/* Per-session KV buffers (same sizing as gemma4_load). Session-owned; never
 * freed by gemma4_free. install swaps layer pointers onto kv; release restores
 * the primary pointers and copies m->kv_len back. */
typedef struct Gemma4Kv Gemma4Kv;
Gemma4Kv* gemma4_kv_new(const Gemma4Model* m);
void gemma4_kv_free(Gemma4Kv* kv);
void gemma4_kv_clear(Gemma4Kv* kv); /* zero buffers, kv_len=0 */
void gemma4_kv_install(Gemma4Model* m, Gemma4Kv* kv);
void gemma4_kv_release(Gemma4Model* m, Gemma4Kv* kv);

/* Prefill: n_tokens at positions pos0 .. pos0+n_tokens-1 in ONE pass.
 * Numerically equivalent to that many sequential gemma4_forward calls (the
 * batched==sequential test is the acceptance criterion for it), but each weight
 * row is read from DRAM once for the whole batch instead of once per token.
 * Splits internally at m->batch, so any n_tokens is fine. need_logits returns
 * the LAST token's logits; NULL otherwise, or on error. */
float* gemma4_forward_batch(Gemma4Model* m, const int32_t* tokens, size_t n_tokens,
                            size_t pos0, bool need_logits);

#endif
