/*
 * dflash2.h — DFlash 2 block-diffusion draft model for the C port.
 *
 * Ports the DFlash 2 architecture (z-lab/dflash, dflash/model.py:
 * DFlash2DraftModel) as used by incoai/GLM-5.3-Flash-DFlash2:
 *
 *   - 5-layer Qwen3-style backbone, hidden 4096, 32 Q heads / 8 KV heads,
 *     head_dim 128, RMSNorm q/k per head, RoPE (theta 10k).
 *   - Dual-stream attention: K/V are computed from BOTH the fused target
 *     context (fc @ target_layer hiddens, hidden_norm) and the draft's own
 *     noise embeddings; is_causal=false with a bidirectional sliding
 *     window of 2048, so the KV cache is a fixed-size ring.
 *   - Grouped dynamic depthwise convolution (kernel 2, group 16) around
 *     both the attention output and the MLP output:
 *     out[i,c] = sum_t (base[t,c] + delta[i,t,g(c)]) * x[i-t,c].
 *   - Candidate selector: per position top-k (16) draft logits, then a
 *     greedy path through the lattice scored by
 *     edge(p->c) = <A[p] o proj(h), B[c]> + unary[c],
 *     where A/B are per-token codebooks of rank 256.
 *
 * The draft checkpoint has NO token embedding table and NO lm_head: token
 * identity enters through the selector codebooks, and draft logits are
 * computed with the *target* model's lm_head. This implementation takes
 * the noise embeddings and the lm_head from the caller (the target), and
 * consumes a fixed per-step "target context" feature, matching SGLang's
 * DFLASH speculative worker contract.
 *
 * A single propose step drafts a whole block of tokens in one parallel
 * pass (block_size 8 for GLM-5.3-Flash-DFlash2).
 */
#ifndef OXIDIZE_DFLASH2_H
#define OXIDIZE_DFLASH2_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Hard limits matching the GLM-5.3-Flash-DFlash2 checkpoint. */
#define OC_DFLASH2_MAX_LAYERS        8
#define OC_DFLASH2_MAX_TARGET_LAYERS 16
#define OC_DFLASH2_RANK              256
#define OC_DFLASH2_MAX_TOP_K         32
#define OC_DFLASH2_MAX_BLOCK         32
#define OC_DFLASH2_MAX_GROUPS        (4096 / 16)

/* ─── Config ─────────────────────────────────────────────────────────── */

typedef struct OcDFlash2Config {
    size_t   hidden_size;        /* 4096 */
    size_t   intermediate_size;  /* 12288 */
    size_t   num_hidden_layers;  /* 5 */
    size_t   num_attention_heads;/* 32 */
    size_t   num_key_value_heads;/* 8 */
    size_t   head_dim;           /* 128 */
    size_t   vocab_size;         /* 154880 */
    size_t   num_target_layers;  /* 45 */
    size_t   target_layer_ids[OC_DFLASH2_MAX_TARGET_LAYERS];
    size_t   n_target_layer_ids; /* 5: {5, 14, 24, 33, 42} */
    size_t   block_size;         /* 8 */
    size_t   conv_kernel_size;   /* 2 */
    size_t   conv_group_size;    /* 16 */
    uint32_t mask_token_id;      /* 154856 */
    size_t   selector_rank;      /* 256 */
    size_t   selector_top_k;     /* 16 */
    size_t   sliding_window;     /* 2048 */
    float    rope_theta;         /* 10000.0f */
    float    rms_norm_eps;       /* 1e-5f */
} OcDFlash2Config;

/* GLM-5.3-Flash-DFlash2 defaults. */
void oc_dflash2_config_init(OcDFlash2Config *cfg);

/* Hugepage-aware allocation for the big per-step working sets (weights,
 * lm_head): anonymous mmap + MADV_HUGEPAGE on Linux, malloc elsewhere.
 * Allocations of at least 1 MiB are mmap-backed and return NULL on
 * failure (no malloc fallback, so frees always pair with the matching
 * deallocator via oc_dflash2_free_huge(p, n)). */
void *oc_dflash2_alloc_huge(size_t n);
void oc_dflash2_free_huge(void *p, size_t n);

/* ─── Weights ───────────────────────────────────────────────────────── */

typedef struct OcDFlash2Weight {
    float *data;     /* row-major [rows, cols]; NULL until loaded (or when
                      * the weight is kept in BF16 — see bf16). */
    uint16_t *bf16;  /* optional BF16 rows [rows, cols] (raw file bytes,
                      * little-endian); halves resident + streamed bytes.
                      * Mutually exclusive with data. */
    size_t rows;
    size_t cols;
    size_t alloc_bytes; /* backing-store size for hugepage-aware free of
                         * data/bf16 (0 = plain malloc/free). */
    /* Optional: synthetic weight generator for benchmarks. When data is
     * NULL and generate != NULL, row `r` is materialized into
     * gen_buf[0..cols) on demand (one row per call; r indexes the vocab
     * row, not gen_buf). Used to exercise the full vocab-GEMV cost
     * without a 2.4 GB target lm_head in memory. */
    void (*generate)(size_t row, size_t cols, float *gen_buf, void *user);
    void *gen_user;
} OcDFlash2Weight;

/* Grouped dynamic causal conv weights (per layer, attention + mlp). */
typedef struct OcDFlash2Conv {
    OcDFlash2Weight base_kernel;    /* [kernel, hidden] */
    OcDFlash2Weight kernel_proj;    /* [2*kernel*groups, hidden] */
} OcDFlash2Conv;

typedef struct OcDFlash2Attention {
    OcDFlash2Weight q_proj;   /* [32*128, 4096] */
    OcDFlash2Weight k_proj;   /* [8*128, 4096]  */
    OcDFlash2Weight v_proj;   /* [8*128, 4096]  */
    OcDFlash2Weight o_proj;   /* [4096, 4096]   */
    float *q_norm;             /* [128] */
    float *k_norm;             /* [128] */
} OcDFlash2Attention;

typedef struct OcDFlash2Layer {
    float *input_layernorm;         /* [4096] */
    OcDFlash2Attention attn;
    OcDFlash2Conv attn_conv;
    float *post_attention_layernorm;/* [4096] */
    OcDFlash2Weight mlp_gate;       /* [12288, 4096] */
    OcDFlash2Weight mlp_up;         /* [12288, 4096] */
    OcDFlash2Weight mlp_down;       /* [4096, 12288] */
    OcDFlash2Conv mlp_conv;
} OcDFlash2Layer;

/* Candidate-selector codebooks and projection. */
typedef struct OcDFlash2Selector {
    OcDFlash2Weight predecessor_codebook; /* [vocab, rank] */
    OcDFlash2Weight successor_codebook;   /* [vocab, rank] */
    OcDFlash2Weight hidden_projection;     /* [rank, hidden] */
} OcDFlash2Selector;

/* ─── KV ring cache ─────────────────────────────────────────────────── */

/*
 * The draft attends bidirectionally inside a strict
 * `|pos_q - pos_k| < sliding_window` mask. The absolute position limit is
 * 1M tokens, but attention only ever reaches `sliding_window` positions,
 * so the cache is a ring of capacity `sliding_window + block_size` per
 * layer. K/V for the context stream and the noise stream are appended in
 * one timeline in strictly increasing position order.
 *
 * `total` is the monotonic write count; entries live in slots
 * `w % capacity` for write indices w in [total - len, total). Trimming
 * drops the NEWEST entries (positions >= a cutoff) by decrementing total.
 */
typedef struct OcDFlash2KvRing {
    float *k;            /* [capacity, n_kv_heads, head_dim] */
    float *v;            /* [capacity, n_kv_heads, head_dim] */
    int64_t *pos;        /* [capacity] absolute position of each slot */
    size_t capacity;
    size_t n_kv_heads;
    size_t head_dim;
    size_t total;        /* monotonic writes since clear */
    size_t len;          /* valid entries = min(total, capacity) */
    float *undo_k;       /* slots overwritten by the latest append */
    float *undo_v;
    int64_t *undo_pos;
    size_t undo_total;   /* total before the latest append */
    size_t undo_n;
} OcDFlash2KvRing;

OcError oc_dflash2_kvring_init(OcDFlash2KvRing *ring, size_t capacity,
                               size_t n_kv_heads, size_t head_dim);
void oc_dflash2_kvring_free(OcDFlash2KvRing *ring);
void oc_dflash2_kvring_clear(OcDFlash2KvRing *ring);

/* Append K/V for `n` positions starting at absolute position `pos0`.
 * K/V are already RoPE'd. Layout: k/v [n, n_kv_heads, head_dim]. */
OcError oc_dflash2_kvring_append(OcDFlash2KvRing *ring,
                                 const float *k, const float *v,
                                 int64_t pos0, size_t n);

/* ─── Model ─────────────────────────────────────────────────────────── */

typedef struct OcDFlash2Model {
    OcDFlash2Config cfg;

    /* Top-level weights. */
    OcDFlash2Weight fc;          /* [hidden, n_target_layers*hidden] */
    float *hidden_norm;          /* [hidden] */
    float *norm;                 /* final RMSNorm [hidden] */
    OcDFlash2Selector selector;

    OcDFlash2Layer *layers;      /* [num_hidden_layers] */
    size_t n_layers;

    /* Per-layer sliding-window KV rings. */
    OcDFlash2KvRing *kv;         /* [num_hidden_layers] */

    /* Scratch: fused context feature per attended position.
     * target_ctx holds hidden_norm(fc @ concat(target hiddens)) for the
     * rows set via oc_dflash2_set_context (prefill rows at step 0, then
     * the per-verify rows), up to kv_capacity rows. */
    float *target_ctx;           /* [kv_capacity, hidden] */
    size_t target_ctx_len;       /* number of valid rows */

    /* Per-layer KV ring capacity (sliding_window + block_size). */
    size_t kv_capacity;

    /* RoPE frequency table (head_dim/2 entries), computed once at load
     * with the same powf the reference evaluates per call; cached so
     * propose does not recompute transcendentals every step. */
    float *rope_freq;            /* [head_dim / 2] */
    size_t rope_freq_n;

    /* Absolute position of the next noise token to be drafted. */
    int64_t next_noise_pos;

    /* Final-normed hidden rows [block, hidden] from the latest successful
     * propose call, retained for callers that want logits without
     * re-running the backbone (oc_dflash2_last_hidden). */
    float *last_hidden;
    size_t last_hidden_len;

    /* Deterministic PRNG state for the stochastic selector path
     * (temperature > 0). Explicit state instead of rand(): reproducible
     * across runs, thread-safe, and not limited by RAND_MAX resolution. */
    uint32_t rng_state;

    /* Loaded weights flag. */
    bool loaded;
} OcDFlash2Model;

/* Load all weights from a HuggingFace SafeTensors checkpoint
 * (model.safetensors, BF16) + config.json parsed from the same directory.
 * Returns OC_ERR_FORMAT on missing tensor/shape mismatch. */
OcError oc_dflash2_model_load(OcDFlash2Model *m, const char *st_path,
                              const char *config_json_path);

/* Release all memory. */
void oc_dflash2_model_free(OcDFlash2Model *m);

/* ─── Target-side inputs (per verification step) ────────────────────── */

/*
 * The target model must provide, per produced token:
 *   - `noise_embeddings`: [block, hidden] rows = target's input embedding
 *     for the block tokens (scaled by input_embedding_scale if any).
 *   - `target_context`:   [n_ctx, n_target_layer_ids*hidden] concat of
 *     the target's hidden states at target_layer_ids (the *selected*
 *     layers, not num_target_layers), i.e. exactly what fc consumes. For
 *     GLM-5.3-Flash-DFlash2: 5 layers x 4096 = 20480. The fc weight is
 *     [hidden, n_target_layer_ids*hidden] to match.
 *
 * `oc_dflash2_set_context` fuses via fc + hidden_norm into an internal
 * per-position context row used as the attention context stream.
 */
OcError oc_dflash2_set_context(OcDFlash2Model *m,
                               const float *target_context,
                               size_t n_ctx_rows);

/*
 * Propose a block of draft tokens in one parallel forward pass.
 *
 * Inputs:
 *   anchor_ids     [n_anchor]   token ids already committed (the last
 *                               committed token is the path start), where
 *                               n_anchor >= 1; the path starts from
 *                               anchor_ids[n_anchor-1]. All ids must be
 *                               < cfg.vocab_size (they index the selector
 *                               codebooks).
 *   noise_emb      [block, hidden] noise embeddings for the block tokens
 *                               (block tokens = [mask-padded block ids]).
 *   block_ids      [block]      the block token ids (first is the anchor
 *                               position's committed token). Currently
 *                               ignored: the C port consumes only the
 *                               noise embeddings (which already carry the
 *                               target's token identity), not raw ids.
 *   lm_head        target lm_head [vocab, hidden] row-major, used to turn
 *                               draft hidden states into logits.
 *   temperature    0 => greedy selector path; > 0 => stochastic selector.
 *
 * Outputs:
 *   out_tokens     [block-1]    selected path tokens (draft proposal).
 *   out_top_k      [block-1][top_k] candidate ids per position (for
 *                               rejection sampling; row-major
 *                               [block-1)*top_k]).
 *   out_top_k_probs [block-1][top_k] candidate probs per position
 *                               (temperature-adjusted), row-major.
 *
 * Returns OC_OK on success; `block - 1` tokens are written to out_tokens.
 */
OcError oc_dflash2_propose(OcDFlash2Model *m,
                           const uint32_t *anchor_ids, size_t n_anchor,
                           const float *noise_emb, size_t block,
                           const uint32_t *block_ids,
                           const OcDFlash2Weight *lm_head,
                           float temperature,
                           uint32_t *out_tokens,
                           uint32_t *out_top_k,
                           float *out_top_k_probs);

/* Reset draft caches (KV rings + position counter) between contexts. */
void oc_dflash2_reset(OcDFlash2Model *m);

/* Debug/validation: run the backbone forward ONLY (no selector), writing
 * the final-norm hidden rows for all `block` noise rows into out_hidden
 * [block, hidden]. Same semantics as propose's forward pass: context rows
 * from the last oc_dflash2_set_context, noise rows from `noise_emb`.
 * Positions: ctx rows at [start - n_ctx, start), noise at [start, start +
 * block). */
OcError oc_dflash2_forward_debug(OcDFlash2Model *m,
                                 const float *noise_emb, size_t block,
                                 float *out_hidden);

/* Debug/validation: selector-only on precomputed draft hidden rows.
 * Mirrors CandidateSelector.select with temperature 0: out_tokens
 * [block-1], and out_cand receives the top-k candidate ids per position
 * (unsorted, matching torch.topk(sorted=False) as closely as the C
 * selection allows; ties may differ). */
OcError oc_dflash2_selector_debug(OcDFlash2Model *m,
                                  const float *draft_hidden, size_t n_rows,
                                  const uint32_t *anchor_ids, size_t n_anchor,
                                  const OcDFlash2Weight *lm_head,
                                  uint32_t *out_tokens,
                                  uint32_t *out_cand);

/* Trim KV entries whose absolute position >= pos_keep_exclusive. The next
 * append must be at position pos_keep_exclusive. Used immediately after a
 * propose-step append to roll the ring back to the last committed position;
 * overwritten live slots from that append are restored across ring wrap. */
void oc_dflash2_kvring_trim(OcDFlash2KvRing *ring, int64_t pos_keep_exclusive);

/* Last-step hidden states of the latest propose call, normed: useful for
 * callers that want logits without re-running the backbone. [block, hidden] */
const float *oc_dflash2_last_hidden(const OcDFlash2Model *m, size_t *rows);

/* ─── Micro-kernels exposed for tests/benchmarks ────────────────────── */

/* out[i] = sum_c A[i,c]*B[c] with A row-major [rows, cols]. */
void oc_dflash2_gemv(const float *w, size_t rows, size_t cols,
                     const float *x, float *out);

/* Grouped dynamic causal conv, kernel=2, group_size from cfg.
 * x: [len, hidden]; dyn: [len, kernel*groups]; base: [kernel, hidden].
 * out: [len, hidden]. out[i,c] = sum_t (base[t,c]+dyn[i,t,g(c)])*x[i-t,c]. */
void oc_dflash2_grouped_conv(const float *x, const float *dyn,
                              const float *base,
                              size_t len, size_t hidden,
                              size_t kernel, size_t group_size,
                              float *out);

/* Selector edge scores for one position:
 * scores[k] = <A[p] o proj(h), B[c_k]> + unary[k]. */
void oc_dflash2_selector_scores(const float *proj_h, /* [rank] */
                                const float *A_p,    /* [rank] */
                                const float *B_k,    /* [k*rank] */
                                const float *unary,  /* [k] */
                                size_t k,
                                float *scores);      /* [k] */

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_DFLASH2_H */
