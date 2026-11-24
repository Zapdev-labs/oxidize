/*
 * inkling.h — Inkling-2 loop transformer (inkling2_loop) and its DSpark
 * drafter.
 *
 * Inkling-2-Mini runs physical layers [core_start, core_end) num_passes
 * times per token (24 physical layers, 48 layer visits). Each visit has its
 * own KV cache and short-convolution state. Layers mix earlier block outputs
 * with learned depth-attention queries, attention carries a learned
 * relative-position bias and a per-head sink, K and V pass through causal
 * short convolutions, and the FFN is a 128-expert sigmoid-routed MoE with a
 * soft-capped SiLU ("situ") activation. Loop passes after the first mix a
 * message from the previous pass into the Q, K, V and FFN inputs through
 * low-rank gain/forget controllers.
 *
 * The DSpark drafter (mtp.safetensors) is a 3-layer non-causal block
 * drafter: it reads target features from layer visits 5, 23, 41 and the
 * final norm, drafts block_size - 1 tokens per forward from an anchor token
 * plus mask embeddings, and biases each position's logits with an RNN
 * Markov head conditioned on the token drafted before it.
 */
#ifndef OXIDIZE_INKLING_H
#define OXIDIZE_INKLING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_INK_MAX_LAYERS  24
#define OC_INK_MAX_VISITS  64
#define OC_INK_MAX_BATCH   128  /* tokens per forward call (prefill chunk / verify block) */
#define OC_INK_N_CTRL      4    /* loop controllers: q, k, v, ffn */
#define OC_INK_MAX_TAPS    8

/* ─── conversion ────────────────────────────────────────────────────────── */

typedef enum {
    OC_INK_W_NVFP4 = 0,
    OC_INK_W_BF16  = 1,
} OcInklingWeightType;

typedef struct OcInklingConvertOptions {
    const char *hf_dir;          /* Hugging Face checkpoint directory               */
    const char *out_path;        /* target GGUF (NULL = skip)                        */
    const char *draft_out_path;  /* DSpark drafter GGUF (NULL = skip)                */
    OcInklingWeightType dense_type;   /* attention projections                     */
    OcInklingWeightType output_type;  /* lm head                                    */
    OcInklingWeightType draft_type;   /* drafter projections                        */
} OcInklingConvertOptions;

OcError oc_inkling_convert(const OcInklingConvertOptions *opt);

/* ─── weights ───────────────────────────────────────────────────────────── */

/* One linear weight [rows x cols] or a stack of n_mat of them (experts). */
typedef struct OcInkMat {
    uint32_t       gtype;      /* ggml type: 40 NVFP4, 30 BF16, 0 F32 */
    const uint8_t *data;
    size_t         rows, cols, n_mat;
    size_t         row_bytes, mat_bytes;
    const float   *scale;      /* NVFP4 global scale per matrix, NULL = 1 */
} OcInkMat;

typedef struct OcInkLayer {
    bool swa;
    OcInkMat wq, wk, wv, wo, wgate, wr, sink;
    OcInkMat k_conv_proj, v_conv_proj;            /* full-attention layers only */
    const float *attn_norm;                       /* SWA layers                  */
    const float *q_inp_norm, *k_inp_norm, *v_inp_norm;
    const float *k_conv, *v_conv, *attn_post_conv, *ffn_post_conv;  /* [C][n_conv] */
    const float *q_clip, *rel_clip;               /* [n_head][head_dim], [n_head][n_rel] */
    const float *rel_bias;                        /* [extent][n_rel]              */
    const float *attn_res_query;                  /* [n_res_q][n_embd], NULL on layer 0 */
    uint32_t     n_attn_res_query;
    const float *ffn_res_query, *ffn_norm;
    OcInkMat     router;                          /* F32 [n_expert x n_embd]     */
    const float *router_bias;
    OcInkMat     gate_up, down;                   /* experts                      */
    const float *msg_query;                       /* [4][n_embd], core layers     */
    OcInkMat     ctrl_down[OC_INK_N_CTRL], ctrl_gain[OC_INK_N_CTRL], ctrl_forget[OC_INK_N_CTRL];
    /* owned, built at load: conv taps as [n_conv][C] (k, v, attn post, ffn post)
     * and the relative bias table as [n_rel][extent] */
    float       *conv_t[4];
    float       *rel_bias_t;
} OcInkLayer;

typedef struct OcInklingModel {
    OcGgufMmappedFile map;
    OcTokenizer       tok;
    bool              have_tok;

    uint32_t n_embd, n_layer, n_head, n_head_kv, head_dim, n_vocab;
    uint32_t n_swa, n_rel, rel_extent, n_expert, n_expert_used, n_ff_exp;
    uint32_t res_block, n_conv, core_start, core_end, n_passes, gate_rank, log_floor;
    uint32_t n_ctx_train;
    float    eps, qk_clip, rel_clip, log_alpha, situ_beta, situ_lbeta, route_scale, softcap;

    const uint16_t *tok_embd;                     /* BF16 [n_vocab][n_embd]      */
    const float    *tok_norm, *output_norm, *output_res_query;
    OcInkMat        output;
    OcInkLayer      layers[OC_INK_MAX_LAYERS];

    /* layer visit schedule */
    uint32_t n_visits;
    uint8_t  visit_layer[OC_INK_MAX_VISITS];
    uint8_t  visit_pass[OC_INK_MAX_VISITS];

    char    *chat_template;
    void    *gpu;                                  /* GPU weight mirror, oc_inkling_gpu_load */
} OcInklingModel;

OcError oc_inkling_load(const char *path, OcInklingModel *m);
void    oc_inkling_free(OcInklingModel *m);
bool    oc_inkling_gguf_is_inkling(const char *path);

/* ─── session ───────────────────────────────────────────────────────────── */

typedef struct OcInklingSession OcInklingSession;

/* n_ctx: maximum positions. n_taps / tap_visits: layer visits whose input
 * the forward copies out per token for the drafter (visit == n_visits means
 * the final normed hidden state). */
OcError oc_inkling_session_new(const OcInklingModel *m, uint32_t n_ctx,
                               const int32_t *tap_visits, uint32_t n_taps,
                               OcInklingSession **out);
void    oc_inkling_session_free(OcInklingSession *s);
void    oc_inkling_session_reset(OcInklingSession *s);
uint32_t oc_inkling_session_pos(const OcInklingSession *s);
const OcInklingModel *oc_inkling_session_model(const OcInklingSession *s);

/* Run n tokens (n <= OC_INK_MAX_BATCH) at the next positions. With
 * all_logits, logits receives n rows of n_vocab; otherwise only the last
 * token's row. logits may be NULL. */
OcError oc_inkling_forward(OcInklingSession *s, const uint32_t *tokens, size_t n,
                           float *logits, bool all_logits);

/* Prefill any number of tokens in OC_INK_MAX_BATCH chunks; last logits out. */
OcError oc_inkling_prefill(OcInklingSession *s, const uint32_t *tokens, size_t n, float *logits);

/* Drop the tail of the most recent forward call, keeping its first n_keep
 * tokens (KV entries past the new end are ignored, the convolution states
 * are restored from that call's input history). */
OcError oc_inkling_rollback(OcInklingSession *s, size_t n_keep);

/* Tap features of token i of the most recent forward call: [n_taps][n_embd]. */
const float *oc_inkling_taps(const OcInklingSession *s, size_t i);

/* Most tokens one forward call accepts (larger on the GPU). */
uint32_t oc_inkling_session_max_batch(const OcInklingSession *s);

/* ─── GPU (Vulkan builds, make vulkan) ──────────────────────────────────── */

/* True when this build has the Vulkan forward. */
bool    oc_inkling_gpu_compiled(void);

/* Upload the weights to the GPU once (kept in m->gpu, freed with the model). */
OcError oc_inkling_gpu_load(OcInklingModel *m);

/* Run a fresh session's forward calls on the GPU; needs oc_inkling_gpu_load. */
OcError oc_inkling_session_use_gpu(OcInklingSession *s);

/* ─── DSpark drafter ────────────────────────────────────────────────────── */

typedef struct OcInkDraftLayer {
    bool swa;
    const float *attn_norm, *q_norm, *k_norm, *k_ctx_norm, *ctx_norm, *ctx_mix, *ffn_norm;
    OcInkMat wq, wk, wv, wo, wk_ctx, wv_ctx, ffn_gate, ffn_up, ffn_down;
} OcInkDraftLayer;

typedef struct OcInklingDraft {
    OcGgufMmappedFile map;
    uint32_t n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff, n_swa, block_size, rank, n_vocab;
    float    eps, rope_base;
    uint32_t n_targets;
    int32_t  target_layers[OC_INK_MAX_TAPS];
    OcInkDraftLayer layers[4];
    const float *output_norm, *mask_embd, *markov_ctx_mix, *markov_ctx_norm;
    const float *hidden_out, *hidden_out_b, *rnn_proj, *rnn_proj_b;   /* F32 */
    const uint16_t *markov_w1;                                         /* BF16 [n_vocab][rank] */
    /* owned, built at load: the Markov bias table folded through the
     * target lm head, w2 = lm_head . hidden_out as NVFP4 [n_vocab][rank]
     * (global scale w2_scale) and w2_b = lm_head . hidden_out_b */
    uint8_t *w2;
    float    w2_scale;
    float   *w2_b;
} OcInklingDraft;

OcError oc_inkling_draft_load(const char *path, const OcInklingModel *target, OcInklingDraft *d);
void    oc_inkling_draft_free(OcInklingDraft *d);

typedef struct OcInklingSpecStats {
    uint64_t n_prompt, rounds, drafted, accepted, emitted;
    uint64_t accept_hist[16];
    double   t_prefill, t_decode, t_draft, t_verify;
} OcInklingSpecStats;

typedef struct OcInklingSpec OcInklingSpec;

OcError oc_inkling_spec_new(const OcInklingDraft *d, OcInklingSession *target, uint32_t n_ctx,
                            OcInklingSpec **out);
void    oc_inkling_spec_free(OcInklingSpec *sp);

/* Feed the taps of the most recent target forward call (its first n tokens)
 * to the drafter's context. */
OcError oc_inkling_spec_inject(OcInklingSpec *sp, size_t n);

/* Draft up to n_max tokens after anchor (the sampled token not yet run
 * through the target). Returns the count in *n_out. */
OcError oc_inkling_spec_draft(OcInklingSpec *sp, uint32_t anchor, uint32_t n_max,
                              uint32_t *out, uint32_t *n_out);

/* Greedy generation after prefilling prompt into a fresh session. With a
 * drafter (sp != NULL, draft_max > 0) each round drafts up to draft_max
 * tokens, verifies them in one target forward, and keeps the longest
 * matching prefix plus the target's own next token, so the output is
 * identical to plain greedy decoding. Drafting stops at a slot whose top
 * probability is below p_min (0: never). Emits tokens through cb until
 * n_predict tokens, an end-of-generation token, or cb returning false. */
typedef bool (*OcInklingTokenCb)(uint32_t token, void *user);

OcError oc_inkling_generate(OcInklingSession *s, OcInklingSpec *sp, const uint32_t *prompt,
                            size_t n_prompt, uint32_t n_predict, uint32_t draft_max, float p_min,
                            OcInklingTokenCb cb, void *user, OcInklingSpecStats *stats);

#ifdef __cplusplus
}
#endif

#endif
