/*
 * dflash.h — DFlash speculative decoding for C port.
 *
 * Ports the DFlash algorithm from oxidize-core/src/model/dflash.rs.
 * DFlash uses a small draft model to generate candidate continuations
 * that are verified by the target model in a single forward pass.
 */
#ifndef OXIDIZE_DFLASH_H
#define OXIDIZE_DFLASH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_DFLASH_MAX_DRAFT 16
#define OC_DFLASH_MAX_CONTEXT 4096

typedef struct {
    uint32_t max_draft_tokens;
    uint32_t verification_window;
    float acceptance_threshold;
    bool adaptive;
} OcDFlashConfig;

typedef struct {
    OcDFlashConfig config;
    uint32_t draft_tokens[OC_DFLASH_MAX_DRAFT];
    float draft_logprobs[OC_DFLASH_MAX_DRAFT];
    uint32_t target_tokens[OC_DFLASH_MAX_DRAFT];
    float target_logprobs[OC_DFLASH_MAX_DRAFT];
    uint32_t n_draft;
    uint32_t n_accepted;
    uint64_t total_accepted;
    uint64_t total_proposed;
} OcDFlashState;

OcError oc_dflash_config_init(OcDFlashConfig *cfg);
OcError oc_dflash_state_init(OcDFlashState *state, const OcDFlashConfig *cfg);
OcError oc_dflash_set_draft(OcDFlashState *state, const uint32_t *tokens,
                           const float *logprobs, uint32_t n);
OcError oc_dflash_set_target(OcDFlashState *state, const uint32_t *tokens,
                            const float *logprobs, uint32_t n);
OcError oc_dflash_verify(OcDFlashState *state, uint32_t *out_accepted, uint32_t *out_n);
OcError oc_dflash_get_accepted(const OcDFlashState *state, const uint32_t **out_tokens, uint32_t *out_n);
float oc_dflash_acceptance_rate(const OcDFlashState *state);
uint32_t oc_dflash_avg_acceptance(const OcDFlashState *state);
void oc_dflash_state_free(OcDFlashState *state);

/* ─── Real DFlash draft model (port of dflash.rs) ───────────────────── */

/* DFlash configuration matching HuggingFace config.json. */
typedef struct OcDFlashModelConfig {
    size_t hidden_size;
    size_t num_hidden_layers;
    size_t num_target_layers;
    size_t block_size;
    size_t target_layer_ids[16];
    size_t n_target_layer_ids;
    uint32_t mask_token_id;
    size_t vocab_size;
    size_t num_attention_heads;
    size_t num_key_value_heads;
    size_t intermediate_size;
    float  rms_norm_eps;
    float  rope_theta;
} OcDFlashModelConfig;

/* Lightweight weight (like Rust F32Weight). */
typedef struct OcDFlashWeight {
    float *data;        /* [rows * cols] row-major, or NULL */
    size_t rows;
    size_t cols;
} OcDFlashWeight;

/* DFlash attention layer. */
typedef struct OcDFlashAttention {
    OcDFlashWeight q_proj;
    OcDFlashWeight k_proj;
    OcDFlashWeight v_proj;
    OcDFlashWeight o_proj;
    float *q_norm_weight;   /* [head_dim] */
    float *k_norm_weight;   /* [head_dim] */
} OcDFlashAttention;

/* DFlash decoder layer. */
typedef struct OcDFlashDecoderLayer {
    float *input_layernorm;    /* [hidden_size] */
    OcDFlashAttention attention;
    float *post_attention_layernorm; /* [hidden_size] */
    OcDFlashWeight mlp_gate;
    OcDFlashWeight mlp_up;
    OcDFlashWeight mlp_down;
} OcDFlashDecoderLayer;

/* DFlash KV cache. */
typedef struct OcDFlashKvCache {
    float *keys;
    float *values;
    size_t seq_len;
    size_t capacity;
} OcDFlashKvCache;

/* Full DFlash draft model. */
typedef struct OcDFlashDraftModel {
    OcDFlashModelConfig config;
    OcDFlashWeight fc;                /* feature fusion */
    float *fc_bias;                   /* [hidden_size] */
    float *hidden_norm;               /* [hidden_size] */
    OcDFlashDecoderLayer *layers;    /* [num_hidden_layers] */
    size_t n_layers;
    float *norm;                      /* final norm [hidden_size] */
    OcDFlashWeight output;            /* lm_head */
    OcDFlashWeight tok_embeddings;    /* [vocab, hidden] */
    OcDFlashKvCache *kv_cache;        /* [num_hidden_layers] */
    float *target_hidden_cache;      /* [target_hidden_width] */
    size_t target_hidden_cache_len;
    size_t position_offset;
    bool loaded;
} OcDFlashDraftModel;

/* Initialize config with defaults. */
void oc_dflash_model_config_init(OcDFlashModelConfig *cfg);

/* Get head_dim = hidden_size / num_attention_heads. */
size_t oc_dflash_config_head_dim(const OcDFlashModelConfig *cfg);

/* Get target_hidden_width = hidden_size * num_target_layers. */
size_t oc_dflash_config_target_hidden_width(const OcDFlashModelConfig *cfg);

/* Initialize a draft model from config. Weights are not loaded yet. */
OcError oc_dflash_model_init(OcDFlashDraftModel *m, const OcDFlashModelConfig *cfg);

/* Cache target model hidden states for fusion. */
OcError oc_dflash_cache_target_hidden(OcDFlashDraftModel *m,
                                       const float *hidden, size_t len);

/* Clear speculative caches (target_hidden_cache, KV). */
void oc_dflash_clear_speculative_caches(OcDFlashDraftModel *m);

/* Forward a single token, returning hidden state. */
OcError oc_dflash_forward_token(OcDFlashDraftModel *m,
                                 uint32_t token,
                                 const float *target_hidden, size_t target_hidden_len,
                                 float *out_hidden, size_t hidden_len);

/* Compute logits from a hidden state. */
OcError oc_dflash_logits(const OcDFlashDraftModel *m,
                          const float *hidden, size_t hidden_len,
                          float *out_logits, size_t logits_len);

/* Free the draft model. */
void oc_dflash_model_free(OcDFlashDraftModel *m);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_DFLASH_H */
