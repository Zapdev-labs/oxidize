#ifndef OXIDIZE_EAGLE3_H
#define OXIDIZE_EAGLE3_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_EAGLE_MAX_DRAFT 8
#define OC_EAGLE_MAX_LAYERS 4
#define OC_EAGLE_VOCAB_SIZE 128256

typedef struct {
    uint32_t max_draft_tokens;
    uint32_t n_layers;
    uint32_t hidden_dim;
    float acceptance_threshold;
    bool dynamic_draft;
} OcEagleConfig;

typedef struct {
    OcEagleConfig config;
    float *hidden_states;
    uint32_t *draft_tokens;
    float *draft_probs;
    uint32_t n_draft;
    bool initialized;
} OcEagleState;

OcError oc_eagle_config_init(OcEagleConfig *cfg);
OcError oc_eagle_state_init(OcEagleState *state, const OcEagleConfig *cfg);
OcError oc_eagle_generate_draft(OcEagleState *state, const uint32_t *context,
                               size_t n_context, uint32_t max_tokens);
OcError oc_eagle_get_draft_tokens(const OcEagleState *state,
                                  const uint32_t **out_tokens, uint32_t *out_count);
OcError oc_eagle_get_draft_probs(const OcEagleState *state,
                                 const float **out_probs, uint32_t *out_count);
OcError oc_eagle_update_acceptance(OcEagleState *state, uint32_t n_accepted);
uint32_t oc_eagle_n_draft(const OcEagleState *state);
float oc_eagle_acceptance_rate(const OcEagleState *state);
void oc_eagle_state_free(OcEagleState *state);


/* Eagle3 configuration (parsed from GGUF metadata). */
typedef struct OcEagle3Config {
    size_t hidden_size;
    size_t num_hidden_layers;       /* typically 1 */
    size_t extract_layers[8];       /* target layer indices for feature fusion */
    size_t n_extract_layers;
    size_t target_hidden_size;
    bool   norm_before_residual;
    size_t vocab_size;              /* target vocab */
    size_t draft_vocab_size;       /* draft vocab (may be smaller) */
    size_t num_attention_heads;
    size_t num_key_value_heads;
    size_t head_dim;               /* 0 = auto (hidden_size / num_attention_heads) */
    size_t intermediate_size;
    float  rms_norm_eps;
    float  rope_theta;
} OcEagle3Config;

/* Lightweight weight matrix (like Rust F32Weight). */
typedef struct OcEagle3Weight {
    float *data;          /* [rows * cols] in row-major, or NULL if not loaded */
    size_t rows;
    size_t cols;
} OcEagle3Weight;

/* Eagle3 attention layer. */
typedef struct OcEagle3Attention {
    OcEagle3Weight q_proj;
    OcEagle3Weight k_proj;
    OcEagle3Weight v_proj;
    OcEagle3Weight o_proj;
} OcEagle3Attention;

/* Eagle3 decoder layer (typically just 1). */
typedef struct OcEagle3DecoderLayer {
    float *attn_norm;        /* [hidden_size] */
    float *attn_norm_2;      /* [hidden_size] - norm for g_embeddings */
    OcEagle3Attention attention;
    float *ffn_norm;         /* [hidden_size] */
    OcEagle3Weight mlp_gate;
    OcEagle3Weight mlp_up;
    OcEagle3Weight mlp_down;
} OcEagle3DecoderLayer;

/* Simple KV cache for Eagle3. */
typedef struct OcEagle3KvCache {
    float *keys;            /* [seq_len * kv_len] */
    float *values;          /* same */
    size_t seq_len;
    size_t capacity;        /* max tokens */
} OcEagle3KvCache;

/* Full Eagle3 draft model. */
typedef struct OcEagle3DraftModel {
    OcEagle3Config config;
    OcEagle3Weight fc;              /* feature fusion: [hidden, extract_layers * target_hidden] */
    uint64_t *d2t;                   /* draft-to-target vocab map [draft_vocab_size] */
    size_t n_d2t;
    OcEagle3DecoderLayer layer;     /* single decoder layer */
    float *output_norm;             /* [hidden_size] */
    OcEagle3Weight output;          /* lm_head: [draft_vocab, hidden] */
    OcEagle3Weight tok_embeddings;  /* [vocab, hidden] */
    float *g_embeddings;             /* encoder output [hidden_size] */
    OcEagle3KvCache *kv_caches;     /* [num_hidden_layers] */
    size_t position_offset;
    bool loaded;
} OcEagle3DraftModel;

/* Initialize config with defaults. */
void oc_eagle3_config_init(OcEagle3Config *cfg);

/* Get head_dim (explicit or hidden_size / num_attention_heads). */
size_t oc_eagle3_config_head_dim(const OcEagle3Config *cfg);

/* Get encoder input width = n_extract_layers * target_hidden_size. */
size_t oc_eagle3_config_encoder_input_width(const OcEagle3Config *cfg);

/* Initialize a draft model from config (allocates, but weights not loaded). */
OcError oc_eagle3_model_init(OcEagle3DraftModel *m, const OcEagle3Config *cfg);

/* Encode target model features into g_embeddings via fc GEMV.
 * target_features: [encoder_input_width] */
OcError oc_eagle3_encode_features(OcEagle3DraftModel *m,
                                   const float *target_features, size_t n_features);

/* Forward one token through the decoder. Returns hidden and logits.
 * hidden: [hidden_size], logits: [vocab_size] */
OcError oc_eagle3_forward_decoder(OcEagle3DraftModel *m,
                                   uint32_t token,
                                   float *out_hidden, size_t hidden_len,
                                   float *out_logits, size_t logits_len);

/* Compute logits from a hidden state (rms_norm + lm_head + d2t scatter). */
OcError oc_eagle3_logits_from_hidden(const OcEagle3DraftModel *m,
                                      const float *hidden, size_t hidden_len,
                                      float *out_logits, size_t logits_len);

/* Reset KV cache and position. */
void oc_eagle3_reset_cache(OcEagle3DraftModel *m);

/* Reserve KV cache capacity for n_tokens. */
OcError oc_eagle3_reserve_cache_tokens(OcEagle3DraftModel *m, size_t n_tokens);

/* Free the draft model. */
void oc_eagle3_model_free(OcEagle3DraftModel *m);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_EAGLE3_H */
