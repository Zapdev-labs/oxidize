#ifndef OXIDIZE_INF_MODEL_H
#define OXIDIZE_INF_MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/inference.h"
#include "oxidize/workspace.h"
#include "oxidize/weight_storage.h"
#include "oxidize/layer_weights.h"
#include "oxidize/mtp_weights.h"
#include "oxidize/kv_cache.h"
#include "oxidize/ssm.h"
#include "oxidize/seq_kv.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Config + embeddings. */
    OcInferenceConfig  config;
    OcWeightStorage     tok_embeddings;
    size_t              tok_embeddings_cols;
    float              *norm_weight;        /* [hidden_size] final norm */
    OcWeightStorage     output_weight;      /* lm_head */

    /* Per-layer weights. */
    OcLayerWeights     *layers;
    size_t              n_layers;
    size_t              layers_cap;

    /* MTP/nextn (optional). NULL if not present. */
    OcMtpWeights       *mtp;

    /* KV cache (single-stream timeline). */
    OcKvCache           kv_cache;

    /* Maps absolute layer index to KV cache layer index.
     * -1 = non-attention layer (shortconv/Mamba).
     * Allows KV cache to be sized to only attention layers. */
    int32_t            *kv_layer_map;
    size_t              kv_layer_map_len;

    /* SSM/Mamba persistent state. */
    OcSsmEngine         ssm_engine;

    /* Pre-allocated workspace. */
    OcWorkspace         workspace;

    /* Final output-normalized hidden for MTP. */
    float              *last_output_hidden;
    size_t              last_output_hidden_len;

    /* EAGLE3 hidden state capture. */
    size_t             *eagle3_capture_layers;
    size_t              eagle3_n_capture_layers;
    float             **eagle3_layer_hiddens;  /* per-capture-layer f32* */
    size_t              eagle3_n_hiddens;

    /* State flags. */
    bool                loaded;
} OcInferenceModel;

/* Initialize a model with the given config (allocates layers + workspace). */
OcError oc_inf_model_init(OcInferenceModel *m, const OcInferenceConfig *cfg);

/* Free all model memory. Safe on NULL. */
void oc_inf_model_free(OcInferenceModel *m);

/* Add a layer to the model (grows layers array if needed). */
OcError oc_inf_model_add_layer(OcInferenceModel *m, OcLayerWeights *layer);

/* Set the MTP weights (transfers ownership). */
OcError oc_inf_model_set_mtp(OcInferenceModel *m, OcMtpWeights *mtp);

/* Access the model config. */
const OcInferenceConfig *oc_inf_model_config(const OcInferenceModel *m);

/* Number of attention layers stored in the KV cache. */
size_t oc_inf_model_kv_layer_count(const OcInferenceModel *m);

/* KV row width (num_key_value_heads * kv_head_dim). */
size_t oc_inf_model_kv_row_len(const OcInferenceModel *m);

/* Check if model is loaded. */
bool oc_inf_model_is_loaded(const OcInferenceModel *m);

/* Check if continuous-batching decode is enabled. */
bool oc_inf_model_batched_decode_enabled(void);


/* Compute (q_head_dim, q_heads, kv_head_dim, kv_heads) from config, layer
 * norms, and projection output sizes. Mirrors Rust attention_head_dims(). */
void oc_attention_head_dims(const OcInferenceConfig *cfg,
                             const OcLayerWeights *layer,
                             size_t q_len, size_t kv_len,
                             uint32_t *out_q_head_dim,
                             uint32_t *out_q_heads,
                             uint32_t *out_kv_head_dim,
                             uint32_t *out_kv_heads);

/* GEMV for a single head's slice of a per-head weight matrix.
 * storage is logically [n_heads, rows, cols]; computes the head-th slice. */
OcError oc_gemv_weight_head(const OcWeightStorage *ws,
                             size_t rows, size_t cols,
                             uint32_t head, uint32_t n_heads,
                             const float *input, float *output);


/* Embed a token into workspace.x[..hidden_size].
 * Handles F32 and quantized embeddings, embedding_scale. */
void oc_inf_model_embed_token(OcInferenceModel *m, uint32_t token);

/* Read the current hidden state from workspace.x[..hidden_size]. */
const float *oc_inf_model_hidden_state(const OcInferenceModel *m);

/* Get config hidden_size. */
size_t oc_inf_model_config_hidden_size(const OcInferenceModel *m);

/* Overwrite hidden state with `hidden` (len must == hidden_size). */
OcError oc_inf_model_set_hidden_state(OcInferenceModel *m, const float *hidden, size_t len);

/* Apply final RMSNorm using model's norm_weight and rms_norm_eps. */
OcError oc_inf_model_apply_final_norm(const OcInferenceModel *m,
                                       const float *hidden, float *out, size_t len);

/* Get final norm weight pointer (read-only). */
const float *oc_inf_model_final_norm_weight(const OcInferenceModel *m);

/* Whether this model has a usable MTP/nextn draft block. */
bool oc_inf_model_has_mtp(const OcInferenceModel *m);

/* Number of nextn predict layers from config. */
size_t oc_inf_model_nextn_predict_layers(const OcInferenceModel *m);

/* Get last output hidden (final normed row for MTP). */
const float *oc_inf_model_last_output_hidden(const OcInferenceModel *m);

/* Configure which target layers are snapshotted for EAGLE3 feature fusion. */
OcError oc_inf_model_set_eagle3_capture_layers(OcInferenceModel *m,
                                                  const size_t *layers, size_t n);

/* Concatenate EAGLE3 captured hidden rows into a single flat vector.
 * Returns OC_ERR_NOT_FOUND if any capture layer is missing data.
 * out must have capacity >= n_capture_layers * hidden_size. */
OcError oc_inf_model_concat_eagle3_features(const OcInferenceModel *m,
                                             float *out, size_t out_len);

/* Project already-normalized hidden states through the output (lm_head) matrix.
 * normed: [hidden_size], logits: [vocab_size]. */
OcError oc_inf_model_lm_head_logits_from_normed(const OcInferenceModel *m,
                                                  const float *normed, size_t normed_len,
                                                  float *logits, size_t logits_len);

/* Apply final RMSNorm + lm_head to workspace.x and return logits.
 * Uses workspace.hidden_a for normed, workspace.logits for output.
 * Sets last_output_hidden. Returns logits pointer and length via out/out_len. */
OcError oc_inf_model_final_head_from_workspace(OcInferenceModel *m,
                                                 float **out, size_t *out_len);

/* Forward a single token through all layers (no logits).
 * Updates KV cache, workspace.x, and EAGLE3 captures.
 * position: absolute position of this token. */
OcError oc_inf_model_forward_token(OcInferenceModel *m, uint32_t token, size_t position);

/* Forward a single token through all layers and return logits.
 * Convenience: forward_token + final_head_from_workspace. */
OcError oc_inf_model_forward_token_logits(OcInferenceModel *m, uint32_t token,
                                            size_t position,
                                            float **logits, size_t *logits_len);

/* Run layers [start, end) against the hidden state in workspace.x.
 * Mutates workspace.x in place. pos is the absolute position for KV/RoPE.
 * Used by MTP draft path and pipeline-parallel stages. */
OcError oc_inf_model_run_layer_range(OcInferenceModel *m,
                                       size_t start, size_t end,
                                       size_t pos);


/* Generate draft tokens using the native MTP/nextn block. */
OcError oc_inf_model_draft_mtp_tokens(OcInferenceModel *m,
                                        uint32_t start_token,
                                        const float *start_hidden, size_t hidden_len,
                                        size_t max_tokens,
                                        uint32_t *out_tokens,
                                        float *out_logits,
                                        size_t *out_n);


/* Check if all layers support the batched GEMM path (no Mamba/SSM/MoE).
 * Returns true when the model can use forward_tokens / forward_batch. */
bool oc_inf_model_layers_supported_for_batched(const OcInferenceModel *m);

/* Batched prefill: process multiple tokens of ONE sequence via GEMM. */
OcError oc_inf_model_forward_tokens(OcInferenceModel *m,
                                      const uint32_t *tokens, size_t n_tokens,
                                      size_t start_pos, bool need_logits,
                                      float **out_logits, size_t *out_logits_len);

/* Cross-sequence batched decode: process N sequences (one token each) via GEMM. */
OcError oc_inf_model_forward_batch(OcInferenceModel *m,
                                     const uint32_t *tokens,
                                     const size_t *positions,
                                     OcSeqKv *kvs, size_t n_seqs,
                                     bool need_logits,
                                     float *out_logits);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_INF_MODEL_H */
