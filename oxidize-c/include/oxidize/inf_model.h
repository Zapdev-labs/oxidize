/*
 * inf_model.h — Full inference model struct.
 *
 * Port of oxidize-core/src/model/inference.rs::InferenceModel.
 *
 * Bundles all model state: config, token embeddings, output head, per-layer
 * weights, MTP weights, KV cache, SSM state, workspace, and EAGLE3 capture.
 */
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

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_INF_MODEL_H */
