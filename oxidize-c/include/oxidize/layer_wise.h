/*
 * layer_wise.h — Layer-wise inference for large models.
 *
 * Enables running models larger than available RAM by loading layers
 * one at a time, computing activations, then freeing the layer weights.
 * Port from oxidize-core/src/model/layer_wise.rs.
 */
#ifndef OXIDIZE_LAYER_WISE_H
#define OXIDIZE_LAYER_WISE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_LW_MAX_LAYERS 256
#define OC_LW_MAX_TENSORS 32

typedef struct {
    uint32_t layer_idx;
    uint64_t weight_size;
    bool loaded;
    void *weight_data;
    char tensor_names[OC_LW_MAX_TENSORS][128];
    uint32_t n_tensors;
} OcLwLayerState;

typedef struct {
    uint32_t n_layers;
    uint32_t current_layer;
    uint64_t total_weight_size;
    uint64_t available_memory;
    uint32_t max_concurrent_layers;
    OcLwLayerState layers[OC_LW_MAX_LAYERS];
    bool initialized;
} OcLayerWiseState;

typedef struct {
    uint32_t n_layers;
    uint64_t available_memory;
    uint32_t max_concurrent_layers;
    bool offload_to_disk;
} OcLayerWiseConfig;

OcError oc_lw_config_init(OcLayerWiseConfig *cfg);
OcError oc_lw_state_init(OcLayerWiseState *state, const OcLayerWiseConfig *cfg);
OcError oc_lw_register_layer(OcLayerWiseState *state, uint32_t layer_idx,
                            uint64_t weight_size);
OcError oc_lw_register_tensor(OcLayerWiseState *state, uint32_t layer_idx,
                             const char *tensor_name);
OcError oc_lw_load_layer(OcLayerWiseState *state, uint32_t layer_idx);
OcError oc_lw_unload_layer(OcLayerWiseState *state, uint32_t layer_idx);
OcError oc_lw_get_layer(const OcLayerWiseState *state, uint32_t layer_idx,
                       const OcLwLayerState **out);
uint32_t oc_lw_n_loaded(const OcLayerWiseState *state);
uint64_t oc_lw_loaded_bytes(const OcLayerWiseState *state);
uint32_t oc_lw_n_layers(const OcLayerWiseState *state);
OcError oc_lw_advance(OcLayerWiseState *state);
uint32_t oc_lw_current_layer(const OcLayerWiseState *state);
void oc_lw_state_free(OcLayerWiseState *state);

/* ─── Layer-wise model forward (port of layer_wise/forward.rs) ──────── */

/* Forward a single token through the model, loading/unloading layers
 * one at a time to stay within memory budget.
 *
 * This wraps OcInferenceModel's forward_token with layer lifecycle:
 * 1. Embed token
 * 2. For each layer: ensure loaded, run layer, optionally unload
 * 3. Apply final norm + LM head
 *
 * Parameters:
 *   state    - layer-wise state tracking which layers are loaded
 *   model    - the inference model (provides weights + forward)
 *   token    - token ID to forward
 *   pos      - position in sequence (for RoPE)
 *   logits   - output logits buffer [vocab_size]
 *
 * Returns OC_OK on success. */
OcError oc_lw_forward_single(OcLayerWiseState *state,
                              void *model,
                              uint32_t token,
                              size_t pos,
                              float *logits);

/* Forward an already-normed hidden state through layers [start, end).
 * This is the layer-wise variant: it processes layers one at a time,
 * loading each layer before computing and unloading after (if memory
 * budget requires). The hidden state is modified in-place.
 *
 * Parameters:
 *   state        - layer-wise state
 *   model        - inference model (for layer weights)
 *   hidden       - hidden state [hidden_size], modified in-place
 *   start_layer  - first layer index
 *   end_layer    - one past last layer
 *   pos          - position in sequence */
OcError oc_lw_forward_normed_hidden(OcLayerWiseState *state,
                                     void *model,
                                     float *hidden,
                                     size_t start_layer,
                                     size_t end_layer,
                                     size_t pos);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_LAYER_WISE_H */
