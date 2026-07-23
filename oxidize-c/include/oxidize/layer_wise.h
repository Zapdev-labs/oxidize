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

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_LAYER_WISE_H */
