/*
 * layer_wise.c — Layer-wise inference implementation.
 */
#include "oxidize/layer_wise.h"

#include <stdlib.h>
#include <string.h>

OcError oc_lw_config_init(OcLayerWiseConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_layers = 32;
    cfg->available_memory = 8ULL * 1024 * 1024 * 1024; /* 8 GB */
    cfg->max_concurrent_layers = 4;
    cfg->offload_to_disk = false;
    return OC_OK;
}

OcError oc_lw_state_init(OcLayerWiseState *state, const OcLayerWiseConfig *cfg)
{
    if (!state) return OC_ERR_INVALID_ARG;
    memset(state, 0, sizeof(*state));
    if (cfg) {
        state->n_layers = cfg->n_layers;
        state->available_memory = cfg->available_memory;
        state->max_concurrent_layers = cfg->max_concurrent_layers;
    } else {
        OcLayerWiseConfig default_cfg;
        oc_lw_config_init(&default_cfg);
        state->n_layers = default_cfg.n_layers;
        state->available_memory = default_cfg.available_memory;
        state->max_concurrent_layers = default_cfg.max_concurrent_layers;
    }
    if (state->n_layers > OC_LW_MAX_LAYERS) state->n_layers = OC_LW_MAX_LAYERS;
    state->current_layer = 0;
    state->initialized = true;
    return OC_OK;
}

OcError oc_lw_register_layer(OcLayerWiseState *state, uint32_t layer_idx,
                            uint64_t weight_size)
{
    if (!state || layer_idx >= state->n_layers) return OC_ERR_INVALID_ARG;
    OcLwLayerState *l = &state->layers[layer_idx];
    l->layer_idx = layer_idx;
    l->weight_size = weight_size;
    l->loaded = false;
    l->weight_data = NULL;
    l->n_tensors = 0;
    state->total_weight_size += weight_size;
    return OC_OK;
}

OcError oc_lw_register_tensor(OcLayerWiseState *state, uint32_t layer_idx,
                             const char *tensor_name)
{
    if (!state || !tensor_name || layer_idx >= state->n_layers)
        return OC_ERR_INVALID_ARG;
    OcLwLayerState *l = &state->layers[layer_idx];
    if (l->n_tensors >= OC_LW_MAX_TENSORS) return OC_ERR_OOM;
    size_t nlen = strlen(tensor_name);
    if (nlen >= sizeof(l->tensor_names[l->n_tensors]))
        nlen = sizeof(l->tensor_names[l->n_tensors]) - 1;
    memcpy(l->tensor_names[l->n_tensors], tensor_name, nlen);
    l->tensor_names[l->n_tensors][nlen] = '\0';
    l->n_tensors++;
    return OC_OK;
}

OcError oc_lw_load_layer(OcLayerWiseState *state, uint32_t layer_idx)
{
    if (!state || layer_idx >= state->n_layers) return OC_ERR_INVALID_ARG;
    OcLwLayerState *l = &state->layers[layer_idx];
    if (l->loaded) return OC_OK;

    /* Check if we need to evict layers. */
    uint32_t n_loaded = oc_lw_n_loaded(state);
    while (n_loaded >= state->max_concurrent_layers) {
        /* Find the oldest loaded layer (furthest from current). */
        uint32_t victim = 0;
        uint32_t max_dist = 0;
        for (uint32_t i = 0; i < state->n_layers; i++) {
            if (state->layers[i].loaded) {
                uint32_t dist = (state->current_layer > i)
                    ? (state->current_layer - i) : (i - state->current_layer);
                if (dist > max_dist) { max_dist = dist; victim = i; }
            }
        }
        oc_lw_unload_layer(state, victim);
        n_loaded--;
    }

    /* Allocate weight data (simulated). */
    l->weight_data = malloc(l->weight_size > 0 ? l->weight_size : 1);
    if (!l->weight_data) return OC_ERR_OOM;
    memset(l->weight_data, 0, l->weight_size > 0 ? l->weight_size : 1);
    l->loaded = true;
    return OC_OK;
}

OcError oc_lw_unload_layer(OcLayerWiseState *state, uint32_t layer_idx)
{
    if (!state || layer_idx >= state->n_layers) return OC_ERR_INVALID_ARG;
    OcLwLayerState *l = &state->layers[layer_idx];
    if (!l->loaded) return OC_OK;
    free(l->weight_data);
    l->weight_data = NULL;
    l->loaded = false;
    return OC_OK;
}

OcError oc_lw_get_layer(const OcLayerWiseState *state, uint32_t layer_idx,
                       const OcLwLayerState **out)
{
    if (!state || !out || layer_idx >= state->n_layers) return OC_ERR_INVALID_ARG;
    *out = &state->layers[layer_idx];
    return OC_OK;
}

uint32_t oc_lw_n_loaded(const OcLayerWiseState *state)
{
    if (!state) return 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < state->n_layers; i++)
        if (state->layers[i].loaded) count++;
    return count;
}

uint64_t oc_lw_loaded_bytes(const OcLayerWiseState *state)
{
    if (!state) return 0;
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < state->n_layers; i++)
        if (state->layers[i].loaded) bytes += state->layers[i].weight_size;
    return bytes;
}

uint32_t oc_lw_n_layers(const OcLayerWiseState *state)
{
    return state ? state->n_layers : 0;
}

OcError oc_lw_advance(OcLayerWiseState *state)
{
    if (!state || !state->initialized) return OC_ERR_INVALID_ARG;
    /* Unload current layer, load next. */
    if (state->current_layer < state->n_layers) {
        oc_lw_unload_layer(state, state->current_layer);
    }
    state->current_layer++;
    if (state->current_layer >= state->n_layers) {
        state->current_layer = state->n_layers; /* done */
        return OC_OK;
    }
    return oc_lw_load_layer(state, state->current_layer);
}

uint32_t oc_lw_current_layer(const OcLayerWiseState *state)
{
    return state ? state->current_layer : 0;
}

void oc_lw_state_free(OcLayerWiseState *state)
{
    if (!state) return;
    for (uint32_t i = 0; i < state->n_layers; i++)
        oc_lw_unload_layer(state, i);
    memset(state, 0, sizeof(*state));
}
