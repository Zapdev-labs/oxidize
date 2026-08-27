/*
 * llama_swap.h — Llama-family model swap support.
 *
 * Enables hot-swapping between different quantized models without
 * re-initializing the inference engine. Useful for A/B testing
 * different quantization levels.
 */
#ifndef OXIDIZE_LLAMA_SWAP_H
#define OXIDIZE_LLAMA_SWAP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ──────────────────────────────────────────────────────── */

#define OC_SWAP_MAX_MODELS 8
#define OC_SWAP_MAX_PATH 512
#define OC_SWAP_MAX_NAME 128

/* ─── Types ─────────────────────────────────────────────────────────── */

typedef struct OcSwapModelEntry {
    char path[OC_SWAP_MAX_PATH];
    char name[OC_SWAP_MAX_NAME];
    uint64_t size_bytes;
    char quant_type[32];
    bool loaded;
    void *model_data;       /* opaque pointer to loaded model state */
    size_t model_data_size;
} OcSwapModelEntry;

typedef struct OcModelSwap {
    OcSwapModelEntry models[OC_SWAP_MAX_MODELS];
    size_t n_models;
    int active_idx;
    uint64_t total_loaded_bytes;
} OcModelSwap;

/* ─── API ────────────────────────────────────────────────────────────── */

/* Initialize the swap manager. */
OcError oc_model_swap_init(OcModelSwap *sw);

/* Register a model by path. Returns the model index. */
OcError oc_model_swap_register(OcModelSwap *sw, const char *path,
                               const char *name, int *out_idx);

/* Load a model into memory (if not already loaded). */
OcError oc_model_swap_load(OcModelSwap *sw, int idx);

/* Unload a model from memory (free its data). */
OcError oc_model_swap_unload(OcModelSwap *sw, int idx);

/* Switch the active model. Loads the target if needed, may unload others
 * if memory is constrained. */
OcError oc_model_swap_activate(OcModelSwap *sw, int idx);

/* Get the active model index (-1 if none). */
int oc_model_swap_active(const OcModelSwap *sw);

/* Get info about a registered model. */
OcError oc_model_swap_info(const OcModelSwap *sw, int idx,
                           const OcSwapModelEntry **out_info);

/* List all registered models. Returns count. */
size_t oc_model_swap_list(const OcModelSwap *sw,
                          const OcSwapModelEntry **out_array);

/* Total bytes of loaded model data. */
uint64_t oc_model_swap_loaded_bytes(const OcModelSwap *sw);

/* Free all resources. */
void oc_model_swap_free(OcModelSwap *sw);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_LLAMA_SWAP_H */
