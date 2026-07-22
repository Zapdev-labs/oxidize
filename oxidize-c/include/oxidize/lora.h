/*
 * lora.h — LoRA (Low-Rank Adaptation) adapter inference support.
 *
 * Applies LoRA adapters on top of base model weights during inference.
 * A LoRA adapter consists of low-rank matrices A and B for each target
 * weight, where the effective weight is: W_eff = W + alpha * B @ A.
 *
 * During forward pass, the adapter is applied by computing the delta
 * activation: delta_x = alpha * B @ (A @ x), and adding it to the base
 * output. This avoids modifying the base weights.
 */
#ifndef OXIDIZE_LORA_H
#define OXIDIZE_LORA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/llama.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A single LoRA adapter for one weight matrix. */
typedef struct OcLoraAdapter {
    /* Base weight shape: [rows, cols]. A is [rank, cols], B is [rows, rank]. */
    float *a;               /* [rank, cols] — down-projection           */
    float *b;               /* [rows, rank] — up-projection             */
    uint32_t rank;          /* LoRA rank (typically 8-64)                */
    uint32_t rows;          /* must match the base weight's rows          */
    uint32_t cols;          /* must match the base weight's cols          */
    float   alpha;          /* scaling factor (default: rank)             */
} OcLoraAdapter;

/* A collection of LoRA adapters for all layers in a model. */
typedef struct OcLoraModel {
    OcLoraAdapter *q_adapters;    /* per-layer q_proj adapters       */
    OcLoraAdapter *k_adapters;    /* per-layer k_proj adapters       */
    OcLoraAdapter *v_adapters;    /* per-layer v_proj adapters       */
    OcLoraAdapter *o_adapters;    /* per-layer o_proj adapters       */
    OcLoraAdapter *gate_adapters; /* per-layer ffn_gate adapters     */
    OcLoraAdapter *up_adapters;   /* per-layer ffn_up adapters       */
    OcLoraAdapter *down_adapters; /* per-layer ffn_down adapters     */
    size_t n_layers;
    bool active;
} OcLoraModel;

/* Initialize a LoRA model with `n_layers` layers. All adapters start NULL. */
OcError oc_lora_model_init(OcLoraModel *lm, size_t n_layers);

/* Set a single adapter for a specific layer and weight type.
 * Takes ownership of `a` and `b` (freed on oc_lora_model_free). */
OcError oc_lora_set_adapter(OcLoraModel *lm, size_t layer_idx,
                            const char *weight_name,
                            float *a, float *b,
                            uint32_t rank, uint32_t rows, uint32_t cols,
                            float alpha);

/* Apply LoRA delta to an activation vector.
 * Given the base activation `x` (length `cols`), computes:
 *   delta = alpha * B @ (A @ x)
 * and adds it to `out` (length `rows`).
 *
 * `temp` is a scratch buffer of at least `rank` floats. */
void oc_lora_apply(const OcLoraAdapter *adapter,
                   const float *x, float *out, float *temp);

/* Free a LoRA model and all its adapters. */
void oc_lora_model_free(OcLoraModel *lm);

/* Check if any adapters are loaded. */
bool oc_lora_is_active(const OcLoraModel *lm);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_LORA_H */
