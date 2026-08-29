/* lora.h — LoRA (Low-Rank Adaptation) adapter inference support. */
#ifndef OXIDIZE_LORA_H
#define OXIDIZE_LORA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

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

/* Apply LoRA delta to an activation vector. */
void oc_lora_apply(const OcLoraAdapter *adapter,
                   const float *x, float *out, float *temp);

/* Free a LoRA model and all its adapters. */
void oc_lora_model_free(OcLoraModel *lm);

/* Check if any adapters are loaded. */
bool oc_lora_is_active(const OcLoraModel *lm);


/* Whether the adapter is LoRA (f32/f16 base) or QLoRA (quantized base). */
typedef enum OcAdapterKind {
    OC_ADAPTER_LORA = 0,
    OC_ADAPTER_QLORA = 1,
} OcAdapterKind;

/* A matched (base, lora_a, lora_b) triple. */
typedef struct OcLoraTarget {
    char *base_tensor;    /* e.g. "blk.0.attn_q.weight" */
    char *lora_a_tensor;  /* e.g. "blk.0.attn_q.weight.lora_a.weight" */
    char *lora_b_tensor;  /* e.g. "blk.0.attn_q.weight.lora_b.weight" */
} OcLoraTarget;

/* A plan for applying LoRA adapters to a base model. */
typedef struct OcLoraPlan {
    OcAdapterKind kind;
    OcLoraTarget *targets;
    size_t n_targets;
    char **missing_base_tensors;
    size_t n_missing;
} OcLoraPlan;

/* Error codes for LoRA planning. */
typedef enum OcLoraPlanError {
    OC_LORA_PLAN_OK = 0,
    OC_LORA_PLAN_MISSING_PAIR_FOR_A = 1,  /* lora_a found but no lora_b */
    OC_LORA_PLAN_MISSING_PAIR_FOR_B = 2,  /* lora_b found but no lora_a */
    OC_LORA_PLAN_DUPLICATE_PAIR = 3,
    OC_LORA_PLAN_INVALID_ARG = 4,
} OcLoraPlanError;

/* Auto-match `.lora_a.weight` / `.lora_b.weight` adapter tensor pairs */
OcLoraPlanError oc_lora_plan_application(const char *const *base_tensor_names,
                                          size_t n_base,
                                          const char *const *adapter_tensor_names,
                                          size_t n_adapter,
                                          int base_qtype,
                                          OcLoraPlan *out_plan);

/* Free a LoRA plan. */
void oc_lora_plan_free(OcLoraPlan *plan);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_LORA_H */
