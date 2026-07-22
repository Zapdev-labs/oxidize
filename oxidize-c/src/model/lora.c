/*
 * lora.c — LoRA adapter inference implementation.
 */
#include "oxidize/lora.h"

#include <stdlib.h>
#include <string.h>

OcError oc_lora_model_init(OcLoraModel *lm, size_t n_layers)
{
    if (!lm) return OC_ERR_INVALID_ARG;
    memset(lm, 0, sizeof(*lm));
    lm->n_layers = n_layers;
    lm->q_adapters    = calloc(n_layers, sizeof(OcLoraAdapter));
    lm->k_adapters    = calloc(n_layers, sizeof(OcLoraAdapter));
    lm->v_adapters    = calloc(n_layers, sizeof(OcLoraAdapter));
    lm->o_adapters    = calloc(n_layers, sizeof(OcLoraAdapter));
    lm->gate_adapters = calloc(n_layers, sizeof(OcLoraAdapter));
    lm->up_adapters   = calloc(n_layers, sizeof(OcLoraAdapter));
    lm->down_adapters = calloc(n_layers, sizeof(OcLoraAdapter));
    if (!lm->q_adapters || !lm->k_adapters || !lm->v_adapters ||
        !lm->o_adapters || !lm->gate_adapters || !lm->up_adapters ||
        !lm->down_adapters) {
        oc_lora_model_free(lm);
        return OC_ERR_OOM;
    }
    lm->active = false;
    return OC_OK;
}

OcError oc_lora_set_adapter(OcLoraModel *lm, size_t layer_idx,
                            const char *weight_name,
                            float *a, float *b,
                            uint32_t rank, uint32_t rows, uint32_t cols,
                            float alpha)
{
    if (!lm || !weight_name || layer_idx >= lm->n_layers)
        return OC_ERR_INVALID_ARG;

    OcLoraAdapter *target = NULL;
    if (strcmp(weight_name, "q_proj") == 0 || strcmp(weight_name, "attn_q") == 0) {
        target = &lm->q_adapters[layer_idx];
    } else if (strcmp(weight_name, "k_proj") == 0 || strcmp(weight_name, "attn_k") == 0) {
        target = &lm->k_adapters[layer_idx];
    } else if (strcmp(weight_name, "v_proj") == 0 || strcmp(weight_name, "attn_v") == 0) {
        target = &lm->v_adapters[layer_idx];
    } else if (strcmp(weight_name, "o_proj") == 0 || strcmp(weight_name, "attn_output") == 0) {
        target = &lm->o_adapters[layer_idx];
    } else if (strcmp(weight_name, "gate_proj") == 0 || strcmp(weight_name, "ffn_gate") == 0) {
        target = &lm->gate_adapters[layer_idx];
    } else if (strcmp(weight_name, "up_proj") == 0 || strcmp(weight_name, "ffn_up") == 0) {
        target = &lm->up_adapters[layer_idx];
    } else if (strcmp(weight_name, "down_proj") == 0 || strcmp(weight_name, "ffn_down") == 0) {
        target = &lm->down_adapters[layer_idx];
    } else {
        return OC_ERR_INVALID_ARG;
    }

    /* Free previous adapter if any. */
    free(target->a);
    free(target->b);

    target->a     = a;
    target->b     = b;
    target->rank  = rank;
    target->rows  = rows;
    target->cols  = cols;
    target->alpha = alpha;
    lm->active = true;
    return OC_OK;
}

void oc_lora_apply(const OcLoraAdapter *adapter,
                   const float *x, float *out, float *temp)
{
    if (!adapter || !adapter->a || !adapter->b || !x || !out || !temp)
        return;

    float scale = adapter->alpha;
    if (scale == 0.0f) scale = (float)adapter->rank;

    /* Step 1: temp = A @ x  (temp has length `rank`)
     * A is [rank, cols], x is [cols]. */
    for (uint32_t r = 0; r < adapter->rank; r++) {
        const float *a_row = adapter->a + (size_t)r * adapter->cols;
        float dot = 0.0f;
        for (uint32_t c = 0; c < adapter->cols; c++) {
            dot += a_row[c] * x[c];
        }
        temp[r] = dot;
    }

    /* Step 2: out += scale * B @ temp  (out has length `rows`)
     * B is [rows, rank], temp is [rank]. */
    for (uint32_t r = 0; r < adapter->rows; r++) {
        const float *b_row = adapter->b + (size_t)r * adapter->rank;
        float dot = 0.0f;
        for (uint32_t k = 0; k < adapter->rank; k++) {
            dot += b_row[k] * temp[k];
        }
        out[r] += scale * dot;
    }
}

void oc_lora_model_free(OcLoraModel *lm)
{
    if (!lm) return;
    OcLoraAdapter *arrays[] = {
        lm->q_adapters, lm->k_adapters, lm->v_adapters, lm->o_adapters,
        lm->gate_adapters, lm->up_adapters, lm->down_adapters
    };
    for (size_t i = 0; i < 7; i++) {
        if (arrays[i]) {
            for (size_t l = 0; l < lm->n_layers; l++) {
                free(arrays[i][l].a);
                free(arrays[i][l].b);
            }
            free(arrays[i]);
        }
    }
    memset(lm, 0, sizeof(*lm));
}

bool oc_lora_is_active(const OcLoraModel *lm)
{
    return lm ? lm->active : false;
}
