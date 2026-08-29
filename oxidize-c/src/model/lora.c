/*
 * lora.c — LoRA adapter inference implementation.
 */
#define _POSIX_C_SOURCE 200809L
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


/* Check if a string ends with suffix. Returns pointer to base if yes, NULL if no. */
static const char *strip_suffix(const char *s, const char *suffix, size_t *out_base_len)
{
    size_t sl = strlen(s);
    size_t sul = strlen(suffix);
    if (sl < sul) return NULL;
    if (strcmp(s + sl - sul, suffix) != 0) return NULL;
    *out_base_len = sl - sul;
    return s;
}

OcLoraPlanError oc_lora_plan_application(const char *const *base_tensor_names,
                                          size_t n_base,
                                          const char *const *adapter_tensor_names,
                                          size_t n_adapter,
                                          int base_qtype,
                                          OcLoraPlan *out_plan)
{
    if (!out_plan)
        return OC_LORA_PLAN_INVALID_ARG;

    memset(out_plan, 0, sizeof(*out_plan));

    /* Determine adapter kind. */
    /* base_qtype: 0 = F32/F16/unknown -> Lora, anything else -> Qlora */
    out_plan->kind = (base_qtype == 0) ? OC_ADAPTER_LORA : OC_ADAPTER_QLORA;

    /* Collect lora_a and lora_b base names. */
    /* We use simple arrays since we don't have a hashmap. */
    const char *a_bases[256];
    const char *a_names[256];
    size_t n_a = 0;
    const char *b_bases[256];
    const char *b_names[256];
    size_t n_b = 0;

    const char *suffix_a = ".lora_a.weight";
    const char *suffix_b = ".lora_b.weight";

    for (size_t i = 0; i < n_adapter; i++) {
        if (!adapter_tensor_names[i]) continue;
        size_t base_len;
        const char *base = strip_suffix(adapter_tensor_names[i], suffix_a, &base_len);
        if (base) {
            if (n_a >= 256) return OC_LORA_PLAN_INVALID_ARG;
            /* Check for duplicate. */
            for (size_t j = 0; j < n_a; j++) {
                size_t existing_len = 0;
                strip_suffix(a_names[j], suffix_a, &existing_len);
                if (existing_len == base_len &&
                    strncmp(a_bases[j], base, base_len) == 0)
                    return OC_LORA_PLAN_DUPLICATE_PAIR;
            }
            a_bases[n_a] = base;
            a_names[n_a] = adapter_tensor_names[i];
            n_a++;
            continue;
        }
        base = strip_suffix(adapter_tensor_names[i], suffix_b, &base_len);
        if (base) {
            if (n_b >= 256) return OC_LORA_PLAN_INVALID_ARG;
            for (size_t j = 0; j < n_b; j++) {
                size_t existing_len = 0;
                strip_suffix(b_names[j], suffix_b, &existing_len);
                if (existing_len == base_len &&
                    strncmp(b_bases[j], base, base_len) == 0)
                    return OC_LORA_PLAN_DUPLICATE_PAIR;
            }
            b_bases[n_b] = base;
            b_names[n_b] = adapter_tensor_names[i];
            n_b++;
        }
    }

    /* Match pairs: for each a_base, find matching b_base. */
    /* Collect unique base names from both a and b. */
    /* We iterate a and find matching b. */
    size_t max_targets = n_a + n_b;
    if (max_targets == 0) {
        out_plan->targets = NULL;
        out_plan->n_targets = 0;
        out_plan->missing_base_tensors = NULL;
        out_plan->n_missing = 0;
        return OC_LORA_PLAN_OK;
    }

    OcLoraTarget *targets = calloc(max_targets, sizeof(OcLoraTarget));
    if (!targets) return OC_LORA_PLAN_INVALID_ARG;
    size_t n_targets = 0;

    /* For each lora_a, find the matching lora_b. */
    for (size_t i = 0; i < n_a; i++) {
        size_t a_base_len = 0;
        strip_suffix(a_names[i], suffix_a, &a_base_len);
        const char *a_base = a_bases[i];

        bool found_b = false;
        for (size_t j = 0; j < n_b; j++) {
            size_t b_base_len = 0;
            strip_suffix(b_names[j], suffix_b, &b_base_len);
            if (b_base_len == a_base_len &&
                strncmp(b_bases[j], a_base, a_base_len) == 0) {
                /* Found pair. */
                char *base_str = malloc(a_base_len + 1);
                memcpy(base_str, a_base, a_base_len);
                base_str[a_base_len] = '\0';

                targets[n_targets].base_tensor = base_str;
                targets[n_targets].lora_a_tensor = strdup(a_names[i]);
                targets[n_targets].lora_b_tensor = strdup(b_names[j]);
                n_targets++;
                found_b = true;
                break;
            }
        }
        if (!found_b) {
            free(targets);
            return OC_LORA_PLAN_MISSING_PAIR_FOR_A;
        }
    }

    /* Check for lora_b without matching lora_a. */
    for (size_t j = 0; j < n_b; j++) {
        size_t b_base_len = 0;
        strip_suffix(b_names[j], suffix_b, &b_base_len);
        const char *b_base = b_bases[j];

        bool found_a = false;
        for (size_t i = 0; i < n_a; i++) {
            size_t a_base_len = 0;
            strip_suffix(a_names[i], suffix_a, &a_base_len);
            if (a_base_len == b_base_len &&
                strncmp(a_bases[i], b_base, b_base_len) == 0) {
                found_a = true;
                break;
            }
        }
        if (!found_a) {
            free(targets);
            return OC_LORA_PLAN_MISSING_PAIR_FOR_B;
        }
    }

    /* Find missing base tensors. */
    char **missing = calloc(n_targets + 1, sizeof(char *));
    size_t n_missing = 0;
    for (size_t i = 0; i < n_targets; i++) {
        bool found = false;
        for (size_t j = 0; j < n_base; j++) {
            if (base_tensor_names[j] &&
                strcmp(base_tensor_names[j], targets[i].base_tensor) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            missing[n_missing] = strdup(targets[i].base_tensor);
            n_missing++;
        }
    }

    out_plan->targets = targets;
    out_plan->n_targets = n_targets;
    out_plan->missing_base_tensors = missing;
    out_plan->n_missing = n_missing;
    return OC_LORA_PLAN_OK;
}

void oc_lora_plan_free(OcLoraPlan *plan)
{
    if (!plan) return;
    for (size_t i = 0; i < plan->n_targets; i++) {
        free(plan->targets[i].base_tensor);
        free(plan->targets[i].lora_a_tensor);
        free(plan->targets[i].lora_b_tensor);
    }
    free(plan->targets);
    for (size_t i = 0; i < plan->n_missing; i++)
        free(plan->missing_base_tensors[i]);
    free(plan->missing_base_tensors);
    memset(plan, 0, sizeof(*plan));
}
