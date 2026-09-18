/*
 * lora.c — LoRA adapter inference implementation.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/lora.h"

#include "oxidize/flash_attention.h"
#include "oxidize/log.h"
#include "oxidize/safetensors.h"

#include <stdio.h>
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
    lm->shexp_gate_adapters = calloc(n_layers, sizeof(OcLoraAdapter));
    lm->shexp_up_adapters   = calloc(n_layers, sizeof(OcLoraAdapter));
    lm->shexp_down_adapters = calloc(n_layers, sizeof(OcLoraAdapter));
    lm->ssm_qkv_adapters    = calloc(n_layers, sizeof(OcLoraAdapter));
    lm->ssm_gate_adapters   = calloc(n_layers, sizeof(OcLoraAdapter));
    lm->ssm_out_adapters    = calloc(n_layers, sizeof(OcLoraAdapter));
    lm->ssm_alpha_adapters  = calloc(n_layers, sizeof(OcLoraAdapter));
    lm->ssm_beta_adapters   = calloc(n_layers, sizeof(OcLoraAdapter));
    if (!lm->q_adapters || !lm->k_adapters || !lm->v_adapters ||
        !lm->o_adapters || !lm->gate_adapters || !lm->up_adapters ||
        !lm->down_adapters || !lm->shexp_gate_adapters ||
        !lm->shexp_up_adapters || !lm->shexp_down_adapters ||
        !lm->ssm_qkv_adapters || !lm->ssm_gate_adapters ||
        !lm->ssm_out_adapters || !lm->ssm_alpha_adapters ||
        !lm->ssm_beta_adapters) {
        oc_lora_model_free(lm);
        return OC_ERR_OOM;
    }
    lm->active = false;
    lm->scale = 2.0f;
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
    } else if (strcmp(weight_name, "shexp_gate") == 0 ||
               strcmp(weight_name, "shared_expert.gate_proj") == 0) {
        target = &lm->shexp_gate_adapters[layer_idx];
    } else if (strcmp(weight_name, "shexp_up") == 0 ||
               strcmp(weight_name, "shared_expert.up_proj") == 0) {
        target = &lm->shexp_up_adapters[layer_idx];
    } else if (strcmp(weight_name, "shexp_down") == 0 ||
               strcmp(weight_name, "shared_expert.down_proj") == 0) {
        target = &lm->shexp_down_adapters[layer_idx];
    } else if (strcmp(weight_name, "ssm_qkv") == 0 ||
               strcmp(weight_name, "in_proj_qkv") == 0) {
        target = &lm->ssm_qkv_adapters[layer_idx];
    } else if (strcmp(weight_name, "ssm_gate") == 0 ||
               strcmp(weight_name, "in_proj_z") == 0) {
        target = &lm->ssm_gate_adapters[layer_idx];
    } else if (strcmp(weight_name, "ssm_out") == 0 ||
               strcmp(weight_name, "out_proj") == 0) {
        target = &lm->ssm_out_adapters[layer_idx];
    } else if (strcmp(weight_name, "ssm_alpha") == 0 ||
               strcmp(weight_name, "in_proj_a") == 0) {
        target = &lm->ssm_alpha_adapters[layer_idx];
    } else if (strcmp(weight_name, "ssm_beta") == 0 ||
               strcmp(weight_name, "in_proj_b") == 0) {
        target = &lm->ssm_beta_adapters[layer_idx];
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
    if (rank > lm->max_rank) lm->max_rank = rank;
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
        lm->gate_adapters, lm->up_adapters, lm->down_adapters,
        lm->shexp_gate_adapters, lm->shexp_up_adapters, lm->shexp_down_adapters,
        lm->ssm_qkv_adapters, lm->ssm_gate_adapters, lm->ssm_out_adapters,
        lm->ssm_alpha_adapters, lm->ssm_beta_adapters
    };
    for (size_t i = 0; i < sizeof(arrays) / sizeof(arrays[0]); i++) {
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

enum {
    OC_LORA_KIND_Q = 0,
    OC_LORA_KIND_K,
    OC_LORA_KIND_V,
    OC_LORA_KIND_O,
    OC_LORA_KIND_SHEXP_GATE,
    OC_LORA_KIND_SHEXP_UP,
    OC_LORA_KIND_SHEXP_DOWN,
    OC_LORA_KIND_SSM_QKV,
    OC_LORA_KIND_SSM_GATE,
    OC_LORA_KIND_SSM_OUT,
    OC_LORA_KIND_SSM_ALPHA,
    OC_LORA_KIND_SSM_BETA,
    OC_LORA_KIND_COUNT
};

static const char *lora_kind_name(int kind)
{
    switch (kind) {
    case OC_LORA_KIND_Q: return "q_proj";
    case OC_LORA_KIND_K: return "k_proj";
    case OC_LORA_KIND_V: return "v_proj";
    case OC_LORA_KIND_O: return "o_proj";
    case OC_LORA_KIND_SHEXP_GATE: return "shexp_gate";
    case OC_LORA_KIND_SHEXP_UP: return "shexp_up";
    case OC_LORA_KIND_SHEXP_DOWN: return "shexp_down";
    case OC_LORA_KIND_SSM_QKV: return "ssm_qkv";
    case OC_LORA_KIND_SSM_GATE: return "ssm_gate";
    case OC_LORA_KIND_SSM_OUT: return "ssm_out";
    case OC_LORA_KIND_SSM_ALPHA: return "ssm_alpha";
    case OC_LORA_KIND_SSM_BETA: return "ssm_beta";
    default: return NULL;
    }
}

static int lora_parse_kind(const char *rest)
{
    if (strstr(rest, "self_attn.q_proj")) return OC_LORA_KIND_Q;
    if (strstr(rest, "self_attn.k_proj")) return OC_LORA_KIND_K;
    if (strstr(rest, "self_attn.v_proj")) return OC_LORA_KIND_V;
    if (strstr(rest, "self_attn.o_proj")) return OC_LORA_KIND_O;
    if (strstr(rest, "shared_expert.gate_proj") || strstr(rest, "ffn_gate_shexp"))
        return OC_LORA_KIND_SHEXP_GATE;
    if (strstr(rest, "shared_expert.up_proj") || strstr(rest, "ffn_up_shexp"))
        return OC_LORA_KIND_SHEXP_UP;
    if (strstr(rest, "shared_expert.down_proj") || strstr(rest, "ffn_down_shexp"))
        return OC_LORA_KIND_SHEXP_DOWN;
    if (strstr(rest, "in_proj_qkv") || strstr(rest, "attn_qkv"))
        return OC_LORA_KIND_SSM_QKV;
    if (strstr(rest, "in_proj_z") || strstr(rest, "attn_gate"))
        return OC_LORA_KIND_SSM_GATE;
    if (strstr(rest, "out_proj") || strstr(rest, "ssm_out"))
        return OC_LORA_KIND_SSM_OUT;
    if (strstr(rest, "in_proj_a") || strstr(rest, "ssm_alpha"))
        return OC_LORA_KIND_SSM_ALPHA;
    if (strstr(rest, "in_proj_b") || strstr(rest, "ssm_beta"))
        return OC_LORA_KIND_SSM_BETA;
    return -1;
}

static bool lora_name_is_a(const char *name)
{
    size_t n = strlen(name);
    if (n >= 6 && strcmp(name + n - 6, "lora_A") == 0) return true;
    if (n >= 6 && strcmp(name + n - 6, "lora_a") == 0) return true;
    if (n >= 14 && strcmp(name + n - 14, "lora_A.weight") == 0) return true;
    if (n >= 14 && strcmp(name + n - 14, "lora_a.weight") == 0) return true;
    return false;
}

static bool lora_name_is_b(const char *name)
{
    size_t n = strlen(name);
    if (n >= 6 && strcmp(name + n - 6, "lora_B") == 0) return true;
    if (n >= 6 && strcmp(name + n - 6, "lora_b") == 0) return true;
    if (n >= 14 && strcmp(name + n - 14, "lora_B.weight") == 0) return true;
    if (n >= 14 && strcmp(name + n - 14, "lora_b.weight") == 0) return true;
    return false;
}

static OcError lora_copy_f32(const OcSafetensorsFile *st,
                             const OcSafetensorsTensor *t, float **out,
                             uint32_t *rows, uint32_t *cols)
{
    if (!t || t->n_dims != 2) return OC_ERR_FORMAT;
    const void *raw = NULL;
    OcError e = oc_safetensors_get_tensor_data(st, t, &raw);
    if (e != OC_OK) return e;
    uint32_t r = (uint32_t)t->shape[0];
    uint32_t c = (uint32_t)t->shape[1];
    size_t n = (size_t)r * c;
    float *buf = malloc(n * sizeof(float));
    if (!buf) return OC_ERR_OOM;
        if (strcmp(t->dtype, "F32") == 0) {
        memcpy(buf, raw, n * sizeof(float));
    } else if (strcmp(t->dtype, "F16") == 0) {
        const uint16_t *h = (const uint16_t *)raw;
        for (size_t i = 0; i < n; i++) buf[i] = oc_f16_to_f32_bits(h[i]);
    } else if (strcmp(t->dtype, "BF16") == 0) {
        const uint16_t *h = (const uint16_t *)raw;
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = ((uint32_t)h[i]) << 16;
            memcpy(&buf[i], &bits, sizeof(float));
        }
    } else {
        free(buf);
        return OC_ERR_FORMAT;
    }
    *out = buf;
    *rows = r;
    *cols = c;
    return OC_OK;
}

typedef struct {
    float *a;
    float *b;
    uint32_t a_rows, a_cols, b_rows, b_cols;
} OcLoraPending;

static void lora_pending_free(OcLoraPending *pend, size_t nslot)
{
    if (!pend) return;
    for (size_t i = 0; i < nslot; i++) {
        free(pend[i].a);
        free(pend[i].b);
    }
    free(pend);
}

OcError oc_lora_load_safetensors(OcLoraModel *lm, const char *path, float scale)
{
    if (!lm || !path) return OC_ERR_INVALID_ARG;
    if (scale <= 0.0f) scale = 2.0f;
    lm->scale = scale;

    OcSafetensorsFile st;
    OcError e = oc_safetensors_open(path, &st);
    if (e != OC_OK) return e;

    size_t nslot = lm->n_layers * (size_t)OC_LORA_KIND_COUNT;
    OcLoraPending *pend = calloc(nslot, sizeof(*pend));
    if (!pend) {
        oc_safetensors_close(&st);
        return OC_ERR_OOM;
    }

    uint32_t loaded = 0;
    for (size_t i = 0; i < st.n_tensors; i++) {
        const OcSafetensorsTensor *t = &st.tensors[i];
        unsigned layer = 0;
        char rest[96];
        rest[0] = '\0';
        if (sscanf(t->name, "language_model.model.layers.%u.%95s", &layer, rest) != 2 &&
            sscanf(t->name, "model.layers.%u.%95s", &layer, rest) != 2 &&
            sscanf(t->name, "layers.%u.%95s", &layer, rest) != 2) {
            continue;
        }
        if ((size_t)layer >= lm->n_layers) continue;
        int kind = lora_parse_kind(rest);
        if (kind < 0) continue;
        OcLoraPending *slot = &pend[(size_t)layer * OC_LORA_KIND_COUNT + (size_t)kind];
        float *buf = NULL;
        uint32_t rows = 0, cols = 0;
        e = lora_copy_f32(&st, t, &buf, &rows, &cols);
        if (e != OC_OK) {
            lora_pending_free(pend, nslot);
            oc_safetensors_close(&st);
            return e;
        }
        if (lora_name_is_a(t->name)) {
            free(slot->a);
            slot->a = buf;
            slot->a_rows = rows;
            slot->a_cols = cols;
        } else if (lora_name_is_b(t->name)) {
            free(slot->b);
            slot->b = buf;
            slot->b_rows = rows;
            slot->b_cols = cols;
        } else {
            free(buf);
        }
    }
    oc_safetensors_close(&st);

    for (size_t layer = 0; layer < lm->n_layers; layer++) {
        for (int kind = 0; kind < OC_LORA_KIND_COUNT; kind++) {
            OcLoraPending *slot = &pend[layer * OC_LORA_KIND_COUNT + (size_t)kind];
            if (!slot->a || !slot->b) {
                free(slot->a);
                free(slot->b);
                continue;
            }
            uint32_t rank = slot->a_rows;
            if (slot->b_cols != rank) {
                lora_pending_free(pend, nslot);
                return OC_ERR_TENSOR;
            }
            e = oc_lora_set_adapter(lm, layer, lora_kind_name(kind),
                                    slot->a, slot->b, rank,
                                    slot->b_rows, slot->a_cols, scale);
            if (e != OC_OK) {
                lora_pending_free(pend, nslot);
                return e;
            }
            slot->a = NULL;
            slot->b = NULL;
            loaded++;
        }
    }
    lora_pending_free(pend, nslot);
    if (loaded == 0) return OC_ERR_MODEL;
    oc_log(OC_LOG_INFO, "lora: loaded %u adapters from %s (scale=%.3f, max_rank=%u)",
           loaded, path, (double)scale, lm->max_rank);
    return OC_OK;
}

/* ─── LoRA plan (auto-matching) ─────────────────────────────────────── */

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
