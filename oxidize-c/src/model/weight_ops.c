#define _POSIX_C_SOURCE 200809L
#include "oxidize/weight_ops.h"
#include "oxidize/quant.h"
#include "oxidize/activation.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>


OcError oc_gemv_f32(const float *weights, size_t rows, size_t cols,
                     const float *input, float *output)
{
    if (!weights || !input || !output) return OC_ERR_INVALID_ARG;
    if (rows == 0 || cols == 0) return OC_ERR_INVALID_ARG;

    for (size_t j = 0; j < rows; j++) {
        float sum = 0.0f;
        const float *row = &weights[j * cols];
        for (size_t i = 0; i < cols; i++)
            sum += row[i] * input[i];
        output[j] = sum;
    }
    return OC_OK;
}


OcError oc_gemv_weight(const OcWeightStorage *ws,
                       size_t rows, size_t cols,
                       const float *input, float *output)
{
    if (!ws || !input || !output) return OC_ERR_INVALID_ARG;
    if (rows == 0 || cols == 0) return OC_ERR_INVALID_ARG;

    switch (ws->type) {
    case OC_WEIGHT_F32:
        if (ws->f32_len < rows * cols) return OC_ERR_INVALID_ARG;
        return oc_gemv_f32(ws->f32_data, rows, cols, input, output);

    case OC_WEIGHT_QUANTIZED:
    case OC_WEIGHT_MMAP_QUANTIZED: {
        size_t sz;
        const uint8_t *data = oc_weight_storage_quant_bytes(ws, &sz);
        if (!data) return OC_ERR_INVALID_ARG;
        /* Dequantize the full weight matrix row by row. */
        size_t row_bytes = oc_quantized_size(ws->qtype, cols);
        if (row_bytes == 0) return OC_ERR_INVALID_ARG;
        if (row_bytes * rows > sz) return OC_ERR_INVALID_ARG;

        float *row_f32 = malloc(cols * sizeof(float));
        if (!row_f32) return OC_ERR_OOM;

        for (size_t j = 0; j < rows; j++) {
            OcError e = oc_quant_dequant_row_scalar(ws->qtype,
                                                     data + j * row_bytes,
                                                     row_bytes,
                                                     row_f32, cols);
            if (e != OC_OK) {
                free(row_f32);
                return e;
            }
            float sum = 0.0f;
            for (size_t i = 0; i < cols; i++)
                sum += row_f32[i] * input[i];
            output[j] = sum;
        }
        free(row_f32);
        return OC_OK;
    }
    }
    return OC_ERR_INVALID_ARG;
}


OcError oc_gemv_expert_weight(const OcWeightStorage *ws,
                               size_t expert_idx, size_t n_experts,
                               size_t rows, size_t cols,
                               const float *input, float *output)
{
    if (!ws || !input || !output) return OC_ERR_INVALID_ARG;
    if (expert_idx >= n_experts) return OC_ERR_INVALID_ARG;
    if (rows == 0 || cols == 0) return OC_ERR_INVALID_ARG;

    switch (ws->type) {
    case OC_WEIGHT_F32: {
        size_t values_per_expert = rows * cols;
        size_t start = expert_idx * values_per_expert;
        if (start + values_per_expert > ws->f32_len) return OC_ERR_INVALID_ARG;
        return oc_gemv_f32(&ws->f32_data[start], rows, cols, input, output);
    }
    case OC_WEIGHT_QUANTIZED:
    case OC_WEIGHT_MMAP_QUANTIZED: {
        size_t sz;
        const uint8_t *data = oc_weight_storage_quant_bytes(ws, &sz);
        if (!data) return OC_ERR_INVALID_ARG;
        size_t row_bytes = oc_quantized_size(ws->qtype, cols);
        if (row_bytes == 0) return OC_ERR_INVALID_ARG;
        size_t expert_size = row_bytes * rows;
        size_t start = expert_idx * expert_size;
        if (start + expert_size > sz) return OC_ERR_INVALID_ARG;

        float *row_f32 = malloc(cols * sizeof(float));
        if (!row_f32) return OC_ERR_OOM;

        for (size_t j = 0; j < rows; j++) {
            OcError e = oc_quant_dequant_row_scalar(ws->qtype,
                                                     data + start + j * row_bytes,
                                                     row_bytes,
                                                     row_f32, cols);
            if (e != OC_OK) {
                free(row_f32);
                return e;
            }
            float sum = 0.0f;
            for (size_t i = 0; i < cols; i++)
                sum += row_f32[i] * input[i];
            output[j] = sum;
        }
        free(row_f32);
        return OC_OK;
    }
    }
    return OC_ERR_INVALID_ARG;
}


OcError oc_gemv_weight_fused(OcGemvPart *parts, size_t n_parts,
                              size_t cols, const float *input)
{
    if (!parts || !input) return OC_ERR_INVALID_ARG;
    if (n_parts == 0) return OC_OK;

    for (size_t p = 0; p < n_parts; p++) {
        if (parts[p].rows == 0) continue;
        OcError e = oc_gemv_weight(parts[p].storage, parts[p].rows, cols,
                                    input, parts[p].output);
        if (e != OC_OK) return e;
    }
    return OC_OK;
}


OcError oc_gemm_weight(const OcWeightStorage *ws,
                        size_t rows, size_t cols,
                        const float *inputs, float *outputs, size_t batch)
{
    if (!ws || !inputs || !outputs) return OC_ERR_INVALID_ARG;
    if (rows == 0 || cols == 0 || batch == 0) return OC_ERR_INVALID_ARG;

    /* For batch=1, use GEMV. */
    if (batch == 1)
        return oc_gemv_weight(ws, rows, cols, inputs, outputs);

    switch (ws->type) {
    case OC_WEIGHT_F32: {
        if (ws->f32_len < rows * cols) return OC_ERR_INVALID_ARG;
        for (size_t b = 0; b < batch; b++) {
            const float *in = &inputs[b * cols];
            float *out = &outputs[b * rows];
            for (size_t j = 0; j < rows; j++) {
                float sum = 0.0f;
                const float *row = &ws->f32_data[j * cols];
                for (size_t i = 0; i < cols; i++)
                    sum += row[i] * in[i];
                out[j] = sum;
            }
        }
        return OC_OK;
    }
    case OC_WEIGHT_QUANTIZED:
    case OC_WEIGHT_MMAP_QUANTIZED: {
        size_t sz;
        const uint8_t *data = oc_weight_storage_quant_bytes(ws, &sz);
        if (!data) return OC_ERR_INVALID_ARG;
        size_t row_bytes = oc_quantized_size(ws->qtype, cols);
        if (row_bytes == 0) return OC_ERR_INVALID_ARG;
        if (row_bytes * rows > sz) return OC_ERR_INVALID_ARG;

        float *row_f32 = malloc(cols * sizeof(float));
        if (!row_f32) return OC_ERR_OOM;

        for (size_t b = 0; b < batch; b++) {
            const float *in = &inputs[b * cols];
            float *out = &outputs[b * rows];
            for (size_t j = 0; j < rows; j++) {
                OcError e = oc_quant_dequant_row_scalar(ws->qtype,
                                                         data + j * row_bytes,
                                                         row_bytes,
                                                         row_f32, cols);
                if (e != OC_OK) {
                    free(row_f32);
                    return e;
                }
                float sum = 0.0f;
                for (size_t i = 0; i < cols; i++)
                    sum += row_f32[i] * in[i];
                out[j] = sum;
            }
        }
        free(row_f32);
        return OC_OK;
    }
    }
    return OC_ERR_INVALID_ARG;
}


void oc_add_repeating_bias(float *buf, size_t buf_len,
                            const float *bias, size_t bias_len)
{
    if (!buf || !bias || bias_len == 0) return;
    for (size_t i = 0; i < buf_len; i++)
        buf[i] += bias[i % bias_len];
}


static int compare_expert_scores(const void *a, const void *b)
{
    const OcExpertScore *sa = a, *sb = b;
    if (sa->select > sb->select) return -1;
    if (sa->select < sb->select) return 1;
    return 0;
}

OcError oc_moe_ffn_forward(const OcWeightStorage *gate_inp,
                             const OcWeightStorage *gate_exps,
                             const OcWeightStorage *up_exps,
                             const OcWeightStorage *down_exps,
                             const float *exp_probs_b,
                             const OcInferenceConfig *cfg,
                             const float *normed, float *ffn_out,
                             float *gate_scratch, float *up_scratch,
                             float *expert_out,
                             float *router_logits,
                             OcExpertScore *expert_scores)
{
    if (!gate_inp || !gate_exps || !up_exps || !down_exps || !cfg ||
        !normed || !ffn_out || !gate_scratch || !up_scratch || !expert_out ||
        !router_logits || !expert_scores)
        return OC_ERR_INVALID_ARG;

    size_t h = cfg->hidden_size;
    size_t i_size = cfg->expert_intermediate_size > 0 ? cfg->expert_intermediate_size : cfg->intermediate_size;
    size_t n_experts = cfg->num_experts;
    /* LongCat appends `zero_expert_count` identity experts after the routed them -- a token can route part of its mass to "do nothing". */
    size_t n_zero  = cfg->zero_expert_count;
    size_t n_slots = n_experts + n_zero;
    size_t n_per_tok = cfg->num_experts_per_tok;
    if (n_per_tok == 0) n_per_tok = 1;
    if (n_per_tok > n_slots) n_per_tok = n_slots;
    bool sigmoid_gating = cfg->expert_gating_sigmoid;

    /* Zero output. */
    memset(ffn_out, 0, h * sizeof(float));

    memset(router_logits, 0, n_slots * sizeof(float));
    OcError e = oc_gemv_weight(gate_inp, n_slots, h, normed, router_logits);
    if (e != OC_OK) return e;

    if (sigmoid_gating) {
        for (size_t i = 0; i < n_slots; i++)
            router_logits[i] = 1.0f / (1.0f + expf(-router_logits[i]));
        for (size_t i = 0; i < n_slots; i++) {
            float bias = exp_probs_b ? exp_probs_b[i] : 0.0f;
            expert_scores[i].idx = i;
            /* Historically the bias was folded into the weight here. It still
             * is for sigmoid gating, which is what LFM2MoE expects; only the
             * softmax path below separates selection from weight. */
            expert_scores[i].weight = router_logits[i] + bias;
            expert_scores[i].select = expert_scores[i].weight;
        }
    } else {
        float max_logit = router_logits[0];
        for (size_t i = 1; i < n_slots; i++)
            if (router_logits[i] > max_logit) max_logit = router_logits[i];
        float sum_exp = 0.0f;
        for (size_t i = 0; i < n_slots; i++) {
            router_logits[i] = expf(router_logits[i] - max_logit);
            sum_exp += router_logits[i];
        }
        if (sum_exp > 0.0f) {
            for (size_t i = 0; i < n_slots; i++)
                router_logits[i] /= sum_exp;
        }
        for (size_t i = 0; i < n_slots; i++) {
            expert_scores[i].idx = i;
            expert_scores[i].weight = router_logits[i];
            /* exp_probs_b steers WHICH experts win top-k without changing
             * how much their output counts. Folding it into the weight
             * instead would double-count the bias. */
            float bias = (n_zero > 0 && exp_probs_b) ? exp_probs_b[i] : 0.0f;
            expert_scores[i].select = router_logits[i] + bias;
        }
    }

    if (cfg->expert_group_count > 1 &&
        cfg->expert_group_used_count > 0 &&
        cfg->expert_group_used_count < cfg->expert_group_count &&
        n_slots % cfg->expert_group_count == 0)
    {
        size_t n_group = cfg->expert_group_count;
        size_t group_size = n_slots / n_group;
        size_t n_group_used = cfg->expert_group_used_count;

        /* Compute per-group max score. */
        float *group_max = malloc(n_group * sizeof(float));
        if (!group_max) return OC_ERR_OOM;
        for (size_t g = 0; g < n_group; g++) group_max[g] = -INFINITY;
        for (size_t i = 0; i < n_slots; i++) {
            size_t g = i / group_size;
            if (expert_scores[i].select > group_max[g])
                group_max[g] = expert_scores[i].select;
        }
        /* Find top n_group_used groups. */
        /* Simple: mark groups as selected or not. */
        bool *group_selected = calloc(n_group, sizeof(bool));
        if (!group_selected) { free(group_max); return OC_ERR_OOM; }
        for (size_t sel = 0; sel < n_group_used; sel++) {
            float best_score = -1e30f;
            size_t best_g = 0;
            for (size_t g = 0; g < n_group; g++) {
                if (!group_selected[g] && group_max[g] > best_score) {
                    best_score = group_max[g];
                    best_g = g;
                }
            }
            group_selected[best_g] = true;
        }
        /* Mask out experts in non-selected groups. */
        for (size_t i = 0; i < n_slots; i++) {
            size_t g = i / group_size;
            if (!group_selected[g])
                expert_scores[i].select = -1e30f;
        }
        free(group_max);
        free(group_selected);
    }

    qsort(expert_scores, n_slots, sizeof(OcExpertScore), compare_expert_scores);

    bool renormalize = (n_zero == 0);
    float weight_sum = 0.0f;
    if (renormalize) {
        for (size_t k = 0; k < n_per_tok; k++)
            weight_sum += expert_scores[k].weight;
    }
    /* Apply expert_weights_scale. */
    float scale = cfg->expert_weights_scale;

    for (size_t k = 0; k < n_per_tok; k++) {
        size_t expert_idx = expert_scores[k].idx;
        float weight = expert_scores[k].weight;
        if (expert_scores[k].select <= -1e29f) continue;  /* masked out */

        float normalized_weight;
        if (renormalize) {
            normalized_weight = (weight_sum > 0.0f)
                ? (weight / weight_sum) * scale : 0.0f;
        } else {
            normalized_weight = weight * scale;
        }

        if (expert_idx >= n_experts) {
            for (size_t i = 0; i < h; i++)
                ffn_out[i] += normalized_weight * normed[i];
            continue;
        }

        e = oc_gemv_expert_weight(gate_exps, expert_idx, n_experts,
                                   i_size, h, normed, gate_scratch);
        if (e != OC_OK) return e;

        e = oc_gemv_expert_weight(up_exps, expert_idx, n_experts,
                                   i_size, h, normed, up_scratch);
        if (e != OC_OK) return e;

        /* SwiGLU: gate[i] = silu(gate[i]) * up[i] */
        oc_swiglu_inplace_f32(gate_scratch, up_scratch, i_size);

        e = oc_gemv_expert_weight(down_exps, expert_idx, n_experts,
                                   h, i_size, gate_scratch, expert_out);
        if (e != OC_OK) return e;

        /* Accumulate: ffn_out += weight * expert_out */
        for (size_t i = 0; i < h; i++)
            ffn_out[i] += normalized_weight * expert_out[i];
    }

    return OC_OK;
}
