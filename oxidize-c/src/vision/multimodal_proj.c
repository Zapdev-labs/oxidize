#include "oxidize/multimodal_proj.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


static float mm_gelu(float x)
{
    /* Exact GeLU: 0.5 * x * (1 + erf(x / sqrt(2))) */
    return 0.5f * x * (1.0f + erff(x / 1.4142135623730951f));
}

static float mm_relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

static float mm_silu(float x)
{
    return x * (1.0f / (1.0f + expf(-x)));
}

static float mm_activate(OcMultimodalActivation act, float x)
{
    switch (act) {
    case OC_MM_ACT_GELU: return mm_gelu(x);
    case OC_MM_ACT_RELU: return mm_relu(x);
    case OC_MM_ACT_SILU: return mm_silu(x);
    default: return mm_gelu(x);
    }
}


OcMultimodalProjection *oc_mm_proj_init(const OcMultimodalProjectionConfig *config)
{
    if (!config) return NULL;
    if (config->input_dim == 0 || config->output_dim == 0) return NULL;
    if (config->n_layers == 0) return NULL;

    OcMultimodalProjection *proj = calloc(1, sizeof(*proj));
    if (!proj) return NULL;

    proj->config = *config;
    if (proj->config.n_layers == 0) proj->config.n_layers = 2;
    if (proj->config.hidden_dim == 0) proj->config.hidden_dim = config->input_dim;

    uint32_t n = proj->config.n_layers;
    proj->n_weights = n;
    proj->weights = calloc(n, sizeof(float *));
    proj->biases = calloc(n, sizeof(float *));
    proj->in_dims = calloc(n, sizeof(uint32_t));
    proj->out_dims = calloc(n, sizeof(uint32_t));
    if (!proj->weights || !proj->biases || !proj->in_dims || !proj->out_dims) {
        free(proj->weights);
        free(proj->biases);
        free(proj->in_dims);
        free(proj->out_dims);
        free(proj);
        return NULL;
    }

    /* Compute per-layer dims: */
    for (uint32_t l = 0; l < n; l++) {
        proj->in_dims[l] = (l == 0) ? config->input_dim : proj->out_dims[l - 1];
        if (n == 1) {
            proj->out_dims[l] = config->output_dim;
        } else {
            proj->out_dims[l] = (l == n - 1) ? config->output_dim
                                              : proj->config.hidden_dim;
        }
    }

    proj->initialized = true;
    return proj;
}

void oc_mm_proj_free(OcMultimodalProjection *proj)
{
    if (!proj) return;
    if (proj->weights) {
        for (uint32_t l = 0; l < proj->n_weights; l++) {
            free(proj->weights[l]);
        }
        free(proj->weights);
    }
    if (proj->biases) {
        for (uint32_t l = 0; l < proj->n_weights; l++) {
            free(proj->biases[l]);
        }
        free(proj->biases);
    }
    free(proj->in_dims);
    free(proj->out_dims);
    proj->weights = NULL;
    proj->biases = NULL;
    proj->in_dims = NULL;
    proj->out_dims = NULL;
    proj->n_weights = 0;
    proj->initialized = false;
    free(proj);
}


OcError oc_mm_proj_load_weights(OcMultimodalProjection *proj,
                                  const float *data, size_t data_size)
{
    if (!proj || !proj->initialized || !data) return OC_ERR_INVALID_ARG;

    size_t offset = 0;
    for (uint32_t l = 0; l < proj->n_weights; l++) {
        size_t w_count = (size_t)proj->out_dims[l] * proj->in_dims[l];
        size_t b_count = proj->out_dims[l];
        size_t needed = (offset + w_count + b_count) * sizeof(float);
        if (needed > data_size) return OC_ERR_INVALID_ARG;

        /* Weight matrix. */
        free(proj->weights[l]);
        proj->weights[l] = malloc(w_count * sizeof(float));
        if (!proj->weights[l]) return OC_ERR_OOM;
        memcpy(proj->weights[l], data + offset, w_count * sizeof(float));
        offset += w_count;

        /* Bias vector. */
        free(proj->biases[l]);
        proj->biases[l] = malloc(b_count * sizeof(float));
        if (!proj->biases[l]) return OC_ERR_OOM;
        memcpy(proj->biases[l], data + offset, b_count * sizeof(float));
        offset += b_count;
    }

    return OC_OK;
}

OcError oc_mm_proj_set_layer_weight(OcMultimodalProjection *proj,
                                      uint32_t layer_idx,
                                      const float *weight_data,
                                      size_t n_elements)
{
    if (!proj || !proj->initialized || !weight_data) return OC_ERR_INVALID_ARG;
    if (layer_idx >= proj->n_weights) return OC_ERR_INVALID_ARG;
    size_t expected = (size_t)proj->out_dims[layer_idx] * proj->in_dims[layer_idx];
    if (n_elements != expected) return OC_ERR_INVALID_ARG;

    free(proj->weights[layer_idx]);
    proj->weights[layer_idx] = malloc(n_elements * sizeof(float));
    if (!proj->weights[layer_idx]) return OC_ERR_OOM;
    memcpy(proj->weights[layer_idx], weight_data, n_elements * sizeof(float));
    return OC_OK;
}

OcError oc_mm_proj_set_layer_bias(OcMultimodalProjection *proj,
                                    uint32_t layer_idx,
                                    const float *bias_data,
                                    size_t n_elements)
{
    if (!proj || !proj->initialized || !bias_data) return OC_ERR_INVALID_ARG;
    if (layer_idx >= proj->n_weights) return OC_ERR_INVALID_ARG;
    if (n_elements != proj->out_dims[layer_idx]) return OC_ERR_INVALID_ARG;

    free(proj->biases[layer_idx]);
    proj->biases[layer_idx] = malloc(n_elements * sizeof(float));
    if (!proj->biases[layer_idx]) return OC_ERR_OOM;
    memcpy(proj->biases[layer_idx], bias_data, n_elements * sizeof(float));
    return OC_OK;
}


float *oc_mm_proj_forward(OcMultimodalProjection *proj,
                            const float *input, size_t n_tokens)
{
    if (!proj || !proj->initialized || !input || n_tokens == 0) return NULL;

    /* Verify all layers have weights set. */
    for (uint32_t l = 0; l < proj->n_weights; l++) {
        if (!proj->weights[l] || !proj->biases[l]) return NULL;
    }

    const float *current = input;
    size_t current_dim = proj->config.input_dim;
    size_t current_tokens = n_tokens;
    bool owns_current = false;

    for (uint32_t l = 0; l < proj->n_weights; l++) {
        uint32_t in_d = proj->in_dims[l];
        uint32_t out_d = proj->out_dims[l];
        const float *W = proj->weights[l];
        const float *b = proj->biases[l];

        if (in_d != current_dim) {
            if (owns_current) free((void *)current);
            return NULL;
        }

        float *out = malloc(current_tokens * out_d * sizeof(float));
        if (!out) {
            if (owns_current) free((void *)current);
            return NULL;
        }

        /* Linear: out[t, j] = sum_i W[j, i] * current[t, i] + b[j] */
        for (size_t t = 0; t < current_tokens; t++) {
            const float *in_row = current + t * in_d;
            float *out_row = out + t * out_d;
            for (uint32_t j = 0; j < out_d; j++) {
                const float *w_row = W + (size_t)j * in_d;
                float dot = 0.0f;
                for (uint32_t i = 0; i < in_d; i++) {
                    dot += w_row[i] * in_row[i];
                }
                out_row[j] = dot + b[j];
            }
        }

        /* Apply activation between layers (not after the final layer). */
        if (l < proj->n_weights - 1) {
            for (size_t t = 0; t < current_tokens; t++) {
                float *row = out + t * out_d;
                for (uint32_t j = 0; j < out_d; j++) {
                    row[j] = mm_activate(proj->config.activation, row[j]);
                }
            }
        }

        if (owns_current) free((void *)current);
        current = out;
        current_dim = out_d;
        owns_current = true;
    }

    /* If we never allocated (shouldn't happen with n_layers >= 1), return NULL. */
    if (!owns_current) return NULL;

    return (float *)current;
}


float *oc_mm_proj_concat_prompt(OcMultimodalProjection *proj,
                                  const float *text_embeds, size_t n_text,
                                  const float *mm_embeds, size_t n_mm)
{
    if (!proj || !proj->initialized) return NULL;
    if (n_text == 0 && n_mm == 0) return NULL;

    uint32_t out_d = proj->config.output_dim;
    size_t total = n_text + n_mm;
    float *out = malloc(total * out_d * sizeof(float));
    if (!out) return NULL;

    size_t offset = 0;
    if (n_text > 0 && text_embeds) {
        memcpy(out + offset, text_embeds, n_text * out_d * sizeof(float));
        offset += n_text * out_d;
    }
    if (n_mm > 0 && mm_embeds) {
        memcpy(out + offset, mm_embeds, n_mm * out_d * sizeof(float));
        offset += n_mm * out_d;
    }

    return out;
}
