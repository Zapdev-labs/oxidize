#include "oxidize/qwen35_delta.h"

#include "oxidize/parallel.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

static bool checked_mul(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool geometry_sizes(const OcQwen35DeltaGeometry *g,
                           size_t *key_dim, size_t *value_dim,
                           size_t *conv_dim, size_t *conv_state_len,
                           size_t *recurrent_state_len)
{
    size_t twice_key;
    if (g == NULL || g->n_key_heads == 0 || g->n_value_heads == 0 ||
        g->key_head_dim == 0 || g->value_head_dim == 0 ||
        g->conv_kernel < 2 || g->n_value_heads % g->n_key_heads != 0) {
        return false;
    }
    if (!checked_mul(g->n_key_heads, g->key_head_dim, key_dim) ||
        !checked_mul(g->n_value_heads, g->value_head_dim, value_dim) ||
        !checked_mul(*key_dim, 2, &twice_key) ||
        twice_key > SIZE_MAX - *value_dim) {
        return false;
    }
    *conv_dim = twice_key + *value_dim;
    if (!checked_mul(*conv_dim, g->conv_kernel - 1, conv_state_len) ||
        !checked_mul(*value_dim, g->key_head_dim, recurrent_state_len)) {
        return false;
    }
    return true;
}

OcError oc_qwen35_delta_state_init(OcQwen35DeltaState *state,
                                   const OcQwen35DeltaGeometry *geometry,
                                   float *conv_state, size_t conv_state_len,
                                   float *recurrent_state,
                                   size_t recurrent_state_len)
{
    size_t key_dim, value_dim, conv_dim, want_conv, want_recurrent;
    if (state == NULL || !geometry_sizes(geometry, &key_dim, &value_dim,
                                         &conv_dim, &want_conv,
                                         &want_recurrent) ||
        conv_state == NULL || recurrent_state == NULL ||
        conv_state_len < want_conv || recurrent_state_len < want_recurrent) {
        return OC_ERR_INVALID_ARG;
    }
    (void)key_dim;
    (void)value_dim;
    (void)conv_dim;
    state->geometry = *geometry;
    state->conv_state = conv_state;
    state->conv_state_len = want_conv;
    state->recurrent_state = recurrent_state;
    state->recurrent_state_len = want_recurrent;
    oc_qwen35_delta_state_reset(state);
    return OC_OK;
}

void oc_qwen35_delta_state_reset(OcQwen35DeltaState *state)
{
    if (state == NULL) return;
    if (state->conv_state != NULL) {
        memset(state->conv_state, 0, state->conv_state_len * sizeof(float));
    }
    if (state->recurrent_state != NULL) {
        memset(state->recurrent_state, 0,
               state->recurrent_state_len * sizeof(float));
    }
}

void oc_qwen35_delta_state_free(OcQwen35DeltaState *state)
{
    if (state != NULL) memset(state, 0, sizeof(*state));
}

static float sigmoid_stable(float x)
{
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float e = expf(x);
    return e / (1.0f + e);
}

static float softplus_stable(float x)
{
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static float silu_stable(float x)
{
    if (x >= 0.0f) return x / (1.0f + expf(-x));
    float e = expf(x);
    return x * e / (1.0f + e);
}

typedef struct {
    const OcQwen35DeltaGeometry *g;
    float *state;
    const float *weight;
    const float *input;
    float *output;
} ConvCtx;

static void conv_channels(size_t begin, size_t end, size_t tid, void *user_data)
{
    (void)tid;
    ConvCtx *ctx = user_data;
    size_t kernel = ctx->g->conv_kernel;
    size_t history = kernel - 1;
    for (size_t channel = begin; channel < end; channel++) {
        float *past = ctx->state + channel * history;
        const float *weight = ctx->weight + channel * kernel;
        float sum = weight[kernel - 1] * ctx->input[channel];
        for (size_t i = 0; i < history; i++) sum += weight[i] * past[i];
        memmove(past, past + 1, (history - 1) * sizeof(float));
        past[history - 1] = ctx->input[channel];
        ctx->output[channel] = silu_stable(sum);
    }
}

typedef struct {
    const OcQwen35DeltaGeometry *g;
    float *conv_output;
    float eps;
} NormalizeCtx;

static void normalize_key_heads(size_t begin, size_t end, size_t tid,
                                void *user_data)
{
    (void)tid;
    NormalizeCtx *ctx = user_data;
    size_t dim = ctx->g->key_head_dim;
    size_t key_total = ctx->g->n_key_heads * dim;
    float q_scale = 1.0f / sqrtf((float)dim);
    for (size_t head = begin; head < end; head++) {
        float *q = ctx->conv_output + head * dim;
        float *k = ctx->conv_output + key_total + head * dim;
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            q_sum += q[i] * q[i];
            k_sum += k[i] * k[i];
        }
        float q_inv = 1.0f / sqrtf(q_sum > ctx->eps ? q_sum : ctx->eps);
        float k_inv = 1.0f / sqrtf(k_sum > ctx->eps ? k_sum : ctx->eps);
        for (size_t i = 0; i < dim; i++) {
            q[i] *= q_inv * q_scale;
            k[i] *= k_inv;
        }
    }
}

typedef struct {
    const OcQwen35DeltaGeometry *g;
    float *recurrent;
    const float *conv_output;
    const OcQwen35DeltaParams *params;
    const OcQwen35DeltaInput *input;
    float *output;
} DeltaCtx;

static void delta_heads(size_t begin, size_t end, size_t tid, void *user_data)
{
    (void)tid;
    DeltaCtx *ctx = user_data;
    size_t dk = ctx->g->key_head_dim;
    size_t dv = ctx->g->value_head_dim;
    size_t key_total = ctx->g->n_key_heads * dk;
    const float *values = ctx->conv_output + 2 * key_total;
    for (size_t head = begin; head < end; head++) {
        size_t key_head = head % ctx->g->n_key_heads;
        const float *q = ctx->conv_output + key_head * dk;
        const float *k = ctx->conv_output + key_total + key_head * dk;
        const float *v = values + head * dv;
        float *matrix = ctx->recurrent + head * dv * dk;
        float *out = ctx->output + head * dv;
        float beta = sigmoid_stable(ctx->input->beta[head]);
        float decay = expf(ctx->params->ssm_a[head] *
                           softplus_stable(ctx->input->alpha[head] +
                                           ctx->params->dt_bias[head]));
        for (size_t i = 0; i < dv; i++) {
            float *row = matrix + i * dk;
            float state_k = 0.0f;
            for (size_t j = 0; j < dk; j++) {
                row[j] *= decay;
                state_k += row[j] * k[j];
            }
            float delta = (v[i] - state_k) * beta;
            float value = 0.0f;
            for (size_t j = 0; j < dk; j++) {
                row[j] += delta * k[j];
                value += row[j] * q[j];
            }
            out[i] = value;
        }
    }
}

typedef struct {
    const OcQwen35DeltaGeometry *g;
    const OcQwen35DeltaParams *params;
    const float *gate;
    float *output;
} GateCtx;

static void gate_heads(size_t begin, size_t end, size_t tid, void *user_data)
{
    (void)tid;
    GateCtx *ctx = user_data;
    size_t dv = ctx->g->value_head_dim;
    for (size_t head = begin; head < end; head++) {
        float *out = ctx->output + head * dv;
        const float *gate = ctx->gate + head * dv;
        float sum = 0.0f;
        for (size_t i = 0; i < dv; i++) sum += out[i] * out[i];
        float inv = 1.0f / sqrtf(sum / (float)dv + ctx->params->norm_eps);
        for (size_t i = 0; i < dv; i++) {
            out[i] = out[i] * inv * ctx->params->norm_weight[i] *
                     silu_stable(gate[i]);
        }
    }
}

OcError oc_qwen35_delta_step(OcQwen35DeltaState *state,
                             const OcQwen35DeltaParams *params,
                             const OcQwen35DeltaInput *input,
                             float *conv_output, size_t conv_output_len,
                             float *output, size_t output_len)
{
    size_t key_dim, value_dim, conv_dim, conv_state_len, recurrent_state_len;
    if (state == NULL || params == NULL || input == NULL ||
        !geometry_sizes(&state->geometry, &key_dim, &value_dim, &conv_dim,
                        &conv_state_len, &recurrent_state_len) ||
        state->conv_state == NULL || state->recurrent_state == NULL ||
        state->conv_state_len < conv_state_len ||
        state->recurrent_state_len < recurrent_state_len ||
        params->conv_weight == NULL || params->ssm_a == NULL ||
        params->dt_bias == NULL || params->norm_weight == NULL ||
        !isfinite(params->norm_eps) || params->norm_eps <= 0.0f ||
        params->conv_weight_len / state->geometry.conv_kernel < conv_dim ||
        input->qkv == NULL || input->gate == NULL || input->beta == NULL ||
        input->alpha == NULL || input->qkv_len < conv_dim ||
        input->gate_len < value_dim ||
        input->beta_len < state->geometry.n_value_heads ||
        input->alpha_len < state->geometry.n_value_heads ||
        conv_output == NULL || conv_output_len < conv_dim ||
        output == NULL || output_len < value_dim) {
        return OC_ERR_INVALID_ARG;
    }
    ConvCtx conv = {&state->geometry, state->conv_state,
                    params->conv_weight, input->qkv, conv_output};
    oc_parallel_for(conv_dim, conv_channels, &conv);

    NormalizeCtx normalize = {&state->geometry, conv_output, params->norm_eps};
    oc_parallel_for(state->geometry.n_key_heads, normalize_key_heads, &normalize);

    DeltaCtx delta = {&state->geometry, state->recurrent_state, conv_output,
                      params, input, output};
    oc_parallel_for(state->geometry.n_value_heads, delta_heads, &delta);

    GateCtx gate = {&state->geometry, params, input->gate, output};
    oc_parallel_for(state->geometry.n_value_heads, gate_heads, &gate);
    return OC_OK;
}
