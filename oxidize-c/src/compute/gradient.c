/*
 * gradient.c — Gradient computation + optimizer stubs implementation.
 */
#include "oxidize/gradient.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


OcError oc_grad_config_init(OcGradientConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    cfg->learning_rate  = OC_GRAD_DEFAULT_LR;
    cfg->clip_norm      = OC_GRAD_DEFAULT_CLIP_NORM;
    cfg->weight_decay   = OC_GRAD_DEFAULT_WEIGHT_DECAY;
    return OC_OK;
}

OcError oc_grad_init(const OcGradientConfig *config, size_t n_params,
                     OcGradientState *out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    if (n_params == 0) return OC_ERR_INVALID_ARG;

    memset(out, 0, sizeof(*out));

    if (config) {
        out->config = *config;
    } else {
        oc_grad_config_init(&out->config);
    }

    size_t bytes = n_params * sizeof(float);
    out->gradients = malloc(bytes);
    out->params    = malloc(bytes);
    out->momentum  = malloc(bytes);
    out->velocity  = malloc(bytes);
    if (!out->gradients || !out->params || !out->momentum || !out->velocity) {
        oc_grad_free(out);
        return OC_ERR_OOM;
    }

    memset(out->gradients, 0, bytes);
    memset(out->params,    0, bytes);
    memset(out->momentum,  0, bytes);
    memset(out->velocity,  0, bytes);

    out->n_params   = n_params;
    out->step_count = 0;
    return OC_OK;
}

void oc_grad_free(OcGradientState *state)
{
    if (!state) return;
    free(state->gradients);
    free(state->params);
    free(state->momentum);
    free(state->velocity);
    memset(state, 0, sizeof(*state));
}


OcError oc_grad_zero(OcGradientState *state)
{
    if (!state) return OC_ERR_INVALID_ARG;
    if (state->n_params == 0) return OC_OK;
    memset(state->gradients, 0, state->n_params * sizeof(float));
    return OC_OK;
}

OcError oc_grad_accumulate(OcGradientState *state, const float *grad, size_t n)
{
    if (!state || !grad) return OC_ERR_INVALID_ARG;
    if (n == 0 || n > state->n_params) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < n; i++) {
        state->gradients[i] += grad[i];
    }
    return OC_OK;
}

OcError oc_grad_clip(OcGradientState *state)
{
    if (!state) return OC_ERR_INVALID_ARG;
    if (state->n_params == 0 || state->config.clip_norm <= 0.0f)
        return OC_OK;

    /* Global L2 norm of the gradient buffer. */
    double sum_sq = 0.0;
    for (size_t i = 0; i < state->n_params; i++) {
        double g = (double)state->gradients[i];
        sum_sq += g * g;
    }
    float norm = (float)sqrt(sum_sq);
    if (norm <= state->config.clip_norm) return OC_OK;

    float scale = state->config.clip_norm / norm;
    for (size_t i = 0; i < state->n_params; i++) {
        state->gradients[i] *= scale;
    }
    return OC_OK;
}


OcError oc_grad_sgd_step(OcGradientState *state, float *params, size_t n)
{
    if (!state || !params) return OC_ERR_INVALID_ARG;
    if (n == 0 || n > state->n_params) return OC_ERR_INVALID_ARG;

    float lr = state->config.learning_rate;
    for (size_t i = 0; i < n; i++) {
        params[i] -= lr * state->gradients[i];
    }
    state->step_count++;
    return OC_OK;
}

OcError oc_grad_adam_step(OcGradientState *state, float *params, size_t n)
{
    if (!state || !params) return OC_ERR_INVALID_ARG;
    if (n == 0 || n > state->n_params) return OC_ERR_INVALID_ARG;

    const float beta1 = OC_GRAD_ADAM_BETA1;
    const float beta2 = OC_GRAD_ADAM_BETA2;
    const float eps   = OC_GRAD_ADAM_EPS;

    state->step_count++;
    uint64_t t = state->step_count;
    float bias1 = 1.0f - powf(beta1, (float)t);
    float bias2 = 1.0f - powf(beta2, (float)t);
    float lr    = state->config.learning_rate;

    for (size_t i = 0; i < n; i++) {
        float g = state->gradients[i];
        state->momentum[i] = beta1 * state->momentum[i] + (1.0f - beta1) * g;
        state->velocity[i] = beta2 * state->velocity[i]
                             + (1.0f - beta2) * g * g;
        float m_hat = state->momentum[i] / bias1;
        float v_hat = state->velocity[i] / bias2;
        params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
    return OC_OK;
}

OcError oc_grad_adamw_step(OcGradientState *state, float *params, size_t n)
{
    if (!state || !params) return OC_ERR_INVALID_ARG;
    if (n == 0 || n > state->n_params) return OC_ERR_INVALID_ARG;

    const float beta1 = OC_GRAD_ADAM_BETA1;
    const float beta2 = OC_GRAD_ADAM_BETA2;
    const float eps   = OC_GRAD_ADAM_EPS;
    const float wd    = state->config.weight_decay;

    state->step_count++;
    uint64_t t = state->step_count;
    float bias1 = 1.0f - powf(beta1, (float)t);
    float bias2 = 1.0f - powf(beta2, (float)t);
    float lr    = state->config.learning_rate;

    for (size_t i = 0; i < n; i++) {
        /* Decoupled weight decay applied directly to params. */
        params[i] -= lr * wd * params[i];

        float g = state->gradients[i];
        state->momentum[i] = beta1 * state->momentum[i] + (1.0f - beta1) * g;
        state->velocity[i] = beta2 * state->velocity[i]
                             + (1.0f - beta2) * g * g;
        float m_hat = state->momentum[i] / bias1;
        float v_hat = state->velocity[i] / bias2;
        params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
    return OC_OK;
}


OcError oc_grad_set_lr(OcGradientState *state, float lr)
{
    if (!state) return OC_ERR_INVALID_ARG;
    if (lr < 0.0f) return OC_ERR_INVALID_ARG;
    state->config.learning_rate = lr;
    return OC_OK;
}

float oc_grad_get_lr(const OcGradientState *state)
{
    if (!state) return 0.0f;
    return state->config.learning_rate;
}


/* Forward declarations of the scalar activation kernels we reuse from
 * activation.c so the gradient path stays bit-exact with the forward path. */
float oc_gelu_exact_f32(float x);

__attribute__((unused))
static float oc_silu_f32(float x)
{
    return x / (1.0f + expf(-x));
}

OcError oc_grad_compute_linear_backward(
    const float *input, const float *weight, const float *grad_output,
    size_t in_features, size_t out_features,
    float *grad_input, float *grad_weight)
{
    if (!input || !weight || !grad_output || !grad_input || !grad_weight)
        return OC_ERR_INVALID_ARG;
    if (in_features == 0 || out_features == 0) return OC_ERR_INVALID_ARG;

    for (size_t i = 0; i < in_features; i++) {
        float acc = 0.0f;
        for (size_t j = 0; j < out_features; j++) {
            acc += grad_output[j] * weight[j * in_features + i];
        }
        grad_input[i] = acc;
    }

    for (size_t j = 0; j < out_features; j++) {
        float go = grad_output[j];
        float *gw_row = grad_weight + j * in_features;
        for (size_t i = 0; i < in_features; i++) {
            gw_row[i] = go * input[i];
        }
    }
    return OC_OK;
}

OcError oc_grad_compute_activation_backward(
    const float *grad_output, const float *input, size_t n,
    OcGradActivationType activation_type, float *out)
{
    if (!grad_output || !input || !out) return OC_ERR_INVALID_ARG;
    if (n == 0) return OC_ERR_INVALID_ARG;

    switch (activation_type) {
    case OC_GRAD_ACT_RELU:
        for (size_t i = 0; i < n; i++) {
            out[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f;
        }
        break;

    case OC_GRAD_ACT_GELU: {
        /* d/dx gelu_exact(x) = 0.5*(1 + erf(x/sqrt2)) + x*pdf(x/sqrt2)/sqrt2
         * where pdf(z) = exp(-z^2/2)/sqrt(2*pi). */
        const float inv_sqrt2 = 0.70710678118654752f;
        const float inv_sqrt_2pi = 0.39894228040143268f;
        for (size_t i = 0; i < n; i++) {
            float x = input[i];
            float z = x * inv_sqrt2;
            float pdf = inv_sqrt_2pi * expf(-0.5f * z * z);
            float d = 0.5f * (1.0f + erff(z)) + x * pdf * inv_sqrt2;
            out[i] = grad_output[i] * d;
        }
        break;
    }

    case OC_GRAD_ACT_SILU: {
        /* silu(x)  = x*sigmoid(x)
         * silu'(x) = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
         *          = sigmoid(x)*(1 + x*(1-sigmoid(x))) */
        for (size_t i = 0; i < n; i++) {
            float x = input[i];
            float sig = 1.0f / (1.0f + expf(-x));
            float d = sig * (1.0f + x * (1.0f - sig));
            out[i] = grad_output[i] * d;
        }
        break;
    }

    case OC_GRAD_ACT_TANH: {
        for (size_t i = 0; i < n; i++) {
            float t = tanhf(input[i]);
            out[i] = grad_output[i] * (1.0f - t * t);
        }
        break;
    }

    default:
        return OC_ERR_INVALID_ARG;
    }
    return OC_OK;
}
