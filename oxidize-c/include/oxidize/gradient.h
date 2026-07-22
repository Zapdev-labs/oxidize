/*
 * gradient.h — Gradient computation + optimizer stubs for LoRA training.
 *
 * Provides basic backprop primitives (linear layer backward, activation
 * backward) and three optimizer step functions (SGD, Adam, AdamW) operating
 * on flat float32 parameter vectors. Intended as the training-side companion
 * to the LoRA adapter inference path (see lora.h).
 *
 * All buffers are caller-owned except those allocated inside OcGradientState.
 */
#ifndef OXIDIZE_GRADIENT_H
#define OXIDIZE_GRADIENT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ──────────────────────────────────────────────────────── */

#define OC_GRAD_DEFAULT_LR          0.001f
#define OC_GRAD_DEFAULT_CLIP_NORM   1.0f
#define OC_GRAD_DEFAULT_WEIGHT_DECAY 0.01f
#define OC_GRAD_ADAM_BETA1          0.9f
#define OC_GRAD_ADAM_BETA2          0.999f
#define OC_GRAD_ADAM_EPS            1e-8f

/* ─── Types ─────────────────────────────────────────────────────────── */

/* Activation type selector for the backward pass. Values are stable. */
typedef enum {
    OC_GRAD_ACT_RELU = 0,
    OC_GRAD_ACT_GELU = 1,
    OC_GRAD_ACT_SILU = 2,
    OC_GRAD_ACT_TANH = 3,
} OcGradActivationType;

/* Optimizer hyperparameters. */
typedef struct OcGradientConfig {
    float learning_rate;   /* step size (default 0.001)                   */
    float clip_norm;       /* max global grad norm before clip (default 1.0) */
    float weight_decay;    /* L2 decay coefficient (default 0.01)         */
} OcGradientConfig;

/* Persistent optimizer state across steps. Buffers are owned (freed by
 * oc_grad_free). All buffers have length `n_params`. */
typedef struct OcGradientState {
    OcGradientConfig config;
    float  *gradients;    /* accumulated gradient buffer                 */
    float  *params;       /* working parameter snapshot                  */
    float  *momentum;     /* Adam first moment (m)                       */
    float  *velocity;     /* Adam second moment (v)                      */
    size_t  n_params;
    uint64_t step_count;   /* incremented by each optimizer step          */
} OcGradientState;

/* ─── Lifecycle ─────────────────────────────────────────────────────── */

/* Initialize config with defaults. */
OcError oc_grad_config_init(OcGradientConfig *cfg);

/* Allocate state for `n_params` parameters. All buffers zeroed. */
OcError oc_grad_init(const OcGradientConfig *config, size_t n_params,
                     OcGradientState *out);

/* Free all owned buffers and reset state. Safe on already-freed state. */
void oc_grad_free(OcGradientState *state);

/* ─── Gradient buffer ops ──────────────────────────────────────────── */

/* Zero the gradient buffer. */
OcError oc_grad_zero(OcGradientState *state);

/* Accumulate `grad[0..n-1]` into the state's gradient buffer.
 * `n` must be <= state->n_params. */
OcError oc_grad_accumulate(OcGradientState *state, const float *grad, size_t n);

/* Clip the gradient buffer to `config.clip_norm` global L2 norm. */
OcError oc_grad_clip(OcGradientState *state);

/* ─── Optimizer steps ──────────────────────────────────────────────── */

/* SGD update: params[i] -= lr * grad[i]. */
OcError oc_grad_sgd_step(OcGradientState *state, float *params, size_t n);

/* Adam update: maintains m, v; uses bias correction. */
OcError oc_grad_adam_step(OcGradientState *state, float *params, size_t n);

/* AdamW update: same as Adam but weight decay applied to params directly. */
OcError oc_grad_adamw_step(OcGradientState *state, float *params, size_t n);

/* ─── Learning rate ────────────────────────────────────────────────── */

OcError oc_grad_set_lr(OcGradientState *state, float lr);
float     oc_grad_get_lr(const OcGradientState *state);

/* ─── Backprop primitives ─────────────────────────────────────────── */

/* Linear layer backward pass.
 *
 *   input        : [in_features]                      (forward input)
 *   weight       : [out_features, in_features] row-major
 *   grad_output  : [out_features]                     (upstream gradient)
 *   grad_input   : [in_features]  (out, caller-allocated)
 *   grad_weight  : [out_features, in_features] (out, caller-allocated)
 *
 * Forward:  output[j] = sum_i(input[i] * weight[j*in + i])
 * Backward:
 *   grad_input[i]  = sum_j(grad_output[j] * weight[j*in + i])
 *   grad_weight[j*in + i] = grad_output[j] * input[i]
 */
OcError oc_grad_compute_linear_backward(
    const float *input, const float *weight, const float *grad_output,
    size_t in_features, size_t out_features,
    float *grad_input, float *grad_weight);

/* Activation backward pass.
 *
 *   grad_output : [n]  upstream gradient
 *   input       : [n]  forward pre-activation
 *   out         : [n]  grad w.r.t. input (caller-allocated)
 *   activation_type : one of OC_GRAD_ACT_*
 *
 *   ReLU:  grad = grad_output * (input > 0)
 *   GELU:  grad = grad_output * gelu'(input)
 *   SiLU:  grad = grad_output * silu'(input)
 *   Tanh:  grad = grad_output * (1 - tanh(input)^2)
 */
OcError oc_grad_compute_activation_backward(
    const float *grad_output, const float *input, size_t n,
    OcGradActivationType activation_type, float *out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_GRADIENT_H */
