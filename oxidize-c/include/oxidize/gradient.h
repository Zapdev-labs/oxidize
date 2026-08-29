/* gradient.h — Gradient computation + optimizer stubs for LoRA training. */
#ifndef OXIDIZE_GRADIENT_H
#define OXIDIZE_GRADIENT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OC_GRAD_DEFAULT_LR          0.001f
#define OC_GRAD_DEFAULT_CLIP_NORM   1.0f
#define OC_GRAD_DEFAULT_WEIGHT_DECAY 0.01f
#define OC_GRAD_ADAM_BETA1          0.9f
#define OC_GRAD_ADAM_BETA2          0.999f
#define OC_GRAD_ADAM_EPS            1e-8f


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


/* Initialize config with defaults. */
OcError oc_grad_config_init(OcGradientConfig *cfg);

/* Allocate state for `n_params` parameters. All buffers zeroed. */
OcError oc_grad_init(const OcGradientConfig *config, size_t n_params,
                     OcGradientState *out);

/* Free all owned buffers and reset state. Safe on already-freed state. */
void oc_grad_free(OcGradientState *state);


/* Zero the gradient buffer. */
OcError oc_grad_zero(OcGradientState *state);

/* Accumulate `grad[0..n-1]` into the state's gradient buffer.
 * `n` must be <= state->n_params. */
OcError oc_grad_accumulate(OcGradientState *state, const float *grad, size_t n);

/* Clip the gradient buffer to `config.clip_norm` global L2 norm. */
OcError oc_grad_clip(OcGradientState *state);


/* SGD update: params[i] -= lr * grad[i]. */
OcError oc_grad_sgd_step(OcGradientState *state, float *params, size_t n);

/* Adam update: maintains m, v; uses bias correction. */
OcError oc_grad_adam_step(OcGradientState *state, float *params, size_t n);

/* AdamW update: same as Adam but weight decay applied to params directly. */
OcError oc_grad_adamw_step(OcGradientState *state, float *params, size_t n);


OcError oc_grad_set_lr(OcGradientState *state, float lr);
float     oc_grad_get_lr(const OcGradientState *state);


/* Linear layer backward pass. */
OcError oc_grad_compute_linear_backward(
    const float *input, const float *weight, const float *grad_output,
    size_t in_features, size_t out_features,
    float *grad_input, float *grad_weight);

/* Activation backward pass. */
OcError oc_grad_compute_activation_backward(
    const float *grad_output, const float *input, size_t n,
    OcGradActivationType activation_type, float *out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_GRADIENT_H */
