#include <criterion/criterion.h>
#include <math.h>
#include <string.h>

#include "oxidize/gradient.h"


Test(grad, config_init_defaults)
{
    OcGradientConfig cfg;
    cr_assert_eq(oc_grad_config_init(&cfg), OC_OK);
    cr_assert_float_eq(cfg.learning_rate, OC_GRAD_DEFAULT_LR, 1e-9f);
    cr_assert_float_eq(cfg.clip_norm, OC_GRAD_DEFAULT_CLIP_NORM, 1e-9f);
    cr_assert_float_eq(cfg.weight_decay, OC_GRAD_DEFAULT_WEIGHT_DECAY, 1e-9f);
}

Test(grad, config_init_null)
{
    cr_assert_neq(oc_grad_config_init(NULL), OC_OK);
}

Test(grad, init_free)
{
    OcGradientConfig cfg;
    oc_grad_config_init(&cfg);
    OcGradientState st;
    cr_assert_eq(oc_grad_init(&cfg, 16, &st), OC_OK);
    cr_assert_eq(st.n_params, 16u);
    cr_assert_eq(st.step_count, 0u);
    for (size_t i = 0; i < 16; i++) {
        cr_assert_float_eq(st.gradients[i], 0.0f, 1e-9f);
        cr_assert_float_eq(st.momentum[i], 0.0f, 1e-9f);
        cr_assert_float_eq(st.velocity[i], 0.0f, 1e-9f);
    }
    oc_grad_free(&st);
    cr_assert_null(st.gradients);
}

Test(grad, init_default_config)
{
    OcGradientState st;
    cr_assert_eq(oc_grad_init(NULL, 4, &st), OC_OK);
    cr_assert_float_eq(st.config.learning_rate, OC_GRAD_DEFAULT_LR, 1e-9f);
    oc_grad_free(&st);
}

Test(grad, init_bad_args)
{
    OcGradientState st;
    cr_assert_neq(oc_grad_init(NULL, 0, &st), OC_OK);
    cr_assert_neq(oc_grad_init(NULL, 4, NULL), OC_OK);
}

Test(grad, free_null_safe)
{
    oc_grad_free(NULL);
    OcGradientState st;
    memset(&st, 0, sizeof(st));
    oc_grad_free(&st);
}


Test(grad, zero_gradients)
{
    OcGradientState st;
    cr_assert_eq(oc_grad_init(NULL, 8, &st), OC_OK);
    for (size_t i = 0; i < 8; i++) st.gradients[i] = (float)i;
    cr_assert_eq(oc_grad_zero(&st), OC_OK);
    for (size_t i = 0; i < 8; i++)
        cr_assert_float_eq(st.gradients[i], 0.0f, 1e-9f);
    oc_grad_free(&st);
}

Test(grad, accumulate)
{
    OcGradientState st;
    cr_assert_eq(oc_grad_init(NULL, 4, &st), OC_OK);
    float g[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    cr_assert_eq(oc_grad_accumulate(&st, g, 4), OC_OK);
    cr_assert_eq(oc_grad_accumulate(&st, g, 4), OC_OK);
    for (size_t i = 0; i < 4; i++)
        cr_assert_float_eq(st.gradients[i], 2.0f * (float)(i + 1), 1e-6f);
    oc_grad_free(&st);
}

Test(grad, accumulate_bad_n)
{
    OcGradientState st;
    cr_assert_eq(oc_grad_init(NULL, 4, &st), OC_OK);
    float g[2] = {1.0f, 2.0f};
    cr_assert_neq(oc_grad_accumulate(&st, g, 0), OC_OK);
    cr_assert_neq(oc_grad_accumulate(&st, g, 5), OC_OK);
    oc_grad_free(&st);
}

Test(grad, clip_below_norm_no_change)
{
    OcGradientState st;
    cr_assert_eq(oc_grad_init(NULL, 3, &st), OC_OK);
    st.gradients[0] = 0.1f;
    st.gradients[1] = 0.0f;
    st.gradients[2] = 0.0f;
    cr_assert_eq(oc_grad_clip(&st), OC_OK);
    cr_assert_float_eq(st.gradients[0], 0.1f, 1e-9f);
    oc_grad_free(&st);
}

Test(grad, clip_above_norm_scales)
{
    OcGradientState st;
    cr_assert_eq(oc_grad_init(NULL, 2, &st), OC_OK);
    /* norm = sqrt(3^2 + 4^2) = 5; clip_norm default 1.0 -> scale 0.2 */
    st.gradients[0] = 3.0f;
    st.gradients[1] = 4.0f;
    cr_assert_eq(oc_grad_clip(&st), OC_OK);
    cr_assert_float_eq(st.gradients[0], 0.6f, 1e-6f);
    cr_assert_float_eq(st.gradients[1], 0.8f, 1e-6f);
    /* New norm should equal clip_norm. */
    float n = sqrtf(st.gradients[0] * st.gradients[0]
                    + st.gradients[1] * st.gradients[1]);
    cr_assert_float_eq(n, 1.0f, 1e-6f);
    oc_grad_free(&st);
}


Test(grad, sgd_step_updates_params)
{
    OcGradientState st;
    cr_assert_eq(oc_grad_init(NULL, 3, &st), OC_OK);
    st.gradients[0] = 1.0f; st.gradients[1] = 2.0f; st.gradients[2] = 3.0f;
    float params[3] = {10.0f, 20.0f, 30.0f};
    cr_assert_eq(oc_grad_sgd_step(&st, params, 3), OC_OK);
    cr_assert_float_eq(params[0], 10.0f - OC_GRAD_DEFAULT_LR * 1.0f, 1e-6f);
    cr_assert_float_eq(params[1], 20.0f - OC_GRAD_DEFAULT_LR * 2.0f, 1e-6f);
    cr_assert_float_eq(params[2], 30.0f - OC_GRAD_DEFAULT_LR * 3.0f, 1e-6f);
    cr_assert_eq(st.step_count, 1u);
    oc_grad_free(&st);
}

Test(grad, sgd_step_bad_args)
{
    OcGradientState st;
    cr_assert_eq(oc_grad_init(NULL, 3, &st), OC_OK);
    float p[3] = {0};
    cr_assert_neq(oc_grad_sgd_step(&st, p, 5), OC_OK);
    cr_assert_neq(oc_grad_sgd_step(NULL, p, 3), OC_OK);
    oc_grad_free(&st);
}

Test(grad, adam_step_reduces_loss_direction)
{
    /* With grad = +1 everywhere, params should decrease. */
    OcGradientState st;
    cr_assert_eq(oc_grad_init(NULL, 2, &st), OC_OK);
    st.gradients[0] = 1.0f;
    st.gradients[1] = 1.0f;
    float p[2] = {1.0f, 1.0f};
    cr_assert_eq(oc_grad_adam_step(&st, p, 2), OC_OK);
    cr_assert(p[0] < 1.0f);
    cr_assert(p[1] < 1.0f);
    cr_assert_eq(st.step_count, 1u);
    /* Second step: momentum/v are nonzero; params should drop further. */
    st.gradients[0] = 1.0f;
    st.gradients[1] = 1.0f;
    float p1[2] = {p[0], p[1]};
    cr_assert_eq(oc_grad_adam_step(&st, p, 2), OC_OK);
    cr_assert(p[0] < p1[0]);
    oc_grad_free(&st);
}

Test(grad, adamw_step_has_weight_decay)
{
    /* AdamW decouples weight decay: with grad=0, params should shrink by wd. */
    OcGradientState st;
    cr_assert_eq(oc_grad_init(NULL, 1, &st), OC_OK);
    st.gradients[0] = 0.0f;
    float p = 1.0f;
    cr_assert_eq(oc_grad_adamw_step(&st, &p, 1), OC_OK);
    /* Expected: p -= lr * wd * p; with v=0, m=0 the adam delta is 0. */
    float expected = 1.0f
        - OC_GRAD_DEFAULT_LR * OC_GRAD_DEFAULT_WEIGHT_DECAY * 1.0f;
    cr_assert_float_eq(p, expected, 1e-6f);
    oc_grad_free(&st);
}

Test(grad, adam_step_with_zero_grad_no_change)
{
    OcGradientState st;
    cr_assert_eq(oc_grad_init(NULL, 1, &st), OC_OK);
    st.gradients[0] = 0.0f;
    float p = 5.0f;
    cr_assert_eq(oc_grad_adam_step(&st, &p, 1), OC_OK);
    /* m=0, v=0 -> delta = 0; params unchanged. */
    cr_assert_float_eq(p, 5.0f, 1e-9f);
    oc_grad_free(&st);
}


Test(grad, set_get_lr)
{
    OcGradientState st;
    cr_assert_eq(oc_grad_init(NULL, 2, &st), OC_OK);
    cr_assert_float_eq(oc_grad_get_lr(&st), OC_GRAD_DEFAULT_LR, 1e-9f);
    cr_assert_eq(oc_grad_set_lr(&st, 0.5f), OC_OK);
    cr_assert_float_eq(oc_grad_get_lr(&st), 0.5f, 1e-9f);
    cr_assert_neq(oc_grad_set_lr(&st, -1.0f), OC_OK);
    oc_grad_free(&st);
}

Test(grad, get_lr_null)
{
    cr_assert_float_eq(oc_grad_get_lr(NULL), 0.0f, 1e-9f);
}


Test(grad, linear_backward_grad_input)
{
    /* input = [1, 2], weight = [[1,0],[0,1]] (identity 2x2), */
    float input[2]  = {1.0f, 2.0f};
    float weight[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float grad_out[2] = {3.0f, 4.0f};
    float gi[2] = {0};
    float gw[4] = {0};
    cr_assert_eq(oc_grad_compute_linear_backward(input, weight, grad_out,
                                                 2, 2, gi, gw), OC_OK);
    cr_assert_float_eq(gi[0], 3.0f, 1e-6f);
    cr_assert_float_eq(gi[1], 4.0f, 1e-6f);
    cr_assert_float_eq(gw[0], 3.0f, 1e-6f);
    cr_assert_float_eq(gw[1], 6.0f, 1e-6f);
    cr_assert_float_eq(gw[2], 4.0f, 1e-6f);
    cr_assert_float_eq(gw[3], 8.0f, 1e-6f);
}

Test(grad, linear_backward_bad_args)
{
    float in_[2] = {0};
    float w[4]   = {0};
    float go[2]  = {0};
    float gi[2]  = {0};
    float gw[4]  = {0};
    cr_assert_neq(oc_grad_compute_linear_backward(NULL, w, go, 2, 2, gi, gw),
                  OC_OK);
    cr_assert_neq(oc_grad_compute_linear_backward(in_, w, go, 0, 2, gi, gw),
                  OC_OK);
}


Test(grad, activation_relu_backward)
{
    float go[3] = {1.0f, 1.0f, 1.0f};
    float in_[3] = {-1.0f, 0.0f, 2.0f};
    float out[3] = {0};
    cr_assert_eq(oc_grad_compute_activation_backward(go, in_, 3,
                  OC_GRAD_ACT_RELU, out), OC_OK);
    cr_assert_float_eq(out[0], 0.0f, 1e-9f);
    cr_assert_float_eq(out[1], 0.0f, 1e-9f); /* ReLU at 0 is 0 */
    cr_assert_float_eq(out[2], 1.0f, 1e-9f);
}

Test(grad, activation_tanh_backward)
{
    /* d/dx tanh(x) = 1 - tanh(x)^2; at x=0 -> 1; grad_out=1 -> out=1 */
    float go[1] = {1.0f};
    float in_[1] = {0.0f};
    float out[1] = {0};
    cr_assert_eq(oc_grad_compute_activation_backward(go, in_, 1,
                  OC_GRAD_ACT_TANH, out), OC_OK);
    cr_assert_float_eq(out[0], 1.0f, 1e-6f);
}

Test(grad, activation_silu_backward_at_zero)
{
    /* silu'(0) = sigmoid(0)*(1 + 0*(1-sigmoid(0))) = 0.5 */
    float go[1] = {1.0f};
    float in_[1] = {0.0f};
    float out[1] = {0};
    cr_assert_eq(oc_grad_compute_activation_backward(go, in_, 1,
                  OC_GRAD_ACT_SILU, out), OC_OK);
    cr_assert_float_eq(out[0], 0.5f, 1e-6f);
}

Test(grad, activation_gelu_backward_at_zero)
{
    /* gelu'(0) = 0.5*(1 + erf(0)) + 0*pdf(0)/sqrt2 = 0.5 */
    float go[1] = {1.0f};
    float in_[1] = {0.0f};
    float out[1] = {0};
    cr_assert_eq(oc_grad_compute_activation_backward(go, in_, 1,
                  OC_GRAD_ACT_GELU, out), OC_OK);
    cr_assert_float_eq(out[0], 0.5f, 1e-5f);
}

Test(grad, activation_backward_bad_type)
{
    float go[1] = {1.0f};
    float in_[1] = {0.0f};
    float out[1] = {0};
    cr_assert_neq(oc_grad_compute_activation_backward(go, in_, 1,
                  (OcGradActivationType)99, out), OC_OK);
}
