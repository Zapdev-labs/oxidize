/* test_lora.c — LoRA adapter inference tests. */
#include <criterion/criterion.h>
#include "oxidize/lora.h"

Test(lora, model_init_free)
{
    OcLoraModel lm;
    cr_assert_eq(oc_lora_model_init(&lm, 4), OC_OK);
    cr_assert_eq(lm.n_layers, 4);
    cr_assert_not_null(lm.q_adapters);
    cr_assert(!oc_lora_is_active(&lm));
    oc_lora_model_free(&lm);
    cr_assert_null(lm.q_adapters);
}

Test(lora, set_adapter)
{
    OcLoraModel lm;
    oc_lora_model_init(&lm, 2);

    /* Create a simple rank-2 adapter for q_proj of layer 0.
     * a: [2, 4] (rank=2, cols=4)
     * b: [3, 2] (rows=3, rank=2) */
    float a[] = {1, 0, 0, 0,  /* row 0 */
                 0, 1, 0, 0}; /* row 1 */
    float b[] = {1, 0,  /* row 0 */
                 0, 1,  /* row 1 */
                 1, 1}; /* row 2 */
    /* We need to malloc since lora takes ownership. */
    float *a_copy = malloc(sizeof(a));
    float *b_copy = malloc(sizeof(b));
    memcpy(a_copy, a, sizeof(a));
    memcpy(b_copy, b, sizeof(b));

    cr_assert_eq(oc_lora_set_adapter(&lm, 0, "q_proj",
        a_copy, b_copy, 2, 3, 4, 2.0f), OC_OK);
    cr_assert(oc_lora_is_active(&lm));
    cr_assert_eq(lm.q_adapters[0].rank, 2);
    cr_assert_eq(lm.q_adapters[0].rows, 3);
    cr_assert_eq(lm.q_adapters[0].cols, 4);
    cr_assert_float_eq(lm.q_adapters[0].alpha, 2.0f, 1e-6f);

    oc_lora_model_free(&lm);
}

Test(lora, apply_identity)
{
    /* A is identity (rank=2, cols=2), B is identity (rows=2, rank=2).
     * delta = alpha * B @ (A @ x) = alpha * x. */
    float a[] = {1, 0, 0, 1};
    float b[] = {1, 0, 0, 1};
    OcLoraAdapter adapter = {
        .a = a, .b = b, .rank = 2, .rows = 2, .cols = 2, .alpha = 1.0f
    };

    float x[] = {3.0f, 5.0f};
    float out[] = {0.0f, 0.0f};
    float temp[2];
    oc_lora_apply(&adapter, x, out, temp);

    cr_assert_float_eq(out[0], 3.0f, 1e-5f);
    cr_assert_float_eq(out[1], 5.0f, 1e-5f);
}

Test(lora, apply_scaling)
{
    /* With alpha=2, output should be 2x. */
    float a[] = {1, 0, 0, 1};
    float b[] = {1, 0, 0, 1};
    OcLoraAdapter adapter = {
        .a = a, .b = b, .rank = 2, .rows = 2, .cols = 2, .alpha = 2.0f
    };
    float x[] = {3.0f, 5.0f};
    float out[] = {10.0f, 20.0f}; /* existing values */
    float temp[2];
    oc_lora_apply(&adapter, x, out, temp);
    /* out += 2 * x */
    cr_assert_float_eq(out[0], 16.0f, 1e-5f);
    cr_assert_float_eq(out[1], 30.0f, 1e-5f);
}

Test(lora, apply_null_safety)
{
    /* Should not crash with NULL inputs. */
    OcLoraAdapter adapter = {0};
    float x[] = {1.0f};
    float out[] = {0.0f};
    float temp[1];
    oc_lora_apply(&adapter, x, out, temp);
    /* out should be unchanged. */
    cr_assert_float_eq(out[0], 0.0f, 1e-6f);
}

Test(lora, set_invalid_weight_name)
{
    OcLoraModel lm;
    oc_lora_model_init(&lm, 1);
    float dummy[1] = {0};
    cr_assert_neq(oc_lora_set_adapter(&lm, 0, "invalid_proj",
        dummy, dummy, 1, 1, 1, 1.0f), OC_OK);
    oc_lora_model_free(&lm);
}
