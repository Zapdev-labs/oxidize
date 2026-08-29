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
    cr_assert_eq(oc_lora_model_init(&lm, 2), OC_OK);

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
    cr_assert_not_null(a_copy, "malloc");
    cr_assert_not_null(b_copy, "malloc");
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
    cr_assert_eq(oc_lora_model_init(&lm, 1), OC_OK);
    float dummy[1] = {0};
    cr_assert_neq(oc_lora_set_adapter(&lm, 0, "invalid_proj",
        dummy, dummy, 1, 1, 1, 1.0f), OC_OK);
    oc_lora_model_free(&lm);
}


Test(lora_plan, basic_match)
{
    const char *base[] = {"blk.0.attn_q.weight", "blk.0.attn_v.weight"};
    const char *adapter[] = {
        "blk.0.attn_q.weight.lora_a.weight",
        "blk.0.attn_q.weight.lora_b.weight",
    };

    OcLoraPlan plan;
    OcLoraPlanError e = oc_lora_plan_application(base, 2, adapter, 2, 0, &plan);
    cr_assert_eq(e, OC_LORA_PLAN_OK);
    cr_assert_eq(plan.kind, OC_ADAPTER_LORA);
    cr_assert_eq(plan.n_targets, 1);
    cr_assert_str_eq(plan.targets[0].base_tensor, "blk.0.attn_q.weight");
    cr_assert_eq(plan.n_missing, 0);
    oc_lora_plan_free(&plan);
}

Test(lora_plan, qlora_kind)
{
    const char *base[] = {"blk.0.attn_q.weight"};
    const char *adapter[] = {
        "blk.0.attn_q.weight.lora_a.weight",
        "blk.0.attn_q.weight.lora_b.weight",
    };

    OcLoraPlan plan;
    OcLoraPlanError e = oc_lora_plan_application(base, 1, adapter, 2, 1, &plan);
    cr_assert_eq(e, OC_LORA_PLAN_OK);
    cr_assert_eq(plan.kind, OC_ADAPTER_QLORA);
    oc_lora_plan_free(&plan);
}

Test(lora_plan, missing_base_tensor)
{
    const char *base[] = {"blk.0.attn_q.weight"};
    const char *adapter[] = {
        "blk.0.attn_q.weight.lora_a.weight",
        "blk.0.attn_q.weight.lora_b.weight",
        "blk.0.attn_v.weight.lora_a.weight",
        "blk.0.attn_v.weight.lora_b.weight",
    };

    OcLoraPlan plan;
    OcLoraPlanError e = oc_lora_plan_application(base, 1, adapter, 4, 0, &plan);
    cr_assert_eq(e, OC_LORA_PLAN_OK);
    cr_assert_eq(plan.n_targets, 2);
    cr_assert_eq(plan.n_missing, 1);
    cr_assert_str_eq(plan.missing_base_tensors[0], "blk.0.attn_v.weight");
    oc_lora_plan_free(&plan);
}

Test(lora_plan, missing_pair_for_a)
{
    const char *base[] = {"blk.0.attn_q.weight"};
    const char *adapter[] = {
        "blk.0.attn_q.weight.lora_a.weight",
        /* no lora_b */
    };

    OcLoraPlan plan;
    OcLoraPlanError e = oc_lora_plan_application(base, 1, adapter, 1, 0, &plan);
    cr_assert_eq(e, OC_LORA_PLAN_MISSING_PAIR_FOR_A);
}

Test(lora_plan, missing_pair_for_b)
{
    const char *base[] = {"blk.0.attn_q.weight"};
    const char *adapter[] = {
        "blk.0.attn_q.weight.lora_b.weight",
    };

    OcLoraPlan plan;
    OcLoraPlanError e = oc_lora_plan_application(base, 1, adapter, 1, 0, &plan);
    cr_assert_eq(e, OC_LORA_PLAN_MISSING_PAIR_FOR_B);
}

Test(lora_plan, duplicate_pair)
{
    const char *base[] = {"blk.0.attn_q.weight"};
    const char *adapter[] = {
        "blk.0.attn_q.weight.lora_a.weight",
        "blk.0.attn_q.weight.lora_a.weight",
        "blk.0.attn_q.weight.lora_b.weight",
    };

    OcLoraPlan plan;
    OcLoraPlanError e = oc_lora_plan_application(base, 1, adapter, 3, 0, &plan);
    cr_assert_eq(e, OC_LORA_PLAN_DUPLICATE_PAIR);
}

Test(lora_plan, multi_pair)
{
    const char *base[] = {
        "blk.0.attn_q.weight",
        "blk.0.attn_v.weight",
        "blk.1.ffn_gate.weight",
    };
    const char *adapter[] = {
        "blk.0.attn_q.weight.lora_a.weight",
        "blk.0.attn_q.weight.lora_b.weight",
        "blk.0.attn_v.weight.lora_a.weight",
        "blk.0.attn_v.weight.lora_b.weight",
        "blk.1.ffn_gate.weight.lora_a.weight",
        "blk.1.ffn_gate.weight.lora_b.weight",
    };

    OcLoraPlan plan;
    OcLoraPlanError e = oc_lora_plan_application(base, 3, adapter, 6, 0, &plan);
    cr_assert_eq(e, OC_LORA_PLAN_OK);
    cr_assert_eq(plan.n_targets, 3);
    cr_assert_eq(plan.n_missing, 0);
    oc_lora_plan_free(&plan);
}

Test(lora_plan, empty_adapters)
{
    const char *base[] = {"blk.0.attn_q.weight"};

    OcLoraPlan plan;
    OcLoraPlanError e = oc_lora_plan_application(base, 1, NULL, 0, 0, &plan);
    cr_assert_eq(e, OC_LORA_PLAN_OK);
    cr_assert_eq(plan.n_targets, 0);
    oc_lora_plan_free(&plan);
}

Test(lora_plan, null_safety)
{
    OcLoraPlan plan;
    cr_assert_eq(oc_lora_plan_application(NULL, 0, NULL, 0, 0, NULL), OC_LORA_PLAN_INVALID_ARG);
    const char *base[] = {"x"};
    cr_assert_eq(oc_lora_plan_application(base, 1, NULL, 0, 0, NULL), OC_LORA_PLAN_INVALID_ARG);
}

Test(lora_plan, no_adapters_on_base)
{
    const char *base[] = {"blk.0.attn_q.weight"};
    const char *adapter[] = {"blk.0.attn_q.weight"};

    OcLoraPlan plan;
    OcLoraPlanError e = oc_lora_plan_application(base, 1, adapter, 1, 0, &plan);
    cr_assert_eq(e, OC_LORA_PLAN_OK);
    cr_assert_eq(plan.n_targets, 0);
    oc_lora_plan_free(&plan);
}
