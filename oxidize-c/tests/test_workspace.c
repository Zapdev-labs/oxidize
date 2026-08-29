/* test_workspace.c — Inference workspace tests. */
#include "framework.h"
#include "oxidize/workspace.h"
#include <string.h>

Test(ws, for_config_default)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    OcWorkspace ws;
    cr_assert_eq(oc_workspace_for_config(&ws, &cfg), OC_OK);
    cr_assert_not_null(ws.x);
    cr_assert_not_null(ws.hidden_a);
    cr_assert_not_null(ws.hidden_b);
    cr_assert_not_null(ws.intermediate_a);
    cr_assert_not_null(ws.q_full);
    cr_assert_not_null(ws.k_vec);
    cr_assert_not_null(ws.v_vec);
    cr_assert_not_null(ws.attn_result);
    cr_assert_not_null(ws.logits);
    cr_assert_eq(ws.hidden_size, 4096);
    cr_assert_eq(ws.vocab_size, 32000);
    cr_assert_gt(ws.max_qkv, 0);
    oc_workspace_free(&ws);
}

OC_TEST_NULL_SAFE(ws, for_config_null,
        cr_assert_neq(oc_workspace_for_config(NULL, NULL), OC_OK);)

Test(ws, for_config_null_ws)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cr_assert_neq(oc_workspace_for_config(NULL, &cfg), OC_OK);
}

Test(ws, for_config_null_cfg)
{
    OcWorkspace ws;
    cr_assert_neq(oc_workspace_for_config(&ws, NULL), OC_OK);
}

OC_TEST_NULL_SAFE(ws, free_null,
        oc_workspace_free(NULL);)

Test(ws, zero)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    OcWorkspace ws;
    oc_workspace_for_config(&ws, &cfg);
    /* Write some data. */
    ws.x[0] = 1.0f;
    ws.logits[0] = 2.0f;
    oc_workspace_zero(&ws);
    cr_assert_float_eq(ws.x[0], 0.0f, 0.001f);
    cr_assert_float_eq(ws.logits[0], 0.0f, 0.001f);
    oc_workspace_free(&ws);
}

Test(ws, size_bytes)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    OcWorkspace ws;
    oc_workspace_for_config(&ws, &cfg);
    size_t sz = oc_workspace_size_bytes(&ws);
    cr_assert_gt(sz, 0);
    /* At least hidden_size * 3 * sizeof(float) for x + hidden_a + hidden_b. */
    cr_assert_gt(sz, 4096 * 3 * sizeof(float));
    oc_workspace_free(&ws);
}

OC_TEST_NULL_SAFE(ws, size_bytes_null,
        cr_assert_eq(oc_workspace_size_bytes(NULL), 0);)

Test(ws, moe_config)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.num_experts = 8;
    cfg.num_experts_per_tok = 2;
    cfg.expert_intermediate_size = 1792;
    OcWorkspace ws;
    cr_assert_eq(oc_workspace_for_config(&ws, &cfg), OC_OK);
    cr_assert_eq(ws.n_experts, 8);
    cr_assert_eq(ws.n_experts_per_tok, 2);
    cr_assert_eq(ws.expert_inter, 1792);
    cr_assert_not_null(ws.moe_router_logits);
    cr_assert_not_null(ws.moe_gate_all);
    cr_assert_not_null(ws.moe_up_all);
    cr_assert_not_null(ws.moe_down_all);
    oc_workspace_free(&ws);
}

Test(ws, custom_dims)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.hidden_size = 2048;
    cfg.intermediate_size = 5632;
    cfg.num_attention_heads = 16;
    cfg.vocab_size = 64000;
    cfg.context_size = 8192;
    OcWorkspace ws;
    cr_assert_eq(oc_workspace_for_config(&ws, &cfg), OC_OK);
    cr_assert_eq(ws.hidden_size, 2048);
    cr_assert_eq(ws.vocab_size, 64000);
    cr_assert_eq(ws.intermediate_size, 5632);
    oc_workspace_free(&ws);
}

Test(ws, buffers_initialized_zero)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    OcWorkspace ws;
    oc_workspace_for_config(&ws, &cfg);
    /* All buffers should be zero-initialized. */
    cr_assert_float_eq(ws.x[0], 0.0f, 0.001f);
    cr_assert_float_eq(ws.logits[0], 0.0f, 0.001f);
    cr_assert_float_eq(ws.q_full[0], 0.0f, 0.001f);
    oc_workspace_free(&ws);
}

Test(ws, mamba_scratch)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.hidden_size = 256;  /* Less than 576 */
    OcWorkspace ws;
    oc_workspace_for_config(&ws, &cfg);
    cr_assert_not_null(ws.mamba_scratch);
    oc_workspace_free(&ws);
}

Test(ws, shortconv_buffers)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.shortconv_l_cache = 4;
    OcWorkspace ws;
    oc_workspace_for_config(&ws, &cfg);
    cr_assert_not_null(ws.shortconv_bcx);
    cr_assert_not_null(ws.shortconv_bx);
    oc_workspace_free(&ws);
}

Test(ws, double_free_safe)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    OcWorkspace ws;
    oc_workspace_for_config(&ws, &cfg);
    oc_workspace_free(&ws);
    /* Should not crash. */
    oc_workspace_free(&ws);
}
