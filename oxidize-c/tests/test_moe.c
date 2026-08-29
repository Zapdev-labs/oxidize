#include <criterion/criterion.h>
#include <math.h>
#include <string.h>
#include "oxidize/moe.h"

/* Helper: build a default 4-expert, top-2, hidden=4, expert_size=3 config. */
static OcMoeConfig default_config(void)
{
    OcMoeConfig c;
    memset(&c, 0, sizeof(c));
    c.n_experts        = 4;
    c.n_active_experts = 2;
    c.expert_size      = 3;
    c.hidden_dim       = 4;
    c.routing_method   = OC_MOE_ROUTE_TOP_K;
    c.top_p            = 0.0f;
    c.router_z_loss    = 0.0f;
    c.router_aux_loss  = 0.0f;
    c.normalize_weights = true;
    return c;
}


Test(moe, router_init_free)
{
    OcMoeConfig cfg = default_config();
    OcMoeRouter r;
    cr_assert_eq(oc_moe_router_init(&r, &cfg), OC_OK);
    cr_assert_eq(r.config.n_experts, 4);
    cr_assert_eq(r.config.n_active_experts, 2);
    cr_assert_not_null(r.gate_weights);
    cr_assert_not_null(r.expert_weights);
    cr_assert_not_null(r.expert_up);
    cr_assert_not_null(r.expert_down);
    cr_assert_not_null(r.expert_usage_counts);
    cr_assert_eq(r.total_tokens, 0);
    oc_moe_router_free(&r);
    cr_assert_null(r.gate_weights);
    cr_assert_null(r.expert_weights);
}

Test(moe, router_init_null)
{
    OcMoeConfig cfg = default_config();
    cr_assert_neq(oc_moe_router_init(NULL, &cfg), OC_OK);

    OcMoeRouter r;
    cr_assert_neq(oc_moe_router_init(&r, NULL), OC_OK);
}

Test(moe, router_init_bad_config)
{
    OcMoeRouter r;
    OcMoeConfig c0 = default_config(); c0.n_experts = 0;
    cr_assert_neq(oc_moe_router_init(&r, &c0), OC_OK);

    /* n_active_experts == 0 is the documented "default to 1" case. */
    OcMoeConfig c1 = default_config(); c1.n_active_experts = 0;
    cr_assert_eq(oc_moe_router_init(&r, &c1), OC_OK);
    cr_assert_eq(r.config.n_active_experts, 1);
    oc_moe_router_free(&r);

    OcMoeConfig c2 = default_config(); c2.n_active_experts = 100; /* > n_experts */
    cr_assert_neq(oc_moe_router_init(&r, &c2), OC_OK);

    OcMoeConfig c3 = default_config(); c3.hidden_dim = 0;
    cr_assert_neq(oc_moe_router_init(&r, &c3), OC_OK);

    OcMoeConfig c4 = default_config(); c4.expert_size = 0;
    cr_assert_neq(oc_moe_router_init(&r, &c4), OC_OK);
}

Test(moe, router_free_null_safety)
{
    oc_moe_router_free(NULL);  /* should not crash */
}


Test(moe, router_set_gate)
{
    OcMoeConfig cfg = default_config();
    OcMoeRouter r;
    oc_moe_router_init(&r, &cfg);

    /* Gate weights: [4 experts × 4 hidden]. Expert 0 has all ones, rest zero.
     * So gate logit for expert 0 = sum(hidden), others = 0. */
    float gate[16] = {0};
    for (int i = 0; i < 4; i++) gate[i] = 1.0f;  /* expert 0, all dims = 1 */
    cr_assert_eq(oc_moe_router_set_gate(&r, gate), OC_OK);

    /* Verify copy. */
    for (int i = 0; i < 4; i++) {
        cr_assert_float_eq(r.gate_weights[i], 1.0f, 1e-6f);
    }
    for (int i = 4; i < 16; i++) {
        cr_assert_float_eq(r.gate_weights[i], 0.0f, 1e-6f);
    }

    oc_moe_router_free(&r);
}

Test(moe, router_set_experts)
{
    OcMoeConfig cfg = default_config();
    OcMoeRouter r;
    oc_moe_router_init(&r, &cfg);

    /* expert_weights: [4 experts × 3 expert_size × 4 hidden] = 48 floats.
     * Set expert 0 to identity-like, rest zero. */
    float exp[48] = {0};
    exp[0] = 2.0f;  /* expert 0, row 0, dim 0 */
    cr_assert_eq(oc_moe_router_set_experts(&r, exp, NULL, NULL), OC_OK);
    cr_assert_float_eq(r.expert_weights[0], 2.0f, 1e-6f);

    oc_moe_router_free(&r);
}

Test(moe, router_set_gate_null_router)
{
    cr_assert_neq(oc_moe_router_set_gate(NULL, NULL), OC_OK);
}


Test(moe, route_top_k_basic)
{
    OcMoeConfig cfg = default_config();
    OcMoeRouter r;
    oc_moe_router_init(&r, &cfg);

    /* Gate: expert 0 → strong positive logits, expert 1 → mild positive,
     * experts 2,3 → zero. With hidden = [1,1,1,1]:
     *   logit[0] = 4.0, logit[1] = 2.0, logit[2] = 0, logit[3] = 0. */
    float gate[16] = {0};
    gate[0] = gate[1] = gate[2] = gate[3] = 1.0f;       /* expert 0 */
    gate[4] = gate[5] = gate[6] = gate[7] = 0.5f;       /* expert 1 */
    oc_moe_router_set_gate(&r, gate);

    float hidden[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    OcMoeRouteResult res;
    float temp[4];
    cr_assert_eq(oc_moe_route(&r, hidden, &res, temp), OC_OK);

    cr_assert_eq(res.n_selected, 2);
    /* Top-2 should be experts 0 and 1. */
    cr_assert_eq(res.expert_indices[0], 0);
    cr_assert_eq(res.expert_indices[1], 1);
    /* Weights renormalized to sum 1. */
    float wsum = res.expert_weights[0] + res.expert_weights[1];
    cr_assert_float_eq(wsum, 1.0f, 1e-5f);
    /* Expert 0 should have higher weight than expert 1. */
    cr_assert_gt(res.expert_weights[0], res.expert_weights[1]);

    oc_moe_router_free(&r);
}

Test(moe, route_softmax_all_experts)
{
    OcMoeConfig cfg = default_config();
    cfg.routing_method = OC_MOE_ROUTE_SOFTMAX;
    cfg.normalize_weights = true;
    OcMoeRouter r;
    oc_moe_router_init(&r, &cfg);

    float gate[16] = {0};
    for (int i = 0; i < 16; i++) gate[i] = 0.25f;
    oc_moe_router_set_gate(&r, gate);

    float hidden[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    OcMoeRouteResult res;
    float temp[4];
    cr_assert_eq(oc_moe_route(&r, hidden, &res, temp), OC_OK);

    /* Softmax routing selects ALL experts. */
    cr_assert_eq(res.n_selected, 4);
    /* All weights roughly equal (since all logits equal). */
    for (uint32_t i = 0; i < 4; i++) {
        cr_assert_float_eq(res.expert_weights[i], 0.25f, 1e-4f);
    }

    oc_moe_router_free(&r);
}

Test(moe, route_top_p)
{
    OcMoeConfig cfg = default_config();
    cfg.routing_method = OC_MOE_ROUTE_TOP_P;
    cfg.top_p = 0.9f;
    cfg.n_active_experts = 1;  /* min experts */
    OcMoeRouter r;
    oc_moe_router_init(&r, &cfg);

    /* Expert 0 dominates: logit = 10.0, rest near 0. */
    float gate[16] = {0};
    gate[0] = gate[1] = gate[2] = gate[3] = 10.0f;  /* expert 0 */
    oc_moe_router_set_gate(&r, gate);

    float hidden[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    OcMoeRouteResult res;
    float temp[4];
    cr_assert_eq(oc_moe_route(&r, hidden, &res, temp), OC_OK);

    /* Expert 0 dominates (logit 40 vs 0), so top_p=0.9 selects exactly it. */
    cr_assert_eq(res.expert_indices[0], 0);
    cr_assert_eq(res.n_selected, 1);

    oc_moe_router_free(&r);
}

Test(moe, route_null_args)
{
    OcMoeConfig cfg = default_config();
    OcMoeRouter r;
    oc_moe_router_init(&r, &cfg);

    float hidden[4] = {1.0f};
    OcMoeRouteResult res;
    float temp[4];

    cr_assert_neq(oc_moe_route(NULL, hidden, &res, temp), OC_OK);
    cr_assert_neq(oc_moe_route(&r, NULL, &res, temp), OC_OK);
    cr_assert_neq(oc_moe_route(&r, hidden, NULL, temp), OC_OK);

    oc_moe_router_free(&r);
}

Test(moe, route_without_temp_buffer)
{
    /* Should malloc internally if temp_logits is NULL. */
    OcMoeConfig cfg = default_config();
    OcMoeRouter r;
    oc_moe_router_init(&r, &cfg);

    float gate[16] = {0};
    gate[0] = 1.0f;
    oc_moe_router_set_gate(&r, gate);

    float hidden[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    OcMoeRouteResult res;
    cr_assert_eq(oc_moe_route(&r, hidden, &res, NULL), OC_OK);
    cr_assert_eq(res.n_selected, 2);

    oc_moe_router_free(&r);
}


Test(moe, expert_forward_single_projection)
{
    OcMoeConfig cfg = default_config();
    OcMoeRouter r;
    oc_moe_router_init(&r, &cfg);

    /* Expert 0: identity-like (2x), others zero.
     * expert_weights[0] is [3 × 4], row 0 = [2,0,0,0], rest 0. */
    float exp[48] = {0};
    exp[0] = 2.0f;
    oc_moe_router_set_experts(&r, exp, NULL, NULL);

    /* Zero out up/down so single-projection path is used. */
    free(r.expert_up);
    free(r.expert_down);
    r.expert_up = NULL;
    r.expert_down = NULL;

    float x[4] = {3.0f, 5.0f, 7.0f, 9.0f};
    float out[3] = {0};
    size_t out_len = 0;
    cr_assert_eq(oc_moe_expert_forward(&r, 0, x, out, &out_len, NULL), OC_OK);
    cr_assert_eq(out_len, 3);
    /* out[0] = 2 * 3 = 6, out[1] = 0, out[2] = 0. */
    cr_assert_float_eq(out[0], 6.0f, 1e-5f);
    cr_assert_float_eq(out[1], 0.0f, 1e-5f);
    cr_assert_float_eq(out[2], 0.0f, 1e-5f);

    oc_moe_router_free(&r);
}

Test(moe, expert_forward_swiglu)
{
    OcMoeConfig cfg = default_config();
    /* Use hidden=2, expert_size=2 for simplicity. */
    cfg.hidden_dim = 2;
    cfg.expert_size = 2;
    OcMoeRouter r;
    oc_moe_router_init(&r, &cfg);

    /* Expert 0: gate=identity, up=ones, down=identity. */
    float gate_proj[16] = {0};  /* [4 experts × 2 es × 2 hd] */
    float up_proj[16]   = {0};
    float down_proj[16]  = {0};
    /* Expert 0: gate = [[1,0],[0,1]], up = [[1,0],[0,1]], down = [[1,0],[0,1]] */
    gate_proj[0] = 1.0f; gate_proj[3] = 1.0f;  /* I */
    up_proj[0]   = 1.0f; up_proj[3]   = 1.0f;
    down_proj[0] = 1.0f; down_proj[3]  = 1.0f;
    oc_moe_router_set_experts(&r, gate_proj, up_proj, down_proj);

    float x[2] = {2.0f, 3.0f};
    float out[2] = {0};
    size_t out_len = 0;
    float temp[2];
    cr_assert_eq(oc_moe_expert_forward(&r, 0, x, out, &out_len, temp), OC_OK);
    cr_assert_eq(out_len, 2);

    /* silu(2) = 2/(1+e^-2) ≈ 1.7616, out[0] = silu(2)*2 ≈ 3.5232 */
    /* silu(3) = 3/(1+e^-3) ≈ 2.8557, out[1] = silu(3)*3 ≈ 8.5672 */
    float expected0 = (2.0f / (1.0f + expf(-2.0f))) * 2.0f;
    float expected1 = (3.0f / (1.0f + expf(-3.0f))) * 3.0f;
    cr_assert_float_eq(out[0], expected0, 1e-4f);
    cr_assert_float_eq(out[1], expected1, 1e-4f);

    oc_moe_router_free(&r);
}

Test(moe, expert_forward_invalid_idx)
{
    OcMoeConfig cfg = default_config();
    OcMoeRouter r;
    oc_moe_router_init(&r, &cfg);

    float x[4] = {1.0f};
    float out[3] = {0};
    size_t out_len = 0;
    /* expert_idx 100 >= n_experts(4) → OC_ERR_MODEL */
    cr_assert_eq(oc_moe_expert_forward(&r, 100, x, out, &out_len, NULL),
                 OC_ERR_MODEL);

    oc_moe_router_free(&r);
}

Test(moe, expert_forward_null_args)
{
    OcMoeConfig cfg = default_config();
    OcMoeRouter r;
    oc_moe_router_init(&r, &cfg);

    float x[4] = {1.0f};
    float out[3] = {0};
    size_t out_len = 0;

    cr_assert_neq(oc_moe_expert_forward(NULL, 0, x, out, &out_len, NULL),
                  OC_OK);
    cr_assert_neq(oc_moe_expert_forward(&r, 0, NULL, out, &out_len, NULL),
                  OC_OK);
    cr_assert_neq(oc_moe_expert_forward(&r, 0, x, NULL, &out_len, NULL),
                  OC_OK);
    cr_assert_neq(oc_moe_expert_forward(&r, 0, x, out, NULL, NULL),
                  OC_OK);

    oc_moe_router_free(&r);
}


Test(moe, combine_weighted_sum)
{
    OcMoeRouteResult res;
    memset(&res, 0, sizeof(res));
    res.n_selected = 2;
    res.expert_weights[0] = 0.75f;
    res.expert_weights[1] = 0.25f;

    /* Two experts, each output length 2. */
    float e0[2] = {4.0f, 8.0f};
    float e1[2] = {0.0f, 4.0f};
    const float *outs[2] = {e0, e1};

    float combined[2] = {0};
    cr_assert_eq(oc_moe_combine(&res, outs, 2, 2, combined), OC_OK);
    /* combined[0] = 0.75*4 + 0.25*0 = 3.0 */
    /* combined[1] = 0.75*8 + 0.25*4 = 7.0 */
    cr_assert_float_eq(combined[0], 3.0f, 1e-5f);
    cr_assert_float_eq(combined[1], 7.0f, 1e-5f);
}

Test(moe, combine_null_args)
{
    OcMoeRouteResult res;
    memset(&res, 0, sizeof(res));
    res.n_selected = 1;
    res.expert_weights[0] = 1.0f;

    float e0[1] = {1.0f};
    const float *outs[1] = {e0};
    float combined[1] = {0};

    cr_assert_neq(oc_moe_combine(NULL, outs, 1, 1, combined), OC_OK);
    cr_assert_neq(oc_moe_combine(&res, NULL, 1, 1, combined), OC_OK);
    cr_assert_neq(oc_moe_combine(&res, outs, 1, 1, NULL), OC_OK);
}

Test(moe, combine_zero_selected)
{
    OcMoeRouteResult res;
    memset(&res, 0, sizeof(res));
    res.n_selected = 0;

    const float *outs[1] = {NULL};
    float combined[1] = {0};
    cr_assert_neq(oc_moe_combine(&res, outs, 0, 1, combined), OC_OK);
}

Test(moe, combine_mismatched_n)
{
    OcMoeRouteResult res;
    memset(&res, 0, sizeof(res));
    res.n_selected = 2;

    float e0[1] = {1.0f};
    const float *outs[1] = {e0};
    float combined[1] = {0};
    /* n_selected=1 but result->n_selected=2 → error. */
    cr_assert_neq(oc_moe_combine(&res, outs, 1, 1, combined), OC_OK);
}


Test(moe, stats_tracking)
{
    OcMoeConfig cfg = default_config();
    OcMoeRouter r;
    oc_moe_router_init(&r, &cfg);

    /* Set gate so expert 0 always wins. */
    float gate[16] = {0};
    gate[0] = gate[1] = gate[2] = gate[3] = 5.0f;
    gate[4] = gate[5] = gate[6] = gate[7] = 1.0f;
    oc_moe_router_set_gate(&r, gate);

    float hidden[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    OcMoeRouteResult res;
    float temp[4];

    /* Route 5 tokens. */
    for (int i = 0; i < 5; i++) {
        cr_assert_eq(oc_moe_route(&r, hidden, &res, temp), OC_OK);
    }

    OcMoeStats stats;
    cr_assert_eq(oc_moe_get_stats(&r, &stats), OC_OK);
    cr_assert_eq(stats.total_tokens, 5);
    cr_assert_eq(stats.n_experts, 4);
    cr_assert_eq(stats.n_active_experts, 2);

    /* Expert 0 should be used 5 times, expert 1 should be used 5 times
     * (as the second pick). Experts 2,3 unused. */
    cr_assert_eq(stats.expert_usage_counts[0], 5);
    cr_assert_eq(stats.expert_usage_counts[1], 5);
    cr_assert_eq(stats.expert_usage_counts[2], 0);
    cr_assert_eq(stats.expert_usage_counts[3], 0);

    /* Entropy should be > 0 (non-degenerate distribution). */
    cr_assert_gt(stats.routing_entropy, 0.0);

    oc_moe_stats_free(&stats);
    cr_assert_null(stats.expert_usage_counts);
    oc_moe_router_free(&r);
}

Test(moe, stats_null_args)
{
    OcMoeStats stats;
    cr_assert_neq(oc_moe_get_stats(NULL, &stats), OC_OK);

    OcMoeConfig cfg = default_config();
    OcMoeRouter r;
    oc_moe_router_init(&r, &cfg);
    cr_assert_neq(oc_moe_get_stats(&r, NULL), OC_OK);
    oc_moe_router_free(&r);
}

Test(moe, stats_free_null)
{
    oc_moe_stats_free(NULL);  /* should not crash */
}


Test(moe, stats_format_json)
{
    OcMoeStats stats;
    memset(&stats, 0, sizeof(stats));
    stats.total_tokens = 100;
    stats.n_experts = 4;
    stats.n_active_experts = 2;
    stats.routing_entropy = 1.5;
    stats.expert_usage_counts = malloc(4 * sizeof(uint64_t));
    stats.expert_usage_counts[0] = 30;
    stats.expert_usage_counts[1] = 25;
    stats.expert_usage_counts[2] = 25;
    stats.expert_usage_counts[3] = 20;

    char buf[512];
    size_t n = oc_moe_stats_format(&stats, buf, sizeof(buf));
    cr_assert_gt(n, 0);
    cr_assert_lt(n, sizeof(buf));

    /* Verify JSON contains expected keys. */
    cr_assert(strstr(buf, "\"total_tokens\": 100") != NULL);
    cr_assert(strstr(buf, "\"n_experts\": 4") != NULL);
    cr_assert(strstr(buf, "\"n_active_experts\": 2") != NULL);
    cr_assert(strstr(buf, "\"routing_entropy\": 1.500000") != NULL);
    cr_assert(strstr(buf, "30") != NULL);
    cr_assert(strstr(buf, "25") != NULL);
    cr_assert(strstr(buf, "20") != NULL);

    oc_moe_stats_free(&stats);
}

Test(moe, stats_format_null)
{
    char buf[64];
    cr_assert_eq(oc_moe_stats_format(NULL, buf, sizeof(buf)), 0);
}

Test(moe, stats_format_size_query)
{
    OcMoeStats stats;
    memset(&stats, 0, sizeof(stats));
    stats.total_tokens = 42;
    stats.n_experts = 2;
    stats.n_active_experts = 1;
    stats.routing_entropy = 0.5;
    stats.expert_usage_counts = malloc(2 * sizeof(uint64_t));
    stats.expert_usage_counts[0] = 21;
    stats.expert_usage_counts[1] = 21;

    /* NULL buf → should return needed length. */
    size_t needed = oc_moe_stats_format(&stats, NULL, 0);
    cr_assert_gt(needed, 0);

    /* Small buf → should still return > 0 (truncated write is OK). */
    char small[4];
    size_t n = oc_moe_stats_format(&stats, small, sizeof(small));
    cr_assert_gt(n, 0);
    (void)needed;

    oc_moe_stats_free(&stats);
}


Test(moe, routing_method_name)
{
    cr_assert_str_eq(oc_moe_routing_method_name(OC_MOE_ROUTE_TOP_K), "top_k");
    cr_assert_str_eq(oc_moe_routing_method_name(OC_MOE_ROUTE_TOP_P), "top_p");
    cr_assert_str_eq(oc_moe_routing_method_name(OC_MOE_ROUTE_SOFTMAX), "softmax");
    cr_assert_str_eq(oc_moe_routing_method_name(99), "unknown");
}


Test(moe, e2e_route_forward_combine)
{
    OcMoeConfig cfg = default_config();
    /* hidden=2, expert_size=2, 3 experts, top-1. */
    cfg.hidden_dim = 2;
    cfg.expert_size = 2;
    cfg.n_experts = 3;
    cfg.n_active_experts = 1;
    cfg.routing_method = OC_MOE_ROUTE_TOP_K;
    cfg.normalize_weights = true;
    OcMoeRouter r;
    oc_moe_router_init(&r, &cfg);

    /* Gate: expert 0 dominates. */
    float gate[6] = {0};  /* [3 × 2] */
    gate[0] = 10.0f; gate[1] = 10.0f;  /* expert 0 */
    oc_moe_router_set_gate(&r, gate);

    /* Expert weights: expert 0 is identity (2x). */
    float exp[12] = {0};  /* [3 × 2 × 2] */
    exp[0] = 2.0f; exp[3] = 2.0f;  /* expert 0 = 2*I */
    oc_moe_router_set_experts(&r, exp, NULL, NULL);
    /* Disable SwiGLU path. */
    free(r.expert_up);
    free(r.expert_down);
    r.expert_up = NULL;
    r.expert_down = NULL;

    float x[2] = {3.0f, 7.0f};
    OcMoeRouteResult res;
    float temp[3];
    cr_assert_eq(oc_moe_route(&r, x, &res, temp), OC_OK);
    cr_assert_eq(res.n_selected, 1);
    cr_assert_eq(res.expert_indices[0], 0);

    float exp_out[2] = {0};
    size_t out_len = 0;
    cr_assert_eq(oc_moe_expert_forward(&r, 0, x, exp_out, &out_len, NULL), OC_OK);
    cr_assert_eq(out_len, 2);
    cr_assert_float_eq(exp_out[0], 6.0f, 1e-5f);
    cr_assert_float_eq(exp_out[1], 14.0f, 1e-5f);

    /* Combine (single expert, weight = 1.0 after normalization). */
    const float *outs[1] = {exp_out};
    float combined[2] = {0};
    cr_assert_eq(oc_moe_combine(&res, outs, 1, 2, combined), OC_OK);
    cr_assert_float_eq(combined[0], 6.0f, 1e-5f);
    cr_assert_float_eq(combined[1], 14.0f, 1e-5f);

    oc_moe_router_free(&r);
}
