/* test_layer_range.c — run_layer_range + speculative stats tests. */
#include <criterion/criterion.h>
#include "oxidize/inf_model.h"
#include "oxidize/weight_storage.h"
#include "oxidize/layer_weights.h"
#include "oxidize/inference.h"
#include "oxidize/speculative.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Reuse tiny model setup. */
static void setup_tiny_model(OcInferenceModel *m)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.key_value_head_dim = 0;
    cfg.intermediate_size = 8;
    cfg.vocab_size = 16;
    cfg.context_size = 64;
    cfg.layer_count = 2;
    cfg.rms_norm_eps = 1e-5f;
    cfg.rope_theta = 10000.0f;
    cfg.embedding_scale = 1.0f;
    cfg.gelu_ffn = false;
    cfg.sandwich_norm = false;
    cfg.num_experts = 0;
    cfg.num_experts_per_tok = 0;

    oc_inf_model_init(m, &cfg);

    float *embed = malloc(64 * sizeof(float));
    for (size_t i = 0; i < 16; i++)
        for (size_t j = 0; j < 4; j++)
            embed[i * 4 + j] = (float)i;
    oc_weight_storage_f32(&m->tok_embeddings, embed, 64);

    m->norm_weight = malloc(4 * sizeof(float));
    for (size_t i = 0; i < 4; i++) m->norm_weight[i] = 1.0f;

    float *out_w = malloc(64 * sizeof(float));
    memset(out_w, 0, 64 * sizeof(float));
    for (size_t i = 0; i < 16; i++) out_w[i * 4 + (i % 4)] = 1.0f;
    oc_weight_storage_f32(&m->output_weight, out_w, 64);

    for (size_t li = 0; li < 2; li++) {
        OcLayerWeights layer;
        oc_layer_weights_init(&layer);
        layer.attn_norm = malloc(4 * sizeof(float));
        for (size_t i = 0; i < 4; i++) layer.attn_norm[i] = 1.0f;
        layer.n_attn_norm = 4;

        float *q_w = malloc(16 * sizeof(float));
        memset(q_w, 0, 16 * sizeof(float));
        for (size_t i = 0; i < 4; i++) q_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.attn_q, q_w, 16);

        float *k_w = malloc(16 * sizeof(float));
        memset(k_w, 0, 16 * sizeof(float));
        for (size_t i = 0; i < 4; i++) k_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.attn_k, k_w, 16);

        float *v_w = malloc(16 * sizeof(float));
        memset(v_w, 0, 16 * sizeof(float));
        for (size_t i = 0; i < 4; i++) v_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.attn_v, v_w, 16);

        float *ao_w = malloc(16 * sizeof(float));
        memset(ao_w, 0, 16 * sizeof(float));
        for (size_t i = 0; i < 4; i++) ao_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.attn_output, ao_w, 16);

        layer.ffn_norm = malloc(4 * sizeof(float));
        for (size_t i = 0; i < 4; i++) layer.ffn_norm[i] = 1.0f;
        layer.n_ffn_norm = 4;

        float *fg_w = malloc(32 * sizeof(float));
        memset(fg_w, 0, 32 * sizeof(float));
        for (size_t i = 0; i < 4; i++) fg_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.ffn_gate, fg_w, 32);

        float *fu_w = malloc(32 * sizeof(float));
        memset(fu_w, 0, 32 * sizeof(float));
        for (size_t i = 0; i < 4; i++) fu_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.ffn_up, fu_w, 32);

        float *fd_w = malloc(32 * sizeof(float));
        memset(fd_w, 0, 32 * sizeof(float));
        for (size_t i = 0; i < 4; i++) fd_w[i * 8 + i] = 1.0f;
        oc_weight_storage_f32(&layer.ffn_down, fd_w, 32);

        oc_inf_model_add_layer(m, &layer);
    }
    m->loaded = true;
}

Test(layer_range, run_single_layer)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    /* Embed token 5. */
    oc_inf_model_embed_token(&m, 5);

    /* Run only layer 0. */
    OcError e = oc_inf_model_run_layer_range(&m, 0, 1, 0);
    cr_assert_eq(e, OC_OK);

    /* Hidden state should be non-zero. */
    const float *x = oc_inf_model_hidden_state(&m);
    bool nonzero = false;
    for (int i = 0; i < 4; i++)
        if (fabsf(x[i]) > 0.001f) nonzero = true;
    cr_assert(nonzero);

    /* KV cache should have 1 token. */
    cr_assert_eq(oc_kv_cache_n_tokens(&m.kv_cache), 1);

    oc_inf_model_free(&m);
}

Test(layer_range, run_all_layers_split)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    oc_inf_model_embed_token(&m, 3);

    /* Run layer 0, then layer 1 separately. */
    OcError e = oc_inf_model_run_layer_range(&m, 0, 1, 0);
    cr_assert_eq(e, OC_OK);
    e = oc_inf_model_run_layer_range(&m, 1, 2, 0);
    cr_assert_eq(e, OC_OK);

    /* KV cache should have entries from both layers. */
    cr_assert_geq(oc_kv_cache_n_tokens(&m.kv_cache), 1);

    oc_inf_model_free(&m);
}

Test(layer_range, empty_range)
{
    OcInferenceModel m;
    setup_tiny_model(&m);
    /* start >= end -> no-op. */
    OcError e = oc_inf_model_run_layer_range(&m, 1, 1, 0);
    cr_assert_eq(e, OC_OK);
    oc_inf_model_free(&m);
}

Test(layer_range, out_of_bounds)
{
    OcInferenceModel m;
    setup_tiny_model(&m);
    cr_assert_neq(oc_inf_model_run_layer_range(&m, 0, 99, 0), OC_OK);
    cr_assert_neq(oc_inf_model_run_layer_range(NULL, 0, 1, 0), OC_OK);
    oc_inf_model_free(&m);
}

/* ─── Speculative stats accessors ──────────────────────────────────────── */

Test(spec_stats, acceptance_rate)
{
    OcSpeculativeStats stats = {
        .total_draft_tokens = 100,
        .accepted_draft_tokens = 75,
        .target_forward_passes = 25,
        .draft_forward_passes = 25,
        .emitted_tokens = 100,
    };
    cr_assert_float_eq(oc_speculative_acceptance_rate(&stats), 0.75, 0.001);
}

Test(spec_stats, acceptance_rate_zero)
{
    OcSpeculativeStats stats = {0, 0, 0, 0, 0};
    cr_assert_float_eq(oc_speculative_acceptance_rate(&stats), 0.0, 0.001);
}

Test(spec_stats, tokens_per_forward)
{
    OcSpeculativeStats stats = {
        .total_draft_tokens = 100,
        .accepted_draft_tokens = 75,
        .target_forward_passes = 25,
        .draft_forward_passes = 25,
        .emitted_tokens = 100,
    };
    /* 100 / 25 = 4.0 tokens per target forward. */
    cr_assert_float_eq(oc_speculative_tokens_per_target_forward(&stats), 4.0, 0.001);
}

Test(spec_stats, estimated_speedup)
{
    OcSpeculativeStats stats = {
        .total_draft_tokens = 80,
        .accepted_draft_tokens = 60,
        .target_forward_passes = 20,
        .draft_forward_passes = 20,
        .emitted_tokens = 80,
    };
    /* 80 / 20 = 4.0x speedup. */
    cr_assert_float_eq(oc_speculative_estimated_speedup(&stats), 4.0, 0.001);
}

Test(spec_stats, null_safety)
{
    cr_assert_float_eq(oc_speculative_acceptance_rate(NULL), 0.0, 0.001);
    cr_assert_float_eq(oc_speculative_tokens_per_target_forward(NULL), 0.0, 0.001);
    cr_assert_float_eq(oc_speculative_estimated_speedup(NULL), 0.0, 0.001);
}

Test(spec_stats, load_draft_null)
{
    cr_assert_neq(oc_speculative_load_draft(NULL, NULL, NULL), OC_OK);
}
