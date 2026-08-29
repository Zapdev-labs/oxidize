/* test_gen_loop.c — Full generation loop tests. */
#include "framework.h"
#include "oxidize/generation.h"
#include "oxidize/inf_model.h"
#include "oxidize/weight_storage.h"
#include "oxidize/layer_weights.h"
#include "oxidize/inference.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Reuse the tiny model setup from test_inf_forward.c. */
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

    /* Token embeddings: token i -> [i, i, i, i]. */
    float *embed = malloc(64 * sizeof(float));
    for (size_t i = 0; i < 16; i++)
        for (size_t j = 0; j < 4; j++)
            embed[i * 4 + j] = (float)i;
    oc_weight_storage_f32(&m->tok_embeddings, embed, 64);

    /* Norm weight: all ones. */
    m->norm_weight = malloc(4 * sizeof(float));
    for (size_t i = 0; i < 4; i++) m->norm_weight[i] = 1.0f;

    /* Output weight: identity-ish. */
    float *out_w = malloc(64 * sizeof(float));
    memset(out_w, 0, 64 * sizeof(float));
    for (size_t i = 0; i < 16; i++)
        out_w[i * 4 + (i % 4)] = 1.0f;
    oc_weight_storage_f32(&m->output_weight, out_w, 64);

    /* Create 2 layers (same as test_inf_forward.c). */
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

static int s_callback_count = 0;
static int token_cb(uint32_t token, const char *piece, void *user)
{
    (void)token; (void)piece; (void)user;
    s_callback_count++;
    return 0;
}

Test(gen_loop, basic_run)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    OcGenConfig cfg;
    oc_gen_config_init(&cfg);
    cfg.max_tokens = 5;
    cfg.temperature = 0.0f;  /* greedy */
    cfg.stop_on_eos = false;

    OcGenResult result;
    oc_gen_result_init(&result, 256);

    uint32_t prompt[] = {1, 2, 3};
    s_callback_count = 0;
    OcError e = oc_gen_run(&m, prompt, 3, &cfg, &result, token_cb, NULL);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(result.n_tokens, 5);
    cr_assert_eq(result.n_prompt_tokens, 3);
    cr_assert_eq(s_callback_count, 5);
    cr_assert_gt(result.tokens_per_sec, 0.0);

    oc_gen_result_free(&result);
    oc_inf_model_free(&m);
}

Test(gen_loop, stop_on_eos)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    /* Set token 0 as stop by using greedy with stop_on_eos. */
    OcGenConfig cfg;
    oc_gen_config_init(&cfg);
    cfg.max_tokens = 10;
    cfg.temperature = 0.0f;
    cfg.stop_on_eos = true;

    OcGenResult result;
    oc_gen_result_init(&result, 256);

    uint32_t prompt[] = {1, 2};
    OcError e = oc_gen_run(&m, prompt, 2, &cfg, &result, NULL, NULL);
    cr_assert_eq(e, OC_OK);
    /* Should stop when token 0 is generated (or at max_tokens). */
    cr_assert_leq(result.n_tokens, 10);
    cr_assert_geq(result.n_tokens, 1);

    oc_gen_result_free(&result);
    oc_inf_model_free(&m);
}

Test(gen_loop, no_prompt)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    OcGenConfig cfg;
    oc_gen_config_init(&cfg);
    cfg.max_tokens = 3;
    cfg.temperature = 0.0f;
    cfg.stop_on_eos = false;

    OcGenResult result;
    oc_gen_result_init(&result, 256);

    s_callback_count = 0;
    OcError e = oc_gen_run(&m, NULL, 0, &cfg, &result, token_cb, NULL);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(result.n_tokens, 3);
    cr_assert_eq(s_callback_count, 3);

    oc_gen_result_free(&result);
    oc_inf_model_free(&m);
}

Test(gen_loop, mtp_fallback)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    /* Model has no MTP -> should fall back to standard gen. */
    OcGenConfig cfg;
    oc_gen_config_init(&cfg);
    cfg.max_tokens = 4;
    cfg.temperature = 0.0f;
    cfg.stop_on_eos = false;

    OcGenResult result;
    oc_gen_result_init(&result, 256);

    uint32_t prompt[] = {5};
    OcError e = oc_gen_run_mtp(&m, prompt, 1, &cfg, 4, &result, NULL, NULL);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(result.n_tokens, 4);

    oc_gen_result_free(&result);
    oc_inf_model_free(&m);
}

Test(gen_loop, null_safety)
{
    OcGenConfig cfg;
    oc_gen_config_init(&cfg);
    OcGenResult result;
    oc_gen_result_init(&result, 256);

    cr_assert_neq(oc_gen_run(NULL, NULL, 0, &cfg, &result, NULL, NULL), OC_OK);
    cr_assert_neq(oc_gen_run(NULL, NULL, 0, NULL, &result, NULL, NULL), OC_OK);
    cr_assert_neq(oc_gen_run_mtp(NULL, NULL, 0, &cfg, 4, &result, NULL, NULL), OC_OK);

    oc_gen_result_free(&result);
}

Test(gen_loop, result_init_free)
{
    OcGenResult result;
    OcError e = oc_gen_result_init(&result, 128);
    cr_assert_eq(e, OC_OK);
    cr_assert_not_null(result.tokens);
    cr_assert_eq(result.n_tokens, 0);
    oc_gen_result_free(&result);
    cr_assert_null(result.tokens);
}

Test(gen_loop, config_init_defaults)
{
    OcGenConfig cfg;
    oc_gen_config_init(&cfg);
    cr_assert_eq(cfg.max_tokens, 256);
    cr_assert_float_eq(cfg.temperature, 0.8f, 0.001f);
    cr_assert_float_eq(cfg.top_p, 0.9f, 0.001f);
    cr_assert_eq(cfg.top_k, 40);
    cr_assert_float_eq(cfg.repeat_penalty, 1.1f, 0.001f);
    cr_assert(cfg.stream);
    cr_assert(cfg.stop_on_eos);
}

Test(gen_loop, stop_reason)
{
    OcGenResult result;
    memset(&result, 0, sizeof(result));
    cr_assert_str_eq(oc_gen_stop_reason(&result), "error");

    result.n_tokens = 5;
    result.stopped_on_eos = false;
    cr_assert_str_eq(oc_gen_stop_reason(&result), "length");

    result.stopped_on_eos = true;
    cr_assert_str_eq(oc_gen_stop_reason(&result), "stop");

    cr_assert_str_eq(oc_gen_stop_reason(NULL), "unknown");
}
