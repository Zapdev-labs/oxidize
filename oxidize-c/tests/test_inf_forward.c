/* test_inf_forward.c — Forward pass tests for OcInferenceModel. */
#include <criterion/criterion.h>
#include "oxidize/inf_model.h"
#include "oxidize/weight_storage.h"
#include "oxidize/layer_weights.h"
#include "oxidize/inference.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Helper: create a minimal 2-layer model with identity-like weights. */
static void setup_tiny_model(OcInferenceModel *m)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.hidden_size = 4;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.key_value_head_dim = 0;  /* = hidden/heads = 2 */
    cfg.intermediate_size = 8;
    cfg.vocab_size = 16;
    cfg.context_size = 32;
    cfg.layer_count = 2;
    cfg.rms_norm_eps = 1e-5f;
    cfg.rope_theta = 10000.0f;
    cfg.embedding_scale = 1.0f;
    cfg.gelu_ffn = false;
    cfg.sandwich_norm = false;
    cfg.num_experts = 0;
    cfg.num_experts_per_tok = 0;

    oc_inf_model_init(m, &cfg);

    /* Token embeddings: F32, vocab_size * hidden = 16*4 = 64 floats.
     * token i -> [i, i, i, i] */
    float *embed = malloc(64 * sizeof(float));
    for (size_t i = 0; i < 16; i++)
        for (size_t j = 0; j < 4; j++)
            embed[i * 4 + j] = (float)i;
    oc_weight_storage_f32(&m->tok_embeddings, embed, 64);

    /* Norm weight: all ones. */
    m->norm_weight = malloc(4 * sizeof(float));
    for (size_t i = 0; i < 4; i++) m->norm_weight[i] = 1.0f;

    /* Output weight: identity-ish [vocab, hidden] = [16, 4].
     * Row i has 1.0 in column (i%4). */
    float *out_w = malloc(64 * sizeof(float));
    memset(out_w, 0, 64 * sizeof(float));
    for (size_t i = 0; i < 16; i++)
        out_w[i * 4 + (i % 4)] = 1.0f;
    oc_weight_storage_f32(&m->output_weight, out_w, 64);

    /* Create 2 layers. */
    for (size_t li = 0; li < 2; li++) {
        OcLayerWeights layer;
        oc_layer_weights_init(&layer);

        /* Attn norm: all ones. */
        layer.attn_norm = malloc(4 * sizeof(float));
        for (size_t i = 0; i < 4; i++) layer.attn_norm[i] = 1.0f;
        layer.n_attn_norm = 4;

        /* Q projection: identity [4, 4]. */
        float *q_w = malloc(16 * sizeof(float));
        memset(q_w, 0, 16 * sizeof(float));
        for (size_t i = 0; i < 4; i++) q_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.attn_q, q_w, 16);

        /* K projection: identity. */
        float *k_w = malloc(16 * sizeof(float));
        memset(k_w, 0, 16 * sizeof(float));
        for (size_t i = 0; i < 4; i++) k_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.attn_k, k_w, 16);

        /* V projection: identity. */
        float *v_w = malloc(16 * sizeof(float));
        memset(v_w, 0, 16 * sizeof(float));
        for (size_t i = 0; i < 4; i++) v_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.attn_v, v_w, 16);

        /* Attn output: identity. */
        float *ao_w = malloc(16 * sizeof(float));
        memset(ao_w, 0, 16 * sizeof(float));
        for (size_t i = 0; i < 4; i++) ao_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.attn_output, ao_w, 16);

        /* FFN norm: all ones. */
        layer.ffn_norm = malloc(4 * sizeof(float));
        for (size_t i = 0; i < 4; i++) layer.ffn_norm[i] = 1.0f;
        layer.n_ffn_norm = 4;

        /* FFN gate: [8, 4] - first 4 rows identity, next 4 zero. */
        float *fg_w = malloc(32 * sizeof(float));
        memset(fg_w, 0, 32 * sizeof(float));
        for (size_t i = 0; i < 4; i++) fg_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.ffn_gate, fg_w, 32);

        /* FFN up: [8, 4] - first 4 rows identity. */
        float *fu_w = malloc(32 * sizeof(float));
        memset(fu_w, 0, 32 * sizeof(float));
        for (size_t i = 0; i < 4; i++) fu_w[i * 4 + i] = 1.0f;
        oc_weight_storage_f32(&layer.ffn_up, fu_w, 32);

        /* FFN down: [4, 8] - first 4 cols identity. */
        float *fd_w = malloc(32 * sizeof(float));
        memset(fd_w, 0, 32 * sizeof(float));
        for (size_t i = 0; i < 4; i++) fd_w[i * 8 + i] = 1.0f;
        oc_weight_storage_f32(&layer.ffn_down, fd_w, 32);

        oc_inf_model_add_layer(m, &layer);
    }

    m->loaded = true;
}

Test(inf_fwd, embed_token)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    oc_inf_model_embed_token(&m, 3);
    const float *x = oc_inf_model_hidden_state(&m);
    cr_assert_not_null(x);
    /* Token 3 -> [3, 3, 3, 3]. */
    for (int i = 0; i < 4; i++)
        cr_assert_float_eq(x[i], 3.0f, 0.001f);

    oc_inf_model_free(&m);
}

Test(inf_fwd, embed_token_clamp)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    /* Token beyond vocab -> clamped to last (15). */
    oc_inf_model_embed_token(&m, 999);
    const float *x = oc_inf_model_hidden_state(&m);
    for (int i = 0; i < 4; i++)
        cr_assert_float_eq(x[i], 15.0f, 0.001f);

    oc_inf_model_free(&m);
}

Test(inf_fwd, embed_token_scale)
{
    OcInferenceModel m;
    setup_tiny_model(&m);
    m.config.embedding_scale = 2.0f;

    oc_inf_model_embed_token(&m, 3);
    const float *x = oc_inf_model_hidden_state(&m);
    for (int i = 0; i < 4; i++)
        cr_assert_float_eq(x[i], 6.0f, 0.001f);

    oc_inf_model_free(&m);
}

Test(inf_fwd, hidden_state_and_set)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    float vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    OcError e = oc_inf_model_set_hidden_state(&m, vals, 4);
    cr_assert_eq(e, OC_OK);
    const float *x = oc_inf_model_hidden_state(&m);
    for (int i = 0; i < 4; i++)
        cr_assert_float_eq(x[i], vals[i], 0.001f);

    /* Wrong len. */
    e = oc_inf_model_set_hidden_state(&m, vals, 3);
    cr_assert_neq(e, OC_OK);

    oc_inf_model_free(&m);
}

Test(inf_fwd, config_hidden_size)
{
    OcInferenceModel m;
    setup_tiny_model(&m);
    cr_assert_eq(oc_inf_model_config_hidden_size(&m), 4);
    cr_assert_eq(oc_inf_model_config_hidden_size(NULL), 0);
    oc_inf_model_free(&m);
}

Test(inf_fwd, apply_final_norm)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    float input[4] = {3.0f, 4.0f, 0.0f, 0.0f};
    float out[4];
    OcError e = oc_inf_model_apply_final_norm(&m, input, out, 4);
    cr_assert_eq(e, OC_OK);

    /* RMSNorm: rms = sqrt((9+16+0+0)/4 + eps) = sqrt(25/4 + eps) ~= 2.5
     * out = input / rms * weight(=1) = [1.2, 1.6, 0, 0]. */
    cr_assert_float_eq(out[0], 1.2f, 0.01f);
    cr_assert_float_eq(out[1], 1.6f, 0.01f);
    cr_assert_float_eq(out[2], 0.0f, 0.01f);
    cr_assert_float_eq(out[3], 0.0f, 0.01f);

    oc_inf_model_free(&m);
}

Test(inf_fwd, apply_final_norm_bad_len)
{
    OcInferenceModel m;
    setup_tiny_model(&m);
    float in[4] = {1, 1, 1, 1};
    float out[4];
    cr_assert_neq(oc_inf_model_apply_final_norm(&m, in, out, 3), OC_OK);
    oc_inf_model_free(&m);
}

Test(inf_fwd, final_norm_weight)
{
    OcInferenceModel m;
    setup_tiny_model(&m);
    const float *w = oc_inf_model_final_norm_weight(&m);
    cr_assert_not_null(w);
    cr_assert_float_eq(w[0], 1.0f, 0.001f);
    oc_inf_model_free(&m);
}

Test(inf_fwd, has_mtp_no)
{
    OcInferenceModel m;
    setup_tiny_model(&m);
    cr_assert_eq(oc_inf_model_has_mtp(&m), false);
    cr_assert_eq(oc_inf_model_nextn_predict_layers(&m), 0);
    oc_inf_model_free(&m);
}

Test(inf_fwd, lm_head_logits)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    float normed[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    float logits[16];
    OcError e = oc_inf_model_lm_head_logits_from_normed(&m, normed, 4, logits, 16);
    cr_assert_eq(e, OC_OK);

    /* Output weight row i has 1.0 at col (i%4).
     * normed = [0, 1, 0, 0] -> logits[i] = out_w[i*4 + (i%4)] * normed[i%4].
     * For i%4==1: logits = 1*1 = 1. Others = 0. */
    for (int i = 0; i < 16; i++) {
        if (i % 4 == 1)
            cr_assert_float_eq(logits[i], 1.0f, 0.01f);
        else
            cr_assert_float_eq(logits[i], 0.0f, 0.01f);
    }

    oc_inf_model_free(&m);
}

Test(inf_fwd, lm_head_bad_len)
{
    OcInferenceModel m;
    setup_tiny_model(&m);
    float n[4] = {1, 1, 1, 1};
    float l[16];
    cr_assert_neq(oc_inf_model_lm_head_logits_from_normed(&m, n, 3, l, 16), OC_OK);
    cr_assert_neq(oc_inf_model_lm_head_logits_from_normed(&m, n, 4, l, 15), OC_OK);
    oc_inf_model_free(&m);
}

Test(inf_fwd, final_head_from_workspace)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    /* Embed token 5 -> [5,5,5,5]. */
    oc_inf_model_embed_token(&m, 5);

    float *logits = NULL;
    size_t logits_len = 0;
    OcError e = oc_inf_model_final_head_from_workspace(&m, &logits, &logits_len);
    cr_assert_eq(e, OC_OK);
    cr_assert_not_null(logits);
    cr_assert_eq(logits_len, 16);

    /* After RMSNorm of [5,5,5,5] with weight=1: rms = sqrt(25+eps) ~= 5.
     * normed = [1,1,1,1]. Logits = out_w @ normed = each row sums col j=normed[j].
     * Row i has 1.0 at col (i%4), so logits[i] = 1.0. */
    for (int i = 0; i < 16; i++)
        cr_assert_float_eq(logits[i], 1.0f, 0.01f);

    /* last_output_hidden should be ~[1,1,1,1]. */
    const float *loh = oc_inf_model_last_output_hidden(&m);
    cr_assert_not_null(loh);
    for (int i = 0; i < 4; i++)
        cr_assert_float_eq(loh[i], 1.0f, 0.01f);

    oc_inf_model_free(&m);
}

Test(inf_fwd, forward_token_position0)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    /* Forward token 3 at position 0. */
    OcError e = oc_inf_model_forward_token(&m, 3, 0);
    cr_assert_eq(e, OC_OK);

    /* Hidden state should be non-zero after forward pass. */
    const float *x = oc_inf_model_hidden_state(&m);
    bool nonzero = false;
    for (int i = 0; i < 4; i++)
        if (fabsf(x[i]) > 0.001f) nonzero = true;
    cr_assert(nonzero);

    /* KV cache should have 1 token. */
    cr_assert_eq(oc_kv_cache_n_tokens(&m.kv_cache), 1);

    oc_inf_model_free(&m);
}

Test(inf_fwd, forward_token_logits)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    float *logits = NULL;
    size_t logits_len = 0;
    OcError e = oc_inf_model_forward_token_logits(&m, 5, 0, &logits, &logits_len);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(logits_len, 16);

    /* Logits should be non-trivial. */
    bool nonzero = false;
    for (size_t i = 0; i < logits_len; i++)
        if (fabsf(logits[i]) > 0.001f) nonzero = true;
    cr_assert(nonzero);

    oc_inf_model_free(&m);
}

Test(inf_fwd, forward_multi_token)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    /* Forward 3 tokens. */
    cr_assert_eq(oc_inf_model_forward_token(&m, 1, 0), OC_OK);
    cr_assert_eq(oc_inf_model_forward_token(&m, 2, 1), OC_OK);
    cr_assert_eq(oc_inf_model_forward_token(&m, 3, 2), OC_OK);

    /* KV cache should have 3 tokens. */
    cr_assert_eq(oc_kv_cache_n_tokens(&m.kv_cache), 3);

    oc_inf_model_free(&m);
}

Test(inf_fwd, forward_context_exceeded)
{
    OcInferenceModel m;
    setup_tiny_model(&m);
    m.config.context_size = 4;

    cr_assert_eq(oc_inf_model_forward_token(&m, 1, 0), OC_OK);
    cr_assert_eq(oc_inf_model_forward_token(&m, 2, 1), OC_OK);
    cr_assert_eq(oc_inf_model_forward_token(&m, 3, 2), OC_OK);
    cr_assert_eq(oc_inf_model_forward_token(&m, 4, 3), OC_OK);
    /* Position 4 exceeds context_size=4. */
    cr_assert_neq(oc_inf_model_forward_token(&m, 5, 4), OC_OK);

    oc_inf_model_free(&m);
}

Test(inf_fwd, eagle3_capture)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    /* Capture layer 0 and 1. */
    size_t layers[2] = {0, 1};
    OcError e = oc_inf_model_set_eagle3_capture_layers(&m, layers, 2);
    cr_assert_eq(e, OC_OK);

    /* Forward a token. */
    cr_assert_eq(oc_inf_model_forward_token(&m, 5, 0), OC_OK);

    /* Concat EAGLE3 features. */
    float features[8]; /* 2 layers * 4 hidden */
    e = oc_inf_model_concat_eagle3_features(&m, features, 8);
    cr_assert_eq(e, OC_OK);

    /* Features should be non-zero. */
    bool nonzero = false;
    for (int i = 0; i < 8; i++)
        if (fabsf(features[i]) > 0.001f) nonzero = true;
    cr_assert(nonzero);

    oc_inf_model_free(&m);
}

Test(inf_fwd, eagle3_missing)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    /* Capture layer 0 but don't forward. */
    size_t layers[1] = {0};
    oc_inf_model_set_eagle3_capture_layers(&m, layers, 1);

    float features[4];
    OcError e = oc_inf_model_concat_eagle3_features(&m, features, 4);
    cr_assert_neq(e, OC_OK); /* Should fail - no capture yet. */

    oc_inf_model_free(&m);
}

Test(inf_fwd, eagle3_clear)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    size_t layers[1] = {0};
    oc_inf_model_set_eagle3_capture_layers(&m, layers, 1);
    cr_assert_eq(m.eagle3_n_capture_layers, 1);

    /* Clear. */
    oc_inf_model_set_eagle3_capture_layers(&m, NULL, 0);
    cr_assert_eq(m.eagle3_n_capture_layers, 0);

    oc_inf_model_free(&m);
}

Test(inf_fwd, null_safety)
{
    cr_assert_eq(oc_inf_model_forward_token(NULL, 0, 0), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_inf_model_forward_token_logits(NULL, 0, 0, NULL, NULL), OC_ERR_INVALID_ARG);
    cr_assert_null(oc_inf_model_hidden_state(NULL));
    cr_assert_eq(oc_inf_model_config_hidden_size(NULL), 0);
    cr_assert_eq(oc_inf_model_set_hidden_state(NULL, NULL, 0), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_inf_model_apply_final_norm(NULL, NULL, NULL, 0), OC_ERR_INVALID_ARG);
    cr_assert_null(oc_inf_model_final_norm_weight(NULL));
    cr_assert_eq(oc_inf_model_has_mtp(NULL), false);
    cr_assert_eq(oc_inf_model_nextn_predict_layers(NULL), 0);
    cr_assert_null(oc_inf_model_last_output_hidden(NULL));
}

/* ─── MTP draft generation tests ───────────────────────────────────────── */

Test(inf_fwd, mtp_draft_no_mtp)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    /* Model has no MTP block, should fail. */
    uint32_t tokens[4];
    float logits[64]; /* 4 * 16 */
    size_t n;
    OcError e = oc_inf_model_draft_mtp_tokens(&m, 0, m.workspace.x, 4,
                                                 4, tokens, logits, &n);
    cr_assert_neq(e, OC_OK);

    oc_inf_model_free(&m);
}

Test(inf_fwd, mtp_draft_max_zero)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    /* max_tokens=0 should return OK with n=0. */
    uint32_t tokens[1];
    float logits[16];
    size_t n = 999;
    OcError e = oc_inf_model_draft_mtp_tokens(&m, 0, m.workspace.x, 4,
                                                 0, tokens, logits, &n);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(n, 0);

    oc_inf_model_free(&m);
}

Test(inf_fwd, mtp_draft_null_safety)
{
    OcInferenceModel m;
    setup_tiny_model(&m);

    uint32_t tokens[1];
    float logits[16];
    size_t n;
    /* NULL model. */
    cr_assert_neq(oc_inf_model_draft_mtp_tokens(NULL, 0, NULL, 0, 1,
                                                   tokens, logits, &n), OC_OK);
    /* NULL hidden. */
    cr_assert_neq(oc_inf_model_draft_mtp_tokens(&m, 0, NULL, 4, 1,
                                                   tokens, logits, &n), OC_OK);
    /* Wrong hidden len. */
    cr_assert_neq(oc_inf_model_draft_mtp_tokens(&m, 0, m.workspace.x, 3, 1,
                                                   tokens, logits, &n), OC_OK);
    /* NULL outputs. */
    cr_assert_neq(oc_inf_model_draft_mtp_tokens(&m, 0, m.workspace.x, 4, 1,
                                                   NULL, logits, &n), OC_OK);

    oc_inf_model_free(&m);
}
