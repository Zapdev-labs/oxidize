/* test_inf_forward.c — Forward pass tests for OcInferenceModel. */
#include "framework.h"
#include "tiny_model.h"
#include "oxidize/inf_model.h"
#include "oxidize/weight_storage.h"
#include "oxidize/layer_weights.h"
#include "oxidize/inference.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Helper: create a minimal 2-layer model with identity-like weights. */
Test(inf_fwd, embed_token)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 32);

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
    oc_test_setup_tiny_model(&m, 32);

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
    oc_test_setup_tiny_model(&m, 32);
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
    oc_test_setup_tiny_model(&m, 32);

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
    oc_test_setup_tiny_model(&m, 32);
    cr_assert_eq(oc_inf_model_config_hidden_size(&m), 4);
    cr_assert_eq(oc_inf_model_config_hidden_size(NULL), 0);
    oc_inf_model_free(&m);
}

Test(inf_fwd, apply_final_norm)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 32);

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
    oc_test_setup_tiny_model(&m, 32);
    float in[4] = {1, 1, 1, 1};
    float out[4];
    cr_assert_neq(oc_inf_model_apply_final_norm(&m, in, out, 3), OC_OK);
    oc_inf_model_free(&m);
}

Test(inf_fwd, final_norm_weight)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 32);
    const float *w = oc_inf_model_final_norm_weight(&m);
    cr_assert_not_null(w);
    cr_assert_float_eq(w[0], 1.0f, 0.001f);
    oc_inf_model_free(&m);
}

Test(inf_fwd, has_mtp_no)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 32);
    cr_assert_eq(oc_inf_model_has_mtp(&m), false);
    cr_assert_eq(oc_inf_model_nextn_predict_layers(&m), 0);
    oc_inf_model_free(&m);
}

Test(inf_fwd, lm_head_logits)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 32);

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
    oc_test_setup_tiny_model(&m, 32);
    float n[4] = {1, 1, 1, 1};
    float l[16];
    cr_assert_neq(oc_inf_model_lm_head_logits_from_normed(&m, n, 3, l, 16), OC_OK);
    cr_assert_neq(oc_inf_model_lm_head_logits_from_normed(&m, n, 4, l, 15), OC_OK);
    oc_inf_model_free(&m);
}

Test(inf_fwd, final_head_from_workspace)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 32);

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
    oc_test_setup_tiny_model(&m, 32);

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
    oc_test_setup_tiny_model(&m, 32);

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
    oc_test_setup_tiny_model(&m, 32);

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
    oc_test_setup_tiny_model(&m, 32);
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
    oc_test_setup_tiny_model(&m, 32);

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
    oc_test_setup_tiny_model(&m, 32);

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
    oc_test_setup_tiny_model(&m, 32);

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
    oc_test_setup_tiny_model(&m, 32);

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
    oc_test_setup_tiny_model(&m, 32);

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
    oc_test_setup_tiny_model(&m, 32);

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

/* ─── attention_head_dims + gemv_weight_head tests ───────────────────── */

Test(inf_fwd, attention_head_dims_basic)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.hidden_size = 8;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.key_value_head_dim = 0;

    OcLayerWeights layer;
    oc_layer_weights_init(&layer);

    uint32_t q_hd, q_h, kv_hd, kv_h;
    oc_attention_head_dims(&cfg, &layer, 8, 4, &q_hd, &q_h, &kv_hd, &kv_h);
    /* q_len=8, n_heads=4 -> q_head_dim=2, q_heads=4.
     * kv_len=4, kvh=2 -> kv_head_dim=2, kv_heads=2. */
    cr_assert_eq(q_hd, 2);
    cr_assert_eq(q_h, 4);
    cr_assert_eq(kv_hd, 2);
    cr_assert_eq(kv_h, 2);
}

Test(inf_fwd, attention_head_dims_with_q_norm)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.hidden_size = 8;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 4;

    OcLayerWeights layer;
    oc_layer_weights_init(&layer);
    /* q_norm of len 2 divides q_len=8 -> q_head_dim=2. */
    layer.attn_q_norm = malloc(2 * sizeof(float));
    layer.n_q_norm = 2;

    uint32_t q_hd, q_h, kv_hd, kv_h;
    oc_attention_head_dims(&cfg, &layer, 8, 8, &q_hd, &q_h, &kv_hd, &kv_h);
    cr_assert_eq(q_hd, 2);
    cr_assert_eq(q_h, 4);

    free(layer.attn_q_norm);
}

Test(inf_fwd, attention_head_dims_null_layer)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;

    uint32_t q_hd, q_h, kv_hd, kv_h;
    oc_attention_head_dims(&cfg, NULL, 4, 4, &q_hd, &q_h, &kv_hd, &kv_h);
    cr_assert_eq(q_hd, 2);
    cr_assert_eq(q_h, 2);
    cr_assert_eq(kv_hd, 2);
    cr_assert_eq(kv_h, 2);
}

Test(inf_fwd, gemv_weight_head_basic)
{
    /* 2 heads, each [2, 2] = identity. */
    OcWeightStorage ws;
    oc_weight_storage_init(&ws);
    float *data = malloc(8 * sizeof(float));
    /* head 0: identity [1,0,0,1], head 1: identity [1,0,0,1] */
    data[0]=1; data[1]=0; data[2]=0; data[3]=1;
    data[4]=1; data[5]=0; data[6]=0; data[7]=1;
    oc_weight_storage_f32(&ws, data, 8);

    float input[] = {3.0f, 4.0f};
    float output[2];

    /* Head 0: identity @ [3,4] = [3,4]. */
    OcError e = oc_gemv_weight_head(&ws, 2, 2, 0, 2, input, output);
    cr_assert_eq(e, OC_OK);
    cr_assert_float_eq(output[0], 3.0f, 0.01f);
    cr_assert_float_eq(output[1], 4.0f, 0.01f);

    /* Head 1: identity @ [3,4] = [3,4]. */
    e = oc_gemv_weight_head(&ws, 2, 2, 1, 2, input, output);
    cr_assert_eq(e, OC_OK);
    cr_assert_float_eq(output[0], 3.0f, 0.01f);
    cr_assert_float_eq(output[1], 4.0f, 0.01f);

    oc_weight_storage_free(&ws);
}

Test(inf_fwd, gemv_weight_head_null)
{
    float input[] = {1.0f};
    float output[1];
    cr_assert_neq(oc_gemv_weight_head(NULL, 1, 1, 0, 1, input, output), OC_OK);
    cr_assert_neq(oc_gemv_weight_head(NULL, 1, 1, 0, 0, input, output), OC_OK);
}

/* ─── Batched forward tests ───────────────────────────────────────────── */

Test(inf_fwd, layers_supported_for_batched)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 32);
    /* Standard attention + dense FFN -> should be supported. */
    cr_assert(oc_inf_model_layers_supported_for_batched(&m));
    oc_inf_model_free(&m);
}

OC_TEST_NULL_SAFE(inf_fwd, layers_supported_null,
        cr_assert_eq(oc_inf_model_layers_supported_for_batched(NULL), false);)

Test(inf_fwd, forward_tokens_batched)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 32);

    uint32_t tokens[] = {1, 2, 3};
    float *logits = NULL;
    size_t logits_len = 0;
    OcError e = oc_inf_model_forward_tokens(&m, tokens, 3, 0, true,
                                               &logits, &logits_len);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(logits_len, 16);
    cr_assert_not_null(logits);

    /* KV cache should have 3 tokens. */
    cr_assert_eq(oc_kv_cache_n_tokens(&m.kv_cache), 3);

    /* Logits should be non-zero. */
    bool nonzero = false;
    for (size_t i = 0; i < logits_len; i++)
        if (fabsf(logits[i]) > 0.001f) nonzero = true;
    cr_assert(nonzero);

    oc_inf_model_free(&m);
}

Test(inf_fwd, forward_tokens_no_logits)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 32);

    uint32_t tokens[] = {5, 3};
    OcError e = oc_inf_model_forward_tokens(&m, tokens, 2, 0, false, NULL, NULL);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(oc_kv_cache_n_tokens(&m.kv_cache), 2);

    oc_inf_model_free(&m);
}

Test(inf_fwd, forward_tokens_null)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 32);
    cr_assert_neq(oc_inf_model_forward_tokens(&m, NULL, 3, 0, false, NULL, NULL), OC_OK);
    cr_assert_neq(oc_inf_model_forward_tokens(NULL, (uint32_t[]){1}, 1, 0, false, NULL, NULL), OC_OK);
    oc_inf_model_free(&m);
}

Test(inf_fwd, forward_batch_basic)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 32);

    /* Create 2 sequences with their own SeqKv buffers. */
    size_t kv_layer_count = m.config.layer_count;
    size_t kv_len = (size_t)m.config.num_key_value_heads * oc_inference_config_kv_head_dim(&m.config);
    size_t cap = 32;

    OcSeqKv kv0, kv1;
    oc_seq_kv_init(&kv0, kv_layer_count, cap, kv_len);
    oc_seq_kv_init(&kv1, kv_layer_count, cap, kv_len);

    uint32_t tokens[] = {3, 5};
    size_t positions[] = {0, 0};
    OcSeqKv kvs[] = {kv0, kv1};

    float *logits = malloc(2 * 16 * sizeof(float));
    OcError e = oc_inf_model_forward_batch(&m, tokens, positions, kvs, 2,
                                              true, logits);
    cr_assert_eq(e, OC_OK);

    /* Each SeqKv should have 1 token written. */
    cr_assert_eq(kvs[0].len, 1);
    cr_assert_eq(kvs[1].len, 1);

    /* Logits should be non-zero. */
    bool nonzero = false;
    for (size_t i = 0; i < 2 * 16; i++)
        if (fabsf(logits[i]) > 0.001f) nonzero = true;
    cr_assert(nonzero);

    free(logits);
    oc_seq_kv_free(&kvs[0]);
    oc_seq_kv_free(&kvs[1]);
    oc_inf_model_free(&m);
}

Test(inf_fwd, forward_batch_null)
{
    OcInferenceModel m;
    oc_test_setup_tiny_model(&m, 32);
    cr_assert_neq(oc_inf_model_forward_batch(&m, NULL, NULL, NULL, 0, false, NULL), OC_OK);
    cr_assert_neq(oc_inf_model_forward_batch(NULL, NULL, NULL, NULL, 0, false, NULL), OC_OK);
    oc_inf_model_free(&m);
}
