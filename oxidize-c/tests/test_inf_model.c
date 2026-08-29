/* test_inf_model.c — Inference model struct tests. */
#include "framework.h"
#include "oxidize/inf_model.h"
#include <stdlib.h>

Test(infm, init)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    OcInferenceModel m;
    cr_assert_eq(oc_inf_model_init(&m, &cfg), OC_OK);
    cr_assert_not_null(m.layers);
    cr_assert_eq(m.n_layers, 0);
    cr_assert_not_null(m.kv_layer_map);
    cr_assert_not_null(m.last_output_hidden);
    cr_assert_not_null(m.workspace.x);
    cr_assert_null(m.mtp);
    cr_assert(!m.loaded);
    oc_inf_model_free(&m);
}

Test(infm, init_null)
{
    cr_assert_neq(oc_inf_model_init(NULL, NULL), OC_OK);
}

Test(infm, init_bad_config)
{
    OcInferenceModel m;
    OcInferenceConfig cfg = {0};
    /* hidden_size=0 -> workspace_for_config will fail */
    cr_assert_neq(oc_inf_model_init(&m, &cfg), OC_OK);
}

Test(infm, add_layer)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    OcInferenceModel m;
    oc_inf_model_init(&m, &cfg);

    OcLayerWeights lw;
    oc_layer_weights_init(&lw);
    cr_assert_eq(oc_inf_model_add_layer(&m, &lw), OC_OK);
    cr_assert_eq(m.n_layers, 1);
    oc_inf_model_free(&m);
}

Test(infm, add_layer_grows)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.layer_count = 2;
    OcInferenceModel m;
    oc_inf_model_init(&m, &cfg);

    /* Add more layers than initial cap. */
    for (int i = 0; i < 5; i++) {
        OcLayerWeights lw;
        oc_layer_weights_init(&lw);
        cr_assert_eq(oc_inf_model_add_layer(&m, &lw), OC_OK);
    }
    cr_assert_eq(m.n_layers, 5);
    oc_inf_model_free(&m);
}

Test(infm, add_layer_null)
{
    cr_assert_neq(oc_inf_model_add_layer(NULL, NULL), OC_OK);
}

Test(infm, set_mtp)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    OcInferenceModel m;
    oc_inf_model_init(&m, &cfg);

    OcMtpWeights *mtp = malloc(sizeof(OcMtpWeights));
    oc_mtp_weights_init(mtp);
    cr_assert_eq(oc_inf_model_set_mtp(&m, mtp), OC_OK);
    cr_assert_not_null(m.mtp);
    oc_inf_model_free(&m);
}

Test(infm, set_mtp_replaces)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    OcInferenceModel m;
    oc_inf_model_init(&m, &cfg);

    OcMtpWeights *mtp1 = malloc(sizeof(OcMtpWeights));
    oc_mtp_weights_init(mtp1);
    oc_inf_model_set_mtp(&m, mtp1);

    OcMtpWeights *mtp2 = malloc(sizeof(OcMtpWeights));
    oc_mtp_weights_init(mtp2);
    oc_inf_model_set_mtp(&m, mtp2);
    cr_assert_eq(m.mtp, mtp2);
    oc_inf_model_free(&m);
}

Test(infm, config)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.hidden_size = 2048;
    OcInferenceModel m;
    oc_inf_model_init(&m, &cfg);
    const OcInferenceConfig *c = oc_inf_model_config(&m);
    cr_assert_eq(c->hidden_size, 2048);
    oc_inf_model_free(&m);
}

Test(infm, config_null)
{
    cr_assert_null(oc_inf_model_config(NULL));
}

Test(infm, kv_layer_count)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.layer_count = 16;
    OcInferenceModel m;
    oc_inf_model_init(&m, &cfg);
    cr_assert_eq(oc_inf_model_kv_layer_count(&m), 16);
    oc_inf_model_free(&m);
}

Test(infm, kv_row_len)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    /* num_key_value_heads=32, head_dim=128 -> kv_row_len=32*128=4096 */
    OcInferenceModel m;
    oc_inf_model_init(&m, &cfg);
    size_t rlen = oc_inf_model_kv_row_len(&m);
    cr_assert_gt(rlen, 0);
    oc_inf_model_free(&m);
}

Test(infm, is_loaded)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    OcInferenceModel m;
    oc_inf_model_init(&m, &cfg);
    cr_assert(!oc_inf_model_is_loaded(&m));
    m.loaded = true;
    cr_assert(oc_inf_model_is_loaded(&m));
    oc_inf_model_free(&m);
}

Test(infm, free_null)
{
    oc_inf_model_free(NULL);
}

Test(infm, batched_decode_default)
{
    /* Should be false by default (no env var set). */
    cr_assert(!oc_inf_model_batched_decode_enabled());
}

Test(infm, kv_layer_map_default)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    cfg.layer_count = 8;
    OcInferenceModel m;
    oc_inf_model_init(&m, &cfg);
    /* Default: all layers are attention layers (identity map). */
    for (size_t i = 0; i < 8; i++)
        cr_assert_eq(m.kv_layer_map[i], (int32_t)i);
    oc_inf_model_free(&m);
}

Test(infm, workspace_allocated)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    OcInferenceModel m;
    oc_inf_model_init(&m, &cfg);
    cr_assert_not_null(m.workspace.x);
    cr_assert_not_null(m.workspace.logits);
    cr_assert_not_null(m.workspace.hidden_a);
    cr_assert_eq(m.workspace.hidden_size, 4096);
    cr_assert_eq(m.workspace.vocab_size, 32000);
    oc_inf_model_free(&m);
}

Test(infm, last_output_hidden)
{
    OcInferenceConfig cfg;
    oc_inference_config_init(&cfg);
    OcInferenceModel m;
    oc_inf_model_init(&m, &cfg);
    cr_assert_not_null(m.last_output_hidden);
    cr_assert_eq(m.last_output_hidden_len, cfg.hidden_size);
    /* Should be zero-initialized. */
    cr_assert_float_eq(m.last_output_hidden[0], 0.0f, 0.001f);
    oc_inf_model_free(&m);
}
