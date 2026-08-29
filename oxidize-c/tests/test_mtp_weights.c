/* test_mtp_weights.c — MTP weights tests. */
#include "framework.h"
#include "oxidize/mtp_weights.h"
#include "oxidize/inference.h"
#include <stdlib.h>

Test(mtpw, init)
{
    OcMtpWeights mw;
    oc_mtp_weights_init(&mw);
    cr_assert(oc_weight_storage_is_empty(&mw.eh_proj));
    cr_assert_null(mw.enorm);
    cr_assert_null(mw.hnorm);
    oc_mtp_weights_free(&mw);
}

Test(mtpw, init_null)
{
    oc_mtp_weights_init(NULL);
    oc_mtp_weights_free(NULL);
}

Test(mtpw, is_usable_empty)
{
    OcMtpWeights mw;
    OcInferenceConfig cfg;
    oc_mtp_weights_init(&mw);
    oc_inference_config_init(&cfg);
    cr_assert(!oc_mtp_weights_is_usable(&mw, &cfg));
    oc_mtp_weights_free(&mw);
}

Test(mtpw, is_usable_null)
{
    cr_assert(!oc_mtp_weights_is_usable(NULL, NULL));
}

Test(mtpw, is_usable_complete)
{
    OcMtpWeights mw;
    OcInferenceConfig cfg;
    oc_mtp_weights_init(&mw);
    oc_inference_config_init(&cfg);
    cfg.hidden_size = 4;

    /* Set up eh_proj: [2*h, h] = [8, 4] -> 32 f32 elements, output_dim(8) = 4. */
    float *eh = malloc(32 * sizeof(float));
    oc_weight_storage_f32(&mw.eh_proj, eh, 32);

    /* Set up norms. */
    mw.enorm = malloc(4 * sizeof(float));
    mw.hnorm = malloc(4 * sizeof(float));

    /* Set up layer weights. */
    float *norm = malloc(4 * sizeof(float));
    mw.layer.attn_norm = norm;
    mw.layer.n_attn_norm = 4;

    float *q = malloc(16 * sizeof(float));
    oc_weight_storage_f32(&mw.layer.attn_q, q, 16);

    float *k = malloc(16 * sizeof(float));
    oc_weight_storage_f32(&mw.layer.attn_k, k, 16);

    float *v = malloc(16 * sizeof(float));
    oc_weight_storage_f32(&mw.layer.attn_v, v, 16);

    float *out = malloc(16 * sizeof(float));
    oc_weight_storage_f32(&mw.layer.attn_output, out, 16);

    float *gate = malloc(16 * sizeof(float));
    oc_weight_storage_f32(&mw.layer.ffn_gate, gate, 16);

    float *up = malloc(16 * sizeof(float));
    oc_weight_storage_f32(&mw.layer.ffn_up, up, 16);

    float *down = malloc(16 * sizeof(float));
    oc_weight_storage_f32(&mw.layer.ffn_down, down, 16);

    cr_assert(oc_mtp_weights_is_usable(&mw, &cfg));
    oc_mtp_weights_free(&mw);
}

Test(mtpw, is_usable_missing_ffn)
{
    OcMtpWeights mw;
    OcInferenceConfig cfg;
    oc_mtp_weights_init(&mw);
    oc_inference_config_init(&cfg);
    cfg.hidden_size = 4;

    float *eh = malloc(32 * sizeof(float));
    oc_weight_storage_f32(&mw.eh_proj, eh, 32);
    mw.enorm = malloc(4 * sizeof(float));
    mw.hnorm = malloc(4 * sizeof(float));

    /* Set up attention but NOT FFN. */
    float *norm = malloc(4 * sizeof(float));
    mw.layer.attn_norm = norm;
    float *q = malloc(16 * sizeof(float));
    oc_weight_storage_f32(&mw.layer.attn_q, q, 16);
    float *k = malloc(16 * sizeof(float));
    oc_weight_storage_f32(&mw.layer.attn_k, k, 16);
    float *v = malloc(16 * sizeof(float));
    oc_weight_storage_f32(&mw.layer.attn_v, v, 16);
    float *out = malloc(16 * sizeof(float));
    oc_weight_storage_f32(&mw.layer.attn_output, out, 16);

    cr_assert(!oc_mtp_weights_is_usable(&mw, &cfg));
    oc_mtp_weights_free(&mw);
}

Test(mtpw, free_releases_all)
{
    OcMtpWeights mw;
    oc_mtp_weights_init(&mw);
    float *eh = malloc(8 * sizeof(float));
    oc_weight_storage_f32(&mw.eh_proj, eh, 8);
    mw.enorm = malloc(4 * sizeof(float));
    mw.hnorm = malloc(4 * sizeof(float));
    float *norm = malloc(4 * sizeof(float));
    mw.layer.attn_norm = norm;
    oc_mtp_weights_free(&mw);
    /* After free, should be zeroed. */
    cr_assert(oc_weight_storage_is_empty(&mw.eh_proj));
    cr_assert_null(mw.enorm);
    cr_assert_null(mw.hnorm);
}
