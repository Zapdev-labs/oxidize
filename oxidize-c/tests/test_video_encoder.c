#include <criterion/criterion.h>
#include <string.h>

#include "oxidize/video_encoder.h"


Test(video_encoder, init_good)
{
    OcVideoEncoderConfig cfg = {8, 8, 16};
    OcVideoEncoder enc;
    cr_assert_eq(oc_video_encoder_init(&enc, &cfg), OC_OK, "");
    cr_assert_eq(enc.config.vision_hidden, 8u, "");
    cr_assert_eq(enc.config.temporal_hidden, 8u, "");
    cr_assert_eq(enc.config.llm_hidden, 16u, "");
    cr_assert_null(enc.output_tokens, "");
    cr_assert_eq(enc.n_tokens, 0u, "");
    oc_video_encoder_free(&enc);
}

Test(video_encoder, init_null_enc)
{
    OcVideoEncoderConfig cfg = {8, 8, 16};
    cr_assert_eq(oc_video_encoder_init(NULL, &cfg), OC_ERR_INVALID_ARG, "");
}

Test(video_encoder, init_null_cfg)
{
    OcVideoEncoder enc;
    cr_assert_eq(oc_video_encoder_init(&enc, NULL), OC_ERR_INVALID_ARG, "");
}

Test(video_encoder, init_bad_dims)
{
    OcVideoEncoder enc;
    OcVideoEncoderConfig cfg;
    cfg.vision_hidden = 0; cfg.temporal_hidden = 8; cfg.llm_hidden = 16;
    cr_assert_eq(oc_video_encoder_init(&enc, &cfg), OC_ERR_INVALID_ARG, "");
    cfg.vision_hidden = 8; cfg.temporal_hidden = 0;
    cr_assert_eq(oc_video_encoder_init(&enc, &cfg), OC_ERR_INVALID_ARG, "");
    cfg.temporal_hidden = 8; cfg.llm_hidden = 0;
    cr_assert_eq(oc_video_encoder_init(&enc, &cfg), OC_ERR_INVALID_ARG, "");
}

Test(video_encoder, free_null_is_safe)
{
    oc_video_encoder_free(NULL);
    cr_assert(true, "");
}


Test(video_encoder, encode_basic)
{
    OcVideoEncoderConfig cfg = {4, 4, 4};
    OcVideoEncoder enc;
    oc_video_encoder_init(&enc, &cfg);
    float emb[8] = { /* 2 frames x 4 dims */
        0.10f, 0.20f, 0.30f, 0.40f,
        0.50f, 0.60f, 0.70f, 0.80f,
    };
    cr_assert_eq(oc_video_encoder_encode(&enc, emb, 2, 4), OC_OK, "");
    cr_assert_eq(enc.n_tokens, 2u, "");
    cr_assert_not_null(enc.output_tokens, "");
    /* frame 0 */
    cr_assert_float_eq(enc.output_tokens[0], 0.10f, 1e-6, "");
    cr_assert_float_eq(enc.output_tokens[3], 0.40f, 1e-6, "");
    /* frame 1 */
    cr_assert_float_eq(enc.output_tokens[4], 0.50f, 1e-6, "");
    cr_assert_float_eq(enc.output_tokens[7], 0.80f, 1e-6, "");
    oc_video_encoder_free(&enc);
}

Test(video_encoder, encode_zero_frames)
{
    OcVideoEncoderConfig cfg = {4, 4, 4};
    OcVideoEncoder enc;
    oc_video_encoder_init(&enc, &cfg);
    float emb[1] = {0.f};
    cr_assert_eq(oc_video_encoder_encode(&enc, emb, 0, 4), OC_OK, "");
    cr_assert_eq(enc.n_tokens, 0u, "");
    oc_video_encoder_free(&enc);
}

Test(video_encoder, encode_dim_mismatch)
{
    OcVideoEncoderConfig cfg = {4, 4, 4};
    OcVideoEncoder enc;
    oc_video_encoder_init(&enc, &cfg);
    float emb[8] = {0};
    cr_assert_eq(oc_video_encoder_encode(&enc, emb, 2, 8),
                 OC_ERR_INVALID_ARG, "frame_dim != vision_hidden");
    oc_video_encoder_free(&enc);
}

Test(video_encoder, encode_null_args)
{
    OcVideoEncoderConfig cfg = {4, 4, 4};
    OcVideoEncoder enc;
    oc_video_encoder_init(&enc, &cfg);
    float emb[4] = {0};
    cr_assert_eq(oc_video_encoder_encode(NULL, emb, 1, 4), OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_video_encoder_encode(&enc, NULL, 1, 4), OC_ERR_INVALID_ARG, "");
    oc_video_encoder_free(&enc);
}

Test(video_encoder, encode_pads_when_llm_larger)
{
    OcVideoEncoderConfig cfg = {2, 2, 4};
    OcVideoEncoder enc;
    oc_video_encoder_init(&enc, &cfg);
    float emb[2] = {0.5f, 0.7f}; /* 1 frame x 2 dims */
    cr_assert_eq(oc_video_encoder_encode(&enc, emb, 1, 2), OC_OK, "");
    cr_assert_float_eq(enc.output_tokens[0], 0.5f, 1e-6, "");
    cr_assert_float_eq(enc.output_tokens[1], 0.7f, 1e-6, "");
    cr_assert_float_eq(enc.output_tokens[2], 0.f, 1e-6, "padded zero");
    cr_assert_float_eq(enc.output_tokens[3], 0.f, 1e-6, "padded zero");
    oc_video_encoder_free(&enc);
}

Test(video_encoder, encode_multiple_calls_overwrite)
{
    OcVideoEncoderConfig cfg = {2, 2, 2};
    OcVideoEncoder enc;
    oc_video_encoder_init(&enc, &cfg);
    float a[4] = {1.f, 2.f, 3.f, 4.f}; /* 2 frames x 2 */
    float b[2] = {9.f, 8.f};          /* 1 frame  x 2 */
    cr_assert_eq(oc_video_encoder_encode(&enc, a, 2, 2), OC_OK, "");
    cr_assert_eq(enc.n_tokens, 2u, "");
    cr_assert_eq(oc_video_encoder_encode(&enc, b, 1, 2), OC_OK, "");
    cr_assert_eq(enc.n_tokens, 1u, "overwritten count");
    cr_assert_float_eq(enc.output_tokens[0], 9.f, 1e-6, "");
    oc_video_encoder_free(&enc);
}


Test(video_encoder, n_tokens_before_encode)
{
    OcVideoEncoderConfig cfg = {2, 2, 2};
    OcVideoEncoder enc;
    oc_video_encoder_init(&enc, &cfg);
    cr_assert_eq(oc_video_encoder_n_tokens(&enc), 0u, "");
    cr_assert_eq(oc_video_encoder_n_tokens(NULL), 0u, "");
    oc_video_encoder_free(&enc);
}

Test(video_encoder, output_ptr_before_encode)
{
    OcVideoEncoderConfig cfg = {2, 2, 2};
    OcVideoEncoder enc;
    oc_video_encoder_init(&enc, &cfg);
    cr_assert_null(oc_video_encoder_output(&enc), "");
    cr_assert_null(oc_video_encoder_output(NULL), "");
    oc_video_encoder_free(&enc);
}

Test(video_encoder, output_ptr_after_encode)
{
    OcVideoEncoderConfig cfg = {2, 2, 2};
    OcVideoEncoder enc;
    oc_video_encoder_init(&enc, &cfg);
    float emb[2] = {0.f, 0.f};
    oc_video_encoder_encode(&enc, emb, 1, 2);
    cr_assert_not_null(oc_video_encoder_output(&enc), "");
    oc_video_encoder_free(&enc);
}

Test(video_encoder, get_token_basic)
{
    OcVideoEncoderConfig cfg = {2, 2, 2};
    OcVideoEncoder enc;
    oc_video_encoder_init(&enc, &cfg);
    float emb[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    oc_video_encoder_encode(&enc, emb, 2, 2);
    float v = -1.f;
    cr_assert_eq(oc_video_encoder_get_token(&enc, 0, 0, &v), OC_OK, "");
    cr_assert_float_eq(v, 0.1f, 1e-6, "");
    cr_assert_eq(oc_video_encoder_get_token(&enc, 1, 1, &v), OC_OK, "");
    cr_assert_float_eq(v, 0.4f, 1e-6, "");
    oc_video_encoder_free(&enc);
}

Test(video_encoder, get_token_out_of_range)
{
    OcVideoEncoderConfig cfg = {2, 2, 2};
    OcVideoEncoder enc;
    oc_video_encoder_init(&enc, &cfg);
    float emb[2] = {0.f, 0.f};
    oc_video_encoder_encode(&enc, emb, 1, 2);
    float v = 0.f;
    cr_assert_eq(oc_video_encoder_get_token(&enc, 5, 0, &v),
                 OC_ERR_INVALID_ARG, "frame OOR");
    cr_assert_eq(oc_video_encoder_get_token(&enc, 0, 5, &v),
                 OC_ERR_INVALID_ARG, "dim OOR");
    oc_video_encoder_free(&enc);
}

Test(video_encoder, get_token_null_args)
{
    OcVideoEncoderConfig cfg = {2, 2, 2};
    OcVideoEncoder enc;
    oc_video_encoder_init(&enc, &cfg);
    float emb[2] = {0.f, 0.f};
    oc_video_encoder_encode(&enc, emb, 1, 2);
    float v = 0.f;
    cr_assert_eq(oc_video_encoder_get_token(NULL, 0, 0, &v),
                 OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_video_encoder_get_token(&enc, 0, 0, NULL),
                 OC_ERR_INVALID_ARG, "");
    oc_video_encoder_free(&enc);
}
