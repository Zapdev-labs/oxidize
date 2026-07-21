/* test_vision.c — vision encoder stub tests. */
#include <criterion/criterion.h>
#include "oxidize/vision.h"
#include <stdlib.h>
#include <string.h>

Test(vision, init_and_encode)
{
    OcVisionEncoder enc;
    OcVisionConfig cfg = { .image_size=224, .patch_size=14, .hidden_size=768,
                           .n_layers=12, .n_heads=12, .n_patches=0 };
    cr_assert_eq(oc_vision_init(&enc, &cfg), OC_OK);
    cr_assert(enc.initialized);
    cr_assert_eq(enc.config.n_patches, 256); /* (224/14)^2 = 256 */

    float *emb = malloc(256 * 768 * sizeof(*emb));
    cr_assert_not_null(emb);
    size_t out_len = 0;
    OcImage img = { .data=(const uint8_t*)"dummy", .width=224, .height=224,
                    .channels=3, .format=OC_IMAGE_RGB };
    cr_assert_eq(oc_vision_encode(&enc, &img, emb, &out_len), OC_OK);
    cr_assert_eq(out_len, 256 * 768);

    free(emb);
    oc_vision_free(&enc);
    cr_assert(!enc.initialized);
}

Test(vision, resize)
{
    uint8_t src_data[] = {0, 64, 128, 255}; /* 2x2 gray */
    OcImage src = { .data=src_data, .width=2, .height=2, .channels=1, .format=OC_IMAGE_GRAY };
    uint8_t out[4]; /* 2x2 output */
    cr_assert_eq(oc_vision_resize(&src, 2, 2, out), OC_OK);
    /* At same size, should be close to original. */
    cr_assert_eq(out[0], 0);
    cr_assert_eq(out[3], 255);
}

Test(vision, normalize)
{
    uint8_t pixels[] = {0, 128, 255};
    OcImage img = { .data=pixels, .width=1, .height=1, .channels=3, .format=OC_IMAGE_RGB };
    float out[3];
    cr_assert_eq(oc_vision_normalize(&img, out), OC_OK);
    cr_assert_float_eq(out[0], -1.0f, 1e-5f, "0 -> -1.0");
    cr_assert_float_eq(out[1], 128.0f/127.5f - 1.0f, 1e-4f, "128 -> ~0.004");
    cr_assert_float_eq(out[2], 255.0f/127.5f - 1.0f, 1e-4f, "255 -> ~1.0");
}

Test(vision, init_rejects_bad_config)
{
    OcVisionEncoder enc;
    OcVisionConfig cfg = {0};
    cr_assert_eq(oc_vision_init(&enc, &cfg), OC_ERR_INVALID_ARG);
}

Test(vision, multimodal_create)
{
    uint32_t tokens[] = {1, 2, 3};
    float emb[] = {0.1f, 0.2f};
    OcMultimodalPrompt p;
    cr_assert_eq(oc_multimodal_create(tokens, 3, emb, 2, 99, &p), OC_OK);
    cr_assert_eq(p.n_text_tokens, 3);
    cr_assert_eq(p.image_token_id, 99);
    cr_assert_eq(p.n_image_embeddings, 2);
    oc_multimodal_free(&p);
}
