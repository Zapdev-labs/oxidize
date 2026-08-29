#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>
#include <string.h>
#include "oxidize/vision_encoder.h"


Test(vision_encoder, config_init_defaults)
{
    OcVisionEncoderConfig cfg;
    cr_assert_eq(oc_vision_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.image_size, OC_VISION_DEFAULT_IMAGE_SIZE);
    cr_assert_eq(cfg.patch_size, OC_VISION_DEFAULT_PATCH_SIZE);
    cr_assert_eq(cfg.n_channels, OC_VISION_DEFAULT_N_CHANNELS);
    cr_assert_eq(cfg.n_layers,   OC_VISION_DEFAULT_N_LAYERS);
    cr_assert_eq(cfg.hidden_dim, OC_VISION_DEFAULT_HIDDEN_DIM);
    cr_assert_eq(cfg.n_heads,    OC_VISION_DEFAULT_N_HEADS);
}

Test(vision_encoder, config_init_null)
{
    cr_assert_eq(oc_vision_config_init(NULL), OC_ERR_INVALID_ARG);
}


Test(vision_encoder, init_with_default_config)
{
    OcVisionEncoder *e = NULL;
    cr_assert_eq(oc_vision_encoder_init(&e, NULL), OC_OK);
    cr_assert_not_null(e);
    cr_assert(e->initialized);
    cr_assert_eq(e->config.image_size, OC_VISION_DEFAULT_IMAGE_SIZE);
    cr_assert_eq(e->config.hidden_dim, OC_VISION_DEFAULT_HIDDEN_DIM);
    cr_assert_null(e->weight_data);
    cr_assert_eq(e->weight_size, 0u);
    oc_vision_encoder_free(e);
}

Test(vision_encoder, init_with_custom_config)
{
    OcVisionEncoderConfig cfg;
    oc_vision_config_init(&cfg);
    cfg.image_size = 224;
    cfg.patch_size = 14;
    cfg.hidden_dim = 512;
    OcVisionEncoder *e = NULL;
    cr_assert_eq(oc_vision_encoder_init(&e, &cfg), OC_OK);
    cr_assert_eq(e->config.image_size, 224u);
    cr_assert_eq(e->config.patch_size, 14u);
    cr_assert_eq(e->config.hidden_dim, 512u);
    oc_vision_encoder_free(e);
}

Test(vision_encoder, init_rejects_zero_image_size)
{
    OcVisionEncoderConfig cfg;
    oc_vision_config_init(&cfg);
    cfg.image_size = 0;
    OcVisionEncoder *e = NULL;
    cr_assert_eq(oc_vision_encoder_init(&e, &cfg), OC_ERR_INVALID_ARG);
    cr_assert_null(e);
}

Test(vision_encoder, init_rejects_non_divisible_patch)
{
    OcVisionEncoderConfig cfg;
    oc_vision_config_init(&cfg);
    cfg.image_size = 224;
    cfg.patch_size = 15; /* 224 % 15 != 0 */
    OcVisionEncoder *e = NULL;
    cr_assert_eq(oc_vision_encoder_init(&e, &cfg), OC_ERR_INVALID_ARG);
    cr_assert_null(e);
}

Test(vision_encoder, init_rejects_zero_heads)
{
    OcVisionEncoderConfig cfg;
    oc_vision_config_init(&cfg);
    cfg.n_heads = 0;
    OcVisionEncoder *e = NULL;
    cr_assert_eq(oc_vision_encoder_init(&e, &cfg), OC_ERR_INVALID_ARG);
    cr_assert_null(e);
}

Test(vision_encoder, free_null_is_safe)
{
    oc_vision_encoder_free(NULL);
    cr_assert(true);
}


Test(vision_encoder, load_weights_copies_data)
{
    OcVisionEncoder *e = NULL;
    cr_assert_eq(oc_vision_encoder_init(&e, NULL), OC_OK);
    float weights[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    cr_assert_eq(oc_vision_encoder_load_weights(e, weights, sizeof(weights)), OC_OK);
    cr_assert_not_null(e->weight_data);
    cr_assert_eq(e->weight_size, sizeof(weights));
    cr_assert_neq(e->weight_data, weights); /* must be a copy */
    cr_assert(memcmp(e->weight_data, weights, sizeof(weights)) == 0);
    oc_vision_encoder_free(e);
}

Test(vision_encoder, load_weights_replaces_previous)
{
    OcVisionEncoder *e = NULL;
    cr_assert_eq(oc_vision_encoder_init(&e, NULL), OC_OK);
    float w1[2] = {1.0f, 2.0f};
    float w2[3] = {3.0f, 4.0f, 5.0f};
    cr_assert_eq(oc_vision_encoder_load_weights(e, w1, sizeof(w1)), OC_OK);
    cr_assert_eq(e->weight_size, sizeof(w1));
    cr_assert_eq(oc_vision_encoder_load_weights(e, w2, sizeof(w2)), OC_OK);
    cr_assert_eq(e->weight_size, sizeof(w2));
    cr_assert(memcmp(e->weight_data, w2, sizeof(w2)) == 0);
    oc_vision_encoder_free(e);
}

Test(vision_encoder, load_weights_zero_size_clears)
{
    OcVisionEncoder *e = NULL;
    cr_assert_eq(oc_vision_encoder_init(&e, NULL), OC_OK);
    float w[2] = {1.0f, 2.0f};
    cr_assert_eq(oc_vision_encoder_load_weights(e, w, sizeof(w)), OC_OK);
    cr_assert_eq(oc_vision_encoder_load_weights(e, NULL, 0), OC_OK);
    cr_assert_null(e->weight_data);
    cr_assert_eq(e->weight_size, 0u);
    oc_vision_encoder_free(e);
}


Test(vision_encoder, encode_returns_features)
{
    OcVisionEncoder *e = NULL;
    cr_assert_eq(oc_vision_encoder_init(&e, NULL), OC_OK);
    OcImagePatch *img = NULL;
    cr_assert_eq(oc_image_patch_init(224, 224, 3, &img), OC_OK);
    float *feats = NULL;
    size_t n = 0;
    cr_assert_eq(oc_vision_encoder_encode(e, img, &feats, &n), OC_OK);
    cr_assert_not_null(feats);
    uint32_t n_patches = (224 / 16) * (224 / 16);
    cr_assert_eq(n, (size_t)n_patches * OC_VISION_DEFAULT_HIDDEN_DIM);
    free(feats);
    oc_image_patch_free(img);
    oc_vision_encoder_free(e);
}

Test(vision_encoder, encode_rejects_uninitialized_encoder)
{
    OcVisionEncoder e = {0}; /* initialized = false */
    OcImagePatch *img = NULL;
    cr_assert_eq(oc_image_patch_init(16, 16, 3, &img), OC_OK);
    float *feats = NULL;
    size_t n = 0;
    cr_assert_eq(oc_vision_encoder_encode(&e, img, &feats, &n), OC_ERR_INVALID_ARG);
    oc_image_patch_free(img);
}

Test(vision_encoder, encode_rejects_null_image_pixels)
{
    OcVisionEncoder *e = NULL;
    cr_assert_eq(oc_vision_encoder_init(&e, NULL), OC_OK);
    OcImagePatch img = {0};
    img.width = 16; img.height = 16; img.channels = 3; /* pixels NULL */
    float *feats = NULL;
    size_t n = 0;
    cr_assert_eq(oc_vision_encoder_encode(e, &img, &feats, &n), OC_ERR_INVALID_ARG);
    oc_vision_encoder_free(e);
}

Test(vision_encoder, patch_embed_returns_patches)
{
    OcVisionEncoder *e = NULL;
    cr_assert_eq(oc_vision_encoder_init(&e, NULL), OC_OK);
    OcImagePatch *img = NULL;
    cr_assert_eq(oc_image_patch_init(224, 224, 3, &img), OC_OK);
    float *patches = NULL;
    size_t n = 0;
    cr_assert_eq(oc_vision_encoder_patch_embed(e, img, &patches, &n), OC_OK);
    cr_assert_not_null(patches);
    uint32_t n_patches = (224 / 16) * (224 / 16);
    size_t patch_dim = 16 * 16 * 3;
    cr_assert_eq(n, (size_t)n_patches * patch_dim);
    free(patches);
    oc_image_patch_free(img);
    oc_vision_encoder_free(e);
}

Test(vision_encoder, patch_embed_extracts_real_pixels)
{
    OcVisionEncoder *e = NULL;
    cr_assert_eq(oc_vision_encoder_init(&e, NULL), OC_OK);
    /* Create a 32x32 image with 3 channels, fill with known values. */
    OcImagePatch *img = NULL;
    cr_assert_eq(oc_image_patch_init(32, 32, 3, &img), OC_OK);
    /* Fill with pattern: pixel(x,y,c) = (x + y + c) / 10.0. */
    for (uint32_t y = 0; y < 32; y++) {
        for (uint32_t x = 0; x < 32; x++) {
            for (uint32_t c = 0; c < 3; c++) {
                img->pixels[(y * 32 + x) * 3 + c] = (float)(x + y + c) / 10.0f;
            }
        }
    }
    float *patches = NULL;
    size_t n = 0;
    cr_assert_eq(oc_vision_encoder_patch_embed(e, img, &patches, &n), OC_OK);
    /* Default: image_size=224, patch_size=16 → n_patches=196.
     * But image is only 32x32, so many patches will be zero. */
    uint32_t n_patches = (224 / 16) * (224 / 16);
    size_t patch_dim = 16 * 16 * 3;
    cr_assert_eq(n, (size_t)n_patches * patch_dim);
    /* Verify first patch (px=0, py=0) extracts pixel data correctly.
     * patch[0] = pixel(0,0,0) = 0.0. */
    cr_assert_float_eq(patches[0], 0.0f, 0.001f);
    /* patch[1] = pixel(1,0,0) = 1/10 = 0.1. */
    cr_assert_float_eq(patches[1], 0.1f, 0.001f);
    /* patch[256] = pixel(0,0,1) = 1/10 = 0.1 (channel 1, dy=0, dx=0). */
    cr_assert_float_eq(patches[256], 0.1f, 0.001f);
    /* Verify non-zero data exists (not all zeros like the old stub). */
    bool has_nonzero = false;
    for (size_t i = 0; i < n; i++) {
        if (patches[i] != 0.0f) { has_nonzero = true; break; }
    }
    cr_assert(has_nonzero, "patch_embed should extract real pixel data");
    free(patches);
    oc_image_patch_free(img);
    oc_vision_encoder_free(e);
}


Test(vision_encoder, image_patch_init_allocates)
{
    OcImagePatch *p = NULL;
    cr_assert_eq(oc_image_patch_init(16, 16, 3, &p), OC_OK);
    cr_assert_not_null(p);
    cr_assert_not_null(p->pixels);
    cr_assert_eq(p->width, 16u);
    cr_assert_eq(p->height, 16u);
    cr_assert_eq(p->channels, 3u);
    /* Pixels are zeroed. */
    cr_assert_eq(p->pixels[0], 0.0f);
    cr_assert_eq(p->pixels[(16 * 16 * 3) - 1], 0.0f);
    oc_image_patch_free(p);
}

Test(vision_encoder, image_patch_init_rejects_zero_dims)
{
    OcImagePatch *p = NULL;
    cr_assert_eq(oc_image_patch_init(0, 16, 3, &p), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_image_patch_init(16, 0, 3, &p), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_image_patch_init(16, 16, 0, &p), OC_ERR_INVALID_ARG);
    cr_assert_null(p);
}

Test(vision_encoder, image_patch_free_null_is_safe)
{
    oc_image_patch_free(NULL);
    cr_assert(true);
}

Test(vision_encoder, encode_features_are_deterministic)
{
    OcVisionEncoder *e = NULL;
    cr_assert_eq(oc_vision_encoder_init(&e, NULL), OC_OK);
    OcImagePatch *img = NULL;
    cr_assert_eq(oc_image_patch_init(224, 224, 3, &img), OC_OK);
    float *f1 = NULL; size_t n1 = 0;
    float *f2 = NULL; size_t n2 = 0;
    cr_assert_eq(oc_vision_encoder_encode(e, img, &f1, &n1), OC_OK);
    cr_assert_eq(oc_vision_encoder_encode(e, img, &f2, &n2), OC_OK);
    cr_assert_eq(n1, n2);
    cr_assert(memcmp(f1, f2, n1 * sizeof(float)) == 0);
    free(f1);
    free(f2);
    oc_image_patch_free(img);
    oc_vision_encoder_free(e);
}
