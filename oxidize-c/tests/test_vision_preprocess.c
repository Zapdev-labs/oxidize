/* test_vision_preprocess.c — Vision preprocessing tests. */
#include "framework.h"
#include "oxidize/vision_preprocess.h"
#include <string.h>
#include <math.h>

Test(vpre, image_init)
{
    OcImage img;
    cr_assert_eq(oc_image_init(&img, 10, 20), OC_OK);
    cr_assert_eq(img.width, 10);
    cr_assert_eq(img.height, 20);
    cr_assert_not_null(img.pixels);
    oc_image_free(&img);
}

Test(vpre, image_init_null)
{
    cr_assert_neq(oc_image_init(NULL, 10, 10), OC_OK);
}

Test(vpre, image_from_rgb)
{
    OcImage img;
    uint8_t rgb[] = {255, 0, 0, 0, 255, 0, 0, 0, 255};
    cr_assert_eq(oc_image_from_rgb(&img, 3, 1, rgb), OC_OK);
    cr_assert_eq(img.width, 3);
    cr_assert_eq(img.height, 1);
    cr_assert_eq(img.pixels[0].r, 255);
    cr_assert_eq(img.pixels[1].g, 255);
    cr_assert_eq(img.pixels[2].b, 255);
    oc_image_free(&img);
}

Test(vpre, image_free)
{
    OcImage img;
    oc_image_init(&img, 5, 5);
    cr_assert_eq(oc_image_free(&img), OC_OK);
    cr_assert_null(img.pixels);
    cr_assert_eq(img.width, 0);
}

Test(vpre, image_free_null)
{
    cr_assert_eq(oc_image_free(NULL), OC_OK);
}

Test(vpre, config_init)
{
    OcPreprocessConfig cfg;
    cr_assert_eq(oc_preprocess_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.target_size, 224);
    cr_assert(cfg.rescale);
    cr_assert(cfg.to_rgb);
    cr_assert(cfg.center_crop);
    cr_assert_float_eq(cfg.mean[0], 0.48145466f, 0.0001f);
}

Test(vpre, config_init_null)
{
    cr_assert_neq(oc_preprocess_config_init(NULL), OC_OK);
}

Test(vpre, resize)
{
    OcImage src;
    oc_image_init(&src, 4, 4);
    for (int i = 0; i < 16; i++) {
        src.pixels[i].r = 100;
        src.pixels[i].g = 150;
        src.pixels[i].b = 200;
    }
    OcImage dst;
    cr_assert_eq(oc_preprocess_resize(&src, &dst, 2, 2), OC_OK);
    cr_assert_eq(dst.width, 2);
    cr_assert_eq(dst.height, 2);
    cr_assert_eq(dst.pixels[0].r, 100);
    oc_image_free(&src);
    oc_image_free(&dst);
}

Test(vpre, resize_null)
{
    cr_assert_neq(oc_preprocess_resize(NULL, NULL, 0, 0), OC_OK);
}

Test(vpre, center_crop)
{
    OcImage src;
    oc_image_init(&src, 10, 10);
    for (int i = 0; i < 100; i++) {
        src.pixels[i].r = (uint8_t)(i % 256);
    }
    OcImage dst;
    cr_assert_eq(oc_preprocess_center_crop(&src, &dst, 4, 4), OC_OK);
    cr_assert_eq(dst.width, 4);
    cr_assert_eq(dst.height, 4);
    /* Center of 10x10 starts at (3,3). */
    cr_assert_eq(dst.pixels[0].r, src.pixels[3 * 10 + 3].r);
    oc_image_free(&src);
    oc_image_free(&dst);
}

Test(vpre, center_crop_too_large)
{
    OcImage src;
    oc_image_init(&src, 5, 5);
    OcImage dst;
    cr_assert_neq(oc_preprocess_center_crop(&src, &dst, 10, 10), OC_OK);
    oc_image_free(&src);
}

Test(vpre, normalize)
{
    OcImage img;
    oc_image_init(&img, 2, 2);
    for (int i = 0; i < 4; i++) {
        img.pixels[i].r = 255;
        img.pixels[i].g = 128;
        img.pixels[i].b = 0;
    }
    OcPreprocessConfig cfg;
    oc_preprocess_config_init(&cfg);
    float out[12]; /* 4 pixels * 3 channels */
    cr_assert_eq(oc_preprocess_normalize(&img, out, &cfg), OC_OK);
    /* r=255 → 1.0 → (1.0 - mean[0]) / std[0]. */
    cr_assert_float_eq(out[0], (1.0f - cfg.mean[0]) / cfg.std[0], 0.01f);
    cr_assert_float_eq(out[1], (128.0f/255.0f - cfg.mean[1]) / cfg.std[1], 0.01f);
    cr_assert_float_eq(out[2], (0.0f - cfg.mean[2]) / cfg.std[2], 0.01f);
    oc_image_free(&img);
}

Test(vpre, normalize_null)
{
    cr_assert_neq(oc_preprocess_normalize(NULL, NULL, NULL), OC_OK);
}

Test(vpre, full_pipeline)
{
    OcImage img;
    oc_image_init(&img, 100, 100);
    for (int i = 0; i < 10000; i++) {
        img.pixels[i].r = 128;
        img.pixels[i].g = 64;
        img.pixels[i].b = 32;
    }
    OcPreprocessConfig cfg;
    oc_preprocess_config_init(&cfg);
    cfg.target_size = 32;
    float *out = malloc(32 * 32 * 3 * sizeof(float));
    cr_assert_eq(oc_preprocess_full(&img, out, &cfg), OC_OK);
    /* Check first pixel is normalized. */
    cr_assert_lt(out[0], 1.0f);
    cr_assert_gt(out[0], -1.0f);
    free(out);
    oc_image_free(&img);
}

Test(vpre, full_pipeline_null)
{
    cr_assert_neq(oc_preprocess_full(NULL, NULL, NULL), OC_OK);
}

Test(vpre, resize_uneven)
{
    OcImage src;
    oc_image_init(&src, 10, 5);
    for (int i = 0; i < 50; i++) {
        src.pixels[i].r = 50;
        src.pixels[i].g = 100;
        src.pixels[i].b = 150;
    }
    OcImage dst;
    cr_assert_eq(oc_preprocess_resize(&src, &dst, 5, 10), OC_OK);
    cr_assert_eq(dst.width, 5);
    cr_assert_eq(dst.height, 10);
    oc_image_free(&src);
    oc_image_free(&dst);
}

Test(vpre, normalize_no_rescale)
{
    OcImage img;
    oc_image_init(&img, 1, 1);
    img.pixels[0].r = 255;
    img.pixels[0].g = 255;
    img.pixels[0].b = 255;
    OcPreprocessConfig cfg;
    oc_preprocess_config_init(&cfg);
    cfg.rescale = false;
    float out[3];
    cr_assert_eq(oc_preprocess_normalize(&img, out, &cfg), OC_OK);
    cr_assert_float_eq(out[0], (255.0f - cfg.mean[0]) / cfg.std[0], 0.01f);
    oc_image_free(&img);
}
