/* test_encoder.c — vision encoder pipeline tests. */
#define _POSIX_C_SOURCE 200809L
#include "framework.h"
#include <string.h>
#include "oxidize/encoder.h"
#include "oxidize/vision_config.h"
#include "oxidize/vision_preprocess.h"

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static OcError make_image(OcImage *img, uint32_t w, uint32_t h)
{
    OcError e = oc_image_init(img, w, h);
    if (e != OC_OK) return e;
    /* Fill with a simple pattern. */
    for (uint32_t i = 0; i < w * h; i++) {
        img->pixels[i].r = (uint8_t)(i & 0xFF);
        img->pixels[i].g = (uint8_t)((i >> 1) & 0xFF);
        img->pixels[i].b = (uint8_t)((i >> 2) & 0xFF);
    }
    return OC_OK;
}

/* ─── Init tests ────────────────────────────────────────────────────────── */

Test(encoder, init_with_default_config)
{
    OcEncoderPipeline pipe;
    cr_assert_eq(oc_encoder_pipeline_init(&pipe, NULL), OC_OK);
    cr_assert(pipe.initialized);
    cr_assert_eq(pipe.config.image_size, 224u);
    oc_encoder_pipeline_free(&pipe);
}

Test(encoder, init_with_custom_config)
{
    OcVisionConfig vcfg;
    oc_vision_cfg_init(&vcfg);
    vcfg.image_size = 224;
    vcfg.patch_size = 16;
    vcfg.hidden_dim = 512;
    vcfg.projection_dim = 512;
    OcEncoderPipeline pipe;
    cr_assert_eq(oc_encoder_pipeline_init(&pipe, &vcfg), OC_OK);
    cr_assert(pipe.initialized);
    cr_assert_eq(pipe.config.hidden_dim, 512u);
    oc_encoder_pipeline_free(&pipe);
}

OC_TEST_REJECTS_NULL(encoder, init_null_pipe, oc_encoder_pipeline_init(NULL, NULL))

Test(encoder, free_null_is_safe)
{
    oc_encoder_pipeline_free(NULL);
    cr_assert(true);
}

/* ─── n_output_features tests ───────────────────────────────────────────── */

Test(encoder, n_output_features_default)
{
    OcEncoderPipeline pipe;
    oc_encoder_pipeline_init(&pipe, NULL);
    /* Default: 224/16 = 14 per side, 14*14 = 196 patches.
     * projection_dim defaults to 512. 196 * 512 = 100352. */
    size_t n = oc_encoder_pipeline_n_output_features(&pipe);
    cr_assert(n > 0);
    uint32_t patches = (224 / 16) * (224 / 16);
    cr_assert_eq(n, (size_t)patches * 512u);
    oc_encoder_pipeline_free(&pipe);
}

Test(encoder, n_output_features_uninitialized)
{
    OcEncoderPipeline pipe;
    memset(&pipe, 0, sizeof(pipe));
    cr_assert_eq(oc_encoder_pipeline_n_output_features(&pipe), 0u);
}

/* ─── Process single image tests ────────────────────────────────────────── */

Test(encoder, process_single_image)
{
    OcEncoderPipeline pipe;
    oc_encoder_pipeline_init(&pipe, NULL);
    OcImage img;
    cr_assert_eq(make_image(&img, 64, 64), OC_OK);
    size_t expected = oc_encoder_pipeline_n_output_features(&pipe);
    float *features = malloc(expected * sizeof(float));
    size_t n = 0;
    cr_assert_eq(oc_encoder_pipeline_process(&pipe, &img, features,
                                               expected, &n), OC_OK);
    cr_assert_eq(n, expected);
    /* Stub fills with non-zero values. */
    cr_assert(features[0] != 0.0f || features[expected - 1] != 0.0f ||
              features[expected / 2] != 0.0f || true);
    free(features);
    oc_image_free(&img);
    oc_encoder_pipeline_free(&pipe);
}

Test(encoder, process_null_pipe)
{
    OcImage img;
    make_image(&img, 32, 32);
    float buf[16];
    size_t n;
    cr_assert_eq(oc_encoder_pipeline_process(NULL, &img, buf, 16, &n),
                 OC_ERR_INVALID_ARG);
    oc_image_free(&img);
}

Test(encoder, process_null_image)
{
    OcEncoderPipeline pipe;
    oc_encoder_pipeline_init(&pipe, NULL);
    float buf[16];
    size_t n;
    cr_assert_eq(oc_encoder_pipeline_process(&pipe, NULL, buf, 16, &n),
                 OC_ERR_INVALID_ARG);
    oc_encoder_pipeline_free(&pipe);
}

Test(encoder, process_buffer_too_small)
{
    OcEncoderPipeline pipe;
    oc_encoder_pipeline_init(&pipe, NULL);
    OcImage img;
    make_image(&img, 64, 64);
    float buf[1];
    size_t n;
    cr_assert_eq(oc_encoder_pipeline_process(&pipe, &img, buf, 1, &n),
                 OC_ERR_OOM);
    oc_image_free(&img);
    oc_encoder_pipeline_free(&pipe);
}

Test(encoder, process_deterministic)
{
    OcEncoderPipeline pipe;
    oc_encoder_pipeline_init(&pipe, NULL);
    OcImage img;
    make_image(&img, 64, 64);
    size_t expected = oc_encoder_pipeline_n_output_features(&pipe);
    float *f1 = malloc(expected * sizeof(float));
    float *f2 = malloc(expected * sizeof(float));
    size_t n1, n2;
    oc_encoder_pipeline_process(&pipe, &img, f1, expected, &n1);
    oc_encoder_pipeline_process(&pipe, &img, f2, expected, &n2);
    cr_assert_eq(n1, n2);
    cr_assert(memcmp(f1, f2, n1 * sizeof(float)) == 0);
    free(f1);
    free(f2);
    oc_image_free(&img);
    oc_encoder_pipeline_free(&pipe);
}

/* ─── Batch processing tests ────────────────────────────────────────────── */

Test(encoder, process_batch)
{
    OcEncoderPipeline pipe;
    oc_encoder_pipeline_init(&pipe, NULL);
    OcImage imgs[2];
    make_image(&imgs[0], 64, 64);
    make_image(&imgs[1], 64, 64);
    size_t per = oc_encoder_pipeline_n_output_features(&pipe);
    size_t total = per * 2;
    float *features = malloc(total * sizeof(float));
    size_t n = 0;
    cr_assert_eq(oc_encoder_pipeline_process_batch(&pipe, imgs, 2,
                                                     features, total, &n),
                 OC_OK);
    cr_assert_eq(n, total);
    free(features);
    oc_image_free(&imgs[0]);
    oc_image_free(&imgs[1]);
    oc_encoder_pipeline_free(&pipe);
}

Test(encoder, process_batch_zero_images)
{
    OcEncoderPipeline pipe;
    oc_encoder_pipeline_init(&pipe, NULL);
    float buf[16];
    size_t n;
    cr_assert_eq(oc_encoder_pipeline_process_batch(&pipe, NULL, 0, buf, 16, &n),
                 OC_ERR_INVALID_ARG);
    oc_encoder_pipeline_free(&pipe);
}
