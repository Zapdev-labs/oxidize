/*
 * vision_preprocess.c — Image preprocessing implementation.
 */
#include "oxidize/vision_preprocess.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

OcError oc_image_init(OcImage *img, uint32_t width, uint32_t height)
{
    if (!img || width == 0 || height == 0) return OC_ERR_INVALID_ARG;
    img->pixels = malloc((size_t)width * height * sizeof(OcRgbPixel));
    if (!img->pixels) return OC_ERR_OOM;
    memset(img->pixels, 0, (size_t)width * height * sizeof(OcRgbPixel));
    img->width = width;
    img->height = height;
    return OC_OK;
}

OcError oc_image_from_rgb(OcImage *img, uint32_t width, uint32_t height,
                          const uint8_t *rgb_data)
{
    if (!img || !rgb_data || width == 0 || height == 0) return OC_ERR_INVALID_ARG;
    size_t n = (size_t)width * height;
    img->pixels = malloc(n * sizeof(OcRgbPixel));
    if (!img->pixels) return OC_ERR_OOM;
    for (size_t i = 0; i < n; i++) {
        img->pixels[i].r = rgb_data[i * 3];
        img->pixels[i].g = rgb_data[i * 3 + 1];
        img->pixels[i].b = rgb_data[i * 3 + 2];
    }
    img->width = width;
    img->height = height;
    return OC_OK;
}

OcError oc_image_free(OcImage *img)
{
    if (!img) return OC_OK;
    free(img->pixels);
    img->pixels = NULL;
    img->width = 0;
    img->height = 0;
    return OC_OK;
}

OcError oc_preprocess_config_init(OcPreprocessConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->target_size = 224;
    /* CLIP default normalization. */
    cfg->mean[0] = 0.48145466f;
    cfg->mean[1] = 0.4578275f;
    cfg->mean[2] = 0.40821073f;
    cfg->std[0] = 0.26862954f;
    cfg->std[1] = 0.26130258f;
    cfg->std[2] = 0.27577711f;
    cfg->rescale = true;
    cfg->to_rgb = true;
    cfg->center_crop = true;
    return OC_OK;
}

OcError oc_preprocess_resize(const OcImage *src, OcImage *dst,
                            uint32_t target_w, uint32_t target_h)
{
    if (!src || !dst || !src->pixels) return OC_ERR_INVALID_ARG;
    if (target_w == 0 || target_h == 0) return OC_ERR_INVALID_ARG;

    OcError e = oc_image_init(dst, target_w, target_h);
    if (e != OC_OK) return e;

    /* Bilinear interpolation. */
    for (uint32_t y = 0; y < target_h; y++) {
        float src_y = (float)y * src->height / target_h;
        uint32_t y0 = (uint32_t)src_y;
        uint32_t y1 = (y0 + 1 < src->height) ? y0 + 1 : y0;
        float fy = src_y - y0;
        for (uint32_t x = 0; x < target_w; x++) {
            float src_x = (float)x * src->width / target_w;
            uint32_t x0 = (uint32_t)src_x;
            uint32_t x1 = (x0 + 1 < src->width) ? x0 + 1 : x0;
            float fx = src_x - x0;

            const OcRgbPixel *p00 = &src->pixels[y0 * src->width + x0];
            const OcRgbPixel *p01 = &src->pixels[y0 * src->width + x1];
            const OcRgbPixel *p10 = &src->pixels[y1 * src->width + x0];
            const OcRgbPixel *p11 = &src->pixels[y1 * src->width + x1];

            OcRgbPixel *d = &dst->pixels[y * target_w + x];
            d->r = (uint8_t)((1-fx)*(1-fy)*p00->r + fx*(1-fy)*p01->r +
                           (1-fx)*fy*p10->r + fx*fy*p11->r);
            d->g = (uint8_t)((1-fx)*(1-fy)*p00->g + fx*(1-fy)*p01->g +
                           (1-fx)*fy*p10->g + fx*fy*p11->g);
            d->b = (uint8_t)((1-fx)*(1-fy)*p00->b + fx*(1-fy)*p01->b +
                           (1-fx)*fy*p10->b + fx*fy*p11->b);
        }
    }
    return OC_OK;
}

OcError oc_preprocess_center_crop(const OcImage *src, OcImage *dst,
                                 uint32_t crop_w, uint32_t crop_h)
{
    if (!src || !dst || !src->pixels) return OC_ERR_INVALID_ARG;
    if (crop_w == 0 || crop_h == 0) return OC_ERR_INVALID_ARG;
    if (crop_w > src->width || crop_h > src->height) return OC_ERR_INVALID_ARG;

    OcError e = oc_image_init(dst, crop_w, crop_h);
    if (e != OC_OK) return e;

    uint32_t off_x = (src->width - crop_w) / 2;
    uint32_t off_y = (src->height - crop_h) / 2;

    for (uint32_t y = 0; y < crop_h; y++)
        for (uint32_t x = 0; x < crop_w; x++)
            dst->pixels[y * crop_w + x] = src->pixels[(y + off_y) * src->width + (x + off_x)];

    return OC_OK;
}

OcError oc_preprocess_normalize(const OcImage *src, float *out,
                               const OcPreprocessConfig *cfg)
{
    if (!src || !out || !cfg || !src->pixels) return OC_ERR_INVALID_ARG;

    size_t n = (size_t)src->width * src->height;
    for (size_t i = 0; i < n; i++) {
        float r = (float)src->pixels[i].r;
        float g = (float)src->pixels[i].g;
        float b = (float)src->pixels[i].b;
        if (cfg->rescale) {
            r /= 255.0f;
            g /= 255.0f;
            b /= 255.0f;
        }
        if (cfg->to_rgb) {
            out[i * 3]     = (r - cfg->mean[0]) / cfg->std[0];
            out[i * 3 + 1] = (g - cfg->mean[1]) / cfg->std[1];
            out[i * 3 + 2] = (b - cfg->mean[2]) / cfg->std[2];
        } else {
            out[i * 3]     = (b - cfg->mean[2]) / cfg->std[2];
            out[i * 3 + 1] = (g - cfg->mean[1]) / cfg->std[1];
            out[i * 3 + 2] = (r - cfg->mean[0]) / cfg->std[0];
        }
    }
    return OC_OK;
}

OcError oc_preprocess_full(const OcImage *src, float *out,
                         const OcPreprocessConfig *cfg)
{
    if (!src || !out || !cfg) return OC_ERR_INVALID_ARG;

    OcImage resized = {0};
    OcImage cropped = {0};
    const OcImage *work = src;

    /* Resize to target size. */
    OcError e = oc_preprocess_resize(src, &resized, cfg->target_size, cfg->target_size);
    if (e != OC_OK) return e;
    work = &resized;

    /* Center crop if enabled. */
    if (cfg->center_crop) {
        e = oc_preprocess_center_crop(work, &cropped, cfg->target_size, cfg->target_size);
        if (e != OC_OK) { oc_image_free(&resized); return e; }
        work = &cropped;
    }

    /* Normalize. */
    e = oc_preprocess_normalize(work, out, cfg);

    oc_image_free(&resized);
    oc_image_free(&cropped);
    return e;
}
