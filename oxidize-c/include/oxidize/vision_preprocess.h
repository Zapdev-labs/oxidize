/*
 * vision_preprocess.h — Image preprocessing for vision encoder.
 *
 * Resize, normalize, and augment images for vision model input.
 * Port from oxidize-core/src/vision/preprocess.rs.
 */
#ifndef OXIDIZE_VISION_PREPROCESS_H
#define OXIDIZE_VISION_PREPROCESS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t r, g, b;
} OcRgbPixel;

typedef struct {
    OcRgbPixel *pixels;
    uint32_t width;
    uint32_t height;
} OcImage;

typedef struct {
    uint32_t target_size;
    float mean[3];
    float std[3];
    bool rescale;
    bool to_rgb;
    bool center_crop;
} OcPreprocessConfig;

OcError oc_image_init(OcImage *img, uint32_t width, uint32_t height);
OcError oc_image_from_rgb(OcImage *img, uint32_t width, uint32_t height,
                          const uint8_t *rgb_data);
OcError oc_image_free(OcImage *img);
OcError oc_preprocess_config_init(OcPreprocessConfig *cfg);
OcError oc_preprocess_resize(const OcImage *src, OcImage *dst,
                            uint32_t target_w, uint32_t target_h);
OcError oc_preprocess_center_crop(const OcImage *src, OcImage *dst,
                                 uint32_t crop_w, uint32_t crop_h);
OcError oc_preprocess_normalize(const OcImage *src, float *out,
                               const OcPreprocessConfig *cfg);
OcError oc_preprocess_full(const OcImage *src, float *out,
                         const OcPreprocessConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_VISION_PREPROCESS_H */
