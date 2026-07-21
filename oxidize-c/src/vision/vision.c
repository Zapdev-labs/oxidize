/*
 * vision.c — CLIP-style vision encoder stub implementation.
 *
 * Provides the API surface for multimodal image encoding. Currently returns
 * placeholder embeddings (zeros) — the full CLIP ViT implementation will be
 * added when a real vision GGUF model is available for testing.
 */
#include "oxidize/vision.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

OcError oc_vision_init(OcVisionEncoder *enc, const OcVisionConfig *cfg)
{
    if (!enc || !cfg) return OC_ERR_INVALID_ARG;
    memset(enc, 0, sizeof(*enc));
    enc->config = *cfg;
    if (cfg->image_size == 0 || cfg->patch_size == 0 || cfg->hidden_size == 0)
        return OC_ERR_INVALID_ARG;
    uint32_t patches_per_side = cfg->image_size / cfg->patch_size;
    enc->config.n_patches = patches_per_side * patches_per_side;
    enc->initialized = true;
    return OC_OK;
}

OcError oc_vision_encode(OcVisionEncoder *enc, const OcImage *img,
                         float *out_embeddings, size_t *out_len)
{
    if (!enc || !enc->initialized || !img || !img->data || img->width == 0 ||
        img->height == 0 || img->channels == 0 || !out_embeddings || !out_len)
        return OC_ERR_INVALID_ARG;
    size_t total = (size_t)enc->config.n_patches * enc->config.hidden_size;
    /* Stub: return zeros. Real implementation would run CLIP ViT forward. */
    memset(out_embeddings, 0, total * sizeof(float));
    *out_len = total;
    return OC_OK;
}

OcError oc_vision_resize(const OcImage *src, uint32_t target_w,
                          uint32_t target_h, uint8_t *out)
{
    if (!src || !src->data || src->width == 0 || src->height == 0 ||
        src->channels == 0 || target_w == 0 || target_h == 0 || !out)
        return OC_ERR_INVALID_ARG;
    /* Bilinear resize. */
    for (uint32_t y = 0; y < target_h; y++) {
        float src_y = (float)y * src->height / target_h;
        uint32_t y0 = (uint32_t)src_y;
        uint32_t y1 = (y0 + 1 < src->height) ? y0 + 1 : y0;
        float wy = src_y - y0;
        for (uint32_t x = 0; x < target_w; x++) {
            float src_x = (float)x * src->width / target_w;
            uint32_t x0 = (uint32_t)src_x;
            uint32_t x1 = (x0 + 1 < src->width) ? x0 + 1 : x0;
            float wx = src_x - x0;
            for (uint32_t c = 0; c < src->channels; c++) {
                size_t i00 = ((size_t)y0 * src->width + x0) * src->channels + c;
                size_t i01 = ((size_t)y0 * src->width + x1) * src->channels + c;
                size_t i10 = ((size_t)y1 * src->width + x0) * src->channels + c;
                size_t i11 = ((size_t)y1 * src->width + x1) * src->channels + c;
                float v = (1 - wx) * (1 - wy) * src->data[i00]
                       + wx * (1 - wy) * src->data[i01]
                       + (1 - wx) * wy * src->data[i10]
                       + wx * wy * src->data[i11];
                size_t out_idx = ((size_t)y * target_w + x) * src->channels + c;
                out[out_idx] = (uint8_t)(v + 0.5f);
            }
        }
    }
    return OC_OK;
}

OcError oc_vision_normalize(const OcImage *img, float *out)
{
    if (!img || !img->data || img->width == 0 || img->height == 0 ||
        img->channels == 0 || !out)
        return OC_ERR_INVALID_ARG;
    size_t n = (size_t)img->width * img->height * img->channels;
    for (size_t i = 0; i < n; i++) {
        out[i] = ((float)img->data[i] / 127.5f) - 1.0f;
    }
    return OC_OK;
}

void oc_vision_free(OcVisionEncoder *enc)
{
    if (!enc) return;
    memset(enc, 0, sizeof(*enc));
}

OcError oc_multimodal_create(const uint32_t *text_tokens, size_t n_text,
                              float *image_embeddings, size_t n_img_emb,
                              uint32_t image_token_id,
                              OcMultimodalPrompt *out)
{
    if (!text_tokens || !out) return OC_ERR_INVALID_ARG;
    out->text_tokens = malloc(n_text * sizeof(uint32_t));
    if (!out->text_tokens) return OC_ERR_OOM;
    memcpy(out->text_tokens, text_tokens, n_text * sizeof(uint32_t));
    out->n_text_tokens = n_text;
    out->image_embeddings = image_embeddings;
    out->n_image_embeddings = n_img_emb;
    out->image_token_id = image_token_id;
    return OC_OK;
}

void oc_multimodal_free(OcMultimodalPrompt *prompt)
{
    if (!prompt) return;
    free(prompt->text_tokens);
    memset(prompt, 0, sizeof(*prompt));
}
