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

    if (enc->patch_proj != NULL) {
        uint32_t p = enc->config.patch_size;
        uint32_t c = img->channels;
        uint32_t n_side = enc->config.image_size / p;
        uint32_t hidden = enc->config.hidden_size;

        uint8_t *rgb = NULL;
        OcImage rgb_img;
        const OcImage *eff_img = img;
        if (c == 4) {
            /* Projections are trained on RGB; drop the alpha channel so
             * patch_dim matches the weight layout instead of reading past
             * patch_proj. */
            size_t npix = (size_t)img->width * img->height;
            rgb = malloc(npix * 3);
            if (!rgb) return OC_ERR_OOM;
            for (size_t i = 0; i < npix; i++) {
                rgb[i * 3 + 0] = img->data[i * 4 + 0];
                rgb[i * 3 + 1] = img->data[i * 4 + 1];
                rgb[i * 3 + 2] = img->data[i * 4 + 2];
            }
            rgb_img = (OcImage){ .data=rgb, .width=img->width,
                                 .height=img->height, .channels=3,
                                 .format=OC_IMAGE_RGB };
            eff_img = &rgb_img;
            c = 3;
        }
        uint32_t patch_dim = p * p * c;

        uint8_t *resized = NULL;
        if (eff_img->width != enc->config.image_size ||
            eff_img->height != enc->config.image_size) {
            resized = malloc((size_t)enc->config.image_size *
                             enc->config.image_size * c);
            if (!resized) { free(rgb); return OC_ERR_OOM; }
            oc_vision_resize(eff_img, enc->config.image_size,
                            enc->config.image_size, resized);
            rgb_img = (OcImage){ .data=resized, .width=enc->config.image_size,
                          .height=enc->config.image_size, .channels=c,
                          .format=eff_img->format };
            free(rgb);
            rgb = NULL;
            eff_img = &rgb_img;
        }

        float *patch_pixels = malloc(patch_dim * sizeof(float));
        if (!patch_pixels) { free(resized); free(rgb); return OC_ERR_OOM; }

        for (uint32_t py = 0; py < n_side; py++) {
            for (uint32_t px = 0; px < n_side; px++) {
                uint32_t patch_idx = py * n_side + px;
                for (uint32_t dy = 0; dy < p; dy++) {
                    for (uint32_t dx = 0; dx < p; dx++) {
                        for (uint32_t ch = 0; ch < c; ch++) {
                            uint32_t x = px * p + dx;
                            uint32_t y = py * p + dy;
                            size_t idx = ((size_t)y * eff_img->width + x) * c + ch;
                            patch_pixels[(dy * p + dx) * c + ch] =
                                (float)eff_img->data[idx] / 127.5f - 1.0f;
                        }
                    }
                }
                float *out_patch = out_embeddings + (size_t)patch_idx * hidden;
                for (uint32_t h = 0; h < hidden; h++) {
                    const float *w_row = enc->patch_proj + (size_t)h * patch_dim;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < patch_dim; d++)
                        dot += w_row[d] * patch_pixels[d];
                    out_patch[h] = dot;
                }
                if (enc->pos_emb != NULL) {
                    const float *pos = enc->pos_emb + (size_t)patch_idx * hidden;
                    for (uint32_t h = 0; h < hidden; h++)
                        out_patch[h] += pos[h];
                }
            }
        }
        free(patch_pixels);
        free(resized);
        free(rgb);

        if (enc->ln_weight != NULL && enc->ln_bias != NULL) {
            for (uint32_t pi = 0; pi < enc->config.n_patches; pi++) {
                float *emb = out_embeddings + (size_t)pi * hidden;
                /* True LayerNorm: center by per-patch mean, then scale by
                 * the standard deviation. */
                float mean = 0.0f;
                for (uint32_t h = 0; h < hidden; h++) mean += emb[h];
                mean /= hidden;
                float ss = 0.0f;
                for (uint32_t h = 0; h < hidden; h++) {
                    float d = emb[h] - mean;
                    ss += d * d;
                }
                float inv = 1.0f / sqrtf(ss / hidden + 1e-6f);
                for (uint32_t h = 0; h < hidden; h++)
                    emb[h] = (emb[h] - mean) * inv * enc->ln_weight[h] +
                             enc->ln_bias[h];
            }
        }
    } else {
        memset(out_embeddings, 0, total * sizeof(float));
    }
    *out_len = total;
    return OC_OK;
}

OcError oc_vision_set_weights(OcVisionEncoder *enc,
                               const float *patch_proj,
                               const float *pos_emb,
                               const float *ln_weight,
                               const float *ln_bias,
                               const float *cls_emb)
{
    if (!enc) return OC_ERR_INVALID_ARG;
    /* CLS-token prepending is not implemented: the encode output contract is
     * exactly n_patches * hidden_size floats. Reject rather than silently
     * dropping the CLS embedding. */
    if (cls_emb != NULL) return OC_ERR_INVALID_ARG;
    enc->patch_proj = patch_proj;
    enc->pos_emb = pos_emb;
    enc->ln_weight = ln_weight;
    enc->ln_bias = ln_bias;
    enc->cls_emb = cls_emb;
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
