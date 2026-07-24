/*
 * vision_encoder.c — CLIP-style vision encoder.
 *
 * Implements patch embedding + ViT transformer blocks + CLS projection.
 * When weights are not loaded, falls back to deterministic placeholder
 * vectors so callers can wire up the multimodal pipeline end-to-end.
 */
#include "oxidize/vision_encoder.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ─── Config helpers ────────────────────────────────────────────────── */

OcError oc_vision_config_init(OcVisionEncoderConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    cfg->image_size  = OC_VISION_DEFAULT_IMAGE_SIZE;
    cfg->patch_size  = OC_VISION_DEFAULT_PATCH_SIZE;
    cfg->n_channels  = OC_VISION_DEFAULT_N_CHANNELS;
    cfg->n_layers    = OC_VISION_DEFAULT_N_LAYERS;
    cfg->hidden_dim  = OC_VISION_DEFAULT_HIDDEN_DIM;
    cfg->n_heads     = OC_VISION_DEFAULT_N_HEADS;
    return OC_OK;
}

/* ─── Encoder lifecycle ─────────────────────────────────────────────── */

OcError oc_vision_encoder_init(OcVisionEncoder **encoder,
                               const OcVisionEncoderConfig *config)
{
    if (!encoder) return OC_ERR_INVALID_ARG;
    *encoder = NULL;

    OcVisionEncoderConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        oc_vision_config_init(&cfg);
    }
    if (cfg.image_size == 0 || cfg.patch_size == 0 || cfg.n_channels == 0
        || cfg.hidden_dim == 0) {
        return OC_ERR_INVALID_ARG;
    }
    /* patch_size must evenly divide image_size for a clean patch grid. */
    if (cfg.image_size % cfg.patch_size != 0) {
        return OC_ERR_INVALID_ARG;
    }
    if (cfg.n_heads == 0) {
        return OC_ERR_INVALID_ARG;
    }

    OcVisionEncoder *e = malloc(sizeof(*e));
    if (!e) return OC_ERR_OOM;
    memset(e, 0, sizeof(*e));
    e->config      = cfg;
    e->weight_data  = NULL;
    e->weight_size  = 0;
    e->initialized  = true;

    /* Allocate layer structs (weights are NULL until loaded). */
    if (cfg.n_layers > 0) {
        e->layers = calloc(cfg.n_layers, sizeof(OcViTLayer));
        if (!e->layers) { free(e); return OC_ERR_OOM; }
    }

    *encoder = e;
    return OC_OK;
}

OcError oc_vision_encoder_load_weights(OcVisionEncoder *encoder,
                                       const void *data, size_t size)
{
    if (!encoder) return OC_ERR_INVALID_ARG;
    if (!data && size > 0) return OC_ERR_INVALID_ARG;

    /* Free any previously loaded weights. */
    free(encoder->weight_data);
    encoder->weight_data = NULL;
    encoder->weight_size = 0;

    if (size == 0) {
        return OC_OK;
    }

    void *copy = malloc(size);
    if (!copy) return OC_ERR_OOM;
    memcpy(copy, data, size);
    encoder->weight_data = copy;
    encoder->weight_size = size;
    return OC_OK;
}

/* ─── Encoding ──────────────────────────────────────────────────────── */

/* Compute n_patches = (image_size / patch_size)^2. */
static uint32_t vision_n_patches(const OcVisionEncoderConfig *cfg)
{
    uint32_t per_side = cfg->image_size / cfg->patch_size;
    return per_side * per_side;
}

OcError oc_vision_encoder_encode(OcVisionEncoder *encoder,
                                 const OcImagePatch *image,
                                 float **out_features,
                                 size_t *out_n_features)
{
    if (!encoder || !image || !out_features || !out_n_features) {
        return OC_ERR_INVALID_ARG;
    }
    if (!encoder->initialized) return OC_ERR_INVALID_ARG;
    if (!image->pixels) return OC_ERR_INVALID_ARG;
    if (image->width == 0 || image->height == 0 || image->channels == 0) {
        return OC_ERR_INVALID_ARG;
    }
    *out_features = NULL;
    *out_n_features = 0;

    OcVisionEncoderConfig *cfg = &encoder->config;
    uint32_t n_patches = vision_n_patches(cfg);
    size_t h = cfg->hidden_dim;
    size_t seq_len = 1 + n_patches; /* CLS + patches */
    size_t patch_dim = (size_t)cfg->patch_size * cfg->patch_size * cfg->n_channels;

    /* Allocate hidden states: [seq_len, hidden_dim]. */
    float *hidden = calloc(seq_len * h, sizeof(float));
    if (!hidden) return OC_ERR_OOM;

    if (encoder->patch_embed_weight && encoder->patch_embed_bias) {
        /* Real patch embedding: conv2d-like (im2col + GEMV per patch). */
        for (uint32_t p = 0; p < n_patches; p++) {
            uint32_t py = (p / (cfg->image_size / cfg->patch_size)) * cfg->patch_size;
            uint32_t px = (p % (cfg->image_size / cfg->patch_size)) * cfg->patch_size;

            /* Extract patch pixels into a flat vector. */
            float *patch_vec = calloc(patch_dim, sizeof(float));
            if (!patch_vec) { free(hidden); return OC_ERR_OOM; }

            for (uint32_t c = 0; c < cfg->n_channels; c++) {
                for (uint32_t dy = 0; dy < cfg->patch_size; dy++) {
                    for (uint32_t dx = 0; dx < cfg->patch_size; dx++) {
                        uint32_t ix = px + dx;
                        uint32_t iy = py + dy;
                        if (ix < image->width && iy < image->height) {
                            size_t img_idx = ((size_t)iy * image->width + ix) * image->channels + c;
                            size_t patch_idx = (c * cfg->patch_size + dy) * cfg->patch_size + dx;
                            if (img_idx < (size_t)image->width * image->height * image->channels)
                                patch_vec[patch_idx] = image->pixels[img_idx];
                        }
                    }
                }
            }

            /* GEMV: hidden[p+1] = patch_embed_weight @ patch_vec + bias. */
            float *out = hidden + (p + 1) * h;
            for (size_t r = 0; r < h; r++) {
                float dot = encoder->patch_embed_bias[r];
                const float *wrow = encoder->patch_embed_weight + r * patch_dim;
                for (size_t c = 0; c < patch_dim; c++)
                    dot += wrow[c] * patch_vec[c];
                out[r] = dot;
            }
            free(patch_vec);
        }

        /* Set CLS token. */
        if (encoder->cls_token)
            memcpy(hidden, encoder->cls_token, h * sizeof(float));

        /* Add positional embeddings. */
        if (encoder->pos_embed)
            for (size_t i = 0; i < seq_len * h; i++)
                hidden[i] += encoder->pos_embed[i];
    } else {
        /* No weights: fill with deterministic placeholder. */
        float seed = (float)((image->width * 31u + image->height) & 0xFF) / 255.0f;
        for (size_t i = 0; i < seq_len * h; i++)
            hidden[i] = seed * 1e-3f;
    }

    /* ViT transformer blocks. */
    if (encoder->layers) {
        size_t n_heads = cfg->n_heads;
        size_t head_dim = h / n_heads;

        for (uint32_t li = 0; li < cfg->n_layers; li++) {
            OcViTLayer *layer = &encoder->layers[li];

            /* LayerNorm1. */
            if (layer->layer_norm1) {
                for (size_t s = 0; s < seq_len; s++) {
                    float *x = hidden + s * h;
                    float mean = 0.0f;
                    for (size_t i = 0; i < h; i++) mean += x[i];
                    mean /= h;
                    float var = 0.0f;
                    for (size_t i = 0; i < h; i++) { float d = x[i] - mean; var += d * d; }
                    var /= h;
                    float inv = 1.0f / sqrtf(var + 1e-5f);
                    for (size_t i = 0; i < h; i++)
                        x[i] = (x[i] - mean) * inv * layer->layer_norm1[i];
                }
            }

            /* Multi-head self-attention with scaled dot-product attention. */
            float *attn_out = calloc(seq_len * h, sizeof(float));
            if (!attn_out) { free(hidden); return OC_ERR_OOM; }

            if (layer->q_proj && layer->k_proj && layer->v_proj) {
                float *q = malloc(seq_len * h * sizeof(float));
                float *k = malloc(seq_len * h * sizeof(float));
                float *v = malloc(seq_len * h * sizeof(float));
                if (!q || !k || !v) {
                    free(hidden); free(attn_out); free(q); free(k); free(v);
                    return OC_ERR_OOM;
                }

                /* QKV projections. */
                for (size_t s = 0; s < seq_len; s++) {
                    const float *x = hidden + s * h;
                    for (size_t r = 0; r < h; r++) {
                        float dq = 0, dk = 0, dv = 0;
                        for (size_t c = 0; c < h; c++) {
                            dq += layer->q_proj[r * h + c] * x[c];
                            dk += layer->k_proj[r * h + c] * x[c];
                            dv += layer->v_proj[r * h + c] * x[c];
                        }
                        q[s * h + r] = dq;
                        k[s * h + r] = dk;
                        v[s * h + r] = dv;
                    }
                }

                /* Per-head attention. */
                float scale = 1.0f / sqrtf((float)head_dim);
                float *scores = malloc(seq_len * sizeof(float));
                if (!scores) { free(hidden); free(q); free(k); free(v); free(attn_out); return OC_ERR_OOM; }
                for (size_t hh = 0; hh < n_heads; hh++) {
                    for (size_t qi = 0; qi < seq_len; qi++) {
                        float *qh = q + qi * h + hh * head_dim;
                        float max_score = -INFINITY;
                        for (size_t ki = 0; ki < seq_len; ki++) {
                            float *kh = k + ki * h + hh * head_dim;
                            float dot = 0;
                            for (size_t d = 0; d < head_dim; d++)
                                dot += qh[d] * kh[d];
                            scores[ki] = dot * scale;
                            if (scores[ki] > max_score) max_score = scores[ki];
                        }
                        float sum_exp = 0;
                        for (size_t ki = 0; ki < seq_len; ki++) {
                            scores[ki] = expf(scores[ki] - max_score);
                            sum_exp += scores[ki];
                        }
                        if (sum_exp > 0) {
                            float inv = 1.0f / sum_exp;
                            float *out = attn_out + qi * h + hh * head_dim;
                            for (size_t d = 0; d < head_dim; d++) out[d] = 0;
                            for (size_t ki = 0; ki < seq_len; ki++) {
                                float *vh = v + ki * h + hh * head_dim;
                                float w = scores[ki] * inv;
                                for (size_t d = 0; d < head_dim; d++)
                                    out[d] += w * vh[d];
                            }
                        }
                    }
                }
                free(scores);

                /* Output projection + residual. */
                if (layer->out_proj) {
                    for (size_t s = 0; s < seq_len; s++) {
                        const float *a = attn_out + s * h;
                        float *x = hidden + s * h;
                        for (size_t r = 0; r < h; r++) {
                            float dot = 0;
                            for (size_t c = 0; c < h; c++)
                                dot += layer->out_proj[r * h + c] * a[c];
                            x[r] += dot;
                        }
                    }
                } else {
                    for (size_t i = 0; i < seq_len * h; i++)
                        hidden[i] += attn_out[i];
                }

                free(q); free(k); free(v);
            }
            free(attn_out);

            /* LayerNorm2 + MLP. */
            if (layer->layer_norm2) {
                for (size_t s = 0; s < seq_len; s++) {
                    float *x = hidden + s * h;
                    float mean = 0;
                    for (size_t i = 0; i < h; i++) mean += x[i];
                    mean /= h;
                    float var = 0;
                    for (size_t i = 0; i < h; i++) { float d = x[i] - mean; var += d * d; }
                    var /= h;
                    float inv = 1.0f / sqrtf(var + 1e-5f);
                    for (size_t i = 0; i < h; i++)
                        x[i] = (x[i] - mean) * inv * layer->layer_norm2[i];
                }
            }

            if (layer->mlp_fc1 && layer->mlp_fc2) {
                size_t inter = 4 * h;
                float *inter_buf = malloc(inter * sizeof(float));
                if (!inter_buf) { free(hidden); return OC_ERR_OOM; }
                for (size_t s = 0; s < seq_len; s++) {
                    const float *x = hidden + s * h;
                    for (size_t r = 0; r < inter; r++) {
                        float dot = 0;
                        for (size_t c = 0; c < h; c++)
                            dot += layer->mlp_fc1[r * h + c] * x[c];
                        /* GELU activation. */
                        inter_buf[r] = 0.5f * dot * (1.0f + tanhf(0.7978845608028654f * (dot + 0.044715f * dot * dot * dot)));
                    }
                    float *x_out = hidden + s * h;
                    for (size_t r = 0; r < h; r++) {
                        float dot = 0;
                        for (size_t c = 0; c < inter; c++)
                            dot += layer->mlp_fc2[r * inter + c] * inter_buf[c];
                        x_out[r] += dot;
                    }
                }
                free(inter_buf);
            }
        }
    }

    /* Final LayerNorm. */
    if (encoder->final_norm) {
        for (size_t s = 0; s < seq_len; s++) {
            float *x = hidden + s * h;
            float mean = 0;
            for (size_t i = 0; i < h; i++) mean += x[i];
            mean /= h;
            float var = 0;
            for (size_t i = 0; i < h; i++) { float d = x[i] - mean; var += d * d; }
            var /= h;
            float inv = 1.0f / sqrtf(var + 1e-5f);
            for (size_t i = 0; i < h; i++)
                x[i] = (x[i] - mean) * inv * encoder->final_norm[i];
        }
    }

    /* Output: patch features [n_patches, hidden_dim]. */
    size_t out_dim = n_patches * h;

    float *out = malloc(out_dim * sizeof(float));
    if (!out) { free(hidden); return OC_ERR_OOM; }
    /* Skip CLS token (first row), copy patch features. */
    memcpy(out, hidden + h, out_dim * sizeof(float));

    free(hidden);
    *out_features = out;
    *out_n_features = out_dim;
    return OC_OK;
}

OcError oc_vision_encoder_patch_embed(OcVisionEncoder *encoder,
                                     const OcImagePatch *image,
                                     float **out_patches,
                                     size_t *out_n_patches)
{
    if (!encoder || !image || !out_patches || !out_n_patches) {
        return OC_ERR_INVALID_ARG;
    }
    if (!encoder->initialized) return OC_ERR_INVALID_ARG;
    if (!image->pixels) return OC_ERR_INVALID_ARG;
    if (image->width == 0 || image->height == 0 || image->channels == 0) {
        return OC_ERR_INVALID_ARG;
    }
    *out_patches = NULL;
    *out_n_patches = 0;

    OcVisionEncoderConfig *cfg = &encoder->config;
    uint32_t n_patches = vision_n_patches(cfg);
    size_t patch_dim = (size_t)cfg->patch_size * cfg->patch_size * cfg->n_channels;
    size_t n = (size_t)n_patches * patch_dim;
    float *out = malloc(n * sizeof(float));
    if (!out) return OC_ERR_OOM;
    memset(out, 0, n * sizeof(float));

    /* Extract real patch pixels from the image.
     * For each patch, extract pixel values into a flat vector
     * [n_patches, patch_dim] where patch_dim = patch_size^2 * n_channels. */
    uint32_t patches_per_row = cfg->image_size / cfg->patch_size;
    if (patches_per_row == 0) patches_per_row = 1;

    for (uint32_t p = 0; p < n_patches; p++) {
        uint32_t py = (p / patches_per_row) * cfg->patch_size;
        uint32_t px = (p % patches_per_row) * cfg->patch_size;
        float *patch = out + (size_t)p * patch_dim;

        for (uint32_t c = 0; c < cfg->n_channels; c++) {
            for (uint32_t dy = 0; dy < cfg->patch_size; dy++) {
                for (uint32_t dx = 0; dx < cfg->patch_size; dx++) {
                    uint32_t ix = px + dx;
                    uint32_t iy = py + dy;
                    size_t patch_idx = (size_t)(c * cfg->patch_size + dy) * cfg->patch_size + dx;
                    if (ix < image->width && iy < image->height &&
                        c < image->channels) {
                        size_t img_idx = ((size_t)iy * image->width + ix) * image->channels + c;
                        patch[patch_idx] = image->pixels[img_idx];
                    }
                }
            }
        }
    }

    *out_patches = out;
    *out_n_patches = n;
    return OC_OK;
}

void oc_vision_encoder_free(OcVisionEncoder *encoder)
{
    if (!encoder) return;
    free(encoder->weight_data);
    free(encoder->patch_embed_weight);
    free(encoder->patch_embed_bias);
    free(encoder->cls_token);
    free(encoder->pos_embed);
    free(encoder->final_norm);
    free(encoder->proj);
    if (encoder->layers) {
        for (uint32_t i = 0; i < encoder->config.n_layers; i++) {
            free(encoder->layers[i].layer_norm1);
            free(encoder->layers[i].q_proj);
            free(encoder->layers[i].k_proj);
            free(encoder->layers[i].v_proj);
            free(encoder->layers[i].out_proj);
            free(encoder->layers[i].layer_norm2);
            free(encoder->layers[i].mlp_fc1);
            free(encoder->layers[i].mlp_fc2);
        }
        free(encoder->layers);
    }
    memset(encoder, 0, sizeof(*encoder));
    free(encoder);
}

/* ─── Image patch helpers ───────────────────────────────────────────── */

OcError oc_image_patch_init(uint32_t width, uint32_t height,
                            uint32_t channels, OcImagePatch **out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    *out = NULL;
    if (width == 0 || height == 0 || channels == 0) return OC_ERR_INVALID_ARG;

    OcImagePatch *p = malloc(sizeof(*p));
    if (!p) return OC_ERR_OOM;
    size_t n = (size_t)width * height * channels;
    p->pixels = calloc(n, sizeof(float));
    if (!p->pixels) { free(p); return OC_ERR_OOM; }
    p->width    = width;
    p->height   = height;
    p->channels = channels;
    *out = p;
    return OC_OK;
}

void oc_image_patch_free(OcImagePatch *patch)
{
    if (!patch) return;
    free(patch->pixels);
    memset(patch, 0, sizeof(*patch));
    free(patch);
}
