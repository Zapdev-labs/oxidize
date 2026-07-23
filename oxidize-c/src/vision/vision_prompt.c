/*
 * vision_prompt.c — Multimodal prompt construction implementation.
 */
#include "oxidize/vision_prompt.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static void copy_str(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0 || !src) { if (dst && cap > 0) dst[0] = '\0'; return; }
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

OcError oc_vision_prompt_init(OcVisionPrompt *vp, OcVisionPromptFormat fmt)
{
    if (!vp) return OC_ERR_INVALID_ARG;
    memset(vp, 0, sizeof(*vp));
    vp->format = fmt;
    return OC_OK;
}

OcError oc_vision_prompt_set_text(OcVisionPrompt *vp, const char *text)
{
    if (!vp || !text) return OC_ERR_INVALID_ARG;
    copy_str(vp->text, sizeof(vp->text), text);
    return OC_OK;
}

OcError oc_vision_prompt_add_image(OcVisionPrompt *vp, const float *features,
                                   size_t n_features)
{
    if (!vp || !features || n_features == 0) return OC_ERR_INVALID_ARG;
    if (vp->n_images >= OC_VP_MAX_IMAGES) return OC_ERR_OOM;

    OcVisionFeature *f = &vp->images[vp->n_images];
    f->features = malloc(n_features * sizeof(float));
    if (!f->features) return OC_ERR_OOM;
    memcpy(f->features, features, n_features * sizeof(float));
    f->n_features = n_features;
    vp->n_images++;
    return OC_OK;
}

OcError oc_vision_prompt_render(const OcVisionPrompt *vp, char *out, size_t out_size)
{
    if (!vp || !out || out_size == 0) return OC_ERR_INVALID_ARG;

    size_t pos = 0;
    int written;

    switch (vp->format) {
    case OC_VP_FORMAT_LLAVA:
        written = snprintf(out + pos, out_size - pos, "<image>\n");
        if (written < 0 || (size_t)written >= out_size - pos) return OC_ERR_OOM;
        pos += written;
        break;
    case OC_VP_FORMAT_QWEN_VL:
        for (uint32_t i = 0; i < vp->n_images && pos < out_size; i++) {
            written = snprintf(out + pos, out_size - pos, "<|image_%u|>", i + 1);
            if (written < 0 || (size_t)written >= out_size - pos) return OC_ERR_OOM;
            pos += written;
        }
        if (pos < out_size) { out[pos++] = '\n'; }
        break;
    case OC_VP_FORMAT_INTERNVL:
        written = snprintf(out + pos, out_size - pos, "<image>");
        if (written < 0 || (size_t)written >= out_size - pos) return OC_ERR_OOM;
        pos += written;
        break;
    case OC_VP_FORMAT_MPLUG_OWL:
        /* mPLUG-Owl uses image tokens at end. */
        break;
    }

    /* Append text. */
    size_t text_len = strlen(vp->text);
    if (pos + text_len + 1 >= out_size) {
        size_t remaining = out_size - pos - 1;
        memcpy(out + pos, vp->text, remaining);
        pos += remaining;
    } else {
        memcpy(out + pos, vp->text, text_len);
        pos += text_len;
    }
    out[pos] = '\0';

    /* mPLUG-Owl image tokens at end. */
    if (vp->format == OC_VP_FORMAT_MPLUG_OWL && vp->n_images > 0) {
        size_t remaining = out_size - pos - 1;
        written = snprintf(out + pos, remaining, " <image>");
        if (written > 0 && (size_t)written < remaining) pos += written;
    }

    return OC_OK;
}

OcError oc_vision_prompt_render_tokens(const OcVisionPrompt *vp,
                                      uint32_t *out_tokens, size_t max_tokens,
                                      size_t *out_n)
{
    if (!vp || !out_tokens || !out_n) return OC_ERR_INVALID_ARG;
    *out_n = 0;

    /* Stub: render image placeholders as token 1, text as token 0. */
    for (uint32_t i = 0; i < vp->n_images && *out_n < max_tokens; i++) {
        out_tokens[(*out_n)++] = 1; /* IMAGE token */
    }
    /* Tokenize text simply (one token per char). */
    for (size_t i = 0; vp->text[i] && *out_n < max_tokens; i++) {
        out_tokens[(*out_n)++] = (uint32_t)(unsigned char)vp->text[i];
    }
    return OC_OK;
}

uint32_t oc_vision_prompt_n_images(const OcVisionPrompt *vp)
{
    return vp ? vp->n_images : 0;
}

const char *oc_vision_prompt_format_name(OcVisionPromptFormat fmt)
{
    switch (fmt) {
    case OC_VP_FORMAT_LLAVA:     return "llava";
    case OC_VP_FORMAT_QWEN_VL:   return "qwen_vl";
    case OC_VP_FORMAT_INTERNVL:  return "internvl";
    case OC_VP_FORMAT_MPLUG_OWL: return "mplug_owl";
    default: return "unknown";
    }
}

void oc_vision_prompt_free(OcVisionPrompt *vp)
{
    if (!vp) return;
    for (uint32_t i = 0; i < vp->n_images; i++)
        free(vp->images[i].features);
    memset(vp, 0, sizeof(*vp));
}
