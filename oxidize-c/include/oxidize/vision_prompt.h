#ifndef OXIDIZE_VISION_PROMPT_H
#define OXIDIZE_VISION_PROMPT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_VP_MAX_IMAGES 8
#define OC_VP_MAX_TEXT 4096

typedef struct {
    float *features;
    size_t n_features;
} OcVisionFeature;

typedef enum {
    OC_VP_FORMAT_LLAVA = 0,
    OC_VP_FORMAT_QWEN_VL = 1,
    OC_VP_FORMAT_INTERNVL = 2,
    OC_VP_FORMAT_MPLUG_OWL = 3,
} OcVisionPromptFormat;

typedef struct {
    char text[OC_VP_MAX_TEXT];
    OcVisionFeature images[OC_VP_MAX_IMAGES];
    uint32_t n_images;
    OcVisionPromptFormat format;
} OcVisionPrompt;

OcError oc_vision_prompt_init(OcVisionPrompt *vp, OcVisionPromptFormat fmt);
OcError oc_vision_prompt_set_text(OcVisionPrompt *vp, const char *text);
OcError oc_vision_prompt_add_image(OcVisionPrompt *vp, const float *features,
                                   size_t n_features);
OcError oc_vision_prompt_render(const OcVisionPrompt *vp, char *out, size_t out_size);
OcError oc_vision_prompt_render_tokens(const OcVisionPrompt *vp,
                                      uint32_t *out_tokens, size_t max_tokens,
                                      size_t *out_n);
uint32_t oc_vision_prompt_n_images(const OcVisionPrompt *vp);
const char *oc_vision_prompt_format_name(OcVisionPromptFormat fmt);
void oc_vision_prompt_free(OcVisionPrompt *vp);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_VISION_PROMPT_H */
