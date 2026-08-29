/* test_vision_prompt.c — Vision prompt tests. */
#include "framework.h"
#include "oxidize/vision_prompt.h"
#include <string.h>

Test(vp, init)
{
    OcVisionPrompt vp;
    cr_assert_eq(oc_vision_prompt_init(&vp, OC_VP_FORMAT_LLAVA), OC_OK);
    cr_assert_eq(vp.format, OC_VP_FORMAT_LLAVA);
    cr_assert_eq(vp.n_images, 0);
    cr_assert_eq(vp.text[0], '\0');
    oc_vision_prompt_free(&vp);
}

OC_TEST_NULL_SAFE(vp, init_null,
        cr_assert_neq(oc_vision_prompt_init(NULL, OC_VP_FORMAT_LLAVA), OC_OK);)

Test(vp, set_text)
{
    OcVisionPrompt vp;
    oc_vision_prompt_init(&vp, OC_VP_FORMAT_LLAVA);
    cr_assert_eq(oc_vision_prompt_set_text(&vp, "hello"), OC_OK);
    cr_assert_str_eq(vp.text, "hello");
    oc_vision_prompt_free(&vp);
}

OC_TEST_NULL_SAFE(vp, set_text_null,
        cr_assert_neq(oc_vision_prompt_set_text(NULL, NULL), OC_OK);)

Test(vp, add_image)
{
    OcVisionPrompt vp;
    oc_vision_prompt_init(&vp, OC_VP_FORMAT_LLAVA);
    float features[] = {1.0f, 2.0f, 3.0f};
    cr_assert_eq(oc_vision_prompt_add_image(&vp, features, 3), OC_OK);
    cr_assert_eq(vp.n_images, 1);
    cr_assert_eq(vp.images[0].n_features, 3);
    cr_assert_float_eq(vp.images[0].features[0], 1.0f, 0.001f);
    oc_vision_prompt_free(&vp);
}

OC_TEST_NULL_SAFE(vp, add_image_null,
        cr_assert_neq(oc_vision_prompt_add_image(NULL, NULL, 0), OC_OK);)

Test(vp, add_multiple_images)
{
    OcVisionPrompt vp;
    oc_vision_prompt_init(&vp, OC_VP_FORMAT_QWEN_VL);
    float f1[] = {1.0f};
    float f2[] = {2.0f, 3.0f};
    oc_vision_prompt_add_image(&vp, f1, 1);
    oc_vision_prompt_add_image(&vp, f2, 2);
    cr_assert_eq(vp.n_images, 2);
    oc_vision_prompt_free(&vp);
}

Test(vp, render_llava)
{
    OcVisionPrompt vp;
    oc_vision_prompt_init(&vp, OC_VP_FORMAT_LLAVA);
    oc_vision_prompt_set_text(&vp, "What is this?");
    char out[256];
    cr_assert_eq(oc_vision_prompt_render(&vp, out, sizeof(out)), OC_OK);
    cr_assert(strstr(out, "<image>") != NULL);
    cr_assert(strstr(out, "What is this?") != NULL);
    oc_vision_prompt_free(&vp);
}

Test(vp, render_qwen_vl)
{
    OcVisionPrompt vp;
    oc_vision_prompt_init(&vp, OC_VP_FORMAT_QWEN_VL);
    float f[] = {1.0f};
    oc_vision_prompt_add_image(&vp, f, 1);
    oc_vision_prompt_set_text(&vp, "Describe");
    char out[256];
    cr_assert_eq(oc_vision_prompt_render(&vp, out, sizeof(out)), OC_OK);
    cr_assert(strstr(out, "<|image_1|>") != NULL);
    oc_vision_prompt_free(&vp);
}

Test(vp, render_internvl)
{
    OcVisionPrompt vp;
    oc_vision_prompt_init(&vp, OC_VP_FORMAT_INTERNVL);
    oc_vision_prompt_set_text(&vp, "test");
    char out[256];
    cr_assert_eq(oc_vision_prompt_render(&vp, out, sizeof(out)), OC_OK);
    cr_assert(strstr(out, "<image>") != NULL);
    oc_vision_prompt_free(&vp);
}

Test(vp, render_mplug_owl)
{
    OcVisionPrompt vp;
    oc_vision_prompt_init(&vp, OC_VP_FORMAT_MPLUG_OWL);
    float f[] = {1.0f};
    oc_vision_prompt_add_image(&vp, f, 1);
    oc_vision_prompt_set_text(&vp, "test");
    char out[256];
    cr_assert_eq(oc_vision_prompt_render(&vp, out, sizeof(out)), OC_OK);
    cr_assert(strstr(out, "<image>") != NULL);
    oc_vision_prompt_free(&vp);
}

OC_TEST_NULL_SAFE(vp, render_null,
        cr_assert_neq(oc_vision_prompt_render(NULL, NULL, 0), OC_OK);)

Test(vp, render_tokens)
{
    OcVisionPrompt vp;
    oc_vision_prompt_init(&vp, OC_VP_FORMAT_LLAVA);
    float f[] = {1.0f};
    oc_vision_prompt_add_image(&vp, f, 1);
    oc_vision_prompt_set_text(&vp, "hi");
    uint32_t tokens[100];
    size_t n;
    cr_assert_eq(oc_vision_prompt_render_tokens(&vp, tokens, 100, &n), OC_OK);
    /* LLaVA: 1 image token + 1 word token ("hi") = 2 tokens. */
    cr_assert_eq(n, 2);
    /* Image token should be in the special range. */
    cr_assert(tokens[0] >= 0x7FFFFFF0);
    /* Text token should be in normal vocabulary range (1-32000). */
    cr_assert(tokens[1] >= 1 && tokens[1] <= 32000);
    oc_vision_prompt_free(&vp);
}

Test(vp, render_tokens_multi_word)
{
    OcVisionPrompt vp;
    oc_vision_prompt_init(&vp, OC_VP_FORMAT_LLAVA);
    float f[] = {1.0f};
    oc_vision_prompt_add_image(&vp, f, 1);
    oc_vision_prompt_set_text(&vp, "hello world foo");
    uint32_t tokens[100];
    size_t n;
    cr_assert_eq(oc_vision_prompt_render_tokens(&vp, tokens, 100, &n), OC_OK);
    /* 1 image + 3 words + 2 spaces = 6 tokens. */
    cr_assert_eq(n, 6, "expected 6 tokens, got %zu", n);
    /* First token is image. */
    cr_assert(tokens[0] >= 0x7FFFFFF0);
    /* Word tokens (djb2 hash) should be non-zero. */
    cr_assert(tokens[1] > 0);  /* "hello" */
    /* Space tokens should be 0. */
    cr_assert_eq(tokens[2], 0);  /* space */
    cr_assert(tokens[3] > 0);  /* "world" */
    oc_vision_prompt_free(&vp);
}

Test(vp, render_tokens_qwen_vl)
{
    OcVisionPrompt vp;
    oc_vision_prompt_init(&vp, OC_VP_FORMAT_QWEN_VL);
    float f[] = {1.0f, 2.0f};
    oc_vision_prompt_add_image(&vp, f, 2);
    oc_vision_prompt_add_image(&vp, f, 2);
    oc_vision_prompt_set_text(&vp, "test");
    uint32_t tokens[100];
    size_t n;
    cr_assert_eq(oc_vision_prompt_render_tokens(&vp, tokens, 100, &n), OC_OK);
    /* Qwen-VL: 2 numbered image tokens + 1 word = 3 tokens. */
    cr_assert_eq(n, 3);
    /* Image tokens should be sequential (IMAGE_TOKEN_BASE + 0, +1). */
    cr_assert_eq(tokens[0], 0x7FFFFFF0);
    cr_assert_eq(tokens[1], 0x7FFFFFF1);
    oc_vision_prompt_free(&vp);
}

Test(vp, render_tokens_mplug_owl)
{
    OcVisionPrompt vp;
    oc_vision_prompt_init(&vp, OC_VP_FORMAT_MPLUG_OWL);
    float f[] = {1.0f};
    oc_vision_prompt_add_image(&vp, f, 1);
    oc_vision_prompt_set_text(&vp, "test");
    uint32_t tokens[100];
    size_t n;
    cr_assert_eq(oc_vision_prompt_render_tokens(&vp, tokens, 100, &n), OC_OK);
    /* mPLUG-Owl: text first, then image at end. */
    cr_assert_eq(n, 2);  /* 1 word + 1 image */
    cr_assert(tokens[0] > 0 && tokens[0] <= 32000);  /* word */
    cr_assert(tokens[1] >= 0x7FFFFFF0);  /* image */
    oc_vision_prompt_free(&vp);
}

OC_TEST_NULL_SAFE(vp, render_tokens_null,
        cr_assert_neq(oc_vision_prompt_render_tokens(NULL, NULL, 0, NULL), OC_OK);)

Test(vp, n_images)
{
    OcVisionPrompt vp;
    oc_vision_prompt_init(&vp, OC_VP_FORMAT_LLAVA);
    cr_assert_eq(oc_vision_prompt_n_images(&vp), 0);
    float f[] = {1.0f};
    oc_vision_prompt_add_image(&vp, f, 1);
    cr_assert_eq(oc_vision_prompt_n_images(&vp), 1);
    oc_vision_prompt_free(&vp);
}

OC_TEST_NULL_SAFE(vp, n_images_null,
        cr_assert_eq(oc_vision_prompt_n_images(NULL), 0);)

Test(vp, format_name)
{
    cr_assert_str_eq(oc_vision_prompt_format_name(OC_VP_FORMAT_LLAVA), "llava");
    cr_assert_str_eq(oc_vision_prompt_format_name(OC_VP_FORMAT_QWEN_VL), "qwen_vl");
    cr_assert_str_eq(oc_vision_prompt_format_name(OC_VP_FORMAT_INTERNVL), "internvl");
    cr_assert_str_eq(oc_vision_prompt_format_name(OC_VP_FORMAT_MPLUG_OWL), "mplug_owl");
}

OC_TEST_NULL_SAFE(vp, free_null,
        oc_vision_prompt_free(NULL);)
