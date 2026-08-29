/* test_vision_config.c — Vision config tests. */
#include "framework.h"
#include "oxidize/vision_config.h"

Test(vcfg, init)
{
    OcVisionConfig cfg;
    cr_assert_eq(oc_vision_cfg_init(&cfg), OC_OK);
    cr_assert_eq(cfg.model_type, OC_VISION_MODEL_CLIP);
    cr_assert_eq(cfg.image_size, 224);
    cr_assert_eq(cfg.patch_size, 16);
    cr_assert_eq(cfg.n_layers, 12);
    cr_assert_eq(cfg.hidden_dim, 768);
}

OC_TEST_NULL_SAFE(vcfg, init_null,
        cr_assert_neq(oc_vision_cfg_init(NULL), OC_OK);)

Test(vcfg, clip_base)
{
    OcVisionConfig cfg;
    cr_assert_eq(oc_vision_cfg_clip_base(&cfg), OC_OK);
    cr_assert_eq(cfg.image_size, 224);
    cr_assert_eq(cfg.n_layers, 12);
    cr_assert_eq(cfg.hidden_dim, 768);
}

Test(vcfg, clip_large)
{
    OcVisionConfig cfg;
    cr_assert_eq(oc_vision_cfg_clip_large(&cfg), OC_OK);
    cr_assert_eq(cfg.image_size, 336);
    cr_assert_eq(cfg.n_layers, 24);
    cr_assert_eq(cfg.hidden_dim, 1024);
}

Test(vcfg, siglip)
{
    OcVisionConfig cfg;
    cr_assert_eq(oc_vision_cfg_siglip(&cfg), OC_OK);
    cr_assert_eq(cfg.model_type, OC_VISION_MODEL_SIGLIP);
    cr_assert_eq(cfg.image_size, 384);
    cr_assert_eq(cfg.n_layers, 27);
    cr_assert(!cfg.use_abs_pos_emb);
    cr_assert(cfg.use_rotary_emb);
}

Test(vcfg, validate_ok)
{
    OcVisionConfig cfg;
    oc_vision_cfg_init(&cfg);
    cr_assert_eq(oc_vision_cfg_validate(&cfg), OC_OK);
}

OC_TEST_NULL_SAFE(vcfg, validate_null,
        cr_assert_neq(oc_vision_cfg_validate(NULL), OC_OK);)

Test(vcfg, validate_bad_patch)
{
    OcVisionConfig cfg;
    oc_vision_cfg_init(&cfg);
    cfg.patch_size = 0;
    cr_assert_neq(oc_vision_cfg_validate(&cfg), OC_OK);
}

Test(vcfg, validate_non_divisible)
{
    OcVisionConfig cfg;
    oc_vision_cfg_init(&cfg);
    cfg.image_size = 100;
    cfg.patch_size = 16;
    cr_assert_neq(oc_vision_cfg_validate(&cfg), OC_OK);
}

Test(vcfg, validate_bad_heads)
{
    OcVisionConfig cfg;
    oc_vision_cfg_init(&cfg);
    cfg.n_heads = 7;
    cfg.hidden_dim = 768;
    cr_assert_neq(oc_vision_cfg_validate(&cfg), OC_OK);
}

Test(vcfg, n_patches)
{
    OcVisionConfig cfg;
    oc_vision_cfg_init(&cfg);
    cr_assert_eq(oc_vision_cfg_n_patches(&cfg), 196); /* (224/16)^2 */
}

Test(vcfg, n_patches_total)
{
    OcVisionConfig cfg;
    oc_vision_cfg_init(&cfg);
    cr_assert_eq(oc_vision_cfg_n_patches_total(&cfg), 197); /* +CLS */
}

Test(vcfg, n_patches_clip_large)
{
    OcVisionConfig cfg;
    oc_vision_cfg_clip_large(&cfg);
    cr_assert_eq(oc_vision_cfg_n_patches(&cfg), 576); /* (336/14)^2 */
}

Test(vcfg, patch_dim)
{
    OcVisionConfig cfg;
    oc_vision_cfg_init(&cfg);
    cr_assert_eq(oc_vision_cfg_patch_dim(&cfg), 768); /* 16*16*3 */
}

Test(vcfg, model_type_name)
{
    cr_assert_str_eq(oc_vision_model_type_name(OC_VISION_MODEL_CLIP), "clip");
    cr_assert_str_eq(oc_vision_model_type_name(OC_VISION_MODEL_SIGLIP), "siglip");
    cr_assert_str_eq(oc_vision_model_type_name(OC_VISION_MODEL_DINO), "dino");
    cr_assert_str_eq(oc_vision_model_type_name(OC_VISION_MODEL_INTERN_VIT), "intern_vit");
}

Test(vcfg, is_valid)
{
    OcVisionConfig cfg;
    oc_vision_cfg_init(&cfg);
    cr_assert(oc_vision_cfg_is_valid(&cfg));
}

Test(vcfg, is_valid_false)
{
    OcVisionConfig cfg = {0};
    cr_assert(!oc_vision_cfg_is_valid(&cfg));
}
