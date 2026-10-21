#include <criterion/criterion.h>
#include "oxidize/eagle4.h"

#include <stdio.h>
#include <string.h>

static OcEagle4Config tiny_cfg(void)
{
    OcEagle4Config c;
    oc_eagle4_config_init(&c);
    c.hidden_size = 4;
    c.target_hidden_size = 4;
    c.n_aux = 1;
    c.extract_layers[0] = 0;
    c.vocab_size = 5;
    c.n_heads = 1;
    c.n_kv_heads = 1;
    c.head_dim = 4;
    c.intermediate_size = 4;
    c.sliding_window = 2;
    c.lm_rank = 2;
    c.n_candidates = 3;
    c.use_rope = 0;
    return c;
}

Test(eagle4, zero_weights_draft_token_zero)
{
    OcEagle4Config cfg = tiny_cfg();
    OcEagle4Model m;
    cr_assert_eq(oc_eagle4_init(&m, &cfg), OC_OK);
    float aux[4] = {1, 0, 0, 0};
    uint32_t tok = 99;
    float logit = 0;
    cr_assert_eq(oc_eagle4_step(&m, 1, aux, NULL, &tok, &logit), OC_OK);
    cr_assert_eq(tok, 0);
    uint32_t chain[2];
    cr_assert_eq(oc_eagle4_draft(&m, 1, aux, 2, chain), OC_OK);
    cr_assert_eq(chain[0], 0);
    cr_assert_eq(chain[1], 0);
    oc_eagle4_free(&m);
}

Test(eagle4, sliding_window_keeps_two)
{
    OcEagle4Config cfg = tiny_cfg();
    OcEagle4Model m;
    cr_assert_eq(oc_eagle4_init(&m, &cfg), OC_OK);
    float aux[4] = {0.2f, -0.1f, 0.3f, 0.4f};
    cr_assert_eq(oc_eagle4_step(&m, 0, aux, NULL, NULL, NULL), OC_OK);
    cr_assert_eq(oc_eagle4_step(&m, 1, NULL, NULL, NULL, NULL), OC_OK);
    cr_assert_eq(oc_eagle4_step(&m, 2, NULL, NULL, NULL, NULL), OC_OK);
    cr_assert_eq(m.cache_len, 3);
    cr_assert(m.cfg.sliding_window == 2);
    oc_eagle4_free(&m);
}

Test(eagle4, capture_roundtrip)
{
    uint32_t layers[2] = {1, 3};
    OcEagleFeatureCapture *cap = oc_eagle4_capture_new(layers, 2, 2, 4);
    cr_assert_not_null(cap);
    float h0[2] = {1, 2};
    float h1[2] = {3, 4};
    oc_eagle4_capture_at(cap, 1, 0, 7, h0, 2);
    oc_eagle4_capture_at(cap, 3, 0, 7, h1, 2);
    const char *path = "/tmp/eagle4-cap-test.bin";
    cr_assert_eq(oc_eagle4_capture_save(cap, path), OC_OK);
    FILE *f = fopen(path, "rb");
    cr_assert_not_null(f);
    char magic[4];
    cr_assert_eq(fread(magic, 1, 4, f), 4);
    cr_assert_eq(memcmp(magic, "E4H1", 4), 0);
    uint32_t hdr[3];
    cr_assert_eq(fread(hdr, sizeof hdr, 1, f), 1);
    cr_assert_eq(hdr[0], 1);
    cr_assert_eq(hdr[1], 2);
    cr_assert_eq(hdr[2], 2);
    fclose(f);
    remove(path);
    float cat[4];
    cr_assert_eq(oc_eagle4_capture_concat(cap, 0, cat, 4), OC_OK);
    cr_assert_float_eq(cat[0], 1, 1e-6);
    cr_assert_float_eq(cat[3], 4, 1e-6);
    oc_eagle4_capture_free(cap);
}

Test(eagle4, rejects_bad_config)
{
    OcEagle4Config cfg = tiny_cfg();
    cfg.n_candidates = 99;
    OcEagle4Model m;
    cr_assert_neq(oc_eagle4_init(&m, &cfg), OC_OK);
    cr_assert_neq(oc_eagle4_step(NULL, 0, NULL, NULL, NULL, NULL), OC_OK);
}
