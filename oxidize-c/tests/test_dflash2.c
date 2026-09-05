#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>

#include "oxidize/dflash2.h"

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

Test(dflash2, trim_restores_committed_entry_after_wrap)
{
    /* Given: a full ring followed by one committed row and one speculative
     * row, where the speculative row overwrites still-live history. */
    OcDFlash2KvRing ring;
    cr_assert_eq(oc_dflash2_kvring_init(&ring, 4, 1, 1), OC_OK);
    const float initial[] = { 0.0f, 1.0f, 2.0f, 3.0f };
    cr_assert_eq(oc_dflash2_kvring_append(&ring, initial, initial, 0, 4),
                 OC_OK);
    const float next[] = { 4.0f, 5.0f };
    cr_assert_eq(oc_dflash2_kvring_append(&ring, next, next, 4, 2), OC_OK);

    /* When: the speculative position is rolled back. */
    oc_dflash2_kvring_trim(&ring, 5);

    /* Then: the committed position that shared its slot is restored. */
    cr_assert_eq(ring.total, 5);
    cr_assert_eq(ring.len, 4);
    cr_assert_eq(ring.pos[1], 1);
    cr_assert_float_eq(ring.k[1], 1.0f, 0.0f);
    cr_assert_float_eq(ring.v[1], 1.0f, 0.0f);
    oc_dflash2_kvring_free(&ring);
}

Test(dflash2, trim_restores_entry_when_append_crosses_wrap)
{
    /* Given: a ring with one free slot and a two-row append that crosses
     * the wrap boundary. */
    OcDFlash2KvRing ring;
    cr_assert_eq(oc_dflash2_kvring_init(&ring, 4, 1, 1), OC_OK);
    const float initial[] = { 0.0f, 1.0f, 2.0f };
    cr_assert_eq(oc_dflash2_kvring_append(&ring, initial, initial, 0, 3),
                 OC_OK);
    const float next[] = { 3.0f, 4.0f };
    cr_assert_eq(oc_dflash2_kvring_append(&ring, next, next, 3, 2), OC_OK);

    /* When: the wrapped speculative row is rolled back. */
    oc_dflash2_kvring_trim(&ring, 4);

    /* Then: all four committed rows remain addressable. */
    cr_assert_eq(ring.total, 4);
    cr_assert_eq(ring.len, 4);
    cr_assert_eq(ring.pos[0], 0);
    cr_assert_float_eq(ring.k[0], 0.0f, 0.0f);
    cr_assert_float_eq(ring.v[0], 0.0f, 0.0f);
    oc_dflash2_kvring_free(&ring);
}

Test(dflash2, trim_full_capacity_append_restores_all_history)
{
    /* Given: a full ring whose entire physical storage is overwritten by
     * one speculative append. */
    OcDFlash2KvRing ring;
    cr_assert_eq(oc_dflash2_kvring_init(&ring, 4, 1, 1), OC_OK);
    const float initial[] = { 0.0f, 1.0f, 2.0f, 3.0f };
    const float speculative[] = { 4.0f, 5.0f, 6.0f, 7.0f };
    cr_assert_eq(oc_dflash2_kvring_append(&ring, initial, initial, 0, 4),
                 OC_OK);
    cr_assert_eq(oc_dflash2_kvring_append(&ring, speculative, speculative,
                                          4, 4), OC_OK);

    /* When: every row from the latest append is rolled back. */
    oc_dflash2_kvring_trim(&ring, 4);

    /* Then: the original logical history and payloads are intact. */
    cr_assert_eq(ring.total, 4);
    cr_assert_eq(ring.len, 4);
    for (size_t i = 0; i < 4; i++) {
        cr_assert_eq(ring.pos[i], (int64_t)i);
        cr_assert_float_eq(ring.k[i], initial[i], 0.0f);
        cr_assert_float_eq(ring.v[i], initial[i], 0.0f);
    }
    oc_dflash2_kvring_free(&ring);
}

Test(dflash2, failed_append_preserves_prior_rollback_journal)
{
    /* Given: a wrapped append with a live rollback journal. */
    OcDFlash2KvRing ring;
    cr_assert_eq(oc_dflash2_kvring_init(&ring, 2, 1, 1), OC_OK);
    const float initial[] = { 0.0f, 1.0f };
    const float speculative = 2.0f;
    cr_assert_eq(oc_dflash2_kvring_append(&ring, initial, initial, 0, 2),
                 OC_OK);
    cr_assert_eq(oc_dflash2_kvring_append(&ring, &speculative, &speculative,
                                          2, 1), OC_OK);

    /* When: the next append is rejected before it can mutate the ring. */
    const float invalid[] = { 3.0f, 4.0f, 5.0f };
    cr_assert_eq(oc_dflash2_kvring_append(&ring, invalid, invalid, 3, 3),
                 OC_ERR_INVALID_ARG);
    oc_dflash2_kvring_trim(&ring, 2);

    /* Then: the earlier journal still restores the overwritten row. */
    cr_assert_eq(ring.total, 2);
    cr_assert_eq(ring.pos[0], 0);
    cr_assert_float_eq(ring.k[0], 0.0f, 0.0f);
    cr_assert_float_eq(ring.v[0], 0.0f, 0.0f);
    oc_dflash2_kvring_free(&ring);
}

static void init_scalar_weight(OcDFlash2Weight *weight, float *data,
                               size_t rows)
{
    weight->data = data;
    weight->rows = rows;
    weight->cols = 1;
}

Test(dflash2, later_layer_append_failure_restores_exact_ring_state)
{
    float one = 1.0f;
    float pair[] = { 1.0f, 1.0f };
    OcDFlash2Layer layers[2] = { 0 };
    OcDFlash2KvRing rings[2] = { 0 };
    OcDFlash2Model model = { 0 };
    model.loaded = true;
    model.cfg.hidden_size = 1;
    model.cfg.intermediate_size = 1;
    model.cfg.num_attention_heads = 1;
    model.cfg.num_key_value_heads = 1;
    model.cfg.head_dim = 1;
    model.cfg.conv_kernel_size = 1;
    model.cfg.conv_group_size = 1;
    model.cfg.rms_norm_eps = 1e-5f;
    model.layers = layers;
    model.n_layers = 2;
    model.kv = rings;
    model.norm = &one;
    model.next_noise_pos = 3;

    for (size_t li = 0; li < 2; li++) {
        layers[li].input_layernorm = &one;
        layers[li].post_attention_layernorm = &one;
        layers[li].attn.q_norm = &one;
        layers[li].attn.k_norm = &one;
        init_scalar_weight(&layers[li].attn.q_proj, &one, 1);
        init_scalar_weight(&layers[li].attn.k_proj, &one, 1);
        init_scalar_weight(&layers[li].attn.v_proj, &one, 1);
        init_scalar_weight(&layers[li].attn.o_proj, &one, 1);
        init_scalar_weight(&layers[li].attn_conv.base_kernel, pair, 2);
        init_scalar_weight(&layers[li].attn_conv.kernel_proj, pair, 2);
        init_scalar_weight(&layers[li].mlp_gate, &one, 1);
        init_scalar_weight(&layers[li].mlp_up, &one, 1);
        init_scalar_weight(&layers[li].mlp_down, &one, 1);
        init_scalar_weight(&layers[li].mlp_conv.base_kernel, pair, 2);
        init_scalar_weight(&layers[li].mlp_conv.kernel_proj, pair, 2);
        cr_assert_eq(oc_dflash2_kvring_init(&rings[li], 2, 1, 1), OC_OK);
        const float initial[] = { 0.0f, 1.0f };
        const float prior = 2.0f;
        cr_assert_eq(oc_dflash2_kvring_append(&rings[li], initial, initial,
                                              0, 2), OC_OK);
        cr_assert_eq(oc_dflash2_kvring_append(&rings[li], &prior, &prior,
                                              2, 1), OC_OK);
    }

    float *undo_k[2] = { rings[0].undo_k, rings[1].undo_k };
    float *undo_v[2] = { rings[0].undo_v, rings[1].undo_v };
    int64_t *undo_pos[2] = { rings[0].undo_pos, rings[1].undo_pos };
    const float noise = 0.5f;
    float out = 0.0f;
    oc_dflash2_test_fail_wrapped_append_after(1);

    cr_assert_eq(oc_dflash2_forward_debug(&model, &noise, 1, &out),
                 OC_ERR_OOM);
    for (size_t li = 0; li < 2; li++) {
        cr_assert_eq(rings[li].total, 3);
        cr_assert_eq(rings[li].len, 2);
        cr_assert_eq(rings[li].pos[0], 2);
        cr_assert_eq(rings[li].pos[1], 1);
        cr_assert_float_eq(rings[li].k[1], 1.0f, 0.0f);
        cr_assert_float_eq(rings[li].v[1], 1.0f, 0.0f);
        cr_assert_float_eq(rings[li].k[0], 2.0f, 0.0f);
        cr_assert_float_eq(rings[li].v[0], 2.0f, 0.0f);
        cr_assert_eq(rings[li].undo_k, undo_k[li]);
        cr_assert_eq(rings[li].undo_v, undo_v[li]);
        cr_assert_eq(rings[li].undo_pos, undo_pos[li]);
        cr_assert_eq(rings[li].undo_total, 2);
        cr_assert_eq(rings[li].undo_n, 1);
        oc_dflash2_kvring_free(&rings[li]);
    }
}

Test(dflash2, model_load_rejects_overflowing_config_before_weights)
{
    char path[] = "/tmp/oxidize-dflash2-config-XXXXXX";
    int fd = mkstemp(path);
    cr_assert_geq(fd, 0);
    FILE *config = fdopen(fd, "w");
    cr_assert_not_null(config);
    cr_assert_gt(fputs("{\"hidden_size\":4294967296,"
                       "\"intermediate_size\":4294967296}", config), 0);
    cr_assert_eq(fclose(config), 0);

    OcDFlash2Model model;
    cr_assert_eq(oc_dflash2_model_load(&model, "/does/not/exist", path),
                 OC_ERR_FORMAT);
    cr_assert_eq(unlink(path), 0);
}

Test(dflash2, kvring_init_rejects_overflowing_geometry)
{
    OcDFlash2KvRing ring;
    cr_assert_eq(oc_dflash2_kvring_init(&ring, 8, SIZE_MAX / 8 + 1, 1),
                 OC_ERR_INVALID_ARG);
}

Test(dflash2, model_load_rejects_non_integer_config_before_weights)
{
    char path[] = "/tmp/oxidize-dflash2-config-XXXXXX";
    int fd = mkstemp(path);
    cr_assert_geq(fd, 0);
    FILE *config = fdopen(fd, "w");
    cr_assert_not_null(config);
    cr_assert_gt(fputs("{\"hidden_size\":4096.5}", config), 0);
    cr_assert_eq(fclose(config), 0);

    OcDFlash2Model model;
    cr_assert_eq(oc_dflash2_model_load(&model, "/does/not/exist", path),
                 OC_ERR_FORMAT);
    cr_assert_eq(unlink(path), 0);
}

Test(dflash2, selector_debug_rejects_overflowing_row_extent)
{
    OcDFlash2Model model = { 0 };
    model.loaded = true;
    model.cfg.hidden_size = 1;
    model.cfg.vocab_size = 1;
    model.cfg.selector_top_k = 1;
    model.cfg.selector_rank = 2;
    OcDFlash2Weight lm_head = { 0 };
    float weight = 0.0f;
    lm_head.data = &weight;
    lm_head.rows = 1;
    lm_head.cols = 1;
    float hidden = 0.0f;
    uint32_t anchor = 0;
    uint32_t token = 0;

    cr_assert_eq(oc_dflash2_selector_debug(&model, &hidden, SIZE_MAX,
                                           &anchor, 1, &lm_head, &token,
                                           NULL),
                 OC_ERR_INVALID_ARG);
}
