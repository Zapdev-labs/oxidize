#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>

#include <stdio.h>
#include <unistd.h>

#include "oxidize/dflash2.h"

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
