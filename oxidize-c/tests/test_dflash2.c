#include <criterion/criterion.h>

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
