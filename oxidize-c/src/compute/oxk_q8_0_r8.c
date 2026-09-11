#include "oxidize/oxk_q8_0_r8.h"

#include "oxidize/quant.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define QK   32u
#define Q8B  34u
#define R8B  OC_Q8_0_R8_SB_BYTES

/* Shared shape validation + overflow-checked span computation.
 *
 * Every byte offset either function forms is bounded by the total span
 * `ngroups * gbytes`, so proving that product safe proves every derived
 * offset safe:
 *   - repack/unpack row offsets peak at (nrows-1)*nblock*34 + (nblock-1)*34
 *     + 34 == nrows*nblock*34 == ngroups*gbytes (because 8*34 == 272).
 *   - superblock offsets peak at g*gbytes + (nblock-1)*272 + 272
 *     == (g+1)*gbytes <= ngroups*gbytes.
 *
 * Returns false (caller reports -1) when the shape is invalid or when any of
 * `nblock * 272` or `ngroups * gbytes` would wrap size_t. */
static bool r8_span(size_t nrows, size_t cols,
                    size_t *out_nblock, size_t *out_ngroups, size_t *out_gbytes)
{
    if ((cols % QK) != 0 || (nrows % OC_Q8_0_R8_PACK) != 0) return false;

    const size_t nblock = cols / QK;
    /* One full R8 superblock per (group, block) pair: guard nblock * 272. */
    if (nblock != 0 && nblock > SIZE_MAX / R8B) return false;
    const size_t gbytes = nblock * R8B;

    const size_t ngroups = nrows / OC_Q8_0_R8_PACK;
    /* Guard the complete grouped span ngroups * gbytes. */
    if (ngroups != 0 && gbytes != 0 && gbytes > SIZE_MAX / ngroups)
        return false;

    *out_nblock  = nblock;
    *out_ngroups = ngroups;
    *out_gbytes  = gbytes;
    return true;
}

int oc_q8_0_r8_repack(const uint8_t *q8_0, size_t nrows, size_t cols,
                      uint8_t *dst)
{
    size_t nblock, ngroups, gbytes;
    if (q8_0 == NULL || dst == NULL) return -1;
    if (!r8_span(nrows, cols, &nblock, &ngroups, &gbytes)) return -1;

    for (size_t g = 0; g < ngroups; g++) {
        const uint8_t *x8[OC_Q8_0_R8_PACK];
        uint8_t *y = dst + g * gbytes;
        for (size_t k = 0; k < OC_Q8_0_R8_PACK; k++)
            x8[k] = q8_0 + ((g * OC_Q8_0_R8_PACK + k) * nblock * Q8B);
        for (size_t ib = 0; ib < nblock; ib++) {
            uint8_t *blk = y + ib * R8B;
            /* qs is `int8_t *` over a `uint8_t` buffer: int8_t is a character
             * type, so this neither breaks alignment nor aliasing. The two
             * scale bytes are moved with memcpy instead of a uint16_t cast —
             * `blk` carries no alignment guarantee. */
            int8_t *qs = (int8_t *)(void *)(blk + 16);
            for (size_t k = 0; k < OC_Q8_0_R8_PACK; k++) {
                const uint8_t *src = x8[k] + ib * Q8B;
                memcpy(blk + 2u * k, src, 2);
                const int8_t *sqs = (const int8_t *)(const void *)(src + 2);
                for (size_t l = 0; l < 4; l++) {
                    for (size_t i = 0; i < 4; i++) {
                        qs[32 * l + 4 * k + i] = sqs[i + 4 * l];
                        qs[32 * l + 4 * k + i + 128] = sqs[i + 4 * l + 16];
                    }
                }
            }
        }
    }
    return 0;
}

int oc_q8_0_r8_unpack_to_q8_0_inplace(uint8_t *data, size_t nrows, size_t cols)
{
    size_t nblock, ngroups, gbytes;
    if (data == NULL) return -1;
    if (!r8_span(nrows, cols, &nblock, &ngroups, &gbytes)) return -1;
    /* Nothing to do: malloc(0) may legitimately return NULL, so bail out
     * before allocating rather than reporting a spurious failure. */
    if (gbytes == 0 || ngroups == 0) return 0;

    uint8_t *scratch = (uint8_t *)malloc(gbytes);
    if (scratch == NULL) return -1;
    for (size_t g = 0; g < ngroups; g++) {
        uint8_t *src = data + g * gbytes;
        memcpy(scratch, src, gbytes);
        for (size_t ib = 0; ib < nblock; ib++) {
            const uint8_t *blk = scratch + ib * R8B;
            const int8_t *qs = (const int8_t *)(const void *)(blk + 16);
            for (size_t k = 0; k < OC_Q8_0_R8_PACK; k++) {
                uint8_t *dst = src + k * nblock * Q8B + ib * Q8B;
                memcpy(dst, blk + 2u * k, 2);
                int8_t *dqs = (int8_t *)(void *)(dst + 2);
                for (size_t l = 0; l < 4; l++) {
                    for (size_t i = 0; i < 4; i++) {
                        dqs[i + 4 * l] = qs[32 * l + 4 * k + i];
                        dqs[i + 4 * l + 16] = qs[32 * l + 4 * k + i + 128];
                    }
                }
            }
        }
    }
    free(scratch);
    return 0;
}
