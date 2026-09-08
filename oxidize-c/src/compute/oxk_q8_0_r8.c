#include "oxidize/oxk_q8_0_r8.h"

#include "oxidize/quant.h"

#include <stdlib.h>
#include <string.h>

#define QK   32u
#define Q8B  34u
#define R8B  OC_Q8_0_R8_SB_BYTES

void oc_q8_0_r8_repack(const uint8_t *q8_0, size_t nrows, size_t cols,
                       uint8_t *dst)
{
    const size_t nblock = cols / QK;
    if (q8_0 == NULL || dst == NULL || nblock * QK != cols ||
        (nrows % OC_Q8_0_R8_PACK) != 0)
        return;
    for (size_t g = 0; g < nrows / OC_Q8_0_R8_PACK; g++) {
        const uint8_t *x8[8];
        uint8_t *y = dst + g * nblock * R8B;
        for (int k = 0; k < 8; k++)
            x8[k] = q8_0 + ((g * 8u + (size_t)k) * nblock * Q8B);
        for (size_t ib = 0; ib < nblock; ib++) {
            uint8_t *blk = y + ib * R8B;
            uint16_t *d = (uint16_t *)(void *)blk;
            int8_t *qs = (int8_t *)(void *)(blk + 16);
            for (int k = 0; k < 8; k++) {
                const uint8_t *src = x8[k] + ib * Q8B;
                memcpy(&d[k], src, 2);
                const int8_t *sqs = (const int8_t *)(const void *)(src + 2);
                for (int l = 0; l < 4; l++) {
                    for (int i = 0; i < 4; i++) {
                        qs[32 * l + 4 * k + i] = sqs[i + 4 * l];
                        qs[32 * l + 4 * k + i + 128] = sqs[i + 4 * l + 16];
                    }
                }
            }
        }
    }
}

int oc_q8_0_r8_unpack_to_q8_0_inplace(uint8_t *data, size_t nrows, size_t cols)
{
    const size_t nblock = cols / QK;
    if (data == NULL || nblock * QK != cols || (nrows % OC_Q8_0_R8_PACK) != 0)
        return -1;
    const size_t gbytes = nblock * R8B;
    uint8_t *scratch = (uint8_t *)malloc(gbytes);
    if (scratch == NULL) return -1;
    for (size_t g = 0; g < nrows / OC_Q8_0_R8_PACK; g++) {
        uint8_t *src = data + g * gbytes;
        memcpy(scratch, src, gbytes);
        for (size_t ib = 0; ib < nblock; ib++) {
            const uint8_t *blk = scratch + ib * R8B;
            const uint16_t *d = (const uint16_t *)(const void *)blk;
            const int8_t *qs = (const int8_t *)(const void *)(blk + 16);
            for (int k = 0; k < 8; k++) {
                uint8_t *dst = src + (size_t)k * nblock * Q8B + ib * Q8B;
                memcpy(dst, &d[k], 2);
                int8_t *dqs = (int8_t *)(void *)(dst + 2);
                for (int l = 0; l < 4; l++) {
                    for (int i = 0; i < 4; i++) {
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
