/* standalone_q4k_multi_parity.c — AVX2 Q4_K prepped-multi kernel vs the
 * scalar reference on pseudo-random rows (bit-exactness check, no
 * criterion dependency; built ad hoc with: cc -O2 -Iinclude -mavx2). */
#include "oxidize/oxk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t s = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void)
{
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return (uint32_t)(s >> 32);
}

int main(void)
{
    const size_t block_sizes[] = { 1, 2, 3, 8, 64 };
    for (size_t bi = 0; bi < sizeof block_sizes / sizeof block_sizes[0]; bi++) {
        const size_t blocks = block_sizes[bi];
        const size_t row_bytes = blocks * OC_OXK_BLOCK_Q4_K_SIZE;
        uint8_t *row = malloc(row_bytes);
        void *prep = malloc(oc_oxk_q4_k_prep_bytes(blocks));
        const size_t as = blocks * OC_OXK_BLOCK_Q8_K_SIZE;
        const size_t n_act = 9;
        uint8_t *acts = malloc(n_act * as);

        for (size_t t = 0; t < 32; t++) {
            /* Random codes/scales; f16 d/dmin restricted to small normal
             * values (0x3800..0x3C00) so no NaN/Inf enters the sums. */
            for (size_t b = 0; b < blocks; b++) {
                uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_K_SIZE;
                for (size_t i = 4; i < OC_OXK_BLOCK_Q4_K_SIZE; i++)
                    wb[i] = (uint8_t)rnd();
                wb[0] = (uint8_t)(0x38 + (rnd() % 4)); wb[1] = 0x00;
                wb[2] = (uint8_t)(0x38 + (rnd() % 4)); wb[3] = 0x00;
            }
            for (size_t v = 0; v < n_act; v++) {
                uint8_t *qb = acts + v * as;
                for (size_t b = 0; b < blocks; b++) {
                    uint8_t *bl = qb + b * OC_OXK_BLOCK_Q8_K_SIZE;
                    for (size_t i = 4; i < OC_OXK_BLOCK_Q8_K_SIZE; i++)
                        bl[i] = (uint8_t)rnd();
                    float dq = 0.5f + (float)(rnd() % 100) / 100.0f;
                    memcpy(bl, &dq, 4);
                }
            }
            oc_oxk_q4_k_prep_row(row, blocks, prep);
            float ref[16], got[16];
            for (size_t v = 0; v < n_act; v++)
                ref[v] = oc_oxk_dot_q4_k_prepped(prep, blocks,
                                                 acts + v * as);
            oc_oxk_dot_q4_k_prepped_multi_avx2(prep, blocks, acts, as,
                                               n_act, got);
            for (size_t v = 0; v < n_act; v++) {
                if (memcmp(&ref[v], &got[v], sizeof(float)) != 0) {
                    fprintf(stderr,
                            "MISMATCH blocks=%zu t=%zu v=%zu: %+.9g != %+.9g\n",
                            blocks, t, v, (double)ref[v], (double)got[v]);
                    return 1;
                }
            }
        }
        free(row); free(prep); free(acts);
    }
    printf("q4_k prepped multi AVX2: bit-exact with scalar reference\n");
    return 0;
}
