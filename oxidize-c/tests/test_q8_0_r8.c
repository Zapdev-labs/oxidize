#include <criterion/criterion.h>

#include "oxidize/oxk_q8_0_r8.h"
#include "oxidize/quant.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static float frand(uint32_t *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (float)((int32_t)(*s >> 8) % 2001 - 1000) / 1000.0f;
}

Test(q8_0_r8, ggml_id_and_size)
{
    cr_assert_eq(oc_quant_type_from_ggml_id(208), OC_QUANT_Q8_0_R8);
    cr_assert_eq(oc_quant_type_to_ggml_id(OC_QUANT_Q8_0_R8), 208u);
    cr_assert_str_eq(oc_quant_type_name(OC_QUANT_Q8_0_R8), "Q8_0_R8");
    cr_assert_eq(oc_quantized_size(OC_QUANT_Q8_0_R8, 256),
                 oc_quantized_size(OC_QUANT_Q8_0, 256));
}

Test(q8_0_r8, unpack_matches_row_major_q8_0)
{
    const size_t cols = 256;
    const size_t rows = 16;
    const size_t row_bytes = oc_quantized_size(OC_QUANT_Q8_0, cols);
    uint8_t *q8 = malloc(rows * row_bytes);
    uint8_t *r8 = malloc(rows * row_bytes);
    uint8_t *back = malloc(rows * row_bytes);
    float *src = malloc(cols * sizeof(float));
    cr_assert(q8 && r8 && back && src);

    uint32_t seed = 11;
    for (size_t r = 0; r < rows; r++) {
        for (size_t i = 0; i < cols; i++) src[i] = frand(&seed);
        cr_assert_eq(oc_quant_pack_row(OC_QUANT_Q8_0, src, cols,
                                       q8 + r * row_bytes, row_bytes), OC_OK);
    }
    oc_q8_0_r8_repack(q8, rows, cols, r8);
    memcpy(back, r8, rows * row_bytes);
    cr_assert_eq(oc_q8_0_r8_unpack_to_q8_0_inplace(back, rows, cols), 0);
    cr_assert_eq(memcmp(back, q8, rows * row_bytes), 0);

    free(q8); free(r8); free(back); free(src);
}
