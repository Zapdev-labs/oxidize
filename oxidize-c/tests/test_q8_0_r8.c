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

/* ─── Golden known-answer fixture ────────────────────────────────────────
 *
 * One superblock: 8 rows x 32 cols, so nblock == 1 and both the row-major
 * Q8_0 form and the Q8_0_R8 form are exactly 8*34 == 272 bytes.
 *
 * Both vectors below were derived BY HAND from the layout documented in
 * include/oxidize/oxk_q8_0_r8.h, not by running the implementation. Index
 * mapping, with `k` the row (0..7), `l` in 0..3, `i` in 0..3, and
 * qs = superblock + 16:
 *
 *   scale bytes:  r8[2*k + 0..1]           = q8_0[34*k + 0..1]
 *   low  16 el.:  qs[32*l + 4*k + i]       = q8_0[34*k + 2 + (4*l + i)]
 *   high 16 el.:  qs[32*l + 4*k + i + 128] = q8_0[34*k + 2 + (4*l + i + 16)]
 *
 * Chosen source values make every permutation error visible:
 *   scale bytes of row k = { 0x10 + k, 0x20 + k }
 *   element j of row k   = (uint8_t)(32*k + j)      — all 256 values distinct
 *
 * Reading the golden R8 vector confirms the mapping directly: the first 16
 * bytes are the eight interleaved scale pairs (10 20 | 11 21 | ... | 17 27),
 * and the quants then appear as 4-element runs that step +0x20 per row
 * (00 01 02 03 | 20 21 22 23 | 40 41 42 43 | ...) — i.e. l-major, then k,
 * then i, with elements 16..31 of every row living in the second 128 bytes.
 */
static const uint8_t k_golden_q8_0[272] = {
    0x10, 0x20, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
    0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
    0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x11, 0x21,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
    0x2C, 0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x12, 0x22, 0x40, 0x41,
    0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D,
    0x4E, 0x4F, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F, 0x13, 0x23, 0x60, 0x61, 0x62, 0x63,
    0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F,
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x7B,
    0x7C, 0x7D, 0x7E, 0x7F, 0x14, 0x24, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85,
    0x86, 0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F, 0x90, 0x91,
    0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D,
    0x9E, 0x9F, 0x15, 0x25, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF, 0xB0, 0xB1, 0xB2, 0xB3,
    0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF,
    0x16, 0x26, 0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
    0xCA, 0xCB, 0xCC, 0xCD, 0xCE, 0xCF, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5,
    0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF, 0x17, 0x27,
    0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xEB,
    0xEC, 0xED, 0xEE, 0xEF, 0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,
    0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF,
};

static const uint8_t k_golden_r8[272] = {
    /* [0,16): interleaved scale pairs, row 0..7. */
    0x10, 0x20, 0x11, 0x21, 0x12, 0x22, 0x13, 0x23, 0x14, 0x24, 0x15, 0x25,
    0x16, 0x26, 0x17, 0x27,
    /* [16,144): elements 0..15 of every row, l-major then k then i. */
    0x00, 0x01, 0x02, 0x03, 0x20, 0x21, 0x22, 0x23,
    0x40, 0x41, 0x42, 0x43, 0x60, 0x61, 0x62, 0x63, 0x80, 0x81, 0x82, 0x83,
    0xA0, 0xA1, 0xA2, 0xA3, 0xC0, 0xC1, 0xC2, 0xC3, 0xE0, 0xE1, 0xE2, 0xE3,
    0x04, 0x05, 0x06, 0x07, 0x24, 0x25, 0x26, 0x27, 0x44, 0x45, 0x46, 0x47,
    0x64, 0x65, 0x66, 0x67, 0x84, 0x85, 0x86, 0x87, 0xA4, 0xA5, 0xA6, 0xA7,
    0xC4, 0xC5, 0xC6, 0xC7, 0xE4, 0xE5, 0xE6, 0xE7, 0x08, 0x09, 0x0A, 0x0B,
    0x28, 0x29, 0x2A, 0x2B, 0x48, 0x49, 0x4A, 0x4B, 0x68, 0x69, 0x6A, 0x6B,
    0x88, 0x89, 0x8A, 0x8B, 0xA8, 0xA9, 0xAA, 0xAB, 0xC8, 0xC9, 0xCA, 0xCB,
    0xE8, 0xE9, 0xEA, 0xEB, 0x0C, 0x0D, 0x0E, 0x0F, 0x2C, 0x2D, 0x2E, 0x2F,
    0x4C, 0x4D, 0x4E, 0x4F, 0x6C, 0x6D, 0x6E, 0x6F, 0x8C, 0x8D, 0x8E, 0x8F,
    0xAC, 0xAD, 0xAE, 0xAF, 0xCC, 0xCD, 0xCE, 0xCF, 0xEC, 0xED, 0xEE, 0xEF,
    /* [144,272): elements 16..31 of every row, same ordering. */
    0x10, 0x11, 0x12, 0x13, 0x30, 0x31, 0x32, 0x33, 0x50, 0x51, 0x52, 0x53,
    0x70, 0x71, 0x72, 0x73, 0x90, 0x91, 0x92, 0x93, 0xB0, 0xB1, 0xB2, 0xB3,
    0xD0, 0xD1, 0xD2, 0xD3, 0xF0, 0xF1, 0xF2, 0xF3, 0x14, 0x15, 0x16, 0x17,
    0x34, 0x35, 0x36, 0x37, 0x54, 0x55, 0x56, 0x57, 0x74, 0x75, 0x76, 0x77,
    0x94, 0x95, 0x96, 0x97, 0xB4, 0xB5, 0xB6, 0xB7, 0xD4, 0xD5, 0xD6, 0xD7,
    0xF4, 0xF5, 0xF6, 0xF7, 0x18, 0x19, 0x1A, 0x1B, 0x38, 0x39, 0x3A, 0x3B,
    0x58, 0x59, 0x5A, 0x5B, 0x78, 0x79, 0x7A, 0x7B, 0x98, 0x99, 0x9A, 0x9B,
    0xB8, 0xB9, 0xBA, 0xBB, 0xD8, 0xD9, 0xDA, 0xDB, 0xF8, 0xF9, 0xFA, 0xFB,
    0x1C, 0x1D, 0x1E, 0x1F, 0x3C, 0x3D, 0x3E, 0x3F, 0x5C, 0x5D, 0x5E, 0x5F,
    0x7C, 0x7D, 0x7E, 0x7F, 0x9C, 0x9D, 0x9E, 0x9F, 0xBC, 0xBD, 0xBE, 0xBF,
    0xDC, 0xDD, 0xDE, 0xDF, 0xFC, 0xFD, 0xFE, 0xFF,
};

Test(q8_0_r8, ggml_id_and_name)
{
    cr_assert_eq(oc_quant_type_from_ggml_id(208), OC_QUANT_Q8_0_R8);
    cr_assert_eq(oc_quant_type_to_ggml_id(OC_QUANT_Q8_0_R8), 208u);
    cr_assert_str_eq(oc_quant_type_name(OC_QUANT_Q8_0_R8), "Q8_0_R8");
}

/* The type is known but NOT runnable: the interleaved bytes are not
 * row-strided and there is no decoder, so every runtime sizing path must
 * refuse it rather than treat it as row-major Q8_0. */
Test(q8_0_r8, rejected_by_runtime_sizing_and_decode)
{
    cr_assert(oc_quant_is_runtime_supported(OC_QUANT_Q8_0) );
    cr_assert_not(oc_quant_is_runtime_supported(OC_QUANT_Q8_0_R8));
    cr_assert_not(oc_quant_is_runtime_supported(OC_QUANT_UNKNOWN));

    OcQuantBlockLayout bl = oc_quant_block_size(OC_QUANT_Q8_0_R8);
    cr_assert_eq(bl.elements_per_block, 0u);
    cr_assert_eq(bl.bytes_per_block, 0u);
    cr_assert_eq(oc_quantized_size(OC_QUANT_Q8_0_R8, 256), 0u);

    float out[32];
    uint8_t buf[34] = {0};
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_Q8_0_R8, buf, sizeof(buf),
                                      out, 32), OC_ERR_QUANT);
    cr_assert_eq(oc_quant_dequant_row_scalar(OC_QUANT_Q8_0_R8, buf,
                                             sizeof(buf), out, 32),
                 OC_ERR_QUANT);
    float fin[32] = {0};
    cr_assert_eq(oc_quant_pack_row(OC_QUANT_Q8_0_R8, fin, 32, buf,
                                   sizeof(buf)), OC_ERR_QUANT);
}

/* Golden: repack must produce EXACTLY the hand-derived interleaved bytes. */
Test(q8_0_r8, repack_matches_golden_bytes)
{
    uint8_t got[272];
    memset(got, 0xAA, sizeof(got));
    cr_assert_eq(oc_q8_0_r8_repack(k_golden_q8_0, 8, 32, got), 0);
    cr_assert_arr_eq(got, k_golden_r8, sizeof(got));
}

/* Golden: unpack of the hand-written interleaved fixture must produce EXACTLY
 * the row-major Q8_0 bytes. Independent of repack. */
Test(q8_0_r8, unpack_matches_golden_bytes)
{
    uint8_t got[272];
    memcpy(got, k_golden_r8, sizeof(got));
    cr_assert_eq(oc_q8_0_r8_unpack_to_q8_0_inplace(got, 8, 32), 0);
    cr_assert_arr_eq(got, k_golden_q8_0, sizeof(got));
}

Test(q8_0_r8, roundtrip_multi_group_multi_block)
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
    cr_assert_eq(oc_q8_0_r8_repack(q8, rows, cols, r8), 0);
    /* The repack must actually move bytes around, otherwise the round trip
     * below would pass for a pair of no-ops. */
    cr_assert_neq(memcmp(r8, q8, rows * row_bytes), 0);
    memcpy(back, r8, rows * row_bytes);
    cr_assert_eq(oc_q8_0_r8_unpack_to_q8_0_inplace(back, rows, cols), 0);
    cr_assert_eq(memcmp(back, q8, rows * row_bytes), 0);

    free(q8); free(r8); free(back); free(src);
}

/* Repack on an unaligned destination: the scale bytes must be moved with
 * memcpy, so an odd offset is legal. */
Test(q8_0_r8, repack_tolerates_unaligned_buffers)
{
    uint8_t src[273];
    uint8_t dst[273];
    memcpy(src + 1, k_golden_q8_0, 272);
    memset(dst, 0xAA, sizeof(dst));
    cr_assert_eq(oc_q8_0_r8_repack(src + 1, 8, 32, dst + 1), 0);
    cr_assert_arr_eq(dst + 1, k_golden_r8, 272);
    cr_assert_eq(oc_q8_0_r8_unpack_to_q8_0_inplace(dst + 1, 8, 32), 0);
    cr_assert_arr_eq(dst + 1, k_golden_q8_0, 272);
}

Test(q8_0_r8, rejects_bad_arguments)
{
    uint8_t buf[272] = {0};
    uint8_t out[272] = {0};

    /* NULL arguments. */
    cr_assert_eq(oc_q8_0_r8_repack(NULL, 8, 32, out), -1);
    cr_assert_eq(oc_q8_0_r8_repack(buf, 8, 32, NULL), -1);
    cr_assert_eq(oc_q8_0_r8_unpack_to_q8_0_inplace(NULL, 8, 32), -1);

    /* cols not a multiple of 32. */
    cr_assert_eq(oc_q8_0_r8_repack(buf, 8, 33, out), -1);
    cr_assert_eq(oc_q8_0_r8_repack(buf, 8, 31, out), -1);
    cr_assert_eq(oc_q8_0_r8_unpack_to_q8_0_inplace(buf, 8, 33), -1);

    /* nrows not a multiple of 8. */
    cr_assert_eq(oc_q8_0_r8_repack(buf, 7, 32, out), -1);
    cr_assert_eq(oc_q8_0_r8_repack(buf, 9, 32, out), -1);
    cr_assert_eq(oc_q8_0_r8_unpack_to_q8_0_inplace(buf, 7, 32), -1);
}

/* cols == 0 and nrows == 0 are valid degenerate shapes with no work to do:
 * they must report success, not the malloc(0)-may-return-NULL failure. */
Test(q8_0_r8, empty_shapes_succeed)
{
    uint8_t buf[8] = {0};
    uint8_t out[8] = {0};

    cr_assert_eq(oc_q8_0_r8_repack(buf, 8, 0, out), 0);
    cr_assert_eq(oc_q8_0_r8_unpack_to_q8_0_inplace(buf, 8, 0), 0);
    cr_assert_eq(oc_q8_0_r8_repack(buf, 0, 32, out), 0);
    cr_assert_eq(oc_q8_0_r8_unpack_to_q8_0_inplace(buf, 0, 32), 0);
    cr_assert_eq(oc_q8_0_r8_repack(buf, 0, 0, out), 0);
    cr_assert_eq(oc_q8_0_r8_unpack_to_q8_0_inplace(buf, 0, 0), 0);
}

/* Dimensions whose derived byte span wraps size_t must be rejected before any
 * allocation or pointer arithmetic. Both buffers stay untouched. */
Test(q8_0_r8, rejects_overflowing_dimensions)
{
    uint8_t buf[8] = {0};
    uint8_t out[8] = {0};

    /* nblock * 272 overflows: cols/32 > SIZE_MAX/272. */
    const size_t huge_cols = (SIZE_MAX / 4u) & ~(size_t)31u;
    cr_assert_eq(oc_q8_0_r8_repack(buf, 8, huge_cols, out), -1);
    cr_assert_eq(oc_q8_0_r8_unpack_to_q8_0_inplace(buf, 8, huge_cols), -1);

    /* ngroups * gbytes overflows: nrows/8 > SIZE_MAX/272 with nblock == 1. */
    const size_t huge_rows = (SIZE_MAX / 16u) & ~(size_t)7u;
    cr_assert_eq(oc_q8_0_r8_repack(buf, huge_rows, 32, out), -1);
    cr_assert_eq(oc_q8_0_r8_unpack_to_q8_0_inplace(buf, huge_rows, 32), -1);
}
