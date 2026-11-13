/*
 * test_oxk_asm.c: hand-written assembly decode kernels vs the scalar
 * reference (src/compute/asm/oxk_x86_64.S).
 *
 * The asm kernels follow the facc contract: integer block dots are exact,
 * block results accumulate in f32 lanes, so they agree with the
 * block-sequential scalar dot to reassociation tolerance rather than bit
 * for bit. Row counts 1..9 cover the 4-row tile plus every tail length;
 * block counts include 0, 1 and odd values.
 *
 * Skipped on hosts where the asm is not installed (non-x86-64, no
 * AVX2/FMA/F16C, or OC_ASM=0).
 */
#include <criterion/criterion.h>

#include "oxidize/oxk.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) && defined(__ELF__) && !defined(OC_NO_ASM)
void oc_asm_dot_rows_q4_0_q8_0_facc(const uint8_t *rows, size_t row_bytes,
                                    size_t n_rows, size_t blocks,
                                    const uint8_t *q8, const int32_t *qsums,
                                    float *out);
void oc_asm_dot_rows_q8_0_q8_0_facc(const uint8_t *rows, size_t row_bytes,
                                    size_t n_rows, size_t blocks,
                                    const uint8_t *q8, float *out);

static uint32_t rng(uint32_t *s)
{
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}

/* Normal positive/negative f16 in roughly [2^-8, 2^1). */
static void put_f16(uint8_t *p, uint32_t *s)
{
    const uint32_t r = rng(s);
    const uint16_t h = (uint16_t)(((r & 1u) << 15) |
                                  ((7u + (r >> 1) % 9u) << 10) |
                                  ((r >> 8) & 0x3FFu));
    memcpy(p, &h, 2);
}

static void fill_q4_0(uint8_t *row, size_t blocks, uint32_t *s)
{
    for (size_t b = 0; b < blocks; b++) {
        put_f16(row + b * 18, s);
        for (size_t i = 0; i < 16; i++)
            row[b * 18 + 2 + i] = (uint8_t)rng(s);
    }
}

/* Q8_0 codes over the full int8 range, including -128. */
static void fill_q8_0(uint8_t *row, size_t blocks, uint32_t *s)
{
    for (size_t b = 0; b < blocks; b++) {
        put_f16(row + b * 34, s);
        for (size_t i = 0; i < 32; i++)
            row[b * 34 + 2 + i] = (uint8_t)rng(s);
    }
}

static void act_sums(const uint8_t *q8, size_t blocks, int32_t *qsums)
{
    for (size_t b = 0; b < blocks; b++) {
        int32_t sum = 0;
        for (size_t i = 0; i < 32; i++)
            sum += (int8_t)q8[b * 34 + 2 + i];
        qsums[b] = sum;
    }
}

static void check_close(float got, float want, size_t r, size_t blocks)
{
    const double tol = 1e-5 * (1.0 + fabs((double)want)) * (1.0 + (double)blocks);
    cr_assert_lt(fabs((double)got - (double)want), tol,
                 "row %zu blocks %zu: asm %.9g != scalar %.9g",
                 r, blocks, (double)got, (double)want);
}

Test(oxk_asm, q4_0_facc_matches_scalar)
{
    if (!oc_oxk_asm_active()) cr_skip_test("asm kernels not active");
    static const size_t block_counts[] = { 0, 1, 2, 3, 7, 28, 37, 152 };
    uint32_t s = 0x5EEDu;
    for (size_t bi = 0; bi < sizeof block_counts / sizeof block_counts[0]; bi++) {
        const size_t blocks = block_counts[bi];
        const size_t rb = blocks * 18;
        for (size_t n_rows = 1; n_rows <= 9; n_rows++) {
            uint8_t *w = malloc(rb * n_rows + 64);
            uint8_t *a = malloc(blocks * 34 + 64);
            int32_t *qs = malloc((blocks + 1) * sizeof *qs);
            float *out = malloc((n_rows + 1) * sizeof *out);
            cr_assert(w && a && qs && out);
            for (size_t r = 0; r < n_rows; r++) fill_q4_0(w + r * rb, blocks, &s);
            fill_q8_0(a, blocks, &s);
            act_sums(a, blocks, qs);
            out[n_rows] = 12345.0f; /* canary: must not be written */
            oc_asm_dot_rows_q4_0_q8_0_facc(w, rb, n_rows, blocks, a, qs, out);
            for (size_t r = 0; r < n_rows; r++)
                check_close(out[r],
                            oc_oxk_dot_q4_0_q8_0_scalar(w + r * rb, blocks, a),
                            r, blocks);
            cr_assert_eq(out[n_rows], 12345.0f, "wrote past n_rows");
            free(w); free(a); free(qs); free(out);
        }
    }
}

Test(oxk_asm, q8_0_facc_matches_scalar)
{
    if (!oc_oxk_asm_active()) cr_skip_test("asm kernels not active");
    static const size_t block_counts[] = { 0, 1, 2, 3, 7, 28, 37, 152 };
    uint32_t s = 0xC0FFEEu;
    for (size_t bi = 0; bi < sizeof block_counts / sizeof block_counts[0]; bi++) {
        const size_t blocks = block_counts[bi];
        const size_t rb = blocks * 34;
        for (size_t n_rows = 1; n_rows <= 9; n_rows++) {
            uint8_t *w = malloc(rb * n_rows + 64);
            uint8_t *a = malloc(blocks * 34 + 64);
            float *out = malloc((n_rows + 1) * sizeof *out);
            cr_assert(w && a && out);
            for (size_t r = 0; r < n_rows; r++) fill_q8_0(w + r * rb, blocks, &s);
            fill_q8_0(a, blocks, &s);
            out[n_rows] = 12345.0f;
            oc_asm_dot_rows_q8_0_q8_0_facc(w, rb, n_rows, blocks, a, out);
            for (size_t r = 0; r < n_rows; r++)
                check_close(out[r],
                            oc_oxk_dot_q8_0_q8_0_scalar(w + r * rb, blocks, a),
                            r, blocks);
            cr_assert_eq(out[n_rows], 12345.0f, "wrote past n_rows");
            free(w); free(a); free(out);
        }
    }
}

/* A row's bits must not depend on whether it was reduced inside a 4-row
 * tile or alone in the tail: the matvec splits rows across threads, so the
 * tile/tail boundary moves with the thread count. */
Test(oxk_asm, tile_and_tail_bit_identical)
{
    if (!oc_oxk_asm_active()) cr_skip_test("asm kernels not active");
    const size_t blocks = 37, n_rows = 11;
    uint32_t s = 0x7117u;
    for (int q8 = 0; q8 < 2; q8++) {
        const size_t rb = blocks * (q8 ? 34 : 18);
        uint8_t *w = malloc(rb * n_rows);
        uint8_t *a = malloc(blocks * 34);
        int32_t *qs = malloc(blocks * sizeof *qs);
        float tiled[11], single;
        cr_assert(w && a && qs);
        for (size_t r = 0; r < n_rows; r++) {
            if (q8) fill_q8_0(w + r * rb, blocks, &s);
            else fill_q4_0(w + r * rb, blocks, &s);
        }
        fill_q8_0(a, blocks, &s);
        act_sums(a, blocks, qs);
        if (q8) oc_asm_dot_rows_q8_0_q8_0_facc(w, rb, n_rows, blocks, a, tiled);
        else oc_asm_dot_rows_q4_0_q8_0_facc(w, rb, n_rows, blocks, a, qs, tiled);
        for (size_t r = 0; r < n_rows; r++) {
            if (q8) oc_asm_dot_rows_q8_0_q8_0_facc(w + r * rb, rb, 1, blocks, a, &single);
            else oc_asm_dot_rows_q4_0_q8_0_facc(w + r * rb, rb, 1, blocks, a, qs, &single);
            uint32_t x, y;
            memcpy(&x, &tiled[r], 4);
            memcpy(&y, &single, 4);
            cr_assert_eq(x, y, "q8=%d row %zu: tiled %a vs single %a", q8, r,
                         (double)tiled[r], (double)single);
        }
        free(w); free(a); free(qs);
    }
}

/* The public facc entry points must route to the asm when it is active and
 * still agree with the C intrinsics kernels. */
Test(oxk_asm, public_facc_matches_c_kernel)
{
    if (!oc_oxk_asm_active()) cr_skip_test("asm kernels not active");
    const size_t blocks = 56, n_rows = 13, rb = blocks * 18;
    uint32_t s = 0xABCDu;
    uint8_t *w = malloc(rb * n_rows);
    uint8_t *a = malloc(blocks * 34);
    int32_t *qs = malloc(blocks * sizeof *qs);
    float *got = malloc(n_rows * sizeof *got);
    float *want = malloc(n_rows * sizeof *want);
    cr_assert(w && a && qs && got && want);
    for (size_t r = 0; r < n_rows; r++) fill_q4_0(w + r * rb, blocks, &s);
    fill_q8_0(a, blocks, &s);
    act_sums(a, blocks, qs);
    oc_oxk_dot_rows_q4_0_q8_0_facc(w, rb, n_rows, blocks, a, qs, got);
    oc_oxk_dot_rows_q4_0_q8_0_facc_avx2(w, rb, n_rows, blocks, a, qs, want);
    for (size_t r = 0; r < n_rows; r++) check_close(got[r], want[r], r, blocks);
    free(w); free(a); free(qs); free(got); free(want);
}
#else
Test(oxk_asm, not_built) { cr_skip_test("asm kernels are x86-64 ELF only"); }
#endif
