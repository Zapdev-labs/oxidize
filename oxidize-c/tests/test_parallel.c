/* test_parallel.c — worker pool + threaded matvec. */
/* raw float bits, not a tolerance, because a tolerance would hide exactly the */
#include <criterion/criterion.h>

#include "oxidize/matvec.h"
#include "oxidize/parallel.h"
#include "oxidize/quant.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Deterministic filler — rand() would make a failure unreproducible. */
static float frand(uint32_t *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (float)((int32_t)(*s >> 8) % 2000 - 1000) / 1000.0f;
}


Test(parallel, defaults_to_inline)
{
    cr_assert_eq(oc_parallel_n_threads(), 1u,
                 "pool must start inline so results are deterministic");
}

Test(parallel, set_threads_reports_count)
{
    cr_assert_eq(oc_parallel_set_threads(4), OC_OK);
    cr_assert_eq(oc_parallel_n_threads(), 4u);
    cr_assert_eq(oc_parallel_set_threads(1), OC_OK);
    cr_assert_eq(oc_parallel_n_threads(), 1u);
}

/* Slices must tile [0, n) exactly once — no gaps (wrong answers) and no
 * overlaps (two threads writing one slot). */
typedef struct { uint8_t *hits; } CoverCtx;

static void cover_fn(size_t begin, size_t end, size_t tid, void *ud)
{
    (void)tid;
    CoverCtx *c = (CoverCtx *)ud;
    for (size_t i = begin; i < end; i++) c->hits[i]++;
}

Test(parallel, slices_tile_the_range_exactly)
{
    const size_t n = 1000;
    uint8_t *hits = calloc(n, 1);
    cr_assert_not_null(hits);

    cr_assert_eq(oc_parallel_set_threads(7), OC_OK);
    CoverCtx ctx = { hits };
    oc_parallel_for(n, cover_fn, &ctx);
    oc_parallel_set_threads(1);

    for (size_t i = 0; i < n; i++) {
        cr_assert_eq(hits[i], 1, "index %zu covered %u times", i, hits[i]);
    }
    free(hits);
}

/* A range smaller than the split threshold still has to run completely. */
Test(parallel, tiny_range_still_runs)
{
    uint8_t hits[3] = {0};
    cr_assert_eq(oc_parallel_set_threads(8), OC_OK);
    CoverCtx ctx = { hits };
    oc_parallel_for(3, cover_fn, &ctx);
    oc_parallel_set_threads(1);
    for (size_t i = 0; i < 3; i++) cr_assert_eq(hits[i], 1);
}

Test(parallel, more_threads_than_items)
{
    uint8_t hits[2] = {0};
    cr_assert_eq(oc_parallel_set_threads(16), OC_OK);
    CoverCtx ctx = { hits };
    oc_parallel_for(2, cover_fn, &ctx);
    oc_parallel_set_threads(1);
    cr_assert_eq(hits[0], 1);
    cr_assert_eq(hits[1], 1);
}


Test(parallel, matvec_f32_bit_identical_across_thread_counts)
{
    const size_t rows = 257;   /* prime-ish, so the remainder path is used */
    const size_t cols = 193;
    float *w = malloc(rows * cols * sizeof(float));
    float *x = malloc(cols * sizeof(float));
    float *serial = malloc(rows * sizeof(float));
    float *par = malloc(rows * sizeof(float));
    cr_assert_not_null(w); cr_assert_not_null(x);
    cr_assert_not_null(serial); cr_assert_not_null(par);

    uint32_t s = 12345;
    for (size_t i = 0; i < rows * cols; i++) w[i] = frand(&s);
    for (size_t i = 0; i < cols; i++) x[i] = frand(&s);

    oc_parallel_set_threads(1);
    oc_matvec_f32(w, rows, cols, x, serial);

    const size_t counts[] = {2, 3, 5, 8, 16};
    for (size_t k = 0; k < sizeof(counts) / sizeof(counts[0]); k++) {
        memset(par, 0, rows * sizeof(float));
        cr_assert_eq(oc_parallel_set_threads(counts[k]), OC_OK);
        oc_matvec_f32(w, rows, cols, x, par);
        oc_parallel_set_threads(1);
        cr_assert_arr_eq(par, serial, rows * sizeof(float),
                         "f32 matvec differs at %zu threads", counts[k]);
    }
    free(w); free(x); free(serial); free(par);
}


/* Q8_0 keeps the fixture simple: 32-value blocks, f16 scale + 32 int8. */
Test(parallel, matvec_quantized_bit_identical_across_thread_counts)
{
    const size_t rows = 131;
    const size_t cols = 256;               /* 8 Q8_0 blocks per row */
    const size_t row_bytes = oc_quantized_size(OC_QUANT_Q8_0, cols);
    cr_assert_gt(row_bytes, 0);

    uint8_t *w = malloc(rows * row_bytes);
    float *x = malloc(cols * sizeof(float));
    float *temp = malloc(cols * sizeof(float));
    float *serial = malloc(rows * sizeof(float));
    float *par = malloc(rows * sizeof(float));
    cr_assert_not_null(w); cr_assert_not_null(x); cr_assert_not_null(temp);
    cr_assert_not_null(serial); cr_assert_not_null(par);

    uint32_t s = 999;
    for (size_t i = 0; i < rows * row_bytes; i++) {
        s = s * 1664525u + 1013904223u;
        w[i] = (uint8_t)(s >> 16);
    }
    for (size_t i = 0; i < cols; i++) x[i] = frand(&s);

    oc_parallel_set_threads(1);
    oc_matvec_quantized(OC_QUANT_Q8_0, w, rows, cols, row_bytes, x,
                        serial, temp);

    const size_t counts[] = {2, 4, 6, 12};
    for (size_t k = 0; k < sizeof(counts) / sizeof(counts[0]); k++) {
        memset(par, 0, rows * sizeof(float));
        cr_assert_eq(oc_parallel_set_threads(counts[k]), OC_OK);
        oc_matvec_quantized(OC_QUANT_Q8_0, w, rows, cols, row_bytes, x,
                            par, temp);
        oc_parallel_set_threads(1);
        cr_assert_arr_eq(par, serial, rows * sizeof(float),
                         "quantized matvec differs at %zu threads", counts[k]);
    }
    free(w); free(x); free(temp); free(serial); free(par);
}

/* Threads above tid 0 use pool scratch instead of the caller's `temp`. If a
 * slice ever shared one buffer, concurrent dequantization would corrupt it;
 * a wide matrix with many threads is the shape that exposes it. */
Test(parallel, quantized_scratch_is_per_thread)
{
    const size_t rows = 512;
    const size_t cols = 512;
    const size_t row_bytes = oc_quantized_size(OC_QUANT_Q8_0, cols);
    uint8_t *w = malloc(rows * row_bytes);
    float *x = malloc(cols * sizeof(float));
    float *temp = malloc(cols * sizeof(float));
    float *serial = malloc(rows * sizeof(float));
    float *par = malloc(rows * sizeof(float));
    cr_assert_not_null(w); cr_assert_not_null(x); cr_assert_not_null(temp);
    cr_assert_not_null(serial); cr_assert_not_null(par);

    uint32_t s = 4242;
    for (size_t i = 0; i < rows * row_bytes; i++) {
        s = s * 1664525u + 1013904223u;
        w[i] = (uint8_t)(s >> 16);
    }
    for (size_t i = 0; i < cols; i++) x[i] = frand(&s);

    oc_parallel_set_threads(1);
    oc_matvec_quantized(OC_QUANT_Q8_0, w, rows, cols, row_bytes, x,
                        serial, temp);

    /* Repeat so a rare interleaving has several chances to show up. */
    for (int rep = 0; rep < 8; rep++) {
        memset(par, 0, rows * sizeof(float));
        cr_assert_eq(oc_parallel_set_threads(16), OC_OK);
        oc_matvec_quantized(OC_QUANT_Q8_0, w, rows, cols, row_bytes, x,
                            par, temp);
        oc_parallel_set_threads(1);
        cr_assert_arr_eq(par, serial, rows * sizeof(float),
                         "scratch appears shared between threads (rep %d)", rep);
    }
    free(w); free(x); free(temp); free(serial); free(par);
}

/* Back-to-back regions are the real dispatch pattern (~200 per token), and
 * exercise the generation-counter handoff far more than one region does. */
Test(parallel, many_sequential_regions)
{
    const size_t n = 64;
    uint8_t *hits = calloc(n, 1);
    cr_assert_not_null(hits);
    cr_assert_eq(oc_parallel_set_threads(6), OC_OK);
    CoverCtx ctx = { hits };
    for (int i = 0; i < 500; i++) oc_parallel_for(n, cover_fn, &ctx);
    oc_parallel_set_threads(1);
    for (size_t i = 0; i < n; i++) {
        cr_assert_eq(hits[i], (uint8_t)244 /* 500 mod 256 */,
                     "index %zu hit %u times across 500 regions", i, hits[i]);
    }
    free(hits);
}
