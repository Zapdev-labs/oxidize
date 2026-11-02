/* standalone_q4k_multi_bench.c — microbench the Q4_K prepped-multi path
 * the way oc_matvec_quantized_batch drives it: rows split across
 * oc_parallel, prep per row, multi against n activations.
 * Build: cc -O2 -ffp-contract=off -Iinclude *.c -o bench -lm -lpthread */
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
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    const long rows_arg = argc > 1 ? atol(argv[1]) : 9728;
    const long cols_arg = argc > 2 ? atol(argv[2]) : 2560;
    const long n_arg    = argc > 3 ? atol(argv[3]) : 96;
    const long threads_arg = argc > 4 ? atol(argv[4]) : 8;
    if (rows_arg <= 0 || cols_arg <= 0 || n_arg <= 0 || threads_arg <= 0 ||
        threads_arg > rows_arg) {
        fprintf(stderr,
                "usage: %s [rows] [cols] [n_act] [threads]\n"
                "rows, cols, n_act, and threads must be positive, "
                "and threads <= rows\n",
                argv[0]);
        return 1;
    }
    const size_t rows = (size_t)rows_arg;
    const size_t cols = (size_t)cols_arg;
    const size_t n    = (size_t)n_arg;
    const size_t threads = (size_t)threads_arg;
    const size_t blocks = cols / OC_OXK_QK_K;
    const size_t ncols = blocks * OC_OXK_QK_K;
    const size_t row_bytes = blocks * OC_OXK_BLOCK_Q4_K_SIZE;
    const size_t as = blocks * OC_OXK_BLOCK_Q8_K_SIZE;

    uint8_t *w = malloc(rows * row_bytes);
    uint8_t *acts = malloc(n * as);
    float *out = malloc(rows * n * sizeof(float));
    void *prep = malloc(oc_oxk_q4_k_prep_bytes(blocks) + 64);
    for (size_t r = 0; r < rows; r++) {
        uint8_t *wb = w + r * row_bytes;
        for (size_t i = 4; i < row_bytes; i++) wb[i] = (uint8_t)rnd();
        for (size_t b = 0; b < blocks; b++) {
            uint8_t *bl = wb + b * OC_OXK_BLOCK_Q4_K_SIZE;
            bl[0] = 0x00; bl[1] = 0x38; /* f16 0.5 */
            bl[2] = 0x00; bl[3] = 0x38;
        }
    }
    for (size_t v = 0; v < n; v++) {
        uint8_t *qb = acts + v * as;
        for (size_t b = 0; b < blocks; b++) {
            uint8_t *bl = qb + b * OC_OXK_BLOCK_Q8_K_SIZE;
            for (size_t i = 4; i < OC_OXK_BLOCK_Q8_K_SIZE; i++)
                bl[i] = (uint8_t)rnd();
            float dq = 0.75f;
            memcpy(bl, &dq, 4);
        }
    }

    /* Single-thread kernel-only measurement over 1/threads of rows: */
    const size_t slice = rows / threads;
    double t0 = now();
    for (size_t r = 0; r < slice; r++) {
        oc_oxk_q4_k_prep_row(w + r * row_bytes, blocks, prep);
        oc_oxk_dot_q4_k_prepped_multi_avx2(prep, blocks, acts, as, n,
                                           out + r * n);
    }
    double dt = now() - t0;
    double gflop = 2.0 * (double)slice * (double)ncols * (double)n / 1e9;
    printf("kernel-only single-thread: %.3fs for %zu rows x %zu cols x %zu acts = %.1f GFLOPS (slice %zu rows)\n",
           dt, slice, ncols, n, gflop / dt, slice);

    /* 8-thread through the real parallel machinery is exercised by the
     * engine; this isolates the kernel itself. */
    free(w); free(acts); free(out); free(prep);
    return 0;
}
