/* standalone_q4k_dot_bench.c — single-act Q4_K dot (the tg path):
 * measures GB/s of weight streaming for oc_oxk_dot_q4_k_q8_k.
 * Build: cc -O2 -ffp-contract=off -Iinclude <this> oxk*.c simd*.c log.c
 *        error.c quantization.c -o bench -lm
 */
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
    const size_t rows = argc > 1 ? (size_t)atol(argv[1]) : 9728;
    const size_t cols = argc > 2 ? (size_t)atol(argv[2]) : 2560;
    const size_t reps = argc > 3 ? (size_t)atol(argv[3]) : 5;
    const size_t blocks = cols / OC_OXK_QK_K;
    const size_t row_bytes = blocks * OC_OXK_BLOCK_Q4_K_SIZE;
    const size_t asize = blocks * OC_OXK_BLOCK_Q8_K_SIZE;

    uint8_t *w = malloc(rows * row_bytes);
    uint8_t *act = malloc(asize);
    float *out = malloc(rows * sizeof(float));
    for (size_t r = 0; r < rows; r++) {
        uint8_t *wb = w + r * row_bytes;
        for (size_t i = 4; i < row_bytes; i++) wb[i] = (uint8_t)rnd();
        for (size_t b = 0; b < blocks; b++) {
            wb[b * OC_OXK_BLOCK_Q4_K_SIZE]     = 0x38;
            wb[b * OC_OXK_BLOCK_Q4_K_SIZE + 2] = 0x38;
        }
    }
    for (size_t i = 4; i < asize; i++) act[i] = (uint8_t)rnd();
    { float dq = 0.75f; memcpy(act, &dq, 4); }

    /* warm */
    double t0 = now();
    for (size_t r = 0; r < rows; r++)
        out[r] = oc_oxk_dot_q4_k_q8_k(w + r * row_bytes, blocks, act);
    double dt = now() - t0;

    t0 = now();
    for (size_t t = 0; t < reps; t++)
        for (size_t r = 0; r < rows; r++)
            out[r] = oc_oxk_dot_q4_k_q8_k(w + r * row_bytes, blocks, act);
    dt = now() - t0;

    double bytes = (double)rows * row_bytes * reps;
    double gflop = 2.0 * (double)rows * cols * reps / 1e9;
    printf("tg-dot %zu x %zu: %.3fs  %.1f GB/s  %.1f GFLOPS\n",
           rows, cols, dt, bytes / dt / 1e9, gflop / dt);
    free(w); free(act); free(out);
    return 0;
}
