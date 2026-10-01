/* Compare naive GEMV (oxidize-cpp-train style) vs CBLAS SGEMM.
 * Prints ms and GFLOP/s for Y[T,rows] = X[T,cols] * W^T.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cblas.h>

/* atoi reports neither malformed input nor overflow: a zero iters count would
 * divide by zero in time_ms and a negative dimension is undefined in cblas. */
static int parse_dim(const char *s, const char *what) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno == ERANGE || !end || end == s || *end || v < 1 || v > 1 << 20) {
        fprintf(stderr, "bad %s: %s (want 1..%d)\n", what, s, 1 << 20);
        exit(1);
    }
    return (int)v;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float *malloc_f(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(float)) != 0 || !p) {
        fprintf(stderr, "oom\n");
        exit(1);
    }
    memset(p, 0, n * sizeof(float));
    return (float *)p;
}

static void fill(float *a, size_t n, unsigned seed) {
    unsigned x = seed ? seed : 1u;
    for (size_t i = 0; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        a[i] = ((int)(x >> 9) / 8388608.0f) - 1.0f;
    }
}

static void gemv_naive(float *Y, const float *W, const float *X, int T, int rows, int cols) {
    for (int t = 0; t < T; t++) {
        for (int r = 0; r < rows; r++) {
            const float *w = W + (size_t)r * (size_t)cols;
            const float *x = X + (size_t)t * (size_t)cols;
            float s = 0.0f;
            for (int c = 0; c < cols; c++) s += w[c] * x[c];
            Y[(size_t)t * (size_t)rows + (size_t)r] = s;
        }
    }
}

static double time_ms(void (*fn)(void *), void *ctx, int iters) {
    fn(ctx);
    double t0 = now();
    for (int i = 0; i < iters; i++) fn(ctx);
    return (now() - t0) * 1e3 / iters;
}

struct SgemmCtx {
    float *Y, *W, *X;
    int T, rows, cols;
};

static void run_naive(void *p) {
    struct SgemmCtx *c = p;
    gemv_naive(c->Y, c->W, c->X, c->T, c->rows, c->cols);
}

static void run_sgemm(void *p) {
    struct SgemmCtx *c = p;
    /* Y = X * W^T : Y[T,rows], X[T,cols], W[rows,cols] row-major
     * cblas: C = A * B^T with A=X (T x cols), B=W (rows x cols)
     */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                c->T, c->rows, c->cols,
                1.0f, c->X, c->cols, c->W, c->cols,
                0.0f, c->Y, c->rows);
}

int main(int argc, char **argv) {
    int T = 256, rows = 1024, cols = 384;
    int iters = 8;
    if (argc >= 2 && argc < 4) {
        fprintf(stderr, "usage: %s [T rows cols [iters]]\n", argv[0]);
        return 1;
    }
    if (argc >= 4) {
        T = parse_dim(argv[1], "T");
        rows = parse_dim(argv[2], "rows");
        cols = parse_dim(argv[3], "cols");
    }
    if (argc >= 5) iters = parse_dim(argv[4], "iters");

    struct SgemmCtx c = {
        .Y = malloc_f((size_t)T * (size_t)rows),
        .W = malloc_f((size_t)rows * (size_t)cols),
        .X = malloc_f((size_t)T * (size_t)cols),
        .T = T,
        .rows = rows,
        .cols = cols,
    };
    fill(c.W, (size_t)rows * (size_t)cols, 1);
    fill(c.X, (size_t)T * (size_t)cols, 2);

    double flops = 2.0 * (double)T * (double)rows * (double)cols;
    printf("shape T=%d rows=%d cols=%d iters=%d  (Y = X * W^T)\n", T, rows, cols, iters);

    double ms_naive = time_ms(run_naive, &c, iters < 3 ? iters : 3);
    printf("naive_gemv  %8.2f ms  %6.1f GFLOP/s\n", ms_naive, flops / (ms_naive * 1e6));

    double ms_blas = time_ms(run_sgemm, &c, iters);
    printf("cblas_sgemm %8.2f ms  %6.1f GFLOP/s\n", ms_blas, flops / (ms_blas * 1e6));
    printf("speedup     %8.2fx\n", ms_naive / ms_blas);

    /* residual check on a tiny slice */
    float *Y2 = malloc_f((size_t)T * (size_t)rows);
    gemv_naive(Y2, c.W, c.X, c.T, c.rows, c.cols);
    run_sgemm(&c);
    float max_abs = 0.0f;
    for (size_t i = 0; i < (size_t)T * (size_t)rows; i++) {
        float d = c.Y[i] - Y2[i];
        if (d < 0) d = -d;
        if (d > max_abs) max_abs = d;
    }
    printf("max_abs_diff %.6g\n", max_abs);
    if (max_abs > 1e-2f) {
        fprintf(stderr, "FAIL numeric\n");
        return 2;
    }
    free(Y2);
    free(c.Y);
    free(c.W);
    free(c.X);
    return 0;
}
