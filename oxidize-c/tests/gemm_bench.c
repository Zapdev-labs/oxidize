/* oc_matmul vs n x oc_matvec at gemma-4-31B's real layer shapes, in RAM.
 *
 * The end-to-end prefill number on a 13 GB model is only honest on a box that
 * can hold it; anywhere else it measures the disk. This measures the thing that
 * actually changed — the weights are synthetic but the shapes, the quant type
 * and the kernels are the real ones.
 *
 * n=1 is the decode path (must not regress). n>1 is prefill: the same FLOPs per
 * token, but the weight matrix is streamed from DRAM once for the whole batch
 * instead of once per token.
 *
 *   ./build/gemm-bench [type] [threads]      type: al5xs (default) | q4k | q6k
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/quant.h"
#include "../src/tensor.h"

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* gemma-4-31B: hidden 5376, ffn 21504, 60 layers. One layer's projections. */
static const struct {
  const char* name;
  size_t rows, cols;
} SHAPES[] = {
    {"attn_q  ", 8192, 5376},  {"attn_k  ", 4096, 5376},
    {"attn_out", 5376, 8192},  {"ffn_gate", 21504, 5376},
    {"ffn_up  ", 21504, 5376}, {"ffn_down", 5376, 21504},
};
#define N_SHAPES (sizeof SHAPES / sizeof SHAPES[0])
#define N_LAYERS 60

int main(int argc, char** argv) {
  const char* tname = argc > 1 ? argv[1] : "al5xs";
  int threads = argc > 2 ? atoi(argv[2]) : 0;
  uint32_t type = strcmp(tname, "q4k") == 0   ? OC_Q4_K
                  : strcmp(tname, "q6k") == 0 ? OC_Q6_K
                                              : OC_AL5_XS;
  const size_t toks[] = {1, 8, 16, 32, 64};
  const size_t n_toks = sizeof toks / sizeof toks[0];

  oc_pool_init(threads);
  OcCtx* c = oc_ctx_new();
  if (!c) return 2;

  size_t max_w = 0, max_x = 0, max_y = 0;
  for (size_t s = 0; s < N_SHAPES; ++s) {
    size_t wb = oc_row_bytes(type, SHAPES[s].cols) * SHAPES[s].rows;
    if (wb > max_w) max_w = wb;
    if (SHAPES[s].cols > max_x) max_x = SHAPES[s].cols;
    if (SHAPES[s].rows > max_y) max_y = SHAPES[s].rows;
  }
  uint8_t* W = malloc(max_w);
  float* X = malloc(max_x * 96 * sizeof(float));
  float* Y = malloc(max_y * 96 * sizeof(float));
  if (!W || !X || !Y) return 2;
  for (size_t i = 0; i < max_w; ++i) W[i] = (uint8_t)(i * 37u + 11u);
  /* sane f16 block scales; random bit patterns are inf/NaN often enough to
   * turn this into a denormal benchmark */
  size_t blk = type == OC_AL5_XS ? OC_BLK_AL5_XS : type == OC_Q4_K ? OC_BLK_Q4_K : OC_BLK_Q6_K;
  size_t soff = type == OC_Q6_K ? 208 : 0;
  for (size_t b = 0; b + blk <= max_w; b += blk) {
    uint16_t h = oc_f32_to_f16(0.02f);
    W[b + soff] = (uint8_t)(h & 0xff);
    W[b + soff + 1] = (uint8_t)(h >> 8);
    if (type == OC_Q4_K) { /* the min is an f16 too */
      uint16_t m = oc_f32_to_f16(0.01f);
      W[b + 2] = (uint8_t)(m & 0xff);
      W[b + 3] = (uint8_t)(m >> 8);
    }
  }
  for (size_t i = 0; i < max_x * 96; ++i) X[i] = (float)(i % 63) * 0.01f - 0.3f;

  printf("gemm-bench: type=%s threads=%d isa=%s  (gemma-4-31B shapes x %d layers)\n",
         tname, oc_pool_size(), oc_isa_active_name(), N_LAYERS);
  printf("%-6s %12s %12s %9s %10s\n", "tokens", "matvec x n", "matmul", "speedup",
         "prefill");
  printf("%-6s %12s %12s %9s %10s\n", "", "s/token", "s/token", "", "tok/s");

  double base = 0;
  for (size_t ti = 0; ti < n_toks; ++ti) {
    size_t n = toks[ti];
    double t_vec = 0, t_mat = 0;

    for (size_t s = 0; s < N_SHAPES; ++s) {
      size_t rows = SHAPES[s].rows, cols = SHAPES[s].cols;
      /* old prefill: one matvec per token */
      double t0 = now_s();
      for (size_t i = 0; i < n; ++i)
        oc_matvec(c, Y + i * rows, type, W, rows, cols, X + i * cols);
      t_vec += now_s() - t0;
      /* new prefill: one matmul for the batch */
      t0 = now_s();
      oc_matmul(c, Y, type, W, rows, cols, X, n);
      t_mat += now_s() - t0;
    }
    /* one layer -> whole model, per token */
    double vec_tok = t_vec * N_LAYERS / (double)n;
    double mat_tok = t_mat * N_LAYERS / (double)n;
    if (ti == 0) base = vec_tok;
    printf("%-6zu %12.4f %12.4f %8.2fx %10.2f\n", n, vec_tok, mat_tok,
           vec_tok / mat_tok, 1.0 / mat_tok);
  }
  printf("\ndecode (n=1) reference: %.4f s/token = %.2f tok/s\n", base, 1.0 / base);

  free(W);
  free(X);
  free(Y);
  oc_ctx_free(c);
  oc_pool_free();
  return 0;
}
