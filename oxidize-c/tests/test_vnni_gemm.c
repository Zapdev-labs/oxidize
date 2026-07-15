/* Does oc_matmul (prefill) agree with oc_matvec (decode) on an AVX512-VNNI CPU?
 *
 * oc_matvec takes the int8-activation path when cols%256==0 && rows>=48 &&
 * oc_q8_dot_supported(t) -- true for Q4_K/Q5_K/Q6_K on any VNNI CPU (the Xeon
 * Gold 5220R bench box). oc_matmul (n_tokens >= 8) never does: it dequantizes
 * the weights to f32 and runs an f32 GEMM against the RAW f32 activations.
 *
 * This CPU has no VNNI, so the VNNI kernel cannot be executed here. It CAN be
 * modelled exactly: oc_dot_q4_k_vnni (src/quant_vnni.c:16-51) computes
 *     sum_b d8[b] * ( sum_i (dd*sc)*wq_i*xq_i - min*m*sum_i xq_i )
 *   = dot( dequant(W_row), x_q8 ),  x_q8[i] = d8[i/256] * xq[i]
 * so what oc_matvec RETURNS on a VNNI box is dot(dequant(W_r), x_q8), with x_q8
 * produced by the shipped oc_q8_quantize. Everything below is real product code
 * (oc_matmul, oc_matvec, oc_q8_quantize, oc_dequant_row); only the VNNI
 * kernel's multiply-add ORDER is modelled, which is worth <1ulp.
 *
 * CASE A is bit-for-bit the fixture, shape, X and tolerance of test_matmul
 * (tests/test_quant.c:237-283) -- so a failure here IS that test failing on
 * .132. CASE B uses properly-quantized gaussian weights and activations, to
 * show the divergence is a property of the int8 path, not of the fixture. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/quant.h"
#include "../src/tensor.h"

#define COLS 512 /* == test_quant.c COLS */
#define ROWS 70  /* == test_quant.c MM_ROWS; >= 48 so matvec takes the q8 path */
#define NTOK 8   /* >= OC_GEMM_MIN_TOKENS so matmul takes the GEMM path */
#define RB (COLS / 256 * OC_BLK_Q4_K)

static unsigned rnd_state = 1;
static uint8_t rnd_byte(void) {
  rnd_state = rnd_state * 1103515245u + 12345u;
  return (uint8_t)(rnd_state >> 16);
}
static float rnd_f(void) { return (float)(rnd_byte() - 128) * (1.0f / 128.0f); }

static void put_f16(uint8_t* p, float v) {
  uint16_t h = oc_f32_to_f16(v);
  p[0] = (uint8_t)(h & 0xff);
  p[1] = (uint8_t)(h >> 8);
}

/* verbatim from tests/test_quant.c make_row(), OC_Q4_K branch */
static void make_row_q4k(uint8_t* row, unsigned seed) {
  rnd_state = seed;
  for (size_t i = 0; i < RB; ++i) row[i] = rnd_byte();
  for (size_t b = 0; b < COLS / OC_QK_K; ++b) {
    uint8_t* blk = row + b * OC_BLK_Q4_K;
    put_f16(blk, 0.021f + 0.003f * (float)b);
    put_f16(blk + 2, 0.011f);
  }
}

static float gauss(void) {
  float u = 0;
  for (int i = 0; i < 12; ++i) u += (float)rnd_byte() / 255.0f;
  return u - 6.0f;
}

/* A real Q4_K encode of gaussian weights: per-32 affine fit, 6-bit sub-scales,
 * f16 super-scales. Gives the value distribution a real GGUF has. */
static void encode_row_q4k(uint8_t* row, const float* w) {
  memset(row, 0, RB);
  for (size_t b = 0; b < COLS / OC_QK_K; ++b) {
    uint8_t* blk = row + b * OC_BLK_Q4_K;
    const float* wb = w + b * OC_QK_K;
    float sc[8], mn[8];
    float dmax = 0, mmax = 0;
    for (int g = 0; g < 8; ++g) {
      float lo = wb[g * 32], hi = wb[g * 32];
      for (int i = 1; i < 32; ++i) {
        if (wb[g * 32 + i] < lo) lo = wb[g * 32 + i];
        if (wb[g * 32 + i] > hi) hi = wb[g * 32 + i];
      }
      sc[g] = (hi - lo) / 15.0f;
      mn[g] = -lo;
      if (sc[g] > dmax) dmax = sc[g];
      if (mn[g] > mmax) mmax = mn[g];
    }
    put_f16(blk, dmax / 63.0f);
    put_f16(blk + 2, mmax / 63.0f);
    float d = oc_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
    float mi = oc_f16_to_f32((uint16_t)(blk[2] | (blk[3] << 8)));
    uint8_t s6[8], m6[8];
    for (int g = 0; g < 8; ++g) {
      int a = d > 0 ? (int)lrintf(sc[g] / d) : 0;
      int c = mi > 0 ? (int)lrintf(mn[g] / mi) : 0;
      s6[g] = (uint8_t)(a < 0 ? 0 : a > 63 ? 63 : a);
      m6[g] = (uint8_t)(c < 0 ? 0 : c > 63 ? 63 : c);
    }
    uint8_t* scb = blk + 4;
    for (int g = 0; g < 4; ++g) {
      scb[g] = s6[g];
      scb[g + 4] = m6[g];
    }
    for (int g = 4; g < 8; ++g) {
      scb[g + 4] = (uint8_t)((s6[g] & 0xF) | ((m6[g] & 0xF) << 4));
      scb[g - 4] |= (uint8_t)((uint8_t)(s6[g] >> 4) << 6);
      scb[g] |= (uint8_t)((uint8_t)(m6[g] >> 4) << 6);
    }
    uint8_t* qs = blk + 16;
    for (int g = 0; g < 8; ++g) {
      float ds = d * s6[g], ms = mi * m6[g];
      size_t gp = (size_t)g / 2, half = (size_t)g & 1;
      for (int i = 0; i < 32; ++i) {
        int q = ds > 0 ? (int)lrintf((wb[g * 32 + i] + ms) / ds) : 0;
        if (q < 0) q = 0;
        if (q > 15) q = 15;
        qs[gp * 32 + (size_t)i] |= (uint8_t)(half ? q << 4 : q);
      }
    }
  }
}

static uint8_t W[ROWS * RB];
static float Wf[ROWS * COLS];
static float X[NTOK * COLS], Y[NTOK * ROWS], ref_f32[ROWS];
static float deq[COLS], xq8[COLS];
static int8_t q[COLS];
static float qd[COLS / 256];
static int32_t bsum[COLS / 16];

static int run(const char* name) {
  OcCtx* c = oc_ctx_new();
  oc_matmul(c, Y, OC_Q4_K, W, ROWS, COLS, X, NTOK); /* == prefill */

  int fails = 0;
  double sum_rel = 0;
  float worst = 0, wg = 0, ww = 0;
  size_t wr = 0, wt = 0;

  for (size_t i = 0; i < NTOK; ++i) {
    const float* x = X + i * COLS;
    oc_matvec(c, ref_f32, OC_Q4_K, W, ROWS, COLS, x); /* decode HERE (no VNNI) */
    oc_q8_quantize(x, COLS, q, qd, bsum);
    for (size_t k = 0; k < COLS; ++k) xq8[k] = (float)q[k] * qd[k / 256];

    for (size_t r = 0; r < ROWS; ++r) {
      oc_dequant_row(OC_Q4_K, W + r * RB, deq, COLS);
      float vnni = oc_dot_f32(deq, xq8, COLS); /* == oc_matvec ON A VNNI BOX */
      float got = Y[i * ROWS + r];

      if (fabsf(got - ref_f32[r]) > 1e-4f * (1.0f + fabsf(ref_f32[r]))) {
        fprintf(stderr, "UNEXPECTED: f32 matmul != f32 matvec, tok %zu row %zu\n", i, r);
        exit(2);
      }
      float rel = fabsf(got - vnni) / (1.0f + fabsf(vnni));
      sum_rel += rel;
      if (rel > 1e-4f) ++fails; /* the exact assertion at test_quant.c:270 */
      if (rel > worst) { worst = rel; wg = got; ww = vnni; wr = r; wt = i; }
    }
  }
  printf("--- %s: Q4_K %d rows x %d cols, %d tokens\n", name, ROWS, COLS, NTOK);
  printf("    matmul(prefill) vs matvec-on-VNNI(decode), tol 1e-4 (test_quant.c:270)\n");
  printf("    FAILING elements : %d / %d\n", fails, NTOK * ROWS);
  printf("    mean rel error   : %.3g\n", sum_rel / (NTOK * ROWS));
  printf("    worst  tok %zu row %zu: matmul=%.9g  matvec(VNNI)=%.9g  rel=%.3g\n",
         wt, wr, (double)wg, (double)ww, (double)worst);
  oc_ctx_free(c);
  return fails;
}

int main(void) {
  oc_pool_init(1);
  int fails = 0;

  /* CASE A: byte-identical to test_matmul's fixture, shape, X and tolerance. */
  for (size_t r = 0; r < ROWS; ++r)
    make_row_q4k(W + r * RB, 0xBEEFu + (unsigned)(4 * ROWS + r) * 31u);
  rnd_state = 7u;
  for (size_t i = 0; i < NTOK * COLS; ++i) X[i] = rnd_f();
  fails += run("A  test_matmul's own fixture");

  /* CASE B: gaussian weights, really Q4_K-encoded; gaussian activations. */
  rnd_state = 20260713u;
  for (size_t i = 0; i < ROWS * COLS; ++i) Wf[i] = gauss() * 0.02f;
  for (size_t r = 0; r < ROWS; ++r) encode_row_q4k(W + r * RB, Wf + r * COLS);
  for (size_t i = 0; i < NTOK * COLS; ++i) X[i] = gauss();
  fails += run("B  real Q4_K-encoded gaussian weights");

  oc_pool_free();
  printf(fails ? "\nFAIL: prefill and decode do not agree on an AVX512-VNNI CPU\n"
               : "\nok\n");
  return fails ? 1 : 0;
}
