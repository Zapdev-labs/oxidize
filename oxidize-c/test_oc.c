/* Self-check: fused integer/f16 row dots must agree with dequant-then-f32-dot
 * within int8-activation-quantization tolerance, and the GGUF fixture parses.
 * Run via `make test`. */
#include "oc.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rstate = 12345;
static uint32_t rnd(void) {
  rstate = rstate * 1664525u + 1013904223u;
  return rstate;
}

/* reference: dequant row and f32-dot against raw x (not the q8 blocks), so the
 * comparison bounds the activation-quantization error too */
static void check_quant(oc_quant q, size_t cols) {
  size_t rb = oc_row_bytes(q, cols);
  uint8_t *row = malloc(rb);
  for (size_t i = 0; i < rb; ++i) row[i] = (uint8_t)(rnd() >> 13);
  /* overwrite f16 scale fields with sane small values to avoid inf/nan */
  size_t bb = oc_block_bytes(q), nv = oc_block_values(q);
  for (size_t b = 0; b < cols / nv; ++b) {
    uint8_t *blk = row + b * bb;
    uint16_t half = 0x2c00 | (rnd() & 0xFF); /* ~[0.06, 0.12) */
    size_t off = q == OC_Q2_K ? 80 : q == OC_Q3_K ? 108 : q == OC_Q6_K ? 208 : 0;
    memcpy(blk + off, &half, 2);
    if (q == OC_Q4_K || q == OC_Q5_K || q == OC_Q2_K || q == OC_Q4_1 ||
        q == OC_Q5_1) {
      uint16_t mh = 0x2800 | (rnd() & 0xFF);
      memcpy(blk + (q == OC_Q2_K ? 82 : 2), &mh, 2);
    }
  }

  float *x = malloc(cols * sizeof(float));
  for (size_t i = 0; i < cols; ++i)
    x[i] = ((float)(rnd() & 0xFFFF) / 65536.0f - 0.5f) * 2.0f;

  float *dq = malloc(cols * sizeof(float));
  oc_dequant_row(q, row, dq, cols);
  float ref = 0;
  for (size_t i = 0; i < cols; ++i) ref += dq[i] * x[i];

  oc_weight w = {.quantized = true, .quant = q, .data = row, .rows = 1, .cols = cols};
  float got;
  oc_gemv(&w, 1, cols, x, NULL, &got);

  float mag = 0;
  for (size_t i = 0; i < cols; ++i) mag += fabsf(dq[i] * x[i]);
  float tol = 0.02f * (mag > 1.0f ? mag : 1.0f); /* int8 act quant error bound */
  if (fabsf(got - ref) > tol) {
    fprintf(stderr, "FAIL %s: got %f want %f (tol %f)\n", oc_quant_name(q), got,
            ref, tol);
    exit(1);
  }
  printf("ok %-5s fused=%.5f ref=%.5f\n", oc_quant_name(q), got, ref);
  free(row); free(x); free(dq);
}

int main(void) {
  size_t cols = 512;
  oc_quant types[] = {OC_F16, OC_BF16, OC_Q4_0, OC_Q4_1, OC_Q5_0, OC_Q5_1,
                      OC_Q8_0, OC_Q2_K, OC_Q3_K, OC_Q4_K, OC_Q5_K, OC_Q6_K};
  for (size_t i = 0; i < sizeof(types) / sizeof(*types); ++i)
    check_quant(types[i], cols);

  /* GGUF fixture parse */
  oc_gguf *g = oc_gguf_load(
      "../oxidize-core/tests/fixtures/valid-v3.gguf");
  assert(g->n_tensors > 0 || g->n_meta > 0);
  printf("ok gguf fixture: %zu tensors, %zu meta keys\n", g->n_tensors, g->n_meta);
  oc_gguf_free(g);

  printf("all checks passed\n");
  return 0;
}
