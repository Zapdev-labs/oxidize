/* Self-check: fused integer/f16 row dots must agree with dequant-then-f32-dot
 * within int8-activation-quantization tolerance, and the GGUF fixture parses.
 * Run via `make test`. */
#include "oc.h"
#include "batch.h"

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

static void check_gemma4_projection_metadata(void) {
  const size_t hidden = 5376, q_dim = 16384, intermediate = 21504;
  oc_weight q = {.rows = q_dim, .cols = hidden};
  oc_weight wo = {.rows = q_dim, .cols = hidden};
  oc_weight down = {.rows = intermediate, .cols = hidden};
  oc_weight canonical_wo = {.rows = hidden, .cols = q_dim};

  oc_gemma4_canonicalize_projection_metadata(&q, hidden, false);
  assert(q.rows == q_dim && q.cols == hidden);
  oc_gemma4_canonicalize_projection_metadata(&wo, hidden, true);
  assert(wo.rows == hidden && wo.cols == q_dim);
  oc_gemma4_canonicalize_projection_metadata(&down, hidden, true);
  assert(down.rows == hidden && down.cols == intermediate);
  oc_gemma4_canonicalize_projection_metadata(&canonical_wo, hidden, true);
  assert(canonical_wo.rows == hidden && canonical_wo.cols == q_dim);
  printf("ok gemma4 projection metadata canonicalization\n");
}

/* reference: dequant row and f32-dot against raw x (not the q8 blocks), so the
 * comparison bounds the activation-quantization error too */
static void check_quant(oc_quant q, size_t cols) {
  size_t rb = oc_row_bytes(q, cols);
  uint8_t *row = malloc(rb);
  for (size_t i = 0; i < rb; ++i) row[i] = (uint8_t)(rnd() >> 13);
  if (q == OC_F16 || q == OC_BF16 || q == OC_F32 || q == OC_Q8_0 || q == OC_AL8) {
    float *tmp = malloc(cols * sizeof(float));
    for (size_t i = 0; i < cols; ++i)
      tmp[i] = ((float)(rnd() & 0xFFFF) / 65536.0f - 0.5f) * 2.0f;
    assert(oc_quantize_row(q, tmp, row, cols));
    free(tmp);
  }
  /* overwrite f16 scale fields with sane small values to avoid inf/nan */
  size_t bb = oc_block_bytes(q), nv = oc_block_values(q);
  if (q != OC_F32 && q != OC_F16 && q != OC_BF16) {
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

/* ---- prune + finetune primitives ---- */

static void check_iq_reference_dequant(void) {
  /* mirrors oxidize-core quantization/tests.rs iq4_nl + iq2/iq3 smoke cases */
  uint8_t nl[18] = {0};
  nl[1] = 0x3c; /* f16 1.0 */
  nl[2] = 0x10; /* nibbles 0 and 1 */
  float out_nl[32];
  oc_dequant_row(OC_IQ4_NL, nl, out_nl, 32);
  if (out_nl[0] != -127.0f || out_nl[16] != -104.0f) {
    fprintf(stderr, "FAIL IQ4_NL ref: got %f %f\n", out_nl[0], out_nl[16]);
    exit(1);
  }
  printf("ok IQ4_NL reference dequant\n");

  struct { oc_quant q; size_t bs; size_t n; } cases[] = {
    {OC_IQ2_XXS, 66, QK_K}, {OC_IQ2_XS, 74, QK_K}, {OC_IQ2_S, 82, QK_K},
    {OC_IQ3_XXS, 98, QK_K}, {OC_IQ3_S, 110, QK_K},
  };
  for (size_t k = 0; k < sizeof(cases) / sizeof(*cases); ++k) {
    uint8_t *blk = calloc(1, cases[k].bs);
    blk[1] = 0x3c;
    for (size_t i = 2; i < cases[k].bs; ++i) blk[i] = (uint8_t)(i % 251);
    float *out = malloc(cases[k].n * sizeof(float));
    oc_dequant_row(cases[k].q, blk, out, cases[k].n);
    for (size_t i = 0; i < cases[k].n; ++i) {
      if (!isfinite(out[i])) {
        fprintf(stderr, "FAIL %s ref: non-finite at %zu\n",
                oc_quant_name(cases[k].q), i);
        exit(1);
      }
    }
    printf("ok %-7s reference dequant (finite)\n", oc_quant_name(cases[k].q));
    free(blk);
    free(out);
  }

  struct { uint32_t ggml; oc_quant q; } map[] = {
    {16, OC_IQ2_XXS}, {17, OC_IQ2_XS}, {18, OC_IQ3_XXS}, {20, OC_IQ4_NL},
    {21, OC_IQ3_S}, {22, OC_IQ2_S}, {23, OC_IQ4_XS},
  };
  for (size_t k = 0; k < sizeof(map) / sizeof(*map); ++k) {
    if (oc_from_ggml_type(map[k].ggml) != map[k].q ||
        oc_to_ggml_type(map[k].q) != map[k].ggml) {
      fprintf(stderr, "FAIL ggml map type %u <-> %s\n", map[k].ggml,
              oc_quant_name(map[k].q));
      exit(1);
    }
  }
  printf("ok IQ ggml type round-trip\n");
}

static void check_prune_masks(void) {
  /* magnitude keeps top half per row */
  float w[16];
  for (int i = 0; i < 16; ++i) w[i] = (float)i;
  bool mask[16];
  oc_magnitude_mask(w, 2, 8, 0.5f, mask);
  for (int r = 0; r < 2; ++r) {
    int kept = 0;
    for (int c = 0; c < 8; ++c) kept += mask[r * 8 + c];
    assert(kept == 4);
  }
  for (int c = 0; c < 4; ++c) assert(!mask[c]);
  for (int c = 4; c < 8; ++c) assert(mask[c]);

  /* wanda prefers high activation columns */
  float w2[6] = {10, 10, 10, 1, 1, 1};
  float norms[6] = {0, 0, 0, 10, 10, 10};
  bool m2[6];
  oc_wanda_mask(w2, norms, 1, 6, 0.5f, m2);
  for (int c = 0; c < 3; ++c) assert(!m2[c]);
  for (int c = 3; c < 6; ++c) assert(m2[c]);

  /* N:M caps kept per block: ascending scores keep the top n of each m */
  float s[8];
  for (int i = 0; i < 8; ++i) s[i] = (float)(i + 1);
  bool m3[8];
  memset(m3, 1, 8);
  oc_nm_mask(s, 1, 8, 2, 4, m3);
  assert(!m3[0] && !m3[1] && m3[2] && m3[3]);
  assert(!m3[4] && !m3[5] && m3[6] && m3[7]);
  printf("ok prune masks (magnitude, wanda, 2:4)\n");
}

static void check_quantize_roundtrip(void) {
  enum { N = 512 };
  float x[N], dq[N];
  for (int i = 0; i < N; ++i)
    x[i] = ((float)(rnd() & 0xFFFF) / 65536.0f - 0.5f) * 2.0f;
  uint8_t buf[N * 4];
  struct { oc_quant q; float tol; } cases[] = {
      {OC_F32, 0.0f},    {OC_F16, 0.001f},  {OC_BF16, 0.01f},
      {OC_Q8_0, 0.01f},  {OC_Q4_0, 0.14f},  {OC_Q4_1, 0.08f},
      {OC_Q5_0, 0.07f},  {OC_Q5_1, 0.04f},  {OC_Q2_K, 0.45f},
      {OC_Q3_K, 0.20f},  {OC_Q4_K, 0.09f},  {OC_Q5_K, 0.05f},
      {OC_Q6_K, 0.04f},  {OC_IQ4_XS, 0.15f}, {OC_AL5, 0.14f},
  };
  for (size_t k = 0; k < sizeof(cases) / sizeof(*cases); ++k) {
    assert(oc_quantize_row(cases[k].q, x, buf, N));
    oc_dequant_row(cases[k].q, buf, dq, N);
    float worst = 0;
    for (int i = 0; i < N; ++i) {
      float e = fabsf(dq[i] - x[i]);
      if (e > worst) worst = e;
    }
    if (worst > cases[k].tol) {
      fprintf(stderr, "FAIL roundtrip %s: worst err %f > tol %f\n",
              oc_quant_name(cases[k].q), worst, cases[k].tol);
      exit(1);
    }
    printf("ok requant %-6s worst=%.4f\n", oc_quant_name(cases[k].q), worst);
  }

  /* AL5 must beat Q4_0 in RMSE at identical 18-byte size. */
  float rmse[2] = {0, 0};
  oc_quant q40 = OC_Q4_0, al5 = OC_AL5;
  for (int p = 0; p < 2; ++p) {
    oc_quant q = p ? al5 : q40;
    oc_quantize_row(q, x, buf, N);
    oc_dequant_row(q, buf, dq, N);
    double se = 0;
    for (int i = 0; i < N; ++i) se += (double)(dq[i] - x[i]) * (dq[i] - x[i]);
    rmse[p] = (float)sqrt(se / N);
  }
  printf("ok AL5 rmse=%.5f vs Q4_0 rmse=%.5f (%.1f%% lower)\n", rmse[1],
         rmse[0], 100.0f * (rmse[0] - rmse[1]) / rmse[0]);
  assert(rmse[1] < rmse[0]);
}

static void check_lora_gradients(void) {
  /* finite-difference check: loss = sum(out), grad_out = 1 */
  size_t in = 4, out = 3, rank = 2;
  oc_lora *l = oc_lora_new(in, out, rank, 4.0f, 7);
  for (size_t i = 0; i < out * rank; ++i)
    l->b[i] = ((float)i - 2.5f) * 0.05f;
  float x[4] = {0.3f, -0.7f, 1.1f, 0.05f};
  float g[3] = {1.0f, 1.0f, 1.0f};
  oc_lora_backward(l, x, g, 1);

  float eps = 1e-3f;
  for (size_t idx = 0; idx < rank * in; ++idx) {
    float orig = l->a[idx];
    float o1[3] = {0}, o2[3] = {0};
    l->a[idx] = orig + eps;
    oc_lora_forward(l, x, o1, 1);
    l->a[idx] = orig - eps;
    oc_lora_forward(l, x, o2, 1);
    l->a[idx] = orig;
    float fd = ((o1[0] + o1[1] + o1[2]) - (o2[0] + o2[1] + o2[2])) / (2 * eps);
    assert(fabsf(fd - l->grad_a[idx]) < 1e-2f);
  }
  for (size_t idx = 0; idx < out * rank; ++idx) {
    float orig = l->b[idx];
    float o1[3] = {0}, o2[3] = {0};
    l->b[idx] = orig + eps;
    oc_lora_forward(l, x, o1, 1);
    l->b[idx] = orig - eps;
    oc_lora_forward(l, x, o2, 1);
    l->b[idx] = orig;
    float fd = ((o1[0] + o1[1] + o1[2]) - (o2[0] + o2[1] + o2[2])) / (2 * eps);
    assert(fabsf(fd - l->grad_b[idx]) < 1e-2f);
  }
  oc_lora_free(l);
  printf("ok lora gradients (finite differences)\n");
}

static void check_ce_grad(void) {
  size_t vocab = 7, count = 4;
  float logits[28];
  for (size_t i = 0; i < 28; ++i) logits[i] = sinf((float)i * 0.31f);
  uint32_t targets[4] = {0, 3, 6, 2};
  size_t n = 0;
  float loss = oc_ce_grad(logits, targets, count, vocab, 1.0f, &n);
  assert(n == count);
  assert(loss > 0);
  for (size_t t = 0; t < count; ++t) { /* grad rows sum to ~0 */
    float s = 0;
    for (size_t i = 0; i < vocab; ++i) s += logits[t * vocab + i];
    assert(fabsf(s) < 1e-4f);
  }
  printf("ok cross-entropy grad (rows sum to 0)\n");
}

/* end-to-end: build a tiny valid GGUF, magnitude-prune it, reload + verify */
static void check_prune_e2e(void) {
  const char *in = "/tmp/oc_test_in.gguf", *out = "/tmp/oc_test_out.gguf";
  size_t rows = 8, cols = 64;
  FILE *f = fopen(in, "wb");
  assert(f);
  fwrite("GGUF", 1, 4, f);
  uint32_t v32 = 3;
  fwrite(&v32, 4, 1, f);
  uint64_t v64 = 1; fwrite(&v64, 8, 1, f);      /* n_tensors */
  v64 = 1; fwrite(&v64, 8, 1, f);               /* n_kv */
  const char *key = "general.alignment";
  v64 = strlen(key); fwrite(&v64, 8, 1, f); fwrite(key, 1, strlen(key), f);
  v32 = 4; fwrite(&v32, 4, 1, f);               /* u32 */
  v32 = 32; fwrite(&v32, 4, 1, f);
  const char *tname = "blk.0.attn_q.weight";
  v64 = strlen(tname); fwrite(&v64, 8, 1, f); fwrite(tname, 1, strlen(tname), f);
  v32 = 2; fwrite(&v32, 4, 1, f);               /* n_dims */
  v64 = cols; fwrite(&v64, 8, 1, f);
  v64 = rows; fwrite(&v64, 8, 1, f);
  v32 = 0; fwrite(&v32, 4, 1, f);               /* F32 */
  v64 = 0; fwrite(&v64, 8, 1, f);               /* offset */
  long pos = ftell(f);
  while (pos % 32) { fputc(0, f); pos++; }      /* align data */
  for (size_t i = 0; i < rows * cols; ++i) {
    float x = (float)(i % cols) + 1.0f;         /* row-ascending magnitudes */
    fwrite(&x, 4, 1, f);
  }
  fclose(f);

  char *args[] = {"--input", (char *)in, "--output", (char *)out,
                  "--method", "magnitude", "--sparsity", "0.5",
                  "--keep-name", "no-such-substring"};
  assert(oc_prune_main(10, args) == 0);

  oc_gguf *g = oc_gguf_load(out);
  const oc_tensor_info *ti = oc_find_tensor(g, tname);
  assert(ti && ti->quant == OC_F32 && ti->dims[0] == cols && ti->dims[1] == rows);
  const float *w = (const float *)(g->base + ti->offset);
  for (size_t r = 0; r < rows; ++r) {
    size_t zeros = 0;
    for (size_t c = 0; c < cols; ++c) zeros += w[r * cols + c] == 0.0f;
    assert(zeros == cols / 2);
    for (size_t c = 0; c < cols / 2; ++c) assert(w[r * cols + c] == 0.0f);
    for (size_t c = cols / 2; c < cols; ++c)
      assert(w[r * cols + c] == (float)c + 1.0f);
  }
  oc_gguf_free(g);
  remove(in);
  remove(out);
  printf("ok prune end-to-end (magnitude 0.5, gguf write/reload)\n");
}

/* int8 KV attention must track the f32 reference within int8 tolerance. */
static void check_kv_int8(void) {
  size_t n_heads = 8, kv_heads = 2, hd = 64, seq = 40;
  size_t kv_len = kv_heads * hd;
  float *q = malloc(n_heads * hd * sizeof(float));
  float *kf = malloc(seq * kv_len * sizeof(float));
  float *vf = malloc(seq * kv_len * sizeof(float));
  int8_t *k8 = malloc(seq * kv_len), *v8 = malloc(seq * kv_len);
  float *ks = malloc(seq * kv_heads * sizeof(float));
  float *vs = malloc(seq * kv_heads * sizeof(float));
  float *o_ref = malloc(n_heads * hd * sizeof(float));
  float *o_q8 = malloc(n_heads * hd * sizeof(float));
  for (size_t i = 0; i < n_heads * hd; ++i)
    q[i] = ((float)(rnd() >> 8) / 8388608.0f) - 1.0f;
  for (size_t t = 0; t < seq; ++t) {
    for (size_t i = 0; i < kv_len; ++i) {
      kf[t * kv_len + i] = ((float)(rnd() >> 8) / 8388608.0f) - 1.0f;
      vf[t * kv_len + i] = ((float)(rnd() >> 8) / 8388608.0f) - 1.0f;
    }
    oc_quantize_kv(kf + t * kv_len, k8 + t * kv_len, ks + t * kv_heads, kv_heads, hd);
    oc_quantize_kv(vf + t * kv_len, v8 + t * kv_len, vs + t * kv_heads, kv_heads, hd);
  }
  oc_attention(o_ref, q, kf, vf, seq, n_heads, kv_heads, hd, 0.0f);
  oc_attention_q8(o_q8, q, k8, ks, v8, vs, seq, n_heads, kv_heads, hd, 0.0f);
  float worst = 0.0f;
  for (size_t i = 0; i < n_heads * hd; ++i) {
    float e = fabsf(o_ref[i] - o_q8[i]);
    if (e > worst) worst = e;
  }
  if (worst > 0.02f) {
    fprintf(stderr, "FAIL kv-int8: worst diff %f\n", worst);
    exit(1);
  }
  printf("ok kv-int8 attention worst=%.5f\n", worst);
  free(q); free(kf); free(vf); free(k8); free(v8);
  free(ks); free(vs); free(o_ref); free(o_q8);
}

static void check_batch_queue_capacity_and_slot_reclaim(void) {
  /* Given: two slots and a FIFO that can retain two pending tokens. */
  const int gpu_ids[] = {0, 1};
  const oc_batch_config config = {
      .capacity = 2, .queue_capacity = 2, .gpu_ids = gpu_ids, .gpu_count = 2};
  oc_batch_scheduler *scheduler = oc_batch_scheduler_new(&config);
  size_t first_slot = 0, second_slot = 0, reclaimed_slot = 0;
  uint64_t sequence_id = 0;
  uint32_t token = 0;
  assert(scheduler);

  /* When: capacity is filled, one sequence is completed, then another arrives. */
  assert(oc_batch_scheduler_admit(scheduler, 11, &first_slot));
  assert(oc_batch_scheduler_admit(scheduler, 22, &second_slot));
  assert(!oc_batch_scheduler_admit(scheduler, 33, &reclaimed_slot));
  assert(oc_batch_scheduler_enqueue_token(scheduler, first_slot, 101));
  assert(oc_batch_scheduler_enqueue_token(scheduler, second_slot, 202));
  assert(oc_batch_scheduler_next(scheduler, &sequence_id, &token));
  assert(sequence_id == 11 && token == 101);
  assert(oc_batch_scheduler_complete(scheduler, first_slot));
  assert(oc_batch_scheduler_admit(scheduler, 33, &reclaimed_slot));

  /* Then: rejection does not corrupt FIFO work and the freed slot is reused. */
  assert(reclaimed_slot == first_slot);
  assert(oc_batch_scheduler_next(scheduler, &sequence_id, &token));
  assert(sequence_id == 22 && token == 202);
  assert(!oc_batch_scheduler_next(scheduler, &sequence_id, &token));
  oc_batch_scheduler_free(scheduler);
  printf("ok batch queue capacity and slot reclaim\n");
}

static void check_single_sequence_scheduler_order(void) {
  /* Given: one sequence in a single-slot scheduler. */
  const int gpu_ids[] = {0};
  const oc_batch_config config = {
      .capacity = 1, .queue_capacity = 3, .gpu_ids = gpu_ids, .gpu_count = 1};
  oc_batch_scheduler *scheduler = oc_batch_scheduler_new(&config);
  size_t slot = 0;
  uint64_t sequence_id = 0;
  uint32_t token = 0;
  assert(scheduler);
  assert(oc_batch_scheduler_admit(scheduler, 77, &slot));

  /* When: token work is added in generation order. */
  assert(oc_batch_scheduler_enqueue_token(scheduler, slot, 7));
  assert(oc_batch_scheduler_enqueue_token(scheduler, slot, 8));
  assert(oc_batch_scheduler_enqueue_token(scheduler, slot, 9));

  /* Then: the scheduler emits the sequence's tokens in FIFO order. */
  assert(oc_batch_scheduler_next(scheduler, &sequence_id, &token));
  assert(sequence_id == 77 && token == 7);
  assert(oc_batch_scheduler_next(scheduler, &sequence_id, &token));
  assert(sequence_id == 77 && token == 8);
  assert(oc_batch_scheduler_next(scheduler, &sequence_id, &token));
  assert(sequence_id == 77 && token == 9);
  assert(!oc_batch_scheduler_next(scheduler, &sequence_id, &token));
  oc_batch_scheduler_free(scheduler);
  printf("ok batch single-sequence FIFO order\n");
}

static void check_worker_configuration_validation(void) {
  /* Given: configurations at each scheduler setup boundary. */
  const int duplicate_gpu_ids[] = {0, 0};
  const int unique_gpu_ids[] = {0, 1};
  const oc_batch_config duplicate = {
      .capacity = 2, .queue_capacity = 2, .gpu_ids = duplicate_gpu_ids, .gpu_count = 2};
  const oc_batch_config zero_capacity = {
      .capacity = 0, .queue_capacity = 1, .gpu_ids = unique_gpu_ids, .gpu_count = 2};
  const oc_batch_config short_queue = {
      .capacity = 2, .queue_capacity = 1, .gpu_ids = unique_gpu_ids, .gpu_count = 2};
  const oc_batch_config valid = {
      .capacity = 2, .queue_capacity = 2, .gpu_ids = unique_gpu_ids, .gpu_count = 2};

  /* When: each configuration is validated before worker startup. */
  const bool duplicate_ok = oc_batch_config_valid(&duplicate);
  const bool zero_capacity_ok = oc_batch_config_valid(&zero_capacity);
  const bool short_queue_ok = oc_batch_config_valid(&short_queue);
  const bool valid_ok = oc_batch_config_valid(&valid);

  /* Then: malformed worker topology and impossible queue sizing are rejected. */
  assert(!duplicate_ok);
  assert(!zero_capacity_ok);
  assert(!short_queue_ok);
  assert(valid_ok);
  printf("ok batch worker configuration validation\n");
}

static void check_sequence_scheduler_mixed_prefill_decode(void) {
  /* Given: one sequence ready to decode and another with prompt work remaining. */
  const int gpu_ids[] = {0};
  const uint32_t decode_prompt[] = {41};
  const uint32_t prefill_prompt[] = {51, 52};
  const oc_batch_config config = {
      .capacity = 2, .queue_capacity = 2, .gpu_ids = gpu_ids, .gpu_count = 1};
  const oc_sequence_request decode_request = {
      .request_id = 1001, .prompt_tokens = decode_prompt, .prompt_count = 1,
      .max_tokens = 4, .rng_seed = 7};
  const oc_sequence_request prefill_request = {
      .request_id = 1002, .prompt_tokens = prefill_prompt, .prompt_count = 2,
      .max_tokens = 4, .rng_seed = 9};
  oc_batch_scheduler *scheduler = oc_batch_scheduler_new(&config);
  oc_batch_lane lanes[2];
  size_t decode_slot = 0, prefill_slot = 0, lane_count = 0;
  assert(scheduler);
  assert(oc_batch_scheduler_admit_sequence(scheduler, &decode_request, &decode_slot));
  assert(oc_batch_scheduler_admit_sequence(scheduler, &prefill_request, &prefill_slot));

  /* When: the final prompt token is processed, then both sequences are scheduled. */
  assert(oc_batch_scheduler_next_lanes(scheduler, lanes, 2, &lane_count));
  assert(lane_count == 2);
  assert(lanes[0].slot == decode_slot);
  assert(lanes[0].phase == OC_SEQUENCE_PREFILL);
  assert(lanes[0].position == 0 && lanes[0].token == 41);
  assert(oc_batch_scheduler_finish_lane(scheduler, &lanes[1]));
  assert(oc_batch_scheduler_finish_lane(scheduler, &lanes[0]));

  assert(oc_batch_scheduler_next_lanes(scheduler, lanes, 2, &lane_count));

  /* Then: one lane is decode and one is prefill, each retaining its own position. */
  assert(lane_count == 2);
  assert(lanes[0].slot == decode_slot);
  assert(lanes[0].phase == OC_SEQUENCE_DECODE);
  assert(lanes[0].position == 1 && lanes[0].token == 41);
  assert(lanes[1].slot == prefill_slot);
  assert(lanes[1].phase == OC_SEQUENCE_PREFILL);
  assert(lanes[1].position == 1 && lanes[1].token == 52);
  oc_batch_scheduler_free(scheduler);
  printf("ok sequence scheduler mixed prefill and decode\n");
}

static void check_sequence_scheduler_final_prompt_output_contract(void) {
  /* Given: a two-token prompt whose final token must sample its first completion. */
  const int gpu_ids[] = {0};
  const uint32_t prompt[] = {91, 92};
  const oc_batch_config config = {
      .capacity = 1, .queue_capacity = 3, .gpu_ids = gpu_ids, .gpu_count = 1};
  const oc_sequence_request request = {
      .request_id = 4001, .prompt_tokens = prompt, .prompt_count = 2,
      .max_tokens = 3, .rng_seed = 17};
  oc_batch_scheduler *scheduler = oc_batch_scheduler_new(&config);
  oc_batch_lane lane;
  size_t slot = 0, lane_count = 0;
  uint64_t request_id = 0;
  uint32_t token = 0;
  assert(scheduler);
  assert(oc_batch_scheduler_admit_sequence(scheduler, &request, &slot));

  /* When: the non-final and final prompt tokens are dispatched. */
  assert(oc_batch_scheduler_next_lanes(scheduler, &lane, 1, &lane_count));
  assert(lane_count == 1 && lane.phase == OC_SEQUENCE_PREFILL);
  assert(lane.token == 91 && lane.position == 0 && !lane.produces_output);
  assert(!oc_batch_scheduler_record_output(scheduler, &lane, 999, false));
  assert(oc_batch_scheduler_finish_lane(scheduler, &lane));

  assert(oc_batch_scheduler_next_lanes(scheduler, &lane, 1, &lane_count));
  assert(lane_count == 1 && lane.phase == OC_SEQUENCE_PREFILL);
  assert(lane.token == 92 && lane.position == 1 && lane.produces_output);
  assert(oc_batch_scheduler_record_output(scheduler, &lane, 93, false));

  /* Then: exactly the sampled completion is visible; no prompt token leaks as output. */
  assert(oc_batch_scheduler_output_next(scheduler, &request_id, &token));
  assert(request_id == 4001 && token == 93);
  assert(!oc_batch_scheduler_output_next(scheduler, &request_id, &token));

  assert(oc_batch_scheduler_next_lanes(scheduler, &lane, 1, &lane_count));
  assert(lane_count == 1 && lane.phase == OC_SEQUENCE_DECODE);
  assert(lane.token == 93 && lane.position == 2 && lane.produces_output);
  assert(oc_batch_scheduler_record_output(scheduler, &lane, 94, false));
  assert(oc_batch_scheduler_output_next(scheduler, &request_id, &token));
  assert(request_id == 4001 && token == 94);
  assert(!oc_batch_scheduler_output_next(scheduler, &request_id, &token));
  oc_batch_scheduler_free(scheduler);
  printf("ok sequence scheduler final prompt output contract\n");
}

static void check_sequence_scheduler_eog_is_terminal_not_output(void) {
  /* Given: a final prompt lane whose sampled token is an end-of-generation marker. */
  const int gpu_ids[] = {0};
  const uint32_t prompt[] = {95};
  const oc_batch_config config = {
      .capacity = 1, .queue_capacity = 2, .gpu_ids = gpu_ids, .gpu_count = 1};
  const oc_sequence_request request = {
      .request_id = 4002, .prompt_tokens = prompt, .prompt_count = 1,
      .max_tokens = 3, .rng_seed = 18};
  oc_batch_scheduler *scheduler = oc_batch_scheduler_new(&config);
  oc_batch_lane lane;
  size_t slot = 0, lane_count = 0;
  uint64_t request_id = 0;
  uint32_t token = 0;
  assert(scheduler);
  assert(oc_batch_scheduler_admit_sequence(scheduler, &request, &slot));
  assert(oc_batch_scheduler_next_lanes(scheduler, &lane, 1, &lane_count));
  assert(lane_count == 1 && lane.produces_output);

  /* When: the final-prompt forward samples EOG. */
  assert(oc_batch_scheduler_record_output(scheduler, &lane, 96, true));

  /* Then: it terminates and reclaims without exposing EOG as completion text. */
  assert(!oc_batch_scheduler_output_next(scheduler, &request_id, &token));
  assert(oc_batch_scheduler_reclaim(scheduler, slot));
  oc_batch_scheduler_free(scheduler);
  printf("ok sequence scheduler EOG terminal exclusion\n");
}

static void check_sequence_scheduler_mixed_output_lanes(void) {
  /* Given: one sequence decoding while another dispatches its final prompt token. */
  const int gpu_ids[] = {0};
  const uint32_t one_token_prompt[] = {101};
  const uint32_t two_token_prompt[] = {201, 202};
  const oc_batch_config config = {
      .capacity = 2, .queue_capacity = 3, .gpu_ids = gpu_ids, .gpu_count = 1};
  const oc_sequence_request decode_request = {
      .request_id = 4101, .prompt_tokens = one_token_prompt, .prompt_count = 1,
      .max_tokens = 3, .rng_seed = 19};
  const oc_sequence_request prefill_request = {
      .request_id = 4102, .prompt_tokens = two_token_prompt, .prompt_count = 2,
      .max_tokens = 3, .rng_seed = 23};
  oc_batch_scheduler *scheduler = oc_batch_scheduler_new(&config);
  oc_batch_lane lanes[2];
  size_t decode_slot = 0, prefill_slot = 0, lane_count = 0;
  assert(scheduler);
  assert(oc_batch_scheduler_admit_sequence(scheduler, &decode_request, &decode_slot));
  assert(oc_batch_scheduler_admit_sequence(scheduler, &prefill_request, &prefill_slot));

  /* When: the first round samples request 4101 and forwards request 4102's first prompt token. */
  assert(oc_batch_scheduler_next_lanes(scheduler, lanes, 2, &lane_count));
  assert(lane_count == 2);
  assert(lanes[0].slot == decode_slot && lanes[0].phase == OC_SEQUENCE_PREFILL &&
         lanes[0].produces_output);
  assert(lanes[1].slot == prefill_slot && lanes[1].phase == OC_SEQUENCE_PREFILL &&
         !lanes[1].produces_output);
  assert(oc_batch_scheduler_record_output(scheduler, &lanes[0], 102, false));
  assert(oc_batch_scheduler_finish_lane(scheduler, &lanes[1]));

  assert(oc_batch_scheduler_next_lanes(scheduler, lanes, 2, &lane_count));

  /* Then: decode and final-prefill lanes are both response-producing, but distinguishable. */
  assert(lane_count == 2);
  assert(lanes[0].slot == decode_slot && lanes[0].phase == OC_SEQUENCE_DECODE &&
         lanes[0].token == 102 && lanes[0].produces_output);
  assert(lanes[1].slot == prefill_slot && lanes[1].phase == OC_SEQUENCE_PREFILL &&
         lanes[1].token == 202 && lanes[1].produces_output);
  oc_batch_scheduler_free(scheduler);
  printf("ok sequence scheduler mixed response-producing lanes\n");
}

static void check_sequence_scheduler_cancel_final_prompt_slot_reuse(void) {
  /* Given: a final-prompt lane whose model result has not returned yet. */
  const int gpu_ids[] = {0};
  const uint32_t prompt[] = {301};
  const oc_batch_config config = {
      .capacity = 1, .queue_capacity = 2, .gpu_ids = gpu_ids, .gpu_count = 1};
  const oc_sequence_request request = {
      .request_id = 4201, .prompt_tokens = prompt, .prompt_count = 1,
      .max_tokens = 3, .rng_seed = 29};
  const oc_sequence_request replacement = {
      .request_id = 4202, .prompt_tokens = prompt, .prompt_count = 1,
      .max_tokens = 3, .rng_seed = 31};
  oc_batch_scheduler *scheduler = oc_batch_scheduler_new(&config);
  oc_batch_lane lane;
  size_t slot = 0, replacement_slot = 0, lane_count = 0;
  uint64_t request_id = 0;
  uint32_t token = 0;
  assert(scheduler);
  assert(oc_batch_scheduler_admit_sequence(scheduler, &request, &slot));
  assert(oc_batch_scheduler_next_lanes(scheduler, &lane, 1, &lane_count));
  assert(lane_count == 1 && lane.phase == OC_SEQUENCE_PREFILL && lane.produces_output);

  /* When: cancellation races the returned final-prompt result and the slot is reused. */
  assert(oc_batch_scheduler_cancel(scheduler, slot));
  assert(oc_batch_scheduler_reclaim(scheduler, slot));
  assert(oc_batch_scheduler_admit_sequence(scheduler, &replacement, &replacement_slot));

  /* Then: the stale response cannot leak into the replacement request. */
  assert(replacement_slot == slot);
  assert(!oc_batch_scheduler_record_output(scheduler, &lane, 302, false));
  assert(!oc_batch_scheduler_output_next(scheduler, &request_id, &token));
  oc_batch_scheduler_free(scheduler);
  printf("ok sequence scheduler cancel final prompt slot reuse\n");
}

static void check_sequence_scheduler_cancel_and_slot_reuse(void) {
  /* Given: a sequence with a buffered generated token in a single-slot scheduler. */
  const int gpu_ids[] = {0};
  const uint32_t prompt[] = {71};
  const oc_batch_config config = {
      .capacity = 1, .queue_capacity = 2, .gpu_ids = gpu_ids, .gpu_count = 1};
  const oc_sequence_request request = {
      .request_id = 2001, .prompt_tokens = prompt, .prompt_count = 1,
      .max_tokens = 4, .rng_seed = 11};
  const oc_sequence_request replacement = {
      .request_id = 2002, .prompt_tokens = prompt, .prompt_count = 1,
      .max_tokens = 4, .rng_seed = 12};
  oc_batch_scheduler *scheduler = oc_batch_scheduler_new(&config);
  oc_batch_lane lane;
  size_t slot = 0, replacement_slot = 0, lane_count = 0;
  uint64_t request_id = 0;
  uint32_t token = 0;
  assert(scheduler);
  assert(oc_batch_scheduler_admit_sequence(scheduler, &request, &slot));
  assert(oc_batch_scheduler_next_lanes(scheduler, &lane, 1, &lane_count));
  assert(lane_count == 1);
  assert(oc_batch_scheduler_record_output(scheduler, &lane, 72, false));

  /* When: the request is cancelled before the client drains its output. */
  assert(oc_batch_scheduler_cancel(scheduler, slot));
  assert(!oc_batch_scheduler_output_next(scheduler, &request_id, &token));
  assert(oc_batch_scheduler_reclaim(scheduler, slot));
  assert(oc_batch_scheduler_admit_sequence(scheduler, &replacement, &replacement_slot));

  /* Then: cancellation emits nothing later and permits safe slot reuse. */
  assert(replacement_slot == slot);
  assert(!oc_batch_scheduler_record_output(scheduler, &lane, 73, false));
  oc_batch_scheduler_free(scheduler);
  printf("ok sequence scheduler cancel and slot reuse\n");
}

static void check_sequence_scheduler_rng_isolation(void) {
  /* Given: matching sequence seeds in two schedulers with different lane interleavings. */
  const int gpu_ids[] = {0};
  const uint32_t prompt[] = {81};
  const oc_batch_config config = {
      .capacity = 2, .queue_capacity = 2, .gpu_ids = gpu_ids, .gpu_count = 1};
  const oc_sequence_request request = {
      .request_id = 3001, .prompt_tokens = prompt, .prompt_count = 1,
      .max_tokens = 4, .rng_seed = 0x12345678u};
  const oc_sequence_request interleaved = {
      .request_id = 3002, .prompt_tokens = prompt, .prompt_count = 1,
      .max_tokens = 4, .rng_seed = 0x87654321u};
  oc_batch_scheduler *serial = oc_batch_scheduler_new(&config);
  oc_batch_scheduler *mixed = oc_batch_scheduler_new(&config);
  oc_batch_lane lanes[2];
  size_t serial_slot = 0, mixed_slot = 0, other_slot = 0, lane_count = 0;
  uint64_t serial_values[3], mixed_values[3];
  assert(serial && mixed);
  assert(oc_batch_scheduler_admit_sequence(serial, &request, &serial_slot));
  assert(oc_batch_scheduler_admit_sequence(mixed, &request, &mixed_slot));
  assert(oc_batch_scheduler_admit_sequence(mixed, &interleaved, &other_slot));

  /* When: random draws for the matching request are separated by other lane work. */
  for (size_t i = 0; i < 3; ++i) {
    assert(oc_batch_scheduler_next_random(serial, serial_slot, &serial_values[i]));
    assert(oc_batch_scheduler_next_lanes(mixed, lanes, 2, &lane_count));
    for (size_t lane_index = 0; lane_index < lane_count; ++lane_index)
      assert(oc_batch_scheduler_finish_lane(mixed, &lanes[lane_index]));
    assert(oc_batch_scheduler_next_random(mixed, mixed_slot, &mixed_values[i]));
    uint64_t ignored = 0;
    assert(oc_batch_scheduler_next_random(mixed, other_slot, &ignored));
  }

  /* Then: each request's deterministic stream is unaffected by the interleaving. */
  assert(memcmp(serial_values, mixed_values, sizeof(serial_values)) == 0);
  oc_batch_scheduler_free(mixed);
  oc_batch_scheduler_free(serial);
  printf("ok sequence scheduler RNG isolation\n");
}

static char *read_source_file(const char *path) {
  FILE *file = fopen(path, "rb");
  if (!file) return NULL;
  if (fseek(file, 0, SEEK_END) != 0) {
    fclose(file);
    return NULL;
  }
  long length = ftell(file);
  if (length < 0 || fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return NULL;
  }
  char *text = malloc((size_t)length + 1);
  if (!text || fread(text, 1, (size_t)length, file) != (size_t)length) {
    free(text);
    fclose(file);
    return NULL;
  }
  text[length] = '\0';
  fclose(file);
  return text;
}

static void check_batch_startup_releases_scalar_cuda_before_reconfigure(void) {
  char *server = read_source_file("server.c");
  char *cli = read_source_file("main.c");
  if (!server || !cli) {
    fprintf(stderr, "FAIL batch startup sequencing: cannot read server.c/main.c\n");
    exit(1);
  }

  const char *start = strstr(server, "static batch_runtime *batch_runtime_start");
  const char *validation = start ? strstr(start, "model->cfg.kv_int8") : NULL;
  const char *release = validation ? strstr(validation, "oc_cuda_release(model);") : NULL;
  const char *configure = validation ? strstr(validation, "oc_cuda_configure_batch(model") : NULL;
  const char *worker = configure ? strstr(configure, "pthread_create(&runtime->worker") : NULL;
  const char *preload_reject = strstr(cli, "if (serve && max_seqs > 1 && kv_int8)");
  const char *model_load = strstr(cli, "oc_model_load(model_path, ctx, kv_int8)");
  if (!validation || !release || !configure || !worker ||
      !(validation < release && release < configure && configure < worker) ||
      !preload_reject || !model_load || !(preload_reject < model_load)) {
    fprintf(stderr,
            "FAIL batch startup sequencing: require validate -> release -> configure -> "
            "worker and pre-load kv-int8 rejection\n");
    free(cli);
    free(server);
    exit(1);
  }
  free(cli);
  free(server);
  printf("ok batch startup releases scalar CUDA before reconfigure\n");
}

int main(void) {
  size_t cols = 512;
  oc_quant types[] = {OC_F16, OC_BF16, OC_Q4_0, OC_Q4_1, OC_Q5_0, OC_Q5_1,
                      OC_Q8_0, OC_Q2_K, OC_Q3_K, OC_Q4_K, OC_Q5_K, OC_Q6_K,
                      OC_AL5, OC_IQ4_XS, OC_IQ2_XXS, OC_IQ2_XS, OC_IQ2_S,
                      OC_IQ3_XXS, OC_IQ3_S};
  for (size_t i = 0; i < sizeof(types) / sizeof(*types); ++i)
    check_quant(types[i], cols);
  check_quant(OC_IQ4_NL, 256);

  check_iq_reference_dequant();
  check_gemma4_projection_metadata();

  /* GGUF fixture parse */
  oc_gguf *g = oc_gguf_load(
      "../oxidize-core/tests/fixtures/valid-v3.gguf");
  assert(g->n_tensors > 0 || g->n_meta > 0);
  printf("ok gguf fixture: %zu tensors, %zu meta keys\n", g->n_tensors, g->n_meta);
  oc_gguf_free(g);

  check_prune_masks();
  check_prune_e2e();
  check_quantize_roundtrip();
  check_lora_gradients();
  check_ce_grad();
  check_kv_int8();
  check_batch_queue_capacity_and_slot_reclaim();
  check_single_sequence_scheduler_order();
  check_worker_configuration_validation();
  check_sequence_scheduler_mixed_prefill_decode();
  check_sequence_scheduler_final_prompt_output_contract();
  check_sequence_scheduler_eog_is_terminal_not_output();
  check_sequence_scheduler_mixed_output_lanes();
  check_sequence_scheduler_cancel_final_prompt_slot_reuse();
  check_sequence_scheduler_cancel_and_slot_reuse();
  check_sequence_scheduler_rng_isolation();
  check_batch_startup_releases_scalar_cuda_before_reconfigure();

  printf("all checks passed\n");
  return 0;
}
