/* In-memory GGUF v3 builder for the model tests.
 *
 * Lifted verbatim out of tests/test_model.c so the CUDA equivalence gate
 * (tests/cuda_equiv.c) builds its fixture through the SAME writer the CPU
 * suites do — a fixture the CPU test trusts is a fixture the GPU test trusts.
 * Header-only statics: two TUs include it and each uses a subset. */
#ifndef OC_TESTS_GGUF_BUILD_H
#define OC_TESTS_GGUF_BUILD_H
#pragma GCC diagnostic ignored "-Wunused-function"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../src/gguf.h"
#include "../src/quant.h"
#include "tests.h"

typedef struct {
  uint8_t* b;
  size_t n, cap;
} Buf;

static void put(Buf* z, const void* p, size_t n) {
  if (z->n + n > z->cap) {
    z->cap = (z->n + n) * 2 + 4096;
    z->b = realloc(z->b, z->cap);
    CHECK(z->b != NULL);
  }
  memcpy(z->b + z->n, p, n);
  z->n += n;
}
static void put_u32(Buf* z, uint32_t v) { put(z, &v, 4); }
static void put_u64(Buf* z, uint64_t v) { put(z, &v, 8); }
static void put_f32(Buf* z, float v) { put(z, &v, 4); }
static void put_str(Buf* z, const char* s) {
  put_u64(z, strlen(s));
  put(z, s, strlen(s));
}
static void put_pad32(Buf* z) {
  const uint8_t zero = 0;
  while (z->n % 32) put(z, &zero, 1);
}

typedef struct {
  const char* name;
  uint64_t dims[3]; /* 3-D for MoE expert stacks [cols, rows, n_expert] */
  uint32_t n_dims;
  uint32_t type;   /* ggml type id; 0 (OC_F32) for the plain tsr() path */
  size_t n_vals;
  size_t nbytes;   /* serialized byte length when quantized (qbytes != NULL) */
  float* data;     /* f32 payload (OC_F32 path) */
  uint8_t* qbytes; /* quantized payload (tsr_q8 path); NULL for f32 */
} Tsr;

typedef struct {
  Buf kv;
  size_t n_kv;
  Tsr t[128];
  size_t n_t;
} GgufB;

static void kv_u32(GgufB* m, const char* k, uint32_t v) {
  put_str(&m->kv, k);
  put_u32(&m->kv, GGUF_T_U32);
  put_u32(&m->kv, v);
  m->n_kv++;
}
static void kv_f32(GgufB* m, const char* k, float v) {
  put_str(&m->kv, k);
  put_u32(&m->kv, GGUF_T_F32);
  put_f32(&m->kv, v);
  m->n_kv++;
}
static void kv_str(GgufB* m, const char* k, const char* v) {
  put_str(&m->kv, k);
  put_u32(&m->kv, GGUF_T_STRING);
  put_str(&m->kv, v);
  m->n_kv++;
}

static unsigned rs = 12345u;
static float rndf(void) { /* [-1, 1) */
  rs = rs * 1103515245u + 12345u;
  return (float)((int)((rs >> 16) & 0xffff) - 32768) * (1.0f / 32768.0f);
}
static uint8_t rndb(void) {
  rs = rs * 1103515245u + 12345u;
  return (uint8_t)(rs >> 16);
}

/* A tensor of `rows` x `cols` (GGUF dims are [cols, rows]) filled with
 * `centre + spread * U(-1,1)`. 1-D when rows == 0. */
static void tsr(GgufB* m, const char* name, size_t rows, size_t cols, float centre,
                float spread) {
  CHECK(m->n_t < 128);
  Tsr* t = &m->t[m->n_t++];
  t->name = name;
  t->n_dims = rows ? 2 : 1;
  t->dims[0] = cols;
  t->dims[1] = rows;
  t->n_vals = cols * (rows ? rows : 1);
  t->data = malloc(t->n_vals * sizeof(float));
  CHECK(t->data != NULL);
  for (size_t i = 0; i < t->n_vals; ++i) t->data[i] = centre + spread * rndf();
}

/* Quantize `n` (a multiple of 32) f32 values into a Q8_0 byte row: per 32-value
 * block a little-endian f16 scale d = amax/127 followed by 32 int8 codes
 * round(x/d). Exactly ggml's block_q8_0 layout, so oc_dequant_row/dot decode it.
 * A quantized expert stack is what makes a byte-stride bug observable — its row
 * bytes (34/block) differ from both the f32 stride and the value count. */
static void q8_0_row(const float* x, uint8_t* out, size_t n) {
  for (size_t b = 0; b < n / 32; ++b) {
    const float* xb = x + b * 32;
    uint8_t* blk = out + b * OC_BLK_Q8_0;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
      float a = fabsf(xb[i]);
      if (a > amax) amax = a;
    }
    float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
    uint16_t dh = oc_f32_to_f16(d);
    blk[0] = (uint8_t)(dh & 0xff);
    blk[1] = (uint8_t)(dh >> 8);
    for (int i = 0; i < 32; ++i) {
      long q = lroundf(xb[i] * id);
      if (q > 127) q = 127; else if (q < -128) q = -128;
      blk[2 + i] = (uint8_t)(int8_t)q;
    }
  }
}

/* A Q8_0 tensor. n_expert == 0 => 2-D [cols, rows] (a shared/dense expert);
 * else a 3-D stack [cols, rows, n_expert]. Each of the (n_expert?:1)*rows rows
 * is an independent Q8_0 row of `cols` values (cols % 32 == 0), laid out
 * expert-major exactly as ggml stacks them. Distinct random payload per row so a
 * wrong per-expert stride lands on a DIFFERENT expert and the reference diverges. */
static void tsr_q8(GgufB* m, const char* name, size_t n_expert, size_t rows,
                   size_t cols, float spread) {
  CHECK(m->n_t < 128 && cols % 32 == 0);
  Tsr* t = &m->t[m->n_t++];
  size_t ne = n_expert ? n_expert : 1;
  t->name = name;
  t->n_dims = n_expert ? 3 : 2;
  t->dims[0] = cols;
  t->dims[1] = rows;
  t->dims[2] = n_expert; /* 0 when 2-D; build() writes only n_dims dims */
  t->type = OC_Q8_0;
  t->n_vals = cols * rows * ne;
  size_t rb = oc_row_bytes(OC_Q8_0, cols);
  t->nbytes = rows * ne * rb;
  t->qbytes = malloc(t->nbytes);
  CHECK(t->qbytes != NULL);
  float* fr = malloc(cols * sizeof(float));
  CHECK(fr != NULL);
  for (size_t r = 0; r < rows * ne; ++r) {
    for (size_t c = 0; c < cols; ++c) fr[c] = spread * rndf();
    q8_0_row(fr, t->qbytes + r * rb, cols);
  }
  free(fr);
}

/* ---- arbitrary-type weight rows (the CUDA quant-coverage fixtures) ----------
 * F32/F16/Q8_0/Q4_0/AL5_XS go through the real encoders. The K-quants have no
 * encoder in this tree, and writing one just to test a DEQUANT would put the
 * thing under test on both sides of the comparison. Instead: emit a random but
 * STRUCTURALLY VALID block — every bit pattern of a K-quant block decodes, so
 * only the f16 scales need taming (small, so the weights stay O(0.1)). The
 * equivalence gate compares the CPU and GPU decode of the SAME bytes, so the
 * encoder's quality is irrelevant; its layout is not, and there is none here to
 * get wrong. */
static void qrow_any(uint32_t type, const float* x, uint8_t* out, size_t n) {
  switch (type) {
    case OC_F32: memcpy(out, x, n * 4); return;
    case OC_F16:
      for (size_t i = 0; i < n; ++i) {
        uint16_t h = oc_f32_to_f16(x[i]);
        out[2 * i] = (uint8_t)(h & 0xff);
        out[2 * i + 1] = (uint8_t)(h >> 8);
      }
      return;
    case OC_Q8_0: q8_0_row(x, out, n); return;
    case OC_Q4_0: oc_quantize_row_q4_0(x, out, n); return;
    case OC_AL5_XS:
      for (size_t b = 0; b < n / 32; ++b)
        oc_al5xs_encode_block(x + b * 32, out + b * OC_BLK_AL5_XS);
      return;
    default: break;
  }
  size_t blk = oc_row_bytes(type, OC_QK_K); /* K-quants: 256 values/block */
  CHECK(blk != 0 && n % OC_QK_K == 0);
  for (size_t b = 0; b < n / OC_QK_K; ++b) {
    uint8_t* p = out + b * blk;
    for (size_t i = 0; i < blk; ++i) p[i] = rndb();
    /* tame the f16 scale(s): Q4_K/Q5_K carry {d, dmin} at [0..3], Q6_K a single
     * d at [208]. Values then land in ~[-1, 1]. */
    uint16_t d = oc_f32_to_f16(type == OC_Q6_K ? 5e-4f : 1e-3f);
    uint16_t dm = oc_f32_to_f16(1e-3f);
    if (type == OC_Q6_K) {
      p[208] = (uint8_t)(d & 0xff);
      p[209] = (uint8_t)(d >> 8);
    } else {
      p[0] = (uint8_t)(d & 0xff);
      p[1] = (uint8_t)(d >> 8);
      p[2] = (uint8_t)(dm & 0xff);
      p[3] = (uint8_t)(dm >> 8);
    }
  }
}

/* 2-D [cols, rows] weight tensor of any type qrow_any() can emit. */
static void tsr_any(GgufB* m, const char* name, uint32_t type, size_t rows,
                    size_t cols, float spread) {
  CHECK(m->n_t < 128);
  Tsr* t = &m->t[m->n_t++];
  size_t rb = oc_row_bytes(type, cols);
  CHECK(rb != 0);
  t->name = name;
  t->n_dims = 2;
  t->dims[0] = cols;
  t->dims[1] = rows;
  t->type = type;
  t->n_vals = cols * rows;
  t->nbytes = rows * rb;
  t->qbytes = malloc(t->nbytes);
  CHECK(t->qbytes != NULL);
  float* fr = malloc(cols * sizeof(float));
  CHECK(fr != NULL);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) fr[c] = spread * rndf();
    qrow_any(type, fr, t->qbytes + r * rb, cols);
  }
  free(fr);
}

/* Serialize to GGUF v3 (alignment 32; f32 or per-tensor quantized). Caller frees. */
static uint8_t* build(GgufB* m, size_t* len_out) {
  Buf z = {NULL, 0, 0};
  put(&z, "GGUF", 4);
  put_u32(&z, 3);
  put_u64(&z, m->n_t);
  put_u64(&z, m->n_kv);
  put(&z, m->kv.b, m->kv.n);

  uint64_t off = 0; /* tensor offsets are relative to the data section */
  for (size_t i = 0; i < m->n_t; ++i) {
    Tsr* t = &m->t[i];
    size_t nb = t->qbytes ? t->nbytes : t->n_vals * 4;
    put_str(&z, t->name);
    put_u32(&z, t->n_dims);
    for (uint32_t d = 0; d < t->n_dims; ++d) put_u64(&z, t->dims[d]);
    put_u32(&z, t->type); /* 0 == OC_F32 for the plain tsr() path */
    put_u64(&z, off);
    off += (nb + 31) & ~(uint64_t)31;
  }
  put_pad32(&z);

  for (size_t i = 0; i < m->n_t; ++i) {
    Tsr* t = &m->t[i];
    size_t nb = t->qbytes ? t->nbytes : t->n_vals * 4;
    put(&z, t->qbytes ? (void*)t->qbytes : (void*)t->data, nb);
    put_pad32(&z);
    free(t->data);
    free(t->qbytes);
    t->data = NULL;
    t->qbytes = NULL;
  }
  free(m->kv.b);
  *len_out = z.n;
  return z.b;
}

#endif
