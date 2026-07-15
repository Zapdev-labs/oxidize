/* GGUF v3 writer + row encoders shared by the offline tools. */
#include "gguf_write.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../src/quant.h"
#include "../src/tensor.h"

/* ---- little-endian primitives (gguf.c is read-only, so we write by hand) --- */

static void w_bytes(GwWriter* w, const void* p, size_t n) {
  if (w->err) return;
  if (fwrite(p, 1, n, w->f) != n) {
    perror("fwrite");
    w->err = 1;
  }
}
static void w_u8(GwWriter* w, uint8_t v) { w_bytes(w, &v, 1); }
static void w_u16(GwWriter* w, uint16_t v) {
  uint8_t b[2] = {(uint8_t)v, (uint8_t)(v >> 8)};
  w_bytes(w, b, 2);
}
static void w_u32(GwWriter* w, uint32_t v) {
  uint8_t b[4];
  for (int i = 0; i < 4; ++i) b[i] = (uint8_t)(v >> (8 * i));
  w_bytes(w, b, 4);
}
static void w_u64(GwWriter* w, uint64_t v) {
  uint8_t b[8];
  for (int i = 0; i < 8; ++i) b[i] = (uint8_t)(v >> (8 * i));
  w_bytes(w, b, 8);
}
static void w_f32v(GwWriter* w, float v) {
  uint32_t u;
  memcpy(&u, &v, 4);
  w_u32(w, u);
}
static void w_f64v(GwWriter* w, double v) {
  uint64_t u;
  memcpy(&u, &v, 8);
  w_u64(w, u);
}
static void w_str(GwWriter* w, const char* s, size_t len) {
  w_u64(w, len);
  w_bytes(w, s, len);
}

static void w_value(GwWriter* w, const GgufValue* v) {
  switch (v->kind) {
    case GGUF_T_U8: w_u8(w, (uint8_t)v->v.u); break;
    case GGUF_T_BOOL: w_u8(w, v->v.u ? 1 : 0); break;
    case GGUF_T_I8: w_u8(w, (uint8_t)(int8_t)v->v.i); break;
    case GGUF_T_U16: w_u16(w, (uint16_t)v->v.u); break;
    case GGUF_T_I16: w_u16(w, (uint16_t)(int16_t)v->v.i); break;
    case GGUF_T_U32: w_u32(w, (uint32_t)v->v.u); break;
    case GGUF_T_I32: w_u32(w, (uint32_t)(int32_t)v->v.i); break;
    case GGUF_T_U64: w_u64(w, v->v.u); break;
    case GGUF_T_I64: w_u64(w, (uint64_t)v->v.i); break;
    case GGUF_T_F32: w_f32v(w, (float)v->v.f); break;
    case GGUF_T_F64: w_f64v(w, v->v.f); break;
    case GGUF_T_STRING: w_str(w, v->v.str.ptr, v->v.str.len); break;
    case GGUF_T_ARRAY:
      w_u32(w, (uint32_t)v->v.arr.elem_kind);
      w_u64(w, v->v.arr.n);
      for (size_t i = 0; i < v->v.arr.n; ++i) w_value(w, &v->v.arr.items[i]);
      break;
    default:
      fprintf(stderr, "gguf_write: unknown KV kind %d\n", v->kind);
      w->err = 1;
  }
}

static uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

int gw_open(GwWriter* w, const char* path, const GgufKv* kvs, size_t n_kv,
            uint64_t align, const GwTensor* ts, size_t nt, int file_type) {
  w->f = fopen(path, "wb");
  w->align = align ? align : 32;
  w->pos = 0;
  w->err = 0;
  if (!w->f) {
    perror(path);
    return -1;
  }

  int have_ft = 0;
  for (size_t i = 0; i < n_kv; ++i)
    if (!strcmp(kvs[i].key, "general.file_type")) have_ft = 1;
  int append_ft = file_type >= 0 && !have_ft;

  w_bytes(w, "GGUF", 4);
  w_u32(w, 3);
  w_u64(w, nt);
  w_u64(w, n_kv + (append_ft ? 1u : 0u));
  for (size_t i = 0; i < n_kv; ++i) {
    w_str(w, kvs[i].key, strlen(kvs[i].key));
    if (file_type >= 0 && !strcmp(kvs[i].key, "general.file_type")) {
      w_u32(w, GGUF_T_U32);
      w_u32(w, (uint32_t)file_type);
      continue;
    }
    w_u32(w, (uint32_t)kvs[i].val.kind);
    w_value(w, &kvs[i].val);
  }
  if (append_ft) {
    w_str(w, "general.file_type", strlen("general.file_type"));
    w_u32(w, GGUF_T_U32);
    w_u32(w, (uint32_t)file_type);
  }

  uint64_t off = 0;
  for (size_t i = 0; i < nt; ++i) {
    w_str(w, ts[i].name, strlen(ts[i].name));
    w_u32(w, ts[i].n_dims);
    for (uint32_t d = 0; d < ts[i].n_dims; ++d) w_u64(w, ts[i].dims[d]);
    w_u32(w, ts[i].type);
    w_u64(w, off);
    off = align_up(off + ts[i].size, w->align);
  }

  long hdr_end = ftell(w->f);
  if (hdr_end < 0) {
    perror("ftell");
    w->err = 1;
  }
  for (uint64_t p = (uint64_t)hdr_end; p < align_up((uint64_t)hdr_end, w->align); ++p)
    w_u8(w, 0);
  if (w->err) {
    fclose(w->f);
    w->f = NULL;
    return -1;
  }
  return 0;
}

int gw_tensor(GwWriter* w, const void* data, uint64_t size) {
  w_bytes(w, data, size);
  w->pos += size;
  for (uint64_t p = w->pos; p < align_up(w->pos, w->align); ++p) w_u8(w, 0);
  w->pos = align_up(w->pos, w->align);
  return w->err ? -1 : 0;
}

int gw_close(GwWriter* w) {
  int err = w->err;
  if (w->f && fclose(w->f) != 0) {
    perror("fclose");
    err = 1;
  }
  w->f = NULL;
  return err ? -1 : 0;
}

/* Does `bytes` of tensor payload actually live inside one of the file's maps?
 * gguf.c hands out a raw pointer without checking the declared size against the
 * file length, and the tiny parser fixtures declare huge tensors over an empty
 * data section — reading them SIGBUSes. Every tool checks before it touches. */
int gw_data_ok(const GgufFile* f, const GgufTensorInfo* t, uint64_t bytes) {
  const uint8_t* p = t->data;
  if (!p) return 0;
  const uint8_t* base = (const uint8_t*)f->map;
  if (base && p >= base && p <= base + f->size)
    return (uint64_t)(base + f->size - p) >= bytes;
  for (size_t i = 0; i < f->n_shards; ++i) {
    const uint8_t* sb = (const uint8_t*)f->shards[i].map;
    if (sb && p >= sb && p <= sb + f->shards[i].size)
      return (uint64_t)(sb + f->shards[i].size - p) >= bytes;
  }
  return 0;
}

/* ---- encoders -------------------------------------------------------------
 * Only the types we can round-trip-verify against the ggml block layout:
 * F32, F16, Q8_0, Q4_0 and the custom AL5_XS. The K-quants (Q4_K/Q6_K/...)
 * are DEQUANT-ONLY here: their super-block scale/min search is not implemented,
 * and a wrong K-quant encoder is silent quality loss. */

uint32_t gw_type_id(const char* name) {
  if (!strcmp(name, "F32")) return OC_F32;
  if (!strcmp(name, "F16")) return OC_F16;
  if (!strcmp(name, "Q8_0")) return OC_Q8_0;
  if (!strcmp(name, "Q4_0")) return OC_Q4_0;
  if (!strcmp(name, "AL5_XS")) return OC_AL5_XS;
  return UINT32_MAX;
}

int gw_encodable(uint32_t type) {
  return type == OC_F32 || type == OC_F16 || type == OC_Q8_0 ||
         type == OC_Q4_0 || type == OC_AL5_XS;
}

static void encode_q8_0(const float* x, uint8_t* dst, size_t n) {
  for (size_t b = 0; b < n / 32; ++b) {
    const float* v = x + b * 32;
    uint8_t* o = dst + b * OC_BLK_Q8_0;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i)
      if (fabsf(v[i]) > amax) amax = fabsf(v[i]);
    uint16_t d16 = oc_f32_to_f16(amax / 127.0f);
    float d = oc_f16_to_f32(d16);
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    o[0] = (uint8_t)(d16 & 0xff);
    o[1] = (uint8_t)(d16 >> 8);
    for (int i = 0; i < 32; ++i) {
      long q = lrintf(v[i] * id);
      if (q > 127) q = 127;
      if (q < -128) q = -128;
      o[2 + i] = (uint8_t)(int8_t)q;
    }
  }
}

int gw_encode_row(uint32_t type, const float* x, uint8_t* dst, size_t n) {
  switch (type) {
    case OC_F32: memcpy(dst, x, n * 4); return 0;
    case OC_F16:
      for (size_t i = 0; i < n; ++i) {
        uint16_t h = oc_f32_to_f16(x[i]);
        dst[2 * i] = (uint8_t)(h & 0xff);
        dst[2 * i + 1] = (uint8_t)(h >> 8);
      }
      return 0;
    case OC_Q8_0:
      if (n % 32) return -1;
      encode_q8_0(x, dst, n);
      return 0;
    case OC_Q4_0:
      if (n % 32) return -1;
      oc_quantize_row_q4_0(x, dst, n);
      return 0;
    case OC_AL5_XS:
      if (n % 32) return -1;
      for (size_t b = 0; b < n / 32; ++b)
        oc_al5xs_encode_block(x + b * 32, dst + b * OC_BLK_AL5_XS);
      return 0;
    default: return -1;
  }
}

/* ---- threaded requant ----------------------------------------------------- */

typedef struct {
  const uint8_t* src;
  uint8_t* dst;
  uint32_t src_type, dst_type;
  size_t cols, src_rb, dst_rb;
} ReqJob;

static void requant_rows(void* ctx, size_t r0, size_t r1) {
  ReqJob* j = (ReqJob*)ctx;
  float* row = malloc(j->cols * sizeof(float));
  if (!row) abort();
  for (size_t r = r0; r < r1; ++r) {
    if (oc_dequant_row(j->src_type, j->src + r * j->src_rb, row, j->cols) != 0 ||
        gw_encode_row(j->dst_type, row, j->dst + r * j->dst_rb, j->cols) != 0) {
      fprintf(stderr, "requant: type %u -> %u failed\n", j->src_type, j->dst_type);
      exit(1);
    }
  }
  free(row);
}

void gw_requant(const uint8_t* src, uint32_t src_type, uint32_t dst_type,
                size_t rows, size_t cols, uint8_t* dst) {
  ReqJob j = {.src = src,
              .dst = dst,
              .src_type = src_type,
              .dst_type = dst_type,
              .cols = cols,
              .src_rb = oc_row_bytes(src_type, cols),
              .dst_rb = oc_row_bytes(dst_type, cols)};
  oc_pool_init(0);
  oc_parallel_for(rows, requant_rows, &j);
}
