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
 * F32/F16/Q8_0/Q4_0/AL5_XS plus ggml-compatible K-quants (Q4_K/Q5_K/Q6_K).
 * Q4_K/Q5_K use make_qkx1 (same as oxidize-core quantize_q4_k_scalar); Q6_K
 * uses make_qx_quants. Layout matches oc_dequant_row. */

uint32_t gw_type_id(const char* name) {
  if (!strcmp(name, "F32")) return OC_F32;
  if (!strcmp(name, "F16")) return OC_F16;
  if (!strcmp(name, "Q8_0")) return OC_Q8_0;
  if (!strcmp(name, "Q4_0")) return OC_Q4_0;
  if (!strcmp(name, "Q4_K")) return OC_Q4_K;
  if (!strcmp(name, "Q5_K")) return OC_Q5_K;
  if (!strcmp(name, "Q6_K")) return OC_Q6_K;
  if (!strcmp(name, "AL5_XS")) return OC_AL5_XS;
  return UINT32_MAX;
}

int gw_encodable(uint32_t type) {
  return type == OC_F32 || type == OC_F16 || type == OC_Q8_0 ||
         type == OC_Q4_0 || type == OC_Q4_K || type == OC_Q5_K ||
         type == OC_Q6_K || type == OC_AL5_XS;
}

static int gw_nearest_int(float fval) {
  float val = fval + 12582912.0f;
  uint32_t u;
  memcpy(&u, &val, 4);
  return (int)(u & 0x007fffffu) - 0x00400000;
}

static float make_qkx1(int n, int nmax, const float* x, uint8_t* L, float* the_min,
                       int ntry, float alpha) {
  float min = x[0], max = x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] < min) min = x[i];
    if (x[i] > max) max = x[i];
  }
  if (max == min) {
    memset(L, 0, (size_t)n);
    *the_min = 0.0f;
    return 0.0f;
  }
  if (min > 0.0f) min = 0.0f;
  float iscale = (float)nmax / (max - min);
  float scale = 1.0f / iscale;
  for (int itry = 0; itry < ntry; ++itry) {
    float sumlx = 0.0f;
    int suml2 = 0;
    int did_change = 0;
    for (int i = 0; i < n; ++i) {
      int l = gw_nearest_int(iscale * (x[i] - min));
      if (l < 0) l = 0;
      if (l > nmax) l = nmax;
      if (l != (int)L[i]) {
        L[i] = (uint8_t)l;
        did_change = 1;
      }
      sumlx += (x[i] - min) * (float)l;
      suml2 += l * l;
    }
    scale = suml2 > 0 ? sumlx / (float)suml2 : 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += x[i] - scale * (float)L[i];
    min = alpha * min + (1.0f - alpha) * sum / (float)n;
    if (min > 0.0f) min = 0.0f;
    iscale = scale != 0.0f ? 1.0f / scale : 0.0f;
    if (!did_change) break;
  }
  *the_min = -min;
  return scale;
}

static float make_qx(int n, int nmax, const float* x, int8_t* L) {
  float max = 0.0f, amax = 0.0f;
  for (int i = 0; i < n; ++i) {
    float ax = fabsf(x[i]);
    if (ax > amax) {
      amax = ax;
      max = x[i];
    }
  }
  if (amax < 1e-15f) {
    memset(L, 0, (size_t)n);
    return 0.0f;
  }
  float iscale = -(float)nmax / max;
  float sumlx = 0.0f, suml2 = 0.0f;
  for (int i = 0; i < n; ++i) {
    int l = gw_nearest_int(iscale * x[i]);
    if (l < -nmax) l = -nmax;
    if (l > nmax - 1) l = nmax - 1;
    L[i] = (int8_t)(l + nmax);
    float w = x[i] * x[i];
    sumlx += w * x[i] * (float)l;
    suml2 += w * (float)(l * l);
  }
  float scale = suml2 > 0.0f ? sumlx / suml2 : 0.0f;
  float best = scale * sumlx;
  for (int is = -9; is <= 9; ++is) {
    if (is == 0) continue;
    iscale = -((float)nmax + 0.1f * (float)is) / max;
    sumlx = suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
      int l = gw_nearest_int(iscale * x[i]);
      if (l < -nmax) l = -nmax;
      if (l > nmax - 1) l = nmax - 1;
      float w = x[i] * x[i];
      sumlx += w * x[i] * (float)l;
      suml2 += w * (float)(l * l);
    }
    if (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
      for (int i = 0; i < n; ++i) {
        int l = gw_nearest_int(iscale * x[i]);
        if (l < -nmax) l = -nmax;
        if (l > nmax - 1) l = nmax - 1;
        L[i] = (int8_t)(l + nmax);
      }
      scale = sumlx / suml2;
      best = scale * sumlx;
    }
  }
  return scale;
}

static void wr16(uint8_t* p, uint16_t v) {
  p[0] = (uint8_t)(v & 0xff);
  p[1] = (uint8_t)(v >> 8);
}

static void pack_scale_min_k4(uint8_t* scales, int j, uint8_t ls, uint8_t lm) {
  if (j < 4) {
    scales[j] = ls;
    scales[j + 4] = lm;
  } else {
    scales[j + 4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
    scales[j - 4] |= (uint8_t)((ls >> 4) << 6);
    scales[j] |= (uint8_t)((lm >> 4) << 6);
  }
}

static void encode_qkx_minmax(const float* x, uint8_t* dst, size_t n, int nmax,
                              int is_q5) {
  uint8_t L[OC_QK_K];
  float mins[OC_QK_K / 32], scales[OC_QK_K / 32];
  for (size_t b = 0; b < n / OC_QK_K; ++b) {
    const float* xb = x + b * OC_QK_K;
    uint8_t* blk = dst + b * (is_q5 ? OC_BLK_Q5_K : OC_BLK_Q4_K);
    memset(blk, 0, is_q5 ? OC_BLK_Q5_K : OC_BLK_Q4_K);
    float max_scale = 0.0f, max_min = 0.0f;
    for (int j = 0; j < OC_QK_K / 32; ++j) {
      scales[j] =
          make_qkx1(32, nmax, xb + 32 * j, L + 32 * j, &mins[j], 5, 0.5f);
      if (scales[j] > max_scale) max_scale = scales[j];
      if (mins[j] > max_min) max_min = mins[j];
    }
    float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
    float inv_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;
    uint8_t* scb = blk + 4;
    for (int j = 0; j < OC_QK_K / 32; ++j) {
      int ls = gw_nearest_int(inv_scale * scales[j]);
      int lm = gw_nearest_int(inv_min * mins[j]);
      if (ls < 0) ls = 0;
      if (ls > 63) ls = 63;
      if (lm < 0) lm = 0;
      if (lm > 63) lm = 63;
      pack_scale_min_k4(scb, j, (uint8_t)ls, (uint8_t)lm);
    }
    wr16(blk, oc_f32_to_f16(max_scale / 63.0f));
    wr16(blk + 2, oc_f32_to_f16(max_min / 63.0f));
    float d0 = oc_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
    float m0 = oc_f16_to_f32((uint16_t)(blk[2] | (blk[3] << 8)));
    for (int j = 0; j < OC_QK_K / 32; ++j) {
      uint8_t sc, m;
      if (j < 4) {
        sc = scb[j] & 63;
        m = scb[j + 4] & 63;
      } else {
        sc = (uint8_t)((scb[j + 4] & 0xF) | ((scb[j - 4] >> 6) << 4));
        m = (uint8_t)((scb[j + 4] >> 4) | ((scb[j] >> 6) << 4));
      }
      float d = d0 * (float)sc;
      if (d == 0.0f) continue;
      float dm = m0 * (float)m;
      for (int ii = 0; ii < 32; ++ii) {
        int l = gw_nearest_int((xb[32 * j + ii] + dm) / d);
        if (l < 0) l = 0;
        if (l > nmax) l = nmax;
        L[32 * j + ii] = (uint8_t)l;
      }
    }
    if (!is_q5) {
      uint8_t* q = blk + 16;
      for (int j = 0; j < OC_QK_K; j += 64) {
        for (int l = 0; l < 32; ++l)
          q[l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 4));
        q += 32;
      }
    } else {
      uint8_t* qh = blk + 16;
      uint8_t* ql = blk + 48;
      memset(qh, 0, OC_QK_K / 8);
      uint8_t m1 = 1, m2 = 2;
      for (int n0 = 0; n0 < OC_QK_K; n0 += 64) {
        for (int j = 0; j < 32; ++j) {
          int l1 = L[n0 + j], l2 = L[n0 + j + 32];
          if (l1 > 15) {
            l1 -= 16;
            qh[j] |= m1;
          }
          if (l2 > 15) {
            l2 -= 16;
            qh[j] |= m2;
          }
          ql[j] = (uint8_t)(l1 | (l2 << 4));
        }
        m1 = (uint8_t)(m1 << 2);
        m2 = (uint8_t)(m2 << 2);
        ql += 32;
      }
    }
  }
}

static void encode_q6_k(const float* x, uint8_t* dst, size_t n) {
  int8_t L[OC_QK_K];
  float scales[OC_QK_K / 16];
  for (size_t b = 0; b < n / OC_QK_K; ++b) {
    const float* xb = x + b * OC_QK_K;
    uint8_t* blk = dst + b * OC_BLK_Q6_K;
    memset(blk, 0, OC_BLK_Q6_K);
    float max_scale = 0.0f, max_abs = 0.0f;
    for (int ib = 0; ib < OC_QK_K / 16; ++ib) {
      scales[ib] = make_qx(16, 32, xb + 16 * ib, L + 16 * ib);
      float a = fabsf(scales[ib]);
      if (a > max_abs) {
        max_abs = a;
        max_scale = scales[ib];
      }
    }
    if (max_abs < 1e-15f) {
      wr16(blk + 208, oc_f32_to_f16(0.0f));
      continue;
    }
    float iscale = -128.0f / max_scale;
    wr16(blk + 208, oc_f32_to_f16(1.0f / iscale));
    int8_t* sc = (int8_t*)(blk + 192);
    for (int ib = 0; ib < OC_QK_K / 16; ++ib) {
      int v = gw_nearest_int(iscale * scales[ib]);
      if (v < -128) v = -128;
      if (v > 127) v = 127;
      sc[ib] = (int8_t)v;
    }
    float d0 = oc_f16_to_f32((uint16_t)(blk[208] | (blk[209] << 8)));
    for (int j = 0; j < OC_QK_K / 16; ++j) {
      float d = d0 * (float)sc[j];
      if (d == 0.0f) continue;
      for (int ii = 0; ii < 16; ++ii) {
        int l = gw_nearest_int(xb[16 * j + ii] / d);
        if (l < -32) l = -32;
        if (l > 31) l = 31;
        L[16 * j + ii] = (int8_t)(l + 32);
      }
    }
    uint8_t* ql = blk;
    uint8_t* qh = blk + 128;
    for (int j = 0; j < OC_QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        uint8_t q1 = (uint8_t)L[j + l] & 0xF;
        uint8_t q2 = (uint8_t)L[j + l + 32] & 0xF;
        uint8_t q3 = (uint8_t)L[j + l + 64] & 0xF;
        uint8_t q4 = (uint8_t)L[j + l + 96] & 0xF;
        ql[l] = (uint8_t)(q1 | (q3 << 4));
        ql[l + 32] = (uint8_t)(q2 | (q4 << 4));
        qh[l] = (uint8_t)((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) |
                          ((L[j + l + 64] >> 4) << 4) |
                          ((L[j + l + 96] >> 4) << 6));
      }
      ql += 64;
      qh += 32;
    }
  }
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
    case OC_Q4_K:
      if (n % OC_QK_K) return -1;
      encode_qkx_minmax(x, dst, n, 15, 0);
      return 0;
    case OC_Q5_K:
      if (n % OC_QK_K) return -1;
      encode_qkx_minmax(x, dst, n, 31, 1);
      return 0;
    case OC_Q6_K:
      if (n % OC_QK_K) return -1;
      encode_q6_k(x, dst, n);
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
