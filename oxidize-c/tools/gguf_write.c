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
 * F32/F16/Q8_0/Q4_0/AL5_XS plus ggml-compatible K-quants
 * (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K). Layout matches oc_dequant_row. */

uint32_t gw_type_id(const char* name) {
  if (!strcmp(name, "F32")) return OC_F32;
  if (!strcmp(name, "F16")) return OC_F16;
  if (!strcmp(name, "Q8_0")) return OC_Q8_0;
  if (!strcmp(name, "Q4_0")) return OC_Q4_0;
  if (!strcmp(name, "Q2_K")) return OC_Q2_K;
  if (!strcmp(name, "Q3_K")) return OC_Q3_K;
  if (!strcmp(name, "Q4_K")) return OC_Q4_K;
  if (!strcmp(name, "Q5_K")) return OC_Q5_K;
  if (!strcmp(name, "Q6_K")) return OC_Q6_K;
  if (!strcmp(name, "IQ4_XS")) return OC_IQ4_XS;
  if (!strcmp(name, "AL5_XS")) return OC_AL5_XS;
  return UINT32_MAX;
}

int gw_encodable(uint32_t type) {
  return type == OC_F32 || type == OC_F16 || type == OC_Q8_0 ||
         type == OC_Q4_0 || type == OC_Q2_K || type == OC_Q3_K ||
         type == OC_Q4_K || type == OC_Q5_K || type == OC_Q6_K ||
         type == OC_IQ4_XS || type == OC_AL5_XS;
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

/* ggml make_qkx2_quants (weights = |x|), used by Q2_K. */
static float make_qkx2(int n, int nmax, const float* x, uint8_t* L, float* the_min,
                       uint8_t* Laux, float rmin, float rdelta, int nstep) {
  float min = x[0], max = x[0], sum_w = fabsf(x[0]), sum_x = sum_w * x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] < min) min = x[i];
    if (x[i] > max) max = x[i];
    float w = fabsf(x[i]);
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min > 0.0f) min = 0.0f;
  if (max == min) {
    memset(L, 0, (size_t)n);
    *the_min = -min;
    return 0.0f;
  }
  float iscale = (float)nmax / (max - min);
  float scale = 1.0f / iscale;
  float best_err = 0.0f;
  for (int i = 0; i < n; ++i) {
    int l = gw_nearest_int(iscale * (x[i] - min));
    if (l < 0) l = 0;
    if (l > nmax) l = nmax;
    L[i] = (uint8_t)l;
    float diff = scale * (float)L[i] + min - x[i];
    best_err += fabsf(x[i]) * fabsf(diff);
  }
  for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta * (float)is + (float)nmax) / (max - min);
    float sum_l = 0.0f, sum_l2 = 0.0f, sum_xl = 0.0f;
    for (int i = 0; i < n; ++i) {
      int l = gw_nearest_int(iscale * (x[i] - min));
      if (l < 0) l = 0;
      if (l > nmax) l = nmax;
      Laux[i] = (uint8_t)l;
      float w = fabsf(x[i]);
      sum_l += w * (float)l;
      sum_l2 += w * (float)(l * l);
      sum_xl += w * (float)l * x[i];
    }
    float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0.0f) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0.0f) {
        this_min = 0.0f;
        this_scale = sum_l2 > 0.0f ? sum_xl / sum_l2 : 0.0f;
      }
      float cur = 0.0f;
      for (int i = 0; i < n; ++i) {
        float diff = this_scale * (float)Laux[i] + this_min - x[i];
        cur += fabsf(x[i]) * fabsf(diff);
      }
      if (cur < best_err) {
        memcpy(L, Laux, (size_t)n);
        best_err = cur;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

static void encode_q2_k(const float* x, uint8_t* dst, size_t n) {
  uint8_t L[OC_QK_K], Laux[16];
  float mins[OC_QK_K / 16], scales[OC_QK_K / 16];
  for (size_t b = 0; b < n / OC_QK_K; ++b) {
    const float* xb = x + b * OC_QK_K;
    uint8_t* blk = dst + b * OC_BLK_Q2_K;
    memset(blk, 0, OC_BLK_Q2_K);
    float max_scale = 0.0f, max_min = 0.0f;
    for (int j = 0; j < OC_QK_K / 16; ++j) {
      scales[j] = make_qkx2(16, 3, xb + 16 * j, L + 16 * j, &mins[j], Laux,
                            -0.5f, 0.1f, 15);
      if (scales[j] > max_scale) max_scale = scales[j];
      if (mins[j] > max_min) max_min = mins[j];
    }
    uint8_t* scb = blk;
    if (max_scale > 0.0f) {
      float iscale = 15.0f / max_scale;
      for (int j = 0; j < OC_QK_K / 16; ++j) {
        int l = gw_nearest_int(iscale * scales[j]);
        if (l < 0) l = 0;
        if (l > 15) l = 15;
        scb[j] = (uint8_t)l;
      }
      wr16(blk + 80, oc_f32_to_f16(max_scale / 15.0f));
    } else {
      wr16(blk + 80, oc_f32_to_f16(0.0f));
    }
    if (max_min > 0.0f) {
      float iscale = 15.0f / max_min;
      for (int j = 0; j < OC_QK_K / 16; ++j) {
        int l = gw_nearest_int(iscale * mins[j]);
        if (l < 0) l = 0;
        if (l > 15) l = 15;
        scb[j] = (uint8_t)(scb[j] | (l << 4));
      }
      wr16(blk + 82, oc_f32_to_f16(max_min / 15.0f));
    } else {
      wr16(blk + 82, oc_f32_to_f16(0.0f));
    }
    float d0 = oc_f16_to_f32((uint16_t)(blk[80] | (blk[81] << 8)));
    float m0 = oc_f16_to_f32((uint16_t)(blk[82] | (blk[83] << 8)));
    for (int j = 0; j < OC_QK_K / 16; ++j) {
      float d = d0 * (float)(scb[j] & 0xF);
      if (d == 0.0f) continue;
      float dm = m0 * (float)(scb[j] >> 4);
      for (int ii = 0; ii < 16; ++ii) {
        int l = gw_nearest_int((xb[16 * j + ii] + dm) / d);
        if (l < 0) l = 0;
        if (l > 3) l = 3;
        L[16 * j + ii] = (uint8_t)l;
      }
    }
    uint8_t* qs = blk + 16;
    for (int j = 0; j < OC_QK_K; j += 128) {
      for (int l = 0; l < 32; ++l)
        qs[j / 4 + l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 2) |
                                  (L[j + l + 64] << 4) | (L[j + l + 96] << 6));
    }
  }
}

static void encode_q3_k(const float* x, uint8_t* dst, size_t n) {
  int8_t L[OC_QK_K];
  float scales[OC_QK_K / 16];
  for (size_t b = 0; b < n / OC_QK_K; ++b) {
    const float* xb = x + b * OC_QK_K;
    uint8_t* blk = dst + b * OC_BLK_Q3_K;
    memset(blk, 0, OC_BLK_Q3_K);
    float max_scale = 0.0f, amax = 0.0f;
    for (int j = 0; j < OC_QK_K / 16; ++j) {
      scales[j] = make_qx(16, 4, xb + 16 * j, L + 16 * j);
      float a = fabsf(scales[j]);
      if (a > amax) {
        amax = a;
        max_scale = scales[j];
      }
    }
    uint8_t* scb = blk + 96;
    if (max_scale != 0.0f) {
      float iscale = -32.0f / max_scale;
      for (int j = 0; j < OC_QK_K / 16; ++j) {
        int l = gw_nearest_int(iscale * scales[j]);
        if (l < -32) l = -32;
        if (l > 31) l = 31;
        l += 32;
        if (j < 8)
          scb[j] = (uint8_t)(l & 0xF);
        else
          scb[j - 8] = (uint8_t)(scb[j - 8] | ((l & 0xF) << 4));
        scb[j % 4 + 8] =
            (uint8_t)(scb[j % 4 + 8] | ((l >> 4) << (2 * (j / 4))));
      }
      wr16(blk + 108, oc_f32_to_f16(1.0f / iscale));
    } else {
      wr16(blk + 108, oc_f32_to_f16(0.0f));
    }
    float d0 = oc_f16_to_f32((uint16_t)(blk[108] | (blk[109] << 8)));
    for (int j = 0; j < OC_QK_K / 16; ++j) {
      int sc = j < 8 ? scb[j] & 0xF : scb[j - 8] >> 4;
      sc = (sc | (((scb[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
      float d = d0 * (float)sc;
      if (d == 0.0f) continue;
      for (int ii = 0; ii < 16; ++ii) {
        int l = gw_nearest_int(xb[16 * j + ii] / d);
        if (l < -4) l = -4;
        if (l > 3) l = 3;
        L[16 * j + ii] = (int8_t)(l + 4);
      }
    }
    uint8_t* hm = blk;
    int m = 0;
    uint8_t bit = 1;
    for (int j = 0; j < OC_QK_K; ++j) {
      if (L[j] > 3) {
        hm[m] |= bit;
        L[j] = (int8_t)(L[j] - 4);
      }
      if (++m == OC_QK_K / 8) {
        m = 0;
        bit = (uint8_t)(bit << 1);
      }
    }
    uint8_t* qs = blk + 32;
    for (int j = 0; j < OC_QK_K; j += 128) {
      for (int l = 0; l < 32; ++l)
        qs[j / 4 + l] = (uint8_t)((uint8_t)L[j + l] | ((uint8_t)L[j + l + 32] << 2) |
                                  ((uint8_t)L[j + l + 64] << 4) |
                                  ((uint8_t)L[j + l + 96] << 6));
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

/* IQ4_NL codebook (ggml kvalues_iq4nl). */
static const int8_t gw_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static int best_iq4nl(float x) {
  int best = 0;
  float bestd = fabsf(x - (float)gw_kvalues_iq4nl[0]);
  for (int i = 1; i < 16; ++i) {
    float d = fabsf(x - (float)gw_kvalues_iq4nl[i]);
    if (d < bestd) {
      bestd = d;
      best = i;
    }
  }
  return best;
}

static void encode_iq4_xs(const float* x, uint8_t* dst, size_t n) {
  uint8_t L[OC_QK_K];
  float scales[8], weight[32];
  for (size_t b = 0; b < n / OC_QK_K; ++b) {
    const float* xb = x + b * OC_QK_K;
    uint8_t* blk = dst + b * OC_BLK_IQ4_XS;
    memset(blk, 0, OC_BLK_IQ4_XS);
    float max_scale = 0.0f, amax_scale = 0.0f;
    for (int ib = 0; ib < 8; ++ib) {
      const float* xg = xb + ib * 32;
      float amax = 0.0f, maxv = 0.0f;
      for (int j = 0; j < 32; ++j) {
        weight[j] = xg[j] * xg[j];
        float ax = fabsf(xg[j]);
        if (ax > amax) {
          amax = ax;
          maxv = xg[j];
        }
      }
      if (amax < 1e-15f) {
        scales[ib] = 0.0f;
        continue;
      }
      float d = -maxv / (float)gw_kvalues_iq4nl[0];
      float id = 1.0f / d;
      float sumqx = 0.0f, sumq2 = 0.0f;
      for (int j = 0; j < 32; ++j) {
        int l = best_iq4nl(id * xg[j]);
        L[ib * 32 + j] = (uint8_t)l;
        float q = (float)gw_kvalues_iq4nl[l];
        sumqx += weight[j] * q * xg[j];
        sumq2 += weight[j] * q * q;
      }
      d = sumq2 > 0.0f ? sumqx / sumq2 : 0.0f;
      float best = d * sumqx;
      for (int itry = -7; itry <= 7; ++itry) {
        id = ((float)itry + (float)gw_kvalues_iq4nl[0]) / maxv;
        sumqx = sumq2 = 0.0f;
        for (int j = 0; j < 32; ++j) {
          int l = best_iq4nl(id * xg[j]);
          float q = (float)gw_kvalues_iq4nl[l];
          sumqx += weight[j] * q * xg[j];
          sumq2 += weight[j] * q * q;
        }
        if (sumq2 > 0.0f && sumqx * sumqx > best * sumq2) {
          d = sumqx / sumq2;
          best = d * sumqx;
        }
      }
      scales[ib] = d;
      float ad = fabsf(d);
      if (ad > amax_scale) {
        amax_scale = ad;
        max_scale = d;
      }
    }
    float dsuper = amax_scale > 0.0f ? -max_scale / 32.0f : 0.0f;
    wr16(blk, oc_f32_to_f16(dsuper));
    float idsuper = dsuper != 0.0f ? 1.0f / dsuper : 0.0f;
    uint16_t scales_h = 0;
    uint8_t* scales_l = blk + 4;
    for (int ib = 0; ib < 8; ++ib) {
      int l = gw_nearest_int(idsuper * scales[ib]);
      if (l < -32) l = -32;
      if (l > 31) l = 31;
      float dl = dsuper * (float)l;
      float idl = dl != 0.0f ? 1.0f / dl : 0.0f;
      const float* xg = xb + ib * 32;
      for (int j = 0; j < 32; ++j) L[ib * 32 + j] = (uint8_t)best_iq4nl(idl * xg[j]);
      l += 32;
      uint8_t ll = (uint8_t)(l & 0xf), lh = (uint8_t)(l >> 4);
      if ((ib & 1) == 0)
        scales_l[ib / 2] = ll;
      else
        scales_l[ib / 2] = (uint8_t)(scales_l[ib / 2] | (ll << 4));
      scales_h = (uint16_t)(scales_h | (lh << (2 * (ib % 8))));
    }
    wr16(blk + 2, scales_h);
    uint8_t* qs = blk + 8;
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 16; ++j)
        qs[16 * i + j] =
            (uint8_t)(L[32 * i + j] | (L[32 * i + 16 + j] << 4));
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
    case OC_Q2_K:
      if (n % OC_QK_K) return -1;
      encode_q2_k(x, dst, n);
      return 0;
    case OC_Q3_K:
      if (n % OC_QK_K) return -1;
      encode_q3_k(x, dst, n);
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
    case OC_IQ4_XS:
      if (n % OC_QK_K) return -1;
      encode_iq4_xs(x, dst, n);
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
