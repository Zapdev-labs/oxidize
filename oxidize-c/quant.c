/* Dequantization kernels. Byte/bit-faithful port of oxidize-cpp/src/quant.cpp
 * (itself ported from oxidize-core quantization.rs / ggml layouts). */
#include "oc.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void oc_die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "error: ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(1);
}

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

float oc_f16_to_f32(const uint8_t *le2) {
  uint16_t bits = rd16(le2);
  uint32_t sign = (uint32_t)(bits >> 15) & 1, exp = (bits >> 10) & 0x1F,
           frac = bits & 0x3FF, out;
  if (exp == 0) {
    if (frac == 0) {
      out = sign << 31;
    } else {
      uint32_t f = frac;
      int e = -14;
      while ((f & 0x400) == 0) { f <<= 1; e--; }
      out = (sign << 31) | ((uint32_t)(e + 127) << 23) | ((f & 0x3FF) << 13);
    }
  } else if (exp == 0x1F) {
    out = (sign << 31) | 0x7F800000u | (frac << 13);
  } else {
    out = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
  }
  float v;
  memcpy(&v, &out, 4);
  return v;
}

size_t oc_block_values(oc_quant q) {
  switch (q) {
    case OC_F32: case OC_F16: case OC_BF16: return 1;
    case OC_Q4_0: case OC_Q4_1: case OC_Q5_0: case OC_Q5_1: case OC_Q8_0: return QK;
    case OC_Q2_K: case OC_Q3_K: case OC_Q4_K: case OC_Q5_K: case OC_Q6_K: return QK_K;
    default: oc_die("quant: no block layout (type %d)", (int)q);
  }
  return 0;
}

size_t oc_block_bytes(oc_quant q) {
  switch (q) {
    case OC_F32: return 4;
    case OC_F16: case OC_BF16: return 2;
    case OC_Q4_0: return 18;
    case OC_Q4_1: return 20;
    case OC_Q5_0: return 22;
    case OC_Q5_1: return 24;
    case OC_Q8_0: return 34;
    case OC_Q2_K: return 84;
    case OC_Q3_K: return 110;
    case OC_Q4_K: return 144;
    case OC_Q5_K: return 176;
    case OC_Q6_K: return 210;
    default: oc_die("quant: no block layout (type %d)", (int)q);
  }
  return 0;
}

size_t oc_row_bytes(oc_quant q, size_t cols) {
  size_t v = oc_block_values(q);
  if (cols % v != 0) oc_die("quant: cols %zu not multiple of block %zu", cols, v);
  return cols / v * oc_block_bytes(q);
}

oc_quant oc_from_ggml_type(uint32_t t) {
  switch (t) {
    case 0: return OC_F32;
    case 1: return OC_F16;
    case 2: return OC_Q4_0;
    case 3: return OC_Q4_1;
    case 6: return OC_Q5_0;
    case 7: return OC_Q5_1;
    case 8: return OC_Q8_0;
    case 10: return OC_Q2_K;
    case 11: return OC_Q3_K;
    case 12: return OC_Q4_K;
    case 13: return OC_Q5_K;
    case 14: return OC_Q6_K;
    case 30: return OC_BF16;
    default: return OC_UNKNOWN;
  }
}

const char *oc_quant_name(oc_quant q) {
  static const char *n[] = {"F32","F16","BF16","Q4_0","Q4_1","Q5_0","Q5_1",
                            "Q8_0","Q2_K","Q3_K","Q4_K","Q5_K","Q6_K","?"};
  return n[q <= OC_UNKNOWN ? q : OC_UNKNOWN];
}

/* K-quant 6-bit scale/min extraction (ggml get_scale_min_k4). */
static void scale_min_k4(size_t j, const uint8_t *s, uint8_t *sc, uint8_t *m) {
  if (j < 4) {
    *sc = s[j] & 63;
    *m = s[j + 4] & 63;
  } else {
    *sc = (uint8_t)((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4));
    *m = (uint8_t)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
  }
}

void oc_dequant_row(oc_quant q, const uint8_t *src, float *dst, size_t n) {
  size_t b, i, l;
  switch (q) {
    case OC_F32:
      memcpy(dst, src, n * 4);
      return;
    case OC_F16:
      for (i = 0; i < n; ++i) dst[i] = oc_f16_to_f32(src + 2 * i);
      return;
    case OC_BF16:
      for (i = 0; i < n; ++i) {
        uint32_t bits = (uint32_t)rd16(src + 2 * i) << 16;
        memcpy(&dst[i], &bits, 4);
      }
      return;
    case OC_Q4_0:
      for (b = 0; b < n / QK; ++b) {
        const uint8_t *blk = src + b * 18;
        float d = oc_f16_to_f32(blk);
        float *o = dst + b * QK;
        for (i = 0; i < 16; ++i) {
          o[i] = (float)((int)(blk[2 + i] & 0xF) - 8) * d;
          o[i + 16] = (float)((int)(blk[2 + i] >> 4) - 8) * d;
        }
      }
      return;
    case OC_Q4_1:
      for (b = 0; b < n / QK; ++b) {
        const uint8_t *blk = src + b * 20;
        float d = oc_f16_to_f32(blk), m = oc_f16_to_f32(blk + 2);
        float *o = dst + b * QK;
        for (i = 0; i < 16; ++i) {
          o[i] = (float)(blk[4 + i] & 0xF) * d + m;
          o[i + 16] = (float)(blk[4 + i] >> 4) * d + m;
        }
      }
      return;
    case OC_Q5_0:
      for (b = 0; b < n / QK; ++b) {
        const uint8_t *blk = src + b * 22;
        float d = oc_f16_to_f32(blk);
        uint32_t qh = rd32(blk + 2);
        const uint8_t *qs = blk + 6;
        float *o = dst + b * QK;
        for (i = 0; i < 16; ++i) {
          int q0 = (int)((qs[i] & 0xF) | (((qh >> i) & 1) << 4)) - 16;
          int q1 = (int)((qs[i] >> 4) | (((qh >> (i + 16)) & 1) << 4)) - 16;
          o[i] = (float)q0 * d;
          o[i + 16] = (float)q1 * d;
        }
      }
      return;
    case OC_Q5_1:
      for (b = 0; b < n / QK; ++b) {
        const uint8_t *blk = src + b * 24;
        float d = oc_f16_to_f32(blk), m = oc_f16_to_f32(blk + 2);
        uint32_t qh = rd32(blk + 4);
        const uint8_t *qs = blk + 8;
        float *o = dst + b * QK;
        for (i = 0; i < 16; ++i) {
          o[i] = (float)((qs[i] & 0xF) | (((qh >> i) & 1) << 4)) * d + m;
          o[i + 16] = (float)((qs[i] >> 4) | (((qh >> (i + 16)) & 1) << 4)) * d + m;
        }
      }
      return;
    case OC_Q8_0:
      for (b = 0; b < n / QK; ++b) {
        const uint8_t *blk = src + b * 34;
        float d = oc_f16_to_f32(blk);
        float *o = dst + b * QK;
        for (i = 0; i < QK; ++i) o[i] = (float)(int8_t)blk[2 + i] * d;
      }
      return;
    case OC_Q2_K:
      for (b = 0; b < n / QK_K; ++b) {
        const uint8_t *blk = src + b * 84;
        float d = oc_f16_to_f32(blk + 80), min = oc_f16_to_f32(blk + 82);
        const uint8_t *scales = blk, *qs = blk + 16;
        float *o = dst + b * QK_K;
        size_t qp = 0, is = 0;
        for (int outer = 0; outer < 2; ++outer) {
          size_t qb = (size_t)outer * 32;
          for (int k = 0; k < 4; ++k) {
            uint8_t sc1 = scales[is++];
            float dl1 = d * (float)(sc1 & 0xF), ml1 = min * (float)(sc1 >> 4);
            uint8_t sc2 = scales[is++];
            float dl2 = d * (float)(sc2 & 0xF), ml2 = min * (float)(sc2 >> 4);
            size_t shift = ((is / 2 - 1) % 4) * 2;
            for (l = 0; l < 16; ++l)
              o[qp + l] = dl1 * (float)((qs[qb + l] >> shift) & 3) - ml1;
            for (l = 0; l < 16; ++l)
              o[qp + 16 + l] = dl2 * (float)((qs[qb + 16 + l] >> shift) & 3) - ml2;
            qp += 32;
          }
        }
      }
      return;
    case OC_Q3_K:
      for (b = 0; b < n / QK_K; ++b) {
        const uint8_t *blk = src + b * 110;
        float d_all = oc_f16_to_f32(blk + 108);
        const uint8_t *hmask = blk, *qs = blk + 32;
        uint32_t sr[4];
        sr[0] = rd32(blk + 96); sr[1] = rd32(blk + 100);
        uint32_t tmp = rd32(blk + 104);
        sr[2] = ((sr[0] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 4) & 0x03030303u) << 4);
        sr[3] = ((sr[1] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 6) & 0x03030303u) << 4);
        sr[0] = (sr[0] & 0x0F0F0F0Fu) | ((tmp & 0x03030303u) << 4);
        sr[1] = (sr[1] & 0x0F0F0F0Fu) | (((tmp >> 2) & 0x03030303u) << 4);
        int8_t scales[16];
        memcpy(scales, sr, 16);
        float *o = dst + b * QK_K;
        size_t qp = 0, is = 0;
        uint8_t m = 1;
        for (int g = 0; g < 2; ++g) {
          for (int k = 0; k < 4; ++k) {
            float dl = d_all * (float)((int)scales[is] - 32);
            ++is;
            size_t shift = (size_t)k * 2;
            for (l = 0; l < 16; ++l) {
              int qv = (qs[l] >> shift) & 3;
              o[qp + l] = dl * (float)(qv - ((hmask[l] & m) ? 0 : 4));
            }
            float dl2 = d_all * (float)((int)scales[is] - 32);
            ++is;
            for (l = 0; l < 16; ++l) {
              int qv = (qs[l + 16] >> shift) & 3;
              o[qp + 16 + l] = dl2 * (float)(qv - ((hmask[l + 16] & m) ? 0 : 4));
            }
            qp += 32;
            m <<= 1;
          }
          qs += 32; /* ggml advances qs every 128 values; hmask stays fixed */
        }
      }
      return;
    case OC_Q4_K:
      for (b = 0; b < n / QK_K; ++b) {
        const uint8_t *blk = src + b * 144;
        float d = oc_f16_to_f32(blk), min = oc_f16_to_f32(blk + 2);
        const uint8_t *scales = blk + 4, *qs = blk + 16;
        float *o = dst + b * QK_K;
        size_t op = 0, is = 0;
        for (int gp = 0; gp < 4; ++gp) {
          size_t qb = (size_t)gp * 32;
          uint8_t sc1, m1, sc2, m2;
          scale_min_k4(is, scales, &sc1, &m1);
          scale_min_k4(is + 1, scales, &sc2, &m2);
          float d1 = d * sc1, mn1 = min * m1, d2 = d * sc2, mn2 = min * m2;
          for (l = 0; l < 32; ++l) o[op + l] = d1 * (float)(qs[qb + l] & 0xF) - mn1;
          for (l = 0; l < 32; ++l) o[op + 32 + l] = d2 * (float)(qs[qb + l] >> 4) - mn2;
          op += 64;
          is += 2;
        }
      }
      return;
    case OC_Q5_K:
      for (b = 0; b < n / QK_K; ++b) {
        const uint8_t *blk = src + b * 176;
        float d = oc_f16_to_f32(blk), min = oc_f16_to_f32(blk + 2);
        const uint8_t *scales = blk + 4, *qh = blk + 16, *qs = blk + 48;
        float *o = dst + b * QK_K;
        size_t qp = 0, is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int g = 0; g < 4; ++g) {
          uint8_t sc1, m1, sc2, m2;
          scale_min_k4(is, scales, &sc1, &m1);
          scale_min_k4(is + 1, scales, &sc2, &m2);
          float d1 = d * sc1, mn1 = min * m1, d2 = d * sc2, mn2 = min * m2;
          for (l = 0; l < 32; ++l)
            o[qp + l] = d1 * (float)((qs[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - mn1;
          for (l = 0; l < 32; ++l)
            o[qp + 32 + l] = d2 * (float)((qs[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - mn2;
          qp += 64;
          is += 2;
          u1 <<= 2;
          u2 <<= 2;
          qs += 32;
        }
      }
      return;
    case OC_Q6_K:
      for (b = 0; b < n / QK_K; ++b) {
        const uint8_t *blk = src + b * 210;
        float d = oc_f16_to_f32(blk + 208);
        const uint8_t *ql = blk, *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        float *o = dst + b * QK_K;
        size_t qp = 0;
        for (int g = 0; g < 2; ++g) {
          size_t qlo = (size_t)g * 64, qho = (size_t)g * 32, sco = (size_t)g * 8;
          for (l = 0; l < 32; ++l) {
            size_t is = l / 16;
            int q1 = ((int)(ql[qlo + l] & 0xF) | ((int)(qh[qho + l] & 3) << 4)) - 32;
            int q2 = ((int)(ql[qlo + l + 32] & 0xF) | ((int)((qh[qho + l] >> 2) & 3) << 4)) - 32;
            int q3 = ((int)(ql[qlo + l] >> 4) | ((int)((qh[qho + l] >> 4) & 3) << 4)) - 32;
            int q4 = ((int)(ql[qlo + l + 32] >> 4) | ((int)((qh[qho + l] >> 6) & 3) << 4)) - 32;
            o[qp + l] = d * (float)sc[sco + is] * (float)q1;
            o[qp + 32 + l] = d * (float)sc[sco + is + 2] * (float)q2;
            o[qp + 64 + l] = d * (float)sc[sco + is + 4] * (float)q3;
            o[qp + 96 + l] = d * (float)sc[sco + is + 6] * (float)q4;
          }
          qp += 128;
        }
      }
      return;
    default:
      oc_die("quant: unsupported dequant type %d", (int)q);
  }
}
