/* Dequantization kernels. Byte/bit-faithful port of oxidize-cpp/src/quant.cpp
 * (itself ported from oxidize-core quantization.rs / ggml layouts). */
#include "oc.h"
#include "oc_iq_grids.h"

#include <math.h>
#include <float.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

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
    case OC_Q4_0: case OC_Q4_1: case OC_Q5_0: case OC_Q5_1: case OC_Q8_0:
    case OC_AL5: return QK;
    case OC_AL8: case OC_AL6: case OC_AL5_XS:
    case OC_Q2_K: case OC_Q3_K: case OC_Q4_K: case OC_Q5_K: case OC_Q6_K:
    case OC_IQ4_XS: case OC_IQ2_XXS: case OC_IQ2_XS: case OC_IQ2_S:
    case OC_IQ3_XXS: case OC_IQ3_S: return QK_K;
    case OC_IQ4_NL: return QK4_NL;
    default: oc_die("quant: no block layout (type %d)", (int)q);
  }
  return 0;
}

size_t oc_block_bytes(oc_quant q) {
  switch (q) {
    case OC_F32: return 4;
    case OC_F16: case OC_BF16: return 2;
    case OC_Q4_0: case OC_AL5: return 18;
    case OC_AL8: return 34;
    case OC_AL6: return 22;
    case OC_AL5_XS: return 14;
    case OC_Q4_1: return 20;
    case OC_Q5_0: return 22;
    case OC_Q5_1: return 24;
    case OC_Q8_0: return 34;
    case OC_Q2_K: return 84;
    case OC_Q3_K: return 110;
    case OC_Q4_K: return 144;
    case OC_Q5_K: return 176;
    case OC_Q6_K: return 210;
    case OC_IQ4_XS: return 136;
    case OC_IQ2_XXS: return 66;
    case OC_IQ2_XS: return 74;
    case OC_IQ2_S: return 82;
    case OC_IQ3_XXS: return 98;
    case OC_IQ3_S: return 110;
    case OC_IQ4_NL: return 18;
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
    case 16: return OC_IQ2_XXS;
    case 17: return OC_IQ2_XS;
    case 18: return OC_IQ3_XXS;
    case 20: return OC_IQ4_NL;
    case 21: return OC_IQ3_S;
    case 22: return OC_IQ2_S;
    case 23: return OC_IQ4_XS;
    case 30: return OC_BF16;
    case 240: return OC_AL5; /* custom, oxidize-c only */
    case 241: return OC_AL8;
    case 242: return OC_AL6;
    case 243: return OC_AL5_XS;
    default: return OC_UNKNOWN;
  }
}

const char *oc_quant_name(oc_quant q) {
  static const char *n[] = {"F32","F16","BF16","Q4_0","Q4_1","Q5_0","Q5_1",
                            "Q8_0","Q2_K","Q3_K","Q4_K","Q5_K","Q6_K","IQ4_XS",
                            "IQ2_XXS","IQ2_XS","IQ2_S","IQ3_XXS","IQ3_S","IQ4_NL",
                            "AL5","AL8","AL6","AL5_XS","?"};
  return n[q <= OC_UNKNOWN ? q : OC_UNKNOWN];
}

uint32_t oc_to_ggml_type(oc_quant q) {
  switch (q) {
    case OC_F32: return 0;
    case OC_F16: return 1;
    case OC_Q4_0: return 2;
    case OC_Q4_1: return 3;
    case OC_Q5_0: return 6;
    case OC_Q5_1: return 7;
    case OC_Q8_0: return 8;
    case OC_Q2_K: return 10;
    case OC_Q3_K: return 11;
    case OC_Q4_K: return 12;
    case OC_Q5_K: return 13;
    case OC_Q6_K: return 14;
    case OC_IQ4_XS: return 23;
    case OC_IQ2_XXS: return 16;
    case OC_IQ2_XS: return 17;
    case OC_IQ3_XXS: return 18;
    case OC_IQ4_NL: return 20;
    case OC_IQ3_S: return 21;
    case OC_IQ2_S: return 22;
    case OC_BF16: return 30;
    case OC_AL5: return 240; /* custom, oxidize-c only */
    case OC_AL8: return 241;
    case OC_AL6: return 242;
    case OC_AL5_XS: return 243;
    default: oc_die("quant: no ggml type for %d", (int)q);
  }
  return 0;
}

oc_quant oc_quant_parse(const char *name) {
  for (int q = OC_F32; q < OC_UNKNOWN; ++q)
    if (strcasecmp(name, oc_quant_name((oc_quant)q)) == 0) return (oc_quant)q;
  return OC_UNKNOWN;
}

uint16_t oc_f32_to_f16(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFF;
  if (exp >= 31) {
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) {
      uint16_t payload = (uint16_t)((mant >> 13) & 0x03FFu);
      if (payload == 0) payload = 1;
      return (uint16_t)(sign | 0x7C00u | payload);       /* NaN */
    }
    return (uint16_t)(sign | 0x7C00u);                   /* inf / overflow */
  }
  if (exp <= 0) {                                        /* subnormal / zero */
    if (exp < -10) return (uint16_t)sign;
    mant |= 0x800000;
    uint32_t shift = (uint32_t)(14 - exp);
    uint32_t half = (mant >> shift) + ((mant >> (shift - 1)) & 1); /* rne-ish */
    return (uint16_t)(sign | half);
  }
  uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
  if (mant & 0x1000) h++; /* round to nearest */
  return h;
}

static int nearest_i(float x) { return (int)lrintf(x); }
static int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
static void wr16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }

/* AL family: MSE-optimal per-block scale with multi-seed + iterative refinement. */
static float al_refine_scale(const float *x, size_t n, float d, int lo, int hi) {
  if (d == 0.0f) return 0.0f;
  float id = 1.0f / d;
  float sumlx = 0.0f, suml2 = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    int l = clampi((int)lrintf(x[i] * id), lo, hi);
    sumlx += x[i] * (float)l;
    suml2 += (float)(l * l);
  }
  return suml2 > 0.0f ? sumlx / suml2 : d;
}

static float al_refine_scale_iter(const float *x, size_t n, float d, int lo, int hi) {
  for (int it = 0; it < 4; ++it) {
    float nd = al_refine_scale(x, n, d, lo, hi);
    if (fabsf(nd - d) <= FLT_EPSILON * fmaxf(fabsf(nd), 1.0f)) return nd;
    d = nd;
  }
  return d;
}

static float al_block_mse(const float *x, size_t n, float d, int lo, int hi) {
  if (d == 0.0f) return 0.0f;
  float id = 1.0f / d, mse = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    int l = clampi((int)lrintf(x[i] * id), lo, hi);
    float err = x[i] - (float)l * d;
    mse += err * err;
  }
  return mse / (float)n;
}

static float al_best_initial_scale(const float *x, size_t n, float mx, float amax,
                                   int lo, int hi) {
  float seeds[3] = {mx / -(float)lo, amax / (float)hi, -amax / (float)lo};
  float best_d = seeds[0], best_mse = FLT_MAX;
  for (int s = 0; s < 3; ++s) {
    float seed = seeds[s];
    if (!isfinite(seed) || seed == 0.0f) continue;
    float d = al_refine_scale_iter(x, n, seed, lo, hi);
    float mse = al_block_mse(x, n, d, lo, hi);
    if (mse < best_mse) {
      best_mse = mse;
      best_d = d;
    }
  }
  return best_d;
}

static void al5_try_seed(const float *x, float seed, float *best_d, int *levels,
                         float *best_mse) {
  if (!isfinite(seed) || seed == 0.0f) return;
  int lv[QK];
  float id = 1.0f / seed;
  for (size_t i = 0; i < QK; ++i)
    lv[i] = clampi((int)lrintf(x[i] * id), -8, 7);
  float sumlx = 0.0f, suml2 = 0.0f;
  for (size_t i = 0; i < QK; ++i) {
    float l = (float)lv[i];
    sumlx += x[i] * l;
    suml2 += l * l;
  }
  float d = suml2 > 0.0f ? sumlx / suml2 : seed;
  if (fabsf(d - seed) > FLT_EPSILON * fmaxf(fabsf(d), 1.0f)) {
    id = 1.0f / d;
    for (size_t i = 0; i < QK; ++i)
      lv[i] = clampi((int)lrintf(x[i] * id), -8, 7);
  }
  float mse = 0.0f;
  for (size_t i = 0; i < QK; ++i) {
    float err = x[i] - (float)lv[i] * d;
    mse += err * err;
  }
  mse /= (float)QK;
  if (mse < *best_mse) {
    *best_mse = mse;
    *best_d = d;
    for (size_t i = 0; i < QK; ++i) levels[i] = lv[i];
  }
}

static void quantize_block_al5(const float *x, uint8_t *o) {
  float amax = 0.0f, mx = 0.0f;
  for (size_t i = 0; i < QK; ++i) {
    float a = fabsf(x[i]);
    if (a > amax) {
      amax = a;
      mx = x[i];
    }
  }
  if (amax == 0.0f) {
    wr16(o, oc_f32_to_f16(0.0f));
    memset(o + 2, 0x88, 16);
    return;
  }

  float best_d = mx / -8.0f;
  float best_mse = FLT_MAX;
  int levels[QK];
  al5_try_seed(x, mx / -8.0f, &best_d, levels, &best_mse);
  int saturated = 0;
  for (size_t i = 0; i < QK; ++i) {
    if (levels[i] == -8 || levels[i] == 7) {
      saturated = 1;
      break;
    }
  }
  if (saturated) al5_try_seed(x, amax / 7.0f, &best_d, levels, &best_mse);

  wr16(o, oc_f32_to_f16(best_d));
  for (size_t i = 0; i < QK / 2; ++i) {
    int lo = levels[i] + 8;
    int hi = levels[i + QK / 2] + 8;
    o[2 + i] = (uint8_t)((hi << 4) | lo);
  }
}
static void wr32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

/* affine range of a group: m = -min clamped >= 0, sc = (max+m)/levels */
static void group_affine(const float *x, size_t n, int levels, float *sc, float *m) {
  float mn = x[0], mx = x[0];
  for (size_t i = 1; i < n; ++i) {
    if (x[i] < mn) mn = x[i];
    if (x[i] > mx) mx = x[i];
  }
  if (mn > 0) mn = 0;
  *m = -mn;
  *sc = (mx - mn) / (float)levels;
}

static float group_amax(const float *x, size_t n) {
  float a = 0;
  for (size_t i = 0; i < n; ++i) {
    float v = fabsf(x[i]);
    if (v > a) a = v;
  }
  return a;
}

/* ggml K-quant 6-bit scale/min packing (inverse of scale_min_k4) */
static void pack_k4(uint8_t *s, const uint8_t *sc6, const uint8_t *m6) {
  for (int j = 0; j < 4; ++j) { s[j] = sc6[j]; s[j + 4] = m6[j]; }
  for (int j = 4; j < 8; ++j) {
    s[j + 4] = (uint8_t)((sc6[j] & 0xF) | ((m6[j] & 0xF) << 4));
    s[j - 4] |= (uint8_t)((sc6[j] >> 4) << 6);
    s[j] |= (uint8_t)((m6[j] >> 4) << 6);
  }
}

bool oc_quantize_row(oc_quant q, const float *src, uint8_t *dst, size_t n) {
  switch (q) {
    case OC_F32:
      memcpy(dst, src, n * sizeof(float));
      return true;
    case OC_F16:
      for (size_t i = 0; i < n; ++i) {
        uint16_t h = oc_f32_to_f16(src[i]);
        memcpy(dst + 2 * i, &h, 2);
      }
      return true;
    case OC_BF16:
      for (size_t i = 0; i < n; ++i) {
        uint32_t bits;
        memcpy(&bits, &src[i], 4);
        bits += 0x8000; /* round to nearest */
        wr16(dst + 2 * i, (uint16_t)(bits >> 16));
      }
      return true;
    case OC_Q4_1:
      for (size_t b = 0; b < n / QK; ++b) {
        const float *x = src + b * QK;
        uint8_t *o = dst + b * 20;
        float mnv = x[0], mxv = x[0];
        for (size_t i = 1; i < QK; ++i) {
          if (x[i] < mnv) mnv = x[i];
          if (x[i] > mxv) mxv = x[i];
        }
        float d = (mxv - mnv) / 15.0f;
        float id = d > 0 ? 1.0f / d : 0.0f;
        wr16(o, oc_f32_to_f16(d));
        wr16(o + 2, oc_f32_to_f16(mnv));
        for (size_t i = 0; i < 16; ++i) {
          int lo = clampi(nearest_i((x[i] - mnv) * id), 0, 15);
          int hi = clampi(nearest_i((x[i + 16] - mnv) * id), 0, 15);
          o[4 + i] = (uint8_t)(lo | (hi << 4));
        }
      }
      return true;
    case OC_Q5_0:
      for (size_t b = 0; b < n / QK; ++b) {
        const float *x = src + b * QK;
        uint8_t *o = dst + b * 22;
        float amax = 0, mx = 0;
        for (size_t i = 0; i < QK; ++i) {
          float a = fabsf(x[i]);
          if (a > amax) { amax = a; mx = x[i]; }
        }
        float d = mx / -16.0f;
        float id = d != 0 ? 1.0f / d : 0.0f;
        wr16(o, oc_f32_to_f16(d));
        uint32_t qh = 0;
        for (size_t i = 0; i < 16; ++i) {
          int q0 = clampi(nearest_i(x[i] * id) + 16, 0, 31);
          int q1 = clampi(nearest_i(x[i + 16] * id) + 16, 0, 31);
          o[6 + i] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
          qh |= (uint32_t)(q0 >> 4) << i;
          qh |= (uint32_t)(q1 >> 4) << (i + 16);
        }
        wr32(o + 2, qh);
      }
      return true;
    case OC_Q5_1:
      for (size_t b = 0; b < n / QK; ++b) {
        const float *x = src + b * QK;
        uint8_t *o = dst + b * 24;
        float mnv = x[0], mxv = x[0];
        for (size_t i = 1; i < QK; ++i) {
          if (x[i] < mnv) mnv = x[i];
          if (x[i] > mxv) mxv = x[i];
        }
        float d = (mxv - mnv) / 31.0f;
        float id = d > 0 ? 1.0f / d : 0.0f;
        wr16(o, oc_f32_to_f16(d));
        wr16(o + 2, oc_f32_to_f16(mnv));
        uint32_t qh = 0;
        for (size_t i = 0; i < 16; ++i) {
          int q0 = clampi(nearest_i((x[i] - mnv) * id), 0, 31);
          int q1 = clampi(nearest_i((x[i + 16] - mnv) * id), 0, 31);
          o[8 + i] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
          qh |= (uint32_t)(q0 >> 4) << i;
          qh |= (uint32_t)(q1 >> 4) << (i + 16);
        }
        wr32(o + 4, qh);
      }
      return true;
    /* K-quants: simple per-group min/max (affine) or amax (symmetric) fits.
     * ponytail: no llama.cpp rmse scale search — byte-layout compatible,
     * slightly higher quantization error; add the search if perplexity on
     * requantized outputs ever matters. */
    case OC_Q2_K: /* 84B: scales[16] (sc|m nibbles), qs[64], d, dmin */
      for (size_t b = 0; b < n / QK_K; ++b) {
        const float *x = src + b * QK_K;
        uint8_t *o = dst + b * 84;
        memset(o, 0, 84);
        float scf[16], mf[16], max_sc = 0, max_m = 0;
        for (int g = 0; g < 16; ++g) {
          group_affine(x + g * 16, 16, 3, &scf[g], &mf[g]);
          if (scf[g] > max_sc) max_sc = scf[g];
          if (mf[g] > max_m) max_m = mf[g];
        }
        float d = max_sc / 15.0f, dmin = max_m / 15.0f;
        wr16(o + 80, oc_f32_to_f16(d));
        wr16(o + 82, oc_f32_to_f16(dmin));
        for (int g = 0; g < 16; ++g) {
          int s4 = d > 0 ? clampi(nearest_i(scf[g] / d), 0, 15) : 0;
          int m4 = dmin > 0 ? clampi(nearest_i(mf[g] / dmin), 0, 15) : 0;
          o[g] = (uint8_t)(s4 | (m4 << 4));
          float dl = d * (float)s4, ml = dmin * (float)m4;
          float idl = dl > 0 ? 1.0f / dl : 0.0f;
          int outer = g / 8, k = (g % 8) / 2, half = g % 2;
          uint8_t *qs = o + 16 + outer * 32 + half * 16;
          for (int l = 0; l < 16; ++l) {
            int qv = clampi(nearest_i((x[g * 16 + l] + ml) * idl), 0, 3);
            qs[l] |= (uint8_t)(qv << (2 * k));
          }
        }
      }
      return true;
    case OC_Q3_K: /* 110B: hmask[32], qs[64], packed scales[12], d */
      for (size_t b = 0; b < n / QK_K; ++b) {
        const float *x = src + b * QK_K;
        uint8_t *o = dst + b * 110;
        memset(o, 0, 110);
        float dg[16], maxd = 0;
        for (int g = 0; g < 16; ++g) {
          /* grid is [-4,3]; amax/3.5 splits the clamp error across both ends */
          dg[g] = group_amax(x + g * 16, 16) / 3.5f;
          if (dg[g] > maxd) maxd = dg[g];
        }
        float d_all = maxd / 31.0f;
        wr16(o + 108, oc_f32_to_f16(d_all));
        uint8_t s6[16];
        for (int g = 0; g < 16; ++g) {
          int sc = d_all > 0 ? clampi(nearest_i(dg[g] / d_all), 0, 31) : 0;
          s6[g] = (uint8_t)(sc + 32);
          float dl = d_all * (float)sc;
          float idl = dl > 0 ? 1.0f / dl : 0.0f;
          int outer = g / 8, k = (g % 8) / 2, half = g % 2;
          uint8_t *qs = o + 32 + outer * 32 + half * 16;
          uint8_t *hm = o + half * 16;
          uint8_t hbit = (uint8_t)(1 << (outer * 4 + k));
          for (int l = 0; l < 16; ++l) {
            int qf = clampi(nearest_i(x[g * 16 + l] * idl), -4, 3) + 4; /* 0..7 */
            if (qf >= 4) hm[l] |= hbit;      /* h set => no -4 offset */
            qs[l] |= (uint8_t)((qf & 3) << (2 * k));
          }
        }
        /* pack 16 6-bit scales into 12 bytes (inverse of the loader) */
        uint32_t la0 = 0, la1 = 0, ha = 0;
        for (int i = 0; i < 4; ++i) {
          la0 |= (uint32_t)((s6[i] & 0xF) | ((s6[8 + i] & 0xF) << 4)) << (8 * i);
          la1 |= (uint32_t)((s6[4 + i] & 0xF) | ((s6[12 + i] & 0xF) << 4)) << (8 * i);
          ha |= (uint32_t)((s6[i] >> 4) | ((s6[4 + i] >> 4) << 2) |
                           ((s6[8 + i] >> 4) << 4) | ((s6[12 + i] >> 4) << 6))
                << (8 * i);
        }
        wr32(o + 96, la0);
        wr32(o + 100, la1);
        wr32(o + 104, ha);
      }
      return true;
    case OC_Q4_K: /* 144B: d, dmin, scales[12], qs[128] */
      for (size_t b = 0; b < n / QK_K; ++b) {
        const float *x = src + b * QK_K;
        uint8_t *o = dst + b * 144;
        memset(o, 0, 144);
        float scf[8], mf[8], max_sc = 0, max_m = 0;
        for (int g = 0; g < 8; ++g) {
          group_affine(x + g * 32, 32, 15, &scf[g], &mf[g]);
          if (scf[g] > max_sc) max_sc = scf[g];
          if (mf[g] > max_m) max_m = mf[g];
        }
        float d = max_sc / 63.0f, dmin = max_m / 63.0f;
        wr16(o, oc_f32_to_f16(d));
        wr16(o + 2, oc_f32_to_f16(dmin));
        uint8_t sc6[8], m6[8];
        for (int g = 0; g < 8; ++g) {
          sc6[g] = (uint8_t)(d > 0 ? clampi(nearest_i(scf[g] / d), 0, 63) : 0);
          m6[g] = (uint8_t)(dmin > 0 ? clampi(nearest_i(mf[g] / dmin), 0, 63) : 0);
        }
        pack_k4(o + 4, sc6, m6);
        for (int gp = 0; gp < 4; ++gp) {
          float d1 = d * (float)sc6[2 * gp], mn1 = dmin * (float)m6[2 * gp];
          float d2 = d * (float)sc6[2 * gp + 1], mn2 = dmin * (float)m6[2 * gp + 1];
          float id1 = d1 > 0 ? 1.0f / d1 : 0.0f, id2 = d2 > 0 ? 1.0f / d2 : 0.0f;
          uint8_t *qs = o + 16 + gp * 32;
          for (int l = 0; l < 32; ++l) {
            int lo = clampi(nearest_i((x[gp * 64 + l] + mn1) * id1), 0, 15);
            int hi = clampi(nearest_i((x[gp * 64 + 32 + l] + mn2) * id2), 0, 15);
            qs[l] = (uint8_t)(lo | (hi << 4));
          }
        }
      }
      return true;
    case OC_Q5_K: /* 176B: d, dmin, scales[12], qh[32], qs[128] */
      for (size_t b = 0; b < n / QK_K; ++b) {
        const float *x = src + b * QK_K;
        uint8_t *o = dst + b * 176;
        memset(o, 0, 176);
        float scf[8], mf[8], max_sc = 0, max_m = 0;
        for (int g = 0; g < 8; ++g) {
          group_affine(x + g * 32, 32, 31, &scf[g], &mf[g]);
          if (scf[g] > max_sc) max_sc = scf[g];
          if (mf[g] > max_m) max_m = mf[g];
        }
        float d = max_sc / 63.0f, dmin = max_m / 63.0f;
        wr16(o, oc_f32_to_f16(d));
        wr16(o + 2, oc_f32_to_f16(dmin));
        uint8_t sc6[8], m6[8];
        for (int g = 0; g < 8; ++g) {
          sc6[g] = (uint8_t)(d > 0 ? clampi(nearest_i(scf[g] / d), 0, 63) : 0);
          m6[g] = (uint8_t)(dmin > 0 ? clampi(nearest_i(mf[g] / dmin), 0, 63) : 0);
        }
        pack_k4(o + 4, sc6, m6);
        uint8_t *qh = o + 16;
        for (int g = 0; g < 4; ++g) {
          float d1 = d * (float)sc6[2 * g], mn1 = dmin * (float)m6[2 * g];
          float d2 = d * (float)sc6[2 * g + 1], mn2 = dmin * (float)m6[2 * g + 1];
          float id1 = d1 > 0 ? 1.0f / d1 : 0.0f, id2 = d2 > 0 ? 1.0f / d2 : 0.0f;
          uint8_t *qs = o + 48 + g * 32;
          for (int l = 0; l < 32; ++l) {
            int q1 = clampi(nearest_i((x[g * 64 + l] + mn1) * id1), 0, 31);
            int q2 = clampi(nearest_i((x[g * 64 + 32 + l] + mn2) * id2), 0, 31);
            qs[l] = (uint8_t)((q1 & 0xF) | ((q2 & 0xF) << 4));
            if (q1 >> 4) qh[l] |= (uint8_t)(1 << (2 * g));
            if (q2 >> 4) qh[l] |= (uint8_t)(2 << (2 * g));
          }
        }
      }
      return true;
    case OC_Q6_K: /* 210B: ql[128], qh[64], int8 scales[16], d */
      for (size_t b = 0; b < n / QK_K; ++b) {
        const float *x = src + b * QK_K;
        uint8_t *o = dst + b * 210;
        memset(o, 0, 210);
        float dg[16], maxd = 0;
        for (int g = 0; g < 16; ++g) {
          dg[g] = group_amax(x + g * 16, 16) / 31.0f;
          if (dg[g] > maxd) maxd = dg[g];
        }
        float d = maxd / 127.0f;
        wr16(o + 208, oc_f32_to_f16(d));
        int8_t *sc = (int8_t *)(o + 192);
        for (int g = 0; g < 16; ++g)
          sc[g] = (int8_t)(d > 0 ? clampi(nearest_i(dg[g] / d), 0, 127) : 0);
        for (int g2 = 0; g2 < 2; ++g2) {
          uint8_t *ql = o + g2 * 64, *qh = o + 128 + g2 * 32;
          for (int l = 0; l < 32; ++l) {
            int L[4];
            for (int off = 0; off < 4; ++off) {
              int gidx = g2 * 8 + off * 2 + l / 16;
              float dl = d * (float)sc[gidx];
              float idl = dl > 0 ? 1.0f / dl : 0.0f;
              float v = x[g2 * 128 + off * 32 + l];
              L[off] = clampi(nearest_i(v * idl), -32, 31) + 32; /* 0..63 */
            }
            ql[l] = (uint8_t)((L[0] & 0xF) | ((L[2] & 0xF) << 4));
            ql[l + 32] = (uint8_t)((L[1] & 0xF) | ((L[3] & 0xF) << 4));
            qh[l] = (uint8_t)((L[0] >> 4) | ((L[1] >> 4) << 2) |
                              ((L[2] >> 4) << 4) | ((L[3] >> 4) << 6));
          }
        }
      }
      return true;
    case OC_IQ4_XS: { /* 136B: d, scales_h u16, scales_l[4], qs[128] */
      static const int8_t kv16[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                      1, 13, 25, 38, 53, 69, 89, 113};
      for (size_t b = 0; b < n / QK_K; ++b) {
        const float *x = src + b * QK_K;
        uint8_t *o = dst + b * 136;
        memset(o, 0, 136);
        float dg[8], maxd = 0;
        for (int g = 0; g < 8; ++g) {
          dg[g] = group_amax(x + g * 32, 32) / 127.0f;
          if (dg[g] > maxd) maxd = dg[g];
        }
        float d = maxd / 31.0f;
        wr16(o, oc_f32_to_f16(d));
        uint16_t sh = 0;
        for (int g = 0; g < 8; ++g) {
          int ls = d > 0 ? clampi(nearest_i(dg[g] / d), 0, 31) : 0;
          int stored = ls + 32; /* 6-bit, dequant subtracts 32 */
          o[4 + g / 2] |= (uint8_t)((stored & 0xF) << (4 * (g % 2)));
          sh |= (uint16_t)((stored >> 4) << (2 * g));
          float dl = d * (float)ls;
          float idl = dl > 0 ? 1.0f / dl : 0.0f;
          uint8_t *qs = o + 8 + g * 16;
          for (int l = 0; l < 16; ++l) {
            int bi = 0, bj = 0;
            float t0 = x[g * 32 + l] * idl, t1 = x[g * 32 + 16 + l] * idl;
            float e0 = 1e30f, e1 = 1e30f;
            for (int k = 0; k < 16; ++k) {
              float d0 = fabsf(t0 - (float)kv16[k]);
              float d1 = fabsf(t1 - (float)kv16[k]);
              if (d0 < e0) { e0 = d0; bi = k; }
              if (d1 < e1) { e1 = d1; bj = k; }
            }
            qs[l] = (uint8_t)(bi | (bj << 4));
          }
        }
        wr16(o + 2, sh);
      }
      return true;
    }
    case OC_Q8_0: /* per-32 block: f16 d + 32 int8, v = d*q */
      for (size_t b = 0; b < n / QK; ++b) {
        const float *x = src + b * QK;
        uint8_t *o = dst + b * 34;
        float amax = 0;
        for (size_t i = 0; i < QK; ++i) {
          float a = fabsf(x[i]);
          if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float id = d > 0 ? 1.0f / d : 0.0f;
        uint16_t dh = oc_f32_to_f16(d);
        memcpy(o, &dh, 2);
        for (size_t i = 0; i < QK; ++i) {
          int v = (int)lrintf(x[i] * id);
          if (v > 127) v = 127;
          if (v < -128) v = -128;
          ((int8_t *)(o + 2))[i] = (int8_t)v;
        }
      }
      return true;
    case OC_Q4_0: /* per-32 block: f16 d + 16 nibble bytes, v = d*(nib-8) */
      for (size_t b = 0; b < n / QK; ++b) {
        const float *x = src + b * QK;
        uint8_t *o = dst + b * 18;
        float amax = 0, mx = 0;
        for (size_t i = 0; i < QK; ++i) {
          float a = fabsf(x[i]);
          if (a > amax) { amax = a; mx = x[i]; }
        }
        float d = mx / -8.0f;
        float id = d != 0 ? 1.0f / d : 0.0f;
        uint16_t dh = oc_f32_to_f16(d);
        memcpy(o, &dh, 2);
        for (size_t i = 0; i < QK / 2; ++i) {
          int lo = (int)(x[i] * id + 8.5f);
          int hi = (int)(x[i + QK / 2] * id + 8.5f);
          if (lo > 15) lo = 15;
          if (lo < 0) lo = 0;
          if (hi > 15) hi = 15;
          if (hi < 0) hi = 0;
          o[2 + i] = (uint8_t)((hi << 4) | lo);
        }
      }
      return true;
    case OC_AL5:
      for (size_t b = 0; b < n / QK; ++b)
        quantize_block_al5(src + b * QK, dst + b * 18);
      return true;
    case OC_AL8:
      return oc_quantize_row(OC_Q8_0, src, dst, n);
    case OC_AL6:
      return oc_quantize_row(OC_Q5_0, src, dst, n);
    default:
      return false;
  }
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
    case OC_AL5: /* same bitstream as Q4_0: (nib-8)*d */
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
    case OC_AL6:
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
    case OC_AL8:
    case OC_AL5_XS:
      for (b = 0; b < n / QK; ++b) {
        const uint8_t *blk = src + b * 14;
        float d = oc_f16_to_f32(blk);
        float *o = dst + b * QK;
        uint32_t bitpos = 0;
        for (i = 0; i < QK; ++i) {
          uint8_t lvl = 0;
          for (int bit = 0; bit < 3; ++bit) {
            size_t byte_idx = (size_t)(bitpos / 8);
            size_t bit_idx = (size_t)(bitpos % 8);
            if ((blk[2 + byte_idx] >> bit_idx) & 1) lvl |= (uint8_t)(1u << bit);
            ++bitpos;
          }
          o[i] = (float)((int)lvl - 4) * d;
        }
      }
      return;
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
    case OC_IQ4_XS:
      /* 136-byte superblocks: fp16 d, u16 scales_h, u8 scales_l[4], qs[128].
       * value = d * (6-bit scale - 32) * kvalues[nibble] (ggml block_iq4_xs). */
      for (b = 0; b < n / QK_K; ++b) {
        static const int8_t kv16[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                        1, 13, 25, 38, 53, 69, 89, 113};
        const uint8_t *blk = src + b * 136;
        float d = oc_f16_to_f32(blk);
        uint16_t sh = rd16(blk + 2);
        const uint8_t *sl = blk + 4, *qs = blk + 8;
        float *o = dst + b * QK_K;
        for (int ib = 0; ib < 8; ++ib) {
          int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4);
          float dl = d * (float)(ls - 32);
          for (l = 0; l < 16; ++l) {
            o[l] = dl * (float)kv16[qs[l] & 0xF];
            o[l + 16] = dl * (float)kv16[qs[l] >> 4];
          }
          o += 32;
          qs += 16;
        }
      }
      return;
    case OC_IQ4_NL:
      for (b = 0; b < n / QK4_NL; ++b) {
        const uint8_t *blk = src + b * 18;
        float d = oc_f16_to_f32(blk);
        const uint8_t *qs = blk + 2;
        float *o = dst + b * QK4_NL;
        for (l = 0; l < QK4_NL / 2; ++l) {
          o[l] = d * (float)oc_kvalues_iq4nl[qs[l] & 0xF];
          o[l + QK4_NL / 2] = d * (float)oc_kvalues_iq4nl[qs[l] >> 4];
        }
      }
      return;
    case OC_IQ2_XXS:
      for (b = 0; b < n / QK_K; ++b) {
        const uint8_t *blk = src + b * 66;
        float d = oc_f16_to_f32(blk);
        const uint8_t *qs = blk + 2;
        float *y = dst + b * QK_K;
        for (size_t ib32 = 0; ib32 < QK_K / 32; ++ib32) {
          uint32_t aux0 = rd32(qs + 8 * ib32);
          uint32_t aux1 = rd32(qs + 8 * ib32 + 4);
          const uint8_t *aux8 = (const uint8_t *)&aux0;
          float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
          for (size_t ll = 0; ll < 4; ++ll) {
            uint64_t grid_u = oc_iq2xxs_grid[aux8[ll]];
            const uint8_t *grid = (const uint8_t *)&grid_u;
            uint8_t signs = oc_ksigns_iq2xs[(aux1 >> (7 * ll)) & 127];
            for (size_t j = 0; j < 8; ++j)
              y[j] = db * (float)grid[j] *
                     ((signs & oc_kmask_iq2xs[j]) ? -1.0f : 1.0f);
            y += 8;
          }
        }
      }
      return;
    case OC_IQ2_XS:
      for (b = 0; b < n / QK_K; ++b) {
        const uint8_t *blk = src + b * 74;
        float d = oc_f16_to_f32(blk);
        const uint8_t *scales = blk + 2 + QK_K / 4;
        float *y = dst + b * QK_K;
        size_t out_ptr = 0;
        for (size_t ib32 = 0; ib32 < QK_K / 32; ++ib32) {
          uint8_t sc = scales[ib32];
          float db[2] = {d * (0.5f + (float)(sc & 0xf)) * 0.25f,
                         d * (0.5f + (float)(sc >> 4)) * 0.25f};
          for (size_t ll = 0; ll < 4; ++ll) {
            size_t qs_off = 2 + 2 * (4 * ib32 + ll);
            uint16_t qs_val = rd16(blk + qs_off);
            uint64_t grid_u = oc_iq2xs_grid[qs_val & 511];
            const uint8_t *grid = (const uint8_t *)&grid_u;
            uint8_t signs = oc_ksigns_iq2xs[qs_val >> 9];
            float dl = db[ll / 2];
            for (size_t j = 0; j < 8; ++j)
              y[out_ptr + j] = dl * (float)grid[j] *
                               ((signs & oc_kmask_iq2xs[j]) ? -1.0f : 1.0f);
            out_ptr += 8;
          }
        }
      }
      return;
    case OC_IQ2_S:
      for (b = 0; b < n / QK_K; ++b) {
        const uint8_t *blk = src + b * 82;
        float d = oc_f16_to_f32(blk);
        const uint8_t *qs = blk + 2;
        const uint8_t *qh = blk + 2 + QK_K / 4;
        const uint8_t *scales = blk + 2 + QK_K / 4 + QK_K / 32;
        float *y = dst + b * QK_K;
        size_t qs_ptr = 0, signs_ptr = QK_K / 8, out_ptr = 0;
        for (size_t ib32 = 0; ib32 < QK_K / 32; ++ib32) {
          uint8_t sc = scales[ib32];
          float db[2] = {d * (0.5f + (float)(sc & 0xf)) * 0.25f,
                         d * (0.5f + (float)(sc >> 4)) * 0.25f};
          for (size_t ll = 0; ll < 4; ++ll) {
            float dl = db[ll / 2];
            size_t grid_idx = (size_t)qs[qs_ptr + ll] |
                              (((size_t)qh[ib32] << (8 - 2 * ll)) & 0x300);
            uint64_t grid_u = oc_iq2s_grid[grid_idx];
            const uint8_t *grid = (const uint8_t *)&grid_u;
            uint8_t signs = qs[signs_ptr + ll];
            for (size_t j = 0; j < 8; ++j)
              y[out_ptr + j] = dl * (float)grid[j] *
                               ((signs & oc_kmask_iq2xs[j]) ? -1.0f : 1.0f);
            out_ptr += 8;
          }
          qs_ptr += 4;
          signs_ptr += 4;
        }
      }
      return;
    case OC_IQ3_XXS:
      for (b = 0; b < n / QK_K; ++b) {
        const uint8_t *blk = src + b * 98;
        float d = oc_f16_to_f32(blk);
        const uint8_t *qs = blk + 2;
        const uint8_t *scales_and_signs = qs + QK_K / 4;
        float *y = dst + b * QK_K;
        for (size_t ib32 = 0; ib32 < QK_K / 32; ++ib32) {
          uint32_t aux32 = rd32(scales_and_signs + 4 * ib32);
          float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
          for (size_t ll = 0; ll < 4; ++ll) {
            uint8_t signs = oc_ksigns_iq2xs[(aux32 >> (7 * ll)) & 127];
            uint32_t g1 = oc_iq3xxs_grid[qs[2 * ll + 0]];
            uint32_t g2 = oc_iq3xxs_grid[qs[2 * ll + 1]];
            const uint8_t *grid1 = (const uint8_t *)&g1;
            const uint8_t *grid2 = (const uint8_t *)&g2;
            for (size_t j = 0; j < 4; ++j) {
              y[j] = db * (float)grid1[j] *
                     ((signs & oc_kmask_iq2xs[j]) ? -1.0f : 1.0f);
              y[j + 4] = db * (float)grid2[j] *
                         ((signs & oc_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
            }
            y += 8;
          }
          qs += 8;
        }
      }
      return;
    case OC_IQ3_S:
      for (b = 0; b < n / QK_K; ++b) {
        const uint8_t *blk = src + b * 110;
        float d = oc_f16_to_f32(blk);
        const uint8_t *qs = blk + 2;
        const uint8_t *qh = blk + 66;
        const uint8_t *signs = blk + 74;
        const uint8_t *scales = blk + 106;
        float *o = dst + b * QK_K;
        size_t qs_o = 0, qh_o = 0, sg_o = 0, y = 0, ib32 = 0;
        while (ib32 < QK_K / 32) {
          float db1 = d * (float)(1 + 2 * (scales[ib32 / 2] & 0xf));
          float db2 = d * (float)(1 + 2 * (scales[ib32 / 2] >> 4));
          for (size_t ll = 0; ll < 4; ++ll) {
            size_t h = qh[qh_o];
            size_t i1 = (size_t)qs[qs_o + 2 * ll] | ((h << (8 - 2 * ll)) & 256);
            size_t i2 = (size_t)qs[qs_o + 2 * ll + 1] | ((h << (7 - 2 * ll)) & 256);
            uint8_t s = signs[sg_o + ll];
            for (size_t j = 0; j < 4; ++j) {
              float f1 = (s & oc_kmask_iq2xs[j]) ? -1.0f : 1.0f;
              float f2 = (s & oc_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f;
              o[y + j] = db1 * (float)((oc_iq3s_grid[i1] >> (8 * j)) & 0xff) * f1;
              o[y + j + 4] = db1 * (float)((oc_iq3s_grid[i2] >> (8 * j)) & 0xff) * f2;
            }
            y += 8;
          }
          qs_o += 8;
          sg_o += 4;
          for (size_t ll = 0; ll < 4; ++ll) {
            size_t h = qh[qh_o + 1];
            size_t i1 = (size_t)qs[qs_o + 2 * ll] | ((h << (8 - 2 * ll)) & 256);
            size_t i2 = (size_t)qs[qs_o + 2 * ll + 1] | ((h << (7 - 2 * ll)) & 256);
            uint8_t s = signs[sg_o + ll];
            for (size_t j = 0; j < 4; ++j) {
              float f1 = (s & oc_kmask_iq2xs[j]) ? -1.0f : 1.0f;
              float f2 = (s & oc_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f;
              o[y + j] = db2 * (float)((oc_iq3s_grid[i1] >> (8 * j)) & 0xff) * f1;
              o[y + j + 4] = db2 * (float)((oc_iq3s_grid[i2] >> (8 * j)) & 0xff) * f2;
            }
            y += 8;
          }
          qh_o += 2;
          qs_o += 8;
          sg_o += 4;
          ib32 += 2;
        }
      }
      return;
    default:
      oc_die("quant: unsupported dequant type %d", (int)q);
  }
}
