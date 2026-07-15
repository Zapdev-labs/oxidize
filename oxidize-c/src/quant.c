#include "quant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "quant_impl.h"
#include "quant_iq_grids.h"

float oc_f16_to_f32(uint16_t bits) {
  uint32_t sign = ((uint32_t)bits >> 15) & 1;
  uint32_t exp = ((uint32_t)bits >> 10) & 0x1F;
  uint32_t frac = (uint32_t)bits & 0x3FF;
  uint32_t f32;
  if (exp == 0) {
    if (frac == 0) {
      f32 = sign << 31;
    } else {
      int e = -14;
      while ((frac & 0x400) == 0) {
        frac <<= 1;
        e -= 1;
      }
      frac &= 0x3FF;
      f32 = (sign << 31) | ((uint32_t)(e + 127) << 23) | (frac << 13);
    }
  } else if (exp == 0x1F) {
    f32 = (sign << 31) | 0x7F800000u | (frac << 13);
  } else {
    f32 = (sign << 31) | ((uint32_t)((int)exp - 15 + 127) << 23) | (frac << 13);
  }
  float v;
  memcpy(&v, &f32, 4);
  return v;
}

uint16_t oc_f32_to_f16(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  uint32_t e = (x >> 23) & 0xff;
  uint32_t mant = x & 0x7fffffu;
  if (e == 0xff) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
  int32_t exp = (int32_t)e - 127 + 15;
  if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;
    mant |= 0x800000u;
    int shift = 14 - exp;
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1u);
    uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (half & 1u))) half++;
    return (uint16_t)(sign | half);
  }
  uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
  uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
  return h;
}

/* ---- AL5_XS (type 243): 32 weights / 14 bytes ---------------------------- */

float al5xs_lut(unsigned code, float scale) {
  /* w = (code - 4) * scale for 3-bit codes 0..7 (verified vs BF16 source, see evidence/al5xs-decode.md).
   * NOTE: this mapping is being verified empirically against reference output
   * and may be adjusted (e.g. non-uniform codebook). Keep it here only. */
  return ((float)code - 4.0f) * scale;
}

void al5xs_unpack(const uint8_t qs[12], uint8_t codes[32]) {
  /* LSB-first: code i = bits [3i, 3i+3) of the 96-bit LE bitstream.
   * Isolated so the packing order can be flipped if empirically wrong. */
  for (int i = 0; i < 32; ++i) {
    unsigned bit = (unsigned)(3 * i);
    unsigned byte = bit >> 3, off = bit & 7;
    unsigned v = qs[byte] >> off;
    if (off > 5) v |= (unsigned)qs[byte + 1] << (8 - off);
    codes[i] = (uint8_t)(v & 7);
  }
}

void al5xs_pack(const uint8_t codes[32], uint8_t qs[12]) {
  memset(qs, 0, 12);
  for (int i = 0; i < 32; ++i) {
    unsigned bit = (unsigned)(3 * i);
    unsigned byte = bit >> 3, off = bit & 7;
    unsigned v = codes[i] & 7u;
    qs[byte] |= (uint8_t)(v << off);
    if (off > 5) qs[byte + 1] |= (uint8_t)(v >> (8 - off));
  }
}

/* Encoder: pick scale d minimizing sum (w - (q-4)*d)^2 with
 * q = clamp(round(w/d + 4), 0, 7). Search 31 candidates around max|w|/4 plus
 * an alternating least-squares fit, then re-derive codes from the f16-rounded
 * winner so the emitted block is exactly what a decoder will see. */
static float al5xs_mse(const float w[32], float d, uint8_t codes[32]) {
  float mse = 0.0f;
  for (int i = 0; i < 32; ++i) {
    int q = d > 0.0f ? (int)lrintf(w[i] / d + 4.0f) : 4;
    if (q < 0) q = 0;
    if (q > 7) q = 7;
    codes[i] = (uint8_t)q;
    float e = w[i] - ((float)q - 4.0f) * d;
    mse += e * e;
  }
  return mse;
}

void oc_al5xs_encode_block(const float w[32], uint8_t out[14]) {
  float amax = 0.0f;
  for (int i = 0; i < 32; ++i) {
    float a = fabsf(w[i]);
    if (a > amax) amax = a;
  }
  float best_d = 0.0f;
  uint8_t codes[32];
  for (int i = 0; i < 32; ++i) codes[i] = 4;
  if (amax > 0.0f) {
    float d0 = amax / 4.0f;
    float best_mse = al5xs_mse(w, d0, codes);
    best_d = d0;
    for (int k = 0; k < 31; ++k) {
      float d = d0 * (0.8f + 0.4f * (float)k / 30.0f);
      float m = al5xs_mse(w, d, codes);
      if (m < best_mse) { best_mse = m; best_d = d; }
    }
    /* Alternating fit: q from d, then d = <w,q-4>/<q-4,q-4>. */
    float d = best_d;
    for (int it = 0; it < 5; ++it) {
      al5xs_mse(w, d, codes);
      float num = 0.0f, den = 0.0f;
      for (int i = 0; i < 32; ++i) {
        float q = (float)codes[i] - 4.0f;
        num += w[i] * q;
        den += q * q;
      }
      if (den <= 0.0f || num <= 0.0f) break;
      d = num / den;
      float m = al5xs_mse(w, d, codes);
      if (m < best_mse) { best_mse = m; best_d = d; }
    }
  }
  if (!isfinite(best_d)) best_d = 0.0f; /* never emit non-finite scales */
  uint16_t h = oc_f32_to_f16(best_d);
  float dq = oc_f16_to_f32(h); /* decoder sees the f16 value; fit codes to it */
  if (dq > 0.0f) {
    al5xs_mse(w, dq, codes);
  } else {
    h = 0;
    for (int i = 0; i < 32; ++i) codes[i] = 4;
  }
  out[0] = (uint8_t)(h & 0xff);
  out[1] = (uint8_t)(h >> 8);
  al5xs_pack(codes, out + 2);
}

static void dequant_al5xs_block(const uint8_t* blk, float* o) {
  float scale = al5xs_scale(blk);
  uint8_t codes[32];
  al5xs_unpack(blk + 2, codes);
  for (int i = 0; i < 32; ++i) o[i] = al5xs_lut(codes[i], scale);
}

/* ---- K-quant block dequant ------------------------------------------------ */

static void dequant_q4_k_block(const uint8_t* block, float* o) {
  float d = oc_f16_to_f32(rd16(block));
  float min = oc_f16_to_f32(rd16(block + 2));
  const uint8_t* scales = block + 4;
  const uint8_t* qs = block + 16;
  size_t out_ptr = 0, is = 0;
  for (int gp = 0; gp < 4; ++gp) {
    size_t q_base = (size_t)gp * 32;
    uint8_t sc1, m1, sc2, m2;
    get_scale_min_k4(is, scales, &sc1, &m1);
    get_scale_min_k4(is + 1, scales, &sc2, &m2);
    float d1 = d * sc1, min1 = min * m1, d2 = d * sc2, min2 = min * m2;
    for (size_t l = 0; l < 32; ++l) o[out_ptr + l] = d1 * (float)(qs[q_base + l] & 0xF) - min1;
    for (size_t l = 0; l < 32; ++l) o[out_ptr + 32 + l] = d2 * (float)(qs[q_base + l] >> 4) - min2;
    out_ptr += 64;
    is += 2;
  }
}

static void dequant_q5_k_block(const uint8_t* block, float* o) {
  float d = oc_f16_to_f32(rd16(block));
  float min = oc_f16_to_f32(rd16(block + 2));
  const uint8_t* scales = block + 4;
  const uint8_t* qh = block + 16;
  const uint8_t* ql = block + 48;
  size_t is = 0;
  uint8_t u1 = 1, u2 = 2;
  for (int gp = 0; gp < 4; ++gp) {
    uint8_t sc1, m1, sc2, m2;
    get_scale_min_k4(is, scales, &sc1, &m1);
    get_scale_min_k4(is + 1, scales, &sc2, &m2);
    float d1 = d * sc1, min1 = min * m1, d2 = d * sc2, min2 = min * m2;
    for (size_t l = 0; l < 32; ++l)
      *o++ = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - min1;
    for (size_t l = 0; l < 32; ++l)
      *o++ = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - min2;
    ql += 32;
    is += 2;
    u1 <<= 2;
    u2 <<= 2;
  }
}

static void dequant_q6_k_block(const uint8_t* block, float* o) {
  float d = oc_f16_to_f32(rd16(block + 208));
  const uint8_t* ql = block;
  const uint8_t* qh = block + 128;
  const int8_t* sc = (const int8_t*)(block + 192);
  size_t q_ptr = 0;
  for (int group = 0; group < 2; ++group) {
    size_t ql_off = (size_t)group * 64, qh_off = (size_t)group * 32, sc_off = (size_t)group * 8;
    for (size_t l = 0; l < 32; ++l) {
      size_t is = l / 16;
      int q1 = ((int)(ql[ql_off + l] & 0xF) | ((int)(qh[qh_off + l] & 3) << 4)) - 32;
      int q2 = ((int)(ql[ql_off + l + 32] & 0xF) | ((int)((qh[qh_off + l] >> 2) & 3) << 4)) - 32;
      int q3 = ((int)(ql[ql_off + l] >> 4) | ((int)((qh[qh_off + l] >> 4) & 3) << 4)) - 32;
      int q4 = ((int)(ql[ql_off + l + 32] >> 4) | ((int)((qh[qh_off + l] >> 6) & 3) << 4)) - 32;
      o[q_ptr + l] = d * sc[sc_off + is] * (float)q1;
      o[q_ptr + 32 + l] = d * sc[sc_off + is + 2] * (float)q2;
      o[q_ptr + 64 + l] = d * sc[sc_off + is + 4] * (float)q3;
      o[q_ptr + 96 + l] = d * sc[sc_off + is + 6] * (float)q4;
    }
    q_ptr += 128;
  }
}

/* ---- simple legacy quants (32 values/block, ggml split j/j+16 order) ------- */
/* rd32 now lives in quant_impl.h (also used by the Q3_K AVX2 kernel). */

static void dequant_q4_1_block(const uint8_t* blk, float* o) {
  float d = oc_f16_to_f32(rd16(blk));
  float m = oc_f16_to_f32(rd16(blk + 2));
  const uint8_t* qs = blk + 4;
  for (int j = 0; j < 16; ++j) {
    o[j] = (float)(qs[j] & 0xF) * d + m;
    o[j + 16] = (float)(qs[j] >> 4) * d + m;
  }
}

static void dequant_q5_0_block(const uint8_t* blk, float* o) {
  float d = oc_f16_to_f32(rd16(blk));
  uint32_t qh = rd32(blk + 2);
  const uint8_t* qs = blk + 6;
  for (int j = 0; j < 16; ++j) {
    int xh0 = (int)((qh >> j) & 1u) << 4;
    int xh1 = (int)((qh >> (j + 16)) & 1u) << 4;
    o[j] = (float)(((int)(qs[j] & 0xF) | xh0) - 16) * d;
    o[j + 16] = (float)(((int)(qs[j] >> 4) | xh1) - 16) * d;
  }
}

static void dequant_q5_1_block(const uint8_t* blk, float* o) {
  float d = oc_f16_to_f32(rd16(blk));
  float m = oc_f16_to_f32(rd16(blk + 2));
  uint32_t qh = rd32(blk + 4);
  const uint8_t* qs = blk + 8;
  for (int j = 0; j < 16; ++j) {
    int xh0 = (int)((qh >> j) & 1u) << 4;
    int xh1 = (int)((qh >> (j + 16)) & 1u) << 4;
    o[j] = (float)((int)(qs[j] & 0xF) | xh0) * d + m;
    o[j + 16] = (float)((int)(qs[j] >> 4) | xh1) * d + m;
  }
}

/* ---- Q2_K / Q3_K / IQ4_XS block dequant (ggml layout) --------------------- */

static void dequant_q2_k_block(const uint8_t* block, float* o) {
  const uint8_t* scales = block;      /* 16 */
  const uint8_t* qs = block + 16;     /* 64 */
  float d = oc_f16_to_f32(rd16(block + 80));
  float dmin = oc_f16_to_f32(rd16(block + 82));
  size_t y = 0, is = 0;
  for (int n = 0; n < 256; n += 128) {
    const uint8_t* q = qs + (size_t)(n / 128) * 32;
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
      uint8_t sc = scales[is++];
      float dl = d * (float)(sc & 0xF), ml = dmin * (float)(sc >> 4);
      for (int l = 0; l < 16; ++l)
        o[y + l] = dl * (float)((q[l] >> shift) & 3) - ml;
      sc = scales[is++];
      dl = d * (float)(sc & 0xF);
      ml = dmin * (float)(sc >> 4);
      for (int l = 0; l < 16; ++l)
        o[y + 16 + l] = dl * (float)((q[l + 16] >> shift) & 3) - ml;
      y += 32;
      shift += 2;
    }
  }
}

static void dequant_q3_k_block(const uint8_t* block, float* o) {
  const uint8_t* hm = block;          /* 32 hmask */
  const uint8_t* qs = block + 32;     /* 64 */
  float d_all = oc_f16_to_f32(rd16(block + 108));
  /* Unpack the 12 packed bytes into 16 signed 6-bit scales (ggml scheme). */
  uint32_t aux[4];
  for (int k = 0; k < 3; ++k) aux[k] = rd32(block + 96 + 4 * k);
  uint32_t tmp = aux[2];
  aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
  aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
  aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);
  aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);
  int8_t sb[16];
  for (int k = 0; k < 4; ++k)
    for (int b = 0; b < 4; ++b) sb[k * 4 + b] = (int8_t)((aux[k] >> (8 * b)) & 0xff);
  size_t y = 0, is = 0;
  uint8_t m = 1;
  for (int n = 0; n < 256; n += 128) {
    const uint8_t* q = qs + (size_t)(n / 128) * 32;
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
      float dl = d_all * (float)((int)sb[is++] - 32);
      for (int l = 0; l < 16; ++l) {
        int qv = (q[l] >> shift) & 3, hbit = (hm[l] & m) ? 0 : 4;
        o[y + l] = dl * (float)(qv - hbit);
      }
      dl = d_all * (float)((int)sb[is++] - 32);
      for (int l = 0; l < 16; ++l) {
        int qv = (q[l + 16] >> shift) & 3, hbit = (hm[l + 16] & m) ? 0 : 4;
        o[y + 16 + l] = dl * (float)(qv - hbit);
      }
      y += 32;
      shift += 2;
      m <<= 1;
    }
  }
}

/* IQ4_NL 16-value nonlinear codebook, shared by IQ4_XS (ggml-common.h). */
static const int8_t OC_KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static void dequant_iq4_xs_block(const uint8_t* block, float* o) {
  float d = oc_f16_to_f32(rd16(block));
  uint16_t scales_h = rd16(block + 2);
  const uint8_t* scales_l = block + 4; /* 4 */
  const uint8_t* qs = block + 8;       /* 128 */
  for (int ib = 0; ib < 8; ++ib) {
    int ls_l = (scales_l[ib / 2] >> (4 * (ib & 1))) & 0xf;
    int ls_h = (int)((scales_h >> (2 * ib)) & 3) << 4;
    float dl = d * (float)((ls_l | ls_h) - 32);
    const uint8_t* q = qs + (size_t)ib * 16;
    float* ob = o + (size_t)ib * 32;
    for (int j = 0; j < 16; ++j) {
      ob[j] = dl * (float)OC_KVALUES_IQ4NL[q[j] & 0xf];
      ob[j + 16] = dl * (float)OC_KVALUES_IQ4NL[q[j] >> 4];
    }
  }
}

/* ---- IQ grid-codebook block dequant (ggml ggml-quants.c layout) -----------
 * Scalar only: each value is a byte gathered from a 2-16 KB codebook grid, an
 * optional sign flip, and a per-32 scale — a gather that does not vectorize
 * into anything that beats the scalar loop, so there is deliberately no AVX2
 * path (grids live in quant_iq_grids.h, copied verbatim from ggml). The sign
 * tables oc_ksigns_iq2xs / oc_kmask_iq2xs and the grids all come from there.
 * IQ1S_DELTA is the fixed ±0.125 codebook offset shared by IQ1_S and IQ1_M. */
#define OC_IQ1_DELTA 0.125f

static void dequant_iq2_xxs_block(const uint8_t* block, float* o) {
  float d = oc_f16_to_f32(rd16(block));
  const uint8_t* qs = block + 2; /* 32 u16 */
  for (int ib32 = 0; ib32 < 8; ++ib32) {
    const uint8_t* q8 = qs + 8 * ib32; /* aux8[0..3] = grid idx, aux32[1] next */
    uint32_t a1 = rd32(q8 + 4);
    float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
    for (int l = 0; l < 4; ++l) {
      const uint8_t* grid = (const uint8_t*)&oc_iq2xxs_grid[q8[l]];
      uint8_t signs = oc_ksigns_iq2xs[(a1 >> (7 * l)) & 127];
      for (int j = 0; j < 8; ++j)
        o[j] = db * (float)grid[j] * (signs & oc_kmask_iq2xs[j] ? -1.f : 1.f);
      o += 8;
    }
  }
}

static void dequant_iq2_xs_block(const uint8_t* block, float* o) {
  float d = oc_f16_to_f32(rd16(block));
  const uint8_t* qs = block + 2;        /* 32 u16 */
  const uint8_t* scales = block + 2 + 64; /* 8 */
  for (int ib32 = 0; ib32 < 8; ++ib32) {
    float db[2];
    db[0] = d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f;
    db[1] = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;
    for (int l = 0; l < 4; ++l) {
      uint16_t q = rd16(qs + 2 * (4 * ib32 + l));
      const uint8_t* grid = (const uint8_t*)&oc_iq2xs_grid[q & 511];
      uint8_t signs = oc_ksigns_iq2xs[q >> 9];
      float dl = db[l / 2];
      for (int j = 0; j < 8; ++j)
        o[j] = dl * (float)grid[j] * (signs & oc_kmask_iq2xs[j] ? -1.f : 1.f);
      o += 8;
    }
  }
}

static void dequant_iq2_s_block(const uint8_t* block, float* o) {
  float d = oc_f16_to_f32(rd16(block));
  const uint8_t* qs = block + 2;          /* 64: low grid idx (32) + signs (32) */
  const uint8_t* qh = block + 2 + 64;     /* 8 */
  const uint8_t* scales = block + 2 + 72; /* 8 */
  const uint8_t* signs = qs + 32;         /* second half of the qs region */
  for (int ib32 = 0; ib32 < 8; ++ib32) {
    float db[2];
    db[0] = d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f;
    db[1] = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;
    const uint8_t* qb = qs + 4 * ib32;
    const uint8_t* sb = signs + 4 * ib32;
    for (int l = 0; l < 4; ++l) {
      float dl = db[l / 2];
      unsigned idx = qb[l] | (((unsigned)qh[ib32] << (8 - 2 * l)) & 0x300);
      const uint8_t* grid = (const uint8_t*)&oc_iq2s_grid[idx];
      for (int j = 0; j < 8; ++j)
        o[j] = dl * (float)grid[j] * (sb[l] & oc_kmask_iq2xs[j] ? -1.f : 1.f);
      o += 8;
    }
  }
}

static void dequant_iq3_xxs_block(const uint8_t* block, float* o) {
  float d = oc_f16_to_f32(rd16(block));
  const uint8_t* qs = block + 2;           /* 64 */
  const uint8_t* sas = block + 2 + 64;     /* 32: scales+signs, 8 u32 */
  for (int ib32 = 0; ib32 < 8; ++ib32) {
    uint32_t a = rd32(sas + 4 * ib32);
    float db = d * (0.5f + (float)(a >> 28)) * 0.5f;
    const uint8_t* qb = qs + 8 * ib32;
    for (int l = 0; l < 4; ++l) {
      uint8_t signs = oc_ksigns_iq2xs[(a >> (7 * l)) & 127];
      const uint8_t* g1 = (const uint8_t*)&oc_iq3xxs_grid[qb[2 * l + 0]];
      const uint8_t* g2 = (const uint8_t*)&oc_iq3xxs_grid[qb[2 * l + 1]];
      for (int j = 0; j < 4; ++j) {
        o[j + 0] = db * (float)g1[j] * (signs & oc_kmask_iq2xs[j + 0] ? -1.f : 1.f);
        o[j + 4] = db * (float)g2[j] * (signs & oc_kmask_iq2xs[j + 4] ? -1.f : 1.f);
      }
      o += 8;
    }
  }
}

static void dequant_iq3_s_block(const uint8_t* block, float* o) {
  float d = oc_f16_to_f32(rd16(block));
  const uint8_t* qs = block + 2;            /* 64 */
  const uint8_t* qh = block + 2 + 64;       /* 8 */
  const uint8_t* signs = block + 2 + 72;    /* 32 */
  const uint8_t* scales = block + 2 + 104;  /* 4 */
  for (int ib32 = 0; ib32 < 8; ib32 += 2) {
    float db1 = d * (float)(1 + 2 * (scales[ib32 / 2] & 0xf));
    float db2 = d * (float)(1 + 2 * (scales[ib32 / 2] >> 4));
    for (int l = 0; l < 4; ++l) {
      unsigned i1 = qs[2 * l + 0] | (((unsigned)qh[0] << (8 - 2 * l)) & 256);
      unsigned i2 = qs[2 * l + 1] | (((unsigned)qh[0] << (7 - 2 * l)) & 256);
      const uint8_t* g1 = (const uint8_t*)&oc_iq3s_grid[i1];
      const uint8_t* g2 = (const uint8_t*)&oc_iq3s_grid[i2];
      for (int j = 0; j < 4; ++j) {
        o[j + 0] = db1 * (float)g1[j] * (signs[l] & oc_kmask_iq2xs[j + 0] ? -1.f : 1.f);
        o[j + 4] = db1 * (float)g2[j] * (signs[l] & oc_kmask_iq2xs[j + 4] ? -1.f : 1.f);
      }
      o += 8;
    }
    qs += 8;
    signs += 4;
    for (int l = 0; l < 4; ++l) {
      unsigned i1 = qs[2 * l + 0] | (((unsigned)qh[1] << (8 - 2 * l)) & 256);
      unsigned i2 = qs[2 * l + 1] | (((unsigned)qh[1] << (7 - 2 * l)) & 256);
      const uint8_t* g1 = (const uint8_t*)&oc_iq3s_grid[i1];
      const uint8_t* g2 = (const uint8_t*)&oc_iq3s_grid[i2];
      for (int j = 0; j < 4; ++j) {
        o[j + 0] = db2 * (float)g1[j] * (signs[l] & oc_kmask_iq2xs[j + 0] ? -1.f : 1.f);
        o[j + 4] = db2 * (float)g2[j] * (signs[l] & oc_kmask_iq2xs[j + 4] ? -1.f : 1.f);
      }
      o += 8;
    }
    qh += 2;
    qs += 8;
    signs += 4;
  }
}

static void dequant_iq1_s_block(const uint8_t* block, float* o) {
  float d = oc_f16_to_f32(rd16(block));
  const uint8_t* qs = block + 2;      /* 32 */
  const uint8_t* qh = block + 2 + 32; /* 8 u16 */
  for (int ib = 0; ib < 8; ++ib) {
    uint16_t h = rd16(qh + 2 * ib);
    float dl = d * (float)(2 * ((h >> 12) & 7) + 1);
    float delta = (h & 0x8000) ? -OC_IQ1_DELTA : OC_IQ1_DELTA;
    for (int l = 0; l < 4; ++l) {
      unsigned idx = qs[l] | (((unsigned)(h >> (3 * l)) & 7) << 8);
      const int8_t* grid = (const int8_t*)&oc_iq1s_grid[idx];
      for (int j = 0; j < 8; ++j) o[j] = dl * ((float)grid[j] + delta);
      o += 8;
    }
    qs += 4;
  }
}

static void dequant_iq1_m_block(const uint8_t* block, float* o) {
  const uint8_t* qs = block;           /* 32 */
  const uint8_t* qh = block + 32;      /* 16 */
  const uint8_t* sc_b = block + 48;    /* 8 == 4 u16 */
  uint16_t sc[4];
  for (int k = 0; k < 4; ++k) sc[k] = rd16(sc_b + 2 * k);
  /* f16 scale reassembled from the top nibble of each of the 4 u16 scales. */
  uint16_t s = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                          ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
  float d = oc_f16_to_f32(s);
  for (int ib = 0; ib < 8; ++ib) {
    float dl1 = d * (float)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
    float dl2 = d * (float)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
    unsigned idx[4];
    float dl[4];
    idx[0] = qs[0] | (((unsigned)qh[0] << 8) & 0x700);
    idx[1] = qs[1] | (((unsigned)qh[0] << 4) & 0x700);
    idx[2] = qs[2] | (((unsigned)qh[1] << 8) & 0x700);
    idx[3] = qs[3] | (((unsigned)qh[1] << 4) & 0x700);
    dl[0] = (qh[0] & 0x08) ? -OC_IQ1_DELTA : OC_IQ1_DELTA;
    dl[1] = (qh[0] & 0x80) ? -OC_IQ1_DELTA : OC_IQ1_DELTA;
    dl[2] = (qh[1] & 0x08) ? -OC_IQ1_DELTA : OC_IQ1_DELTA;
    dl[3] = (qh[1] & 0x80) ? -OC_IQ1_DELTA : OC_IQ1_DELTA;
    for (int l = 0; l < 4; ++l) {
      const int8_t* grid = (const int8_t*)&oc_iq1s_grid[idx[l]];
      float s2 = l < 2 ? dl1 : dl2;
      for (int j = 0; j < 8; ++j) o[j] = s2 * ((float)grid[j] + dl[l]);
      o += 8;
    }
    qs += 4;
    qh += 2;
  }
}

/* ---- row geometry + dequant ----------------------------------------------- */

size_t oc_row_bytes(uint32_t t, size_t cols) {
  switch (t) {
    case OC_F32: return cols * 4;
    case OC_F16: return cols * 2;
    case OC_BF16: return cols * 2;
    case OC_Q4_0: return cols % OC_QK ? 0 : cols / OC_QK * OC_BLK_Q4_0;
    case OC_Q4_1: return cols % OC_QK ? 0 : cols / OC_QK * OC_BLK_Q4_1;
    case OC_Q5_0: return cols % OC_QK ? 0 : cols / OC_QK * OC_BLK_Q5_0;
    case OC_Q5_1: return cols % OC_QK ? 0 : cols / OC_QK * OC_BLK_Q5_1;
    case OC_Q8_0: return cols % OC_QK ? 0 : cols / OC_QK * OC_BLK_Q8_0;
    case OC_Q2_K: return cols % OC_QK_K ? 0 : cols / OC_QK_K * OC_BLK_Q2_K;
    case OC_Q3_K: return cols % OC_QK_K ? 0 : cols / OC_QK_K * OC_BLK_Q3_K;
    case OC_Q4_K: return cols % OC_QK_K ? 0 : cols / OC_QK_K * OC_BLK_Q4_K;
    case OC_Q5_K: return cols % OC_QK_K ? 0 : cols / OC_QK_K * OC_BLK_Q5_K;
    case OC_Q6_K: return cols % OC_QK_K ? 0 : cols / OC_QK_K * OC_BLK_Q6_K;
    case OC_IQ4_XS: return cols % OC_QK_K ? 0 : cols / OC_QK_K * OC_BLK_IQ4_XS;
    case OC_IQ2_XXS: return cols % OC_QK_K ? 0 : cols / OC_QK_K * OC_BLK_IQ2_XXS;
    case OC_IQ2_XS: return cols % OC_QK_K ? 0 : cols / OC_QK_K * OC_BLK_IQ2_XS;
    case OC_IQ2_S: return cols % OC_QK_K ? 0 : cols / OC_QK_K * OC_BLK_IQ2_S;
    case OC_IQ3_XXS: return cols % OC_QK_K ? 0 : cols / OC_QK_K * OC_BLK_IQ3_XXS;
    case OC_IQ3_S: return cols % OC_QK_K ? 0 : cols / OC_QK_K * OC_BLK_IQ3_S;
    case OC_IQ1_S: return cols % OC_QK_K ? 0 : cols / OC_QK_K * OC_BLK_IQ1_S;
    case OC_IQ1_M: return cols % OC_QK_K ? 0 : cols / OC_QK_K * OC_BLK_IQ1_M;
    case OC_AL5_XS: return cols % OC_QK ? 0 : cols / OC_QK * OC_BLK_AL5_XS;
    default: return 0;
  }
}

static int dequant_row_scalar(uint32_t t, const uint8_t* src, float* dst, size_t n) {
  switch (t) {
    case OC_F32:
      memcpy(dst, src, n * 4);
      return 0;
    case OC_F16:
      for (size_t i = 0; i < n; ++i) dst[i] = oc_f16_to_f32(rd16(src + 2 * i));
      return 0;
    case OC_BF16:
      /* high 16 bits of an IEEE-754 f32, zero-filled mantissa: exact widen. */
      for (size_t i = 0; i < n; ++i) {
        uint32_t bits = (uint32_t)rd16(src + 2 * i) << 16;
        memcpy(&dst[i], &bits, 4);
      }
      return 0;
    case OC_Q4_1:
      for (size_t b = 0; b < n / OC_QK; ++b)
        dequant_q4_1_block(src + b * OC_BLK_Q4_1, dst + b * OC_QK);
      return 0;
    case OC_Q5_0:
      for (size_t b = 0; b < n / OC_QK; ++b)
        dequant_q5_0_block(src + b * OC_BLK_Q5_0, dst + b * OC_QK);
      return 0;
    case OC_Q5_1:
      for (size_t b = 0; b < n / OC_QK; ++b)
        dequant_q5_1_block(src + b * OC_BLK_Q5_1, dst + b * OC_QK);
      return 0;
    case OC_Q2_K:
      for (size_t b = 0; b < n / OC_QK_K; ++b)
        dequant_q2_k_block(src + b * OC_BLK_Q2_K, dst + b * OC_QK_K);
      return 0;
    case OC_Q3_K:
      for (size_t b = 0; b < n / OC_QK_K; ++b)
        dequant_q3_k_block(src + b * OC_BLK_Q3_K, dst + b * OC_QK_K);
      return 0;
    case OC_IQ4_XS:
      for (size_t b = 0; b < n / OC_QK_K; ++b)
        dequant_iq4_xs_block(src + b * OC_BLK_IQ4_XS, dst + b * OC_QK_K);
      return 0;
    case OC_IQ2_XXS:
      for (size_t b = 0; b < n / OC_QK_K; ++b)
        dequant_iq2_xxs_block(src + b * OC_BLK_IQ2_XXS, dst + b * OC_QK_K);
      return 0;
    case OC_IQ2_XS:
      for (size_t b = 0; b < n / OC_QK_K; ++b)
        dequant_iq2_xs_block(src + b * OC_BLK_IQ2_XS, dst + b * OC_QK_K);
      return 0;
    case OC_IQ2_S:
      for (size_t b = 0; b < n / OC_QK_K; ++b)
        dequant_iq2_s_block(src + b * OC_BLK_IQ2_S, dst + b * OC_QK_K);
      return 0;
    case OC_IQ3_XXS:
      for (size_t b = 0; b < n / OC_QK_K; ++b)
        dequant_iq3_xxs_block(src + b * OC_BLK_IQ3_XXS, dst + b * OC_QK_K);
      return 0;
    case OC_IQ3_S:
      for (size_t b = 0; b < n / OC_QK_K; ++b)
        dequant_iq3_s_block(src + b * OC_BLK_IQ3_S, dst + b * OC_QK_K);
      return 0;
    case OC_IQ1_S:
      for (size_t b = 0; b < n / OC_QK_K; ++b)
        dequant_iq1_s_block(src + b * OC_BLK_IQ1_S, dst + b * OC_QK_K);
      return 0;
    case OC_IQ1_M:
      for (size_t b = 0; b < n / OC_QK_K; ++b)
        dequant_iq1_m_block(src + b * OC_BLK_IQ1_M, dst + b * OC_QK_K);
      return 0;
    case OC_Q4_0:
      for (size_t b = 0; b < n / OC_QK; ++b) {
        const uint8_t* blk = src + b * OC_BLK_Q4_0;
        float* o = dst + b * OC_QK;
        float d = oc_f16_to_f32(rd16(blk));
        for (size_t i = 0; i < 16; ++i) {
          uint8_t p = blk[2 + i];
          o[i] = (float)((int)(p & 0x0F) - 8) * d;
          o[i + 16] = (float)((int)(p >> 4) - 8) * d;
        }
      }
      return 0;
    case OC_Q8_0:
      for (size_t b = 0; b < n / OC_QK; ++b) {
        const uint8_t* blk = src + b * OC_BLK_Q8_0;
        float* o = dst + b * OC_QK;
        float d = oc_f16_to_f32(rd16(blk));
        for (size_t i = 0; i < OC_QK; ++i) o[i] = (float)(int8_t)blk[2 + i] * d;
      }
      return 0;
    case OC_Q4_K:
      for (size_t b = 0; b < n / OC_QK_K; ++b)
        dequant_q4_k_block(src + b * OC_BLK_Q4_K, dst + b * OC_QK_K);
      return 0;
    case OC_Q5_K:
      for (size_t b = 0; b < n / OC_QK_K; ++b)
        dequant_q5_k_block(src + b * OC_BLK_Q5_K, dst + b * OC_QK_K);
      return 0;
    case OC_Q6_K:
      for (size_t b = 0; b < n / OC_QK_K; ++b)
        dequant_q6_k_block(src + b * OC_BLK_Q6_K, dst + b * OC_QK_K);
      return 0;
    case OC_AL5_XS:
      for (size_t b = 0; b < n / OC_QK; ++b)
        dequant_al5xs_block(src + b * OC_BLK_AL5_XS, dst + b * OC_QK);
      return 0;
    default:
      return -1;
  }
}

/* ---- scalar kernels (always compiled, always reachable) -------------------- */

/* f32 dot with 8 independent accumulators so the compiler vectorizes. */
static float dot_f32(const float* restrict a, const float* restrict b, size_t n) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
    s4 += a[i + 4] * b[i + 4];
    s5 += a[i + 5] * b[i + 5];
    s6 += a[i + 6] * b[i + 6];
    s7 += a[i + 7] * b[i + 7];
  }
  float sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
  for (; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

static float dot_f32_row(const uint8_t* row, const float* x, size_t cols) {
  return dot_f32((const float*)(const void*)row, x, cols);
}

static float dot_f16_row(const uint8_t* row, const float* x, size_t cols) {
  float sum = 0.0f;
  for (size_t i = 0; i < cols; ++i) sum += oc_f16_to_f32(rd16(row + 2 * i)) * x[i];
  return sum;
}

static float dot_q4_0_row(const uint8_t* row, const float* x, size_t cols) {
  float sum = 0.0f;
  for (size_t b = 0; b < cols / OC_QK; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q4_0;
    const float* xb = x + b * OC_QK;
    float d = oc_f16_to_f32(rd16(blk));
    float lo = 0, hi = 0;
    for (int j = 0; j < 16; ++j) {
      lo += (float)((int)(blk[2 + j] & 0x0F) - 8) * xb[j];
      hi += (float)((int)(blk[2 + j] >> 4) - 8) * xb[j + 16];
    }
    sum += d * (lo + hi);
  }
  return sum;
}

static float dot_q8_0_row(const uint8_t* row, const float* x, size_t cols) {
  float sum = 0.0f;
  for (size_t b = 0; b < cols / OC_QK; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q8_0;
    const float* xb = x + b * OC_QK;
    float d = oc_f16_to_f32(rd16(blk));
    float acc = 0;
    for (int j = 0; j < OC_QK; ++j) acc += (float)(int8_t)blk[2 + j] * xb[j];
    sum += d * acc;
  }
  return sum;
}

static float dot_bf16_row(const uint8_t* row, const float* x, size_t cols) {
  float sum = 0.0f;
  for (size_t i = 0; i < cols; ++i) {
    uint32_t bits = (uint32_t)rd16(row + 2 * i) << 16;
    float v;
    memcpy(&v, &bits, 4);
    sum += v * x[i];
  }
  return sum;
}

/* Legacy 32-value and K/IQ blocks with no fused kernel yet: dequant one block
 * into scratch, then dot_f32. Correct and self-checking against the golden
 * vectors; a fused SIMD path is a later package. */
#define OC_SCALAR_BLK_DOT(NAME, VPB, BLK, DEQ)                         \
  static float NAME(const uint8_t* row, const float* x, size_t cols) { \
    float scratch[VPB];                                                \
    float sum = 0.0f;                                                  \
    for (size_t b = 0; b < cols / (VPB); ++b) {                        \
      DEQ(row + b * (BLK), scratch);                                   \
      sum += dot_f32(scratch, x + b * (VPB), (VPB));                   \
    }                                                                  \
    return sum;                                                        \
  }
OC_SCALAR_BLK_DOT(dot_q4_k_row, OC_QK_K, OC_BLK_Q4_K, dequant_q4_k_block)
OC_SCALAR_BLK_DOT(dot_q5_k_row, OC_QK_K, OC_BLK_Q5_K, dequant_q5_k_block)
OC_SCALAR_BLK_DOT(dot_q6_k_row, OC_QK_K, OC_BLK_Q6_K, dequant_q6_k_block)
OC_SCALAR_BLK_DOT(dot_q2_k_row, OC_QK_K, OC_BLK_Q2_K, dequant_q2_k_block)
OC_SCALAR_BLK_DOT(dot_q3_k_row, OC_QK_K, OC_BLK_Q3_K, dequant_q3_k_block)
OC_SCALAR_BLK_DOT(dot_iq4_xs_row, OC_QK_K, OC_BLK_IQ4_XS, dequant_iq4_xs_block)
OC_SCALAR_BLK_DOT(dot_iq2_xxs_row, OC_QK_K, OC_BLK_IQ2_XXS, dequant_iq2_xxs_block)
OC_SCALAR_BLK_DOT(dot_iq2_xs_row, OC_QK_K, OC_BLK_IQ2_XS, dequant_iq2_xs_block)
OC_SCALAR_BLK_DOT(dot_iq2_s_row, OC_QK_K, OC_BLK_IQ2_S, dequant_iq2_s_block)
OC_SCALAR_BLK_DOT(dot_iq3_xxs_row, OC_QK_K, OC_BLK_IQ3_XXS, dequant_iq3_xxs_block)
OC_SCALAR_BLK_DOT(dot_iq3_s_row, OC_QK_K, OC_BLK_IQ3_S, dequant_iq3_s_block)
OC_SCALAR_BLK_DOT(dot_iq1_s_row, OC_QK_K, OC_BLK_IQ1_S, dequant_iq1_s_block)
OC_SCALAR_BLK_DOT(dot_iq1_m_row, OC_QK_K, OC_BLK_IQ1_M, dequant_iq1_m_block)
OC_SCALAR_BLK_DOT(dot_q4_1_row, OC_QK, OC_BLK_Q4_1, dequant_q4_1_block)
OC_SCALAR_BLK_DOT(dot_q5_0_row, OC_QK, OC_BLK_Q5_0, dequant_q5_0_block)
OC_SCALAR_BLK_DOT(dot_q5_1_row, OC_QK, OC_BLK_Q5_1, dequant_q5_1_block)
#undef OC_SCALAR_BLK_DOT

static float dot_al5xs_row(const uint8_t* row, const float* x, size_t cols) {
  float sum = 0.0f;
  for (size_t b = 0; b < cols / OC_QK; ++b) {
    const uint8_t* blk = row + b * OC_BLK_AL5_XS;
    const float* xb = x + b * OC_QK;
    float scale = al5xs_scale(blk);
    uint8_t codes[32];
    al5xs_unpack(blk + 2, codes);
    float acc = 0;
    for (int j = 0; j < 32; ++j) acc += ((float)codes[j] - 4.0f) * xb[j];
    sum += scale * acc; /* same mapping as al5xs_lut, hoisted scale */
  }
  return sum;
}

/* Reference rank-kb update. Each lane t accumulates over k in order, which is
 * exactly what the SIMD versions do per lane — so they agree to within one FMA
 * rounding, and the differential test can be tight. */
static void gemm_row_scalar(float* acc, const float* w, const float* xp,
                            size_t kb, size_t n) {
  for (size_t k = 0; k < kb; ++k) {
    float a = w[k];
    const float* xs = xp + k * n;
    for (size_t t = 0; t < n; ++t) acc[t] += a * xs[t];
  }
}

static void gemm_row4_scalar(float* acc, const float* w, size_t ws,
                             const float* xp, size_t kb, size_t n) {
  for (size_t i = 0; i < 4; ++i) gemm_row_scalar(acc + i * n, w + i * ws, xp, kb, n);
}

static void q8_quantize_scalar(const float* x, size_t n, int8_t* q, float* d,
                               int32_t* bsum) {
  for (size_t b = 0; b < n / OC_QK_K; ++b) {
    const float* xb = x + b * OC_QK_K;
    float amax = 0.0f;
    for (size_t i = 0; i < OC_QK_K; ++i) {
      float a = fabsf(xb[i]);
      if (a > amax) amax = a;
    }
    float scale = amax / 127.0f;
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    d[b] = scale;
    int8_t* qb = q + b * OC_QK_K;
    int32_t* sb = bsum + b * 16;
    for (int g = 0; g < 16; ++g) {
      int32_t s = 0;
      for (int i = 0; i < 16; ++i) {
        int v = (int)lrintf(xb[g * 16 + i] * inv);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        qb[g * 16 + i] = (int8_t)v;
        s += v;
      }
      sb[g] = s;
    }
  }
}

/* ---- runtime CPU feature detection ---------------------------------------- */

typedef struct {
  int avx2, fma, f16c;
  int avx512f, avx512bw, avx512vl, avx512dq, avx512vnni;
  int neon, dotprod; /* aarch64 */
} OcCpu;

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>

/* XCR0: the CPU can have AVX-512 and the OS still not save ZMM state on a
 * context switch, in which case using it faults. Check both. */
static uint64_t oc_xcr0(void) {
  uint32_t lo, hi;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return ((uint64_t)hi << 32) | lo;
}

static OcCpu cpu_detect(void) {
  OcCpu c = {0};
  uint32_t a, b, cx, d;
  if (!__get_cpuid(1, &a, &b, &cx, &d)) return c;
  if (!((cx >> 27) & 1) || !((cx >> 28) & 1)) return c; /* no OSXSAVE / no AVX */
  uint64_t xcr0 = oc_xcr0();
  if ((xcr0 & 0x6) != 0x6) return c; /* OS does not save YMM: SSE only */
  c.fma = (int)((cx >> 12) & 1);
  c.f16c = (int)((cx >> 29) & 1);
  if (__get_cpuid_max(0, NULL) < 7) return c;
  __cpuid_count(7, 0, a, b, cx, d);
  c.avx2 = (int)((b >> 5) & 1);
  if ((xcr0 & 0xe6) == 0xe6) { /* opmask + zmm_hi256 + hi16_zmm saved */
    c.avx512f = (int)((b >> 16) & 1);
    c.avx512dq = (int)((b >> 17) & 1);
    c.avx512bw = (int)((b >> 30) & 1);
    c.avx512vl = (int)((b >> 31) & 1);
    c.avx512vnni = (int)((cx >> 11) & 1);
  }
  return c;
}

/* Each SIMD TU is compiled with all of -mavx2 -mfma -mf16c (plus its own
 * flags), so the compiler may emit any of those instructions anywhere in it:
 * a kernel family is usable only if the CPU has every feature its TU was
 * built with, not merely the ones the intrinsics name.
 *
 * No Skylake-SP AVX-512 downclock gate here (oxidize-kernels/src/cpu.rs has
 * one). That gate is about a *float* GEMV losing to AVX2 on model 85/86 — but
 * those parts are also the only ones on the bench box with VNNI, where the
 * int8 path is a different kernel entirely. Auto-downgrading would turn VNNI
 * off on the exact machine it was written for, on the strength of a benchmark
 * nobody has run here. OC_ISA=avx2 is the knob; measure, then decide. */
static OcIsa isa_from_cpu(const OcCpu* c) {
  if (!c->avx2 || !c->fma || !c->f16c) return OC_ISA_SCALAR;
  if (c->avx512f && c->avx512bw && c->avx512vl && c->avx512dq) return OC_ISA_AVX512;
  return OC_ISA_AVX2;
}
#elif defined(__aarch64__)
#include <sys/auxv.h>
/* ASIMD (NEON) is mandatory on aarch64 — nothing to test for the baseline
 * kernels. dotprod (sdot/udot) is optional; we only report it in the ISA name,
 * the bound kernels are baseline-NEON so they run on every aarch64 part. */
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
static OcCpu cpu_detect(void) {
  OcCpu c = {0};
  c.neon = 1;
  c.dotprod = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) ? 1 : 0;
  return c;
}
static OcIsa isa_from_cpu(const OcCpu* c) {
  return c->neon ? OC_ISA_NEON : OC_ISA_SCALAR;
}
#else
static OcCpu cpu_detect(void) {
  OcCpu c = {0};
  return c; /* unknown arch: scalar only */
}
static OcIsa isa_from_cpu(const OcCpu* c) {
  (void)c;
  return OC_ISA_SCALAR;
}
#endif

/* ---- dispatch table -------------------------------------------------------- */

typedef float (*OcDotFn)(const uint8_t* row, const float* x, size_t cols);
typedef float (*OcDotQ8Fn)(const uint8_t* row, const OcQ8Act* a, size_t cols);
typedef void (*OcQ8QuantFn)(const float* x, size_t n, int8_t* q, float* d,
                            int32_t* bsum);
typedef void (*OcGemmRowFn)(float* acc, const float* w, const float* xp,
                            size_t kb, size_t n);
typedef void (*OcGemmRow4Fn)(float* acc, const float* w, size_t ws,
                             const float* xp, size_t kb, size_t n);
typedef void (*OcDeqFn)(const uint8_t* row, float* out, size_t n);

enum {
  TI_F32, TI_F16, TI_Q4_0, TI_Q8_0, TI_Q4_K, TI_Q5_K, TI_Q6_K, TI_AL5_XS,
  TI_BF16, TI_Q4_1, TI_Q5_0, TI_Q5_1, TI_Q2_K, TI_Q3_K, TI_IQ4_XS,
  TI_IQ2_XXS, TI_IQ2_XS, TI_IQ2_S, TI_IQ3_XXS, TI_IQ3_S, TI_IQ1_S, TI_IQ1_M,
  TI_N
};

static int type_index(uint32_t t) {
  switch (t) {
    case OC_F32: return TI_F32;
    case OC_F16: return TI_F16;
    case OC_BF16: return TI_BF16;
    case OC_Q4_0: return TI_Q4_0;
    case OC_Q4_1: return TI_Q4_1;
    case OC_Q5_0: return TI_Q5_0;
    case OC_Q5_1: return TI_Q5_1;
    case OC_Q8_0: return TI_Q8_0;
    case OC_Q2_K: return TI_Q2_K;
    case OC_Q3_K: return TI_Q3_K;
    case OC_Q4_K: return TI_Q4_K;
    case OC_Q5_K: return TI_Q5_K;
    case OC_Q6_K: return TI_Q6_K;
    case OC_IQ4_XS: return TI_IQ4_XS;
    case OC_IQ2_XXS: return TI_IQ2_XXS;
    case OC_IQ2_XS: return TI_IQ2_XS;
    case OC_IQ2_S: return TI_IQ2_S;
    case OC_IQ3_XXS: return TI_IQ3_XXS;
    case OC_IQ3_S: return TI_IQ3_S;
    case OC_IQ1_S: return TI_IQ1_S;
    case OC_IQ1_M: return TI_IQ1_M;
    case OC_AL5_XS: return TI_AL5_XS;
    default: return -1;
  }
}

/* Positional, indexed by TI_*. New scalar-only types stay on these kernels at
 * every ISA level (bind_table only overrides the SIMD-accelerated ones). */
static const OcDotFn DOT_SCALAR[TI_N] = {
    [TI_F32] = dot_f32_row,      [TI_F16] = dot_f16_row,
    [TI_Q4_0] = dot_q4_0_row,    [TI_Q8_0] = dot_q8_0_row,
    [TI_Q4_K] = dot_q4_k_row,    [TI_Q5_K] = dot_q5_k_row,
    [TI_Q6_K] = dot_q6_k_row,    [TI_AL5_XS] = dot_al5xs_row,
    [TI_BF16] = dot_bf16_row,    [TI_Q4_1] = dot_q4_1_row,
    [TI_Q5_0] = dot_q5_0_row,    [TI_Q5_1] = dot_q5_1_row,
    [TI_Q2_K] = dot_q2_k_row,    [TI_Q3_K] = dot_q3_k_row,
    [TI_IQ4_XS] = dot_iq4_xs_row,
    [TI_IQ2_XXS] = dot_iq2_xxs_row, [TI_IQ2_XS] = dot_iq2_xs_row,
    [TI_IQ2_S] = dot_iq2_s_row,     [TI_IQ3_XXS] = dot_iq3_xxs_row,
    [TI_IQ3_S] = dot_iq3_s_row,     [TI_IQ1_S] = dot_iq1_s_row,
    [TI_IQ1_M] = dot_iq1_m_row,
};

static OcDotFn g_dot[TI_N];
static OcDotQ8Fn g_dot_q8[TI_N];      /* NULL = no int8 kernel for this type */
static OcQ8QuantFn g_q8_quantize = q8_quantize_scalar;
static OcGemmRowFn g_gemm_row = gemm_row_scalar;
static OcGemmRow4Fn g_gemm_row4 = gemm_row4_scalar;
/* Vectorized whole-row dequant. Only where it pays: oc_matmul unpacks every
 * weight in the model once per batch and cannot fuse that into the FMA the way
 * the dot kernels do, so a scalar unpack there costs MORE than the GEMM it
 * feeds. NULL => the scalar switch. */
static OcDeqFn g_deq[TI_N];
static OcCpu g_cpu;
static OcIsa g_forced = OC_ISA_AUTO;
static OcIsa g_active = OC_ISA_SCALAR;
static const char* g_active_name = "scalar";

static void bind_table(void) {
  OcIsa best = isa_from_cpu(&g_cpu);
  OcIsa want = g_forced == OC_ISA_AUTO || g_forced > best ? best : g_forced;

  memcpy(g_dot, DOT_SCALAR, sizeof g_dot);
  memset(g_dot_q8, 0, sizeof g_dot_q8);
  g_q8_quantize = q8_quantize_scalar;
  g_gemm_row = gemm_row_scalar;
  g_gemm_row4 = gemm_row4_scalar;
  memset(g_deq, 0, sizeof g_deq);

#if defined(__x86_64__) || defined(__i386__)
  int vnni = want >= OC_ISA_AVX512 && g_cpu.avx512vnni;
  if (want >= OC_ISA_AVX2) {
    g_dot[TI_Q4_K] = oc_dot_q4_k_avx2;
    g_dot[TI_Q5_K] = oc_dot_q5_k_avx2;
    g_dot[TI_Q6_K] = oc_dot_q6_k_avx2;
    g_dot[TI_Q2_K] = oc_dot_q2_k_avx2;
    g_dot[TI_Q3_K] = oc_dot_q3_k_avx2;
    g_dot[TI_IQ4_XS] = oc_dot_iq4_xs_avx2;
    g_dot[TI_BF16] = oc_dot_bf16_avx2;
    g_dot[TI_AL5_XS] = oc_dot_al5xs_avx2;
    g_gemm_row = oc_gemm_row_avx2;
    g_gemm_row4 = oc_gemm_row4_avx2;
    g_deq[TI_Q4_K] = oc_dequant_q4_k_avx2;
    g_deq[TI_Q5_K] = oc_dequant_q5_k_avx2;
    g_deq[TI_Q6_K] = oc_dequant_q6_k_avx2;
    g_deq[TI_AL5_XS] = oc_dequant_al5xs_avx2;
  }
  if (want >= OC_ISA_AVX512) {
    /* Q5_K/Q6_K get wider AVX-512 dots; Q2_K/Q3_K/IQ4_XS/BF16 keep the AVX2
     * kernels above (they run fine 8-wide on an AVX-512 core). */
    g_dot[TI_Q4_K] = oc_dot_q4_k_avx512;
    g_dot[TI_Q5_K] = oc_dot_q5_k_avx512;
    g_dot[TI_Q6_K] = oc_dot_q6_k_avx512;
    g_dot[TI_AL5_XS] = oc_dot_al5xs_avx512;
    g_q8_quantize = oc_q8_quantize_avx512;
    g_gemm_row = oc_gemm_row_avx512;
    g_gemm_row4 = oc_gemm_row4_avx512;
  }
  if (vnni) {
    g_dot_q8[TI_Q4_K] = oc_dot_q4_k_vnni;
    g_dot_q8[TI_Q5_K] = oc_dot_q5_k_vnni;
    g_dot_q8[TI_Q6_K] = oc_dot_q6_k_vnni;
  }
  g_active = want;
  g_active_name = want == OC_ISA_AVX512 ? (vnni ? "avx512+vnni" : "avx512")
                  : want == OC_ISA_AVX2 ? "avx2"
                                        : "scalar";
#elif defined(__aarch64__)
  /* NEON is the only SIMD tier on aarch64; want<NEON means scalar was forced.
   * Q4_0/Q8_0 additionally get NEON dots here (they stay scalar on x86). */
  int neon = want >= OC_ISA_NEON;
  if (neon) {
    g_dot[TI_Q4_0] = oc_dot_q4_0_neon;
    g_dot[TI_Q8_0] = oc_dot_q8_0_neon;
    g_dot[TI_Q4_K] = oc_dot_q4_k_neon;
    g_dot[TI_Q5_K] = oc_dot_q5_k_neon;
    g_dot[TI_Q6_K] = oc_dot_q6_k_neon;
    g_dot[TI_BF16] = oc_dot_bf16_neon;
    g_gemm_row = oc_gemm_row_neon;
    g_gemm_row4 = oc_gemm_row4_neon;
    g_deq[TI_Q4_K] = oc_dequant_q4_k_neon;
    g_deq[TI_Q5_K] = oc_dequant_q5_k_neon;
    g_deq[TI_Q6_K] = oc_dequant_q6_k_neon;
  }
  g_active = neon ? OC_ISA_NEON : OC_ISA_SCALAR;
  g_active_name = neon ? (g_cpu.dotprod ? "neon+dotprod" : "neon") : "scalar";
#else
  g_active = OC_ISA_SCALAR;
  g_active_name = "scalar";
#endif
}

/* Resolved before main(): the table must be live for any caller, including the
 * pool workers, without a per-call branch or a data race on lazy init. */
__attribute__((constructor)) static void oc_isa_init(void) {
  g_cpu = cpu_detect();
  const char* e = getenv("OC_ISA");
  if (e) {
    if (strcmp(e, "scalar") == 0) g_forced = OC_ISA_SCALAR;
    else if (strcmp(e, "avx2") == 0) g_forced = OC_ISA_AVX2;
    else if (strcmp(e, "avx512") == 0) g_forced = OC_ISA_AVX512;
    else if (strcmp(e, "neon") == 0) g_forced = OC_ISA_NEON;
  }
  bind_table();
}

void oc_force_isa(OcIsa isa) {
  g_forced = isa;
  bind_table();
}

OcIsa oc_isa_available(void) { return isa_from_cpu(&g_cpu); }
OcIsa oc_isa_active(void) { return g_active; }
const char* oc_isa_active_name(void) { return g_active_name; }

const void* oc_dot_impl(uint32_t t) {
  int i = type_index(t);
  return i < 0 ? NULL : (const void*)g_dot[i];
}

/* ---- dispatched entry points ----------------------------------------------- */

int oc_dequant_row(uint32_t t, const uint8_t* src, float* dst, size_t n) {
  int i = type_index(t);
  if (i >= 0 && g_deq[i]) {
    g_deq[i](src, dst, n);
    return 0;
  }
  return dequant_row_scalar(t, src, dst, n);
}

float oc_dot_row(uint32_t t, const uint8_t* row, const float* x, size_t cols) {
  int i = type_index(t);
  if (i < 0) return 0.0f; /* callers validate the type at load time */
  return g_dot[i](row, x, cols);
}

int oc_q8_dot_supported(uint32_t t) {
  int i = type_index(t);
  return i >= 0 && g_dot_q8[i] != NULL;
}

void oc_q8_quantize(const float* x, size_t n, int8_t* q, float* d, int32_t* bsum) {
  g_q8_quantize(x, n, q, d, bsum);
}

float oc_dot_row_q8(uint32_t t, const uint8_t* row, const OcQ8Act* a, size_t cols) {
  int i = type_index(t);
  if (i < 0 || !g_dot_q8[i]) return 0.0f; /* callers must check oc_q8_dot_supported */
  return g_dot_q8[i](row, a, cols);
}

void oc_gemm_row(float* acc, const float* w, const float* xp, size_t kb, size_t n) {
  g_gemm_row(acc, w, xp, kb, n);
}

void oc_gemm_row4(float* acc, const float* w, size_t ws, const float* xp,
                  size_t kb, size_t n) {
  g_gemm_row4(acc, w, ws, xp, kb, n);
}

const void* oc_gemm_impl(void) { return (const void*)g_gemm_row; }

const void* oc_dequant_impl(uint32_t t) {
  int i = type_index(t);
  return i < 0 ? NULL : (const void*)g_deq[i];
}

/* ---- misc quantizers ------------------------------------------------------- */

void oc_quantize_row_q4_0(const float* x, uint8_t* out, size_t n) {
  for (size_t b = 0; b < n / OC_QK; ++b) {
    const float* xb = x + b * OC_QK;
    uint8_t* o = out + b * OC_BLK_Q4_0;
    float amax = 0.0f, maxv = 0.0f;
    for (int i = 0; i < OC_QK; ++i) {
      float a = fabsf(xb[i]);
      if (a > amax) { amax = a; maxv = xb[i]; }
    }
    float d = maxv / -8.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    uint16_t dh = oc_f32_to_f16(d);
    o[0] = (uint8_t)(dh & 0xff);
    o[1] = (uint8_t)(dh >> 8);
    for (int i = 0; i < 16; ++i) {
      int lo = (int)(xb[i] * id + 8.5f);
      int hi = (int)(xb[i + 16] * id + 8.5f);
      if (lo > 15) lo = 15;
      if (lo < 0) lo = 0;
      if (hi > 15) hi = 15;
      if (hi < 0) hi = 0;
      o[2 + i] = (uint8_t)(lo | (hi << 4));
    }
  }
}

/* ---- rotoquant KV cache helpers ---- */

void oc_fht(float* v, size_t n) {
  for (size_t len = 1; len < n; len <<= 1)
    for (size_t i = 0; i < n; i += len << 1)
      for (size_t j = i; j < i + len; ++j) {
        float a = v[j], b = v[j + len];
        v[j] = a + b;
        v[j + len] = a - b;
      }
  float s = 1.0f / sqrtf((float)n);
  for (size_t i = 0; i < n; ++i) v[i] *= s;
}

void oc_kvq_encode(const float* x, size_t n, uint8_t* out, float* meta) {
  float mn = x[0], mx = x[0];
  for (size_t i = 1; i < n; ++i) {
    if (x[i] < mn) mn = x[i];
    if (x[i] > mx) mx = x[i];
  }
  float s = (mx - mn) / 15.0f;
  float inv = s > 0.0f ? 1.0f / s : 0.0f;
  meta[0] = s;
  meta[1] = mn;
  for (size_t i = 0; i < n; i += 2) {
    int q0 = (int)((x[i] - mn) * inv + 0.5f);
    int q1 = (int)((x[i + 1] - mn) * inv + 0.5f);
    if (q0 > 15) q0 = 15;
    if (q1 > 15) q1 = 15;
    out[i >> 1] = (uint8_t)(q0 | (q1 << 4));
  }
}

void oc_kvq_decode(const uint8_t* in, size_t n, const float* meta, float* x) {
  float s = meta[0], mn = meta[1];
  for (size_t i = 0; i < n; i += 2) {
    uint8_t b = in[i >> 1];
    x[i] = mn + s * (float)(b & 15u);
    x[i + 1] = mn + s * (float)(b >> 4);
  }
}
