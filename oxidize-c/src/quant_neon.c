/* ARM NEON / ASIMD kernels. Compiled only on aarch64 (armv8-a NEON is baseline,
 * so no -march feature flag is required); reachable only through the dispatch
 * table in quant.c. The decode math is a lane-for-lane port of the scalar
 * reference in quant.c (and mirrors quant_avx2.c 128-bit-wide), so every kernel
 * is == the scalar path the differential in tests/test_quant.c checks. */
#include "quant_impl.h"

#if !defined(__aarch64__)
#error "build quant_neon.c only on aarch64"
#endif

#include <arm_neon.h>

/* Horizontal sum of a float32x4. vaddvq_f32 is a single ASIMD instruction. */
static inline float hsum128(float32x4_t v) { return vaddvq_f32(v); }

/* Widen 16 unsigned bytes to four float32x4 (lanes 0-3, 4-7, 8-11, 12-15). */
static inline void neon_u8_to_f32(uint8x16_t v, float32x4_t o[4]) {
  uint16x8_t lo = vmovl_u8(vget_low_u8(v));
  uint16x8_t hi = vmovl_u8(vget_high_u8(v));
  o[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
  o[1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo)));
  o[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
  o[3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi)));
}

/* Same, signed. */
static inline void neon_s8_to_f32(int8x16_t v, float32x4_t o[4]) {
  int16x8_t lo = vmovl_s8(vget_low_s8(v));
  int16x8_t hi = vmovl_s8(vget_high_s8(v));
  o[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
  o[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
  o[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
  o[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
}

/* ---- Q4_0: d * sum((nibble-8) * x). Mirrors dot_q4_0_row. ------------------ */
float oc_dot_q4_0_neon(const uint8_t* row, const float* x, size_t cols) {
  const uint8x16_t maskF = vdupq_n_u8(0x0F);
  const float32x4_t eight = vdupq_n_f32(8.0f);
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (size_t b = 0; b < cols / OC_QK; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q4_0;
    const float* xb = x + b * OC_QK;
    float d = oc_f16_fast(rd16(blk));
    uint8x16_t q = vld1q_u8(blk + 2); /* 16 bytes -> 32 nibbles */
    float32x4_t lo[4], hi[4];
    neon_u8_to_f32(vandq_u8(q, maskF), lo);
    neon_u8_to_f32(vshrq_n_u8(q, 4), hi);
    float32x4_t bacc = vdupq_n_f32(0.0f);
    for (int k = 0; k < 4; ++k)
      bacc = vfmaq_f32(bacc, vsubq_f32(lo[k], eight), vld1q_f32(xb + k * 4));
    for (int k = 0; k < 4; ++k)
      bacc = vfmaq_f32(bacc, vsubq_f32(hi[k], eight), vld1q_f32(xb + 16 + k * 4));
    acc = vfmaq_f32(acc, vdupq_n_f32(d), bacc);
  }
  return hsum128(acc);
}

/* ---- Q8_0: d * sum(int8 * x). Mirrors dot_q8_0_row. ------------------------ */
float oc_dot_q8_0_neon(const uint8_t* row, const float* x, size_t cols) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (size_t b = 0; b < cols / OC_QK; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q8_0;
    const float* xb = x + b * OC_QK;
    float d = oc_f16_fast(rd16(blk));
    float32x4_t q0[4], q1[4];
    neon_s8_to_f32(vld1q_s8((const int8_t*)(blk + 2)), q0);
    neon_s8_to_f32(vld1q_s8((const int8_t*)(blk + 2 + 16)), q1);
    float32x4_t bacc = vdupq_n_f32(0.0f);
    for (int k = 0; k < 4; ++k) bacc = vfmaq_f32(bacc, q0[k], vld1q_f32(xb + k * 4));
    for (int k = 0; k < 4; ++k) bacc = vfmaq_f32(bacc, q1[k], vld1q_f32(xb + 16 + k * 4));
    acc = vfmaq_f32(acc, vdupq_n_f32(d), bacc);
  }
  return hsum128(acc);
}

/* ---- Q4_K: fused dequant+dot. Mirrors oc_dot_q4_k_avx2. -------------------- */
float oc_dot_q4_k_neon(const uint8_t* row, const float* x, size_t cols) {
  const uint8x16_t maskF = vdupq_n_u8(0x0F);
  float sum = 0.0f;
  for (size_t b = 0; b < cols / OC_QK_K; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q4_K;
    float d = oc_f16_fast(rd16(blk));
    float min = oc_f16_fast(rd16(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qs = blk + 16;
    const float* xb = x + b * OC_QK_K;
    size_t is = 0;
    for (int gp = 0; gp < 4; ++gp) {
      uint8_t sc1, m1, sc2, m2;
      get_scale_min_k4(is, scales, &sc1, &m1);
      get_scale_min_k4(is + 1, scales, &sc2, &m2);
      const uint8_t* q = qs + (size_t)gp * 32;
      const float* xlo = xb + (size_t)gp * 64;
      const float* xhi = xlo + 32;
      float32x4_t alo = vdupq_n_f32(0), ahi = vdupq_n_f32(0);
      float32x4_t slo = vdupq_n_f32(0), shi = vdupq_n_f32(0);
      for (int half = 0; half < 2; ++half) {
        uint8x16_t qb = vld1q_u8(q + half * 16);
        float32x4_t lo[4], hi[4];
        neon_u8_to_f32(vandq_u8(qb, maskF), lo);
        neon_u8_to_f32(vshrq_n_u8(qb, 4), hi);
        for (int k = 0; k < 4; ++k) {
          float32x4_t xl = vld1q_f32(xlo + half * 16 + k * 4);
          float32x4_t xh = vld1q_f32(xhi + half * 16 + k * 4);
          alo = vfmaq_f32(alo, lo[k], xl);
          ahi = vfmaq_f32(ahi, hi[k], xh);
          slo = vaddq_f32(slo, xl);
          shi = vaddq_f32(shi, xh);
        }
      }
      sum += d * sc1 * hsum128(alo) - min * m1 * hsum128(slo);
      sum += d * sc2 * hsum128(ahi) - min * m2 * hsum128(shi);
      is += 2;
    }
  }
  return sum;
}

/* ---- Q5_K: fused. 4 low bits + 1 high bit (per-32 bit lane), min via macc.
 * Mirrors oc_dot_q5_k_avx2. The variable bit select uses vtstq (bit-set ->
 * 0xFF) & 16 instead of a runtime immediate shift. --------------------------- */
float oc_dot_q5_k_neon(const uint8_t* row, const float* x, size_t cols) {
  const uint8x16_t maskF = vdupq_n_u8(0x0F);
  const uint8x16_t c16 = vdupq_n_u8(16);
  float sum = 0.0f;
  for (size_t b = 0; b < cols / OC_QK_K; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q5_K;
    float d = oc_f16_fast(rd16(blk));
    float min = oc_f16_fast(rd16(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qh = blk + 16;
    const uint8_t* ql = blk + 48;
    const float* xb = x + b * OC_QK_K;
    size_t is = 0;
    float32x4_t acc = vdupq_n_f32(0), macc = vdupq_n_f32(0);
    for (int gp = 0; gp < 4; ++gp) {
      uint8_t sc1, m1, sc2, m2;
      get_scale_min_k4(is, scales, &sc1, &m1);
      get_scale_min_k4(is + 1, scales, &sc2, &m2);
      const float* xlo = xb + (size_t)gp * 64;
      const float* xhi = xlo + 32;
      float32x4_t vs1 = vdupq_n_f32(d * sc1), vm1 = vdupq_n_f32(min * m1);
      float32x4_t vs2 = vdupq_n_f32(d * sc2), vm2 = vdupq_n_f32(min * m2);
      uint8x16_t bit_lo = vdupq_n_u8((uint8_t)(1u << (2 * gp)));
      uint8x16_t bit_hi = vdupq_n_u8((uint8_t)(1u << (2 * gp + 1)));
      for (int half = 0; half < 2; ++half) {
        uint8x16_t qb = vld1q_u8(ql + half * 16);
        uint8x16_t hb = vld1q_u8(qh + half * 16);
        uint8x16_t lo5 = vaddq_u8(vandq_u8(qb, maskF),
                                  vandq_u8(vtstq_u8(hb, bit_lo), c16));
        uint8x16_t hi5 = vaddq_u8(vshrq_n_u8(qb, 4),
                                  vandq_u8(vtstq_u8(hb, bit_hi), c16));
        float32x4_t lo[4], hi[4];
        neon_u8_to_f32(lo5, lo);
        neon_u8_to_f32(hi5, hi);
        for (int k = 0; k < 4; ++k) {
          float32x4_t xl = vld1q_f32(xlo + half * 16 + k * 4);
          float32x4_t xh = vld1q_f32(xhi + half * 16 + k * 4);
          acc = vfmaq_f32(acc, vmulq_f32(lo[k], vs1), xl);
          acc = vfmaq_f32(acc, vmulq_f32(hi[k], vs2), xh);
          macc = vfmaq_f32(macc, vm1, xl);
          macc = vfmaq_f32(macc, vm2, xh);
        }
      }
      ql += 32;
      is += 2;
    }
    sum += hsum128(acc) - hsum128(macc);
  }
  return sum;
}

/* ---- Q6_K: fused. ql nibbles + 2 qh bits, signed via -32, per-16 int8 scale.
 * Mirrors oc_dot_q6_k_avx2. --------------------------------------------------- */
float oc_dot_q6_k_neon(const uint8_t* row, const float* x, size_t cols) {
  const uint8x16_t maskF = vdupq_n_u8(0x0F);
  const uint8x16_t mask3 = vdupq_n_u8(0x03);
  const float32x4_t c32 = vdupq_n_f32(32.0f);
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (size_t b = 0; b < cols / OC_QK_K; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q6_K;
    float d = oc_f16_fast(rd16(blk + 208));
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    const float* xb = x + b * OC_QK_K;
    for (int gp = 0; gp < 2; ++gp) {
      size_t ql_off = (size_t)gp * 64, qh_off = (size_t)gp * 32, sc_off = (size_t)gp * 8;
      const float* xg = xb + (size_t)gp * 128;
      for (int j = 0; j < 32; j += 16) {
        size_t is = (size_t)j / 16;
        uint8x16_t l0 = vld1q_u8(ql + ql_off + j);
        uint8x16_t l1 = vld1q_u8(ql + ql_off + j + 32);
        uint8x16_t hbits = vld1q_u8(qh + qh_off + j);
        uint8x16_t q1u = vorrq_u8(vandq_u8(l0, maskF),
                                  vshlq_n_u8(vandq_u8(hbits, mask3), 4));
        uint8x16_t q2u = vorrq_u8(vandq_u8(l1, maskF),
                                  vshlq_n_u8(vandq_u8(vshrq_n_u8(hbits, 2), mask3), 4));
        uint8x16_t q3u = vorrq_u8(vshrq_n_u8(l0, 4),
                                  vshlq_n_u8(vandq_u8(vshrq_n_u8(hbits, 4), mask3), 4));
        uint8x16_t q4u = vorrq_u8(vshrq_n_u8(l1, 4),
                                  vshlq_n_u8(vandq_u8(vshrq_n_u8(hbits, 6), mask3), 4));
        float32x4_t q1[4], q2[4], q3[4], q4[4];
        neon_u8_to_f32(q1u, q1);
        neon_u8_to_f32(q2u, q2);
        neon_u8_to_f32(q3u, q3);
        neon_u8_to_f32(q4u, q4);
        float32x4_t s1 = vdupq_n_f32(d * sc[sc_off + is]);
        float32x4_t s2 = vdupq_n_f32(d * sc[sc_off + is + 2]);
        float32x4_t s3 = vdupq_n_f32(d * sc[sc_off + is + 4]);
        float32x4_t s4 = vdupq_n_f32(d * sc[sc_off + is + 6]);
        for (int k = 0; k < 4; ++k) {
          acc = vfmaq_f32(acc, vmulq_f32(vsubq_f32(q1[k], c32), s1),
                          vld1q_f32(xg + j + k * 4));
          acc = vfmaq_f32(acc, vmulq_f32(vsubq_f32(q2[k], c32), s2),
                          vld1q_f32(xg + 32 + j + k * 4));
          acc = vfmaq_f32(acc, vmulq_f32(vsubq_f32(q3[k], c32), s3),
                          vld1q_f32(xg + 64 + j + k * 4));
          acc = vfmaq_f32(acc, vmulq_f32(vsubq_f32(q4[k], c32), s4),
                          vld1q_f32(xg + 96 + j + k * 4));
        }
      }
    }
  }
  return hsum128(acc);
}

/* ---- BF16: widen top-16-bit float into the f32 mantissa, then FMA.
 * Mirrors oc_dot_bf16_avx2. --------------------------------------------------- */
float oc_dot_bf16_neon(const uint8_t* row, const float* x, size_t cols) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 8 <= cols; i += 8) {
    uint16x8_t u16 = vld1q_u16((const uint16_t*)(row + 2 * i));
    uint32x4_t lo = vshll_n_u16(vget_low_u16(u16), 16);
    uint32x4_t hi = vshll_n_u16(vget_high_u16(u16), 16);
    acc = vfmaq_f32(acc, vreinterpretq_f32_u32(lo), vld1q_f32(x + i));
    acc = vfmaq_f32(acc, vreinterpretq_f32_u32(hi), vld1q_f32(x + i + 4));
  }
  float sum = hsum128(acc);
  for (; i < cols; ++i) {
    union { uint32_t u; float f; } cvt;
    cvt.u = (uint32_t)rd16(row + 2 * i) << 16;
    sum += cvt.f * x[i];
  }
  return sum;
}

/* ---- whole-row dequant (feeds oc_matmul). Mirror the AVX2 dequant kernels,
 * storing d*q - min instead of folding into an FMA. ------------------------- */
void oc_dequant_q4_k_neon(const uint8_t* row, float* out, size_t n) {
  const uint8x16_t maskF = vdupq_n_u8(0x0F);
  for (size_t b = 0; b < n / OC_QK_K; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q4_K;
    float d = oc_f16_fast(rd16(blk));
    float mn = oc_f16_fast(rd16(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qs = blk + 16;
    float* o = out + b * OC_QK_K;
    size_t is = 0;
    for (int gp = 0; gp < 4; ++gp) {
      uint8_t sc1, m1, sc2, m2;
      get_scale_min_k4(is, scales, &sc1, &m1);
      get_scale_min_k4(is + 1, scales, &sc2, &m2);
      float32x4_t d1 = vdupq_n_f32(d * sc1), min1 = vdupq_n_f32(mn * m1);
      float32x4_t d2 = vdupq_n_f32(d * sc2), min2 = vdupq_n_f32(mn * m2);
      const uint8_t* q = qs + (size_t)gp * 32;
      float* olo = o + (size_t)gp * 64;
      float* ohi = olo + 32;
      for (int half = 0; half < 2; ++half) {
        uint8x16_t qb = vld1q_u8(q + half * 16);
        float32x4_t lo[4], hi[4];
        neon_u8_to_f32(vandq_u8(qb, maskF), lo);
        neon_u8_to_f32(vshrq_n_u8(qb, 4), hi);
        for (int k = 0; k < 4; ++k) {
          vst1q_f32(olo + half * 16 + k * 4, vsubq_f32(vmulq_f32(lo[k], d1), min1));
          vst1q_f32(ohi + half * 16 + k * 4, vsubq_f32(vmulq_f32(hi[k], d2), min2));
        }
      }
      is += 2;
    }
  }
}

void oc_dequant_q5_k_neon(const uint8_t* row, float* out, size_t n) {
  const uint8x16_t maskF = vdupq_n_u8(0x0F);
  const uint8x16_t c16 = vdupq_n_u8(16);
  for (size_t b = 0; b < n / OC_QK_K; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q5_K;
    float d = oc_f16_fast(rd16(blk));
    float mn = oc_f16_fast(rd16(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qh = blk + 16;
    const uint8_t* ql = blk + 48;
    float* o = out + b * OC_QK_K;
    size_t is = 0;
    for (int gp = 0; gp < 4; ++gp) {
      uint8_t sc1, m1, sc2, m2;
      get_scale_min_k4(is, scales, &sc1, &m1);
      get_scale_min_k4(is + 1, scales, &sc2, &m2);
      float32x4_t d1 = vdupq_n_f32(d * sc1), min1 = vdupq_n_f32(mn * m1);
      float32x4_t d2 = vdupq_n_f32(d * sc2), min2 = vdupq_n_f32(mn * m2);
      float* olo = o + (size_t)gp * 64;
      float* ohi = olo + 32;
      uint8x16_t bit_lo = vdupq_n_u8((uint8_t)(1u << (2 * gp)));
      uint8x16_t bit_hi = vdupq_n_u8((uint8_t)(1u << (2 * gp + 1)));
      for (int half = 0; half < 2; ++half) {
        uint8x16_t qb = vld1q_u8(ql + half * 16);
        uint8x16_t hb = vld1q_u8(qh + half * 16);
        uint8x16_t lo5 = vaddq_u8(vandq_u8(qb, maskF),
                                  vandq_u8(vtstq_u8(hb, bit_lo), c16));
        uint8x16_t hi5 = vaddq_u8(vshrq_n_u8(qb, 4),
                                  vandq_u8(vtstq_u8(hb, bit_hi), c16));
        float32x4_t lo[4], hi[4];
        neon_u8_to_f32(lo5, lo);
        neon_u8_to_f32(hi5, hi);
        for (int k = 0; k < 4; ++k) {
          vst1q_f32(olo + half * 16 + k * 4, vsubq_f32(vmulq_f32(lo[k], d1), min1));
          vst1q_f32(ohi + half * 16 + k * 4, vsubq_f32(vmulq_f32(hi[k], d2), min2));
        }
      }
      ql += 32;
      is += 2;
    }
  }
}

void oc_dequant_q6_k_neon(const uint8_t* row, float* out, size_t n) {
  const uint8x16_t maskF = vdupq_n_u8(0x0F);
  const uint8x16_t mask3 = vdupq_n_u8(0x03);
  const float32x4_t c32 = vdupq_n_f32(32.0f);
  for (size_t b = 0; b < n / OC_QK_K; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q6_K;
    float d = oc_f16_fast(rd16(blk + 208));
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    float* o = out + b * OC_QK_K;
    for (int gp = 0; gp < 2; ++gp) {
      size_t ql_off = (size_t)gp * 64, qh_off = (size_t)gp * 32, sc_off = (size_t)gp * 8;
      float* og = o + (size_t)gp * 128;
      for (int j = 0; j < 32; j += 16) {
        size_t is = (size_t)j / 16;
        uint8x16_t l0 = vld1q_u8(ql + ql_off + j);
        uint8x16_t l1 = vld1q_u8(ql + ql_off + j + 32);
        uint8x16_t hbits = vld1q_u8(qh + qh_off + j);
        uint8x16_t q1u = vorrq_u8(vandq_u8(l0, maskF),
                                  vshlq_n_u8(vandq_u8(hbits, mask3), 4));
        uint8x16_t q2u = vorrq_u8(vandq_u8(l1, maskF),
                                  vshlq_n_u8(vandq_u8(vshrq_n_u8(hbits, 2), mask3), 4));
        uint8x16_t q3u = vorrq_u8(vshrq_n_u8(l0, 4),
                                  vshlq_n_u8(vandq_u8(vshrq_n_u8(hbits, 4), mask3), 4));
        uint8x16_t q4u = vorrq_u8(vshrq_n_u8(l1, 4),
                                  vshlq_n_u8(vandq_u8(vshrq_n_u8(hbits, 6), mask3), 4));
        float32x4_t q1[4], q2[4], q3[4], q4[4];
        neon_u8_to_f32(q1u, q1);
        neon_u8_to_f32(q2u, q2);
        neon_u8_to_f32(q3u, q3);
        neon_u8_to_f32(q4u, q4);
        float32x4_t s1 = vdupq_n_f32(d * sc[sc_off + is]);
        float32x4_t s2 = vdupq_n_f32(d * sc[sc_off + is + 2]);
        float32x4_t s3 = vdupq_n_f32(d * sc[sc_off + is + 4]);
        float32x4_t s4 = vdupq_n_f32(d * sc[sc_off + is + 6]);
        for (int k = 0; k < 4; ++k) {
          vst1q_f32(og + j + k * 4, vmulq_f32(vsubq_f32(q1[k], c32), s1));
          vst1q_f32(og + 32 + j + k * 4, vmulq_f32(vsubq_f32(q2[k], c32), s2));
          vst1q_f32(og + 64 + j + k * 4, vmulq_f32(vsubq_f32(q3[k], c32), s3));
          vst1q_f32(og + 96 + j + k * 4, vmulq_f32(vsubq_f32(q4[k], c32), s4));
        }
      }
    }
  }
}

/* ---- GEMM inner kernels. Pure float FMA; 16-token tile (four q-regs), lanes
 * accumulate in k order exactly like gemm_row_scalar. n is a multiple of 16. -- */
void oc_gemm_row_neon(float* acc, const float* w, const float* xp, size_t kb,
                      size_t n) {
  for (size_t t = 0; t + 16 <= n; t += 16) {
    float32x4_t a0 = vld1q_f32(acc + t), a1 = vld1q_f32(acc + t + 4);
    float32x4_t a2 = vld1q_f32(acc + t + 8), a3 = vld1q_f32(acc + t + 12);
    const float* xs = xp + t;
    for (size_t k = 0; k < kb; ++k, xs += n) {
      float32x4_t wk = vdupq_n_f32(w[k]);
      a0 = vfmaq_f32(a0, wk, vld1q_f32(xs));
      a1 = vfmaq_f32(a1, wk, vld1q_f32(xs + 4));
      a2 = vfmaq_f32(a2, wk, vld1q_f32(xs + 8));
      a3 = vfmaq_f32(a3, wk, vld1q_f32(xs + 12));
    }
    vst1q_f32(acc + t, a0);
    vst1q_f32(acc + t + 4, a1);
    vst1q_f32(acc + t + 8, a2);
    vst1q_f32(acc + t + 12, a3);
  }
}

void oc_gemm_row4_neon(float* acc, const float* w, size_t ws, const float* xp,
                       size_t kb, size_t n) {
  const float *w0 = w, *w1 = w + ws, *w2 = w + 2 * ws, *w3 = w + 3 * ws;
  float *p1 = acc + n, *p2 = acc + 2 * n, *p3 = acc + 3 * n;
  for (size_t t = 0; t + 16 <= n; t += 16) {
    float32x4_t a0 = vld1q_f32(acc + t), a1 = vld1q_f32(acc + t + 4);
    float32x4_t a2 = vld1q_f32(acc + t + 8), a3 = vld1q_f32(acc + t + 12);
    float32x4_t b0 = vld1q_f32(p1 + t), b1 = vld1q_f32(p1 + t + 4);
    float32x4_t b2 = vld1q_f32(p1 + t + 8), b3 = vld1q_f32(p1 + t + 12);
    float32x4_t c0 = vld1q_f32(p2 + t), c1 = vld1q_f32(p2 + t + 4);
    float32x4_t c2 = vld1q_f32(p2 + t + 8), c3 = vld1q_f32(p2 + t + 12);
    float32x4_t d0 = vld1q_f32(p3 + t), d1 = vld1q_f32(p3 + t + 4);
    float32x4_t d2 = vld1q_f32(p3 + t + 8), d3 = vld1q_f32(p3 + t + 12);
    const float* xs = xp + t;
    for (size_t k = 0; k < kb; ++k, xs += n) {
      float32x4_t x0 = vld1q_f32(xs), x1 = vld1q_f32(xs + 4);
      float32x4_t x2 = vld1q_f32(xs + 8), x3 = vld1q_f32(xs + 12);
      float32x4_t v = vdupq_n_f32(w0[k]);
      a0 = vfmaq_f32(a0, v, x0); a1 = vfmaq_f32(a1, v, x1);
      a2 = vfmaq_f32(a2, v, x2); a3 = vfmaq_f32(a3, v, x3);
      v = vdupq_n_f32(w1[k]);
      b0 = vfmaq_f32(b0, v, x0); b1 = vfmaq_f32(b1, v, x1);
      b2 = vfmaq_f32(b2, v, x2); b3 = vfmaq_f32(b3, v, x3);
      v = vdupq_n_f32(w2[k]);
      c0 = vfmaq_f32(c0, v, x0); c1 = vfmaq_f32(c1, v, x1);
      c2 = vfmaq_f32(c2, v, x2); c3 = vfmaq_f32(c3, v, x3);
      v = vdupq_n_f32(w3[k]);
      d0 = vfmaq_f32(d0, v, x0); d1 = vfmaq_f32(d1, v, x1);
      d2 = vfmaq_f32(d2, v, x2); d3 = vfmaq_f32(d3, v, x3);
    }
    vst1q_f32(acc + t, a0); vst1q_f32(acc + t + 4, a1);
    vst1q_f32(acc + t + 8, a2); vst1q_f32(acc + t + 12, a3);
    vst1q_f32(p1 + t, b0); vst1q_f32(p1 + t + 4, b1);
    vst1q_f32(p1 + t + 8, b2); vst1q_f32(p1 + t + 12, b3);
    vst1q_f32(p2 + t, c0); vst1q_f32(p2 + t + 4, c1);
    vst1q_f32(p2 + t + 8, c2); vst1q_f32(p2 + t + 12, c3);
    vst1q_f32(p3 + t, d0); vst1q_f32(p3 + t + 4, d1);
    vst1q_f32(p3 + t + 8, d2); vst1q_f32(p3 + t + 12, d3);
  }
}
