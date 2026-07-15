/* AVX2 + FMA + F16C kernels. Compiled with -mavx2 -mfma -mf16c; reachable only
 * through the dispatch table in quant.c. */
#include "quant_impl.h"

#if !defined(__AVX2__) || !defined(__FMA__) || !defined(__F16C__)
#error "build quant_avx2.c with -mavx2 -mfma -mf16c"
#endif

#include <immintrin.h>

static inline float hsum256(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}

/* AL5_XS: each 3 bytes of qs hold exactly 8 3-bit codes (LSB-first), so one
 * 24-bit load + per-lane variable shift decodes 8 codes at once. */
float oc_dot_al5xs_avx2(const uint8_t* row, const float* x, size_t cols) {
  const __m256i shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
  const __m256i mask7 = _mm256_set1_epi32(7);
  const __m256 four = _mm256_set1_ps(4.0f);
  __m256 acc = _mm256_setzero_ps();
  for (size_t b = 0; b < cols / OC_QK; ++b) {
    const uint8_t* blk = row + b * OC_BLK_AL5_XS;
    float scale = al5xs_scale(blk);
    const uint8_t* qs = blk + 2;
    const float* xb = x + b * OC_QK;
    __m256 bacc = _mm256_setzero_ps();
    for (int g = 0; g < 4; ++g) {
      uint32_t bits = (uint32_t)qs[g * 3] | (uint32_t)qs[g * 3 + 1] << 8 |
                      (uint32_t)qs[g * 3 + 2] << 16;
      __m256i codes = _mm256_and_si256(
          _mm256_srlv_epi32(_mm256_set1_epi32((int)bits), shifts), mask7);
      __m256 w = _mm256_sub_ps(_mm256_cvtepi32_ps(codes), four);
      bacc = _mm256_fmadd_ps(w, _mm256_loadu_ps(xb + g * 8), bacc);
    }
    acc = _mm256_fmadd_ps(_mm256_set1_ps(scale), bacc, acc);
  }
  return hsum256(acc);
}

/* Rank-kb update, 16-token tile. n is a multiple of 16 (oc_matmul pads the
 * panel), so there is no token tail. Accumulators live in registers across the
 * whole kb loop; reload them per k instead and the kernel goes store-bound.
 * Lanes accumulate in k order, matching gemm_row_scalar exactly. */
void oc_gemm_row_avx2(float* acc, const float* w, const float* xp, size_t kb,
                      size_t n) {
  for (size_t t = 0; t + 16 <= n; t += 16) {
    __m256 a0 = _mm256_loadu_ps(acc + t);
    __m256 a1 = _mm256_loadu_ps(acc + t + 8);
    const float* xs = xp + t;
    for (size_t k = 0; k < kb; ++k, xs += n) {
      __m256 wk = _mm256_set1_ps(w[k]);
      a0 = _mm256_fmadd_ps(wk, _mm256_loadu_ps(xs), a0);
      a1 = _mm256_fmadd_ps(wk, _mm256_loadu_ps(xs + 8), a1);
    }
    _mm256_storeu_ps(acc + t, a0);
    _mm256_storeu_ps(acc + t + 8, a1);
  }
}

/* Four adjacent weight rows. Eight accumulator chains (4 rows x 2 vectors) is
 * what it takes to cover the FMA latency and keep both FMA units fed; the two
 * panel loads are also amortized over four broadcasts instead of one. 11 of 16
 * ymm registers live. */
void oc_gemm_row4_avx2(float* acc, const float* w, size_t ws, const float* xp,
                       size_t kb, size_t n) {
  const float *w0 = w, *w1 = w + ws, *w2 = w + 2 * ws, *w3 = w + 3 * ws;
  float *p1 = acc + n, *p2 = acc + 2 * n, *p3 = acc + 3 * n;
  for (size_t t = 0; t + 16 <= n; t += 16) {
    __m256 a0 = _mm256_loadu_ps(acc + t), a1 = _mm256_loadu_ps(acc + t + 8);
    __m256 b0 = _mm256_loadu_ps(p1 + t), b1 = _mm256_loadu_ps(p1 + t + 8);
    __m256 c0 = _mm256_loadu_ps(p2 + t), c1 = _mm256_loadu_ps(p2 + t + 8);
    __m256 d0 = _mm256_loadu_ps(p3 + t), d1 = _mm256_loadu_ps(p3 + t + 8);
    const float* xs = xp + t;
    for (size_t k = 0; k < kb; ++k, xs += n) {
      __m256 x0 = _mm256_loadu_ps(xs), x1 = _mm256_loadu_ps(xs + 8);
      __m256 v = _mm256_set1_ps(w0[k]);
      a0 = _mm256_fmadd_ps(v, x0, a0);
      a1 = _mm256_fmadd_ps(v, x1, a1);
      v = _mm256_set1_ps(w1[k]);
      b0 = _mm256_fmadd_ps(v, x0, b0);
      b1 = _mm256_fmadd_ps(v, x1, b1);
      v = _mm256_set1_ps(w2[k]);
      c0 = _mm256_fmadd_ps(v, x0, c0);
      c1 = _mm256_fmadd_ps(v, x1, c1);
      v = _mm256_set1_ps(w3[k]);
      d0 = _mm256_fmadd_ps(v, x0, d0);
      d1 = _mm256_fmadd_ps(v, x1, d1);
    }
    _mm256_storeu_ps(acc + t, a0);
    _mm256_storeu_ps(acc + t + 8, a1);
    _mm256_storeu_ps(p1 + t, b0);
    _mm256_storeu_ps(p1 + t + 8, b1);
    _mm256_storeu_ps(p2 + t, c0);
    _mm256_storeu_ps(p2 + t + 8, c1);
    _mm256_storeu_ps(p3 + t, d0);
    _mm256_storeu_ps(p3 + t + 8, d1);
  }
}

/* ---- dequant to a buffer ----
 * oc_matmul dequantizes each weight row-block once and reuses it across the
 * batch, so unlike the dot kernels it cannot fuse the unpack into the FMA. That
 * makes the unpack a standalone pass — and a SCALAR one costs ~10 cycles per
 * weight against ~2 for the vectorized FMA that follows it, i.e. it becomes 80%
 * of the GEMM. Same decode as the dot kernels above, writing to memory. */
void oc_dequant_al5xs_avx2(const uint8_t* row, float* out, size_t n) {
  const __m256i shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
  const __m256i mask7 = _mm256_set1_epi32(7);
  const __m256 four = _mm256_set1_ps(4.0f);
  for (size_t b = 0; b < n / OC_QK; ++b) {
    const uint8_t* blk = row + b * OC_BLK_AL5_XS;
    __m256 scale = _mm256_set1_ps(al5xs_scale(blk));
    const uint8_t* qs = blk + 2;
    float* o = out + b * OC_QK;
    for (int g = 0; g < 4; ++g) {
      uint32_t bits = (uint32_t)qs[g * 3] | (uint32_t)qs[g * 3 + 1] << 8 |
                      (uint32_t)qs[g * 3 + 2] << 16;
      __m256i codes = _mm256_and_si256(
          _mm256_srlv_epi32(_mm256_set1_epi32((int)bits), shifts), mask7);
      __m256 w = _mm256_sub_ps(_mm256_cvtepi32_ps(codes), four);
      _mm256_storeu_ps(o + g * 8, _mm256_mul_ps(w, scale));
    }
  }
}

void oc_dequant_q4_k_avx2(const uint8_t* row, float* out, size_t n) {
  const __m256i maskF = _mm256_set1_epi32(0xF);
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
      __m256 d1 = _mm256_set1_ps(d * sc1), min1 = _mm256_set1_ps(mn * m1);
      __m256 d2 = _mm256_set1_ps(d * sc2), min2 = _mm256_set1_ps(mn * m2);
      const uint8_t* q = qs + (size_t)gp * 32;
      float* olo = o + (size_t)gp * 64;
      float* ohi = olo + 32;
      for (int j = 0; j < 32; j += 8) {
        __m256i qi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(q + j)));
        __m256 lo = _mm256_cvtepi32_ps(_mm256_and_si256(qi, maskF));
        __m256 hi = _mm256_cvtepi32_ps(_mm256_srli_epi32(qi, 4));
        _mm256_storeu_ps(olo + j, _mm256_fmsub_ps(lo, d1, min1)); /* lo*d1 - min1 */
        _mm256_storeu_ps(ohi + j, _mm256_fmsub_ps(hi, d2, min2));
      }
      is += 2;
    }
  }
}

/* Q5_K: 4 low bits from ql + 1 high bit from qh (per-32 bit lane 2*gp / 2*gp+1),
 * per-32 scale d*sc and min*m. fmsub folds the min subtraction into the FMA. */
void oc_dequant_q5_k_avx2(const uint8_t* row, float* out, size_t n) {
  const __m256i maskF = _mm256_set1_epi32(0xF);
  const __m256i one = _mm256_set1_epi32(1);
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
      __m256 d1 = _mm256_set1_ps(d * sc1), min1 = _mm256_set1_ps(mn * m1);
      __m256 d2 = _mm256_set1_ps(d * sc2), min2 = _mm256_set1_ps(mn * m2);
      float* olo = o + (size_t)gp * 64;
      float* ohi = olo + 32;
      for (int j = 0; j < 32; j += 8) {
        __m256i qi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(ql + j)));
        __m256i hb = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(qh + j)));
        __m256i lo5 = _mm256_or_si256(_mm256_and_si256(qi, maskF),
            _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hb, 2 * gp), one), 4));
        __m256i hi5 = _mm256_or_si256(_mm256_srli_epi32(qi, 4),
            _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hb, 2 * gp + 1), one), 4));
        _mm256_storeu_ps(olo + j, _mm256_fmsub_ps(_mm256_cvtepi32_ps(lo5), d1, min1));
        _mm256_storeu_ps(ohi + j, _mm256_fmsub_ps(_mm256_cvtepi32_ps(hi5), d2, min2));
      }
      ql += 32;
      is += 2;
    }
  }
}

/* Q6_K: ql nibbles + 2 high bits from qh, signed, per-16 int8 scales. */
void oc_dequant_q6_k_avx2(const uint8_t* row, float* out, size_t n) {
  const __m256i maskF = _mm256_set1_epi32(0xF);
  const __m256i mask3 = _mm256_set1_epi32(3);
  const __m256i c32 = _mm256_set1_epi32(32);
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
      for (int j = 0; j < 32; j += 8) {
        size_t is = (size_t)j / 16; /* 8-wide chunks never straddle the 16 boundary */
        __m256i l0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(ql + ql_off + j)));
        __m256i l1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(ql + ql_off + j + 32)));
        __m256i hbits = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(qh + qh_off + j)));
        __m256i q1 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_and_si256(l0, maskF),
                            _mm256_slli_epi32(_mm256_and_si256(hbits, mask3), 4)), c32);
        __m256i q2 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_and_si256(l1, maskF),
                            _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hbits, 2), mask3), 4)), c32);
        __m256i q3 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_srli_epi32(l0, 4),
                            _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hbits, 4), mask3), 4)), c32);
        __m256i q4 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_srli_epi32(l1, 4),
                            _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hbits, 6), mask3), 4)), c32);
        _mm256_storeu_ps(og + j, _mm256_mul_ps(_mm256_cvtepi32_ps(q1),
                                               _mm256_set1_ps(d * sc[sc_off + is])));
        _mm256_storeu_ps(og + j + 32, _mm256_mul_ps(_mm256_cvtepi32_ps(q2),
                                                    _mm256_set1_ps(d * sc[sc_off + is + 2])));
        _mm256_storeu_ps(og + j + 64, _mm256_mul_ps(_mm256_cvtepi32_ps(q3),
                                                    _mm256_set1_ps(d * sc[sc_off + is + 4])));
        _mm256_storeu_ps(og + j + 96, _mm256_mul_ps(_mm256_cvtepi32_ps(q4),
                                                    _mm256_set1_ps(d * sc[sc_off + is + 6])));
      }
    }
  }
}

/* Q4_K: fused dequant+dot, no scratch round-trip. */
float oc_dot_q4_k_avx2(const uint8_t* row, const float* x, size_t cols) {
  const __m256i maskF = _mm256_set1_epi32(0xF);
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
      __m256 alo = _mm256_setzero_ps(), ahi = _mm256_setzero_ps();
      __m256 slo = _mm256_setzero_ps(), shi = _mm256_setzero_ps();
      for (int j = 0; j < 32; j += 8) {
        __m256i qi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(q + j)));
        __m256 lo = _mm256_cvtepi32_ps(_mm256_and_si256(qi, maskF));
        __m256 hi = _mm256_cvtepi32_ps(_mm256_srli_epi32(qi, 4));
        __m256 xl = _mm256_loadu_ps(xlo + j), xh = _mm256_loadu_ps(xhi + j);
        alo = _mm256_fmadd_ps(lo, xl, alo);
        ahi = _mm256_fmadd_ps(hi, xh, ahi);
        slo = _mm256_add_ps(slo, xl);
        shi = _mm256_add_ps(shi, xh);
      }
      sum += d * sc1 * hsum256(alo) - min * m1 * hsum256(slo);
      sum += d * sc2 * hsum256(ahi) - min * m2 * hsum256(shi);
      is += 2;
    }
  }
  return sum;
}

/* Q5_K: fused. Same decode as oc_dequant_q5_k_avx2, but the per-32 scale/min
 * fold happens once at the end via two running accumulators: acc collects
 * d*sc*q*x, macc collects min*m*x, and the block result is hsum(acc)-hsum(macc)
 * (== sum_i x_i*(d*sc*q_i - min*m), the dequant value dotted with x). */
float oc_dot_q5_k_avx2(const uint8_t* row, const float* x, size_t cols) {
  const __m256i maskF = _mm256_set1_epi32(0xF);
  const __m256i one = _mm256_set1_epi32(1);
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
    __m256 acc = _mm256_setzero_ps(), macc = _mm256_setzero_ps();
    for (int gp = 0; gp < 4; ++gp) {
      uint8_t sc1, m1, sc2, m2;
      get_scale_min_k4(is, scales, &sc1, &m1);
      get_scale_min_k4(is + 1, scales, &sc2, &m2);
      const float* xlo = xb + (size_t)gp * 64;
      const float* xhi = xlo + 32;
      __m256 vs1 = _mm256_set1_ps(d * sc1), vm1 = _mm256_set1_ps(min * m1);
      __m256 vs2 = _mm256_set1_ps(d * sc2), vm2 = _mm256_set1_ps(min * m2);
      for (int j = 0; j < 32; j += 8) {
        __m256i qi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(ql + j)));
        __m256i hb = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(qh + j)));
        __m256i lo5 = _mm256_or_si256(_mm256_and_si256(qi, maskF),
            _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hb, 2 * gp), one), 4));
        __m256i hi5 = _mm256_or_si256(_mm256_srli_epi32(qi, 4),
            _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hb, 2 * gp + 1), one), 4));
        __m256 xl = _mm256_loadu_ps(xlo + j), xh = _mm256_loadu_ps(xhi + j);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(lo5), vs1), xl, acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(hi5), vs2), xh, acc);
        macc = _mm256_fmadd_ps(vm1, xl, macc);
        macc = _mm256_fmadd_ps(vm2, xh, macc);
      }
      ql += 32;
      is += 2;
    }
    sum += hsum256(acc) - hsum256(macc);
  }
  return sum;
}

/* Q6_K: fused. Same decode as oc_dequant_q6_k_avx2 (ql nibbles + 2 qh bits,
 * signed via -32), each per-16 scale applied inline then FMA'd with x. */
float oc_dot_q6_k_avx2(const uint8_t* row, const float* x, size_t cols) {
  const __m256i maskF = _mm256_set1_epi32(0xF);
  const __m256i mask3 = _mm256_set1_epi32(3);
  const __m256i c32 = _mm256_set1_epi32(32);
  __m256 acc = _mm256_setzero_ps();
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
      for (int j = 0; j < 32; j += 8) {
        size_t is = (size_t)j / 16; /* 8-wide chunks never straddle the 16 boundary */
        __m256i l0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(ql + ql_off + j)));
        __m256i l1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(ql + ql_off + j + 32)));
        __m256i hbits = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(qh + qh_off + j)));
        __m256i q1 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_and_si256(l0, maskF),
                            _mm256_slli_epi32(_mm256_and_si256(hbits, mask3), 4)), c32);
        __m256i q2 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_and_si256(l1, maskF),
                            _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hbits, 2), mask3), 4)), c32);
        __m256i q3 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_srli_epi32(l0, 4),
                            _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hbits, 4), mask3), 4)), c32);
        __m256i q4 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_srli_epi32(l1, 4),
                            _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hbits, 6), mask3), 4)), c32);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q1),
                  _mm256_set1_ps(d * sc[sc_off + is])), _mm256_loadu_ps(xg + j), acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q2),
                  _mm256_set1_ps(d * sc[sc_off + is + 2])), _mm256_loadu_ps(xg + 32 + j), acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q3),
                  _mm256_set1_ps(d * sc[sc_off + is + 4])), _mm256_loadu_ps(xg + 64 + j), acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q4),
                  _mm256_set1_ps(d * sc[sc_off + is + 6])), _mm256_loadu_ps(xg + 96 + j), acc);
      }
    }
  }
  return hsum256(acc);
}

/* Q2_K: fused. 2-bit codes, per-16 scale d*(sc&F) and min dmin*(sc>>4); the min
 * folds through a running macc as in Q5_K. Variable shift via srlv (the shift
 * amount lives in the nested loop, not an unrollable immediate). */
float oc_dot_q2_k_avx2(const uint8_t* row, const float* x, size_t cols) {
  const __m256i mask3 = _mm256_set1_epi32(3);
  float sum = 0.0f;
  for (size_t b = 0; b < cols / OC_QK_K; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q2_K;
    const uint8_t* scales = blk;
    const uint8_t* qs = blk + 16;
    float d = oc_f16_fast(rd16(blk + 80));
    float dmin = oc_f16_fast(rd16(blk + 82));
    const float* xb = x + b * OC_QK_K;
    __m256 acc = _mm256_setzero_ps(), macc = _mm256_setzero_ps();
    size_t y = 0, is = 0;
    for (int n = 0; n < 256; n += 128) {
      const uint8_t* q = qs + (size_t)(n / 128) * 32;
      for (int j = 0; j < 4; ++j) {
        __m256i vshift = _mm256_set1_epi32(2 * j);
        uint8_t sc1 = scales[is], sc2 = scales[is + 1];
        __m256 dl1 = _mm256_set1_ps(d * (float)(sc1 & 0xF));
        __m256 ml1 = _mm256_set1_ps(dmin * (float)(sc1 >> 4));
        __m256 dl2 = _mm256_set1_ps(d * (float)(sc2 & 0xF));
        __m256 ml2 = _mm256_set1_ps(dmin * (float)(sc2 >> 4));
        for (int c = 0; c < 16; c += 8) {
          __m256i q1 = _mm256_and_si256(_mm256_srlv_epi32(
              _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(q + c))), vshift), mask3);
          __m256 xl = _mm256_loadu_ps(xb + y + c);
          acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q1), dl1), xl, acc);
          macc = _mm256_fmadd_ps(ml1, xl, macc);
          __m256i q2 = _mm256_and_si256(_mm256_srlv_epi32(
              _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(q + 16 + c))), vshift), mask3);
          __m256 xh = _mm256_loadu_ps(xb + y + 16 + c);
          acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q2), dl2), xh, acc);
          macc = _mm256_fmadd_ps(ml2, xh, macc);
        }
        y += 32;
        is += 2;
      }
    }
    sum += hsum256(acc) - hsum256(macc);
  }
  return sum;
}

/* Q3_K: fused. 2 low bits from qs + an inverted high bit from hmask (bit set =>
 * value stays 0..3, clear => value-4), per-16 signed 6-bit scale. The 12 packed
 * scale bytes are unpacked scalar once per block (tiny); the value loop is SIMD. */
float oc_dot_q3_k_avx2(const uint8_t* row, const float* x, size_t cols) {
  const __m256i mask3 = _mm256_set1_epi32(3);
  const __m256i four = _mm256_set1_epi32(4);
  const __m256i zero = _mm256_setzero_si256();
  float sum = 0.0f;
  for (size_t b = 0; b < cols / OC_QK_K; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q3_K;
    const uint8_t* hm = blk;
    const uint8_t* qs = blk + 32;
    float d_all = oc_f16_fast(rd16(blk + 108));
    uint32_t aux[4];
    for (int k = 0; k < 3; ++k) aux[k] = rd32(blk + 96 + 4 * k);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
    aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);
    aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);
    int8_t sb[16];
    for (int k = 0; k < 4; ++k)
      for (int bb = 0; bb < 4; ++bb) sb[k * 4 + bb] = (int8_t)((aux[k] >> (8 * bb)) & 0xff);
    const float* xb = x + b * OC_QK_K;
    __m256 acc = _mm256_setzero_ps();
    size_t y = 0, is = 0;
    uint8_t m = 1;
    for (int n = 0; n < 256; n += 128) {
      const uint8_t* q = qs + (size_t)(n / 128) * 32;
      for (int j = 0; j < 4; ++j) {
        __m256i vshift = _mm256_set1_epi32(2 * j);
        __m256i mvec = _mm256_set1_epi32(m);
        __m256 sl = _mm256_set1_ps(d_all * (float)((int)sb[is] - 32));
        __m256 sh = _mm256_set1_ps(d_all * (float)((int)sb[is + 1] - 32));
        for (int c = 0; c < 16; c += 8) {
          __m256i qv = _mm256_and_si256(_mm256_srlv_epi32(
              _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(q + c))), vshift), mask3);
          __m256i hb = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(hm + c)));
          __m256i hbit = _mm256_and_si256(
              _mm256_cmpeq_epi32(_mm256_and_si256(hb, mvec), zero), four);
          __m256 fv = _mm256_cvtepi32_ps(_mm256_sub_epi32(qv, hbit));
          acc = _mm256_fmadd_ps(_mm256_mul_ps(fv, sl), _mm256_loadu_ps(xb + y + c), acc);
          __m256i qv2 = _mm256_and_si256(_mm256_srlv_epi32(
              _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(q + 16 + c))), vshift), mask3);
          __m256i hb2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(hm + 16 + c)));
          __m256i hbit2 = _mm256_and_si256(
              _mm256_cmpeq_epi32(_mm256_and_si256(hb2, mvec), zero), four);
          __m256 fv2 = _mm256_cvtepi32_ps(_mm256_sub_epi32(qv2, hbit2));
          acc = _mm256_fmadd_ps(_mm256_mul_ps(fv2, sh), _mm256_loadu_ps(xb + y + 16 + c), acc);
        }
        y += 32;
        is += 2;
        m <<= 1;
      }
    }
    sum += hsum256(acc);
  }
  return sum;
}

/* IQ4_XS: fused. The 16-entry IQ4_NL codebook maps a 4-bit code to a signed
 * value; vpshufb does that gather in-lane for 16 codes at once. Per-32 scale. */
static const int8_t OC_IQ4NL_AVX2[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

float oc_dot_iq4_xs_avx2(const uint8_t* row, const float* x, size_t cols) {
  const __m128i lut = _mm_loadu_si128((const __m128i*)OC_IQ4NL_AVX2);
  const __m128i maskF = _mm_set1_epi8(0x0F);
  __m256 acc = _mm256_setzero_ps();
  for (size_t b = 0; b < cols / OC_QK_K; ++b) {
    const uint8_t* blk = row + b * OC_BLK_IQ4_XS;
    float d = oc_f16_fast(rd16(blk));
    uint16_t scales_h = rd16(blk + 2);
    const uint8_t* scales_l = blk + 4;
    const uint8_t* qs = blk + 8;
    const float* xb = x + b * OC_QK_K;
    for (int ib = 0; ib < 8; ++ib) {
      int ls_l = (scales_l[ib / 2] >> (4 * (ib & 1))) & 0xf;
      int ls_h = (int)((scales_h >> (2 * ib)) & 3) << 4;
      __m256 vdl = _mm256_set1_ps(d * (float)((ls_l | ls_h) - 32));
      __m128i q = _mm_loadu_si128((const __m128i*)(qs + (size_t)ib * 16));
      __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, maskF));
      __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), maskF));
      const float* xob = xb + (size_t)ib * 32;
      __m256 lo0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
      __m256 lo1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
      __m256 hi0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
      __m256 hi1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
      acc = _mm256_fmadd_ps(_mm256_mul_ps(lo0, vdl), _mm256_loadu_ps(xob), acc);
      acc = _mm256_fmadd_ps(_mm256_mul_ps(lo1, vdl), _mm256_loadu_ps(xob + 8), acc);
      acc = _mm256_fmadd_ps(_mm256_mul_ps(hi0, vdl), _mm256_loadu_ps(xob + 16), acc);
      acc = _mm256_fmadd_ps(_mm256_mul_ps(hi1, vdl), _mm256_loadu_ps(xob + 24), acc);
    }
  }
  return hsum256(acc);
}

/* BF16: widen the top-16-bit truncated float by shifting into the f32 mantissa
 * (exact, no rounding), then FMA. Tail handled scalar. */
float oc_dot_bf16_avx2(const uint8_t* row, const float* x, size_t cols) {
  __m256 acc = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 8 <= cols; i += 8) {
    __m128i u16 = _mm_loadu_si128((const __m128i*)(row + 2 * i));
    __m256i u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(u16), 16);
    acc = _mm256_fmadd_ps(_mm256_castsi256_ps(u32), _mm256_loadu_ps(x + i), acc);
  }
  float sum = hsum256(acc);
  for (; i < cols; ++i) {
    union { uint32_t u; float f; } cvt;
    cvt.u = (uint32_t)rd16(row + 2 * i) << 16;
    sum += cvt.f * x[i];
  }
  return sum;
}
