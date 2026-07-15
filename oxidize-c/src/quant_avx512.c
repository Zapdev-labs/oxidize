/* AVX512F/BW/VL/DQ kernels (16 lanes, half the iterations of the AVX2 paths).
 * DQ is not optional: _mm512_and_ps in the activation quantizer is a DQ
 * instruction. Every AVX-512 part that is not a Xeon Phi has F/BW/VL/DQ/CD.
 * Deliberately NOT compiled with -mavx512vnni, so the compiler cannot slip a
 * vpdpbusd into a kernel that must also run on a Skylake-X, which has no VNNI.
 * Reachable only through the dispatch table in quant.c. */
#include "quant_impl.h"

#if !defined(__AVX512F__) || !defined(__AVX512BW__) || !defined(__AVX512VL__) || \
    !defined(__AVX512DQ__) || !defined(__FMA__) || !defined(__F16C__)
#error "build quant_avx512.c with -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma -mf16c"
#endif

#include <immintrin.h>

/* Rank-kb update, 16-token tile (one zmm). n is a multiple of 16, so no tail. */
void oc_gemm_row_avx512(float* acc, const float* w, const float* xp, size_t kb,
                        size_t n) {
  for (size_t t = 0; t + 16 <= n; t += 16) {
    __m512 a0 = _mm512_loadu_ps(acc + t);
    const float* xs = xp + t;
    for (size_t k = 0; k < kb; ++k, xs += n)
      a0 = _mm512_fmadd_ps(_mm512_set1_ps(w[k]), _mm512_loadu_ps(xs), a0);
    _mm512_storeu_ps(acc + t, a0);
  }
}

/* Four adjacent weight rows, 16-token zmm tile. Four accumulator chains and one
 * panel load feeding four FMAs. See the AVX2 twin for why four. */
void oc_gemm_row4_avx512(float* acc, const float* w, size_t ws, const float* xp,
                         size_t kb, size_t n) {
  const float *w0 = w, *w1 = w + ws, *w2 = w + 2 * ws, *w3 = w + 3 * ws;
  float *p1 = acc + n, *p2 = acc + 2 * n, *p3 = acc + 3 * n;
  for (size_t t = 0; t + 16 <= n; t += 16) {
    __m512 a = _mm512_loadu_ps(acc + t);
    __m512 b = _mm512_loadu_ps(p1 + t);
    __m512 c = _mm512_loadu_ps(p2 + t);
    __m512 d = _mm512_loadu_ps(p3 + t);
    const float* xs = xp + t;
    for (size_t k = 0; k < kb; ++k, xs += n) {
      __m512 x0 = _mm512_loadu_ps(xs);
      a = _mm512_fmadd_ps(_mm512_set1_ps(w0[k]), x0, a);
      b = _mm512_fmadd_ps(_mm512_set1_ps(w1[k]), x0, b);
      c = _mm512_fmadd_ps(_mm512_set1_ps(w2[k]), x0, c);
      d = _mm512_fmadd_ps(_mm512_set1_ps(w3[k]), x0, d);
    }
    _mm512_storeu_ps(acc + t, a);
    _mm512_storeu_ps(p1 + t, b);
    _mm512_storeu_ps(p2 + t, c);
    _mm512_storeu_ps(p3 + t, d);
  }
}

float oc_dot_al5xs_avx512(const uint8_t* row, const float* x, size_t cols) {
  /* Whole 14-byte block decoded from one fault-safe masked load: broadcast the
   * block to all four 128-lanes, pshufb each dword to the 4 bytes holding its
   * 3-bit code, then per-lane shift+mask. P(b) = byte indices b..b+3. */
#define OC_P(b) (int)((unsigned)(b) | ((unsigned)(b) + 1) << 8 | \
                      ((unsigned)(b) + 2) << 16 | ((unsigned)(b) + 3) << 24)
  const __m512i pat0 = _mm512_setr_epi32(OC_P(2), OC_P(2), OC_P(2), OC_P(2),
                                         OC_P(2), OC_P(2), OC_P(2), OC_P(2),
                                         OC_P(5), OC_P(5), OC_P(5), OC_P(5),
                                         OC_P(5), OC_P(5), OC_P(5), OC_P(5));
  const __m512i pat1 = _mm512_setr_epi32(OC_P(8), OC_P(8), OC_P(8), OC_P(8),
                                         OC_P(8), OC_P(8), OC_P(8), OC_P(8),
                                         OC_P(11), OC_P(11), OC_P(11), OC_P(11),
                                         OC_P(11), OC_P(11), OC_P(11), OC_P(11));
#undef OC_P
  const __m512i shifts = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21,
                                           0, 3, 6, 9, 12, 15, 18, 21);
  const __m512i mask7 = _mm512_set1_epi32(7);
  const __m512 four = _mm512_set1_ps(4.0f);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  for (size_t b = 0; b < cols / OC_QK; ++b) {
    const uint8_t* blk = row + b * OC_BLK_AL5_XS;
    float scale = al5xs_scale(blk);
    const float* xb = x + b * OC_QK;
    __m512i q = _mm512_broadcast_i32x4(_mm_maskz_loadu_epi8(0x3FFF, blk));
    __m512i c0 = _mm512_and_si512(
        _mm512_srlv_epi32(_mm512_shuffle_epi8(q, pat0), shifts), mask7);
    __m512i c1 = _mm512_and_si512(
        _mm512_srlv_epi32(_mm512_shuffle_epi8(q, pat1), shifts), mask7);
    __m512 s = _mm512_set1_ps(scale);
    __m512 w0 = _mm512_mul_ps(_mm512_sub_ps(_mm512_cvtepi32_ps(c0), four), s);
    __m512 w1 = _mm512_mul_ps(_mm512_sub_ps(_mm512_cvtepi32_ps(c1), four), s);
    acc0 = _mm512_fmadd_ps(w0, _mm512_loadu_ps(xb), acc0);
    acc1 = _mm512_fmadd_ps(w1, _mm512_loadu_ps(xb + 16), acc1);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

float oc_dot_q4_k_avx512(const uint8_t* row, const float* x, size_t cols) {
  const __m512i maskF = _mm512_set1_epi32(0xF);
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
      __m512 alo = _mm512_setzero_ps(), ahi = _mm512_setzero_ps();
      __m512 slo = _mm512_setzero_ps(), shi = _mm512_setzero_ps();
      for (int j = 0; j < 32; j += 16) {
        __m512i qi = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(q + j)));
        __m512 lo = _mm512_cvtepi32_ps(_mm512_and_si512(qi, maskF));
        __m512 hi = _mm512_cvtepi32_ps(_mm512_srli_epi32(qi, 4));
        __m512 xl = _mm512_loadu_ps(xlo + j), xh = _mm512_loadu_ps(xhi + j);
        alo = _mm512_fmadd_ps(lo, xl, alo);
        ahi = _mm512_fmadd_ps(hi, xh, ahi);
        slo = _mm512_add_ps(slo, xl);
        shi = _mm512_add_ps(shi, xh);
      }
      sum += d * sc1 * _mm512_reduce_add_ps(alo) - min * m1 * _mm512_reduce_add_ps(slo);
      sum += d * sc2 * _mm512_reduce_add_ps(ahi) - min * m2 * _mm512_reduce_add_ps(shi);
      is += 2;
    }
  }
  return sum;
}

float oc_dot_q5_k_avx512(const uint8_t* row, const float* x, size_t cols) {
  const __m512i maskF = _mm512_set1_epi32(0xF);
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
    __m512 acc = _mm512_setzero_ps(), macc = _mm512_setzero_ps();
    for (int gp = 0; gp < 4; ++gp) {
      uint8_t sc1, m1, sc2, m2;
      get_scale_min_k4(is, scales, &sc1, &m1);
      get_scale_min_k4(is + 1, scales, &sc2, &m2);
      const float* xlo = xb + (size_t)gp * 64;
      const float* xhi = xlo + 32;
      __m512 vs1 = _mm512_set1_ps(d * sc1), vm1 = _mm512_set1_ps(min * m1);
      __m512 vs2 = _mm512_set1_ps(d * sc2), vm2 = _mm512_set1_ps(min * m2);
      for (int j = 0; j < 32; j += 16) {
        __m512i qi = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(ql + j)));
        __m512i hi = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(qh + j)));
        __m512i lo5 = _mm512_or_si512(_mm512_and_si512(qi, maskF),
            _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(hi, 2 * gp),
                                               _mm512_set1_epi32(1)), 4));
        __m512i hi5 = _mm512_or_si512(_mm512_srli_epi32(qi, 4),
            _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(hi, 2 * gp + 1),
                                               _mm512_set1_epi32(1)), 4));
        __m512 xl = _mm512_loadu_ps(xlo + j), xh = _mm512_loadu_ps(xhi + j);
        acc = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(lo5), vs1), xl, acc);
        acc = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(hi5), vs2), xh, acc);
        macc = _mm512_fmadd_ps(vm1, xl, macc);
        macc = _mm512_fmadd_ps(vm2, xh, macc);
      }
      ql += 32;
      is += 2;
    }
    sum += _mm512_reduce_add_ps(acc) - _mm512_reduce_add_ps(macc);
  }
  return sum;
}

float oc_dot_q6_k_avx512(const uint8_t* row, const float* x, size_t cols) {
  const __m512i maskF = _mm512_set1_epi32(0xF);
  const __m512i mask3 = _mm512_set1_epi32(3);
  const __m512i off32 = _mm512_set1_epi32(32);
  float sum = 0.0f;
  for (size_t b = 0; b < cols / OC_QK_K; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q6_K;
    float d = oc_f16_fast(rd16(blk + 208));
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    const float* xb = x + b * OC_QK_K;
    __m512 acc = _mm512_setzero_ps();
    for (int g = 0; g < 2; ++g) {
      const uint8_t* qlg = ql + g * 64;
      const uint8_t* qhg = qh + g * 32;
      const int8_t* scg = sc + g * 8;
      const float* xg = xb + g * 128;
      for (int j = 0; j < 32; j += 16) {
        __m512i l0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(qlg + j)));
        __m512i l32 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(qlg + j + 32)));
        __m512i h = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(qhg + j)));
        __m512i q1 = _mm512_sub_epi32(_mm512_or_si512(_mm512_and_si512(l0, maskF),
            _mm512_slli_epi32(_mm512_and_si512(h, mask3), 4)), off32);
        __m512i q2 = _mm512_sub_epi32(_mm512_or_si512(_mm512_and_si512(l32, maskF),
            _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(h, 2), mask3), 4)), off32);
        __m512i q3 = _mm512_sub_epi32(_mm512_or_si512(_mm512_srli_epi32(l0, 4),
            _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(h, 4), mask3), 4)), off32);
        __m512i q4 = _mm512_sub_epi32(_mm512_or_si512(_mm512_srli_epi32(l32, 4),
            _mm512_slli_epi32(_mm512_srli_epi32(h, 6), 4)), off32);
        int isj = j / 16;
        acc = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(q1),
            _mm512_set1_ps(d * scg[isj])), _mm512_loadu_ps(xg + j), acc);
        acc = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(q2),
            _mm512_set1_ps(d * scg[isj + 2])), _mm512_loadu_ps(xg + 32 + j), acc);
        acc = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(q3),
            _mm512_set1_ps(d * scg[isj + 4])), _mm512_loadu_ps(xg + 64 + j), acc);
        acc = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(q4),
            _mm512_set1_ps(d * scg[isj + 6])), _mm512_loadu_ps(xg + 96 + j), acc);
      }
    }
    sum += _mm512_reduce_add_ps(acc);
  }
  return sum;
}

/* Activation quantizer for the VNNI dot kernels. AVX512F only: it is bound
 * whenever AVX-512 is selected, VNNI or not (the dot kernels that consume it
 * are the ones that need VNNI). */
void oc_q8_quantize_avx512(const float* x, size_t n, int8_t* q, float* d,
                           int32_t* bsum) {
  const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
  for (size_t b = 0; b < n / OC_QK_K; ++b) {
    const float* xb = x + b * OC_QK_K;
    __m512 vmax = _mm512_setzero_ps();
    for (int i = 0; i < OC_QK_K; i += 16)
      vmax = _mm512_max_ps(vmax, _mm512_and_ps(_mm512_loadu_ps(xb + i), sign_mask));
    float amax = _mm512_reduce_max_ps(vmax);
    d[b] = amax / 127.0f;
    __m512 vinv = _mm512_set1_ps(amax > 0.0f ? 127.0f / amax : 0.0f);
    int8_t* qb = q + b * OC_QK_K;
    int32_t* sb = bsum + b * 16;
    for (int g = 0; g < 16; ++g) {
      __m512i vi = _mm512_cvtps_epi32( /* round-to-nearest-even */
          _mm512_mul_ps(_mm512_loadu_ps(xb + g * 16), vinv));
      _mm_storeu_si128((__m128i*)(qb + g * 16), _mm512_cvtsepi32_epi8(vi));
      sb[g] = _mm512_reduce_add_epi32(vi);
    }
  }
}
