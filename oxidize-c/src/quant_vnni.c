/* AVX512-VNNI int8-activation dot kernels (vpdpbusd). Compiled with
 * -mavx512vnni on top of the AVX-512 flags; the only TU allowed to see VNNI.
 * Reachable only through the dispatch table in quant.c, which binds these
 * exclusively when cpuid reports AVX512VNNI. */
#include "quant_impl.h"

#if !defined(__AVX512VNNI__) || !defined(__AVX512F__) || !defined(__AVX512BW__) || \
    !defined(__AVX512VL__) || !defined(__AVX512DQ__) || !defined(__F16C__)
#error "build quant_vnni.c with -mavx512vnni -mavx512f -mavx512bw -mavx512vl -mavx512dq -mf16c"
#endif

#include <immintrin.h>

/* Q4_K: dpbusd over 64 u8 nibbles/instruction; per-32 scales via a two-half
 * float multiplier vector; mins folded through the per-32 activation sums. */
float oc_dot_q4_k_vnni(const uint8_t* row, const OcQ8Act* a, size_t cols) {
  const __m256i maskF = _mm256_set1_epi8(0x0F);
  float sum = 0.0f;
  for (size_t b = 0; b < cols / OC_QK_K; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q4_K;
    float dd = oc_f16_fast(rd16(blk));
    float min = oc_f16_fast(rd16(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qs = blk + 16;
    const int8_t* xq = a->q + b * OC_QK_K;
    const int32_t* bs = a->bsum + b * 16;
    float d8 = a->d[b];
    __m512 facc = _mm512_setzero_ps();
    float mins = 0.0f;
    size_t is = 0;
    for (int gp = 0; gp < 4; ++gp) {
      uint8_t sc1, m1, sc2, m2;
      get_scale_min_k4(is, scales, &sc1, &m1);
      get_scale_min_k4(is + 1, scales, &sc2, &m2);
      __m256i bytes = _mm256_loadu_si256((const __m256i*)(qs + gp * 32));
      __m256i lo = _mm256_and_si256(bytes, maskF);
      __m256i hi = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), maskF);
      __m512i qv = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
      __m512i xv = _mm512_loadu_si512((const void*)(xq + gp * 64));
      __m512i idot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), qv, xv);
      __m512 sc = _mm512_insertf32x8(
          _mm512_castps256_ps512(_mm256_set1_ps(dd * sc1)),
          _mm256_set1_ps(dd * sc2), 1);
      facc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(idot), sc, facc);
      int32_t s1 = bs[gp * 4] + bs[gp * 4 + 1], s2 = bs[gp * 4 + 2] + bs[gp * 4 + 3];
      mins += min * (m1 * (float)s1 + m2 * (float)s2);
      is += 2;
    }
    sum += d8 * (_mm512_reduce_add_ps(facc) - mins);
  }
  return sum;
}

/* Q5_K: Q4_K plus the fifth bit from qh. */
float oc_dot_q5_k_vnni(const uint8_t* row, const OcQ8Act* a, size_t cols) {
  const __m256i maskF = _mm256_set1_epi8(0x0F);
  const __m256i one = _mm256_set1_epi8(1);
  const __m256i bit4 = _mm256_set1_epi8(0x10);
  float sum = 0.0f;
  for (size_t b = 0; b < cols / OC_QK_K; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q5_K;
    float dd = oc_f16_fast(rd16(blk));
    float min = oc_f16_fast(rd16(blk + 2));
    const uint8_t* scales = blk + 4;
    const __m256i qhv = _mm256_loadu_si256((const __m256i*)(blk + 16));
    const uint8_t* qs = blk + 48;
    const int8_t* xq = a->q + b * OC_QK_K;
    const int32_t* bs = a->bsum + b * 16;
    float d8 = a->d[b];
    __m512 facc = _mm512_setzero_ps();
    float mins = 0.0f;
    size_t is = 0;
    for (int gp = 0; gp < 4; ++gp) {
      uint8_t sc1, m1, sc2, m2;
      get_scale_min_k4(is, scales, &sc1, &m1);
      get_scale_min_k4(is + 1, scales, &sc2, &m2);
      __m256i bytes = _mm256_loadu_si256((const __m256i*)(qs + gp * 32));
      __m256i hb_lo = _mm256_and_si256(
          _mm256_slli_epi16(_mm256_and_si256(
              _mm256_srli_epi16(qhv, 2 * gp), one), 4), bit4);
      __m256i hb_hi = _mm256_and_si256(
          _mm256_slli_epi16(_mm256_and_si256(
              _mm256_srli_epi16(qhv, 2 * gp + 1), one), 4), bit4);
      __m256i lo = _mm256_or_si256(_mm256_and_si256(bytes, maskF), hb_lo);
      __m256i hi = _mm256_or_si256(
          _mm256_and_si256(_mm256_srli_epi16(bytes, 4), maskF), hb_hi);
      __m512i qv = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
      __m512i xv = _mm512_loadu_si512((const void*)(xq + gp * 64));
      __m512i idot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), qv, xv);
      __m512 sc = _mm512_insertf32x8(
          _mm512_castps256_ps512(_mm256_set1_ps(dd * sc1)),
          _mm256_set1_ps(dd * sc2), 1);
      facc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(idot), sc, facc);
      int32_t s1 = bs[gp * 4] + bs[gp * 4 + 1], s2 = bs[gp * 4 + 2] + bs[gp * 4 + 3];
      mins += min * (m1 * (float)s1 + m2 * (float)s2);
      is += 2;
    }
    sum += d8 * (_mm512_reduce_add_ps(facc) - mins);
  }
  return sum;
}

/* Q6_K: unsigned 6-bit codes via dpbusd; the -32 offset folds through the
 * per-16 activation sums (scales are per 16 values). */
float oc_dot_q6_k_vnni(const uint8_t* row, const OcQ8Act* a, size_t cols) {
  const __m256i maskF = _mm256_set1_epi8(0x0F);
  const __m256i mask30 = _mm256_set1_epi8(0x30);
  float sum = 0.0f;
  for (size_t b = 0; b < cols / OC_QK_K; ++b) {
    const uint8_t* blk = row + b * OC_BLK_Q6_K;
    float dd = oc_f16_fast(rd16(blk + 208));
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    const int8_t* xq = a->q + b * OC_QK_K;
    const int32_t* bs = a->bsum + b * 16;
    float d8 = a->d[b];
    __m256 facc = _mm256_setzero_ps();
    float corr = 0.0f; /* 32 * sc_g * bsum16_g */
    for (int g2 = 0; g2 < 2; ++g2) {
      __m256i l0 = _mm256_loadu_si256((const __m256i*)(ql + g2 * 64));
      __m256i l1 = _mm256_loadu_si256((const __m256i*)(ql + g2 * 64 + 32));
      __m256i h = _mm256_loadu_si256((const __m256i*)(qh + g2 * 32));
      const int8_t* scg = sc + g2 * 8;
      const int32_t* bsg = bs + g2 * 8;
      __m256i runs[4];
      runs[0] = _mm256_or_si256(_mm256_and_si256(l0, maskF),
                                _mm256_and_si256(_mm256_slli_epi16(h, 4), mask30));
      runs[1] = _mm256_or_si256(_mm256_and_si256(l1, maskF),
                                _mm256_and_si256(_mm256_slli_epi16(h, 2), mask30));
      runs[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), maskF),
                                _mm256_and_si256(h, mask30));
      runs[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), maskF),
                                _mm256_and_si256(_mm256_srli_epi16(h, 2), mask30));
      for (int r = 0; r < 4; ++r) {
        __m256i xv = _mm256_loadu_si256((const __m256i*)(xq + g2 * 128 + r * 32));
        __m256i idot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), runs[r], xv);
        float sa = dd * scg[2 * r], sb = dd * scg[2 * r + 1];
        __m256 scv = _mm256_set_m128(_mm_set1_ps(sb), _mm_set1_ps(sa));
        facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(idot), scv, facc);
        corr += 32.0f * (sa * (float)bsg[2 * r] + sb * (float)bsg[2 * r + 1]);
      }
    }
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(facc),
                           _mm256_extractf128_ps(facc, 1));
    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4 = _mm_add_ss(s4, _mm_shuffle_ps(s4, s4, 1));
    sum += d8 * (_mm_cvtss_f32(s4) - corr);
  }
  return sum;
}
