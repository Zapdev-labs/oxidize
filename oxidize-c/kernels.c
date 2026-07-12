/* CPU kernels: rms_norm, RoPE, SwiGLU, softmax, attention, and the gemv/gemm
 * hot path. Speed strategy vs oxidize-cpp: the activation vector is quantized
 * to int8 blocks ONCE per matvec, then Q4_0/Q8_0/Q4_K/Q5_K/Q6_K weight rows are
 * dotted with AVX2 integer maddubs ops (llama.cpp-style) instead of the C++'s
 * nibble->float->FMA path. F16/BF16 rows use fused F16C/shift converts.
 * OpenMP parallelizes over output rows. */
#include "oc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define OC_AVX2 1
#include <immintrin.h>
#endif

/* ---- basic vector ops ---- */

void oc_rms_norm(float *out, const float *x, const float *w, size_t n, float eps) {
  float ss = 0.0f;
  for (size_t i = 0; i < n; ++i) ss += x[i] * x[i];
  float inv = 1.0f / sqrtf(ss / (float)n + eps);
  for (size_t i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}

/* YaRN (ggml rope_yarn): NTK-by-parts frequency interpolation + magnitude
 * scale. yarn_factor <= 1 disables. beta_fast=32, beta_slow=1 (ggml defaults). */
static float yarn_corr_dim(float n_dims, float orig_ctx, float n_rot, float base) {
  return n_dims * logf(orig_ctx / (n_rot * 2.0f * (float)M_PI)) / (2.0f * logf(base));
}

void oc_rope(float *vec, size_t head_dim, size_t n_heads, size_t pos,
             float theta, size_t rope_dim, float yarn_factor, float yarn_orig_ctx,
             const float *ff) {
  size_t rl = rope_dim == 0 || rope_dim > head_dim ? head_dim : rope_dim;
  if (rl == 0) return;
  size_t half = rl / 2;
  float posf = (float)pos;
  float mult = powf(theta, -2.0f / (float)rl);
  bool yarn = yarn_factor > 1.0f && yarn_orig_ctx > 0.0f;
  float mscale = 1.0f, corr_lo = 0.0f, corr_hi = 0.0f, freq_scale = 1.0f;
  if (yarn) {
    freq_scale = 1.0f / yarn_factor;
    mscale = 1.0f + 0.1f * logf(yarn_factor);
    corr_lo = floorf(yarn_corr_dim((float)rl, yarn_orig_ctx, 32.0f, theta));
    corr_hi = ceilf(yarn_corr_dim((float)rl, yarn_orig_ctx, 1.0f, theta));
    if (corr_lo < 0) corr_lo = 0;
    if (corr_hi > (float)rl - 1) corr_hi = (float)rl - 1;
  } else if (pos == 0 && !ff) {
    return; /* identity without yarn magnitude scaling */
  }
  for (size_t h = 0; h < n_heads; ++h) {
    float *hp = vec + h * head_dim;
    float freq = 1.0f;
    for (size_t i = 0; i < half; ++i) {
      float f_i = ff ? freq / ff[i] : freq;
      float ang = posf * f_i;
      float c, s;
      if (yarn) {
        float theta_extrap = posf * f_i;
        float theta_interp = theta_extrap * freq_scale;
        float denom = corr_hi - corr_lo;
        float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, ((float)(2 * i) / 2.0f - corr_lo) /
                                                     (denom > 0.001f ? denom : 0.001f)));
        ang = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
        c = cosf(ang) * mscale;
        s = sinf(ang) * mscale;
      } else {
        c = cosf(ang);
        s = sinf(ang);
      }
      float x0 = hp[i], x1 = hp[half + i];
      hp[i] = x0 * c - x1 * s;
      hp[half + i] = x0 * s + x1 * c;
      freq *= mult;
    }
  }
}

void oc_swiglu(float *gate, const float *up, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float g = gate[i];
    gate[i] = g / (1.0f + expf(-g)) * up[i];
  }
}

void oc_geglu(float *gate, const float *up, size_t n) {
  const float k = 0.7978845608028654f; /* sqrt(2/pi) */
  for (size_t i = 0; i < n; ++i) {
    float g = gate[i];
    gate[i] = 0.5f * g * (1.0f + tanhf(k * (g + 0.044715f * g * g * g))) * up[i];
  }
}

void oc_softmax(float *x, size_t n) {
  if (n == 0) return;
  float mx = x[0];
  for (size_t i = 1; i < n; ++i)
    if (x[i] > mx) mx = x[i];
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    x[i] = expf(x[i] - mx);
    sum += x[i];
  }
  float inv = (float)(1.0 / sum);
  for (size_t i = 0; i < n; ++i) x[i] *= inv;
}

/* fast inline f16 scale load for the per-block hot path (one per 18-34 bytes
 * of weights; the cross-TU oc_f16_to_f32 call cost ~15% of decode) */
#ifdef OC_AVX2
static inline float f16s(const uint8_t *p) {
  uint16_t bits;
  memcpy(&bits, p, 2);
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}
#else
#define f16s oc_f16_to_f32
#endif

#ifdef OC_AVX2
static inline float hsum8(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}
/* sum of 32 signed-int8 pairwise products as 8 x i32 */
static inline __m256i i8dot(__m256i x, __m256i y) {
  __m256i ax = _mm256_sign_epi8(x, x);          /* |x| */
  __m256i sy = _mm256_sign_epi8(y, x);          /* y * sign(x) */
  __m256i p16 = _mm256_maddubs_epi16(ax, sy);   /* u8*i8 -> i16 pairs */
  return _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
}
/* unsigned x (0..15 nibbles) times signed y */
static inline __m256i u8dot(__m256i x, __m256i y) {
  __m256i p16 = _mm256_maddubs_epi16(x, y);
  return _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
}
#endif

float oc_dot_f32(const float *a, const float *b, size_t n) {
#ifdef OC_AVX2
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), a2);
    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), a3);
  }
  for (; i + 8 <= n; i += 8)
    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
  float sum = hsum8(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
  for (; i < n; ++i) sum += a[i] * b[i];
  return sum;
#else
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  float sum = (s0 + s1) + (s2 + s3);
  for (; i < n; ++i) sum += a[i] * b[i];
  return sum;
#endif
}

/* ---- activation quantization (f32 -> int8 blocks of 32) ---- */

void oc_quantize_act(const float *x, oc_q8blk *out, size_t n) {
  for (size_t b = 0; b < n / QK; ++b) {
    const float *xb = x + b * QK;
    float amax = 0.0f;
    for (size_t i = 0; i < QK; ++i) {
      float a = fabsf(xb[i]);
      if (a > amax) amax = a;
    }
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    int sum = 0;
    for (size_t i = 0; i < QK; ++i) {
      int q = (int)lrintf(xb[i] * id);
      if (q > 127) q = 127;
      if (q < -127) q = -127;
      out[b].q[i] = (int8_t)q;
      sum += q;
    }
    out[b].d = d;
    out[b].s = d * (float)sum;
  }
}

/* ---- fused row dots ---- */

/* Q4_0: 18-byte blocks, value = (nib-8)*d; lo nibble -> j, hi -> j+16. */
static float dot_q4_0(const uint8_t *row, const oc_q8blk *xq, size_t cols) {
  size_t nb = cols / QK;
#ifdef OC_AVX2
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  size_t b = 0;
  for (; b + 2 <= nb; b += 2) {
    const uint8_t *blk0 = row + b * 18, *blk1 = blk0 + 18;
    __m128i q40 = _mm_loadu_si128((const __m128i *)(blk0 + 2));
    __m128i q41 = _mm_loadu_si128((const __m128i *)(blk1 + 2));
    __m256i w0 = _mm256_sub_epi8(
        _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q40, 4), _mm_set1_epi8(0x0F)),
                         _mm_and_si128(q40, _mm_set1_epi8(0x0F))),
        _mm256_set1_epi8(8));
    __m256i w1 = _mm256_sub_epi8(
        _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q41, 4), _mm_set1_epi8(0x0F)),
                         _mm_and_si128(q41, _mm_set1_epi8(0x0F))),
        _mm256_set1_epi8(8));
    __m256i y0 = _mm256_loadu_si256((const __m256i *)xq[b].q);
    __m256i y1 = _mm256_loadu_si256((const __m256i *)xq[b + 1].q);
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i8dot(w0, y0)),
                           _mm256_set1_ps(f16s(blk0) * xq[b].d), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i8dot(w1, y1)),
                           _mm256_set1_ps(f16s(blk1) * xq[b + 1].d), acc1);
  }
  for (; b < nb; ++b) {
    const uint8_t *blk = row + b * 18;
    __m128i q4 = _mm_loadu_si128((const __m128i *)(blk + 2));
    __m128i lo = _mm_and_si128(q4, _mm_set1_epi8(0x0F));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(q4, 4), _mm_set1_epi8(0x0F));
    __m256i w = _mm256_sub_epi8(_mm256_set_m128i(hi, lo), _mm256_set1_epi8(8));
    __m256i y = _mm256_loadu_si256((const __m256i *)xq[b].q);
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i8dot(w, y)),
                           _mm256_set1_ps(f16s(blk) * xq[b].d), acc0);
  }
  return hsum8(_mm256_add_ps(acc0, acc1));
#else
  float sum = 0.0f;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t *blk = row + b * 18;
    int isum = 0;
    for (int j = 0; j < 16; ++j) {
      isum += ((blk[2 + j] & 0xF) - 8) * xq[b].q[j];
      isum += ((blk[2 + j] >> 4) - 8) * xq[b].q[j + 16];
    }
    sum += f16s(blk) * xq[b].d * (float)isum;
  }
  return sum;
#endif
}

/* IQ2 superblocks: decode one block at a time instead of the full row. */
static float dot_iq_block(const uint8_t *row, size_t cols, oc_quant q,
                          const float *x) {
  size_t nv = oc_block_values(q);
  size_t bb = oc_block_bytes(q);
  size_t nb = cols / nv;
  float sum = 0.0f;
  float block[QK_K];
  for (size_t b = 0; b < nb; ++b) {
    oc_dequant_row(q, row + b * bb, block, nv);
    sum += oc_dot_f32(block, x + b * nv, nv);
  }
  return sum;
}

static float dot_iq2_xxs(const uint8_t *row, const float *x, size_t cols) {
  return dot_iq_block(row, cols, OC_IQ2_XXS, x);
}
static float dot_iq2_xs(const uint8_t *row, const float *x, size_t cols) {
  return dot_iq_block(row, cols, OC_IQ2_XS, x);
}
static float dot_iq2_s(const uint8_t *row, const float *x, size_t cols) {
  return dot_iq_block(row, cols, OC_IQ2_S, x);
}

/* Q8_0: 34-byte blocks. */
static float dot_q8_0(const uint8_t *row, const oc_q8blk *xq, size_t cols) {
  size_t nb = cols / QK;
#ifdef OC_AVX2
  __m256 acc = _mm256_setzero_ps();
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t *blk = row + b * 34;
    __m256i w = _mm256_loadu_si256((const __m256i *)(blk + 2));
    __m256i y = _mm256_loadu_si256((const __m256i *)xq[b].q);
    __m256 df = _mm256_set1_ps(f16s(blk) * xq[b].d);
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i8dot(w, y)), df, acc);
  }
  return hsum8(acc);
#else
  float sum = 0.0f;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t *blk = row + b * 34;
    int isum = 0;
    for (int j = 0; j < QK; ++j) isum += (int8_t)blk[2 + j] * xq[b].q[j];
    sum += f16s(blk) * xq[b].d * (float)isum;
  }
  return sum;
#endif
}

static void scale_min_k4(size_t j, const uint8_t *s, uint8_t *sc, uint8_t *m) {
  if (j < 4) {
    *sc = s[j] & 63;
    *m = s[j + 4] & 63;
  } else {
    *sc = (uint8_t)((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4));
    *m = (uint8_t)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
  }
}

/* Q4_K: 144-byte superblocks of 256. Per 32-value group g: value = d*sc_g*nib
 * - min*m_g. group dot = d*sc*dx*idot(nib,q8) - min*m * xq.s (precomputed
 * block sum). Nibbles are unsigned 0..15 -> maddubs directly. */
static float dot_q4_k(const uint8_t *row, const oc_q8blk *xq, size_t cols) {
  size_t nsb = cols / QK_K;
  float sum = 0.0f;
  for (size_t sb = 0; sb < nsb; ++sb) {
    const uint8_t *blk = row + sb * 144;
    float d = f16s(blk), min = f16s(blk + 2);
    const uint8_t *scales = blk + 4, *qs = blk + 16;
    const oc_q8blk *x8 = xq + sb * (QK_K / QK);
#ifdef OC_AVX2
    __m256 acc = _mm256_setzero_ps();
    float msum = 0.0f;
    for (int p = 0; p < 4; ++p) {
      uint8_t sc1, m1, sc2, m2;
      scale_min_k4((size_t)p * 2, scales, &sc1, &m1);
      scale_min_k4((size_t)p * 2 + 1, scales, &sc2, &m2);
      __m256i q = _mm256_loadu_si256((const __m256i *)(qs + p * 32));
      __m256i lo = _mm256_and_si256(q, _mm256_set1_epi8(0x0F));
      __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), _mm256_set1_epi8(0x0F));
      const oc_q8blk *b1 = &x8[p * 2], *b2 = &x8[p * 2 + 1];
      __m256i y1 = _mm256_loadu_si256((const __m256i *)b1->q);
      __m256i y2 = _mm256_loadu_si256((const __m256i *)b2->q);
      acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(u8dot(lo, y1)),
                            _mm256_set1_ps(d * (float)sc1 * b1->d), acc);
      acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(u8dot(hi, y2)),
                            _mm256_set1_ps(d * (float)sc2 * b2->d), acc);
      msum += min * (float)m1 * b1->s + min * (float)m2 * b2->s;
    }
    sum += hsum8(acc) - msum;
#else
    for (int p = 0; p < 4; ++p) {
      uint8_t sc1, m1, sc2, m2;
      scale_min_k4((size_t)p * 2, scales, &sc1, &m1);
      scale_min_k4((size_t)p * 2 + 1, scales, &sc2, &m2);
      const oc_q8blk *b1 = &x8[p * 2], *b2 = &x8[p * 2 + 1];
      int i1 = 0, i2 = 0;
      for (int l = 0; l < 32; ++l) {
        i1 += (qs[p * 32 + l] & 0xF) * b1->q[l];
        i2 += (qs[p * 32 + l] >> 4) * b2->q[l];
      }
      sum += d * (float)sc1 * b1->d * (float)i1 - min * (float)m1 * b1->s;
      sum += d * (float)sc2 * b2->d * (float)i2 - min * (float)m2 * b2->s;
    }
#endif
  }
  return sum;
}

/* Q6_K: 210-byte superblocks; value = d*sc[g16]* (6-bit q - 32) where the -32
 * folds into xq.s like a min. Group granularity is 16, so pair two 16-groups
 * per 32-value x block. */
static float dot_q6_k(const uint8_t *row, const oc_q8blk *xq, size_t cols) {
  size_t nsb = cols / QK_K;
  float sum = 0.0f;
  for (size_t sb = 0; sb < nsb; ++sb) {
    const uint8_t *blk = row + sb * 210;
    float d = f16s(blk + 208);
    const uint8_t *ql = blk, *qh = blk + 128;
    const int8_t *sc = (const int8_t *)(blk + 192);
    const oc_q8blk *x8 = xq + sb * (QK_K / QK);
    /* scalar int path (Q6_K rows are rare in decode-heavy tensors; keep simple) */
    for (int g = 0; g < 2; ++g) {
      size_t qlo = (size_t)g * 64, qho = (size_t)g * 32, sco = (size_t)g * 8;
      for (int half = 0; half < 4; ++half) {
        /* halves map to the 4 interleaved 32-value output ranges */
        const oc_q8blk *xb = &x8[g * 4 + half];
        int isum0 = 0, isum1 = 0; /* two 16-value scale groups per 32 values */
        for (int l = 0; l < 32; ++l) {
          int q;
          switch (half) {
            case 0: q = ((ql[qlo + l] & 0xF) | ((qh[qho + l] & 3) << 4)) - 32; break;
            case 1: q = ((ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4)) - 32; break;
            case 2: q = ((ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) - 32; break;
            default: q = ((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) - 32; break;
          }
          if (l < 16) isum0 += q * xb->q[l];
          else isum1 += q * xb->q[l];
        }
        int s0 = sc[sco + (size_t)half * 2], s1 = sc[sco + (size_t)half * 2 + 1];
        sum += d * xb->d * ((float)s0 * (float)isum0 + (float)s1 * (float)isum1);
      }
    }
  }
  return sum;
}

/* IQ4_XS: 136-byte superblocks; value = d*(scale6-32)*kvalues[nib]. Integer
 * dot vs q8 activations; nibbles mapped through the nonlinear LUT. */
static const int8_t oc_kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22,
                                            -10, 1, 13, 25, 38, 53, 69, 89, 113};

static float dot_iq4_xs(const uint8_t *row, const oc_q8blk *xq, size_t cols) {
  size_t nsb = cols / QK_K;
  float sum = 0.0f;
  for (size_t sb = 0; sb < nsb; ++sb) {
    const uint8_t *blk = row + sb * 136;
    float d = f16s(blk);
    uint16_t sh = (uint16_t)(blk[2] | (blk[3] << 8));
    const uint8_t *sl = blk + 4, *qs = blk + 8;
    const oc_q8blk *x8 = xq + sb * (QK_K / QK);
    for (int ib = 0; ib < 8; ++ib) {
      int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4);
      const oc_q8blk *xb = &x8[ib];
      const uint8_t *qp = qs + ib * 16;
#ifdef OC_AVX2
      __m128i q16 = _mm_loadu_si128((const __m128i *)qp);
      __m128i lut = _mm_loadu_si128((const __m128i *)oc_kvalues_iq4nl);
      __m128i lo = _mm_and_si128(q16, _mm_set1_epi8(0x0F));
      __m128i hi = _mm_and_si128(_mm_srli_epi16(q16, 4), _mm_set1_epi8(0x0F));
      __m256i w = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi),
                                   _mm_shuffle_epi8(lut, lo));
      __m256i y = _mm256_loadu_si256((const __m256i *)xb->q);
      __m256i dsum = i8dot(w, y);
      __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(dsum),
                                 _mm256_extracti128_si256(dsum, 1));
      s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
      s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
      int isum = _mm_cvtsi128_si32(s4);
      sum += d * (float)(ls - 32) * xb->d * (float)isum;
#else
      int isum = 0;
      for (int l = 0; l < 16; ++l) {
        isum += oc_kvalues_iq4nl[qp[l] & 0xF] * xb->q[l];
        isum += oc_kvalues_iq4nl[qp[l] >> 4] * xb->q[l + 16];
      }
      sum += d * (float)(ls - 32) * xb->d * (float)isum;
#endif
    }
  }
  return sum;
}

/* IQ4_NL: 18-byte blocks; value = d*kvalues_iq4nl[nib]. Same qs layout as Q4_0. */
static float dot_iq4_nl(const uint8_t *row, const oc_q8blk *xq, size_t cols) {
  size_t nb = cols / QK4_NL;
#ifdef OC_AVX2
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m128i lut = _mm_loadu_si128((const __m128i *)oc_kvalues_iq4nl);
  size_t b = 0;
  for (; b + 2 <= nb; b += 2) {
    const uint8_t *blk0 = row + b * 18, *blk1 = blk0 + 18;
    __m128i q40 = _mm_loadu_si128((const __m128i *)(blk0 + 2));
    __m128i q41 = _mm_loadu_si128((const __m128i *)(blk1 + 2));
    __m128i lo0 = _mm_and_si128(q40, _mm_set1_epi8(0x0F));
    __m128i hi0 = _mm_and_si128(_mm_srli_epi16(q40, 4), _mm_set1_epi8(0x0F));
    __m128i lo1 = _mm_and_si128(q41, _mm_set1_epi8(0x0F));
    __m128i hi1 = _mm_and_si128(_mm_srli_epi16(q41, 4), _mm_set1_epi8(0x0F));
    __m256i w0 = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi0),
                                  _mm_shuffle_epi8(lut, lo0));
    __m256i w1 = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi1),
                                  _mm_shuffle_epi8(lut, lo1));
    __m256i y0 = _mm256_loadu_si256((const __m256i *)xq[b].q);
    __m256i y1 = _mm256_loadu_si256((const __m256i *)xq[b + 1].q);
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i8dot(w0, y0)),
                           _mm256_set1_ps(f16s(blk0) * xq[b].d), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i8dot(w1, y1)),
                           _mm256_set1_ps(f16s(blk1) * xq[b + 1].d), acc1);
  }
  for (; b < nb; ++b) {
    const uint8_t *blk = row + b * 18;
    __m128i q4 = _mm_loadu_si128((const __m128i *)(blk + 2));
    __m128i lo = _mm_and_si128(q4, _mm_set1_epi8(0x0F));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(q4, 4), _mm_set1_epi8(0x0F));
    __m256i w = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi),
                                 _mm_shuffle_epi8(lut, lo));
    __m256i y = _mm256_loadu_si256((const __m256i *)xq[b].q);
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i8dot(w, y)),
                           _mm256_set1_ps(f16s(blk) * xq[b].d), acc0);
  }
  return hsum8(_mm256_add_ps(acc0, acc1));
#else
  float sum = 0.0f;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t *blk = row + b * 18;
    int isum = 0;
    for (int j = 0; j < 16; ++j) {
      isum += oc_kvalues_iq4nl[blk[2 + j] & 0xF] * xq[b].q[j];
      isum += oc_kvalues_iq4nl[blk[2 + j] >> 4] * xq[b].q[j + 16];
    }
    sum += f16s(blk) * xq[b].d * (float)isum;
  }
  return sum;
#endif
}

/* F16 row fused dot */
static float dot_f16(const uint8_t *row, const float *x, size_t n) {
#ifdef OC_AVX2
  const uint16_t *w = (const uint16_t *)row;
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i))),
                         _mm256_loadu_ps(x + i), a0);
    a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 8))),
                         _mm256_loadu_ps(x + i + 8), a1);
    a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 16))),
                         _mm256_loadu_ps(x + i + 16), a2);
    a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 24))),
                         _mm256_loadu_ps(x + i + 24), a3);
  }
  for (; i + 8 <= n; i += 8)
    a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i))),
                         _mm256_loadu_ps(x + i), a0);
  float sum = hsum8(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
  for (; i < n; ++i) sum += oc_f16_to_f32(row + 2 * i) * x[i];
  return sum;
#else
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) sum += oc_f16_to_f32(row + 2 * i) * x[i];
  return sum;
#endif
}

static float dot_bf16(const uint8_t *row, const float *x, size_t n) {
#ifdef OC_AVX2
  const uint16_t *w = (const uint16_t *)row;
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m128i h0 = _mm_loadu_si128((const __m128i *)(w + i));
    __m128i h1 = _mm_loadu_si128((const __m128i *)(w + i + 8));
    __m256 f0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h0), 16));
    __m256 f1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h1), 16));
    a0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x + i), a0);
    a1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x + i + 8), a1);
  }
  float sum = hsum8(_mm256_add_ps(a0, a1));
  for (; i < n; ++i) {
    uint32_t bits = (uint32_t)w[i] << 16;
    float v;
    memcpy(&v, &bits, 4);
    sum += v * x[i];
  }
  return sum;
#else
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    uint32_t bits = (uint32_t)(row[2 * i] | (row[2 * i + 1] << 8)) << 16;
    float v;
    memcpy(&v, &bits, 4);
    sum += v * x[i];
  }
  return sum;
#endif
}

/* one weight row x one input; dispatch on quant. */
static float row_dot(const oc_weight *w, size_t r, size_t cols, const float *x,
                     const oc_q8blk *xq, float *scratch) {
  if (!w->quantized) return oc_dot_f32(w->f32 + r * cols, x, cols);
  size_t rb = oc_row_bytes(w->quant, cols);
  const uint8_t *row = w->data + r * rb;
  switch (w->quant) {
    case OC_F32: return oc_dot_f32((const float *)row, x, cols);
    case OC_F16: return dot_f16(row, x, cols);
    case OC_BF16: return dot_bf16(row, x, cols);
    case OC_Q4_0: case OC_AL5: return dot_q4_0(row, xq, cols);
    case OC_Q8_0: case OC_AL8: return dot_q8_0(row, xq, cols);
    case OC_Q4_K: return dot_q4_k(row, xq, cols);
    case OC_Q6_K: return dot_q6_k(row, xq, cols);
    case OC_IQ4_XS: return dot_iq4_xs(row, xq, cols);
    case OC_IQ4_NL: return dot_iq4_nl(row, xq, cols);
    case OC_IQ2_XXS: return dot_iq2_xxs(row, x, cols);
    case OC_IQ2_XS: return dot_iq2_xs(row, x, cols);
    case OC_IQ2_S: return dot_iq2_s(row, x, cols);
    default:
      oc_dequant_row(w->quant, row, scratch, cols);
      return oc_dot_f32(scratch, x, cols);
  }
}

static bool needs_q8(const oc_weight *w) {
  return w->quantized && (w->quant == OC_Q4_0 || w->quant == OC_AL5 ||
                          w->quant == OC_Q8_0 ||
                          w->quant == OC_Q4_K || w->quant == OC_Q6_K ||
                          w->quant == OC_IQ4_XS || w->quant == OC_IQ4_NL);
}

void oc_gemv(const oc_weight *w, size_t rows, size_t cols, const float *x,
             const oc_q8blk *xq, float *y) {
  oc_q8blk *own = NULL;
  if (needs_q8(w) && xq == NULL) {
    own = malloc(cols / QK * sizeof(oc_q8blk));
    oc_quantize_act(x, own, cols);
    xq = own;
  }
#pragma omp parallel
  {
    float *scratch = NULL;
    if (w->quantized && w->quant != OC_F32 && w->quant != OC_F16 &&
        w->quant != OC_BF16 && !needs_q8(w))
      scratch = malloc(cols * sizeof(float));
#pragma omp for schedule(static)
    for (long long r = 0; r < (long long)rows; ++r)
      y[r] = row_dot(w, (size_t)r, cols, x, xq, scratch);
    free(scratch);
  }
  free(own);
}

void oc_gemm(const oc_weight *w, size_t rows, size_t cols, const float *in,
             const oc_q8blk *inq, float *out, size_t batch) {
  if (batch == 1) {
    oc_gemv(w, rows, cols, in, inq, out);
    return;
  }
  oc_q8blk *own = NULL;
  size_t xstride = cols / QK;
  if (needs_q8(w) && inq == NULL) {
    own = malloc(batch * xstride * sizeof(oc_q8blk));
    for (size_t b = 0; b < batch; ++b)
      oc_quantize_act(in + b * cols, own + b * xstride, cols);
    inq = own;
  }
#pragma omp parallel
  {
    float *scratch = NULL;
    if (w->quantized && w->quant != OC_F32 && w->quant != OC_F16 &&
        w->quant != OC_BF16 && !needs_q8(w))
      scratch = malloc(cols * sizeof(float));
#pragma omp for schedule(static)
    for (long long r = 0; r < (long long)rows; ++r) {
      for (size_t b = 0; b < batch; ++b) {
        out[b * rows + (size_t)r] =
            row_dot(w, (size_t)r, cols, in + b * cols,
                    inq ? inq + b * xstride : NULL, scratch);
      }
    }
    free(scratch);
  }
  free(own);
}

void oc_quantize_kv(const float *x, int8_t *q, float *scale, size_t n_kv,
                    size_t hd) {
  for (size_t h = 0; h < n_kv; ++h) {
    const float *xr = x + h * hd;
    int8_t *qr = q + h * hd;
    float amax = 0.0f;
    for (size_t d = 0; d < hd; ++d) {
      float a = fabsf(xr[d]);
      if (a > amax) amax = a;
    }
    float s = amax > 0.0f ? amax / 127.0f : 0.0f;
    float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (size_t d = 0; d < hd; ++d) {
      int v = (int)lrintf(xr[d] * inv);
      if (v > 127) v = 127;
      else if (v < -127) v = -127;
      qr[d] = (int8_t)v;
    }
    scale[h] = s;
  }
}

/* int8 KV attention. Mirrors oc_attention but dequantizes each cached K/V row
 * (int8, one scale per head) on read. Query stays f32.
 * K is dotted directly against the f32 query, avoiding fixed-size scratch. */
void oc_attention_q8(float *out, const float *q, const int8_t *k8,
                     const float *ks, const int8_t *v8, const float *vs,
                     size_t seq_len, size_t n_heads, size_t kv_heads,
                     size_t head_dim, float scale) {
  size_t group = n_heads / kv_heads;
  size_t kv_len = kv_heads * head_dim;
  if (scale == 0.0f) scale = 1.0f / sqrtf((float)head_dim);
#pragma omp parallel for schedule(static)
  for (long long hh = 0; hh < (long long)n_heads; ++hh) {
    size_t h = (size_t)hh;
    size_t kh = h / group;
    size_t kv_off = kh * head_dim;
    const float *qh = q + h * head_dim;
    float *oh = out + h * head_dim;
    float rmax = -INFINITY, rsum = 0.0f;
    memset(oh, 0, head_dim * sizeof(float));
    for (size_t t = 0; t < seq_len; ++t) {
      const int8_t *kr = k8 + t * kv_len + kv_off;
      float sk = ks[t * kv_heads + kh];
      float dot = 0.0f;
      for (size_t d = 0; d < head_dim; ++d) dot += qh[d] * (float)kr[d];
      float score = dot * sk * scale;
      float nmax = rmax > score ? rmax : score;
      float ef = expf(rmax - nmax), es = expf(score - nmax);
      if (ef != 1.0f)
        for (size_t d = 0; d < head_dim; ++d) oh[d] *= ef;
      const int8_t *vr = v8 + t * kv_len + kv_off;
      float sv = vs[t * kv_heads + kh] * es;
      for (size_t d = 0; d < head_dim; ++d) oh[d] += sv * (float)vr[d];
      rsum = rsum * ef + es;
      rmax = nmax;
    }
    if (rsum > 0.0f) {
      float inv = 1.0f / rsum;
      for (size_t d = 0; d < head_dim; ++d) oh[d] *= inv;
    }
  }
}

void oc_attention(float *out, const float *q, const float *k_cache,
                  const float *v_cache, size_t seq_len, size_t n_heads,
                  size_t kv_heads, size_t head_dim, float scale) {
  size_t group = n_heads / kv_heads;
  size_t kv_len = kv_heads * head_dim;
  if (scale == 0.0f) scale = 1.0f / sqrtf((float)head_dim);
#pragma omp parallel for schedule(static)
  for (long long hh = 0; hh < (long long)n_heads; ++hh) {
    size_t h = (size_t)hh;
    size_t kv_off = (h / group) * head_dim;
    const float *qh = q + h * head_dim;
    float *oh = out + h * head_dim;
    float rmax = -INFINITY, rsum = 0.0f;
    memset(oh, 0, head_dim * sizeof(float));
    for (size_t t = 0; t < seq_len; ++t) {
      float score = oc_dot_f32(qh, k_cache + t * kv_len + kv_off, head_dim) * scale;
      float nmax = rmax > score ? rmax : score;
      float ef = expf(rmax - nmax), es = expf(score - nmax);
      if (ef != 1.0f)
        for (size_t d = 0; d < head_dim; ++d) oh[d] *= ef;
      const float *vr = v_cache + t * kv_len + kv_off;
      for (size_t d = 0; d < head_dim; ++d) oh[d] += es * vr[d];
      rsum = rsum * ef + es;
      rmax = nmax;
    }
    if (rsum > 0.0f) {
      float inv = 1.0f / rsum;
      for (size_t d = 0; d < head_dim; ++d) oh[d] *= inv;
    }
  }
}
