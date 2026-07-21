/*
 * simd_avx512.c — AVX-512 BW + DQ + VNNI dequant kernels.
 *
 * Compiled into EVERY build; each kernel is annotated
 * `__attribute__((target("avx512f,avx512bw,avx512dq,avx512vnni")))` so the
 * same binary runs on non-AVX512 hosts (dispatcher selects them only on
 * capable CPUs).
 *
 * Processes 16 elements per vector op (vs 8 for AVX2). Bit-exactness contract
 * is identical to simd_avx2.c — see that file's header comment.
 */
#include "oxidize/simd.h"

#if defined(__x86_64__) || defined(__i386__)

#include <immintrin.h>
#include <stdint.h>

/* Broadcast a little-endian f16 pair at `p` to 16×f32 via AVX-512 F16C. */
static inline __m512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vnni")))
f16_broadcast_f32x16(const uint8_t *p)
{
    uint16_t h = (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
    __m128i hi = _mm_set1_epi16((int16_t)h);
    /* _mm512_cvtph_ps converts the low 16 i16 lanes of a 256-bit input. */
    __m256i wide = _mm256_broadcastsi128_si256(hi); /* 16 copies of h */
    return _mm512_cvtph_ps(wide);                    /* 16×f32 */
}

/* ─── Q8_0 (block: f16 d, 32×int8) ────────────────────────────────────── */

__attribute__((target("avx512f,avx512bw,avx512dq,avx512vnni")))
bool oc_simd_dequant_q8_0_avx512(const uint8_t *src, size_t src_len,
                                 float *dst, size_t value_count)
{
    if (src_len % OC_BLOCK_Q8_0_SIZE != 0) return false;
    size_t n_blocks = src_len / OC_BLOCK_Q8_0_SIZE;
    if (value_count != n_blocks * OC_QK8_0) return false;

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = src + b * OC_BLOCK_Q8_0_SIZE;
        float *out = dst + b * OC_QK8_0;
        __m512 d = f16_broadcast_f32x16(blk);
        const __m256i *qs = (const __m256i *)(blk + 2);
        __m256i v = _mm256_loadu_si256(qs);          /* 32 int8 */
        /* Two 16-lane chunks (low 16 + high 16), each int8 → int32 → f32 → *d. */
        __m512 lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            _mm256_castsi256_si128(v)));
        __m512 hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            _mm256_extracti128_si256(v, 1)));
        _mm512_storeu_ps(out + 0,  _mm512_mul_ps(lo, d));
        _mm512_storeu_ps(out + 16, _mm512_mul_ps(hi, d));
    }
    return true;
}

/* ─── Q4_0 (block: f16 d, 16 packed bytes → 32 outputs) ────────────────── */

__attribute__((target("avx512f,avx512bw,avx512dq,avx512vnni")))
bool oc_simd_dequant_q4_0_avx512(const uint8_t *src, size_t src_len,
                                 float *dst, size_t value_count)
{
    if (src_len % OC_BLOCK_Q4_0_SIZE != 0) return false;
    size_t n_blocks = src_len / OC_BLOCK_Q4_0_SIZE;
    if (value_count != n_blocks * OC_QK4_0) return false;

    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    const __m512i eight   = _mm512_set1_epi32(8);

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = src + b * OC_BLOCK_Q4_0_SIZE;
        float *out = dst + b * OC_QK4_0;
        __m512 d = f16_broadcast_f32x16(blk);
        __m128i packed = _mm_loadu_si128((const __m128i *)(blk + 2));

        __m128i lo = _mm_and_si128(packed, lo_mask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);

        __m512 l = _mm512_cvtepi32_ps(_mm512_sub_epi32(
            _mm512_cvtepu8_epi32(lo), eight));
        __m512 h = _mm512_cvtepi32_ps(_mm512_sub_epi32(
            _mm512_cvtepu8_epi32(hi), eight));

        _mm512_storeu_ps(out + 0,  _mm512_mul_ps(l, d));
        _mm512_storeu_ps(out + 16, _mm512_mul_ps(h, d));
    }
    return true;
}

/* ─── Q4_1 (block: f16 d, f16 m, 16 packed bytes → 32 outputs) ─────────── */

__attribute__((target("avx512f,avx512bw,avx512dq,avx512vnni")))
bool oc_simd_dequant_q4_1_avx512(const uint8_t *src, size_t src_len,
                                 float *dst, size_t value_count)
{
    if (src_len % OC_BLOCK_Q4_1_SIZE != 0) return false;
    size_t n_blocks = src_len / OC_BLOCK_Q4_1_SIZE;
    if (value_count != n_blocks * OC_QK4_1) return false;

    const __m128i lo_mask = _mm_set1_epi8(0x0F);

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = src + b * OC_BLOCK_Q4_1_SIZE;
        float *out = dst + b * OC_QK4_1;
        __m512 d = f16_broadcast_f32x16(blk);
        __m512 m = f16_broadcast_f32x16(blk + 2);
        __m128i packed = _mm_loadu_si128((const __m128i *)(blk + 4));

        __m128i lo = _mm_and_si128(packed, lo_mask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);

        /* Interleaved layout: out[2i]=low, out[2i+1]=high (see scalar). */
        __m128i int_lo = _mm_unpacklo_epi8(lo, hi); /* [l0,h0,...,l7,h7]   */
        __m128i int_hi = _mm_unpackhi_epi8(lo, hi); /* [l8,h8,...,l15,h15] */

        __m512 il = _mm512_add_ps(_mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(int_lo)), d), m);
        __m512 ih = _mm512_add_ps(_mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(int_hi)), d), m);

        _mm512_storeu_ps(out + 0,  il);
        _mm512_storeu_ps(out + 16, ih);
    }
    return true;
}

/* ─── Q4_K ────────────────────────────────────────────────────────────── */

static inline void get_scale_min_k4_scalar_avx512(size_t j, const uint8_t *scales,
                                                   uint8_t *sc, uint8_t *m)
{
    if (j < 4) {
        *sc = scales[j] & 63u;
        *m  = scales[j + 4] & 63u;
    } else {
        *sc = (uint8_t)((scales[j + 4] & 0x0Fu) | ((scales[j - 4] >> 6) << 4));
        *m  = (uint8_t)(((scales[j + 4] >> 4) & 0x0Fu) | ((scales[j] >> 6) << 4));
    }
}

__attribute__((target("avx512f,avx512bw,avx512dq,avx512vnni")))
bool oc_simd_dequant_q4_k_avx512(const uint8_t *src, size_t src_len,
                                 float *dst, size_t value_count)
{
    if (src_len % OC_BLOCK_Q4_K_SIZE != 0) return false;
    size_t n_blocks = src_len / OC_BLOCK_Q4_K_SIZE;
    if (value_count != n_blocks * OC_QK_K) return false;

    const __m256i lo_mask = _mm256_set1_epi8(0x0F);

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = src + b * OC_BLOCK_Q4_K_SIZE;
        float *out = dst + b * OC_QK_K;
        __m512 d_global   = f16_broadcast_f32x16(blk);
        __m512 min_global = f16_broadcast_f32x16(blk + 2);
        const uint8_t *scales = blk + 4;
        const uint8_t *qs = blk + 16;

        size_t is = 0;
        for (size_t gp = 0; gp < 4; gp++) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4_scalar_avx512(is,     scales, &sc1, &m1);
            get_scale_min_k4_scalar_avx512(is + 1, scales, &sc2, &m2);

            __m512 d1  = _mm512_mul_ps(d_global,   _mm512_set1_ps((float)sc1));
            __m512 m1v = _mm512_mul_ps(min_global, _mm512_set1_ps((float)m1));
            __m512 d2  = _mm512_mul_ps(d_global,   _mm512_set1_ps((float)sc2));
            __m512 m2v = _mm512_mul_ps(min_global, _mm512_set1_ps((float)m2));

            /* 32 packed bytes (full 256-bit load). Low → sub-group 1,
             * high → sub-group 2; each split into two 16-lane chunks. */
            __m256i packed = _mm256_loadu_si256((const __m256i *)(qs + gp * 32));
            __m256i lo = _mm256_and_si256(packed, lo_mask);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), lo_mask);

            __m128i lo_low  = _mm256_castsi256_si128(lo);
            __m128i lo_high = _mm256_extracti128_si256(lo, 1);
            __m128i hi_low  = _mm256_castsi256_si128(hi);
            __m128i hi_high = _mm256_extracti128_si256(hi, 1);

            /* Sub-group 1 (low) → out[gp*64 + 0..32] with (d1,m1). */
            _mm512_storeu_ps(out + gp * 64 + 0,
                _mm512_sub_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo_low)),  d1), m1v));
            _mm512_storeu_ps(out + gp * 64 + 16,
                _mm512_sub_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo_high)), d1), m1v));

            /* Sub-group 2 (high) → out[gp*64 + 32..64] with (d2,m2). */
            _mm512_storeu_ps(out + gp * 64 + 32,
                _mm512_sub_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi_low)),  d2), m2v));
            _mm512_storeu_ps(out + gp * 64 + 48,
                _mm512_sub_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi_high)), d2), m2v));

            is += 2;
        }
    }
    return true;
}

#else

bool oc_simd_dequant_q4_0_avx512(const uint8_t *src, size_t src_len,
                                 float *dst, size_t value_count)
{
    (void)src; (void)src_len; (void)dst; (void)value_count;
    return false;
}

bool oc_simd_dequant_q4_1_avx512(const uint8_t *src, size_t src_len,
                                 float *dst, size_t value_count)
{
    (void)src; (void)src_len; (void)dst; (void)value_count;
    return false;
}

bool oc_simd_dequant_q8_0_avx512(const uint8_t *src, size_t src_len,
                                 float *dst, size_t value_count)
{
    (void)src; (void)src_len; (void)dst; (void)value_count;
    return false;
}

bool oc_simd_dequant_q4_k_avx512(const uint8_t *src, size_t src_len,
                                 float *dst, size_t value_count)
{
    (void)src; (void)src_len; (void)dst; (void)value_count;
    return false;
}

#endif
