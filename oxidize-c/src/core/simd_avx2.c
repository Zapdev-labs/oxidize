/*
 * simd_avx2.c — AVX2 + FMA + F16C dequant kernels.
 *
 * Compiled into EVERY build (no -mavx2 on the compile line). Each kernel is
 * annotated `__attribute__((target("avx2,fma,f16c")))` so gcc/clang emit
 * AVX2 instructions for that function only; the dispatcher calls them solely
 * on hosts where `__builtin_cpu_supports("avx2")` (and fma + f16c) are true.
 *
 * Bit-exactness with the scalar reference in src/compute/quantization.c:
 *   - Q4_0: out = (nibble - 8) * d            → single rounded mul, exact.
 *   - Q4_1: out = nibble * d + m             → vmulps + vaddps (NOT FMA).
 *   - Q8_0: out = (int8) * d                  → single rounded mul, exact.
 *   - Q4_K: out = d_sub * nibble - min_sub   → vmulps + vsubps (NOT FMA).
 *   - f16 → f32: vcvtph2ps is the canonical conversion (f16 is a strict
 *     subset of f32); bit-identical to the scalar bit-twiddle.
 *
 * Tests/test_simd.c asserts byte-for-byte equality between each kernel and
 * the scalar reference on randomized inputs (VAL-SIMD-001..004).
 */
#include "oxidize/simd.h"

#if defined(__x86_64__) || defined(__i386__)

#include <immintrin.h>
#include <stdint.h>

/* ─── Shared helpers ──────────────────────────────────────────────────── */

/* Broadcast a little-endian f16 pair at `p` to 8×f32 via F16C. The f16 → f32
 * conversion is the canonical one (f16 strictly representable in f32), so
 * the result is bit-identical to the scalar f16_le_to_f32 in quantization.c. */
static inline __m256 __attribute__((target("avx2,fma,f16c")))
f16_broadcast_f32x8(const uint8_t *p)
{
    uint16_t h = (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
    __m128i hi = _mm_set1_epi16((int16_t)h);
    __m128  f4 = _mm_cvtph_ps(hi);                 /* 4×f32, all equal */
    __m256  f8 = _mm256_castps128_ps256(f4);
    return _mm256_insertf128_ps(f8, f4, 1);        /* 8×f32 */
}

/* 16×uint8 → two 8×int32 chunks (low 8, high 8). Idiomatic high-half move. */
static inline __m256i __attribute__((target("avx2,fma,f16c")))
cvtepu8_low(const __m128i v)
{
    return _mm256_cvtepu8_epi32(v);
}
static inline __m256i __attribute__((target("avx2,fma,f16c")))
cvtepu8_high(const __m128i v)
{
    return _mm256_cvtepu8_epi32(_mm_srli_si128(v, 8));
}

/* ─── Q8_0 (block: f16 d, 32×int8) ────────────────────────────────────── */

__attribute__((target("avx2,fma,f16c")))
bool oc_simd_dequant_q8_0_avx2(const uint8_t *src, size_t src_len,
                               float *dst, size_t value_count)
{
    if (src_len % OC_BLOCK_Q8_0_SIZE != 0) return false;
    size_t n_blocks = src_len / OC_BLOCK_Q8_0_SIZE;
    if (value_count != n_blocks * OC_QK8_0) return false;

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = src + b * OC_BLOCK_Q8_0_SIZE;
        float *out = dst + b * OC_QK8_0;
        __m256 d = f16_broadcast_f32x8(blk);
        /* 32 int8 = two 16-byte loads; each load gives low-8 + high-8
         * int8 chunks (via cvtepi8 on the load and its shifted half). */
        const __m128i *qs = (const __m128i *)(blk + 2);
        __m128i v0 = _mm_loadu_si128(qs + 0);   /* int8 bytes  0..15 */
        __m128i v1 = _mm_loadu_si128(qs + 1);   /* int8 bytes 16..31 */
        _mm256_storeu_ps(out + 0,
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v0)), d));
        _mm256_storeu_ps(out + 8,
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v0, 8))), d));
        _mm256_storeu_ps(out + 16,
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v1)), d));
        _mm256_storeu_ps(out + 24,
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v1, 8))), d));
    }
    return true;
}

/* ─── Q4_0 (block: f16 d, 16 packed bytes → 32 outputs) ────────────────── */

__attribute__((target("avx2,fma,f16c")))
bool oc_simd_dequant_q4_0_avx2(const uint8_t *src, size_t src_len,
                               float *dst, size_t value_count)
{
    if (src_len % OC_BLOCK_Q4_0_SIZE != 0) return false;
    size_t n_blocks = src_len / OC_BLOCK_Q4_0_SIZE;
    if (value_count != n_blocks * OC_QK4_0) return false;

    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    const __m256i eight    = _mm256_set1_epi32(8);

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = src + b * OC_BLOCK_Q4_0_SIZE;
        float *out = dst + b * OC_QK4_0;
        __m256 d = f16_broadcast_f32x8(blk);
        __m128i packed = _mm_loadu_si128((const __m128i *)(blk + 2)); /* 16 bytes */

        __m128i lo = _mm_and_si128(packed, lo_mask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);

        /* Subtract 8 in int domain (exact), convert to f32, mul by d. */
        __m256 l0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(cvtepu8_low(lo),  eight));
        __m256 l1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(cvtepu8_high(lo), eight));
        __m256 h0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(cvtepu8_low(hi),  eight));
        __m256 h1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(cvtepu8_high(hi), eight));

        _mm256_storeu_ps(out + 0,  _mm256_mul_ps(l0, d));
        _mm256_storeu_ps(out + 8,  _mm256_mul_ps(l1, d));
        _mm256_storeu_ps(out + 16, _mm256_mul_ps(h0, d));
        _mm256_storeu_ps(out + 24, _mm256_mul_ps(h1, d));
    }
    return true;
}

/* ─── Q4_1 (block: f16 d, f16 m, 16 packed bytes → 32 outputs) ─────────── */

__attribute__((target("avx2,fma,f16c")))
bool oc_simd_dequant_q4_1_avx2(const uint8_t *src, size_t src_len,
                               float *dst, size_t value_count)
{
    if (src_len % OC_BLOCK_Q4_1_SIZE != 0) return false;
    size_t n_blocks = src_len / OC_BLOCK_Q4_1_SIZE;
    if (value_count != n_blocks * OC_QK4_1) return false;

    const __m128i lo_mask = _mm_set1_epi8(0x0F);

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = src + b * OC_BLOCK_Q4_1_SIZE;
        float *out = dst + b * OC_QK4_1;
        __m256 d = f16_broadcast_f32x8(blk);
        __m256 m = f16_broadcast_f32x8(blk + 2);
        __m128i packed = _mm_loadu_si128((const __m128i *)(blk + 4));

        __m128i lo = _mm_and_si128(packed, lo_mask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);

        /* Q4_1 layout is INTERLEAVED: out[2i]=low(byte i), out[2i+1]=high.
         * Interleave lo/hi via punpcklbw (low 8 pairs) + punpckhbw (high 8). */
        __m128i int_lo = _mm_unpacklo_epi8(lo, hi); /* [l0,h0,...,l7,h7]   */
        __m128i int_hi = _mm_unpackhi_epi8(lo, hi); /* [l8,h8,...,l15,h15] */

        /* out = nibble*d + m  — separate vmulps + vaddps (NO FMA) for parity. */
        __m256 il0 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(cvtepu8_low(int_lo)),  d), m);
        __m256 il1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(cvtepu8_high(int_lo)), d), m);
        __m256 ih0 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(cvtepu8_low(int_hi)),  d), m);
        __m256 ih1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(cvtepu8_high(int_hi)), d), m);

        _mm256_storeu_ps(out + 0,  il0);
        _mm256_storeu_ps(out + 8,  il1);
        _mm256_storeu_ps(out + 16, ih0);
        _mm256_storeu_ps(out + 24, ih1);
    }
    return true;
}

/* ─── Q4_K (block: f16 d, f16 min, 12 scale bytes, 128 packed bytes) ─────
 *
 * 256 outputs per block, 4 groups × (2 sub-groups × 32 outputs). Each group
 * reads 32 bytes of qs: low nibbles → 32 outputs with (d1,min1), high
 * nibbles of the SAME 32 bytes → 32 outputs with (d2,min2). The 8 (sc,m)
 * pairs are extracted scalarly via the exact get_scale_min_k4 layout; the
 * nibble unpack + mul + sub is SIMD.
 */
static inline void get_scale_min_k4_scalar(size_t j, const uint8_t *scales,
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

__attribute__((target("avx2,fma,f16c")))
bool oc_simd_dequant_q4_k_avx2(const uint8_t *src, size_t src_len,
                               float *dst, size_t value_count)
{
    if (src_len % OC_BLOCK_Q4_K_SIZE != 0) return false;
    size_t n_blocks = src_len / OC_BLOCK_Q4_K_SIZE;
    if (value_count != n_blocks * OC_QK_K) return false;

    const __m256i lo_mask = _mm256_set1_epi8(0x0F);

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = src + b * OC_BLOCK_Q4_K_SIZE;
        float *out = dst + b * OC_QK_K;
        __m256 d_global   = f16_broadcast_f32x8(blk);
        __m256 min_global = f16_broadcast_f32x8(blk + 2);
        const uint8_t *scales = blk + 4;
        const uint8_t *qs = blk + 16;

        size_t is = 0;
        for (size_t gp = 0; gp < 4; gp++) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4_scalar(is,     scales, &sc1, &m1);
            get_scale_min_k4_scalar(is + 1, scales, &sc2, &m2);

            /* d_sub = d * sc  (single mul, exact: sc is integer ≤ 63). */
            __m256 d1  = _mm256_mul_ps(d_global,   _mm256_set1_ps((float)sc1));
            __m256 m1v = _mm256_mul_ps(min_global, _mm256_set1_ps((float)m1));
            __m256 d2  = _mm256_mul_ps(d_global,   _mm256_set1_ps((float)sc2));
            __m256 m2v = _mm256_mul_ps(min_global, _mm256_set1_ps((float)m2));

            /* 32 packed bytes for this group (full 256-bit load). Low
             * nibbles → sub-group 1 (32 outputs), high nibbles → sub-group 2. */
            __m256i packed = _mm256_loadu_si256((const __m256i *)(qs + gp * 32));
            __m256i lo = _mm256_and_si256(packed, lo_mask);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), lo_mask);

            /* Split each 32-byte vector into two 16-byte halves, then each
             * half into two 8-lane int32 chunks (4 chunks × 8 = 32 outputs). */
            __m128i lo_low  = _mm256_castsi256_si128(lo);
            __m128i lo_high = _mm256_extracti128_si256(lo, 1);
            __m128i hi_low  = _mm256_castsi256_si128(hi);
            __m128i hi_high = _mm256_extracti128_si256(hi, 1);

            /* Sub-group 1 (low) → out[gp*64 + 0..32] with (d1,m1). */
            _mm256_storeu_ps(out + gp * 64 + 0,
                _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo_low)),  d1), m1v));
            _mm256_storeu_ps(out + gp * 64 + 8,
                _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo_low, 8))),  d1), m1v));
            _mm256_storeu_ps(out + gp * 64 + 16,
                _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo_high)), d1), m1v));
            _mm256_storeu_ps(out + gp * 64 + 24,
                _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo_high, 8))), d1), m1v));

            /* Sub-group 2 (high) → out[gp*64 + 32..64] with (d2,m2). */
            _mm256_storeu_ps(out + gp * 64 + 32,
                _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi_low)),  d2), m2v));
            _mm256_storeu_ps(out + gp * 64 + 40,
                _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi_low, 8))),  d2), m2v));
            _mm256_storeu_ps(out + gp * 64 + 48,
                _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi_high)), d2), m2v));
            _mm256_storeu_ps(out + gp * 64 + 56,
                _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi_high, 8))), d2), m2v));

            is += 2;
        }
    }
    return true;
}

#else

bool oc_simd_dequant_q4_0_avx2(const uint8_t *src, size_t src_len,
                               float *dst, size_t value_count)
{
    (void)src; (void)src_len; (void)dst; (void)value_count;
    return false;
}

bool oc_simd_dequant_q4_1_avx2(const uint8_t *src, size_t src_len,
                               float *dst, size_t value_count)
{
    (void)src; (void)src_len; (void)dst; (void)value_count;
    return false;
}

bool oc_simd_dequant_q8_0_avx2(const uint8_t *src, size_t src_len,
                               float *dst, size_t value_count)
{
    (void)src; (void)src_len; (void)dst; (void)value_count;
    return false;
}

bool oc_simd_dequant_q4_k_avx2(const uint8_t *src, size_t src_len,
                               float *dst, size_t value_count)
{
    (void)src; (void)src_len; (void)dst; (void)value_count;
    return false;
}

#endif
