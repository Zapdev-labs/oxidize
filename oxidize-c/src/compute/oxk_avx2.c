/*
 * oxk_avx2.c — AVX2 optimized OXK kernels.
 *
 * Real AVX2 implementations for the K-quant dots (Q4_K, Q5_K, Q6_K,
 * Q2_K, Q3_K, IQ3_S), the Q8_0-activation family, and the prepared-row
 * kernels the batched path runs. Q4_K matters most for K-quant models;
 * Q2_K dominates 2-bit quants (UD-Q2_K_XL is ~19% Q2_K by weight bytes).
 *
 * Every vectorized kernel here must be BIT-EXACT against its scalar
 * counterpart, not merely close; test_oxk_avx2_parity.c enforces that with a
 * raw float comparison.
 *
 * All functions use __attribute__((target("avx2,fma,f16c"))) and are
 * present in every build. Callers must check oc_oxk_caps()->level >=
 * OC_OXK_AVX2 before calling.
 */
#include "oxidize/oxk.h"

#include <string.h>

/* AVX2/AVX-512 intrinsics are x86-only. On other targets (e.g. aarch64
 * cross-compile) the real bodies are replaced by scalar-forwarding stubs at
 * the bottom of this file so the OXK symbols still link — the dispatcher in
 * oxk.c never selects them there (oc_simd_caps() reports SCALAR). */
#if defined(__x86_64__) || defined(__i386__)

#include <immintrin.h>

#include "quant_tables.h"

/* Software prefetch distance for the streaming dot kernels. Eight threads
 * reading separate weight streams leave the hardware prefetcher short of
 * lookahead on the compute-bound IQ3_S kernel: prefetching ~1 KB ahead took
 * an 8-thread IQ3_S matvec from 27.9 to 30.7 GB/s on a Ryzen 6850H (512 B
 * and 2 KB were within 2%). Prefetches past the end of a matrix are
 * harmless hints. */
#define OXK_PREFETCH_AHEAD 1024

/* ─── Shared helpers ───────────────────────────────────────────────────── */

/* Horizontal sum of eight int32 lanes. */
__attribute__((target("avx2,f16c")))
static inline int32_t hsum_i32_8(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/* Block scale without a call: the out-of-line bit-twiddle converter forced
 * a vzeroupper and a reload of every vector constant once per block, which
 * cost ~15% of single-thread matvec throughput (3.3 -> 3.9 GB/s on a
 * Ryzen 6850H). F16C is exact for every f16 input. */
__attribute__((target("avx2,f16c")))
static inline float oxk_f16(const uint8_t *p)
{
    uint16_t h;
    memcpy(&h, p, 2);
    return _cvtsh_ss(h);
}

/* Sum of a packed Q8_0 activation block's 32 int8 codes as one int32.
 * sad_epu8 against zero sums unsigned bytes; biasing each code by +128 via
 * the 0x80 xor turns s8 into u8, so the lane total is sum + 32*128. Exact
 * integer math — used by the min-term quants (Q4_1/Q5_1) and the deferred
 * offset correction in the unsigned-code kernels (Q5_0). */
__attribute__((target("avx2,f16c")))
static inline int32_t oxk_act_sum_i8x32(__m256i a)
{
    const __m256i sad = _mm256_sad_epu8(
        _mm256_xor_si256(a, _mm256_set1_epi8((char)0x80)),
        _mm256_setzero_si256());
    const __m128i s = _mm_add_epi64(_mm256_castsi256_si128(sad),
                                    _mm256_extracti128_si256(sad, 1));
    const __m128i t = _mm_add_epi64(s,
        _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    return (int32_t)_mm_cvtsi128_si64(t) - 32 * 128;
}

/* Paired horizontal sum: s[0] = hsum(p0), s[1] = hsum(p1) in ~6 uops
 * instead of two independent hsum_i32_8 chains (~14). Both reduction trees
 * share the same hadd shuffles, so the two block sums fall out of one lane
 * pair. Same integer result either way — this changes no bits. */
__attribute__((target("avx2")))
static inline __m128i hsum2_i32_8(__m256i p0, __m256i p1)
{
    const __m256i h  = _mm256_hadd_epi32(p0, p1);
    const __m256i h2 = _mm256_hadd_epi32(h, h);
    return _mm_add_epi32(_mm256_castsi256_si128(h2),
                         _mm256_extracti128_si256(h2, 1));
}

/* Horizontal sum of eight float lanes. */
__attribute__((target("avx2")))
static inline float hsum_f32_8(__m256 v)
{
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v),
                                _mm256_extractf128_ps(v, 1));
    const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 1)));
}

/* Broadcast table for the 16-group-of-16 weight layouts (Q6_K, Q2_K,
 * Q3_K): entry k selects scale bytes 2k and 2k+1, so a 128-bit pshufb of
 * the 16 scale bytes followed by a widening move puts sc[2k] on the first
 * eight int16 lanes of a maddubs result and sc[2k+1] on the last eight —
 * i.e. the two scale groups a 32-byte code chunk covers. */
static const uint8_t oxk_g16_sc_shuf[8][16] = {
    { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 },
    { 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 },
    { 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5 },
    { 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7 },
    { 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9 },
    { 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11 },
    { 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13 },
    { 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15 },
};

/* ─── AVX2 Q8_0 × Q8_0 dot product ───────────────────────────────────────
 *
 * maddubs cannot be used here: |w| can reach 128 (the -128 code's unsigned
 * |w| is fine) but a -128 activation sign-negated by psignb wraps back to
 * -128, and pushing w into the unsigned slot as w+128 lets pair sums hit
 * 2*255*128 — saturated either way. Widen both to i16 instead: two
 * cvtepi8_epi16 per operand, two madd_epi16, exact for every input. This
 * replaces a body that loaded only 16 of the block's 32 bytes. */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float q8_0_dot_body(const uint8_t *row, size_t blocks,
                                 const uint8_t *q8)
{
    float sum = 0.0f;
    size_t b = 0;
    /* Two blocks per iteration: identical `sum +=` order (still bit-exact),
     * but the widen+hsum chains of the two blocks overlap in the OoO
     * window. */
    for (; b + 2 <= blocks; b += 2) {
        const uint8_t *w0 = row + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *w1 = w0 + OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *q0 = q8 + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *q1 = q0 + OC_OXK_BLOCK_Q8_0_SIZE;
        _mm_prefetch((const char *)(w0 + OXK_PREFETCH_AHEAD), _MM_HINT_T0);
        const __m256i w0v = _mm256_loadu_si256((const __m256i *)(w0 + 2));
        const __m256i w1v = _mm256_loadu_si256((const __m256i *)(w1 + 2));
        const __m256i a0v = _mm256_loadu_si256((const __m256i *)(q0 + 2));
        const __m256i a1v = _mm256_loadu_si256((const __m256i *)(q1 + 2));
        const __m256i p0 = _mm256_add_epi32(
            _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w0v)),
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a0v))),
            _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w0v, 1)),
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a0v, 1))));
        const __m256i p1 = _mm256_add_epi32(
            _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w1v)),
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a1v))),
            _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w1v, 1)),
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a1v, 1))));
        const __m128i s2 = hsum2_i32_8(p0, p1);
        sum += oxk_f16(w0) * oxk_f16(q0)
             * (float)_mm_cvtsi128_si32(s2);
        sum += oxk_f16(w1) * oxk_f16(q1)
             * (float)_mm_extract_epi32(s2, 1);
    }
    for (; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const __m256i w = _mm256_loadu_si256((const __m256i *)(wb + 2));
        const __m256i a = _mm256_loadu_si256((const __m256i *)(qb + 2));
        const __m256i p = _mm256_add_epi32(
            _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w)),
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a))),
            _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1)),
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1))));
        sum += oxk_f16(wb) * oxk_f16(qb) * (float)hsum_i32_8(p);
    }
    return sum;
}

/* ─── Exported single-row forms and the Q8_0-activation dot family ────── */

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q8_0_q8_0_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    return q8_0_dot_body(row, blocks, q8);
}

/* ─── AVX2 Q4_0 × Q8_0 dot product ─────────────────────────────────────── */

/* maddubs form on the raw unsigned codes: w = code - 8, so the integer
 * isum = sum(code*a) - 8*sum(a). Keeping codes in the unsigned slot avoids
 * psignb's -128 wraparound entirely; the activation block sum is integer
 * exact via oxk_act_sum_i8x32. Pair bound 2*15*128 = 3840 — no saturation,
 * so the i32 result is the scalar isum exactly. */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float q4_0_dot_body(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    const __m128i nibble = _mm_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_0_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;
        _mm_prefetch((const char *)(wb + OXK_PREFETCH_AHEAD), _MM_HINT_T0);
        const float dw = oxk_f16(wb);
        const float dq = oxk_f16(qb);
        const __m128i qs = _mm_loadu_si128((const __m128i *)(wb + 2));
        const __m128i lo = _mm_and_si128(qs, nibble);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), nibble);
        /* [lo | hi] in one register matches activation order 0..31. */
        const __m256i c = _mm256_set_m128i(hi, lo);
        const __m256i a = _mm256_loadu_si256((const __m256i *)(qb + 2));
        const int32_t dot = hsum_i32_8(_mm256_madd_epi16(
            _mm256_maddubs_epi16(c, a), ones));
        const int32_t qsum = oxk_act_sum_i8x32(a);
        sum += dw * dq * (float)(dot - 8 * qsum);
    }
    return sum;
}

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q4_0_q8_0_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    return q4_0_dot_body(row, blocks, q8);
}

/* Same kernel with the per-block activation sums supplied rather than
 * re-derived: the matvec quantizes the activation once and dots it against
 * thousands of rows, so hoisting the ~8-op sad chain out of the block loop
 * removes about a quarter of the per-block work. Bit-identical: qsums[b]
 * is exactly the integer oxk_act_sum_i8x32 computes. */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float q4_0_dot_body_qs(const uint8_t *row, size_t blocks,
                                     const uint8_t *q8,
                                     const int32_t *qsums)
{
    const __m128i nibble = _mm_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    float sum = 0.0f;
    size_t b = 0;
    /* Two blocks per iteration: identical `sum +=` order (still bit-exact),
     * but the hsum chains of the two blocks overlap in the OoO window. */
    for (; b + 2 <= blocks; b += 2) {
        const uint8_t *w0 = row + b * OC_OXK_BLOCK_Q4_0_SIZE;
        const uint8_t *w1 = w0 + OC_OXK_BLOCK_Q4_0_SIZE;
        const uint8_t *q0 = q8 + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *q1 = q0 + OC_OXK_BLOCK_Q8_0_SIZE;
        _mm_prefetch((const char *)(w0 + OXK_PREFETCH_AHEAD), _MM_HINT_T0);
        const __m128i q0b = _mm_loadu_si128((const __m128i *)(w0 + 2));
        const __m128i q1b = _mm_loadu_si128((const __m128i *)(w1 + 2));
        const __m256i c0 = _mm256_set_m128i(
            _mm_and_si128(_mm_srli_epi16(q0b, 4), nibble),
            _mm_and_si128(q0b, nibble));
        const __m256i c1 = _mm256_set_m128i(
            _mm_and_si128(_mm_srli_epi16(q1b, 4), nibble),
            _mm_and_si128(q1b, nibble));
        const __m256i a0 = _mm256_loadu_si256((const __m256i *)(q0 + 2));
        const __m256i a1 = _mm256_loadu_si256((const __m256i *)(q1 + 2));
        const __m128i d2 = hsum2_i32_8(
            _mm256_madd_epi16(_mm256_maddubs_epi16(c0, a0), ones),
            _mm256_madd_epi16(_mm256_maddubs_epi16(c1, a1), ones));
        sum += oxk_f16(w0) * oxk_f16(q0)
             * (float)(_mm_cvtsi128_si32(d2) - 8 * qsums[b]);
        sum += oxk_f16(w1) * oxk_f16(q1)
             * (float)(_mm_extract_epi32(d2, 1) - 8 * qsums[b + 1]);
    }
    for (; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_0_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const __m128i qs = _mm_loadu_si128((const __m128i *)(wb + 2));
        const __m256i c = _mm256_set_m128i(
            _mm_and_si128(_mm_srli_epi16(qs, 4), nibble),
            _mm_and_si128(qs, nibble));
        const __m256i a = _mm256_loadu_si256((const __m256i *)(qb + 2));
        const int32_t dot = hsum_i32_8(_mm256_madd_epi16(
            _mm256_maddubs_epi16(c, a), ones));
        sum += oxk_f16(wb) * oxk_f16(qb) * (float)(dot - 8 * qsums[b]);
    }
    return sum;
}

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q4_0_q8_0_qs_avx2(const uint8_t *row, size_t blocks,
                                   const uint8_t *q8, const int32_t *qsums)
{
    return q4_0_dot_body_qs(row, blocks, q8, qsums);
}

/* facc rows forms — the fused-matvec decode path. The per-block integer
 * dots are still exact, but the t*isum products accumulate in f32 lanes
 * (one horizontal sum per row) instead of the reference's block-sequential
 * scalar chain, so results differ from the public dot by a few ulp — the
 * same tradeoff packed_multi already documents for prefill. The public
 * dot/rows kernels stay bit-exact; only the fused matvec calls these. */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float q8_0_dot_body_facc(const uint8_t *row, size_t blocks,
                                       const uint8_t *q8)
{
    __m256 f0 = _mm256_setzero_ps();
    __m256 f1 = _mm256_setzero_ps();
    size_t b = 0;
    for (; b + 2 <= blocks; b += 2) {
        const uint8_t *w0 = row + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *w1 = w0 + OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *q0 = q8 + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *q1 = q0 + OC_OXK_BLOCK_Q8_0_SIZE;
        _mm_prefetch((const char *)(w0 + OXK_PREFETCH_AHEAD), _MM_HINT_T0);
        const __m256i w0v = _mm256_loadu_si256((const __m256i *)(w0 + 2));
        const __m256i w1v = _mm256_loadu_si256((const __m256i *)(w1 + 2));
        const __m256i a0v = _mm256_loadu_si256((const __m256i *)(q0 + 2));
        const __m256i a1v = _mm256_loadu_si256((const __m256i *)(q1 + 2));
        const __m256i p0 = _mm256_add_epi32(
            _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w0v)),
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a0v))),
            _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w0v, 1)),
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a0v, 1))));
        const __m256i p1 = _mm256_add_epi32(
            _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w1v)),
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a1v))),
            _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w1v, 1)),
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a1v, 1))));
        f0 = _mm256_fmadd_ps(
            _mm256_set1_ps(oxk_f16(w0) * oxk_f16(q0)),
            _mm256_cvtepi32_ps(p0), f0);
        f1 = _mm256_fmadd_ps(
            _mm256_set1_ps(oxk_f16(w1) * oxk_f16(q1)),
            _mm256_cvtepi32_ps(p1), f1);
    }
    for (; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const __m256i w = _mm256_loadu_si256((const __m256i *)(wb + 2));
        const __m256i a = _mm256_loadu_si256((const __m256i *)(qb + 2));
        const __m256i p = _mm256_add_epi32(
            _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w)),
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a))),
            _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1)),
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1))));
        f0 = _mm256_fmadd_ps(
            _mm256_set1_ps(oxk_f16(wb) * oxk_f16(qb)),
            _mm256_cvtepi32_ps(p), f0);
    }
    return hsum_f32_8(_mm256_add_ps(f0, f1));
}

__attribute__((target("avx2,fma,f16c")))
void oc_oxk_dot_rows_q8_0_q8_0_facc_avx2(const uint8_t *rows,
                                         size_t row_bytes, size_t n_rows,
                                         size_t blocks, const uint8_t *q8,
                                         float *out)
{
    for (size_t r = 0; r < n_rows; r++) {
        _mm_prefetch((const char *)(rows + (r + 1) * row_bytes),
                     _MM_HINT_T0);
        out[r] = q8_0_dot_body_facc(rows + r * row_bytes, blocks, q8);
    }
}

/* Q4_0 facc: maddubs stays on the raw unsigned codes (no psignb hazard),
 * the -8*sum(a) correction accumulates in a scalar chain on otherwise-idle
 * ports — result = hsum(facc) - corr. */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float q4_0_dot_body_facc(const uint8_t *row, size_t blocks,
                                       const uint8_t *q8,
                                       const int32_t *qsums)
{
    const __m128i nibble = _mm_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 f0 = _mm256_setzero_ps();
    __m256 f1 = _mm256_setzero_ps();
    float corr = 0.0f;
    size_t b = 0;
    for (; b + 2 <= blocks; b += 2) {
        const uint8_t *w0 = row + b * OC_OXK_BLOCK_Q4_0_SIZE;
        const uint8_t *w1 = w0 + OC_OXK_BLOCK_Q4_0_SIZE;
        const uint8_t *q0 = q8 + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *q1 = q0 + OC_OXK_BLOCK_Q8_0_SIZE;
        _mm_prefetch((const char *)(w0 + OXK_PREFETCH_AHEAD), _MM_HINT_T0);
        const __m128i q0b = _mm_loadu_si128((const __m128i *)(w0 + 2));
        const __m128i q1b = _mm_loadu_si128((const __m128i *)(w1 + 2));
        const __m256i c0 = _mm256_set_m128i(
            _mm_and_si128(_mm_srli_epi16(q0b, 4), nibble),
            _mm_and_si128(q0b, nibble));
        const __m256i c1 = _mm256_set_m128i(
            _mm_and_si128(_mm_srli_epi16(q1b, 4), nibble),
            _mm_and_si128(q1b, nibble));
        const __m256i a0 = _mm256_loadu_si256((const __m256i *)(q0 + 2));
        const __m256i a1 = _mm256_loadu_si256((const __m256i *)(q1 + 2));
        const __m256i i0 = _mm256_madd_epi16(
            _mm256_maddubs_epi16(c0, a0), ones);
        const __m256i i1 = _mm256_madd_epi16(
            _mm256_maddubs_epi16(c1, a1), ones);
        const float t0 = oxk_f16(w0) * oxk_f16(q0);
        const float t1 = oxk_f16(w1) * oxk_f16(q1);
        f0 = _mm256_fmadd_ps(_mm256_set1_ps(t0), _mm256_cvtepi32_ps(i0), f0);
        f1 = _mm256_fmadd_ps(_mm256_set1_ps(t1), _mm256_cvtepi32_ps(i1), f1);
        corr += t0 * (float)(8 * qsums[b]);
        corr += t1 * (float)(8 * qsums[b + 1]);
    }
    for (; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_0_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const __m128i qs = _mm_loadu_si128((const __m128i *)(wb + 2));
        const __m256i c = _mm256_set_m128i(
            _mm_and_si128(_mm_srli_epi16(qs, 4), nibble),
            _mm_and_si128(qs, nibble));
        const __m256i a = _mm256_loadu_si256((const __m256i *)(qb + 2));
        const __m256i i = _mm256_madd_epi16(
            _mm256_maddubs_epi16(c, a), ones);
        const float t = oxk_f16(wb) * oxk_f16(qb);
        f0 = _mm256_fmadd_ps(_mm256_set1_ps(t), _mm256_cvtepi32_ps(i), f0);
        corr += t * (float)(8 * qsums[b]);
    }
    return hsum_f32_8(_mm256_add_ps(f0, f1)) - corr;
}

__attribute__((target("avx2,fma,f16c")))
void oc_oxk_dot_rows_q4_0_q8_0_facc_avx2(const uint8_t *rows,
                                         size_t row_bytes, size_t n_rows,
                                         size_t blocks, const uint8_t *q8,
                                         const int32_t *qsums, float *out)
{
    for (size_t r = 0; r < n_rows; r++) {
        _mm_prefetch((const char *)(rows + (r + 1) * row_bytes),
                     _MM_HINT_T0);
        out[r] = q4_0_dot_body_facc(rows + r * row_bytes, blocks, q8, qsums);
    }
}

__attribute__((target("avx2,fma,f16c")))
void oc_oxk_dot_rows_q4_0_q8_0_qs_avx2(const uint8_t *rows,
                                       size_t row_bytes, size_t n_rows,
                                       size_t blocks, const uint8_t *q8,
                                       const int32_t *qsums, float *out)
{
    for (size_t r = 0; r < n_rows; r++) {
        _mm_prefetch((const char *)(rows + (r + 1) * row_bytes),
                     _MM_HINT_T0);
        out[r] = q4_0_dot_body_qs(rows + r * row_bytes, blocks, q8, qsums);
    }
}

/* ─── AVX2 Q4_1 × Q8_0 dot product ───────────────────────────────────────
 *
 * Real kernel — this symbol forwarded to scalar until now. The value offset
 * stays an addend: the dot scales the raw 0..15 codes and the minimum term
 * mw*dq*q8_sum carries the -8 average, exactly as the scalar reference
 * (including its `sum += A + B` float shape). */

/* Codes stay unsigned: maddubs(u8 nibble, s8 activation) sums q*a pairs
 * (bound 2*15*127, no saturation). The activation block sum the min term
 * needs comes from oxk_act_sum_i8x32 — same integer the scalar reference
 * accumulates, so both products in `dw*dq*dot + mw*dq*qsum` are exact. */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float q4_1_dot_body(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    const __m128i nibble = _mm_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_1_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;
        _mm_prefetch((const char *)(wb + OXK_PREFETCH_AHEAD), _MM_HINT_T0);
        const float dw = oxk_f16(wb);
        const float mw = oxk_f16(wb + 2);
        const float dq = oxk_f16(qb);
        const __m128i qs = _mm_loadu_si128((const __m128i *)(wb + 4));
        const __m128i lo = _mm_and_si128(qs, nibble);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), nibble);
        const __m256i w = _mm256_set_m128i(hi, lo);
        const __m256i a = _mm256_loadu_si256((const __m256i *)(qb + 2));
        const int32_t dot = hsum_i32_8(_mm256_madd_epi16(
            _mm256_maddubs_epi16(w, a), ones));
        /* q8_sum: all 32 activation codes as one integer. */
        const int32_t qsum = oxk_act_sum_i8x32(a);
        const float t = dw * dq * (float)dot + mw * dq * (float)qsum;
        sum += t;
    }
    return sum;
}

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q4_1_q8_0_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    return q4_1_dot_body(row, blocks, q8);
}

/* ─── AVX2 Q5_0 × Q8_0 dot product ───────────────────────────────────────
 *
 * AL6 stores the same bitstream, so it aliases this kernel. The qh dword
 * carries element j's fifth bit: bit j for the low-nibble half (elements
 * 0..15), bit 16+j for the high-nibble half (elements 16..31). */

/* 16 or 0 per byte lane j, iff bit j of a 16-bit word is set — the Q5_0 and
 * Q5_1 high-bit rows as a byte mask that ORs into the 4-bit nibbles. */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline __m128i q5_hbit16(uint32_t bits16)
{
    /* sel: byte j>>3 of the word (little-endian) in lane j; bmask: bit j&7
     * of that byte. Their AND is nonzero exactly when bit j is set. */
    const __m128i src = _mm_cvtsi32_si128((int)(bits16 & 0xFFFFu));
    const __m128i sel = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                       1, 1, 1, 1, 1, 1, 1, 1);
    const __m128i bmask = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char)0x80,
                                        1, 2, 4, 8, 16, 32, 64, (char)0x80);
    const __m128i t = _mm_and_si128(_mm_shuffle_epi8(src, sel), bmask);
    return _mm_andnot_si128(_mm_cmpeq_epi8(t, _mm_setzero_si128()),
                            _mm_set1_epi8(16));
}

__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float q5_0_dot_body(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    const __m128i nibble = _mm_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q5_0_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;
        _mm_prefetch((const char *)(wb + OXK_PREFETCH_AHEAD), _MM_HINT_T0);
        const float dw = oxk_f16(wb);
        const float dq = oxk_f16(qb);
        uint32_t qh;
        memcpy(&qh, wb + 2, 4);
        const __m128i qs = _mm_loadu_si128((const __m128i *)(wb + 6));
        const __m128i lo = _mm_and_si128(qs, nibble);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), nibble);
        const __m128i hlo = q5_hbit16(qh);
        const __m128i hhi = q5_hbit16(qh >> 16);
        /* Codes stay unsigned (0..31): w = code - 16, so
         * isum = sum(code*a) - 16*sum(a). Unsigned codes in the maddubs u8
         * slot sidestep psignb's -128 wraparound; the pair bound
         * 2*31*128 = 7936 never saturates. */
        const __m256i c = _mm256_set_m128i(_mm_or_si128(hi, hhi),
                                         _mm_or_si128(lo, hlo));
        const __m256i a = _mm256_loadu_si256((const __m256i *)(qb + 2));
        const int32_t dot = hsum_i32_8(_mm256_madd_epi16(
            _mm256_maddubs_epi16(c, a), ones));
        const int32_t qsum = oxk_act_sum_i8x32(a);
        sum += dw * dq * (float)(dot - 16 * qsum);
    }
    return sum;
}

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q5_0_q8_0_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    return q5_0_dot_body(row, blocks, q8);
}

/* ─── AVX2 Q5_1 × Q8_0 dot product ─────────────────────────────────────── */

__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float q5_1_dot_body(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    const __m128i nibble = _mm_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q5_1_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;
        _mm_prefetch((const char *)(wb + OXK_PREFETCH_AHEAD), _MM_HINT_T0);
        const float dw = oxk_f16(wb);
        const float mw = oxk_f16(wb + 2);
        const float dq = oxk_f16(qb);
        uint32_t qh;
        memcpy(&qh, wb + 4, 4);
        const __m128i qs = _mm_loadu_si128((const __m128i *)(wb + 8));
        const __m128i lo = _mm_and_si128(qs, nibble);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), nibble);
        const __m128i hlo = q5_hbit16(qh);
        const __m128i hhi = q5_hbit16(qh >> 16);
        const __m256i w = _mm256_set_m128i(_mm_or_si128(hi, hhi),
                                         _mm_or_si128(lo, hlo));
        const __m256i a = _mm256_loadu_si256((const __m256i *)(qb + 2));
        /* Unsigned 5-bit codes 0..31: pair bound 2*31*127 = 7874, safe. */
        const int32_t dot = hsum_i32_8(_mm256_madd_epi16(
            _mm256_maddubs_epi16(w, a), ones));
        const int32_t qsum = oxk_act_sum_i8x32(a);
        const float t = dw * dq * (float)dot + mw * dq * (float)qsum;
        sum += t;
    }
    return sum;
}

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q5_1_q8_0_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    return q5_1_dot_body(row, blocks, q8);
}

/* ─── Packed multi-activation dots (batched prefill) ─────────────────────
 *
 * One weight row dotted against `n_act` packed Q8_0 activations. The batch
 * path otherwise calls the single-row kernel once per (row, activation)
 * pair, which re-decodes the row — nibble unpack, block scale, prefetch —
 * once per activation. Here the decode happens once per block per
 * eight-activation group, the i32 pair-sums convert straight to f32 and
 * accumulate in vector lanes scaled by dw*dq, and the horizontal sum runs
 * once per (row, activation) instead of once per block.
 *
 * Q4_0 puts the BIASED activation a+128 (one xor) in maddubs' unsigned slot
 * and the signed weight w = code-8 in the signed slot, so the pair dot is
 * sum(a*w) + 128*sum(w): the correction is weight-side, computed once per
 * block and shared by every activation — no per-pair activation sum, which
 * is what the single-row kernel spends ~6 ops per block on. Pair bound
 * 2*255*8 = 4080 — no saturation. The -128*wsum correction lands in lane 0
 * of the i32 partial vector before the f32 scale, so each block contributes
 * exactly dw*dq*truedot — the same value the sequential kernel adds, merely
 * associated into eight lanes first (a few ulp of reassociation, like any
 * tiled GEMM; the integer dots themselves are exact). */

__attribute__((target("avx2,fma,f16c")))
void oc_oxk_dot_q4_0_q8_0_packed_multi_avx2(
    const uint8_t *row, size_t blocks, const uint8_t *acts,
    size_t act_stride, size_t n_act, float *out)
{
    const __m128i nibble = _mm_set1_epi8(0x0F);
    const __m256i ones   = _mm256_set1_epi16(1);
    const __m256i sign8  = _mm256_set1_epi8((char)0x80);
    const __m256i eight8 = _mm256_set1_epi8(8);

    for (size_t v0 = 0; v0 < n_act; v0 += 8) {
        const size_t nv = (n_act - v0 < 8u) ? n_act - v0 : 8u;
        /* A fixed bound of 8 keeps the accumulators in registers; tail
         * activations recompute against slot 0 and their results are
         * simply never written. */
        const uint8_t *ap[8];
        for (size_t v = 0; v < 8; v++)
            ap[v] = acts + (v0 + (v < nv ? v : 0)) * act_stride;
        __m256 facc[8] = { _mm256_setzero_ps(), _mm256_setzero_ps(),
                           _mm256_setzero_ps(), _mm256_setzero_ps(),
                           _mm256_setzero_ps(), _mm256_setzero_ps(),
                           _mm256_setzero_ps(), _mm256_setzero_ps() };
        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_0_SIZE;
            _mm_prefetch((const char *)(wb + OXK_PREFETCH_AHEAD),
                         _MM_HINT_T0);
            const float dw = oxk_f16(wb);
            const __m128i qs = _mm_loadu_si128((const __m128i *)(wb + 2));
            const __m128i lo = _mm_and_si128(qs, nibble);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), nibble);
            const __m256i c = _mm256_set_m128i(hi, lo);
            const __m256i w = _mm256_sub_epi8(c, eight8);
            /* +128*wsum into lane 0 only (subtracted below); wsum =
             * sum(c) - 32*8. truedot = sum((a+128)*w) - 128*sum(w). */
            const __m256i sad = _mm256_sad_epu8(c, _mm256_setzero_si256());
            const __m128i ss = _mm_add_epi64(
                _mm256_castsi256_si128(sad),
                _mm256_extracti128_si256(sad, 1));
            const int32_t csum = (int32_t)_mm_cvtsi128_si64(_mm_add_epi64(
                ss, _mm_shuffle_epi32(ss, _MM_SHUFFLE(1, 0, 3, 2))));
            const __m256i corr = _mm256_set_epi32(
                0, 0, 0, 0, 0, 0, 0, 128 * (csum - 256));
            for (size_t v = 0; v < 8; v++) {
                const uint8_t *qb = ap[v] + b * OC_OXK_BLOCK_Q8_0_SIZE;
                const __m256i a = _mm256_xor_si256(
                    _mm256_loadu_si256((const __m256i *)(qb + 2)), sign8);
                const __m256i p = _mm256_sub_epi32(
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a, w), ones),
                    corr);
                facc[v] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p),
                        _mm256_set1_ps(dw * oxk_f16(qb)), facc[v]);
            }
        }
        for (size_t v = 0; v < nv; v++)
            out[v0 + v] = hsum_f32_8(facc[v]);
    }
}

/* Q8_0 cannot use maddubs (a -128 weight in the signed slot lets pair sums
 * hit 2*255*128), so the weight widens to i16 once per block and each
 * activation widens per pair — still exact i32 truedot partials, same
 * f32-lane accumulation as above. */
__attribute__((target("avx2,fma,f16c")))
void oc_oxk_dot_q8_0_q8_0_packed_multi_avx2(
    const uint8_t *row, size_t blocks, const uint8_t *acts,
    size_t act_stride, size_t n_act, float *out)
{
    for (size_t v0 = 0; v0 < n_act; v0 += 8) {
        const size_t nv = (n_act - v0 < 8u) ? n_act - v0 : 8u;
        const uint8_t *ap[8];
        for (size_t v = 0; v < 8; v++)
            ap[v] = acts + (v0 + (v < nv ? v : 0)) * act_stride;
        __m256 facc[8] = { _mm256_setzero_ps(), _mm256_setzero_ps(),
                           _mm256_setzero_ps(), _mm256_setzero_ps(),
                           _mm256_setzero_ps(), _mm256_setzero_ps(),
                           _mm256_setzero_ps(), _mm256_setzero_ps() };
        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *wb = row + b * OC_OXK_BLOCK_Q8_0_SIZE;
            _mm_prefetch((const char *)(wb + OXK_PREFETCH_AHEAD),
                         _MM_HINT_T0);
            const float dw = oxk_f16(wb);
            const __m256i w = _mm256_loadu_si256((const __m256i *)(wb + 2));
            const __m256i wlo = _mm256_cvtepi8_epi16(
                _mm256_castsi256_si128(w));
            const __m256i whi = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(w, 1));
            for (size_t v = 0; v < 8; v++) {
                const uint8_t *qb = ap[v] + b * OC_OXK_BLOCK_Q8_0_SIZE;
                const __m256i a = _mm256_loadu_si256(
                    (const __m256i *)(qb + 2));
                const __m256i p = _mm256_add_epi32(
                    _mm256_madd_epi16(wlo,
                        _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a))),
                    _mm256_madd_epi16(whi,
                        _mm256_cvtepi8_epi16(
                            _mm256_extracti128_si256(a, 1))));
                facc[v] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p),
                        _mm256_set1_ps(dw * oxk_f16(qb)), facc[v]);
            }
        }
        for (size_t v = 0; v < nv; v++)
            out[v0 + v] = hsum_f32_8(facc[v]);
    }
}

/* ─── Quad-blocked Q4_0 × Q8_0 multi-activation dot ──────────────────────
 *
 * packed_multi still pays the full row decode — nibble unpack plus the sad
 * chain for wsum — once per eight-activation group, so a 512-token prefill
 * tile decodes every row 64 times. The quad path instead decodes FOUR rows
 * once per tile into per-thread scratch, then runs a register-blocked
 * 4-row x 4-act inner loop on the prepared form.
 *
 * Prepared quad-block layout (160 bytes per block):
 *   [f32 dw[4]]       block scale per row
 *   [f32 cw[4]]       128*dw*wsum per row — the maddubs bias correction
 *   [i8  w[4][32]]    signed weights c-8
 *
 * Per block the inner loop loads each activation vector once and maddubs it
 * against all four weight rows (a+128 unsigned, w signed, as in
 * packed_multi), so the per-(row,activation) work is just
 * maddubs/madd/cvt/fma plus one vmulps for the dw*dq scale — about five
 * ops per pair where packed_multi spends ~12. The bias correction is not
 * applied per pair: corr[v] accumulates dq_v * cw[r] vectorized over the
 * four rows and is subtracted once at row end, since
 *   out[r,v] = sum_b dw*dq*sum((a+128)*w) - sum_b dq * 128*dw*wsum.
 * Same exact integer dots, same f32 reassociation as packed_multi. */
__attribute__((target("avx2,fma,f16c")))
void oc_oxk_q4_0_quad_prep_avx2(const uint8_t *rows, size_t row_bytes,
                                size_t n_rows, size_t blocks, void *scratch)
{
    const __m128i nibble = _mm_set1_epi8(0x0F);
    const __m256i eight8 = _mm256_set1_epi8(8);
    for (size_t b = 0; b < blocks; b++) {
        uint8_t *pb = (uint8_t *)scratch + b * OC_OXK_Q4_0_QUAD_BLOCK_BYTES;
        float  *dwq = (float *)pb;
        float  *cwq = (float *)(pb + 16);
        int8_t *wq  = (int8_t *)(pb + 32);
        for (size_t r = 0; r < 4; r++) {
            if (r < n_rows) {
                const uint8_t *wb = rows + r * row_bytes
                                      + b * OC_OXK_BLOCK_Q4_0_SIZE;
                const float dw = oxk_f16(wb);
                const __m128i qs = _mm_loadu_si128((const __m128i *)(wb + 2));
                const __m128i lo = _mm_and_si128(qs, nibble);
                const __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4),
                                                 nibble);
                const __m256i c = _mm256_set_m128i(hi, lo);
                const __m256i sad = _mm256_sad_epu8(
                    c, _mm256_setzero_si256());
                const __m128i ss = _mm_add_epi64(
                    _mm256_castsi256_si128(sad),
                    _mm256_extracti128_si256(sad, 1));
                const int32_t csum = (int32_t)_mm_cvtsi128_si64(
                    _mm_add_epi64(ss,
                        _mm_shuffle_epi32(ss, _MM_SHUFFLE(1, 0, 3, 2))));
                dwq[r] = dw;
                cwq[r] = 128.0f * dw * (float)(csum - 256);
                _mm256_storeu_si256((__m256i *)(wq + r * 32),
                                    _mm256_sub_epi8(c, eight8));
            } else {
                /* Padding rows: zero weight and scale produce an exact
                 * zero result that the caller simply never writes out. */
                dwq[r] = 0.0f;
                cwq[r] = 0.0f;
                _mm256_storeu_si256((__m256i *)(wq + r * 32),
                                    _mm256_setzero_si256());
            }
        }
    }
}

__attribute__((target("avx2,fma,f16c"), optimize("O3,unroll-loops")))
void oc_oxk_dot_q4_0_q8_0_quad_multi_avx2(
    const void *prep4, size_t blocks, const uint8_t *acts,
    size_t act_stride, size_t dq_off, size_t n_act, float *res4)
{
    const __m256i ones  = _mm256_set1_epi16(1);
    const __m256i sign8 = _mm256_set1_epi8((char)0x80);
    const uint8_t *prep = (const uint8_t *)prep4;

    for (size_t v = 0; v < n_act; v++) {
        const uint8_t *act = acts + v * act_stride;
        const float   *dp  = (const float *)(act + dq_off);
        __m256 f0 = _mm256_setzero_ps(), f1 = _mm256_setzero_ps(),
               f2 = _mm256_setzero_ps(), f3 = _mm256_setzero_ps();
        __m128 c = _mm_setzero_ps();
        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *pb = prep + b * OC_OXK_Q4_0_QUAD_BLOCK_BYTES;
            _mm_prefetch((const char *)(pb + OXK_PREFETCH_AHEAD),
                         _MM_HINT_T0);
            const uint8_t *qb = act + b * OC_OXK_BLOCK_Q8_0_SIZE;
            const __m256i a = _mm256_xor_si256(
                _mm256_loadu_si256((const __m256i *)(qb + 2)), sign8);
            const __m256 dqv = dq_off ? _mm256_broadcast_ss(dp + b)
                                      : _mm256_set1_ps(oxk_f16(qb));
            const __m128 cw = _mm_loadu_ps((const float *)(pb + 16));
            c = _mm_fmadd_ps(cw, _mm256_castps256_ps128(dqv), c);
            f0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_madd_epi16(
                    _mm256_maddubs_epi16(a, _mm256_loadu_si256(
                        (const __m256i *)(pb + 32))), ones)),
                _mm256_mul_ps(dqv,
                    _mm256_broadcast_ss((const float *)pb)), f0);
            f1 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_madd_epi16(
                    _mm256_maddubs_epi16(a, _mm256_loadu_si256(
                        (const __m256i *)(pb + 64))), ones)),
                _mm256_mul_ps(dqv,
                    _mm256_broadcast_ss((const float *)pb + 1)), f1);
            f2 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_madd_epi16(
                    _mm256_maddubs_epi16(a, _mm256_loadu_si256(
                        (const __m256i *)(pb + 96))), ones)),
                _mm256_mul_ps(dqv,
                    _mm256_broadcast_ss((const float *)pb + 2)), f2);
            f3 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_madd_epi16(
                    _mm256_maddubs_epi16(a, _mm256_loadu_si256(
                        (const __m256i *)(pb + 128))), ones)),
                _mm256_mul_ps(dqv,
                    _mm256_broadcast_ss((const float *)pb + 3)), f3);
        }
        float corr[4];
        _mm_storeu_ps(corr, c);
        float *rp = res4 + v * 4;
        rp[0] = hsum_f32_8(f0) - corr[0];
        rp[1] = hsum_f32_8(f1) - corr[1];
        rp[2] = hsum_f32_8(f2) - corr[2];
        rp[3] = hsum_f32_8(f3) - corr[3];
    }
}

/* ─── AVX2 AL5_XS × Q8_0 dot product ─────────────────────────────────────
 *
 * The custom 14-byte 3-bit block: f16 d + 12 bytes of LSB-first 3-bit codes,
 * code value minus 4. The scalar reference gathers one bit at a time (96
 * shifts/ands per block); here each aligned 3-byte group is one 24-bit
 * little-endian window and eight codes come out of a single srlv+mask. The
 * bytes are read as five in-bounds scalar loads: one u32 load would stride a
 * byte past the last block of the last row of a tensor. */

__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float al5_xs_dot_body(const uint8_t *row, size_t blocks,
                                   const uint8_t *q8)
{
    const __m256i shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    const __m256i v7 = _mm256_set1_epi32(7);
    const __m256i four = _mm256_set1_epi16(4);
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_AL5_XS_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;
        _mm_prefetch((const char *)(wb + OXK_PREFETCH_AHEAD), _MM_HINT_T0);
        const float dw = oxk_f16(wb);
        const float dq = oxk_f16(qb);
        const uint8_t *packed = wb + 2;
        uint32_t p03, p811, L[4];
        uint16_t p45;
        memcpy(&p03, packed, 4);
        memcpy(&p811, packed + 8, 4);
        memcpy(&p45, packed + 4, 2);
        /* Group g covers codes 8g..8g+7 = the 24 bits of bytes 3g..3g+2. */
        L[0] = p03 & 0xFFFFFFu;
        L[1] = ((p03 >> 24) & 0xFFu) | ((uint32_t)p45 << 8);
        L[2] = (uint32_t)packed[6] | ((uint32_t)packed[7] << 8)
              | ((p811 & 0xFFu) << 16);
        L[3] = (p811 >> 8) & 0xFFFFFFu;
        __m256i pos = _mm256_setzero_si256();
        for (int g = 0; g < 4; g += 2) {
            const __m256i c0 = _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32((int)L[g]), shifts), v7);
            const __m256i c1 = _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32((int)L[g + 1]), shifts), v7);
            /* packus+permute put codes 16g..16g+15... correction: elements
             * 8g..8g+15 (two 8-code groups) into 16-bit lane order. */
            const __m256i w = _mm256_sub_epi16(
                _mm256_permute4x64_epi64(
                    _mm256_packus_epi32(c0, c1), _MM_SHUFFLE(3, 1, 2, 0)),
                four);
            const __m256i a = _mm256_cvtepi8_epi16(
                _mm_loadu_si128((const __m128i *)(qb + 2 + 8 * g)));
            pos = _mm256_add_epi32(pos, _mm256_madd_epi16(w, a));
        }
        sum += dw * dq * (float)hsum_i32_8(pos);
    }
    return sum;
}

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_al5_xs_q8_0_avx2(const uint8_t *row, size_t blocks,
                                 const uint8_t *q8)
{
    return al5_xs_dot_body(row, blocks, q8);
}

/* ─── AVX2 Q4_K × Q8_K dot product ─────────────────────────────────────── */

/* Q4_K is the format most models ship in, so this is the kernel that decides
 * throughput in practice.
 *
 * The nibble products are computed with _mm256_maddubs_epi16 (unsigned
 * nibble 0..15 times signed int8 activation -> int16 pair sums) and scaled
 * with _mm256_madd_epi16 against the broadcast 6-bit sub-block scale, which
 * both widens to int32 and applies the scale in one instruction. The mins
 * term is one madd of the eight 6-bit mins against the pairwise-added
 * activation block sums. Everything stays integer until one multiply-add per
 * block, exactly as the scalar reference does, so this is bit-exact against
 * it, not merely close.
 *
 * No out-of-line helper is called inside the block loop: each call forced a
 * vzeroupper and a reload of every vector constant, and the eighteen per
 * block (f16 scales, scale/min unpack, bsum reads) made this kernel ~5x
 * slower than ggml's on a Ryzen 6850H. */
#define OXK_KMASK1 0x3f3f3f3fu
#define OXK_KMASK2 0x0f0f0f0fu
#define OXK_KMASK3 0x03030303u

__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float q4_k_dot_body(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    const __m256i lownib = _mm256_set1_epi8(0x0F);
    float sum = 0.0f;

    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_K_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_K_SIZE;
        _mm_prefetch((const char *)(wb + OXK_PREFETCH_AHEAD), _MM_HINT_T0);
        const float dw   = oxk_f16(wb);
        const float dmin = oxk_f16(wb + 2);
        const uint8_t *qs = wb + 16;
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + 256;

        /* Unpack the twelve scale bytes into sc[0..7] (lanes 0..7) and
         * m[0..7] (lanes 8..15): the same values oc_oxk_get_scale_min_k4
         * yields for j = 0..7 (ggml's utmp shuffle). Kept in registers: a
         * byte array here costs two store-forwarding stalls per block. */
        uint32_t u0, u1, u2;
        memcpy(&u0, wb + 4, 4);
        memcpy(&u1, wb + 8, 4);
        memcpy(&u2, wb + 12, 4);
        const uint32_t s0 = u0 & OXK_KMASK1;
        const uint32_t s1 = (u2 & OXK_KMASK2) | (((u0 >> 6) & OXK_KMASK3) << 4);
        const uint32_t m0 = u1 & OXK_KMASK1;
        const uint32_t m1 = ((u2 >> 4) & OXK_KMASK2) | (((u1 >> 6) & OXK_KMASK3) << 4);
        const __m256i sm16 = _mm256_cvtepu8_epi16(
            _mm_set_epi32((int)m1, (int)m0, (int)s1, (int)s0));

        /* min_acc = sum_j m_j * (bsum[2j] + bsum[2j+1]): each min is
         * duplicated onto its two int16 bsums so madd forms the products and
         * their pair sums directly in int32, where nothing can wrap. */
        const __m128i mhi = _mm256_extracti128_si256(sm16, 1);
        const __m256i mdup = _mm256_set_m128i(_mm_unpackhi_epi16(mhi, mhi),
                                              _mm_unpacklo_epi16(mhi, mhi));
        const int32_t min_acc = hsum_i32_8(_mm256_madd_epi16(
            mdup, _mm256_loadu_si256((const __m256i *)bsums)));

        /* Broadcast each sub-block scale to all 16 int16 lanes. */
        const __m128i sc128 = _mm256_castsi256_si128(sm16);
        const __m256i scales = _mm256_set_m128i(sc128, sc128);

        __m256i pos_v = _mm256_setzero_si256();
        for (int gp = 0; gp < 4; gp++) {
            /* 32 packed bytes = 64 nibbles = sub-blocks 2gp (low) and
             * 2gp+1 (high). */
            const __m256i packed = _mm256_loadu_si256((const __m256i *)(qs + gp * 32));
            const __m256i nib_lo = _mm256_and_si256(packed, lownib);
            const __m256i nib_hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), lownib);
            const __m256i a_lo = _mm256_loadu_si256((const __m256i *)(q8v + gp * 64));
            const __m256i a_hi = _mm256_loadu_si256((const __m256i *)(q8v + gp * 64 + 32));
            const __m256i sc_lo = _mm256_shuffle_epi8(scales,
                _mm256_set1_epi16((short)(0x0100 * (4 * gp + 1) + 4 * gp)));
            const __m256i sc_hi = _mm256_shuffle_epi8(scales,
                _mm256_set1_epi16((short)(0x0100 * (4 * gp + 3) + 4 * gp + 2)));
            /* 15*128*2 = 3840 per int16 pair; times a 6-bit scale and paired
             * again it is < 2^19, far from int32 overflow. */
            const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(nib_lo, a_lo), sc_lo);
            const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(nib_hi, a_hi), sc_hi);
            pos_v = _mm256_add_epi32(pos_v, _mm256_add_epi32(p1, p2));
        }
        sum += dw * dq * (float)hsum_i32_8(pos_v) - dmin * dq * (float)min_acc;
    }
    return sum;
}

/* ─── AVX2 Q5_K × Q8_K dot product ───────────────────────────────────────
 *
 * Real kernel — this symbol forwarded to scalar until now, which meant every
 * AVX2-only host (all Ryzen consumer CPUs) ran Q5_K_M/Q5_K_S files scalar
 * while AVX-512 hosts had a VNNI kernel. Same structure as Q4_K: the eight
 * 6-bit scales/mins come out of the utmp-style decode (scale k lives in the
 * low byte of sm16 lane k, so a shuffle byte selector of 4gp picks scale
 * 2gp and the adjacent odd byte is a guaranteed zero), the mins term is the
 * bsums madd, and the nibble products are maddubs+madd.
 *
 * Q5_K's extra degree of freedom is the qh row: 32 bytes where byte l holds
 * the fifth bit of group l's low half at bit 2gp and of its high half at bit
 * 2gp+1. Every group re-reads the same 32 bytes and masks a different bit
 * pair, so the high-bit term is two and/andnot pairs per group. */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float q5_k_dot_body(const uint8_t *row, size_t blocks,
                                 const uint8_t *q8)
{
    const __m256i lownib = _mm256_set1_epi8(0x0F);
    float sum = 0.0f;

    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q5_K_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_K_SIZE;
        _mm_prefetch((const char *)(wb + OXK_PREFETCH_AHEAD), _MM_HINT_T0);
        const float dw   = oxk_f16(wb);
        const float dmin = oxk_f16(wb + 2);
        const uint8_t *qh = wb + 16;   /* 32 bytes: 256 high bits */
        const uint8_t *qs = wb + 48;   /* 128 bytes: 256 nibbles */
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + 256;

        uint32_t u0, u1, u2;
        memcpy(&u0, wb + 4, 4);
        memcpy(&u1, wb + 8, 4);
        memcpy(&u2, wb + 12, 4);
        const uint32_t s0 = u0 & OXK_KMASK1;
        const uint32_t s1 = (u2 & OXK_KMASK2) | (((u0 >> 6) & OXK_KMASK3) << 4);
        const uint32_t m0 = u1 & OXK_KMASK1;
        const uint32_t m1 = ((u2 >> 4) & OXK_KMASK2) | (((u1 >> 6) & OXK_KMASK3) << 4);
        const __m256i sm16 = _mm256_cvtepu8_epi16(
            _mm_set_epi32((int)m1, (int)m0, (int)s1, (int)s0));

        /* min_acc = sum_j m_j * (bsum[2j] + bsum[2j+1]), as in Q4_K. */
        const __m128i mhi = _mm256_extracti128_si256(sm16, 1);
        const __m256i mdup = _mm256_set_m128i(_mm_unpackhi_epi16(mhi, mhi),
                                              _mm_unpacklo_epi16(mhi, mhi));
        const int32_t min_acc = hsum_i32_8(_mm256_madd_epi16(
            mdup, _mm256_loadu_si256((const __m256i *)bsums)));

        const __m128i sc128 = _mm256_castsi256_si128(sm16);
        const __m256i scales = _mm256_set_m128i(sc128, sc128);
        const __m256i qhv = _mm256_loadu_si256((const __m256i *)qh);

        __m256i pos_v = _mm256_setzero_si256();
        for (int gp = 0; gp < 4; gp++) {
            const __m256i packed = _mm256_loadu_si256((const __m256i *)(qs + gp * 32));
            const __m256i nib_lo = _mm256_and_si256(packed, lownib);
            const __m256i nib_hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), lownib);
            const __m256i hb_lo = _mm256_and_si256(
                qhv, _mm256_set1_epi8((char)(1 << (2 * gp))));
            const __m256i hb_hi = _mm256_and_si256(
                qhv, _mm256_set1_epi8((char)(1 << (2 * gp + 1))));
            const __m256i h_lo = _mm256_andnot_si256(
                _mm256_cmpeq_epi8(hb_lo, _mm256_setzero_si256()),
                _mm256_set1_epi8(16));
            const __m256i h_hi = _mm256_andnot_si256(
                _mm256_cmpeq_epi8(hb_hi, _mm256_setzero_si256()),
                _mm256_set1_epi8(16));
            const __m256i v_lo = _mm256_or_si256(nib_lo, h_lo);
            const __m256i v_hi = _mm256_or_si256(nib_hi, h_hi);
            const __m256i a_lo = _mm256_loadu_si256((const __m256i *)(q8v + gp * 64));
            const __m256i a_hi = _mm256_loadu_si256((const __m256i *)(q8v + gp * 64 + 32));
            const __m256i sc_lo = _mm256_shuffle_epi8(scales,
                _mm256_set1_epi16((short)(0x0100 * (4 * gp + 1) + 4 * gp)));
            const __m256i sc_hi = _mm256_shuffle_epi8(scales,
                _mm256_set1_epi16((short)(0x0100 * (4 * gp + 3) + 4 * gp + 2)));
            /* 31*128*2 = 7936 per int16 pair; times a 6-bit scale and paired
             * again it is < 2^21, far from int32 overflow. */
            const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(v_lo, a_lo), sc_lo);
            const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(v_hi, a_hi), sc_hi);
            pos_v = _mm256_add_epi32(pos_v, _mm256_add_epi32(p1, p2));
        }
        sum += dw * dq * (float)hsum_i32_8(pos_v) - dmin * dq * (float)min_acc;
    }
    return sum;
}

/* ─── AVX2 Q6_K × Q8_K dot product ─────────────────────────────────────
 *
 * Q6_K block (210 bytes): [128 ql][64 qh][16 int8 scales][f16 d].
 * 256 six-bit values: the low 4 bits come from ql, the high 2 from qh.
 *
 * Bit-exact against oc_oxk_dot_q6_k_q8_k_scalar. That reference was already
 * restructured to accumulate each 16-element scale group in int32 and take
 * one float multiply per block, precisely so a SIMD version could match it
 * exactly rather than approximately — integer reassociation is exact, so the
 * lane-wise sums and the hadd reduction below change nothing. The -32 value
 * offset stays folded out through the activation block sums, as there.
 *
 * The 16-wide chunking is not arbitrary: with base = 128*n + l, every scale
 * group boundary in the reference falls on l = 0 and l = 16, so one 16-byte
 * chunk of l maps to exactly one group per q-slot. That makes each of the
 * four unpacked slots a single 16-element dot, which is one maddubs. */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float q6_k_dot_body(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m2 = _mm256_set1_epi8(0x03);
    /* oxk_g16_sc_shuf[k]: bytes [2k x8, 2k+1 x8] — broadcasts scale group
     * 2k to the first eight int16 lanes of a maddubs result and 2k+1 to
     * the last eight, i.e. to elements 0..15 and 16..31 of one 32-value
     * slot. Shared with the Q2_K/Q3_K kernels (file scope). */

    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q6_K_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_K_SIZE;
        const uint8_t *ql = wb;
        const uint8_t *qh = wb + 128;
        const float dw = oxk_f16(wb + 208);
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + 256;

        const __m128i sc8 = _mm_loadu_si128((const __m128i *)(wb + 192));

        /* minc = sum_g sc[g] * bsum[g]: one madd over the 16 groups. */
        const __m256i minv = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(sc8),
            _mm256_loadu_si256((const __m256i *)bsums));

        __m256i pos_v = _mm256_setzero_si256();
        for (int n = 0; n < 2; n++) {
            const __m256i lo = _mm256_loadu_si256((const __m256i *)(ql + 64 * n));
            const __m256i hi = _mm256_loadu_si256((const __m256i *)(ql + 64 * n + 32));
            const __m256i hb = _mm256_loadu_si256((const __m256i *)(qh + 32 * n));
            /* The same bit selection as the scalar reference: slot k takes
             * its high two bits from qh >> 2k. Masks are applied after the
             * 16-bit shifts so bits bled from the neighbouring byte drop. */
            const __m256i q1 = _mm256_or_si256(_mm256_and_si256(lo, m4),
                _mm256_slli_epi16(_mm256_and_si256(hb, m2), 4));
            const __m256i q2 = _mm256_or_si256(_mm256_and_si256(hi, m4),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 2), m2), 4));
            const __m256i q3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(lo, 4), m4),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 4), m2), 4));
            const __m256i q4 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(hi, 4), m4),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 6), m2), 4));
            const int8_t *av = q8v + 128 * n;
            /* q is 0..63 unsigned, a signed: 63*128*2 = 16128 < 32767 per
             * int16 pair, and times an int8 scale it stays well in int32. */
#define OXK_Q6K_SLOT(Q, K)                                                     \
    pos_v = _mm256_add_epi32(pos_v, _mm256_madd_epi16(                         \
        _mm256_cvtepi8_epi16(_mm_shuffle_epi8(sc8,                             \
            _mm_loadu_si128((const __m128i *)oxk_g16_sc_shuf[4 * n + (K)]))),  \
        _mm256_maddubs_epi16((Q),                                              \
            _mm256_loadu_si256((const __m256i *)(av + 32 * (K))))))
            OXK_Q6K_SLOT(q1, 0);
            OXK_Q6K_SLOT(q2, 1);
            OXK_Q6K_SLOT(q3, 2);
            OXK_Q6K_SLOT(q4, 3);
#undef OXK_Q6K_SLOT
        }
        const int32_t pos  = hsum_i32_8(pos_v);
        const int32_t minc = hsum_i32_8(minv);
        sum += dw * dq * (float)(pos - 32 * minc);
    }
    return sum;
}

/* ─── AVX2 Q2_K / Q3_K × Q8_K dot products ──────────────────────────────
 *
 * Both types pack 256 weights as sixteen groups of sixteen whose 2-bit
 * codes decode as eight contiguous 32-byte passes (q2k_unpack_block /
 * q3k_unpack_block in oxk.c): step k = 4*outer + inner reads
 * qs[32*outer .. +32) shifted right by 2*inner and produces elements
 * 32k..32k+31, i.e. groups 2k and 2k+1. The maddubs pair sums then hold
 * group 2k on i16 lanes 0..7 and group 2k+1 on lanes 8..15, so one madd
 * against an oxk_g16_sc_shuf broadcast of the two scale bytes finishes
 * both groups' scale-weighted sums — the same trick q6_k_dot_body uses.
 *
 * Every term stays in int32 until the per-block float epilogue, which is
 * the scalar reference's expression verbatim: integer reassociation is
 * exact, so these are bit-exact against oc_oxk_dot_q2_k_q8_k_scalar /
 * oc_oxk_dot_q3_k_q8_k_scalar. That matters beyond style — matvec.c
 * routes every Q2_K/Q3_K row through these bodies on AVX2 hosts.
 */

/* Q2_K block (84 bytes): [16 scale|min bytes][64 packed 2-bit][f16 d]
 * [f16 dmin]. Byte g of the scale field is group g's 4-bit scale in the
 * low nibble and its 4-bit minimum in the high nibble; the min folds out
 * through the Q8_K block sums exactly as in the scalar reference. */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float q2_k_dot_body(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    const __m256i two = _mm256_set1_epi8(0x03);
    const __m128i nib = _mm_set1_epi8(0x0F);
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q2_K_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_K_SIZE;
        _mm_prefetch((const char *)(wb + OXK_PREFETCH_AHEAD), _MM_HINT_T0);
        const uint8_t *qs = wb + 16;
        const float dw   = oxk_f16(wb + 80);
        const float dmin = oxk_f16(wb + 82);
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + OC_OXK_QK_K;

        /* sc[g] = scales[g] & 15, mn[g] = scales[g] >> 4: the 16-bit
         * shift smears the neighbour's low nibble into each byte's high
         * nibble, which the 0x0F mask then drops. */
        const __m128i sm  = _mm_loadu_si128((const __m128i *)wb);
        const __m128i sc8 = _mm_and_si128(sm, nib);
        const __m128i mn8 = _mm_and_si128(_mm_srli_epi16(sm, 4), nib);

        /* min_acc = sum_g mn[g] * bsum[g]: widen the sixteen nibble mins
         * to i16 and madd against the sixteen i16 activation sums —
         * exact int32, like the scalar accumulation. */
        const int32_t min_acc = hsum_i32_8(_mm256_madd_epi16(
            _mm256_cvtepu8_epi16(mn8),
            _mm256_loadu_si256((const __m256i *)bsums)));

        __m256i pos_v = _mm256_setzero_si256();
        for (int outer = 0; outer < 2; outer++) {
            const __m256i qbits = _mm256_loadu_si256(
                (const __m256i *)(qs + 32 * outer));
            for (int inner = 0; inner < 4; inner++) {
                const __m256i c = _mm256_and_si256(
                    _mm256_srli_epi16(qbits, 2 * inner), two);
                const __m256i a = _mm256_loadu_si256(
                    (const __m256i *)(q8v + 32 * (outer * 4 + inner)));
                const __m256i scv = _mm256_cvtepu8_epi16(
                    _mm_shuffle_epi8(sc8, _mm_loadu_si128(
                        (const __m128i *)
                        oxk_g16_sc_shuf[outer * 4 + inner])));
                /* 2-bit unsigned code x i8 activation: pair sums are
                 * <= 2*3*128 < 2^10, and times a 4-bit scale the madd
                 * lanes stay under 2^15 — int32 is far away. */
                pos_v = _mm256_add_epi32(pos_v, _mm256_madd_epi16(
                    _mm256_maddubs_epi16(c, a), scv));
            }
        }
        const int32_t pos = hsum_i32_8(pos_v);
        sum += dq * (dw * (float)pos - dmin * (float)min_acc);
    }
    return sum;
}

/* Q3_K block (110 bytes): [32 hmask bytes][64 packed 2-bit][12 packed
 * 6-bit scales][f16 d]. The third bit of element 32k+l is bit k of
 * hmask[l]; the scalar packs it into the code (raw = q | bit<<2, a
 * uniform -4 offset folded through the block sums) and so does this —
 * the only difference from the Q2_K inner loop is the hmask or and the
 * signed scale decode. */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float q3_k_dot_body(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    const __m256i two = _mm256_set1_epi8(0x03);
    const __m256i one = _mm256_set1_epi8(0x01);
    /* Constants for the packed 6-bit scale decode (see below). */
    const __m128i lo_nib  = _mm_set1_epi8(0x0F);
    const __m128i hi_bits = _mm_set1_epi8(0x03);
    const __m128i tshifts = _mm_setr_epi32(0, 2, 4, 6);
    const __m128i hi_sel  = _mm_set_epi32(-1, -1, 0, 0);
    const __m128i bias32  = _mm_set1_epi8(32);
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q3_K_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_K_SIZE;
        _mm_prefetch((const char *)(wb + OXK_PREFETCH_AHEAD), _MM_HINT_T0);
        const uint8_t *qs = wb + 32;
        const float dw = oxk_f16(wb + 108);
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + OC_OXK_QK_K;

        /* Sixteen signed 6-bit scales packed in twelve bytes — the same
         * aux[0..3] shuffle q3k_unpack_block runs, on dword lanes: lane i
         * of sc8 holds sc[4i..4i+3]. aux[0..1] are the low qword of the
         * scale field, aux[2] the four bytes at wb+104. */
        const __m128i a01 = _mm_loadl_epi64((const __m128i *)(wb + 96));
        uint32_t tmp;
        memcpy(&tmp, wb + 104, 4);
        const __m128i A = _mm_unpacklo_epi64(a01, a01);
        const __m128i T = _mm_shuffle_epi32(_mm_cvtsi32_si128((int)tmp),
                                            _MM_SHUFFLE(0, 0, 0, 0));
        /* Lanes 0,1 take (aux & 0x0F..), lanes 2,3 take ((aux>>4) &
         * 0x0F..): low nibbles of each byte. */
        const __m128i lo = _mm_blendv_epi8(_mm_and_si128(A, lo_nib),
            _mm_and_si128(_mm_srli_epi32(A, 4), lo_nib), hi_sel);
        /* High two bits of each scale come from aux[2] >> (0,2,4,6),
         * nibble-masked and shifted into bits 4..5. */
        const __m128i hi = _mm_slli_epi32(
            _mm_and_si128(_mm_srlv_epi32(T, tshifts), hi_bits), 4);
        /* Scale bytes minus the 32 bias, interpreted signed — the
         * scalar's (int8)(byte - 32). */
        const __m128i sc8 = _mm_sub_epi8(_mm_or_si128(lo, hi), bias32);

        /* off = sum_g sc[g] * bsum[g]: folds the uniform -4 code offset
         * out through the activation sums, with the SIGNED scales. */
        const int32_t off = hsum_i32_8(_mm256_madd_epi16(
            _mm256_cvtepi8_epi16(sc8),
            _mm256_loadu_si256((const __m256i *)bsums)));

        const __m256i hb = _mm256_loadu_si256((const __m256i *)wb);
        __m256i pos_v = _mm256_setzero_si256();
        for (int outer = 0; outer < 2; outer++) {
            const __m256i qbits = _mm256_loadu_si256(
                (const __m256i *)(qs + 32 * outer));
            for (int inner = 0; inner < 4; inner++) {
                const int k = outer * 4 + inner;
                /* Byte l of step k's code vector takes its third bit from
                 * bit k of hmask[l]: shift it to bit 0, isolate, shift
                 * into place — bits bled from the neighbouring byte are
                 * dropped by the 0x01 mask before the left shift. */
                const __m256i c = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(qbits, 2 * inner),
                                     two),
                    _mm256_slli_epi16(_mm256_and_si256(
                        _mm256_srli_epi16(hb, k), one), 2));
                const __m256i a = _mm256_loadu_si256(
                    (const __m256i *)(q8v + 32 * k));
                const __m256i scv = _mm256_cvtepi8_epi16(
                    _mm_shuffle_epi8(sc8, _mm_loadu_si128(
                        (const __m128i *)oxk_g16_sc_shuf[k])));
                /* 3-bit unsigned code x i8: pair sums <= 2*7*128 < 2^11;
                 * times a signed 6-bit scale the madd lanes stay under
                 * 2^17. */
                pos_v = _mm256_add_epi32(pos_v, _mm256_madd_epi16(
                    _mm256_maddubs_epi16(c, a), scv));
            }
        }
        sum += dw * dq * (float)(hsum_i32_8(pos_v) - 4 * off);
    }
    return sum;
}

/* ─── AVX2 prepared-row dots ─────────────────────────────────────────────
 *
 * Same accumulation against the decoded-row layouts oc_oxk_q2_k_prep_row
 * and oc_oxk_q3_k_prep_row write: the codes arrive as bytes, so the inner
 * loop is eight maddubs+madd passes per block plus the block-sum fold.
 * Q3_K consumes the Q6_K prepared layout (unsigned codes + sixteen signed
 * group scales + a fixed code offset), so one shared body carries both —
 * and Q6_K itself — with the offset a parameter the compiler folds. All
 * are bit-exact with their scalar prepared dots: identical int32 terms,
 * identical per-block float expression, activation-at-a-time order. */

/* Q2_K prepared row: d[blocks], dmin[blocks], codes[blocks*256] (u8),
 * sc[blocks*16] (u8), mn[blocks*16] (u8). */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float q2_k_prepped_body(const void *prep, size_t blocks,
                                      const uint8_t *act)
{
    const float   *d     = (const float *)prep;
    const float   *dmin  = d + blocks;
    const uint8_t *codes = (const uint8_t *)(d + 2 * blocks);
    const uint8_t *sc    = codes + blocks * OC_OXK_QK_K;
    const uint8_t *mn    = sc + blocks * 16;

    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *cb = codes + b * OC_OXK_QK_K;
        const uint8_t *qb = act + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t *q8v = (const int8_t *)(qb + 4);

        const __m128i sc8 = _mm_loadu_si128(
            (const __m128i *)(sc + b * 16));
        const int32_t min_acc = hsum_i32_8(_mm256_madd_epi16(
            _mm256_cvtepu8_epi16(
                _mm_loadu_si128((const __m128i *)(mn + b * 16))),
            _mm256_loadu_si256((const __m256i *)(qb + 4 + OC_OXK_QK_K))));

        __m256i pos_v = _mm256_setzero_si256();
        for (int k = 0; k < 8; k++) {
            const __m256i c = _mm256_loadu_si256(
                (const __m256i *)(cb + 32 * k));
            const __m256i a = _mm256_loadu_si256(
                (const __m256i *)(q8v + 32 * k));
            const __m256i scv = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(sc8,
                _mm_loadu_si128((const __m128i *)oxk_g16_sc_shuf[k])));
            pos_v = _mm256_add_epi32(pos_v, _mm256_madd_epi16(
                _mm256_maddubs_epi16(c, a), scv));
        }
        const int32_t pos = hsum_i32_8(pos_v);
        sum += dq * (d[b] * (float)pos - dmin[b] * (float)min_acc);
    }
    return sum;
}

/* Q6_K-shaped prepared row (also what oc_oxk_q3_k_prep_row writes):
 * d[blocks], codes[blocks*256] (u8), sc[blocks*16] (i8). `offset` is the
 * code bias folded through the block sums — 32 for Q6_K, 4 for Q3_K. */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float q3q6_prepped_body(const void *prep, size_t blocks,
                                      const uint8_t *act, int32_t offset)
{
    const float   *d     = (const float *)prep;
    const uint8_t *codes = (const uint8_t *)(d + blocks);
    const int8_t  *sc    = (const int8_t *)(codes + blocks * OC_OXK_QK_K);

    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *cb  = codes + b * OC_OXK_QK_K;
        const int8_t  *scb = sc + b * 16;
        const uint8_t *qb  = act + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t *q8v = (const int8_t *)(qb + 4);

        const __m128i sc8 = _mm_loadu_si128((const __m128i *)scb);
        const int32_t off = hsum_i32_8(_mm256_madd_epi16(
            _mm256_cvtepi8_epi16(sc8),
            _mm256_loadu_si256((const __m256i *)(qb + 4 + OC_OXK_QK_K))));

        __m256i pos_v = _mm256_setzero_si256();
        for (int k = 0; k < 8; k++) {
            const __m256i c = _mm256_loadu_si256(
                (const __m256i *)(cb + 32 * k));
            const __m256i a = _mm256_loadu_si256(
                (const __m256i *)(q8v + 32 * k));
            const __m256i scv = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(sc8,
                _mm_loadu_si128((const __m128i *)oxk_g16_sc_shuf[k])));
            pos_v = _mm256_add_epi32(pos_v, _mm256_madd_epi16(
                _mm256_maddubs_epi16(c, a), scv));
        }
        const int32_t pos = hsum_i32_8(pos_v);
        sum += d[b] * dq * (float)(pos - offset * off);
    }
    return sum;
}

/* One prepared Q2_K row against n_act activations, four at a time so the
 * codes and scale broadcasts load once per four dots — the same
 * scheduling the AVX-512 prepped kernels use. Bit-exact per activation
 * with the single-activation body: every block's int32 terms are
 * identical, and each activation keeps its own scalar float chain. */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline void q2_k_prepped_multi_body(const void *prep, size_t blocks,
                                           const uint8_t *acts,
                                           size_t act_stride, size_t n_act,
                                           float *out)
{
    const float   *d     = (const float *)prep;
    const float   *dmin  = d + blocks;
    const uint8_t *codes = (const uint8_t *)(d + 2 * blocks);
    const uint8_t *sc    = codes + blocks * OC_OXK_QK_K;
    const uint8_t *mn    = sc + blocks * 16;

    size_t a0 = 0;
    for (; a0 + 4 <= n_act; a0 += 4) {
        const uint8_t *act[4] = {
            acts + (a0 + 0) * act_stride, acts + (a0 + 1) * act_stride,
            acts + (a0 + 2) * act_stride, acts + (a0 + 3) * act_stride,
        };
        float sums[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *cb  = codes + b * OC_OXK_QK_K;
            const uint8_t *scb = sc + b * 16;
            const uint8_t *mnb = mn + b * 16;
            const __m128i sc8  = _mm_loadu_si128((const __m128i *)scb);
            const __m256i mn16 = _mm256_cvtepu8_epi16(
                _mm_loadu_si128((const __m128i *)mnb));
            __m256i acc[4] = { _mm256_setzero_si256(),
                               _mm256_setzero_si256(),
                               _mm256_setzero_si256(),
                               _mm256_setzero_si256() };
            for (int k = 0; k < 8; k++) {
                const __m256i c = _mm256_loadu_si256(
                    (const __m256i *)(cb + 32 * k));
                const __m256i scv = _mm256_cvtepu8_epi16(
                    _mm_shuffle_epi8(sc8, _mm_loadu_si128(
                        (const __m128i *)oxk_g16_sc_shuf[k])));
                const size_t qoff = b * OC_OXK_BLOCK_Q8_K_SIZE + 4u +
                                    32u * (size_t)k;
                for (int a = 0; a < 4; a++) {
                    const __m256i q8v = _mm256_loadu_si256(
                        (const __m256i *)(act[a] + qoff));
                    acc[a] = _mm256_add_epi32(acc[a], _mm256_madd_epi16(
                        _mm256_maddubs_epi16(c, q8v), scv));
                }
            }
            for (int a = 0; a < 4; a++) {
                const uint8_t *qb = act[a] + b * OC_OXK_BLOCK_Q8_K_SIZE;
                const int32_t min_acc = hsum_i32_8(_mm256_madd_epi16(mn16,
                    _mm256_loadu_si256(
                        (const __m256i *)(qb + 4 + OC_OXK_QK_K))));
                const int32_t pos = hsum_i32_8(acc[a]);
                float dq;
                memcpy(&dq, qb, 4);
                sums[a] += dq * (d[b] * (float)pos -
                                 dmin[b] * (float)min_acc);
            }
        }
        for (int a = 0; a < 4; a++) out[a0 + (size_t)a] = sums[a];
    }
    for (; a0 < n_act; a0++)
        out[a0] = q2_k_prepped_body(prep, blocks, acts + a0 * act_stride);
}

/* Shared multi-activation body for the Q6_K-shaped prepared layouts
 * (Q3_K offset 4, Q6_K offset 32). Same structure as the Q2_K multi:
 * four activations per pass over the row, each keeping its own int32
 * accumulator and scalar float chain — bit-exact per activation against
 * the single-activation body. */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline void q3q6_prepped_multi_body(const void *prep, size_t blocks,
                                           const uint8_t *acts,
                                           size_t act_stride, size_t n_act,
                                           float *out, int32_t offset)
{
    const float   *d     = (const float *)prep;
    const uint8_t *codes = (const uint8_t *)(d + blocks);
    const int8_t  *sc    = (const int8_t *)(codes + blocks * OC_OXK_QK_K);

    size_t a0 = 0;
    for (; a0 + 4 <= n_act; a0 += 4) {
        const uint8_t *act[4] = {
            acts + (a0 + 0) * act_stride, acts + (a0 + 1) * act_stride,
            acts + (a0 + 2) * act_stride, acts + (a0 + 3) * act_stride,
        };
        float sums[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *cb  = codes + b * OC_OXK_QK_K;
            const int8_t  *scb = sc + b * 16;
            const __m128i sc8  = _mm_loadu_si128((const __m128i *)scb);
            const __m256i sc16 = _mm256_cvtepi8_epi16(sc8);
            __m256i acc[4] = { _mm256_setzero_si256(),
                               _mm256_setzero_si256(),
                               _mm256_setzero_si256(),
                               _mm256_setzero_si256() };
            for (int k = 0; k < 8; k++) {
                const __m256i c = _mm256_loadu_si256(
                    (const __m256i *)(cb + 32 * k));
                const __m256i scv = _mm256_cvtepi8_epi16(
                    _mm_shuffle_epi8(sc8, _mm_loadu_si128(
                        (const __m128i *)oxk_g16_sc_shuf[k])));
                const size_t qoff = b * OC_OXK_BLOCK_Q8_K_SIZE + 4u +
                                    32u * (size_t)k;
                for (int a = 0; a < 4; a++) {
                    const __m256i q8v = _mm256_loadu_si256(
                        (const __m256i *)(act[a] + qoff));
                    acc[a] = _mm256_add_epi32(acc[a], _mm256_madd_epi16(
                        _mm256_maddubs_epi16(c, q8v), scv));
                }
            }
            for (int a = 0; a < 4; a++) {
                const uint8_t *qb = act[a] + b * OC_OXK_BLOCK_Q8_K_SIZE;
                const int32_t off = hsum_i32_8(_mm256_madd_epi16(sc16,
                    _mm256_loadu_si256(
                        (const __m256i *)(qb + 4 + OC_OXK_QK_K))));
                const int32_t pos = hsum_i32_8(acc[a]);
                float dq;
                memcpy(&dq, qb, 4);
                sums[a] += d[b] * dq * (float)(pos - offset * off);
            }
        }
        for (int a = 0; a < 4; a++) out[a0 + (size_t)a] = sums[a];
    }
    for (; a0 < n_act; a0++)
        out[a0] = q3q6_prepped_body(prep, blocks, acts + a0 * act_stride,
                                    offset);
}

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q2_k_prepped_avx2(const void *prep, size_t blocks,
                                   const uint8_t *act)
{
    return q2_k_prepped_body(prep, blocks, act);
}

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q3_k_prepped_avx2(const void *prep, size_t blocks,
                                   const uint8_t *act)
{
    return q3q6_prepped_body(prep, blocks, act, 4);
}

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q6_k_prepped_avx2(const void *prep, size_t blocks,
                                   const uint8_t *act)
{
    return q3q6_prepped_body(prep, blocks, act, 32);
}

__attribute__((target("avx2,fma,f16c")))
void oc_oxk_dot_q2_k_prepped_multi_avx2(const void *prep, size_t blocks,
                                        const uint8_t *acts,
                                        size_t act_stride, size_t n_act,
                                        float *out)
{
    q2_k_prepped_multi_body(prep, blocks, acts, act_stride, n_act, out);
}

__attribute__((target("avx2,fma,f16c")))
void oc_oxk_dot_q3_k_prepped_multi_avx2(const void *prep, size_t blocks,
                                        const uint8_t *acts,
                                        size_t act_stride, size_t n_act,
                                        float *out)
{
    q3q6_prepped_multi_body(prep, blocks, acts, act_stride, n_act, out, 4);
}

__attribute__((target("avx2,fma,f16c")))
void oc_oxk_dot_q6_k_prepped_multi_avx2(const void *prep, size_t blocks,
                                        const uint8_t *acts,
                                        size_t act_stride, size_t n_act,
                                        float *out)
{
    q3q6_prepped_multi_body(prep, blocks, acts, act_stride, n_act, out, 32);
}

/* ─── AVX2 matvec (forward to scalar) ──────────────────────────────────── */

__attribute__((target("avx2,fma,f16c")))
void oc_oxk_matvec_q4_0_f32_avx2(const uint8_t *w, size_t n_rows,
                                 size_t row_bytes, const float *x, float *out)
{
    oc_oxk_matvec_q4_0_f32_scalar(w, n_rows, row_bytes, x, out);
}

__attribute__((target("avx2,fma,f16c")))
void oc_oxk_matvec_q4_k_f32_avx2(const uint8_t *w, size_t n_rows,
                                 size_t row_bytes, const float *x, float *out)
{
    oc_oxk_matvec_q4_k_f32_scalar(w, n_rows, row_bytes, x, out);
}

__attribute__((target("avx2,fma,f16c")))
void oc_oxk_matvec_q8_0_f32_avx2(const uint8_t *w, size_t n_rows,
                                 size_t row_bytes, const float *x, float *out)
{
    oc_oxk_matvec_q8_0_f32_scalar(w, n_rows, row_bytes, x, out);
}

/* ─── AVX-512 stubs (forward to scalar) ────────────────────────────────── */

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q4_0_q8_0_avx512(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    return oc_oxk_dot_q4_0_q8_0_scalar(row, blocks, q8);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q4_1_q8_0_avx512(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    return oc_oxk_dot_q4_1_q8_0_scalar(row, blocks, q8);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q8_0_q8_0_avx512(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    return oc_oxk_dot_q8_0_q8_0_scalar(row, blocks, q8);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
/* AVX-512 hosts run the AVX2 kernel.
 *
 * A VNNI version (dpbusd, which folds the maddubs+madd pair into a single
 * instruction) was written and measured: bit-exact, and about 6% faster on a
 * Cascade Lake Xeon. It is not kept because clang rejects the intrinsic even
 * with a matching target attribute -- it is stricter than GCC about inlining
 * intrinsics whose own target string differs from the caller's -- and a 6%
 * gain on one compiler is not worth failing the clang build. Q4_K is no
 * longer the bottleneck regardless; what remains of the decode gap is in the
 * still-scalar attention, norm and RoPE code, not here.
 *
 * The point of this entry point is that it no longer forwards to *scalar*,
 * which is what it did before and which made an AVX-512 host run the slowest
 * kernel available. */
__attribute__((target("avx512bw,avx512dq")))
float oc_oxk_dot_q4_k_q8_k_avx512(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    return oc_oxk_dot_q4_k_q8_k_avx2(row, blocks, q8);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q5_k_q8_k_avx512(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    return oc_oxk_dot_q5_k_q8_k_scalar(row, blocks, q8);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
float oc_oxk_dot_q6_k_q8_k_avx512(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    return oc_oxk_dot_q6_k_q8_k_scalar(row, blocks, q8);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
void oc_oxk_matvec_q4_0_f32_avx512(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out)
{
    oc_oxk_matvec_q4_0_f32_scalar(w, n_rows, row_bytes, x, out);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
void oc_oxk_matvec_q4_k_f32_avx512(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out)
{
    oc_oxk_matvec_q4_k_f32_scalar(w, n_rows, row_bytes, x, out);
}

__attribute__((target("avx512bw,avx512dq,avx512vnni")))
void oc_oxk_matvec_q8_0_f32_avx512(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out)
{
    oc_oxk_matvec_q8_0_f32_scalar(w, n_rows, row_bytes, x, out);
}

/* ─── AVX2 IQ3_S × Q8_K ──────────────────────────────────────────────────
 *
 * Port of ggml's ggml_vec_dot_iq3_s_q8_K AVX2 path. Per 64 weights (one
 * ib32 pair): 16 grid entries looked up from 9-bit indices (qs byte plus one
 * qh bit), sign bits expanded to byte masks with a shuffle/compare, the signs
 * folded into the activation, then _mm256_maddubs_epi16 (grid values are
 * unsigned 1..15) and _mm256_madd_epi16 against the odd sub-block scale.
 *
 * Unlike ggml, the block's int32 lanes are reduced to one scalar before the
 * float multiply-add, so the float accumulation order is exactly the scalar
 * reference's and the result is bit-identical (test_oxk_iq3_s.c). The extra
 * horizontal add is a handful of cycles per 256 weights.
 *
 * Throughput is bound by the 64 table lookups per block rather than by
 * DRAM: ~28 ns per 110-byte block per core on Zen 3+, so a matvec tops out
 * near 4 GB/s per core (~27 GB/s on 8 cores, about half of what the memory
 * system can stream). Gathers, 64-bit pair packing and whole-block index
 * precompute were all measured and none beat plain scalar lookups. */

/* Eight grid entries (32 unsigned weights) of one 32-wide sub-block. Each
 * 9-bit index is qs[k] plus bit k of qh. Measured on Zen 3+ (Ryzen 6850H),
 * building the vector straight from scalar table loads beats both
 * _mm256_i32gather_epi32 (~9% slower) and ggml's store-indices-then-reload
 * form (~2% slower); the lookups, not the arithmetic, bound this kernel. */
__attribute__((target("avx2")))
static inline __m256i iq3_s_grid8(const uint8_t *qs, uint32_t qh)
{
    return _mm256_set_epi32(
        (int)IQ3S_GRID[qs[7] | ((qh << 1) & 256u)],
        (int)IQ3S_GRID[qs[6] | ((qh << 2) & 256u)],
        (int)IQ3S_GRID[qs[5] | ((qh << 3) & 256u)],
        (int)IQ3S_GRID[qs[4] | ((qh << 4) & 256u)],
        (int)IQ3S_GRID[qs[3] | ((qh << 5) & 256u)],
        (int)IQ3S_GRID[qs[2] | ((qh << 6) & 256u)],
        (int)IQ3S_GRID[qs[1] | ((qh << 7) & 256u)],
        (int)IQ3S_GRID[qs[0] | ((qh << 8) & 256u)]);
}

/* Block scale without a call: the out-of-line bit-twiddle converter forced
 * a vzeroupper and a reload of every vector constant once per block, which
 * cost ~15% of single-thread matvec throughput (3.3 -> 3.9 GB/s on a
 * Ryzen 6850H). F16C is exact for every f16 input. */
__attribute__((target("avx2,f16c")))
static inline float iq3_s_f16(const uint8_t *p)
{
    uint16_t h;
    memcpy(&h, p, 2);
    return _cvtsh_ss(h);
}

/* The IQ3_S lane reduction (see iq3_s_lanes_reduce in oxk.c):
 * ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)). */
__attribute__((target("avx2,fma,f16c")))
static inline float iq3_s_hsum_f32_8(__m256 v)
{
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/* acc + d * (float)lanes, as a separate multiply and add (bit-identical to
 * the scalar reference, which is compiled with -ffp-contract=off). */
__attribute__((target("avx2,fma,f16c")))
static inline __m256 iq3_s_lane_acc(__m256 acc, float d, __m256i lanes)
{
    return _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d),
                                            _mm256_cvtepi32_ps(lanes)));
}

/* Grid bytes (unsigned) and sign byte-masks (0xFF = negative) for the 64
 * weights of sub-blocks ib32 and ib32+1. */
__attribute__((target("avx2")))
static inline void iq3_s_load_pair(const uint8_t *qs, const uint8_t *qh,
                                   const uint8_t *signs,
                                   __m256i *g1, __m256i *g2,
                                   __m256i *s1, __m256i *s2)
{
    const __m256i mask1 = _mm256_set_epi64x(0x0303030303030303LL,
                                            0x0202020202020202LL,
                                            0x0101010101010101LL, 0);
    const __m256i mask2 = _mm256_set1_epi64x((long long)0x8040201008040201ULL);

    *g1 = iq3_s_grid8(qs, qh[0]);
    *g2 = iq3_s_grid8(qs + 8, qh[1]);

    uint32_t sg0, sg1;
    memcpy(&sg0, signs, 4);
    memcpy(&sg1, signs + 4, 4);
    __m256i a = _mm256_and_si256(
        _mm256_shuffle_epi8(_mm256_set1_epi32((int)sg0), mask1), mask2);
    *s1 = _mm256_cmpeq_epi8(a, mask2);
    a = _mm256_and_si256(
        _mm256_shuffle_epi8(_mm256_set1_epi32((int)sg1), mask1), mask2);
    *s2 = _mm256_cmpeq_epi8(a, mask2);
}

__attribute__((target("avx2,fma,f16c"), always_inline))
static inline float iq3_s_dot_body(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    __m256 accf = _mm256_setzero_ps();
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *xb = row + b * OC_OXK_BLOCK_IQ3_S_SIZE;
        const uint8_t *yb = q8  + b * OC_OXK_BLOCK_Q8_K_SIZE;
        _mm_prefetch((const char *)(xb + OXK_PREFETCH_AHEAD), _MM_HINT_T0);
        float yd;
        memcpy(&yd, yb, 4);
        const float d = iq3_s_f16(xb) * yd;
        const uint8_t *qs = xb + 2;
        const uint8_t *qh = xb + 66;
        const uint8_t *sg = xb + 74;
        const uint8_t *sc = xb + 106;
        const int8_t  *qa = (const int8_t *)(yb + 4);
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)qa);
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)(qa + 32));
            qa += 64;
            __m256i g1, g2, s1, s2;
            iq3_s_load_pair(qs, qh + ib32, sg, &g1, &g2, &s1, &s2);
            qs += 16;
            sg += 8;
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s1, q8_1), s1);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2, q8_2), s2);
            const __m256i dot1 = _mm256_maddubs_epi16(g1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(g2, q8s_2);
            const int16_t ls1 = (int16_t)(2 * (sc[ib32 / 2] & 0x0F) + 1);
            const int16_t ls2 = (int16_t)(2 * (sc[ib32 / 2] >> 4) + 1);
            sumi1 = _mm256_add_epi32(sumi1,
                        _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1)));
            sumi2 = _mm256_add_epi32(sumi2,
                        _mm256_madd_epi16(dot2, _mm256_set1_epi16(ls2)));
        }
        accf = iq3_s_lane_acc(accf, d, _mm256_add_epi32(sumi1, sumi2));
    }
    return iq3_s_hsum_f32_8(accf);
}


/* One IQ3_S row against NA (2..4) activations: the grid gathers and sign
 * expansion are done once per sub-block pair and shared, the per-activation
 * work is the sign-apply + maddubs. Per activation the integer sums, the
 * scale and the lane accumulation are exactly iq3_s_dot_body's, so every
 * output is bit-identical to the single-activation kernel. */
__attribute__((target("avx2,fma,f16c"), always_inline))
static inline void iq3_s_dot_body_multi(const uint8_t *row, size_t blocks,
                                        const uint8_t *acts, size_t act_stride,
                                        const int NA, float *out)
{
    __m256 accf[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(),
                       _mm256_setzero_ps(), _mm256_setzero_ps() };
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *xb = row + b * OC_OXK_BLOCK_IQ3_S_SIZE;
        _mm_prefetch((const char *)(xb + OXK_PREFETCH_AHEAD), _MM_HINT_T0);
        const float dw = iq3_s_f16(xb);
        const uint8_t *qs = xb + 2;
        const uint8_t *qh = xb + 66;
        const uint8_t *sg = xb + 74;
        const uint8_t *sc = xb + 106;
        __m256i sumi1[4], sumi2[4];
        for (int a = 0; a < NA; a++) {
            sumi1[a] = _mm256_setzero_si256();
            sumi2[a] = _mm256_setzero_si256();
        }
        const size_t yo = b * OC_OXK_BLOCK_Q8_K_SIZE;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            __m256i g1, g2, s1, s2;
            iq3_s_load_pair(qs, qh + ib32, sg, &g1, &g2, &s1, &s2);
            qs += 16;
            sg += 8;
            const __m256i vls1 = _mm256_set1_epi16(
                (int16_t)(2 * (sc[ib32 / 2] & 0x0F) + 1));
            const __m256i vls2 = _mm256_set1_epi16(
                (int16_t)(2 * (sc[ib32 / 2] >> 4) + 1));
            for (int a = 0; a < NA; a++) {
                const int8_t *qa = (const int8_t *)(acts + a * act_stride + yo + 4)
                                 + 32 * ib32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)qa);
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)(qa + 32));
                const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s1, q8_1), s1);
                const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2, q8_2), s2);
                sumi1[a] = _mm256_add_epi32(sumi1[a], _mm256_madd_epi16(
                               _mm256_maddubs_epi16(g1, q8s_1), vls1));
                sumi2[a] = _mm256_add_epi32(sumi2[a], _mm256_madd_epi16(
                               _mm256_maddubs_epi16(g2, q8s_2), vls2));
            }
        }
        for (int a = 0; a < NA; a++) {
            float yd;
            memcpy(&yd, acts + a * act_stride + yo, 4);
            accf[a] = iq3_s_lane_acc(accf[a], dw * yd,
                                     _mm256_add_epi32(sumi1[a], sumi2[a]));
        }
    }
    for (int a = 0; a < NA; a++) out[a] = iq3_s_hsum_f32_8(accf[a]);
}

/* out[a * out_stride + r]. The 2/3/4 paths use the multi-row kernel; every
 * other n_act stores straight into `out` (res[] is only 4 floats). */
__attribute__((target("avx2,fma,f16c")))
void oc_oxk_dot_rows_iq3_s_q8_k_multi_avx2(const uint8_t *rows, size_t row_bytes,
                                           size_t n_rows, size_t blocks,
                                           const uint8_t *acts,
                                           size_t act_stride, size_t n_act,
                                           float *out, size_t out_stride)
{
    float res[4];
    for (size_t r = 0; r < n_rows; r++) {
        const uint8_t *row = rows + r * row_bytes;
        switch (n_act) {
        case 2: iq3_s_dot_body_multi(row, blocks, acts, act_stride, 2, res); break;
        case 3: iq3_s_dot_body_multi(row, blocks, acts, act_stride, 3, res); break;
        case 4: iq3_s_dot_body_multi(row, blocks, acts, act_stride, 4, res); break;
        default:
            for (size_t a = 0; a < n_act; a++)
                out[a * out_stride + r] = iq3_s_dot_body(
                    row, blocks, acts + a * act_stride);
            continue;
        }
        for (size_t a = 0; a < n_act; a++) out[a * out_stride + r] = res[a];
    }
}

/* ─── Exported single-row and rows forms (shared inlined bodies) ───────── */

#define OXK_ROWS_WRAPPERS(NAME, BODY)                                          \
__attribute__((target("avx2,fma,f16c")))                                       \
float oc_oxk_dot_##NAME##_q8_k_avx2(const uint8_t *row, size_t blocks,         \
                                    const uint8_t *q8)                         \
{                                                                              \
    return BODY(row, blocks, q8);                                              \
}                                                                              \
__attribute__((target("avx2,fma,f16c")))                                       \
void oc_oxk_dot_rows_##NAME##_q8_k_avx2(const uint8_t *rows, size_t row_bytes, \
                                        size_t n_rows, size_t blocks,          \
                                        const uint8_t *q8, float *out)         \
{                                                                              \
    for (size_t r = 0; r < n_rows; r++)                                        \
        out[r] = BODY(rows + r * row_bytes, blocks, q8);                       \
}

OXK_ROWS_WRAPPERS(q4_k, q4_k_dot_body)
OXK_ROWS_WRAPPERS(q5_k, q5_k_dot_body)
OXK_ROWS_WRAPPERS(q6_k, q6_k_dot_body)
OXK_ROWS_WRAPPERS(q2_k, q2_k_dot_body)
OXK_ROWS_WRAPPERS(q3_k, q3_k_dot_body)
OXK_ROWS_WRAPPERS(iq3_s, iq3_s_dot_body)

#undef OXK_ROWS_WRAPPERS

/* The Q8_0-activation family (classic quants plus the AL aliases) previously
 * had no rows form: matvec.c dispatched one indirect row-dot call per row.
 * These share the single-row bodies, so per-row results are bit-identical,
 * while the caller gets one call per slice segment, inlined block loops and
 * the software prefetch inside the loop. */
#define OXK_ROWS_WRAPPERS_Q80(NAME, BODY)                                      \
__attribute__((target("avx2,fma,f16c")))                                        \
void oc_oxk_dot_rows_##NAME##_q8_0_avx2(const uint8_t *rows,                    \
                                        size_t row_bytes, size_t n_rows,        \
                                        size_t blocks, const uint8_t *q8,       \
                                        float *out)                            \
{                                                                              \
    for (size_t r = 0; r < n_rows; r++)                                         \
        out[r] = BODY(rows + r * row_bytes, blocks, q8);                       \
}

OXK_ROWS_WRAPPERS_Q80(q4_0, q4_0_dot_body)
OXK_ROWS_WRAPPERS_Q80(q4_1, q4_1_dot_body)
OXK_ROWS_WRAPPERS_Q80(q5_0, q5_0_dot_body)
OXK_ROWS_WRAPPERS_Q80(q5_1, q5_1_dot_body)
OXK_ROWS_WRAPPERS_Q80(q8_0, q8_0_dot_body)
OXK_ROWS_WRAPPERS_Q80(al5_xs, al5_xs_dot_body)

#undef OXK_ROWS_WRAPPERS_Q80

__attribute__((target("avx2,fma,f16c")))
void oc_oxk_iq3_s_prep_row_avx2(const uint8_t *row, size_t blocks,
                                void *scratch)
{
    uint8_t *out = (uint8_t *)scratch;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *xb = row + b * OC_OXK_BLOCK_IQ3_S_SIZE;
        uint8_t *pb = out + b * OC_OXK_IQ3_S_PREP_BLOCK;
        const float d = iq3_s_f16(xb);
        memset(pb, 0, 16);
        memcpy(pb, &d, 4);
        int16_t ls[8];
        for (int k = 0; k < 4; k++) {
            ls[2 * k]     = (int16_t)(2 * (xb[106 + k] & 0x0F) + 1);
            ls[2 * k + 1] = (int16_t)(2 * (xb[106 + k] >> 4) + 1);
        }
        memcpy(pb + 16, ls, 16);
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            __m256i g1, g2, s1, s2;
            iq3_s_load_pair(xb + 2 + 8 * ib32, xb + 66 + ib32,
                            xb + 74 + 4 * ib32, &g1, &g2, &s1, &s2);
            _mm256_storeu_si256((__m256i *)(pb + 32 + 32 * ib32),
                                _mm256_sub_epi8(_mm256_xor_si256(s1, g1), s1));
            _mm256_storeu_si256((__m256i *)(pb + 64 + 32 * ib32),
                                _mm256_sub_epi8(_mm256_xor_si256(s2, g2), s2));
        }
    }
}

/* One prepared row against n_act activations, four at a time so each weight
 * vector is loaded once per four dots. w*q is computed as
 * maddubs(|q|, sign(w, q)): the sign flips w (|w| <= 15, never -128, so
 * psignb cannot wrap) and |q| sits in the unsigned slot (0..128). int16 pair
 * sums stay under 2*128*15, so no saturation. sign(w, q) is used rather
 * than sign(q, w) because a -128 activation negated by psignb wraps back to
 * -128 and would flip the product's sign. */
__attribute__((target("avx2,fma,f16c")))
void oc_oxk_dot_iq3_s_prepped_multi_avx2(const void *scratch, size_t blocks,
                                         const uint8_t *acts,
                                         size_t act_stride, size_t n_act,
                                         float *out)
{
    const uint8_t *prep = (const uint8_t *)scratch;
    size_t a = 0;
    for (; a + 4 <= n_act; a += 4) {
        const uint8_t *act0 = acts + (a + 0) * act_stride;
        const uint8_t *act1 = acts + (a + 1) * act_stride;
        const uint8_t *act2 = acts + (a + 2) * act_stride;
        const uint8_t *act3 = acts + (a + 3) * act_stride;
        __m256 f0 = _mm256_setzero_ps(), f1 = f0, f2 = f0, f3 = f0;
        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *pb = prep + b * OC_OXK_IQ3_S_PREP_BLOCK;
            const size_t yo = b * OC_OXK_BLOCK_Q8_K_SIZE;
            float wd;
            memcpy(&wd, pb, 4);
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();
            for (int ib32 = 0; ib32 < 8; ib32++) {
                int16_t ls;
                memcpy(&ls, pb + 16 + 2 * ib32, 2);
                const __m256i vls = _mm256_set1_epi16(ls);
                const __m256i w =
                    _mm256_loadu_si256((const __m256i *)(pb + 32 + 32 * ib32));
                const size_t qo = yo + 4 + 32 * (size_t)ib32;
#define OC_IQ3S_MULTI_STEP(ACC, ACT)                                           \
    do {                                                                       \
        const __m256i q = _mm256_loadu_si256((const __m256i *)((ACT) + qo));   \
        const __m256i p = _mm256_maddubs_epi16(_mm256_abs_epi8(q),             \
                                             _mm256_sign_epi8(w, q));        \
        ACC = _mm256_add_epi32(ACC, _mm256_madd_epi16(p, vls));                \
    } while (0)
                OC_IQ3S_MULTI_STEP(acc0, act0);
                OC_IQ3S_MULTI_STEP(acc1, act1);
                OC_IQ3S_MULTI_STEP(acc2, act2);
                OC_IQ3S_MULTI_STEP(acc3, act3);
            }
            float yd;
            memcpy(&yd, act0 + yo, 4); f0 = iq3_s_lane_acc(f0, wd * yd, acc0);
            memcpy(&yd, act1 + yo, 4); f1 = iq3_s_lane_acc(f1, wd * yd, acc1);
            memcpy(&yd, act2 + yo, 4); f2 = iq3_s_lane_acc(f2, wd * yd, acc2);
            memcpy(&yd, act3 + yo, 4); f3 = iq3_s_lane_acc(f3, wd * yd, acc3);
        }
        out[a + 0] = iq3_s_hsum_f32_8(f0);
        out[a + 1] = iq3_s_hsum_f32_8(f1);
        out[a + 2] = iq3_s_hsum_f32_8(f2);
        out[a + 3] = iq3_s_hsum_f32_8(f3);
    }
    for (; a < n_act; a++) {
        const uint8_t *act = acts + a * act_stride;
        __m256 f = _mm256_setzero_ps();
        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *pb = prep + b * OC_OXK_IQ3_S_PREP_BLOCK;
            const size_t yo = b * OC_OXK_BLOCK_Q8_K_SIZE;
            float wd, yd;
            memcpy(&wd, pb, 4);
            __m256i acc = _mm256_setzero_si256();
            for (int ib32 = 0; ib32 < 8; ib32++) {
                int16_t ls;
                memcpy(&ls, pb + 16 + 2 * ib32, 2);
                const __m256i vls = _mm256_set1_epi16(ls);
                const __m256i w =
                    _mm256_loadu_si256((const __m256i *)(pb + 32 + 32 * ib32));
                const size_t qo = yo + 4 + 32 * (size_t)ib32;
                OC_IQ3S_MULTI_STEP(acc, act);
            }
            memcpy(&yd, act + yo, 4);
            f = iq3_s_lane_acc(f, wd * yd, acc);
        }
        out[a] = iq3_s_hsum_f32_8(f);
    }
#undef OC_IQ3S_MULTI_STEP
}

#else  /* non-x86: forward every AVX symbol to the scalar reference. */

float oc_oxk_dot_q8_0_q8_0_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q8_0_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_q4_0_q8_0_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q4_0_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_q4_1_q8_0_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q4_1_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_q4_k_q8_k_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q4_k_q8_k_scalar(row, blocks, q8); }
float oc_oxk_dot_q5_k_q8_k_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q5_k_q8_k_scalar(row, blocks, q8); }
float oc_oxk_dot_q6_k_q8_k_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q6_k_q8_k_scalar(row, blocks, q8); }
float oc_oxk_dot_q2_k_q8_k_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q2_k_q8_k_scalar(row, blocks, q8); }
float oc_oxk_dot_q3_k_q8_k_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q3_k_q8_k_scalar(row, blocks, q8); }
float oc_oxk_dot_q2_k_prepped_avx2(const void *prep, size_t blocks, const uint8_t *act)
{ return oc_oxk_dot_q2_k_prepped(prep, blocks, act); }
float oc_oxk_dot_q3_k_prepped_avx2(const void *prep, size_t blocks, const uint8_t *act)
{ return oc_oxk_dot_q3_k_prepped(prep, blocks, act); }
float oc_oxk_dot_q6_k_prepped_avx2(const void *prep, size_t blocks, const uint8_t *act)
{ return oc_oxk_dot_q6_k_prepped(prep, blocks, act); }
void oc_oxk_dot_q2_k_prepped_multi_avx2(const void *prep, size_t blocks, const uint8_t *acts, size_t act_stride, size_t n_act, float *out)
{ for (size_t v = 0; v < n_act; v++) out[v] = oc_oxk_dot_q2_k_prepped(prep, blocks, acts + v * act_stride); }
void oc_oxk_dot_q3_k_prepped_multi_avx2(const void *prep, size_t blocks, const uint8_t *acts, size_t act_stride, size_t n_act, float *out)
{ for (size_t v = 0; v < n_act; v++) out[v] = oc_oxk_dot_q3_k_prepped(prep, blocks, acts + v * act_stride); }
void oc_oxk_dot_q6_k_prepped_multi_avx2(const void *prep, size_t blocks, const uint8_t *acts, size_t act_stride, size_t n_act, float *out)
{ for (size_t v = 0; v < n_act; v++) out[v] = oc_oxk_dot_q6_k_prepped(prep, blocks, acts + v * act_stride); }
void oc_oxk_matvec_q4_0_f32_avx2(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_matvec_q4_0_f32_scalar(w, n_rows, row_bytes, x, out); }
void oc_oxk_matvec_q4_k_f32_avx2(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_matvec_q4_k_f32_scalar(w, n_rows, row_bytes, x, out); }
void oc_oxk_matvec_q8_0_f32_avx2(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_matvec_q8_0_f32_scalar(w, n_rows, row_bytes, x, out); }

float oc_oxk_dot_q4_0_q8_0_avx512(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q4_0_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_q4_1_q8_0_avx512(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q4_1_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_q8_0_q8_0_avx512(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q8_0_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_q4_k_q8_k_avx512(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q4_k_q8_k_scalar(row, blocks, q8); }
float oc_oxk_dot_q5_k_q8_k_avx512(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q5_k_q8_k_scalar(row, blocks, q8); }
float oc_oxk_dot_q6_k_q8_k_avx512(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q6_k_q8_k_scalar(row, blocks, q8); }
void oc_oxk_matvec_q4_0_f32_avx512(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_matvec_q4_0_f32_scalar(w, n_rows, row_bytes, x, out); }
void oc_oxk_matvec_q4_k_f32_avx512(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_matvec_q4_k_f32_scalar(w, n_rows, row_bytes, x, out); }
void oc_oxk_matvec_q8_0_f32_avx512(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_matvec_q8_0_f32_scalar(w, n_rows, row_bytes, x, out); }

float oc_oxk_dot_iq3_s_q8_k_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_iq3_s_q8_k_scalar(row, blocks, q8); }
void oc_oxk_iq3_s_prep_row_avx2(const uint8_t *row, size_t blocks, void *scratch)
{ oc_oxk_iq3_s_prep_row_scalar(row, blocks, scratch); }
void oc_oxk_dot_iq3_s_prepped_multi_avx2(const void *scratch, size_t blocks,
                                         const uint8_t *acts, size_t act_stride,
                                         size_t n_act, float *out)
{ oc_oxk_dot_iq3_s_prepped_multi_scalar(scratch, blocks, acts, act_stride, n_act, out); }

void oc_oxk_dot_rows_q4_k_q8_k_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, float *out)
{ for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_q4_k_q8_k_scalar(rows + r * row_bytes, blocks, q8); }
void oc_oxk_dot_rows_q6_k_q8_k_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, float *out)
{ for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_q6_k_q8_k_scalar(rows + r * row_bytes, blocks, q8); }
void oc_oxk_dot_rows_q2_k_q8_k_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, float *out)
{ for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_q2_k_q8_k_scalar(rows + r * row_bytes, blocks, q8); }
void oc_oxk_dot_rows_q3_k_q8_k_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, float *out)
{ for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_q3_k_q8_k_scalar(rows + r * row_bytes, blocks, q8); }
void oc_oxk_dot_rows_iq3_s_q8_k_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, float *out)
{ for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_iq3_s_q8_k_scalar(rows + r * row_bytes, blocks, q8); }
void oc_oxk_dot_rows_iq3_s_q8_k_multi_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *acts, size_t act_stride, size_t n_act, float *out, size_t out_stride)
{ for (size_t a = 0; a < n_act; a++) for (size_t r = 0; r < n_rows; r++) out[a * out_stride + r] = oc_oxk_dot_iq3_s_q8_k_scalar(rows + r * row_bytes, blocks, acts + a * act_stride); }

float oc_oxk_dot_q5_0_q8_0_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q5_0_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_q5_1_q8_0_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_q5_1_q8_0_scalar(row, blocks, q8); }
float oc_oxk_dot_al5_xs_q8_0_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ return oc_oxk_dot_al5_xs_q8_0_scalar(row, blocks, q8); }

void oc_oxk_dot_rows_q5_k_q8_k_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, float *out)
{ for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_q5_k_q8_k_scalar(rows + r * row_bytes, blocks, q8); }
void oc_oxk_dot_rows_q4_0_q8_0_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, float *out)
{ for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_q4_0_q8_0_scalar(rows + r * row_bytes, blocks, q8); }
void oc_oxk_dot_rows_q4_1_q8_0_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, float *out)
{ for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_q4_1_q8_0_scalar(rows + r * row_bytes, blocks, q8); }
void oc_oxk_dot_rows_q5_0_q8_0_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, float *out)
{ for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_q5_0_q8_0_scalar(rows + r * row_bytes, blocks, q8); }
void oc_oxk_dot_rows_q5_1_q8_0_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, float *out)
{ for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_q5_1_q8_0_scalar(rows + r * row_bytes, blocks, q8); }
void oc_oxk_dot_rows_q8_0_q8_0_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, float *out)
{ for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_q8_0_q8_0_scalar(rows + r * row_bytes, blocks, q8); }
void oc_oxk_dot_rows_al5_xs_q8_0_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, float *out)
{ for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_al5_xs_q8_0_scalar(rows + r * row_bytes, blocks, q8); }

float oc_oxk_dot_q4_0_q8_0_qs_avx2(const uint8_t *row, size_t blocks, const uint8_t *q8, const int32_t *qsums)
{ (void)qsums; return oc_oxk_dot_q4_0_q8_0_scalar(row, blocks, q8); }
void oc_oxk_dot_rows_q4_0_q8_0_qs_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, const int32_t *qsums, float *out)
{ (void)qsums; for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_q4_0_q8_0_scalar(rows + r * row_bytes, blocks, q8); }
void oc_oxk_dot_rows_q4_0_q8_0_facc_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, const int32_t *qsums, float *out)
{ (void)qsums; for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_q4_0_q8_0_scalar(rows + r * row_bytes, blocks, q8); }
void oc_oxk_dot_rows_q8_0_q8_0_facc_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, float *out)
{ for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_q8_0_q8_0_scalar(rows + r * row_bytes, blocks, q8); }

void oc_oxk_q4_0_quad_prep_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, void *scratch)
{ oc_oxk_q4_0_quad_prep(rows, row_bytes, n_rows, blocks, scratch); }
void oc_oxk_dot_q4_0_q8_0_quad_multi_avx2(const void *prep4, size_t blocks, const uint8_t *acts, size_t act_stride, size_t dq_off, size_t n_act, float *res4)
{ oc_oxk_dot_q4_0_q8_0_quad_multi(prep4, blocks, acts, act_stride, dq_off, n_act, res4); }

#endif  /* __x86_64__ || __i386__ */
