/*
 * oxk_avx2.c — AVX2 optimized OXK kernels.
 *
 * Real AVX2 implementations for Q8_0 and Q4_K; Q4_0/Q4_1/Q5_K/Q6_K still
 * forward to scalar. Q4_K matters most in practice — it is what most models
 * ship as — and running it on the scalar path made Q4_K_M slower than the
 * physically larger Q8_0.
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

/* ─── AVX2 Q8_0 × Q8_0 dot product ─────────────────────────────────────── */

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q8_0_q8_0_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    __m256 acc = _mm256_setzero_ps();

    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;

        /* Load f16 scales and convert to f32. */
        __m128i d_w_h = _mm_loadl_epi64((const __m128i *)wb);
        __m128i d_q_h = _mm_loadl_epi64((const __m128i *)qb);
        __m256 dw = _mm256_cvtph_ps(d_w_h);
        __m256 dq = _mm256_cvtph_ps(d_q_h);

        /* Extract scale values (first element of each conversion). */
        float dw_f = _mm256_cvtss_f32(dw);
        float dq_f = _mm256_cvtss_f32(dq);
        __m256 scale = _mm256_set1_ps(dw_f * dq_f);

        /* Load 32 int8 values from weight and activation. */
        const int8_t *wv = (const int8_t *)(wb + 2);
        const int8_t *qv = (const int8_t *)(qb + 2);

        /* Process 32 bytes in 4 chunks of 8 (using _mm256_cvtepi8_ps). */
        __m256i w0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)wv));
        __m256i w1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(wv + 8)));
        __m256i q0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)qv));
        __m256i q1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qv + 8)));

        __m256 p0 = _mm256_mul_ps(_mm256_cvtepi32_ps(w0), _mm256_cvtepi32_ps(q0));
        __m256 p1 = _mm256_mul_ps(_mm256_cvtepi32_ps(w1), _mm256_cvtepi32_ps(q1));

        __m256i w2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(wv + 16)));
        __m256i w3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(wv + 24)));
        __m256i q2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qv + 16)));
        __m256i q3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qv + 24)));

        __m256 p2 = _mm256_mul_ps(_mm256_cvtepi32_ps(w2), _mm256_cvtepi32_ps(q2));
        __m256 p3 = _mm256_mul_ps(_mm256_cvtepi32_ps(w3), _mm256_cvtepi32_ps(q3));

        /* Horizontal sum of products. */
        __m256 sum01 = _mm256_add_ps(p0, p1);
        __m256 sum23 = _mm256_add_ps(p2, p3);
        __m256 sum = _mm256_add_ps(sum01, sum23);

        /* Scale by dw * dq. */
        __m256 scaled = _mm256_mul_ps(sum, scale);

        /* Horizontal add to accumulator. */
        acc = _mm256_add_ps(acc, scaled);
    }

    /* Horizontal sum of acc. */
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

/* ─── AVX2 Q4_0 × Q8_0 dot product ─────────────────────────────────────── */

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q4_0_q8_0_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    /* For now, forward to scalar for correctness.
     * A true AVX2 implementation would unpack 4-bit nibbles using
     * _mm256_and_si256 and _mm256_srli_epi16, then multiply by Q8 values. */
    return oc_oxk_dot_q4_0_q8_0_scalar(row, blocks, q8);
}

/* ─── AVX2 Q4_1 × Q8_0 dot product ─────────────────────────────────────── */

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q4_1_q8_0_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    return oc_oxk_dot_q4_1_q8_0_scalar(row, blocks, q8);
}

/* ─── AVX2 Q4_K × Q8_K dot product ─────────────────────────────────────── */

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

__attribute__((target("avx2,fma,f16c")))
static inline float oxk_f16(const uint8_t *p)
{
    uint16_t h;
    memcpy(&h, p, 2);
    return _cvtsh_ss(h);
}

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

/* ─── AVX2 k-quants (forward to scalar) ────────────────────────────────── */

__attribute__((target("avx2,fma,f16c")))
float oc_oxk_dot_q5_k_q8_k_avx2(const uint8_t *row, size_t blocks,
                                const uint8_t *q8)
{
    return oc_oxk_dot_q5_k_q8_k_scalar(row, blocks, q8);
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
    /* shuf[i]: bytes [2i x8, 2i+1 x8] — broadcasts scale group 2i to the
     * first eight int16 lanes of a maddubs result and 2i+1 to the last
     * eight, i.e. to elements 0..15 and 16..31 of one 32-value slot. */
    static const uint8_t shuf[8][16] = {
        { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 },
        { 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5 },
        { 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7 },
        { 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9 },
        { 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11 },
        { 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13 },
        { 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15 },
    };

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
            _mm_loadu_si128((const __m128i *)shuf[4 * n + (K)]))),             \
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
OXK_ROWS_WRAPPERS(q6_k, q6_k_dot_body)
OXK_ROWS_WRAPPERS(iq3_s, iq3_s_dot_body)

#undef OXK_ROWS_WRAPPERS

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
 * vector (and its abs) is loaded once per four dots. w*q is computed as
 * maddubs(|w|, sign(q, w)); |w| <= 15 so the int16 pair sums cannot
 * saturate. */
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
                const __m256i aw = _mm256_abs_epi8(w);
                const size_t qo = yo + 4 + 32 * (size_t)ib32;
#define OC_IQ3S_MULTI_STEP(ACC, ACT)                                           \
    do {                                                                       \
        const __m256i q = _mm256_loadu_si256((const __m256i *)((ACT) + qo));   \
        const __m256i p = _mm256_maddubs_epi16(aw, _mm256_sign_epi8(q, w));    \
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
                const __m256i aw = _mm256_abs_epi8(w);
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
void oc_oxk_dot_rows_iq3_s_q8_k_avx2(const uint8_t *rows, size_t row_bytes, size_t n_rows, size_t blocks, const uint8_t *q8, float *out)
{ for (size_t r = 0; r < n_rows; r++) out[r] = oc_oxk_dot_iq3_s_q8_k_scalar(rows + r * row_bytes, blocks, q8); }

#endif  /* __x86_64__ || __i386__ */
