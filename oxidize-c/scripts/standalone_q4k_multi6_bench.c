/* standalone_q4k_multi6_bench.c — A/B: committed 4-act prepped-multi
 * kernel vs N-act register-blocked variants (more ymm accumulator chains;
 * the kernel was latency-bound at 8 chains). All variants are bit-identical
 * per act (same per-block integer math and float accumulation order).
 * Build: cc -O2 -ffp-contract=off -march=native -Iinclude
 *        scripts/standalone_q4k_multi6_bench.c src/compute/oxk_avx2.c
 *        src/compute/oxk.c src/compute/oxk_avx512.c
 *        src/compute/quantization.c src/core/log.c src/core/error.c
 *        src/core/simd.c src/core/simd_avx2.c src/core/simd_avx512.c
 *        -o /tmp/m6 -lm -lpthread
 */
#include "oxidize/oxk.h"

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t s = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void)
{
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return (uint32_t)(s >> 32);
}
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static inline int32_t hsum_i32_8(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i t  = _mm_add_epi32(lo, hi);
    t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1, 0, 3, 2)));
    t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(t);
}
static inline __m128i hsum2_i32_8(__m256i p0, __m256i p1)
{
    const __m256i h  = _mm256_hadd_epi32(p0, p1);
    const __m256i h2 = _mm256_hadd_epi32(h, h);
    return _mm_add_epi32(_mm256_castsi256_si128(h2),
                         _mm256_extracti128_si256(h2, 1));
}

/* N-act variant of oc_oxk_dot_q4_k_prepped_multi_avx2 (same math). */
#define MULTI_BODY(FN, NACT)                                                   \
__attribute__((target("avx2,fma,f16c")))                                       \
static void FN(const void *prep, size_t blocks, const uint8_t *acts,           \
               size_t act_stride, size_t n_act, float *out)                    \
{                                                                              \
    const float   *d     = (const float *)prep;                                \
    const float   *dmin  = d + blocks;                                         \
    const uint8_t *codes = (const uint8_t *)(d + 2 * blocks);                  \
    const uint8_t *sc    = codes + blocks * OC_OXK_QK_K;                       \
    const uint8_t *mn    = sc + blocks * (OC_OXK_QK_K / 32u);                  \
    size_t a0 = 0;                                                             \
    for (; a0 + (NACT) <= n_act; a0 += (NACT)) {                               \
        const uint8_t *act[(NACT)];                                            \
        float sums[(NACT)];                                                    \
        for (int a = 0; a < (NACT); a++) {                                     \
            act[a] = acts + (a0 + (size_t)a) * act_stride;                     \
            sums[a] = 0.0f;                                                    \
        }                                                                      \
        for (size_t b = 0; b < blocks; b++) {                                  \
            const uint8_t *cb  = codes + b * OC_OXK_QK_K;                      \
            const uint8_t *scb = sc + b * (OC_OXK_QK_K / 32u);                 \
            const uint8_t *mnb = mn + b * (OC_OXK_QK_K / 32u);                 \
            __m256i acc_lo[(NACT)], acc_hi[(NACT)];                            \
            for (int a = 0; a < (NACT); a++) {                                 \
                acc_lo[a] = _mm256_setzero_si256();                            \
                acc_hi[a] = _mm256_setzero_si256();                            \
            }                                                                  \
            for (int gp = 0; gp < 4; gp++) {                                   \
                const __m256i cz0 = _mm256_loadu_si256(                        \
                    (const __m256i *)(cb + (size_t)gp * 64));                  \
                const __m256i cz1 = _mm256_loadu_si256(                        \
                    (const __m256i *)(cb + (size_t)gp * 64 + 32));             \
                const __m256i sl = _mm256_set1_epi16(                          \
                    (int16_t)scb[2 * gp]);                                     \
                const __m256i sh = _mm256_set1_epi16(                          \
                    (int16_t)scb[2 * gp + 1]);                                 \
                const size_t qoff = b * OC_OXK_BLOCK_Q8_K_SIZE + 4 +           \
                                    (size_t)gp * 64;                           \
                for (int a = 0; a < (NACT); a++) {                             \
                    const __m256i p0 = _mm256_maddubs_epi16(                   \
                        cz0, _mm256_loadu_si256(                               \
                            (const __m256i *)(act[a] + qoff)));                \
                    const __m256i p1 = _mm256_maddubs_epi16(                   \
                        cz1, _mm256_loadu_si256(                               \
                            (const __m256i *)(act[a] + qoff + 32)));           \
                    acc_lo[a] = _mm256_add_epi32(acc_lo[a],                    \
                        _mm256_madd_epi16(p0, sl));                            \
                    acc_hi[a] = _mm256_add_epi32(acc_hi[a],                    \
                        _mm256_madd_epi16(p1, sh));                            \
                }                                                              \
            }                                                                  \
            const __m256i mn16 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(        \
                _mm_loadl_epi64((const __m128i *)mnb),                         \
                _mm_setr_epi8(0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7)));              \
            for (int a = 0; a < (NACT); a += 2) {                              \
                const __m256i t0 =                                             \
                    _mm256_add_epi32(acc_lo[a], acc_hi[a]);                    \
                const __m256i t1 =                                             \
                    _mm256_add_epi32(acc_lo[a + 1], acc_hi[a + 1]);            \
                const __m128i h = hsum2_i32_8(t0, t1);                         \
                const int32_t pos0 = _mm_cvtsi128_si32(h);                     \
                const int32_t pos1 = _mm_extract_epi32(h, 1);                  \
                const uint8_t *qb0 =                                           \
                    act[a] + b * OC_OXK_BLOCK_Q8_K_SIZE;                       \
                const uint8_t *qb1 =                                           \
                    act[a + 1] + b * OC_OXK_BLOCK_Q8_K_SIZE;                   \
                float dq0, dq1;                                                \
                memcpy(&dq0, qb0, 4);                                          \
                memcpy(&dq1, qb1, 4);                                          \
                const int32_t min0 = hsum_i32_8(_mm256_madd_epi16(             \
                    mn16, _mm256_loadu_si256(                                  \
                        (const __m256i *)(qb0 + 4 + OC_OXK_QK_K))));           \
                const int32_t min1 = hsum_i32_8(_mm256_madd_epi16(             \
                    mn16, _mm256_loadu_si256(                                  \
                        (const __m256i *)(qb1 + 4 + OC_OXK_QK_K))));           \
                sums[a] += d[b] * dq0 * (float)pos0                            \
                         - dmin[b] * dq0 * (float)min0;                        \
                sums[a + 1] += d[b] * dq1 * (float)pos1                        \
                             - dmin[b] * dq1 * (float)min1;                    \
            }                                                                  \
        }                                                                      \
        for (int a = 0; a < (NACT); a++)                                       \
            out[a0 + (size_t)a] = sums[a];                                     \
    }                                                                          \
    for (; a0 < n_act; a0++)                                                   \
        out[a0] = oc_oxk_dot_q4_k_prepped(prep, blocks,                        \
                                          acts + a0 * act_stride);             \
}


MULTI_BODY(multi4, 4)
MULTI_BODY(multi6, 6)
MULTI_BODY(multi8, 8)

int main(int argc, char **argv)
{
    const size_t rows = argc > 1 ? (size_t)atol(argv[1]) : 512;
    const size_t cols = argc > 2 ? (size_t)atol(argv[2]) : 2560;
    const size_t n    = argc > 3 ? (size_t)atol(argv[3]) : 501;
    const size_t blocks = cols / OC_OXK_QK_K;
    const size_t row_bytes = blocks * OC_OXK_BLOCK_Q4_K_SIZE;
    const size_t as = blocks * OC_OXK_BLOCK_Q8_K_SIZE;

    uint8_t *w = malloc(rows * row_bytes);
    uint8_t *acts = malloc(n * as);
    float *o4 = malloc(n * sizeof(float));
    float *o5 = malloc(n * sizeof(float));
    float *o6 = malloc(n * sizeof(float));
    float *o8 = malloc(n * sizeof(float));
    void *prep = malloc(oc_oxk_q4_k_prep_bytes(blocks) + 64);
    for (size_t r = 0; r < rows; r++) {
        uint8_t *wb = w + r * row_bytes;
        for (size_t i = 4; i < row_bytes; i++)
            wb[i] = (uint8_t)((i % OC_OXK_BLOCK_Q4_K_SIZE) >= 16
                                  ? rnd() & 0x0F : rnd());
        for (size_t b = 0; b < blocks; b++) {
            wb[b * OC_OXK_BLOCK_Q4_K_SIZE]     = 0x38;
            wb[b * OC_OXK_BLOCK_Q4_K_SIZE + 2] = 0x38;
        }
    }
    for (size_t v = 0; v < n; v++) {
        uint8_t *qb = acts + v * as;
        for (size_t b = 0; b < blocks; b++) {
            uint8_t *bl = qb + b * OC_OXK_BLOCK_Q8_K_SIZE;
            for (size_t i = 4; i < OC_OXK_BLOCK_Q8_K_SIZE; i++)
                bl[i] = (uint8_t)rnd();
            float dq = 0.75f;
            memcpy(bl, &dq, 4);
        }
    }

    /* bit-exactness: committed kernel vs variants */
    for (size_t r = 0; r < 4; r++) {
        oc_oxk_q4_k_prep_row(w + r * row_bytes, blocks, prep);
        oc_oxk_dot_q4_k_prepped_multi_avx2(prep, blocks, acts, as, n, o4);
        
        multi6(prep, blocks, acts, as, n, o6);
        multi8(prep, blocks, acts, as, n, o8);
        for (size_t v = 0; v < n; v++) {
            uint32_t b4, b6, b8;
            memcpy(&b4, &o4[v], 4);
            
            memcpy(&b6, &o6[v], 4);
            memcpy(&b8, &o8[v], 4);
            if (b4 != b6 || b4 != b8) {
                fprintf(stderr, "MISMATCH r=%zu v=%zu: %.9g %.9g %.9g\n",
                        r, v, (double)o4[v], (double)o6[v], (double)o8[v]);
                return 1;
            }
        }
    }
    printf("bit-exact vs committed kernel\n");

    const double gflop = 2.0 * (double)rows * cols * n / 1e9;
    const struct { const char *name;
                   void (*fn)(const void *, size_t, const uint8_t *,
                              size_t, size_t, float *); } vs[] = {
        { "4-act (committed)", multi4 },
        
        { "6-act", multi6 },
        { "8-act", multi8 },
    };
    for (int k = 0; k < 3; k++) {
        const double t0 = now();
        for (size_t r = 0; r < rows; r++) {
            oc_oxk_q4_k_prep_row(w + r * row_bytes, blocks, prep);
            vs[k].fn(prep, blocks, acts, as, n, o4);
        }
        const double dt = now() - t0;
        printf("%s: %.3fs = %.1f GFLOPS\n", vs[k].name, dt, gflop / dt);
    }
    return 0;
}
