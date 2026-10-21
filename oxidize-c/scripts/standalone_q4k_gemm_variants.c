/* standalone_q4k_gemm_variants.c — f32-FMA GEMM variants for the Q4_K
 * batched prefill, single-thread GFLOPS race:
 *   A: dequant per (row, block) to an L1 buffer, all acts streamed
 *   D: 6-act register-blocked GEMM — 12 ymm accumulators, dequant result
 *      reused across the 6 acts from L1, act chunk L1-resident
 * Numeric check vs the scalar dequant reference, then timing.
 * Build: cc -O2 -ffp-contract=off -march=native -Iinclude
 *        scripts/standalone_q4k_gemm_variants.c src/compute/quantization.c
 *        src/core/log.c src/core/error.c -o /tmp/vbench -lm
 */
#include "oxidize/oxk.h"
#include "oxidize/quant.h"

/* Standalone build: force the scalar reference dequant in quantization.c
 * (the SIMD hook lives in autotune.c, which we don't link). */
bool oc_simd_try_dequant(OcGgufQuantizationType qtype, const uint8_t *src,
                         size_t src_len, float *dst, size_t value_count)
{
    (void)qtype; (void)src; (void)src_len; (void)dst; (void)value_count;
    return false;
}

#include <immintrin.h>
#include <math.h>
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
static inline float f16of(const uint8_t *p)
{
    uint16_t h; memcpy(&h, p, 2);
    return _cvtsh_ss(h);
}

/* Port of quantization.c get_scale_min_k4 for j = 0..7 (12 scale bytes). */
static inline void dq_scales(const uint8_t *scales, uint8_t sc[8],
                             uint8_t mn[8])
{
    for (int j = 0; j < 4; j++) {
        sc[j] = (uint8_t)(scales[j] & 63u);
        mn[j] = (uint8_t)(scales[j + 4] & 63u);
    }
    for (int j = 4; j < 8; j++) {
        sc[j] = (uint8_t)((scales[j + 4] & 0x0Fu)
                          | ((scales[j - 4] >> 6) << 4));
        mn[j] = (uint8_t)(((scales[j + 4] >> 4) & 0x0Fu)
                          | ((scales[j] >> 6) << 4));
    }
}

/* Dequantize quarter gp (64 floats) of one Q4_K block into out[0..7].
 * Layout: elements 0..31 = low nibbles of qs[gp*32..], scale 2*gp;
 *         elements 32..63 = high nibbles of the same bytes, scale 2*gp+1.
 * value = d*sc*nib - dmin*mn. */
__attribute__((target("avx2,fma,f16c")))
static inline void dq_block_q(const uint8_t *wb, int gp, __m256 out[8])
{
    const float d = f16of(wb);
    const float dmin = f16of(wb + 2);
    uint8_t sc[8], mn[8];
    dq_scales(wb + 4, sc, mn);
    const __m256 dsc0 = _mm256_set1_ps(d * (float)sc[2 * gp]);
    const __m256 dmn0 = _mm256_set1_ps(-(dmin * (float)mn[2 * gp]));
    const __m256 dsc1 = _mm256_set1_ps(d * (float)sc[2 * gp + 1]);
    const __m256 dmn1 = _mm256_set1_ps(-(dmin * (float)mn[2 * gp + 1]));

    const __m128i p0 = _mm_loadu_si128((const __m128i *)(wb + 16 + gp * 32));
    const __m128i p1 = _mm_loadu_si128((const __m128i *)(wb + 32 + gp * 32));
    const __m128i lownib = _mm_set1_epi8(0x0F);
    const __m128i p0lo = _mm_and_si128(p0, lownib);
    const __m128i p1lo = _mm_and_si128(p1, lownib);
    const __m128i p0hi = _mm_and_si128(_mm_srli_epi16(p0, 4), lownib);
    const __m128i p1hi = _mm_and_si128(_mm_srli_epi16(p1, 4), lownib);
    const __m128i sub0[2] = { p0lo, p1lo };  /* elements 0..31,  scale 2gp   */
    const __m128i sub1[2] = { p0hi, p1hi };  /* elements 32..63, scale 2gp+1 */
    for (int c = 0; c < 4; c++) {
        const __m128i *src = (c >> 1) ? sub1 : sub0;
        const __m128i x0 = src[c & 1];
        const __m128i x1 = _mm_shuffle_epi32(x0, _MM_SHUFFLE(3, 2, 3, 2));
        const __m256 ds = (c >> 1) ? dsc1 : dsc0;
        const __m256 dm = (c >> 1) ? dmn1 : dmn0;
        out[2 * c] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(x0)), ds, dm);
        out[2 * c + 1] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(x1)), ds, dm);
    }
}

static float hsum8(__m256 v)
{
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v),
                                _mm256_extractf128_ps(v, 1));
    const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 1)));
}

/* Variant A: dequant per (row, block) into an L1 buffer, acts streamed. */
__attribute__((target("avx2,fma,f16c")))
static void varA(const uint8_t *row, size_t blocks, const float *acts,
                 size_t stride, size_t n, float *out)
{
    float dq[256];
    __m256 *acc = aligned_alloc(32, ((n * sizeof(__m256) + 63) / 64) * 64);
    for (size_t v = 0; v < n; v++) acc[v] = _mm256_setzero_ps();
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_K_SIZE;
        _mm_prefetch((const char *)(wb + 640), _MM_HINT_T0);
        for (int gp = 0; gp < 4; gp++) {
            __m256 w[8];
            dq_block_q(wb, gp, w);
            _mm256_storeu_ps(dq + gp * 64, w[0]);
            _mm256_storeu_ps(dq + gp * 64 + 8, w[1]);
            _mm256_storeu_ps(dq + gp * 64 + 16, w[2]);
            _mm256_storeu_ps(dq + gp * 64 + 24, w[3]);
            _mm256_storeu_ps(dq + gp * 64 + 32, w[4]);
            _mm256_storeu_ps(dq + gp * 64 + 40, w[5]);
            _mm256_storeu_ps(dq + gp * 64 + 48, w[6]);
            _mm256_storeu_ps(dq + gp * 64 + 56, w[7]);
        }
        const size_t aoff = b * 256;
        for (size_t v = 0; v < n; v++) {
            const float *a = acts + v * stride + aoff;
            __m256 p0 = _mm256_setzero_ps(), p1 = _mm256_setzero_ps();
            for (unsigned kh = 0; kh < 256; kh += 16) {
                p0 = _mm256_fmadd_ps(_mm256_loadu_ps(dq + kh),
                                     _mm256_loadu_ps(a + kh), p0);
                p1 = _mm256_fmadd_ps(_mm256_loadu_ps(dq + kh + 8),
                                     _mm256_loadu_ps(a + kh + 8), p1);
            }
            acc[v] = _mm256_add_ps(acc[v], _mm256_add_ps(p0, p1));
        }
    }
    for (size_t v = 0; v < n; v++) out[v] = hsum8(acc[v]);
    free(acc);
}

/* Variant D: 6-act register-blocked GEMM. Per (row, 6-act chunk, block):
 * dequant once, reuse across 6 acts; 12 ymm accumulators stay in registers
 * across blocks; act chunk (6 KB) and dq buffer (1 KB) stay in L1. */
#define GACT 6
__attribute__((target("avx2,fma,f16c")))
static void varD(const uint8_t *row, size_t blocks, const float *acts,
                 size_t stride, size_t n, float *out)
{
    float dq[256];
    const size_t nchunk = n / GACT;
    for (size_t c = 0; c < nchunk; c++) {
        __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
        __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
        __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
        __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
        __m256 a40 = _mm256_setzero_ps(), a41 = _mm256_setzero_ps();
        __m256 a50 = _mm256_setzero_ps(), a51 = _mm256_setzero_ps();
        const float *ap[GACT];
        for (int v = 0; v < GACT; v++)
            ap[v] = acts + (c * GACT + v) * stride;
        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_K_SIZE;
            _mm_prefetch((const char *)(wb + 640), _MM_HINT_T0);
            for (int gp = 0; gp < 4; gp++) {
                __m256 w[8];
                dq_block_q(wb, gp, w);
                _mm256_storeu_ps(dq + gp * 64, w[0]);
                _mm256_storeu_ps(dq + gp * 64 + 8, w[1]);
                _mm256_storeu_ps(dq + gp * 64 + 16, w[2]);
                _mm256_storeu_ps(dq + gp * 64 + 24, w[3]);
                _mm256_storeu_ps(dq + gp * 64 + 32, w[4]);
                _mm256_storeu_ps(dq + gp * 64 + 40, w[5]);
                _mm256_storeu_ps(dq + gp * 64 + 48, w[6]);
                _mm256_storeu_ps(dq + gp * 64 + 56, w[7]);
            }
            const size_t aoff = b * 256;
            for (unsigned kh = 0; kh < 256; kh += 16) {
                const __m256 d0 = _mm256_loadu_ps(dq + kh);
                const __m256 d1 = _mm256_loadu_ps(dq + kh + 8);
                a00 = _mm256_fmadd_ps(d0, _mm256_loadu_ps(ap[0] + aoff + kh), a00);
                a01 = _mm256_fmadd_ps(d1, _mm256_loadu_ps(ap[0] + aoff + kh + 8), a01);
                a10 = _mm256_fmadd_ps(d0, _mm256_loadu_ps(ap[1] + aoff + kh), a10);
                a11 = _mm256_fmadd_ps(d1, _mm256_loadu_ps(ap[1] + aoff + kh + 8), a11);
                a20 = _mm256_fmadd_ps(d0, _mm256_loadu_ps(ap[2] + aoff + kh), a20);
                a21 = _mm256_fmadd_ps(d1, _mm256_loadu_ps(ap[2] + aoff + kh + 8), a21);
                a30 = _mm256_fmadd_ps(d0, _mm256_loadu_ps(ap[3] + aoff + kh), a30);
                a31 = _mm256_fmadd_ps(d1, _mm256_loadu_ps(ap[3] + aoff + kh + 8), a31);
                a40 = _mm256_fmadd_ps(d0, _mm256_loadu_ps(ap[4] + aoff + kh), a40);
                a41 = _mm256_fmadd_ps(d1, _mm256_loadu_ps(ap[4] + aoff + kh + 8), a41);
                a50 = _mm256_fmadd_ps(d0, _mm256_loadu_ps(ap[5] + aoff + kh), a50);
                a51 = _mm256_fmadd_ps(d1, _mm256_loadu_ps(ap[5] + aoff + kh + 8), a51);
            }
        }
        out[c * GACT + 0] = hsum8(a00) + hsum8(a01);
        out[c * GACT + 1] = hsum8(a10) + hsum8(a11);
        out[c * GACT + 2] = hsum8(a20) + hsum8(a21);
        out[c * GACT + 3] = hsum8(a30) + hsum8(a31);
        out[c * GACT + 4] = hsum8(a40) + hsum8(a41);
        out[c * GACT + 5] = hsum8(a50) + hsum8(a51);
    }
    /* tail acts, scalar-ish f32 path (same math, few acts) */
    for (size_t v = nchunk * GACT; v < n; v++) {
        float sum = 0.0f;
        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_K_SIZE;
            __m256 w[8];
            float ldq[256];
            for (int gp = 0; gp < 4; gp++) {
                dq_block_q(wb, gp, w);
                _mm256_storeu_ps(ldq + gp * 64, w[0]);
                _mm256_storeu_ps(ldq + gp * 64 + 8, w[1]);
                _mm256_storeu_ps(ldq + gp * 64 + 16, w[2]);
                _mm256_storeu_ps(ldq + gp * 64 + 24, w[3]);
                _mm256_storeu_ps(ldq + gp * 64 + 32, w[4]);
                _mm256_storeu_ps(ldq + gp * 64 + 40, w[5]);
                _mm256_storeu_ps(ldq + gp * 64 + 48, w[6]);
                _mm256_storeu_ps(ldq + gp * 64 + 56, w[7]);
            }
            const float *a = acts + v * stride + b * 256;
            for (unsigned i = 0; i < 256; i++) sum += ldq[i] * a[i];
        }
        out[v] = sum;
    }
}

int main(int argc, char **argv)
{
    const size_t rows = argc > 1 ? (size_t)atol(argv[1]) : 512;
    const size_t cols = argc > 2 ? (size_t)atol(argv[2]) : 2560;
    const size_t n = argc > 3 ? (size_t)atol(argv[3]) : 501;
    const size_t blocks = cols / OC_OXK_QK_K;
    const size_t row_bytes = blocks * OC_OXK_BLOCK_Q4_K_SIZE;

    uint8_t *w = malloc(rows * row_bytes);
    float *acts = malloc(n * cols * sizeof(float));
    float *ref = malloc(n * sizeof(float));
    float *out = malloc(n * sizeof(float));
    float *deq = malloc(cols * sizeof(float));
    for (size_t r = 0; r < rows; r++) {
        uint8_t *wb = w + r * row_bytes;
        for (size_t i = 4; i < row_bytes; i++) wb[i] = (uint8_t)rnd();
        for (size_t b = 0; b < blocks; b++) {
            uint8_t *bl = wb + b * OC_OXK_BLOCK_Q4_K_SIZE;
            bl[0] = 0x00; bl[1] = 0x38; /* f16 0.5 */
            bl[2] = 0x00; bl[3] = 0x38;
        }
    }
    for (size_t i = 0; i < n * cols; i++)
        acts[i] = (float)((int)(rnd() % 2001) - 1000) / 500.0f;

    /* numeric check of each variant against the scalar dequant reference,
     * row 0 */
    oc_quant_dequant_row(OC_QUANT_Q4_K_M, w, row_bytes, deq, cols);
    double maxerr = 0.0;
    {
        double mags = 0.0;
        for (size_t v = 0; v < n; v++) {
            double sum = 0.0, mag = 0.0;
            for (size_t i = 0; i < cols; i++) {
                const double p = (double)deq[i] * (double)acts[v * cols + i];
                sum += p; mag += fabs(p);
            }
            ref[v] = (float)sum;
            mags = mag > mags ? mag : mags;
        }
        void (*fns[2])(const uint8_t *, size_t, const float *, size_t,
                       size_t, float *) = { varA, varD };
        const char *names[2] = { "A", "D" };
        for (int k = 0; k < 2; k++) {
            fns[k](w, blocks, acts, cols, n, out);
            for (size_t v = 0; v < n; v++) {
                const double err = fabs((double)out[v] - (double)ref[v]) /
                                   (mags + 1e-9);
                if (err > maxerr) maxerr = err;
                if (!(err <= 1e-5)) {
                    fprintf(stderr, "variant %s FAIL v=%zu err=%.3g\n",
                            names[k], v, err);
                    return 1;
                }
            }
        }
    }
    printf("numeric: max err %.3g (sum|terms|-normalized) — all variants\n",
           maxerr);

    const struct { const char *name;
                   void (*fn)(const uint8_t *, size_t, const float *,
                              size_t, size_t, float *); } vs[] = {
        { "A dq-buf stream", varA },
        { "D reg-blk 6-act", varD },
    };
    for (int k = 0; k < 2; k++) {
        const double t0 = now();
        for (size_t r = 0; r < rows; r++)
            vs[k].fn(w + r * row_bytes, blocks, acts, cols, n, out);
        const double dt = now() - t0;
        const double gflop = 2.0 * (double)rows * cols * n / 1e9;
        printf("%s: %.3fs = %.1f GFLOPS single-thread\n",
               vs[k].name, dt, gflop / dt);
    }
    return 0;
}
