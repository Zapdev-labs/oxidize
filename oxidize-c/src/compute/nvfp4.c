/*
 * nvfp4.c — NVFP4 activation quantization, quantization, and GEMV/GEMM.
 * See nvfp4.h for the layout and numeric convention.
 */
#include "oxidize/nvfp4.h"

#include <math.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define OC_NVFP4_X86 1
#endif

static const int8_t E2M1_X2[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

/* ue4m3(b) * 0.5: the half folds the doubled E2M1 integers back to e2m1. */
static float g_ue4m3_half[256];
static int   g_tables_ready;
static int   g_have_avx2 = -1;

static void init_tables(void)
{
    if (g_tables_ready) return;
    for (int b = 0; b < 256; b++) g_ue4m3_half[b] = 0.5f * oc_nvfp4_ue4m3((uint8_t)b);
    g_tables_ready = 1;
}

size_t oc_nvfp4_row_bytes(size_t cols)
{
    return cols / OC_NVFP4_BLOCK * OC_NVFP4_BLOCK_BYTES;
}

float oc_nvfp4_ue4m3(uint8_t b)
{
    b &= 0x7F;
    if (b == 0 || b == 0x7F) return 0.0f;
    int e = (b >> 3) & 0xF;
    int m = b & 7;
    if (e == 0) return ldexpf((float)m, -9);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

uint8_t oc_nvfp4_ue4m3_encode(float v)
{
    if (!(v > 0.0f)) return 0;
    if (v >= 448.0f) return 0x7E;
    int lo = 0, hi = 0x7E;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (oc_nvfp4_ue4m3((uint8_t)mid) <= v) lo = mid; else hi = mid;
    }
    float dl = v - oc_nvfp4_ue4m3((uint8_t)lo);
    float dh = oc_nvfp4_ue4m3((uint8_t)hi) - v;
    return (uint8_t)(dh <= dl ? hi : lo);
}

bool oc_nvfp4_have_avx2(void)
{
    if (g_have_avx2 < 0) {
#ifdef OC_NVFP4_X86
        g_have_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
        g_have_avx2 = 0;
#endif
    }
    return g_have_avx2 == 1;
}

void oc_nvfp4_quantize_act(const float *x, size_t cols, OcNvfp4Act out)
{
    for (size_t b = 0; b < cols / OC_NVFP4_BLOCK; b++) {
        const float *xb = x + b * OC_NVFP4_BLOCK;
        float amax = 0.0f;
        for (size_t i = 0; i < OC_NVFP4_BLOCK; i++) {
            float a = fabsf(xb[i]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        out.d[b] = d;
        int8_t *q = out.q + b * OC_NVFP4_BLOCK;
        for (size_t s = 0; s < 4; s++) {
            for (size_t j = 0; j < 8; j++) {
                q[s * 8 + j]      = (int8_t)lrintf(xb[s * 16 + j] * id);
                q[32 + s * 8 + j] = (int8_t)lrintf(xb[s * 16 + 8 + j] * id);
            }
        }
    }
}

void oc_nvfp4_dequant_row(const uint8_t *row, size_t cols, float *out)
{
    init_tables();
    for (size_t b = 0; b < cols / OC_NVFP4_BLOCK; b++) {
        const uint8_t *blk = row + b * OC_NVFP4_BLOCK_BYTES;
        float *o = out + b * OC_NVFP4_BLOCK;
        for (size_t s = 0; s < 4; s++) {
            float d = g_ue4m3_half[blk[s]];
            for (size_t j = 0; j < 8; j++) {
                uint8_t v = blk[4 + s * 8 + j];
                o[s * 16 + j]     = d * (float)E2M1_X2[v & 15];
                o[s * 16 + 8 + j] = d * (float)E2M1_X2[v >> 4];
            }
        }
    }
}

void oc_nvfp4_gemv_rows_scalar(const uint8_t *w, size_t cols, size_t r0, size_t r1,
                               float gscale, OcNvfp4Act act, float *y)
{
    init_tables();
    const size_t nb = cols / OC_NVFP4_BLOCK;
    const size_t rb = oc_nvfp4_row_bytes(cols);
    for (size_t r = r0; r < r1; r++) {
        const uint8_t *row = w + r * rb;
        float acc = 0.0f;
        for (size_t b = 0; b < nb; b++) {
            const uint8_t *blk = row + b * OC_NVFP4_BLOCK_BYTES;
            const int8_t *q = act.q + b * OC_NVFP4_BLOCK;
            float sb = 0.0f;
            for (size_t s = 0; s < 4; s++) {
                int32_t si = 0;
                for (size_t j = 0; j < 8; j++) {
                    uint8_t v = blk[4 + s * 8 + j];
                    si += (int32_t)E2M1_X2[v & 15] * q[s * 8 + j];
                    si += (int32_t)E2M1_X2[v >> 4] * q[32 + s * 8 + j];
                }
                sb += (float)si * g_ue4m3_half[blk[s]];
            }
            acc += sb * act.d[b];
        }
        y[r] = acc * gscale;
    }
}

#ifdef OC_NVFP4_X86
__attribute__((target("avx2,fma")))
static inline float hsum8(__m256 v)
{
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/* Up to 4 activation vectors per pass over a row: the weight block is
 * decoded once and reused, so a verify batch reads each row once per tile. */
#define NVFP4_TILE 4

__attribute__((target("avx2,fma")))
static void gemm_rows_avx2(const uint8_t *w, size_t cols, size_t r0, size_t r1,
                           float gscale, const OcNvfp4Act *acts, size_t n_vec,
                           float *y, size_t y_stride)
{
    const size_t nb = cols / OC_NVFP4_BLOCK;
    const size_t rb = oc_nvfp4_row_bytes(cols);
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)E2M1_X2));
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i pair = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);

    for (size_t v0 = 0; v0 < n_vec; v0 += NVFP4_TILE) {
        const size_t nv = n_vec - v0 < NVFP4_TILE ? n_vec - v0 : NVFP4_TILE;
        for (size_t r = r0; r < r1; r++) {
            const uint8_t *row = w + r * rb;
            __m256 acc[NVFP4_TILE];
            for (size_t t = 0; t < NVFP4_TILE; t++) acc[t] = _mm256_setzero_ps();
            for (size_t b = 0; b < nb; b++) {
                const uint8_t *blk = row + b * OC_NVFP4_BLOCK_BYTES;
                _mm_prefetch((const char *)blk + 1024, _MM_HINT_T0);
                const __m256i qs = _mm256_loadu_si256((const __m256i *)(blk + 4));
                const __m256i wlo = _mm256_shuffle_epi8(lut, _mm256_and_si256(qs, m4));
                const __m256i whi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(qs, 4), m4));
                const __m256i alo_w = _mm256_sign_epi8(wlo, wlo);
                const __m256i ahi_w = _mm256_sign_epi8(whi, whi);
                const __m128 d4 = _mm_setr_ps(g_ue4m3_half[blk[0]], g_ue4m3_half[blk[1]],
                                              g_ue4m3_half[blk[2]], g_ue4m3_half[blk[3]]);
                const __m256 dw = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(d4), pair);
                for (size_t t = 0; t < nv; t++) {
                    const OcNvfp4Act *a = &acts[v0 + t];
                    const int8_t *q = a->q + b * OC_NVFP4_BLOCK;
                    const __m256i qlo = _mm256_loadu_si256((const __m256i *)q);
                    const __m256i qhi = _mm256_loadu_si256((const __m256i *)(q + 32));
                    const __m256i plo = _mm256_maddubs_epi16(alo_w, _mm256_sign_epi8(qlo, wlo));
                    const __m256i phi = _mm256_maddubs_epi16(ahi_w, _mm256_sign_epi8(qhi, whi));
                    const __m256i s32 = _mm256_madd_epi16(_mm256_add_epi16(plo, phi), ones);
                    const __m256 sc = _mm256_mul_ps(dw, _mm256_set1_ps(a->d[b]));
                    acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s32), sc, acc[t]);
                }
            }
            for (size_t t = 0; t < nv; t++) y[(v0 + t) * y_stride + r] = hsum8(acc[t]) * gscale;
        }
    }
}
#endif

void oc_nvfp4_gemm_rows(const uint8_t *w, size_t cols, size_t r0, size_t r1,
                        float gscale, const OcNvfp4Act *acts, size_t n_vec,
                        float *y, size_t y_stride)
{
    init_tables();
#ifdef OC_NVFP4_X86
    if (oc_nvfp4_have_avx2()) {
        gemm_rows_avx2(w, cols, r0, r1, gscale, acts, n_vec, y, y_stride);
        return;
    }
#endif
    for (size_t v = 0; v < n_vec; v++)
        oc_nvfp4_gemv_rows_scalar(w, cols, r0, r1, gscale, acts[v], y + v * y_stride);
}

void oc_nvfp4_planar(const uint8_t *w, size_t rows, size_t cols, uint8_t *out)
{
    const size_t n = rows * (cols / OC_NVFP4_BLOCK);
    uint8_t *sc = out + n * 32;
    for (size_t i = 0; i < n; i++) {
        memcpy(out + i * 32, w + i * OC_NVFP4_BLOCK_BYTES + 4, 32);
        memcpy(sc + i * 4, w + i * OC_NVFP4_BLOCK_BYTES, 4);
    }
}

float oc_nvfp4_global_scale(const float *x, size_t n)
{
    float amax = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    return amax > 0.0f ? amax / (6.0f * 448.0f) : 1.0f;
}

static uint8_t e2m1_nearest(float v)
{
    static const float mag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    float a = fabsf(v);
    int best = 0;
    float be = a;
    for (int i = 1; i < 8; i++) {
        float e = fabsf(a - mag[i]);
        if (e < be) { be = e; best = i; }
    }
    return (uint8_t)(best | (v < 0.0f && best ? 8 : 0));
}

void oc_nvfp4_quantize_row(const float *x, size_t n, float gscale, uint8_t *out)
{
    init_tables();
    for (size_t b = 0; b < n / OC_NVFP4_BLOCK; b++) {
        const float *xb = x + b * OC_NVFP4_BLOCK;
        uint8_t *blk = out + b * OC_NVFP4_BLOCK_BYTES;
        for (size_t s = 0; s < 4; s++) {
            const float *xs = xb + s * 16;
            float amax = 0.0f;
            for (size_t j = 0; j < 16; j++) {
                float a = fabsf(xs[j]);
                if (a > amax) amax = a;
            }
            uint8_t best_code = 0;
            uint8_t best_q[16] = { 0 };
            float best_err = INFINITY;
            if (amax > 0.0f) {
                uint8_t c0 = oc_nvfp4_ue4m3_encode(amax / (6.0f * gscale));
                for (int dc = -2; dc <= 2; dc++) {
                    int c = (int)c0 + dc;
                    if (c < 1 || c > 0x7E) continue;
                    float d = oc_nvfp4_ue4m3((uint8_t)c) * gscale;
                    float id = 1.0f / d, err = 0.0f;
                    uint8_t q[16];
                    for (size_t j = 0; j < 16; j++) {
                        q[j] = e2m1_nearest(xs[j] * id);
                        float r = xs[j] - d * 0.5f * (float)E2M1_X2[q[j]];
                        err += r * r;
                    }
                    if (err < best_err) {
                        best_err = err;
                        best_code = (uint8_t)c;
                        memcpy(best_q, q, sizeof q);
                    }
                }
            }
            blk[s] = best_code;
            for (size_t j = 0; j < 8; j++)
                blk[4 + s * 8 + j] = (uint8_t)(best_q[j] | (best_q[j + 8] << 4));
        }
    }
}
