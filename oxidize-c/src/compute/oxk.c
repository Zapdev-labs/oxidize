/*
 * oxk.c — OXK (Oxidize Kernels) scalar reference implementations + dispatcher.
 *
 * All SIMD variants (AVX2/AVX-512) currently forward to these scalar
 * implementations for correctness. They can be optimized later.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/oxk.h"

#include <string.h>
#include <threads.h>

/* ─── f16 → f32 (bit-twiddle, no libm) ──────────────────────────────────── */

float oc_oxk_f16_le_to_f32(const uint8_t p[2])
{
    uint16_t h = (uint16_t)(p[0] | (p[1] << 8));
    uint32_t sign = (uint32_t)(h >> 15);
    uint32_t exp  = (uint32_t)((h >> 10) & 0x1F);
    uint32_t mant = (uint32_t)(h & 0x3FF);

    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            /* Subnormal: normalize. Each left shift halves the effective
             * exponent, so it decreases as we normalize. */
            uint32_t e = 0;
            while ((mant & 0x400) == 0) { mant <<= 1; e++; }
            mant &= 0x3FF;
            exp = 127 - 15 + 1 - e;
            bits = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        uint32_t new_exp = exp - 15 + 127;
        bits = (sign << 31) | (new_exp << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* ─── Q4_K scale/min extraction ─────────────────────────────────────────── */

void oc_oxk_get_scale_min_k4(unsigned j, const uint8_t scales[12],
                              uint8_t *scale, uint8_t *min)
{
    if (j < 4) {
        *scale = scales[j] & 63;
        *min   = scales[j + 4] & 63;
    } else {
        *scale = (scales[j - 4] >> 6) | ((scales[j] << 2) & 0x3C);
        *min   = (scales[j] >> 6) | ((scales[j + 4] << 2) & 0x3C);
    }
}

/* ─── Q8_K bsum read ────────────────────────────────────────────────────── */

int16_t oc_oxk_read_q8_k_bsum(const uint8_t *bsums, size_t index)
{
    /* 16 i16 values packed at offset 260 of Q8_K block (bsums start). */
    const uint8_t *p = bsums + index * 2;
    return (int16_t)(p[0] | (p[1] << 8));
}

/* ─── Scalar dot products ──────────────────────────────────────────────── */

float oc_oxk_dot_q4_0_q8_0_scalar(const uint8_t *row, size_t blocks,
                                   const uint8_t *q8)
{
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_0_SIZE; /* f16 d + 16 packed bytes */
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;  /* f16 d + 32 int8 */
        float dw = oc_oxk_f16_le_to_f32(wb);
        float dq = oc_oxk_f16_le_to_f32(qb);
        const uint8_t *qs = wb + 2; /* 16 packed bytes → 32 nibbles */
        const int8_t  *qv = (const int8_t *)(qb + 2); /* 32 int8 values */
        int32_t isum = 0;
        for (int i = 0; i < 16; i++) {
            int lo = qs[i] & 0x0F;
            int hi = qs[i] >> 4;
            isum += (lo - 8) * (int)qv[i * 2];
            isum += (hi - 8) * (int)qv[i * 2 + 1];
        }
        sum += dw * dq * (float)isum;
    }
    return sum;
}

float oc_oxk_dot_q4_1_q8_0_scalar(const uint8_t *row, size_t blocks,
                                   const uint8_t *q8)
{
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_1_SIZE; /* f16 d + f16 m + 16 packed */
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;
        float dw  = oc_oxk_f16_le_to_f32(wb);
        float mw  = oc_oxk_f16_le_to_f32(wb + 2);
        float dq  = oc_oxk_f16_le_to_f32(qb);
        const uint8_t *qs = wb + 4; /* 16 packed bytes → 32 nibbles */
        const int8_t  *qv = (const int8_t *)(qb + 2);
        int32_t dot_prod = 0;
        int32_t q8_sum   = 0;
        for (int i = 0; i < 16; i++) {
            int lo = qs[i] & 0x0F;
            int hi = qs[i] >> 4;
            dot_prod += lo * (int)qv[i * 2] + hi * (int)qv[i * 2 + 1];
            q8_sum   += (int)qv[i * 2] + (int)qv[i * 2 + 1];
        }
        sum += dw * dq * (float)dot_prod + mw * dq * (float)q8_sum;
    }
    return sum;
}

float oc_oxk_dot_q8_0_q8_0_scalar(const uint8_t *row, size_t blocks,
                                   const uint8_t *q8)
{
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q8_0_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_0_SIZE;
        float dw = oc_oxk_f16_le_to_f32(wb);
        float dq = oc_oxk_f16_le_to_f32(qb);
        const int8_t *wv = (const int8_t *)(wb + 2);
        const int8_t *qv = (const int8_t *)(qb + 2);
        int32_t isum = 0;
        for (int i = 0; i < 32; i++)
            isum += (int)wv[i] * (int)qv[i];
        sum += dw * dq * (float)isum;
    }
    return sum;
}

float oc_oxk_dot_q4_k_q8_k_scalar(const uint8_t *row, size_t blocks,
                                   const uint8_t *q8)
{
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_K_SIZE;  /* 144 bytes */
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_K_SIZE;  /* 292 bytes */
        float dw    = oc_oxk_f16_le_to_f32(wb);
        float dmin  = oc_oxk_f16_le_to_f32(wb + 2);
        const uint8_t *scales = wb + 4;  /* 12 bytes */
        const uint8_t *qs     = wb + 16; /* 128 bytes: 256 nibbles */
        /* Q8_K: [f32 d (4)][256 int8 (256)][16 i16 bsums (32)] */
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v    = (const int8_t *)(qb + 4);
        const uint8_t *bsums  = qb + 4 + 256; /* 16 i16 = 32 bytes */

        for (int gp = 0; gp < 4; gp++) {
            uint8_t sc1, m1, sc2, m2;
            oc_oxk_get_scale_min_k4(gp * 2,     scales, &sc1, &m1);
            oc_oxk_get_scale_min_k4(gp * 2 + 1, scales, &sc2, &m2);

            int32_t sum1 = 0, sum2 = 0;
            for (int l = 0; l < 32; l++) {
                uint8_t byte = qs[gp * 32 + l];
                sum1 += (byte & 0x0F) * (int)q8v[gp * 64 + l];
                sum2 += (byte >> 4)   * (int)q8v[gp * 64 + 32 + l];
            }

            int32_t bs1 = oc_oxk_read_q8_k_bsum(bsums, gp * 4) +
                          oc_oxk_read_q8_k_bsum(bsums, gp * 4 + 1);
            int32_t bs2 = oc_oxk_read_q8_k_bsum(bsums, gp * 4 + 2) +
                          oc_oxk_read_q8_k_bsum(bsums, gp * 4 + 3);

            float term1 = dw * dq * (float)sc1 * (float)sum1 - dw * dmin * dq * (float)m1 * (float)bs1;
            float term2 = dw * dq * (float)sc2 * (float)sum2 - dw * dmin * dq * (float)m2 * (float)bs2;
            sum += term1 + term2;
        }
    }
    return sum;
}

float oc_oxk_dot_q5_k_q8_k_scalar(const uint8_t *row, size_t blocks,
                                   const uint8_t *q8)
{
    /* Q5_K block (176 bytes): [f16 d (2)][f16 dmin (2)][12 scales (12)]
     *                          [32 qh bytes (32)][128 qs bytes (128)] */
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q5_K_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float dw   = oc_oxk_f16_le_to_f32(wb);
        float dmin = oc_oxk_f16_le_to_f32(wb + 2);
        const uint8_t *scales = wb + 4;
        const uint8_t *qh     = wb + 16;  /* 32 bytes: 256 high bits */
        const uint8_t *qs     = wb + 48;  /* 128 bytes: 256 nibbles */
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + 256;

        for (int gp = 0; gp < 4; gp++) {
            uint8_t sc1, m1, sc2, m2;
            oc_oxk_get_scale_min_k4(gp * 2,     scales, &sc1, &m1);
            oc_oxk_get_scale_min_k4(gp * 2 + 1, scales, &sc2, &m2);

            int32_t sum1 = 0, sum2 = 0;
            for (int l = 0; l < 32; l++) {
                uint8_t byte = qs[gp * 32 + l];
                int lo = (byte & 0x0F) + (((qh[(gp * 64 + l) / 8] >> ((gp * 64 + l) % 8)) & 1) << 4);
                int hi = (byte >> 4)   + (((qh[(gp * 64 + 32 + l) / 8] >> ((gp * 64 + 32 + l) % 8)) & 1) << 4);
                sum1 += lo * (int)q8v[gp * 64 + l];
                sum2 += hi * (int)q8v[gp * 64 + 32 + l];
            }

            int32_t bs1 = oc_oxk_read_q8_k_bsum(bsums, gp * 4) +
                          oc_oxk_read_q8_k_bsum(bsums, gp * 4 + 1);
            int32_t bs2 = oc_oxk_read_q8_k_bsum(bsums, gp * 4 + 2) +
                          oc_oxk_read_q8_k_bsum(bsums, gp * 4 + 3);

            float term1 = dw * dq * (float)sc1 * (float)sum1 - dw * dmin * dq * (float)m1 * (float)bs1;
            float term2 = dw * dq * (float)sc2 * (float)sum2 - dw * dmin * dq * (float)m2 * (float)bs2;
            sum += term1 + term2;
        }
    }
    return sum;
}

float oc_oxk_dot_q6_k_q8_k_scalar(const uint8_t *row, size_t blocks,
                                   const uint8_t *q8)
{
    /* Q6_K block (210 bytes): [128 ql bytes][64 qh bytes][16 int8 scales][f16 d] */
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q6_K_SIZE;
        const uint8_t *qb = q8  + b * OC_OXK_BLOCK_Q8_K_SIZE;
        const uint8_t *ql = wb;         /* 128 bytes */
        const uint8_t *qh = wb + 128;   /* 64 bytes */
        const int8_t  *sc = (const int8_t *)(wb + 192); /* 16 int8 scales */
        float dw = oc_oxk_f16_le_to_f32(wb + 208);  /* f16 d at offset 208 */

        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        /* bsums not needed for Q6_K dot (we compute per-element). */

        for (int n = 0; n < 2; n++) {
            const uint8_t *ql_chunk = ql + n * 64;
            const uint8_t *qh_chunk = qh + n * 32;
            const int8_t  *sc_chunk = sc + n * 8;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = ((ql_chunk[l] & 0x0F) | (((qh_chunk[l] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql_chunk[l + 32] & 0x0F) | (((qh_chunk[l] >> 2) & 3) << 4)) - 32;
                int q3 = ((ql_chunk[l] >> 4) | (((qh_chunk[l] >> 4) & 3) << 4)) - 32;
                int q4 = ((ql_chunk[l + 32] >> 4) | (((qh_chunk[l] >> 6) & 3) << 4)) - 32;

                int base = n * 128 + l;
                int g1 = base;
                int g2 = base + 32;
                int g3 = base + 64;
                int g4 = base + 96;

                sum += dw * dq * (float)sc_chunk[is + 0] * (float)(q1 * (int)q8v[g1]);
                sum += dw * dq * (float)sc_chunk[is + 2] * (float)(q2 * (int)q8v[g2]);
                sum += dw * dq * (float)sc_chunk[is + 4] * (float)(q3 * (int)q8v[g3]);
                sum += dw * dq * (float)sc_chunk[is + 6] * (float)(q4 * (int)q8v[g4]);
            }
        }
    }
    return sum;
}

/* ─── Scalar matvec ────────────────────────────────────────────────────── */

void oc_oxk_matvec_q4_0_f32_scalar(const uint8_t *w, size_t n_rows,
                                    size_t row_bytes, const float *x, float *out)
{
    size_t blocks = row_bytes / OC_OXK_BLOCK_Q4_0_SIZE;
    for (size_t r = 0; r < n_rows; r++) {
        /* Dequantize weight row and dot with f32 input. */
        const uint8_t *row = w + r * row_bytes;
        float sum = 0.0f;
        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_0_SIZE;
            float dw = oc_oxk_f16_le_to_f32(wb);
            const uint8_t *qs = wb + 2;
            for (int i = 0; i < 16; i++) {
                int lo = (qs[i] & 0x0F) - 8;
                int hi = (qs[i] >> 4) - 8;
                sum += dw * (float)lo * x[b * 32 + i * 2];
                sum += dw * (float)hi * x[b * 32 + i * 2 + 1];
            }
        }
        out[r] = sum;
    }
}

void oc_oxk_matvec_q4_k_f32_scalar(const uint8_t *w, size_t n_rows,
                                    size_t row_bytes, const float *x, float *out)
{
    size_t blocks = row_bytes / OC_OXK_BLOCK_Q4_K_SIZE;
    for (size_t r = 0; r < n_rows; r++) {
        const uint8_t *row = w + r * row_bytes;
        float sum = 0.0f;
        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_K_SIZE;
            float dw = oc_oxk_f16_le_to_f32(wb);
            float dmin = oc_oxk_f16_le_to_f32(wb + 2);
            const uint8_t *scales = wb + 4;
            const uint8_t *qs = wb + 16;
            for (int gp = 0; gp < 4; gp++) {
                uint8_t sc1, m1, sc2, m2;
                oc_oxk_get_scale_min_k4(gp * 2,     scales, &sc1, &m1);
                oc_oxk_get_scale_min_k4(gp * 2 + 1, scales, &sc2, &m2);
                for (int l = 0; l < 32; l++) {
                    uint8_t byte = qs[gp * 32 + l];
                    int lo = byte & 0x0F;
                    int hi = byte >> 4;
                    float w1 = dw * (float)sc1 * (float)lo - dw * dmin * (float)m1;
                    float w2 = dw * (float)sc2 * (float)hi - dw * dmin * (float)m2;
                    sum += w1 * x[b * 256 + gp * 64 + l];
                    sum += w2 * x[b * 256 + gp * 64 + 32 + l];
                }
            }
        }
        out[r] = sum;
    }
}

void oc_oxk_matvec_q8_0_f32_scalar(const uint8_t *w, size_t n_rows,
                                    size_t row_bytes, const float *x, float *out)
{
    size_t blocks = row_bytes / OC_OXK_BLOCK_Q8_0_SIZE;
    for (size_t r = 0; r < n_rows; r++) {
        const uint8_t *row = w + r * row_bytes;
        float sum = 0.0f;
        for (size_t b = 0; b < blocks; b++) {
            const uint8_t *wb = row + b * OC_OXK_BLOCK_Q8_0_SIZE;
            float dw = oc_oxk_f16_le_to_f32(wb);
            const int8_t *wv = (const int8_t *)(wb + 2);
            for (int i = 0; i < 32; i++)
                sum += dw * (float)wv[i] * x[b * 32 + i];
        }
        out[r] = sum;
    }
}

/* AVX2 / AVX-512 implementations are in oxk_avx2.c */

/* ─── Capability detection + dispatcher ──────────────────────────────────── */

static OcOxkContext g_ctx;
static once_flag g_once = ONCE_FLAG_INIT;

static void oc_oxk_init_once(void)
{
    OcOxkLevel level = OC_OXK_SCALAR;
    bool has_f16c = false, has_fma = false, has_vnni = false;
    const char *name = "scalar";

#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq")) {
        level = OC_OXK_AVX512;
        has_f16c = true;
        has_fma  = true;
        has_vnni = __builtin_cpu_supports("avx512vnni");
        name = "avx512";
    } else if (__builtin_cpu_supports("avx2")) {
        level = OC_OXK_AVX2;
        has_f16c = __builtin_cpu_supports("f16c");
        has_fma  = __builtin_cpu_supports("fma");
        name = "avx2";
    }
#endif

    g_ctx.caps.level    = level;
    g_ctx.caps.has_f16c = has_f16c;
    g_ctx.caps.has_fma  = has_fma;
    g_ctx.caps.has_vnni = has_vnni;
    g_ctx.caps.name     = name;

    /* Dispatch table — all forward to scalar for now. */
    g_ctx.dot_q4_0_q8_0 = oc_oxk_dot_q4_0_q8_0_scalar;
    g_ctx.dot_q4_1_q8_0 = oc_oxk_dot_q4_1_q8_0_scalar;
    g_ctx.dot_q4_k_q8_k = oc_oxk_dot_q4_k_q8_k_scalar;
    g_ctx.dot_q5_k_q8_k = oc_oxk_dot_q5_k_q8_k_scalar;
    g_ctx.dot_q6_k_q8_k = oc_oxk_dot_q6_k_q8_k_scalar;
    g_ctx.dot_q8_0_q8_0 = oc_oxk_dot_q8_0_q8_0_scalar;

    g_ctx.matvec_q4_0_f32 = oc_oxk_matvec_q4_0_f32_scalar;
    g_ctx.matvec_q4_k_f32 = oc_oxk_matvec_q4_k_f32_scalar;
    g_ctx.matvec_q8_0_f32 = oc_oxk_matvec_q8_0_f32_scalar;
}

const OcOxkContext *oc_oxk_init(void)
{
    call_once(&g_once, oc_oxk_init_once);
    return &g_ctx;
}

const OcOxkCaps *oc_oxk_caps(void)
{
    oc_oxk_init();
    return &g_ctx.caps;
}

/* ─── Dispatched entry points ───────────────────────────────────────────── */

float oc_oxk_dot_q4_0_q8_0(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ oc_oxk_init(); return g_ctx.dot_q4_0_q8_0(row, blocks, q8); }

float oc_oxk_dot_q4_1_q8_0(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ oc_oxk_init(); return g_ctx.dot_q4_1_q8_0(row, blocks, q8); }

float oc_oxk_dot_q4_k_q8_k(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ oc_oxk_init(); return g_ctx.dot_q4_k_q8_k(row, blocks, q8); }

float oc_oxk_dot_q5_k_q8_k(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ oc_oxk_init(); return g_ctx.dot_q5_k_q8_k(row, blocks, q8); }

float oc_oxk_dot_q6_k_q8_k(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ oc_oxk_init(); return g_ctx.dot_q6_k_q8_k(row, blocks, q8); }

float oc_oxk_dot_q8_0_q8_0(const uint8_t *row, size_t blocks, const uint8_t *q8)
{ oc_oxk_init(); return g_ctx.dot_q8_0_q8_0(row, blocks, q8); }

void oc_oxk_matvec_q4_0_f32(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_init(); g_ctx.matvec_q4_0_f32(w, n_rows, row_bytes, x, out); }

void oc_oxk_matvec_q4_k_f32(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_init(); g_ctx.matvec_q4_k_f32(w, n_rows, row_bytes, x, out); }

void oc_oxk_matvec_q8_0_f32(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_init(); g_ctx.matvec_q8_0_f32(w, n_rows, row_bytes, x, out); }
