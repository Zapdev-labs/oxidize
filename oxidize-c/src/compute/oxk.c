/* oxk.c — OXK (Oxidize Kernels) scalar reference implementations + dispatcher. */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/oxk.h"
#include "oxidize/log.h"
#include "oxidize/oxk_avx512.h"
#include "oxidize/oxk_neon.h"
#include "oxidize/simd.h"

#include <string.h>
/* pthread_once, not C11 <threads.h>: Apple's libc does not ship <threads.h>,
 * so call_once/once_flag break the macOS build. pthread is already linked. */
#include <pthread.h>


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


void oc_oxk_get_scale_min_k4(unsigned j, const uint8_t scales[12],
                              uint8_t *scale, uint8_t *min)
{
    /* Must match quantization.c's get_scale_min_k4, which is bit-identical to ggml's. */
    if (j < 4) {
        *scale = scales[j] & 63;
        *min   = scales[j + 4] & 63;
    } else {
        *scale = (uint8_t)((scales[j + 4] & 0x0Fu) | ((scales[j - 4] >> 6) << 4));
        *min   = (uint8_t)(((scales[j + 4] >> 4) & 0x0Fu) | ((scales[j] >> 6) << 4));
    }
}


int16_t oc_oxk_read_q8_k_bsum(const uint8_t *bsums, size_t index)
{
    /* 16 i16 values packed at offset 260 of Q8_K block (bsums start). */
    const uint8_t *p = bsums + index * 2;
    return (int16_t)(p[0] | (p[1] << 8));
}


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
        /* GGUF Q4_0 splits a block into halves: the low nibble of byte i is */
        for (int i = 0; i < 16; i++) {
            int lo = qs[i] & 0x0F;
            int hi = qs[i] >> 4;
            isum += (lo - 8) * (int)qv[i];
            isum += (hi - 8) * (int)qv[i + 16];
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
        /* Same half-split layout as Q4_0 — see the note there. */
        for (int i = 0; i < 16; i++) {
            int lo = qs[i] & 0x0F;
            int hi = qs[i] >> 4;
            dot_prod += lo * (int)qv[i] + hi * (int)qv[i + 16];
            q8_sum   += (int)qv[i] + (int)qv[i + 16];
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

        int32_t pos = 0, min_acc = 0;
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

            /* Accumulate the scaled sums in int32 across the whole block and what lets a vectorized kernel be bit-exact against this one: */
            pos     += (int32_t)sc1 * sum1 + (int32_t)sc2 * sum2;
            min_acc += (int32_t)m1  * bs1  + (int32_t)m2  * bs2;
        }

        /* The offset term is scaled by dmin alone, NOT by dw*dmin: the dequantized weight is d*sc*q - dmin*m, so the minimum carries its own scale. */
        sum += dw * dq * (float)pos - dmin * dq * (float)min_acc;
    }
    return sum;
}

#define OC_Q4K_SUBGROUPS 8u

size_t oc_oxk_q4_k_prep_bytes(size_t blocks)
{
    return blocks * (2u * sizeof(float) + OC_OXK_QK_K + 2u * OC_Q4K_SUBGROUPS);
}

static void q4k_prep_ptrs(void *scratch, size_t blocks, float **d, float **dmin,
                          uint8_t **codes, uint8_t **sc, uint8_t **mn)
{
    float *f = (float *)scratch;
    *d    = f;
    *dmin = f + blocks;
    uint8_t *u = (uint8_t *)(f + 2 * blocks);
    *codes = u;
    *sc    = u + blocks * OC_OXK_QK_K;
    *mn    = *sc + blocks * OC_Q4K_SUBGROUPS;
}

void oc_oxk_q4_k_prep_row(const uint8_t *row, size_t blocks, void *scratch)
{
    float *d, *dmin;
    uint8_t *codes, *sc, *mn;
    q4k_prep_ptrs(scratch, blocks, &d, &dmin, &codes, &sc, &mn);

    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q4_K_SIZE;
        d[b]    = oc_oxk_f16_le_to_f32(wb);
        dmin[b] = oc_oxk_f16_le_to_f32(wb + 2);
        const uint8_t *scales = wb + 4;
        const uint8_t *qs     = wb + 16;

        for (unsigned j = 0; j < OC_Q4K_SUBGROUPS; j++) {
            oc_oxk_get_scale_min_k4(j, scales, &sc[b * OC_Q4K_SUBGROUPS + j],
                                    &mn[b * OC_Q4K_SUBGROUPS + j]);
        }

        /* Nibble order: within group gp, the low nibble of qs[gp*32+l] is */
        uint8_t *c = codes + b * OC_OXK_QK_K;
        for (unsigned gp = 0; gp < 4; gp++) {
            const uint8_t *src = qs + gp * 32;
            uint8_t *lo = c + gp * 64;
            uint8_t *hi = lo + 32;
            for (unsigned l = 0; l < 32; l++) {
                lo[l] = (uint8_t)(src[l] & 0x0Fu);
                hi[l] = (uint8_t)(src[l] >> 4);
            }
        }
    }
}

/* Q5_K prep into the SAME layout as Q4_K: codes are the 5-bit values oc_oxk_dot_q4_k_prepped()/_multi() kernels, bit-exact against */
void oc_oxk_q5_k_prep_row(const uint8_t *row, size_t blocks, void *scratch)
{
    float *d, *dmin;
    uint8_t *codes, *sc, *mn;
    q4k_prep_ptrs(scratch, blocks, &d, &dmin, &codes, &sc, &mn);

    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q5_K_SIZE;
        d[b]    = oc_oxk_f16_le_to_f32(wb);
        dmin[b] = oc_oxk_f16_le_to_f32(wb + 2);
        const uint8_t *scales = wb + 4;
        const uint8_t *qh     = wb + 16;
        const uint8_t *qs     = wb + 48;

        for (unsigned j = 0; j < OC_Q4K_SUBGROUPS; j++) {
            oc_oxk_get_scale_min_k4(j, scales, &sc[b * OC_Q4K_SUBGROUPS + j],
                                    &mn[b * OC_Q4K_SUBGROUPS + j]);
        }

        /* qh[l] carries one high bit per 64-element group: bit 2*gp for the
         * low-nibble half, bit 2*gp+1 for the high-nibble half. */
        uint8_t *c = codes + b * OC_OXK_QK_K;
        for (unsigned gp = 0; gp < 4; gp++) {
            const uint8_t *src = qs + gp * 32;
            uint8_t *lo = c + gp * 64;
            uint8_t *hi = lo + 32;
            for (unsigned l = 0; l < 32; l++) {
                lo[l] = (uint8_t)((src[l] & 0x0Fu) |
                                  (((qh[l] >> (2 * gp)) & 1u) << 4));
                hi[l] = (uint8_t)((src[l] >> 4) |
                                  (((qh[l] >> (2 * gp + 1)) & 1u) << 4));
            }
        }
    }
}

size_t oc_oxk_q6_k_prep_bytes(size_t blocks)
{
    return blocks * (sizeof(float) + OC_OXK_QK_K + 16u);
}

static void q6k_prep_ptrs(void *scratch, size_t blocks, float **d,
                          uint8_t **codes, int8_t **sc)
{
    float *f = (float *)scratch;
    *d     = f;
    *codes = (uint8_t *)(f + blocks);
    *sc    = (int8_t *)(*codes + blocks * OC_OXK_QK_K);
}

void oc_oxk_q6_k_prep_row(const uint8_t *row, size_t blocks, void *scratch)
{
    float *d;
    uint8_t *codes;
    int8_t *sc;
    q6k_prep_ptrs(scratch, blocks, &d, &codes, &sc);

    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *wb = row + b * OC_OXK_BLOCK_Q6_K_SIZE;
        const uint8_t *ql = wb;
        const uint8_t *qh = wb + 128;
        memcpy(sc + b * 16, wb + 192, 16);
        d[b] = oc_oxk_f16_le_to_f32(wb + 208);

        uint8_t *c = codes + b * OC_OXK_QK_K;
        for (int n = 0; n < 2; n++) {
            const uint8_t *ql_chunk = ql + n * 64;
            const uint8_t *qh_chunk = qh + n * 32;
            uint8_t *base = c + n * 128;
            for (int l = 0; l < 32; l++) {
                base[l]      = (uint8_t)((ql_chunk[l] & 0x0F) |
                                         (((qh_chunk[l] >> 0) & 3) << 4));
                base[l + 32] = (uint8_t)((ql_chunk[l + 32] & 0x0F) |
                                         (((qh_chunk[l] >> 2) & 3) << 4));
                base[l + 64] = (uint8_t)((ql_chunk[l] >> 4) |
                                         (((qh_chunk[l] >> 4) & 3) << 4));
                base[l + 96] = (uint8_t)((ql_chunk[l + 32] >> 4) |
                                         (((qh_chunk[l] >> 6) & 3) << 4));
            }
        }
    }
}

float oc_oxk_dot_q6_k_prepped(const void *scratch, size_t blocks,
                              const uint8_t *q8)
{
    float *d;
    uint8_t *codes;
    int8_t *sc;
    q6k_prep_ptrs((void *)(uintptr_t)scratch, blocks, &d, &codes, &sc);

    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *qb = q8 + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + OC_OXK_QK_K;
        const uint8_t *c     = codes + b * OC_OXK_QK_K;
        const int8_t  *scb   = sc + b * 16;

        int32_t pos = 0, minc = 0;
        for (unsigned g = 0; g < 16; g++) {
            const uint8_t *cg = c + g * 16;
            const int8_t  *qg = q8v + g * 16;
            int32_t s = 0;
            for (unsigned l = 0; l < 16; l++)
                s += (int32_t)cg[l] * (int32_t)qg[l];
            pos  += (int32_t)scb[g] * s;
            minc += (int32_t)scb[g] *
                    (int32_t)oc_oxk_read_q8_k_bsum(bsums, g);
        }
        sum += d[b] * dq * (float)(pos - 32 * minc);
    }
    return sum;
}


/* Q2_K block: scales[16] (4-bit scale | 4-bit min), qs[64], f16 d, f16 dmin.
 * codes are the raw 2-bit values 0..3; the group minimum is returned
 * separately and folded out through the activation block sums. */
static void q2k_unpack_block(const uint8_t *wb, uint8_t *codes, uint8_t *sc,
                             uint8_t *mn, float *d, float *dmin)
{
    const uint8_t *scales = wb;
    const uint8_t *qs     = wb + 16;
    *d    = oc_oxk_f16_le_to_f32(wb + 80);
    *dmin = oc_oxk_f16_le_to_f32(wb + 82);
    for (unsigned g = 0; g < 16; g++) {
        sc[g] = (uint8_t)(scales[g] & 0x0Fu);
        mn[g] = (uint8_t)(scales[g] >> 4);
    }
    /* Groups 2k and 2k+1 are adjacent in `codes` and their source bytes are */
    for (unsigned outer = 0; outer < 2; outer++) {
        for (unsigned inner = 0; inner < 4; inner++) {
            const uint8_t *q = qs + outer * 32;
            uint8_t *c = codes + (outer * 8 + inner * 2) * 16;
            const unsigned shift = inner * 2;
            for (unsigned l = 0; l < 32; l++)
                c[l] = (uint8_t)((q[l] >> shift) & 3u);
        }
    }
}

/* Q3_K block: hmask[32], qs[64], 12 packed 6-bit scales, f16 d. */
static void q3k_unpack_block(const uint8_t *wb, uint8_t *codes, int8_t *sc,
                             float *d)
{
    const uint8_t *hmask = wb;
    const uint8_t *qs    = wb + 32;
    *d = oc_oxk_f16_le_to_f32(wb + 108);

    /* 6-bit scales: 16 values packed into 12 bytes, low 4 bits in the first
     * 8 bytes and the high 2 bits spread across the last 4. Same shuffle as
     * dequant_q3_k. */
    uint32_t aux[4];
    for (unsigned i = 0; i < 3; i++) {
        const uint8_t *p = wb + 96 + i * 4;
        aux[i] = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                 ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    }
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 4) & 0x03030303u) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 6) & 0x03030303u) << 4);
    aux[0] = (aux[0] & 0x0F0F0F0Fu) | ((tmp & 0x03030303u) << 4);
    aux[1] = (aux[1] & 0x0F0F0F0Fu) | (((tmp >> 2) & 0x03030303u) << 4);
    for (unsigned i = 0; i < 4; i++) {
        for (unsigned j = 0; j < 4; j++)
            sc[i * 4 + j] = (int8_t)(int32_t)
                (((aux[i] >> (j * 8)) & 0xFFu) - 32u);
    }

    /* Same contiguous 32-byte decode as Q2_K (see the note there), with the
     * third bit pulled out of hmask arithmetically rather than with a branch
     * so the loop stays vectorizable. */
    for (unsigned outer = 0; outer < 2; outer++) {
        for (unsigned inner = 0; inner < 4; inner++) {
            const uint8_t *q = qs + outer * 32;
            uint8_t *c = codes + (outer * 8 + inner * 2) * 16;
            const unsigned shift = inner * 2;
            const unsigned bit = outer * 4 + inner;
            for (unsigned l = 0; l < 32; l++) {
                c[l] = (uint8_t)(((q[l] >> shift) & 3u) |
                                 (((hmask[l] >> bit) & 1u) << 2));
            }
        }
    }
}

float oc_oxk_dot_q2_k_q8_k_scalar(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        uint8_t codes[OC_OXK_QK_K], sc[16], mn[16];
        float dw, dmin;
        q2k_unpack_block(row + b * OC_OXK_BLOCK_Q2_K_SIZE, codes, sc, mn,
                         &dw, &dmin);

        const uint8_t *qb = q8 + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + OC_OXK_QK_K;

        int32_t pos = 0, min_acc = 0;
        for (unsigned g = 0; g < 16; g++) {
            const uint8_t *cg = codes + g * 16;
            const int8_t  *qg = q8v + g * 16;
            int32_t acc = 0;
            for (unsigned l = 0; l < 16; l++)
                acc += (int32_t)cg[l] * (int32_t)qg[l];
            pos     += (int32_t)sc[g] * acc;
            min_acc += (int32_t)mn[g] *
                       (int32_t)oc_oxk_read_q8_k_bsum(bsums, g);
        }
        sum += dq * (dw * (float)pos - dmin * (float)min_acc);
    }
    return sum;
}

float oc_oxk_dot_q3_k_q8_k_scalar(const uint8_t *row, size_t blocks,
                                  const uint8_t *q8)
{
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        uint8_t codes[OC_OXK_QK_K];
        int8_t sc[16];
        float dw;
        q3k_unpack_block(row + b * OC_OXK_BLOCK_Q3_K_SIZE, codes, sc, &dw);

        const uint8_t *qb = q8 + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + OC_OXK_QK_K;

        int32_t pos = 0, off = 0;
        for (unsigned g = 0; g < 16; g++) {
            const uint8_t *cg = codes + g * 16;
            const int8_t  *qg = q8v + g * 16;
            int32_t acc = 0;
            for (unsigned l = 0; l < 16; l++)
                acc += (int32_t)cg[l] * (int32_t)qg[l];
            pos += (int32_t)sc[g] * acc;
            off += (int32_t)sc[g] *
                   (int32_t)oc_oxk_read_q8_k_bsum(bsums, g);
        }
        sum += dw * dq * (float)(pos - 4 * off);
    }
    return sum;
}

/* Q2_K prepared row: d[blocks], dmin[blocks], codes[blocks*256],
 * sc[blocks*16], mn[blocks*16]. */
size_t oc_oxk_q2_k_prep_bytes(size_t blocks)
{
    return blocks * (2u * sizeof(float) + OC_OXK_QK_K + 32u);
}

static void q2k_prep_ptrs(void *scratch, size_t blocks, float **d, float **dmin,
                          uint8_t **codes, uint8_t **sc, uint8_t **mn)
{
    float *f = (float *)scratch;
    *d     = f;
    *dmin  = f + blocks;
    *codes = (uint8_t *)(f + 2 * blocks);
    *sc    = *codes + blocks * OC_OXK_QK_K;
    *mn    = *sc + blocks * 16u;
}

void oc_oxk_q2_k_prep_row(const uint8_t *row, size_t blocks, void *scratch)
{
    float *d, *dmin;
    uint8_t *codes, *sc, *mn;
    q2k_prep_ptrs(scratch, blocks, &d, &dmin, &codes, &sc, &mn);
    for (size_t b = 0; b < blocks; b++) {
        q2k_unpack_block(row + b * OC_OXK_BLOCK_Q2_K_SIZE,
                         codes + b * OC_OXK_QK_K, sc + b * 16, mn + b * 16,
                         &d[b], &dmin[b]);
    }
}

/* Q3_K prepared row uses the Q6_K layout verbatim (see oc_oxk_q6_k_prep_row):
 * d[blocks], codes[blocks*256], int8 sc[blocks*16]. */
void oc_oxk_q3_k_prep_row(const uint8_t *row, size_t blocks, void *scratch)
{
    float *f = (float *)scratch;
    uint8_t *codes = (uint8_t *)(f + blocks);
    int8_t  *sc    = (int8_t *)(codes + blocks * OC_OXK_QK_K);
    for (size_t b = 0; b < blocks; b++) {
        q3k_unpack_block(row + b * OC_OXK_BLOCK_Q3_K_SIZE,
                         codes + b * OC_OXK_QK_K, sc + b * 16, &f[b]);
    }
}

float oc_oxk_dot_q2_k_prepped(const void *scratch, size_t blocks,
                              const uint8_t *q8)
{
    float *d, *dmin;
    uint8_t *codes, *sc, *mn;
    q2k_prep_ptrs((void *)(uintptr_t)scratch, blocks, &d, &dmin, &codes, &sc,
                  &mn);

    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *qb = q8 + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + OC_OXK_QK_K;
        const uint8_t *c     = codes + b * OC_OXK_QK_K;
        const uint8_t *scb   = sc + b * 16;
        const uint8_t *mnb   = mn + b * 16;

        int32_t pos = 0, min_acc = 0;
        for (unsigned g = 0; g < 16; g++) {
            const uint8_t *cg = c + g * 16;
            const int8_t  *qg = q8v + g * 16;
            int32_t acc = 0;
            for (unsigned l = 0; l < 16; l++)
                acc += (int32_t)cg[l] * (int32_t)qg[l];
            pos     += (int32_t)scb[g] * acc;
            min_acc += (int32_t)mnb[g] *
                       (int32_t)oc_oxk_read_q8_k_bsum(bsums, g);
        }
        sum += dq * (d[b] * (float)pos - dmin[b] * (float)min_acc);
    }
    return sum;
}

float oc_oxk_dot_q3_k_prepped(const void *scratch, size_t blocks,
                              const uint8_t *q8)
{
    const float   *d     = (const float *)scratch;
    const uint8_t *codes = (const uint8_t *)(d + blocks);
    const int8_t  *sc    = (const int8_t *)(codes + blocks * OC_OXK_QK_K);

    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *qb = q8 + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + OC_OXK_QK_K;
        const uint8_t *c     = codes + b * OC_OXK_QK_K;
        const int8_t  *scb   = sc + b * 16;

        int32_t pos = 0, off = 0;
        for (unsigned g = 0; g < 16; g++) {
            const uint8_t *cg = c + g * 16;
            const int8_t  *qg = q8v + g * 16;
            int32_t acc = 0;
            for (unsigned l = 0; l < 16; l++)
                acc += (int32_t)cg[l] * (int32_t)qg[l];
            pos += (int32_t)scb[g] * acc;
            off += (int32_t)scb[g] *
                   (int32_t)oc_oxk_read_q8_k_bsum(bsums, g);
        }
        sum += d[b] * dq * (float)(pos - 4 * off);
    }
    return sum;
}

static void dot_q2_k_prepped_multi_scalar(const void *scratch, size_t blocks,
                                          const uint8_t *acts,
                                          size_t act_stride, size_t n_act,
                                          float *out)
{
    for (size_t v = 0; v < n_act; v++)
        out[v] = oc_oxk_dot_q2_k_prepped(scratch, blocks,
                                         acts + v * act_stride);
}

static void dot_q3_k_prepped_multi_scalar(const void *scratch, size_t blocks,
                                          const uint8_t *acts,
                                          size_t act_stride, size_t n_act,
                                          float *out)
{
    for (size_t v = 0; v < n_act; v++)
        out[v] = oc_oxk_dot_q3_k_prepped(scratch, blocks,
                                         acts + v * act_stride);
}

float oc_oxk_dot_q4_k_prepped(const void *scratch, size_t blocks,
                              const uint8_t *q8)
{
    float *d, *dmin;
    uint8_t *codes, *sc, *mn;
    q4k_prep_ptrs((void *)(uintptr_t)scratch, blocks, &d, &dmin, &codes, &sc,
                  &mn);

    float sum = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t *qb = q8 + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float dq;
        memcpy(&dq, qb, 4);
        const int8_t  *q8v   = (const int8_t *)(qb + 4);
        const uint8_t *bsums = qb + 4 + OC_OXK_QK_K;
        const uint8_t *c     = codes + b * OC_OXK_QK_K;
        const uint8_t *scb   = sc + b * OC_Q4K_SUBGROUPS;
        const uint8_t *mnb   = mn + b * OC_Q4K_SUBGROUPS;

        int32_t pos = 0, min_acc = 0;
        for (unsigned j = 0; j < OC_Q4K_SUBGROUPS; j++) {
            const uint8_t *cj = c + j * 32;
            const int8_t  *qj = q8v + j * 32;
            /* Unsigned code x signed activation, widening to int32 — the
             * exact shape of VNNI's vpdpbusd, and a pattern gcc vectorizes
             * on its own when the build allows it. */
            int32_t s = 0;
            for (unsigned l = 0; l < 32; l++) s += (int32_t)cj[l] * (int32_t)qj[l];
            pos += (int32_t)scb[j] * s;
            min_acc += (int32_t)mnb[j] *
                       ((int32_t)oc_oxk_read_q8_k_bsum(bsums, j * 2) +
                        (int32_t)oc_oxk_read_q8_k_bsum(bsums, j * 2 + 1));
        }
        /* Same per-block float accumulation as the packed kernel, so the
         * two agree bit-for-bit rather than merely closely. */
        sum += d[b] * dq * (float)pos - dmin[b] * dq * (float)min_acc;
    }
    return sum;
}

static void dot_q4_k_prepped_multi_scalar(const void *scratch, size_t blocks,
                                          const uint8_t *acts,
                                          size_t act_stride, size_t n_act,
                                          float *out)
{
    for (size_t v = 0; v < n_act; v++) {
        out[v] = oc_oxk_dot_q4_k_prepped(scratch, blocks,
                                         acts + v * act_stride);
    }
}

static void dot_q6_k_prepped_multi_scalar(const void *scratch, size_t blocks,
                                          const uint8_t *acts,
                                          size_t act_stride, size_t n_act,
                                          float *out)
{
    for (size_t v = 0; v < n_act; v++) {
        out[v] = oc_oxk_dot_q6_k_prepped(scratch, blocks,
                                         acts + v * act_stride);
    }
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

        int32_t pos = 0, min_acc = 0;
        for (int gp = 0; gp < 4; gp++) {
            uint8_t sc1, m1, sc2, m2;
            oc_oxk_get_scale_min_k4(gp * 2,     scales, &sc1, &m1);
            oc_oxk_get_scale_min_k4(gp * 2 + 1, scales, &sc2, &m2);

            int32_t sum1 = 0, sum2 = 0;
            for (int l = 0; l < 32; l++) {
                uint8_t byte = qs[gp * 32 + l];
                int lo = (byte & 0x0F) + (((qh[l] >> (2 * gp))     & 1) << 4);
                int hi = (byte >> 4)   + (((qh[l] >> (2 * gp + 1)) & 1) << 4);
                sum1 += lo * (int)q8v[gp * 64 + l];
                sum2 += hi * (int)q8v[gp * 64 + 32 + l];
            }

            int32_t bs1 = oc_oxk_read_q8_k_bsum(bsums, gp * 4) +
                          oc_oxk_read_q8_k_bsum(bsums, gp * 4 + 1);
            int32_t bs2 = oc_oxk_read_q8_k_bsum(bsums, gp * 4 + 2) +
                          oc_oxk_read_q8_k_bsum(bsums, gp * 4 + 3);

            /* Accumulate the scaled sums in int32 across the whole block and what lets a vectorized kernel be bit-exact against this one: */
            pos     += (int32_t)sc1 * sum1 + (int32_t)sc2 * sum2;
            min_acc += (int32_t)m1  * bs1  + (int32_t)m2  * bs2;
        }

        /* The offset term is scaled by dmin alone, NOT by dw*dmin: the dequantized weight is d*sc*q - dmin*m, so the minimum carries its own scale. */
        sum += dw * dq * (float)pos - dmin * dq * (float)min_acc;
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
        const uint8_t *bsums = qb + 4 + 256;

        /* Integer accumulation per 16-element scale group, one float accumulation so the SIMD kernels can be bit-exact against this */
        int32_t grp[16] = {0};
        for (int n = 0; n < 2; n++) {
            const uint8_t *ql_chunk = ql + n * 64;
            const uint8_t *qh_chunk = qh + n * 32;
            for (int l = 0; l < 32; l++) {
                int q1 = (ql_chunk[l] & 0x0F) | (((qh_chunk[l] >> 0) & 3) << 4);
                int q2 = (ql_chunk[l + 32] & 0x0F) | (((qh_chunk[l] >> 2) & 3) << 4);
                int q3 = (ql_chunk[l] >> 4) | (((qh_chunk[l] >> 4) & 3) << 4);
                int q4 = (ql_chunk[l + 32] >> 4) | (((qh_chunk[l] >> 6) & 3) << 4);

                int base = n * 128 + l;
                grp[base / 16]        += q1 * (int)q8v[base];
                grp[(base + 32) / 16] += q2 * (int)q8v[base + 32];
                grp[(base + 64) / 16] += q3 * (int)q8v[base + 64];
                grp[(base + 96) / 16] += q4 * (int)q8v[base + 96];
            }
        }
        int32_t pos = 0, minc = 0;
        for (int g = 0; g < 16; g++) {
            pos  += (int32_t)sc[g] * grp[g];
            minc += (int32_t)sc[g] *
                    (int32_t)oc_oxk_read_q8_k_bsum(bsums, (size_t)g);
        }
        sum += dw * dq * (float)(pos - 32 * minc);
    }
    return sum;
}


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


static OcOxkContext g_ctx;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static void oc_oxk_init_once(void)
{
    OcOxkLevel level = OC_OXK_SCALAR;
    bool has_f16c = false, has_fma = false, has_vnni = false, has_neon = false;
    const char *name = "scalar";

    /* Reuse core/simd.c's cpuid-based detection: __builtin_cpu_supports
     * rejects some feature strings on older clang and doesn't exist for
     * aarch64 targets. */
    const OcSimdCaps *sc = oc_simd_caps();
    if (sc->level == OC_SIMD_AVX512) {
        level = OC_OXK_AVX512;
        has_f16c = true;
        has_fma  = true;
        has_vnni = sc->has_vnni;
        name = "avx512";
    } else if (sc->level == OC_SIMD_AVX2) {
        level = OC_OXK_AVX2;
        has_f16c = sc->has_f16c;
        has_fma  = sc->has_fma;
        name = "avx2";
    } else if (sc->level == OC_SIMD_NEON) {
        level = OC_OXK_NEON;
        has_neon = true;
        name = "neon";
    }

    g_ctx.caps.level    = level;
    g_ctx.caps.has_f16c = has_f16c;
    g_ctx.caps.has_fma  = has_fma;
    g_ctx.caps.has_vnni = has_vnni;
    g_ctx.caps.has_neon = has_neon;
    g_ctx.caps.name     = name;

    /* Dispatch table. Scalar is the baseline every architecture falls back implement. Each is bit-exact against the scalar reference invariant here, so installing them changes speed and nothing else. */
    g_ctx.dot_q4_0_q8_0 = oc_oxk_dot_q4_0_q8_0_scalar;
    g_ctx.dot_q4_1_q8_0 = oc_oxk_dot_q4_1_q8_0_scalar;
    g_ctx.dot_q4_k_q8_k = oc_oxk_dot_q4_k_q8_k_scalar;
    g_ctx.dot_q5_k_q8_k = oc_oxk_dot_q5_k_q8_k_scalar;
    g_ctx.dot_q6_k_q8_k = oc_oxk_dot_q6_k_q8_k_scalar;
    g_ctx.dot_q8_0_q8_0 = oc_oxk_dot_q8_0_q8_0_scalar;
    g_ctx.dot_q2_k_q8_k = oc_oxk_dot_q2_k_q8_k_scalar;
    g_ctx.dot_q3_k_q8_k = oc_oxk_dot_q3_k_q8_k_scalar;
    g_ctx.dot_q4_k_prepped_multi = dot_q4_k_prepped_multi_scalar;
    g_ctx.dot_q6_k_prepped_multi = dot_q6_k_prepped_multi_scalar;
    g_ctx.dot_q2_k_prepped_multi = dot_q2_k_prepped_multi_scalar;
    g_ctx.dot_q3_k_prepped_multi = dot_q3_k_prepped_multi_scalar;
    g_ctx.dot_q2_k_prepped_1 = oc_oxk_dot_q2_k_prepped;
    g_ctx.dot_q3_k_prepped_1 = oc_oxk_dot_q3_k_prepped;

#if defined(__x86_64__) || defined(__i386__)
    /* oxk_avx2.c carries real AVX2 implementations of the Q4_K and Q8_0 dots, but nothing ever installed them: this table was written when every x86 variant forwarded to scalar and was not revisited when the kernels landed. */
    if (level >= OC_OXK_AVX2) {
        g_ctx.dot_q4_k_q8_k = oc_oxk_dot_q4_k_q8_k_avx2;
        g_ctx.dot_q8_0_q8_0 = oc_oxk_dot_q8_0_q8_0_avx2;
    }
    /* The VNNI kernels use _mm*_dpbusd_epi32, so AVX-512 alone is not
     * enough — Skylake-SP has AVX-512F/BW but no VNNI. */
    if (level >= OC_OXK_AVX512 && has_vnni) {
        g_ctx.dot_q5_k_q8_k = oc_oxk_dot_q5_k_q8_k_avx512vnni;
        g_ctx.dot_q6_k_q8_k = oc_oxk_dot_q6_k_q8_k_avx512vnni;
        g_ctx.dot_q4_k_prepped_multi = oc_oxk_dot_q4_k_prepped_multi_avx512;
        g_ctx.dot_q6_k_prepped_multi = oc_oxk_dot_q6_k_prepped_multi_avx512;
        g_ctx.dot_q2_k_prepped_multi = oc_oxk_dot_q2_k_prepped_multi_avx512;
        g_ctx.dot_q3_k_prepped_multi = oc_oxk_dot_q3_k_prepped_multi_avx512;
        g_ctx.dot_q2_k_prepped_1 = oc_oxk_dot_q2_k_prepped_avx512;
        g_ctx.dot_q3_k_prepped_1 = oc_oxk_dot_q3_k_prepped_avx512;
    }
#endif

#if defined(__aarch64__)
    if (level == OC_OXK_NEON) {
        g_ctx.dot_q4_0_q8_0 = oc_oxk_dot_q4_0_q8_0_neon;
        g_ctx.dot_q4_1_q8_0 = oc_oxk_dot_q4_1_q8_0_neon;
        g_ctx.dot_q4_k_q8_k = oc_oxk_dot_q4_k_q8_k_neon;
        g_ctx.dot_q5_k_q8_k = oc_oxk_dot_q5_k_q8_k_neon;
        g_ctx.dot_q6_k_q8_k = oc_oxk_dot_q6_k_q8_k_neon;
        g_ctx.dot_q8_0_q8_0 = oc_oxk_dot_q8_0_q8_0_neon;
    }
    /* Matvecs stay scalar on NEON too: their reference accumulates f32
     * per element, and vectorizing reassociates that sum (see oxk_neon.h). */
#endif

    oc_log(OC_LOG_DEBUG, "oxk: level=%s vnni=%d q4_k_dot=%s", name,
           (int)has_vnni,
           g_ctx.dot_q4_k_q8_k == oc_oxk_dot_q4_k_q8_k_scalar ? "scalar"
                                                              : "simd");

    g_ctx.matvec_q4_0_f32 = oc_oxk_matvec_q4_0_f32_scalar;
    g_ctx.matvec_q4_k_f32 = oc_oxk_matvec_q4_k_f32_scalar;
    g_ctx.matvec_q8_0_f32 = oc_oxk_matvec_q8_0_f32_scalar;
}

const OcOxkContext *oc_oxk_init(void)
{
    pthread_once(&g_once, oc_oxk_init_once);
    return &g_ctx;
}

const OcOxkCaps *oc_oxk_caps(void)
{
    oc_oxk_init();
    return &g_ctx.caps;
}


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

void oc_oxk_dot_q4_k_prepped_multi(const void *scratch, size_t blocks,
                                   const uint8_t *acts, size_t act_stride,
                                   size_t n_act, float *out)
{ oc_oxk_init(); g_ctx.dot_q4_k_prepped_multi(scratch, blocks, acts,
                                              act_stride, n_act, out); }

void oc_oxk_dot_q6_k_prepped_multi(const void *scratch, size_t blocks,
                                   const uint8_t *acts, size_t act_stride,
                                   size_t n_act, float *out)
{ oc_oxk_init(); g_ctx.dot_q6_k_prepped_multi(scratch, blocks, acts,
                                              act_stride, n_act, out); }

float oc_oxk_dot_q2_k_q8_k(const uint8_t *row, size_t blocks,
                           const uint8_t *q8)
{ oc_oxk_init(); return g_ctx.dot_q2_k_q8_k(row, blocks, q8); }

float oc_oxk_dot_q3_k_q8_k(const uint8_t *row, size_t blocks,
                           const uint8_t *q8)
{ oc_oxk_init(); return g_ctx.dot_q3_k_q8_k(row, blocks, q8); }

void oc_oxk_dot_q2_k_prepped_multi(const void *scratch, size_t blocks,
                                   const uint8_t *acts, size_t act_stride,
                                   size_t n_act, float *out)
{ oc_oxk_init(); g_ctx.dot_q2_k_prepped_multi(scratch, blocks, acts,
                                              act_stride, n_act, out); }

void oc_oxk_dot_q3_k_prepped_multi(const void *scratch, size_t blocks,
                                   const uint8_t *acts, size_t act_stride,
                                   size_t n_act, float *out)
{ oc_oxk_init(); g_ctx.dot_q3_k_prepped_multi(scratch, blocks, acts,
                                              act_stride, n_act, out); }

float oc_oxk_dot_q2_k_prepped_1(const void *prep, size_t blocks,
                                const uint8_t *act)
{ oc_oxk_init(); return g_ctx.dot_q2_k_prepped_1(prep, blocks, act); }

float oc_oxk_dot_q3_k_prepped_1(const void *prep, size_t blocks,
                                const uint8_t *act)
{ oc_oxk_init(); return g_ctx.dot_q3_k_prepped_1(prep, blocks, act); }

void oc_oxk_matvec_q4_0_f32(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_init(); g_ctx.matvec_q4_0_f32(w, n_rows, row_bytes, x, out); }

void oc_oxk_matvec_q4_k_f32(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_init(); g_ctx.matvec_q4_k_f32(w, n_rows, row_bytes, x, out); }

void oc_oxk_matvec_q8_0_f32(const uint8_t *w, size_t n_rows, size_t row_bytes, const float *x, float *out)
{ oc_oxk_init(); g_ctx.matvec_q8_0_f32(w, n_rows, row_bytes, x, out); }
