/*
 * quantization.c — scalar GGUF quantization reference path.
 *
 * Port of oxidize-core/src/compute/quantization.rs and the scalar dequant
 * helpers in quantization/{quant_simple.rs,quant_k_blocks.rs,quant_utils.rs}.
 * Bit-exact parity with the Rust scalar reference is a hard invariant
 * (VAL-QUANT-001..015).
 *
 * Scope (quant-standard-types feature):
 *   F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
 *   Q2_K, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M, Q6_K,
 *   I8, I16, I32, I64, F64.
 *
 * SIMD dispatch (scalar/AVX2/AVX-512/NEON) is layered on by the
 * `quant-simd-dispatch` feature; this file is the scalar reference and the
 * only path used by the standard-types tests.
 *
 * AL-family / IQ-family / NVFP4 packings are added by the
 * `quant-al-iq-nvfp4` feature; their enum slots exist but their dispatch
 * arms return OC_ERR_QUANT here.
 */
#include "oxidize/quant.h"
#include "oxidize/log.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* ─── Bit-exact f16/bf16 helpers (port of quant_utils.rs) ─────────────── */

static float f16_le_to_f32(const uint8_t b0, const uint8_t b1)
{
    uint16_t bits = (uint16_t)((uint16_t)b0 | ((uint16_t)b1 << 8));
    uint32_t sign = (uint32_t)((bits >> 15) & 1u);
    uint32_t exp  = (uint32_t)((bits >> 10) & 0x1Fu);
    uint32_t frac = (uint32_t)(bits & 0x03FFu);

    uint32_t f32_bits;
    if (exp == 0u) {
        if (frac == 0u) {
            f32_bits = sign << 31;
        } else {
            /* Subnormal: normalize the mantissa. */
            uint32_t frac_norm = frac;
            int32_t e = -14;
            while ((frac_norm & 0x0400u) == 0u) {
                frac_norm <<= 1;
                e -= 1;
            }
            frac_norm &= 0x03FFu;
            f32_bits = (sign << 31) | (((uint32_t)(e + 127)) << 23) | (frac_norm << 13);
        }
    } else if (exp == 0x1Fu) {
        f32_bits = (sign << 31) | 0x7F800000u | (frac << 13);
    } else {
        int32_t e = (int32_t)exp - 15 + 127;
        f32_bits = (sign << 31) | (((uint32_t)e) << 23) | (frac << 13);
    }

    float out;
    memcpy(&out, &f32_bits, sizeof(out));
    return out;
}

static uint16_t f32_to_f16_bits(float value)
{
    uint32_t x;
    memcpy(&x, &value, sizeof(x));
    uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu);
    uint32_t frac = x & 0x007FFFFFu;

    if (exp == 0xFF) {
        if (frac == 0u) {
            return (uint16_t)(sign | 0x7C00u);
        }
        uint16_t nan = (uint16_t)(frac >> 13);
        uint16_t out = (uint16_t)(sign | 0x7C00u | nan);
        if (nan == 0u) out |= 1u;
        return out;
    }

    int32_t exp16 = exp - 127 + 15;
    if (exp16 >= 0x1F) {
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exp16 <= 0) {
        if (exp16 < -10) {
            return sign;
        }
        uint32_t mant = frac | 0x00800000u;
        uint32_t shift = (uint32_t)(14 - exp16);
        uint16_t half_frac = (uint16_t)(mant >> shift);
        if (((mant >> (shift - 1)) & 1u) != 0u) {
            half_frac = (uint16_t)(half_frac + 1u);
        }
        return (uint16_t)(sign | half_frac);
    }

    uint16_t half_exp  = (uint16_t)(((uint16_t)exp16) << 10);
    uint16_t half_frac = (uint16_t)(frac >> 13);
    if ((frac & 0x00001000u) != 0u) {
        half_frac = (uint16_t)(half_frac + 1u);
        if ((half_frac & 0x0400u) != 0u) {
            half_frac = 0;
            half_exp = (uint16_t)(half_exp + 0x0400u);
            if (half_exp >= 0x7C00u) {
                return (uint16_t)(sign | 0x7C00u);
            }
        }
    }
    return (uint16_t)(sign | half_exp | half_frac);
}

static void f16_le_write(uint8_t *dst, float value)
{
    uint16_t bits = f32_to_f16_bits(value);
    dst[0] = (uint8_t)(bits & 0xFFu);
    dst[1] = (uint8_t)((bits >> 8) & 0xFFu);
}

/* ─── Bit-packing helpers (port of quant_utils.rs::write_bits) ────────── */

static void write_bits(uint8_t *bitstream, size_t index, size_t bits, uint32_t value)
{
    size_t bit_offset = index * bits;
    size_t byte_index = bit_offset / 8;
    size_t shift      = bit_offset % 8;
    uint32_t mask = ((1u << (uint32_t)bits) - 1u) << (uint32_t)shift;

    uint32_t acc = 0;
    for (size_t i = 0; i < 4; i++) {
        acc |= ((uint32_t)bitstream[byte_index + i]) << (8u * (uint32_t)i);
    }
    acc = (acc & ~mask) | ((value << (uint32_t)shift) & mask);
    for (size_t i = 0; i < 4; i++) {
        bitstream[byte_index + i] = (uint8_t)((acc >> (8u * (uint32_t)i)) & 0xFFu);
    }
}

/* llama.cpp `nearest_int` — fast round-to-nearest for quant heuristics. */
static int32_t nearest_int_f(float fval)
{
    float val = fval + 12582912.0f;
    uint32_t bits;
    memcpy(&bits, &val, sizeof(bits));
    return (int32_t)(bits & 0x007FFFFFu) - 0x00400000;
}

/* ─── Layout validation (port of quant_utils.rs::validate_layout) ─────── */

typedef struct {
    size_t input_block_size;
    size_t values_per_block;
} LayoutInfo;

static LayoutInfo layout_for(OcGgufQuantizationType qtype)
{
    switch (qtype) {
    case OC_QUANT_F32:    return (LayoutInfo){ 4,  1 };
    case OC_QUANT_F16:    return (LayoutInfo){ 2,  1 };
    case OC_QUANT_BF16:   return (LayoutInfo){ 2,  1 };
    case OC_QUANT_I8:     return (LayoutInfo){ 1,  1 };
    case OC_QUANT_I16:    return (LayoutInfo){ 2,  1 };
    case OC_QUANT_I32:    return (LayoutInfo){ 4,  1 };
    case OC_QUANT_I64:    return (LayoutInfo){ 8,  1 };
    case OC_QUANT_F64:    return (LayoutInfo){ 8,  1 };
    case OC_QUANT_Q4_0:   return (LayoutInfo){ OC_BLOCK_Q4_0_SIZE, OC_QK4_0 };
    case OC_QUANT_Q4_1:   return (LayoutInfo){ OC_BLOCK_Q4_1_SIZE, OC_QK4_1 };
    case OC_QUANT_Q5_0:   return (LayoutInfo){ OC_BLOCK_Q5_0_SIZE, OC_QK5_0 };
    case OC_QUANT_Q5_1:   return (LayoutInfo){ OC_BLOCK_Q5_1_SIZE, OC_QK5_1 };
    case OC_QUANT_Q8_0:   return (LayoutInfo){ OC_BLOCK_Q8_0_SIZE, OC_QK8_0 };
    case OC_QUANT_Q2_K:   return (LayoutInfo){ OC_BLOCK_Q2_K_SIZE, OC_QK_K };
    case OC_QUANT_Q3_K_S:
    case OC_QUANT_Q3_K_M:
    case OC_QUANT_Q3_K_L: return (LayoutInfo){ OC_BLOCK_Q3_K_SIZE, OC_QK_K };
    case OC_QUANT_Q4_K_S:
    case OC_QUANT_Q4_K_M: return (LayoutInfo){ OC_BLOCK_Q4_K_SIZE, OC_QK_K };
    case OC_QUANT_Q5_K_S:
    case OC_QUANT_Q5_K_M: return (LayoutInfo){ OC_BLOCK_Q5_K_SIZE, OC_QK_K };
    case OC_QUANT_Q6_K:   return (LayoutInfo){ OC_BLOCK_Q6_K_SIZE, OC_QK_K };
    /* AL/IQ/NVFP4 — block-size table filled by their features; default to
     * (0,0) so unknown paths return OC_ERR_QUANT. */
    default:              return (LayoutInfo){ 0, 0 };
    }
}

static bool validate_layout(OcGgufQuantizationType qtype,
                            size_t src_len, size_t value_count,
                            size_t input_block_size, size_t values_per_block)
{
    (void)qtype;
    if (input_block_size == 0 || values_per_block == 0) {
        return false;
    }
    if (src_len % input_block_size != 0) {
        return false;
    }
    size_t expected_output = (src_len / input_block_size) * values_per_block;
    return value_count == expected_output;
}

/* ─── Plain-storage dequant: F32/F16/BF16/I8/I16/I32/I64/F64 ─────────── */

static OcError dequant_f32(const uint8_t *src, size_t src_len,
                           float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_F32, src_len, value_count, 4, 1)) {
        return OC_ERR_INVALID_ARG;
    }
    for (size_t i = 0; i < value_count; i++) {
        uint32_t bits = (uint32_t)src[4 * i]
                      | ((uint32_t)src[4 * i + 1] << 8)
                      | ((uint32_t)src[4 * i + 2] << 16)
                      | ((uint32_t)src[4 * i + 3] << 24);
        memcpy(&dst[i], &bits, sizeof(bits));
    }
    return OC_OK;
}

static OcError dequant_f16(const uint8_t *src, size_t src_len,
                           float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_F16, src_len, value_count, 2, 1)) {
        return OC_ERR_INVALID_ARG;
    }
    for (size_t i = 0; i < value_count; i++) {
        dst[i] = f16_le_to_f32(src[2 * i], src[2 * i + 1]);
    }
    return OC_OK;
}

static OcError dequant_bf16(const uint8_t *src, size_t src_len,
                             float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_BF16, src_len, value_count, 2, 1)) {
        return OC_ERR_INVALID_ARG;
    }
    /* BF16 = upper 16 bits of f32; widening is exact (left-shift by 16). */
    for (size_t i = 0; i < value_count; i++) {
        uint32_t bits = (uint32_t)((uint16_t)src[2 * i]
                                   | ((uint16_t)src[2 * i + 1] << 8)) << 16;
        memcpy(&dst[i], &bits, sizeof(bits));
    }
    return OC_OK;
}

static OcError dequant_i8(const uint8_t *src, size_t src_len,
                          float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_I8, src_len, value_count, 1, 1)) {
        return OC_ERR_INVALID_ARG;
    }
    for (size_t i = 0; i < value_count; i++) {
        dst[i] = (float)((int8_t)src[i]);
    }
    return OC_OK;
}

static OcError dequant_i16(const uint8_t *src, size_t src_len,
                           float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_I16, src_len, value_count, 2, 1)) {
        return OC_ERR_INVALID_ARG;
    }
    for (size_t i = 0; i < value_count; i++) {
        int16_t v = (int16_t)((uint16_t)src[2 * i]
                              | ((uint16_t)src[2 * i + 1] << 8));
        dst[i] = (float)v;
    }
    return OC_OK;
}

static OcError dequant_i32(const uint8_t *src, size_t src_len,
                           float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_I32, src_len, value_count, 4, 1)) {
        return OC_ERR_INVALID_ARG;
    }
    for (size_t i = 0; i < value_count; i++) {
        int32_t v = (int32_t)((uint32_t)src[4 * i]
                              | ((uint32_t)src[4 * i + 1] << 8)
                              | ((uint32_t)src[4 * i + 2] << 16)
                              | ((uint32_t)src[4 * i + 3] << 24));
        dst[i] = (float)v;
    }
    return OC_OK;
}

static OcError dequant_i64(const uint8_t *src, size_t src_len,
                           float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_I64, src_len, value_count, 8, 1)) {
        return OC_ERR_INVALID_ARG;
    }
    for (size_t i = 0; i < value_count; i++) {
        uint64_t v = (uint64_t)src[8 * i]
                   | ((uint64_t)src[8 * i + 1] << 8)
                   | ((uint64_t)src[8 * i + 2] << 16)
                   | ((uint64_t)src[8 * i + 3] << 24)
                   | ((uint64_t)src[8 * i + 4] << 32)
                   | ((uint64_t)src[8 * i + 5] << 40)
                   | ((uint64_t)src[8 * i + 6] << 48)
                   | ((uint64_t)src[8 * i + 7] << 56);
        int64_t sv;
        memcpy(&sv, &v, sizeof(sv));
        dst[i] = (float)sv;
    }
    return OC_OK;
}

static OcError dequant_f64(const uint8_t *src, size_t src_len,
                           float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_F64, src_len, value_count, 8, 1)) {
        return OC_ERR_INVALID_ARG;
    }
    for (size_t i = 0; i < value_count; i++) {
        uint64_t v = (uint64_t)src[8 * i]
                   | ((uint64_t)src[8 * i + 1] << 8)
                   | ((uint64_t)src[8 * i + 2] << 16)
                   | ((uint64_t)src[8 * i + 3] << 24)
                   | ((uint64_t)src[8 * i + 4] << 32)
                   | ((uint64_t)src[8 * i + 5] << 40)
                   | ((uint64_t)src[8 * i + 6] << 48)
                   | ((uint64_t)src[8 * i + 7] << 56);
        double d;
        memcpy(&d, &v, sizeof(d));
        dst[i] = (float)d;
    }
    return OC_OK;
}

/* ─── Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 dequant ────────────────────────── */

static OcError dequant_q4_0(const uint8_t *src, size_t src_len,
                            float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_Q4_0, src_len, value_count,
                         OC_BLOCK_Q4_0_SIZE, OC_QK4_0)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_Q4_0_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_Q4_0_SIZE;
        float *out = dst + b * OC_QK4_0;
        float d = f16_le_to_f32(block[0], block[1]);
        for (size_t i = 0; i < 16; i++) {
            uint8_t packed = block[2 + i];
            out[i]      = (float)((int32_t)(packed & 0x0Fu) - 8) * d;
            out[i + 16] = (float)((int32_t)((packed >> 4) & 0x0Fu) - 8) * d;
        }
    }
    return OC_OK;
}

static OcError dequant_q4_1(const uint8_t *src, size_t src_len,
                            float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_Q4_1, src_len, value_count,
                         OC_BLOCK_Q4_1_SIZE, OC_QK4_1)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_Q4_1_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_Q4_1_SIZE;
        float *out = dst + b * OC_QK4_1;
        float d = f16_le_to_f32(block[0], block[1]);
        float m = f16_le_to_f32(block[2], block[3]);
        for (size_t i = 0; i < 16; i++) {
            uint8_t packed = block[4 + i];
            out[2 * i]     = (float)(packed & 0x0Fu) * d + m;
            out[2 * i + 1] = (float)((packed >> 4) & 0x0Fu) * d + m;
        }
    }
    return OC_OK;
}

static OcError dequant_q5_0(const uint8_t *src, size_t src_len,
                            float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_Q5_0, src_len, value_count,
                         OC_BLOCK_Q5_0_SIZE, OC_QK5_0)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_Q5_0_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_Q5_0_SIZE;
        float *out = dst + b * OC_QK5_0;
        float d = f16_le_to_f32(block[0], block[1]);
        const uint8_t *qh = &block[2];
        const uint8_t *qs = &block[6];
        for (size_t i = 0; i < OC_QK5_0; i++) {
            uint8_t low = (i % 2 == 0)
                ? (uint8_t)(qs[i / 2] & 0x0Fu)
                : (uint8_t)((qs[i / 2] >> 4) & 0x0Fu);
            uint8_t high = (uint8_t)((qh[i / 8] >> (i % 8)) & 0x01u);
            uint8_t q = (uint8_t)(low | (high << 4));
            out[i] = (float)((int8_t)q - 16) * d;
        }
    }
    return OC_OK;
}

static OcError dequant_q5_1(const uint8_t *src, size_t src_len,
                            float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_Q5_1, src_len, value_count,
                         OC_BLOCK_Q5_1_SIZE, OC_QK5_1)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_Q5_1_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_Q5_1_SIZE;
        float *out = dst + b * OC_QK5_1;
        float d = f16_le_to_f32(block[0], block[1]);
        float m = f16_le_to_f32(block[2], block[3]);
        const uint8_t *qh = &block[4];
        const uint8_t *qs = &block[8];
        for (size_t i = 0; i < OC_QK5_1; i++) {
            uint8_t low = (i % 2 == 0)
                ? (uint8_t)(qs[i / 2] & 0x0Fu)
                : (uint8_t)((qs[i / 2] >> 4) & 0x0Fu);
            uint8_t high = (uint8_t)((qh[i / 8] >> (i % 8)) & 0x01u);
            uint8_t q = (uint8_t)(low | (high << 4));
            out[i] = (float)q * d + m;
        }
    }
    return OC_OK;
}

static OcError dequant_q8_0(const uint8_t *src, size_t src_len,
                            float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_Q8_0, src_len, value_count,
                         OC_BLOCK_Q8_0_SIZE, OC_QK8_0)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_Q8_0_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_Q8_0_SIZE;
        float *out = dst + b * OC_QK8_0;
        float d = f16_le_to_f32(block[0], block[1]);
        for (size_t i = 0; i < OC_QK8_0; i++) {
            out[i] = (float)((int8_t)block[2 + i]) * d;
        }
    }
    return OC_OK;
}

/* ─── K-family dequant (port of quant_k_blocks.rs) ────────────────────── */

static void get_scale_min_k4(size_t j, const uint8_t *scales,
                             uint8_t *out_sc, uint8_t *out_m)
{
    if (j < 4) {
        *out_sc = (uint8_t)(scales[j] & 63u);
        *out_m  = (uint8_t)(scales[j + 4] & 63u);
    } else {
        *out_sc = (uint8_t)((scales[j + 4] & 0x0Fu)
                            | ((scales[j - 4] >> 6) << 4));
        *out_m  = (uint8_t)(((scales[j + 4] >> 4) & 0x0Fu)
                            | ((scales[j] >> 6) << 4));
    }
}

static OcError dequant_q2_k(const uint8_t *src, size_t src_len,
                            float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_Q2_K, src_len, value_count,
                         OC_BLOCK_Q2_K_SIZE, OC_QK_K)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_Q2_K_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_Q2_K_SIZE;
        float *out = dst + b * OC_QK_K;
        float d   = f16_le_to_f32(block[80], block[81]);
        float min = f16_le_to_f32(block[82], block[83]);
        const uint8_t *scales = &block[0];
        const uint8_t *qs = &block[16];
        size_t q_ptr = 0;
        size_t is = 0;
        for (size_t outer = 0; outer < 2; outer++) {
            size_t qs_base = outer * 32;
            for (size_t inner = 0; inner < 4; inner++) {
                uint8_t sc1 = scales[is];
                float dl1 = d * (float)(sc1 & 0x0Fu);
                float ml1 = min * (float)(sc1 >> 4);
                is += 1;
                uint8_t sc2 = scales[is];
                float dl2 = d * (float)(sc2 & 0x0Fu);
                float ml2 = min * (float)(sc2 >> 4);
                is += 1;
                size_t shift = ((is / 2 - 1) % 4) * 2;
                for (size_t l = 0; l < 16; l++) {
                    out[q_ptr + l] = dl1 * (float)((qs[qs_base + l] >> shift) & 3u) - ml1;
                }
                for (size_t l = 0; l < 16; l++) {
                    out[q_ptr + 16 + l] = dl2 * (float)((qs[qs_base + 16 + l] >> shift) & 3u) - ml2;
                }
                q_ptr += 32;
            }
        }
    }
    return OC_OK;
}

static OcError dequant_q3_k(const uint8_t *src, size_t src_len,
                            float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_Q3_K_S, src_len, value_count,
                         OC_BLOCK_Q3_K_SIZE, OC_QK_K)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_Q3_K_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_Q3_K_SIZE;
        float *out = dst + b * OC_QK_K;
        float d_all = f16_le_to_f32(block[108], block[109]);
        const uint8_t *hmask = &block[0];
        const uint8_t *qs = &block[32];
        uint32_t scales_raw[4];
        scales_raw[0] = (uint32_t)block[96] | ((uint32_t)block[97] << 8)
                      | ((uint32_t)block[98] << 16) | ((uint32_t)block[99] << 24);
        scales_raw[1] = (uint32_t)block[100] | ((uint32_t)block[101] << 8)
                      | ((uint32_t)block[102] << 16) | ((uint32_t)block[103] << 24);
        scales_raw[2] = (uint32_t)block[104] | ((uint32_t)block[105] << 8)
                      | ((uint32_t)block[106] << 16) | ((uint32_t)block[107] << 24);
        uint32_t tmp = scales_raw[2];
        scales_raw[2] = ((scales_raw[0] >> 4) & 0x0F0F0F0Fu)
                      | (((tmp >> 4) & 0x03030303u) << 4);
        scales_raw[3] = ((scales_raw[1] >> 4) & 0x0F0F0F0Fu)
                      | (((tmp >> 6) & 0x03030303u) << 4);
        scales_raw[0] = (scales_raw[0] & 0x0F0F0F0Fu)
                      | ((tmp & 0x03030303u) << 4);
        scales_raw[1] = (scales_raw[1] & 0x0F0F0F0Fu)
                      | (((tmp >> 2) & 0x03030303u) << 4);

        uint8_t scale_bytes[16];
        for (size_t i = 0; i < 4; i++) {
            scale_bytes[i * 4 + 0] = (uint8_t)(scales_raw[i] & 0xFFu);
            scale_bytes[i * 4 + 1] = (uint8_t)((scales_raw[i] >> 8) & 0xFFu);
            scale_bytes[i * 4 + 2] = (uint8_t)((scales_raw[i] >> 16) & 0xFFu);
            scale_bytes[i * 4 + 3] = (uint8_t)((scales_raw[i] >> 24) & 0xFFu);
        }

        size_t q_ptr = 0;
        size_t is = 0;
        uint8_t m = 1u;
        for (size_t outer = 0; outer < 2; outer++) {
            for (size_t inner = 0; inner < 4; inner++) {
                float dl = d_all * (float)((int32_t)(int8_t)scale_bytes[is] - 32);
                is += 1;
                size_t shift = ((is - 1) % 4) * 2;
                for (size_t l = 0; l < 16; l++) {
                    int32_t qv = (int32_t)((qs[l] >> shift) & 3u);
                    int32_t hbit = ((hmask[l] & m) != 0) ? 0 : 4;
                    out[q_ptr + l] = dl * (float)(qv - hbit);
                }
                float dl2 = d_all * (float)((int32_t)(int8_t)scale_bytes[is] - 32);
                is += 1;
                for (size_t l = 0; l < 16; l++) {
                    int32_t qv = (int32_t)((qs[l + 16] >> shift) & 3u);
                    int32_t hbit = ((hmask[l + 16] & m) != 0) ? 0 : 4;
                    out[q_ptr + 16 + l] = dl2 * (float)(qv - hbit);
                }
                q_ptr += 32;
                m = (uint8_t)(m << 1);
            }
        }
    }
    return OC_OK;
}

static OcError dequant_q4_k(const uint8_t *src, size_t src_len,
                            float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_Q4_K_S, src_len, value_count,
                         OC_BLOCK_Q4_K_SIZE, OC_QK_K)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_Q4_K_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_Q4_K_SIZE;
        float *out = dst + b * OC_QK_K;
        float d   = f16_le_to_f32(block[0], block[1]);
        float min = f16_le_to_f32(block[2], block[3]);
        const uint8_t *scales = &block[4];
        const uint8_t *qs = &block[16];
        size_t out_ptr = 0;
        size_t is = 0;
        for (size_t gp = 0; gp < 4; gp++) {
            size_t q_base = gp * 32;
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is,     scales, &sc1, &m1);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);
            float d1 = d * (float)sc1;
            float min1 = min * (float)m1;
            float d2 = d * (float)sc2;
            float min2 = min * (float)m2;
            for (size_t l = 0; l < 32; l++) {
                out[out_ptr + l] = d1 * (float)(qs[q_base + l] & 0x0Fu) - min1;
            }
            for (size_t l = 0; l < 32; l++) {
                out[out_ptr + 32 + l] = d2 * (float)(qs[q_base + l] >> 4) - min2;
            }
            out_ptr += 64;
            is += 2;
        }
    }
    return OC_OK;
}

static OcError dequant_q5_k(const uint8_t *src, size_t src_len,
                            float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_Q5_K_S, src_len, value_count,
                         OC_BLOCK_Q5_K_SIZE, OC_QK_K)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_Q5_K_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_Q5_K_SIZE;
        float *out = dst + b * OC_QK_K;
        float d   = f16_le_to_f32(block[0], block[1]);
        float min = f16_le_to_f32(block[2], block[3]);
        const uint8_t *scales = &block[4];
        const uint8_t *qh = &block[16];
        const uint8_t *qs = &block[48];
        size_t q_ptr = 0;
        size_t is = 0;
        uint8_t u1 = 1u;
        uint8_t u2 = 2u;
        for (size_t outer = 0; outer < 4; outer++) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is,     scales, &sc1, &m1);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);
            float d1 = d * (float)sc1;
            float min1 = min * (float)m1;
            float d2 = d * (float)sc2;
            float min2 = min * (float)m2;
            for (size_t l = 0; l < 32; l++) {
                uint32_t qv1 = (uint32_t)(qs[l] & 0x0Fu)
                             + (((qh[l] & u1) != 0) ? 16u : 0u);
                out[q_ptr + l] = d1 * (float)qv1 - min1;
            }
            for (size_t l = 0; l < 32; l++) {
                uint32_t qv2 = (uint32_t)(qs[l] >> 4)
                             + (((qh[l] & u2) != 0) ? 16u : 0u);
                out[q_ptr + 32 + l] = d2 * (float)qv2 - min2;
            }
            q_ptr += 64;
            is += 2;
            u1 = (uint8_t)(u1 << 2);
            u2 = (uint8_t)(u2 << 2);
        }
    }
    return OC_OK;
}

static OcError dequant_q6_k(const uint8_t *src, size_t src_len,
                            float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_Q6_K, src_len, value_count,
                         OC_BLOCK_Q6_K_SIZE, OC_QK_K)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_Q6_K_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_Q6_K_SIZE;
        float *out = dst + b * OC_QK_K;
        float d = f16_le_to_f32(block[208], block[209]);
        const uint8_t *ql = &block[0];
        const uint8_t *qh = &block[128];
        const uint8_t *sc = &block[192];
        size_t q_ptr = 0;
        for (size_t group = 0; group < 2; group++) {
            size_t ql_off = group * 64;
            size_t qh_off = group * 32;
            size_t sc_off = group * 8;
            for (size_t l = 0; l < 32; l++) {
                size_t is_idx = l / 16;
                int32_t q1 = ((int32_t)(ql[ql_off + l] & 0x0Fu)
                              | (((int32_t)(qh[qh_off + l] & 3u)) << 4)) - 32;
                int32_t q2 = ((int32_t)(ql[ql_off + l + 32] & 0x0Fu)
                              | ((((int32_t)((qh[qh_off + l] >> 2) & 3u))) << 4)) - 32;
                int32_t q3 = ((int32_t)(ql[ql_off + l] >> 4)
                              | ((((int32_t)((qh[qh_off + l] >> 4) & 3u))) << 4)) - 32;
                int32_t q4 = ((int32_t)(ql[ql_off + l + 32] >> 4)
                              | ((((int32_t)((qh[qh_off + l] >> 6) & 3u))) << 4)) - 32;
                out[q_ptr + l]      = d * (float)(int8_t)sc[sc_off + is_idx]     * (float)q1;
                out[q_ptr + 32 + l] = d * (float)(int8_t)sc[sc_off + is_idx + 2] * (float)q2;
                out[q_ptr + 64 + l] = d * (float)(int8_t)sc[sc_off + is_idx + 4] * (float)q3;
                out[q_ptr + 96 + l] = d * (float)(int8_t)sc[sc_off + is_idx + 6] * (float)q4;
            }
            q_ptr += 128;
        }
    }
    return OC_OK;
}

/* ─── Pack-from-f32 (port of quant_simple.rs + quant_k_blocks.rs) ────── */

static OcError pack_f32(const float *src, size_t value_count,
                        uint8_t *dst, size_t dst_len)
{
    if (dst_len != value_count * 4) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < value_count; i++) {
        uint32_t bits;
        memcpy(&bits, &src[i], sizeof(bits));
        dst[4 * i + 0] = (uint8_t)(bits & 0xFFu);
        dst[4 * i + 1] = (uint8_t)((bits >> 8) & 0xFFu);
        dst[4 * i + 2] = (uint8_t)((bits >> 16) & 0xFFu);
        dst[4 * i + 3] = (uint8_t)((bits >> 24) & 0xFFu);
    }
    return OC_OK;
}

static OcError pack_f16(const float *src, size_t value_count,
                        uint8_t *dst, size_t dst_len)
{
    if (dst_len != value_count * 2) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < value_count; i++) {
        f16_le_write(&dst[2 * i], src[i]);
    }
    return OC_OK;
}

static OcError pack_i8(const float *src, size_t value_count,
                       uint8_t *dst, size_t dst_len)
{
    if (dst_len != value_count) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < value_count; i++) {
        int8_t v = (int8_t)(int32_t)src[i];
        dst[i] = (uint8_t)v;
    }
    return OC_OK;
}

static OcError pack_i16(const float *src, size_t value_count,
                        uint8_t *dst, size_t dst_len)
{
    if (dst_len != value_count * 2) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < value_count; i++) {
        int16_t v = (int16_t)(int32_t)src[i];
        dst[2 * i + 0] = (uint8_t)((uint16_t)v & 0xFFu);
        dst[2 * i + 1] = (uint8_t)(((uint16_t)v >> 8) & 0xFFu);
    }
    return OC_OK;
}

static OcError pack_i32(const float *src, size_t value_count,
                        uint8_t *dst, size_t dst_len)
{
    if (dst_len != value_count * 4) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < value_count; i++) {
        int32_t v = (int32_t)src[i];
        uint32_t uv;
        memcpy(&uv, &v, sizeof(uv));
        dst[4 * i + 0] = (uint8_t)(uv & 0xFFu);
        dst[4 * i + 1] = (uint8_t)((uv >> 8) & 0xFFu);
        dst[4 * i + 2] = (uint8_t)((uv >> 16) & 0xFFu);
        dst[4 * i + 3] = (uint8_t)((uv >> 24) & 0xFFu);
    }
    return OC_OK;
}

static OcError pack_i64(const float *src, size_t value_count,
                        uint8_t *dst, size_t dst_len)
{
    if (dst_len != value_count * 8) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < value_count; i++) {
        int64_t v = (int64_t)src[i];
        uint64_t uv;
        memcpy(&uv, &v, sizeof(uv));
        for (int k = 0; k < 8; k++) {
            dst[8 * i + k] = (uint8_t)((uv >> (8 * k)) & 0xFFu);
        }
    }
    return OC_OK;
}

static OcError pack_f64(const float *src, size_t value_count,
                        uint8_t *dst, size_t dst_len)
{
    if (dst_len != value_count * 8) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < value_count; i++) {
        double d = (double)src[i];
        uint64_t uv;
        memcpy(&uv, &d, sizeof(uv));
        for (int k = 0; k < 8; k++) {
            dst[8 * i + k] = (uint8_t)((uv >> (8 * k)) & 0xFFu);
        }
    }
    return OC_OK;
}

static OcError pack_q4_0(const float *src, size_t value_count,
                         uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK4_0 != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK4_0) * OC_BLOCK_Q4_0_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK4_0;
    for (size_t b = 0; b < n_blocks; b++) {
        const float *in_block = src + b * OC_QK4_0;
        uint8_t *out_block = dst + b * OC_BLOCK_Q4_0_SIZE;
        float amax = 0.0f, mx = 0.0f;
        for (size_t i = 0; i < OC_QK4_0; i++) {
            float v = in_block[i];
            float a = fabsf(v);
            if (a > amax) { amax = a; mx = v; }
        }
        float d = (mx != 0.0f) ? (mx / -8.0f) : 0.0f;
        float inv_d = (d != 0.0f) ? (1.0f / d) : 0.0f;
        f16_le_write(&out_block[0], d);
        for (size_t i = 0; i < 16; i++) {
            uint8_t lo = (d == 0.0f) ? 8u
                       : (uint8_t)fminf(fmaxf(truncf(in_block[i] * inv_d + 8.5f), 0.0f), 15.0f);
            uint8_t hi = (d == 0.0f) ? 8u
                       : (uint8_t)fminf(fmaxf(truncf(in_block[i + 16] * inv_d + 8.5f), 0.0f), 15.0f);
            out_block[2 + i] = (uint8_t)(lo | (hi << 4));
        }
    }
    return OC_OK;
}

static OcError pack_q4_1(const float *src, size_t value_count,
                         uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK4_1 != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK4_1) * OC_BLOCK_Q4_1_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK4_1;
    for (size_t b = 0; b < n_blocks; b++) {
        const float *in_block = src + b * OC_QK4_1;
        uint8_t *out_block = dst + b * OC_BLOCK_Q4_1_SIZE;
        float mn = INFINITY, mx = -INFINITY;
        for (size_t i = 0; i < OC_QK4_1; i++) {
            if (in_block[i] < mn) mn = in_block[i];
            if (in_block[i] > mx) mx = in_block[i];
        }
        float d = (mx > mn) ? ((mx - mn) / 15.0f) : 0.0f;
        f16_le_write(&out_block[0], d);
        f16_le_write(&out_block[2], mn);
        for (size_t i = 0; i < OC_QK4_1 / 2; i++) {
            uint8_t q_low  = (d == 0.0f) ? 0u
                           : (uint8_t)fminf(fmaxf(roundf((in_block[2 * i] - mn) / d), 0.0f), 15.0f);
            uint8_t q_high = (d == 0.0f) ? 0u
                           : (uint8_t)fminf(fmaxf(roundf((in_block[2 * i + 1] - mn) / d), 0.0f), 15.0f);
            out_block[4 + i] = (uint8_t)(q_low | (q_high << 4));
        }
    }
    return OC_OK;
}

static OcError pack_q5_0(const float *src, size_t value_count,
                         uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK5_0 != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK5_0) * OC_BLOCK_Q5_0_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK5_0;
    for (size_t b = 0; b < n_blocks; b++) {
        const float *in_block = src + b * OC_QK5_0;
        uint8_t *out_block = dst + b * OC_BLOCK_Q5_0_SIZE;
        float max_abs = 0.0f;
        for (size_t i = 0; i < OC_QK5_0; i++) {
            float a = fabsf(in_block[i]);
            if (a > max_abs) max_abs = a;
        }
        float d = (max_abs == 0.0f) ? 0.0f : (max_abs / 16.0f);
        f16_le_write(&out_block[0], d);
        out_block[2] = out_block[3] = out_block[4] = out_block[5] = 0u;
        for (size_t i = 0; i < OC_QK5_0; i++) {
            uint8_t q = (d == 0.0f) ? 16u
                      : (uint8_t)fminf(fmaxf((float)((int32_t)roundf(in_block[i] / d) + 16), 0.0f), 31.0f);
            if ((q & 0x10u) != 0u) {
                out_block[2 + i / 8] |= (uint8_t)(1u << (i % 8));
            }
            uint8_t low = q & 0x0Fu;
            size_t qs_index = 6 + i / 2;
            if (i % 2 == 0) {
                out_block[qs_index] = low;
            } else {
                out_block[qs_index] |= (uint8_t)(low << 4);
            }
        }
    }
    return OC_OK;
}

static OcError pack_q5_1(const float *src, size_t value_count,
                         uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK5_1 != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK5_1) * OC_BLOCK_Q5_1_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK5_1;
    for (size_t b = 0; b < n_blocks; b++) {
        const float *in_block = src + b * OC_QK5_1;
        uint8_t *out_block = dst + b * OC_BLOCK_Q5_1_SIZE;
        float mn = INFINITY, mx = -INFINITY;
        for (size_t i = 0; i < OC_QK5_1; i++) {
            if (in_block[i] < mn) mn = in_block[i];
            if (in_block[i] > mx) mx = in_block[i];
        }
        float d = (mx > mn) ? ((mx - mn) / 31.0f) : 0.0f;
        f16_le_write(&out_block[0], d);
        f16_le_write(&out_block[2], mn);
        out_block[4] = out_block[5] = out_block[6] = out_block[7] = 0u;
        for (size_t i = 0; i < OC_QK5_1; i++) {
            uint8_t q = (d == 0.0f) ? 0u
                      : (uint8_t)fminf(fmaxf(roundf((in_block[i] - mn) / d), 0.0f), 31.0f);
            if ((q & 0x10u) != 0u) {
                out_block[4 + i / 8] |= (uint8_t)(1u << (i % 8));
            }
            uint8_t low = q & 0x0Fu;
            size_t qs_index = 8 + i / 2;
            if (i % 2 == 0) {
                out_block[qs_index] = low;
            } else {
                out_block[qs_index] |= (uint8_t)(low << 4);
            }
        }
    }
    return OC_OK;
}

static OcError pack_q8_0(const float *src, size_t value_count,
                         uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK8_0 != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK8_0) * OC_BLOCK_Q8_0_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK8_0;
    for (size_t b = 0; b < n_blocks; b++) {
        const float *in_block = src + b * OC_QK8_0;
        uint8_t *out_block = dst + b * OC_BLOCK_Q8_0_SIZE;
        float max_abs = 0.0f;
        for (size_t i = 0; i < OC_QK8_0; i++) {
            float a = fabsf(in_block[i]);
            if (a > max_abs) max_abs = a;
        }
        float d = (max_abs == 0.0f) ? 0.0f : (max_abs / 127.0f);
        f16_le_write(&out_block[0], d);
        for (size_t i = 0; i < OC_QK8_0; i++) {
            int32_t q = (d == 0.0f) ? 0
                      : (int32_t)fminf(fmaxf(roundf(in_block[i] / d), -128.0f), 127.0f);
            out_block[2 + i] = (uint8_t)(int8_t)q;
        }
    }
    return OC_OK;
}

/* llama.cpp `make_qkx1_quants` — used by Q4_K pack. */
static float make_qkx1_quants(const float *x, size_t n, uint8_t *l,
                             float *out_min, int ntry, float alpha)
{
    size_t nmax = 15;
    float mn = x[0];
    float mx = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] < mn) mn = x[i];
        if (x[i] > mx) mx = x[i];
    }
    if (mx == mn) {
        for (size_t i = 0; i < n; i++) l[i] = 0;
        *out_min = 0.0f;
        return 0.0f;
    }
    if (mn > 0.0f) mn = 0.0f;

    float iscale = (float)nmax / (mx - mn);
    float scale = 1.0f / iscale;

    for (int itry = 0; itry < ntry; itry++) {
        float sumlx = 0.0f;
        int32_t suml2 = 0;
        bool did_change = false;
        for (size_t i = 0; i < n; i++) {
            int32_t ql = nearest_int_f(iscale * (x[i] - mn));
            if (ql < 0) ql = 0;
            if (ql > (int32_t)nmax) ql = (int32_t)nmax;
            if (l[i] != (uint8_t)ql) { l[i] = (uint8_t)ql; did_change = true; }
            sumlx += (x[i] - mn) * (float)ql;
            suml2 += ql * ql;
        }
        if (suml2 > 0) scale = sumlx / (float)suml2;
        float sum = 0.0f;
        for (size_t i = 0; i < n; i++) {
            sum += x[i] - scale * (float)l[i];
        }
        mn = alpha * mn + (1.0f - alpha) * sum / (float)n;
        if (mn > 0.0f) mn = 0.0f;
        iscale = 1.0f / scale;
        if (!did_change) break;
    }

    *out_min = -mn;
    return scale;
}

/* Q4_K pack (port of quant_k_blocks.rs::quantize_q4_k_scalar). */
static OcError pack_q4_k(const float *src, size_t value_count,
                         uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK_K != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK_K) * OC_BLOCK_Q4_K_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;

    size_t n_blocks = value_count / OC_QK_K;
    uint8_t l[OC_QK_K];
    float mins[OC_QK_K / 32];
    float scales[OC_QK_K / 32];
    for (size_t b = 0; b < n_blocks; b++) {
        const float *in_block = src + b * OC_QK_K;
        uint8_t *out_block = dst + b * OC_BLOCK_Q4_K_SIZE;
        float max_scale = 0.0f, max_min = 0.0f;
        for (size_t j = 0; j < OC_QK_K / 32; j++) {
            scales[j] = make_qkx1_quants(&in_block[32 * j], 32,
                                          &l[32 * j], &mins[j], 5, 0.5f);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }
        float inv_scale = (max_scale > 0.0f) ? (63.0f / max_scale) : 0.0f;
        float inv_min   = (max_min   > 0.0f) ? (63.0f / max_min)   : 0.0f;

        for (size_t k = 4; k < 16; k++) out_block[k] = 0u;
        for (size_t j = 0; j < OC_QK_K / 32; j++) {
            int32_t ls = nearest_int_f(inv_scale * scales[j]);
            if (ls < 0) { ls = 0; }
            if (ls > 63) { ls = 63; }
            int32_t lm = nearest_int_f(inv_min * mins[j]);
            if (lm < 0) { lm = 0; }
            if (lm > 63) { lm = 63; }
            if (j < 4) {
                out_block[4 + j]      = (uint8_t)ls;
                out_block[4 + j + 4]  = (uint8_t)lm;
            } else {
                out_block[4 + j + 4]  = (uint8_t)((ls & 0x0Fu) | ((lm & 0x0Fu) << 4));
                out_block[4 + j - 4] |= (uint8_t)((ls >> 4) << 6);
                out_block[4 + j]     |= (uint8_t)((lm >> 4) << 6);
            }
        }

        f16_le_write(&out_block[0], max_scale / 63.0f);
        f16_le_write(&out_block[2], max_min   / 63.0f);

        for (size_t j = 0; j < OC_QK_K / 32; j++) {
            uint8_t sc, m;
            get_scale_min_k4(j, &out_block[4], &sc, &m);
            float d = f16_le_to_f32(out_block[0], out_block[1]) * (float)sc;
            if (d == 0.0f) continue;
            float dm = f16_le_to_f32(out_block[2], out_block[3]) * (float)m;
            for (size_t ii = 0; ii < 32; ii++) {
                int32_t ql = nearest_int_f((in_block[32 * j + ii] + dm) / d);
                if (ql < 0) { ql = 0; }
                if (ql > 15) { ql = 15; }
                l[32 * j + ii] = (uint8_t)ql;
            }
        }

        for (size_t k = 16; k < 144; k++) out_block[k] = 0u;
        for (size_t j = 0; j < OC_QK_K; j += 64) {
            for (size_t l_idx = 0; l_idx < 32; l_idx++) {
                out_block[16 + (j / 64) * 32 + l_idx]
                    = (uint8_t)(l[j + l_idx] | (l[j + l_idx + 32] << 4));
            }
        }
    }
    return OC_OK;
}

/* Generic K-packed encoder for symmetric-with-zero-point layouts (Q2_K, Q3_K,
 * Q5_K, Q6_K). Port of quant_k_blocks.rs::quantize_k_packed_scalar. */
static OcError pack_k_packed(const float *src, size_t value_count,
                             uint8_t *dst, size_t dst_len,
                             size_t block_size, size_t bits, float zero_point)
{
    if (value_count % OC_QK_K != 0) return OC_ERR_INVALID_ARG;
    if (dst_len != (value_count / OC_QK_K) * block_size) return OC_ERR_INVALID_ARG;

    float max_q = (float)((1u << (uint32_t)bits) - 1u);
    float positive_span = (zero_point > (max_q - zero_point)) ? zero_point : (max_q - zero_point);
    size_t n_blocks = value_count / OC_QK_K;
    for (size_t b = 0; b < n_blocks; b++) {
        const float *in_block = src + b * OC_QK_K;
        uint8_t *out_block = dst + b * block_size;
        float max_abs = 0.0f;
        for (size_t i = 0; i < OC_QK_K; i++) {
            float a = fabsf(in_block[i]);
            if (a > max_abs) max_abs = a;
        }
        float d = (max_abs == 0.0f) ? 0.0f : (max_abs / positive_span);
        f16_le_write(&out_block[0], d);
        for (size_t k = 2; k < block_size; k++) out_block[k] = 0u;
        for (size_t i = 0; i < OC_QK_K; i++) {
            uint32_t q;
            if (d == 0.0f) {
                q = (uint32_t)roundf(zero_point);
            } else {
                float v = (in_block[i] / d) + zero_point;
                if (v < 0.0f) v = 0.0f;
                if (v > max_q) v = max_q;
                q = (uint32_t)roundf(v);
            }
            write_bits(&out_block[2], i, bits, q);
        }
    }
    return OC_OK;
}

/* ─── Public dispatch ────────────────────────────────────────────────── */

OcQuantBlockLayout oc_quant_block_size(OcGgufQuantizationType qtype)
{
    LayoutInfo l = layout_for(qtype);
    OcQuantBlockLayout out = { l.values_per_block, l.input_block_size };
    return out;
}

size_t oc_quantized_size(OcGgufQuantizationType qtype, size_t value_count)
{
    OcQuantBlockLayout bs = oc_quant_block_size(qtype);
    if (bs.elements_per_block == 0) return 0;
    if (value_count % bs.elements_per_block != 0) return 0;
    return (value_count / bs.elements_per_block) * bs.bytes_per_block;
}

OcError oc_quant_dequant_row(OcGgufQuantizationType qtype,
                             const uint8_t *src, size_t src_len,
                             float *dst, size_t value_count)
{
    if (src == NULL || dst == NULL) return OC_ERR_INVALID_ARG;
    switch (qtype) {
    case OC_QUANT_F32:    return dequant_f32(src, src_len, dst, value_count);
    case OC_QUANT_F16:    return dequant_f16(src, src_len, dst, value_count);
    case OC_QUANT_BF16:   return dequant_bf16(src, src_len, dst, value_count);
    case OC_QUANT_I8:     return dequant_i8(src, src_len, dst, value_count);
    case OC_QUANT_I16:    return dequant_i16(src, src_len, dst, value_count);
    case OC_QUANT_I32:    return dequant_i32(src, src_len, dst, value_count);
    case OC_QUANT_I64:    return dequant_i64(src, src_len, dst, value_count);
    case OC_QUANT_F64:    return dequant_f64(src, src_len, dst, value_count);
    case OC_QUANT_Q4_0:   return dequant_q4_0(src, src_len, dst, value_count);
    case OC_QUANT_Q4_1:   return dequant_q4_1(src, src_len, dst, value_count);
    case OC_QUANT_Q5_0:   return dequant_q5_0(src, src_len, dst, value_count);
    case OC_QUANT_Q5_1:   return dequant_q5_1(src, src_len, dst, value_count);
    case OC_QUANT_Q8_0:   return dequant_q8_0(src, src_len, dst, value_count);
    case OC_QUANT_Q2_K:   return dequant_q2_k(src, src_len, dst, value_count);
    case OC_QUANT_Q3_K_S:
    case OC_QUANT_Q3_K_M:
    case OC_QUANT_Q3_K_L: return dequant_q3_k(src, src_len, dst, value_count);
    case OC_QUANT_Q4_K_S:
    case OC_QUANT_Q4_K_M: return dequant_q4_k(src, src_len, dst, value_count);
    case OC_QUANT_Q5_K_S:
    case OC_QUANT_Q5_K_M: return dequant_q5_k(src, src_len, dst, value_count);
    case OC_QUANT_Q6_K:   return dequant_q6_k(src, src_len, dst, value_count);
    default:
        /* AL-family / IQ-family / NVFP4 — not implemented in this feature.
         * Unknown types return OC_ERR_QUANT (no crash). */
        return OC_ERR_QUANT;
    }
}

OcError oc_quant_pack_row(OcGgufQuantizationType qtype,
                          const float *src, size_t value_count,
                          uint8_t *dst, size_t dst_len)
{
    if (src == NULL || dst == NULL) return OC_ERR_INVALID_ARG;
    switch (qtype) {
    case OC_QUANT_F32:    return pack_f32(src, value_count, dst, dst_len);
    case OC_QUANT_F16:    return pack_f16(src, value_count, dst, dst_len);
    case OC_QUANT_I8:     return pack_i8(src, value_count, dst, dst_len);
    case OC_QUANT_I16:    return pack_i16(src, value_count, dst, dst_len);
    case OC_QUANT_I32:    return pack_i32(src, value_count, dst, dst_len);
    case OC_QUANT_I64:    return pack_i64(src, value_count, dst, dst_len);
    case OC_QUANT_F64:    return pack_f64(src, value_count, dst, dst_len);
    case OC_QUANT_Q4_0:   return pack_q4_0(src, value_count, dst, dst_len);
    case OC_QUANT_Q4_1:   return pack_q4_1(src, value_count, dst, dst_len);
    case OC_QUANT_Q5_0:   return pack_q5_0(src, value_count, dst, dst_len);
    case OC_QUANT_Q5_1:   return pack_q5_1(src, value_count, dst, dst_len);
    case OC_QUANT_Q8_0:   return pack_q8_0(src, value_count, dst, dst_len);
    case OC_QUANT_Q4_K_S:
    case OC_QUANT_Q4_K_M: return pack_q4_k(src, value_count, dst, dst_len);
    case OC_QUANT_Q2_K:
        return pack_k_packed(src, value_count, dst, dst_len,
                             OC_BLOCK_Q2_K_SIZE, 2, 1.5f);
    case OC_QUANT_Q3_K_S:
    case OC_QUANT_Q3_K_M:
    case OC_QUANT_Q3_K_L:
        return pack_k_packed(src, value_count, dst, dst_len,
                             OC_BLOCK_Q3_K_SIZE, 3, 3.5f);
    case OC_QUANT_Q5_K_S:
    case OC_QUANT_Q5_K_M:
        return pack_k_packed(src, value_count, dst, dst_len,
                             OC_BLOCK_Q5_K_SIZE, 5, 16.0f);
    case OC_QUANT_Q6_K:
        return pack_k_packed(src, value_count, dst, dst_len,
                             OC_BLOCK_Q6_K_SIZE, 6, 32.0f);
    default:
        /* BF16 pack + AL/IQ/NVFP4 pack — added by their features. */
        return OC_ERR_QUANT;
    }
}

OcError oc_quant_pack_block(OcGgufQuantizationType qtype,
                            const float *src, uint8_t *dst)
{
    if (src == NULL || dst == NULL) return OC_ERR_INVALID_ARG;
    OcQuantBlockLayout bs = oc_quant_block_size(qtype);
    if (bs.elements_per_block == 0 || bs.bytes_per_block == 0) {
        return OC_ERR_QUANT;
    }
    return oc_quant_pack_row(qtype, src, bs.elements_per_block, dst, bs.bytes_per_block);
}

/* ─── Naming + ggml-id mapping ────────────────────────────────────────── */

static const struct {
    OcGgufQuantizationType q;
    const char *name;
    uint32_t    ggml_id;
} k_quant_table[] = {
    { OC_QUANT_F32,     "F32",     0  },
    { OC_QUANT_F16,     "F16",     1  },
    { OC_QUANT_BF16,    "BF16",    30 },
    { OC_QUANT_Q4_0,    "Q4_0",    2  },
    { OC_QUANT_Q4_1,    "Q4_1",    3  },
    { OC_QUANT_Q5_0,    "Q5_0",    6  },
    { OC_QUANT_Q5_1,    "Q5_1",    7  },
    { OC_QUANT_Q8_0,    "Q8_0",    8  },
    { OC_QUANT_Q2_K,    "Q2_K",    10 },
    { OC_QUANT_Q3_K_S,  "Q3_K_S",  11 },
    { OC_QUANT_Q3_K_M,  "Q3_K_M",  12 },
    { OC_QUANT_Q3_K_L,  "Q3_K_L",  13 },
    { OC_QUANT_Q4_K_S,  "Q4_K_S",  14 },
    { OC_QUANT_Q4_K_M,  "Q4_K_M",  15 },
    { OC_QUANT_Q5_K_S,  "Q5_K_S",  16 },
    { OC_QUANT_Q5_K_M,  "Q5_K_M",  17 },
    { OC_QUANT_Q6_K,    "Q6_K",    18 },
    /* AL-family (ggml ids 240-243) — names match oxidize-core. */
    { OC_QUANT_AL5,     "AL5",     240 },
    { OC_QUANT_AL5_XS,  "AL5_XS",  241 },
    { OC_QUANT_AL6,     "AL6",     242 },
    { OC_QUANT_AL8,     "AL8",     243 },
    /* IQ-family (ggml ids 24-29, 32-37, etc.). */
    { OC_QUANT_IQ2_XXS, "IQ2_XXS", 24 },
    { OC_QUANT_IQ2_XS,  "IQ2_XS",  25 },
    { OC_QUANT_IQ2_S,   "IQ2_S",   29 },
    { OC_QUANT_IQ3_XXS, "IQ3_XXS", 27 },
    { OC_QUANT_IQ3_S,   "IQ3_S",   35 },
    { OC_QUANT_IQ4_NL,  "IQ4_NL",  28 },
    { OC_QUANT_IQ4_XS,  "IQ4_XS",  32 },
    { OC_QUANT_IQ1_S,   "IQ1_S",   33 },
    { OC_QUANT_IQ1_M,   "IQ1_M",   36 },
    { OC_QUANT_NVFP4,   "NVFP4",   38 },
    /* Plain integer / wide-float storage. Use ggml id 0..7 mapping per the
     * GGUF spec draft for I8/I16/I32/I64/F64 (these are oxidize-c internal
     * extensions; values >= 64 are oxidize-c-local). */
    { OC_QUANT_I8,      "I8",      64 },
    { OC_QUANT_I16,     "I16",     65 },
    { OC_QUANT_I32,     "I32",     66 },
    { OC_QUANT_I64,     "I64",     67 },
    { OC_QUANT_F64,     "F64",     68 },
};

#define K_QUANT_TABLE_LEN \
    (sizeof(k_quant_table) / sizeof(k_quant_table[0]))

const char *oc_quant_type_name(OcGgufQuantizationType qtype)
{
    for (size_t i = 0; i < K_QUANT_TABLE_LEN; i++) {
        if (k_quant_table[i].q == qtype) return k_quant_table[i].name;
    }
    return "?";
}

OcGgufQuantizationType oc_quant_type_from_ggml_id(uint32_t ggml_type)
{
    for (size_t i = 0; i < K_QUANT_TABLE_LEN; i++) {
        if (k_quant_table[i].ggml_id == ggml_type) return k_quant_table[i].q;
    }
    return OC_QUANT_UNKNOWN;
}

uint32_t oc_quant_type_to_ggml_id(OcGgufQuantizationType qtype)
{
    for (size_t i = 0; i < K_QUANT_TABLE_LEN; i++) {
        if (k_quant_table[i].q == qtype) return k_quant_table[i].ggml_id;
    }
    return 0xffffffffu;
}
