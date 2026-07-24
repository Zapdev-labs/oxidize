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
#include "oxidize/simd.h"

/* Bit-exact lookup tables for AL/IQ/NVFP4 dequant (port of
 * oxidize-core/src/compute/quantization{.rs,/iq_grids.rs,
 * /iq1s_grid_fragment.rs, /iq2s_grid_fragment.rs, /iq2xs_grid_fragment.rs}).
 * Generated verbatim from ggml-common.h — do not hand-edit; regenerate via
 * scripts/gen_quant_tables.py. SHA256 of each table matches the Rust source
 * (validated by tests/test_quant.c::al_iq_constant_table_sha256). */
#include "quant_tables.h"

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

/* BF16: bfloat16 format = sign(1) + exponent(8) + mantissa(7).
 * Conversion from f32: truncate the low 16 mantissa bits of the f32,
 * with round-to-nearest-even. */
static uint16_t f32_to_bf16_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    /* Round to nearest even: add 0x7FFF + (LSB of result) before truncating. */
    uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    uint16_t bf16 = (uint16_t)((bits + rounding_bias) >> 16);
    return bf16;
}

static void bf16_le_write(uint8_t *dst, float value)
{
    uint16_t bits = f32_to_bf16_bits(value);
    dst[0] = (uint8_t)(bits & 0xFFu);
    dst[1] = (uint8_t)((bits >> 8) & 0xFFu);
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
    /* AL-family (ggml ids 240-243). AL5/AL6/AL8 reuse the Q4_0/Q5_0/Q8_0
     * block layouts (identical byte layout, different encoder); AL5_XS uses
     * its own 14-byte 3-bit-packed block. Port of
     * oxidize-core/src/compute/quantization.rs::quant_block_layout. */
    case OC_QUANT_AL5:    return (LayoutInfo){ OC_BLOCK_Q4_0_SIZE, OC_QK_AL };
    case OC_QUANT_AL5_XS: return (LayoutInfo){ OC_BLOCK_AL5_XS_SIZE, OC_QK_AL };
    case OC_QUANT_AL6:    return (LayoutInfo){ OC_BLOCK_Q5_0_SIZE, OC_QK_AL };
    case OC_QUANT_AL8:    return (LayoutInfo){ OC_BLOCK_Q8_0_SIZE, OC_QK_AL };
    /* IQ-family — block sizes ported from oxidize-core (all use QK_K=256
     * except IQ4_NL which uses QK4_NL=32, and NVFP4 which uses QK_NVFP4=64). */
    case OC_QUANT_IQ1_S:   return (LayoutInfo){ OC_BLOCK_IQ1_S_SIZE,   OC_QK_K };
    case OC_QUANT_IQ1_M:   return (LayoutInfo){ OC_BLOCK_IQ1_M_SIZE,   OC_QK_K };
    case OC_QUANT_IQ2_XXS: return (LayoutInfo){ OC_BLOCK_IQ2_XXS_SIZE, OC_QK_K };
    case OC_QUANT_IQ2_XS:  return (LayoutInfo){ OC_BLOCK_IQ2_XS_SIZE,  OC_QK_K };
    case OC_QUANT_IQ2_S:   return (LayoutInfo){ OC_BLOCK_IQ2_S_SIZE,   OC_QK_K };
    case OC_QUANT_IQ3_XXS: return (LayoutInfo){ OC_BLOCK_IQ3_XXS_SIZE, OC_QK_K };
    case OC_QUANT_IQ3_S:   return (LayoutInfo){ OC_BLOCK_IQ3_S_SIZE,   OC_QK_K };
    case OC_QUANT_IQ4_NL:  return (LayoutInfo){ OC_BLOCK_IQ4_NL_SIZE,  OC_QK4_NL };
    case OC_QUANT_IQ4_XS:  return (LayoutInfo){ OC_BLOCK_IQ4_XS_SIZE,  OC_QK_K };
    case OC_QUANT_NVFP4:   return (LayoutInfo){ OC_BLOCK_NVFP4_SIZE,   OC_QK_NVFP4 };
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
            /* llama.cpp advances `q` by 32 per 128-element group, and the
             * 2-bit shift restarts at 0 within each group. */
            const uint8_t *q = &qs[outer * 32];
            for (size_t inner = 0; inner < 4; inner++) {
                size_t shift = inner * 2;
                float dl = d_all * (float)((int32_t)(int8_t)scale_bytes[is] - 32);
                is += 1;
                for (size_t l = 0; l < 16; l++) {
                    int32_t qv = (int32_t)((q[l] >> shift) & 3u);
                    int32_t hbit = ((hmask[l] & m) != 0) ? 0 : 4;
                    out[q_ptr + l] = dl * (float)(qv - hbit);
                }
                float dl2 = d_all * (float)((int32_t)(int8_t)scale_bytes[is] - 32);
                is += 1;
                for (size_t l = 0; l < 16; l++) {
                    int32_t qv = (int32_t)((q[l + 16] >> shift) & 3u);
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
            /* Each 64-element group consumes its own 32 low-nibble bytes
             * (llama.cpp advances `ql` by 32 per group); the 32 qh bytes are
             * reused with the u1/u2 bit masks stepping instead. */
            const uint8_t *ql = &qs[outer * 32];
            for (size_t l = 0; l < 32; l++) {
                uint32_t qv1 = (uint32_t)(ql[l] & 0x0Fu)
                             + (((qh[l] & u1) != 0) ? 16u : 0u);
                out[q_ptr + l] = d1 * (float)qv1 - min1;
            }
            for (size_t l = 0; l < 32; l++) {
                uint32_t qv2 = (uint32_t)(ql[l] >> 4)
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

/* ─── AL-family / IQ-family / NVFP4 dequant ────────────────────────────
 *
 * Bit-exact port of:
 *   - oxidize-core/src/compute/quantization/al_family.rs (AL5_XS 3-bit
 *     unpack)
 *   - oxidize-core/src/compute/quantization/quant_simple.rs (AL5 dequant ==
 *     Q4_0 dequant)
 *   - oxidize-core/src/compute/quantization/quant_iq_series.rs (IQ1_S,
 *     IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ4_NL)
 *   - oxidize-core/src/compute/quantization/quant_k_blocks.rs (IQ3_S,
 *     IQ4_XS)
 *   - oxidize-core/src/compute/quantization/quant_nvfp4.rs (NVFP4)
 *
 * Hard parity invariant: output must match Rust `dequantize_row_*` byte-for-
 * byte on the same packed input (VAL-QUANT-006/007/009).
 */

/* AL5 dequant: identical to Q4_0 (split-halves 4-bit nibble block). */
static OcError dequant_al5(const uint8_t *src, size_t src_len,
                           float *dst, size_t value_count)
{
    return dequant_q4_0(src, src_len, dst, value_count);
}

/* AL8 dequant: identical to Q8_0 (signed 8-bit). */
static OcError dequant_al8(const uint8_t *src, size_t src_len,
                          float *dst, size_t value_count)
{
    return dequant_q8_0(src, src_len, dst, value_count);
}

/* AL6 dequant: identical to Q5_0 (5-bit + 1-bit high). */
static OcError dequant_al6(const uint8_t *src, size_t src_len,
                          float *dst, size_t value_count)
{
    return dequant_q5_0(src, src_len, dst, value_count);
}

/* AL5_XS dequant: 3-bit packed levels in 12 bytes (32 levels × 3 bits = 96
 * bits = 12 bytes), preceded by a 2-byte f16 scale `d`. Each level l decodes
 * to (l - 4) * d (range -4..+3). Port of al_family.rs::unpack3bit. */
static OcError dequant_al5_xs(const uint8_t *src, size_t src_len,
                              float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_AL5_XS, src_len, value_count,
                         OC_BLOCK_AL5_XS_SIZE, OC_QK_AL)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_AL5_XS_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_AL5_XS_SIZE;
        float *out = dst + b * OC_QK_AL;
        float d = f16_le_to_f32(block[0], block[1]);
        const uint8_t *packed = &block[2];
        uint32_t bitpos = 0u;
        for (size_t i = 0; i < OC_QK_AL; i++) {
            uint8_t v = 0u;
            for (int bb = 0; bb < 3; bb++) {
                size_t byte_idx = bitpos / 8u;
                size_t bit_idx = bitpos % 8u;
                if ((packed[byte_idx] >> bit_idx) & 1u) {
                    v |= (uint8_t)(1u << bb);
                }
                bitpos += 1u;
            }
            out[i] = (float)((int32_t)v - 4) * d;
        }
    }
    return OC_OK;
}

/* IQ1_S dequant: 20-byte block, 256 values. Port of
 * quant_iq_series.rs::dequantize_iq1_s_scalar. Layout:
 *   d[0..2] (f16), qs[2..34] (32 u8), qh[34..50] (8 u16 LE).
 * Each 32-value sub-block has scale `d * (2*((qh>>12)&7)+1)` and a ±0.125
 * delta from the high bit (qh & 0x8000). 4 sub-blocks × 8 grid entries. */
#define OC_IQ1S_DELTA 0.125f

static void iq1s_grid_decode(uint16_t index, int8_t *out8)
{
    /* Decode 11-bit iq1s_grid index into 8 ternary values {-1, 0, +1} stored
     * as i8. The grid table is u64 little-endian; port of
     * iq_grids.rs::iq1s_grid_decode. */
    uint64_t packed = IQ1S_GRID[index & 0x7FFu];
    for (int i = 0; i < 8; i++) {
        out8[i] = (int8_t)((packed >> (8u * (uint32_t)i)) & 0xFFu);
    }
}

static OcError dequant_iq1_s(const uint8_t *src, size_t src_len,
                            float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_IQ1_S, src_len, value_count,
                         OC_BLOCK_IQ1_S_SIZE, OC_QK_K)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_IQ1_S_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_IQ1_S_SIZE;
        float *out = dst + b * OC_QK_K;
        float d = f16_le_to_f32(block[0], block[1]);
        const uint8_t *qs = &block[2];
        const uint8_t *qh = &block[34];
        uint16_t qh_u16[8];
        for (size_t i = 0; i < 8; i++) {
            qh_u16[i] = (uint16_t)((uint16_t)qh[2 * i]
                                   | ((uint16_t)qh[2 * i + 1] << 8));
        }
        size_t out_ptr = 0;
        for (size_t ib = 0; ib < OC_QK_K / 32; ib++) {
            float dl = d * (2.0f * (float)((qh_u16[ib] >> 12) & 7u) + 1.0f);
            float delta = (qh_u16[ib] & 0x8000u) ? -OC_IQ1S_DELTA : OC_IQ1S_DELTA;
            for (int l = 0; l < 4; l++) {
                uint16_t grid_idx = (uint16_t)qs[l + ib * 4]
                                  | (uint16_t)(((qh_u16[ib] >> (3 * l)) & 7u) << 8);
                int8_t grid_vals[8];
                iq1s_grid_decode(grid_idx, grid_vals);
                for (int j = 0; j < 8; j++) {
                    out[out_ptr + j] = dl * ((float)grid_vals[j] + delta);
                }
                out_ptr += 8;
            }
        }
    }
    return OC_OK;
}

/* IQ1_M dequant: 20-byte block, 256 values. Port of
 * quant_iq_series.rs::dequantize_iq1_m_scalar. Layout:
 *   qs[0..32] (32 u8), qh[32..48] (16 u8), scales[48..56] (8 u8). The d
 * scale is reconstructed from 4 u16 values packed across `scales`:
 *   scale_u16 = (sc0>>12) | ((sc1>>8)&0xf0) | ((sc2>>4)&0xf00) | (sc3&0xf000)
 * Each 32-value sub-block has its own 3-bit scale nibbles packed in
 * `scales[ib/2]` (one nibble pair per ib32). */
static OcError dequant_iq1_m(const uint8_t *src, size_t src_len,
                            float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_IQ1_M, src_len, value_count,
                         OC_BLOCK_IQ1_M_SIZE, OC_QK_K)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_IQ1_M_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_IQ1_M_SIZE;
        float *out = dst + b * OC_QK_K;
        const uint8_t *qs = &block[0];
        const uint8_t *qh = &block[32];
        const uint8_t *scales = &block[48];

        uint16_t sc[4];
        for (size_t i = 0; i < 4; i++) {
            sc[i] = (uint16_t)((uint16_t)scales[2 * i]
                               | ((uint16_t)scales[2 * i + 1] << 8));
        }
        uint16_t scale_u16 = (uint16_t)((sc[0] >> 12)
                                       | ((sc[1] >> 8) & 0x00F0u)
                                       | ((sc[2] >> 4) & 0x0F00u)
                                       | (sc[3] & 0xF000u));
        /* f16_bits_to_f32 — same as f16_le_to_f32 but takes the assembled bits. */
        uint32_t f32_bits;
        {
            uint32_t sign = (uint32_t)((scale_u16 >> 15) & 1u);
            uint32_t exp  = (uint32_t)((scale_u16 >> 10) & 0x1Fu);
            uint32_t frac = (uint32_t)(scale_u16 & 0x03FFu);
            if (exp == 0u) {
                if (frac == 0u) {
                    f32_bits = sign << 31;
                } else {
                    uint32_t fn = frac;
                    int32_t e = -14;
                    while ((fn & 0x0400u) == 0u) { fn <<= 1; e -= 1; }
                    fn &= 0x03FFu;
                    f32_bits = (sign << 31) | (((uint32_t)(e + 127)) << 23) | (fn << 13);
                }
            } else if (exp == 0x1Fu) {
                f32_bits = (sign << 31) | 0x7F800000u | (frac << 13);
            } else {
                int32_t e = (int32_t)exp - 15 + 127;
                f32_bits = (sign << 31) | (((uint32_t)e) << 23) | (frac << 13);
            }
        }
        float d;
        memcpy(&d, &f32_bits, sizeof(d));

        size_t out_ptr = 0;
        for (size_t ib = 0; ib < OC_QK_K / 32; ib++) {
            uint8_t sc_ib = scales[ib / 2];
            /* Note: Rust reads two 3-bit scales from sc_ib at offsets 6*(ib%2)
             * and 6*(ib%2)+3, both masked with 0x7. */
            uint32_t dl1_bits = (sc_ib >> (6u * (ib % 2u))) & 0x7u;
            uint32_t dl2_bits = (sc_ib >> (6u * (ib % 2u) + 3u)) & 0x7u;
            float dl1 = d * (2.0f * (float)dl1_bits + 1.0f);
            float dl2 = d * (2.0f * (float)dl2_bits + 1.0f);

            uint16_t idx[4];
            idx[0] = (uint16_t)qs[ib * 4]     | (uint16_t)((qh[ib * 2]     << 8) & 0x0700u);
            idx[1] = (uint16_t)qs[ib * 4 + 1] | (uint16_t)((qh[ib * 2]     << 4) & 0x0700u);
            idx[2] = (uint16_t)qs[ib * 4 + 2] | (uint16_t)((qh[ib * 2 + 1] << 8) & 0x0700u);
            idx[3] = (uint16_t)qs[ib * 4 + 3] | (uint16_t)((qh[ib * 2 + 1] << 4) & 0x0700u);

            float dl_arr[4] = { dl1, dl1, dl2, dl2 };
            float delta_arr[4];
            delta_arr[0] = (qh[ib * 2]     & 0x08u) ? -OC_IQ1S_DELTA : OC_IQ1S_DELTA;
            delta_arr[1] = (qh[ib * 2]     & 0x80u) ? -OC_IQ1S_DELTA : OC_IQ1S_DELTA;
            delta_arr[2] = (qh[ib * 2 + 1] & 0x08u) ? -OC_IQ1S_DELTA : OC_IQ1S_DELTA;
            delta_arr[3] = (qh[ib * 2 + 1] & 0x80u) ? -OC_IQ1S_DELTA : OC_IQ1S_DELTA;

            for (int l = 0; l < 4; l++) {
                int8_t grid_vals[8];
                iq1s_grid_decode(idx[l], grid_vals);
                for (int j = 0; j < 8; j++) {
                    out[out_ptr + j] = dl_arr[l] * ((float)grid_vals[j] + delta_arr[l]);
                }
                out_ptr += 8;
            }
        }
    }
    return OC_OK;
}

/* IQ2_XXS dequant: 18-byte block, 256 values. Port of
 * quant_iq_series.rs::dequantize_iq2_xxs_scalar. Layout:
 *   d[0..2] (f16), qs[2..18] (16 bytes = 4 ib32 × 4 bytes). Each ib32 has
 *   8 bytes: aux0 (4 bytes, 4 grid indices) + aux1 (4 bytes, 4 7-bit sign
 *   selectors + 4-bit scale). */
static OcError dequant_iq2_xxs(const uint8_t *src, size_t src_len,
                              float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_IQ2_XXS, src_len, value_count,
                         OC_BLOCK_IQ2_XXS_SIZE, OC_QK_K)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_IQ2_XXS_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_IQ2_XXS_SIZE;
        float *out = dst + b * OC_QK_K;
        float d = f16_le_to_f32(block[0], block[1]);
        const uint8_t *qs = &block[2];
        size_t out_ptr = 0;
        for (size_t ib32 = 0; ib32 < OC_QK_K / 32; ib32++) {
            uint32_t aux0 = (uint32_t)qs[4 * ib32]
                          | ((uint32_t)qs[4 * ib32 + 1] << 8)
                          | ((uint32_t)qs[4 * ib32 + 2] << 16)
                          | ((uint32_t)qs[4 * ib32 + 3] << 24);
            uint32_t aux1 = (uint32_t)qs[4 * ib32 + 4]
                          | ((uint32_t)qs[4 * ib32 + 5] << 8)
                          | ((uint32_t)qs[4 * ib32 + 6] << 16)
                          | ((uint32_t)qs[4 * ib32 + 7] << 24);
            uint8_t aux_bytes[4];
            aux_bytes[0] = (uint8_t)(aux0 & 0xFFu);
            aux_bytes[1] = (uint8_t)((aux0 >> 8) & 0xFFu);
            aux_bytes[2] = (uint8_t)((aux0 >> 16) & 0xFFu);
            aux_bytes[3] = (uint8_t)((aux0 >> 24) & 0xFFu);
            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                size_t grid_idx = (size_t)aux_bytes[l];
                uint64_t grid_u64 = IQ2XXS_GRID[grid_idx];
                uint8_t grid[8];
                for (int j = 0; j < 8; j++) {
                    grid[j] = (uint8_t)((grid_u64 >> (8u * (uint32_t)j)) & 0xFFu);
                }
                uint8_t signs = KSIGNS_IQ2XS[(aux1 >> (7 * l)) & 127u];
                for (int j = 0; j < 8; j++) {
                    float sign = (signs & KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
                    out[out_ptr + j] = db * (float)grid[j] * sign;
                }
                out_ptr += 8;
            }
        }
    }
    return OC_OK;
}

/* IQ2_XS dequant: 74-byte block, 256 values. Port of
 * quant_iq_series.rs::dequantize_iq2_xs_scalar. Layout:
 *   d[0..2] (f16), qs[2..66] (64 bytes = 8 ib32 × 4 × 2-byte u16 indices),
 *   scales[66..74] (8 bytes, one per ib32). Each ib32 has 4 grid lookups;
 *   scale nibbles sc&0xf and sc>>4 split the 4 lookups into 2 dl values. */
static OcError dequant_iq2_xs(const uint8_t *src, size_t src_len,
                             float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_IQ2_XS, src_len, value_count,
                         OC_BLOCK_IQ2_XS_SIZE, OC_QK_K)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_IQ2_XS_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_IQ2_XS_SIZE;
        float *out = dst + b * OC_QK_K;
        float d = f16_le_to_f32(block[0], block[1]);
        size_t qs_base = 2;
        const uint8_t *scales = &block[2 + OC_QK_K / 4];
        size_t out_ptr = 0;
        for (size_t ib32 = 0; ib32 < OC_QK_K / 32; ib32++) {
            uint8_t sc = scales[ib32];
            float db[2];
            db[0] = d * (0.5f + (float)(sc & 0x0Fu)) * 0.25f;
            db[1] = d * (0.5f + (float)(sc >> 4)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                size_t qs_off = qs_base + 2 * (4 * ib32 + l);
                uint16_t qs_val = (uint16_t)((uint16_t)block[qs_off]
                                              | ((uint16_t)block[qs_off + 1] << 8));
                uint64_t grid_u64 = IQ2XS_GRID[(size_t)(qs_val & 0x1FFu)];
                uint8_t grid[8];
                for (int j = 0; j < 8; j++) {
                    grid[j] = (uint8_t)((grid_u64 >> (8u * (uint32_t)j)) & 0xFFu);
                }
                uint8_t signs = KSIGNS_IQ2XS[(size_t)(qs_val >> 9)];
                float dl = db[l / 2];
                for (int j = 0; j < 8; j++) {
                    float sign = (signs & KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
                    out[out_ptr + j] = dl * (float)grid[j] * sign;
                }
                out_ptr += 8;
            }
        }
    }
    return OC_OK;
}

/* IQ2_S dequant: 82-byte block, 256 values. Port of
 * quant_iq_series.rs::dequantize_iq2_s_scalar. Layout:
 *   d[0..2] (f16), qs[2..66] (64 bytes = QK_K/4), qh[66..74] (8 bytes =
 *   QK_K/32), scales[74..82] (8 bytes = QK_K/32). Signs are stored in the
 *   qs buffer at offsets QK_K/8..QK_K/4 (i.e. the second half of qs). */
static OcError dequant_iq2_s(const uint8_t *src, size_t src_len,
                            float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_IQ2_S, src_len, value_count,
                         OC_BLOCK_IQ2_S_SIZE, OC_QK_K)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_IQ2_S_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_IQ2_S_SIZE;
        float *out = dst + b * OC_QK_K;
        float d = f16_le_to_f32(block[0], block[1]);
        const uint8_t *qs = &block[2];
        const uint8_t *qh = &block[2 + OC_QK_K / 4];
        const uint8_t *scales = &block[2 + OC_QK_K / 4 + OC_QK_K / 32];
        size_t qs_ptr = 0;
        size_t signs_ptr = OC_QK_K / 8;
        size_t out_ptr = 0;
        for (size_t ib32 = 0; ib32 < OC_QK_K / 32; ib32++) {
            uint8_t sc = scales[ib32];
            float db[2];
            db[0] = d * (0.5f + (float)(sc & 0x0Fu)) * 0.25f;
            db[1] = d * (0.5f + (float)(sc >> 4)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                float dl = db[l / 2];
                size_t grid_idx = (size_t)qs[qs_ptr + l]
                                | (((size_t)qh[ib32] << (8 - 2 * l)) & 0x300u);
                uint64_t grid_u64 = IQ2S_GRID[grid_idx];
                uint8_t grid[8];
                for (int j = 0; j < 8; j++) {
                    grid[j] = (uint8_t)((grid_u64 >> (8u * (uint32_t)j)) & 0xFFu);
                }
                uint8_t signs = qs[signs_ptr + l];
                for (int j = 0; j < 8; j++) {
                    float sign = (signs & KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
                    out[out_ptr + j] = dl * (float)grid[j] * sign;
                }
                out_ptr += 8;
            }
            qs_ptr += 4;
            signs_ptr += 4;
        }
    }
    return OC_OK;
}

/* IQ3_XXS dequant: 98-byte block, 256 values. Port of
 * quant_iq_series.rs::dequantize_iq3_xxs_scalar. Layout:
 *   d[0..2] (f16), qs[2..66] (64 bytes = QK_K/4), scales_and_signs[66..98]
 *   (32 bytes = 3 * (QK_K/8) — actually 8 ib32 × 4 bytes = 32 bytes).
 *   Each ib32 reads 8 qs bytes (4 grid pairs × 2 bytes) and 4 bytes of
 *   scales/signs (aux32). */
static OcError dequant_iq3_xxs(const uint8_t *src, size_t src_len,
                              float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_IQ3_XXS, src_len, value_count,
                         OC_BLOCK_IQ3_XXS_SIZE, OC_QK_K)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_IQ3_XXS_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_IQ3_XXS_SIZE;
        float *out = dst + b * OC_QK_K;
        float d = f16_le_to_f32(block[0], block[1]);
        const uint8_t *qs = &block[2];
        const uint8_t *scales_and_signs = &block[2 + OC_QK_K / 4];
        size_t qs_ptr = 0;
        size_t out_ptr = 0;
        for (size_t ib32 = 0; ib32 < OC_QK_K / 32; ib32++) {
            uint32_t aux32 = (uint32_t)scales_and_signs[4 * ib32]
                           | ((uint32_t)scales_and_signs[4 * ib32 + 1] << 8)
                           | ((uint32_t)scales_and_signs[4 * ib32 + 2] << 16)
                           | ((uint32_t)scales_and_signs[4 * ib32 + 3] << 24);
            float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; l++) {
                uint8_t signs = KSIGNS_IQ2XS[(aux32 >> (7 * l)) & 127u];
                uint32_t g1 = IQ3XXS_GRID[(size_t)qs[qs_ptr + 2 * l]];
                uint32_t g2 = IQ3XXS_GRID[(size_t)qs[qs_ptr + 2 * l + 1]];
                for (int j = 0; j < 4; j++) {
                    float sign_lo = (signs & KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
                    float sign_hi = (signs & KMASK_IQ2XS[j + 4]) ? -1.0f : 1.0f;
                    uint8_t v1 = (uint8_t)((g1 >> (8u * (uint32_t)j)) & 0xFFu);
                    uint8_t v2 = (uint8_t)((g2 >> (8u * (uint32_t)j)) & 0xFFu);
                    out[out_ptr + j]     = db * (float)v1 * sign_lo;
                    out[out_ptr + j + 4] = db * (float)v2 * sign_hi;
                }
                out_ptr += 8;
            }
            qs_ptr += 8;
        }
    }
    return OC_OK;
}

/* IQ3_S dequant: 110-byte block, 256 values. Port of
 * quant_k_blocks.rs::dequantize_iq3_s_scalar. Layout:
 *   d[0..2] (f16), qs[2..66] (64 bytes = QK_K/4), qh[66..74] (8 bytes =
 *   QK_K/32), signs[74..106] (32 bytes = QK_K/8), scales[106..110] (4 bytes
 *   = QK_K/64). Each ib32 pair shares one scales byte (low/high nibble). */
static OcError dequant_iq3_s(const uint8_t *src, size_t src_len,
                             float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_IQ3_S, src_len, value_count,
                         OC_BLOCK_IQ3_S_SIZE, OC_QK_K)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_IQ3_S_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_IQ3_S_SIZE;
        float *out = dst + b * OC_QK_K;
        float d = f16_le_to_f32(block[0], block[1]);
        const uint8_t *qs = &block[2];        /* 64 bytes */
        const uint8_t *qh = &block[66];       /* 8 bytes */
        const uint8_t *signs = &block[74];    /* 32 bytes */
        const uint8_t *scales = &block[106];   /* 4 bytes */
        size_t qs_o = 0;
        size_t qh_o = 0;
        size_t sg_o = 0;
        size_t y = 0;
        size_t ib32 = 0;
        while (ib32 < OC_QK_K / 32) {
            float db1 = d * (float)(1 + 2 * (int32_t)(scales[ib32 / 2] & 0x0Fu));
            float db2 = d * (float)(1 + 2 * (int32_t)(scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; l++) {
                size_t h = (size_t)qh[qh_o];
                size_t i1 = (size_t)qs[qs_o + 2 * l]
                          | ((h << (8 - 2 * l)) & 256u);
                size_t i2 = (size_t)qs[qs_o + 2 * l + 1]
                          | ((h << (7 - 2 * l)) & 256u);
                uint8_t s = signs[sg_o + l];
                uint32_t g1 = IQ3S_GRID[i1];
                uint32_t g2 = IQ3S_GRID[i2];
                for (int j = 0; j < 4; j++) {
                    float f1 = (s & KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
                    float f2 = (s & KMASK_IQ2XS[j + 4]) ? -1.0f : 1.0f;
                    uint8_t v1 = (uint8_t)((g1 >> (8u * (uint32_t)j)) & 0xFFu);
                    uint8_t v2 = (uint8_t)((g2 >> (8u * (uint32_t)j)) & 0xFFu);
                    out[y + j]     = db1 * (float)v1 * f1;
                    out[y + j + 4] = db1 * (float)v2 * f2;
                }
                y += 8;
            }
            qs_o += 8;
            sg_o += 4;
            for (int l = 0; l < 4; l++) {
                size_t h = (size_t)qh[qh_o + 1];
                size_t i1 = (size_t)qs[qs_o + 2 * l]
                          | ((h << (8 - 2 * l)) & 256u);
                size_t i2 = (size_t)qs[qs_o + 2 * l + 1]
                          | ((h << (7 - 2 * l)) & 256u);
                uint8_t s = signs[sg_o + l];
                uint32_t g1 = IQ3S_GRID[i1];
                uint32_t g2 = IQ3S_GRID[i2];
                for (int j = 0; j < 4; j++) {
                    float f1 = (s & KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
                    float f2 = (s & KMASK_IQ2XS[j + 4]) ? -1.0f : 1.0f;
                    uint8_t v1 = (uint8_t)((g1 >> (8u * (uint32_t)j)) & 0xFFu);
                    uint8_t v2 = (uint8_t)((g2 >> (8u * (uint32_t)j)) & 0xFFu);
                    out[y + j]     = db2 * (float)v1 * f1;
                    out[y + j + 4] = db2 * (float)v2 * f2;
                }
                y += 8;
            }
            qh_o += 2;
            qs_o += 8;
            sg_o += 4;
            ib32 += 2;
        }
    }
    return OC_OK;
}

/* IQ4_NL dequant: 18-byte block, 32 values. Port of
 * quant_iq_series.rs::dequantize_iq4_nl_scalar. Layout:
 *   d[0..2] (f16), qs[2..18] (16 bytes = QK4_NL/2 nibbles). Each nibble
 *   indexes the 16-entry KVALUES_IQ4NL codebook (nonlinear 4-bit). */
static OcError dequant_iq4_nl(const uint8_t *src, size_t src_len,
                             float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_IQ4_NL, src_len, value_count,
                         OC_BLOCK_IQ4_NL_SIZE, OC_QK4_NL)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_IQ4_NL_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_IQ4_NL_SIZE;
        float *out = dst + b * OC_QK4_NL;
        float d = f16_le_to_f32(block[0], block[1]);
        const uint8_t *qs = &block[2];
        for (size_t j = 0; j < OC_QK4_NL / 2; j++) {
            uint8_t packed = qs[j];
            out[j] = d * (float)KVALUES_IQ4NL[(size_t)(packed & 0x0Fu)];
            out[j + OC_QK4_NL / 2] = d * (float)KVALUES_IQ4NL[(size_t)(packed >> 4)];
        }
    }
    return OC_OK;
}

/* IQ4_XS dequant: 136-byte block, 256 values. Port of
 * quant_k_blocks.rs::dequantize_iq4_xs_scalar. Layout:
 *   d[0..2] (f16), scales_h[2..4] (u16 LE), scales_l[4..8] (4 bytes =
 *   QK_K/64), qs[8..136] (128 bytes = QK_K/2 nibbles). Each of 8 ib32
 *   sub-blocks has a 6-bit signed scale (ls_l 4 bits + ls_h 2 bits) and 16
 *   nibble indices into KVALUES_IQ4NL. */
static OcError dequant_iq4_xs(const uint8_t *src, size_t src_len,
                             float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_IQ4_XS, src_len, value_count,
                         OC_BLOCK_IQ4_XS_SIZE, OC_QK_K)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_IQ4_XS_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_IQ4_XS_SIZE;
        float *out = dst + b * OC_QK_K;
        float d = f16_le_to_f32(block[0], block[1]);
        uint16_t scales_h = (uint16_t)((uint16_t)block[2]
                                        | ((uint16_t)block[3] << 8));
        const uint8_t *scales_l = &block[4];
        const uint8_t *qs = &block[8];
        for (size_t ib = 0; ib < OC_QK_K / 32; ib++) {
            int32_t ls_l = (int32_t)((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0Fu);
            int32_t ls_h = (int32_t)(((scales_h >> (2 * ib)) & 3u)) << 4;
            float dl = d * (float)((ls_l | ls_h) - 32);
            size_t qoff = ib * 16;
            size_t ooff = ib * 32;
            for (size_t j = 0; j < 16; j++) {
                uint8_t bv = qs[qoff + j];
                out[ooff + j]      = dl * (float)KVALUES_IQ4NL[(size_t)(bv & 0x0Fu)];
                out[ooff + j + 16] = dl * (float)KVALUES_IQ4NL[(size_t)(bv >> 4)];
            }
        }
    }
    return OC_OK;
}

/* NVFP4 dequant: 36-byte block, 64 values. Port of
 * quant_nvfp4.rs::dequantize_nvfp4_scalar. Layout:
 *   scales[0..4] (4 UE4M3 scales, one per 16-value sub-block), qs[4..36]
 *   (32 bytes = QK_NVFP4/2 packed E2M1 nibbles). Each nibble indexes
 *   E2M1_DOUBLED_VALUES[16] (nonlinear 4-bit codebook). */
static float ue4m3_to_f32(uint8_t byte)
{
    /* Port of quant_nvfp4.rs::ue4m3_to_f32. Unsigned 4-bit exponent + 3-bit
     * mantissa (E4M3 with no sign bit). */
    uint32_t exp = (uint32_t)((byte >> 3) & 0x0Fu);
    uint32_t mant = (uint32_t)(byte & 0x07u);
    if (exp == 0u) {
        return (float)mant * 0.001953125f;  /* 2^-9 */
    }
    /* (1 + mant/8) * 2^(exp - 7) */
    return (1.0f + (float)mant / 8.0f) * ldexpf(1.0f, (int)exp - 7);
}

static OcError dequant_nvfp4(const uint8_t *src, size_t src_len,
                            float *dst, size_t value_count)
{
    if (!validate_layout(OC_QUANT_NVFP4, src_len, value_count,
                         OC_BLOCK_NVFP4_SIZE, OC_QK_NVFP4)) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n_blocks = src_len / OC_BLOCK_NVFP4_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * OC_BLOCK_NVFP4_SIZE;
        float *out = dst + b * OC_QK_NVFP4;
        const uint8_t *scales = &block[0];
        const uint8_t *qs = &block[OC_QK_NVFP4 / OC_QK_NVFP4_SUB];
        for (size_t sub = 0; sub < OC_QK_NVFP4 / OC_QK_NVFP4_SUB; sub++) {
            float scale = ue4m3_to_f32(scales[sub]);
            size_t base_q = sub * (OC_QK_NVFP4_SUB / 2);
            size_t base_out = sub * OC_QK_NVFP4_SUB;
            for (size_t j = 0; j < OC_QK_NVFP4_SUB / 2; j++) {
                uint8_t packed = qs[base_q + j];
                out[base_out + j] = scale * E2M1_DOUBLED_VALUES[(size_t)(packed & 0x0Fu)];
                out[base_out + j + OC_QK_NVFP4_SUB / 2]
                    = scale * E2M1_DOUBLED_VALUES[(size_t)(packed >> 4)];
            }
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

static OcError pack_bf16(const float *src, size_t value_count,
                         uint8_t *dst, size_t dst_len)
{
    if (dst_len != value_count * 2) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < value_count; i++) {
        bf16_le_write(&dst[2 * i], src[i]);
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

/* llama.cpp `make_qkx1_quants` — affine (scale + min) fit for the
 * K-family packers. `nmax` is the top quantized level: 3 for Q2_K,
 * 15 for Q4_K, 31 for Q5_K. */
static float make_qkx1_quants(const float *x, size_t n, uint8_t *l,
                             float *out_min, size_t nmax, int ntry, float alpha)
{
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
                                          &l[32 * j], &mins[j], 15, 5, 0.5f);
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

/* ─── AL-family pack (MSE-optimized encoders, port of al_family.rs) ──── */

/* al_refine_scale: least-squares optimal d for fixed integer grid [lo,hi].
 * Port of al_family.rs::al_refine_scale. */
static float al_refine_scale(const float *block, size_t n, float d, int lo, int hi)
{
    if (d == 0.0f) return 0.0f;
    float inv_d = 1.0f / d;
    float sumlx = 0.0f, suml2 = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float v = block[i] * inv_d;
        float l = roundf(v);
        if (l < (float)lo) l = (float)lo;
        if (l > (float)hi) l = (float)hi;
        sumlx += block[i] * l;
        suml2 += l * l;
    }
    if (suml2 > 0.0f) return sumlx / suml2;
    return d;
}

static void block_amax_mx(const float *block, size_t n, float *amax, float *mx)
{
    float a = 0.0f, m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float av = fabsf(block[i]);
        if (av > a) { a = av; m = block[i]; }
    }
    *amax = a;
    *mx = m;
}

/* pack3bit: pack 32 3-bit levels into 12 bytes (96 bits). Port of
 * al_family.rs::pack3bit. Levels are written LSB-first per level. */
static void pack3bit(const uint8_t *levels, uint8_t *out)
{
    for (size_t i = 0; i < 12; i++) out[i] = 0u;
    uint32_t bitpos = 0u;
    for (size_t i = 0; i < OC_QK_AL; i++) {
        uint8_t l = levels[i];
        for (int b = 0; b < 3; b++) {
            if ((l >> b) & 1u) {
                size_t byte_idx = bitpos / 8u;
                size_t bit_idx = bitpos % 8u;
                out[byte_idx] |= (uint8_t)(1u << bit_idx);
            }
            bitpos += 1u;
        }
    }
}

/* AL5 pack: MSE-optimal 4-bit packing using make_qx_quants-style grid
 * search. Port of quant_simple.rs::quantize_block_al5 (no imatrix). Output
 * is the same 18-byte Q4_0 layout. */
static void pack_block_al5(const float *in_block, uint8_t *out_block)
{
    float amax, mx;
    block_amax_mx(in_block, OC_QK_AL, &amax, &mx);
    if (amax == 0.0f) {
        f16_le_write(&out_block[0], 0.0f);
        for (size_t i = 2; i < OC_BLOCK_Q4_0_SIZE; i++) out_block[i] = 0x88u;
        return;
    }
    float best_d = mx / -8.0f;
    int32_t best_levels[OC_QK_AL];
    for (size_t i = 0; i < OC_QK_AL; i++) best_levels[i] = 0;
    float best_obj = -1.0f;
    int32_t levels[OC_QK_AL];
    for (int is = -9; is <= 9; is++) {
        float iscale = -(8.0f + 0.1f * (float)is) / mx;
        float sumlx = 0.0f, suml2 = 0.0f;
        for (size_t i = 0; i < OC_QK_AL; i++) {
            int32_t l = (int32_t)roundf(in_block[i] * iscale);
            if (l < -8) l = -8;
            if (l > 7) l = 7;
            levels[i] = l;
            float w = in_block[i] * in_block[i];
            float lf = (float)l;
            sumlx += w * in_block[i] * lf;
            suml2 += w * lf * lf;
        }
        if (suml2 > 0.0f) {
            float obj = sumlx * sumlx / suml2;
            if (obj > best_obj) {
                best_obj = obj;
                best_d = sumlx / suml2;
                for (size_t i = 0; i < OC_QK_AL; i++) best_levels[i] = levels[i];
            }
        }
    }
    f16_le_write(&out_block[0], best_d);
    for (size_t i = 0; i < 16; i++) {
        uint8_t lo = (uint8_t)(best_levels[i] + 8);
        uint8_t hi = (uint8_t)(best_levels[i + 16] + 8);
        out_block[2 + i] = (uint8_t)(lo | (hi << 4));
    }
}

/* ─── K-quant pack encoders (Q2_K, Q3_K, Q5_K, Q6_K) ────────────────── */

/* Value of the block element with the largest magnitude (sign preserved). */
static float extreme_value(const float *x, size_t n)
{
    float best = 0.0f;
    float amax = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; best = x[i]; }
    }
    return best;
}


/* Q2_K pack: 2-bit quantization with per-sub-block scale + min.
 * Block layout (84 bytes): 16 bytes scales, 64 bytes 2-bit qs, 4 bytes f16 d/min. */
static OcError pack_q2_k(const float *src, size_t value_count,
                         uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK_K != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK_K) * OC_BLOCK_Q2_K_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK_K;

    /* Q2_K carries 16 scale/min nibble pairs, one per 16 elements — matching
     * dequant_q2_k, which consumes two scale bytes per 32-element step. */
    const size_t n_sub = OC_QK_K / 16;   /* 16 sub-blocks of 16 */

    for (size_t b = 0; b < n_blocks; b++) {
        const float *in_block = src + b * OC_QK_K;
        uint8_t *out_block = dst + b * OC_BLOCK_Q2_K_SIZE;
        memset(out_block, 0, OC_BLOCK_Q2_K_SIZE);

        uint8_t l[OC_QK_K];
        float scales[OC_QK_K / 16];
        float mins[OC_QK_K / 16];

        float max_scale = 0.0f, max_min = 0.0f;
        for (size_t i = 0; i < n_sub; i++) {
            scales[i] = make_qkx1_quants(&in_block[16 * i], 16,
                                         &l[16 * i], &mins[i], 3, 5, 0.5f);
            if (scales[i] > max_scale) max_scale = scales[i];
            if (mins[i] > max_min) max_min = mins[i];
        }

        /* Scale and min each get a 4-bit field, so the super-block scales
         * are max/15. */
        float d    = max_scale / 15.0f;
        float dmin = max_min / 15.0f;
        f16_le_write(&out_block[80], d);
        f16_le_write(&out_block[82], dmin);
        /* Re-read what f16 rounding actually stored, so the value loop
         * quantizes against the scales the decoder will see. */
        d    = f16_le_to_f32(out_block[80], out_block[81]);
        dmin = f16_le_to_f32(out_block[82], out_block[83]);

        for (size_t i = 0; i < n_sub; i++) {
            int32_t ls = (d > 0.0f) ? nearest_int_f(scales[i] / d) : 0;
            if (ls < 0) ls = 0;
            if (ls > 15) ls = 15;
            int32_t lm = (dmin > 0.0f) ? nearest_int_f(mins[i] / dmin) : 0;
            if (lm < 0) lm = 0;
            if (lm > 15) lm = 15;
            out_block[i] = (uint8_t)(ls | (lm << 4));
        }

        /* Pack 2-bit values into the 64-byte qs area. Sub-block i sits at
         * qs[outer * 32 + half * 16 + t] with a (inner * 2) bit shift, where
         * k = i / 2, half = i % 2, outer = k / 4, inner = k % 4 — the exact
         * addressing dequant_q2_k walks. */
        uint8_t *qs = &out_block[16];
        for (size_t i = 0; i < n_sub; i++) {
            float dl = d * (float)(out_block[i] & 0x0Fu);
            float ml = dmin * (float)(out_block[i] >> 4);
            size_t k = i / 2, half = i % 2;
            size_t qs_base = (k / 4) * 32 + half * 16;
            size_t shift = (k % 4) * 2;
            for (size_t t = 0; t < 16; t++) {
                int32_t q = 0;
                if (dl > 0.0f) {
                    q = nearest_int_f((in_block[16 * i + t] + ml) / dl);
                    if (q < 0) q = 0;
                    if (q > 3) q = 3;
                }
                qs[qs_base + t] |= (uint8_t)((uint32_t)q << shift);
            }
        }
    }
    return OC_OK;
}

/* Q3_K pack: 3-bit quantization with per-sub-block scale.
 * Block layout (110 bytes): 32 bytes hmask, 64 bytes 3-bit qs, 12 bytes scales, 2 bytes f16 d. */
/* Write the 16 six-bit Q3_K sub-block scales into the 12-byte packed field,
 * as the exact inverse of the unpacking in dequant_q3_k. Sub-block i lands in
 * group g = i / 4 and lane k = i % 4:
 *   g 0: low nibble of S[k],     high 2 bits at S[8+k] bits 0-1
 *   g 1: low nibble of S[4+k],   high 2 bits at S[8+k] bits 2-3
 *   g 2: high nibble of S[k],    high 2 bits at S[8+k] bits 4-5
 *   g 3: high nibble of S[4+k],  high 2 bits at S[8+k] bits 6-7 */
static void q3k_write_scales(uint8_t *s12, const int32_t *mult, size_t n_sub)
{
    memset(s12, 0, 12);
    for (size_t i = 0; i < n_sub; i++) {
        uint32_t v = (uint32_t)(mult[i] + 32) & 0x3Fu;   /* 6-bit, bias 32 */
        uint32_t lo = v & 0x0Fu;
        uint32_t hi = (v >> 4) & 0x03u;
        size_t k = i % 4, g = i / 4;
        switch (g) {
        case 0: s12[k]     |= (uint8_t)lo;         break;
        case 1: s12[4 + k] |= (uint8_t)lo;         break;
        case 2: s12[k]     |= (uint8_t)(lo << 4);  break;
        default: s12[4 + k] |= (uint8_t)(lo << 4); break;
        }
        s12[8 + k] |= (uint8_t)(hi << (2 * g));
    }
}

static OcError pack_q3_k(const float *src, size_t value_count,
                         uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK_K != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK_K) * OC_BLOCK_Q3_K_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK_K;

    /* 16 sub-blocks of 16, each with a 6-bit signed scale multiplier.
     * Levels are 3-bit signed: qs holds a 2-bit magnitude and hmask the
     * high bit, reconstructing q in [-4, 3]. */
    const size_t n_sub = OC_QK_K / 16;

    for (size_t b = 0; b < n_blocks; b++) {
        const float *in_block = src + b * OC_QK_K;
        uint8_t *out_block = dst + b * OC_BLOCK_Q3_K_SIZE;
        memset(out_block, 0, OC_BLOCK_Q3_K_SIZE);

        /* Place each sub-block's extreme element on level -4 (the widest
         * side of the signed range) so nothing clips, whatever its sign. */
        float sub_scale[OC_QK_K / 16];
        float max_abs = 0.0f;
        for (size_t i = 0; i < n_sub; i++) {
            float ext = extreme_value(&in_block[16 * i], 16);
            sub_scale[i] = ext / -4.0f;
            float a = fabsf(sub_scale[i]);
            if (a > max_abs) max_abs = a;
        }

        float d_all = max_abs / 31.0f;
        int32_t mult[OC_QK_K / 16];
        for (size_t i = 0; i < n_sub; i++) {
            int32_t m = (d_all > 0.0f) ? nearest_int_f(sub_scale[i] / d_all) : 0;
            if (m < -32) m = -32;
            if (m > 31) m = 31;
            mult[i] = m;
        }

        f16_le_write(&out_block[108], d_all);
        d_all = f16_le_to_f32(out_block[108], out_block[109]);
        q3k_write_scales(&out_block[96], mult, n_sub);

        uint8_t *hmask = &out_block[0];
        uint8_t *qs = &out_block[32];
        for (size_t i = 0; i < n_sub; i++) {
            float dl = d_all * (float)mult[i];
            /* Addressing mirrors dequant_q3_k: k = i / 2, half = i % 2,
             * outer = k / 4, inner = k % 4. */
            size_t k = i / 2, half = i % 2;
            size_t outer = k / 4, inner = k % 4;
            size_t qs_base = outer * 32 + half * 16;
            size_t shift = inner * 2;
            uint8_t m = (uint8_t)(1u << k);
            for (size_t t = 0; t < 16; t++) {
                int32_t q = 0;
                if (dl != 0.0f) {
                    q = nearest_int_f(in_block[16 * i + t] / dl);
                    if (q < -4) q = -4;
                    if (q > 3) q = 3;
                }
                /* The decoder computes q = qv - (hmask bit set ? 0 : 4), so a
                 * set bit means a non-negative level. */
                uint32_t qv = (q >= 0) ? (uint32_t)q : (uint32_t)(q + 4);
                if (q >= 0) hmask[half * 16 + t] |= m;
                qs[qs_base + t] |= (uint8_t)((qv & 3u) << shift);
            }
        }
    }
    return OC_OK;
}

/* Q5_K pack: 5-bit quantization with per-sub-block scale + min.
 * Block layout (176 bytes): 4 bytes f16 d/min, 12 bytes scales, 32 bytes qh, 128 bytes qs. */
static OcError pack_q5_k(const float *src, size_t value_count,
                         uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK_K != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK_K) * OC_BLOCK_Q5_K_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK_K;

    /* Same shape as Q4_K — 8 sub-blocks of 32 with 6-bit scale/min pairs —
     * but 5-bit levels: the low 4 bits live in qs, the 5th in qh. */
    const size_t n_sub = OC_QK_K / 32;

    for (size_t b = 0; b < n_blocks; b++) {
        const float *in_block = src + b * OC_QK_K;
        uint8_t *out_block = dst + b * OC_BLOCK_Q5_K_SIZE;
        memset(out_block, 0, OC_BLOCK_Q5_K_SIZE);

        uint8_t l[OC_QK_K];
        float scales[OC_QK_K / 32];
        float mins[OC_QK_K / 32];

        float max_scale = 0.0f, max_min = 0.0f;
        for (size_t j = 0; j < n_sub; j++) {
            scales[j] = make_qkx1_quants(&in_block[32 * j], 32,
                                         &l[32 * j], &mins[j], 31, 5, 0.5f);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }

        float inv_scale = (max_scale > 0.0f) ? (63.0f / max_scale) : 0.0f;
        float inv_min   = (max_min   > 0.0f) ? (63.0f / max_min)   : 0.0f;

        /* 6-bit scale/min fields, packed exactly as get_scale_min_k4 reads. */
        for (size_t j = 0; j < n_sub; j++) {
            int32_t ls = nearest_int_f(inv_scale * scales[j]);
            if (ls < 0) ls = 0;
            if (ls > 63) ls = 63;
            int32_t lm = nearest_int_f(inv_min * mins[j]);
            if (lm < 0) lm = 0;
            if (lm > 63) lm = 63;
            if (j < 4) {
                out_block[4 + j]     = (uint8_t)ls;
                out_block[4 + j + 4] = (uint8_t)lm;
            } else {
                out_block[4 + j + 4]  = (uint8_t)((ls & 0x0Fu) | ((lm & 0x0Fu) << 4));
                out_block[4 + j - 4] |= (uint8_t)((ls >> 4) << 6);
                out_block[4 + j]     |= (uint8_t)((lm >> 4) << 6);
            }
        }

        f16_le_write(&out_block[0], max_scale / 63.0f);
        f16_le_write(&out_block[2], max_min   / 63.0f);
        float d    = f16_le_to_f32(out_block[0], out_block[1]);
        float dmin = f16_le_to_f32(out_block[2], out_block[3]);

        uint8_t *qh = &out_block[16];
        uint8_t *qs = &out_block[48];
        for (size_t j = 0; j < n_sub; j++) {
            uint8_t sc, m;
            get_scale_min_k4(j, &out_block[4], &sc, &m);
            float dl = d * (float)sc;
            float ml = dmin * (float)m;
            /* Sub-block j = 2 * outer + half: its 32 low nibbles live in
             * qs[outer * 32 + t], and its 5th bits in qh[t] bit (2*outer+half). */
            size_t outer = j / 2, half = j % 2;
            uint8_t hbit = (uint8_t)(1u << (2 * outer + half));
            for (size_t t = 0; t < 32; t++) {
                int32_t q = 0;
                if (dl > 0.0f) {
                    q = nearest_int_f((in_block[32 * j + t] + ml) / dl);
                    if (q < 0) q = 0;
                    if (q > 31) q = 31;
                }
                qs[outer * 32 + t] |= (uint8_t)(((uint32_t)q & 0x0Fu) << (4 * half));
                if (q & 16) qh[t] |= hbit;
            }
        }
    }
    return OC_OK;
}

/* Q6_K pack: 6-bit quantization with per-sub-block scale (signed).
 * Block layout (210 bytes): 128 bytes ql, 32 bytes qh, 16 bytes scales, 2 bytes f16 d. */
static OcError pack_q6_k(const float *src, size_t value_count,
                         uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK_K != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK_K) * OC_BLOCK_Q6_K_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK_K;

    for (size_t b = 0; b < n_blocks; b++) {
        const float *in_block = src + b * OC_QK_K;
        uint8_t *out_block = dst + b * OC_BLOCK_Q6_K_SIZE;

        /* Compute per-sub-block scale (16 sub-blocks of 16 elements each). */
        float scales[16];
        float max_scale = 0.0f;
        for (size_t j = 0; j < 16; j++) {
            float amax = 0.0f;
            for (size_t ii = 0; ii < 16; ii++) {
                float ax = fabsf(in_block[j * 16 + ii]);
                if (ax > amax) amax = ax;
            }
            scales[j] = amax / 31.0f;
            if (scales[j] > max_scale) max_scale = scales[j];
        }

        /* dequant_q6_k reads each sub-block scale as a plain int8 and
         * multiplies by d, so the super-block scale must span the full
         * int8 range — and the stored byte carries no +32 bias. */
        float d = max_scale / 127.0f;
        if (d == 0.0f) d = 1.0f;
        f16_le_write(&out_block[208], d);

        /* Pack scales as signed 8-bit values at offset 192. */
        uint8_t *sc_out = &out_block[192];
        for (size_t j = 0; j < 16; j++) {
            int32_t qs = nearest_int_f(scales[j] / d);
            if (qs < -128) qs = -128;
            if (qs > 127) qs = 127;
            sc_out[j] = (uint8_t)(int8_t)qs;
        }

        /* Quantize values: 6-bit signed (4 in ql + 2 in qh). */
        uint8_t *ql = &out_block[0];
        uint8_t *qh = &out_block[128];
        memset(ql, 0, 128);
        /* qh is 64 bytes (two 32-byte groups), not 32: the loop below ORs
         * into qh[group * 32 + l], so clearing only the first group left
         * the second reading uninitialized memory. */
        memset(qh, 0, 64);

        for (size_t group = 0; group < 2; group++) {
            size_t ql_off = group * 64;
            size_t qh_off = group * 32;
            size_t sc_off = group * 8;
            for (size_t l = 0; l < 32; l++) {
                size_t is_idx = l / 16;
                float dl1 = d * (float)(int8_t)sc_out[sc_off + is_idx];
                float dl2 = d * (float)(int8_t)sc_out[sc_off + is_idx + 2];
                float dl3 = d * (float)(int8_t)sc_out[sc_off + is_idx + 4];
                float dl4 = d * (float)(int8_t)sc_out[sc_off + is_idx + 6];

                /* 4 values per output position, each 6-bit signed (-32..31). */
                size_t base = group * 128 + l;

                if (dl1 != 0.0f) {
                    int32_t q1 = nearest_int_f(in_block[base] / dl1) + 32;
                    if (q1 < 0) q1 = 0;
                    if (q1 > 63) q1 = 63;
                    ql[ql_off + l] |= (uint8_t)(q1 & 0x0F);
                    qh[qh_off + l] |= (uint8_t)(((q1 >> 4) & 3u));
                }
                if (dl2 != 0.0f) {
                    int32_t q2 = nearest_int_f(in_block[base + 32] / dl2) + 32;
                    if (q2 < 0) q2 = 0;
                    if (q2 > 63) q2 = 63;
                    ql[ql_off + l + 32] |= (uint8_t)(q2 & 0x0F);
                    qh[qh_off + l] |= (uint8_t)(((q2 >> 4) & 3u) << 2);
                }
                if (dl3 != 0.0f) {
                    int32_t q3 = nearest_int_f(in_block[base + 64] / dl3) + 32;
                    if (q3 < 0) q3 = 0;
                    if (q3 > 63) q3 = 63;
                    ql[ql_off + l] |= (uint8_t)((q3 & 0x0F) << 4);
                    qh[qh_off + l] |= (uint8_t)(((q3 >> 4) & 3u) << 4);
                }
                if (dl4 != 0.0f) {
                    int32_t q4 = nearest_int_f(in_block[base + 96] / dl4) + 32;
                    if (q4 < 0) q4 = 0;
                    if (q4 > 63) q4 = 63;
                    ql[ql_off + l + 32] |= (uint8_t)((q4 & 0x0F) << 4);
                    qh[qh_off + l] |= (uint8_t)(((q4 >> 4) & 3u) << 6);
                }
            }
        }
    }
    return OC_OK;
}

static OcError pack_al5(const float *src, size_t value_count,
                        uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK_AL != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK_AL) * OC_BLOCK_Q4_0_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK_AL;
    for (size_t b = 0; b < n_blocks; b++) {
        pack_block_al5(src + b * OC_QK_AL, dst + b * OC_BLOCK_Q4_0_SIZE);
    }
    return OC_OK;
}

/* AL8 pack: MSE-optimal 8-bit packing with al_refine_scale. Port of
 * al_family.rs::quantize_block_al8. Output is the same 34-byte Q8_0 layout. */
static void pack_block_al8(const float *in_block, uint8_t *out_block)
{
    float amax, mx;
    block_amax_mx(in_block, OC_QK_AL, &amax, &mx);
    if (amax == 0.0f) {
        f16_le_write(&out_block[0], 0.0f);
        for (size_t i = 2; i < OC_BLOCK_Q8_0_SIZE; i++) out_block[i] = 0u;
        return;
    }
    float d = mx / -127.0f;
    d = al_refine_scale(in_block, OC_QK_AL, d, -127, 127);
    float inv_d = (d != 0.0f) ? 1.0f / d : 0.0f;
    f16_le_write(&out_block[0], d);
    for (size_t i = 0; i < OC_QK_AL; i++) {
        int32_t q;
        if (d == 0.0f) {
            q = 0;
        } else {
            float v = in_block[i] * inv_d;
            float r = roundf(v);
            if (r < -128.0f) r = -128.0f;
            if (r > 127.0f) r = 127.0f;
            q = (int32_t)r;
        }
        out_block[2 + i] = (uint8_t)(int8_t)q;
    }
}

static OcError pack_al8(const float *src, size_t value_count,
                        uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK_AL != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK_AL) * OC_BLOCK_Q8_0_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK_AL;
    for (size_t b = 0; b < n_blocks; b++) {
        pack_block_al8(src + b * OC_QK_AL, dst + b * OC_BLOCK_Q8_0_SIZE);
    }
    return OC_OK;
}

/* AL6 pack: MSE-optimal 5-bit packing with al_refine_scale. Port of
 * al_family.rs::quantize_block_al6. Output is the same 22-byte Q5_0 layout. */
static void pack_block_al6(const float *in_block, uint8_t *out_block)
{
    float amax, mx;
    block_amax_mx(in_block, OC_QK_AL, &amax, &mx);
    if (amax == 0.0f) {
        f16_le_write(&out_block[0], 0.0f);
        for (size_t i = 2; i < 6; i++) out_block[i] = 0u;
        for (size_t i = 6; i < OC_BLOCK_Q5_0_SIZE; i++) out_block[i] = 0x10u;
        return;
    }
    float d = mx / -16.0f;
    d = al_refine_scale(in_block, OC_QK_AL, d, -16, 15);
    float inv_d = (d != 0.0f) ? 1.0f / d : 0.0f;
    f16_le_write(&out_block[0], d);
    for (size_t i = 2; i < 6; i++) out_block[i] = 0u;
    for (size_t i = 0; i < OC_QK_AL; i++) {
        uint8_t q;
        if (d == 0.0f) {
            q = 16u;
        } else {
            float v = in_block[i] * inv_d + 16.5f;
            float r = truncf(v);
            if (r < 0.0f) r = 0.0f;
            if (r > 31.0f) r = 31.0f;
            q = (uint8_t)r;
        }
        if (q & 0x10u) {
            out_block[2 + i / 8] |= (uint8_t)(1u << (i % 8));
        }
        uint8_t low = (uint8_t)(q & 0x0Fu);
        size_t qs_index = 6 + i / 2;
        if (i % 2 == 0) {
            out_block[qs_index] = low;
        } else {
            out_block[qs_index] |= (uint8_t)(low << 4);
        }
    }
}

static OcError pack_al6(const float *src, size_t value_count,
                        uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK_AL != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK_AL) * OC_BLOCK_Q5_0_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK_AL;
    for (size_t b = 0; b < n_blocks; b++) {
        pack_block_al6(src + b * OC_QK_AL, dst + b * OC_BLOCK_Q5_0_SIZE);
    }
    return OC_OK;
}

/* AL5_XS pack: MSE-optimal 3-bit packing with al_refine_scale. Port of
 * al_family.rs::quantize_block_al5_xs. Output is the 14-byte AL5_XS layout
 * (2-byte f16 d + 12-byte 3-bit packed levels). */
static void pack_block_al5_xs(const float *in_block, uint8_t *out_block)
{
    float amax, mx;
    block_amax_mx(in_block, OC_QK_AL, &amax, &mx);
    if (amax == 0.0f) {
        f16_le_write(&out_block[0], 0.0f);
        for (size_t i = 2; i < OC_BLOCK_AL5_XS_SIZE; i++) out_block[i] = 0u;
        return;
    }
    float d = mx / -4.0f;
    d = al_refine_scale(in_block, OC_QK_AL, d, -4, 3);
    float inv_d = (d != 0.0f) ? 1.0f / d : 0.0f;
    f16_le_write(&out_block[0], d);
    uint8_t levels[OC_QK_AL];
    for (size_t i = 0; i < OC_QK_AL; i++) {
        uint8_t q;
        if (d == 0.0f) {
            q = 4u;
        } else {
            float v = in_block[i] * inv_d + 4.5f;
            float r = truncf(v);
            if (r < 0.0f) r = 0.0f;
            if (r > 7.0f) r = 7.0f;
            q = (uint8_t)r;
        }
        levels[i] = q;
    }
    pack3bit(levels, &out_block[2]);
}

static OcError pack_al5_xs(const float *src, size_t value_count,
                           uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK_AL != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK_AL) * OC_BLOCK_AL5_XS_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK_AL;
    for (size_t b = 0; b < n_blocks; b++) {
        pack_block_al5_xs(src + b * OC_QK_AL, dst + b * OC_BLOCK_AL5_XS_SIZE);
    }
    return OC_OK;
}

/* ─── IQ4 / NVFP4 pack encoders ───────────────────────────────────────
 *
 * These three types quantize against a fixed 16-entry codebook rather than
 * a learned grid, so encoding is a nearest-codebook search plus a scale
 * choice — no lattice lookup like IQ1/IQ2/IQ3 need.
 *
 * Scale choice, in both IQ4 encoders: map the largest-magnitude element of
 * the block onto KVALUES_IQ4NL[0] == -127, the codebook's extreme entry.
 * That guarantees no element clips, whatever its sign. A least-squares pass
 * over the chosen indices then rescales to minimize squared error. */

/* Index of the KVALUES_IQ4NL entry closest to `v` (v already divided by
 * the block scale). The table is sorted ascending, but a 16-entry linear
 * scan is cheap and keeps the tie-breaking obvious. */
static size_t iq4_nearest(float v)
{
    size_t best = 0;
    float best_err = fabsf(v - (float)KVALUES_IQ4NL[0]);
    for (size_t i = 1; i < 16; i++) {
        float err = fabsf(v - (float)KVALUES_IQ4NL[i]);
        if (err < best_err) { best_err = err; best = i; }
    }
    return best;
}

/* Least-squares rescale: given fixed codebook indices, the scale that
 * minimizes sum (x - d*kv)^2 is sum(x*kv)/sum(kv*kv). Returns `fallback`
 * when the denominator vanishes (an all-zero block). */
static float iq4_lsq_scale(const float *x, const size_t *idx, size_t n,
                           const float *weight, float fallback)
{
    float num = 0.0f, den = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float kv = (float)KVALUES_IQ4NL[idx[i]];
        float w = weight ? weight[i] : 1.0f;
        num += w * x[i] * kv;
        den += w * kv * kv;
    }
    if (den <= 0.0f) return fallback;
    return num / den;
}

static OcError pack_iq4_nl(const float *src, size_t value_count,
                           uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK4_NL != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK4_NL) * OC_BLOCK_IQ4_NL_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK4_NL;

    for (size_t b = 0; b < n_blocks; b++) {
        const float *in = src + b * OC_QK4_NL;
        uint8_t *out = dst + b * OC_BLOCK_IQ4_NL_SIZE;

        float ext = extreme_value(in, OC_QK4_NL);
        float d = ext / (float)KVALUES_IQ4NL[0];   /* maps ext -> -127 */

        size_t idx[OC_QK4_NL];
        if (d == 0.0f) {
            /* All-zero block: any index reconstructs to 0 with d == 0. */
            memset(idx, 0, sizeof(idx));
        } else {
            for (size_t i = 0; i < OC_QK4_NL; i++)
                idx[i] = iq4_nearest(in[i] / d);
            d = iq4_lsq_scale(in, idx, OC_QK4_NL, NULL, d);
        }

        f16_le_write(&out[0], d);
        /* Nibble layout mirrors the dequantizer: element j in the low
         * nibble of qs[j], element j + 16 in the high nibble. */
        for (size_t j = 0; j < OC_QK4_NL / 2; j++) {
            out[2 + j] = (uint8_t)((idx[j] & 0x0Fu)
                                   | ((idx[j + OC_QK4_NL / 2] & 0x0Fu) << 4));
        }
    }
    return OC_OK;
}

static OcError pack_iq4_xs(const float *src, size_t value_count,
                           uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK_K != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK_K) * OC_BLOCK_IQ4_XS_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK_K;

    const size_t n_sub = OC_QK_K / 32;   /* 8 sub-blocks of 32 elements */

    for (size_t b = 0; b < n_blocks; b++) {
        const float *in = src + b * OC_QK_K;
        uint8_t *out = dst + b * OC_BLOCK_IQ4_XS_SIZE;
        memset(out, 0, OC_BLOCK_IQ4_XS_SIZE);

        /* Per-sub-block scale that places its extreme element on -127. */
        float sub_scale[OC_QK_K / 32];
        float max_abs_scale = 0.0f;
        for (size_t ib = 0; ib < n_sub; ib++) {
            float ext = extreme_value(in + ib * 32, 32);
            sub_scale[ib] = ext / (float)KVALUES_IQ4NL[0];
            float a = fabsf(sub_scale[ib]);
            if (a > max_abs_scale) max_abs_scale = a;
        }

        /* Sub-block scales are stored as a 6-bit field ls, decoded as
         * d * (ls - 32), so the integer multiplier spans [-32, 31]. */
        float d = max_abs_scale / 31.0f;
        int32_t mult[OC_QK_K / 32];
        for (size_t ib = 0; ib < n_sub; ib++) {
            int32_t m = (d > 0.0f) ? nearest_int_f(sub_scale[ib] / d) : 0;
            if (m < -32) m = -32;
            if (m > 31) m = 31;
            mult[ib] = m;
        }

        /* Pick codebook indices against the quantized sub-block scales. */
        size_t idx[OC_QK_K];
        for (size_t ib = 0; ib < n_sub; ib++) {
            float dl = d * (float)mult[ib];
            for (size_t j = 0; j < 32; j++) {
                size_t i = ib * 32 + j;
                idx[i] = (dl != 0.0f) ? iq4_nearest(in[i] / dl) : 0;
            }
        }

        /* Refine d by least squares over the whole block. Each element's
         * reconstruction is d * mult[ib] * kv, so mult[ib] folds into the
         * per-element weight of the fit. */
        if (d > 0.0f) {
            float num = 0.0f, den = 0.0f;
            for (size_t ib = 0; ib < n_sub; ib++) {
                float m = (float)mult[ib];
                for (size_t j = 0; j < 32; j++) {
                    size_t i = ib * 32 + j;
                    float basis = m * (float)KVALUES_IQ4NL[idx[i]];
                    num += in[i] * basis;
                    den += basis * basis;
                }
            }
            if (den > 0.0f) d = num / den;
        }

        f16_le_write(&out[0], d);

        /* ls = mult + 32, split into 4 low bits (scales_l, nibble per
         * sub-block) and 2 high bits (scales_h, 2 bits per sub-block). */
        uint16_t scales_h = 0;
        uint8_t *scales_l = &out[4];
        for (size_t ib = 0; ib < n_sub; ib++) {
            uint32_t ls = (uint32_t)(mult[ib] + 32);   /* 0..63 */
            scales_l[ib / 2] |= (uint8_t)((ls & 0x0Fu) << (4 * (ib % 2)));
            scales_h |= (uint16_t)(((ls >> 4) & 3u) << (2 * ib));
        }
        out[2] = (uint8_t)(scales_h & 0xFFu);
        out[3] = (uint8_t)((scales_h >> 8) & 0xFFu);

        /* Nibbles: element j of a sub-block low, element j + 16 high. */
        uint8_t *qs = &out[8];
        for (size_t ib = 0; ib < n_sub; ib++) {
            for (size_t j = 0; j < 16; j++) {
                size_t lo = ib * 32 + j;
                size_t hi = lo + 16;
                qs[ib * 16 + j] = (uint8_t)((idx[lo] & 0x0Fu)
                                            | ((idx[hi] & 0x0Fu) << 4));
            }
        }
    }
    return OC_OK;
}

/* Nearest UE4M3 code for a non-negative scale. Only the low 7 bits are
 * meaningful (4-bit exponent + 3-bit mantissa, no sign), so an exhaustive
 * 128-entry search is both exact and cheap — it runs once per 16 values. */
static uint8_t f32_to_ue4m3(float v)
{
    if (!(v > 0.0f)) return 0;   /* also catches NaN */
    uint8_t best = 0;
    float best_err = fabsf(v - ue4m3_to_f32(0));
    for (uint32_t code = 1; code < 128u; code++) {
        float err = fabsf(v - ue4m3_to_f32((uint8_t)code));
        if (err < best_err) { best_err = err; best = (uint8_t)code; }
    }
    return best;
}

/* Index of the E2M1 codebook entry closest to `v`. Entry 8 is -0.0, which
 * compares equal to entry 0, so the scan naturally prefers +0. */
static size_t e2m1_nearest(float v)
{
    size_t best = 0;
    float best_err = fabsf(v - E2M1_DOUBLED_VALUES[0]);
    for (size_t i = 1; i < 16; i++) {
        float err = fabsf(v - E2M1_DOUBLED_VALUES[i]);
        if (err < best_err) { best_err = err; best = i; }
    }
    return best;
}

static OcError pack_nvfp4(const float *src, size_t value_count,
                          uint8_t *dst, size_t dst_len)
{
    if (value_count % OC_QK_NVFP4 != 0) return OC_ERR_INVALID_ARG;
    size_t expected = (value_count / OC_QK_NVFP4) * OC_BLOCK_NVFP4_SIZE;
    if (dst_len != expected) return OC_ERR_INVALID_ARG;
    size_t n_blocks = value_count / OC_QK_NVFP4;

    const size_t n_sub = OC_QK_NVFP4 / OC_QK_NVFP4_SUB;   /* 4 sub-blocks */
    /* Largest magnitude in the E2M1 codebook. */
    const float e2m1_max = 12.0f;

    for (size_t b = 0; b < n_blocks; b++) {
        const float *in = src + b * OC_QK_NVFP4;
        uint8_t *out = dst + b * OC_BLOCK_NVFP4_SIZE;
        uint8_t *scales = &out[0];
        uint8_t *qs = &out[n_sub];

        for (size_t sub = 0; sub < n_sub; sub++) {
            const float *x = in + sub * OC_QK_NVFP4_SUB;

            float amax = 0.0f;
            for (size_t j = 0; j < OC_QK_NVFP4_SUB; j++) {
                float ax = fabsf(x[j]);
                if (ax > amax) amax = ax;
            }

            /* Scale so the largest element lands on the codebook extreme,
             * then round the scale itself to the UE4M3 grid. Quantizing
             * against the *decoded* scale keeps encoder and decoder in
             * agreement even when rounding moved it. */
            scales[sub] = f32_to_ue4m3(amax / e2m1_max);
            float scale = ue4m3_to_f32(scales[sub]);

            size_t base_q = sub * (OC_QK_NVFP4_SUB / 2);
            for (size_t j = 0; j < OC_QK_NVFP4_SUB / 2; j++) {
                size_t lo_i = j;
                size_t hi_i = j + OC_QK_NVFP4_SUB / 2;
                size_t lo = (scale > 0.0f) ? e2m1_nearest(x[lo_i] / scale) : 0;
                size_t hi = (scale > 0.0f) ? e2m1_nearest(x[hi_i] / scale) : 0;
                qs[base_q + j] = (uint8_t)((lo & 0x0Fu) | ((hi & 0x0Fu) << 4));
            }
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
    /* SIMD fast path: if a kernel is available for this (qtype, host), it
     * produces output byte-identical to the scalar reference (VAL-SIMD-001..
     * 004). On false, fall through to the scalar switch. Layout errors are
     * also reported as false here so the scalar path returns the canonical
     * OC_ERR_INVALID_ARG. */
    if (oc_simd_try_dequant(qtype, src, src_len, dst, value_count)) {
        return OC_OK;
    }
    return oc_quant_dequant_row_scalar(qtype, src, src_len, dst, value_count);
}

OcError oc_quant_dequant_row_scalar(OcGgufQuantizationType qtype,
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
    /* AL-family (ggml ids 240-243) — MSE-optimized 4-bit packing. AL5/AL6/
     * AL8 reuse the Q4_0/Q5_0/Q8_0 dequant; AL5_XS uses its own 3-bit unpack. */
    case OC_QUANT_AL5:    return dequant_al5(src, src_len, dst, value_count);
    case OC_QUANT_AL5_XS: return dequant_al5_xs(src, src_len, dst, value_count);
    case OC_QUANT_AL6:    return dequant_al6(src, src_len, dst, value_count);
    case OC_QUANT_AL8:    return dequant_al8(src, src_len, dst, value_count);
    /* IQ-family — lookup-table-based importance quants. */
    case OC_QUANT_IQ1_S:   return dequant_iq1_s(src, src_len, dst, value_count);
    case OC_QUANT_IQ1_M:   return dequant_iq1_m(src, src_len, dst, value_count);
    case OC_QUANT_IQ2_XXS: return dequant_iq2_xxs(src, src_len, dst, value_count);
    case OC_QUANT_IQ2_XS:  return dequant_iq2_xs(src, src_len, dst, value_count);
    case OC_QUANT_IQ2_S:   return dequant_iq2_s(src, src_len, dst, value_count);
    case OC_QUANT_IQ3_XXS: return dequant_iq3_xxs(src, src_len, dst, value_count);
    case OC_QUANT_IQ3_S:   return dequant_iq3_s(src, src_len, dst, value_count);
    case OC_QUANT_IQ4_NL:  return dequant_iq4_nl(src, src_len, dst, value_count);
    case OC_QUANT_IQ4_XS:  return dequant_iq4_xs(src, src_len, dst, value_count);
    case OC_QUANT_NVFP4:   return dequant_nvfp4(src, src_len, dst, value_count);
    default:
        /* Unknown types return OC_ERR_QUANT (no crash). */
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
    case OC_QUANT_BF16:   return pack_bf16(src, value_count, dst, dst_len);
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
        return pack_q2_k(src, value_count, dst, dst_len);
    case OC_QUANT_Q3_K_S:
    case OC_QUANT_Q3_K_M:
    case OC_QUANT_Q3_K_L:
        return pack_q3_k(src, value_count, dst, dst_len);
    case OC_QUANT_Q5_K_S:
    case OC_QUANT_Q5_K_M:
        return pack_q5_k(src, value_count, dst, dst_len);
    case OC_QUANT_Q6_K:
        return pack_q6_k(src, value_count, dst, dst_len);
    /* AL-family (MSE-optimized encoders). */
    case OC_QUANT_AL5:    return pack_al5(src, value_count, dst, dst_len);
    case OC_QUANT_AL5_XS: return pack_al5_xs(src, value_count, dst, dst_len);
    case OC_QUANT_AL6:    return pack_al6(src, value_count, dst, dst_len);
    case OC_QUANT_AL8:    return pack_al8(src, value_count, dst, dst_len);
    /* Codebook-quantized types (fixed 16-entry tables, no lattice). */
    case OC_QUANT_IQ4_NL: return pack_iq4_nl(src, value_count, dst, dst_len);
    case OC_QUANT_IQ4_XS: return pack_iq4_xs(src, value_count, dst, dst_len);
    case OC_QUANT_NVFP4:  return pack_nvfp4(src, value_count, dst, dst_len);
    default:
        /* IQ1/IQ2/IQ3 encoders need a search over the E8 lattice grids in
         * quant_tables.h, which is not implemented — those types stay
         * dequant-only. Unknown types return OC_ERR_QUANT (no crash). */
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
    { OC_QUANT_Q3_K_M,  "Q3_K_M",  11 },
    { OC_QUANT_Q3_K_L,  "Q3_K_L",  11 },
    { OC_QUANT_Q4_K_S,  "Q4_K_S",  12 },
    { OC_QUANT_Q4_K_M,  "Q4_K_M",  12 },
    { OC_QUANT_Q5_K_S,  "Q5_K_S",  13 },
    { OC_QUANT_Q5_K_M,  "Q5_K_M",  13 },
    { OC_QUANT_Q6_K,    "Q6_K",    14 },
    /* AL-family (ggml ids 240-243) — names match oxidize-core. */
    { OC_QUANT_AL5,     "AL5",     240 },
    { OC_QUANT_AL5_XS,  "AL5_XS",  243 },
    { OC_QUANT_AL6,     "AL6",     242 },
    { OC_QUANT_AL8,     "AL8",     241 },
    /* IQ-family (ggml ids 24-29, 32-37, etc.). */
    { OC_QUANT_IQ2_XXS, "IQ2_XXS", 16 },
    { OC_QUANT_IQ2_XS,  "IQ2_XS",  17 },
    { OC_QUANT_IQ2_S,   "IQ2_S",   22 },
    { OC_QUANT_IQ3_XXS, "IQ3_XXS", 18 },
    { OC_QUANT_IQ3_S,   "IQ3_S",   21 },
    { OC_QUANT_IQ4_NL,  "IQ4_NL",  20 },
    { OC_QUANT_IQ4_XS,  "IQ4_XS",  23 },
    { OC_QUANT_IQ1_S,   "IQ1_S",   19 },
    { OC_QUANT_IQ1_M,   "IQ1_M",   29 },
    { OC_QUANT_NVFP4,   "NVFP4",   40 },
    { OC_QUANT_I8,      "I8",      24 },
    { OC_QUANT_I16,     "I16",     25 },
    { OC_QUANT_I32,     "I32",     26 },
    { OC_QUANT_I64,     "I64",     27 },
    { OC_QUANT_F64,     "F64",     28 },
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
    if (ggml_type == 12) return OC_QUANT_Q4_K_M;
    if (ggml_type == 13) return OC_QUANT_Q5_K_M;
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
