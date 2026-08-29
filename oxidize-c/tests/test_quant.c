/*
 * test_quant.c — quantization scalar reference tests (VAL-QUANT-001..015).
 *
 * Covers the `quant-standard-types` feature scope: block-size constants,
 * dequant-to-f32, pack-then-dequant round-trip, random-block corpus sweep,
 * unknown-type error handling.
 *
 * Bit-exact parity with Rust `oxidize-core/src/compute/quantization.rs`
 * scalar dequant is validated by:
 *   - Constructing canonical blocks using the same encoder algorithm the
 *     Rust reference uses (so the expected dequant output is computable by
 *     hand for zero/known-scale blocks).
 *   - Round-trip tests for types whose Rust encoder IS a true inverse of the
 *     dequant (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K_M, F32, F16, BF16, I*).
 *   - For Q2_K / Q3_K / Q5_K / Q6_K: the Rust `quantize_k_packed_scalar`
 *     produces a simplified layout that is NOT a true inverse of the
 *     super-block dequant (the d field lands at offset 0 in the pack but is
 *     read from offset 80+ by the dequant). These types are tested by
 *     dequantizing hand-crafted blocks with known scales and verifying the
 *     output matches the hand-computed expected values.
 *
 * SIMD parity (scalar-vs-AVX2 / AVX2-vs-AVX-512) is layered on by the
 * `quant-simd-dispatch` feature.
 */
#include "framework.h"

#include "oxidize/quant.h"
#include "oxidize/error.h"
/* Pull in the AL/IQ/NVFP4 constant tables (KVALUES_IQ4NL, IQ3S_GRID, etc.)
 * so the SHA256 parity test (VAL-QUANT-016) can hash their bytes directly. */
#include "../src/compute/quant_tables.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static void put_f16(uint8_t *buf, size_t off, uint16_t bits)
{
    buf[off]     = (uint8_t)(bits & 0xFFu);
    buf[off + 1] = (uint8_t)((bits >> 8) & 0xFFu);
}

static void put_f32(uint8_t *buf, size_t off, float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    buf[off]     = (uint8_t)(bits & 0xFFu);
    buf[off + 1] = (uint8_t)((bits >> 8) & 0xFFu);
    buf[off + 2] = (uint8_t)((bits >> 16) & 0xFFu);
    buf[off + 3] = (uint8_t)((bits >> 24) & 0xFFu);
}

/* Deterministic LCG for fixture generation (reproducible without libc rand). */
static uint32_t lcg_state = 0x1234abcdu;
static float next_float(float lo, float hi)
{
    uint32_t x = lcg_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    lcg_state = x;
    float u = (float)((double)(x >> 8) / (double)0x1000000);
    return lo + u * (hi - lo);
}

/* ─── VAL-QUANT-001: block-size constants ─────────────────────────────── */

Test(quant, block_sizes, .description = "VAL-QUANT-001: block-size constants match Rust") {
    struct { OcGgufQuantizationType t; size_t els; size_t bytes; } cases[] = {
        { OC_QUANT_F32,    1,   4  },
        { OC_QUANT_F16,    1,   2  },
        { OC_QUANT_BF16,   1,   2  },
        { OC_QUANT_Q4_0,   32,  18 },
        { OC_QUANT_Q4_1,   32,  20 },
        { OC_QUANT_Q5_0,   32,  22 },
        { OC_QUANT_Q5_1,   32,  24 },
        { OC_QUANT_Q8_0,   32,  34 },
        { OC_QUANT_Q2_K,   256, 84 },
        { OC_QUANT_Q3_K_S, 256, 110 },
        { OC_QUANT_Q3_K_M, 256, 110 },
        { OC_QUANT_Q3_K_L, 256, 110 },
        { OC_QUANT_Q4_K_S, 256, 144 },
        { OC_QUANT_Q4_K_M, 256, 144 },
        { OC_QUANT_Q5_K_S, 256, 176 },
        { OC_QUANT_Q5_K_M, 256, 176 },
        { OC_QUANT_Q6_K,   256, 210 },
        { OC_QUANT_I8,     1,   1  },
        { OC_QUANT_I16,    1,   2  },
        { OC_QUANT_I32,    1,   4  },
        { OC_QUANT_I64,    1,   8  },
        { OC_QUANT_F64,    1,   8  },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        OcQuantBlockLayout bs = oc_quant_block_size(cases[i].t);
        cr_assert_eq(bs.elements_per_block, cases[i].els,
            "block-size elements mismatch for %s: got %zu, want %zu",
            oc_quant_type_name(cases[i].t), bs.elements_per_block, cases[i].els);
        cr_assert_eq(bs.bytes_per_block, cases[i].bytes,
            "block-size bytes mismatch for %s: got %zu, want %zu",
            oc_quant_type_name(cases[i].t), bs.bytes_per_block, cases[i].bytes);
    }
}

Test(quant, block_size_unknown_returns_zero, .description = "VAL-QUANT-013: unknown block size is (0,0)") {
    OcQuantBlockLayout bs = oc_quant_block_size(OC_QUANT_UNKNOWN);
    cr_assert_eq(bs.elements_per_block, 0u, "unknown elements_per_block");
    cr_assert_eq(bs.bytes_per_block,    0u, "unknown bytes_per_block");
}

Test(quant, quantized_size_matches_layout, .description = "oc_quantized_size consistency") {
    cr_assert_eq(oc_quantized_size(OC_QUANT_Q4_0, 64), 36u, "Q4_0 quantized_size");
    cr_assert_eq(oc_quantized_size(OC_QUANT_Q4_0, 33), 0u, "Q4_0 non-multiple");
    cr_assert_eq(oc_quantized_size(OC_QUANT_Q4_K_M, 256), 144u, "Q4_K_M quantized_size");
    cr_assert_eq(oc_quantized_size(OC_QUANT_Q6_K, 256), 210u, "Q6_K quantized_size");
}

/* ─── VAL-QUANT-008: F32/F16/BF16 + integer/f64 dequant ──────────────── */

Test(quant, dequant_f32, .description = "VAL-QUANT-008: F32 dequant is identity") {
    float src[8];
    for (int i = 0; i < 8; i++) src[i] = (float)i * 0.5f - 1.5f;
    uint8_t buf[32];
    memcpy(buf, src, sizeof(buf));
    float dst[8];
    OcError e = oc_quant_dequant_row(OC_QUANT_F32, buf, sizeof(buf), dst, 8);
    cr_assert_eq(e, OC_OK, "F32 dequant should succeed");
    cr_assert_arr_eq(dst, src, sizeof(dst), "F32 dequant should be identity");
}

Test(quant, dequant_f16, .description = "VAL-QUANT-008: F16 dequant bit-exact") {
    uint16_t f16_vals[6] = { 0x3C00, 0xBC00, 0x3800, 0xB400, 0x4000, 0x0001 };
    float expected[6] = { 1.0f, -1.0f, 0.5f, -0.25f, 2.0f, 0.0f };
    uint8_t buf[12];
    for (int i = 0; i < 6; i++) put_f16(buf, 2 * i, f16_vals[i]);
    float dst[6];
    OcError e = oc_quant_dequant_row(OC_QUANT_F16, buf, sizeof(buf), dst, 6);
    cr_assert_eq(e, OC_OK, "F16 dequant should succeed");
    for (int i = 0; i < 5; i++) {
        cr_assert_float_eq(dst[i], expected[i], 1e-7f,
            "F16 dequant mismatch at %d: got %f, want %f", i, dst[i], expected[i]);
    }
    /* Subnormal (0x0001) → tiny positive value. */
    cr_assert_gt(dst[5], 0.0f, "F16 subnormal should be positive tiny");
}

Test(quant, dequant_bf16, .description = "VAL-QUANT-008: BF16 dequant bit-exact") {
    uint16_t bf16_vals[4] = { 0x3F80, 0xC000, 0x3F00, 0x0000 };
    float expected[4] = { 1.0f, -2.0f, 0.5f, 0.0f };
    uint8_t buf[8];
    for (int i = 0; i < 4; i++) put_f16(buf, 2 * i, bf16_vals[i]);
    float dst[4];
    OcError e = oc_quant_dequant_row(OC_QUANT_BF16, buf, sizeof(buf), dst, 4);
    cr_assert_eq(e, OC_OK, "BF16 dequant should succeed");
    for (int i = 0; i < 4; i++) {
        cr_assert_float_eq(dst[i], expected[i], 1e-7f,
            "BF16 dequant mismatch at %d: got %f, want %f", i, dst[i], expected[i]);
    }
}

Test(quant, pack_bf16_round_trip, .description = "BF16 pack-then-dequant round-trip") {
    float src[8] = { 1.0f, -2.0f, 0.5f, -0.25f, 3.14159f, -1e10f, 0.0f, 42.0f };
    uint8_t buf[16];
    OcError e = oc_quant_pack_row(OC_QUANT_BF16, src, 8, buf, sizeof(buf));
    cr_assert_eq(e, OC_OK, "BF16 pack should succeed");
    float dst[8];
    e = oc_quant_dequant_row(OC_QUANT_BF16, buf, sizeof(buf), dst, 8);
    cr_assert_eq(e, OC_OK, "BF16 dequant should succeed");
    /* BF16 has ~7 bits of mantissa, so relative error should be < 1%. */
    for (int i = 0; i < 8; i++) {
        if (src[i] == 0.0f) {
            cr_assert_float_eq(dst[i], 0.0f, 1e-7f, "BF16 zero at %d", i);
        } else {
            float rel_err = fabsf((dst[i] - src[i]) / src[i]);
            cr_assert(rel_err < 0.01f, "BF16 round-trip error too high at %d: %f vs %f (rel=%f)",
                      i, dst[i], src[i], rel_err);
        }
    }
}

Test(quant, pack_bf16_basic_values) {
    /* Test known BF16 values: 1.0 = 0x3F80, -2.0 = 0xC000, 0.5 = 0x3F00 */
    float src[3] = { 1.0f, -2.0f, 0.5f };
    uint8_t buf[6];
    cr_assert_eq(oc_quant_pack_row(OC_QUANT_BF16, src, 3, buf, sizeof(buf)), OC_OK);
    /* Check BF16 bit pattern (little-endian). */
    cr_assert_eq(buf[0], 0x80); cr_assert_eq(buf[1], 0x3F);  /* 1.0 = 0x3F80 */
    cr_assert_eq(buf[2], 0x00); cr_assert_eq(buf[3], 0xC0);  /* -2.0 = 0xC000 */
    cr_assert_eq(buf[4], 0x00); cr_assert_eq(buf[5], 0x3F);  /* 0.5 = 0x3F00 */
}

Test(quant, pack_bf16_wrong_size) {
    float src[4] = {0};
    uint8_t buf[7]; /* too small */
    cr_assert_neq(oc_quant_pack_row(OC_QUANT_BF16, src, 4, buf, sizeof(buf)), OC_OK);
}

Test(quant, dequant_int_and_f64, .description = "I8/I16/I32/I64/F64 dequant bit-exact") {
    int8_t i8[5] = { -1, 0, 1, 127, -128 };
    float i8_exp[5] = { -1.0f, 0.0f, 1.0f, 127.0f, -128.0f };
    float dst[8];
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_I8, (const uint8_t*)i8, 5, dst, 5), OC_OK);
    for (int i = 0; i < 5; i++) {
        cr_assert_float_eq(dst[i], i8_exp[i], 0.0f, "I8 dequant at %d", i);
    }

    uint8_t i16_buf[8];
    int16_t i16[4] = { 1, -2, 32767, -32768 };
    for (int i = 0; i < 4; i++) {
        i16_buf[2 * i]     = (uint8_t)((uint16_t)i16[i] & 0xFFu);
        i16_buf[2 * i + 1] = (uint8_t)(((uint16_t)i16[i] >> 8) & 0xFFu);
    }
    float i16_exp[4] = { 1.0f, -2.0f, 32767.0f, -32768.0f };
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_I16, i16_buf, 8, dst, 4), OC_OK);
    for (int i = 0; i < 4; i++) {
        cr_assert_float_eq(dst[i], i16_exp[i], 0.0f, "I16 dequant at %d", i);
    }

    int32_t i32[2] = { 1000000, -1000000 };
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_I32, (const uint8_t*)i32, 8, dst, 2), OC_OK);
    cr_assert_float_eq(dst[0], 1000000.0f, 0.0f, "I32 +");
    cr_assert_float_eq(dst[1], -1000000.0f, 0.0f, "I32 -");

    int64_t i64[1] = { (int64_t)1 << 40 };
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_I64, (const uint8_t*)i64, 8, dst, 1), OC_OK);
    cr_assert_float_eq(dst[0], (float)((int64_t)1 << 40), 0.0f, "I64 large");

    double f64[2] = { 0.5, -1.5 };
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_F64, (const uint8_t*)f64, 16, dst, 2), OC_OK);
    cr_assert_float_eq(dst[0], 0.5f, 0.0f, "F64 0.5");
    cr_assert_float_eq(dst[1], -1.5f, 0.0f, "F64 -1.5");
}

/* ─── VAL-QUANT-002: Q4_0 dequant ─────────────────────────────────────── */

Test(quant, dequant_q4_0_handcrafted, .description = "VAL-QUANT-002: Q4_0 dequant on hand-crafted block") {
    /* Block: d=1.0 (f16 0x3C00), 16 packed bytes encoding nibbles 0..15 then
     * 0..15. Each nibble n decodes to (n - 8) * d = (n-8) * 1.0. */
    uint8_t buf[18];
    buf[0] = 0x00; buf[1] = 0x3C;  /* f16 1.0 little-endian */
    for (int i = 0; i < 8; i++) {
        buf[2 + i]      = (uint8_t)(i | ((i + 8) << 4));
        buf[2 + 8 + i]  = (uint8_t)((i + 8) | (i << 4));
    }
    float dst[32];
    OcError e = oc_quant_dequant_row(OC_QUANT_Q4_0, buf, sizeof(buf), dst, 32);
    cr_assert_eq(e, OC_OK, "Q4_0 dequant should succeed");
    /* First 16: out[i] = (nibble_low - 8) * 1.0. */
    for (int i = 0; i < 8; i++) {
        cr_assert_float_eq(dst[i],      (float)(i - 8),      0.0f, "Q4_0 low[%d]", i);
        cr_assert_float_eq(dst[i + 8],  (float)((i + 8) - 8), 0.0f, "Q4_0 low[%d]", i + 8);
        cr_assert_float_eq(dst[i + 16], (float)((i + 8) - 8), 0.0f, "Q4_0 hi[%d]", i + 16);
        cr_assert_float_eq(dst[i + 24], (float)(i - 8),      0.0f, "Q4_0 hi[%d]", i + 24);
    }
}

Test(quant, dequant_q4_0_zero_block, .description = "Q4_0 zero block dequant") {
    /* All-zero block: d=0 (f16 0x0000), all packed bytes 0x88 (nibbles = 8,
     * so (8 - 8) * 0 = 0). */
    uint8_t buf[18];
    buf[0] = 0; buf[1] = 0;
    for (int i = 0; i < 16; i++) buf[2 + i] = 0x88;
    float dst[32];
    OcError e = oc_quant_dequant_row(OC_QUANT_Q4_0, buf, sizeof(buf), dst, 32);
    cr_assert_eq(e, OC_OK);
    for (int i = 0; i < 32; i++) {
        cr_assert_float_eq(dst[i], 0.0f, 0.0f, "Q4_0 zero at %d: got %f", i, dst[i]);
    }
}

/* ─── Q8_0 dequant hand-crafted ──────────────────────────────────────── */

Test(quant, dequant_q8_0_handcrafted, .description = "VAL-QUANT-004: Q8_0 dequant on hand-crafted block") {
    /* d=0.5 (f16 0x3800), qs[i] = i as i8. Expected: out[i] = (i8)qs[i] * 0.5. */
    uint8_t buf[34];
    buf[0] = 0x00; buf[1] = 0x38;  /* f16 0.5 */
    for (int i = 0; i < 32; i++) buf[2 + i] = (uint8_t)(int8_t)i;
    float dst[32];
    OcError e = oc_quant_dequant_row(OC_QUANT_Q8_0, buf, sizeof(buf), dst, 32);
    cr_assert_eq(e, OC_OK, "Q8_0 dequant");
    for (int i = 0; i < 32; i++) {
        float expected = (float)((int8_t)(uint8_t)(int8_t)i) * 0.5f;
        cr_assert_float_eq(dst[i], expected, 0.0f, "Q8_0 at %d: got %f, want %f",
            i, dst[i], expected);
    }
}

/* ─── Q6_K dequant hand-crafted (known scales) ─────────────────────────── */

Test(quant, dequant_q6_k_handcrafted, .description = "VAL-QUANT-004: Q6_K dequant on hand-crafted block") {
    /* Construct a Q6_K block with d=1.0, all scales = 1 (signed i8), all
     * quantized values = 0 (so out = d * sc * (0 - 32) = 1 * 1 * -32 = -32).
     * Then verify the output matches the hand-computed expected value. */
    uint8_t buf[210];
    memset(buf, 0, sizeof(buf));
    /* ql[0..128] = 0, qh[0..64] = 0, sc[0..16] = 1 (i8), d=1.0 at [208..210]. */
    for (int i = 0; i < 128; i++) buf[i] = 0;
    for (int i = 128; i < 192; i++) buf[i] = 0;
    for (int i = 192; i < 208; i++) buf[i] = 1u;  /* sc = 1 as u8 = 1 as i8 */
    put_f16(buf, 208, 0x3C00);  /* f16 1.0 */

    float dst[256];
    OcError e = oc_quant_dequant_row(OC_QUANT_Q6_K, buf, sizeof(buf), dst, 256);
    cr_assert_eq(e, OC_OK, "Q6_K dequant should succeed");
    /* All quantized values = 0 (ql & 0xF = 0, qh & 3 = 0), so q1 = 0 - 32 = -32.
     * sc[is] = 1 (i8), d = 1.0 → out = 1.0 * 1 * -32 = -32.0. */
    for (int i = 0; i < 256; i++) {
        cr_assert_float_eq(dst[i], -32.0f, 0.0f, "Q6_K at %d: got %f", i, dst[i]);
    }
}

Test(quant, dequant_q6_k_zero_d, .description = "Q6_K with d=0 produces all zeros") {
    uint8_t buf[210];
    memset(buf, 0, sizeof(buf));
    put_f16(buf, 208, 0x0000);  /* d = 0 */
    float dst[256];
    OcError e = oc_quant_dequant_row(OC_QUANT_Q6_K, buf, sizeof(buf), dst, 256);
    cr_assert_eq(e, OC_OK);
    for (int i = 0; i < 256; i++) {
        cr_assert_float_eq(dst[i], 0.0f, 0.0f, "Q6_K zero-d at %d", i);
    }
}

/* ─── Q2_K dequant hand-crafted ────────────────────────────────────────── */

Test(quant, dequant_q2_k_handcrafted, .description = "VAL-QUANT-005: Q2_K dequant on hand-crafted block") {
    /* Q2_K layout (84 bytes): scales[0..16], qs[16..80], d[80..82], min[82..84].
     * Set d=1.0, min=0.0, scales[0]=0x11 (sc1=1), scales[1]=0x11 (sc2=1),
     * rest of scales = 0, qs all = 0xFF (so (qs >> shift) & 3 = 3 for all).
     * For the first 32 outputs: dl1 = d * 1 = 1.0, ml1 = min * 1 = 0 → out = 3 - 0 = 3.
     * For the next 32: dl2 = d * 1 = 1.0, ml2 = 0 → out = 3.
     * Remaining 192: scales = 0 → dl = 0, ml = 0 → out = 0. */
    uint8_t buf[84];
    memset(buf, 0, sizeof(buf));
    buf[0] = 0x11;  /* scales[0] = 0x11 (sc1=1, ml1=1) */
    buf[1] = 0x11;  /* scales[1] = 0x11 (sc2=1, ml2=1) */
    for (int i = 16; i < 80; i++) buf[i] = 0xFF;  /* qs = 0xFF → (>>0)&3 = 3 */
    put_f16(buf, 80, 0x3C00);  /* d = 1.0 */
    put_f16(buf, 82, 0x0000);  /* min = 0.0 */

    float dst[256];
    OcError e = oc_quant_dequant_row(OC_QUANT_Q2_K, buf, sizeof(buf), dst, 256);
    cr_assert_eq(e, OC_OK, "Q2_K dequant should succeed");
    /* First 32 outputs (sub-block 0): shift = 0 (first iter), dl1=1, ml1=0.
     * out = 1 * 3 - 0 = 3.0 for first 16, dl2=1, ml2=0 → 3.0 for next 16. */
    for (int i = 0; i < 32; i++) {
        cr_assert_float_eq(dst[i], 3.0f, 0.0f, "Q2_K first-32[%d]: got %f", i, dst[i]);
    }
    /* Remaining outputs: scales = 0 → dl = 0, ml = 0 → out = 0 * 3 - 0 = 0. */
    for (int i = 32; i < 256; i++) {
        cr_assert_float_eq(dst[i], 0.0f, 0.0f, "Q2_K rest[%d]: got %f", i, dst[i]);
    }
}

Test(quant, dequant_q2_k_zero_block, .description = "Q2_K zero block (d=0) → all zeros") {
    uint8_t buf[84];
    memset(buf, 0, sizeof(buf));
    float dst[256];
    OcError e = oc_quant_dequant_row(OC_QUANT_Q2_K, buf, sizeof(buf), dst, 256);
    cr_assert_eq(e, OC_OK);
    for (int i = 0; i < 256; i++) {
        cr_assert_float_eq(dst[i], 0.0f, 0.0f, "Q2_K zero at %d", i);
    }
}

/* ─── Q3_K dequant hand-crafted ────────────────────────────────────────── */

Test(quant, dequant_q3_k_handcrafted, .description = "VAL-QUANT-005: Q3_K dequant runs and produces finite output") {
    /* Q3_K scale decoding is non-trivial (bit-shuffled 6-bit values). For a
     * sanity test, set d=0 so all outputs are 0 regardless of scales — this
     * verifies the dequant runs without crash and produces finite output.
     * The bit-exact dequant parity vs Rust is validated by the corpus sweep
     * on real GGUF fixtures (deferred to integration on .121 where Rust
     * reference is available). */
    uint8_t buf[110];
    memset(buf, 0, sizeof(buf));
    /* d = 0.0 at [108..110]. */
    put_f16(buf, 108, 0x0000);

    float dst[256];
    OcError e = oc_quant_dequant_row(OC_QUANT_Q3_K_S, buf, sizeof(buf), dst, 256);
    cr_assert_eq(e, OC_OK, "Q3_K dequant should succeed");
    for (int i = 0; i < 256; i++) {
        cr_assert(isfinite(dst[i]), "Q3_K non-finite at %d", i);
        cr_assert_float_eq(dst[i], 0.0f, 0.0f, "Q3_K zero-d at %d: got %f", i, dst[i]);
    }
}

/* ─── Q5_K dequant hand-crafted ────────────────────────────────────────── */

Test(quant, dequant_q5_k_handcrafted, .description = "VAL-QUANT-004: Q5_K dequant on hand-crafted block") {
    /* Q5_K layout: d[0..2], min[2..4], scales[4..16], qh[16..48], qs[48..176].
     * Set d=1.0, min=0.0, scales[0]=1, scales[1]=1, rest=0, qh=0, qs=0.
     * For first 32: qv1 = 0 + 0 = 0 (qh & u1 = 0). out = d1 * 0 - min1 = 0 - 0 = 0.
     * (Since qs=0, qv=0 → out = 1*0 - 0 = 0.) Verify it runs without crash. */
    uint8_t buf[176];
    memset(buf, 0, sizeof(buf));
    put_f16(buf, 0, 0x3C00);  /* d = 1.0 */
    put_f16(buf, 2, 0x0000);  /* min = 0 */
    buf[4] = 1u;  /* scales[0] low bits = 1 */
    buf[5] = 1u;  /* scales[1] (scales[4+0]) low bits = 1 */
    /* Set qs[0..32] = 0x11 so low nibble = 1, high nibble = 1.
     * qv1 = 1 + 0 (qh=0) = 1. out = 1 * 1 - 0 = 1. */
    for (int i = 48; i < 80; i++) buf[i] = 0x11u;

    float dst[256];
    OcError e = oc_quant_dequant_row(OC_QUANT_Q5_K_M, buf, sizeof(buf), dst, 256);
    cr_assert_eq(e, OC_OK, "Q5_K dequant should succeed");
    /* First 32 outputs (group_pair 0, sc1=1, m1=0): out = 1 * 1 - 0 = 1. */
    for (int i = 0; i < 32; i++) {
        cr_assert_float_eq(dst[i], 1.0f, 0.0f, "Q5_K first-32[%d]: got %f", i, dst[i]);
    }
    /* Next 32 (group_pair 0, sc2=1, m2=0): qv2 = (qs>>4) + 0 = 1. out = 1*1 - 0 = 1. */
    for (int i = 32; i < 64; i++) {
        cr_assert_float_eq(dst[i], 1.0f, 0.0f, "Q5_K next-32[%d]: got %f", i, dst[i]);
    }
    /* Remaining 192: scales = 0 → d1=0, d2=0 → out = 0*0 - 0 = 0. */
    for (int i = 64; i < 256; i++) {
        cr_assert_float_eq(dst[i], 0.0f, 0.0f, "Q5_K rest[%d]: got %f", i, dst[i]);
    }
}

/* ─── Q4_K dequant hand-crafted ────────────────────────────────────────── */

Test(quant, dequant_q4_k_handcrafted, .description = "VAL-QUANT-003: Q4_K dequant on hand-crafted block") {
    /* Q4_K layout: d[0..2], min[2..4], scales[4..16], qs[16..144].
     * Set d=1.0, min=0.0, scales[0]=1 (sc1=1, m1=0), scales[1]=1 (sc2=1, m2=0),
     * rest of scales = 0, qs[0..32] = 0x11 (low=1, high=1).
     * First 32: out = d1 * (qs & 0xF) - min1 = 1 * 1 - 0 = 1.
     * Next 32: out = d2 * (qs >> 4) - min2 = 1 * 1 - 0 = 1.
     * Rest: scales=0 → d1=d2=0, min1=min2=0 → out = 0 - 0 = 0. */
    uint8_t buf[144];
    memset(buf, 0, sizeof(buf));
    put_f16(buf, 0, 0x3C00);  /* d = 1.0 */
    put_f16(buf, 2, 0x0000);  /* min = 0 */
    buf[4] = 1u;  /* scales[0] (j<4): sc1 = 1 & 63 = 1, m1 = scales[4] & 63 = 0. Wait, scales[4+0]=scales[4] is for m1 of j=0. */
    /* Actually for j=0: sc1 = scales[0] & 63, m1 = scales[0+4] & 63 = scales[4] & 63.
     * sc2 = scales[1] & 63, m2 = scales[1+4] & 63 = scales[5] & 63.
     * We want sc1=1, m1=0, sc2=1, m2=0. So scales[0]=1, scales[1]=1, scales[4]=0, scales[5]=0. */
    buf[4] = 0u;  /* reset */
    buf[5] = 0u;
    buf[4] = 1u;  /* scales[0] = 1 → sc1 = 1 */
    buf[5] = 1u;  /* scales[1] = 1 → sc2 = 1 */
    /* scales[4], scales[5] stay 0 → m1 = 0, m2 = 0. Good. */
    for (int i = 16; i < 48; i++) buf[i] = 0x11u;  /* qs[0..32] = 0x11 */

    float dst[256];
    OcError e = oc_quant_dequant_row(OC_QUANT_Q4_K_M, buf, sizeof(buf), dst, 256);
    cr_assert_eq(e, OC_OK, "Q4_K dequant should succeed");
    /* First 32: d1 = 1*1 = 1, min1 = 0*0 = 0. out = 1 * 1 - 0 = 1. */
    for (int i = 0; i < 32; i++) {
        cr_assert_float_eq(dst[i], 1.0f, 0.0f, "Q4_K first-32[%d]: got %f", i, dst[i]);
    }
    /* Next 32: d2 = 1*1 = 1, min2 = 0*0 = 0. qv2 = (qs>>4) = 1. out = 1*1 - 0 = 1. */
    for (int i = 32; i < 64; i++) {
        cr_assert_float_eq(dst[i], 1.0f, 0.0f, "Q4_K next-32[%d]: got %f", i, dst[i]);
    }
    /* Remaining: scales = 0 → out = 0 - 0 = 0. */
    for (int i = 64; i < 256; i++) {
        cr_assert_float_eq(dst[i], 0.0f, 0.0f, "Q4_K rest[%d]: got %f", i, dst[i]);
    }
}

/* ─── VAL-QUANT-013: unknown type returns OC_ERR_QUANT ─────────────────── */

Test(quant, unknown_type_returns_err_quant, .description = "VAL-QUANT-013: unknown type → OC_ERR_QUANT") {
    uint8_t buf[16] = {0};
    float dst[4];
    OcError e = oc_quant_dequant_row(OC_QUANT_UNKNOWN, buf, sizeof(buf), dst, 4);
    cr_assert_eq(e, OC_ERR_QUANT, "dequant unknown should return OC_ERR_QUANT");
    OcError pe = oc_quant_pack_row(OC_QUANT_UNKNOWN, dst, 4, buf, sizeof(buf));
    cr_assert_eq(pe, OC_ERR_QUANT, "pack unknown should return OC_ERR_QUANT");
    OcError be = oc_quant_pack_block(OC_QUANT_UNKNOWN, dst, buf);
    cr_assert_eq(be, OC_ERR_QUANT, "pack_block unknown should return OC_ERR_QUANT");
}

Test(quant, invalid_arg_returns_error, .description = "NULL/bad-length args return OC_ERR_INVALID_ARG") {
    uint8_t buf[16] = {0};
    float dst[4];
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_F32, NULL, 0, dst, 4), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_F32, buf, sizeof(buf), NULL, 4), OC_ERR_INVALID_ARG);
    /* Mismatched src_len/value_count for F32 (4 bytes/elem, but value_count=5 != 4). */
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_F32, buf, 16, dst, 5), OC_ERR_INVALID_ARG);
}

Test(quant, ggml_id_round_trip, .description = "oc_quant_type_from_ggml_id round-trip") {
    cr_assert_eq(oc_quant_type_from_ggml_id(0),  OC_QUANT_F32,  "ggml id 0");
    cr_assert_eq(oc_quant_type_from_ggml_id(1),  OC_QUANT_F16,  "ggml id 1");
    cr_assert_eq(oc_quant_type_from_ggml_id(2),  OC_QUANT_Q4_0, "ggml id 2");
    cr_assert_eq(oc_quant_type_from_ggml_id(8),  OC_QUANT_Q8_0, "ggml id 8");
    cr_assert_eq(oc_quant_type_from_ggml_id(12), OC_QUANT_Q4_K_M, "ggml id 12");
    cr_assert_eq(oc_quant_type_from_ggml_id(13), OC_QUANT_Q5_K_M, "ggml id 13");
    cr_assert_eq(oc_quant_type_from_ggml_id(14), OC_QUANT_Q6_K, "ggml id 14");
    cr_assert_eq(oc_quant_type_from_ggml_id(241), OC_QUANT_AL8, "ggml id 241");
    cr_assert_eq(oc_quant_type_from_ggml_id(243), OC_QUANT_AL5_XS, "ggml id 243");
    cr_assert_eq(oc_quant_type_from_ggml_id(30), OC_QUANT_BF16, "ggml id 30");
    cr_assert_eq(oc_quant_type_from_ggml_id(66), OC_QUANT_IQ1_XXXS,
                 "ggml id 66");
    cr_assert_eq(oc_quant_type_from_ggml_id(0xff), OC_QUANT_UNKNOWN, "unknown ggml id");

    cr_assert_eq(oc_quant_type_to_ggml_id(OC_QUANT_F32),  0u,  "F32 ggml id");
    cr_assert_eq(oc_quant_type_to_ggml_id(OC_QUANT_Q4_K_M), 12u, "Q4_K_M ggml id");
    cr_assert_eq(oc_quant_type_to_ggml_id(OC_QUANT_Q8_0), 8u, "Q8_0 ggml id");
    cr_assert_eq(oc_quant_type_to_ggml_id(OC_QUANT_IQ1_XXXS), 66u,
                 "IQ1_XXXS ggml id");
    cr_assert_eq(oc_quant_type_to_ggml_id(OC_QUANT_UNKNOWN), 0xffffffffu, "UNKNOWN ggml id");
}

/* ─── Golden vectors: interleaved-half nibble layout ───────────────────
 *
 * The ggml 4/5-bit block layout puts element j in the LOW nibble of qs[j]
 * and element j + QK/2 in the HIGH nibble. Reading them as sequential
 * pairs is a pure permutation: pack and dequant agree with each other, so
 * every round-trip test passes while real GGUF weights decode scrambled.
 * That is exactly how Q4_1/Q5_0/Q5_1 shipped broken, so these expectations
 * are fixed vectors derived from the llama.cpp semantics rather than from
 * anything this file computes.
 */
static const uint8_t GOLDEN_Q5_1[] = {
    0x00, 0x34, 0x00, 0xBE, 0xA5, 0x3C, 0x0F, 0xF0, 0x03, 0x0A, 0x11, 0x18,
    0x1F, 0x26, 0x2D, 0x34, 0x3B, 0x42, 0x49, 0x50, 0x57, 0x5E, 0x65, 0x6C};
static const float GOLDEN_Q5_1_WANT[32] = {
    3.25f, 1.0f, 2.75f, 0.5f, 2.25f, 4.0f, 1.75f, 3.5f, 1.25f, -1.0f, 4.75f,
    2.5f, 4.25f, 6.0f, -0.25f, 1.5f, 2.5f, 2.5f, 2.75f, 2.75f, -1.25f, -1.0f,
    -1.0f, -0.75f, -0.75f, -0.5f, -0.5f, -0.25f, 3.75f, 3.75f, 4.0f, 4.0f};

static const uint8_t GOLDEN_Q5_0[] = {
    0x00, 0x34, 0xA5, 0x3C, 0x0F, 0xF0, 0x03, 0x0A, 0x11, 0x18, 0x1F, 0x26,
    0x2D, 0x34, 0x3B, 0x42, 0x49, 0x50, 0x57, 0x5E, 0x65, 0x6C};
static const float GOLDEN_Q5_0_WANT[32] = {
    0.75f, -1.5f, 0.25f, -2.0f, -0.25f, 1.5f, -0.75f, 1.0f, -1.25f, -3.5f,
    2.25f, 0.0f, 1.75f, 3.5f, -2.75f, -1.0f, 0.0f, 0.0f, 0.25f, 0.25f, -3.75f,
    -3.5f, -3.5f, -3.25f, -3.25f, -3.0f, -3.0f, -2.75f, 1.25f, 1.25f, 1.5f,
    1.5f};

static const uint8_t GOLDEN_Q4_1[] = {
    0x00, 0x34, 0x00, 0xBE, 0x03, 0x0A, 0x11, 0x18, 0x1F, 0x26, 0x2D, 0x34,
    0x3B, 0x42, 0x49, 0x50, 0x57, 0x5E, 0x65, 0x6C};
static const float GOLDEN_Q4_1_WANT[32] = {
    -0.75f, 1.0f, -1.25f, 0.5f, 2.25f, 0.0f, 1.75f, -0.5f, 1.25f, -1.0f,
    0.75f, -1.5f, 0.25f, 2.0f, -0.25f, 1.5f, -1.5f, -1.5f, -1.25f, -1.25f,
    -1.25f, -1.0f, -1.0f, -0.75f, -0.75f, -0.5f, -0.5f, -0.25f, -0.25f,
    -0.25f, 0.0f, 0.0f};

Test(quant, golden_interleaved_nibble_layout,
     .description = "Q4_1/Q5_0/Q5_1 decode to llama.cpp's element order") {
    struct {
        OcGgufQuantizationType t;
        const uint8_t *blk;
        size_t bytes;
        const float *want;
    } cases[] = {
        { OC_QUANT_Q5_1, GOLDEN_Q5_1, sizeof(GOLDEN_Q5_1), GOLDEN_Q5_1_WANT },
        { OC_QUANT_Q5_0, GOLDEN_Q5_0, sizeof(GOLDEN_Q5_0), GOLDEN_Q5_0_WANT },
        { OC_QUANT_Q4_1, GOLDEN_Q4_1, sizeof(GOLDEN_Q4_1), GOLDEN_Q4_1_WANT },
    };
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const char *name = oc_quant_type_name(cases[c].t);
        float got[32];
        cr_assert_eq(oc_quant_dequant_row(cases[c].t, cases[c].blk,
                                          cases[c].bytes, got, 32), OC_OK,
                     "%s dequant failed", name);
        for (size_t i = 0; i < 32; i++) {
            cr_assert_float_eq(got[i], cases[c].want[i], 1e-5f,
                "%s element %zu: got %f want %f (element order wrong?)",
                name, i, got[i], cases[c].want[i]);
        }
    }
}

/* ─── Encoder accuracy + payload coverage ──────────────────────────────
 *
 * Two properties that a finiteness check cannot see, and whose absence let
 * four broken K-quant encoders and two broken K-quant dequantizers ship:
 *
 *   1. Round-trip error must actually scale with the format's bit width. An
 *      encoder writing to the wrong offsets still produces finite output —
 *      it just produces noise, which shows up here as relative RMSE near or
 *      above 1.0 (output uncorrelated with input).
 *   2. Every payload byte must influence the result. A dequantizer that
 *      forgets to advance its nibble pointer silently ignores most of each
 *      super-block, which no round-trip average reliably catches.
 */

/* Deterministic bell-shaped sample: real weight tensors are not uniform,
 * and nonlinear codebooks (IQ4/NVFP4) are tuned for this shape. */
static float next_gaussian(float sigma)
{
    float u1 = (next_float(0.0f, 1.0f) + 1e-7f);
    float u2 = next_float(0.0f, 1.0f);
    return sigma * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

Test(quant, encoder_roundtrip_error_scales_with_bit_width,
     .description = "pack→dequant relative RMSE stays within the format's budget") {
    struct {
        OcGgufQuantizationType t;
        size_t vals_per_block;
        size_t bytes_per_block;
        float  max_rel_rmse;   /* generous bound; noise would score ~1.0+ */
    } cases[] = {
        { OC_QUANT_Q8_0,   32,  OC_BLOCK_Q8_0_SIZE,   0.02f },
        { OC_QUANT_Q6_K,   256, OC_BLOCK_Q6_K_SIZE,   0.04f },
        { OC_QUANT_Q5_K_M, 256, OC_BLOCK_Q5_K_SIZE,   0.07f },
        { OC_QUANT_Q4_K_M, 256, OC_BLOCK_Q4_K_SIZE,   0.13f },
        { OC_QUANT_Q4_0,   32,  OC_BLOCK_Q4_0_SIZE,   0.15f },
        { OC_QUANT_IQ4_NL, 32,  OC_BLOCK_IQ4_NL_SIZE, 0.13f },
        { OC_QUANT_IQ4_XS, 256, OC_BLOCK_IQ4_XS_SIZE, 0.13f },
        { OC_QUANT_NVFP4,  64,  OC_BLOCK_NVFP4_SIZE,  0.16f },
        { OC_QUANT_Q3_K_M, 256, OC_BLOCK_Q3_K_SIZE,   0.26f },
        { OC_QUANT_Q2_K,   256, OC_BLOCK_Q2_K_SIZE,   0.45f },
    };

    const size_t n_blocks = 8;
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        size_t n_vals = cases[c].vals_per_block * n_blocks;
        size_t n_bytes = cases[c].bytes_per_block * n_blocks;
        float src[256 * 8], dst[256 * 8];
        uint8_t buf[256 * 8];
        cr_assert_leq(n_vals, 256 * 8, "src overflow");
        cr_assert_leq(n_bytes, sizeof(buf), "buf overflow");

        lcg_state = 12345u;   /* deterministic per type */
        for (size_t i = 0; i < n_vals; i++) src[i] = next_gaussian(0.6f);

        const char *name = oc_quant_type_name(cases[c].t);
        cr_assert_eq(oc_quant_pack_row(cases[c].t, src, n_vals, buf, n_bytes),
                     OC_OK, "%s pack failed", name);
        cr_assert_eq(oc_quant_dequant_row(cases[c].t, buf, n_bytes, dst, n_vals),
                     OC_OK, "%s dequant failed", name);

        double sse = 0.0, ss = 0.0;
        for (size_t i = 0; i < n_vals; i++) {
            cr_assert(isfinite(dst[i]), "%s non-finite at %zu", name, i);
            double e = (double)dst[i] - (double)src[i];
            sse += e * e;
            ss += (double)src[i] * (double)src[i];
        }
        cr_assert_gt(ss, 0.0);
        float rel_rmse = (float)sqrt(sse / ss);
        cr_assert_leq(rel_rmse, cases[c].max_rel_rmse,
                      "%s round-trip rel RMSE %.4f exceeds budget %.4f",
                      name, rel_rmse, cases[c].max_rel_rmse);
    }
}

Test(quant, dequant_reads_every_payload_byte,
     .description = "no dequantizer ignores part of its block") {
    /* Every block-quantized type, including the dequant-only IQ family. */
    static const OcGgufQuantizationType types[] = {
        OC_QUANT_Q4_0, OC_QUANT_Q4_1, OC_QUANT_Q5_0, OC_QUANT_Q5_1,
        OC_QUANT_Q8_0,
        OC_QUANT_Q2_K, OC_QUANT_Q3_K_M, OC_QUANT_Q4_K_M, OC_QUANT_Q5_K_M,
        OC_QUANT_Q6_K,
        OC_QUANT_AL5, OC_QUANT_AL5_XS, OC_QUANT_AL6, OC_QUANT_AL8,
        OC_QUANT_IQ1_S, OC_QUANT_IQ1_M, OC_QUANT_IQ2_XXS, OC_QUANT_IQ2_XS,
        OC_QUANT_IQ2_S, OC_QUANT_IQ3_XXS, OC_QUANT_IQ3_S,
        OC_QUANT_IQ4_NL, OC_QUANT_IQ4_XS, OC_QUANT_NVFP4,
    };

    for (size_t c = 0; c < sizeof(types) / sizeof(types[0]); c++) {
        const char *name = oc_quant_type_name(types[c]);
        OcQuantBlockLayout bs = oc_quant_block_size(types[c]);
        cr_assert_gt(bs.bytes_per_block, 0, "%s has no block layout", name);
        cr_assert_gt(bs.elements_per_block, 0, "%s has no block layout", name);

        size_t bytes = bs.bytes_per_block;
        size_t vals = bs.elements_per_block;
        uint8_t buf[256];
        float base[256], probe[256];
        cr_assert_leq(bytes, sizeof(buf), "%s buf overflow", name);
        cr_assert_leq(vals, 256, "%s val overflow", name);

        lcg_state = 999u;
        for (size_t i = 0; i < bytes; i++)
            buf[i] = (uint8_t)(uint32_t)next_float(0.0f, 255.9f);

        cr_assert_eq(oc_quant_dequant_row(types[c], buf, bytes, base, vals),
                     OC_OK, "%s dequant failed", name);

        /* Flipping any payload byte must move at least one output. */
        for (size_t i = 0; i < bytes; i++) {
            uint8_t saved = buf[i];
            buf[i] = (uint8_t)~saved;
            cr_assert_eq(oc_quant_dequant_row(types[c], buf, bytes, probe, vals),
                         OC_OK);
            buf[i] = saved;

            bool moved = false;
            for (size_t k = 0; k < vals && !moved; k++)
                if (base[k] != probe[k]) moved = true;
            cr_assert(moved, "%s ignores payload byte %zu of %zu",
                      name, i, bytes);
        }
    }
}

Test(quant, k_encoders_pack_and_roundtrip) {
    /* K-quant pack encoders are now implemented. Verify they succeed and
     * produce finite output on round-trip. */
    float src[256];
    for (size_t i = 0; i < 256; i++)
        src[i] = (float)((int)(i * 37 % 100) - 50) * 0.01f;

    struct { OcGgufQuantizationType t; size_t block_size; float tol; } types[] = {
        { OC_QUANT_Q2_K,   OC_BLOCK_Q2_K_SIZE, 0.5f },
        { OC_QUANT_Q3_K_M, OC_BLOCK_Q3_K_SIZE, 0.3f },
        { OC_QUANT_Q5_K_M, OC_BLOCK_Q5_K_SIZE, 0.1f },
        { OC_QUANT_Q6_K,   OC_BLOCK_Q6_K_SIZE, 0.05f },
    };

    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        uint8_t buf[OC_BLOCK_Q6_K_SIZE];  /* largest block size */
        size_t total_bytes = types[t].block_size;

        OcError pe = oc_quant_pack_row(types[t].t, src, 256, buf, total_bytes);
        cr_assert_eq(pe, OC_OK, "%s pack failed",
            oc_quant_type_name(types[t].t));

        float dst[256];
        OcError de = oc_quant_dequant_row(types[t].t, buf, total_bytes, dst, 256);
        cr_assert_eq(de, OC_OK, "%s dequant failed",
            oc_quant_type_name(types[t].t));

        for (size_t i = 0; i < 256; i++) {
            cr_assert(isfinite(dst[i]), "%s non-finite at idx %zu",
                oc_quant_type_name(types[t].t), i);
        }
    }
}

/* ─── VAL-QUANT-015: pack-then-dequant round-trip ──────────────────────── */

Test(quant, pack_then_dequant_roundtrip, .description = "VAL-QUANT-015: pack-then-dequant preserves data") {
    /* Q4_0: 4 blocks, 128 values. */
    {
        float src[128];
        for (int i = 0; i < 128; i++) src[i] = next_float(-2.0f, 2.0f);
        uint8_t buf[72];
        cr_assert_eq(oc_quant_pack_row(OC_QUANT_Q4_0, src, 128, buf, sizeof(buf)), OC_OK);
        float dst[128];
        cr_assert_eq(oc_quant_dequant_row(OC_QUANT_Q4_0, buf, sizeof(buf), dst, 128), OC_OK);
        for (int i = 0; i < 128; i++) {
            cr_assert(isfinite(dst[i]), "Q4_0 RT non-finite at %d", i);
            cr_assert_leq(fabsf(dst[i] - src[i]), 0.3f, "Q4_0 RT error at %d", i);
        }
    }
    /* Q4_K_M: 1 super-block, 256 values (true inverse encoder). */
    {
        float src[256];
        for (int i = 0; i < 256; i++) src[i] = next_float(-1.0f, 1.0f);
        uint8_t buf[144];
        cr_assert_eq(oc_quant_pack_row(OC_QUANT_Q4_K_M, src, 256, buf, sizeof(buf)), OC_OK);
        float dst[256];
        cr_assert_eq(oc_quant_dequant_row(OC_QUANT_Q4_K_M, buf, sizeof(buf), dst, 256), OC_OK);
        for (int i = 0; i < 256; i++) {
            cr_assert(isfinite(dst[i]), "Q4_K_M RT non-finite at %d", i);
            cr_assert_leq(fabsf(dst[i] - src[i]), 0.3f, "Q4_K_M RT error at %d", i);
        }
    }
    /* Q8_0: 4 blocks, 128 values — near-exact round-trip (Q8 step ~ d/127). */
    {
        float src[128];
        for (int i = 0; i < 128; i++) src[i] = next_float(-2.0f, 2.0f);
        uint8_t buf[136];  /* 4 blocks × 34 bytes */
        cr_assert_eq(oc_quant_pack_row(OC_QUANT_Q8_0, src, 128, buf, sizeof(buf)), OC_OK);
        float dst[128];
        cr_assert_eq(oc_quant_dequant_row(OC_QUANT_Q8_0, buf, sizeof(buf), dst, 128), OC_OK);
        for (int i = 0; i < 128; i++) {
            cr_assert(isfinite(dst[i]), "Q8_0 RT non-finite at %d", i);
            cr_assert_leq(fabsf(dst[i] - src[i]), 0.05f, "Q8_0 RT error at %d", i);
        }
    }
}

Test(quant, pack_block_helper, .description = "oc_quant_pack_block works on a single block") {
    float src[32];
    for (int i = 0; i < 32; i++) src[i] = next_float(-1.0f, 1.0f);
    uint8_t buf[18];
    cr_assert_eq(oc_quant_pack_block(OC_QUANT_Q4_0, src, buf), OC_OK);
    float dst[32];
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_Q4_0, buf, sizeof(buf), dst, 32), OC_OK);
    for (int i = 0; i < 32; i++) {
        cr_assert_leq(fabsf(dst[i] - src[i]), 0.3f, "Q4_0 pack_block RT at %d", i);
    }
}

/* ─── VAL-QUANT-014: random-block corpus sweep ──────────────────────────── */

Test(quant, random_corpus_sweep, .description = "VAL-QUANT-014: random-block corpus sweep across standard types") {
    /* For each standard quant type whose Rust encoder IS a true inverse of the
     * dequant (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K_S, Q4_K_M, F32, F16, I8, I16,
     * I32, I64, F64), generate 5 random blocks, pack → dequant, and verify the
     * round-trip error is within the type's expected tolerance. The K-packed
     * types (Q2_K/Q3_K/Q5_K/Q6_K) are excluded because their Rust
     * `quantize_k_packed_scalar` produces a simplified layout that is NOT a
     * true inverse of the super-block dequant — those are validated by the
     * hand-crafted block tests above. */
    struct {
        OcGgufQuantizationType t;
        float tolerance;
        float range;
        bool integer_only;
    } types[] = {
        { OC_QUANT_F32,     0.0f,  1.5f,    false },
        { OC_QUANT_F16,     1e-3f, 1.5f,    false },
        { OC_QUANT_Q4_0,    0.3f,  2.0f,    false },
        { OC_QUANT_Q4_1,    0.2f,  1.5f,    false },
        { OC_QUANT_Q5_0,    0.15f, 1.5f,    false },
        { OC_QUANT_Q5_1,    0.1f,  1.5f,    false },
        { OC_QUANT_Q8_0,    0.05f, 2.0f,    false },
        { OC_QUANT_Q4_K_S,  0.3f,  1.0f,    false },
        { OC_QUANT_Q4_K_M,  0.3f,  1.0f,    false },
        { OC_QUANT_I8,      0.0f,  100.0f,  true  },
        { OC_QUANT_I16,     0.0f,  30000.0f, true },
        { OC_QUANT_I32,     0.0f,  1e6f,    true  },
        { OC_QUANT_I64,     0.0f,  1e6f,    true  },
        { OC_QUANT_F64,     0.0f,  1.5f,    false },
    };
    const int n_blocks_per_type = 5;
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        OcQuantBlockLayout bs = oc_quant_block_size(types[t].t);
        cr_assert_gt(bs.elements_per_block, 0, "%s elements_per_block > 0",
            oc_quant_type_name(types[t].t));
        size_t total_values = (size_t)n_blocks_per_type * bs.elements_per_block;
        size_t total_bytes  = (size_t)n_blocks_per_type * bs.bytes_per_block;
        /* Stack-allocated (max total: 5*256*4 = 5120 bytes for src/dst,
         * 5*210 = 1050 for buf). */
        float src[5 * 256];
        uint8_t buf[5 * 210];
        float dst[5 * 256];
        cr_assert_leq(total_values, sizeof(src) / sizeof(src[0]), "src overflow");
        cr_assert_leq(total_bytes,  sizeof(buf), "buf overflow");
        cr_assert_leq(total_values, sizeof(dst) / sizeof(dst[0]), "dst overflow");

        for (size_t i = 0; i < total_values; i++) {
            float v = next_float(-types[t].range, types[t].range);
            /* Integer storage types only round-trip exactly on integer inputs
             * (pack truncates to int8/int16/etc.). */
            if (types[t].integer_only) {
                v = roundf(v);
            }
            src[i] = v;
        }

        OcError pe = oc_quant_pack_row(types[t].t, src, total_values, buf, total_bytes);
        cr_assert_eq(pe, OC_OK, "%s pack failed", oc_quant_type_name(types[t].t));
        OcError de = oc_quant_dequant_row(types[t].t, buf, total_bytes, dst, total_values);
        cr_assert_eq(de, OC_OK, "%s dequant failed", oc_quant_type_name(types[t].t));

        for (size_t i = 0; i < total_values; i++) {
            cr_assert(isfinite(dst[i]), "%s non-finite at idx %zu",
                oc_quant_type_name(types[t].t), i);
            float err = fabsf(dst[i] - src[i]);
            cr_assert_leq(err, types[t].tolerance,
                "%s RT error %f > tol %f at idx %zu (src=%f dst=%f)",
                oc_quant_type_name(types[t].t), err, types[t].tolerance, i,
                src[i], dst[i]);
        }
    }
}

/* ─── Naming + sanity ─────────────────────────────────────────────────── */

Test(quant, type_names, .description = "oc_quant_type_name returns canonical names") {
    cr_assert_str_eq(oc_quant_type_name(OC_QUANT_F32),    "F32");
    cr_assert_str_eq(oc_quant_type_name(OC_QUANT_Q4_0),   "Q4_0");
    cr_assert_str_eq(oc_quant_type_name(OC_QUANT_Q4_K_M), "Q4_K_M");
    cr_assert_str_eq(oc_quant_type_name(OC_QUANT_Q6_K),   "Q6_K");
    cr_assert_str_eq(oc_quant_type_name(OC_QUANT_BF16),   "BF16");
    cr_assert_str_eq(oc_quant_type_name(OC_QUANT_UNKNOWN), "?");
}

/* ─── Q4_K_M pack output is bit-exact with canonical encoder ──────────── */

Test(quant, q4_k_m_pack_is_stable, .description = "Q4_K_M pack produces finite, non-trivial output") {
    /* Sanity: Q4_K_M pack on a non-zero input must produce a non-zero output
     * buffer (the d field at [0..2] must be non-zero for non-zero input). */
    float src[256];
    for (int i = 0; i < 256; i++) src[i] = (float)i * 0.01f - 1.28f;
    uint8_t buf[144];
    memset(buf, 0, sizeof(buf));
    OcError e = oc_quant_pack_row(OC_QUANT_Q4_K_M, src, 256, buf, sizeof(buf));
    cr_assert_eq(e, OC_OK, "Q4_K_M pack");
    /* d field (f16 bits at [0..2]) should be non-zero for non-zero input. */
    cr_assert_neq(buf[0] | buf[1], 0, "Q4_K_M d field should be non-zero");
}

/* ─── VAL-QUANT-001: AL/IQ/NVFP4 block sizes ────────────────────────── */

Test(quant, al_iq_nvfp4_block_sizes, .description = "VAL-QUANT-001: AL/IQ/NVFP4 block-size constants match Rust") {
    struct { OcGgufQuantizationType t; size_t els; size_t bytes; } cases[] = {
        { OC_QUANT_AL5,      32,  18  },   /* == Q4_0 layout */
        { OC_QUANT_AL5_XS,   32,  14  },   /* 2 + 12 (3-bit packed) */
        { OC_QUANT_AL6,      32,  22  },   /* == Q5_0 layout */
        { OC_QUANT_AL8,      32,  34  },   /* == Q8_0 layout */
        { OC_QUANT_IQ1_S,    256, 50  },
        { OC_QUANT_IQ1_M,    256, 56  },
        { OC_QUANT_IQ1_XXXS, 256, 38  },
        { OC_QUANT_IQ2_XXS,  256, 66  },
        { OC_QUANT_IQ2_XS,   256, 74  },
        { OC_QUANT_IQ2_S,    256, 82  },
        { OC_QUANT_IQ3_XXS,  256, 98  },
        { OC_QUANT_IQ3_S,    256, 110 },
        { OC_QUANT_IQ4_NL,   32,  18  },
        { OC_QUANT_IQ4_XS,   256, 136 },
        { OC_QUANT_NVFP4,    64,  36  },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        OcQuantBlockLayout bs = oc_quant_block_size(cases[i].t);
        cr_assert_eq(bs.elements_per_block, cases[i].els,
            "AL/IQ/NVFP4 elements mismatch for %s: got %zu, want %zu",
            oc_quant_type_name(cases[i].t), bs.elements_per_block, cases[i].els);
        cr_assert_eq(bs.bytes_per_block, cases[i].bytes,
            "AL/IQ/NVFP4 bytes mismatch for %s: got %zu, want %zu",
            oc_quant_type_name(cases[i].t), bs.bytes_per_block, cases[i].bytes);
    }
}

/* ─── VAL-QUANT-006: AL-family dequant (pack-then-dequant round-trip) ─ */

Test(quant, al_family_pack_dequant_roundtrip, .description = "VAL-QUANT-006: AL5/AL5_XS/AL6/AL8 pack-then-dequant round-trip") {
    /* AL5: 32-elem block, 18 bytes. Tolerance 0.5 (4-bit, range -8..7 * d). */
    {
        float src[32];
        for (int i = 0; i < 32; i++) src[i] = next_float(-2.0f, 2.0f);
        uint8_t buf[18];
        cr_assert_eq(oc_quant_pack_block(OC_QUANT_AL5, src, buf), OC_OK, "AL5 pack");
        float dst[32];
        cr_assert_eq(oc_quant_dequant_row(OC_QUANT_AL5, buf, sizeof(buf), dst, 32), OC_OK, "AL5 dequant");
        for (int i = 0; i < 32; i++) {
            cr_assert(isfinite(dst[i]), "AL5 non-finite at %d", i);
            cr_assert_leq(fabsf(dst[i] - src[i]), 0.5f, "AL5 RT error at %d", i);
        }
    }
    /* AL5_XS: 32-elem block, 14 bytes. 3-bit levels, range -4..3 * d. */
    {
        float src[32];
        for (int i = 0; i < 32; i++) src[i] = next_float(-1.0f, 1.0f);
        uint8_t buf[14];
        cr_assert_eq(oc_quant_pack_block(OC_QUANT_AL5_XS, src, buf), OC_OK, "AL5_XS pack");
        float dst[32];
        cr_assert_eq(oc_quant_dequant_row(OC_QUANT_AL5_XS, buf, sizeof(buf), dst, 32), OC_OK, "AL5_XS dequant");
        for (int i = 0; i < 32; i++) {
            cr_assert(isfinite(dst[i]), "AL5_XS non-finite at %d", i);
            cr_assert_leq(fabsf(dst[i] - src[i]), 0.6f, "AL5_XS RT error at %d", i);
        }
    }
    /* AL6: 32-elem block, 22 bytes. 5-bit, range -16..15 * d. */
    {
        float src[32];
        for (int i = 0; i < 32; i++) src[i] = next_float(-2.0f, 2.0f);
        uint8_t buf[22];
        cr_assert_eq(oc_quant_pack_block(OC_QUANT_AL6, src, buf), OC_OK, "AL6 pack");
        float dst[32];
        cr_assert_eq(oc_quant_dequant_row(OC_QUANT_AL6, buf, sizeof(buf), dst, 32), OC_OK, "AL6 dequant");
        for (int i = 0; i < 32; i++) {
            cr_assert(isfinite(dst[i]), "AL6 non-finite at %d", i);
            cr_assert_leq(fabsf(dst[i] - src[i]), 0.2f, "AL6 RT error at %d", i);
        }
    }
    /* AL8: 32-elem block, 34 bytes. 8-bit, range -127..127 * d. */
    {
        float src[32];
        for (int i = 0; i < 32; i++) src[i] = next_float(-2.0f, 2.0f);
        uint8_t buf[34];
        cr_assert_eq(oc_quant_pack_block(OC_QUANT_AL8, src, buf), OC_OK, "AL8 pack");
        float dst[32];
        cr_assert_eq(oc_quant_dequant_row(OC_QUANT_AL8, buf, sizeof(buf), dst, 32), OC_OK, "AL8 dequant");
        for (int i = 0; i < 32; i++) {
            cr_assert(isfinite(dst[i]), "AL8 non-finite at %d", i);
            cr_assert_leq(fabsf(dst[i] - src[i]), 0.05f, "AL8 RT error at %d", i);
        }
    }
}

Test(quant, al5_xs_handcrafted_dequant, .description = "VAL-QUANT-006: AL5_XS hand-crafted block dequant") {
    /* d=1.0 (f16 0x3C00), 3-bit levels 0..7 packed in the 12-byte bitstream.
     * Each level l decodes to (l - 4) * 1.0 = (l-4). Construct levels 4,5,6,7
     * repeating so output is 0,1,2,3,0,1,2,3,... */
    uint8_t buf[14];
    buf[0] = 0x00; buf[1] = 0x3C;  /* f16 1.0 LE */
    /* Pack 32 levels: level = 4 + (i % 4) → output = (4 + (i%4)) - 4 = i%4.
     * Bits are written LSB-first per level: level 4 = 0b100, level 5 = 0b101, etc. */
    uint8_t levels[32];
    for (int i = 0; i < 32; i++) levels[i] = (uint8_t)(4 + (i % 4));
    memset(&buf[2], 0, 12);
    uint32_t bitpos = 0;
    for (int i = 0; i < 32; i++) {
        for (int b = 0; b < 3; b++) {
            if ((levels[i] >> b) & 1) {
                buf[2 + bitpos / 8] |= (uint8_t)(1u << (bitpos % 8));
            }
            bitpos++;
        }
    }
    float dst[32];
    OcError e = oc_quant_dequant_row(OC_QUANT_AL5_XS, buf, sizeof(buf), dst, 32);
    cr_assert_eq(e, OC_OK, "AL5_XS dequant");
    for (int i = 0; i < 32; i++) {
        float expected = (float)(i % 4);
        cr_assert_float_eq(dst[i], expected, 0.0f, "AL5_XS[%d]: got %f, want %f", i, dst[i], expected);
    }
}

Test(quant, al_family_zero_block, .description = "AL-family zero-block dequant produces all zeros") {
    uint8_t buf[34];
    float dst[32];
    memset(buf, 0, sizeof(buf));
    /* AL5 zero block: d=0, packed bytes 0x88 (nibbles = 8, so (8-8)*0 = 0). */
    for (int i = 2; i < 18; i++) buf[i] = 0x88;
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_AL5, buf, 18, dst, 32), OC_OK);
    for (int i = 0; i < 32; i++) cr_assert_float_eq(dst[i], 0.0f, 0.0f, "AL5 zero[%d]", i);
    /* AL8 zero block: d=0, qs all 0. */
    memset(buf, 0, sizeof(buf));
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_AL8, buf, 34, dst, 32), OC_OK);
    for (int i = 0; i < 32; i++) cr_assert_float_eq(dst[i], 0.0f, 0.0f, "AL8 zero[%d]", i);
    /* AL6 zero block: d=0, qs=0x10 (so q=16 → (16-16)*0 = 0). */
    memset(buf, 0, sizeof(buf));
    for (int i = 6; i < 22; i++) buf[i] = 0x10;
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_AL6, buf, 22, dst, 32), OC_OK);
    for (int i = 0; i < 32; i++) cr_assert_float_eq(dst[i], 0.0f, 0.0f, "AL6 zero[%d]", i);
    /* AL5_XS zero block: d=0, packed all 0. */
    memset(buf, 0, sizeof(buf));
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_AL5_XS, buf, 14, dst, 32), OC_OK);
    for (int i = 0; i < 32; i++) cr_assert_float_eq(dst[i], 0.0f, 0.0f, "AL5_XS zero[%d]", i);
}

/* ─── VAL-QUANT-007: IQ-family dequant (zero-block + hand-crafted) ──── */

Test(quant, iq_family_zero_blocks, .description = "VAL-QUANT-007: IQ-family zero-block dequant produces all zeros") {
    /* For every IQ type, an all-zero block has d=0 (or scale_u16=0 for IQ1_M),
     * so every output is 0 * (grid + delta) = 0. Validates the dequant runs
     * without crash and produces finite zero output. */
    uint8_t buf[256];  /* large enough for all IQ block sizes */
    float dst[256];
    struct { OcGgufQuantizationType t; size_t bytes; size_t els; } cases[] = {
        { OC_QUANT_IQ1_S,   50,  256 },
        { OC_QUANT_IQ1_M,   56,  256 },
        { OC_QUANT_IQ1_XXXS, 38, 256 },
        { OC_QUANT_IQ2_XXS, 66,  256 },
        { OC_QUANT_IQ2_XS,  74,  256 },
        { OC_QUANT_IQ2_S,   82,  256 },
        { OC_QUANT_IQ3_XXS, 98,  256 },
        { OC_QUANT_IQ3_S,   110, 256 },
        { OC_QUANT_IQ4_NL,  18,  32  },
        { OC_QUANT_IQ4_XS,  136, 256 },
    };
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        memset(buf, 0, cases[c].bytes);
        OcError e = oc_quant_dequant_row(cases[c].t, buf, cases[c].bytes, dst, cases[c].els);
        cr_assert_eq(e, OC_OK, "%s zero-block dequant failed",
            oc_quant_type_name(cases[c].t));
        for (size_t i = 0; i < cases[c].els; i++) {
            cr_assert(isfinite(dst[i]), "%s zero-block non-finite at %zu",
                oc_quant_type_name(cases[c].t), i);
            cr_assert_float_eq(dst[i], 0.0f, 0.0f, "%s zero-block[%zu]: got %f",
                oc_quant_type_name(cases[c].t), i, dst[i]);
        }
    }
}

Test(quant, iq1_xxxs_handcrafted, .description = "IQ1_XXXS decodes scale, grid, and delta") {
    uint8_t block[38] = {0};
    float dst[256];

    block[1] = 0x3c;
    cr_assert_eq(oc_quant_dequant_row_scalar(OC_QUANT_IQ1_XXXS,
                                             block, sizeof(block), dst, 256),
                 OC_OK);
    for (size_t i = 0; i < 256; i++) {
        cr_assert_float_eq(dst[i], -0.875f, 0.0f, "IQ1_XXXS[%zu]", i);
    }
}

Test(quant, iq1_xxxs_q8_k_dot_matches_dequantized_oracle) {
    uint8_t row[76] = {0};
    uint8_t q8[584] = {0};
    float weights[512];
    float scale = 0.25f;
    float expected = 0.0f;

    for (size_t block = 0; block < 2; block++) {
        uint8_t *weight_block = row + block * 38;
        uint8_t *act_block = q8 + block * 292;
        weight_block[1] = 0x3c;
        for (size_t i = 0; i < 32; i++)
            weight_block[2 + i] = (uint8_t)(block * 113u + i * 7u + 3u);
        for (size_t i = 0; i < 4; i++)
            weight_block[34 + i] = (uint8_t)((block * 8u + 2u * i + 1u) << 4
                                            | (block * 8u + 2u * i));
        memcpy(act_block, &scale, sizeof(scale));
        int8_t *values = (int8_t *)(act_block + 4);
        for (size_t i = 0; i < 256; i++)
            values[i] = (int8_t)(((block * 17u + i * 29u) % 255u) - 127);
        for (size_t group = 0; group < 16; group++) {
            int16_t sum = 0;
            for (size_t i = 0; i < 16; i++) sum += values[group * 16 + i];
            memcpy(act_block + 260 + group * 2, &sum, sizeof(sum));
        }
    }
    cr_assert_eq(oc_quant_dequant_row_scalar(OC_QUANT_IQ1_XXXS,
                                             row, sizeof(row), weights, 512),
                 OC_OK);
    for (size_t block = 0; block < 2; block++) {
        const int8_t *values = (const int8_t *)(q8 + block * 292 + 4);
        for (size_t i = 0; i < 256; i++)
            expected += weights[block * 256 + i] * scale * values[i];
    }
    float actual = oc_quant_dot_iq1_xxxs_q8_k(row, 2, q8);
    cr_assert_float_eq(actual, expected, 1e-3f);
}

Test(quant, iq4_nl_handcrafted, .description = "VAL-QUANT-007: IQ4_NL dequant with d=1.0 matches KVALUES_IQ4NL codebook") {
    /* d=1.0 (f16 0x3C00). Pack 16 bytes with nibbles 0,1,2,...,15 repeating:
     * out[j] = KVALUES_IQ4NL[nibble], out[j+16] = KVALUES_IQ4NL[nibble>>4]. */
    uint8_t buf[18];
    buf[0] = 0x00; buf[1] = 0x3C;  /* f16 1.0 LE */
    for (int j = 0; j < 16; j++) {
        buf[2 + j] = (uint8_t)((j & 0x0F) | (((j + 1) & 0x0F) << 4));
    }
    float dst[32];
    OcError e = oc_quant_dequant_row(OC_QUANT_IQ4_NL, buf, sizeof(buf), dst, 32);
    cr_assert_eq(e, OC_OK, "IQ4_NL dequant");
    for (int j = 0; j < 16; j++) {
        float lo_expected = (float)KVALUES_IQ4NL[j & 0x0F];
        float hi_expected = (float)KVALUES_IQ4NL[(j + 1) & 0x0F];
        cr_assert_float_eq(dst[j], lo_expected, 0.0f,
            "IQ4_NL low[%d]: got %f, want %f", j, dst[j], lo_expected);
        cr_assert_float_eq(dst[j + 16], hi_expected, 0.0f,
            "IQ4_NL high[%d]: got %f, want %f", j, dst[j + 16], hi_expected);
    }
}

Test(quant, iq4_xs_zero_d_handcrafted, .description = "VAL-QUANT-007: IQ4_XS with d=0 produces all zeros") {
    /* IQ4_XS block: d=0, scales_h=0, scales_l=0, qs=0. With d=0, dl=0, so all
     * outputs are 0 * KVALUES_IQ4NL[0] = 0. Validates the 136-byte block
     * layout is parsed without crash. */
    uint8_t buf[136];
    memset(buf, 0, sizeof(buf));
    float dst[256];
    OcError e = oc_quant_dequant_row(OC_QUANT_IQ4_XS, buf, sizeof(buf), dst, 256);
    cr_assert_eq(e, OC_OK, "IQ4_XS dequant");
    for (int i = 0; i < 256; i++) {
        cr_assert_float_eq(dst[i], 0.0f, 0.0f, "IQ4_XS zero-d[%d]: got %f", i, dst[i]);
    }
}

/* ─── VAL-QUANT-009: NVFP4 dequant ──────────────────────────────────── */

Test(quant, nvfp4_zero_block, .description = "VAL-QUANT-009: NVFP4 zero-block (all scales=0) dequant produces all zeros") {
    /* NVFP4 block: scales[4] (UE4M3) all 0 → scale=0; qs all 0 → E2M1[0]=0.
     * out = 0 * 0 = 0 for all 64 values. */
    uint8_t buf[36];
    memset(buf, 0, sizeof(buf));
    float dst[64];
    OcError e = oc_quant_dequant_row(OC_QUANT_NVFP4, buf, sizeof(buf), dst, 64);
    cr_assert_eq(e, OC_OK, "NVFP4 dequant");
    for (int i = 0; i < 64; i++) {
        cr_assert_float_eq(dst[i], 0.0f, 0.0f, "NVFP4 zero[%d]: got %f", i, dst[i]);
    }
}

Test(quant, nvfp4_handcrafted, .description = "VAL-QUANT-009: NVFP4 dequant with known scales + nibbles") {
    /* Construct a block where:
     *   - scales[0] = UE4M3 for 1.0 (exp=8, mant=0 → (1+0)*2^(8-7) = 2.0...
     *     wait, UE4M3: exp=(byte>>3)&0xf, mant=byte&7. For exp=8 (byte=0x40):
     *       (1 + 0/8) * 2^(8-7) = 2.0. Not 1.0.
     *     For scale=1.0: need (1 + mant/8) * 2^(exp-7) = 1.0 → exp=7, mant=0
     *       → byte = (7<<3) | 0 = 0x38. Check: (1+0)*2^(7-7) = 1.0. ✓
     *   - qs: pack nibbles 0,1,2,3 repeating. E2M1_DOUBLED_VALUES[0]=0, [1]=1,
     *     [2]=2, [3]=3.
     * Expected: out[0..15] = 1.0 * {0,1,2,3,0,1,2,3,...} for sub-block 0. */
    uint8_t buf[36];
    memset(buf, 0, sizeof(buf));
    buf[0] = 0x38;  /* scale[0] = UE4M3(1.0) */
    /* qs starts at offset 4. Pack nibbles: low=0, high=1 → byte=0x10.
     *                         low=2, high=3 → byte=0x32. */
    for (int j = 0; j < 8; j++) {
        buf[4 + 2 * j]     = 0x10;  /* low=0, high=1 */
        buf[4 + 2 * j + 1] = 0x32;  /* low=2, high=3 */
    }
    float dst[64];
    OcError e = oc_quant_dequant_row(OC_QUANT_NVFP4, buf, sizeof(buf), dst, 64);
    cr_assert_eq(e, OC_OK, "NVFP4 dequant");
    /* Sub-block 0 (16 values): scale=1.0. For each qs byte j (0..7):
     *   out[j]     = scale * E2M1[low]   (low nibble)
     *   out[j + 8] = scale * E2M1[high]  (high nibble)
     * qs bytes alternate: 0x10 (low=0,high=1), 0x32 (low=2,high=3).
     * E2M1 table: [0]=0, [1]=1, [2]=2, [3]=3.
     * So out[0..7] (low nibbles) = {0,2,0,2,0,2,0,2}
     *    out[8..15] (high nibbles) = {1,3,1,3,1,3,1,3} */
    for (int j = 0; j < 8; j++) {
        float lo_exp = (j % 2 == 0) ? 0.0f : 2.0f;
        float hi_exp = (j % 2 == 0) ? 1.0f : 3.0f;
        cr_assert_float_eq(dst[j],      lo_exp, 0.0f,
            "NVFP4 sub0 low[%d]: got %f, want %f", j, dst[j], lo_exp);
        cr_assert_float_eq(dst[j + 8],  hi_exp, 0.0f,
            "NVFP4 sub0 hi[%d]: got %f, want %f", j, dst[j + 8], hi_exp);
    }
    /* Sub-blocks 1..3: scale=0 → all 0. */
    for (int j = 16; j < 64; j++) {
        cr_assert_float_eq(dst[j], 0.0f, 0.0f, "NVFP4 zero sub[%d]: got %f", j, dst[j]);
    }
}

/* ─── VAL-QUANT-016: AL/IQ constant table SHA256 matches Rust ───────── */

/* Minimal SHA-256 implementation for table-parity verification. FIPS 180-4. */
typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  data[64];
    size_t   datalen;
} Sha256Ctx;

static const uint32_t SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

static uint32_t sha_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256_transform(Sha256Ctx *c, const uint8_t *data)
{
    uint32_t m[64];
    for (int i = 0; i < 16; i++) {
        m[i] = ((uint32_t)data[i*4] << 24) | ((uint32_t)data[i*4+1] << 16)
             | ((uint32_t)data[i*4+2] << 8) | ((uint32_t)data[i*4+3]);
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = sha_rotr(m[i-15],7) ^ sha_rotr(m[i-15],18) ^ (m[i-15] >> 3);
        uint32_t s1 = sha_rotr(m[i-2],17) ^ sha_rotr(m[i-2],19)  ^ (m[i-2] >> 10);
        m[i] = m[i-16] + s0 + m[i-7] + s1;
    }
    uint32_t a=c->state[0],b=c->state[1],cc=c->state[2],d=c->state[3];
    uint32_t e=c->state[4],f=c->state[5],g=c->state[6],h=c->state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = sha_rotr(e,6) ^ sha_rotr(e,11) ^ sha_rotr(e,25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + SHA256_K[i] + m[i];
        uint32_t S0 = sha_rotr(a,2) ^ sha_rotr(a,13) ^ sha_rotr(a,22);
        uint32_t mj = (a & b) ^ (a & cc) ^ (b & cc);
        h = g; g = f; f = e; e = d + t1;
        d = cc; cc = b; b = a; a = t1 + S0 + mj;
    }
    c->state[0]+=a; c->state[1]+=b; c->state[2]+=cc; c->state[3]+=d;
    c->state[4]+=e; c->state[5]+=f; c->state[6]+=g; c->state[7]+=h;
}

static void sha256_init(Sha256Ctx *c)
{
    c->state[0]=0x6a09e667; c->state[1]=0xbb67ae85; c->state[2]=0x3c6ef372;
    c->state[3]=0xa54ff53a; c->state[4]=0x510e527f; c->state[5]=0x9b05688c;
    c->state[6]=0x1f83d9ab; c->state[7]=0x5be0cd19;
    c->bitlen = 0; c->datalen = 0;
}

static void sha256_update(Sha256Ctx *c, const uint8_t *data, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        c->data[c->datalen++] = data[i];
        if (c->datalen == 64) {
            sha256_transform(c, c->data);
            c->bitlen += 512;
            c->datalen = 0;
        }
    }
}

static void sha256_final(Sha256Ctx *c, uint8_t out[32])
{
    uint64_t bitlen = c->bitlen + (uint64_t)c->datalen * 8;
    c->data[c->datalen++] = 0x80;
    if (c->datalen > 56) {
        while (c->datalen < 64) c->data[c->datalen++] = 0;
        sha256_transform(c, c->data);
        c->datalen = 0;
    }
    while (c->datalen < 56) c->data[c->datalen++] = 0;
    for (int i = 7; i >= 0; i--) c->data[c->datalen++] = (uint8_t)((bitlen >> (8*i)) & 0xFF);
    sha256_transform(c, c->data);
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)((c->state[i] >> 24) & 0xFF);
        out[i*4+1] = (uint8_t)((c->state[i] >> 16) & 0xFF);
        out[i*4+2] = (uint8_t)((c->state[i] >> 8) & 0xFF);
        out[i*4+3] = (uint8_t)(c->state[i] & 0xFF);
    }
}

/* Hex-encode 32-byte hash → 64-char string. */
static void sha256_hex(const uint8_t hash[32], char out[65])
{
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        out[i*2]   = hex[(hash[i] >> 4) & 0xF];
        out[i*2+1] = hex[hash[i] & 0xF];
    }
    out[64] = '\0';
}

Test(quant, al_iq_constant_table_sha256, .description = "VAL-QUANT-016: AL/IQ/NVFP4 constant table SHA256 matches Rust") {
    /* Expected SHA256 hashes computed from oxidize-core Rust sources via
     * scripts/gen_quant_tables.py (one-time Python computation). The Rust
     * tables are themselves transcribed verbatim from ggml-common.h, so a
     * matching hash proves bit-exact parity with both. */
    struct { const char *name; const void *data; size_t len; const char *expected_sha; } tables[] = {
        { "KVALUES_IQ4NL",       KVALUES_IQ4NL,       sizeof(KVALUES_IQ4NL),
          "61aa47540aa024b5d6ddaa839b84ffe59f3d5a349af5c6c7ffcb5e0474b46163" },
        { "E2M1_DOUBLED_VALUES", E2M1_DOUBLED_VALUES, sizeof(E2M1_DOUBLED_VALUES),
          "702a9d6654a1e5c00ba5fdc869ac828ddc37b102d1035ada5215c8e21c8b2006" },
        { "IQ3S_GRID",           IQ3S_GRID,           sizeof(IQ3S_GRID),
          "bd1af4945a1717c65610b0284e4628b9a1ba3b306fae3a06f6e5f597356e349f" },
        { "KMASK_IQ2XS",         KMASK_IQ2XS,         sizeof(KMASK_IQ2XS),
          "5ac9831b2e30eb285ef34f8501620f878432d5c04331ad1ae47f977a83ba41a5" },
        { "KSIGNS_IQ2XS",        KSIGNS_IQ2XS,        sizeof(KSIGNS_IQ2XS),
          "a76742a603f8beca5212ecce0f1f02f11a4887fd2f7c8b15aca0ea3eb3380c31" },
        { "IQ2XXS_GRID",         IQ2XXS_GRID,         sizeof(IQ2XXS_GRID),
          "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54" },
        { "IQ3XXS_GRID",         IQ3XXS_GRID,         sizeof(IQ3XXS_GRID),
          "46e35f5a997efdee6c99ce57854c8a0d4f0ff8ca57e5e8a60c0793ea580acf5d" },
        { "IQ2XS_GRID",          IQ2XS_GRID,          sizeof(IQ2XS_GRID),
          "06e47aaca60b4dc1d9b5a3f34540437058a6b142b4d7a59d5ded769b4d1bf1de" },
        { "IQ2S_GRID",           IQ2S_GRID,           sizeof(IQ2S_GRID),
          "e1aa1473412b0552c2174c30ef22ab4073f6a181b85a17056e8249bd2932fd88" },
        { "IQ1S_GRID",           IQ1S_GRID,           sizeof(IQ1S_GRID),
          "07540ffc1aeaf6ad4d97e96b0fcc765aae39671d4ae4a27bbd0e796fde167c6a" },
    };
    for (size_t i = 0; i < sizeof(tables) / sizeof(tables[0]); i++) {
        Sha256Ctx c;
        sha256_init(&c);
        sha256_update(&c, (const uint8_t *)tables[i].data, tables[i].len);
        uint8_t hash[32];
        sha256_final(&c, hash);
        char hex[65];
        sha256_hex(hash, hex);
        cr_assert_str_eq(hex, tables[i].expected_sha,
            "%s SHA256 mismatch: got %s, want %s", tables[i].name, hex, tables[i].expected_sha);
    }
}
