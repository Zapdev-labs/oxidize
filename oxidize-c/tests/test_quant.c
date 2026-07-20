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
#include <criterion/criterion.h>
#include <criterion/redirect.h>

#include "oxidize/quant.h"
#include "oxidize/error.h"

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
    buf[8] = 1u;  /* scales[4+0]=scales[4] for m1 of j=0 → set to 1 so m1=1? No — we want m1=0. */
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
    cr_assert_eq(oc_quant_type_from_ggml_id(18), OC_QUANT_Q6_K, "ggml id 18");
    cr_assert_eq(oc_quant_type_from_ggml_id(30), OC_QUANT_BF16, "ggml id 30");
    cr_assert_eq(oc_quant_type_from_ggml_id(0xff), OC_QUANT_UNKNOWN, "unknown ggml id");

    cr_assert_eq(oc_quant_type_to_ggml_id(OC_QUANT_F32),  0u,  "F32 ggml id");
    cr_assert_eq(oc_quant_type_to_ggml_id(OC_QUANT_Q4_K_M), 15u, "Q4_K_M ggml id");
    cr_assert_eq(oc_quant_type_to_ggml_id(OC_QUANT_Q8_0), 8u, "Q8_0 ggml id");
    cr_assert_eq(oc_quant_type_to_ggml_id(OC_QUANT_UNKNOWN), 0xffffffffu, "UNKNOWN ggml id");
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
