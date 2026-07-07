use super::*;

#[test]
fn iq_block_sizes_match_ggml_layout() {
    // Verified byte-exact against unsloth/MiniMax-M3-GGUF UD-IQ4_XS tensor
    // offset deltas: IQ4_XS = 136 B / 256 vals, IQ3_S = 110 B / 256 vals.
    assert_eq!(BLOCK_IQ4_XS_SIZE, 136);
    assert_eq!(BLOCK_IQ3_S_SIZE, 110);
    assert_eq!(BLOCK_IQ2_XS_SIZE, 74);
    assert_eq!(BLOCK_IQ2_S_SIZE, 82);
    assert_eq!(BLOCK_IQ4_NL_SIZE, 18);
    assert_eq!(IQ3S_GRID.len(), 512);
    assert_eq!(
        quantized_size(GgufQuantizationType::IQ4_XS, 256).unwrap(),
        136
    );
    assert_eq!(
        quantized_size(GgufQuantizationType::IQ3_S, 256).unwrap(),
        110
    );
    assert_eq!(
        quantized_size(GgufQuantizationType::IQ4_NL, 32).unwrap(),
        18
    );
    assert_eq!(
        quantized_size(GgufQuantizationType::IQ2_S, 256).unwrap(),
        82
    );
}

#[test]
fn iq4_xs_dequant_runs_and_is_finite() {
    // One block: d=1.0 (f16 0x3c00), scales all 0 (=> ls-32 = -32), qs walk.
    let mut block = vec![0u8; BLOCK_IQ4_XS_SIZE];
    block[0] = 0x00;
    block[1] = 0x3c; // f16 1.0
    for (i, b) in block[8..136].iter_mut().enumerate() {
        *b = (i % 256) as u8;
    }
    let mut out = vec![0f32; 256];
    dequantize_iq4_xs_scalar(&block, &mut out).unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
    // scale = -32, low nibble of qs[0]=0 -> codebook[0] = -127 => -32*-127
    assert_eq!(out[0], -32.0 * KVALUES_IQ4NL[0] as f32);
}

/// Deterministic pseudo-Gaussian sample (Box–Muller on a LCG) so tests don't
/// need an RNG dependency but still exercise a realistic weight distribution.
fn gaussian_sample(count: usize) -> Vec<f32> {
    let mut state: u64 = 0x1234_5678_9abc_def1;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32) / ((1u64 << 31) as f32)
    };
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let u1 = next().max(1.0e-7);
        let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        out.push(r * (std::f32::consts::TAU * u2).cos());
        if out.len() < count {
            out.push(r * (std::f32::consts::TAU * u2).sin());
        }
    }
    out
}

fn mse(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = (*x - *y) as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64
}

#[test]
fn iq4_xs_encode_decode_is_self_consistent() {
    let values = gaussian_sample(QK_K * 4);
    let mut encoded = vec![0u8; (values.len() / QK_K) * BLOCK_IQ4_XS_SIZE];
    quantize_iq4_xs_scalar(&values, &mut encoded).expect("encode");
    let mut decoded = vec![0f32; values.len()];
    dequantize_iq4_xs_scalar(&encoded, &mut decoded).expect("decode");
    assert!(decoded.iter().all(|v| v.is_finite()));
    // Re-encoding the decoded values must reproduce identical bytes: the
    // decoded values already sit on the codebook grid, so the encoder is a
    // fixed point there.
    let mut re_encoded = vec![0u8; encoded.len()];
    quantize_iq4_xs_scalar(&decoded, &mut re_encoded).expect("re-encode");
    assert_eq!(encoded, re_encoded);
}

#[test]
fn iq4_xs_beats_q4_0_on_gaussian_weights() {
    let values = gaussian_sample(QK_K * 8);

    let mut iq4 = vec![0u8; (values.len() / QK_K) * BLOCK_IQ4_XS_SIZE];
    quantize_iq4_xs_scalar(&values, &mut iq4).expect("iq4 encode");
    let mut iq4_dec = vec![0f32; values.len()];
    dequantize_iq4_xs_scalar(&iq4, &mut iq4_dec).expect("iq4 decode");

    let mut q40 = vec![0u8; (values.len() / QK4_0) * BLOCK_Q4_0_SIZE];
    quantize_q4_0_scalar(&values, &mut q40).expect("q4_0 encode");
    let mut q40_dec = vec![0f32; values.len()];
    dequantize_q4_0_scalar(&q40, &mut q40_dec).expect("q4_0 decode");

    let iq4_err = mse(&values, &iq4_dec);
    let q40_err = mse(&values, &q40_dec);
    assert!(
        iq4_err < q40_err,
        "IQ4_XS MSE {iq4_err} should beat Q4_0 MSE {q40_err}"
    );
}

#[test]
fn al5_beats_q4_0_mse_on_gaussian_weights() {
    let values = gaussian_sample(QK4_0 * 64);

    let mut q40 = vec![0u8; (values.len() / QK4_0) * BLOCK_Q4_0_SIZE];
    quantize_q4_0_scalar(&values, &mut q40).expect("q4_0 encode");
    let mut q40_dec = vec![0f32; values.len()];
    dequantize_q4_0_scalar(&q40, &mut q40_dec).expect("q4_0 decode");

    let mut al5 = vec![0u8; (values.len() / QK4_0) * BLOCK_Q4_0_SIZE];
    quantize_al5_scalar(&values, &mut al5).expect("al5 encode");
    let mut al5_dec = vec![0f32; values.len()];
    dequantize_al5_scalar(&al5, &mut al5_dec).expect("al5 decode");

    let q40_err = mse(&values, &q40_dec);
    let al5_err = mse(&values, &al5_dec);
    let pct = (1.0 - al5_err / q40_err) * 100.0;
    eprintln!("AL5 MSE {al5_err:.6} vs Q4_0 {q40_err:.6} ({pct:.1}% lower)");
    assert!(
        al5_err < q40_err,
        "AL5 MSE {al5_err} should beat Q4_0 MSE {q40_err}"
    );
}

#[test]
fn iq4_xs_imatrix_lowers_error_on_weighted_columns() {
    let values = gaussian_sample(QK_K * 4);
    // Importance heavily favors the first half of every 32-wide sub-block.
    let weights = (0..values.len())
        .map(|i| if i % 32 < 16 { 8.0 } else { 1.0 })
        .collect::<Vec<_>>();

    let mut plain = vec![0u8; (values.len() / QK_K) * BLOCK_IQ4_XS_SIZE];
    quantize_iq4_xs(&values, None, &mut plain).expect("plain encode");
    let mut plain_dec = vec![0f32; values.len()];
    dequantize_iq4_xs_scalar(&plain, &mut plain_dec).expect("plain decode");

    let mut weighted = vec![0u8; plain.len()];
    quantize_iq4_xs(&values, Some(&weights), &mut weighted).expect("weighted encode");
    let mut weighted_dec = vec![0f32; values.len()];
    dequantize_iq4_xs_scalar(&weighted, &mut weighted_dec).expect("weighted decode");

    // Weighted error on the high-importance columns should drop.
    let important_err = |dec: &[f32]| -> f64 {
        values
            .iter()
            .zip(dec)
            .enumerate()
            .filter(|(i, _)| i % 32 < 16)
            .map(|(_, (x, y))| {
                let d = (*x - *y) as f64;
                d * d
            })
            .sum()
    };
    assert!(
        important_err(&weighted_dec) <= important_err(&plain_dec),
        "imatrix should not increase error on important columns"
    );
    assert!(weighted_dec.iter().all(|v| v.is_finite()));
}

#[test]
fn iq1s_grid_decode_uses_real_table() {
    let mut out = [0_i8; 8];
    iq1s_grid_decode(0, &mut out);
    assert_eq!(out, [-1, -1, -1, -1, -1, -1, -1, -1]);
    iq1s_grid_decode(1, &mut out);
    assert_eq!(out, [1, -1, -1, -1, -1, -1, -1, -1]);
}

#[test]
fn iq4_nl_dequant_and_round_trip() {
    let mut block = vec![0u8; BLOCK_IQ4_NL_SIZE];
    block[0] = 0x00;
    block[1] = 0x3c; // f16 1.0
    block[2] = 0x10; // nibbles 0 and 1
    let mut out = vec![0f32; QK4_NL];
    dequantize_iq4_nl_scalar(&block, &mut out).unwrap();
    assert_eq!(out[0], KVALUES_IQ4NL[0] as f32);
    assert_eq!(out[16], KVALUES_IQ4NL[1] as f32);

    let values = gaussian_sample(QK4_NL * 8);
    let mut encoded = vec![0u8; (values.len() / QK4_NL) * BLOCK_IQ4_NL_SIZE];
    quantize_iq4_nl(&values, &mut encoded).expect("encode");
    let mut decoded = vec![0f32; values.len()];
    dequantize_iq4_nl_scalar(&encoded, &mut decoded).expect("decode");
    assert!(decoded.iter().all(|v| v.is_finite()));

    let mut q4 = vec![0u8; (values.len() / QK4_NL) * BLOCK_Q4_0_SIZE];
    quantize_q4_0_scalar(&values, &mut q4).expect("q4_0");
    let mut q4_dec = vec![0f32; values.len()];
    dequantize_q4_0_scalar(&q4, &mut q4_dec).expect("q4_0 dequant");
    assert!(
        mse(&values, &decoded) <= mse(&values, &q4_dec),
        "IQ4_NL should beat Q4_0 MSE on Gaussian weights"
    );
}

#[test]
fn iq2_xs_and_iq2_s_dequant_are_finite() {
    let mut xs = vec![0u8; BLOCK_IQ2_XS_SIZE];
    xs[0] = 0x00;
    xs[1] = 0x3c;
    for (i, b) in xs[2..].iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    let mut out_xs = vec![0f32; QK_K];
    dequantize_iq2_xs_scalar(&xs, &mut out_xs).unwrap();
    assert!(out_xs.iter().all(|v| v.is_finite()));

    let mut s = vec![0u8; BLOCK_IQ2_S_SIZE];
    s[0] = 0x00;
    s[1] = 0x3c;
    for (i, b) in s[2..].iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    let mut out_s = vec![0f32; QK_K];
    dequantize_iq2_s_scalar(&s, &mut out_s).unwrap();
    assert!(out_s.iter().all(|v| v.is_finite()));
}

#[test]
fn iq3_s_dequant_runs_and_is_finite() {
    let mut block = vec![0u8; BLOCK_IQ3_S_SIZE];
    block[0] = 0x00;
    block[1] = 0x3c; // f16 1.0
    for (i, b) in block[2..66].iter_mut().enumerate() {
        *b = (i % 256) as u8;
    }
    let mut out = vec![0f32; 256];
    dequantize_iq3_s_scalar(&block, &mut out).unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn bf16_dequant_widens_to_exact_f32() {
    // BF16 is the top 16 bits of an f32; widening must be exact (no rounding).
    let values = [0.0_f32, 1.0, -2.0, 0.5, 123.5, -0.015625];
    let mut input = Vec::new();
    for &v in &values {
        let bf16 = (v.to_bits() >> 16) as u16;
        input.extend_from_slice(&bf16.to_le_bytes());
    }
    let mut output = vec![0.0_f32; values.len()];
    dequantize_bf16_scalar(&input, &mut output).expect("bf16 dequant should succeed");
    for (got, want) in output.iter().zip(values.iter()) {
        // All chosen values are exactly representable in BF16.
        assert_eq!(got, want, "bf16 dequant mismatch");
    }
}

#[test]
fn q6_k_dequant_decodes_both_128_groups_independently() {
    // Regression: the second 128-element group of a Q6_K block must advance
    // into ql/qh/scales (ql+=64, qh+=32, scales+=8). With all quant nibbles
    // zero, every value decodes to (0 - 32) = -32 scaled by its group's
    // scale. Distinct scales per group expose a missing-offset bug where the
    // tail of every block is decoded from the head's scales.
    let mut block = vec![0u8; BLOCK_Q6_K_SIZE];
    // scales: bytes 192..208 (16× int8). Group 0 -> 1, group 1 -> 2.
    for s in block.iter_mut().take(208).skip(192).take(8) {
        *s = 1;
    }
    for s in block.iter_mut().take(208).skip(200) {
        *s = 2;
    }
    // super-block scale d (f16) at 208..210 = 1.0.
    block[208..210].copy_from_slice(&half_to_le_bytes_one());

    let mut out = vec![0.0_f32; QK_K];
    dequantize_q6_k_scalar(&block, &mut out).expect("q6_k dequant succeeds");

    // First 128 use group-0 scale (1): -32. Last 128 use group-1 scale (2): -64.
    assert!((out[0] - (-32.0)).abs() < 1e-3, "head: {}", out[0]);
    assert!((out[127] - (-32.0)).abs() < 1e-3, "head end: {}", out[127]);
    assert!((out[128] - (-64.0)).abs() < 1e-3, "tail: {}", out[128]);
    assert!((out[255] - (-64.0)).abs() < 1e-3, "tail end: {}", out[255]);
}

/// Little-endian IEEE half-precision bytes for 1.0 (0x3C00).
fn half_to_le_bytes_one() -> [u8; 2] {
    [0x00, 0x3C]
}

#[test]
fn dequantizes_f32_scalar_values() {
    let mut input = Vec::new();
    input.extend_from_slice(&1.25_f32.to_le_bytes());
    input.extend_from_slice(&(-2.5_f32).to_le_bytes());

    let mut out = vec![0.0_f32; 2];
    dequantize_f32_scalar(&input, &mut out).expect("f32 dequant succeeds");

    assert!((out[0] - 1.25).abs() < 1e-6);
    assert!((out[1] + 2.5).abs() < 1e-6);
}

#[test]
fn dequantizes_f16_scalar_values() {
    let input = vec![
        0x00, 0x3C, // 1.0
        0x00, 0xC1, // -2.5
    ];

    let mut out = vec![0.0_f32; 2];
    dequantize_f16_scalar(&input, &mut out).expect("f16 dequant succeeds");

    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!((out[1] + 2.5).abs() < 1e-6);
}

#[test]
fn dequantizes_q4_0_scalar_block() {
    let mut input = vec![0x00, 0x3C];
    input.extend(std::iter::repeat_n(0x98, 16));

    let mut out = vec![0.0_f32; 32];
    dequantize_q4_0_scalar(&input, &mut out).expect("q4_0 dequant succeeds");

    assert!(out[..16].iter().all(|v| (*v - 0.0).abs() < 1e-6));
    assert!(out[16..].iter().all(|v| (*v - 1.0).abs() < 1e-6));
}

#[test]
fn dequantizes_q5_0_scalar_block() {
    let mut input = vec![0x00, 0x3C];
    input.extend([0x01, 0x00, 0x00, 0x00]);
    input.extend(std::iter::repeat_n(0x00, 16));

    let mut out = vec![0.0_f32; 32];
    dequantize_q5_0_scalar(&input, &mut out).expect("q5_0 dequant succeeds");

    assert!((out[0] - 0.0).abs() < 1e-6);
    assert!((out[1] + 16.0).abs() < 1e-6);
}

#[test]
fn dequantizes_q8_0_scalar_block() {
    let mut input = vec![0x00, 0x3C];
    input.extend(0_u8..32_u8);

    let mut out = vec![0.0_f32; 32];
    dequantize_q8_0_scalar(&input, &mut out).expect("q8_0 dequant succeeds");

    assert!((out[0] - 0.0).abs() < 1e-6);
    assert!((out[31] - 31.0).abs() < 1e-6);
}

#[test]
fn dequantizes_nvfp4_scalar_block() {
    let mut input = vec![0x38, 0x40, 0x30, 0x00];
    input.extend(std::iter::repeat_n(0x21, 8));
    input.extend(std::iter::repeat_n(0xba, 8));
    input.extend(std::iter::repeat_n(0xf7, 8));
    input.extend(std::iter::repeat_n(0x00, 8));

    let mut out = vec![0.0_f32; 64];
    dequantize_nvfp4_scalar(&input, &mut out).expect("nvfp4 dequant succeeds");

    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!((out[8] - 2.0).abs() < 1e-6);
    assert!((out[16] + 4.0).abs() < 1e-6);
    assert!((out[24] + 6.0).abs() < 1e-6);
    assert!((out[32] - 6.0).abs() < 1e-6);
    assert!((out[40] + 6.0).abs() < 1e-6);
    assert!(out[48..64].iter().all(|v| *v == 0.0));
}

#[test]
fn dequantizes_k_quant_scalar_block() {
    let mut input = vec![0x00, 0x3C];
    input.extend(std::iter::repeat_n(0_u8, 82));

    let mut out = vec![0.0_f32; 256];
    dequantize_q2_k_scalar(&input, &mut out).expect("q2_k dequant succeeds");

    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn dispatches_by_quantization_type() {
    let mut input = vec![0x00, 0x3C];
    input.extend(0_u8..32_u8);
    let mut out = vec![0.0_f32; 32];

    dequantize_scalar(GgufQuantizationType::Q8_0, &input, &mut out).expect("dispatch succeeds");
    assert!((out[4] - 4.0).abs() < 1e-6);

    let nvfp4 = vec![0x38; BLOCK_NVFP4_SIZE];
    let mut nvfp4_out = vec![0.0_f32; QK_NVFP4];
    dequantize_scalar(GgufQuantizationType::NVFP4, &nvfp4, &mut nvfp4_out)
        .expect("nvfp4 dispatch succeeds");
}

#[test]
fn dispatches_f16_and_f32_types() {
    let f16_input = vec![0x00, 0x3C, 0x00, 0x40];
    let mut f16_out = vec![0.0_f32; 2];
    dequantize_scalar(GgufQuantizationType::F16, &f16_input, &mut f16_out)
        .expect("f16 dispatch succeeds");
    assert!((f16_out[0] - 1.0).abs() < 1e-6);
    assert!((f16_out[1] - 2.0).abs() < 1e-6);

    let mut f32_input = Vec::new();
    f32_input.extend_from_slice(&3.0_f32.to_le_bytes());
    f32_input.extend_from_slice(&(-4.0_f32).to_le_bytes());
    let mut f32_out = vec![0.0_f32; 2];
    dequantize_scalar(GgufQuantizationType::F32, &f32_input, &mut f32_out)
        .expect("f32 dispatch succeeds");
    assert!((f32_out[0] - 3.0).abs() < 1e-6);
    assert!((f32_out[1] + 4.0).abs() < 1e-6);
}

#[test]
fn validates_output_length() {
    let mut input = vec![0x00, 0x3C];
    input.extend(0_u8..32_u8);
    let mut out = vec![0.0_f32; 31];

    let err = dequantize_q8_0_scalar(&input, &mut out).expect_err("must reject output size");
    assert!(matches!(err, QuantizationError::InvalidOutputLength { .. }));
}

#[test]
fn quantizes_from_f32_to_all_supported_formats() {
    let targets = [
        GgufQuantizationType::F32,
        GgufQuantizationType::F16,
        GgufQuantizationType::Q4_0,
        GgufQuantizationType::Q4_1,
        GgufQuantizationType::Q5_0,
        GgufQuantizationType::Q5_1,
        GgufQuantizationType::Q8_0,
        GgufQuantizationType::Q2_K,
        GgufQuantizationType::Q3_K_S,
        GgufQuantizationType::Q3_K_M,
        GgufQuantizationType::Q3_K_L,
        GgufQuantizationType::Q4_K_S,
        GgufQuantizationType::Q4_K_M,
        GgufQuantizationType::Q5_K_S,
        GgufQuantizationType::Q5_K_M,
        GgufQuantizationType::Q6_K,
    ];

    for target in targets {
        let values = test_values_for_target(target, -8.0, 0.25);
        let mut src = Vec::with_capacity(values.len() * 4);
        for value in &values {
            src.extend_from_slice(&value.to_le_bytes());
        }

        let out_size = quantized_size(target, values.len()).expect("size must be known");
        let mut quantized = vec![0_u8; out_size];
        quantize_scalar(GgufQuantizationType::F32, target, &src, &mut quantized)
            .expect("f32 source quantization must succeed");

        let mut recovered = vec![0.0_f32; values.len()];
        dequantize_scalar(target, &quantized, &mut recovered)
            .expect("dequantization of quantized payload must succeed");
        assert!(recovered.iter().all(|v| v.is_finite()));
        if target == GgufQuantizationType::F32 {
            assert_eq!(src, quantized);
        }
    }
}

#[test]
fn quantizes_from_f16_to_all_supported_formats() {
    let targets = [
        GgufQuantizationType::F32,
        GgufQuantizationType::F16,
        GgufQuantizationType::Q4_0,
        GgufQuantizationType::Q4_1,
        GgufQuantizationType::Q5_0,
        GgufQuantizationType::Q5_1,
        GgufQuantizationType::Q8_0,
        GgufQuantizationType::Q2_K,
        GgufQuantizationType::Q3_K_S,
        GgufQuantizationType::Q3_K_M,
        GgufQuantizationType::Q3_K_L,
        GgufQuantizationType::Q4_K_S,
        GgufQuantizationType::Q4_K_M,
        GgufQuantizationType::Q5_K_S,
        GgufQuantizationType::Q5_K_M,
        GgufQuantizationType::Q6_K,
    ];

    for target in targets {
        let values = test_values_for_target(target, -12.0, 0.2);
        let mut src = Vec::with_capacity(values.len() * 2);
        for value in &values {
            src.extend_from_slice(&f32_to_f16_bits(*value).to_le_bytes());
        }

        let out_size = quantized_size(target, values.len()).expect("size must be known");
        let mut quantized = vec![0_u8; out_size];
        quantize_scalar(GgufQuantizationType::F16, target, &src, &mut quantized)
            .expect("f16 source quantization must succeed");

        let mut recovered = vec![0.0_f32; values.len()];
        dequantize_scalar(target, &quantized, &mut recovered)
            .expect("dequantization of quantized payload must succeed");
        assert!(recovered.iter().all(|v| v.is_finite()));
        if target == GgufQuantizationType::F16 {
            assert_eq!(src, quantized);
        }
    }
}

#[test]
fn q8_0_quantization_uses_independent_scales_per_block() {
    let mut values = vec![0.0_f32; QK8_0 * 2];
    for (i, slot) in values[..QK8_0].iter_mut().enumerate() {
        *slot = i as f32 * 0.5;
    }
    for (i, slot) in values[QK8_0..].iter_mut().enumerate() {
        *slot = i as f32 * 6.0;
    }

    let mut src = Vec::with_capacity(values.len() * 4);
    for value in &values {
        src.extend_from_slice(&value.to_le_bytes());
    }

    let mut quantized = vec![0_u8; BLOCK_Q8_0_SIZE * 2];
    quantize_scalar(
        GgufQuantizationType::F32,
        GgufQuantizationType::Q8_0,
        &src,
        &mut quantized,
    )
    .expect("q8_0 quantization succeeds");

    let first_scale = f16_le_to_f32(&quantized[0..2]);
    let second_scale = f16_le_to_f32(&quantized[BLOCK_Q8_0_SIZE..BLOCK_Q8_0_SIZE + 2]);
    assert!(second_scale > first_scale * 8.0);
}

#[test]
fn q4_1_quantization_uses_independent_scales_per_block() {
    let mut values = vec![0.0_f32; QK4_1 * 2];
    for (i, slot) in values[..QK4_1].iter_mut().enumerate() {
        *slot = -2.0 + i as f32 * 0.1;
    }
    for (i, slot) in values[QK4_1..].iter_mut().enumerate() {
        *slot = -40.0 + i as f32 * 3.0;
    }

    let mut src = Vec::with_capacity(values.len() * 4);
    for value in &values {
        src.extend_from_slice(&value.to_le_bytes());
    }

    let mut quantized = vec![0_u8; BLOCK_Q4_1_SIZE * 2];
    quantize_scalar(
        GgufQuantizationType::F32,
        GgufQuantizationType::Q4_1,
        &src,
        &mut quantized,
    )
    .expect("q4_1 quantization succeeds");

    let first_scale = f16_le_to_f32(&quantized[0..2]);
    let second_scale = f16_le_to_f32(&quantized[BLOCK_Q4_1_SIZE..BLOCK_Q4_1_SIZE + 2]);
    assert!(second_scale > first_scale * 20.0);
}

#[test]
fn creates_valid_imatrix() {
    let matrix = IMatrix::from_values(vec![0.25, 1.0, 2.0]).expect("valid matrix");
    assert_eq!(matrix.values(), &[0.25, 1.0, 2.0]);
}

#[test]
fn rejects_invalid_imatrix_values() {
    let empty = IMatrix::from_values(Vec::new()).expect_err("empty should fail");
    assert!(matches!(
        empty,
        QuantizationError::InvalidImportanceMatrix { .. }
    ));
    let negative = IMatrix::from_values(vec![1.0, -0.1]).expect_err("negative should fail");
    assert!(matches!(
        negative,
        QuantizationError::InvalidImportanceMatrix { .. }
    ));
}

#[test]
fn imatrix_quantization_requires_matching_value_count() {
    let input = vec![0x00, 0x3C, 0x00, 0x40];
    let matrix = IMatrix::from_values(vec![1.0]).expect("matrix should be valid");
    let mut output = vec![0_u8; 4];

    let err = quantize_scalar_with_imatrix(
        GgufQuantizationType::F16,
        GgufQuantizationType::F16,
        &input,
        &mut output,
        &matrix,
    )
    .expect_err("mismatched matrix length should fail");
    assert!(matches!(
        err,
        QuantizationError::InvalidImportanceMatrix { .. }
    ));
}

#[test]
fn imatrix_quantization_biases_encoded_output() {
    let values = [1.0_f32, 2.0_f32];
    let mut input = Vec::new();
    for value in values {
        input.extend_from_slice(&value.to_le_bytes());
    }
    let matrix = IMatrix::from_values(vec![2.0, 0.5]).expect("matrix should be valid");
    let mut with_matrix = vec![0_u8; 8];
    let mut baseline = vec![0_u8; 8];

    quantize_scalar(
        GgufQuantizationType::F32,
        GgufQuantizationType::F32,
        &input,
        &mut baseline,
    )
    .expect("baseline quantization should work");
    quantize_scalar_with_imatrix(
        GgufQuantizationType::F32,
        GgufQuantizationType::F32,
        &input,
        &mut with_matrix,
        &matrix,
    )
    .expect("imatrix quantization should work");

    assert_ne!(with_matrix, baseline);
}

#[test]
fn quantizes_mixed_layers_with_distinct_targets() {
    let first_values = (0..QK8_0).map(|i| i as f32 * 0.25);
    let second_values = [1.0_f32, -2.0_f32];
    let values = first_values.chain(second_values).collect::<Vec<_>>();
    let mut input = Vec::with_capacity(values.len() * 4);
    for value in &values {
        input.extend_from_slice(&value.to_le_bytes());
    }

    let plans = vec![
        MixedLayerPlan {
            name: "blk.0.attn_q.weight".to_owned(),
            value_count: QK8_0,
            target: GgufQuantizationType::Q8_0,
        },
        MixedLayerPlan {
            name: "blk.0.ffn_down.weight".to_owned(),
            value_count: 2,
            target: GgufQuantizationType::F16,
        },
    ];

    let output = quantize_mixed_scalar(GgufQuantizationType::F32, &input, &plans)
        .expect("mixed quantization should succeed");

    assert_eq!(output.len(), 2);
    assert_eq!(output[0].name, "blk.0.attn_q.weight");
    assert_eq!(output[0].target, GgufQuantizationType::Q8_0);
    assert_eq!(output[0].bytes.len(), BLOCK_Q8_0_SIZE);
    assert_eq!(output[1].name, "blk.0.ffn_down.weight");
    assert_eq!(output[1].target, GgufQuantizationType::F16);
    assert_eq!(output[1].bytes.len(), 4);

    let mut recovered_q8 = vec![0.0_f32; QK8_0];
    dequantize_scalar(
        GgufQuantizationType::Q8_0,
        &output[0].bytes,
        &mut recovered_q8,
    )
    .expect("q8 output should dequantize");
    assert!(recovered_q8.iter().all(|v| v.is_finite()));

    let mut recovered_f16 = vec![0.0_f32; 2];
    dequantize_scalar(
        GgufQuantizationType::F16,
        &output[1].bytes,
        &mut recovered_f16,
    )
    .expect("f16 output should dequantize");
    assert!(recovered_f16.iter().all(|v| v.is_finite()));
}

#[test]
fn mixed_quantization_rejects_input_length_mismatch() {
    let input = vec![0_u8; 8];
    let plans = vec![MixedLayerPlan {
        name: "blk.0.attn_q.weight".to_owned(),
        value_count: 4,
        target: GgufQuantizationType::F16,
    }];

    let err = quantize_mixed_scalar(GgufQuantizationType::F32, &input, &plans)
        .expect_err("mismatched source bytes should fail");
    assert!(matches!(
        err,
        QuantizationError::InvalidMixedInputLength { .. }
    ));
}

#[test]
fn mixed_quantization_rejects_empty_layer_name() {
    let input = vec![0_u8; 8];
    let plans = vec![MixedLayerPlan {
        name: String::new(),
        value_count: 2,
        target: GgufQuantizationType::F16,
    }];

    let err = quantize_mixed_scalar(GgufQuantizationType::F32, &input, &plans)
        .expect_err("empty layer name should fail");
    assert!(matches!(
        err,
        QuantizationError::InvalidMixedQuantizationPlan { .. }
    ));
}

#[test]
fn mixed_quantization_parallel_matches_sequential_output() {
    let plans = vec![
        MixedLayerPlan {
            name: "blk.0.attn_q.weight".to_owned(),
            value_count: QK8_0,
            target: GgufQuantizationType::Q8_0,
        },
        MixedLayerPlan {
            name: "blk.0.attn_k.weight".to_owned(),
            value_count: QK8_0,
            target: GgufQuantizationType::Q4_0,
        },
        MixedLayerPlan {
            name: "blk.0.ffn_down.weight".to_owned(),
            value_count: 2,
            target: GgufQuantizationType::F16,
        },
    ];
    let values = (0..(QK8_0 * 2 + 2))
        .map(|i| (i as f32 * 0.25) - 8.0)
        .collect::<Vec<_>>();
    let mut input = Vec::with_capacity(values.len() * 4);
    for value in values {
        input.extend_from_slice(&value.to_le_bytes());
    }

    let sequential = quantize_mixed_scalar_sequential(GgufQuantizationType::F32, 4, &input, &plans)
        .expect("sequential mixed quantization should succeed");
    let parallel = quantize_mixed_scalar(GgufQuantizationType::F32, &input, &plans)
        .expect("parallel mixed quantization should succeed");

    assert_eq!(parallel, sequential);
}

#[test]
fn q4_0_encoding_matches_llama_cpp_reference_block_layout() {
    let mut values = Vec::with_capacity(QK4_0);
    for i in 0..QK4_0 {
        values.push((i % 16) as f32 - 8.0);
    }
    let mut input = Vec::with_capacity(values.len() * 4);
    for value in values {
        input.extend_from_slice(&value.to_le_bytes());
    }

    let mut output = vec![0_u8; BLOCK_Q4_0_SIZE];
    quantize_scalar(
        GgufQuantizationType::F32,
        GgufQuantizationType::Q4_0,
        &input,
        &mut output,
    )
    .expect("q4_0 quantization should succeed");

    let expected = vec![
        0x00, 0x3C, 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE, 0x10, 0x32, 0x54, 0x76, 0x98,
        0xBA, 0xDC, 0xFE,
    ];
    assert_eq!(output, expected);
}

#[test]
fn q5_0_encoding_matches_llama_cpp_reference_block_layout() {
    let mut values = Vec::with_capacity(QK5_0);
    for i in 0..QK5_0 {
        values.push(i as f32 - 16.0);
    }
    let mut input = Vec::with_capacity(values.len() * 4);
    for value in values {
        input.extend_from_slice(&value.to_le_bytes());
    }

    let mut output = vec![0_u8; BLOCK_Q5_0_SIZE];
    quantize_scalar(
        GgufQuantizationType::F32,
        GgufQuantizationType::Q5_0,
        &input,
        &mut output,
    )
    .expect("q5_0 quantization should succeed");

    let expected = vec![
        0x00, 0x3C, // d = 1.0
        0x00, 0x00, 0xFF, 0xFF, // qh
        0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE, // qs[0..8]
        0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE, // qs[8..16]
    ];
    assert_eq!(output, expected);
}

#[test]
fn q8_0_encoding_matches_llama_cpp_reference_block_layout() {
    let mut values = Vec::with_capacity(QK8_0);
    for i in 0..QK8_0 {
        values.push(i as f32 - 16.0);
    }
    let mut input = Vec::with_capacity(values.len() * 4);
    for value in values {
        input.extend_from_slice(&value.to_le_bytes());
    }

    let mut output = vec![0_u8; BLOCK_Q8_0_SIZE];
    quantize_scalar(
        GgufQuantizationType::F32,
        GgufQuantizationType::Q8_0,
        &input,
        &mut output,
    )
    .expect("q8_0 quantization should succeed");

    let expected = vec![
        0x08, 0x30, // d = 16/127 in f16
        129, 137, 145, 153, 161, 169, 177, 185, 192, 200, 208, 216, 224, 232, 240, 248, 0, 8, 16,
        24, 32, 40, 48, 56, 64, 71, 79, 87, 95, 103, 111, 119,
    ];
    assert_eq!(output, expected);
}

#[test]
fn quantized_size_matches_every_supported_scheme() {
    let cases = [
        (GgufQuantizationType::F32, 7, 28),
        (GgufQuantizationType::F16, 7, 14),
        (GgufQuantizationType::Q4_0, QK4_0, BLOCK_Q4_0_SIZE),
        (GgufQuantizationType::Q4_1, QK4_1, BLOCK_Q4_1_SIZE),
        (GgufQuantizationType::Q5_0, QK5_0, BLOCK_Q5_0_SIZE),
        (GgufQuantizationType::Q5_1, QK5_1, BLOCK_Q5_1_SIZE),
        (GgufQuantizationType::Q8_0, QK8_0, BLOCK_Q8_0_SIZE),
        (GgufQuantizationType::Q2_K, QK_K, BLOCK_Q2_K_SIZE),
        (GgufQuantizationType::Q3_K_S, QK_K, BLOCK_Q3_K_SIZE),
        (GgufQuantizationType::Q3_K_M, QK_K, BLOCK_Q3_K_SIZE),
        (GgufQuantizationType::Q3_K_L, QK_K, BLOCK_Q3_K_SIZE),
        (GgufQuantizationType::Q4_K_S, QK_K, BLOCK_Q4_K_SIZE),
        (GgufQuantizationType::Q4_K_M, QK_K, BLOCK_Q4_K_SIZE),
        (GgufQuantizationType::Q5_K_S, QK_K, BLOCK_Q5_K_SIZE),
        (GgufQuantizationType::Q5_K_M, QK_K, BLOCK_Q5_K_SIZE),
        (GgufQuantizationType::Q6_K, QK_K, BLOCK_Q6_K_SIZE),
    ];

    for (quantization, value_count, expected) in cases {
        let actual = quantized_size(quantization, value_count).expect("size should be known");
        assert_eq!(actual, expected, "unexpected size for {quantization:?}");
    }
}

#[test]
fn quantized_size_rejects_invalid_block_lengths() {
    let blocked = [
        (GgufQuantizationType::Q4_0, QK4_0),
        (GgufQuantizationType::Q4_1, QK4_1),
        (GgufQuantizationType::Q5_0, QK5_0),
        (GgufQuantizationType::Q5_1, QK5_1),
        (GgufQuantizationType::Q8_0, QK8_0),
        (GgufQuantizationType::Q2_K, QK_K),
        (GgufQuantizationType::Q3_K_S, QK_K),
        (GgufQuantizationType::Q3_K_M, QK_K),
        (GgufQuantizationType::Q3_K_L, QK_K),
        (GgufQuantizationType::Q4_K_S, QK_K),
        (GgufQuantizationType::Q4_K_M, QK_K),
        (GgufQuantizationType::Q5_K_S, QK_K),
        (GgufQuantizationType::Q5_K_M, QK_K),
        (GgufQuantizationType::Q6_K, QK_K),
    ];

    for (quantization, block_size) in blocked {
        let err = quantized_size(quantization, block_size - 1)
            .expect_err("invalid lengths should be rejected");
        assert!(matches!(err, QuantizationError::InvalidInputLength { .. }));
    }
}

fn test_values_for_target(target: GgufQuantizationType, offset: f32, scale: f32) -> Vec<f32> {
    let value_count = if matches!(
        target,
        GgufQuantizationType::Q2_K
            | GgufQuantizationType::Q3_K_S
            | GgufQuantizationType::Q3_K_M
            | GgufQuantizationType::Q3_K_L
            | GgufQuantizationType::Q4_K_S
            | GgufQuantizationType::Q4_K_M
            | GgufQuantizationType::Q5_K_S
            | GgufQuantizationType::Q5_K_M
            | GgufQuantizationType::Q6_K
    ) {
        QK_K
    } else if matches!(
        target,
        GgufQuantizationType::Q4_0
            | GgufQuantizationType::Q4_1
            | GgufQuantizationType::Q5_0
            | GgufQuantizationType::Q5_1
            | GgufQuantizationType::Q8_0
    ) {
        QK8_0
    } else {
        2
    };
    (0..value_count)
        .map(|i| (i as f32 + offset) * scale)
        .collect()
}
