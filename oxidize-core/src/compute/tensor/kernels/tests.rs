use super::*;

/// Shape/thread/working-set microbenchmark for the Q4_K decode GEMV.
/// Run with:
///   cargo test --release -p oxidize-core --lib -- --ignored --nocapture bench_q4k
#[test]
#[ignore]
fn bench_q4k_gemv_shapes() {
    let shapes: [(usize, usize); 4] = [(9728, 2560), (2560, 9728), (4096, 2560), (1024, 2560)];
    for threads in [1usize, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for &(rows, cols) in &shapes {
            let bpr = cols / QK_K;
            let bytes = rows * bpr * BLOCK_Q4_K_SIZE;
            // 8 copies so the DRAM pass cannot sit in the 16MB L3.
            let copies = 8;
            let weights: Vec<u8> = (0..bytes * copies).map(|i| (i * 37 + 11) as u8).collect();
            let vector: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.001).sin()).collect();
            let mut output = vec![0.0_f32; rows];
            for (label, stride) in [("L3", 0usize), ("DRAM", bytes)] {
                pool.install(|| {
                        for i in 0..copies {
                            let w = &weights[i * stride..i * stride + bytes];
                            gemv_q4_k_q8_k_fused(w, rows, cols, &vector, &mut output).unwrap();
                        }
                        let iters = 24;
                        let t0 = std::time::Instant::now();
                        for i in 0..iters {
                            let w = &weights[(i % copies) * stride..(i % copies) * stride + bytes];
                            gemv_q4_k_q8_k_fused(w, rows, cols, &vector, &mut output).unwrap();
                        }
                        let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
                        eprintln!(
                            "q4k {rows:>5}x{cols:<5} threads={threads} {label:>4}: {:>7.1}us {:>6.1} GB/s",
                            ns / 1e3,
                            bytes as f64 / ns
                        );
                    });
            }
        }
    }
}

/// The fused multi-matrix region must produce bit-identical rows to the
/// sequential per-matrix GEMVs (same row kernels, same per-row order),
/// including mixed Q4_K/Q6_K jobs and non-multiple-of-chunk tails.
#[test]
fn multi_gemv_matches_sequential_bitwise() {
    let cols = 2560;
    let bpr = cols / QK_K;
    let q4_rows = 96_usize;
    let q6_rows = 61_usize;
    let q4: Vec<u8> = (0..q4_rows * bpr * BLOCK_Q4_K_SIZE)
        .map(|i| (i * 31 + 7) as u8)
        .collect();
    let q6: Vec<u8> = (0..q6_rows * bpr * BLOCK_Q6_K_SIZE)
        .map(|i| (i * 17 + 3) as u8)
        .collect();
    let vector: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.01).sin()).collect();

    let mut seq_q4 = vec![0.0_f32; q4_rows];
    let mut seq_q6 = vec![0.0_f32; q6_rows];
    gemv_quantized_f32(
        GgufQuantizationType::Q4_K_M,
        &q4,
        q4_rows,
        cols,
        &vector,
        &mut seq_q4,
    )
    .unwrap();
    gemv_quantized_f32(
        GgufQuantizationType::Q6_K,
        &q6,
        q6_rows,
        cols,
        &vector,
        &mut seq_q6,
    )
    .unwrap();

    let mut multi_q4 = vec![0.0_f32; q4_rows];
    let mut multi_q6 = vec![0.0_f32; q6_rows];
    let mut jobs = [
        GemvJob {
            quantization: GgufQuantizationType::Q4_K_M,
            matrix: &q4,
            rows: q4_rows,
            output: &mut multi_q4,
        },
        GemvJob {
            quantization: GgufQuantizationType::Q6_K,
            matrix: &q6,
            rows: q6_rows,
            output: &mut multi_q6,
        },
    ];
    gemv_quantized_multi_f32(&mut jobs, cols, &vector).unwrap();

    for (i, (a, b)) in seq_q4.iter().zip(&multi_q4).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "q4 row {i}");
    }
    for (i, (a, b)) in seq_q6.iter().zip(&multi_q6).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "q6 row {i}");
    }
}

/// Tolerance for tests that compare CUDA (f16-intermediate) results against
/// CPU references.  The GPU dequantizes to f16 before GEMV, so a small
/// round-trip error (~0.01-0.5) is expected and acceptable.
#[cfg(feature = "cuda")]
const CUDA_TOL: f32 = 0.5;
#[cfg(not(feature = "cuda"))]
const CUDA_TOL: f32 = 1e-4;

/// Gate A (OXK plan): the oxidize-kernels Q4_K row dots must match the
/// legacy tensor.rs kernels bit-for-bit (same integer op sequence and f32
/// combine order), and its Q8_K activation quantizer must be byte-equal.
#[test]
#[cfg(all(feature = "oxk", any(target_arch = "x86", target_arch = "x86_64")))]
fn oxk_q4_k_kernels_match_legacy_exactly() {
    use crate::quantization::{quantize_scalar, quantized_size};
    if !q4_k_q8_k_avx2_available() {
        return;
    }
    let (rows, cols) = (24usize, 512usize);
    let blocks_per_row = cols / QK_K;
    let total = rows * cols;
    let mut bytes = vec![0u8; total * 4];
    for i in 0..total {
        let v = (((i * 31 + 7) % 211) as f32) / 53.0 - 2.0;
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let q_size = quantized_size(GgufQuantizationType::Q4_K_M, total).unwrap();
    let mut q = vec![0u8; q_size];
    quantize_scalar(
        GgufQuantizationType::F32,
        GgufQuantizationType::Q4_K_M,
        &bytes,
        &mut q,
    )
    .unwrap();
    let input: Vec<f32> = (0..cols)
        .map(|i| (((i * 17 + 3) % 113) as f32) / 29.0 - 1.5)
        .collect();

    // Q8_K quantizer parity (byte-exact).
    let mut q8k_legacy = vec![0u8; blocks_per_row * BLOCK_Q8_K_BYTES];
    quantize_vector_q8_k_into(&input, blocks_per_row, &mut q8k_legacy);
    let mut q8k_oxk = vec![0u8; blocks_per_row * BLOCK_Q8_K_BYTES];
    oxidize_kernels::quantize_q8_k_into(&input, blocks_per_row, &mut q8k_oxk);
    assert_eq!(q8k_legacy, q8k_oxk, "Q8_K quantizer bytes differ");

    let row_bytes = blocks_per_row * BLOCK_Q4_K_SIZE;
    // Legacy single-row reference (AVX2 kernel, not VNNI, to pin the exact
    // instruction family OXK replicates; the two are bit-equal anyway).
    let legacy: Vec<f32> = (0..rows)
        .map(|r| unsafe {
            q4_k_q8_k_row_dot_avx2(
                &q[r * row_bytes..(r + 1) * row_bytes],
                blocks_per_row,
                &q8k_legacy,
            )
        })
        .collect();

    // OXK scalar reference vs legacy AVX2: exact.
    for (r, &want) in legacy.iter().enumerate() {
        let got = oxidize_kernels::q4k_q8k_row_dot_scalar(
            &q[r * row_bytes..(r + 1) * row_bytes],
            blocks_per_row,
            &q8k_oxk,
        );
        assert_eq!(got.to_bits(), want.to_bits(), "oxk scalar row {r}");
    }

    // OXK x1 / x4 / x8 vs legacy: exact.
    for (r, &want) in legacy.iter().enumerate() {
        let got = unsafe {
            oxidize_kernels::q4k_q8k_row_dot_avx2(
                &q[r * row_bytes..(r + 1) * row_bytes],
                blocks_per_row,
                &q8k_oxk,
            )
        };
        assert_eq!(got.to_bits(), want.to_bits(), "oxk x1 row {r}");
    }
    let mut quad = [0.0f32; 4];
    unsafe {
        oxidize_kernels::q4k_q8k_row_dot_x4_avx2(
            q.as_ptr(),
            row_bytes,
            blocks_per_row,
            &q8k_oxk,
            &mut quad,
        )
    };
    for (r, &got) in quad.iter().enumerate() {
        assert_eq!(got.to_bits(), legacy[r].to_bits(), "oxk x4 row {r}");
    }
    let mut octet = [0.0f32; 8];
    unsafe {
        oxidize_kernels::q4k_q8k_row_dot_x8_avx2(
            q.as_ptr(),
            row_bytes,
            blocks_per_row,
            &q8k_oxk,
            &mut octet,
        )
    };
    for (r, &got) in octet.iter().enumerate() {
        assert_eq!(got.to_bits(), legacy[r].to_bits(), "oxk x8 row {r}");
    }

    // Range helper over an x8+x4+x1 tail split (24 = 8+8+4+4 tails inside).
    let mut out = vec![0.0f32; rows];
    oxidize_kernels::gemv_q4k_range(&q, blocks_per_row, &q8k_oxk, &mut out);
    for (r, &got) in out.iter().enumerate() {
        assert_eq!(got.to_bits(), legacy[r].to_bits(), "oxk range row {r}");
    }
}

#[test]
#[cfg(not(feature = "cuda"))]
fn q4_k_x4_kernel_matches_single_row_paths() {
    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4 || (a.is_nan() && b.is_nan())
    }

    use crate::quantization::{quantize_scalar, quantized_size};
    // rows multiple of 32 exercises the 4-row expert kernel; rows*cols
    // above PARALLEL_GEMV_MIN_OPS exercises the parallel x4 gemv path.
    let (n_experts, rows, cols) = (2usize, 64usize, 256usize);
    let total = n_experts * rows * cols;
    let mut bytes = vec![0u8; total * 4];
    for i in 0..total {
        let v = (((i * 13 + 11) % 103) as f32) / 51.0 - 1.0;
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let q_size = quantized_size(GgufQuantizationType::Q4_K_M, total).unwrap();
    let mut q = vec![0u8; q_size];
    quantize_scalar(
        GgufQuantizationType::F32,
        GgufQuantizationType::Q4_K_M,
        &bytes,
        &mut q,
    )
    .unwrap();
    let input: Vec<f32> = (0..cols)
        .map(|i| (((i * 7 + 5) % 91) as f32) / 45.0 - 1.0)
        .collect();

    // Reference: per-row single dot via 1-row gemv calls.
    let expert_bytes = q.len() / n_experts;
    let row_bytes = expert_bytes / rows;
    let mut want = vec![0.0f32; n_experts * rows];
    for e in 0..n_experts {
        for r in 0..rows {
            let start = e * expert_bytes + r * row_bytes;
            let mut out = [0.0f32];
            gemv_quantized_f32(
                GgufQuantizationType::Q4_K_M,
                &q[start..start + row_bytes],
                1,
                cols,
                &input,
                &mut out,
            )
            .unwrap();
            want[e * rows + r] = out[0];
        }
    }

    // Expert path (x4 kernel when rows % 32 == 0). Exact equality: the x4
    // kernel performs the identical op sequence per row.
    let selected = [1usize, 0usize];
    let mut batched = vec![0.0f32; selected.len() * rows];
    gemv_quantized_experts_f32(
        GgufQuantizationType::Q4_K_M,
        &q,
        n_experts,
        &selected,
        rows,
        cols,
        &input,
        0,
        &mut batched,
    )
    .unwrap();
    for (slot, &e) in selected.iter().enumerate() {
        for r in 0..rows {
            let got = batched[slot * rows + r];
            let expected = want[e * rows + r];
            assert!(
                approx_eq(got, expected),
                "expert x4: slot {slot} expert {e} row {r}: got {got}, want {expected}"
            );
        }
    }

    // Fused gate+up region must match two separate expert GEMV calls
    // exactly (same kernel per row). Reuse the matrix for both halves.
    let mut fused = vec![0.0f32; 2 * selected.len() * rows];
    gemv_quantized_experts_gate_up_f32(
        GgufQuantizationType::Q4_K_M,
        &q,
        &q,
        n_experts,
        &selected,
        rows,
        cols,
        &input,
        &mut fused,
    )
    .unwrap();
    let half = selected.len() * rows;
    for j in 0..half {
        assert!(approx_eq(fused[j], batched[j]), "fused gate j {j}");
        assert!(approx_eq(fused[half + j], batched[j]), "fused up j {j}");
    }

    // Single-vector gemv path on one expert's matrix (serial branch also
    // routes through the x4 run_range helper).
    let mut gemv_out = vec![0.0f32; rows];
    gemv_quantized_f32(
        GgufQuantizationType::Q4_K_M,
        &q[..expert_bytes],
        rows,
        cols,
        &input,
        &mut gemv_out,
    )
    .unwrap();
    for r in 0..rows {
        assert!(approx_eq(gemv_out[r], want[r]), "gemv x4: row {r}");
    }
}

#[test]
fn gemm_vs_gemv_bit_exact_probe() {
    use crate::quantization::{quantize_scalar, quantized_size};
    let (rows, cols, batch) = (64usize, 512usize, 4usize);
    let total = rows * cols;
    let mut bytes = vec![0u8; total * 4];
    for i in 0..total {
        let v = (((i * 37 + 13) % 211) as f32) / 105.0 - 1.0;
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let q_size = quantized_size(GgufQuantizationType::Q4_K_M, total).unwrap();
    let mut q = vec![0u8; q_size];
    quantize_scalar(
        GgufQuantizationType::F32,
        GgufQuantizationType::Q4_K_M,
        &bytes,
        &mut q,
    )
    .unwrap();
    let mut inputs = vec![0.0f32; batch * cols];
    for (i, x) in inputs.iter_mut().enumerate() {
        *x = (((i * 19 + 7) % 113) as f32) / 56.0 - 1.0;
    }
    let mut gemm_out = vec![0.0f32; batch * rows];
    gemm_quantized_f32(
        GgufQuantizationType::Q4_K_M,
        &q,
        rows,
        cols,
        &inputs,
        &mut gemm_out,
        batch,
    )
    .unwrap();
    let mut mismatches = 0;
    for t in 0..batch {
        let mut gemv_out = vec![0.0f32; rows];
        gemv_quantized_f32(
            GgufQuantizationType::Q4_K_M,
            &q,
            rows,
            cols,
            &inputs[t * cols..(t + 1) * cols],
            &mut gemv_out,
        )
        .unwrap();
        for r in 0..rows {
            if gemm_out[t * rows + r].to_bits() != gemv_out[r].to_bits() {
                if mismatches < 5 {
                    eprintln!(
                        "t={t} r={r}: gemm={} gemv={} diff={}",
                        gemm_out[t * rows + r],
                        gemv_out[r],
                        gemm_out[t * rows + r] - gemv_out[r]
                    );
                }
                mismatches += 1;
            }
        }
    }
    assert_eq!(
        mismatches,
        0,
        "{mismatches} bit mismatches of {}",
        batch * rows
    );
}

#[test]
fn batched_expert_gemv_matches_per_expert_q4_k() {
    use crate::quantization::{quantize_scalar, quantized_size};
    let (n_experts, rows, cols) = (3usize, 4usize, 256usize);
    let total = n_experts * rows * cols;
    let mut bytes = vec![0u8; total * 4];
    for i in 0..total {
        let v = (((i * 31 + 7) % 101) as f32) / 50.0 - 1.0; // spread across [-1, 1]
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let q_size = quantized_size(GgufQuantizationType::Q4_K_M, total).unwrap();
    let mut q = vec![0u8; q_size];
    quantize_scalar(
        GgufQuantizationType::F32,
        GgufQuantizationType::Q4_K_M,
        &bytes,
        &mut q,
    )
    .unwrap();
    let expert_bytes = q.len() / n_experts;
    let selected = [2usize, 0usize];

    // Shared input (gate/up): input_stride = 0.
    let input: Vec<f32> = (0..cols)
        .map(|i| (((i * 17 + 3) % 97) as f32) / 48.0 - 1.0)
        .collect();
    let mut batched = vec![0.0f32; selected.len() * rows];
    gemv_quantized_experts_f32(
        GgufQuantizationType::Q4_K_M,
        &q,
        n_experts,
        &selected,
        rows,
        cols,
        &input,
        0,
        &mut batched,
    )
    .unwrap();
    for (slot, &e) in selected.iter().enumerate() {
        let mut want = vec![0.0f32; rows];
        gemv_quantized_f32(
            GgufQuantizationType::Q4_K_M,
            &q[e * expert_bytes..(e + 1) * expert_bytes],
            rows,
            cols,
            &input,
            &mut want,
        )
        .unwrap();
        for r in 0..rows {
            assert!(
                (batched[slot * rows + r] - want[r]).abs() < 1e-4,
                "shared slot {slot} e {e} row {r}: batched={} want={}",
                batched[slot * rows + r],
                want[r]
            );
        }
    }

    // Per-expert input (down): input_stride = cols.
    let mut inputs = vec![0.0f32; selected.len() * cols];
    for s in 0..selected.len() {
        for i in 0..cols {
            inputs[s * cols + i] = (((i * 23 + s * 5 + 1) % 89) as f32) / 44.0 - 1.0;
        }
    }
    let mut batched2 = vec![0.0f32; selected.len() * rows];
    gemv_quantized_experts_f32(
        GgufQuantizationType::Q4_K_M,
        &q,
        n_experts,
        &selected,
        rows,
        cols,
        &inputs,
        cols,
        &mut batched2,
    )
    .unwrap();
    for (slot, &e) in selected.iter().enumerate() {
        let mut want = vec![0.0f32; rows];
        gemv_quantized_f32(
            GgufQuantizationType::Q4_K_M,
            &q[e * expert_bytes..(e + 1) * expert_bytes],
            rows,
            cols,
            &inputs[slot * cols..(slot + 1) * cols],
            &mut want,
        )
        .unwrap();
        for r in 0..rows {
            assert!((batched2[slot * rows + r] - want[r]).abs() < CUDA_TOL);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn q2_k_dot_avx2_matches_scalar() {
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
        return;
    }
    let mut block = vec![0u8; BLOCK_Q2_K_SIZE];
    let mut state: u32 = 0xCAFE_BABE;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 16) as u8
    };
    for b in block.iter_mut() {
        *b = next();
    }
    // Valid f16 d/dmin in finite range.
    block[80] = 0x00;
    block[81] = 0x3c; // d = 1.0
    block[82] = 0x00;
    block[83] = 0x38; // min = 0.5
    let vector: Vec<f32> = (0..QK_K).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

    let scalar = q2_k_dot_scalar(&block, &vector);
    let simd = unsafe { q2_k_dot_avx2(&block, &vector) };
    assert!(
        (scalar - simd).abs() <= 1e-3 * (1.0 + scalar.abs()),
        "AVX2 Q2_K dot ({simd}) diverged from scalar ({scalar})"
    );
}

#[test]
fn q2_k_dot_scalar_matches_dequant_reference() {
    use crate::quantization::dequantize_q2_k_scalar;
    let mut block = vec![0u8; BLOCK_Q2_K_SIZE];
    let mut state: u32 = 0xDEAD_BEEF;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 16) as u8
    };
    for b in block.iter_mut() {
        *b = next();
    }
    block[80] = 0x00;
    block[81] = 0x3c;
    block[82] = 0x00;
    block[83] = 0x38;
    let vector: Vec<f32> = (0..QK_K).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();

    // Reference: full dequant then plain f32 dot.
    let mut dequant = vec![0.0_f32; QK_K];
    dequantize_q2_k_scalar(&block, &mut dequant).expect("dequant ok");
    let reference: f32 = dequant.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
    let fused = q2_k_dot_scalar(&block, &vector);
    assert!(
        (reference - fused).abs() <= 1e-4 * (1.0 + reference.abs()),
        "Q2_K fused dot ({fused}) diverged from dequant-then-dot reference ({reference})"
    );
}

#[test]
#[cfg(not(feature = "cuda"))]
fn nvfp4_gemv_and_gemm_match_dequant_reference() {
    let rows = 3;
    let blocks_per_row = 2;
    let cols = blocks_per_row * QK_NVFP4;
    let batch = 4;
    let mut weights = vec![0_u8; rows * blocks_per_row * BLOCK_NVFP4_SIZE];
    for (block_idx, block) in weights.chunks_exact_mut(BLOCK_NVFP4_SIZE).enumerate() {
        block[0] = 0x38;
        block[1] = 0x40;
        block[2] = 0x30;
        block[3] = 0x00;
        for (i, q) in block[4..].iter_mut().enumerate() {
            *q = (((i + block_idx) & 0x0f) | ((((15 - (i & 0x0f)) + block_idx) & 0x0f) << 4)) as u8;
        }
    }
    let inputs = (0..batch * cols)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.03125)
        .collect::<Vec<_>>();

    let mut dequant = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let src = (r * blocks_per_row + b) * BLOCK_NVFP4_SIZE;
            let dst = r * cols + b * QK_NVFP4;
            nvfp4_dequantize_block(
                &weights[src..src + BLOCK_NVFP4_SIZE],
                &mut dequant[dst..dst + QK_NVFP4],
            );
        }
    }

    let mut gemv = vec![0.0_f32; rows];
    gemv_quantized_f32(
        GgufQuantizationType::NVFP4,
        &weights,
        rows,
        cols,
        &inputs[..cols],
        &mut gemv,
    )
    .expect("nvfp4 gemv succeeds");
    for r in 0..rows {
        let expected: f32 = dequant[r * cols..(r + 1) * cols]
            .iter()
            .zip(inputs[..cols].iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!((gemv[r] - expected).abs() <= 1e-5 * (1.0 + expected.abs()));
    }

    let mut gemm = vec![0.0_f32; batch * rows];
    gemm_quantized_f32(
        GgufQuantizationType::NVFP4,
        &weights,
        rows,
        cols,
        &inputs,
        &mut gemm,
        batch,
    )
    .expect("nvfp4 gemm succeeds");
    for token in 0..batch {
        for r in 0..rows {
            let expected: f32 = dequant[r * cols..(r + 1) * cols]
                .iter()
                .zip(inputs[token * cols..(token + 1) * cols].iter())
                .map(|(a, b)| a * b)
                .sum();
            let actual = gemm[token * rows + r];
            assert!((actual - expected).abs() <= 1e-5 * (1.0 + expected.abs()));
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn q4_k_dot_avx2_matches_scalar() {
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
        return;
    }
    // Deterministic pseudo-random Q4_K block (BLOCK_Q4_K_SIZE bytes) + vector.
    let mut block = vec![0u8; BLOCK_Q4_K_SIZE];
    let mut state: u32 = 0x1234_5678;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 16) as u8
    };
    for b in block.iter_mut() {
        *b = next();
    }
    // Use small, valid f16 scale/min so values are finite.
    block[0] = 0x00;
    block[1] = 0x3c; // d = 1.0 in f16
    block[2] = 0x00;
    block[3] = 0x38; // min = 0.5 in f16
    let vector: Vec<f32> = (0..QK_K).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();

    let scalar = q4_k_dot_scalar(&block, &vector);
    let simd = unsafe { q4_k_dot_avx2(&block, &vector) };
    assert!(
        (scalar - simd).abs() <= 1e-3 * (1.0 + scalar.abs()),
        "AVX2 Q4_K dot ({simd}) diverged from scalar ({scalar})"
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn batched_q4_k_q8_k_gemm_matches_repeated_q4_k_gemv() {
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
        return;
    }

    let rows = 5;
    let blocks_per_row = 3;
    let cols = blocks_per_row * QK_K;
    let batch = 7;
    let mut weights = vec![0_u8; rows * blocks_per_row * BLOCK_Q4_K_SIZE];
    let mut state: u32 = 0xA11C_E55D;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 16) as u8
    };
    for block in weights.chunks_exact_mut(BLOCK_Q4_K_SIZE) {
        block[0] = 0x00;
        block[1] = 0x34; // d = 0.25
        block[2] = 0x00;
        block[3] = 0x28; // min = 0.03125
        for (i, scale) in block[4..16].iter_mut().enumerate() {
            *scale = ((i * 5 + 3) & 0x3f) as u8;
        }
        for byte in &mut block[16..] {
            *byte = next();
        }
    }
    let inputs = (0..batch * cols)
        .map(|i| ((i as f32 * 0.013).cos() * 0.75) + ((i % 7) as f32 - 3.0) * 0.03)
        .collect::<Vec<_>>();

    let mut batched = vec![0.0_f32; batch * rows];
    gemm_quantized_f32(
        GgufQuantizationType::Q4_K_M,
        &weights,
        rows,
        cols,
        &inputs,
        &mut batched,
        batch,
    )
    .expect("batched q4_k gemm should succeed");

    let mut expected = vec![0.0_f32; batch * rows];
    for token in 0..batch {
        gemv_quantized_f32(
            GgufQuantizationType::Q4_K_M,
            &weights,
            rows,
            cols,
            &inputs[token * cols..(token + 1) * cols],
            &mut expected[token * rows..(token + 1) * rows],
        )
        .expect("q4_k gemv should succeed");
    }

    for (actual, reference) in batched.iter().zip(expected.iter()) {
        assert!(
            (actual - reference).abs() <= CUDA_TOL * (1.0 + reference.abs()),
            "batched q4_k gemm value {actual} diverged from repeated gemv {reference}"
        );
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn q4_k_q8_k_row_dot_vnni_matches_avx2() {
    if !q4_k_q8_k_vnni_available() || !q4_k_q8_k_avx2_available() {
        return; // No VNNI hardware in this environment; nothing to compare.
    }

    let blocks_per_row = 4;
    let cols = blocks_per_row * QK_K;
    let mut row = vec![0_u8; blocks_per_row * BLOCK_Q4_K_SIZE];
    let mut state: u32 = 0x1357_9BDF;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 16) as u8
    };
    for block in row.chunks_exact_mut(BLOCK_Q4_K_SIZE) {
        block[0] = 0x00;
        block[1] = 0x34; // d = 0.25
        block[2] = 0x00;
        block[3] = 0x28; // min = 0.03125
        for (i, scale) in block[4..16].iter_mut().enumerate() {
            *scale = ((i * 7 + 1) & 0x3f) as u8;
        }
        for byte in &mut block[16..] {
            *byte = next();
        }
    }

    let vector = (0..cols)
        .map(|i| ((i as f32 * 0.021).sin() * 0.6) + ((i % 5) as f32 - 2.0) * 0.05)
        .collect::<Vec<_>>();
    let mut q8k = vec![0_u8; blocks_per_row * BLOCK_Q8_K_BYTES];
    quantize_vector_q8_k_into(&vector, blocks_per_row, &mut q8k);

    let avx2 = unsafe { q4_k_q8_k_row_dot_avx2(&row, blocks_per_row, &q8k) };
    let vnni = unsafe { q4_k_q8_k_row_dot_vnni(&row, blocks_per_row, &q8k) };
    assert!(
        (avx2 - vnni).abs() <= 1e-4 * (1.0 + avx2.abs()),
        "VNNI row dot {vnni} diverged from AVX2 {avx2}"
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn q6_k_dot_avx2_matches_scalar() {
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
        return;
    }
    let mut block = vec![0u8; BLOCK_Q6_K_SIZE];
    let mut state: u32 = 0xBADC_0FFE;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 16) as u8
    };
    for b in block.iter_mut() {
        *b = next();
    }
    block[208] = 0x00;
    block[209] = 0x3c; // d = 1.0
    let vector: Vec<f32> = (0..QK_K).map(|i| ((i % 19) as f32 - 9.0) * 0.07).collect();

    let scalar = q6_k_dot_scalar(&block, &vector);
    let simd = unsafe { q6_k_dot_avx2(&block, &vector) };
    assert!(
        (scalar - simd).abs() <= 1e-3 * (1.0 + scalar.abs()),
        "AVX2 Q6_K dot ({simd}) diverged from scalar ({scalar})"
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn q8_0_dot_avx2_matches_scalar() {
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
        return;
    }
    let mut block = vec![0u8; BLOCK_Q8_0_SIZE];
    block[0] = 0x00;
    block[1] = 0x3c; // scale = 1.0
    for (i, q) in block[2..].iter_mut().enumerate() {
        *q = (i as i8 - 16) as u8;
    }
    let vector: Vec<f32> = (0..QK8_0).map(|i| ((i % 7) as f32 - 3.0) * 0.13).collect();

    let scalar = q8_0_dot_scalar(&block, &vector);
    let simd = unsafe { q8_0_dot_avx2(&block, &vector) };
    assert!(
        (scalar - simd).abs() <= 1e-4 * (1.0 + scalar.abs()),
        "AVX2 Q8_0 dot ({simd}) diverged from scalar ({scalar})"
    );
}

fn reference_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    seq_len: usize,
    dim: usize,
    output: &mut [f32],
) {
    let scale = 1.0_f32 / (dim as f32).sqrt();
    let mut scores = vec![0.0_f32; seq_len];

    for (idx, key_row) in key.chunks_exact(dim).enumerate() {
        let dot = query
            .iter()
            .zip(key_row.iter())
            .map(|(q, k)| q * k)
            .sum::<f32>();
        scores[idx] = dot * scale;
    }

    let max_score = scores.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    for score in &mut scores {
        *score = (*score - max_score).exp();
    }
    let sum_exp = scores.iter().sum::<f32>();
    for score in &mut scores {
        *score /= sum_exp;
    }

    output.fill(0.0);
    for (weight, value_row) in scores.iter().zip(value.chunks_exact(dim)) {
        for (out, v) in output.iter_mut().zip(value_row.iter()) {
            *out += weight * v;
        }
    }
}

#[test]
fn creates_tensor_with_shape_strides_and_dtype() {
    let tensor = Tensor::new(vec![4, 8], vec![8, 1], DType::F32);

    assert_eq!(tensor.shape, vec![4, 8]);
    assert_eq!(tensor.strides, vec![8, 1]);
    assert_eq!(tensor.dtype, DType::F32);
}

#[test]
#[should_panic(expected = "shape and strides must have the same rank")]
fn rejects_mismatched_shape_and_strides() {
    let _ = Tensor::new(vec![2, 3], vec![3], DType::I8);
}

#[test]
fn gemv_multiplies_matrix_and_vector() {
    let matrix = vec![
        1.0_f32, 2.0, 3.0, //
        4.0, 5.0, 6.0,
    ];
    let vector = vec![0.5_f32, -1.0, 2.0];
    let mut output = vec![0.0_f32; 2];

    gemv_f32(&matrix, 2, 3, &vector, &mut output).expect("gemv should succeed");

    assert!((output[0] - 4.5).abs() < 1e-6);
    assert!((output[1] - 9.0).abs() < 1e-6);
}

#[test]
fn gemv_rejects_invalid_input_shapes() {
    let mut output = vec![0.0_f32; 2];
    let err = gemv_f32(&[1.0_f32, 2.0, 3.0], 2, 2, &[1.0, 2.0], &mut output)
        .expect_err("matrix length mismatch should fail");
    assert!(matches!(err, GemvError::InvalidMatrixLength { .. }));

    let matrix = vec![1.0_f32, 2.0, 3.0, 4.0];
    let err = gemv_f32(&matrix, 2, 2, &[1.0_f32], &mut output)
        .expect_err("vector length mismatch should fail");
    assert!(matches!(err, GemvError::InvalidVectorLength { .. }));

    let mut short_output = vec![0.0_f32; 1];
    let err = gemv_f32(&matrix, 2, 2, &[1.0_f32, 1.0], &mut short_output)
        .expect_err("output length mismatch should fail");
    assert!(matches!(err, GemvError::InvalidOutputLength { .. }));
}

#[test]
fn quantized_q8_0_gemv_matches_f32_reference() {
    let rows = 2;
    let cols = 32;
    let matrix = (0..rows * cols)
        .map(|i| (i as f32 * 0.125) - 2.0)
        .collect::<Vec<_>>();
    let vector = (0..cols)
        .map(|i| (i as f32 * 0.05) - 0.6)
        .collect::<Vec<_>>();

    let mut matrix_bytes = Vec::with_capacity(matrix.len() * 4);
    for value in &matrix {
        matrix_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let q8_bytes_len =
        crate::quantization::quantized_size(GgufQuantizationType::Q8_0, matrix.len())
            .expect("quantized size is known");
    let mut quantized_matrix = vec![0_u8; q8_bytes_len];
    crate::quantization::quantize_scalar(
        GgufQuantizationType::F32,
        GgufQuantizationType::Q8_0,
        &matrix_bytes,
        &mut quantized_matrix,
    )
    .expect("q8_0 quantization should succeed");

    let mut quantized_out = vec![0.0_f32; rows];
    gemv_quantized_f32(
        GgufQuantizationType::Q8_0,
        &quantized_matrix,
        rows,
        cols,
        &vector,
        &mut quantized_out,
    )
    .expect("quantized gemv should succeed");

    let mut dequantized = vec![0.0_f32; rows * cols];
    crate::quantization::dequantize_scalar(
        GgufQuantizationType::Q8_0,
        &quantized_matrix,
        &mut dequantized,
    )
    .expect("dequantization should succeed");
    let mut reference = vec![0.0_f32; rows];
    gemv_f32(&dequantized, rows, cols, &vector, &mut reference).expect("f32 gemv should succeed");

    for (lhs, rhs) in quantized_out.iter().zip(reference.iter()) {
        assert!((lhs - rhs).abs() < CUDA_TOL);
    }
}

#[test]
fn iq4_xs_gemv_matches_dequantize_then_gemv_reference() {
    let rows = 3;
    let cols = QK_K * 2;
    let matrix = (0..rows * cols)
        .map(|i| ((i as f32 * 0.017).sin() * 3.0) - 0.5)
        .collect::<Vec<_>>();
    let vector = (0..cols)
        .map(|i| ((i as f32 * 0.07).cos() * 0.5) + 0.1)
        .collect::<Vec<_>>();

    let mut matrix_bytes = Vec::with_capacity(matrix.len() * 4);
    for value in &matrix {
        matrix_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let bytes_len = crate::quantization::quantized_size(GgufQuantizationType::IQ4_XS, matrix.len())
        .expect("quantized size is known");
    let mut quantized_matrix = vec![0_u8; bytes_len];
    crate::quantization::quantize_scalar(
        GgufQuantizationType::F32,
        GgufQuantizationType::IQ4_XS,
        &matrix_bytes,
        &mut quantized_matrix,
    )
    .expect("iq4_xs quantization should succeed");

    let mut quantized_out = vec![0.0_f32; rows];
    gemv_quantized_f32(
        GgufQuantizationType::IQ4_XS,
        &quantized_matrix,
        rows,
        cols,
        &vector,
        &mut quantized_out,
    )
    .expect("iq4_xs gemv should succeed");

    let mut dequantized = vec![0.0_f32; rows * cols];
    crate::quantization::dequantize_scalar(
        GgufQuantizationType::IQ4_XS,
        &quantized_matrix,
        &mut dequantized,
    )
    .expect("dequantization should succeed");
    let mut reference = vec![0.0_f32; rows];
    gemv_f32(&dequantized, rows, cols, &vector, &mut reference).expect("f32 gemv should succeed");

    for (lhs, rhs) in quantized_out.iter().zip(reference.iter()) {
        assert!((lhs - rhs).abs() < 1.0e-3, "{lhs} vs {rhs}");
    }
}

#[test]
fn fused_q8_0_gemv_matches_dequantize_then_gemv_reference() {
    let rows = 3;
    let cols = QK8_0 * 2;
    let matrix = (0..rows * cols)
        .map(|i| ((i as f32 * 0.03125).sin() * 6.0) - 1.0)
        .collect::<Vec<_>>();
    let vector = (0..cols)
        .map(|i| ((i as f32 * 0.11).cos() * 0.5) + 0.25)
        .collect::<Vec<_>>();

    let mut matrix_bytes = Vec::with_capacity(matrix.len() * 4);
    for value in &matrix {
        matrix_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let q8_bytes_len =
        crate::quantization::quantized_size(GgufQuantizationType::Q8_0, matrix.len())
            .expect("quantized size is known");
    let mut quantized_matrix = vec![0_u8; q8_bytes_len];
    crate::quantization::quantize_scalar(
        GgufQuantizationType::F32,
        GgufQuantizationType::Q8_0,
        &matrix_bytes,
        &mut quantized_matrix,
    )
    .expect("q8_0 quantization should succeed");

    let mut fused_out = vec![0.0_f32; rows];
    gemv_quantized_f32(
        GgufQuantizationType::Q8_0,
        &quantized_matrix,
        rows,
        cols,
        &vector,
        &mut fused_out,
    )
    .expect("fused q8_0 gemv should succeed");

    let mut dequantized = vec![0.0_f32; rows * cols];
    crate::quantization::dequantize_scalar(
        GgufQuantizationType::Q8_0,
        &quantized_matrix,
        &mut dequantized,
    )
    .expect("dequantization should succeed");
    let mut reference = vec![0.0_f32; rows];
    gemv_f32(&dequantized, rows, cols, &vector, &mut reference).expect("f32 gemv should succeed");

    for (lhs, rhs) in fused_out.iter().zip(reference.iter()) {
        assert!((lhs - rhs).abs() < CUDA_TOL);
    }
}

#[test]
fn gemm_multiplies_matrices() {
    let left = vec![
        1.0_f32, 2.0, 3.0, //
        4.0, 5.0, 6.0,
    ];
    let right = vec![
        7.0_f32, 8.0, //
        9.0, 10.0, //
        11.0, 12.0,
    ];
    let mut output = vec![0.0_f32; 4];

    gemm_f32(&left, 2, 3, &right, 2, &mut output).expect("gemm should succeed");

    let expected = [58.0_f32, 64.0, 139.0, 154.0];
    for (actual, expected) in output.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn gemm_rejects_invalid_input_shapes() {
    let right = vec![
        1.0_f32, 2.0, //
        3.0, 4.0, //
        5.0, 6.0,
    ];
    let mut output = vec![0.0_f32; 4];

    let err = gemm_f32(&[1.0_f32, 2.0, 3.0], 2, 3, &right, 2, &mut output)
        .expect_err("left matrix length mismatch should fail");
    assert!(matches!(err, GemmError::InvalidLeftMatrixLength { .. }));

    let left = vec![
        1.0_f32, 2.0, 3.0, //
        4.0, 5.0, 6.0,
    ];
    let err = gemm_f32(&left, 2, 3, &[1.0_f32, 2.0, 3.0], 2, &mut output)
        .expect_err("right matrix length mismatch should fail");
    assert!(matches!(err, GemmError::InvalidRightMatrixLength { .. }));

    let mut short_output = vec![0.0_f32; 3];
    let err = gemm_f32(&left, 2, 3, &right, 2, &mut short_output)
        .expect_err("output length mismatch should fail");
    assert!(matches!(err, GemmError::InvalidOutputLength { .. }));
}

#[test]
fn gemm_multiplies_non_square_matrices() {
    let left = vec![
        1.0_f32, 2.0, //
        3.0, 4.0, //
        5.0, 6.0,
    ];
    let right = vec![
        7.0_f32, 8.0, 9.0, 10.0, //
        11.0, 12.0, 13.0, 14.0,
    ];
    let mut output = vec![0.0_f32; 12];

    gemm_f32(&left, 3, 2, &right, 4, &mut output).expect("gemm should succeed");

    let expected = [
        29.0_f32, 32.0, 35.0, 38.0, //
        65.0, 72.0, 79.0, 86.0, //
        101.0, 112.0, 123.0, 134.0,
    ];
    for (actual, expected) in output.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn tensor_parallel_gemm_matches_cpu_reference() {
    let rows = 4;
    let shared_dim = 8;
    let cols = 3;
    let left = (0..rows * shared_dim)
        .map(|i| (i as f32 * 0.125) - 1.0)
        .collect::<Vec<_>>();
    let right = (0..shared_dim * cols)
        .map(|i| (i as f32 * 0.05) + 0.25)
        .collect::<Vec<_>>();

    let mut expected = vec![0.0_f32; rows * cols];
    gemm_f32_cpu(&left, rows, shared_dim, &right, cols, &mut expected);

    let mut actual = vec![0.0_f32; rows * cols];
    gemm_f32_tensor_parallel(&left, rows, shared_dim, &right, cols, 4, &mut actual)
        .expect("tensor parallel gemm should succeed");

    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!((lhs - rhs).abs() < 1e-6);
    }
}

#[test]
fn tensor_parallel_gemm_rejects_invalid_shard_count() {
    let left = vec![1.0_f32; 6];
    let right = vec![1.0_f32; 6];
    let mut output = vec![0.0_f32; 4];
    let err = gemm_f32_tensor_parallel(&left, 2, 3, &right, 2, 2, &mut output)
        .expect_err("non-divisible shard count should fail");
    assert_eq!(
        err,
        GemmError::InvalidTensorParallelShardCount {
            shared_dim: 3,
            shard_count: 2
        }
    );
}

#[test]
fn gemm_i8_multiplies_matrices() {
    let left = vec![
        1_i8, -2, 3, //
        4, 0, -1,
    ];
    let right = vec![
        2_i8, -3, //
        1, 5, //
        -4, 2,
    ];
    let mut output = vec![0_i32; 4];

    gemm_i8(&left, 2, 3, &right, 2, &mut output).expect("int8 gemm should succeed");

    assert_eq!(output, vec![-12, -7, 12, -14]);
}

#[test]
fn gemm_i8_rejects_invalid_shapes() {
    let right = vec![
        1_i8, 2, //
        3, 4, //
        5, 6,
    ];
    let mut output = vec![0_i32; 4];

    let err = gemm_i8(&[1_i8, 2, 3], 2, 3, &right, 2, &mut output)
        .expect_err("left matrix length mismatch should fail");
    assert!(matches!(err, GemmError::InvalidLeftMatrixLength { .. }));

    let left = vec![
        1_i8, 2, 3, //
        4, 5, 6,
    ];
    let err = gemm_i8(&left, 2, 3, &[1_i8, 2, 3], 2, &mut output)
        .expect_err("right matrix length mismatch should fail");
    assert!(matches!(err, GemmError::InvalidRightMatrixLength { .. }));

    let mut short_output = vec![0_i32; 3];
    let err = gemm_i8(&left, 2, 3, &right, 2, &mut short_output)
        .expect_err("output length mismatch should fail");
    assert!(matches!(err, GemmError::InvalidOutputLength { .. }));
}

#[test]
fn gemm_i4_multiplies_packed_matrices() {
    let left_values = [1_i8, -2, 3, 4, 0, -1];
    let right_values = [2_i8, -3, 1, 5, -4, 2];
    let left = pack_i4_values(&left_values);
    let right = pack_i4_values(&right_values);
    let mut output = vec![0_i32; 4];

    gemm_i4(&left, 2, 3, &right, 2, &mut output).expect("int4 gemm should succeed");

    assert_eq!(output, vec![-12, -7, 12, -14]);
}

#[test]
fn gemm_i4_supports_odd_value_count() {
    let left_values = [1_i8, -2, 3];
    let right_values = [2_i8, 4, -1];
    let left = pack_i4_values(&left_values);
    let right = pack_i4_values(&right_values);
    let mut output = vec![0_i32; 1];

    gemm_i4(&left, 1, 3, &right, 1, &mut output).expect("odd shared dim int4 gemm should succeed");

    assert_eq!(output, vec![-9]);
}

#[test]
fn gemm_i4_rejects_invalid_shapes() {
    let right = pack_i4_values(&[1_i8, 2, 3, 4, 5, 6]);
    let mut output = vec![0_i32; 4];

    let err = gemm_i4(&pack_i4_values(&[1_i8, 2, 3]), 2, 3, &right, 2, &mut output)
        .expect_err("left matrix byte length mismatch should fail");
    assert!(matches!(err, GemmError::InvalidLeftMatrixLength { .. }));

    let left = pack_i4_values(&[1_i8, 2, 3, 4, 5, 6]);
    let err = gemm_i4(&left, 2, 3, &pack_i4_values(&[1_i8, 2, 3]), 2, &mut output)
        .expect_err("right matrix byte length mismatch should fail");
    assert!(matches!(err, GemmError::InvalidRightMatrixLength { .. }));

    let mut short_output = vec![0_i32; 3];
    let err = gemm_i4(&left, 2, 3, &right, 2, &mut short_output)
        .expect_err("output length mismatch should fail");
    assert!(matches!(err, GemmError::InvalidOutputLength { .. }));
}

fn pack_i4_values(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(2));
    for chunk in values.chunks(2) {
        let low = ((chunk[0] + 8) as u8) & 0x0F;
        let high = if chunk.len() == 2 {
            ((chunk[1] + 8) as u8) & 0x0F
        } else {
            0
        };
        packed.push(low | (high << 4));
    }
    packed
}

#[test]
fn scaled_dot_product_attention_computes_weighted_output() {
    let query = vec![1.0_f32, 0.0];
    let key = vec![
        1.0_f32, 0.0, //
        0.0, 1.0,
    ];
    let value = vec![
        10.0_f32, 0.0, //
        0.0, 20.0,
    ];
    let mut output = vec![0.0_f32; 2];

    scaled_dot_product_attention_f32(&query, &key, &value, 2, 2, &mut output)
        .expect("attention should succeed");

    assert!((output[0] - 6.697615).abs() < 1e-5);
    assert!((output[1] - 6.604_77).abs() < 1e-5);
}

#[test]
fn scaled_dot_product_attention_rejects_invalid_shapes() {
    let mut output = vec![0.0_f32; 2];
    let err = scaled_dot_product_attention_f32(
        &[1.0_f32],
        &[1.0_f32, 0.0],
        &[1.0_f32, 0.0],
        1,
        2,
        &mut output,
    )
    .expect_err("query length mismatch should fail");
    assert!(matches!(err, AttentionError::InvalidQueryLength { .. }));

    let err = scaled_dot_product_attention_f32(
        &[1.0_f32, 0.0],
        &[1.0_f32, 0.0, 1.0],
        &[1.0_f32, 0.0, 1.0, 0.0],
        2,
        2,
        &mut output,
    )
    .expect_err("key length mismatch should fail");
    assert!(matches!(err, AttentionError::InvalidKeyLength { .. }));

    let err = scaled_dot_product_attention_f32(
        &[1.0_f32, 0.0],
        &[1.0_f32, 0.0, 1.0, 0.0],
        &[1.0_f32, 0.0, 1.0],
        2,
        2,
        &mut output,
    )
    .expect_err("value length mismatch should fail");
    assert!(matches!(err, AttentionError::InvalidValueLength { .. }));

    let mut short_output = vec![0.0_f32; 1];
    let err = scaled_dot_product_attention_f32(
        &[1.0_f32, 0.0],
        &[1.0_f32, 0.0, 1.0, 0.0],
        &[1.0_f32, 0.0, 1.0, 0.0],
        2,
        2,
        &mut short_output,
    )
    .expect_err("output length mismatch should fail");
    assert!(matches!(err, AttentionError::InvalidOutputLength { .. }));
}

#[test]
fn scaled_dot_product_attention_matches_reference_softmax() {
    let dim = 4;
    let seq_len = 6;
    let query = vec![0.2_f32, -0.4, 1.2, 0.8];
    let key = vec![
        0.1_f32, 0.4, -0.2, 1.0, //
        0.5, -0.7, 0.3, 0.0, //
        -1.0, 0.1, 0.9, 0.2, //
        0.7, 0.6, -0.8, 0.3, //
        -0.4, 1.3, 0.2, -0.6, //
        0.8, -0.2, -0.5, 0.4,
    ];
    let value = vec![
        1.0_f32, 0.0, 2.0, -1.0, //
        0.3, 1.5, -0.2, 0.7, //
        -1.1, 0.6, 0.2, 0.4, //
        0.9, -0.8, 1.0, 0.5, //
        0.0, 0.2, -0.4, 1.2, //
        -0.7, 0.3, 0.8, -0.9,
    ];
    let mut actual = vec![0.0_f32; dim];
    let mut expected = vec![0.0_f32; dim];

    scaled_dot_product_attention_f32(&query, &key, &value, seq_len, dim, &mut actual)
        .expect("attention should succeed");
    reference_attention(&query, &key, &value, seq_len, dim, &mut expected);

    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!((lhs - rhs).abs() < 1e-6);
    }
}

#[test]
fn scaled_dot_product_attention_matches_pytorch_reference() {
    let query = vec![0.3_f32, -0.8, 1.1, 0.2];
    let key = vec![
        0.4_f32, -0.7, 0.9, -0.1, //
        -1.2, 0.3, 0.5, 0.8, //
        0.6, 1.0, -0.4, 0.2, //
        0.1, -0.2, 0.7, 1.3, //
        -0.5, 0.9, -1.1, 0.4,
    ];
    let value = vec![
        1.0_f32, -0.5, 0.3, 0.9, //
        -0.7, 1.2, 0.4, -1.1, //
        0.6, 0.8, -0.2, 0.5, //
        1.4, -0.9, 1.1, -0.3, //
        -0.4, 0.2, -1.3, 0.7,
    ];
    let expected = [0.704_716_4_f32, -0.158_690_65, 0.412_106_54, 0.145_940_9];
    let mut actual = vec![0.0_f32; 4];

    scaled_dot_product_attention_f32(&query, &key, &value, 5, 4, &mut actual)
        .expect("attention should succeed");

    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!((lhs - rhs).abs() < 1e-6);
    }
}

#[test]
fn scaled_dot_product_attention_is_stable_for_large_logits() {
    let dim = 2;
    let seq_len = 3;
    let query = vec![10_000.0_f32, -10_000.0];
    let key = vec![
        1.0_f32, -1.0, //
        -1.0, 1.0, //
        0.5, -0.5,
    ];
    let value = vec![
        2.0_f32, 4.0, //
        -3.0, 5.0, //
        7.0, -8.0,
    ];
    let mut output = vec![0.0_f32; dim];

    scaled_dot_product_attention_f32(&query, &key, &value, seq_len, dim, &mut output)
        .expect("attention should succeed");

    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn scaled_dot_product_attention_matches_reference_across_flash_blocks() {
    let dim = 8;
    let seq_len = FLASH_ATTENTION_BLOCK_TOKENS * 2 + 7;
    let query = (0..dim)
        .map(|i| (i as f32 * 0.13).sin() - 0.25)
        .collect::<Vec<_>>();
    let key = (0..seq_len * dim)
        .map(|i| ((i as f32 * 0.017).cos() * 1.3) - 0.2)
        .collect::<Vec<_>>();
    let value = (0..seq_len * dim)
        .map(|i| ((i as f32 * 0.031).sin() * 0.9) + 0.1)
        .collect::<Vec<_>>();

    let mut actual = vec![0.0_f32; dim];
    let mut expected = vec![0.0_f32; dim];
    scaled_dot_product_attention_f32(&query, &key, &value, seq_len, dim, &mut actual)
        .expect("flash-style attention should succeed");
    reference_attention(&query, &key, &value, seq_len, dim, &mut expected);

    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!((lhs - rhs).abs() < 1e-5);
    }
}

#[test]
fn scaled_dot_product_attention_matches_reference_for_long_contexts() {
    let dim = 4;
    let seq_len = FLASH_ATTENTION_BLOCK_TOKENS * 64;
    let query = (0..dim)
        .map(|i| ((i as f32 * 0.11).sin() * 0.7) - 0.1)
        .collect::<Vec<_>>();
    let key = (0..seq_len * dim)
        .map(|i| ((i as f32 * 0.007).cos() * 0.8) + 0.05)
        .collect::<Vec<_>>();
    let value = (0..seq_len * dim)
        .map(|i| ((i as f32 * 0.013).sin() * 1.1) - 0.2)
        .collect::<Vec<_>>();

    let mut actual = vec![0.0_f32; dim];
    let mut expected = vec![0.0_f32; dim];
    scaled_dot_product_attention_f32(&query, &key, &value, seq_len, dim, &mut actual)
        .expect("flash attention should support long contexts");
    reference_attention(&query, &key, &value, seq_len, dim, &mut expected);

    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!((lhs - rhs).abs() < 1e-5);
    }
}

#[test]
fn apply_rope_rotates_each_pair_by_position_dependent_angle() {
    let input = vec![1.0_f32, 0.0, 0.0, 1.0];
    let mut output = vec![0.0_f32; 4];

    apply_rope_f32(&input, 1, 4, 10_000.0, &mut output).expect("rope should succeed");

    assert!((output[0] - 1.0_f32.cos()).abs() < 1e-6);
    assert!((output[2] - 1.0_f32.sin()).abs() < 1e-6);

    let angle_1 = 1.0_f32 / 100.0;
    assert!((output[1] + angle_1.sin()).abs() < 1e-6);
    assert!((output[3] - angle_1.cos()).abs() < 1e-6);
}

#[test]
fn apply_rope_position_zero_is_identity() {
    let input = vec![0.25_f32, -0.75, 1.5, 2.0];
    let mut output = vec![0.0_f32; 4];

    apply_rope_f32(&input, 0, 4, 10_000.0, &mut output).expect("rope should succeed");

    for (actual, expected) in output.iter().zip(input.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn apply_rope_rejects_invalid_shapes() {
    let mut output = vec![0.0_f32; 4];
    let err = apply_rope_f32(&[1.0_f32, 2.0], 1, 4, 10_000.0, &mut output)
        .expect_err("input length mismatch should fail");
    assert!(matches!(err, RopeError::InvalidInputLength { .. }));

    let mut short_output = vec![0.0_f32; 2];
    let err = apply_rope_f32(&[1.0_f32, 2.0, 3.0, 4.0], 1, 4, 10_000.0, &mut short_output)
        .expect_err("output length mismatch should fail");
    assert!(matches!(err, RopeError::InvalidOutputLength { .. }));

    let mut odd_output = vec![0.0_f32; 3];
    let err = apply_rope_f32(&[1.0_f32, 2.0, 3.0], 1, 3, 10_000.0, &mut odd_output)
        .expect_err("odd head dimension should fail");
    assert!(matches!(err, RopeError::OddHeadDim { .. }));
}

#[test]
fn swiglu_applies_silu_gate_times_up_projection() {
    let gate = [0.0_f32, 2.0, -2.0];
    let up = [1.0_f32, 3.0, -4.0];
    let mut output = [0.0_f32; 3];

    apply_swiglu_f32(&gate, &up, &mut output).expect("swiglu should succeed");

    assert!((output[0] - 0.0).abs() < 1e-6);
    assert!((output[1] - 5.284_782_4).abs() < 1e-6);
    assert!((output[2] - 0.953_623_35).abs() < 1e-6);
}

#[test]
fn swiglu_rejects_mismatched_input_lengths() {
    let mut output = [0.0_f32; 2];

    let gate_err = apply_swiglu_f32(&[1.0_f32], &[1.0_f32, 2.0], &mut output)
        .expect_err("gate length mismatch should fail");
    assert!(matches!(gate_err, SwiGluError::InvalidGateLength { .. }));

    let up_err = apply_swiglu_f32(&[1.0_f32, 2.0], &[1.0_f32], &mut output)
        .expect_err("up length mismatch should fail");
    assert!(matches!(up_err, SwiGluError::InvalidUpLength { .. }));
}

#[test]
fn linear_activation_fuses_gemv_and_relu() {
    let matrix = [
        1.0_f32, -2.0, 3.0, //
        -4.0, 5.0, -6.0,
    ];
    let vector = [0.5_f32, 2.0, -1.0];
    let mut output = [0.0_f32; 2];

    linear_activation_f32(&matrix, 2, 3, &vector, ActivationFn::Relu, &mut output)
        .expect("fused linear+relu should succeed");

    let expected_linear = [-6.5_f32, 14.0];
    let expected = [expected_linear[0].max(0.0), expected_linear[1].max(0.0)];
    for (actual, expected) in output.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn linear_activation_matches_reference_gelu_and_silu() {
    let matrix = vec![
        0.2_f32, -0.4, 1.1, -0.9, //
        1.4, 0.3, -0.7, 0.5, //
        -0.8, 0.6, 0.9, -1.2,
    ];
    let vector = vec![0.3_f32, -0.2, 1.7, 0.4];
    let mut linear = vec![0.0_f32; 3];
    gemv_f32(&matrix, 3, 4, &vector, &mut linear).expect("gemv should succeed");

    for activation in [ActivationFn::Gelu, ActivationFn::Silu] {
        let mut fused = vec![0.0_f32; 3];
        linear_activation_f32(&matrix, 3, 4, &vector, activation, &mut fused)
            .expect("fused linear activation should succeed");
        let expected = linear
            .iter()
            .map(|value| activate(*value, activation))
            .collect::<Vec<_>>();
        for (actual, expected) in fused.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn linear_activation_rejects_invalid_shapes() {
    let matrix = [1.0_f32, 2.0, 3.0, 4.0];
    let vector = [1.0_f32, 2.0];
    let mut output = [0.0_f32; 2];

    let matrix_err = linear_activation_f32(
        &[1.0_f32, 2.0, 3.0],
        2,
        2,
        &vector,
        ActivationFn::Relu,
        &mut output,
    )
    .expect_err("matrix length mismatch should fail");
    assert!(matches!(
        matrix_err,
        LinearActivationError::InvalidMatrixLength { .. }
    ));

    let vector_err =
        linear_activation_f32(&matrix, 2, 2, &[1.0_f32], ActivationFn::Relu, &mut output)
            .expect_err("vector length mismatch should fail");
    assert!(matches!(
        vector_err,
        LinearActivationError::InvalidVectorLength { .. }
    ));

    let mut short_output = [0.0_f32; 1];
    let output_err = linear_activation_f32(
        &matrix,
        2,
        2,
        &vector,
        ActivationFn::Relu,
        &mut short_output,
    )
    .expect_err("output length mismatch should fail");
    assert!(matches!(
        output_err,
        LinearActivationError::InvalidOutputLength { .. }
    ));
}

#[test]
fn rms_norm_scales_by_root_mean_square() {
    let input = [1.0_f32, 2.0, 3.0];
    let weight = [1.0_f32, 1.0, 1.0];
    let mut output = [0.0_f32; 3];

    rms_norm_f32(&input, &weight, 0.0, &mut output).expect("rms norm should succeed");

    assert!((output[0] - 0.462_910_06).abs() < 1e-6);
    assert!((output[1] - 0.925_820_1).abs() < 1e-6);
    assert!((output[2] - 1.388_730_2).abs() < 1e-6);
}

#[test]
fn rms_norm_matches_pytorch_reference() {
    let input = [0.25_f32, -1.5, 3.0, 0.75, -0.5];
    let weight = [1.2_f32, 0.8, -0.6, 2.0, 0.5];
    let expected = [
        0.192_648_f32,
        -0.770_592_03,
        -1.155_888_1,
        0.963_24,
        -0.160_54,
    ];
    let mut output = [0.0_f32; 5];

    rms_norm_f32(&input, &weight, 1e-5, &mut output).expect("rms norm should succeed");

    for (actual, expected) in output.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn rms_norm_rejects_zero_dimension() {
    let err = rms_norm_f32(&[], &[], 1e-5, &mut []).expect_err("zero-length output should fail");
    assert_eq!(err, RmsNormError::ZeroDimension);
}

#[test]
fn rms_norm_rejects_mismatched_lengths() {
    let mut output = [0.0_f32; 2];
    let input_err = rms_norm_f32(&[1.0_f32], &[1.0_f32, 1.0], 1e-5, &mut output)
        .expect_err("input length mismatch should fail");
    assert!(matches!(input_err, RmsNormError::InvalidInputLength { .. }));

    let weight_err = rms_norm_f32(&[1.0_f32, 2.0], &[1.0_f32], 1e-5, &mut output)
        .expect_err("weight length mismatch should fail");
    assert!(matches!(
        weight_err,
        RmsNormError::InvalidWeightLength { .. }
    ));
}

#[test]
fn layer_norm_normalizes_and_applies_affine_transform() {
    let input = [1.0_f32, 2.0, 3.0];
    let weight = [1.0_f32, 1.0, 1.0];
    let bias = [0.0_f32, 0.0, 0.0];
    let mut output = [0.0_f32; 3];

    layer_norm_f32(&input, &weight, &bias, 0.0, &mut output).expect("layer norm should succeed");

    assert!((output[0] + 1.224_744_8).abs() < 1e-6);
    assert!((output[1] - 0.0).abs() < 1e-6);
    assert!((output[2] - 1.224_744_8).abs() < 1e-6);
}

#[test]
fn layer_norm_matches_pytorch_reference() {
    let input = [0.25_f32, -1.5, 3.0, 0.75, -0.5];
    let weight = [1.2_f32, 0.8, -0.6, 2.0, 0.5];
    let bias = [0.1_f32, -0.2, 0.3, -0.4, 0.9];
    let expected = [
        -0.019_601_732_f32,
        -1.209_970_1,
        -0.736_548_3,
        0.065_117_806,
        0.600_995_66,
    ];
    let mut output = [0.0_f32; 5];

    layer_norm_f32(&input, &weight, &bias, 1e-5, &mut output).expect("layer norm should succeed");

    for (actual, expected) in output.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn layer_norm_rejects_mismatched_lengths() {
    let mut output = [0.0_f32; 2];
    let input_err = layer_norm_f32(
        &[1.0_f32],
        &[1.0_f32, 1.0],
        &[0.0_f32, 0.0],
        1e-5,
        &mut output,
    )
    .expect_err("input length mismatch should fail");
    assert!(matches!(
        input_err,
        LayerNormError::InvalidInputLength { .. }
    ));

    let weight_err = layer_norm_f32(
        &[1.0_f32, 2.0],
        &[1.0_f32],
        &[0.0_f32, 0.0],
        1e-5,
        &mut output,
    )
    .expect_err("weight length mismatch should fail");
    assert!(matches!(
        weight_err,
        LayerNormError::InvalidWeightLength { .. }
    ));

    let bias_err = layer_norm_f32(
        &[1.0_f32, 2.0],
        &[1.0_f32, 1.0],
        &[0.0_f32],
        1e-5,
        &mut output,
    )
    .expect_err("bias length mismatch should fail");
    assert!(matches!(bias_err, LayerNormError::InvalidBiasLength { .. }));
}

#[test]
fn softmax_outputs_probabilities_that_sum_to_one() {
    let input = [1.0_f32, 2.0, 3.0];
    let mut output = [0.0_f32; 3];

    softmax_f32(&input, &mut output).expect("softmax should succeed");

    assert!((output[0] - 0.090_030_57).abs() < 1e-7);
    assert!((output[1] - 0.244_728_48).abs() < 1e-7);
    assert!((output[2] - 0.665_240_94).abs() < 1e-7);
    assert!((output.iter().sum::<f32>() - 1.0).abs() < 1e-7);
}

#[test]
fn softmax_matches_pytorch_reference() {
    let input = [1.25_f32, -0.5, 3.75, 0.0, -2.25];
    let expected = [
        0.073_137_f32,
        0.012_709_305,
        0.890_991_1,
        0.020_954_102,
        0.002_208_546_3,
    ];
    let mut output = [0.0_f32; 5];

    softmax_f32(&input, &mut output).expect("softmax should succeed");

    for (actual, expected) in output.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-7);
    }
}

#[test]
fn softmax_is_numerically_stable_for_large_inputs() {
    let input = [10_000.0_f32, 9_999.0, 9_998.0];
    let mut output = [0.0_f32; 3];

    softmax_f32(&input, &mut output).expect("softmax should succeed");

    assert!(output.iter().all(|value| value.is_finite()));
    assert!((output.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!(output[0] > output[1]);
    assert!(output[1] > output[2]);
}

#[test]
fn softmax_rejects_mismatched_lengths() {
    let mut output = [0.0_f32; 2];
    let err =
        softmax_f32(&[1.0_f32], &mut output).expect_err("mismatched input length should fail");
    assert!(matches!(err, SoftmaxError::InvalidInputLength { .. }));
}
