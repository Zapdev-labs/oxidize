use super::ffn_fusion_cuda_fixture::{
    run_q4k_gate_up_silu_cuda_event_benchmark, run_q4k_gate_up_silu_parity_case,
    run_q4k_gemv_block_size_cuda_event_benchmark,
};

fn assert_bitwise_equal(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "FFN output lengths differ");
    for (row, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual_value.to_bits(),
            expected_value.to_bits(),
            "FFN output row {row} differs: fused={actual_value}, eager={expected_value}"
        );
    }
}

fn median(samples: &[f64]) -> f64 {
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    (ordered[ordered.len() / 2 - 1] + ordered[ordered.len() / 2]) * 0.5
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn q4k_gate_up_silu_fused_matches_eager_exact() {
    // Given a representative 4K-hidden SwiGLU projection geometry.
    let (rows, blocks_per_row) = (11_008, 16);

    // When the existing fused and eager CUDA paths execute on identical inputs.
    let (eager, fused) = run_q4k_gate_up_silu_parity_case(rows, blocks_per_row)
        .expect("Q4_K fused FFN parity fixture must execute");

    // Then every output bit is identical.
    assert_bitwise_equal(&fused, &eager);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn q4k_gate_up_silu_fused_handles_boundary_shapes_exact() {
    // Given minimum, partial-warp-row, and partial-block-row Q4_K geometries.
    for (rows, blocks_per_row) in [(1, 1), (33, 2), (257, 1)] {
        // When both existing CUDA paths execute.
        let (eager, fused) = run_q4k_gate_up_silu_parity_case(rows, blocks_per_row)
            .expect("boundary Q4_K fused FFN parity fixture must execute");

        // Then tail guards preserve bit-exact eager semantics.
        assert_bitwise_equal(&fused, &eager);
    }
}

#[test]
#[ignore = "requires an H100 CUDA GPU"]
fn q4k_gate_up_silu_cuda_event_benchmark() {
    // Given the Mistral-7B 4096-to-11008 Q4_K SwiGLU model shape.
    let benchmark = run_q4k_gate_up_silu_cuda_event_benchmark(11_008, 16)
        .expect("direct CUDA-event FFN benchmark must execute");

    // When eager and fused paths are measured after three warmups with ten samples.
    assert_eq!(benchmark.eager_ms.len(), 10, "eager sample count changed");
    assert_eq!(benchmark.fused_ms.len(), 10, "fused sample count changed");

    // Then output stays bit-exact and both CUDA-event medians are emitted.
    assert_bitwise_equal(&benchmark.fused_output, &benchmark.eager_output);
    println!("eager_ffn_ms={:.6}", median(&benchmark.eager_ms));
    println!("fused_ffn_ms={:.6}", median(&benchmark.fused_ms));
}

#[test]
#[ignore = "requires an H100 CUDA GPU"]
fn q4k_gemv_block_size_sweep_is_exact_and_faster() {
    for (rows, blocks_per_row) in [
        (1024, 16),
        (4096, 16),
        (11_008, 16),
        (32_000, 16),
        (4096, 43),
    ] {
        let mut best = f64::INFINITY;
        let mut eager = 0.0;
        for block_size in [128, 256, 512, 1024] {
            let benchmark = run_q4k_gemv_block_size_cuda_event_benchmark(
                rows,
                blocks_per_row,
                block_size,
                false,
            )
            .expect("Q4_K block-size benchmark must execute");
            assert_bitwise_equal(&benchmark.candidate_output, &benchmark.eager_output);
            assert_eq!(benchmark.eager_ms.len(), 10);
            assert_eq!(benchmark.candidate_ms.len(), 10);
            eager = median(&benchmark.eager_ms);
            let candidate = median(&benchmark.candidate_ms);
            println!("q4k_gemv_{rows}x{blocks_per_row}_block_{block_size}_ms={candidate:.6}");
            best = best.min(candidate);
        }
        println!("eager_q4k_gemv_{rows}x{blocks_per_row}_ms={eager:.6}");
        assert!(
            best <= eager * 1.01,
            "rows={rows} bpr={blocks_per_row} best={best:.6}ms eager={eager:.6}ms"
        );
    }
}

#[test]
#[ignore = "requires an H100 CUDA GPU"]
fn q6k_gemv_block_size_sweep_is_exact() {
    for (rows, blocks_per_row) in [
        (1024, 16),
        (4096, 16),
        (11_008, 16),
        (32_000, 16),
        (4096, 43),
    ] {
        for block_size in [128, 256, 512, 1024] {
            let benchmark = run_q4k_gemv_block_size_cuda_event_benchmark(
                rows,
                blocks_per_row,
                block_size,
                true,
            )
            .expect("Q6_K block-size benchmark must execute");
            assert_bitwise_equal(&benchmark.candidate_output, &benchmark.eager_output);
            let eager = median(&benchmark.eager_ms);
            let candidate = median(&benchmark.candidate_ms);
            println!("q6k_gemv_{rows}x{blocks_per_row}_block_{block_size}_ms={candidate:.6}");
            println!("eager_q6k_gemv_{rows}x{blocks_per_row}_ms={eager:.6}");
        }
    }
}
