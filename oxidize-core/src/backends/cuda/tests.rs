use super::*;

#[test]
fn q4k_gate_up_silu_kernel_source_contract_exists() {
    // Given the checked-in CUDA source used to build the runtime PTX.
    let source = include_str!("../../../kernels/gemv_f32.cu");

    // When the existing eager and fused FFN entry points are inspected.
    let required_entries = [
        "gemv_q4k_f32in_kernel",
        "silu_mul_f32_kernel",
        "gemv_q4k_f32in_gate_up_silu_kernel",
    ];

    // Then all three kernels required by the runtime parity fixture are present.
    for entry in required_entries {
        assert!(source.contains(entry), "missing CUDA entry point: {entry}");
    }
}

#[test]
fn projection_residual_fusion_kernel_source_contract_exists() {
    let source = include_str!("../../../kernels/gemv_f32.cu");
    for entry in [
        "gemv_q4k_f32in_residual_kernel",
        "gemv_q6k_f32in_residual_kernel",
    ] {
        assert!(source.contains(entry), "missing CUDA entry point: {entry}");
    }
}

#[test]
fn ffn_fusion_selection_uses_supported_q4k_geometry() {
    // Given valid production Q4_K gate/up geometry without an opt-in override.
    let selected = super::gpu_native_forward::select_ffn_fusion(true, true, 11_008, 16, true);

    // When the default production selector is evaluated, then fusion is selected.
    assert!(
        selected,
        "supported Q4_K FFN geometry must select fusion by default"
    );
}

#[test]
fn ffn_fusion_selection_falls_back_for_invalid_geometry() {
    // Given an opt-in override with a malformed zero-row geometry.
    let selected = super::gpu_native_forward::select_ffn_fusion(true, true, 0, 16, true);

    // When the production selector validates the shape, then it retains eager fallback.
    assert!(
        !selected,
        "invalid Q4_K FFN geometry must retain eager fallback"
    );
}

#[test]
fn q4k_projection_selector_uses_h100_tuned_block() {
    assert_eq!(
        super::gpu_native_forward::select_projection_gemv_block_size(true, 4096, None),
        1024
    );
}

#[test]
fn q6k_projection_selector_retains_legacy_block() {
    assert_eq!(
        super::gpu_native_forward::select_projection_gemv_block_size(false, 4096, Some(1024)),
        256
    );
}

#[test]
fn malformed_projection_block_override_uses_tuned_default() {
    assert_eq!(
        super::gpu_native_forward::select_projection_gemv_block_size(true, 32_000, Some(7)),
        128
    );
}

#[test]
fn q4k_projection_selector_uses_shape_specific_blocks() {
    use super::gpu_native_forward::select_projection_gemv_block_size;
    assert_eq!(select_projection_gemv_block_size(true, 1024, None), 128);
    assert_eq!(select_projection_gemv_block_size(true, 4096, None), 1024);
    assert_eq!(select_projection_gemv_block_size(true, 11_008, None), 128);
    assert_eq!(select_projection_gemv_block_size(true, 32_000, None), 128);
}

#[test]
fn allocates_cpu_buffer_with_requested_size() {
    let buffer = DeviceBuffer::allocate(MemoryDevice::Cpu, 32).expect("buffer allocates");
    assert_eq!(buffer.device(), MemoryDevice::Cpu);
    assert_eq!(buffer.len(), 32);
}

#[test]
fn supports_host_to_device_and_back_for_cpu_buffer() {
    let mut buffer = DeviceBuffer::allocate(MemoryDevice::Cpu, 4).expect("buffer allocates");
    let input = [1_u8, 2, 3, 4];
    buffer
        .copy_from_host(&input)
        .expect("host to device copy succeeds");

    let mut output = [0_u8; 4];
    buffer
        .copy_to_host(&mut output)
        .expect("device to host copy succeeds");
    assert_eq!(output, input);
}

#[test]
fn rejects_mismatched_transfer_lengths() {
    let mut buffer = DeviceBuffer::allocate(MemoryDevice::Cpu, 3).expect("buffer allocates");

    let h2d_error = buffer
        .copy_from_host(&[1_u8, 2])
        .expect_err("h2d mismatch should fail");
    assert_eq!(
        h2d_error,
        MemoryError::SizeMismatch {
            expected: 3,
            actual: 2
        }
    );

    let mut host = [0_u8; 2];
    let d2h_error = buffer
        .copy_to_host(&mut host)
        .expect_err("d2h mismatch should fail");
    assert_eq!(
        d2h_error,
        MemoryError::SizeMismatch {
            expected: 3,
            actual: 2
        }
    );
}

#[test]
fn validates_gemv_cuda_dimensions() {
    let matrix = [1.0_f32, 2.0, 3.0, 4.0];
    let vector = [1.0_f32, 1.0];
    let output = [0.0_f32; 2];
    validate_gemv_dims(&matrix, 2, 2, &vector, &output).expect("dimensions should be valid");
}

#[test]
fn rejects_gemv_cuda_dimension_mismatch() {
    let err = validate_gemv_dims(&[1.0_f32, 2.0, 3.0], 2, 2, &[1.0_f32, 1.0], &[0.0_f32; 2])
        .expect_err("matrix size mismatch should fail");
    assert!(matches!(err, GemvCudaError::InvalidMatrixLength { .. }));
}

#[test]
fn validates_q8_0_gemv_cuda_dimensions() {
    let rows = 2;
    let cols = 32;
    let matrix = vec![0_u8; rows * BLOCK_Q8_0_SIZE];
    let vector = vec![1.0_f32; cols];
    let output = vec![0.0_f32; rows];
    validate_q8_0_gemv_dims(&matrix, rows, cols, &vector, &output)
        .expect("dimensions should be valid");
}

#[test]
fn rejects_q8_0_gemv_cols_not_aligned() {
    let rows = 1;
    let cols = 31;
    let matrix = vec![0_u8; BLOCK_Q8_0_SIZE];
    let vector = vec![1.0_f32; cols];
    let output = vec![0.0_f32; rows];
    let err = validate_q8_0_gemv_dims(&matrix, rows, cols, &vector, &output)
        .expect_err("non-aligned columns should fail");
    assert!(matches!(err, GemvCudaError::InvalidVectorLength { .. }));
}

#[test]
fn validates_gemm_cuda_dimensions() {
    let left = [1.0_f32, 2.0, 3.0, 4.0];
    let right = [1.0_f32, 2.0, 3.0, 4.0];
    let output = [0.0_f32; 4];
    validate_gemm_dims(&left, 2, 2, &right, 2, &output).expect("dimensions should be valid");
}

#[test]
fn rejects_gemm_cuda_dimension_mismatch() {
    let err = validate_gemm_dims(
        &[1.0_f32, 2.0, 3.0],
        2,
        2,
        &[1.0_f32, 2.0, 3.0, 4.0],
        2,
        &[0.0_f32; 4],
    )
    .expect_err("left matrix size mismatch should fail");
    assert!(matches!(err, GemmCudaError::InvalidLeftMatrixLength { .. }));
}

#[test]
#[cfg(feature = "cuda")]
fn gemv_cuda_kernel_name_matches_ptx_entry() {
    assert!(GEMV_F32_PTX.contains(".entry gemv_f32_kernel"));
    assert!(GEMV_F32_PTX.contains(".entry gemv_q4_k_kernel"));
    assert!(GEMV_F32_PTX.contains(".entry flash_attn_decode_splitk_kernel"));
    assert!(GEMV_F32_PTX.contains(".entry flash_attn_decode_reduce_kernel"));
    assert_eq!(GEMV_KERNEL_NAME, "gemv_f32_kernel");
    assert_eq!(GEMV_Q4_K_DIRECT_KERNEL_NAME, "gemv_q4_k_kernel");
}

#[test]
fn split_k_plan_uses_occupancy_splits_for_short_context_on_h100() {
    // Decode benches often have context < 1024; occupancy-only splits still help.
    let plan = SplitKPlan::select(132, 32, 64).expect("H100 short ctx must split-K");
    assert_eq!(plan.split_count, 9, "132 SM / 32 heads → 9 occupancy splits");
    assert_eq!(plan.block_count, 288);
}

#[test]
fn split_k_plan_selects_legacy_on_tiny_gpu() {
    let plan = SplitKPlan::select(1, 32, 1023);
    assert_eq!(plan, None, "single-SM GPU must retain legacy decode");
}

#[test]
fn split_k_plan_fills_h100_for_long_decode() {
    // Given 32 query heads decoding 2,304 cached tokens on a 132-SM H100.
    let (sm_count, query_heads, seq_len) = (132, 32, 2304);

    // When split-K launch planning runs.
    let plan = SplitKPlan::select(sm_count, query_heads, seq_len)
        .expect("seq_len=2304 must select split-K decode");

    // Then nine splits expose 288 independently schedulable CUDA blocks.
    assert_eq!(plan.split_count, 9, "H100 decode must use nine KV splits");
    assert_eq!(
        plan.block_count, 288,
        "32 heads * 9 splits must launch 288 blocks"
    );
}

#[test]
fn split_k_plan_caps_splits_by_sequence_length() {
    let plan = SplitKPlan::select(132, 32, 5).expect("seq_len=5 allows 5 splits");
    assert_eq!(plan.split_count, 5);
    assert_eq!(SplitKPlan::select(132, 32, 1), None);
}

#[test]
fn split_k_scratch_lengths_reject_zero_and_overflow() {
    assert_eq!(flash_decode_scratch_lengths(0, 2, 64), None);
    assert_eq!(flash_decode_scratch_lengths(2, 0, 64), None);
    assert_eq!(flash_decode_scratch_lengths(2, 2, 0), None);
    assert_eq!(flash_decode_scratch_lengths(usize::MAX, 2, 64), None);
    assert_eq!(flash_decode_scratch_lengths(2, 2, usize::MAX), None);
    assert_eq!(flash_decode_scratch_lengths(8, 7, 64), Some((56, 3584)));
}

#[test]
fn split_k_dispatch_selection_honors_overrides_and_boundaries() {
    assert_eq!(
        select_split_count_for_test(132, 32, 2304, true, Some("9")),
        1
    );
    assert_eq!(
        select_split_count_for_test(132, 32, 2304, false, Some("bad")),
        9
    );
    assert_eq!(
        select_split_count_for_test(132, 32, 2304, false, Some("0")),
        1
    );
    assert_eq!(
        select_split_count_for_test(132, 32, 7, false, Some("99")),
        7
    );
    assert_eq!(select_split_count_for_test(1, 32, 2304, false, None), 1);
}

#[test]
fn flash_decode_launch_contract_accepts_valid_gqa_split_k_shape() {
    // Given a valid two-layer GQA cache and a bounded split-K decode window.
    let result = validate_flash_decode_launch(2, 1, 4096, 8 * 128, 1025, 7, 32, 8, 128, 7);

    // When and then the complete launch boundary is validated.
    assert_eq!(result, Ok(()));
}

#[test]
fn flash_decode_launch_contract_rejects_zero_dimensions() {
    // Given each required launch dimension set to zero in turn.
    let cases = [
        (0, 0, 4096, 1024, 1025, 7, 32, 8, 128, 7),
        (2, 1, 0, 1024, 1025, 7, 32, 8, 128, 7),
        (2, 1, 4096, 0, 1025, 7, 32, 8, 128, 7),
        (2, 1, 4096, 1024, 0, 7, 32, 8, 128, 7),
        (2, 1, 4096, 1024, 1025, 7, 0, 8, 128, 7),
        (2, 1, 4096, 1024, 1025, 7, 32, 0, 128, 7),
        (2, 1, 4096, 1024, 1025, 7, 32, 8, 0, 7),
        (2, 1, 4096, 1024, 1025, 7, 32, 8, 128, 0),
    ];

    // When each malformed shape crosses the boundary, then it is rejected.
    for case in cases {
        let error = validate_flash_decode_launch(
            case.0, case.1, case.2, case.3, case.4, case.5, case.6, case.7, case.8, case.9,
        )
        .expect_err("zero launch fields must be rejected");
        assert!(
            error.contains("must be nonzero"),
            "unexpected error: {error}"
        );
    }
}

#[test]
fn flash_decode_launch_contract_rejects_invalid_attention_geometry() {
    // Given malformed GQA, unsupported head widths, and an over-partitioned decode.
    let cases = [
        (2, 1, 4096, 8 * 128, 1025, 7, 30, 8, 128, 7, "divisible"),
        (2, 1, 4096, 8 * 96, 1025, 7, 32, 8, 96, 7, "power of two"),
        (2, 1, 4096, 8 * 512, 1025, 7, 32, 8, 512, 7, "256"),
        (2, 1, 4096, 8 * 128, 7, 7, 32, 8, 128, 8, "sequence"),
    ];

    // When launch validation runs, then each distinct contract violation is named.
    for case in cases {
        let error = validate_flash_decode_launch(
            case.0, case.1, case.2, case.3, case.4, case.5, case.6, case.7, case.8, case.9,
        )
        .expect_err("invalid attention geometry must be rejected");
        assert!(error.contains(case.10), "unexpected error: {error}");
    }
}

#[test]
fn flash_decode_launch_contract_rejects_cache_shape_and_bounds() {
    // Given an invalid layer, inconsistent KV width, an out-of-context window,
    // and a checked-add overflow at the cache-row boundary.
    let cases = [
        (1, 1, 4096, 1024, 1025, 7, 32, 8, 128, 7, "layer"),
        (2, 1, 4096, 1023, 1025, 7, 32, 8, 128, 7, "KV width"),
        (2, 1, 1024, 1024, 1025, 7, 32, 8, 128, 7, "context"),
        (
            2,
            1,
            usize::MAX,
            1024,
            2,
            usize::MAX,
            32,
            8,
            128,
            2,
            "overflow",
        ),
    ];

    // When launch validation runs, then no malformed cache address reaches CUDA.
    for case in cases {
        let error = validate_flash_decode_launch(
            case.0, case.1, case.2, case.3, case.4, case.5, case.6, case.7, case.8, case.9,
        )
        .expect_err("invalid cache shape or bounds must be rejected");
        assert!(error.contains(case.10), "unexpected error: {error}");
    }
}

#[test]
fn split_k_ranges_exactly_partition_uneven_context() {
    // Given a context that does not divide evenly across five splits.
    let (seq_len, split_count) = (1025, 5);

    // When every split range is constructed.
    let ranges: Vec<_> = (0..split_count)
        .map(|split_idx| split_k_range(seq_len, split_count, split_idx))
        .collect();

    // Then all indices are covered exactly once, with no empty range.
    assert_eq!(ranges.first().map(|range| range.start), Some(0));
    assert_eq!(ranges.last().map(|range| range.end), Some(seq_len));
    assert!(ranges.iter().all(|range| range.start < range.end));
    assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
    assert_eq!(
        ranges.iter().map(|range| range.len()).sum::<usize>(),
        seq_len
    );
}

#[test]
fn split_k_merge_matches_dense_softmax_with_extreme_logits() {
    // Given deterministic split-local online-softmax states, including logits
    // large enough to overflow an unstabilized exponential.
    let logits = [1000.0_f32, 999.0, -1000.0, 998.0, 997.0];
    let values = [
        [1.0_f32, -2.0],
        [3.0, 4.0],
        [20.0, 30.0],
        [-1.0, 2.0],
        [5.0, -3.0],
    ];
    let partials: Vec<(f32, f32, Vec<f32>)> = [(0, 2), (2, 5)]
        .into_iter()
        .map(|(start, end)| {
            let max = logits[start..end]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let denominator = logits[start..end]
                .iter()
                .map(|logit| (*logit - max).exp())
                .sum();
            let numerator = (start..end).fold(vec![0.0_f32; 2], |mut sum, idx| {
                let weight = (logits[idx] - max).exp();
                sum[0] += weight * values[idx][0];
                sum[1] += weight * values[idx][1];
                sum
            });
            (max, denominator, numerator)
        })
        .collect();

    // When split-local states are merged.
    let actual = merge_softmax_partials_for_test(&partials);

    // Then the result matches a stable dense softmax reference.
    let global_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denominator: f32 = logits.iter().map(|logit| (*logit - global_max).exp()).sum();
    let expected = (0..logits.len()).fold([0.0_f32; 2], |mut sum, idx| {
        let weight = (logits[idx] - global_max).exp() / denominator;
        sum[0] += weight * values[idx][0];
        sum[1] += weight * values[idx][1];
        sum
    });
    assert_eq!(actual.len(), expected.len());
    for (component, reference) in actual.iter().zip(expected) {
        assert!(
            (component - reference).abs() < 1.0e-5,
            "merged={component}, dense={reference}"
        );
    }
}

#[test]
fn split_k_cuda_source_defines_decode_and_reduce_entries() {
    // Given the CUDA kernel source compiled into the Oxidize PTX module.
    let source = include_str!("../../../kernels/gemv_f32.cu");

    // When split-K decode entry points are inspected.
    let entries = [
        "flash_attn_decode_splitk_kernel",
        "flash_attn_decode_reduce_kernel",
    ];

    // Then both the partial decode and reduction kernels are defined.
    for entry in entries {
        assert!(
            source.contains(entry),
            "CUDA source is missing extern entry {entry}"
        );
    }
}

#[test]
fn split_k_profile_surface_characterizes_current_phase_one_entries() {
    // Given the checked-in CUDA module and Modal GPU action surface.
    let cuda_module = include_str!("../cuda.rs");
    let modal_app = include_str!("../../../../modal_app.py");

    // When the completed Phase 1 split-K surface is inspected.
    let required_entries = [
        "cuda/flash_decode.rs",
        "cuda/flash_decode_launch.rs",
        "gpu_splitk_bench",
        "gpu_splitk_test",
    ];

    // Then both production modules and their real-GPU validation actions remain present.
    for entry in required_entries {
        assert!(
            cuda_module.contains(entry) || modal_app.contains(entry),
            "current split-K profiling surface is missing {entry}"
        );
    }
}

#[test]
fn profile_summary_requires_direct_ffn_and_cli_wall_totals() {
    // Given the real Modal runner used to choose the next GPU optimization.
    let modal_app = include_str!("../../../../modal_app.py");

    // When its next-speed profiling contract is inspected.
    let required_contract = [
        ("def gpu_next_profile", "gpu_next_profile callable"),
        ("gpu-next-profile", "gpu-next-profile action"),
        ("eager_ffn_ms", "direct eager FFN median"),
        ("fused_ffn_ms", "direct fused FFN median"),
        ("token_wall_off_ms", "unfused CLI token-wall median"),
        ("token_wall_on_ms", "fused CLI token-wall median"),
        ("token_wall_speedup", "CLI speedup"),
    ];

    // Then every mutually checkable time category must be emitted by that action.
    for (marker, category) in required_contract {
        assert!(
            modal_app.contains(marker),
            "gpu_next_profile summary is missing required {category} marker `{marker}`"
        );
    }
}

#[test]
fn gpu_next_profile_avoids_unstable_nsys_trace() {
    let modal_app = include_str!("../../../../modal_app.py");
    let gpu_next_profile = modal_app
        .split("def gpu_next_profile(")
        .nth(1)
        .and_then(|tail| tail.split("\n@app.function").next())
        .expect("gpu_next_profile must remain present in modal_app.py");

    assert!(
        !gpu_next_profile.contains("nsys"),
        "gpu_next_profile must avoid Nsight"
    );
}

#[test]
fn gpu_next_profile_falls_back_from_broken_nsys_to_direct_cuda_events() {
    // Given Nsight profiling proved unreliable on the target H100 image.
    let modal_app = include_str!("../../../../modal_app.py");
    let gpu_next_profile = modal_app
        .split("def gpu_next_profile(")
        .nth(1)
        .and_then(|tail| tail.split("\n@app.function").next())
        .expect("gpu_next_profile must remain present in modal_app.py");
    let fixture = include_str!("ffn_fusion_cuda_fixture.rs");
    let cuda_tests = include_str!("ffn_fusion_cuda_tests.rs");

    // When the replacement profile surface is inspected, then it must use the
    // direct CUDA-event benchmark plus a real CLI OFF/ON parity comparison.
    for marker in [
        "q4k_gate_up_silu_cuda_event_benchmark",
        "eager_ffn_ms",
        "fused_ffn_ms",
        "token_wall_off_ms",
        "token_wall_on_ms",
        "token_wall_speedup",
        "greedy_output_parity=PASS",
        "greedy_logit_trace_parity=PASS",
    ] {
        assert!(
            gpu_next_profile.contains(marker)
                || fixture.contains(marker)
                || cuda_tests.contains(marker),
            "direct-event fallback is missing required marker `{marker}`"
        );
    }
    assert!(
        !gpu_next_profile.contains("nsys"),
        "gpu_next_profile must not invoke broken Nsight profiling"
    );
    assert!(
        gpu_next_profile.contains("range(13)") && gpu_next_profile.contains("iteration >= 3"),
        "real CLI A/B must execute three warmups and ten measured runs per variant"
    );
}

#[test]
fn gpu_next_profile_extracts_real_cli_generation_output() {
    // Given the real CLI streams generated text between its offload-plan and
    // generation-stats records rather than prefixing it with `oxidize:`.
    let modal_app = include_str!("../../../../modal_app.py");
    let gpu_next_profile = modal_app
        .split("def gpu_next_profile(")
        .nth(1)
        .and_then(|tail| tail.split("\n@app.function").next())
        .expect("gpu_next_profile must remain present in modal_app.py");

    // Then the profile harness must parse that bounded CLI output region and
    // retain fail-closed diagnostics for a missing trace or generated output.
    for marker in [
        r#"offload plan:.*?\n(.*?)\ngeneration stats:"#,
        "profile output tail:",
        "profile run omitted greedy logit trace or output",
    ] {
        assert!(
            gpu_next_profile.contains(marker),
            "gpu_next_profile must extract real CLI output; missing `{marker}`"
        );
    }
    assert!(
        !gpu_next_profile.contains(r#"re.findall(r"oxidize: (.*)", blob)"#),
        "gpu_next_profile must not rely on the obsolete `oxidize:` prefix"
    );
}

#[test]
fn gpu_next_profile_reuses_resolved_local_model_for_repeated_trials() {
    // Given repeated CLI measurements must not depend on a remote Hugging Face
    // request succeeding once per trial.
    let modal_app = include_str!("../../../../modal_app.py");
    let gpu_next_profile = modal_app
        .split("def gpu_next_profile(")
        .nth(1)
        .and_then(|tail| tail.split("\n@app.function").next())
        .expect("gpu_next_profile must remain present in modal_app.py");
    let local_model = concat!(
        "/root/.cache/oxidize/hf/",
        "bartowski-Mistral-7B-Instruct-v0-3-GGUF/main/",
        "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
    );

    // Then the six measured subprocesses share one verified local GGUF path,
    // and their command does not invoke repository/file resolution.
    assert!(
        gpu_next_profile.contains(local_model),
        "gpu_next_profile must use the deterministic resolved local model path `{local_model}`"
    );
    assert!(
        gpu_next_profile.contains("if not os.path.isfile(local_model)"),
        "gpu_next_profile must resolve the model once before measured trials"
    );
    let command = gpu_next_profile
        .split("command = [")
        .nth(1)
        .and_then(|tail| tail.split("\n    ]").next())
        .expect("gpu_next_profile must define its measured CLI command");
    assert!(
        command.contains("local_model"),
        "measured CLI command must use the verified local GGUF path"
    );
    assert!(
        !command.contains("\n        model,") && !command.contains(r#""--file""#),
        "measured CLI command must not query Hugging Face or use --file"
    );
}

#[test]
fn modal_harness_exposes_real_dflash_cli_ab() {
    let source = include_str!("../../../../modal_app.py");
    for marker in [
        "def gpu_dflash_ab(",
        "--draft-model",
        "Qwen3-4B-DFlash-q8_0.gguf",
        "dflash_output_parity=PASS",
    ] {
        assert!(source.contains(marker), "missing DFlash A/B marker: {marker}");
    }
}
