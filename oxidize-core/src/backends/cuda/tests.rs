use super::*;

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
fn split_k_plan_selects_legacy_below_threshold() {
    // Given a decode just below the split-K context threshold.
    let seq_len = 1023;

    // When launch planning runs for an H100-sized GPU.
    let plan = SplitKPlan::select(132, 32, seq_len);

    // Then the legacy one-block-per-query-head kernel remains selected.
    assert_eq!(plan, None, "seq_len=1023 must select legacy decode");
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
