use super::{benchmark_flash_decode_case, run_flash_decode_parity_case};

const PARITY_TOLERANCE: f32 = 2.0e-5;

fn assert_lane_parity(actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "CUDA and reference lane counts differ"
    );
    for (lane, (&actual_lane, &expected_lane)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual_lane.is_finite(),
            "CUDA lane {lane} is not finite: {actual_lane}"
        );
        assert!(
            expected_lane.is_finite(),
            "reference lane {lane} is not finite: {expected_lane}"
        );
        assert!(
            (actual_lane - expected_lane).abs() <= PARITY_TOLERANCE,
            "lane {lane} differs: CUDA={actual_lane}, reference={expected_lane}, tolerance={PARITY_TOLERANCE}"
        );
    }
}

#[test]
fn split_k_decode_matches_dense_reference_for_gqa_tail_and_base_row() {
    // Given grouped-query attention, an uneven split tail, and a nonzero cache base row.
    let (query_heads, kv_heads, head_dim, seq_len, base_row) = (8, 2, 64, 1025, 3);

    // When the real split-K CUDA decode is compared with dense CPU attention.
    let (actual, expected) =
        run_flash_decode_parity_case(query_heads, kv_heads, head_dim, seq_len, base_row, false)
            .expect("CUDA parity fixture must execute");

    // Then every output lane is finite and agrees within the f32 kernel tolerance.
    assert_lane_parity(&actual, &expected);
}

#[test]
fn split_k_decode_matches_dense_reference_for_extreme_scores() {
    // Given attention inputs producing extreme positive and negative scores.
    let (query_heads, kv_heads, head_dim, seq_len, base_row) = (8, 2, 64, 1025, 3);

    // When the stabilized split-K CUDA path is compared with the dense reference.
    let (actual, expected) =
        run_flash_decode_parity_case(query_heads, kv_heads, head_dim, seq_len, base_row, true)
            .expect("extreme-score CUDA parity fixture must execute");

    // Then every lane remains finite and matches the stable dense result.
    assert_lane_parity(&actual, &expected);
}

fn sorted_median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

#[test]
#[ignore = "requires an H100-class CUDA GPU"]
fn split_k_h100_benchmark_reports_long_context_speedup() {
    // Given representative H100 decode shapes at two long context lengths.
    let (query_heads, kv_heads, head_dim, warmup_count, sample_count) = (32, 8, 128, 3, 10);

    for seq_len in [8192, 32768] {
        // When legacy and split-K kernels are measured after warmup.
        let (split_count, legacy_ms, split_ms) = benchmark_flash_decode_case(
            query_heads,
            kv_heads,
            head_dim,
            seq_len,
            warmup_count,
            sample_count,
        )
        .expect("H100 flash-decode benchmark must execute");
        let legacy_median_ms = sorted_median(&legacy_ms);
        let split_median_ms = sorted_median(&split_ms);
        println!(
            "seq_len={seq_len} split_count={split_count} legacy_median_ms={legacy_median_ms:.4} split_median_ms={split_median_ms:.4}"
        );

        // Then split-K uses multiple partitions and has a lower finite positive median.
        assert!(
            split_count > 1,
            "seq_len={seq_len} must use multiple KV splits"
        );
        assert_eq!(legacy_ms.len(), sample_count);
        assert_eq!(split_ms.len(), sample_count);
        assert!(
            legacy_ms
                .iter()
                .chain(&split_ms)
                .all(|sample| sample.is_finite() && *sample > 0.0),
            "seq_len={seq_len} timings must all be finite and positive"
        );
        assert!(
            split_median_ms < legacy_median_ms,
            "seq_len={seq_len}: split-K median {split_median_ms:.4} ms must beat legacy {legacy_median_ms:.4} ms"
        );
    }
}
