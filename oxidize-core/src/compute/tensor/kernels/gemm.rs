use super::*;

pub fn gemm_f32(
    left_matrix: &[f32],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[f32],
    cols: usize,
    output: &mut [f32],
) -> Result<(), GemmError> {
    let expected_left_len = rows.saturating_mul(shared_dim);
    if left_matrix.len() != expected_left_len {
        return Err(GemmError::InvalidLeftMatrixLength {
            expected: expected_left_len,
            actual: left_matrix.len(),
        });
    }

    let expected_right_len = shared_dim.saturating_mul(cols);
    if right_matrix.len() != expected_right_len {
        return Err(GemmError::InvalidRightMatrixLength {
            expected: expected_right_len,
            actual: right_matrix.len(),
        });
    }

    let expected_output_len = rows.saturating_mul(cols);
    if output.len() != expected_output_len {
        return Err(GemmError::InvalidOutputLength {
            expected: expected_output_len,
            actual: output.len(),
        });
    }

    #[cfg(feature = "cuda")]
    if crate::cuda::cuda_build_info().detected_at_build {
        return crate::cuda::gemm_f32_cuda(
            left_matrix,
            rows,
            shared_dim,
            right_matrix,
            cols,
            output,
        )
        .map_err(|err| GemmError::Cuda(format!("{err:?}")));
    }

    #[cfg(feature = "webgpu")]
    if crate::webgpu::should_use_webgpu_gemm(rows, shared_dim, cols) {
        crate::webgpu::validate_gemm_dims(
            left_matrix,
            rows,
            shared_dim,
            right_matrix,
            cols,
            output,
        )
        .map_err(|err| GemmError::WebGpu(format!("WebGPU GEMM validation failed: {err:?}")))?;
        gemm_f32_cpu(left_matrix, rows, shared_dim, right_matrix, cols, output);
        return Ok(());
    }

    #[cfg(feature = "metal")]
    if crate::metal::should_use_mps_gemm(rows, shared_dim, cols) {
        crate::metal::validate_gemm_dims(left_matrix, rows, shared_dim, right_matrix, cols, output)
            .map_err(|err| GemmError::Metal(format!("MPS GEMM validation failed: {err:?}")))?;
        gemm_f32_cpu(left_matrix, rows, shared_dim, right_matrix, cols, output);
        return Ok(());
    }

    let shard_count = choose_tensor_parallel_shard_count(shared_dim);
    if shard_count > 1 {
        gemm_f32_tensor_parallel(
            left_matrix,
            rows,
            shared_dim,
            right_matrix,
            cols,
            shard_count,
            output,
        )?;
        return Ok(());
    }

    gemm_f32_cpu(left_matrix, rows, shared_dim, right_matrix, cols, output);
    Ok(())
}

pub fn gemm_f32_tensor_parallel(
    left_matrix: &[f32],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[f32],
    cols: usize,
    shard_count: usize,
    output: &mut [f32],
) -> Result<(), GemmError> {
    validate_tensor_parallel_dims(left_matrix, rows, shared_dim, right_matrix, cols, output)?;
    if shard_count == 0 || !shared_dim.is_multiple_of(shard_count) {
        return Err(GemmError::InvalidTensorParallelShardCount {
            shared_dim,
            shard_count,
        });
    }
    if shard_count == 1 {
        gemm_f32_cpu(left_matrix, rows, shared_dim, right_matrix, cols, output);
        return Ok(());
    }

    let partials = std::thread::scope(|scope| {
        let mut jobs = Vec::with_capacity(shard_count);
        let chunk = shared_dim / shard_count;
        for shard_idx in 0..shard_count {
            let start_k = shard_idx * chunk;
            let end_k = start_k + chunk;
            jobs.push(scope.spawn(move || {
                let mut partial = vec![0.0_f32; rows * cols];
                for row in 0..rows {
                    let out_row = &mut partial[row * cols..(row + 1) * cols];
                    for k in start_k..end_k {
                        let left = left_matrix[row * shared_dim + k];
                        let right_row = &right_matrix[k * cols..(k + 1) * cols];
                        for (col, out_cell) in out_row.iter_mut().enumerate() {
                            *out_cell += left * right_row[col];
                        }
                    }
                }
                partial
            }));
        }
        jobs.into_iter()
            .map(|job| job.join().expect("tensor-parallel worker should not panic"))
            .collect::<Vec<_>>()
    });

    output.fill(0.0);
    for partial in partials {
        for (out, value) in output.iter_mut().zip(partial.iter()) {
            *out += *value;
        }
    }
    Ok(())
}

pub(super) fn validate_tensor_parallel_dims(
    left_matrix: &[f32],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[f32],
    cols: usize,
    output: &[f32],
) -> Result<(), GemmError> {
    let expected_left_len = rows.saturating_mul(shared_dim);
    if left_matrix.len() != expected_left_len {
        return Err(GemmError::InvalidLeftMatrixLength {
            expected: expected_left_len,
            actual: left_matrix.len(),
        });
    }

    let expected_right_len = shared_dim.saturating_mul(cols);
    if right_matrix.len() != expected_right_len {
        return Err(GemmError::InvalidRightMatrixLength {
            expected: expected_right_len,
            actual: right_matrix.len(),
        });
    }

    let expected_output_len = rows.saturating_mul(cols);
    if output.len() != expected_output_len {
        return Err(GemmError::InvalidOutputLength {
            expected: expected_output_len,
            actual: output.len(),
        });
    }
    Ok(())
}

pub(super) fn choose_tensor_parallel_shard_count(shared_dim: usize) -> usize {
    const MIN_SHARED_DIM_FOR_TP: usize = 1024;
    if shared_dim < MIN_SHARED_DIM_FOR_TP {
        return 1;
    }
    let max_threads = std::thread::available_parallelism().map_or(1, usize::from);
    let max_shards = max_threads.min(8).min(shared_dim);
    for shards in (2..=max_shards).rev() {
        if shared_dim.is_multiple_of(shards) {
            return shards;
        }
    }
    1
}

pub fn gemm_i8(
    left_matrix: &[i8],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[i8],
    cols: usize,
    output: &mut [i32],
) -> Result<(), GemmError> {
    let expected_left_len = rows.saturating_mul(shared_dim);
    if left_matrix.len() != expected_left_len {
        return Err(GemmError::InvalidLeftMatrixLength {
            expected: expected_left_len,
            actual: left_matrix.len(),
        });
    }

    let expected_right_len = shared_dim.saturating_mul(cols);
    if right_matrix.len() != expected_right_len {
        return Err(GemmError::InvalidRightMatrixLength {
            expected: expected_right_len,
            actual: right_matrix.len(),
        });
    }

    let expected_output_len = rows.saturating_mul(cols);
    if output.len() != expected_output_len {
        return Err(GemmError::InvalidOutputLength {
            expected: expected_output_len,
            actual: output.len(),
        });
    }

    gemm_i8_cpu(left_matrix, rows, shared_dim, right_matrix, cols, output);
    Ok(())
}

pub fn gemm_i4(
    left_matrix_packed: &[u8],
    rows: usize,
    shared_dim: usize,
    right_matrix_packed: &[u8],
    cols: usize,
    output: &mut [i32],
) -> Result<(), GemmError> {
    let expected_left_values = rows.saturating_mul(shared_dim);
    let expected_left_len = expected_left_values.div_ceil(2);
    if left_matrix_packed.len() != expected_left_len {
        return Err(GemmError::InvalidLeftMatrixLength {
            expected: expected_left_len,
            actual: left_matrix_packed.len(),
        });
    }

    let expected_right_values = shared_dim.saturating_mul(cols);
    let expected_right_len = expected_right_values.div_ceil(2);
    if right_matrix_packed.len() != expected_right_len {
        return Err(GemmError::InvalidRightMatrixLength {
            expected: expected_right_len,
            actual: right_matrix_packed.len(),
        });
    }

    let expected_output_len = rows.saturating_mul(cols);
    if output.len() != expected_output_len {
        return Err(GemmError::InvalidOutputLength {
            expected: expected_output_len,
            actual: output.len(),
        });
    }

    gemm_i4_cpu(
        left_matrix_packed,
        rows,
        shared_dim,
        right_matrix_packed,
        cols,
        output,
    );
    Ok(())
}

pub(super) fn gemv_f32_cpu(matrix: &[f32], cols: usize, vector: &[f32], output: &mut [f32]) {
    // dot_f32_fast (AVX2 FMA, independent accumulators) rather than a scalar
    // iterator sum: LLVM cannot vectorize the f32 reduction (non-associative),
    // leaving a 4-cycle-latency serial FMA chain. The MoE router GEMV runs
    // through here every layer of every token — measured ~24 ms/token of
    // main-thread stall on Qwen3-30B before this change.
    let rows = output.len();
    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        matrix
            .par_chunks_exact(cols)
            .zip(output.par_iter_mut())
            .for_each(|(row_values, out)| {
                *out = dot_f32_fast(row_values, &vector[..cols]);
            });
    } else {
        for (row_values, out) in matrix.chunks_exact(cols).zip(output.iter_mut()) {
            *out = dot_f32_fast(row_values, &vector[..cols]);
        }
    }
}

pub fn linear_activation_f32(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
    activation: ActivationFn,
    output: &mut [f32],
) -> Result<(), LinearActivationError> {
    let expected_matrix_len = rows.saturating_mul(cols);
    if matrix.len() != expected_matrix_len {
        return Err(LinearActivationError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(LinearActivationError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(LinearActivationError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    for (row_values, out) in matrix.chunks_exact(cols).zip(output.iter_mut()) {
        let linear = row_values
            .iter()
            .zip(vector.iter())
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
        *out = activate(linear, activation);
    }

    Ok(())
}

pub(super) fn activate(value: f32, activation: ActivationFn) -> f32 {
    match activation {
        ActivationFn::Relu => value.max(0.0),
        ActivationFn::Gelu => {
            let k = (2.0_f32 / std::f32::consts::PI).sqrt();
            0.5 * value * (1.0 + (k * (value + 0.044_715 * value.powi(3))).tanh())
        }
        ActivationFn::Silu => {
            let sigmoid = 1.0_f32 / (1.0 + (-value).exp());
            value * sigmoid
        }
    }
}

pub(super) fn gemm_f32_cpu(
    left_matrix: &[f32],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[f32],
    cols: usize,
    output: &mut [f32],
) {
    let expected_right_len = shared_dim.saturating_mul(cols);
    let mut right_transposed = vec![0.0_f32; expected_right_len];
    for shared_idx in 0..shared_dim {
        let row_start = shared_idx * cols;
        let row_end = row_start + cols;
        let right_row = &right_matrix[row_start..row_end];
        for (col, value) in right_row.iter().enumerate() {
            right_transposed[col * shared_dim + shared_idx] = *value;
        }
    }

    for row in 0..rows {
        let left_row = &left_matrix[row * shared_dim..(row + 1) * shared_dim];
        let out_row = &mut output[row * cols..(row + 1) * cols];
        for (col, out_cell) in out_row.iter_mut().enumerate() {
            // `right_transposed` makes each column contiguous, so this is a
            // plain dot product — dispatch to the SIMD kernel (AVX-512/AVX2)
            // instead of a scalar loop fenced by a `black_box` "prefetch" that
            // also blocked autovectorization.
            let right_col = &right_transposed[col * shared_dim..(col + 1) * shared_dim];
            *out_cell = crate::flash_attention::dot_product_f32(left_row, right_col);
        }
    }
}

pub(super) fn gemm_i8_cpu(
    left_matrix: &[i8],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[i8],
    cols: usize,
    output: &mut [i32],
) {
    let expected_right_len = shared_dim.saturating_mul(cols);
    let mut right_transposed = vec![0_i8; expected_right_len];
    for shared_idx in 0..shared_dim {
        let row_start = shared_idx * cols;
        let row_end = row_start + cols;
        let right_row = &right_matrix[row_start..row_end];
        for (col, value) in right_row.iter().enumerate() {
            right_transposed[col * shared_dim + shared_idx] = *value;
        }
    }

    for row in 0..rows {
        let left_row = &left_matrix[row * shared_dim..(row + 1) * shared_dim];
        let out_row = &mut output[row * cols..(row + 1) * cols];
        for (col, out_cell) in out_row.iter_mut().enumerate() {
            let right_col = &right_transposed[col * shared_dim..(col + 1) * shared_dim];
            *out_cell = left_row
                .iter()
                .zip(right_col.iter())
                .map(|(l, r)| i32::from(*l) * i32::from(*r))
                .sum();
        }
    }
}

pub(super) fn gemm_i4_cpu(
    left_matrix_packed: &[u8],
    rows: usize,
    shared_dim: usize,
    right_matrix_packed: &[u8],
    cols: usize,
    output: &mut [i32],
) {
    for row in 0..rows {
        let out_row = &mut output[row * cols..(row + 1) * cols];
        for (col, out_cell) in out_row.iter_mut().enumerate() {
            let mut sum = 0_i32;
            for k in 0..shared_dim {
                let left_idx = row * shared_dim + k;
                let right_idx = k * cols + col;
                sum += unpack_i4(left_matrix_packed, left_idx)
                    * unpack_i4(right_matrix_packed, right_idx);
            }
            *out_cell = sum;
        }
    }
}

pub(super) fn unpack_i4(packed: &[u8], value_index: usize) -> i32 {
    let byte = packed[value_index / 2];
    let nibble = if value_index.is_multiple_of(2) {
        byte & 0x0F
    } else {
        (byte >> 4) & 0x0F
    };
    i32::from(nibble as i8 - 8)
}
