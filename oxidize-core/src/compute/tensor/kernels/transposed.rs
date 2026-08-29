use super::*;

pub(super) fn gemv_f32_transposed_cpu(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) {
    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_chunks_mut(TRANSPOSED_GEMV_COL_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let col_start = chunk_idx * TRANSPOSED_GEMV_COL_CHUNK;
                out_chunk.fill(0.0);
                let col_end = col_start + out_chunk.len();
                for (row_values, &vi) in matrix.chunks_exact(cols).zip(vector.iter()).take(rows) {
                    let row_chunk = &row_values[col_start..col_end];
                    for (out, &weight) in out_chunk.iter_mut().zip(row_chunk) {
                        *out += weight * vi;
                    }
                }
            });
    } else {
        output.fill(0.0);
        for (row_values, &vi) in matrix.chunks_exact(cols).zip(vector.iter()).take(rows) {
            for (j, &weight) in row_values.iter().enumerate() {
                output[j] += weight * vi;
            }
        }
    }
}

pub fn gemv_f32_transposed(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let expected_matrix_len = rows.saturating_mul(cols);
    if matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: matrix.len(),
        });
    }
    if vector.len() != rows {
        return Err(GemvError::InvalidVectorLength {
            expected: rows,
            actual: vector.len(),
        });
    }
    if output.len() != cols {
        return Err(GemvError::InvalidOutputLength {
            expected: cols,
            actual: output.len(),
        });
    }
    #[cfg(feature = "cuda")]
    if crate::cuda::cuda_build_info().detected_at_build {
        return crate::cuda::gemv_f32_transposed_cuda(matrix, rows, cols, vector, output)
            .map_err(|err| GemvError::Cuda(format!("{err:?}")));
    }
    gemv_f32_transposed_cpu(matrix, rows, cols, vector, output);
    Ok(())
}
