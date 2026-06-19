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

#[allow(clippy::too_many_arguments, dead_code)]
pub(super) fn gemv_qk_f32_transposed(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
    block_size: usize,
    bits: usize,
    zero_point: f32,
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * block_size;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
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

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        let use_q4_avx512 = bits == 4 && q4_avx512_available();
        let use_q4_avx2 = bits == 4 && !use_q4_avx512 && q4_avx2_available();
        output
            .par_chunks_mut(TRANSPOSED_GEMV_COL_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let col_start = chunk_idx * TRANSPOSED_GEMV_COL_CHUNK;
                let block_start = col_start / QK_K;
                let block_end = ((col_start + out_chunk.len()).div_ceil(QK_K)).min(blocks_per_row);
                out_chunk.fill(0.0);
                for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
                    if vi == 0.0 {
                        continue;
                    }
                    let row_start = row_idx * blocks_per_row * block_size;
                    let row_blocks =
                        &quantized_matrix[row_start..row_start + (blocks_per_row * block_size)];
                    for block_idx in block_start..block_end {
                        let block_offset = block_idx * block_size;
                        let block = &row_blocks[block_offset..block_offset + block_size];
                        let d = f16_le_to_f32([block[0], block[1]]);
                        let factor = d * vi;
                        let local_col = block_idx * QK_K - col_start;
                        let block_output_len = (out_chunk.len() - local_col).min(QK_K);
                        let out_block = &mut out_chunk[local_col..local_col + block_output_len];
                        if bits == 4 && zero_point == 8.0 && block_output_len == QK_K {
                            accumulate_q4_block(
                                &block[2..],
                                factor,
                                out_block,
                                use_q4_avx512,
                                use_q4_avx2,
                            );
                        } else {
                            let bitstream = &block[2..];
                            for (idx, out) in out_block.iter_mut().enumerate() {
                                let q = extract_bits(bitstream, idx, bits) as f32;
                                *out += (q - zero_point) * factor;
                            }
                        }
                    }
                }
            });
    } else {
        output.fill(0.0);
        for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
            let row_start = row_idx * blocks_per_row * block_size;
            let row_blocks =
                &quantized_matrix[row_start..row_start + (blocks_per_row * block_size)];
            for (block_idx, block) in row_blocks.chunks_exact(block_size).enumerate() {
                let d = f16_le_to_f32([block[0], block[1]]);
                let bitstream = &block[2..];
                let col_start = block_idx * QK_K;
                for idx in 0..QK_K {
                    if col_start + idx >= cols {
                        break;
                    }
                    let q = extract_bits(bitstream, idx, bits) as f32;
                    output[col_start + idx] += (q - zero_point) * d * vi;
                }
            }
        }
    }
    Ok(())
}

pub(super) fn gemv_q4_k_f32_transposed(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_Q4_K_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
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

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_chunks_mut(TRANSPOSED_GEMV_COL_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let col_start = chunk_idx * TRANSPOSED_GEMV_COL_CHUNK;
                let block_start = col_start / QK_K;
                let block_end = ((col_start + out_chunk.len()).div_ceil(QK_K)).min(blocks_per_row);
                out_chunk.fill(0.0);
                for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
                    if vi == 0.0 {
                        continue;
                    }
                    let row_start = row_idx * blocks_per_row * BLOCK_Q4_K_SIZE;
                    let row_blocks = &quantized_matrix
                        [row_start..row_start + (blocks_per_row * BLOCK_Q4_K_SIZE)];
                    for block_idx in block_start..block_end {
                        let block_offset = block_idx * BLOCK_Q4_K_SIZE;
                        let block = &row_blocks[block_offset..block_offset + BLOCK_Q4_K_SIZE];
                        let local_col = block_idx * QK_K - col_start;
                        let block_output_len = (out_chunk.len() - local_col).min(QK_K);
                        let out_block = &mut out_chunk[local_col..local_col + block_output_len];
                        if block_output_len == QK_K {
                            accumulate_q4_k_block(block, vi, out_block);
                        } else {
                            for (idx, out) in out_block.iter_mut().enumerate() {
                                *out += q4_k_value(block, idx) * vi;
                            }
                        }
                    }
                }
            });
    } else {
        output.fill(0.0);
        for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
            if vi == 0.0 {
                continue;
            }
            let row_start = row_idx * blocks_per_row * BLOCK_Q4_K_SIZE;
            let row_blocks =
                &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_Q4_K_SIZE)];
            for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q4_K_SIZE).enumerate() {
                let col_start = block_idx * QK_K;
                for idx in 0..QK_K {
                    if col_start + idx >= cols {
                        break;
                    }
                    output[col_start + idx] += q4_k_value(block, idx) * vi;
                }
            }
        }
    }
    Ok(())
}

pub(super) fn gemv_q6_k_f32_transposed(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_Q6_K_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
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

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_chunks_mut(TRANSPOSED_GEMV_COL_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let col_start = chunk_idx * TRANSPOSED_GEMV_COL_CHUNK;
                let block_start = col_start / QK_K;
                let block_end = ((col_start + out_chunk.len()).div_ceil(QK_K)).min(blocks_per_row);
                out_chunk.fill(0.0);
                for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
                    if vi == 0.0 {
                        continue;
                    }
                    let row_start = row_idx * blocks_per_row * BLOCK_Q6_K_SIZE;
                    let row_blocks = &quantized_matrix
                        [row_start..row_start + (blocks_per_row * BLOCK_Q6_K_SIZE)];
                    for block_idx in block_start..block_end {
                        let block_offset = block_idx * BLOCK_Q6_K_SIZE;
                        let block = &row_blocks[block_offset..block_offset + BLOCK_Q6_K_SIZE];
                        let local_col = block_idx * QK_K - col_start;
                        let block_output_len = (out_chunk.len() - local_col).min(QK_K);
                        let out_block = &mut out_chunk[local_col..local_col + block_output_len];
                        if block_output_len == QK_K {
                            accumulate_q6_k_block(block, vi, out_block);
                        } else {
                            for (idx, out) in out_block.iter_mut().enumerate() {
                                *out += q6_k_value(block, idx) * vi;
                            }
                        }
                    }
                }
            });
    } else {
        output.fill(0.0);
        for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
            if vi == 0.0 {
                continue;
            }
            let row_start = row_idx * blocks_per_row * BLOCK_Q6_K_SIZE;
            let row_blocks =
                &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_Q6_K_SIZE)];
            for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q6_K_SIZE).enumerate() {
                let col_start = block_idx * QK_K;
                for idx in 0..QK_K {
                    if col_start + idx >= cols {
                        break;
                    }
                    output[col_start + idx] += q6_k_value(block, idx) * vi;
                }
            }
        }
    }
    Ok(())
}

#[inline]
#[allow(dead_code)]
pub(super) fn q4_avx2_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
#[allow(dead_code)]
pub(super) fn q4_avx512_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
#[allow(dead_code)]
pub(super) fn accumulate_q4_block(
    bitstream: &[u8],
    factor: f32,
    output: &mut [f32],
    use_avx512: bool,
    use_avx2: bool,
) {
    debug_assert!(bitstream.len() >= QK_K / 2);
    debug_assert!(output.len() >= QK_K);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_avx512 {
        unsafe {
            accumulate_q4_block_avx512(bitstream.as_ptr(), factor, output.as_mut_ptr());
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_avx2 {
        unsafe {
            accumulate_q4_block_avx2(bitstream.as_ptr(), factor, output.as_mut_ptr());
        }
        return;
    }

    for (pair_idx, &packed) in bitstream.iter().take(QK_K / 2).enumerate() {
        let out_idx = pair_idx * 2;
        output[out_idx] += ((packed & 0x0F) as f32 - 8.0) * factor;
        output[out_idx + 1] += ((packed >> 4) as f32 - 8.0) * factor;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(dead_code)]
pub(super) unsafe fn accumulate_q4_block_avx512(bitstream: *const u8, factor: f32, output: *mut f32) {
    let mask = _mm_set1_epi8(0x0F);
    let zero_point = _mm_set1_epi8(8);
    let factor = _mm512_set1_ps(factor);

    for byte_offset in (0..QK_K / 2).step_by(16) {
        let packed = unsafe { _mm_loadu_si128(bitstream.add(byte_offset).cast::<__m128i>()) };
        let low = _mm_and_si128(packed, mask);
        let high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
        let interleaved_lo = _mm_sub_epi8(_mm_unpacklo_epi8(low, high), zero_point);
        let interleaved_hi = _mm_sub_epi8(_mm_unpackhi_epi8(low, high), zero_point);

        let groups = [interleaved_lo, interleaved_hi];
        for (group_idx, group) in groups.into_iter().enumerate() {
            let q_i32 = _mm512_cvtepi8_epi32(group);
            let q_f32 = _mm512_cvtepi32_ps(q_i32);
            let out_ptr = unsafe { output.add(byte_offset * 2 + group_idx * 16) };
            let current = unsafe { _mm512_loadu_ps(out_ptr) };
            let updated = _mm512_add_ps(current, _mm512_mul_ps(q_f32, factor));
            unsafe { _mm512_storeu_ps(out_ptr, updated) };
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(dead_code)]
pub(super) unsafe fn accumulate_q4_block_avx2(bitstream: *const u8, factor: f32, output: *mut f32) {
    let mask = _mm_set1_epi8(0x0F);
    let zero_point = _mm_set1_epi8(8);
    let factor = _mm256_set1_ps(factor);

    for byte_offset in (0..QK_K / 2).step_by(16) {
        let packed = unsafe { _mm_loadu_si128(bitstream.add(byte_offset).cast::<__m128i>()) };
        let low = _mm_and_si128(packed, mask);
        let high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
        let interleaved_lo = _mm_sub_epi8(_mm_unpacklo_epi8(low, high), zero_point);
        let interleaved_hi = _mm_sub_epi8(_mm_unpackhi_epi8(low, high), zero_point);

        let groups = [
            interleaved_lo,
            _mm_srli_si128(interleaved_lo, 8),
            interleaved_hi,
            _mm_srli_si128(interleaved_hi, 8),
        ];
        for (group_idx, group) in groups.into_iter().enumerate() {
            let q_i32 = _mm256_cvtepi8_epi32(group);
            let q_f32 = _mm256_cvtepi32_ps(q_i32);
            let out_ptr = unsafe { output.add(byte_offset * 2 + group_idx * 8) };
            let current = unsafe { _mm256_loadu_ps(out_ptr) };
            let updated = _mm256_add_ps(current, _mm256_mul_ps(q_f32, factor));
            unsafe { _mm256_storeu_ps(out_ptr, updated) };
        }
    }
}

pub(super) fn gemv_q8_0_f32_transposed(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK8_0;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_Q8_0_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
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

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_chunks_mut(TRANSPOSED_GEMV_COL_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let col_start = chunk_idx * TRANSPOSED_GEMV_COL_CHUNK;
                let block_start = col_start / QK8_0;
                let block_end = ((col_start + out_chunk.len()).div_ceil(QK8_0)).min(blocks_per_row);
                out_chunk.fill(0.0);
                for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
                    if vi == 0.0 {
                        continue;
                    }
                    let row_start = row_idx * blocks_per_row * BLOCK_Q8_0_SIZE;
                    let row_blocks = &quantized_matrix
                        [row_start..row_start + (blocks_per_row * BLOCK_Q8_0_SIZE)];
                    for block_idx in block_start..block_end {
                        let block_offset = block_idx * BLOCK_Q8_0_SIZE;
                        let block = &row_blocks[block_offset..block_offset + BLOCK_Q8_0_SIZE];
                        let scale = f16_le_to_f32([block[0], block[1]]);
                        let factor = scale * vi;
                        let local_col = block_idx * QK8_0 - col_start;
                        let block_output_len = (out_chunk.len() - local_col).min(QK8_0);
                        let out_block = &mut out_chunk[local_col..local_col + block_output_len];
                        for (out, q_byte) in out_block.iter_mut().zip(block[2..].iter()) {
                            *out += (*q_byte as i8) as f32 * factor;
                        }
                    }
                }
            });
    } else {
        output.fill(0.0);
        for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
            let row_start = row_idx * blocks_per_row * BLOCK_Q8_0_SIZE;
            let row_blocks =
                &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_Q8_0_SIZE)];
            for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q8_0_SIZE).enumerate() {
                let scale = f16_le_to_f32([block[0], block[1]]);
                let col_start = block_idx * QK8_0;
                for (idx, q_byte) in block[2..].iter().enumerate() {
                    if col_start + idx >= cols {
                        break;
                    }
                    output[col_start + idx] += (*q_byte as i8) as f32 * scale * vi;
                }
            }
        }
    }
    Ok(())
}

pub fn gemv_quantized_f32_transposed(
    quantization: GgufQuantizationType,
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    match quantization {
        GgufQuantizationType::Q8_0 => {
            gemv_q8_0_f32_transposed(quantized_matrix, rows, cols, vector, output)
        }
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => {
            gemv_q4_k_f32_transposed(quantized_matrix, rows, cols, vector, output)
        }
        GgufQuantizationType::Q6_K => {
            gemv_q6_k_f32_transposed(quantized_matrix, rows, cols, vector, output)
        }
        _ => Err(GemvError::UnsupportedQuantizationType { quantization }),
    }
}
