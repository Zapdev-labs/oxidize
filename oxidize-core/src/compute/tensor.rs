use crate::gguf::GgufQuantizationType;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const QK8_0: usize = 32;
const BLOCK_Q8_0_SIZE: usize = 2 + QK8_0;
const QK_K: usize = 256;
const BLOCK_Q4_K_SIZE: usize = 2 * std::mem::size_of::<u16>() + 12 + QK_K / 2;
const BLOCK_Q6_K_SIZE: usize = std::mem::size_of::<u16>() + QK_K / 16 + 3 * QK_K / 4;
const FLASH_ATTENTION_BLOCK_TOKENS: usize = 64;
const PARALLEL_GEMV_MIN_OPS: usize = 1 << 20;
const TRANSPOSED_GEMV_COL_CHUNK: usize = QK_K;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    I8,
    I16,
    I32,
    I64,
}

impl DType {
    /// Return the size of a single element in bytes.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemvError {
    InvalidMatrixLength {
        expected: usize,
        actual: usize,
    },
    InvalidVectorLength {
        expected: usize,
        actual: usize,
    },
    InvalidOutputLength {
        expected: usize,
        actual: usize,
    },
    UnsupportedQuantizationType {
        quantization: GgufQuantizationType,
    },
    #[cfg(feature = "cuda")]
    Cuda(String),
    #[cfg(feature = "metal")]
    Metal(String),
    #[cfg(feature = "webgpu")]
    WebGpu(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemmError {
    InvalidLeftMatrixLength {
        expected: usize,
        actual: usize,
    },
    InvalidRightMatrixLength {
        expected: usize,
        actual: usize,
    },
    InvalidOutputLength {
        expected: usize,
        actual: usize,
    },
    #[cfg(feature = "cuda")]
    Cuda(String),
    #[cfg(feature = "metal")]
    Metal(String),
    #[cfg(feature = "webgpu")]
    WebGpu(String),
    InvalidTensorParallelShardCount {
        shared_dim: usize,
        shard_count: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionError {
    ZeroHeadDim,
    InvalidQueryLength { expected: usize, actual: usize },
    InvalidKeyLength { expected: usize, actual: usize },
    InvalidValueLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
    InvalidKvHead { kv_head: usize, kv_heads: usize },
    InvalidHeadGrouping { num_heads: usize, kv_heads: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RopeError {
    InvalidInputLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
    OddHeadDim { head_dim: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwiGluError {
    InvalidGateLength { expected: usize, actual: usize },
    InvalidUpLength { expected: usize, actual: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFn {
    Relu,
    Gelu,
    Silu,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearActivationError {
    InvalidMatrixLength { expected: usize, actual: usize },
    InvalidVectorLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RmsNormError {
    ZeroDimension,
    InvalidInputLength { expected: usize, actual: usize },
    InvalidWeightLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerNormError {
    InvalidInputLength { expected: usize, actual: usize },
    InvalidWeightLength { expected: usize, actual: usize },
    InvalidBiasLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SoftmaxError {
    InvalidInputLength { expected: usize, actual: usize },
}

pub fn gemv_f32(
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
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    #[cfg(feature = "cuda")]
    if crate::cuda::cuda_build_info().detected_at_build {
        return crate::cuda::gemv_f32_cuda(matrix, rows, cols, vector, output)
            .map_err(|err| GemvError::Cuda(format!("{err:?}")));
    }

    #[cfg(feature = "webgpu")]
    if crate::webgpu::should_use_webgpu_gemv(rows, cols) {
        crate::webgpu::validate_gemv_dims(matrix, rows, cols, vector, output)
            .map_err(|err| GemvError::WebGpu(format!("WebGPU GEMV validation failed: {err:?}")))?;
        gemv_f32_cpu(matrix, cols, vector, output);
        return Ok(());
    }

    #[cfg(feature = "metal")]
    if crate::metal::should_use_mps_gemv(rows, cols) {
        crate::metal::validate_gemv_dims(matrix, rows, cols, vector, output)
            .map_err(|err| GemvError::Metal(format!("MPS GEMV validation failed: {err:?}")))?;
        gemv_f32_cpu(matrix, cols, vector, output);
        return Ok(());
    }

    gemv_f32_cpu(matrix, cols, vector, output);
    Ok(())
}

pub fn gemv_quantized_f32(
    quantization: GgufQuantizationType,
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    #[cfg(feature = "cuda")]
    if crate::cuda::cuda_build_info().detected_at_build {
        return crate::cuda::gemv_quantized_cuda(
            quantization,
            quantized_matrix,
            rows,
            cols,
            vector,
            output,
        )
        .map_err(|err| GemvError::Cuda(format!("{err:?}")));
    }

    match quantization {
        GgufQuantizationType::Q8_0 => gemv_q8_0_f32_fused(quantized_matrix, cols, vector, output),
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => gemv_qk_f32_fused(
            quantized_matrix,
            rows,
            cols,
            vector,
            output,
            BLOCK_Q4_K_SIZE,
            4,
            8.0,
        ),
        GgufQuantizationType::Q6_K => gemv_qk_f32_fused(
            quantized_matrix,
            rows,
            cols,
            vector,
            output,
            BLOCK_Q6_K_SIZE,
            6,
            32.0,
        ),
        _ => Err(GemvError::UnsupportedQuantizationType { quantization }),
    }
}

#[allow(clippy::too_many_arguments)]
fn gemv_qk_f32_fused(
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
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    let compute_row = |row_idx: usize| {
        let mut sum = 0.0_f32;
        let row_start = row_idx * blocks_per_row * block_size;
        let row_blocks = &quantized_matrix[row_start..row_start + (blocks_per_row * block_size)];
        for (block_idx, block) in row_blocks.chunks_exact(block_size).enumerate() {
            let d = f16_le_to_f32([block[0], block[1]]);
            let bitstream = &block[2..];
            let vector_offset = block_idx * QK_K;
            for idx in 0..QK_K {
                let q = extract_bits(bitstream, idx, bits) as f32;
                sum += (q - zero_point) * d * vector[vector_offset + idx];
            }
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, out)| *out = compute_row(row_idx));
    } else {
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx);
        }
    }
    Ok(())
}

pub fn extract_bits(bitstream: &[u8], index: usize, bits: usize) -> u32 {
    let bit_offset = index * bits;
    let byte_index = bit_offset / 8;
    let shift = bit_offset % 8;

    let mut acc = 0_u32;
    for i in 0..4 {
        if let Some(byte) = bitstream.get(byte_index + i) {
            acc |= (*byte as u32) << (8 * i);
        }
    }

    (acc >> shift) & ((1_u32 << bits) - 1)
}

fn gemv_q8_0_f32_fused(
    quantized_matrix: &[u8],
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let rows = output.len();
    let blocks_per_row = cols / QK8_0;
    let compute_row = |row_idx: usize| {
        let mut sum = 0.0_f32;
        let row_start = row_idx * blocks_per_row * BLOCK_Q8_0_SIZE;
        let row_blocks =
            &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_Q8_0_SIZE)];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q8_0_SIZE).enumerate() {
            let scale = f16_le_to_f32([block[0], block[1]]);
            let vector_offset = block_idx * QK8_0;
            for (q, v) in block[2..]
                .iter()
                .zip(&vector[vector_offset..vector_offset + QK8_0])
            {
                sum += (*q as i8) as f32 * scale * *v;
            }
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, out)| *out = compute_row(row_idx));
    } else {
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx);
        }
    }
    Ok(())
}

// Transposed GEMV functions for GGUF weight matrices stored as [input_dim, output_dim]
fn gemv_f32_transposed_cpu(
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
    gemv_f32_transposed_cpu(matrix, rows, cols, vector, output);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gemv_qk_f32_transposed(
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

#[inline]
fn q4_avx2_available() -> bool {
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
fn q4_avx512_available() -> bool {
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
fn accumulate_q4_block(
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
unsafe fn accumulate_q4_block_avx512(bitstream: *const u8, factor: f32, output: *mut f32) {
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
unsafe fn accumulate_q4_block_avx2(bitstream: *const u8, factor: f32, output: *mut f32) {
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

fn gemv_q8_0_f32_transposed(
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
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => gemv_qk_f32_transposed(
            quantized_matrix,
            rows,
            cols,
            vector,
            output,
            BLOCK_Q4_K_SIZE,
            4,
            8.0,
        ),
        GgufQuantizationType::Q6_K => gemv_qk_f32_transposed(
            quantized_matrix,
            rows,
            cols,
            vector,
            output,
            BLOCK_Q6_K_SIZE,
            6,
            32.0,
        ),
        _ => Err(GemvError::UnsupportedQuantizationType { quantization }),
    }
}

pub fn f16_le_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x03FF) as u32;

    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut frac_norm = frac;
            let mut e = -14_i32;
            while (frac_norm & 0x0400) == 0 {
                frac_norm <<= 1;
                e -= 1;
            }
            frac_norm &= 0x03FF;
            (sign << 31) | (((e + 127) as u32) << 23) | (frac_norm << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | 0x7F80_0000 | (frac << 13)
    } else {
        let e = exp as i32 - 15 + 127;
        (sign << 31) | ((e as u32) << 23) | (frac << 13)
    };

    f32::from_bits(f32_bits)
}

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

fn validate_tensor_parallel_dims(
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

fn choose_tensor_parallel_shard_count(shared_dim: usize) -> usize {
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

fn gemv_f32_cpu(matrix: &[f32], cols: usize, vector: &[f32], output: &mut [f32]) {
    let rows = output.len();
    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        matrix
            .par_chunks_exact(cols)
            .zip(output.par_iter_mut())
            .for_each(|(row_values, out)| {
                *out = row_values
                    .iter()
                    .zip(vector.iter())
                    .map(|(weight, value)| weight * value)
                    .sum();
            });
    } else {
        for (row_values, out) in matrix.chunks_exact(cols).zip(output.iter_mut()) {
            *out = row_values
                .iter()
                .zip(vector.iter())
                .map(|(weight, value)| weight * value)
                .sum();
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

fn activate(value: f32, activation: ActivationFn) -> f32 {
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

fn gemm_f32_cpu(
    left_matrix: &[f32],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[f32],
    cols: usize,
    output: &mut [f32],
) {
    const PREFETCH_DISTANCE: usize = 16;
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
            let right_col = &right_transposed[col * shared_dim..(col + 1) * shared_dim];
            let mut sum = 0.0_f32;
            for (k, left_value) in left_row.iter().enumerate() {
                if let (Some(next_left), Some(next_right)) = (
                    left_row.get(k + PREFETCH_DISTANCE),
                    right_col.get(k + PREFETCH_DISTANCE),
                ) {
                    std::hint::black_box(*next_left + *next_right);
                }
                sum += left_value * right_col[k];
            }
            *out_cell = sum;
        }
    }
}

fn gemm_i8_cpu(
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

fn gemm_i4_cpu(
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

fn unpack_i4(packed: &[u8], value_index: usize) -> i32 {
    let byte = packed[value_index / 2];
    let nibble = if value_index.is_multiple_of(2) {
        byte & 0x0F
    } else {
        (byte >> 4) & 0x0F
    };
    i32::from(nibble as i8 - 8)
}

pub fn scaled_dot_product_attention_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    seq_len: usize,
    dim: usize,
    output: &mut [f32],
) -> Result<(), AttentionError> {
    if query.len() != dim {
        return Err(AttentionError::InvalidQueryLength {
            expected: dim,
            actual: query.len(),
        });
    }

    let expected_kv_len = seq_len.saturating_mul(dim);
    if key.len() != expected_kv_len {
        return Err(AttentionError::InvalidKeyLength {
            expected: expected_kv_len,
            actual: key.len(),
        });
    }
    if value.len() != expected_kv_len {
        return Err(AttentionError::InvalidValueLength {
            expected: expected_kv_len,
            actual: value.len(),
        });
    }
    if output.len() != dim {
        return Err(AttentionError::InvalidOutputLength {
            expected: dim,
            actual: output.len(),
        });
    }

    output.fill(0.0);
    if seq_len == 0 {
        return Ok(());
    }

    let scale = 1.0_f32 / (dim as f32).sqrt();
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0_f32;
    let mut token_offset = 0_usize;
    let mut block_acc = vec![0.0_f32; dim];
    let mut block_scores = vec![0.0_f32; FLASH_ATTENTION_BLOCK_TOKENS];

    while token_offset < seq_len {
        let block_len = (seq_len - token_offset).min(FLASH_ATTENTION_BLOCK_TOKENS);
        let block_start = token_offset * dim;
        let block_end = block_start + block_len * dim;
        let key_block = &key[block_start..block_end];
        let value_block = &value[block_start..block_end];

        let mut block_max = f32::NEG_INFINITY;
        for (idx, key_row) in key_block.chunks_exact(dim).enumerate() {
            let score = query
                .iter()
                .zip(key_row.iter())
                .map(|(q, k)| q * k)
                .sum::<f32>()
                * scale;
            block_scores[idx] = score;
            block_max = block_max.max(score);
        }

        block_acc.fill(0.0);
        let mut block_sum = 0.0_f32;
        for (idx, value_row) in value_block.chunks_exact(dim).enumerate() {
            let score = block_scores[idx];
            let weight = (score - block_max).exp();
            block_sum += weight;
            for (acc, v) in block_acc.iter_mut().zip(value_row.iter()) {
                *acc += weight * v;
            }
        }

        let merged_max = running_max.max(block_max);
        let running_scale = (running_max - merged_max).exp();
        let block_scale = (block_max - merged_max).exp();
        for (out, acc) in output.iter_mut().zip(block_acc.iter()) {
            *out = *out * running_scale + acc * block_scale;
        }
        running_sum = running_sum * running_scale + block_sum * block_scale;
        running_max = merged_max;
        token_offset += block_len;
    }

    let inv_sum = 1.0_f32 / running_sum;
    for out in output.iter_mut() {
        *out *= inv_sum;
    }

    Ok(())
}

pub fn apply_rope_f32(
    input: &[f32],
    position: usize,
    head_dim: usize,
    theta: f32,
    output: &mut [f32],
) -> Result<(), RopeError> {
    if input.len() != head_dim {
        return Err(RopeError::InvalidInputLength {
            expected: head_dim,
            actual: input.len(),
        });
    }
    if output.len() != head_dim {
        return Err(RopeError::InvalidOutputLength {
            expected: head_dim,
            actual: output.len(),
        });
    }
    if !head_dim.is_multiple_of(2) {
        return Err(RopeError::OddHeadDim { head_dim });
    }

    let position_f = position as f32;
    let half_dim = head_dim / 2;

    for i in 0..half_dim {
        let x0 = input[2 * i];
        let x1 = input[2 * i + 1];
        let freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
        let angle = position_f * freq;
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();

        output[2 * i] = x0 * cos_angle - x1 * sin_angle;
        output[2 * i + 1] = x0 * sin_angle + x1 * cos_angle;
    }

    Ok(())
}

pub fn apply_swiglu_f32(gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<(), SwiGluError> {
    let expected_len = output.len();
    if gate.len() != expected_len {
        return Err(SwiGluError::InvalidGateLength {
            expected: expected_len,
            actual: gate.len(),
        });
    }
    if up.len() != expected_len {
        return Err(SwiGluError::InvalidUpLength {
            expected: expected_len,
            actual: up.len(),
        });
    }

    for ((gate_value, up_value), out) in gate.iter().zip(up.iter()).zip(output.iter_mut()) {
        let sigmoid = 1.0_f32 / (1.0 + (-gate_value).exp());
        *out = gate_value * sigmoid * up_value;
    }

    Ok(())
}

pub fn rms_norm_f32(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    let hidden_dim = output.len();
    if hidden_dim == 0 {
        return Err(RmsNormError::ZeroDimension);
    }
    if input.len() != hidden_dim {
        return Err(RmsNormError::InvalidInputLength {
            expected: hidden_dim,
            actual: input.len(),
        });
    }
    if weight.len() != hidden_dim {
        return Err(RmsNormError::InvalidWeightLength {
            expected: hidden_dim,
            actual: weight.len(),
        });
    }

    let sum_sq = input.iter().map(|value| value * value).sum::<f32>();
    let mean_sq = sum_sq / hidden_dim as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    for ((value, scale), out) in input.iter().zip(weight.iter()).zip(output.iter_mut()) {
        *out = value * inv_rms * scale;
    }
    Ok(())
}

/// Fused RMS-normalization + transposed GEMV for the attention Q projection.
/// Computes RMSNorm of `input`, then performs transposed GEMV using the
/// normalized vector as the GEMV input.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_gemv_f32_transposed(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    matrix: &[f32],
    rows: usize,
    cols: usize,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    if input.len() != rows {
        return Err(RmsNormError::InvalidInputLength {
            expected: rows,
            actual: input.len(),
        });
    }
    if weight.len() != rows {
        return Err(RmsNormError::InvalidWeightLength {
            expected: rows,
            actual: weight.len(),
        });
    }
    if matrix.len() != rows * cols {
        return Err(RmsNormError::InvalidOutputLength {
            expected: rows * cols,
            actual: matrix.len(),
        });
    }
    if output.len() != cols {
        return Err(RmsNormError::InvalidOutputLength {
            expected: cols,
            actual: output.len(),
        });
    }
    if rows == 0 {
        return Err(RmsNormError::ZeroDimension);
    }

    let sum_sq = input.iter().map(|v| v * v).sum::<f32>();
    let mean_sq = sum_sq / rows as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    output.fill(0.0);
    for (i, (value, scale)) in input.iter().zip(weight.iter()).enumerate() {
        let scaled = value * inv_rms * scale;
        let row = &matrix[i * cols..(i + 1) * cols];
        for (j, &mat_val) in row.iter().enumerate() {
            output[j] += mat_val * scaled;
        }
    }
    Ok(())
}

pub fn layer_norm_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), LayerNormError> {
    let hidden_dim = output.len();
    if input.len() != hidden_dim {
        return Err(LayerNormError::InvalidInputLength {
            expected: hidden_dim,
            actual: input.len(),
        });
    }
    if weight.len() != hidden_dim {
        return Err(LayerNormError::InvalidWeightLength {
            expected: hidden_dim,
            actual: weight.len(),
        });
    }
    if bias.len() != hidden_dim {
        return Err(LayerNormError::InvalidBiasLength {
            expected: hidden_dim,
            actual: bias.len(),
        });
    }

    let mean = input.iter().sum::<f32>() / hidden_dim as f32;
    let variance = input
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f32>()
        / hidden_dim as f32;
    let inv_std = 1.0 / (variance + eps).sqrt();

    for (((value, scale), shift), out) in input
        .iter()
        .zip(weight.iter())
        .zip(bias.iter())
        .zip(output.iter_mut())
    {
        *out = (value - mean) * inv_std * scale + shift;
    }
    Ok(())
}

pub fn softmax_f32(input: &[f32], output: &mut [f32]) -> Result<(), SoftmaxError> {
    let expected_len = output.len();
    if input.len() != expected_len {
        return Err(SoftmaxError::InvalidInputLength {
            expected: expected_len,
            actual: input.len(),
        });
    }

    let max_input = input.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let mut sum_exp = 0.0_f64;
    for (x, out) in input.iter().zip(output.iter_mut()) {
        let exp_value = (x - max_input).exp();
        *out = exp_value;
        sum_exp += exp_value as f64;
    }

    let inv_sum = (1.0_f64 / sum_exp) as f32;
    for out in output.iter_mut() {
        *out *= inv_sum;
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub dtype: DType,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, strides: Vec<usize>, dtype: DType) -> Self {
        assert_eq!(
            shape.len(),
            strides.len(),
            "shape and strides must have the same rank"
        );
        Self {
            shape,
            strides,
            dtype,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        gemv_f32(&dequantized, rows, cols, &vector, &mut reference)
            .expect("f32 gemv should succeed");

        for (lhs, rhs) in quantized_out.iter().zip(reference.iter()) {
            assert!((lhs - rhs).abs() < 1e-4);
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
        gemv_f32(&dequantized, rows, cols, &vector, &mut reference)
            .expect("f32 gemv should succeed");

        for (lhs, rhs) in fused_out.iter().zip(reference.iter()) {
            assert!((lhs - rhs).abs() < 1e-4);
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

        gemm_i4(&left, 1, 3, &right, 1, &mut output)
            .expect("odd shared dim int4 gemm should succeed");

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
        assert!((output[1] - 1.0_f32.sin()).abs() < 1e-6);

        let angle_1 = 1.0_f32 / 100.0;
        assert!((output[2] + angle_1.sin()).abs() < 1e-6);
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
        let err =
            rms_norm_f32(&[], &[], 1e-5, &mut []).expect_err("zero-length output should fail");
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

        layer_norm_f32(&input, &weight, &bias, 0.0, &mut output)
            .expect("layer norm should succeed");

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

        layer_norm_f32(&input, &weight, &bias, 1e-5, &mut output)
            .expect("layer norm should succeed");

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
}
