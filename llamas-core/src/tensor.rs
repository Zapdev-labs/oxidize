use crate::gguf::GgufQuantizationType;

const QK8_0: usize = 32;
const BLOCK_Q8_0_SIZE: usize = 2 + QK8_0;
const FLASH_ATTENTION_BLOCK_TOKENS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    I8,
    I16,
    I32,
    I64,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemmError {
    InvalidLeftMatrixLength { expected: usize, actual: usize },
    InvalidRightMatrixLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
    #[cfg(feature = "cuda")]
    Cuda(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionError {
    InvalidQueryLength { expected: usize, actual: usize },
    InvalidKeyLength { expected: usize, actual: usize },
    InvalidValueLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RmsNormError {
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

    for (row_values, out) in matrix.chunks_exact(cols).zip(output.iter_mut()) {
        *out = row_values
            .iter()
            .zip(vector.iter())
            .map(|(weight, value)| weight * value)
            .sum();
    }

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
    if quantization != GgufQuantizationType::Q8_0 {
        return Err(GemvError::UnsupportedQuantizationType { quantization });
    }

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

    crate::cuda::validate_q8_0_gemv_dims(quantized_matrix, rows, cols, vector, output)
        .map_err(|err| match err {
            crate::cuda::GemvCudaError::InvalidMatrixLength { expected, actual } => {
                GemvError::InvalidMatrixLength { expected, actual }
            }
            crate::cuda::GemvCudaError::InvalidVectorLength { expected, actual } => {
                GemvError::InvalidVectorLength { expected, actual }
            }
            crate::cuda::GemvCudaError::InvalidOutputLength { expected, actual } => {
                GemvError::InvalidOutputLength { expected, actual }
            }
            crate::cuda::GemvCudaError::UnsupportedQuantizationType { quantization } => {
                GemvError::UnsupportedQuantizationType { quantization }
            }
            #[cfg(feature = "cuda")]
            crate::cuda::GemvCudaError::Cuda(message) => GemvError::Cuda(message),
        })?;
    gemv_q8_0_f32_fused(quantized_matrix, cols, vector, output)
}

fn gemv_q8_0_f32_fused(
    quantized_matrix: &[u8],
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK8_0;
    for (row_idx, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0_f32;
        let row_start = row_idx * blocks_per_row * BLOCK_Q8_0_SIZE;
        let row_blocks = &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_Q8_0_SIZE)];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q8_0_SIZE).enumerate() {
            let scale = f16_le_to_f32([block[0], block[1]]);
            let vector_offset = block_idx * QK8_0;
            for (q, v) in block[2..].iter().zip(&vector[vector_offset..vector_offset + QK8_0]) {
                sum += (*q as i8) as f32 * scale * *v;
            }
        }
        *out = sum;
    }
    Ok(())
}

fn f16_le_to_f32(bytes: [u8; 2]) -> f32 {
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
    const PREFETCH_DISTANCE: usize = 16;

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
        return crate::cuda::gemm_f32_cuda(left_matrix, rows, shared_dim, right_matrix, cols, output)
            .map_err(|err| GemmError::Cuda(format!("{err:?}")));
    }

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

    Ok(())
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

    while token_offset < seq_len {
        let block_len = (seq_len - token_offset).min(FLASH_ATTENTION_BLOCK_TOKENS);
        let block_start = token_offset * dim;
        let block_end = block_start + block_len * dim;
        let key_block = &key[block_start..block_end];
        let value_block = &value[block_start..block_end];

        let mut block_max = f32::NEG_INFINITY;
        for key_row in key_block.chunks_exact(dim) {
            let score = query
                .iter()
                .zip(key_row.iter())
                .map(|(q, k)| q * k)
                .sum::<f32>()
                * scale;
            block_max = block_max.max(score);
        }

        block_acc.fill(0.0);
        let mut block_sum = 0.0_f32;
        for (key_row, value_row) in key_block.chunks_exact(dim).zip(value_block.chunks_exact(dim)) {
            let score = query
                .iter()
                .zip(key_row.iter())
                .map(|(q, k)| q * k)
                .sum::<f32>()
                * scale;
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
        let vector = (0..cols).map(|i| (i as f32 * 0.05) - 0.6).collect::<Vec<_>>();

        let mut matrix_bytes = Vec::with_capacity(matrix.len() * 4);
        for value in &matrix {
            matrix_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let q8_bytes_len = crate::quantization::quantized_size(GgufQuantizationType::Q8_0, matrix.len())
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
        let q8_bytes_len = crate::quantization::quantized_size(GgufQuantizationType::Q8_0, matrix.len())
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
    fn layer_norm_rejects_mismatched_lengths() {
        let mut output = [0.0_f32; 2];
        let input_err = layer_norm_f32(&[1.0_f32], &[1.0_f32, 1.0], &[0.0_f32, 0.0], 1e-5, &mut output)
            .expect_err("input length mismatch should fail");
        assert!(matches!(input_err, LayerNormError::InvalidInputLength { .. }));

        let weight_err = layer_norm_f32(&[1.0_f32, 2.0], &[1.0_f32], &[0.0_f32, 0.0], 1e-5, &mut output)
            .expect_err("weight length mismatch should fail");
        assert!(matches!(
            weight_err,
            LayerNormError::InvalidWeightLength { .. }
        ));

        let bias_err = layer_norm_f32(&[1.0_f32, 2.0], &[1.0_f32, 1.0], &[0.0_f32], 1e-5, &mut output)
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
        let err = softmax_f32(&[1.0_f32], &mut output)
            .expect_err("mismatched input length should fail");
        assert!(matches!(err, SoftmaxError::InvalidInputLength { .. }));
    }
}
