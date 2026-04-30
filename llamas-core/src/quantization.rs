use crate::gguf::GgufQuantizationType;

const QK4_0: usize = 32;
const QK4_1: usize = 32;
const QK5_0: usize = 32;
const QK5_1: usize = 32;
const QK8_0: usize = 32;
const QK_K: usize = 256;

const BLOCK_Q4_0_SIZE: usize = 2 + 16;
const BLOCK_Q4_1_SIZE: usize = 2 + 2 + 16;
const BLOCK_Q5_0_SIZE: usize = 2 + 4 + 16;
const BLOCK_Q5_1_SIZE: usize = 2 + 2 + 4 + 16;
const BLOCK_Q8_0_SIZE: usize = 2 + 32;

const BLOCK_Q2_K_SIZE: usize = 2 + 64;
const BLOCK_Q3_K_SIZE: usize = 2 + 96;
const BLOCK_Q4_K_SIZE: usize = 2 + 128;
const BLOCK_Q5_K_SIZE: usize = 2 + 160;
const BLOCK_Q6_K_SIZE: usize = 2 + 192;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizationError {
    InvalidInputLength {
        quantization: GgufQuantizationType,
        expected_multiple: usize,
        actual: usize,
    },
    InvalidOutputLength {
        quantization: GgufQuantizationType,
        expected: usize,
        actual: usize,
    },
    UnsupportedQuantizationType(GgufQuantizationType),
}

impl std::fmt::Display for QuantizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInputLength {
                quantization,
                expected_multiple,
                actual,
            } => write!(
                f,
                "invalid input length for {quantization:?}: expected multiple of {expected_multiple}, got {actual}"
            ),
            Self::InvalidOutputLength {
                quantization,
                expected,
                actual,
            } => write!(
                f,
                "invalid output length for {quantization:?}: expected {expected}, got {actual}"
            ),
            Self::UnsupportedQuantizationType(quantization) => {
                write!(f, "unsupported quantization type: {quantization:?}")
            }
        }
    }
}

impl std::error::Error for QuantizationError {}

pub fn quantized_size(
    quantization: GgufQuantizationType,
    value_count: usize,
) -> Result<usize, QuantizationError> {
    let (values_per_block, bytes_per_block) = match quantization {
        GgufQuantizationType::F32 => (1, 4),
        GgufQuantizationType::F16 => (1, 2),
        GgufQuantizationType::Q4_0 => (QK4_0, BLOCK_Q4_0_SIZE),
        GgufQuantizationType::Q4_1 => (QK4_1, BLOCK_Q4_1_SIZE),
        GgufQuantizationType::Q5_0 => (QK5_0, BLOCK_Q5_0_SIZE),
        GgufQuantizationType::Q5_1 => (QK5_1, BLOCK_Q5_1_SIZE),
        GgufQuantizationType::Q8_0 => (QK8_0, BLOCK_Q8_0_SIZE),
        GgufQuantizationType::Q2_K => (QK_K, BLOCK_Q2_K_SIZE),
        GgufQuantizationType::Q3_K_S | GgufQuantizationType::Q3_K_M | GgufQuantizationType::Q3_K_L => {
            (QK_K, BLOCK_Q3_K_SIZE)
        }
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => (QK_K, BLOCK_Q4_K_SIZE),
        GgufQuantizationType::Q5_K_S | GgufQuantizationType::Q5_K_M => (QK_K, BLOCK_Q5_K_SIZE),
        GgufQuantizationType::Q6_K => (QK_K, BLOCK_Q6_K_SIZE),
        other => return Err(QuantizationError::UnsupportedQuantizationType(other)),
    };

    if !value_count.is_multiple_of(values_per_block) {
        return Err(QuantizationError::InvalidInputLength {
            quantization,
            expected_multiple: values_per_block,
            actual: value_count,
        });
    }

    Ok((value_count / values_per_block) * bytes_per_block)
}

pub fn quantize_scalar(
    source: GgufQuantizationType,
    target: GgufQuantizationType,
    input: &[u8],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    let value_count = match source {
        GgufQuantizationType::F32 => {
            if !input.len().is_multiple_of(4) {
                return Err(QuantizationError::InvalidInputLength {
                    quantization: source,
                    expected_multiple: 4,
                    actual: input.len(),
                });
            }
            input.len() / 4
        }
        GgufQuantizationType::F16 => {
            if !input.len().is_multiple_of(2) {
                return Err(QuantizationError::InvalidInputLength {
                    quantization: source,
                    expected_multiple: 2,
                    actual: input.len(),
                });
            }
            input.len() / 2
        }
        other => return Err(QuantizationError::UnsupportedQuantizationType(other)),
    };

    let expected_output = quantized_size(target, value_count)?;
    if output.len() != expected_output {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: target,
            expected: expected_output,
            actual: output.len(),
        });
    }

    let mut values = vec![0.0_f32; value_count];
    dequantize_scalar(source, input, &mut values)?;
    quantize_from_f32_scalar(target, &values, output)
}

pub fn dequantize_scalar(
    quantization: GgufQuantizationType,
    input: &[u8],
    output: &mut [f32],
) -> Result<(), QuantizationError> {
    match quantization {
        GgufQuantizationType::F32 => {
            dequantize_f32_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::F16 => {
            dequantize_f16_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q4_0 => {
            dequantize_q4_0_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q4_1 => {
            dequantize_q4_1_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q5_0 => {
            dequantize_q5_0_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q5_1 => {
            dequantize_q5_1_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q8_0 => {
            dequantize_q8_0_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q2_K => {
            dequantize_q2_k_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q3_K_S
        | GgufQuantizationType::Q3_K_M
        | GgufQuantizationType::Q3_K_L => {
            dequantize_q3_k_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => {
            dequantize_q4_k_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q5_K_S | GgufQuantizationType::Q5_K_M => {
            dequantize_q5_k_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q6_K => {
            dequantize_q6_k_scalar(input, output)?;
            Ok(())
        }
        other => Err(QuantizationError::UnsupportedQuantizationType(other)),
    }
}

fn quantize_from_f32_scalar(
    target: GgufQuantizationType,
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    match target {
        GgufQuantizationType::F32 => quantize_f32_scalar(input, output),
        GgufQuantizationType::F16 => quantize_f16_scalar(input, output),
        GgufQuantizationType::Q4_0 => quantize_q4_0_scalar(input, output),
        GgufQuantizationType::Q4_1 => quantize_q4_1_scalar(input, output),
        GgufQuantizationType::Q5_0 => quantize_q5_0_scalar(input, output),
        GgufQuantizationType::Q5_1 => quantize_q5_1_scalar(input, output),
        GgufQuantizationType::Q8_0 => quantize_q8_0_scalar(input, output),
        GgufQuantizationType::Q2_K => quantize_k_packed_scalar(
            GgufQuantizationType::Q2_K,
            input,
            output,
            BLOCK_Q2_K_SIZE,
            2,
            1.5,
        ),
        GgufQuantizationType::Q3_K_S | GgufQuantizationType::Q3_K_M | GgufQuantizationType::Q3_K_L => {
            quantize_k_packed_scalar(
                target,
                input,
                output,
                BLOCK_Q3_K_SIZE,
                3,
                3.5,
            )
        }
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => quantize_k_packed_scalar(
            target,
            input,
            output,
            BLOCK_Q4_K_SIZE,
            4,
            8.0,
        ),
        GgufQuantizationType::Q5_K_S | GgufQuantizationType::Q5_K_M => quantize_k_packed_scalar(
            target,
            input,
            output,
            BLOCK_Q5_K_SIZE,
            5,
            16.0,
        ),
        GgufQuantizationType::Q6_K => quantize_k_packed_scalar(
            GgufQuantizationType::Q6_K,
            input,
            output,
            BLOCK_Q6_K_SIZE,
            6,
            32.0,
        ),
        other => Err(QuantizationError::UnsupportedQuantizationType(other)),
    }
}

fn quantize_f32_scalar(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
    if output.len() != input.len() * 4 {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::F32,
            expected: input.len() * 4,
            actual: output.len(),
        });
    }
    for (value, bytes) in input.iter().zip(output.chunks_exact_mut(4)) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn quantize_f16_scalar(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
    if output.len() != input.len() * 2 {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::F16,
            expected: input.len() * 2,
            actual: output.len(),
        });
    }
    for (value, bytes) in input.iter().zip(output.chunks_exact_mut(2)) {
        bytes.copy_from_slice(&f32_to_f16_bits(*value).to_le_bytes());
    }
    Ok(())
}

fn quantize_q8_0_scalar(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK8_0) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: GgufQuantizationType::Q8_0,
            expected_multiple: QK8_0,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK8_0) * BLOCK_Q8_0_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::Q8_0,
            expected: (input.len() / QK8_0) * BLOCK_Q8_0_SIZE,
            actual: output.len(),
        });
    }

    for (in_block, out_block) in input
        .chunks_exact(QK8_0)
        .zip(output.chunks_exact_mut(BLOCK_Q8_0_SIZE))
    {
        let max_abs = in_block.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        let d = if max_abs == 0.0 { 0.0 } else { max_abs / 127.0 };
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        for (value, dst) in in_block.iter().zip(out_block[2..].iter_mut()) {
            let q = if d == 0.0 {
                0
            } else {
                (value / d).round().clamp(-128.0, 127.0) as i32
            };
            *dst = (q as i8) as u8;
        }
    }

    Ok(())
}

fn quantize_q4_0_scalar(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK4_0) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: GgufQuantizationType::Q4_0,
            expected_multiple: QK4_0,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK4_0) * BLOCK_Q4_0_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::Q4_0,
            expected: (input.len() / QK4_0) * BLOCK_Q4_0_SIZE,
            actual: output.len(),
        });
    }

    for (in_block, out_block) in input
        .chunks_exact(QK4_0)
        .zip(output.chunks_exact_mut(BLOCK_Q4_0_SIZE))
    {
        let max_abs = in_block.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        let d = if max_abs == 0.0 { 0.0 } else { max_abs / 8.0 };
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        for i in 0..16 {
            let q_low = if d == 0.0 {
                8_u8
            } else {
                ((in_block[2 * i] / d).round() as i32 + 8).clamp(0, 15) as u8
            };
            let q_high = if d == 0.0 {
                8_u8
            } else {
                ((in_block[2 * i + 1] / d).round() as i32 + 8).clamp(0, 15) as u8
            };
            out_block[2 + i] = q_low | (q_high << 4);
        }
    }

    Ok(())
}

fn quantize_q4_1_scalar(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
    quantize_linear_4bit(
        input,
        output,
        GgufQuantizationType::Q4_1,
        BLOCK_Q4_1_SIZE,
        QK4_1,
        4,
        15.0,
    )
}

fn quantize_q5_0_scalar(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK5_0) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: GgufQuantizationType::Q5_0,
            expected_multiple: QK5_0,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK5_0) * BLOCK_Q5_0_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::Q5_0,
            expected: (input.len() / QK5_0) * BLOCK_Q5_0_SIZE,
            actual: output.len(),
        });
    }

    for (in_block, out_block) in input
        .chunks_exact(QK5_0)
        .zip(output.chunks_exact_mut(BLOCK_Q5_0_SIZE))
    {
        let max_abs = in_block.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        let d = if max_abs == 0.0 { 0.0 } else { max_abs / 16.0 };
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out_block[2..6].fill(0);

        for (i, value) in in_block.iter().enumerate() {
            let q = if d == 0.0 {
                16_u8
            } else {
                ((value / d).round() as i32 + 16).clamp(0, 31) as u8
            };
            if q & 0x10 != 0 {
                out_block[2 + i / 8] |= 1 << (i % 8);
            }
            let low = q & 0x0F;
            let qs_index = 6 + i / 2;
            if i % 2 == 0 {
                out_block[qs_index] = low;
            } else {
                out_block[qs_index] |= low << 4;
            }
        }
    }

    Ok(())
}

fn quantize_q5_1_scalar(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK5_1) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: GgufQuantizationType::Q5_1,
            expected_multiple: QK5_1,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK5_1) * BLOCK_Q5_1_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::Q5_1,
            expected: (input.len() / QK5_1) * BLOCK_Q5_1_SIZE,
            actual: output.len(),
        });
    }

    for (in_block, out_block) in input
        .chunks_exact(QK5_1)
        .zip(output.chunks_exact_mut(BLOCK_Q5_1_SIZE))
    {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for value in in_block {
            min = min.min(*value);
            max = max.max(*value);
        }
        let d = if max <= min { 0.0 } else { (max - min) / 31.0 };
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out_block[2..4].copy_from_slice(&f32_to_f16_bits(min).to_le_bytes());
        out_block[4..8].fill(0);

        for (i, value) in in_block.iter().enumerate() {
            let q = if d == 0.0 {
                0_u8
            } else {
                ((value - min) / d).round().clamp(0.0, 31.0) as u8
            };
            if q & 0x10 != 0 {
                out_block[4 + i / 8] |= 1 << (i % 8);
            }
            let low = q & 0x0F;
            let qs_index = 8 + i / 2;
            if i % 2 == 0 {
                out_block[qs_index] = low;
            } else {
                out_block[qs_index] |= low << 4;
            }
        }
    }

    Ok(())
}

fn quantize_linear_4bit(
    input: &[f32],
    output: &mut [u8],
    quantization: GgufQuantizationType,
    block_size: usize,
    values_per_block: usize,
    payload_offset: usize,
    levels: f32,
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(values_per_block) {
        return Err(QuantizationError::InvalidInputLength {
            quantization,
            expected_multiple: values_per_block,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / values_per_block) * block_size {
        return Err(QuantizationError::InvalidOutputLength {
            quantization,
            expected: (input.len() / values_per_block) * block_size,
            actual: output.len(),
        });
    }

    for (in_block, out_block) in input
        .chunks_exact(values_per_block)
        .zip(output.chunks_exact_mut(block_size))
    {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for value in in_block {
            min = min.min(*value);
            max = max.max(*value);
        }
        let d = if max <= min { 0.0 } else { (max - min) / levels };
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out_block[2..4].copy_from_slice(&f32_to_f16_bits(min).to_le_bytes());
        for i in 0..(values_per_block / 2) {
            let q_low = if d == 0.0 {
                0_u8
            } else {
                ((in_block[2 * i] - min) / d).round().clamp(0.0, 15.0) as u8
            };
            let q_high = if d == 0.0 {
                0_u8
            } else {
                ((in_block[2 * i + 1] - min) / d).round().clamp(0.0, 15.0) as u8
            };
            out_block[payload_offset + i] = q_low | (q_high << 4);
        }
    }
    Ok(())
}

fn quantize_k_packed_scalar(
    quantization: GgufQuantizationType,
    input: &[f32],
    output: &mut [u8],
    block_size: usize,
    bits: usize,
    zero_point: f32,
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK_K) {
        return Err(QuantizationError::InvalidInputLength {
            quantization,
            expected_multiple: QK_K,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK_K) * block_size {
        return Err(QuantizationError::InvalidOutputLength {
            quantization,
            expected: (input.len() / QK_K) * block_size,
            actual: output.len(),
        });
    }

    let max_q = ((1_u32 << bits) - 1) as f32;
    let positive_span = zero_point.max(max_q - zero_point);
    for (in_block, out_block) in input.chunks_exact(QK_K).zip(output.chunks_exact_mut(block_size))
    {
        let max_abs = in_block.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        let d = if max_abs == 0.0 {
            0.0
        } else {
            max_abs / positive_span
        };
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out_block[2..].fill(0);

        for (i, value) in in_block.iter().enumerate() {
            let q = if d == 0.0 {
                zero_point.round() as u32
            } else {
                ((value / d) + zero_point).round().clamp(0.0, max_q) as u32
            };
            write_bits(&mut out_block[2..], i, bits, q);
        }
    }

    Ok(())
}

pub fn dequantize_f32_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(GgufQuantizationType::F32, input, output, 4, 1)?;

    for (src, dst) in input.chunks_exact(4).zip(output.iter_mut()) {
        *dst = f32::from_le_bytes([src[0], src[1], src[2], src[3]]);
    }

    Ok(())
}

pub fn dequantize_f16_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(GgufQuantizationType::F16, input, output, 2, 1)?;

    for (src, dst) in input.chunks_exact(2).zip(output.iter_mut()) {
        *dst = f16_le_to_f32(src);
    }

    Ok(())
}

pub fn dequantize_q4_0_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q4_0,
        input,
        output,
        BLOCK_Q4_0_SIZE,
        QK4_0,
    )?;

    for (block, out) in input.chunks_exact(BLOCK_Q4_0_SIZE).zip(output.chunks_exact_mut(QK4_0)) {
        let d = f16_le_to_f32(&block[0..2]);
        for i in 0..16 {
            let packed = block[2 + i];
            out[2 * i] = ((packed & 0x0F) as i8 - 8) as f32 * d;
            out[2 * i + 1] = (((packed >> 4) & 0x0F) as i8 - 8) as f32 * d;
        }
    }

    Ok(())
}

pub fn dequantize_q4_1_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q4_1,
        input,
        output,
        BLOCK_Q4_1_SIZE,
        QK4_1,
    )?;

    for (block, out) in input.chunks_exact(BLOCK_Q4_1_SIZE).zip(output.chunks_exact_mut(QK4_1)) {
        let d = f16_le_to_f32(&block[0..2]);
        let m = f16_le_to_f32(&block[2..4]);
        for i in 0..16 {
            let packed = block[4 + i];
            out[2 * i] = (packed & 0x0F) as f32 * d + m;
            out[2 * i + 1] = ((packed >> 4) & 0x0F) as f32 * d + m;
        }
    }

    Ok(())
}

pub fn dequantize_q5_0_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q5_0,
        input,
        output,
        BLOCK_Q5_0_SIZE,
        QK5_0,
    )?;

    for (block, out) in input.chunks_exact(BLOCK_Q5_0_SIZE).zip(output.chunks_exact_mut(QK5_0)) {
        let d = f16_le_to_f32(&block[0..2]);
        let qh = &block[2..6];
        let qs = &block[6..22];

        for i in 0..QK5_0 {
            let low = if i % 2 == 0 {
                qs[i / 2] & 0x0F
            } else {
                (qs[i / 2] >> 4) & 0x0F
            };
            let high = (qh[i / 8] >> (i % 8)) & 0x01;
            let q = low | (high << 4);
            out[i] = (q as i8 - 16) as f32 * d;
        }
    }

    Ok(())
}

pub fn dequantize_q5_1_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q5_1,
        input,
        output,
        BLOCK_Q5_1_SIZE,
        QK5_1,
    )?;

    for (block, out) in input.chunks_exact(BLOCK_Q5_1_SIZE).zip(output.chunks_exact_mut(QK5_1)) {
        let d = f16_le_to_f32(&block[0..2]);
        let m = f16_le_to_f32(&block[2..4]);
        let qh = &block[4..8];
        let qs = &block[8..24];

        for i in 0..QK5_1 {
            let low = if i % 2 == 0 {
                qs[i / 2] & 0x0F
            } else {
                (qs[i / 2] >> 4) & 0x0F
            };
            let high = (qh[i / 8] >> (i % 8)) & 0x01;
            let q = low | (high << 4);
            out[i] = q as f32 * d + m;
        }
    }

    Ok(())
}

pub fn dequantize_q8_0_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q8_0,
        input,
        output,
        BLOCK_Q8_0_SIZE,
        QK8_0,
    )?;

    for (block, out) in input.chunks_exact(BLOCK_Q8_0_SIZE).zip(output.chunks_exact_mut(QK8_0)) {
        let d = f16_le_to_f32(&block[0..2]);
        for i in 0..QK8_0 {
            out[i] = (block[2 + i] as i8) as f32 * d;
        }
    }

    Ok(())
}

pub fn dequantize_q2_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    dequantize_k_packed(
        GgufQuantizationType::Q2_K,
        input,
        output,
        BLOCK_Q2_K_SIZE,
        2,
        1.5,
    )
}

pub fn dequantize_q3_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    dequantize_k_packed(
        GgufQuantizationType::Q3_K_S,
        input,
        output,
        BLOCK_Q3_K_SIZE,
        3,
        3.5,
    )
}

pub fn dequantize_q4_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    dequantize_k_packed(
        GgufQuantizationType::Q4_K_S,
        input,
        output,
        BLOCK_Q4_K_SIZE,
        4,
        8.0,
    )
}

pub fn dequantize_q5_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    dequantize_k_packed(
        GgufQuantizationType::Q5_K_S,
        input,
        output,
        BLOCK_Q5_K_SIZE,
        5,
        16.0,
    )
}

pub fn dequantize_q6_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    dequantize_k_packed(
        GgufQuantizationType::Q6_K,
        input,
        output,
        BLOCK_Q6_K_SIZE,
        6,
        32.0,
    )
}

fn dequantize_k_packed(
    quantization: GgufQuantizationType,
    input: &[u8],
    output: &mut [f32],
    block_size: usize,
    bits: usize,
    zero_point: f32,
) -> Result<(), QuantizationError> {
    validate_layout(quantization, input, output, block_size, QK_K)?;

    for (block, out) in input.chunks_exact(block_size).zip(output.chunks_exact_mut(QK_K)) {
        let d = f16_le_to_f32(&block[0..2]);
        let bitstream = &block[2..];
        for (idx, slot) in out.iter_mut().enumerate() {
            let q = extract_bits(bitstream, idx, bits) as f32;
            *slot = (q - zero_point) * d;
        }
    }

    Ok(())
}

fn validate_layout(
    quantization: GgufQuantizationType,
    input: &[u8],
    output: &[f32],
    input_block_size: usize,
    values_per_block: usize,
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(input_block_size) {
        return Err(QuantizationError::InvalidInputLength {
            quantization,
            expected_multiple: input_block_size,
            actual: input.len(),
        });
    }

    let expected_output = (input.len() / input_block_size) * values_per_block;
    if output.len() != expected_output {
        return Err(QuantizationError::InvalidOutputLength {
            quantization,
            expected: expected_output,
            actual: output.len(),
        });
    }

    Ok(())
}

fn extract_bits(bitstream: &[u8], index: usize, bits: usize) -> u32 {
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

fn f16_le_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
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

fn f32_to_f16_bits(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let frac = x & 0x007F_FFFF;

    if exp == 0xFF {
        if frac == 0 {
            return sign | 0x7C00;
        }
        let nan = (frac >> 13) as u16;
        return sign | 0x7C00 | nan | if nan == 0 { 1 } else { 0 };
    }

    let exp16 = exp - 127 + 15;
    if exp16 >= 0x1F {
        return sign | 0x7C00;
    }
    if exp16 <= 0 {
        if exp16 < -10 {
            return sign;
        }
        let mant = frac | 0x0080_0000;
        let shift = (14 - exp16) as u32;
        let mut half_frac = (mant >> shift) as u16;
        if ((mant >> (shift - 1)) & 1) != 0 {
            half_frac = half_frac.wrapping_add(1);
        }
        return sign | half_frac;
    }

    let mut half_exp = (exp16 as u16) << 10;
    let mut half_frac = (frac >> 13) as u16;
    if (frac & 0x0000_1000) != 0 {
        half_frac = half_frac.wrapping_add(1);
        if (half_frac & 0x0400) != 0 {
            half_frac = 0;
            half_exp = half_exp.wrapping_add(0x0400);
            if half_exp >= 0x7C00 {
                return sign | 0x7C00;
            }
        }
    }
    sign | half_exp | half_frac
}

fn write_bits(bitstream: &mut [u8], index: usize, bits: usize, value: u32) {
    let bit_offset = index * bits;
    let byte_index = bit_offset / 8;
    let shift = bit_offset % 8;
    let mask = ((1_u32 << bits) - 1) << shift;

    let mut acc = 0_u32;
    for i in 0..4 {
        if let Some(byte) = bitstream.get(byte_index + i) {
            acc |= (*byte as u32) << (8 * i);
        }
    }
    acc = (acc & !mask) | ((value << shift) & mask);
    for i in 0..4 {
        if let Some(byte) = bitstream.get_mut(byte_index + i) {
            *byte = ((acc >> (8 * i)) & 0xFF) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantizes_f32_scalar_values() {
        let mut input = Vec::new();
        input.extend_from_slice(&1.25_f32.to_le_bytes());
        input.extend_from_slice(&(-2.5_f32).to_le_bytes());

        let mut out = vec![0.0_f32; 2];
        dequantize_f32_scalar(&input, &mut out).expect("f32 dequant succeeds");

        assert!((out[0] - 1.25).abs() < 1e-6);
        assert!((out[1] + 2.5).abs() < 1e-6);
    }

    #[test]
    fn dequantizes_f16_scalar_values() {
        let input = vec![
            0x00, 0x3C, // 1.0
            0x00, 0xC1, // -2.5
        ];

        let mut out = vec![0.0_f32; 2];
        dequantize_f16_scalar(&input, &mut out).expect("f16 dequant succeeds");

        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] + 2.5).abs() < 1e-6);
    }

    #[test]
    fn dequantizes_q4_0_scalar_block() {
        let mut input = vec![0x00, 0x3C];
        input.extend(std::iter::repeat_n(0x98, 16));

        let mut out = vec![0.0_f32; 32];
        dequantize_q4_0_scalar(&input, &mut out).expect("q4_0 dequant succeeds");

        assert!(out.iter().step_by(2).all(|v| (*v - 0.0).abs() < 1e-6));
        assert!(out
            .iter()
            .skip(1)
            .step_by(2)
            .all(|v| (*v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn dequantizes_q5_0_scalar_block() {
        let mut input = vec![0x00, 0x3C];
        input.extend([0x01, 0x00, 0x00, 0x00]);
        input.extend(std::iter::repeat_n(0x00, 16));

        let mut out = vec![0.0_f32; 32];
        dequantize_q5_0_scalar(&input, &mut out).expect("q5_0 dequant succeeds");

        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] + 16.0).abs() < 1e-6);
    }

    #[test]
    fn dequantizes_q8_0_scalar_block() {
        let mut input = vec![0x00, 0x3C];
        input.extend(0_u8..32_u8);

        let mut out = vec![0.0_f32; 32];
        dequantize_q8_0_scalar(&input, &mut out).expect("q8_0 dequant succeeds");

        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[31] - 31.0).abs() < 1e-6);
    }

    #[test]
    fn dequantizes_k_quant_scalar_block() {
        let mut input = vec![0x00, 0x3C];
        input.extend(std::iter::repeat_n(0_u8, 64));

        let mut out = vec![0.0_f32; 256];
        dequantize_q2_k_scalar(&input, &mut out).expect("q2_k dequant succeeds");

        assert!(out.iter().all(|v| (*v + 1.5).abs() < 1e-6));
    }

    #[test]
    fn dispatches_by_quantization_type() {
        let mut input = vec![0x00, 0x3C];
        input.extend(0_u8..32_u8);
        let mut out = vec![0.0_f32; 32];

        dequantize_scalar(GgufQuantizationType::Q8_0, &input, &mut out)
            .expect("dispatch succeeds");
        assert!((out[4] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn dispatches_f16_and_f32_types() {
        let f16_input = vec![0x00, 0x3C, 0x00, 0x40];
        let mut f16_out = vec![0.0_f32; 2];
        dequantize_scalar(GgufQuantizationType::F16, &f16_input, &mut f16_out)
            .expect("f16 dispatch succeeds");
        assert!((f16_out[0] - 1.0).abs() < 1e-6);
        assert!((f16_out[1] - 2.0).abs() < 1e-6);

        let mut f32_input = Vec::new();
        f32_input.extend_from_slice(&3.0_f32.to_le_bytes());
        f32_input.extend_from_slice(&(-4.0_f32).to_le_bytes());
        let mut f32_out = vec![0.0_f32; 2];
        dequantize_scalar(GgufQuantizationType::F32, &f32_input, &mut f32_out)
            .expect("f32 dispatch succeeds");
        assert!((f32_out[0] - 3.0).abs() < 1e-6);
        assert!((f32_out[1] + 4.0).abs() < 1e-6);
    }

    #[test]
    fn validates_output_length() {
        let mut input = vec![0x00, 0x3C];
        input.extend(0_u8..32_u8);
        let mut out = vec![0.0_f32; 31];

        let err = dequantize_q8_0_scalar(&input, &mut out).expect_err("must reject output size");
        assert!(matches!(err, QuantizationError::InvalidOutputLength { .. }));
    }

    #[test]
    fn quantizes_from_f32_to_all_supported_formats() {
        let targets = [
            GgufQuantizationType::F32,
            GgufQuantizationType::F16,
            GgufQuantizationType::Q4_0,
            GgufQuantizationType::Q4_1,
            GgufQuantizationType::Q5_0,
            GgufQuantizationType::Q5_1,
            GgufQuantizationType::Q8_0,
            GgufQuantizationType::Q2_K,
            GgufQuantizationType::Q3_K_S,
            GgufQuantizationType::Q3_K_M,
            GgufQuantizationType::Q3_K_L,
            GgufQuantizationType::Q4_K_S,
            GgufQuantizationType::Q4_K_M,
            GgufQuantizationType::Q5_K_S,
            GgufQuantizationType::Q5_K_M,
            GgufQuantizationType::Q6_K,
        ];

        for target in targets {
            let values = test_values_for_target(target, -8.0, 0.25);
            let mut src = Vec::with_capacity(values.len() * 4);
            for value in &values {
                src.extend_from_slice(&value.to_le_bytes());
            }

            let out_size = quantized_size(target, values.len()).expect("size must be known");
            let mut quantized = vec![0_u8; out_size];
            quantize_scalar(GgufQuantizationType::F32, target, &src, &mut quantized)
                .expect("f32 source quantization must succeed");

            let mut recovered = vec![0.0_f32; values.len()];
            dequantize_scalar(target, &quantized, &mut recovered)
                .expect("dequantization of quantized payload must succeed");
            assert!(recovered.iter().all(|v| v.is_finite()));
            if target == GgufQuantizationType::F32 {
                assert_eq!(src, quantized);
            }
        }
    }

    #[test]
    fn quantizes_from_f16_to_all_supported_formats() {
        let targets = [
            GgufQuantizationType::F32,
            GgufQuantizationType::F16,
            GgufQuantizationType::Q4_0,
            GgufQuantizationType::Q4_1,
            GgufQuantizationType::Q5_0,
            GgufQuantizationType::Q5_1,
            GgufQuantizationType::Q8_0,
            GgufQuantizationType::Q2_K,
            GgufQuantizationType::Q3_K_S,
            GgufQuantizationType::Q3_K_M,
            GgufQuantizationType::Q3_K_L,
            GgufQuantizationType::Q4_K_S,
            GgufQuantizationType::Q4_K_M,
            GgufQuantizationType::Q5_K_S,
            GgufQuantizationType::Q5_K_M,
            GgufQuantizationType::Q6_K,
        ];

        for target in targets {
            let values = test_values_for_target(target, -12.0, 0.2);
            let mut src = Vec::with_capacity(values.len() * 2);
            for value in &values {
                src.extend_from_slice(&f32_to_f16_bits(*value).to_le_bytes());
            }

            let out_size = quantized_size(target, values.len()).expect("size must be known");
            let mut quantized = vec![0_u8; out_size];
            quantize_scalar(GgufQuantizationType::F16, target, &src, &mut quantized)
                .expect("f16 source quantization must succeed");

            let mut recovered = vec![0.0_f32; values.len()];
            dequantize_scalar(target, &quantized, &mut recovered)
                .expect("dequantization of quantized payload must succeed");
            assert!(recovered.iter().all(|v| v.is_finite()));
            if target == GgufQuantizationType::F16 {
                assert_eq!(src, quantized);
            }
        }
    }

    #[test]
    fn q8_0_quantization_uses_independent_scales_per_block() {
        let mut values = vec![0.0_f32; QK8_0 * 2];
        for (i, slot) in values[..QK8_0].iter_mut().enumerate() {
            *slot = i as f32 * 0.5;
        }
        for (i, slot) in values[QK8_0..].iter_mut().enumerate() {
            *slot = i as f32 * 6.0;
        }

        let mut src = Vec::with_capacity(values.len() * 4);
        for value in &values {
            src.extend_from_slice(&value.to_le_bytes());
        }

        let mut quantized = vec![0_u8; BLOCK_Q8_0_SIZE * 2];
        quantize_scalar(
            GgufQuantizationType::F32,
            GgufQuantizationType::Q8_0,
            &src,
            &mut quantized,
        )
        .expect("q8_0 quantization succeeds");

        let first_scale = f16_le_to_f32(&quantized[0..2]);
        let second_scale = f16_le_to_f32(&quantized[BLOCK_Q8_0_SIZE..BLOCK_Q8_0_SIZE + 2]);
        assert!(second_scale > first_scale * 8.0);
    }

    #[test]
    fn q4_1_quantization_uses_independent_scales_per_block() {
        let mut values = vec![0.0_f32; QK4_1 * 2];
        for (i, slot) in values[..QK4_1].iter_mut().enumerate() {
            *slot = -2.0 + i as f32 * 0.1;
        }
        for (i, slot) in values[QK4_1..].iter_mut().enumerate() {
            *slot = -40.0 + i as f32 * 3.0;
        }

        let mut src = Vec::with_capacity(values.len() * 4);
        for value in &values {
            src.extend_from_slice(&value.to_le_bytes());
        }

        let mut quantized = vec![0_u8; BLOCK_Q4_1_SIZE * 2];
        quantize_scalar(
            GgufQuantizationType::F32,
            GgufQuantizationType::Q4_1,
            &src,
            &mut quantized,
        )
        .expect("q4_1 quantization succeeds");

        let first_scale = f16_le_to_f32(&quantized[0..2]);
        let second_scale = f16_le_to_f32(&quantized[BLOCK_Q4_1_SIZE..BLOCK_Q4_1_SIZE + 2]);
        assert!(second_scale > first_scale * 20.0);
    }

    fn test_values_for_target(target: GgufQuantizationType, offset: f32, scale: f32) -> Vec<f32> {
        let value_count = if matches!(
            target,
            GgufQuantizationType::Q2_K
                | GgufQuantizationType::Q3_K_S
                | GgufQuantizationType::Q3_K_M
                | GgufQuantizationType::Q3_K_L
                | GgufQuantizationType::Q4_K_S
                | GgufQuantizationType::Q4_K_M
                | GgufQuantizationType::Q5_K_S
                | GgufQuantizationType::Q5_K_M
                | GgufQuantizationType::Q6_K
        ) {
            QK_K
        } else if matches!(
            target,
            GgufQuantizationType::Q4_0
                | GgufQuantizationType::Q4_1
                | GgufQuantizationType::Q5_0
                | GgufQuantizationType::Q5_1
                | GgufQuantizationType::Q8_0
        ) {
            QK8_0
        } else {
            2
        };
        (0..value_count)
            .map(|i| (i as f32 + offset) * scale)
            .collect()
    }
}
