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

pub fn dequantize_scalar(
    quantization: GgufQuantizationType,
    input: &[u8],
    output: &mut [f32],
) -> Result<(), QuantizationError> {
    match quantization {
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn validates_output_length() {
        let mut input = vec![0x00, 0x3C];
        input.extend(0_u8..32_u8);
        let mut out = vec![0.0_f32; 31];

        let err = dequantize_q8_0_scalar(&input, &mut out).expect_err("must reject output size");
        assert!(matches!(err, QuantizationError::InvalidOutputLength { .. }));
    }
}
