use super::*;

pub(super) fn quantize_f32_scalar(
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
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

pub(super) fn quantize_f16_scalar(
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
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

pub(crate) fn quantize_q8_0_scalar(
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
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

pub(super) fn quantize_q4_0_scalar(
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
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
        let mut amax = 0.0_f32;
        let mut mx = 0.0_f32;
        for &v in in_block {
            let a = v.abs();
            if a > amax {
                amax = a;
                mx = v;
            }
        }
        let d = if mx != 0.0 { mx / -8.0 } else { 0.0 };
        let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        for i in 0..16 {
            let lo = if d == 0.0 {
                8_u8
            } else {
                (in_block[i] * inv_d + 8.5).trunc().clamp(0.0, 15.0) as u8
            };
            let hi = if d == 0.0 {
                8_u8
            } else {
                (in_block[i + 16] * inv_d + 8.5)
                    .trunc()
                    .clamp(0.0, 15.0) as u8
            };
            out_block[2 + i] = lo | (hi << 4);
        }
    }

    Ok(())
}

#[inline(always)]
fn al5_pack_levels(out_block: &mut [u8], d: f32, levels: &[i32; QK4_0]) {
    out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
    for i in 0..16 {
        let lo = (levels[i] + 8) as u8;
        let hi = (levels[i + 16] + 8) as u8;
        out_block[2 + i] = lo | (hi << 4);
    }
}

/// Optimized AL5 block quant: `make_qx_quants`-style search (llama.cpp).
/// Grid-searches 19 candidate scales around absmax; for each, takes the
/// least-squares-optimal `d` that minimizes the importance-weighted block error
/// `sum_i w_i (x_i - d*l_i)^2`. `imp` supplies per-column importance (imatrix);
/// when `None` the weight defaults to `x_i^2` — llama.cpp's imatrix-less default,
/// which tracks model quality far better than plain RMSE. Same 18-byte Q4_0
/// bitstream + `dot_q4_0` kernel => zero runtime cost.
#[inline]
fn quantize_block_al5(in_block: &[f32], imp: Option<&[f32]>, out_block: &mut [u8]) {
    let mut amax = 0.0_f32;
    let mut mx = 0.0_f32;
    for &v in in_block {
        let a = v.abs();
        if a > amax {
            amax = a;
            mx = v;
        }
    }
    if amax == 0.0 {
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(0.0).to_le_bytes());
        out_block[2..].fill(0x88);
        return;
    }

    let mut best_d = mx / -8.0;
    let mut best_levels = [0_i32; QK4_0];
    let mut best_obj = -1.0_f32;
    let mut levels = [0_i32; QK4_0];
    for is in -9..=9 {
        let iscale = -(8.0 + 0.1 * is as f32) / mx;
        let mut sumlx = 0.0_f32;
        let mut suml2 = 0.0_f32;
        for (i, &v) in in_block.iter().enumerate() {
            let l = ((v * iscale).round() as i32).clamp(-8, 7);
            levels[i] = l;
            let w = imp.map_or(v * v, |m| m[i]);
            let lf = l as f32;
            sumlx += w * v * lf;
            suml2 += w * lf * lf;
        }
        if suml2 > 0.0 {
            let obj = sumlx * sumlx / suml2; // weighted error reduction; larger = better
            if obj > best_obj {
                best_obj = obj;
                best_d = sumlx / suml2;
                best_levels = levels;
            }
        }
    }

    al5_pack_levels(out_block, best_d, &best_levels);
}

/// ggml split-halves AL5: same 18-byte block as Q4_0, MSE-optimal per-block scale.
pub(super) fn quantize_al5_scalar(
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK4_0) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: GgufQuantizationType::AL5,
            expected_multiple: QK4_0,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK4_0) * BLOCK_Q4_0_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::AL5,
            expected: (input.len() / QK4_0) * BLOCK_Q4_0_SIZE,
            actual: output.len(),
        });
    }

    use rayon::prelude::*;

    input
        .par_chunks_exact(QK4_0)
        .zip(output.par_chunks_exact_mut(BLOCK_Q4_0_SIZE))
        .for_each(|(in_block, out_block)| quantize_block_al5(in_block, None, out_block));

    Ok(())
}

/// Importance-weighted AL5 quantize. Each 32-weight block minimizes the
/// imatrix-weighted error using its per-column importance slice, so weights that
/// multiply high-activation inputs are quantized more faithfully. Same 18-byte
/// output as `quantize_al5_scalar`. `imatrix` is per-element (one importance per
/// input value, tiled across rows by the caller).
pub(super) fn quantize_al5_scalar_weighted(
    input: &[f32],
    imatrix: &[f32],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK4_0) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: GgufQuantizationType::AL5,
            expected_multiple: QK4_0,
            actual: input.len(),
        });
    }
    if imatrix.len() != input.len() {
        return Err(QuantizationError::InvalidImportanceMatrix {
            reason: "matrix length must match input value count",
        });
    }
    if output.len() != (input.len() / QK4_0) * BLOCK_Q4_0_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::AL5,
            expected: (input.len() / QK4_0) * BLOCK_Q4_0_SIZE,
            actual: output.len(),
        });
    }

    use rayon::prelude::*;

    input
        .par_chunks_exact(QK4_0)
        .zip(imatrix.par_chunks_exact(QK4_0))
        .zip(output.par_chunks_exact_mut(BLOCK_Q4_0_SIZE))
        .for_each(|((in_block, imp_block), out_block)| {
            quantize_block_al5(in_block, Some(imp_block), out_block);
        });

    Ok(())
}

/// F16 → AL5 without materializing a full intermediate f32 tensor.
pub(super) fn quantize_f16_to_al5_scalar(
    input: &[u8],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK4_0 * 2) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: GgufQuantizationType::F16,
            expected_multiple: QK4_0 * 2,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / (QK4_0 * 2)) * BLOCK_Q4_0_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::AL5,
            expected: (input.len() / (QK4_0 * 2)) * BLOCK_Q4_0_SIZE,
            actual: output.len(),
        });
    }

    use rayon::prelude::*;

    input
        .par_chunks_exact(QK4_0 * 2)
        .zip(output.par_chunks_exact_mut(BLOCK_Q4_0_SIZE))
        .for_each(|(f16_block, out_block)| {
            let mut block = [0.0_f32; QK4_0];
            for i in 0..QK4_0 {
                block[i] = f16_le_to_f32(&f16_block[2 * i..2 * i + 2]);
            }
            quantize_block_al5(&block, None, out_block);
        });

    Ok(())
}

/// BF16 → AL5 without materializing a full intermediate f32 tensor.
pub(super) fn quantize_bf16_to_al5_scalar(
    input: &[u8],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK4_0 * 2) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: GgufQuantizationType::BF16,
            expected_multiple: QK4_0 * 2,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / (QK4_0 * 2)) * BLOCK_Q4_0_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::AL5,
            expected: (input.len() / (QK4_0 * 2)) * BLOCK_Q4_0_SIZE,
            actual: output.len(),
        });
    }

    use rayon::prelude::*;

    input
        .par_chunks_exact(QK4_0 * 2)
        .zip(output.par_chunks_exact_mut(BLOCK_Q4_0_SIZE))
        .for_each(|(bf16_block, out_block)| {
            let mut block = [0.0_f32; QK4_0];
            for i in 0..QK4_0 {
                let bits = u32::from(u16::from_le_bytes([
                    bf16_block[2 * i],
                    bf16_block[2 * i + 1],
                ])) << 16;
                block[i] = f32::from_bits(bits);
            }
            quantize_block_al5(&block, None, out_block);
        });

    Ok(())
}

pub(super) fn quantize_q4_1_scalar(
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
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

pub(super) fn quantize_q5_0_scalar(
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
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

pub(super) fn quantize_q5_1_scalar(
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
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
        let d = if max <= min {
            0.0
        } else {
            (max - min) / levels
        };
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

pub fn dequantize_bf16_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(GgufQuantizationType::BF16, input, output, 2, 1)?;

    // BF16 is the high 16 bits of an IEEE-754 f32, so widening is a left shift
    // by 16 bits with zero-filled mantissa — exact, no rounding.
    for (src, dst) in input.chunks_exact(2).zip(output.iter_mut()) {
        let bits = u32::from(u16::from_le_bytes([src[0], src[1]])) << 16;
        *dst = f32::from_bits(bits);
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

    for (block, out) in input
        .chunks_exact(BLOCK_Q4_0_SIZE)
        .zip(output.chunks_exact_mut(QK4_0))
    {
        let d = f16_le_to_f32(&block[0..2]);
        for i in 0..16 {
            let packed = block[2 + i];
            out[i] = ((packed & 0x0F) as i32 - 8) as f32 * d;
            out[i + 16] = (((packed >> 4) & 0x0F) as i32 - 8) as f32 * d;
        }
    }

    Ok(())
}

pub fn dequantize_al5_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    dequantize_q4_0_scalar(input, output)
}

pub fn dequantize_q4_1_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q4_1,
        input,
        output,
        BLOCK_Q4_1_SIZE,
        QK4_1,
    )?;

    for (block, out) in input
        .chunks_exact(BLOCK_Q4_1_SIZE)
        .zip(output.chunks_exact_mut(QK4_1))
    {
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

    for (block, out) in input
        .chunks_exact(BLOCK_Q5_0_SIZE)
        .zip(output.chunks_exact_mut(QK5_0))
    {
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

    for (block, out) in input
        .chunks_exact(BLOCK_Q5_1_SIZE)
        .zip(output.chunks_exact_mut(QK5_1))
    {
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

    for (block, out) in input
        .chunks_exact(BLOCK_Q8_0_SIZE)
        .zip(output.chunks_exact_mut(QK8_0))
    {
        let d = f16_le_to_f32(&block[0..2]);
        for i in 0..QK8_0 {
            out[i] = (block[2 + i] as i8) as f32 * d;
        }
    }

    Ok(())
}
