use super::*;

pub const QK_AL: usize = 32;
pub const BLOCK_AL5_XS_SIZE: usize = 14;

#[inline]
fn al_refine_scale(block: &[f32], d: f32, lo: i32, hi: i32) -> f32 {
    if d == 0.0 {
        return 0.0;
    }
    let inv_d = 1.0 / d;
    let mut sumlx = 0.0_f32;
    let mut suml2 = 0.0_f32;
    for &v in block {
        let l = (v * inv_d).round().clamp(lo as f32, hi as f32);
        sumlx += v * l;
        suml2 += l * l;
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        d
    }
}

#[inline]
fn block_amax_mx(block: &[f32]) -> (f32, f32) {
    let mut amax = 0.0_f32;
    let mut mx = 0.0_f32;
    for &v in block {
        let a = v.abs();
        if a > amax {
            amax = a;
            mx = v;
        }
    }
    (amax, mx)
}

fn pack3bit(levels: &[u8; QK_AL], out: &mut [u8; 12]) {
    out.fill(0);
    let mut bitpos = 0u32;
    for &l in levels {
        for b in 0..3u8 {
            if (l >> b) & 1 != 0 {
                let byte_idx = (bitpos / 8) as usize;
                let bit_idx = (bitpos % 8) as u8;
                out[byte_idx] |= 1 << bit_idx;
            }
            bitpos += 1;
        }
    }
}

fn unpack3bit(input: &[u8; 12]) -> [u8; QK_AL] {
    let mut levels = [0u8; QK_AL];
    let mut bitpos = 0u32;
    for lvl in &mut levels {
        let mut v = 0u8;
        for b in 0..3u8 {
            let byte_idx = (bitpos / 8) as usize;
            let bit_idx = (bitpos % 8) as u8;
            if (input[byte_idx] >> bit_idx) & 1 != 0 {
                v |= 1 << b;
            }
            bitpos += 1;
        }
        *lvl = v;
    }
    levels
}

#[inline]
fn quantize_block_al8(in_block: &[f32], out_block: &mut [u8]) {
    let (amax, mx) = block_amax_mx(in_block);
    if amax == 0.0 {
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(0.0).to_le_bytes());
        out_block[2..].fill(0);
        return;
    }
    let mut d = mx / -127.0;
    d = al_refine_scale(in_block, d, -127, 127);
    let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };
    out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
    for (value, dst) in in_block.iter().zip(out_block[2..].iter_mut()) {
        let q = if d == 0.0 {
            0_i32
        } else {
            (value * inv_d).round().clamp(-128.0, 127.0) as i32
        };
        *dst = q as i8 as u8;
    }
}

#[inline]
fn quantize_block_al6(in_block: &[f32], out_block: &mut [u8]) {
    let (amax, mx) = block_amax_mx(in_block);
    if amax == 0.0 {
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(0.0).to_le_bytes());
        out_block[2..6].fill(0);
        out_block[6..].fill(0x10);
        return;
    }
    let mut d = mx / -16.0;
    d = al_refine_scale(in_block, d, -16, 15);
    let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };
    out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
    out_block[2..6].fill(0);
    for (i, value) in in_block.iter().enumerate() {
        let q = if d == 0.0 {
            16_u8
        } else {
            (value * inv_d + 16.5).trunc().clamp(0.0, 31.0) as u8
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

#[inline]
fn quantize_block_al5_xs(in_block: &[f32], out_block: &mut [u8]) {
    let (amax, mx) = block_amax_mx(in_block);
    if amax == 0.0 {
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(0.0).to_le_bytes());
        out_block[2..].fill(0);
        return;
    }
    let mut d = mx / -4.0;
    d = al_refine_scale(in_block, d, -4, 3);
    let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };
    out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
    let mut levels = [0u8; QK_AL];
    for (i, value) in in_block.iter().enumerate() {
        let q = if d == 0.0 {
            4_u8
        } else {
            (value * inv_d + 4.5).trunc().clamp(0.0, 7.0) as u8
        };
        levels[i] = q;
    }
    let mut packed = [0u8; 12];
    pack3bit(&levels, &mut packed);
    out_block[2..].copy_from_slice(&packed);
}

macro_rules! impl_al_scalar {
    ($fn_name:ident, $qtype:expr, $qk:expr, $block:expr, $block_fn:ident) => {
        pub(super) fn $fn_name(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
            if !input.len().is_multiple_of($qk) {
                return Err(QuantizationError::InvalidInputLength {
                    quantization: $qtype,
                    expected_multiple: $qk,
                    actual: input.len(),
                });
            }
            if output.len() != (input.len() / $qk) * $block {
                return Err(QuantizationError::InvalidOutputLength {
                    quantization: $qtype,
                    expected: (input.len() / $qk) * $block,
                    actual: output.len(),
                });
            }
            use rayon::prelude::*;
            input
                .par_chunks_exact($qk)
                .zip(output.par_chunks_exact_mut($block))
                .for_each(|(in_block, out_block)| $block_fn(in_block, out_block));
            Ok(())
        }
    };
}

impl_al_scalar!(
    quantize_al8_scalar,
    GgufQuantizationType::AL8,
    QK8_0,
    BLOCK_Q8_0_SIZE,
    quantize_block_al8
);
impl_al_scalar!(
    quantize_al6_scalar,
    GgufQuantizationType::AL6,
    QK5_0,
    BLOCK_Q5_0_SIZE,
    quantize_block_al6
);
impl_al_scalar!(
    quantize_al5_xs_scalar,
    GgufQuantizationType::AL5_XS,
    QK_AL,
    BLOCK_AL5_XS_SIZE,
    quantize_block_al5_xs
);

pub fn dequantize_al8_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    dequantize_q8_0_scalar(input, output)
}

pub fn dequantize_al6_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    dequantize_q5_0_scalar(input, output)
}

pub fn dequantize_al5_xs_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::AL5_XS,
        input,
        output,
        BLOCK_AL5_XS_SIZE,
        QK_AL,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_AL5_XS_SIZE)
        .zip(output.chunks_exact_mut(QK_AL))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let levels = unpack3bit(block[2..14].try_into().unwrap());
        for (i, &lvl) in levels.iter().enumerate() {
            out[i] = (lvl as i32 - 4) as f32 * d;
        }
    }
    Ok(())
}

pub(super) fn quantize_bf16_to_al8_scalar(
    input: &[u8],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    bf16_to_al_scalar(input, output, quantize_block_al8, QK8_0, BLOCK_Q8_0_SIZE, GgufQuantizationType::AL8)
}

pub(super) fn quantize_bf16_to_al6_scalar(
    input: &[u8],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    bf16_to_al_scalar(input, output, quantize_block_al6, QK5_0, BLOCK_Q5_0_SIZE, GgufQuantizationType::AL6)
}

pub(super) fn quantize_bf16_to_al5_xs_scalar(
    input: &[u8],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    bf16_to_al_scalar(
        input,
        output,
        quantize_block_al5_xs,
        QK_AL,
        BLOCK_AL5_XS_SIZE,
        GgufQuantizationType::AL5_XS,
    )
}

fn bf16_to_al_scalar(
    input: &[u8],
    output: &mut [u8],
    block_fn: fn(&[f32], &mut [u8]),
    qk: usize,
    block_size: usize,
    qtype: GgufQuantizationType,
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(qk * 2) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: GgufQuantizationType::BF16,
            expected_multiple: qk * 2,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / (qk * 2)) * block_size {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: qtype,
            expected: (input.len() / (qk * 2)) * block_size,
            actual: output.len(),
        });
    }
    use rayon::prelude::*;
    input
        .par_chunks_exact(qk * 2)
        .zip(output.par_chunks_exact_mut(block_size))
        .for_each(|(bf16_block, out_block)| {
            let mut block = [0.0_f32; 32];
            let n = qk.min(32);
            for i in 0..n {
                let bits = u32::from(u16::from_le_bytes([
                    bf16_block[2 * i],
                    bf16_block[2 * i + 1],
                ])) << 16;
                block[i] = f32::from_bits(bits);
            }
            block_fn(&block[..qk], out_block);
        });
    Ok(())
}
