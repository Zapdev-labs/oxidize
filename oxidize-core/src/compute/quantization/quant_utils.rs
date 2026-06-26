use super::*;

pub(super) fn validate_layout(
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

pub(super) fn f16_le_to_f32(bytes: &[u8]) -> f32 {
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

pub(super) fn f32_to_f16_bits(value: f32) -> u16 {
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

pub(super) fn write_bits(bitstream: &mut [u8], index: usize, bits: usize, value: u32) {
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
