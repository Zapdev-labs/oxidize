use super::*;
use crate::turboquant::TURBOQUANT_BLOCK_SIZE;

pub(super) fn write_storage(
    storage: &mut KvStorage,
    config: &KvCacheConfig,
    layer: usize,
    position: usize,
    src: &[f32],
) {
    let range = token_range(config, layer, position);
    let token_index = token_slot_index(config, layer, position);
    match storage {
        KvStorage::F32(data) => data[range].copy_from_slice(src),
        KvStorage::F16(data) => {
            for (dst, value) in data[range].iter_mut().zip(src.iter()) {
                *dst = f32_to_f16_bits(*value);
            }
        }
        KvStorage::Q8 { data, scales, mins } => {
            let (min, max) = min_max(src);
            let scale = if max <= min { 0.0 } else { (max - min) / 255.0 };
            scales[token_index] = scale;
            mins[token_index] = min;
            if scale == 0.0 {
                data[range].fill(0);
            } else {
                for (dst, value) in data[range].iter_mut().zip(src.iter()) {
                    let q = ((*value - min) / scale).round().clamp(0.0, 255.0) as u8;
                    *dst = q;
                }
            }
        }
        KvStorage::Q4 { data, scales, mins } => {
            let (min, max) = min_max(src);
            let scale = if max <= min { 0.0 } else { (max - min) / 15.0 };
            scales[token_index] = scale;
            mins[token_index] = min;
            let quantize = |value: f32| -> u8 {
                if scale == 0.0 {
                    0
                } else {
                    ((value - min) / scale).round().clamp(0.0, 15.0) as u8
                }
            };
            let packed_start = range.start / 2;
            if range.start.is_multiple_of(2) {
                for (pair_index, pair) in src.chunks(2).enumerate() {
                    let low = quantize(pair[0]);
                    let high = if pair.len() == 2 {
                        quantize(pair[1])
                    } else {
                        (data[packed_start + pair_index] >> 4) & 0x0F
                    };
                    data[packed_start + pair_index] = (high << 4) | (low & 0x0F);
                }
            } else {
                let first_high = quantize(src[0]) << 4;
                data[packed_start] = (data[packed_start] & 0x0F) | first_high;
                for (pair_index, pair) in src[1..].chunks(2).enumerate() {
                    let low = quantize(pair[0]);
                    let high = if pair.len() == 2 {
                        quantize(pair[1])
                    } else {
                        0
                    };
                    data[packed_start + 1 + pair_index] = (high << 4) | (low & 0x0F);
                }
            }
        }
        KvStorage::TurboQ8 { data, scales } => {
            write_turboquant_token::<8>(data, scales, config, range.start, token_index, src);
        }
        KvStorage::TurboQ4 { data, scales } => {
            write_turboquant_token::<4>(data, scales, config, range.start, token_index, src);
        }
    }
}

/// Encode one token into TurboQuant block-quantized form.
///
/// `BITS` is 4 or 8. Each 32-element block of `src` produces one scale
/// (`max_abs / max_val`) and a stream of unsigned codes `q + max_val` written
/// into `data` starting at element offset `data_element_start`. For `BITS=4`
/// the codes are packed two-per-byte (low nibble first); for `BITS=8` they are
/// written one-byte-per-code.
///
/// Requires `data_element_start` to be even when `BITS=4` (guaranteed by
/// `token_range` because `token_size` is always head_count * head_dim, both even).
fn write_turboquant_token<const BITS: u8>(
    data: &mut [u8],
    scales: &mut [f32],
    config: &KvCacheConfig,
    data_element_start: usize,
    token_index: usize,
    src: &[f32],
) {
    let block_size = TURBOQUANT_BLOCK_SIZE;
    let max_val = ((1u32 << (BITS - 1)) - 1) as f32;
    let blocks_per_token = config.blocks_per_token();
    let scale_off = token_index * blocks_per_token;

    for b in 0..blocks_per_token {
        let s = b * block_size;
        let e = (s + block_size).min(src.len());
        let chunk = &src[s..e];
        let mut max_abs = 0.0_f32;
        for &v in chunk {
            max_abs = max_abs.max(v.abs());
        }
        let scale = if max_abs > 0.0 {
            max_abs / max_val
        } else {
            1.0
        };
        scales[scale_off + b] = scale;

        if BITS == 4 {
            let packed_start = (data_element_start + s) / 2;
            for (i, &v) in chunk.iter().enumerate() {
                let q = (v / scale).round().clamp(-max_val, max_val) as i32;
                let uq = (q + max_val as i32) as u8 & 0x0F;
                let byte_idx = packed_start + (i / 2);
                if i % 2 == 0 {
                    data[byte_idx] = (data[byte_idx] & 0xF0) | uq;
                } else {
                    data[byte_idx] = (data[byte_idx] & 0x0F) | (uq << 4);
                }
            }
        } else {
            let byte_start = data_element_start + s;
            for (i, &v) in chunk.iter().enumerate() {
                let q = (v / scale).round().clamp(-max_val, max_val) as i32;
                data[byte_start + i] = (q + max_val as i32) as u8;
            }
        }
    }
}

pub(super) fn read_storage(
    storage: &KvStorage,
    config: &KvCacheConfig,
    layer: usize,
    position: usize,
    dst: &mut [f32],
) {
    let range = token_range(config, layer, position);
    let token_index = token_slot_index(config, layer, position);
    match storage {
        KvStorage::F32(data) => dst.copy_from_slice(&data[range]),
        KvStorage::F16(data) => {
            for (out, value) in dst.iter_mut().zip(data[range].iter()) {
                *out = f16_bits_to_f32(*value);
            }
        }
        KvStorage::Q8 { data, scales, mins } => {
            let scale = scales[token_index];
            let min = mins[token_index];
            for (out, value) in dst.iter_mut().zip(data[range].iter()) {
                *out = (*value as f32) * scale + min;
            }
        }
        KvStorage::Q4 { data, scales, mins } => {
            let scale = scales[token_index];
            let min = mins[token_index];
            let packed_start = range.start / 2;
            if range.start.is_multiple_of(2) {
                for (pair_index, pair) in dst.chunks_mut(2).enumerate() {
                    let byte = data[packed_start + pair_index];
                    pair[0] = ((byte & 0x0F) as f32) * scale + min;
                    if pair.len() == 2 {
                        pair[1] = (((byte >> 4) & 0x0F) as f32) * scale + min;
                    }
                }
            } else {
                let first_byte = data[packed_start];
                dst[0] = (((first_byte >> 4) & 0x0F) as f32) * scale + min;
                for (pair_index, pair) in dst[1..].chunks_mut(2).enumerate() {
                    let byte = data[packed_start + 1 + pair_index];
                    pair[0] = ((byte & 0x0F) as f32) * scale + min;
                    if pair.len() == 2 {
                        pair[1] = (((byte >> 4) & 0x0F) as f32) * scale + min;
                    }
                }
            }
        }
        KvStorage::TurboQ8 { data, scales } => {
            read_turboquant_token::<8>(data, scales, config, range.start, token_index, dst);
        }
        KvStorage::TurboQ4 { data, scales } => {
            read_turboquant_token::<4>(data, scales, config, range.start, token_index, dst);
        }
    }
}

fn read_turboquant_token<const BITS: u8>(
    data: &[u8],
    scales: &[f32],
    config: &KvCacheConfig,
    data_element_start: usize,
    token_index: usize,
    dst: &mut [f32],
) {
    let block_size = TURBOQUANT_BLOCK_SIZE;
    let max_val = ((1u32 << (BITS - 1)) - 1) as f32;
    let blocks_per_token = config.blocks_per_token();
    let scale_off = token_index * blocks_per_token;

    for b in 0..blocks_per_token {
        let s = b * block_size;
        let e = (s + block_size).min(dst.len());
        let scale = scales[scale_off + b];
        if BITS == 4 {
            let packed_start = (data_element_start + s) / 2;
            for i in 0..(e - s) {
                let byte = data[packed_start + (i / 2)];
                let nibble = if i % 2 == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                dst[s + i] = (nibble as f32 - max_val) * scale;
            }
        } else {
            let byte_start = data_element_start + s;
            for i in 0..(e - s) {
                dst[s + i] = (data[byte_start + i] as f32 - max_val) * scale;
            }
        }
    }
}

pub(super) fn migrate_storage_from_position_major(storage: &mut KvStorage, config: &KvCacheConfig) {
    match storage {
        KvStorage::F32(data) => migrate_flat_elements_from_position_major(data, config),
        KvStorage::F16(data) => migrate_flat_elements_from_position_major(data, config),
        KvStorage::Q8 { data, scales, mins } => {
            migrate_flat_elements_from_position_major(data, config);
            migrate_token_slots_from_position_major(scales, config);
            migrate_token_slots_from_position_major(mins, config);
        }
        KvStorage::Q4 { data, scales, mins } => {
            migrate_q4_elements_from_position_major(data, config);
            migrate_token_slots_from_position_major(scales, config);
            migrate_token_slots_from_position_major(mins, config);
        }
        // TurboQuant variants were introduced after the layer-major migration,
        // so no legacy position-major data ever exists for them.
        KvStorage::TurboQ8 { .. } | KvStorage::TurboQ4 { .. } => {}
    }
}

fn migrate_flat_elements_from_position_major<T: Copy>(data: &mut [T], config: &KvCacheConfig) {
    let token_size = config.token_size();
    let expected = config.element_count();
    if data.len() != expected || token_size == 0 {
        return;
    }

    let old = data.to_owned();
    for layer in 0..config.layer_count {
        for position in 0..config.context_size {
            let old_start = (position * config.layer_count + layer) * token_size;
            let new_start = token_range(config, layer, position).start;
            data[new_start..new_start + token_size]
                .copy_from_slice(&old[old_start..old_start + token_size]);
        }
    }
}

fn migrate_token_slots_from_position_major<T: Copy>(data: &mut [T], config: &KvCacheConfig) {
    let expected = config.layer_count.saturating_mul(config.context_size);
    if data.len() != expected {
        return;
    }

    let old = data.to_owned();
    for layer in 0..config.layer_count {
        for position in 0..config.context_size {
            let old_index = position * config.layer_count + layer;
            let new_index = token_slot_index(config, layer, position);
            data[new_index] = old[old_index];
        }
    }
}

fn migrate_q4_elements_from_position_major(data: &mut [u8], config: &KvCacheConfig) {
    let expected_elements = config.element_count();
    if data.len() != expected_elements.div_ceil(2) {
        return;
    }

    let old_nibbles = unpack_q4_nibbles(data, expected_elements);
    let mut new_nibbles = vec![0_u8; expected_elements];
    let token_size = config.token_size();
    for layer in 0..config.layer_count {
        for position in 0..config.context_size {
            let old_start = (position * config.layer_count + layer) * token_size;
            let new_start = token_range(config, layer, position).start;
            new_nibbles[new_start..new_start + token_size]
                .copy_from_slice(&old_nibbles[old_start..old_start + token_size]);
        }
    }
    pack_q4_nibbles(&new_nibbles, data);
}

fn unpack_q4_nibbles(data: &[u8], element_count: usize) -> Vec<u8> {
    let mut nibbles = Vec::with_capacity(element_count);
    for byte in data {
        nibbles.push(byte & 0x0F);
        if nibbles.len() < element_count {
            nibbles.push((byte >> 4) & 0x0F);
        }
    }
    nibbles
}

fn pack_q4_nibbles(nibbles: &[u8], data: &mut [u8]) {
    for (index, byte) in data.iter_mut().enumerate() {
        let low = nibbles.get(index * 2).copied().unwrap_or(0) & 0x0F;
        let high = nibbles.get(index * 2 + 1).copied().unwrap_or(0) & 0x0F;
        *byte = (high << 4) | low;
    }
}

pub(super) fn token_range(
    config: &KvCacheConfig,
    layer: usize,
    position: usize,
) -> std::ops::Range<usize> {
    let token_size = config.token_size();
    let offset = token_slot_index(config, layer, position) * token_size;
    offset..offset + token_size
}

pub(super) fn token_slot_index(config: &KvCacheConfig, layer: usize, position: usize) -> usize {
    layer * config.context_size + position
}

pub(super) fn min_max(values: &[f32]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in values {
        min = min.min(*value);
        max = max.max(*value);
    }
    (min, max)
}

pub(super) fn f16_bits_to_f32(bits: u16) -> f32 {
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

pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let frac = x & 0x7F_FFFF;

    if exp == 0xFF {
        if frac == 0 {
            return sign | 0x7C00;
        }
        let nan = (frac >> 13) as u16;
        return sign | 0x7C00 | nan | 1;
    }

    let exp16 = exp - 127 + 15;
    if exp16 <= 0 {
        if exp16 < -10 {
            return sign;
        }
        let mant = frac | 0x80_0000;
        let shift = (14 - exp16) as u32;
        let mut half_frac = (mant >> shift) as u16;
        if ((mant >> (shift - 1)) & 1) != 0 {
            half_frac = half_frac.wrapping_add(1);
        }
        return sign | half_frac;
    }

    if exp16 >= 0x1F {
        return sign | 0x7C00;
    }

    let mut half_exp = (exp16 as u16) << 10;
    let mut half_frac = (frac >> 13) as u16;
    if (frac & 0x1000) != 0 {
        half_frac = half_frac.wrapping_add(1);
        if (half_frac & 0x0400) != 0 {
            half_frac = 0;
            half_exp = half_exp.wrapping_add(0x0400);
            if half_exp >= 0x7C00 {
                return sign | 0x7C00;
            }
        }
    }
    sign | half_exp | (half_frac & 0x03FF)
}
