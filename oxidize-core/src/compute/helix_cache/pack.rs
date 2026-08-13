pub(crate) const PI: f32 = std::f32::consts::PI;

pub(crate) fn packed_bits_bytes(bit_count: usize) -> usize {
    bit_count.div_ceil(8)
}

pub(crate) fn packed_nibble_bytes(code_count: usize) -> usize {
    code_count.div_ceil(2)
}

pub(crate) fn set_bit(data: &mut [u8], index: usize, value: bool) {
    let byte = index / 8;
    let mask = 1u8 << (index % 8);
    if value {
        data[byte] |= mask;
    } else {
        data[byte] &= !mask;
    }
}

pub(crate) fn get_bit(data: &[u8], index: usize) -> bool {
    (data[index / 8] & (1u8 << (index % 8))) != 0
}

pub(crate) fn set_nibble(data: &mut [u8], index: usize, value: u8) {
    let byte = index / 2;
    if index & 1 == 0 {
        data[byte] = (data[byte] & 0xF0) | (value & 0x0F);
    } else {
        data[byte] = (data[byte] & 0x0F) | ((value & 0x0F) << 4);
    }
}

pub(crate) fn get_nibble(data: &[u8], index: usize) -> u8 {
    let byte = data[index / 2];
    if index & 1 == 0 {
        byte & 0x0F
    } else {
        (byte >> 4) & 0x0F
    }
}

pub(crate) fn set_int3(data: &mut [u8], index: usize, value: u8) {
    let bit = index * 3;
    for offset in 0..3 {
        let dst = bit + offset;
        let mask = 1u8 << (dst % 8);
        if value & (1u8 << offset) != 0 {
            data[dst / 8] |= mask;
        } else {
            data[dst / 8] &= !mask;
        }
    }
}

pub(crate) fn get_int3(data: &[u8], index: usize) -> u8 {
    let bit = index * 3;
    let mut value = 0u8;
    for offset in 0..3 {
        let src = bit + offset;
        if data[src / 8] & (1u8 << (src % 8)) != 0 {
            value |= 1u8 << offset;
        }
    }
    value
}

pub(crate) fn wrap_angle(mut value: f32) -> f32 {
    while value <= -PI {
        value += 2.0 * PI;
    }
    while value > PI {
        value -= 2.0 * PI;
    }
    value
}

pub(crate) fn rope_frequency(pair: usize, head_dim: usize, theta: f32) -> f32 {
    theta.powf(-2.0 * pair as f32 / head_dim as f32)
}

pub(crate) fn hadamard8(src: &[f32], dst: &mut [f32; 8]) {
    let a0 = src[0] + src[1];
    let a1 = src[0] - src[1];
    let a2 = src[2] + src[3];
    let a3 = src[2] - src[3];
    let a4 = src[4] + src[5];
    let a5 = src[4] - src[5];
    let a6 = src[6] + src[7];
    let a7 = src[6] - src[7];
    let b0 = a0 + a2;
    let b1 = a1 + a3;
    let b2 = a0 - a2;
    let b3 = a1 - a3;
    let b4 = a4 + a6;
    let b5 = a5 + a7;
    let b6 = a4 - a6;
    let b7 = a5 - a7;
    dst[0] = b0 + b4;
    dst[1] = b1 + b5;
    dst[2] = b2 + b6;
    dst[3] = b3 + b7;
    dst[4] = b0 - b4;
    dst[5] = b1 - b5;
    dst[6] = b2 - b6;
    dst[7] = b3 - b7;
}
