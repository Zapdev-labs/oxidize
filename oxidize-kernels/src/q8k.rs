//! Q8_K activation quantization (llama.cpp `block_q8_K` layout).
//!
//! Byte-identical to `quantize_vector_q8_k_into` in oxidize-core's tensor.rs
//! so OXK row dots consume the exact same activation blocks as legacy.

use crate::{BLOCK_Q8_K_BYTES, QK_K};

/// Quantize `vector` (length `n_blocks * 256`) into `n_blocks` Q8_K blocks.
pub fn quantize_q8_k_into(vector: &[f32], n_blocks: usize, out: &mut [u8]) {
    debug_assert_eq!(vector.len(), n_blocks * QK_K);
    debug_assert_eq!(out.len(), n_blocks * BLOCK_Q8_K_BYTES);
    for (b, block_in) in vector
        .as_chunks::<QK_K>()
        .0
        .iter()
        .enumerate()
        .take(n_blocks)
    {
        let block_out = &mut out[b * BLOCK_Q8_K_BYTES..(b + 1) * BLOCK_Q8_K_BYTES];
        quantize_block(block_in, block_out);
    }
}

fn quantize_block(block_in: &[f32], block_out: &mut [u8]) {
    let mut amax = 0.0_f32;
    let mut max = 0.0_f32;
    for &v in block_in {
        let av = v.abs();
        if av > amax {
            amax = av;
            max = v;
        }
    }
    if amax == 0.0 {
        block_out[..4].copy_from_slice(&0.0_f32.to_le_bytes());
        for byte in &mut block_out[4..] {
            *byte = 0;
        }
        return;
    }
    // iscale = -128 / max (sign-preserving symmetry with [-128, 127]).
    let iscale = -128.0_f32 / max;
    let d = 1.0_f32 / iscale;
    block_out[..4].copy_from_slice(&d.to_le_bytes());
    let qs_off = 4;
    for (i, &v) in block_in.iter().enumerate() {
        let q = (iscale * v).round() as i32;
        block_out[qs_off + i] = q.clamp(-128, 127) as i8 as u8;
    }
    let bsums_off = qs_off + QK_K;
    for g in 0..(QK_K / 16) {
        let mut sum: i32 = 0;
        for i in 0..16 {
            sum += (block_out[qs_off + g * 16 + i] as i8) as i32;
        }
        let sum16 = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        block_out[bsums_off + g * 2..bsums_off + g * 2 + 2].copy_from_slice(&sum16.to_le_bytes());
    }
}
