//! Q4_K weight dequantization using the same block layout as OXK GEMV kernels.
//!
//! Bit-identical to `oxidize_core::quantization::dequantize_q4_k_scalar` so
//! pruning scores match the legacy path.

use crate::{BLOCK_Q4_K_SIZE, QK_K, f16_le_to_f32, get_scale_min_k4};

/// Dequantize a contiguous Q4_K byte buffer into row-major `f32`.
pub fn dequantize_q4_k_into(input: &[u8], output: &mut [f32]) {
    let n_blocks = input.len() / BLOCK_Q4_K_SIZE;
    debug_assert_eq!(input.len(), n_blocks * BLOCK_Q4_K_SIZE);
    debug_assert_eq!(output.len(), n_blocks * QK_K);
    for (block, out) in input
        .chunks_exact(BLOCK_Q4_K_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        dequantize_block(block, out);
    }
}

#[inline]
fn dequantize_block(block: &[u8], out: &mut [f32]) {
    let d = f16_le_to_f32([block[0], block[1]]);
    let min = f16_le_to_f32([block[2], block[3]]);
    let scales = &block[4..16];
    let qs = &block[16..144];
    let mut out_ptr = 0;
    let mut is = 0;
    for group_pair in 0..4 {
        let q_base = group_pair * 32;
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d1 = d * sc1 as f32;
        let min1 = min * m1 as f32;
        let d2 = d * sc2 as f32;
        let min2 = min * m2 as f32;
        for l in 0..32 {
            out[out_ptr + l] = d1 * ((qs[q_base + l] & 0xF) as f32) - min1;
        }
        for l in 0..32 {
            out[out_ptr + 32 + l] = d2 * ((qs[q_base + l] >> 4) as f32) - min2;
        }
        out_ptr += 64;
        is += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant_block_count_matches() {
        let mut input = vec![0_u8; 2 * BLOCK_Q4_K_SIZE];
        for (i, b) in input.iter_mut().enumerate() {
            *b = (i % 251) as u8 + 1;
        }
        let mut output = vec![0.0_f32; 2 * QK_K];
        dequantize_q4_k_into(&input, &mut output);
        assert!(output.iter().any(|v| v.is_finite()));
    }
}
