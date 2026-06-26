//! Scalar reference for the Q4_K × Q8_K row dot.
//!
//! Replicates the AVX2 kernel's math exactly: integer group sums (no i16
//! saturation can occur — |q4×q8| pair sums peak at 3810 < i16::MAX) and the
//! same per-block f32 combine order, so SIMD variants must match bit-for-bit.

use crate::{
    BLOCK_Q4_K_SIZE, BLOCK_Q8_K_BYTES, QK_K, f16_le_to_f32, get_scale_min_k4, read_q8_k_bsum,
};

/// Dot one Q4_K row (`blocks_per_row` blocks) against a Q8_K vector.
pub fn q4k_q8k_row_dot_scalar(row: &[u8], blocks_per_row: usize, q8k: &[u8]) -> f32 {
    debug_assert!(row.len() >= blocks_per_row * BLOCK_Q4_K_SIZE);
    debug_assert!(q8k.len() >= blocks_per_row * BLOCK_Q8_K_BYTES);
    let mut acc = 0.0_f32;
    for block_idx in 0..blocks_per_row {
        let w = &row[block_idx * BLOCK_Q4_K_SIZE..(block_idx + 1) * BLOCK_Q4_K_SIZE];
        let q8b = &q8k[block_idx * BLOCK_Q8_K_BYTES..(block_idx + 1) * BLOCK_Q8_K_BYTES];
        let d_w = f16_le_to_f32([w[0], w[1]]);
        let dmin_w = f16_le_to_f32([w[2], w[3]]);
        let d_q8 = f32::from_le_bytes([q8b[0], q8b[1], q8b[2], q8b[3]]);
        let scales = &w[4..16];
        let qs = &w[16..16 + QK_K / 2];
        let q8 = &q8b[4..4 + QK_K];
        let bsums = &q8b[4 + QK_K..];

        let mut pos: i32 = 0;
        let mut min_acc: i32 = 0;
        for gp in 0..4 {
            let g1 = gp * 2;
            let g2 = g1 + 1;
            let (s1, ms1) = get_scale_min_k4(g1, scales);
            let (s2, ms2) = get_scale_min_k4(g2, scales);
            let mut sum1: i32 = 0;
            let mut sum2: i32 = 0;
            for i in 0..32 {
                let byte = qs[gp * 32 + i];
                sum1 += (byte & 0x0f) as i32 * (q8[g1 * 32 + i] as i8) as i32;
                sum2 += (byte >> 4) as i32 * (q8[g2 * 32 + i] as i8) as i32;
            }
            pos += s1 as i32 * sum1 + s2 as i32 * sum2;
            let bs1 =
                read_q8_k_bsum(bsums, g1 * 2) as i32 + read_q8_k_bsum(bsums, g1 * 2 + 1) as i32;
            let bs2 =
                read_q8_k_bsum(bsums, g2 * 2) as i32 + read_q8_k_bsum(bsums, g2 * 2 + 1) as i32;
            min_acc += ms1 as i32 * bs1;
            min_acc += ms2 as i32 * bs2;
        }
        acc += d_w * d_q8 * pos as f32 - dmin_w * d_q8 * min_acc as f32;
    }
    acc
}
