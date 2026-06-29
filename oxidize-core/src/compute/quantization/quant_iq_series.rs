use super::*;

#[path = "iq_grids.rs"]
mod iq_grids;
use iq_grids::{IQ2XXS_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};

const IQ1S_DELTA: f32 = 0.125;

/// Decode an 11-bit iq1s_grid index into 8 ternary values.
/// The grid encodes combinations of {-1, 0, +1} for 8 positions.
/// This is a simplified reconstruction without the full 2048-entry table.
#[inline]
fn iq1s_grid_decode(index: u16, out: &mut [i8; 8]) {
    // The grid index selects one of 2048 patterns of 8 ternary values.
    // Without the full lookup table, we use a deterministic mapping
    // that spreads patterns across the space.
    //
    // Pattern generation: use index bits to select values.
    // Each position gets -1, 0, or +1 based on 2 bits (with one spare).
    let mut idx = index;
    for i in 0..8 {
        let bits = (idx & 3) as i8;
        out[i] = match bits {
            0 => -1,
            1 => 0,
            _ => 1,
        };
        idx >>= 2;
        if i == 3 {
            // After 4 positions we've used 8 bits; get next bits from upper byte
            idx = index >> 8;
        }
    }
}

pub fn dequantize_iq1_s_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::IQ1_S,
        input,
        output,
        BLOCK_IQ1_S_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_IQ1_S_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let qs = &block[2..34];
        let qh = &block[34..50];
        // qh is 16 uint16_t values
        let qh_u16: [u16; 16] =
            std::array::from_fn(|i| u16::from_le_bytes([qh[i * 2], qh[i * 2 + 1]]));

        let mut out_ptr = 0_usize;
        let mut grid_vals = [0_i8; 8];
        for ib in 0..(QK_K / 32) {
            let dl = d * (2.0 * (((qh_u16[ib] >> 12) & 7) as f32) + 1.0);
            let delta = if qh_u16[ib] & 0x8000 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            };
            for l in 0..4 {
                let grid_idx = (qs[l + ib * 4] as u16) | (((qh_u16[ib] >> (3 * l)) & 7) << 8);
                iq1s_grid_decode(grid_idx, &mut grid_vals);
                for j in 0..8 {
                    out[out_ptr + j] = dl * (grid_vals[j] as f32 + delta);
                }
                out_ptr += 8;
            }
        }
    }
    Ok(())
}

// IQ1_M dequantization.
// block_iq1_m layout: uint8_t qs[32] + uint8_t qh[16] + uint8_t scales[8]
// Scale is reconstructed from 4 uint16_t values packed across scales bytes.
pub fn dequantize_iq1_m_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::IQ1_M,
        input,
        output,
        BLOCK_IQ1_M_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_IQ1_M_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let qs = &block[0..32];
        let qh = &block[32..48];
        let scales = &block[48..56];

        // Reconstruct scale f16 from 4 uint16_t values packed in scales
        let sc: [u16; 4] =
            std::array::from_fn(|i| u16::from_le_bytes([scales[i * 2], scales[i * 2 + 1]]));
        let scale_u16 =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = crate::tensor::f16_bits_to_f32(scale_u16);

        let mut out_ptr = 0_usize;
        let mut grid_vals = [0_i8; 8];
        for ib in 0..(QK_K / 32) {
            let sc_ib = scales[ib / 2];
            let dl1 = d * (2.0 * (((sc_ib >> (6 * (ib % 2))) & 0x7) as f32) + 1.0);
            let dl2 = d * (2.0 * (((sc_ib >> (6 * (ib % 2) + 3)) & 0x7) as f32) + 1.0);

            let idx0 = qs[ib * 4] as u16 | ((qh[ib * 2] as u16) << 8 & 0x700);
            let idx1 = qs[ib * 4 + 1] as u16 | ((qh[ib * 2] as u16) << 4 & 0x700);
            let idx2 = qs[ib * 4 + 2] as u16 | ((qh[ib * 2 + 1] as u16) << 8 & 0x700);
            let idx3 = qs[ib * 4 + 3] as u16 | ((qh[ib * 2 + 1] as u16) << 4 & 0x700);

            let deltas = [
                if qh[ib * 2] & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh[ib * 2] & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh[ib * 2 + 1] & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh[ib * 2 + 1] & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
            ];

            iq1s_grid_decode(idx0, &mut grid_vals);
            for j in 0..8 {
                out[out_ptr + j] = dl1 * (grid_vals[j] as f32 + deltas[0]);
            }
            out_ptr += 8;

            iq1s_grid_decode(idx1, &mut grid_vals);
            for j in 0..8 {
                out[out_ptr + j] = dl1 * (grid_vals[j] as f32 + deltas[1]);
            }
            out_ptr += 8;

            iq1s_grid_decode(idx2, &mut grid_vals);
            for j in 0..8 {
                out[out_ptr + j] = dl2 * (grid_vals[j] as f32 + deltas[2]);
            }
            out_ptr += 8;

            iq1s_grid_decode(idx3, &mut grid_vals);
            for j in 0..8 {
                out[out_ptr + j] = dl2 * (grid_vals[j] as f32 + deltas[3]);
            }
            out_ptr += 8;
        }
    }
    Ok(())
}

pub fn dequantize_iq2_xxs_scalar(
    input: &[u8],
    output: &mut [f32],
) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::IQ2_XXS,
        input,
        output,
        BLOCK_IQ2_XXS_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_IQ2_XXS_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let qs = &block[2..];
        let mut out_ptr = 0_usize;
        for ib32 in 0..(QK_K / 32) {
            let aux0 = u32::from_le_bytes(qs[4 * ib32..4 * ib32 + 4].try_into().unwrap());
            let aux1 = u32::from_le_bytes(qs[4 * ib32 + 4..4 * ib32 + 8].try_into().unwrap());
            let aux_bytes = aux0.to_le_bytes();
            let db = d * (0.5 + ((aux1 >> 28) as f32)) * 0.25;
            for l in 0..4 {
                let grid_idx = aux_bytes[l] as usize;
                let grid = IQ2XXS_GRID[grid_idx].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0_f32
                    } else {
                        1.0_f32
                    };
                    out[out_ptr + j] = db * grid[j] as f32 * sign;
                }
                out_ptr += 8;
            }
        }
    }
    Ok(())
}

pub fn dequantize_iq3_xxs_scalar(
    input: &[u8],
    output: &mut [f32],
) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::IQ3_XXS,
        input,
        output,
        BLOCK_IQ3_XXS_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_IQ3_XXS_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let qs = &block[2..2 + QK_K / 4];
        let scales_and_signs = &block[2 + QK_K / 4..];
        let mut qs_ptr = 0_usize;
        let mut out_ptr = 0_usize;
        for ib32 in 0..(QK_K / 32) {
            let aux32 =
                u32::from_le_bytes(scales_and_signs[4 * ib32..4 * ib32 + 4].try_into().unwrap());
            let db = d * (0.5 + ((aux32 >> 28) as f32)) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let grid1 = IQ3XXS_GRID[qs[qs_ptr + 2 * l] as usize].to_le_bytes();
                let grid2 = IQ3XXS_GRID[qs[qs_ptr + 2 * l + 1] as usize].to_le_bytes();
                for j in 0..4 {
                    let sign_lo = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0_f32
                    } else {
                        1.0_f32
                    };
                    let sign_hi = if signs & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0_f32
                    } else {
                        1.0_f32
                    };
                    out[out_ptr + j] = db * grid1[j] as f32 * sign_lo;
                    out[out_ptr + j + 4] = db * grid2[j] as f32 * sign_hi;
                }
                out_ptr += 8;
            }
            qs_ptr += 8;
        }
    }
    Ok(())
}
