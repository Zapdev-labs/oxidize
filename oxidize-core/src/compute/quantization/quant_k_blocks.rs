use super::*;

/// llama.cpp `nearest_int` — fast round-to-nearest for quant heuristics.
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12_582_912.0;
    (val.to_bits() & 0x007f_ffff) as i32 - 0x0040_0000
}

/// Port of llama.cpp `make_qkx1_quants` (ggml-quants.c).
fn make_qkx1_quants(x: &[f32], l: &mut [u8], the_min: &mut f32, ntry: i32, alpha: f32) -> f32 {
    debug_assert_eq!(x.len(), l.len());
    let n = x.len();
    let nmax = 15;

    let mut min = x[0];
    let mut max = x[0];
    for &v in &x[1..] {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    if max == min {
        l.fill(0);
        *the_min = 0.0;
        return 0.0;
    }
    if min > 0.0 {
        min = 0.0;
    }

    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;

    for _ in 0..ntry {
        let mut sumlx = 0.0_f32;
        let mut suml2 = 0_i32;
        let mut did_change = false;
        for (i, &xv) in x.iter().enumerate() {
            let mut ql = nearest_int(iscale * (xv - min));
            ql = ql.clamp(0, nmax);
            if l[i] != ql as u8 {
                l[i] = ql as u8;
                did_change = true;
            }
            sumlx += (xv - min) * ql as f32;
            suml2 += ql * ql;
        }
        if suml2 > 0 {
            scale = sumlx / suml2 as f32;
        }
        let mut sum = 0.0_f32;
        for (i, &xv) in x.iter().enumerate() {
            sum += xv - scale * l[i] as f32;
        }
        min = alpha * min + (1.0 - alpha) * sum / n as f32;
        if min > 0.0 {
            min = 0.0;
        }
        iscale = 1.0 / scale;
        if !did_change {
            break;
        }
    }

    *the_min = -min;
    scale
}

/// llama.cpp-compatible Q4_K block quantizer (`quantize_row_q4_K_ref` with make_qkx1).
pub fn quantize_q4_k_scalar(
    target: GgufQuantizationType,
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK_K) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: target,
            expected_multiple: QK_K,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK_K) * BLOCK_Q4_K_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: target,
            expected: (input.len() / QK_K) * BLOCK_Q4_K_SIZE,
            actual: output.len(),
        });
    }

    let mut l = [0_u8; QK_K];
    let mut mins = [0.0_f32; QK_K / 32];
    let mut scales = [0.0_f32; QK_K / 32];

    for (in_block, out_block) in input
        .chunks_exact(QK_K)
        .zip(output.chunks_exact_mut(BLOCK_Q4_K_SIZE))
    {
        let mut max_scale = 0.0_f32;
        let mut max_min = 0.0_f32;
        for j in 0..QK_K / 32 {
            let chunk = &in_block[32 * j..32 * j + 32];
            let l_chunk = &mut l[32 * j..32 * j + 32];
            scales[j] = make_qkx1_quants(chunk, l_chunk, &mut mins[j], 5, 0.5);
            if scales[j] > max_scale {
                max_scale = scales[j];
            }
            if mins[j] > max_min {
                max_min = mins[j];
            }
        }

        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

        out_block[4..16].fill(0);
        for j in 0..QK_K / 32 {
            let ls = nearest_int(inv_scale * scales[j]).clamp(0, 63) as u8;
            let lm = nearest_int(inv_min * mins[j]).clamp(0, 63) as u8;
            if j < 4 {
                out_block[4 + j] = ls;
                out_block[4 + j + 4] = lm;
            } else {
                out_block[4 + j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                out_block[4 + j - 4] |= (ls >> 4) << 6;
                out_block[4 + j] |= (lm >> 4) << 6;
            }
        }

        out_block[0..2].copy_from_slice(&f32_to_f16_bits(max_scale / 63.0).to_le_bytes());
        out_block[2..4].copy_from_slice(&f32_to_f16_bits(max_min / 63.0).to_le_bytes());

        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &out_block[4..16]);
            let d = f16_le_to_f32(&out_block[0..2]) * sc as f32;
            if d == 0.0 {
                continue;
            }
            let dm = f16_le_to_f32(&out_block[2..4]) * m as f32;
            for ii in 0..32 {
                let ql = nearest_int((in_block[32 * j + ii] + dm) / d).clamp(0, 15) as u8;
                l[32 * j + ii] = ql;
            }
        }

        out_block[16..144].fill(0);
        for j in (0..QK_K).step_by(64) {
            for l_idx in 0..32 {
                out_block[16 + (j / 64) * 32 + l_idx] = l[j + l_idx] | (l[j + l_idx + 32] << 4);
            }
        }
    }

    Ok(())
}

/// Nearest index into the (sorted, asymmetric) `KVALUES_IQ4NL` codebook.
///
/// The codebook is deliberately asymmetric: scaling a finite Gaussian sample so
/// its max-magnitude entry maps to a fixed value yields an asymmetric
/// distribution, so a symmetric table reconstructs poorly (see ikawrakow's
/// IQ4_NL design notes). Linear scan over 16 entries is cheap and branchless.
#[inline]
fn best_index_iq4nl(value: f32) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for (idx, &v) in KVALUES_IQ4NL.iter().enumerate() {
        let dist = (value - v as f32).abs();
        if dist < best_dist {
            best_dist = dist;
            best = idx;
        }
    }
    best
}

/// IQ4_XS encoder (inverse of [`dequantize_iq4_xs_scalar`]).
///
/// Non-linear 4-bit at ~4.25 bpw: super-block of 256 = 8 sub-blocks of 32, each
/// reconstructed as `d * ls * KVALUES_IQ4NL[q]` with a 6-bit signed sub-block
/// scale `ls` and a single f16 super-scale `d`. `weights` (when provided) is the
/// per-value importance used to steer error away from activation-heavy columns;
/// `None` falls back to the `x²` heuristic. The per-sub-block scale search is the
/// `ntry` heuristic from ggml — deliberately *not* an exact RMSE solve, which is
/// known to hurt observed quality.
pub fn quantize_iq4_xs(
    input: &[f32],
    weights: Option<&[f32]>,
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK_K) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: GgufQuantizationType::IQ4_XS,
            expected_multiple: QK_K,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK_K) * BLOCK_IQ4_XS_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::IQ4_XS,
            expected: (input.len() / QK_K) * BLOCK_IQ4_XS_SIZE,
            actual: output.len(),
        });
    }
    if let Some(w) = weights
        && w.len() != input.len()
    {
        return Err(QuantizationError::InvalidImportanceMatrix {
            reason: "matrix length must match input value count",
        });
    }

    const SUB_BLOCKS: usize = QK_K / 32;
    const NTRY: i32 = 7;
    let val0 = KVALUES_IQ4NL[0] as f32;

    let mut scales = [0.0_f32; SUB_BLOCKS];
    let mut l_idx = [0_u8; QK_K];
    let mut weight = [0.0_f32; 32];

    for (block_idx, (in_block, out_block)) in input
        .chunks_exact(QK_K)
        .zip(output.chunks_exact_mut(BLOCK_IQ4_XS_SIZE))
        .enumerate()
    {
        let block_weights = weights.map(|w| &w[block_idx * QK_K..block_idx * QK_K + QK_K]);

        let mut sigma2 = 0.0_f32;
        for &v in in_block {
            sigma2 += v * v;
        }
        sigma2 *= 2.0 / QK_K as f32;

        out_block.fill(0);

        for sb in 0..SUB_BLOCKS {
            let xb = &in_block[sb * 32..sb * 32 + 32];
            match block_weights {
                Some(qw) => {
                    let qw = &qw[sb * 32..sb * 32 + 32];
                    for j in 0..32 {
                        weight[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
                    }
                }
                None => {
                    for j in 0..32 {
                        weight[j] = xb[j] * xb[j];
                    }
                }
            }

            let mut amax = 0.0_f32;
            let mut max = 0.0_f32;
            for &v in xb {
                let ax = v.abs();
                if ax > amax {
                    amax = ax;
                    max = v;
                }
            }
            if amax < 1.0e-8 {
                scales[sb] = 0.0;
                for j in 0..32 {
                    l_idx[sb * 32 + j] = 0;
                }
                continue;
            }

            // Evaluate a candidate inverse-scale: returns (Σ w·q·x, Σ w·q²).
            let eval = |inv_scale: f32| -> (f32, f32) {
                let mut sumqx = 0.0_f32;
                let mut sumq2 = 0.0_f32;
                for j in 0..32 {
                    let q = KVALUES_IQ4NL[best_index_iq4nl(inv_scale * xb[j])] as f32;
                    let w = weight[j];
                    sumqx += w * q * xb[j];
                    sumq2 += w * q * q;
                }
                (sumqx, sumq2)
            };
            let inv0 = -val0 / max;
            let (sumqx, sumq2) = eval(inv0);
            let mut best = if sumq2 > 0.0 { sumqx * sumqx / sumq2 } else { 0.0 };
            let mut best_scale = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
            for itry in -NTRY..=NTRY {
                let inv_scale = (itry as f32 + val0) / max;
                let (sx, s2) = eval(inv_scale);
                if s2 > 0.0 && sx * sx > best * s2 {
                    best = sx * sx / s2;
                    best_scale = sx / s2;
                }
            }

            scales[sb] = best_scale;
            let inv_final = if best_scale != 0.0 { 1.0 / best_scale } else { 0.0 };
            for j in 0..32 {
                l_idx[sb * 32 + j] = best_index_iq4nl(inv_final * xb[j]) as u8;
            }
        }

        // Quantize the eight sub-block scales against one f16 super-scale into
        // signed 6-bit values, matching the decoder's `dl = d * (ls - 32)`.
        let mut amax_scale = 0.0_f32;
        let mut max_scale = 0.0_f32;
        for &s in &scales {
            let a = s.abs();
            if a > amax_scale {
                amax_scale = a;
                max_scale = s;
            }
        }
        if amax_scale == 0.0 {
            continue;
        }

        let d_super = -max_scale / 32.0;
        let inv_super = 1.0 / d_super;
        let mut scales_h: u16 = 0;
        for sb in 0..SUB_BLOCKS {
            let ls = nearest_int(inv_super * scales[sb]).clamp(-32, 31);
            let si = (ls + 32) as u16;
            out_block[4 + sb / 2] |= ((si & 0xf) as u8) << (4 * (sb % 2));
            scales_h |= (si >> 4) << (2 * sb);
        }
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d_super).to_le_bytes());
        out_block[2..4].copy_from_slice(&scales_h.to_le_bytes());

        for sb in 0..SUB_BLOCKS {
            let qoff = 8 + sb * 16;
            for k in 0..16 {
                let lo = l_idx[sb * 32 + k] & 0xf;
                let hi = l_idx[sb * 32 + k + 16] & 0xf;
                out_block[qoff + k] = lo | (hi << 4);
            }
        }
    }

    Ok(())
}

/// IQ4_XS encoder without an importance matrix (uses the `x²` weight heuristic).
pub fn quantize_iq4_xs_scalar(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
    quantize_iq4_xs(input, None, output)
}

pub(super) fn quantize_k_packed_scalar(
    quantization: GgufQuantizationType,
    input: &[f32],
    output: &mut [u8],
    block_size: usize,
    bits: usize,
    zero_point: f32,
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK_K) {
        return Err(QuantizationError::InvalidInputLength {
            quantization,
            expected_multiple: QK_K,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK_K) * block_size {
        return Err(QuantizationError::InvalidOutputLength {
            quantization,
            expected: (input.len() / QK_K) * block_size,
            actual: output.len(),
        });
    }

    let max_q = ((1_u32 << bits) - 1) as f32;
    let positive_span = zero_point.max(max_q - zero_point);
    for (in_block, out_block) in input
        .chunks_exact(QK_K)
        .zip(output.chunks_exact_mut(block_size))
    {
        let max_abs = in_block.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        let d = if max_abs == 0.0 {
            0.0
        } else {
            max_abs / positive_span
        };
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out_block[2..].fill(0);

        for (i, value) in in_block.iter().enumerate() {
            let q = if d == 0.0 {
                zero_point.round() as u32
            } else {
                ((value / d) + zero_point).round().clamp(0.0, max_q) as u32
            };
            write_bits(&mut out_block[2..], i, bits, q);
        }
    }

    Ok(())
}

pub fn dequantize_q2_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q2_K,
        input,
        output,
        BLOCK_Q2_K_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_Q2_K_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[80..82]);
        let min = f16_le_to_f32(&block[82..84]);
        let scales = &block[0..16];
        let qs = &block[16..80];
        let mut q_ptr = 0;
        let mut is = 0;
        // Match llama.cpp dequantize_row_q2_K: each outer iteration consumes a
        // fresh 32-byte slab of qs (qs_base advances by 32), and the four inner
        // iterations re-read the same slab with shifts 0/2/4/6.
        for outer in 0..2 {
            let qs_base = outer * 32;
            for _ in 0..4 {
                let sc1 = scales[is];
                let dl1 = d * ((sc1 & 0xF) as f32);
                let ml1 = min * ((sc1 >> 4) as f32);
                is += 1;
                let sc2 = scales[is];
                let dl2 = d * ((sc2 & 0xF) as f32);
                let ml2 = min * ((sc2 >> 4) as f32);
                is += 1;
                let shift = ((is / 2 - 1) % 4) * 2;
                for l in 0..16 {
                    out[q_ptr + l] = dl1 * (((qs[qs_base + l] >> shift) & 3) as f32) - ml1;
                }
                for l in 0..16 {
                    out[q_ptr + 16 + l] =
                        dl2 * (((qs[qs_base + 16 + l] >> shift) & 3) as f32) - ml2;
                }
                q_ptr += 32;
            }
        }
    }
    Ok(())
}

pub fn dequantize_q3_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q3_K_S,
        input,
        output,
        BLOCK_Q3_K_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_Q3_K_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d_all = f16_le_to_f32(&block[108..110]);
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let mut scales_raw = [0u32; 4];
        scales_raw[0] = u32::from_le_bytes([block[96], block[97], block[98], block[99]]);
        scales_raw[1] = u32::from_le_bytes([block[100], block[101], block[102], block[103]]);
        scales_raw[2] = u32::from_le_bytes([block[104], block[105], block[106], block[107]]);
        let tmp = scales_raw[2];
        scales_raw[2] = ((scales_raw[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4);
        scales_raw[3] = ((scales_raw[1] >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4);
        scales_raw[0] = (scales_raw[0] & 0x0F0F0F0F) | ((tmp & 0x03030303) << 4);
        scales_raw[1] = (scales_raw[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4);
        let mut scale_bytes = [0u8; 16];
        for (i, word) in scales_raw.iter().enumerate() {
            scale_bytes[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }

        let mut q_ptr = 0;
        let mut is = 0;
        let mut m = 1u8;
        for _ in 0..2 {
            for _ in 0..4 {
                let dl = d_all * (scale_bytes[is] as i8 as i32 - 32) as f32;
                is += 1;
                let shift = ((is - 1) % 4) * 2;
                for l in 0..16 {
                    let qv = ((qs[l] >> shift) & 3) as i32;
                    let hbit = if (hmask[l] & m) != 0 { 0 } else { 4 };
                    out[q_ptr + l] = dl * ((qv - hbit) as f32);
                }
                let dl2 = d_all * (scale_bytes[is] as i8 as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let qv = ((qs[l + 16] >> shift) & 3) as i32;
                    let hbit = if (hmask[l + 16] & m) != 0 { 0 } else { 4 };
                    out[q_ptr + 16 + l] = dl2 * ((qv - hbit) as f32);
                }
                q_ptr += 32;
                m <<= 1;
            }
        }
    }
    Ok(())
}

#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

pub fn dequantize_q4_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q4_K_S,
        input,
        output,
        BLOCK_Q4_K_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_Q4_K_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let min = f16_le_to_f32(&block[2..4]);
        let scales = &block[4..16];
        let qs = &block[16..144];
        let mut out_ptr = 0;
        let mut is = 0;
        // Each of 4 group_pairs covers 64 output values (32 from low nibbles + 32 from high nibbles)
        // and reads from qs[group_pair*32 .. group_pair*32+32] — must advance q_base.
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
    Ok(())
}

pub fn dequantize_q5_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q5_K_S,
        input,
        output,
        BLOCK_Q5_K_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_Q5_K_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let min = f16_le_to_f32(&block[2..4]);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let mut q_ptr = 0;
        let mut is = 0;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for _ in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let min1 = min * m1 as f32;
            let d2 = d * sc2 as f32;
            let min2 = min * m2 as f32;
            for l in 0..32 {
                let qv1 = (qs[l] & 0xF) as u32 + if (qh[l] & u1) != 0 { 16 } else { 0 };
                out[q_ptr + l] = d1 * (qv1 as f32) - min1;
            }
            for l in 0..32 {
                let qv2 = (qs[l] >> 4) as u32 + if (qh[l] & u2) != 0 { 16 } else { 0 };
                out[q_ptr + 32 + l] = d2 * (qv2 as f32) - min2;
            }
            q_ptr += 64;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(())
}

pub fn dequantize_q6_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q6_K,
        input,
        output,
        BLOCK_Q6_K_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_Q6_K_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[208..210]);
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        // QK_K=256 values are processed in two 128-element groups. Each group
        // advances into ql/qh/scales (ql+=64, qh+=32, scales+=8), matching the
        // reference dequantize_row_q6_K. Without these per-group offsets the
        // second half of every block is decoded from the first half's data.
        let mut q_ptr = 0;
        for group in 0..2 {
            let ql_off = group * 64;
            let qh_off = group * 32;
            let sc_off = group * 8;
            for l in 0..32 {
                let is = l / 16;
                let q1 =
                    ((ql[ql_off + l] & 0xF) as i32 | (((qh[qh_off + l] & 3) as i32) << 4)) - 32;
                let q2 = ((ql[ql_off + l + 32] & 0xF) as i32
                    | ((((qh[qh_off + l] >> 2) & 3) as i32) << 4))
                    - 32;
                let q3 = ((ql[ql_off + l] >> 4) as i32
                    | ((((qh[qh_off + l] >> 4) & 3) as i32) << 4))
                    - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) as i32
                    | ((((qh[qh_off + l] >> 6) & 3) as i32) << 4))
                    - 32;
                out[q_ptr + l] = d * sc[sc_off + is] as i8 as f32 * q1 as f32;
                out[q_ptr + 32 + l] = d * sc[sc_off + is + 2] as i8 as f32 * q2 as f32;
                out[q_ptr + 64 + l] = d * sc[sc_off + is + 4] as i8 as f32 * q3 as f32;
                out[q_ptr + 96 + l] = d * sc[sc_off + is + 6] as i8 as f32 * q4 as f32;
            }
            q_ptr += 128;
        }
    }
    Ok(())
}

/// IQ4_XS dequantization (ggml `dequantize_row_iq4_xs`). Block = 136 bytes for
/// 256 values: f16 d, u16 scales_h, 4×u8 scales_l, 128×u8 qs (two 4-bit nibbles
/// each). Eight 32-value sub-blocks; per-subblock 6-bit scale (ls-32) selects a
/// scale, and each nibble indexes the shared nonlinear IQ4_NL codebook.
pub fn dequantize_iq4_xs_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::IQ4_XS,
        input,
        output,
        BLOCK_IQ4_XS_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_IQ4_XS_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..136];
        for ib in 0..(QK_K / 32) {
            let ls_l = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32;
            let ls_h = (((scales_h >> (2 * ib)) & 3) as i32) << 4;
            let dl = d * ((ls_l | ls_h) - 32) as f32;
            let qoff = ib * 16;
            let ooff = ib * 32;
            for j in 0..16 {
                let b = qs[qoff + j];
                out[ooff + j] = dl * KVALUES_IQ4NL[(b & 0xf) as usize] as f32;
                out[ooff + j + 16] = dl * KVALUES_IQ4NL[(b >> 4) as usize] as f32;
            }
        }
    }
    Ok(())
}

/// IQ3_S dequantization (ggml `dequantize_row_iq3_s`). Block = 110 bytes for
/// 256 values: f16 d, 64×u8 qs, 8×u8 qh, 32×u8 signs, 4×u8 scales. Each 3-bit
/// index (8th bit from qh) selects a 4-value entry of the iq3s_grid codebook;
/// the sign byte flips signs per kmask; per-32 sub-block scale = d*(1+2*s).
pub fn dequantize_iq3_s_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::IQ3_S,
        input,
        output,
        BLOCK_IQ3_S_SIZE,
        QK_K,
    )?;
    let grid = |idx: usize, j: usize| -> f32 { ((IQ3S_GRID[idx] >> (8 * j)) & 0xff) as f32 };
    for (block, out) in input
        .chunks_exact(BLOCK_IQ3_S_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let qs = &block[2..66]; // 64 bytes
        let qh = &block[66..74]; // 8 bytes
        let signs = &block[74..106]; // 32 bytes
        let scales = &block[106..110]; // 4 bytes
        let mut qs_o = 0usize; // index into qs
        let mut qh_o = 0usize; // index into qh
        let mut sg_o = 0usize; // index into signs
        let mut y = 0usize; // index into out
        let mut ib32 = 0usize;
        while ib32 < QK_K / 32 {
            let db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf) as i32) as f32;
            let db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4) as i32) as f32;
            // first 32: uses qh[qh_o], qs_o..qs_o+8, signs sg_o..sg_o+4
            for l in 0..4 {
                let h = qh[qh_o] as usize;
                let i1 = qs[qs_o + 2 * l] as usize | ((h << (8 - 2 * l)) & 256);
                let i2 = qs[qs_o + 2 * l + 1] as usize | ((h << (7 - 2 * l)) & 256);
                let s = signs[sg_o + l];
                for j in 0..4 {
                    let f1 = if s & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    let f2 = if s & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out[y + j] = db1 * grid(i1, j) * f1;
                    out[y + j + 4] = db1 * grid(i2, j) * f2;
                }
                y += 8;
            }
            qs_o += 8;
            sg_o += 4;
            // second 32: uses qh[qh_o+1], next qs_o..qs_o+8, signs sg_o..sg_o+4
            for l in 0..4 {
                let h = qh[qh_o + 1] as usize;
                let i1 = qs[qs_o + 2 * l] as usize | ((h << (8 - 2 * l)) & 256);
                let i2 = qs[qs_o + 2 * l + 1] as usize | ((h << (7 - 2 * l)) & 256);
                let s = signs[sg_o + l];
                for j in 0..4 {
                    let f1 = if s & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    let f2 = if s & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out[y + j] = db2 * grid(i1, j) * f1;
                    out[y + j + 4] = db2 * grid(i2, j) * f2;
                }
                y += 8;
            }
            qh_o += 2;
            qs_o += 8;
            sg_o += 4;
            ib32 += 2;
        }
    }
    Ok(())
}

pub fn dequantize_q8_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q8_0,
        input,
        output,
        BLOCK_Q8_K_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_Q8_K_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        for j in 0..QK_K {
            out[j] = d * (block[4 + j] as i8) as f32;
        }
    }
    Ok(())
}
