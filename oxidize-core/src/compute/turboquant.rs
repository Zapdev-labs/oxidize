/// TurboQuant — fast block-wise INT4/INT8 quantization for CPU inference.
/// Uses 32-element blocks with per-block scale, optimized for GEMV.
pub const TURBOQUANT_BLOCK_SIZE: usize = 32;
pub const TURBOQUANT_BITS: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboQuantType {
    Int4,
    Int8,
}

/// Block-wise quantized weights: [scale: f32, q0..qN] per block.
#[derive(Debug, Clone, PartialEq)]
pub struct TurboQuantData {
    pub qtype: TurboQuantType,
    pub blocks: Vec<TurboQuantBlock>,
    pub cols: usize,
    pub rows: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TurboQuantBlock {
    pub scale: f32,
    pub values: Vec<u8>,
}

impl TurboQuantData {
    pub fn quantize_f32(src: &[f32], rows: usize, cols: usize, qtype: TurboQuantType) -> Self {
        let block_size = TURBOQUANT_BLOCK_SIZE;
        let bits = if qtype == TurboQuantType::Int4 { 4 } else { 8 };
        let max_val = (1 << (bits - 1)) - 1;
        let blocks_per_row = (cols + block_size - 1) / block_size;
        let total_blocks = rows * blocks_per_row;
        let mut blocks = Vec::with_capacity(total_blocks);

        for r in 0..rows {
            for b in 0..blocks_per_row {
                let start = r * cols + b * block_size;
                let end = (start + block_size).min(r * cols + cols);
                let chunk = &src[start..end];
                let mut max_abs = 0.0_f32;
                for &v in chunk { max_abs = max_abs.max(v.abs()); }
                let scale = if max_abs > 0.0 { max_abs / max_val as f32 } else { 1.0 };
                let mut packed = vec![0u8; if bits == 4 { block_size / 2 } else { block_size }];
                for (i, &v) in chunk.iter().enumerate() {
                    let q = (v / scale).round().clamp(-(max_val as f32), max_val as f32) as i8;
                    let uq = (q + max_val as i8) as u8;
                    if bits == 4 {
                        let byte_idx = i / 2;
                        let nibble = i % 2;
                        if nibble == 0 { packed[byte_idx] |= uq & 0x0F; } else { packed[byte_idx] |= (uq & 0x0F) << 4; }
                    } else {
                        packed[i] = uq;
                    }
                }
                blocks.push(TurboQuantBlock { scale, values: packed });
            }
        }
        Self { qtype, blocks, cols, rows }
    }

    pub fn dequantize_f32(&self, out: &mut [f32]) {
        let block_size = TURBOQUANT_BLOCK_SIZE;
        let bits = if self.qtype == TurboQuantType::Int4 { 4 } else { 8 };
        let max_val = (1 << (bits - 1)) - 1;
        let blocks_per_row = (self.cols + block_size - 1) / block_size;
        for r in 0..self.rows {
            for b in 0..blocks_per_row {
                let block = &self.blocks[r * blocks_per_row + b];
                let start = r * self.cols + b * block_size;
                let end = (start + block_size).min(r * self.cols + self.cols);
                for i in 0..(end - start) {
                    let q = if bits == 4 {
                        let byte = block.values[i / 2];
                        if i % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F }
                    } else {
                        block.values[i]
                    };
                    let val = (q as f32 - max_val as f32) * block.scale;
                    out[start + i] = val;
                }
            }
        }
    }

    pub fn gemv(input: &[f32], tq: &TurboQuantData, out: &mut [f32]) {
        let block_size = TURBOQUANT_BLOCK_SIZE;
        let bits = if tq.qtype == TurboQuantType::Int4 { 4 } else { 8 };
        let max_val = (1 << (bits - 1)) - 1) as f32;
        let blocks_per_row = (tq.cols + block_size - 1) / block_size;
        assert_eq!(input.len(), tq.cols);
        assert_eq!(out.len(), tq.rows);
        for r in 0..tq.rows {
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let block = &tq.blocks[r * blocks_per_row + b];
                let col_start = b * block_size;
                let col_end = (col_start + block_size).min(tq.cols);
                for (j, col) in (col_start..col_end).enumerate() {
                    let q = if bits == 4 {
                        let byte = block.values[j / 2];
                        if j % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F }
                    } else {
                        block.values[j]
                    };
                    let val = (q as f32 - max_val) * block.scale;
                    sum += input[col] * val;
                }
            }
            out[r] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_int4() {
        let src = vec![1.0, -2.0, 3.5, -4.0, 0.5, -0.1, 2.0, -3.0,
                       1.0, -2.0, 3.5, -4.0, 0.5, -0.1, 2.0, -3.0,
                       1.0, -2.0, 3.5, -4.0, 0.5, -0.1, 2.0, -3.0,
                       1.0, -2.0, 3.5, -4.0, 0.5, -0.1, 2.0, -3.0,
                       1.0, -2.0, 3.5, -4.0, 0.5, -0.1, 2.0, -3.0,
                       1.0, -2.0, 3.5, -4.0, 0.5, -0.1, 2.0, -3.0,
                       1.0, -2.0, 3.5, -4.0, 0.5, -0.1, 2.0, -3.0,
                       1.0, -2.0, 3.5, -4.0, 0.5, -0.1, 2.0, -3.0];
        let tq = TurboQuantData::quantize_f32(&src, 2, 32, TurboQuantType::Int4);
        let mut out = vec![0.0_f32; 64];
        tq.dequantize_f32(&mut out);
        for i in 0..src.len() {
            assert!((src[i] - out[i]).abs() < 0.5, "roundtrip failed at {}: {} vs {}", i, src[i], out[i]);
        }
    }

    #[test]
    fn gemv_int4() {
        let src: Vec<f32> = (0..64).map(|i| ((i % 7) as f32 - 3.0)).collect();
        let tq = TurboQuantData::quantize_f32(&src, 2, 32, TurboQuantType::Int4);
        let input = vec![1.0_f32; 32];
        let mut out = vec![0.0_f32; 2];
        TurboQuantData::gemv(&input, &tq, &mut out);
        let mut expected = vec![0.0_f32; 2];
        for r in 0..2 { for c in 0..32 { expected[r] += input[c] * src[r * 32 + c]; } }
        for i in 0..2 { assert!((out[i] - expected[i]).abs() < 1.0, "gemv mismatch: {} vs {}", out[i], expected[i]); }
    }
}
