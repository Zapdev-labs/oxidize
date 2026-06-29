use rayon::prelude::*;

use crate::config::FinetuneConfig;
use crate::error::{FinetuneError, Result};
use crate::lora::{LoRAAdapter, LoRATarget};

// NF4 lookup table — 16 quantisation levels on the normal-float grid.
// Values are ordered so that the index is the 4-bit codeword (0..=15).
const NF4_TABLE: [f32; 16] = [
    -1.000_000_0,
    -0.696_192_8,
    -0.525_073_0,
    -0.394_964_7,
    -0.284_464_7,
    -0.184_774_5,
    -0.091_776_0,
    0.000_000_0,
    0.079_578_5,
    0.160_957_5,
    0.246_155_6,
    0.337_915_2,
    0.440_709_8,
    0.562_617_0,
    0.722_956_8,
    1.000_000_0,
];

/// Find the NF4 codeword (0..=15) closest to `v` in [-1, 1].
#[inline]
fn nearest_nf4(v: f32) -> u8 {
    let mut best = 0u8;
    let mut best_dist = (NF4_TABLE[0] - v).abs();
    for (i, &level) in NF4_TABLE.iter().enumerate().skip(1) {
        let d = (level - v).abs();
        if d < best_dist {
            best_dist = d;
            best = i as u8;
        }
    }
    best
}

/// NF4-quantized block covering up to `block_size` weight scalars.
///
/// Weights are normalised by `absmax` before quantisation so the full dynamic
/// range of [-1, 1] is used; `absmax` restores the original magnitude at dequant.
///
/// Nibble packing: element `i` is stored in
///   - bits 3..0 of `data[i/2]` when `i` is even,
///   - bits 7..4 of `data[i/2]` when `i` is odd.
#[derive(Debug, Clone)]
pub struct NF4Block {
    /// Packed nibbles; `len == ceil(n / 2)`.
    pub data: Vec<u8>,
    /// Number of weights encoded in this block.
    pub len: usize,
    /// Absolute maximum of the original weights (dequant scale).
    pub absmax: f32,
}

impl NF4Block {
    /// Quantize a slice of `f32` weights into one NF4 block.
    ///
    /// If `weights` is empty the block is empty but still valid.
    pub fn quantize(weights: &[f32]) -> Self {
        let n = weights.len();
        let absmax = weights
            .iter()
            .copied()
            .fold(0.0_f32, |acc, w| acc.max(w.abs()));

        // Avoid division by zero for an all-zero block.
        let inv = if absmax > 0.0 { 1.0 / absmax } else { 1.0 };

        let packed_len = n.div_ceil(2);
        let mut data = vec![0u8; packed_len];

        for (i, &w) in weights.iter().enumerate() {
            let normalised = (w * inv).clamp(-1.0, 1.0);
            let code = nearest_nf4(normalised);
            if i % 2 == 0 {
                data[i / 2] |= code & 0x0F;
            } else {
                data[i / 2] |= (code & 0x0F) << 4;
            }
        }

        Self {
            data,
            len: n,
            absmax,
        }
    }

    /// Restore the block to `f32` weights (approximate).
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let byte = self.data[i / 2];
            let nibble = if i % 2 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            out.push(NF4_TABLE[nibble as usize] * self.absmax);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// QLoRAAdapter
// ---------------------------------------------------------------------------

/// QLoRA adapter: the frozen base weight is stored in NF4 blocks, while the
/// trainable LoRA delta (A, B) lives in full f32.
///
/// Forward pass:
///   out[t] = W_dequant · xs[t]  +  scale * B · A · xs[t]
///
/// Backward pass: gradients flow only through the LoRA branch; the base is
/// not updated.
#[derive(Debug, Clone)]
pub struct QLoRAAdapter {
    /// Quantized frozen base weight, partitioned into `block_size`-element
    /// NF4 blocks in row-major order ([out_dim, in_dim] flattened).
    pub base_q: Vec<NF4Block>,
    /// Trainable LoRA delta.
    pub lora: LoRAAdapter,
    /// Number of weights per NF4 block (default 64).
    pub block_size: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    base_dequant: Option<Vec<f32>>,
}

impl QLoRAAdapter {
    /// Build a `QLoRAAdapter` from a dense base weight matrix.
    ///
    /// `base` must be row-major with shape [out_dim, in_dim] (i.e.
    /// `base.len() == out_dim * in_dim`).
    pub fn from_weights(
        base: &[f32],
        in_dim: usize,
        out_dim: usize,
        config: &FinetuneConfig,
    ) -> Self {
        assert_eq!(
            base.len(),
            out_dim * in_dim,
            "base weight length mismatch: got {} expected {}",
            base.len(),
            out_dim * in_dim,
        );

        let block_size = 64_usize;

        // Quantize the base weight in consecutive `block_size` chunks.
        let base_q: Vec<NF4Block> = base.chunks(block_size).map(NF4Block::quantize).collect();

        let lora = LoRAAdapter::new(LoRATarget::AttentionQ, in_dim, out_dim, config);

        Self {
            base_q,
            lora,
            block_size,
            in_dim,
            out_dim,
            base_dequant: None,
        }
    }

    /// Dequantize the full base weight matrix back to f32 ([out_dim * in_dim]).
    fn dequantize_base(&mut self) -> &[f32] {
        if self.base_dequant.is_none() {
            let total = self.out_dim * self.in_dim;
            let mut weights = Vec::with_capacity(total);
            for block in &self.base_q {
                weights.extend_from_slice(&block.dequantize());
            }
            weights.truncate(total);
            self.base_dequant = Some(weights);
        }
        self.base_dequant.as_ref().expect("base_dequant just initialized")
    }

    /// Run the full QLoRA forward pass for `count` input rows.
    ///
    /// Returns `[count, out_dim]` activations.
    ///
    /// Memory layout: both `xs` and the returned slice are row-major with
    /// strides `in_dim` and `out_dim` respectively.
    pub fn forward_batch(&mut self, xs: &[f32], count: usize) -> Result<Vec<f32>> {
        if xs.len() != count * self.in_dim {
            return Err(FinetuneError::Adapter(format!(
                "QLoRAAdapter::forward_batch: xs.len()={} != count*in_dim={}",
                xs.len(),
                count * self.in_dim,
            )));
        }

        let (in_dim, out_dim) = (self.in_dim, self.out_dim);
        let base_w = self.dequantize_base();

        // Base matmul: out[t][o] = sum_i base_w[o * in_dim + i] * xs[t * in_dim + i]
        let mut outs = vec![0.0_f32; count * out_dim];
        outs.par_chunks_mut(out_dim)
            .zip(xs.par_chunks(in_dim))
            .for_each(|(out_row, x_row)| {
                for o in 0..out_dim {
                    let w_row = &base_w[o * in_dim..(o + 1) * in_dim];
                    out_row[o] = dot(w_row, x_row);
                }
            });

        // LoRA addition: out += scale * B A x  (in place)
        self.lora.forward_batch(xs, &mut outs, count)?;

        Ok(outs)
    }

    /// Accumulate LoRA gradients for a batch. The base weight is frozen so
    /// only `lora.grad_a` / `lora.grad_b` are updated.
    ///
    /// `grad_out` is the upstream gradient w.r.t. the full output,
    /// shape [count, out_dim].
    pub fn backward_batch(&mut self, xs: &[f32], grad_out: &[f32], count: usize) -> Result<()> {
        if xs.len() != count * self.in_dim || grad_out.len() != count * self.out_dim {
            return Err(FinetuneError::Adapter(format!(
                "QLoRAAdapter::backward_batch shape mismatch: xs={} grad_out={} count={} in={} out={}",
                xs.len(),
                grad_out.len(),
                count,
                self.in_dim,
                self.out_dim,
            )));
        }

        // Gradient only flows through the LoRA branch (base is frozen).
        self.lora.backward_batch(xs, grad_out, count)
    }

    /// AdamW step for the LoRA parameters with explicit beta/eps arguments.
    ///
    /// This delegates to the inner `LoRAAdapter::step` which uses the hardcoded
    /// beta1=0.9, beta2=0.999, eps=1e-8 from `fused::adamw_step`. The `beta1`,
    /// `beta2`, and `eps` arguments are accepted for API compatibility but the
    /// fused kernel's constants take precedence; pass the matching defaults to
    /// avoid confusion.
    pub fn adam_step(
        &mut self,
        step: usize,
        lr: f32,
        _beta1: f32,
        _beta2: f32,
        _eps: f32,
        wd: f32,
    ) {
        self.lora.step(lr, wd, step);
    }

    /// Zero accumulated LoRA gradients.
    pub fn zero_grad(&mut self) {
        self.lora.zero_grad();
    }

    /// Total trainable parameter count (LoRA only; base is quantized/frozen).
    pub fn trainable_param_count(&self) -> usize {
        self.lora.param_count()
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(rank: usize, alpha: f32) -> FinetuneConfig {
        FinetuneConfig {
            rank,
            alpha,
            ..Default::default()
        }
    }

    // --- NF4Block -----------------------------------------------------------

    #[test]
    fn nf4_roundtrip_all_table_values() {
        // A weight vector that already sits on every NF4 grid point should
        // survive quantisation → dequantisation exactly.
        let block = NF4Block::quantize(&NF4_TABLE);
        let rt = block.dequantize();
        assert_eq!(rt.len(), NF4_TABLE.len());
        for (orig, got) in NF4_TABLE.iter().zip(rt.iter()) {
            assert!(
                (orig - got).abs() < 1e-6,
                "table roundtrip: orig={orig} got={got}"
            );
        }
    }

    #[test]
    fn nf4_quantize_error_bounded() {
        // Random-ish weights in [-1, 1]; max absolute error should be well
        // below 0.1 for these table spacings.
        let weights: Vec<f32> = (0..128).map(|i| ((i as f32 * 0.137).sin())).collect();
        let block = NF4Block::quantize(&weights);
        let rt = block.dequantize();
        assert_eq!(rt.len(), weights.len());
        let max_err = weights
            .iter()
            .zip(rt.iter())
            .map(|(w, r)| (w - r).abs())
            .fold(0.0_f32, f32::max);
        // The widest gap between adjacent NF4 levels is ~0.18 (at the extremes),
        // so the worst-case half-interval error is ~0.09; we allow a comfortable
        // 0.20 headroom here.
        assert!(
            max_err < 0.20,
            "max NF4 quant error {max_err} exceeds bound"
        );
    }

    #[test]
    fn nf4_empty_block() {
        let block = NF4Block::quantize(&[]);
        assert_eq!(block.len, 0);
        assert_eq!(block.dequantize().len(), 0);
    }

    #[test]
    fn nf4_single_element() {
        let block = NF4Block::quantize(&[0.5_f32]);
        let rt = block.dequantize();
        assert_eq!(rt.len(), 1);
        assert!((rt[0] - 0.5).abs() < 0.07, "single-element rt={}", rt[0]);
    }

    #[test]
    fn nf4_odd_length_block() {
        let weights: Vec<f32> = (0..7).map(|i| i as f32 * 0.1 - 0.3).collect();
        let block = NF4Block::quantize(&weights);
        let rt = block.dequantize();
        assert_eq!(rt.len(), weights.len());
    }

    #[test]
    fn nf4_all_zeros() {
        let weights = vec![0.0_f32; 16];
        let block = NF4Block::quantize(&weights);
        let rt = block.dequantize();
        for v in rt {
            assert_eq!(v, 0.0);
        }
    }

    // --- QLoRAAdapter -------------------------------------------------------

    fn make_qlora(in_dim: usize, out_dim: usize) -> QLoRAAdapter {
        let cfg = make_config(4, 8.0);
        // Simple identity-ish base weight.
        let base: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i as f32 * 0.07).sin()) * 0.5)
            .collect();
        QLoRAAdapter::from_weights(&base, in_dim, out_dim, &cfg)
    }

    #[test]
    fn qlora_forward_batch_shape() {
        let mut q = make_qlora(8, 16);
        let xs: Vec<f32> = (0..3 * 8).map(|i| (i as f32 * 0.1).sin()).collect();
        let out = q.forward_batch(&xs, 3).expect("forward_batch");
        assert_eq!(out.len(), 3 * 16);
    }

    #[test]
    fn qlora_forward_single_consistent_with_batch() {
        let mut q = make_qlora(8, 16);
        let xs: Vec<f32> = (0..3 * 8).map(|i| (i as f32 * 0.23).cos()).collect();

        let batched = q.forward_batch(&xs, 3).expect("batch");

        for t in 0..3 {
            let single = q.forward_batch(&xs[t * 8..(t + 1) * 8], 1).expect("single");
            for (b, s) in batched[t * 16..(t + 1) * 16].iter().zip(single.iter()) {
                assert!((b - s).abs() < 1e-5, "t={t} b={b} s={s}");
            }
        }
    }

    #[test]
    fn qlora_backward_updates_lora_grads() {
        let mut q = make_qlora(8, 16);
        let xs = vec![1.0_f32; 8];
        let grad = vec![1.0_f32; 16];
        q.backward_batch(&xs, &grad, 1).expect("backward");
        let any_nonzero =
            q.lora.grad_a.iter().any(|&v| v != 0.0) || q.lora.grad_b.iter().any(|&v| v != 0.0);
        assert!(any_nonzero, "backward produced all-zero gradients");
    }

    #[test]
    fn qlora_adam_step_changes_params() {
        let mut q = make_qlora(4, 8);
        // Seed lora.b so that grad_a also gets a non-zero gradient via the
        // grad_hidden = scale * B^T grad_out path.
        for (i, v) in q.lora.b.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.05;
        }
        let xs = vec![0.5_f32; 4];
        let grad = vec![1.0_f32; 8];
        q.backward_batch(&xs, &grad, 1).expect("backward");
        let a_before = q.lora.a.clone();
        let b_before = q.lora.b.clone();
        q.adam_step(1, 1e-3, 0.9, 0.999, 1e-8, 0.0);
        // Both A and B should be updated (B has grad regardless of A's init).
        let b_changed = q
            .lora
            .b
            .iter()
            .zip(b_before.iter())
            .any(|(new, old)| (new - old).abs() > 1e-9);
        let a_changed = q
            .lora
            .a
            .iter()
            .zip(a_before.iter())
            .any(|(new, old)| (new - old).abs() > 1e-9);
        assert!(b_changed, "adam_step did not update lora.b");
        assert!(a_changed, "adam_step did not update lora.a");
    }

    #[test]
    fn qlora_zero_grad() {
        let mut q = make_qlora(4, 8);
        let xs = vec![1.0_f32; 4];
        let grad = vec![1.0_f32; 8];
        q.backward_batch(&xs, &grad, 1).unwrap();
        q.zero_grad();
        assert!(q.lora.grad_a.iter().all(|&v| v == 0.0));
        assert!(q.lora.grad_b.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn qlora_from_weights_size_mismatch_panics() {
        let cfg = make_config(2, 4.0);
        let result = std::panic::catch_unwind(|| {
            QLoRAAdapter::from_weights(&[1.0_f32; 10], 4, 4, &cfg);
        });
        assert!(result.is_err(), "should panic on bad base weight size");
    }

    #[test]
    fn qlora_trainable_param_count() {
        let q = make_qlora(8, 16);
        // rank=4; A=[4,8]=32, B=[16,4]=64 => 96
        assert_eq!(q.trainable_param_count(), 96);
    }

    #[test]
    fn qlora_forward_wrong_xs_len_returns_err() {
        let mut q = make_qlora(8, 16);
        let result = q.forward_batch(&[0.0_f32; 7], 1);
        assert!(result.is_err());
    }

    #[test]
    fn qlora_backward_wrong_shape_returns_err() {
        let mut q = make_qlora(8, 16);
        let result = q.backward_batch(&[0.0_f32; 8], &[0.0_f32; 8], 1); // grad should be 16
        assert!(result.is_err());
    }
}
