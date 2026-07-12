use rayon::prelude::*;

use crate::config::FinetuneConfig;
use crate::error::{FinetuneError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoRATarget {
    OutputHead,
    AttentionQ,
    AttentionV,
    FfnGate,
    FfnUp,
}

impl LoRATarget {
    pub const fn name(self) -> &'static str {
        match self {
            Self::OutputHead => "OutputHead",
            Self::AttentionQ => "AttentionQ",
            Self::AttentionV => "AttentionV",
            Self::FfnGate => "FfnGate",
            Self::FfnUp => "FfnUp",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "OutputHead" => Some(Self::OutputHead),
            "AttentionQ" => Some(Self::AttentionQ),
            "AttentionV" => Some(Self::AttentionV),
            "FfnGate" => Some(Self::FfnGate),
            "FfnUp" => Some(Self::FfnUp),
            _ => None,
        }
    }
}

/// LoRA adapter trained over a frozen base projection (out = W x + scale * B A x).
///
/// All hot paths are batched: callers pass `count` activation rows at once so
/// the per-row work amortizes into cache-friendly parallel loops instead of
/// one rayon dispatch per token.
#[derive(Debug, Clone)]
pub struct LoRAAdapter {
    pub target: LoRATarget,
    pub in_dim: usize,
    pub out_dim: usize,
    pub rank: usize,
    pub scale: f32,
    /// Down projection, row-major [rank, in_dim].
    pub a: Vec<f32>,
    /// Up projection, row-major [out_dim, rank].
    pub b: Vec<f32>,
    pub grad_a: Vec<f32>,
    pub grad_b: Vec<f32>,
    pub adam_a_m: Vec<f32>,
    pub adam_a_v: Vec<f32>,
    pub adam_b_m: Vec<f32>,
    pub adam_b_v: Vec<f32>,
}

impl LoRAAdapter {
    pub fn new(target: LoRATarget, in_dim: usize, out_dim: usize, config: &FinetuneConfig) -> Self {
        let rank = config.rank.max(1);
        let scale = config.lora_scale();
        let mut a = vec![0.0_f32; rank * in_dim];
        init_lora_a(&mut a, rank, config.seed);
        Self {
            target,
            in_dim,
            out_dim,
            rank,
            scale,
            a,
            b: vec![0.0_f32; out_dim * rank],
            grad_a: vec![0.0_f32; rank * in_dim],
            grad_b: vec![0.0_f32; out_dim * rank],
            adam_a_m: vec![0.0_f32; rank * in_dim],
            adam_a_v: vec![0.0_f32; rank * in_dim],
            adam_b_m: vec![0.0_f32; out_dim * rank],
            adam_b_v: vec![0.0_f32; out_dim * rank],
        }
    }

    pub fn param_count(&self) -> usize {
        self.a.len() + self.b.len()
    }

    fn check_batch(&self, xs: &[f32], outs_len: usize, count: usize) -> Result<()> {
        if xs.len() != count * self.in_dim || outs_len != count * self.out_dim {
            return Err(FinetuneError::Adapter(format!(
                "batch shape mismatch: xs={} outs={} count={} expected in={} out={}",
                xs.len(),
                outs_len,
                count,
                self.in_dim,
                self.out_dim
            )));
        }
        Ok(())
    }

    /// Down-projection for a batch: returns hidden [count, rank].
    fn down_batch(&self, xs: &[f32], count: usize) -> Vec<f32> {
        let (rank, in_dim) = (self.rank, self.in_dim);
        let mut hidden = vec![0.0_f32; count * rank];
        hidden
            .par_chunks_mut(rank)
            .zip(xs.par_chunks(in_dim))
            .for_each(|(hrow, x)| {
                for (r, hv) in hrow.iter_mut().enumerate() {
                    let arow = &self.a[r * in_dim..(r + 1) * in_dim];
                    *hv = dot(arow, x);
                }
            });
        hidden
    }

    /// Adds `scale * B A x` to `count` rows of base projections in place.
    pub fn forward_batch(&self, xs: &[f32], base_outs: &mut [f32], count: usize) -> Result<()> {
        self.check_batch(xs, base_outs.len(), count)?;
        let (rank, out_dim, scale) = (self.rank, self.out_dim, self.scale);
        let hidden = self.down_batch(xs, count);
        base_outs
            .par_chunks_mut(out_dim)
            .zip(hidden.par_chunks(rank))
            .for_each(|(out, hrow)| {
                for (o, ov) in out.iter_mut().enumerate() {
                    let brow = &self.b[o * rank..(o + 1) * rank];
                    *ov += scale * dot(brow, hrow);
                }
            });
        Ok(())
    }

    /// Accumulates gradients for a batch of rows. `grad_outs` is the gradient
    /// of the loss w.r.t. the adapter's (full) output rows, [count, out_dim].
    pub fn backward_batch(&mut self, xs: &[f32], grad_outs: &[f32], count: usize) -> Result<()> {
        self.check_batch(xs, grad_outs.len(), count)?;
        let (rank, in_dim, out_dim, scale) = (self.rank, self.in_dim, self.out_dim, self.scale);
        let hidden = self.down_batch(xs, count);

        // grad_b[o][r] += scale * sum_t grad_outs[t][o] * hidden[t][r]
        let b = &self.b;
        self.grad_b
            .par_chunks_mut(rank)
            .enumerate()
            .for_each(|(o, gb)| {
                for t in 0..count {
                    let g = scale * grad_outs[t * out_dim + o];
                    if g == 0.0 {
                        continue;
                    }
                    let hrow = &hidden[t * rank..(t + 1) * rank];
                    for (gv, hv) in gb.iter_mut().zip(hrow.iter()) {
                        *gv += g * hv;
                    }
                }
            });

        // grad_hidden[t][r] = scale * sum_o grad_outs[t][o] * b[o][r]
        let mut grad_hidden = vec![0.0_f32; count * rank];
        grad_hidden
            .par_chunks_mut(rank)
            .zip(grad_outs.par_chunks(out_dim))
            .for_each(|(gh, grow)| {
                for (o, &g) in grow.iter().enumerate() {
                    if g == 0.0 {
                        continue;
                    }
                    let gs = scale * g;
                    let brow = &b[o * rank..(o + 1) * rank];
                    for (ghv, bv) in gh.iter_mut().zip(brow.iter()) {
                        *ghv += gs * bv;
                    }
                }
            });

        // grad_a[r][i] += sum_t grad_hidden[t][r] * xs[t][i]
        self.grad_a
            .par_chunks_mut(in_dim)
            .enumerate()
            .for_each(|(r, ga)| {
                for t in 0..count {
                    let gh = grad_hidden[t * rank + r];
                    if gh == 0.0 {
                        continue;
                    }
                    let x = &xs[t * in_dim..(t + 1) * in_dim];
                    for (gv, xv) in ga.iter_mut().zip(x.iter()) {
                        *gv += gh * xv;
                    }
                }
            });
        Ok(())
    }

    pub fn zero_grad(&mut self) {
        self.grad_a.fill(0.0);
        self.grad_b.fill(0.0);
    }

    /// AdamW update from the accumulated gradients; grads are NOT zeroed here.
    pub fn step(&mut self, learning_rate: f32, weight_decay: f32, step: usize) {
        crate::fused::adamw_step(
            &mut self.a,
            &self.grad_a,
            &mut self.adam_a_m,
            &mut self.adam_a_v,
            learning_rate,
            weight_decay,
            step,
            true,
        );
        crate::fused::adamw_step(
            &mut self.b,
            &self.grad_b,
            &mut self.adam_b_m,
            &mut self.adam_b_v,
            learning_rate,
            weight_decay,
            step,
            true,
        );
    }

    /// Single-row convenience wrapper (tests, tiny models).
    pub fn forward(&self, x: &[f32], base_out: &mut [f32]) -> Result<()> {
        self.forward_batch(x, base_out, 1)
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn init_lora_a(a: &mut [f32], rank: usize, seed: u64) {
    let scale = 1.0 / (rank as f32).sqrt();
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for v in a.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = ((state >> 32) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        *v = u * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_adapter(in_dim: usize, out_dim: usize) -> LoRAAdapter {
        let cfg = FinetuneConfig {
            rank: 4,
            alpha: 8.0,
            ..Default::default()
        };
        let mut adapter = LoRAAdapter::new(LoRATarget::OutputHead, in_dim, out_dim, &cfg);
        for (i, v) in adapter.b.iter_mut().enumerate() {
            *v = ((i % 13) as f32 - 6.0) * 0.01;
        }
        adapter
    }

    #[test]
    fn lora_forward_changes_output() {
        let adapter = test_adapter(8, 16);
        let x = vec![1.0_f32; 8];
        let mut out = vec![0.0_f32; 16];
        adapter.forward(&x, &mut out).expect("forward");
        assert!(out.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn batched_forward_matches_single_rows() {
        let adapter = test_adapter(8, 16);
        let count = 5;
        let xs: Vec<f32> = (0..count * 8).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut batched = vec![0.0_f32; count * 16];
        adapter
            .forward_batch(&xs, &mut batched, count)
            .expect("batch");
        for t in 0..count {
            let mut single = vec![0.0_f32; 16];
            adapter
                .forward(&xs[t * 8..(t + 1) * 8], &mut single)
                .expect("single");
            for (b, s) in batched[t * 16..(t + 1) * 16].iter().zip(single.iter()) {
                assert!((b - s).abs() < 1e-5, "batched {b} vs single {s}");
            }
        }
    }

    #[test]
    fn backward_batch_matches_sum_of_single_rows() {
        let count = 3;
        let xs: Vec<f32> = (0..count * 8).map(|i| (i as f32 * 0.21).cos()).collect();
        let gs: Vec<f32> = (0..count * 16).map(|i| (i as f32 * 0.11).sin()).collect();

        let mut batched = test_adapter(8, 16);
        batched.backward_batch(&xs, &gs, count).expect("batch");

        let mut single = test_adapter(8, 16);
        for t in 0..count {
            single
                .backward_batch(&xs[t * 8..(t + 1) * 8], &gs[t * 16..(t + 1) * 16], 1)
                .expect("single");
        }
        for (b, s) in batched.grad_a.iter().zip(single.grad_a.iter()) {
            assert!((b - s).abs() < 1e-4, "grad_a {b} vs {s}");
        }
        for (b, s) in batched.grad_b.iter().zip(single.grad_b.iter()) {
            assert!((b - s).abs() < 1e-4, "grad_b {b} vs {s}");
        }
    }

    #[test]
    fn gradient_check_against_finite_differences() {
        // Loss = sum(out); d loss / d param checked by central differences.
        let cfg = FinetuneConfig {
            rank: 2,
            alpha: 4.0,
            ..Default::default()
        };
        let mut adapter = LoRAAdapter::new(LoRATarget::OutputHead, 4, 3, &cfg);
        for (i, v) in adapter.b.iter_mut().enumerate() {
            *v = (i as f32 - 2.5) * 0.05;
        }
        let x = vec![0.3_f32, -0.7, 1.1, 0.05];
        let grad_out = vec![1.0_f32; 3];
        adapter.backward_batch(&x, &grad_out, 1).expect("backward");

        let eps = 1e-3_f32;
        let loss = |a: &LoRAAdapter| -> f32 {
            let mut out = vec![0.0_f32; 3];
            a.forward(&x, &mut out).unwrap();
            out.iter().sum()
        };
        for idx in [0usize, 3, 5] {
            let mut plus = adapter.clone();
            plus.b[idx] += eps;
            let mut minus = adapter.clone();
            minus.b[idx] -= eps;
            let fd = (loss(&plus) - loss(&minus)) / (2.0 * eps);
            let an = adapter.grad_b[idx];
            assert!((fd - an).abs() < 1e-2, "b[{idx}]: fd={fd} analytic={an}");
        }
        for idx in [0usize, 2, 7] {
            let mut plus = adapter.clone();
            plus.a[idx] += eps;
            let mut minus = adapter.clone();
            minus.a[idx] -= eps;
            let fd = (loss(&plus) - loss(&minus)) / (2.0 * eps);
            let an = adapter.grad_a[idx];
            assert!((fd - an).abs() < 1e-2, "a[{idx}]: fd={fd} analytic={an}");
        }
    }
}
