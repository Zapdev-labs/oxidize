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

#[derive(Debug, Clone)]
pub struct LoRAAdapter {
    pub target: LoRATarget,
    pub in_dim: usize,
    pub out_dim: usize,
    pub rank: usize,
    pub scale: f32,
    pub a: Vec<f32>,
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

    pub fn forward(&self, x: &[f32], base_out: &mut [f32]) -> Result<()> {
        if x.len() != self.in_dim || base_out.len() != self.out_dim {
            return Err(FinetuneError::Adapter(format!(
                "shape mismatch: x={} out={} expected in={} out={}",
                x.len(),
                base_out.len(),
                self.in_dim,
                self.out_dim
            )));
        }
        let mut hidden = vec![0.0_f32; self.rank];
        lora_down(&self.a, x, self.in_dim, self.rank, &mut hidden);
        lora_up_add(&self.b, &hidden, self.rank, self.out_dim, self.scale, base_out);
        Ok(())
    }

    pub fn zero_grad(&mut self) {
        self.grad_a.fill(0.0);
        self.grad_b.fill(0.0);
    }

    pub fn backward_and_step(
        &mut self,
        x: &[f32],
        grad_out: &[f32],
        learning_rate: f32,
        weight_decay: f32,
        step: usize,
    ) -> Result<()> {
        let mut hidden = vec![0.0_f32; self.rank];
        lora_down(&self.a, x, self.in_dim, self.rank, &mut hidden);
        let mut grad_hidden = vec![0.0_f32; self.rank];
        lora_up_backward(
            &self.b,
            grad_out,
            &hidden,
            self.rank,
            self.out_dim,
            self.scale,
            &mut grad_hidden,
            &mut self.grad_b,
        );
        lora_down_backward(x, &grad_hidden, self.in_dim, self.rank, &mut self.grad_a);
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
        Ok(())
    }
}

fn init_lora_a(a: &mut [f32], rank: usize, seed: u64) {
    let scale = 1.0 / (rank as f32).sqrt();
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for v in a.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = (state as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        *v = u * scale;
    }
}

fn lora_down(a: &[f32], x: &[f32], in_dim: usize, rank: usize, out: &mut [f32]) {
    out.par_iter_mut().enumerate().for_each(|(r, o)| {
        let row = &a[r * in_dim..(r + 1) * in_dim];
        *o = row.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f32>();
    });
}

fn lora_up_add(b: &[f32], hidden: &[f32], rank: usize, out_dim: usize, scale: f32, out: &mut [f32]) {
    for o in 0..out_dim {
        let row = &b[o * rank..(o + 1) * rank];
        let delta: f32 = row.iter().zip(hidden.iter()).map(|(w, h)| w * h).sum();
        out[o] += scale * delta;
    }
}

fn lora_up_backward(
    b: &[f32],
    grad_out: &[f32],
    hidden: &[f32],
    rank: usize,
    out_dim: usize,
    scale: f32,
    grad_hidden: &mut [f32],
    grad_b: &mut [f32],
) {
    grad_hidden.fill(0.0);
    for o in 0..out_dim {
        let g = grad_out[o] * scale;
        for r in 0..rank {
            grad_b[o * rank + r] += g * hidden[r];
            grad_hidden[r] += b[o * rank + r] * g;
        }
    }
}

fn lora_down_backward(
    x: &[f32],
    grad_hidden: &[f32],
    in_dim: usize,
    rank: usize,
    grad_a: &mut [f32],
) {
    for r in 0..rank {
        let gh = grad_hidden[r];
        for i in 0..in_dim {
            grad_a[r * in_dim + i] += gh * x[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_forward_changes_output() {
        let cfg = FinetuneConfig {
            rank: 4,
            alpha: 8.0,
            ..Default::default()
        };
        let mut adapter = LoRAAdapter::new(LoRATarget::OutputHead, 8, 16, &cfg);
        for (i, v) in adapter.b.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.01;
        }
        let x = vec![1.0_f32; 8];
        let mut out = vec![0.0_f32; 16];
        adapter.forward(&x, &mut out).expect("forward");
        assert!(out.iter().any(|v| *v != 0.0));
    }
}
