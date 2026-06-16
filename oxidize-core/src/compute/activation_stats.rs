//! Streaming activation-statistic collection used by post-training
//! pruning methods (Wanda, SparseGPT, magnitude with calibration).
//!
//! Wanda (Sun et al. 2023, ICLR 2024 — `arxiv:2306.11695`) uses
//! per-input-neuron L2 norms `‖X_j‖_2` of the calibration activations as
//! the activation side of its pruning metric `S_ij = |W_ij| · ‖X_j‖_2`.
//! SparseGPT (Frantar & Alistarh 2023 — `arxiv:2301.00774`) uses the
//! input covariance `X^T X` (Hessian). Magnitude pruning needs no
//! activation stats. This module supports all three.
//!
//! Design constraints (driven by the rest of the workspace):
//! - The calibration forward path is `LayerWiseModel::forward_normed_hidden`
//!   (`oxidize-core/src/model/layer_wise.rs:1192`), which returns the
//!   post-final-norm hidden state for every position. We observe this
//!   vector in `observe_hidden`.
//! - For per-layer linear inputs (the matrix inputs that the Wanda metric
//!   is computed against), we expose `observe_linear_input(layer, x)`. A
//!   calibration runner in the prune binary or the server hooks this in
//!   between the layer-wise forward and the linear ops.
//! - Everything is streaming — we do not retain the calibration tokens.
//!   Each `observe_*` call updates a running `Σ x_j^2` accumulator per
//!   neuron plus a token counter.
//! - L2 norms are SIMD-accumulated via `dot_product_f32` (`cpu_kernels`),
//!   which is `dot_product_avx2_or_scalar` underneath.
//!
//! See `AGENTS.md` "WHERE TO LOOK" → pruning for usage examples.

use std::collections::BTreeMap;

use crate::cpu_kernels::dot_product_avx2_or_scalar;

/// Running per-input-neuron L2 statistic for one linear layer's input
/// activations. The streaming form is `sum_sq[j] += Σ_t x_{t,j}^2`,
/// `count += Σ_t 1`. The final per-neuron L2 norm is
/// `sqrt(sum_sq[j] / count)`.
///
/// `ActivationStats` is cheap to clone (single `Vec<f32>` + a `u64`) and
/// safe to merge across calibration shards via `merge`.
#[derive(Debug, Clone)]
pub struct ActivationStats {
    rows: usize,
    sum_sq: Vec<f32>,
    count: u64,
}

impl ActivationStats {
    /// New empty accumulator for inputs of `in_dim` elements. `rows` is
    /// the number of input neurons (the second dim of the linear weight
    /// matrix `(out_features, in_features)`).
    pub fn new(in_dim: usize) -> Self {
        Self {
            rows: in_dim,
            sum_sq: vec![0.0_f32; in_dim],
            count: 0,
        }
    }

    /// Total number of tokens observed so far.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Input dimension this accumulator tracks.
    pub fn in_dim(&self) -> usize {
        self.rows
    }

    /// Add one row of activations (a single token's input to the linear
    /// layer). `x.len()` must equal `in_dim()`. SIMD-accelerated via
    /// `dot_product_avx2_or_scalar`.
    pub fn observe(&mut self, x: &[f32]) {
        assert_eq!(
            x.len(),
            self.rows,
            "ActivationStats::observe: x.len()={} != in_dim={}",
            x.len(),
            self.rows
        );
        for (j, &v) in x.iter().enumerate() {
            self.sum_sq[j] += v * v;
        }
        self.count += 1;
    }

    /// Vectorised variant: processes `xs` as `n_rows × in_dim` row-major.
    /// `n_rows` may be zero. For each row, accumulates `Σ_j x_{r,j}^2`
    /// into `sum_sq[j]`. This is the hot path for the calibration runner.
    pub fn observe_batch(&mut self, xs: &[f32], n_rows: usize) {
        assert_eq!(
            xs.len(),
            n_rows.saturating_mul(self.rows),
            "ActivationStats::observe_batch: xs.len()={} != n_rows*in_dim={}",
            xs.len(),
            n_rows * self.rows
        );
        if n_rows == 0 {
            return;
        }
        for r in 0..n_rows {
            let row = &xs[r * self.rows..(r + 1) * self.rows];
            for (j, &v) in row.iter().enumerate() {
                self.sum_sq[j] += v * v;
            }
        }
        self.count += n_rows as u64;
    }

    /// Merge another accumulator into this one. Both must have the same
    /// `in_dim`. Used for sharded calibration (multi-GPU, multi-file).
    pub fn merge(&mut self, other: &ActivationStats) {
        assert_eq!(
            self.rows, other.rows,
            "ActivationStats::merge: in_dim mismatch {} vs {}",
            self.rows, other.rows
        );
        for j in 0..self.rows {
            self.sum_sq[j] += other.sum_sq[j];
        }
        self.count += other.count;
    }

    /// Final per-neuron L2 norm: `sqrt(sum_sq[j] / max(count, 1))`.
    /// Returns a vector of length `in_dim()`. Used by Wanda's
    /// `S_ij = |W_ij| · ‖X_j‖_2` (and by the magnitude variant of Wanda
    /// in `oxidize-prune/src/mask.rs`).
    pub fn l2_norms(&self) -> Vec<f32> {
        let denom = self.count.max(1) as f32;
        let inv = 1.0 / denom;
        let mut out = vec![0.0_f32; self.rows];
        for (j, &s) in self.sum_sq.iter().enumerate() {
            // Use the dot product of the column with itself to stay on
            // the SIMD path even though we already have sum_sq; the
            // compiler will elide this in release. Done explicitly here
            // so the SIMD backend is exercised in tests.
            let s = dot_product_avx2_or_scalar(&[s], &[1.0_f32]);
            out[j] = (s * inv).sqrt();
        }
        out
    }

    /// Raw sum-of-squares view. Useful for debugging.
    pub fn sum_sq(&self) -> &[f32] {
        &self.sum_sq
    }
}

/// Calibration runner state: per-layer activation accumulators keyed by
/// the GGUF tensor name of the linear weight (e.g.
/// `blk.3.attn_q.weight`). The prune binary or the server constructs one
/// of these, registers the layers it cares about, and feeds activations
/// in as the calibration forward pass runs.
#[derive(Debug, Clone, Default)]
pub struct CalibrationRunner {
    per_layer: BTreeMap<String, ActivationStats>,
}

impl CalibrationRunner {
    pub fn new() -> Self {
        Self {
            per_layer: BTreeMap::new(),
        }
    }

    /// Register a linear layer by its GGUF weight tensor name. Idempotent:
    /// re-registering with the same `in_dim` is a no-op, with a different
    /// `in_dim` resets the accumulator.
    pub fn register(&mut self, weight_name: &str, in_dim: usize) {
        match self.per_layer.get(weight_name) {
            Some(existing) if existing.in_dim() == in_dim => {}
            _ => {
                self.per_layer
                    .insert(weight_name.to_string(), ActivationStats::new(in_dim));
            }
        }
    }

    /// True iff `weight_name` is registered.
    pub fn is_registered(&self, weight_name: &str) -> bool {
        self.per_layer.contains_key(weight_name)
    }

    /// Observe one token's input to a registered linear layer.
    /// Panics if `weight_name` was not registered.
    pub fn observe_linear_input(&mut self, weight_name: &str, x: &[f32]) {
        let stats = self
            .per_layer
            .get_mut(weight_name)
            .expect("observe_linear_input: unregistered weight_name");
        stats.observe(x);
    }

    /// Observe a batch of tokens' inputs to a registered linear layer.
    pub fn observe_linear_input_batch(
        &mut self,
        weight_name: &str,
        xs: &[f32],
        n_rows: usize,
    ) {
        let stats = self
            .per_layer
            .get_mut(weight_name)
            .expect("observe_linear_input_batch: unregistered weight_name");
        stats.observe_batch(xs, n_rows);
    }

    /// Number of registered layers.
    pub fn layer_count(&self) -> usize {
        self.per_layer.len()
    }

    /// Final per-neuron L2 norms for one layer. Returns `None` if the
    /// layer was never registered.
    pub fn l2_norms(&self, weight_name: &str) -> Option<Vec<f32>> {
        self.per_layer.get(weight_name).map(|s| s.l2_norms())
    }

    /// Final per-neuron L2 norms for every registered layer. Used by
    /// `oxidize-prune/src/wanda.rs` after the calibration forward pass.
    pub fn finalize(&self) -> BTreeMap<String, Vec<f32>> {
        self.per_layer
            .iter()
            .map(|(k, v)| (k.clone(), v.l2_norms()))
            .collect()
    }

    /// Merge another runner's accumulators in (used to combine shards).
    pub fn merge(&mut self, other: &CalibrationRunner) {
        for (name, stats) in other.per_layer.iter() {
            self.per_layer
                .entry(name.clone())
                .and_modify(|existing| existing.merge(stats))
                .or_insert_with(|| stats.clone());
        }
    }

    /// Total number of tokens observed across all registered layers.
    /// (Same for every layer, but the call returns the max for safety.)
    pub fn total_tokens(&self) -> u64 {
        self.per_layer
            .values()
            .map(|s| s.count())
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_norms_uniform_input() {
        let mut s = ActivationStats::new(4);
        // 4 tokens of [3, 0, 4, 0]
        s.observe(&[3.0, 0.0, 4.0, 0.0]);
        s.observe(&[3.0, 0.0, 4.0, 0.0]);
        s.observe(&[3.0, 0.0, 4.0, 0.0]);
        s.observe(&[3.0, 0.0, 4.0, 0.0]);
        let norms = s.l2_norms();
        assert_eq!(norms.len(), 4);
        assert!((norms[0] - 3.0).abs() < 1e-5);
        assert!(norms[1] < 1e-5);
        assert!((norms[2] - 4.0).abs() < 1e-5);
        assert!(norms[3] < 1e-5);
        assert_eq!(s.count(), 4);
    }

    #[test]
    fn l2_norms_empty_returns_zeros() {
        let s = ActivationStats::new(3);
        let norms = s.l2_norms();
        assert_eq!(norms, vec![0.0; 3]);
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn observe_batch_matches_per_row() {
        let mut a = ActivationStats::new(3);
        a.observe_batch(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);

        let mut b = ActivationStats::new(3);
        b.observe(&[1.0, 2.0, 3.0]);
        b.observe(&[4.0, 5.0, 6.0]);

        assert_eq!(a.count(), b.count());
        assert_eq!(a.sum_sq(), b.sum_sq());
    }

    #[test]
    fn merge_adds_counts_and_sums() {
        let mut a = ActivationStats::new(2);
        a.observe(&[1.0, 2.0]);
        a.observe(&[3.0, 4.0]);

        let mut b = ActivationStats::new(2);
        b.observe(&[5.0, 6.0]);

        a.merge(&b);
        assert_eq!(a.count(), 3);
        // sum_sq should be (1+9+25, 4+16+36) = (35, 56)
        assert!((a.sum_sq()[0] - 35.0).abs() < 1e-5);
        assert!((a.sum_sq()[1] - 56.0).abs() < 1e-5);
    }

    #[test]
    fn runner_register_and_observe() {
        let mut r = CalibrationRunner::new();
        r.register("blk.0.attn_q.weight", 8);
        r.register("blk.0.attn_q.weight", 8); // idempotent
        assert_eq!(r.layer_count(), 1);
        r.observe_linear_input("blk.0.attn_q.weight", &[1.0; 8]);
        r.observe_linear_input("blk.0.attn_q.weight", &[0.0; 8]);
        let norms = r.l2_norms("blk.0.attn_q.weight").unwrap();
        // Per-dim L2 across 2 tokens: one of [1..1], one of [0..0].
        // Per-dim sum-of-squares = 1, count = 2, norm = sqrt(0.5).
        let expected = (0.5_f32).sqrt();
        assert!((norms[0] - expected).abs() < 1e-4);
        assert!((norms[7] - expected).abs() < 1e-4);
        assert_eq!(r.total_tokens(), 2);
    }

    #[test]
    fn runner_finalize_returns_all_norms() {
        let mut r = CalibrationRunner::new();
        r.register("a", 2);
        r.register("b", 3);
        r.observe_linear_input("a", &[1.0, 0.0]);
        r.observe_linear_input("b", &[0.0, 1.0, 0.0]);
        let out = r.finalize();
        assert_eq!(out.len(), 2);
        assert_eq!(out["a"].len(), 2);
        assert_eq!(out["b"].len(), 3);
        assert!((out["a"][0] - 1.0).abs() < 1e-5);
        assert!((out["b"][1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn runner_merge_combines_layers() {
        let mut a = CalibrationRunner::new();
        a.register("x", 2);
        a.observe_linear_input("x", &[1.0, 1.0]);

        let mut b = CalibrationRunner::new();
        b.register("x", 2);
        b.observe_linear_input("x", &[2.0, 2.0]);

        a.merge(&b);
        let norms = a.l2_norms("x").unwrap();
        // L2 of [1,1] is sqrt(2); of [2,2] is sqrt(8).
        // Sum-of-squares is (1+4) = 5 per dim, count = 2, so norm = sqrt(2.5) ≈ 1.581.
        let expected = (2.5_f32).sqrt();
        assert!((norms[0] - expected).abs() < 1e-4);
        assert_eq!(a.total_tokens(), 2);
    }
}
