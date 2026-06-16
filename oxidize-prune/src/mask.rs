//! Magnitude + Wanda + structured-N:M masking primitives.
//!
//! Algorithms (all from the literature, see `AGENTS.md` "WHERE TO LOOK"
//! → pruning):
//!
//! - **Magnitude** (Han et al. 2015). Per-output-row: keep the top-k%
//!   weights by `|W|`. We use the per-row comparison group (Sun et al.
//!   2023, Table 7) which the paper shows is the correct default for LLMs
//!   (LLaMA-7B 50% PPL = 8.86 vs 17.29 layer-wise).
//! - **Wanda** (Sun et al. 2023, ICLR 2024 — `arxiv:2306.11695`).
//!   Per-output-row: keep the top-k% weights by `|W_ij| · ‖X_j‖_2`,
//!   where `‖X_j‖_2` is the per-input-neuron L2 norm of the calibration
//!   activations (provided by `oxidize_core::activation_stats`).
//! - **Structured N:M** (Mishra et al. 2021, used by Wanda and SparseGPT
//!   for the 2:4 / 4:8 sparse-tensor-core patterns). For each row and
//!   each block of `M` consecutive input columns, keep at most `N`
//!   weights chosen by the same metric (magnitude or Wanda).
//!
//! The mask returned is a `Vec<bool>` of length `out * in`, where
//! `true = keep`, `false = prune (zero)`. The caller (`wanda.rs`) is
//! responsible for applying the mask to the dequantized weight matrix
//! and re-quantizing.

use anyhow::{Result, bail};

/// Sparsity pattern selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparsityPattern {
    /// Independent unstructured: drop the bottom-k% per output row by
    /// the chosen metric.
    Unstructured,
    /// NVIDIA 2:4 sparse-tensor-core format. Every group of 4
    /// consecutive input columns contains at most 2 kept weights.
    N2of4,
    /// NVIDIA 4:8 sparse-tensor-core format. Every group of 8
    /// consecutive input columns contains at most 4 kept weights.
    N4of8,
}

impl SparsityPattern {
    /// Sparsity (fraction of weights zeroed) implied by this pattern.
    pub fn implied_sparsity(self) -> f32 {
        match self {
            SparsityPattern::Unstructured => 0.5, // caller-driven; the default
            SparsityPattern::N2of4 => 0.5,
            SparsityPattern::N4of8 => 0.5,
        }
    }
}

/// Compute a per-output-row pruning mask by magnitude.
///
/// `weights_f32` is row-major `(rows, cols)`. Returns `Vec<bool>` of
/// length `rows * cols`: `true` = keep. `sparsity` is the fraction to
/// drop, in `[0.0, 1.0)`. Comparison is per-row (the setting the Wanda
/// paper shows is best for LLMs).
pub fn magnitude_mask(weights_f32: &[f32], rows: usize, cols: usize, sparsity: f32) -> Vec<bool> {
    assert_eq!(weights_f32.len(), rows * cols);
    let keep_per_row = ((1.0 - sparsity) * cols as f32).round() as usize;
    let mut mask = vec![true; rows * cols];
    for r in 0..rows {
        let row = &weights_f32[r * cols..(r + 1) * cols];
        // Build (|w|, index) pairs and partial-sort the bottom-k.
        let mut idx: Vec<usize> = (0..cols).collect();
        idx.sort_by(|&a, &b| {
            row[a]
                .abs()
                .partial_cmp(&row[b].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let drop = cols.saturating_sub(keep_per_row);
        for &j in idx.iter().take(drop) {
            mask[r * cols + j] = false;
        }
    }
    mask
}

/// Compute a per-output-row pruning mask by Wanda's metric
/// `S_ij = |W_ij| · ‖X_j‖_2`.
///
/// `act_norms` is the per-input-neuron L2 norm (length `cols`),
/// typically produced by `ActivationStats::l2_norms`. `weights_f32` is
/// row-major `(rows, cols)`.
///
/// Note: the Wanda paper compares within each output row
/// (per-output grouping), which is what we do here. Per Wanda paper
/// §5 / Table 7, the `(output, 1)` group is best for LLMs.
pub fn wanda_mask(
    weights_f32: &[f32],
    act_norms: &[f32],
    rows: usize,
    cols: usize,
    sparsity: f32,
) -> Vec<bool> {
    assert_eq!(weights_f32.len(), rows * cols);
    assert_eq!(act_norms.len(), cols);
    let keep_per_row = ((1.0 - sparsity) * cols as f32).round() as usize;
    let mut mask = vec![true; rows * cols];
    for r in 0..rows {
        let row = &weights_f32[r * cols..(r + 1) * cols];
        let mut idx: Vec<usize> = (0..cols).collect();
        idx.sort_by(|&a, &b| {
            let sa = row[a].abs() * act_norms[a];
            let sb = row[b].abs() * act_norms[b];
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let drop = cols.saturating_sub(keep_per_row);
        for &j in idx.iter().take(drop) {
            mask[r * cols + j] = false;
        }
    }
    mask
}

/// Apply a structured N:M mask on top of a per-row mask. Returns a new
/// mask such that for every row, every block of `m` consecutive input
/// columns contains at most `n` kept weights. Within each block, the
/// `n` weights with the highest score under `score_fn` are kept.
pub fn apply_nm_pattern<F: Fn(usize, usize) -> f32 + Sync>(
    base_mask: &mut Vec<bool>,
    rows: usize,
    cols: usize,
    pattern: SparsityPattern,
    score_fn: F,
) -> Result<()> {
    let (n, m) = match pattern {
        SparsityPattern::N2of4 => (2, 4),
        SparsityPattern::N4of8 => (4, 8),
        SparsityPattern::Unstructured => return Ok(()),
    };
    if !cols.is_multiple_of(m) {
        bail!(
            "N:{} pattern requires cols ({}) to be a multiple of {}",
            n,
            cols,
            m
        );
    }
    for r in 0..rows {
        for blk in 0..(cols / m) {
            let start = blk * m;
            // Among the weights in this row-block, pick the n best by
            // the Wanda/magnitude score. Then force everything else in
            // the block to false.
            let mut block_indices: Vec<usize> = (0..m).collect();
            block_indices.sort_by(|&a, &b| {
                let sa = score_fn(r, start + a);
                let sb = score_fn(r, start + b);
                sa.partial_cmp(&sb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .reverse()
            });
            let keep_set: std::collections::HashSet<usize> =
                block_indices.iter().take(n).copied().collect();
            for k in 0..m {
                let c = start + k;
                if !keep_set.contains(&k) {
                    base_mask[r * cols + c] = false;
                }
            }
        }
    }
    Ok(())
}

/// Apply a mask to a dequantized f32 weight matrix in place.
/// `mask[r * cols + c] == true` means keep.
pub fn apply_mask_inplace(
    weights_f32: &mut [f32],
    mask: &[bool],
    rows: usize,
    cols: usize,
) {
    assert_eq!(weights_f32.len(), rows * cols);
    assert_eq!(mask.len(), rows * cols);
    for i in 0..weights_f32.len() {
        if !mask[i] {
            weights_f32[i] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnitude_mask_keeps_top_per_row() {
        // 2 rows of 8. Sparsity 0.5 -> keep 4 per row.
        let w: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mask = magnitude_mask(&w, 2, 8, 0.5);
        assert_eq!(mask.len(), 16);
        for r in 0..2 {
            let kept: usize = (0..8).map(|c| mask[r * 8 + c] as usize).sum();
            assert_eq!(kept, 4);
        }
        // The top-4 in row 0 are indices 4,5,6,7 (values 4,5,6,7).
        for c in 4..8 {
            assert!(mask[c], "row 0 col {c} should be kept");
        }
        for c in 0..4 {
            assert!(!mask[c], "row 0 col {c} should be pruned");
        }
    }

    #[test]
    fn wanda_mask_prefers_high_activation_columns() {
        // 1 row of 6. Activation norms amplify the right side, so even
        // though the left side has larger weight magnitudes, Wanda
        // should keep the right side.
        let w = vec![10.0, 10.0, 10.0, 1.0, 1.0, 1.0];
        let norms = vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0];
        let mask = wanda_mask(&w, &norms, 1, 6, 0.5);
        // keep 3 of 6.
        for c in 0..3 {
            assert!(!mask[c], "left col {c} should be pruned (low act norm)");
        }
        for c in 3..6 {
            assert!(mask[c], "right col {c} should be kept (high act norm)");
        }
    }

    #[test]
    fn nm_pattern_caps_kept_per_block() {
        // 1 row of 8, 4:8 pattern -> keep 4 per block (one block of 8).
        let w: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let mut mask = vec![true; 8];
        apply_nm_pattern(&mut mask, 1, 8, SparsityPattern::N4of8, |_r, c| w[c]).unwrap();
        let kept: usize = mask.iter().filter(|b| **b).count();
        assert_eq!(kept, 4);
        // The top-4 weights are 5,6,7,8 (cols 4..8).
        for c in 0..4 {
            assert!(!mask[c]);
        }
        for c in 4..8 {
            assert!(mask[c]);
        }
    }

    #[test]
    fn nm_pattern_2of4() {
        // 1 row of 8 -> 2 blocks of 4. 2:4 keeps 2 per block.
        let w: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let mut mask = vec![true; 8];
        apply_nm_pattern(&mut mask, 1, 8, SparsityPattern::N2of4, |_r, c| w[c]).unwrap();
        // Block 0 (cols 0..4): top-2 are cols 2,3.
        assert!(!mask[0]);
        assert!(!mask[1]);
        assert!(mask[2]);
        assert!(mask[3]);
        // Block 1 (cols 4..8): top-2 are cols 6,7.
        assert!(!mask[4]);
        assert!(!mask[5]);
        assert!(mask[6]);
        assert!(mask[7]);
    }

    #[test]
    fn apply_mask_zeros_pruned_entries() {
        let mut w = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, false, true, false];
        apply_mask_inplace(&mut w, &mask, 1, 4);
        assert_eq!(w, vec![1.0, 0.0, 3.0, 0.0]);
    }
}
