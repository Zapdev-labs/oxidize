//! Magnitude + Wanda + structured-N:M masking primitives.
//!
//! Row-wise magnitude / Wanda masks delegate to OXK (`oxidize-kernels::prune`)
//! for SIMD score prep and O(cols) per-row selection.

use anyhow::{Result, bail};
pub use oxidize_kernels::prune::{apply_mask_inplace, magnitude_mask, wanda_mask};

/// Sparsity pattern selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparsityPattern {
    Unstructured,
    N2of4,
    N4of8,
}

impl SparsityPattern {
    pub fn implied_sparsity(self) -> f32 {
        match self {
            SparsityPattern::Unstructured => 0.5,
            SparsityPattern::N2of4 => 0.5,
            SparsityPattern::N4of8 => 0.5,
        }
    }
}

/// Apply a structured N:M mask on top of a per-row mask.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnitude_mask_keeps_top_per_row() {
        let w: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mask = magnitude_mask(&w, 2, 8, 0.5);
        assert_eq!(mask.len(), 16);
        for r in 0..2 {
            let kept: usize = (0..8).map(|c| mask[r * 8 + c] as usize).sum();
            assert_eq!(kept, 4);
        }
        for c in 4..8 {
            assert!(mask[c], "row 0 col {c} should be kept");
        }
        for c in 0..4 {
            assert!(!mask[c], "row 0 col {c} should be pruned");
        }
    }

    #[test]
    fn wanda_mask_prefers_high_activation_columns() {
        let w = vec![10.0, 10.0, 10.0, 1.0, 1.0, 1.0];
        let norms = vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0];
        let mask = wanda_mask(&w, &norms, 1, 6, 0.5);
        for c in 0..3 {
            assert!(!mask[c], "left col {c} should be pruned (low act norm)");
        }
        for c in 3..6 {
            assert!(mask[c], "right col {c} should be kept (high act norm)");
        }
    }

    #[test]
    fn nm_pattern_caps_kept_per_block() {
        let w: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let mut mask = vec![true; 8];
        apply_nm_pattern(&mut mask, 1, 8, SparsityPattern::N4of8, |_r, c| w[c]).unwrap();
        let kept: usize = mask.iter().filter(|b| **b).count();
        assert_eq!(kept, 4);
        for c in 0..4 {
            assert!(!mask[c]);
        }
        for c in 4..8 {
            assert!(mask[c]);
        }
    }

    #[test]
    fn nm_pattern_2of4() {
        let w: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let mut mask = vec![true; 8];
        apply_nm_pattern(&mut mask, 1, 8, SparsityPattern::N2of4, |_r, c| w[c]).unwrap();
        assert!(!mask[0]);
        assert!(!mask[1]);
        assert!(mask[2]);
        assert!(mask[3]);
        assert!(!mask[4]);
        assert!(!mask[5]);
        assert!(mask[6]);
        assert!(mask[7]);
    }

    #[test]
    fn apply_mask_zeros_pruned_entries() {
        let mut w = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, false, true, false];
        apply_mask_inplace(&mut w, &mask);
        assert_eq!(w, vec![1.0, 0.0, 3.0, 0.0]);
    }
}
