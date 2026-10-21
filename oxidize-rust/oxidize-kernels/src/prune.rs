//! OXK pruning kernels: per-row magnitude / Wanda masks and masked zeroing.
//!
//! Uses `select_nth_unstable_by` for O(cols) per-row selection instead of a
//! full sort, and AVX2 where available for score prep and mask application.

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::is_x86_feature_detected;

/// Per-output-row magnitude mask (`true` = keep).
pub fn magnitude_mask(weights_f32: &[f32], rows: usize, cols: usize, sparsity: f32) -> Vec<bool> {
    debug_assert_eq!(weights_f32.len(), rows * cols);
    let keep_per_row = ((1.0 - sparsity) * cols as f32).round() as usize;
    let drop = cols.saturating_sub(keep_per_row);
    let mut mask = vec![true; rows * cols];
    if drop == 0 {
        return mask;
    }
    let mut scratch = vec![0.0_f32; cols];
    let mut indices = vec![0_usize; cols];
    for r in 0..rows {
        let row = &weights_f32[r * cols..(r + 1) * cols];
        fill_abs_scores(row, &mut scratch);
        mask_row_by_scores(
            &scratch,
            &mut indices,
            drop,
            &mut mask[r * cols..(r + 1) * cols],
        );
    }
    mask
}

/// Per-output-row Wanda mask: metric `|W_ij| · ‖X_j‖_2`.
pub fn wanda_mask(
    weights_f32: &[f32],
    act_norms: &[f32],
    rows: usize,
    cols: usize,
    sparsity: f32,
) -> Vec<bool> {
    debug_assert_eq!(weights_f32.len(), rows * cols);
    debug_assert_eq!(act_norms.len(), cols);
    let keep_per_row = ((1.0 - sparsity) * cols as f32).round() as usize;
    let drop = cols.saturating_sub(keep_per_row);
    let mut mask = vec![true; rows * cols];
    if drop == 0 {
        return mask;
    }
    let mut scratch = vec![0.0_f32; cols];
    let mut indices = vec![0_usize; cols];
    for r in 0..rows {
        let row = &weights_f32[r * cols..(r + 1) * cols];
        fill_wanda_scores(row, act_norms, &mut scratch);
        mask_row_by_scores(
            &scratch,
            &mut indices,
            drop,
            &mut mask[r * cols..(r + 1) * cols],
        );
    }
    mask
}

/// Zero pruned entries in a row-major weight matrix (`mask[i] == false` → 0).
pub fn apply_mask_inplace(weights_f32: &mut [f32], mask: &[bool]) {
    // `assert_eq!` (not `debug_assert_eq!`): on a length mismatch `zip` would
    // silently truncate in release builds, leaving weights unzeroed.
    assert_eq!(weights_f32.len(), mask.len());
    for (w, &keep) in weights_f32.iter_mut().zip(mask.iter()) {
        if !keep {
            *w = 0.0;
        }
    }
}

#[inline]
fn fill_abs_scores(row: &[f32], scores: &mut [f32]) {
    debug_assert_eq!(row.len(), scores.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if oxk_avx2_for_prune() {
            unsafe { fill_abs_avx2(row, scores) };
            return;
        }
    }
    for (s, &w) in scores.iter_mut().zip(row.iter()) {
        *s = w.abs();
    }
}

#[inline]
fn fill_wanda_scores(row: &[f32], norms: &[f32], scores: &mut [f32]) {
    debug_assert_eq!(row.len(), scores.len());
    debug_assert_eq!(norms.len(), scores.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if oxk_avx2_for_prune() {
            unsafe { fill_wanda_avx2(row, norms, scores) };
            return;
        }
    }
    for i in 0..scores.len() {
        scores[i] = row[i].abs() * norms[i];
    }
}

#[inline]
fn mask_row_by_scores(scores: &[f32], indices: &mut [usize], drop: usize, row_mask: &mut [bool]) {
    debug_assert_eq!(scores.len(), indices.len());
    debug_assert_eq!(scores.len(), row_mask.len());
    for (i, slot) in indices.iter_mut().enumerate() {
        *slot = i;
    }
    // `total_cmp` gives a strict weak ordering even when scores contain NaN;
    // `partial_cmp(...).unwrap_or(Equal)` does not, which can corrupt the
    // partition produced by `select_nth_unstable_by`.
    indices.select_nth_unstable_by(drop - 1, |&a, &b| scores[a].total_cmp(&scores[b]));
    for &j in indices.iter().take(drop) {
        row_mask[j] = false;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn oxk_avx2_for_prune() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| is_x86_feature_detected!("avx2"))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn fill_abs_avx2(row: &[f32], scores: &mut [f32]) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= row.len() {
        let v = _mm256_loadu_ps(row.as_ptr().add(i));
        let abs_v = _mm256_andnot_ps(_mm256_set1_ps(-0.0), v);
        _mm256_storeu_ps(scores.as_mut_ptr().add(i), abs_v);
        i += 8;
    }
    while i < row.len() {
        scores[i] = row[i].abs();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn fill_wanda_avx2(row: &[f32], norms: &[f32], scores: &mut [f32]) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= row.len() {
        let w = _mm256_loadu_ps(row.as_ptr().add(i));
        let n = _mm256_loadu_ps(norms.as_ptr().add(i));
        let abs_w = _mm256_andnot_ps(_mm256_set1_ps(-0.0), w);
        let prod = _mm256_mul_ps(abs_w, n);
        _mm256_storeu_ps(scores.as_mut_ptr().add(i), prod);
        i += 8;
    }
    while i < row.len() {
        scores[i] = row[i].abs() * norms[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;

    #[test]
    fn magnitude_mask_keeps_top_per_row() {
        let w: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mask = magnitude_mask(&w, 2, 8, 0.5);
        for r in 0..2 {
            let kept: usize = (0..8).map(|c| mask[r * 8 + c] as usize).sum();
            assert_eq!(kept, 4);
        }
        for c in 4..8 {
            assert!(mask[c]);
        }
        for c in 0..4 {
            assert!(!mask[c]);
        }
    }

    #[test]
    fn wanda_mask_prefers_high_activation_columns() {
        let w = vec![10.0, 10.0, 10.0, 1.0, 1.0, 1.0];
        let norms = vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0];
        let mask = wanda_mask(&w, &norms, 1, 6, 0.5);
        for c in 0..3 {
            assert!(!mask[c], "left col {c} should be pruned");
        }
        for c in 3..6 {
            assert!(mask[c], "right col {c} should be kept");
        }
    }

    #[test]
    fn apply_mask_zeros_pruned_entries() {
        let mut w = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, false, true, false];
        apply_mask_inplace(&mut w, &mask);
        assert_eq!(w, vec![1.0, 0.0, 3.0, 0.0]);
    }
}
