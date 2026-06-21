use rayon::prelude::*;

use crate::lora::LoRAAdapter;

/// Strategy used when combining multiple LoRA adapters into one.
#[derive(Debug, Clone, PartialEq)]
pub enum MergeStrategy {
    /// Weighted arithmetic mean of all A and B matrices.
    Linear,
    /// Spherical linear interpolation between exactly two adapters (ignores
    /// weights beyond the first two).
    Slerp,
    /// TIES-merging: trim low-magnitude parameters, resolve sign conflicts by
    /// majority vote, then disjoint-merge the surviving values.
    Ties {
        /// Fraction of parameters to keep (top by absolute value). `0.0 < density <= 1.0`.
        density: f32,
    },
}

/// Collects adapters and merges them according to a chosen strategy.
pub struct AdapterMerger {
    adapters: Vec<(LoRAAdapter, f32)>,
    strategy: MergeStrategy,
}

impl AdapterMerger {
    pub fn new(strategy: MergeStrategy) -> Self {
        Self {
            adapters: Vec::new(),
            strategy,
        }
    }

    /// Builder-style: add an adapter with an associated scalar weight.
    pub fn add(mut self, adapter: LoRAAdapter, weight: f32) -> Self {
        self.adapters.push((adapter, weight));
        self
    }

    /// Merge all added adapters into a single `LoRAAdapter`.
    pub fn merge(self) -> Result<LoRAAdapter, String> {
        if self.adapters.is_empty() {
            return Err("no adapters to merge".into());
        }
        match &self.strategy {
            MergeStrategy::Linear => linear_merge(&self.adapters),
            MergeStrategy::Slerp => {
                if self.adapters.len() < 2 {
                    return Err("slerp requires exactly 2 adapters".into());
                }
                let (a, _) = &self.adapters[0];
                let (b, wb) = &self.adapters[1];
                // Use the second adapter's weight as the interpolation t in [0,1].
                let t = wb.clamp(0.0, 1.0);
                slerp_merge(a, b, t)
            }
            MergeStrategy::Ties { density } => {
                let d = *density;
                ties_merge(&self.adapters, d)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Linear merge
// ---------------------------------------------------------------------------

/// Weighted arithmetic mean of A and B matrices across all adapters.
///
/// Normalises so that weights sum to 1.  The first adapter's metadata
/// (target, dims, rank, scale) is used as the template for the result.
pub fn linear_merge(adapters: &[(LoRAAdapter, f32)]) -> Result<LoRAAdapter, String> {
    let (template, _) = adapters.first().ok_or("empty adapter list")?;
    validate_compatibility(adapters)?;

    let weight_sum: f32 = adapters.iter().map(|(_, w)| w).sum();
    if weight_sum == 0.0 {
        return Err("sum of weights must be non-zero".into());
    }
    let inv = 1.0 / weight_sum;

    let a_len = template.a.len();
    let b_len = template.b.len();

    let merged_a: Vec<f32> = (0..a_len)
        .into_par_iter()
        .map(|i| {
            adapters
                .iter()
                .map(|(ad, w)| ad.a[i] * w * inv)
                .sum::<f32>()
        })
        .collect();

    let merged_b: Vec<f32> = (0..b_len)
        .into_par_iter()
        .map(|i| {
            adapters
                .iter()
                .map(|(ad, w)| ad.b[i] * w * inv)
                .sum::<f32>()
        })
        .collect();

    Ok(build_merged(template, merged_a, merged_b))
}

// ---------------------------------------------------------------------------
// Slerp merge
// ---------------------------------------------------------------------------

/// Spherical linear interpolation between two adapters' flattened parameter
/// vectors.  `t = 0` returns `a`, `t = 1` returns `b`.
///
/// When the two vectors are nearly collinear (|cosθ| ≥ 0.9995) the function
/// falls back to linear interpolation to avoid numerical instability.
pub fn slerp_merge(a: &LoRAAdapter, b: &LoRAAdapter, t: f32) -> Result<LoRAAdapter, String> {
    if a.in_dim != b.in_dim
        || a.out_dim != b.out_dim
        || a.rank != b.rank
        || a.a.len() != b.a.len()
        || a.b.len() != b.b.len()
    {
        return Err("slerp adapters must have identical dimensions".into());
    }

    let sa = slerp_vec(&a.a, &b.a, t);
    let sb = slerp_vec(&a.b, &b.b, t);
    Ok(build_merged(a, sa, sb))
}

fn slerp_vec(va: &[f32], vb: &[f32], t: f32) -> Vec<f32> {
    let norm_a = l2_norm(va);
    let norm_b = l2_norm(vb);

    // Degenerate: either vector is the zero vector — fall back to lerp.
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return lerp_vec(va, vb, t);
    }

    let dot: f32 = va
        .par_iter()
        .zip(vb.par_iter())
        .map(|(x, y)| x * y)
        .sum::<f32>()
        / (norm_a * norm_b);
    let dot = dot.clamp(-1.0, 1.0);

    if dot.abs() >= 0.9995 {
        return lerp_vec(va, vb, t);
    }

    let theta = dot.acos();
    let sin_theta = theta.sin();
    let wa = ((1.0 - t) * theta).sin() / sin_theta;
    let wb = (t * theta).sin() / sin_theta;

    va.par_iter()
        .zip(vb.par_iter())
        .map(|(x, y)| wa * x + wb * y)
        .collect()
}

fn lerp_vec(va: &[f32], vb: &[f32], t: f32) -> Vec<f32> {
    va.par_iter()
        .zip(vb.par_iter())
        .map(|(x, y)| x + t * (y - x))
        .collect()
}

// ---------------------------------------------------------------------------
// TIES merge
// ---------------------------------------------------------------------------

/// TIES-merging (Trimming, Electing, and Disjoint merging).
///
/// Algorithm:
/// 1. Trim each adapter's parameters: set the bottom `(1 - density)` fraction
///    by absolute value to zero.
/// 2. For each parameter position, elect a sign by majority vote across the
///    surviving (non-zero) values, weighted by adapter weights.
/// 3. Keep only values whose sign matches the elected sign, then compute the
///    weighted average of those kept values.
pub fn ties_merge(adapters: &[(LoRAAdapter, f32)], density: f32) -> Result<LoRAAdapter, String> {
    let (template, _) = adapters.first().ok_or("empty adapter list")?;
    validate_compatibility(adapters)?;

    if !(0.0..=1.0).contains(&density) {
        return Err(format!("density must be in [0, 1], got {density}"));
    }

    // Trim each adapter's flat parameter vector.
    let trimmed_a: Vec<Vec<f32>> = adapters
        .iter()
        .map(|(ad, _)| trim_vector(&ad.a, density))
        .collect();
    let trimmed_b: Vec<Vec<f32>> = adapters
        .iter()
        .map(|(ad, _)| trim_vector(&ad.b, density))
        .collect();

    let weights: Vec<f32> = adapters.iter().map(|(_, w)| *w).collect();

    let merged_a = ties_combine(&trimmed_a, &weights);
    let merged_b = ties_combine(&trimmed_b, &weights);

    Ok(build_merged(template, merged_a, merged_b))
}

/// Trim a parameter vector by zeroing out the bottom `(1 - density)` fraction
/// of values ranked by absolute magnitude.
fn trim_vector(v: &[f32], density: f32) -> Vec<f32> {
    if density >= 1.0 {
        return v.to_vec();
    }
    if density <= 0.0 {
        return vec![0.0; v.len()];
    }

    // Determine the threshold: the (1-density) quantile of |v|.
    let mut magnitudes: Vec<f32> = v.iter().map(|x| x.abs()).collect();
    magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff_idx = ((1.0 - density) * magnitudes.len() as f32) as usize;
    let threshold = magnitudes.get(cutoff_idx).copied().unwrap_or(0.0);

    v.iter()
        .map(|x| if x.abs() >= threshold { *x } else { 0.0 })
        .collect()
}

/// Given trimmed vectors (one per adapter) and their weights, elect a sign per
/// position by weighted majority vote, then disjoint-merge the survivors.
fn ties_combine(trimmed: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
    let len = trimmed[0].len();
    let n = trimmed.len();

    (0..len)
        .into_par_iter()
        .map(|i| {
            // Weighted sign vote.
            let pos_mass: f32 = (0..n)
                .filter(|&k| trimmed[k][i] > 0.0)
                .map(|k| weights[k])
                .sum();
            let neg_mass: f32 = (0..n)
                .filter(|&k| trimmed[k][i] < 0.0)
                .map(|k| weights[k])
                .sum();

            let elected_positive = pos_mass >= neg_mass;

            // Disjoint merge: average of values that agree with the elected sign.
            let mut sum = 0.0_f32;
            let mut total_w = 0.0_f32;
            for k in 0..n {
                let v = trimmed[k][i];
                if v == 0.0 {
                    continue;
                }
                if (v > 0.0) == elected_positive {
                    sum += v * weights[k];
                    total_w += weights[k];
                }
            }
            if total_w == 0.0 { 0.0 } else { sum / total_w }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Verify that all adapters share the same dimensions and rank.
fn validate_compatibility(adapters: &[(LoRAAdapter, f32)]) -> Result<(), String> {
    let (first, _) = &adapters[0];
    for (i, (ad, _)) in adapters.iter().enumerate().skip(1) {
        if ad.in_dim != first.in_dim
            || ad.out_dim != first.out_dim
            || ad.rank != first.rank
            || ad.a.len() != first.a.len()
            || ad.b.len() != first.b.len()
        {
            return Err(format!(
                "adapter {i} has incompatible shape (in={} out={} rank={}) \
                 vs template (in={} out={} rank={})",
                ad.in_dim, ad.out_dim, ad.rank, first.in_dim, first.out_dim, first.rank
            ));
        }
    }
    Ok(())
}

/// Construct a merged `LoRAAdapter` from a template (metadata source) and new
/// A / B matrices.  Gradients and Adam state are zeroed — the merged adapter
/// is not yet in training mode.
fn build_merged(template: &LoRAAdapter, a: Vec<f32>, b: Vec<f32>) -> LoRAAdapter {
    LoRAAdapter {
        target: template.target,
        in_dim: template.in_dim,
        out_dim: template.out_dim,
        rank: template.rank,
        scale: template.scale,
        grad_a: vec![0.0; a.len()],
        grad_b: vec![0.0; b.len()],
        adam_a_m: vec![0.0; a.len()],
        adam_a_v: vec![0.0; a.len()],
        adam_b_m: vec![0.0; b.len()],
        adam_b_v: vec![0.0; b.len()],
        a,
        b,
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.par_iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FinetuneConfig;
    use crate::lora::LoRATarget;

    fn make_adapter(
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        fill_a: f32,
        fill_b: f32,
    ) -> LoRAAdapter {
        let cfg = FinetuneConfig {
            rank,
            alpha: rank as f32 * 2.0,
            ..Default::default()
        };
        let mut ad = LoRAAdapter::new(LoRATarget::AttentionQ, in_dim, out_dim, &cfg);
        ad.a.fill(fill_a);
        ad.b.fill(fill_b);
        ad
    }

    #[test]
    fn linear_merge_equal_weights_averages() {
        let a = make_adapter(4, 8, 2, 1.0, 2.0);
        let b = make_adapter(4, 8, 2, 3.0, 4.0);
        let merged = linear_merge(&[(a, 1.0), (b, 1.0)]).expect("linear");
        assert!((merged.a[0] - 2.0).abs() < 1e-5);
        assert!((merged.b[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn linear_merge_weighted() {
        let a = make_adapter(4, 8, 2, 0.0, 0.0);
        let b = make_adapter(4, 8, 2, 4.0, 4.0);
        // weight 0 on `a`, weight 1 on `b` → result should equal `b`
        let merged = linear_merge(&[(a, 0.0), (b, 1.0)]).expect("linear");
        assert!((merged.a[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn linear_merge_incompatible_dims_errors() {
        let a = make_adapter(4, 8, 2, 1.0, 1.0);
        let b = make_adapter(8, 8, 2, 1.0, 1.0); // different in_dim
        assert!(linear_merge(&[(a, 1.0), (b, 1.0)]).is_err());
    }

    #[test]
    fn slerp_t0_returns_first() {
        let a = make_adapter(4, 8, 2, 1.0, 1.0);
        let b = make_adapter(4, 8, 2, 2.0, 2.0);
        let merged = slerp_merge(&a, &b, 0.0).expect("slerp");
        for (m, src) in merged.a.iter().zip(a.a.iter()) {
            assert!((m - src).abs() < 1e-4, "a mismatch: {m} vs {src}");
        }
    }

    #[test]
    fn slerp_t1_returns_second() {
        let a = make_adapter(4, 8, 2, 1.0, 1.0);
        let b = make_adapter(4, 8, 2, 2.0, 2.0);
        let merged = slerp_merge(&a, &b, 1.0).expect("slerp");
        for (m, src) in merged.a.iter().zip(b.a.iter()) {
            assert!((m - src).abs() < 1e-4, "a mismatch: {m} vs {src}");
        }
    }

    #[test]
    fn ties_density_one_equals_weighted_linear() {
        let a = make_adapter(4, 8, 2, 1.0, 1.0);
        let b = make_adapter(4, 8, 2, 3.0, 3.0);
        // With density=1.0 no trimming occurs; both adapters have positive
        // values so elected sign is positive → should closely match linear mean.
        let merged = ties_merge(&[(a, 1.0), (b, 1.0)], 1.0).expect("ties");
        assert!((merged.a[0] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn ties_density_zero_gives_zeros() {
        let a = make_adapter(4, 8, 2, 5.0, 5.0);
        let b = make_adapter(4, 8, 2, 5.0, 5.0);
        let merged = ties_merge(&[(a, 1.0), (b, 1.0)], 0.0).expect("ties");
        assert!(merged.a.iter().all(|&v| v == 0.0));
        assert!(merged.b.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn builder_api_linear() {
        let a = make_adapter(4, 8, 2, 1.0, 1.0);
        let b = make_adapter(4, 8, 2, 3.0, 3.0);
        let merged = AdapterMerger::new(MergeStrategy::Linear)
            .add(a, 1.0)
            .add(b, 1.0)
            .merge()
            .expect("merge");
        assert!((merged.a[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn builder_api_slerp() {
        let a = make_adapter(4, 8, 2, 1.0, 0.0);
        let b = make_adapter(4, 8, 2, 0.0, 1.0);
        // t = 0.5 via the second adapter's weight
        let merged = AdapterMerger::new(MergeStrategy::Slerp)
            .add(a, 1.0)
            .add(b, 0.5)
            .merge()
            .expect("slerp builder");
        // Result should be non-trivial (between the two).
        assert!(merged.a.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn builder_requires_two_for_slerp() {
        let a = make_adapter(4, 8, 2, 1.0, 1.0);
        let result = AdapterMerger::new(MergeStrategy::Slerp).add(a, 1.0).merge();
        assert!(result.is_err());
    }

    #[test]
    fn ties_sign_conflict_cancelled() {
        // a has positive values, b has negative values of equal weight and
        // equal magnitude → tie (pos_mass == neg_mass) defaults to positive,
        // but only `a`'s values survive the disjoint step.
        let a = make_adapter(4, 8, 2, 1.0, 1.0);
        let b = make_adapter(4, 8, 2, -1.0, -1.0);
        let merged = ties_merge(&[(a.clone(), 1.0), (b, 1.0)], 1.0).expect("ties");
        // Elected sign is positive (tie goes to positive); only a's params survive.
        for v in &merged.a {
            assert!(*v >= 0.0, "expected non-negative, got {v}");
        }
    }
}
