use rayon::prelude::*;

#[allow(clippy::too_many_arguments)]
pub fn adamw_step(
    params: &mut [f32],
    grads: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    learning_rate: f32,
    weight_decay: f32,
    step: usize,
    apply_weight_decay: bool,
) {
    let beta1 = 0.9_f32;
    let beta2 = 0.999_f32;
    let eps = 1e-8_f32;
    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);
    params
        .par_iter_mut()
        .zip(grads.par_iter())
        .zip(m.par_iter_mut())
        .zip(v.par_iter_mut())
        .for_each(|(((param, grad), m), v)| {
            if apply_weight_decay {
                *param *= 1.0 - learning_rate * weight_decay;
            }
            *m = beta1 * *m + (1.0 - beta1) * *grad;
            *v = beta2 * *v + (1.0 - beta2) * *grad * *grad;
            let m_hat = *m / bc1;
            let v_hat = *v / bc2;
            *param -= learning_rate * m_hat / (v_hat.sqrt() + eps);
        });
}

/// Batched softmax cross-entropy. Converts `logits` ([count, vocab]) IN PLACE
/// into loss gradients `grad_scale * (softmax(logits) - onehot(target))` and
/// returns the summed (unscaled) per-token loss. Positions whose target is
/// `IGNORE_TARGET` produce zero gradient and no loss.
///
/// `grad_scale` should be `1 / tokens_per_optimizer_step` so accumulated
/// gradients average over the optimizer batch (NOT over vocab size — the old
/// implementation divided by vocab, silently shrinking the effective LR by
/// ~250k for large-vocab models).
pub const IGNORE_TARGET: u32 = u32::MAX;

pub fn cross_entropy_grad_batch(
    logits: &mut [f32],
    targets: &[u32],
    vocab: usize,
    grad_scale: f32,
) -> (f32, usize) {
    assert_eq!(logits.len(), targets.len() * vocab);
    logits
        .par_chunks_mut(vocab)
        .zip(targets.par_iter())
        .map(|(row, &target)| {
            if target == IGNORE_TARGET {
                row.fill(0.0);
                return (0.0_f32, 0usize);
            }
            let target = target as usize;
            if target >= vocab {
                // Out-of-range label = a tokenizer/data bug. Skip it (like an
                // ignored target) instead of silently clamping to the last class
                // and training on the wrong target; assert in dev/test builds.
                debug_assert!(
                    target < vocab,
                    "target {target} out of range for vocab {vocab}"
                );
                row.fill(0.0);
                return (0.0_f32, 0usize);
            }
            let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row.iter().map(|l| (l - max_logit).exp()).sum();
            let log_sum_exp = max_logit + exp_sum.ln();
            let loss = log_sum_exp - row[target];
            for (i, l) in row.iter_mut().enumerate() {
                let p = (*l - log_sum_exp).exp();
                *l = (p - if i == target { 1.0 } else { 0.0 }) * grad_scale;
            }
            (loss, 1usize)
        })
        .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
}

/// Batched loss-only evaluation over [count, vocab] logits.
pub fn softmax_cross_entropy_batch(logits: &[f32], targets: &[u32], vocab: usize) -> (f32, usize) {
    assert_eq!(logits.len(), targets.len() * vocab);
    logits
        .par_chunks(vocab)
        .zip(targets.par_iter())
        .map(|(row, &target)| {
            if target == IGNORE_TARGET {
                return (0.0_f32, 0usize);
            }
            (softmax_cross_entropy(row, target as usize), 1usize)
        })
        .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
}

pub fn softmax_cross_entropy(logits: &[f32], target: usize) -> f32 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
    let log_sum_exp = max_logit + exp_sum.ln();
    log_sum_exp - logits[target.min(logits.len().saturating_sub(1))]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ce_grad_batch_matches_loss_only_and_sums_to_zero_ish() {
        let vocab = 7;
        let count = 4;
        let mut logits: Vec<f32> = (0..count * vocab)
            .map(|i| (i as f32 * 0.31).sin())
            .collect();
        let targets: Vec<u32> = vec![0, 3, 6, 2];
        let expect_loss = softmax_cross_entropy_batch(&logits, &targets, vocab);
        let (loss, n) = cross_entropy_grad_batch(&mut logits, &targets, vocab, 1.0);
        assert_eq!(n, count);
        assert!((loss - expect_loss.0).abs() < 1e-4);
        // softmax grads per row sum to 0 (probabilities sum to 1, minus onehot).
        for row in logits.chunks(vocab) {
            let s: f32 = row.iter().sum();
            assert!(s.abs() < 1e-4, "grad row sum {s}");
        }
    }

    #[test]
    fn ignored_targets_produce_no_loss_or_grad() {
        let vocab = 5;
        let mut logits = vec![0.5_f32; 2 * vocab];
        let targets = vec![1u32, IGNORE_TARGET];
        let (loss, n) = cross_entropy_grad_batch(&mut logits, &targets, vocab, 1.0);
        assert_eq!(n, 1);
        assert!(loss > 0.0);
        assert!(logits[vocab..].iter().all(|g| *g == 0.0));
    }
}
