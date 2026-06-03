use rayon::prelude::*;

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

pub fn cross_entropy_grad(logits: &[f32], target: usize, grad: &mut [f32]) -> f32 {
    let n = logits.len();
    let inv = 1.0 / n.max(1) as f32;
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
    let log_sum_exp = max_logit + exp_sum.ln();
    let mut loss = 0.0_f32;
    for (i, g) in grad.iter_mut().enumerate() {
        let p = (logits[i] - log_sum_exp).exp();
        *g = (p - if i == target { 1.0 } else { 0.0 }) * inv;
        if i == target {
            loss = log_sum_exp - logits[i];
        }
    }
    loss * inv
}

pub fn softmax_cross_entropy(logits: &[f32], target: usize) -> f32 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
    let log_sum_exp = max_logit + exp_sum.ln();
    log_sum_exp - logits[target.min(logits.len().saturating_sub(1))]
}
