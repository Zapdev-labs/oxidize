#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub typical_p: Option<f32>,
    pub tail_free_z: Option<f32>,
    pub locally_typical_tau: Option<f32>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            min_p: None,
            typical_p: None,
            tail_free_z: None,
            locally_typical_tau: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MirostatConfig {
    pub tau: f32,
    pub eta: f32,
    pub mu: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingError {
    EmptyLogits,
    InvalidTemperature,
    InvalidTopK,
    InvalidTopP,
    InvalidMinP,
    InvalidTypicalP,
    InvalidTailFreeZ,
    InvalidLocallyTypicalTau,
    InvalidMirostat,
    InvalidRandom,
}

pub fn greedy(logits: &[f32]) -> Result<u32, SamplingError> {
    let Some((idx, _)) = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
    else {
        return Err(SamplingError::EmptyLogits);
    };
    Ok(idx as u32)
}

pub fn sample(logits: &[f32], config: SamplingConfig, random: f32) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if !config.temperature.is_finite() || config.temperature <= 0.0 {
        return Err(SamplingError::InvalidTemperature);
    }
    if config.top_k == Some(0) {
        return Err(SamplingError::InvalidTopK);
    }
    if let Some(top_p) = config.top_p
        && (!top_p.is_finite() || top_p <= 0.0 || top_p > 1.0)
    {
        return Err(SamplingError::InvalidTopP);
    }
    if let Some(min_p) = config.min_p
        && (!min_p.is_finite() || min_p <= 0.0 || min_p > 1.0)
    {
        return Err(SamplingError::InvalidMinP);
    }
    if let Some(typical_p) = config.typical_p
        && (!typical_p.is_finite() || typical_p <= 0.0 || typical_p > 1.0)
    {
        return Err(SamplingError::InvalidTypicalP);
    }
    if let Some(tail_free_z) = config.tail_free_z
        && (!tail_free_z.is_finite() || tail_free_z <= 0.0 || tail_free_z > 1.0)
    {
        return Err(SamplingError::InvalidTailFreeZ);
    }
    if let Some(locally_typical_tau) = config.locally_typical_tau
        && (!locally_typical_tau.is_finite() || locally_typical_tau <= 0.0)
    {
        return Err(SamplingError::InvalidLocallyTypicalTau);
    }
    if !random.is_finite() || !(0.0..1.0).contains(&random) {
        return Err(SamplingError::InvalidRandom);
    }

    let max_logit = logits
        .iter()
        .copied()
        .max_by(|a, b| a.total_cmp(b))
        .ok_or(SamplingError::EmptyLogits)?;
    let mut indexed_probs: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, logit)| (idx, ((logit - max_logit) / config.temperature).exp()))
        .collect();

    let raw_sum: f32 = indexed_probs.iter().map(|(_, p)| *p).sum();
    if raw_sum <= 0.0 || !raw_sum.is_finite() {
        return greedy(logits);
    }
    for (_, p) in &mut indexed_probs {
        *p /= raw_sum;
    }

    indexed_probs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    if let Some(top_k) = config.top_k
        && indexed_probs.len() > top_k
    {
        indexed_probs.truncate(top_k);
    }

    if let Some(top_p) = config.top_p {
        let mut cumulative = 0.0_f32;
        let cutoff = indexed_probs
            .iter()
            .position(|(_, prob)| {
                cumulative += *prob;
                cumulative >= top_p
            })
            .map_or(indexed_probs.len(), |idx| idx + 1);
        indexed_probs.truncate(cutoff);
    }

    if let Some(min_p) = config.min_p {
        let max_prob = indexed_probs.first().map_or(0.0, |(_, p)| *p);
        let threshold = max_prob * min_p;
        indexed_probs.retain(|(_, prob)| *prob >= threshold);
    }

    if let Some(typical_p) = config.typical_p {
        apply_typical_sampling(&mut indexed_probs, typical_p);
    }

    if let Some(tail_free_z) = config.tail_free_z {
        apply_tail_free_sampling(&mut indexed_probs, tail_free_z);
    }

    if let Some(locally_typical_tau) = config.locally_typical_tau {
        apply_locally_typical_sampling(&mut indexed_probs, locally_typical_tau);
    }

    if indexed_probs.is_empty() {
        return greedy(logits);
    }

    let filtered_sum: f32 = indexed_probs.iter().map(|(_, p)| *p).sum();
    if filtered_sum <= 0.0 || !filtered_sum.is_finite() {
        return greedy(logits);
    }

    let mut cumulative = 0.0_f32;
    let target = random * filtered_sum;
    for (idx, prob) in indexed_probs {
        cumulative += prob;
        if target <= cumulative {
            return Ok(idx as u32);
        }
    }

    greedy(logits)
}

pub fn sample_mirostat(
    logits: &[f32],
    temperature: f32,
    config: MirostatConfig,
    random: f32,
) -> Result<(u32, f32), SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(SamplingError::InvalidTemperature);
    }
    if !config.tau.is_finite()
        || config.tau <= 0.0
        || !config.eta.is_finite()
        || config.eta <= 0.0
        || !config.mu.is_finite()
    {
        return Err(SamplingError::InvalidMirostat);
    }
    if !random.is_finite() || !(0.0..1.0).contains(&random) {
        return Err(SamplingError::InvalidRandom);
    }

    let mut indexed_probs = build_sorted_probs(logits, temperature)?;
    let target_surprisal = config.mu;
    indexed_probs.sort_by(|a, b| {
        let a_surprise = -a.1.max(f32::MIN_POSITIVE).ln();
        let b_surprise = -b.1.max(f32::MIN_POSITIVE).ln();
        (a_surprise - target_surprisal)
            .abs()
            .total_cmp(&(b_surprise - target_surprisal).abs())
    });

    let chosen = weighted_pick(&indexed_probs, random).ok_or(SamplingError::EmptyLogits)?;
    let observed_surprisal = -chosen.1.max(f32::MIN_POSITIVE).ln();
    let updated_mu = config.mu - config.eta * (observed_surprisal - config.tau);

    Ok((chosen.0 as u32, updated_mu))
}

fn build_sorted_probs(logits: &[f32], temperature: f32) -> Result<Vec<(usize, f32)>, SamplingError> {
    let max_logit = logits
        .iter()
        .copied()
        .max_by(|a, b| a.total_cmp(b))
        .ok_or(SamplingError::EmptyLogits)?;
    let mut indexed_probs: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, logit)| (idx, ((logit - max_logit) / temperature).exp()))
        .collect();

    let raw_sum: f32 = indexed_probs.iter().map(|(_, p)| *p).sum();
    if raw_sum <= 0.0 || !raw_sum.is_finite() {
        return Err(SamplingError::EmptyLogits);
    }
    for (_, p) in &mut indexed_probs {
        *p /= raw_sum;
    }

    indexed_probs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    Ok(indexed_probs)
}

fn apply_typical_sampling(indexed_probs: &mut Vec<(usize, f32)>, typical_p: f32) {
    if indexed_probs.is_empty() {
        return;
    }
    let entropy: f32 = indexed_probs
        .iter()
        .map(|(_, p)| {
            let p = p.max(f32::MIN_POSITIVE);
            -p * p.ln()
        })
        .sum();
    let mut by_typicality: Vec<(usize, f32, f32)> = indexed_probs
        .iter()
        .map(|(idx, prob)| {
            let surprise = -prob.max(f32::MIN_POSITIVE).ln();
            (*idx, *prob, (surprise - entropy).abs())
        })
        .collect();
    by_typicality.sort_unstable_by(|a, b| a.2.total_cmp(&b.2));

    let mut cumulative = 0.0_f32;
    let mut keep = Vec::with_capacity(by_typicality.len());
    for (idx, prob, _) in by_typicality {
        keep.push((idx, prob));
        cumulative += prob;
        if cumulative >= typical_p {
            break;
        }
    }
    keep.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    *indexed_probs = keep;
}

fn apply_tail_free_sampling(indexed_probs: &mut Vec<(usize, f32)>, tail_free_z: f32) {
    if indexed_probs.len() <= 2 {
        return;
    }
    let mut second_derivative = Vec::with_capacity(indexed_probs.len().saturating_sub(2));
    for i in 0..indexed_probs.len() - 2 {
        let d1 = indexed_probs[i].1 - indexed_probs[i + 1].1;
        let d2 = indexed_probs[i + 1].1 - indexed_probs[i + 2].1;
        second_derivative.push((d1 - d2).abs());
    }
    let sd_sum: f32 = second_derivative.iter().sum();
    if sd_sum <= 0.0 || !sd_sum.is_finite() {
        return;
    }

    let mut cumulative = 0.0_f32;
    let mut cutoff = indexed_probs.len();
    for (i, sd) in second_derivative.into_iter().enumerate() {
        cumulative += sd / sd_sum;
        if cumulative >= tail_free_z {
            cutoff = (i + 2).max(1);
            break;
        }
    }
    indexed_probs.truncate(cutoff);
}

fn apply_locally_typical_sampling(indexed_probs: &mut Vec<(usize, f32)>, locally_typical_tau: f32) {
    if indexed_probs.is_empty() {
        return;
    }
    let entropy: f32 = indexed_probs
        .iter()
        .map(|(_, p)| {
            let p = p.max(f32::MIN_POSITIVE);
            -p * p.ln()
        })
        .sum();
    let deviation_limit = entropy * locally_typical_tau;
    let mut filtered: Vec<(usize, f32)> = indexed_probs
        .iter()
        .copied()
        .filter(|(_, prob)| {
            let surprise = -prob.max(f32::MIN_POSITIVE).ln();
            (surprise - entropy).abs() <= deviation_limit
        })
        .collect();
    if filtered.is_empty() {
        filtered.push(indexed_probs[0]);
    }
    filtered.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    *indexed_probs = filtered;
}

fn weighted_pick(indexed_probs: &[(usize, f32)], random: f32) -> Option<(usize, f32)> {
    if indexed_probs.is_empty() {
        return None;
    }
    let filtered_sum: f32 = indexed_probs.iter().map(|(_, p)| *p).sum();
    if filtered_sum <= 0.0 || !filtered_sum.is_finite() {
        return None;
    }

    let mut cumulative = 0.0_f32;
    let target = random * filtered_sum;
    for (idx, prob) in indexed_probs.iter().copied() {
        cumulative += prob;
        if target <= cumulative {
            return Some((idx, prob));
        }
    }
    indexed_probs.last().copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_returns_highest_logit_index() {
        assert_eq!(greedy(&[0.5, 2.0, 1.0]).expect("greedy should succeed"), 1);
    }

    #[test]
    fn temperature_changes_sampling_sharpness() {
        let logits = [0.0, 0.2];
        let cooler = sample(
            &logits,
            SamplingConfig {
                temperature: 0.5,
                ..SamplingConfig::default()
            },
            0.55,
        )
        .expect("sampling should succeed");
        let hotter = sample(
            &logits,
            SamplingConfig {
                temperature: 2.0,
                ..SamplingConfig::default()
            },
            0.55,
        )
        .expect("sampling should succeed");

        assert_eq!(cooler, 1);
        assert_eq!(hotter, 0);
    }

    #[test]
    fn top_k_limits_candidate_set() {
        let token = sample(
            &[5.0, 4.0, 3.0, 2.0],
            SamplingConfig {
                top_k: Some(2),
                ..SamplingConfig::default()
            },
            0.99,
        )
        .expect("sampling should succeed");
        assert!(token <= 1);
    }

    #[test]
    fn top_p_uses_nucleus_subset() {
        let token = sample(
            &[5.0, 4.0, 3.0, 2.0],
            SamplingConfig {
                top_p: Some(0.6),
                ..SamplingConfig::default()
            },
            0.99,
        )
        .expect("sampling should succeed");
        assert_eq!(token, 0);
    }

    #[test]
    fn min_p_filters_low_probability_tail() {
        let token = sample(
            &[6.0, 5.9, 2.0, 1.0],
            SamplingConfig {
                min_p: Some(0.9),
                ..SamplingConfig::default()
            },
            0.99,
        )
        .expect("sampling should succeed");
        assert!(token <= 1);
    }

    #[test]
    fn typical_sampling_keeps_typical_tokens() {
        let token = sample(
            &[6.0, 5.95, 4.0, 1.0],
            SamplingConfig {
                typical_p: Some(0.5),
                ..SamplingConfig::default()
            },
            0.95,
        )
        .expect("sampling should succeed");
        assert!(token <= 1);
    }

    #[test]
    fn tail_free_sampling_trims_long_tail() {
        let token = sample(
            &[7.0, 5.0, 3.0, 2.0, 1.0],
            SamplingConfig {
                tail_free_z: Some(0.4),
                ..SamplingConfig::default()
            },
            0.99,
        )
        .expect("sampling should succeed");
        assert!(token <= 2);
    }

    #[test]
    fn locally_typical_sampling_prefers_entropy_band() {
        let token = sample(
            &[6.0, 5.9, 2.0, 1.5],
            SamplingConfig {
                locally_typical_tau: Some(0.7),
                ..SamplingConfig::default()
            },
            0.99,
        )
        .expect("sampling should succeed");
        assert!(token <= 1);
    }

    #[test]
    fn mirostat_sampling_returns_updated_mu() {
        let (token, next_mu) = sample_mirostat(
            &[5.0, 4.0, 3.0, 2.0],
            1.0,
            MirostatConfig {
                tau: 2.0,
                eta: 0.1,
                mu: 4.0,
            },
            0.4,
        )
        .expect("mirostat should succeed");
        assert!(token <= 3);
        assert!(next_mu.is_finite());
        assert_ne!(next_mu, 4.0);
    }

    #[test]
    fn rejects_invalid_sampling_inputs() {
        assert_eq!(
            sample(&[], SamplingConfig::default(), 0.3),
            Err(SamplingError::EmptyLogits)
        );
        assert_eq!(
            sample(
                &[1.0],
                SamplingConfig {
                    temperature: 0.0,
                    ..SamplingConfig::default()
                },
                0.3
            ),
            Err(SamplingError::InvalidTemperature)
        );
        assert_eq!(
            sample(
                &[1.0],
                SamplingConfig {
                    top_k: Some(0),
                    ..SamplingConfig::default()
                },
                0.3
            ),
            Err(SamplingError::InvalidTopK)
        );
        assert_eq!(
            sample(
                &[1.0],
                SamplingConfig {
                    top_p: Some(1.5),
                    ..SamplingConfig::default()
                },
                0.3
            ),
            Err(SamplingError::InvalidTopP)
        );
        assert_eq!(
            sample(
                &[1.0],
                SamplingConfig {
                    min_p: Some(-0.1),
                    ..SamplingConfig::default()
                },
                0.3
            ),
            Err(SamplingError::InvalidMinP)
        );
        assert_eq!(
            sample(
                &[1.0],
                SamplingConfig {
                    typical_p: Some(0.0),
                    ..SamplingConfig::default()
                },
                0.3
            ),
            Err(SamplingError::InvalidTypicalP)
        );
        assert_eq!(
            sample(
                &[1.0],
                SamplingConfig {
                    tail_free_z: Some(1.2),
                    ..SamplingConfig::default()
                },
                0.3
            ),
            Err(SamplingError::InvalidTailFreeZ)
        );
        assert_eq!(
            sample(
                &[1.0],
                SamplingConfig {
                    locally_typical_tau: Some(0.0),
                    ..SamplingConfig::default()
                },
                0.3
            ),
            Err(SamplingError::InvalidLocallyTypicalTau)
        );
        assert_eq!(
            sample(&[1.0], SamplingConfig::default(), 1.0),
            Err(SamplingError::InvalidRandom)
        );
        assert_eq!(
            sample_mirostat(
                &[1.0],
                1.0,
                MirostatConfig {
                    tau: 0.0,
                    eta: 0.1,
                    mu: 1.0
                },
                0.3
            ),
            Err(SamplingError::InvalidMirostat)
        );
    }
}
