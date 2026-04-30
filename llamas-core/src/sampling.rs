#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            min_p: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingError {
    EmptyLogits,
    InvalidTemperature,
    InvalidTopK,
    InvalidTopP,
    InvalidMinP,
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
            sample(&[1.0], SamplingConfig::default(), 1.0),
            Err(SamplingError::InvalidRandom)
        );
    }
}
