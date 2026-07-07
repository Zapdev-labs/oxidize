//! Autoregressive generation with a trainable LoRA adapter on the LM head.

use oxidize_core::layer_wise::LayerWiseModel;
use oxidize_core::model::Model;
use oxidize_core::sampling::{SamplingConfig, greedy, sample};

use crate::error::{FinetuneError, Result};
use crate::lora::LoRAAdapter;

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub eos_token: Option<u32>,
    pub seed: u64,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 0.7,
            top_k: Some(40),
            top_p: Some(0.9),
            eos_token: None,
            seed: 42,
        }
    }
}

/// Simple xorshift64 PRNG returning floats in [0, 1).
fn next_rand(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 32) as u32 as f32) / (u32::MAX as f32)
}

fn sampling_config(generate_cfg: &GenerateConfig) -> SamplingConfig {
    SamplingConfig {
        temperature: generate_cfg.temperature.max(1e-6),
        top_k: generate_cfg.top_k,
        top_p: generate_cfg.top_p,
        ..SamplingConfig::default()
    }
}

fn logits_for_position(
    model: &LayerWiseModel,
    lora: &LoRAAdapter,
    normed: &[f32],
    logits: &mut [f32],
) -> Result<()> {
    let vocab = model.config().vocab_size;
    model
        .lm_head_logits_batch(normed, 1, logits)
        .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
    lora.forward_batch(normed, logits, 1)?;
    Ok(())
}

/// Generate continuation tokens for `prompt` using base weights + LoRA residual.
pub fn generate_with_lora(
    model: &mut LayerWiseModel,
    lora: &LoRAAdapter,
    prompt: &[u32],
    config: &GenerateConfig,
) -> Result<Vec<u32>> {
    if prompt.is_empty() && config.max_new_tokens == 0 {
        return Ok(Vec::new());
    }

    model
        .rewind_to(0)
        .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;

    let vocab = model.config().vocab_size;
    let hidden = model.config().hidden_size;
    let mut logits = vec![0.0_f32; vocab];
    let mut rng = config.seed.wrapping_add(1);
    let samp = sampling_config(config);

    if !prompt.is_empty() {
        let normed = model
            .forward_normed_hidden(prompt, 0)
            .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
        let last = &normed[(prompt.len() - 1) * hidden..prompt.len() * hidden];
        logits_for_position(model, lora, last, &mut logits)?;
    }

    let mut out = Vec::with_capacity(config.max_new_tokens);
    let mut pos = prompt.len();

    for _ in 0..config.max_new_tokens {
        let token = if config.temperature <= 1e-6 {
            greedy(&logits).map_err(|e| FinetuneError::Model(format!("{e:?}")))?
        } else {
            let r = next_rand(&mut rng);
            sample(&logits, samp.clone(), r).map_err(|e| FinetuneError::Model(format!("{e:?}")))?
        };

        if config.eos_token == Some(token) {
            break;
        }
        out.push(token);

        let normed = model
            .forward_normed_hidden(&[token], pos)
            .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
        pos += 1;
        logits_for_position(model, lora, &normed, &mut logits)?;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_config_defaults() {
        let c = GenerateConfig::default();
        assert_eq!(c.max_new_tokens, 128);
        assert!(c.temperature > 0.0);
    }

    #[test]
    fn prng_is_deterministic() {
        let mut a = 42u64;
        let mut b = 42u64;
        for _ in 0..10 {
            assert_eq!(next_rand(&mut a), next_rand(&mut b));
        }
    }
}
