//! Load a GGUF and run architecture-aware DPO / PPO on its frozen forward.

use std::path::Path;
use std::time::Instant;

use oxidize_core::gguf::load_mapped_gguf;
use oxidize_core::inference::InferenceConfig;
use oxidize_core::layer_wise::LayerWiseModel;
use oxidize_core::model::Model;
use oxidize_core::tokenizer::{LoadedTokenizer, load_tokenizer_from_gguf_metadata};

use crate::arch::{ArchPlan, continuation_span};
use crate::config::FinetuneConfig;
use crate::dpo::{DpoConfig, DpoExample, DpoReport, DpoTrainer, sequence_logprob};
use crate::error::{FinetuneError, Result};
use crate::rlhf::{PpoConfig, PpoReport, PpoStepReport, PpoTrainer, RolloutBuffer};

pub struct CausalModel {
    pub model: LayerWiseModel,
    pub tokenizer: LoadedTokenizer,
    pub plan: ArchPlan,
}

pub struct ScoredSpan {
    pub hidden: Vec<f32>,
    pub base_logits: Vec<f32>,
    pub targets: Vec<u32>,
}

pub struct DpoRun {
    pub report: DpoReport,
    pub trainer: DpoTrainer,
}

pub struct PpoRun {
    pub report: PpoReport,
    pub trainer: PpoTrainer,
}

pub fn load_causal_model(
    path: &Path,
    context_cap: usize,
    allow_partial: bool,
) -> Result<CausalModel> {
    let mapped =
        load_mapped_gguf(path).map_err(|e| FinetuneError::Model(format!("load GGUF: {e}")))?;
    let tokenizer = load_tokenizer_from_gguf_metadata(&mapped.parsed().metadata)
        .map_err(|e| FinetuneError::Model(format!("load tokenizer: {e:?}")))?;
    let mut cfg = InferenceConfig::from_gguf(&mapped);
    if context_cap > 0 {
        cfg.context_size = cfg.context_size.min(context_cap);
    }
    let plan = ArchPlan::from_config(&cfg);
    plan.ensure_supported(allow_partial)?;
    let mut model =
        LayerWiseModel::load_from_gguf(&mapped, cfg, 0).map_err(|e| FinetuneError::Model(e))?;
    model
        .warm_layer_cache()
        .map_err(|e| FinetuneError::Model(format!("warm layer cache: {e}")))?;
    Ok(CausalModel {
        model,
        tokenizer,
        plan,
    })
}

/// Keep `prompt + continuation` inside `max_len`, reserving at least one
/// continuation token when the prompt itself is too long.
pub fn clip_prompt_continuation(
    prompt: &[u32],
    continuation: &[u32],
    max_len: usize,
) -> (Vec<u32>, usize) {
    let max_len = max_len.max(2);
    let mut prompt = prompt.to_vec();
    if prompt.len() + 1 > max_len {
        prompt.truncate(max_len - 1);
    }
    let room = max_len - prompt.len();
    let cont_len = continuation.len().min(room);
    let prompt_len = prompt.len();
    prompt.extend_from_slice(&continuation[..cont_len]);
    (prompt, prompt_len)
}

pub fn score_continuation(
    model: &mut LayerWiseModel,
    token_ids: &[u32],
    prompt_len: usize,
    window: usize,
) -> Result<ScoredSpan> {
    let Some((start, n)) = continuation_span(prompt_len, token_ids.len()) else {
        return Err(FinetuneError::EmptyDataset);
    };
    let h = model.config().hidden_size;
    let vocab = model.config().vocab_size;
    let window = window.max(1);
    model
        .rewind_to(0)
        .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
    let inputs = &token_ids[..token_ids.len() - 1];
    let mut hidden = vec![0.0_f32; n * h];
    let mut base_logits = vec![0.0_f32; n * vocab];
    let mut filled = 0usize;
    let mut pos = 0usize;
    while pos < inputs.len() {
        let end = (pos + window).min(inputs.len());
        let kk = end - pos;
        let normed = model
            .forward_normed_hidden(&inputs[pos..end], pos)
            .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
        let mut logits = vec![0.0_f32; kk * vocab];
        model
            .lm_head_logits_batch(&normed, kk, &mut logits)
            .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
        for t in 0..kk {
            let abs = pos + t;
            if abs >= start && filled < n {
                hidden[filled * h..(filled + 1) * h].copy_from_slice(&normed[t * h..(t + 1) * h]);
                base_logits[filled * vocab..(filled + 1) * vocab]
                    .copy_from_slice(&logits[t * vocab..(t + 1) * vocab]);
                filled += 1;
            }
        }
        pos = end;
    }
    if filled != n {
        return Err(FinetuneError::Model(format!(
            "scored {filled} positions, expected {n}"
        )));
    }
    let targets = token_ids[start + 1..start + 1 + n].to_vec();
    Ok(ScoredSpan {
        hidden,
        base_logits,
        targets,
    })
}

pub fn train_dpo_on_model(
    model: &mut LayerWiseModel,
    examples: &mut [DpoExample],
    encode: impl Fn(&str) -> Vec<u32>,
    finetune: &FinetuneConfig,
    dpo_config: DpoConfig,
) -> Result<DpoRun> {
    if examples.is_empty() {
        return Err(FinetuneError::EmptyDataset);
    }
    for example in examples.iter_mut() {
        example.apply_tokenizer(&encode);
    }
    let h = model.config().hidden_size;
    let vocab = model.config().vocab_size;
    let mut trainer = DpoTrainer::new(h, vocab, finetune.clone(), dpo_config);
    let window = finetune.window.max(1);
    let max_len = finetune.max_seq_len.max(2);
    let started = Instant::now();
    let mut total = 0.0_f32;
    let mut steps = 0usize;

    for _ in 0..finetune.epochs.max(1) {
        for example in examples.iter() {
            let (chosen_ids, chosen_prompt) =
                clip_prompt_continuation(&example.prompt, &example.chosen, max_len);
            let (rejected_ids, rejected_prompt) =
                clip_prompt_continuation(&example.prompt, &example.rejected, max_len);
            if chosen_ids.len() <= chosen_prompt || rejected_ids.len() <= rejected_prompt {
                continue;
            }
            let chosen = score_continuation(model, &chosen_ids, chosen_prompt, window)?;
            let rejected = score_continuation(model, &rejected_ids, rejected_prompt, window)?;
            let loss = trainer.train_step_with_base(
                &chosen.targets,
                &rejected.targets,
                &chosen.hidden,
                &rejected.hidden,
                &chosen.base_logits,
                &rejected.base_logits,
                example.ref_chosen_logprob,
                example.ref_rejected_logprob,
            )?;
            total += loss;
            steps += 1;
        }
    }
    if steps == 0 {
        return Err(FinetuneError::EmptyDataset);
    }
    let elapsed = started.elapsed().as_secs_f32();
    Ok(DpoRun {
        report: DpoReport {
            steps,
            mean_loss: total / steps as f32,
            elapsed_seconds: elapsed,
        },
        trainer,
    })
}

pub fn train_ppo_on_model(
    model: &mut LayerWiseModel,
    prompts: &[String],
    encode: impl Fn(&str) -> Vec<u32>,
    finetune: &FinetuneConfig,
    ppo_config: PpoConfig,
    max_new_tokens: usize,
) -> Result<PpoRun> {
    if prompts.is_empty() {
        return Err(FinetuneError::EmptyDataset);
    }
    let h = model.config().hidden_size;
    let vocab = model.config().vocab_size;
    let mut trainer = PpoTrainer::new(h, vocab, finetune.clone(), ppo_config.clone());
    let max_new = max_new_tokens.max(1);
    let max_len = finetune.max_seq_len.max(2);
    let started = Instant::now();
    let mut steps = 0usize;
    let mut transitions = 0usize;
    let mut sum_policy = 0.0_f32;
    let mut sum_value = 0.0_f32;
    let mut sum_entropy = 0.0_f32;
    let mut sum_kl = 0.0_f32;

    for _ in 0..finetune.epochs.max(1) {
        for prompt in prompts {
            let mut ids = encode(prompt);
            ids.truncate(max_len.saturating_sub(1).max(1));
            if ids.is_empty() {
                continue;
            }
            let buffer = rollout(model, &mut trainer, &ids, max_new, finetune.learning_rate)?;
            if buffer.is_empty() {
                continue;
            }
            let mut scored = buffer;
            scored.compute_gae(&ppo_config, 0.0);
            let step = trainer.train_step(&scored);
            accumulate(
                &mut steps,
                &mut transitions,
                &mut sum_policy,
                &mut sum_value,
                &mut sum_entropy,
                &mut sum_kl,
                &step,
            );
        }
    }
    if steps == 0 {
        return Err(FinetuneError::EmptyDataset);
    }
    let n = steps as f32;
    Ok(PpoRun {
        report: PpoReport {
            steps,
            transitions,
            mean_policy_loss: sum_policy / n,
            mean_value_loss: sum_value / n,
            mean_entropy: sum_entropy / n,
            mean_kl: sum_kl / n,
            elapsed_seconds: started.elapsed().as_secs_f32(),
        },
        trainer,
    })
}

fn accumulate(
    steps: &mut usize,
    transitions: &mut usize,
    sum_policy: &mut f32,
    sum_value: &mut f32,
    sum_entropy: &mut f32,
    sum_kl: &mut f32,
    step: &PpoStepReport,
) {
    *steps += 1;
    *transitions += step.n;
    *sum_policy += step.policy_loss;
    *sum_value += step.value_loss;
    *sum_entropy += step.entropy;
    *sum_kl += step.kl;
}

fn rollout(
    model: &mut LayerWiseModel,
    trainer: &mut PpoTrainer,
    prompt: &[u32],
    max_new: usize,
    critic_lr: f32,
) -> Result<RolloutBuffer> {
    let h = model.config().hidden_size;
    let vocab = model.config().vocab_size;
    let ctx = model.config().context_size;
    model
        .rewind_to(0)
        .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
    let normed = model
        .forward_normed_hidden(prompt, 0)
        .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
    if normed.len() < h {
        return Err(FinetuneError::Model(
            "prompt forward returned an empty hidden state".into(),
        ));
    }
    let mut hidden = normed[normed.len() - h..].to_vec();
    let mut pos = prompt.len();
    let mut buffer = RolloutBuffer::new();
    let critic_lr = critic_lr.abs().min(1e-3);

    for _ in 0..max_new {
        if pos >= ctx {
            break;
        }
        let mut base = vec![0.0_f32; vocab];
        model
            .lm_head_logits_batch(&hidden, 1, &mut base)
            .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
        let mut policy = base.clone();
        trainer.lora.forward_batch(&hidden, &mut policy, 1)?;
        let action = argmax(&policy);
        let log_prob = sequence_logprob(&policy, &[action], vocab);
        let reward = sequence_logprob(&base, &[action], vocab);
        let value = trainer.reward_model.score(&hidden);
        buffer.add(hidden.clone(), action, reward, log_prob, value);
        trainer.reward_model.sgd(&hidden, reward, critic_lr);

        let next = model
            .forward_normed_hidden(&[action], pos)
            .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
        hidden = next;
        pos += 1;
    }
    Ok(buffer)
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_reserves_a_continuation_token() {
        let (ids, prompt_len) = clip_prompt_continuation(&[1, 2, 3, 4, 5], &[9, 8, 7], 4);
        assert_eq!(prompt_len, 3);
        assert_eq!(ids, vec![1, 2, 3, 9]);
    }
}
