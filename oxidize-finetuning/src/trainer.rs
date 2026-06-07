use oxidize_core::inference::InferenceModel;
use oxidize_core::model::{Model, Session};

use crate::config::FinetuneConfig;
use crate::dataset::SftExample;
use crate::error::{FinetuneError, Result};
use crate::fused::{cross_entropy_grad, softmax_cross_entropy};
use crate::lora::{LoRAAdapter, LoRATarget};

#[derive(Debug, Clone)]
pub struct FinetuneReport {
    pub steps: usize,
    pub tokens: usize,
    pub mean_loss: f32,
    pub epoch_losses: Vec<f32>,
}

pub struct SftTrainer {
    pub config: FinetuneConfig,
    pub output_lora: LoRAAdapter,
}

impl SftTrainer {
    pub fn for_model(model: &InferenceModel, config: FinetuneConfig) -> Self {
        let h = model.config_hidden_size();
        let vocab = model.config().vocab_size;
        Self {
            config: config.clone(),
            output_lora: LoRAAdapter::new(LoRATarget::OutputHead, h, vocab, &config),
        }
    }

    pub fn tokenize_examples(
        examples: &mut Vec<SftExample>,
        encode: impl Fn(&str) -> Vec<u32>,
        max_seq_len: usize,
    ) -> Result<()> {
        for ex in examples.iter_mut() {
            let mut ids = encode(&ex.text);
            if ids.len() > max_seq_len {
                ids.truncate(max_seq_len);
            }
            if ids.len() < 2 {
                continue;
            }
            ex.token_ids = ids;
        }
        examples.retain(|e| e.token_ids.len() >= 2);
        if examples.is_empty() {
            return Err(FinetuneError::EmptyDataset);
        }
        Ok(())
    }

    pub fn train(
        &mut self,
        model: &mut InferenceModel,
        examples: &[SftExample],
    ) -> Result<FinetuneReport> {
        if examples.is_empty() {
            return Err(FinetuneError::EmptyDataset);
        }
        let h = model.config_hidden_size();
        let vocab = model.config().vocab_size;
        #[allow(unused_assignments)]
        let mut session = Session::new();
        let mut epoch_losses = Vec::with_capacity(self.config.epochs);
        let mut total_loss = 0.0_f32;
        let mut total_steps = 0usize;
        let mut total_tokens = 0usize;
        let mut opt_step = 0usize;
        let mut accum = 0usize;

        let mut normed = vec![0.0_f32; h];
        let mut logits = vec![0.0_f32; vocab];
        let mut grad_logits = vec![0.0_f32; vocab];

        for _epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0_f32;
            let mut epoch_steps = 0usize;

            for example in examples {
                let ids = &example.token_ids;
                if ids.len() < 2 {
                    continue;
                }
                model
                    .rewind_to(0)
                    .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
                session = Session::new();

                for pos in 0..ids.len() - 1 {
                    let token = ids[pos];
                    let target = ids[pos + 1] as usize;

                    model.embed_token_into_workspace(token);
                    model
                        .run_layer_range_in_workspace(pos, 0..model.config().layer_count)
                        .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;

                    let hidden = model.hidden_state();
                    model
                        .apply_final_norm(hidden, &mut normed)
                        .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;

                    logits.fill(0.0_f32);
                    model
                        .lm_head_logits_from_normed(&normed, &mut logits)
                        .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;

                    self.output_lora.forward(&normed, &mut logits)?;

                    grad_logits.fill(0.0_f32);
                    let loss = cross_entropy_grad(&logits, target.min(vocab - 1), &mut grad_logits);
                    epoch_loss += loss;
                    total_loss += loss;
                    epoch_steps += 1;
                    total_steps += 1;
                    total_tokens += 1;
                    accum += 1;

                    if accum >= self.config.gradient_accumulation_steps {
                        opt_step += 1;
                        let lr = warmup_lr(
                            self.config.learning_rate,
                            opt_step,
                            self.config.warmup_steps,
                        );
                        self.output_lora.zero_grad();
                        self.output_lora.backward_and_step(
                            &normed,
                            &grad_logits,
                            lr,
                            self.config.weight_decay,
                            opt_step,
                        )?;
                        accum = 0;
                    }

                    session.record_tokens(1);
                }
            }

            if epoch_steps > 0 {
                epoch_losses.push(epoch_loss / epoch_steps as f32);
            }
        }

        Ok(FinetuneReport {
            steps: total_steps,
            tokens: total_tokens,
            mean_loss: if total_steps > 0 {
                total_loss / total_steps as f32
            } else {
                0.0
            },
            epoch_losses,
        })
    }

    pub fn eval_loss(&self, model: &mut InferenceModel, examples: &[SftExample]) -> Result<f32> {
        let h = model.config_hidden_size();
        let vocab = model.config().vocab_size;
        #[allow(unused_assignments)]
        let mut session = Session::new();
        let mut normed = vec![0.0_f32; h];
        let mut logits = vec![0.0_f32; vocab];
        let mut sum = 0.0_f32;
        let mut n = 0usize;

        for example in examples {
            let ids = &example.token_ids;
            if ids.len() < 2 {
                continue;
            }
            model
                .rewind_to(0)
                .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
            session = Session::new();
            for pos in 0..ids.len() - 1 {
                let token = ids[pos];
                let target = ids[pos + 1] as usize;
                model.embed_token_into_workspace(token);
                model
                    .run_layer_range_in_workspace(pos, 0..model.config().layer_count)
                    .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
                model
                    .apply_final_norm(model.hidden_state(), &mut normed)
                    .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
                logits.fill(0.0_f32);
                model
                    .lm_head_logits_from_normed(&normed, &mut logits)
                    .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
                self.output_lora.forward(&normed, &mut logits)?;
                sum += softmax_cross_entropy(&logits, target.min(vocab - 1));
                n += 1;
                session.record_tokens(1);
            }
        }
        Ok(if n > 0 { sum / n as f32 } else { 0.0 })
    }
}

fn warmup_lr(base: f32, step: usize, warmup: usize) -> f32 {
    if warmup == 0 || step >= warmup {
        base
    } else {
        base * (step as f32 / warmup as f32)
    }
}
