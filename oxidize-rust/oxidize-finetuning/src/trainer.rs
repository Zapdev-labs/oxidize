use std::time::Instant;

use oxidize_core::layer_wise::LayerWiseModel;
use oxidize_core::model::Model;

use crate::config::FinetuneConfig;
use crate::dataset::SftExample;
use crate::error::{FinetuneError, Result};
use crate::fused::{cross_entropy_grad_batch, softmax_cross_entropy_batch};
use crate::lora::{LoRAAdapter, LoRATarget};

#[derive(Debug, Clone)]
pub struct FinetuneReport {
    pub steps: usize,
    pub tokens: usize,
    pub mean_loss: f32,
    pub epoch_losses: Vec<f32>,
    pub tokens_per_second: f32,
    pub elapsed_seconds: f32,
}

/// SFT trainer: frozen quantized base (batched layer-major windows through
/// `LayerWiseModel`) + trainable LoRA on the LM head.
///
/// Throughput design (the "faster than per-token" plan):
/// - windows of `config.window` positions run as GEMMs, amortizing one pass
///   over the quantized weights across the whole window instead of re-reading
///   ~all of the model per token;
/// - logits/grad buffers are allocated once and reused across windows;
/// - cross-entropy converts logits to gradients in place (no second
///   window×vocab buffer);
/// - all LoRA forward/backward/optimizer math is rayon-parallel and batched.
pub struct SftTrainer {
    pub config: FinetuneConfig,
    pub output_lora: LoRAAdapter,
    /// (directory, every_n_optimizer_steps) for periodic adapter checkpoints.
    pub checkpoint: Option<(std::path::PathBuf, usize)>,
}

impl SftTrainer {
    pub fn for_model(model: &LayerWiseModel, config: FinetuneConfig) -> Self {
        let h = model.config().hidden_size;
        let vocab = model.config().vocab_size;
        Self {
            config: config.clone(),
            output_lora: LoRAAdapter::new(LoRATarget::OutputHead, h, vocab, &config),
            checkpoint: None,
        }
    }

    fn save_checkpoint(&self, label: &str) {
        if let Some((dir, _)) = &self.checkpoint {
            match crate::export::export_lora_gguf(
                dir,
                std::slice::from_ref(&self.output_lora),
                self.config.rank,
                self.config.lora_scale(),
            ) {
                Ok(()) => println!("  checkpoint ({label}) -> {}", dir.display()),
                Err(e) => eprintln!("  checkpoint save failed: {e}"),
            }
        }
    }

    pub fn tokenize_examples(
        examples: &mut Vec<SftExample>,
        encode: impl Fn(&str) -> Vec<u32> + Sync,
        max_seq_len: usize,
    ) -> Result<()> {
        use rayon::prelude::*;
        // BPE encoding of a large-vocab tokenizer is the slowest part of setup
        // and is independent per example — run it across all cores.
        let cap = max_seq_len.saturating_mul(4).max(2);
        examples.par_iter_mut().for_each(|ex| {
            let mut ids = encode(&ex.text);
            // Packing splits overlong examples across chunks; still cap single
            // rows to bound pathological inputs.
            ids.truncate(cap);
            ex.token_ids = ids;
        });
        examples.retain(|e| e.token_ids.len() >= 2);
        if examples.is_empty() {
            return Err(FinetuneError::EmptyDataset);
        }
        Ok(())
    }

    /// Train over pre-packed chunks (see `dataset::pack_chunks`).
    pub fn train(
        &mut self,
        model: &mut LayerWiseModel,
        chunks: &[Vec<u32>],
    ) -> Result<FinetuneReport> {
        if chunks.is_empty() {
            return Err(FinetuneError::EmptyDataset);
        }
        let vocab = model.config().vocab_size;
        let window = self.config.window.max(2);
        let tokens_per_step = self.config.tokens_per_step.max(1);
        let grad_scale = 1.0 / tokens_per_step as f32;

        // Reused buffers: window × vocab is the big one (e.g. 64 × 248320 × 4B ≈ 64MB).
        let mut logits = vec![0.0_f32; window * vocab];

        let mut epoch_losses = Vec::with_capacity(self.config.epochs);
        let mut total_loss = 0.0_f32;
        let mut total_tokens = 0usize;
        let mut opt_step = 0usize;
        let mut accum_tokens = 0usize;
        let started = Instant::now();
        let mut last_report = Instant::now();
        let mut tokens_since_report = 0usize;

        for epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0_f32;
            let mut epoch_tokens = 0usize;

            for chunk in chunks {
                if chunk.len() < 2 {
                    continue;
                }
                model
                    .rewind_to(0)
                    .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
                let inputs = &chunk[..chunk.len() - 1];
                let targets = &chunk[1..];

                let mut pos = 0usize;
                while pos < inputs.len() {
                    let end = (pos + window).min(inputs.len());
                    let kk = end - pos;
                    let win_tokens = &inputs[pos..end];
                    let win_targets = &targets[pos..end];

                    let normed = model
                        .forward_normed_hidden(win_tokens, pos)
                        .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
                    let logits_w = &mut logits[..kk * vocab];
                    model
                        .lm_head_logits_batch(&normed, kk, logits_w)
                        .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
                    self.output_lora.forward_batch(&normed, logits_w, kk)?;

                    // In place: logits -> grad_scale * (softmax - onehot).
                    let (loss_sum, n) =
                        cross_entropy_grad_batch(logits_w, win_targets, vocab, grad_scale);
                    self.output_lora.backward_batch(&normed, logits_w, kk)?;

                    epoch_loss += loss_sum;
                    epoch_tokens += n;
                    total_loss += loss_sum;
                    total_tokens += n;
                    accum_tokens += n;
                    tokens_since_report += n;

                    if accum_tokens >= tokens_per_step {
                        opt_step += 1;
                        let lr = warmup_lr(
                            self.config.learning_rate,
                            opt_step,
                            self.config.warmup_steps,
                        );
                        self.output_lora
                            .step(lr, self.config.weight_decay, opt_step);
                        self.output_lora.zero_grad();
                        accum_tokens = 0;

                        if let Some((_, every)) = self.checkpoint
                            && every > 0
                            && opt_step.is_multiple_of(every)
                        {
                            self.save_checkpoint(&format!("step {opt_step}"));
                        }
                    }

                    if last_report.elapsed().as_secs_f32() >= 10.0 {
                        let tps = tokens_since_report as f32 / last_report.elapsed().as_secs_f32();
                        println!(
                            "  epoch {} step {} tokens {} loss {:.4} | {:.2} tok/s",
                            epoch + 1,
                            opt_step,
                            total_tokens,
                            if epoch_tokens > 0 {
                                epoch_loss / epoch_tokens as f32
                            } else {
                                0.0
                            },
                            tps
                        );
                        last_report = Instant::now();
                        tokens_since_report = 0;
                    }

                    pos = end;
                }
            }

            if epoch_tokens > 0 {
                epoch_losses.push(epoch_loss / epoch_tokens as f32);
            }
        }

        // Flush a trailing partial accumulation so its gradients aren't lost.
        if accum_tokens > 0 {
            opt_step += 1;
            let lr = warmup_lr(
                self.config.learning_rate,
                opt_step,
                self.config.warmup_steps,
            );
            self.output_lora
                .step(lr, self.config.weight_decay, opt_step);
            self.output_lora.zero_grad();
        }

        let elapsed = started.elapsed().as_secs_f32();
        Ok(FinetuneReport {
            steps: opt_step,
            tokens: total_tokens,
            mean_loss: if total_tokens > 0 {
                total_loss / total_tokens as f32
            } else {
                0.0
            },
            epoch_losses,
            tokens_per_second: if elapsed > 0.0 {
                total_tokens as f32 / elapsed
            } else {
                0.0
            },
            elapsed_seconds: elapsed,
        })
    }

    /// Mean loss over pre-packed chunks, no gradient work.
    pub fn eval_loss(&self, model: &mut LayerWiseModel, chunks: &[Vec<u32>]) -> Result<f32> {
        let vocab = model.config().vocab_size;
        let window = self.config.window.max(2);
        let mut logits = vec![0.0_f32; window * vocab];
        let mut sum = 0.0_f32;
        let mut n = 0usize;

        for chunk in chunks {
            if chunk.len() < 2 {
                continue;
            }
            model
                .rewind_to(0)
                .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
            let inputs = &chunk[..chunk.len() - 1];
            let targets = &chunk[1..];
            let mut pos = 0usize;
            while pos < inputs.len() {
                let end = (pos + window).min(inputs.len());
                let kk = end - pos;
                let normed = model
                    .forward_normed_hidden(&inputs[pos..end], pos)
                    .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
                let logits_w = &mut logits[..kk * vocab];
                model
                    .lm_head_logits_batch(&normed, kk, logits_w)
                    .map_err(|e| FinetuneError::Model(format!("{e:?}")))?;
                self.output_lora.forward_batch(&normed, logits_w, kk)?;
                let (loss_sum, count) =
                    softmax_cross_entropy_batch(logits_w, &targets[pos..end], vocab);
                sum += loss_sum;
                n += count;
                pos = end;
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
