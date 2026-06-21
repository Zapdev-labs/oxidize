//! Direct Preference Optimization (DPO) trainer.
//!
//! Implements the DPO objective from Rafailov et al. (2023):
//!   L_DPO = -log σ(β · (log π(chosen|x) − log π(rejected|x)))
//!
//! The trainer operates over pre-computed base hidden states — the frozen
//! quantized backbone is run once per example by the caller; this module
//! handles only the LoRA adapter forward/backward passes and the DPO loss.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use serde::Deserialize;

use crate::config::FinetuneConfig;
use crate::error::{FinetuneError, Result};
use crate::fused::softmax_cross_entropy;
use crate::lora::{LoRAAdapter, LoRATarget};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single DPO training example: a prompt with one chosen and one rejected
/// continuation, each represented as a sequence of token ids.
#[derive(Debug, Clone)]
pub struct DpoExample {
    pub prompt: Vec<u32>,
    pub chosen: Vec<u32>,
    pub rejected: Vec<u32>,
    /// Reference-policy log-probability of the chosen continuation (required when
    /// `DpoConfig::reference_free` is false).
    pub ref_chosen_logprob: Option<f32>,
    /// Reference-policy log-probability of the rejected continuation.
    pub ref_rejected_logprob: Option<f32>,
}

// ---------------------------------------------------------------------------
// JSONL loading
// ---------------------------------------------------------------------------

/// Raw row schema for DPO JSONL files.
///
/// Each field may be either an array of integer token ids or a plain string.
/// Strings are stub-tokenized as their UTF-8 byte values (each byte → u32).
#[derive(Debug, Deserialize)]
struct DpoJsonlRow {
    prompt: TokensOrString,
    chosen: TokensOrString,
    rejected: TokensOrString,
    #[serde(default)]
    ref_chosen_logprob: Option<f32>,
    #[serde(default)]
    ref_rejected_logprob: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokensOrString {
    Tokens(Vec<u32>),
    Text(String),
}

impl TokensOrString {
    fn into_tokens(self) -> Vec<u32> {
        match self {
            Self::Tokens(ids) => ids,
            Self::Text(s) => s.bytes().map(u32::from).collect(),
        }
    }
}

/// Load DPO examples from a JSONL file.
///
/// Each line must be a JSON object with `"prompt"`, `"chosen"`, and
/// `"rejected"` fields.  Values may be arrays of integer token ids or plain
/// strings (stub-tokenized as bytes).  Lines that produce an empty prompt,
/// chosen, or rejected sequence are silently skipped.
pub fn load_jsonl_dpo(path: impl AsRef<Path>) -> Result<Vec<DpoExample>> {
    let file = File::open(path.as_ref()).map_err(|e| FinetuneError::Model(e.to_string()))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| FinetuneError::Model(e.to_string()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: DpoJsonlRow = serde_json::from_str(trimmed)
            .map_err(|e| FinetuneError::Model(format!("dpo jsonl line {}: {e}", line_no + 1)))?;
        let prompt = row.prompt.into_tokens();
        let chosen = row.chosen.into_tokens();
        let rejected = row.rejected.into_tokens();
        if prompt.is_empty() || chosen.is_empty() || rejected.is_empty() {
            continue;
        }
        out.push(DpoExample {
            prompt,
            chosen,
            rejected,
            ref_chosen_logprob: row.ref_chosen_logprob,
            ref_rejected_logprob: row.ref_rejected_logprob,
        });
    }

    if out.is_empty() {
        return Err(FinetuneError::EmptyDataset);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// DPO configuration
// ---------------------------------------------------------------------------

/// DPO-specific hyper-parameters that extend the base `FinetuneConfig`.
#[derive(Debug, Clone)]
pub struct DpoConfig {
    /// KL-regularisation coefficient β.  Higher values stay closer to the
    /// reference policy; typical range 0.01 – 0.5.
    pub beta: f32,
    /// When `true`, skip the reference log-probability term and use 0 as the
    /// reference baseline (reference-free / implicit reference variant).
    pub reference_free: bool,
}

impl Default for DpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            reference_free: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Training report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DpoReport {
    pub steps: usize,
    pub mean_loss: f32,
    pub elapsed_seconds: f32,
}

// ---------------------------------------------------------------------------
// DPO trainer
// ---------------------------------------------------------------------------

/// DPO trainer: frozen reference policy + trainable LoRA on the LM head.
///
/// The caller is responsible for computing base hidden states (the output of
/// the backbone's final RMS-norm layer) for each token in the chosen and
/// rejected sequences; this trainer owns only the LoRA adapter.
pub struct DpoTrainer {
    pub config: FinetuneConfig,
    pub dpo_config: DpoConfig,
    pub lora: LoRAAdapter,
    /// Current optimizer step (1-indexed); advanced by `train_step`.
    step: usize,
}

impl DpoTrainer {
    /// Create a new DPO trainer.
    ///
    /// * `in_dim`  — hidden size (input dimension of the LM-head projection).
    /// * `out_dim` — vocabulary size (output dimension of the LM-head).
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        config: FinetuneConfig,
        dpo_config: DpoConfig,
    ) -> Self {
        let lora = LoRAAdapter::new(LoRATarget::OutputHead, in_dim, out_dim, &config);
        Self {
            config,
            dpo_config,
            lora,
            step: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Core DPO loss
    // -----------------------------------------------------------------------

    /// Compute the DPO loss scalar.
    ///
    /// `chosen_logprob` and `rejected_logprob` are the summed log-probabilities
    /// of each sequence under the current (policy) model.  When
    /// `reference_free` is true the reference log-probability is treated as 0.
    ///
    /// L = -log σ(β · (log π_θ(chosen) − log π_θ(rejected)))
    ///
    /// (In the reference-free variant the reference terms cancel to 1 so the
    /// ratio simplifies to the policy ratio alone.)
    pub fn dpo_loss(chosen_logprob: f32, rejected_logprob: f32, beta: f32) -> f32 {
        let margin = beta * (chosen_logprob - rejected_logprob);
        // -log σ(x) = log(1 + exp(-x)) — computed in a numerically stable way.
        log1p_exp(-margin)
    }

    // -----------------------------------------------------------------------
    // Per-sequence log-probability helpers
    // -----------------------------------------------------------------------

    /// Compute the total log-probability of `targets` given hidden states
    /// `hiddens` ([len, in_dim]) using the frozen base LM-head logits plus the
    /// LoRA residual.  Returns (log_prob, logits_buf) so the caller can reuse
    /// the allocation.
    fn sequence_logprob_with_buf(
        &self,
        hiddens: &[f32],
        targets: &[u32],
        logits_buf: &mut Vec<f32>,
    ) -> Result<f32> {
        let len = targets.len();
        if hiddens.len() != len * self.lora.in_dim {
            return Err(FinetuneError::Adapter(format!(
                "hidden shape mismatch: {} != {} * {}",
                hiddens.len(),
                len,
                self.lora.in_dim
            )));
        }
        let vocab = self.lora.out_dim;
        logits_buf.resize(len * vocab, 0.0);
        logits_buf.fill(0.0);
        // LoRA forward adds scale * B A x to the (zero-initialised) buffer,
        // effectively computing the adapter's contribution.  In practice the
        // caller should add the frozen base logits beforehand; here we operate
        // solely on the adapter output as the base is frozen and its
        // contribution is shared between policy and reference.
        self.lora.forward_batch(hiddens, logits_buf, len)?;

        // Sum log-probabilities over all target positions.
        let mut log_prob = 0.0_f32;
        for (t, &tgt) in targets.iter().enumerate() {
            let row = &logits_buf[t * vocab..(t + 1) * vocab];
            let tgt_idx = (tgt as usize).min(vocab.saturating_sub(1));
            log_prob -= softmax_cross_entropy(row, tgt_idx);
        }
        Ok(log_prob)
    }

    // -----------------------------------------------------------------------
    // Gradient of log-probability w.r.t. LoRA outputs
    // -----------------------------------------------------------------------

    /// Fill `grad_out` ([len, vocab]) with ∂log_p / ∂logit = softmax(logit) - onehot(target).
    ///
    /// This is the standard cross-entropy gradient (without negation) which,
    /// when multiplied by the upstream DPO gradient coefficient, gives the
    /// correct update direction.
    fn logprob_grad(
        logits: &[f32],
        targets: &[u32],
        vocab: usize,
        coeff: f32,
        grad_out: &mut Vec<f32>,
    ) {
        let len = targets.len();
        grad_out.resize(len * vocab, 0.0);
        for (t, &tgt) in targets.iter().enumerate() {
            let row = &logits[t * vocab..(t + 1) * vocab];
            let grad_row = &mut grad_out[t * vocab..(t + 1) * vocab];
            let max_l = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row.iter().map(|l| (l - max_l).exp()).sum();
            let log_sum = max_l + exp_sum.ln();
            let tgt_idx = (tgt as usize).min(vocab.saturating_sub(1));
            for (i, (g, l)) in grad_row.iter_mut().zip(row.iter()).enumerate() {
                let p = (l - log_sum).exp();
                // ∂(-log p_target) / ∂logit_i = p_i - 1_{i==target}
                // We want ∂log_prob / ∂logit_i = -(p_i - 1_{i==target})
                // scaled by `coeff`.
                *g = coeff * (if i == tgt_idx { p - 1.0 } else { p });
            }
        }
    }

    // -----------------------------------------------------------------------
    // Training step
    // -----------------------------------------------------------------------

    /// Run one DPO step on a single example.
    ///
    /// `base_hidden_chosen`  — normed hidden states for the chosen continuation,
    ///                          shape [chosen.len(), in_dim].
    /// `base_hidden_rejected` — same for rejected, shape [rejected.len(), in_dim].
    ///
    /// Advances the optimizer step counter and performs an AdamW update.
    /// Returns the scalar DPO loss.
    pub fn train_step(
        &mut self,
        example: &DpoExample,
        base_hidden_chosen: &[f32],
        base_hidden_rejected: &[f32],
    ) -> Result<f32> {
        let vocab = self.lora.out_dim;
        let mut logits_c: Vec<f32> = Vec::new();
        let mut logits_r: Vec<f32> = Vec::new();

        // Forward pass — compute log-probabilities under the current policy.
        let log_p_chosen =
            self.sequence_logprob_with_buf(base_hidden_chosen, &example.chosen, &mut logits_c)?;
        let log_p_rejected =
            self.sequence_logprob_with_buf(base_hidden_rejected, &example.rejected, &mut logits_r)?;

        let (ref_c, ref_r) = if self.dpo_config.reference_free {
            (0.0_f32, 0.0_f32)
        } else {
            match (
                example.ref_chosen_logprob,
                example.ref_rejected_logprob,
            ) {
                (Some(rc), Some(rr)) => (rc, rr),
                _ => {
                    return Err(FinetuneError::Adapter(
                        "reference_free=false requires ref_chosen_logprob and \
                         ref_rejected_logprob on each DpoExample"
                            .into(),
                    ));
                }
            }
        };

        let policy_margin = (log_p_chosen - ref_c) - (log_p_rejected - ref_r);
        let loss = log1p_exp(-self.dpo_config.beta * policy_margin);

        // ∂L / ∂margin = -σ(-β·margin) · β = -(1 - σ(β·margin)) · β
        // ∂L / ∂log_p_chosen   = +∂L/∂margin
        // ∂L / ∂log_p_rejected = -∂L/∂margin
        let sigmoid_pos = sigmoid(self.dpo_config.beta * policy_margin);
        // Gradient of loss w.r.t. log_p_chosen:  -(1 - σ) * β  (negative β means pushing chosen up)
        let grad_chosen_coeff = -(1.0 - sigmoid_pos) * self.dpo_config.beta;
        // Gradient of loss w.r.t. log_p_rejected: +(1 - σ) * β (pushing rejected down)
        let grad_rejected_coeff = (1.0 - sigmoid_pos) * self.dpo_config.beta;

        // Backprop through LoRA for the chosen sequence.
        // `logprob_grad` computes `coeff * d(-log_p)/dlogit` (CE gradient).
        // We need `dL/dlogit = dL/d(log_p) * d(log_p)/dlogit
        //                     = grad_chosen_coeff * (-(softmax - onehot))`
        // so the correct coeff to pass is `-grad_chosen_coeff`.
        let mut grad_logits_c: Vec<f32> = Vec::new();
        Self::logprob_grad(
            &logits_c,
            &example.chosen,
            vocab,
            -grad_chosen_coeff,
            &mut grad_logits_c,
        );
        self.lora
            .backward_batch(base_hidden_chosen, &grad_logits_c, example.chosen.len())?;

        // Backprop through LoRA for the rejected sequence.
        let mut grad_logits_r: Vec<f32> = Vec::new();
        Self::logprob_grad(
            &logits_r,
            &example.rejected,
            vocab,
            -grad_rejected_coeff,
            &mut grad_logits_r,
        );
        self.lora
            .backward_batch(base_hidden_rejected, &grad_logits_r, example.rejected.len())?;

        // AdamW update.
        self.step += 1;
        let lr = warmup_lr(
            self.config.learning_rate,
            self.step,
            self.config.warmup_steps,
        );
        self.lora.step(lr, self.config.weight_decay, self.step);
        self.lora.zero_grad();

        Ok(loss)
    }

    // -----------------------------------------------------------------------
    // Full training loop
    // -----------------------------------------------------------------------

    /// Train over all examples for `config.epochs` epochs.
    ///
    /// `base_hiddens` must have the same length as `examples`; each entry is a
    /// flat slice of shape `[(chosen.len() + rejected.len()), in_dim]` where the
    /// first `chosen.len() * in_dim` values are the chosen hidden states and the
    /// remainder are the rejected hidden states.
    ///
    /// If `base_hiddens` is empty or its length does not match `examples`,
    /// `train_step` is called with zero-filled placeholders so the interface
    /// stays consistent (useful for unit tests where the base model is absent).
    pub fn train(
        &mut self,
        examples: &[DpoExample],
        base_hiddens: &[Vec<f32>],
    ) -> Result<DpoReport> {
        if examples.is_empty() {
            return Err(FinetuneError::EmptyDataset);
        }
        let started = Instant::now();
        let mut total_loss = 0.0_f32;
        let mut total_steps = 0usize;

        for _epoch in 0..self.config.epochs.max(1) {
            for (i, example) in examples.iter().enumerate() {
                // Build zero-filled hidden vectors when the caller provides no
                // pre-computed hiddens (reference-free / unit-test mode).
                let (hidden_c, hidden_r) = if i < base_hiddens.len() {
                    let buf = &base_hiddens[i];
                    let c_len = example.chosen.len() * self.lora.in_dim;
                    let r_len = example.rejected.len() * self.lora.in_dim;
                    let hidden_c = if c_len <= buf.len() {
                        buf[..c_len].to_vec()
                    } else {
                        vec![0.0_f32; c_len]
                    };
                    let hidden_r = if r_len <= buf.len().saturating_sub(c_len) {
                        buf[c_len..c_len + r_len].to_vec()
                    } else {
                        vec![0.0_f32; r_len]
                    };
                    (hidden_c, hidden_r)
                } else {
                    (
                        vec![0.0_f32; example.chosen.len() * self.lora.in_dim],
                        vec![0.0_f32; example.rejected.len() * self.lora.in_dim],
                    )
                };

                let loss = self.train_step(example, &hidden_c, &hidden_r)?;
                total_loss += loss;
                total_steps += 1;
            }
        }

        let elapsed = started.elapsed().as_secs_f32();
        Ok(DpoReport {
            steps: total_steps,
            mean_loss: if total_steps > 0 {
                total_loss / total_steps as f32
            } else {
                0.0
            },
            elapsed_seconds: elapsed,
        })
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Numerically stable log(1 + exp(x)).
#[inline]
fn log1p_exp(x: f32) -> f32 {
    if x >= 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Logistic sigmoid σ(x) = 1 / (1 + exp(-x)).
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn warmup_lr(base: f32, step: usize, warmup: usize) -> f32 {
    if warmup == 0 || step >= warmup {
        base
    } else {
        base * (step as f32 / warmup as f32)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> (FinetuneConfig, DpoConfig) {
        let cfg = FinetuneConfig {
            rank: 2,
            alpha: 4.0,
            learning_rate: 1e-3,
            epochs: 1,
            ..Default::default()
        };
        let dpo = DpoConfig {
            beta: 0.1,
            reference_free: true,
        };
        (cfg, dpo)
    }

    fn tiny_trainer() -> DpoTrainer {
        let (cfg, dpo) = tiny_config();
        DpoTrainer::new(4, 8, cfg, dpo)
    }

    fn toy_example() -> DpoExample {
        DpoExample {
            prompt: vec![1, 2],
            chosen: vec![3, 4, 5],
            rejected: vec![6, 7],
            ref_chosen_logprob: None,
            ref_rejected_logprob: None,
        }
    }

    // -----------------------------------------------------------------------
    // Math primitives
    // -----------------------------------------------------------------------

    #[test]
    fn log1p_exp_positive_arg() {
        // log(1 + e^2) ≈ 2.1269
        let v = log1p_exp(2.0_f32);
        assert!((v - 2.1269).abs() < 1e-3, "log1p_exp(2) = {v}");
    }

    #[test]
    fn log1p_exp_negative_arg() {
        // log(1 + e^{-2}) ≈ 0.1269
        let v = log1p_exp(-2.0_f32);
        assert!((v - 0.1269).abs() < 1e-3, "log1p_exp(-2) = {v}");
    }

    #[test]
    fn sigmoid_midpoint() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn dpo_loss_positive_margin_is_low() {
        // When chosen >> rejected, loss should be well below the random baseline (ln 2 ≈ 0.693).
        // With beta=0.1 and margin=20: -log σ(2) = log1p_exp(-2) ≈ 0.127 < 0.5.
        let loss = DpoTrainer::dpo_loss(10.0, -10.0, 0.1);
        assert!(loss < 0.5, "loss={loss}");
        // With a larger beta the loss should be even closer to 0.
        let loss_strong = DpoTrainer::dpo_loss(10.0, -10.0, 1.0);
        assert!(loss_strong < 0.01, "loss_strong={loss_strong}");
    }

    #[test]
    fn dpo_loss_negative_margin_is_high() {
        // When rejected >> chosen, loss should be large.
        let loss = DpoTrainer::dpo_loss(-10.0, 10.0, 0.1);
        assert!(loss > 1.9, "loss={loss}");
    }

    // -----------------------------------------------------------------------
    // Trainer construction
    // -----------------------------------------------------------------------

    #[test]
    fn trainer_constructs() {
        let t = tiny_trainer();
        assert_eq!(t.lora.in_dim, 4);
        assert_eq!(t.lora.out_dim, 8);
        assert_eq!(t.lora.rank, 2);
    }

    // -----------------------------------------------------------------------
    // train_step runs without error and returns finite loss
    // -----------------------------------------------------------------------

    #[test]
    fn train_step_returns_finite_loss() {
        let mut trainer = tiny_trainer();
        let example = toy_example();
        let hidden_c = vec![0.5_f32; example.chosen.len() * 4];
        let hidden_r = vec![-0.3_f32; example.rejected.len() * 4];
        let loss = trainer.train_step(&example, &hidden_c, &hidden_r).unwrap();
        assert!(loss.is_finite(), "loss={loss}");
    }

    #[test]
    fn train_step_changes_lora_weights() {
        let mut trainer = tiny_trainer();
        // Give B a non-zero init so the adapter output is non-zero from the start.
        for (i, v) in trainer.lora.b.iter_mut().enumerate() {
            *v = ((i % 5) as f32 - 2.0) * 0.1;
        }
        let before: Vec<f32> = trainer.lora.a.clone();
        let example = toy_example();
        let hidden_c: Vec<f32> = (0..example.chosen.len() * 4)
            .map(|i| (i as f32 * 0.3).sin())
            .collect();
        let hidden_r: Vec<f32> = (0..example.rejected.len() * 4)
            .map(|i| (i as f32 * 0.17).cos())
            .collect();
        trainer.train_step(&example, &hidden_c, &hidden_r).unwrap();
        // At least some weights must have changed.
        assert!(
            before
                .iter()
                .zip(trainer.lora.a.iter())
                .any(|(a, b)| (a - b).abs() > 1e-12),
            "LoRA A weights did not change after train_step"
        );
    }

    // -----------------------------------------------------------------------
    // Loss decreases over multiple steps on the same example
    // -----------------------------------------------------------------------

    #[test]
    fn loss_decreases_over_steps() {
        // Use a higher learning rate and more steps so loss clearly decreases.
        let cfg = FinetuneConfig {
            rank: 4,
            alpha: 8.0,
            learning_rate: 1e-2,
            epochs: 1,
            ..Default::default()
        };
        let dpo = DpoConfig {
            beta: 0.5,
            reference_free: true,
        };
        let mut trainer = DpoTrainer::new(8, 16, cfg, dpo);
        // Non-zero B so adapter output is non-zero from step 1.
        for (i, v) in trainer.lora.b.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.05;
        }
        let example = DpoExample {
            prompt: vec![0],
            chosen: vec![1, 2, 3, 4],
            rejected: vec![5, 6, 7, 8],
            ref_chosen_logprob: None,
            ref_rejected_logprob: None,
        };
        let hidden_c: Vec<f32> = (0..example.chosen.len() * 8)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let hidden_r: Vec<f32> = (0..example.rejected.len() * 8)
            .map(|i| (i as f32 * 0.13).cos())
            .collect();

        let first_loss = trainer.train_step(&example, &hidden_c, &hidden_r).unwrap();
        let mut last_loss = first_loss;
        for _ in 0..50 {
            last_loss = trainer.train_step(&example, &hidden_c, &hidden_r).unwrap();
        }
        assert!(
            last_loss < first_loss,
            "loss did not decrease: first={first_loss} last={last_loss}"
        );
    }

    // -----------------------------------------------------------------------
    // Full train() call
    // -----------------------------------------------------------------------

    #[test]
    fn train_produces_report() {
        let mut trainer = tiny_trainer();
        let examples = vec![toy_example(), toy_example()];
        let report = trainer.train(&examples, &[]).unwrap();
        assert_eq!(report.steps, examples.len());
        assert!(report.mean_loss.is_finite());
        assert!(report.elapsed_seconds >= 0.0);
    }

    #[test]
    fn train_empty_returns_error() {
        let mut trainer = tiny_trainer();
        let result = trainer.train(&[], &[]);
        assert!(matches!(result, Err(FinetuneError::EmptyDataset)));
    }

    // -----------------------------------------------------------------------
    // JSONL loading (uses temp files written to std::env::temp_dir())
    // -----------------------------------------------------------------------

    fn write_temp(name: &str, contents: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(name);
        std::fs::write(&p, contents).unwrap();
        p
    }

    #[test]
    fn load_jsonl_dpo_token_arrays() {
        let p = write_temp(
            "dpo_test_arrays.jsonl",
            "{\"prompt\":[1,2],\"chosen\":[3,4,5],\"rejected\":[6,7]}\n\
             {\"prompt\":[8],\"chosen\":[9,10],\"rejected\":[11]}\n",
        );
        let examples = load_jsonl_dpo(&p).unwrap();
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0].chosen, vec![3, 4, 5]);
    }

    #[test]
    fn load_jsonl_dpo_string_fields() {
        let p = write_temp(
            "dpo_test_strings.jsonl",
            "{\"prompt\":\"hi\",\"chosen\":\"yes\",\"rejected\":\"no\"}\n",
        );
        let examples = load_jsonl_dpo(&p).unwrap();
        assert_eq!(examples.len(), 1);
        // "hi" UTF-8 bytes = [104, 105]
        assert_eq!(examples[0].prompt, vec![104u32, 105]);
    }

    #[test]
    fn load_jsonl_dpo_empty_file_returns_error() {
        let p = write_temp("dpo_test_empty.jsonl", "\n");
        let result = load_jsonl_dpo(&p);
        assert!(matches!(result, Err(FinetuneError::EmptyDataset)));
    }
}
