//! Iterative self-regressing SFT: train → checkpoint → self-dialogue → repeat.
//!
//! Each round the model talks to itself on seed prompts, appends synthetic
//! chat examples to the training pool, and continues from the last checkpoint.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use oxidize_core::layer_wise::LayerWiseModel;
use serde::{Deserialize, Serialize};

use crate::config::FinetuneConfig;
use crate::dataset::{SftExample, load_jsonl_sft, pack_chunks};
use crate::error::{FinetuneError, Result};
use crate::export::{export_lora_gguf, load_output_head_lora};
use crate::generate::{GenerateConfig, generate_with_lora};
use crate::lora::LoRAAdapter;
use crate::telemetry::{MetricsLog, TrainingMetrics};
use crate::trainer::{FinetuneReport, SftTrainer};

const CRITIQUE_USER: &str =
    "Review your previous answer. What could be clearer or more accurate? Then give an improved reply.";
const IM_END: &str = concat!("<|", "im_end", "|>");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfTrainConfig {
    pub rounds: usize,
    pub prompts_per_round: usize,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub self_critique: bool,
    pub checkpoint_every_steps: usize,
    pub min_loss_delta: f32,
    pub max_synthetic_per_round: usize,
    pub seed: u64,
}

impl Default for SelfTrainConfig {
    fn default() -> Self {
        Self {
            rounds: 3,
            prompts_per_round: 8,
            max_new_tokens: 128,
            temperature: 0.7,
            self_critique: true,
            checkpoint_every_steps: 50,
            min_loss_delta: 0.001,
            max_synthetic_per_round: 64,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfTrainRoundReport {
    pub round: usize,
    pub train_loss: f32,
    pub eval_loss: Option<f32>,
    pub synthetic_added: usize,
    pub total_examples: usize,
    pub checkpoint: PathBuf,
    pub elapsed_seconds: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfTrainReport {
    pub rounds: Vec<SelfTrainRoundReport>,
    pub final_checkpoint: PathBuf,
    pub synthetic_dataset: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedState {
    pub completed_rounds: usize,
    pub best_eval_loss: Option<f32>,
    pub rounds: Vec<SelfTrainRoundReport>,
}

pub struct SelfTrainLoop {
    pub finetune: FinetuneConfig,
    pub self_config: SelfTrainConfig,
    pub output_dir: PathBuf,
}

impl SelfTrainLoop {
    pub fn new(
        finetune: FinetuneConfig,
        self_config: SelfTrainConfig,
        output_dir: impl AsRef<Path>,
    ) -> Self {
        Self {
            finetune,
            self_config,
            output_dir: output_dir.as_ref().to_path_buf(),
        }
    }

    pub fn run(
        &self,
        model: &mut LayerWiseModel,
        seed_dataset: &[SftExample],
        eval_chunks: &[Vec<u32>],
        encode: impl Fn(&str) -> Vec<u32> + Sync,
        decode: impl Fn(&[u32]) -> String,
        eos: u32,
        resume_adapter: Option<&Path>,
    ) -> Result<SelfTrainReport> {
        fs::create_dir_all(&self.output_dir)
            .map_err(|e| FinetuneError::Model(format!("create output dir: {e}")))?;

        let synthetic_path = self.output_dir.join("synthetic.jsonl");
        let state_path = self.output_dir.join("self_train_state.json");
        let mut state = load_state(&state_path).unwrap_or(PersistedState {
            completed_rounds: 0,
            best_eval_loss: None,
            rounds: Vec::new(),
        });

        let mut pool = seed_dataset.to_vec();
        if synthetic_path.exists() {
            if let Ok(extra) = load_jsonl_sft(&synthetic_path) {
                pool.extend(extra);
            }
        }

        let start_round = state.completed_rounds;
        let mut trainer = SftTrainer::for_model(model, self.finetune.clone());
        if let Some(path) = resume_adapter {
            trainer.output_lora = load_output_head_lora(path)?;
            println!("  loaded adapter checkpoint from {}", path.display());
        }
        if self.self_config.checkpoint_every_steps > 0 {
            trainer.checkpoint = Some((
                self.output_dir.join("checkpoints"),
                self.self_config.checkpoint_every_steps,
            ));
        }

        let mut metrics_log = MetricsLog::new();
        let mut last_checkpoint = self.output_dir.join("adapter");

        for round in start_round..self.self_config.rounds {
            let round_no = round + 1;
            let started = Instant::now();
            println!(
                "self-train round {}/{}: {} examples in pool",
                round_no,
                self.self_config.rounds,
                pool.len()
            );

            let mut examples = pool.clone();
            SftTrainer::tokenize_examples(&mut examples, &encode, self.finetune.max_seq_len)?;
            let train_chunks = pack_chunks(&examples, self.finetune.max_seq_len, eos, self.finetune.pack);

            let report = trainer.train(model, &train_chunks)?;
            log_training_round(&mut metrics_log, round_no, &report);

            let eval_loss = if eval_chunks.is_empty() {
                None
            } else {
                Some(trainer.eval_loss(model, eval_chunks)?)
            };

            let ckpt_dir = self
                .output_dir
                .join(format!("round_{round_no:03}"));
            export_lora_gguf(
                &ckpt_dir,
                std::slice::from_ref(&trainer.output_lora),
                self.finetune.rank,
                self.finetune.lora_scale(),
            )?;
            export_lora_gguf(
                &self.output_dir.join("adapter"),
                std::slice::from_ref(&trainer.output_lora),
                self.finetune.rank,
                self.finetune.lora_scale(),
            )?;
            last_checkpoint = ckpt_dir.clone();

            let prompts = sample_prompts(&pool, self.self_config.prompts_per_round, self.self_config.seed + round as u64);
            let synthetic = self.generate_self_dialogues(
                model,
                &trainer.output_lora,
                &prompts,
                &encode,
                &decode,
                eos,
                round_no,
            )?;
            let added = append_synthetic_jsonl(&synthetic_path, &synthetic)?;
            pool.extend(synthetic);

            let round_report = SelfTrainRoundReport {
                round: round_no,
                train_loss: report.mean_loss,
                eval_loss,
                synthetic_added: added,
                total_examples: pool.len(),
                checkpoint: ckpt_dir.clone(),
                elapsed_seconds: started.elapsed().as_secs_f32(),
            };
            println!(
                "  round {round_no}: loss={:.4} eval={} synthetic=+{} total={} -> {}",
                round_report.train_loss,
                eval_loss
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "n/a".into()),
                added,
                pool.len(),
                ckpt_dir.display()
            );

            state.completed_rounds = round_no;
            state.rounds.push(round_report.clone());
            if let Some(ev) = eval_loss {
                let improved = state
                    .best_eval_loss
                    .map(|b| ev < b - self.self_config.min_loss_delta)
                    .unwrap_or(true);
                if improved {
                    state.best_eval_loss = Some(ev);
                }
            }
            save_state(&state_path, &state)?;

            if round + 1 < self.self_config.rounds {
                if let Ok(loaded) = load_output_head_lora(&last_checkpoint) {
                    trainer.output_lora = loaded;
                }
            }
        }

        let _ = metrics_log.save_csv(&self.output_dir.join("metrics.csv"));

        Ok(SelfTrainReport {
            rounds: state.rounds,
            final_checkpoint: last_checkpoint,
            synthetic_dataset: synthetic_path,
        })
    }

    fn generate_self_dialogues(
        &self,
        model: &mut LayerWiseModel,
        lora: &LoRAAdapter,
        prompts: &[String],
        encode: &impl Fn(&str) -> Vec<u32>,
        decode: &impl Fn(&[u32]) -> String,
        eos: u32,
        round: usize,
    ) -> Result<Vec<SftExample>> {
        let generate_cfg = GenerateConfig {
            max_new_tokens: self.self_config.max_new_tokens,
            temperature: self.self_config.temperature,
            top_k: Some(40),
            top_p: Some(0.9),
            eos_token: Some(eos),
            seed: self.self_config.seed.wrapping_add(round as u64 * 997),
        };

        let mut out = Vec::new();
        for (i, prompt) in prompts.iter().enumerate() {
            if out.len() >= self.self_config.max_synthetic_per_round {
                break;
            }

            let user_block = format!("<|im_start|>user\n{prompt}\n{IM_END}\n");
            let assistant_prefix = "<|im_start|>assistant\n".to_string();
            let prompt_text = format!("{user_block}{assistant_prefix}");
            let prompt_ids = encode(&prompt_text);

            let response_ids = generate_with_lora(model, lora, &prompt_ids, &generate_cfg)?;
            let first_reply = decode(&response_ids);

            let mut messages = vec![
                serde_json::json!({"role": "user", "content": prompt}),
                serde_json::json!({"role": "assistant", "content": first_reply}),
            ];

            if self.self_config.self_critique {
                let critique_prefix = format!(
                    "{user_block}{assistant_prefix}{first_reply}{IM_END}\n<|im_start|>user\n{CRITIQUE_USER}\n{IM_END}\n{assistant_prefix}"
                );
                let critique_ids = encode(&critique_prefix);
                let revised_ids = generate_with_lora(
                    model,
                    lora,
                    &critique_ids,
                    &GenerateConfig {
                        seed: generate_cfg.seed.wrapping_add(i as u64 + 1),
                        ..generate_cfg.clone()
                    },
                )?;
                let revised = decode(&revised_ids);
                messages.push(serde_json::json!({"role": "user", "content": CRITIQUE_USER}));
                messages.push(serde_json::json!({"role": "assistant", "content": revised}));
            }

            out.push(SftExample {
                text: messages_to_chat_text(&messages),
                token_ids: Vec::new(),
            });
        }
        Ok(out)
    }
}

/// Extract user prompts from seed examples for self-dialogue generation.
pub fn sample_prompts(examples: &[SftExample], count: usize, seed: u64) -> Vec<String> {
    let mut prompts: Vec<String> = examples
        .iter()
        .filter_map(|ex| extract_user_prompt(&ex.text))
        .collect();
    if prompts.is_empty() {
        prompts = vec![
            "Explain how gradient descent works.".into(),
            "Write a short function in Rust that reverses a string.".into(),
            "What are three ways to reduce LLM inference latency?".into(),
        ];
    }
    shuffle_prompts(&mut prompts, seed);
    prompts.truncate(count.min(prompts.len()));
    prompts
}

fn extract_user_prompt(text: &str) -> Option<String> {
    if let Some(start) = text.find("<|im_start|>user\n") {
        let rest = &text[start + "<|im_start|>user\n".len()..];
        if let Some(end) = rest.find(IM_END) {
            let p = rest[..end].trim();
            if !p.is_empty() {
                return Some(p.to_string());
            }
        }
    }
    if text.len() > 20 && text.len() < 512 {
        return Some(text.chars().take(256).collect());
    }
    None
}

fn shuffle_prompts(prompts: &mut [String], seed: u64) {
    let mut state = seed | 1;
    for i in (1..prompts.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        prompts.swap(i, j);
    }
}

fn messages_to_chat_text(messages: &[serde_json::Value]) -> String {
    let mut s = String::new();
    for m in messages {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
        s.push_str("<|im_start|>");
        s.push_str(role);
        s.push('\n');
        s.push_str(content);
        s.push_str(IM_END);
        s.push('\n');
    }
    s
}

fn append_synthetic_jsonl(path: &Path, examples: &[SftExample]) -> Result<usize> {
    if examples.is_empty() {
        return Ok(0);
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| FinetuneError::Model(format!("open {}: {e}", path.display())))?;
    let mut added = 0usize;
    for ex in examples {
        let row = format!(
            "{{\"text\":{}}}\n",
            serde_json::to_string(&ex.text).unwrap_or_default()
        );
        file.write_all(row.as_bytes())
            .map_err(|e| FinetuneError::Model(format!("write synthetic: {e}")))?;
        added += 1;
    }
    Ok(added)
}

fn load_state(path: &Path) -> Result<PersistedState> {
    let text = fs::read_to_string(path)
        .map_err(|e| FinetuneError::Model(format!("read state: {e}")))?;
    serde_json::from_str(&text).map_err(|e| FinetuneError::Model(format!("parse state: {e}")))
}

fn save_state(path: &Path, state: &PersistedState) -> Result<()> {
    let json = serde_json::to_string_pretty(state)
        .map_err(|e| FinetuneError::Model(format!("serialize state: {e}")))?;
    fs::write(path, json).map_err(|e| FinetuneError::Model(format!("write state: {e}")))
}

fn log_training_round(log: &mut MetricsLog, round: usize, report: &FinetuneReport) {
    log.record(TrainingMetrics::new(
        report.steps,
        round,
        report.mean_loss,
        0.0,
        report.tokens_per_second,
        0.0,
        None,
    ));
}

/// Load optional one-prompt-per-line seed prompts file.
pub fn load_prompts_file(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let file = File::open(path.as_ref()).map_err(|e| FinetuneError::Model(e.to_string()))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| FinetuneError::Model(e.to_string()))?;
        let t = line.trim();
        if !t.is_empty() {
            out.push(t.to_string());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_prompts_fallback() {
        let p = sample_prompts(&[], 2, 1);
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn extract_user_from_chat_template() {
        let text = format!("<|im_start|>user\nhello world\n{IM_END}\n<|im_start|>assistant\n");
        assert_eq!(
            extract_user_prompt(&text),
            Some("hello world".to_string())
        );
    }

    #[test]
    fn shuffle_is_deterministic() {
        let mut a = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let mut b = a.clone();
        shuffle_prompts(&mut a, 99);
        shuffle_prompts(&mut b, 99);
        assert_eq!(a, b);
    }
}
