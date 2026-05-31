use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use oxidize_core::gguf::load_mapped_gguf;
use oxidize_core::inference::{InferenceConfig, InferenceModel};
use oxidize_core::tokenizer::load_tokenizer_from_gguf_metadata;
use oxidize_finetuning::{FinetuneConfig, SftTrainer, export_lora_gguf, load_jsonl_sft};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "oxidize-finetuning",
    about = "Fast LoRA / SFT fine-tuning for oxidize GGUF models (LFM2, Llama, Qwen, …)"
)]
struct Args {
    /// Base model GGUF path (e.g. LFM2.5-8B-A1B Q4_K_M).
    #[arg(long)]
    model: PathBuf,

    /// JSONL SFT dataset (Alpaca, ShareGPT, or {text} / {messages} rows).
    #[arg(long)]
    dataset: PathBuf,

    /// Output directory for trained LoRA adapter manifest.
    #[arg(long, default_value = "lora-out")]
    output: PathBuf,

    #[arg(long, default_value_t = 16)]
    lora_rank: usize,

    #[arg(long, default_value_t = 32.0)]
    lora_alpha: f32,

    #[arg(long, default_value_t = 2e-4)]
    learning_rate: f32,

    #[arg(long, default_value_t = 1)]
    epochs: usize,

    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,

    #[arg(long, default_value_t = 4)]
    grad_accum: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long)]
    eval_split: Option<f32>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let config = FinetuneConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
        learning_rate: args.learning_rate,
        epochs: args.epochs,
        max_seq_len: args.max_seq_len,
        gradient_accumulation_steps: args.grad_accum.max(1),
        gradient_checkpointing: true,
        seed: args.seed,
        ..FinetuneConfig::default()
    };

    let mapped = load_mapped_gguf(&args.model).context("load GGUF")?;
    let inference_config = InferenceConfig::from_gguf(&mapped);
    let mut model = InferenceModel::load_from_gguf(&mapped, inference_config, true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let tokenizer = load_tokenizer_from_gguf_metadata(&mapped.parsed().metadata)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e:?}"))?;

    let mut examples = load_jsonl_sft(&args.dataset).map_err(|e| anyhow::anyhow!("{e}"))?;
    let encode = |text: &str| -> Vec<u32> { tokenizer.encode(text) };
    SftTrainer::tokenize_examples(&mut examples, encode, config.max_seq_len)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let split = args.eval_split.unwrap_or(0.0).clamp(0.0, 0.5);
    let eval_count = ((examples.len() as f32) * split).round() as usize;
    let (train, eval): (Vec<_>, Vec<_>) = if eval_count > 0 && examples.len() > eval_count {
        let (a, b) = examples.split_at(examples.len() - eval_count);
        (a.to_vec(), b.to_vec())
    } else {
        (examples, Vec::new())
    };

    let mut trainer = SftTrainer::for_model(&model, config.clone());
    println!(
        "oxidize-finetuning: model={} arch={:?} train={} eval={} rank={}",
        args.model.display(),
        model.config().architecture,
        train.len(),
        eval.len(),
        config.rank
    );

    let report = trainer
        .train(&mut model, &train)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!(
        "oxidize-finetuning: steps={} tokens={} mean_loss={:.4}",
        report.steps, report.tokens, report.mean_loss
    );
    for (i, loss) in report.epoch_losses.iter().enumerate() {
        println!("  epoch {} loss={:.4}", i + 1, loss);
    }

    if !eval.is_empty() {
        let eval_loss = trainer
            .eval_loss(&mut model, &eval)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("oxidize-finetuning: eval_loss={:.4}", eval_loss);
    }

    export_lora_gguf(
        &args.output,
        std::slice::from_ref(&trainer.output_lora),
        config.rank,
        config.lora_scale(),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!(
        "oxidize-finetuning: wrote adapter to {}",
        args.output.display()
    );
    Ok(())
}
