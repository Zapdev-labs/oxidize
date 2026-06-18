use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use oxidize_core::gguf::load_mapped_gguf;
use oxidize_core::inference::InferenceConfig;
use oxidize_core::layer_wise::LayerWiseModel;
use oxidize_core::tokenizer::load_tokenizer_from_gguf_metadata;
use oxidize_finetuning::{
    FinetuneConfig, SftTrainer, export_lora_gguf, load_jsonl_sft, pack_chunks,
};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "oxidize-finetuning",
    about = "Fast LoRA / SFT fine-tuning for oxidize GGUF models (Qwen3.5/GDN, Llama, LFM2, …)"
)]
struct Args {
    /// Base model GGUF path (e.g. Qwopus3.6-27B-v2 Q4_K_M).
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

    /// Packed training chunk length.
    #[arg(long, default_value_t = 512)]
    max_seq_len: usize,

    /// Positions per batched forward window (GEMM batch dimension).
    #[arg(long, default_value_t = 64)]
    window: usize,

    /// Optimizer step cadence, in supervised tokens.
    #[arg(long, default_value_t = 256)]
    tokens_per_step: usize,

    /// Disable packing of short examples into full-length chunks.
    #[arg(long, default_value_t = false)]
    no_pack: bool,

    /// Rayon worker threads (0 = rayon default).
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Cap on training tokens per epoch (0 = no cap). Useful for benchmarking.
    #[arg(long, default_value_t = 0)]
    max_tokens: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long)]
    eval_split: Option<f32>,

    /// Save the LoRA adapter to --output every N optimizer steps (0 = only at
    /// the end). Protects long runs against crashes/reboots.
    #[arg(long, default_value_t = 0)]
    checkpoint_every: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("build rayon pool")?;
    }
    let config = FinetuneConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
        learning_rate: args.learning_rate,
        epochs: args.epochs,
        max_seq_len: args.max_seq_len,
        window: args.window,
        tokens_per_step: args.tokens_per_step.max(1),
        pack: !args.no_pack,
        seed: args.seed,
        ..FinetuneConfig::default()
    };

    let mapped = load_mapped_gguf(&args.model).context("load GGUF")?;
    let mut inference_config = InferenceConfig::from_gguf(&mapped);
    // Training never attends beyond one packed chunk; a small context keeps
    // the KV cache allocation proportional to max_seq_len instead of the
    // model's native window (262k for qwen35 → tens of GB of KV).
    inference_config.context_size = inference_config
        .context_size
        .min(args.max_seq_len.max(args.window) + 8);
    let mut model = LayerWiseModel::load_from_gguf(&mapped, inference_config, 0)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    model
        .warm_layer_cache()
        .map_err(|e| anyhow::anyhow!("warm layer cache: {e}"))?;
    let tokenizer = load_tokenizer_from_gguf_metadata(&mapped.parsed().metadata)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e:?}"))?;
    let eos = tokenizer.special_tokens().eos.unwrap_or(0);

    let mut examples = load_jsonl_sft(&args.dataset).map_err(|e| anyhow::anyhow!("{e}"))?;
    let encode = |text: &str| -> Vec<u32> { tokenizer.encode(text) };
    SftTrainer::tokenize_examples(&mut examples, encode, config.max_seq_len)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let split = args.eval_split.unwrap_or(0.0).clamp(0.0, 0.5);
    let eval_count = ((examples.len() as f32) * split).round() as usize;
    let (train_examples, eval_examples) = if eval_count > 0 && examples.len() > eval_count {
        let (a, b) = examples.split_at(examples.len() - eval_count);
        (a.to_vec(), b.to_vec())
    } else {
        (examples, Vec::new())
    };

    let mut train_chunks = pack_chunks(&train_examples, config.max_seq_len, eos, config.pack);
    let eval_chunks = pack_chunks(&eval_examples, config.max_seq_len, eos, config.pack);
    if args.max_tokens > 0 {
        let mut kept = 0usize;
        train_chunks.retain(|c| {
            kept += c.len();
            kept <= args.max_tokens
        });
    }
    let train_tokens: usize = train_chunks.iter().map(|c| c.len()).sum();

    let mut trainer = SftTrainer::for_model(&model, config.clone());
    if args.checkpoint_every > 0 {
        trainer.checkpoint = Some((args.output.clone(), args.checkpoint_every));
        println!(
            "oxidize-finetuning: checkpointing to {} every {} steps",
            args.output.display(),
            args.checkpoint_every
        );
    }
    println!(
        "oxidize-finetuning: model={} arch={:?} layers={} examples={} chunks={} (~{} tokens) eval_chunks={} rank={} window={} tokens/step={}",
        args.model.display(),
        model.config().architecture,
        model.config().layer_count,
        train_examples.len(),
        train_chunks.len(),
        train_tokens,
        eval_chunks.len(),
        config.rank,
        config.window,
        config.tokens_per_step,
    );

    let report = trainer
        .train(&mut model, &train_chunks)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!(
        "oxidize-finetuning: steps={} tokens={} mean_loss={:.4} | {:.2} tok/s over {:.1}s",
        report.steps,
        report.tokens,
        report.mean_loss,
        report.tokens_per_second,
        report.elapsed_seconds,
    );
    for (i, loss) in report.epoch_losses.iter().enumerate() {
        println!("  epoch {} loss={:.4}", i + 1, loss);
    }

    if !eval_chunks.is_empty() {
        let eval_loss = trainer
            .eval_loss(&mut model, &eval_chunks)
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
