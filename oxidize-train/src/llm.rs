use std::path::PathBuf;

use anyhow::{Result, anyhow};
use clap::{Args, Parser, Subcommand};
use oxidize_finetuning::{
    DpoConfig, FinetuneConfig, PpoConfig, SftTrainer, export_lora_gguf, load_causal_model,
    load_jsonl_dpo, load_jsonl_sft, load_prompts_file, pack_chunks, train_dpo_on_model,
    train_ppo_on_model,
};

#[derive(Debug, Parser)]
pub struct LlmArgs {
    #[command(subcommand)]
    command: LlmCommand,
}

#[derive(Debug, Subcommand)]
enum LlmCommand {
    /// Print the training plan for a GGUF without running a step.
    Plan(PlanArgs),
    /// Supervised fine-tuning with an LM-head LoRA on the frozen model.
    Sft(SftArgs),
    /// Direct preference optimization. The frozen LM head is the reference policy.
    Dpo(DpoArgs),
    /// On-policy PPO. Reward is the frozen model's log-probability of the greedy token.
    Ppo(PpoArgs),
}

#[derive(Debug, Args)]
struct PlanArgs {
    #[arg(long)]
    model: PathBuf,
}

#[derive(Debug, Args)]
struct SftArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    dataset: PathBuf,
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
    #[arg(long, default_value_t = 512)]
    max_seq_len: usize,
    #[arg(long, default_value_t = 64)]
    window: usize,
    #[arg(long, default_value_t = 128)]
    tokens_per_step: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = false)]
    allow_partial_arch: bool,
}

#[derive(Debug, Args)]
struct DpoArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    data: PathBuf,
    #[arg(long, default_value = "dpo-out")]
    output: PathBuf,
    #[arg(long, default_value_t = 0.1)]
    beta: f32,
    #[arg(long, default_value_t = 16)]
    rank: usize,
    #[arg(long, default_value_t = 2e-4)]
    lr: f32,
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = 512)]
    max_seq_len: usize,
    #[arg(long, default_value_t = 64)]
    window: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = false)]
    allow_partial_arch: bool,
}

#[derive(Debug, Args)]
struct PpoArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompts: PathBuf,
    #[arg(long, default_value = "ppo-out")]
    output: PathBuf,
    #[arg(long, default_value_t = 16)]
    rank: usize,
    #[arg(long, default_value_t = 2e-4)]
    lr: f32,
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = 256)]
    max_seq_len: usize,
    #[arg(long, default_value_t = 8)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0.2)]
    clip_eps: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = false)]
    allow_partial_arch: bool,
}

pub fn run(args: LlmArgs) -> Result<()> {
    match args.command {
        LlmCommand::Plan(args) => run_plan(args),
        LlmCommand::Sft(args) => run_sft(args),
        LlmCommand::Dpo(args) => run_dpo(args),
        LlmCommand::Ppo(args) => run_ppo(args),
    }
}

fn run_plan(args: PlanArgs) -> Result<()> {
    let loaded = load_causal_model(&args.model, 0, true).map_err(|e| anyhow!("{e}"))?;
    println!("oxidize-train llm:");
    println!("{}", loaded.plan.render());
    println!(
        "layers={} hidden={} vocab={} context={}",
        loaded.model.config().layer_count,
        loaded.model.config().hidden_size,
        loaded.model.config().vocab_size,
        loaded.model.config().context_size,
    );
    Ok(())
}

fn run_sft(args: SftArgs) -> Result<()> {
    let config = FinetuneConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
        learning_rate: args.learning_rate,
        epochs: args.epochs,
        max_seq_len: args.max_seq_len,
        window: args.window,
        tokens_per_step: args.tokens_per_step.max(1),
        seed: args.seed,
        ..FinetuneConfig::default()
    };
    let loaded = load_causal_model(
        &args.model,
        args.max_seq_len.max(args.window) + 8,
        args.allow_partial_arch,
    )
    .map_err(|e| anyhow!("{e}"))?;
    println!("oxidize-train llm sft:");
    println!("{}", loaded.plan.render());
    let oxidize_finetuning::CausalModel {
        mut model,
        tokenizer,
        ..
    } = loaded;
    let eos = tokenizer.special_tokens().eos.unwrap_or(0);
    let mut examples = load_jsonl_sft(&args.dataset).map_err(|e| anyhow!("{e}"))?;
    SftTrainer::tokenize_examples(
        &mut examples,
        |text| tokenizer.encode(text),
        config.max_seq_len,
    )
    .map_err(|e| anyhow!("{e}"))?;
    let chunks = pack_chunks(&examples, config.max_seq_len, eos, config.pack);
    let mut trainer = SftTrainer::for_model(&model, config.clone());
    let report = trainer
        .train(&mut model, &chunks)
        .map_err(|e| anyhow!("{e}"))?;
    export_lora_gguf(
        &args.output,
        std::slice::from_ref(&trainer.output_lora),
        config.rank,
        config.lora_scale(),
    )
    .map_err(|e| anyhow!("{e}"))?;
    println!(
        "oxidize-train llm sft: steps={} tokens={} mean_loss={:.4} -> {}",
        report.steps,
        report.tokens,
        report.mean_loss,
        args.output.display()
    );
    Ok(())
}

fn run_dpo(args: DpoArgs) -> Result<()> {
    let mut examples = load_jsonl_dpo(&args.data).map_err(|e| anyhow!("{e}"))?;
    let loaded = load_causal_model(&args.model, args.max_seq_len + 8, args.allow_partial_arch)
        .map_err(|e| anyhow!("{e}"))?;
    println!("oxidize-train llm dpo:");
    println!("{}", loaded.plan.render());
    let oxidize_finetuning::CausalModel {
        mut model,
        tokenizer,
        ..
    } = loaded;
    let finetune = FinetuneConfig {
        rank: args.rank,
        learning_rate: args.lr,
        epochs: args.epochs,
        max_seq_len: args.max_seq_len,
        window: args.window,
        seed: args.seed,
        ..FinetuneConfig::default()
    };
    let run = train_dpo_on_model(
        &mut model,
        &mut examples,
        |text| tokenizer.encode(text),
        &finetune,
        DpoConfig {
            beta: args.beta,
            reference_free: false,
        },
    )
    .map_err(|e| anyhow!("{e}"))?;
    export_lora_gguf(
        &args.output,
        std::slice::from_ref(&run.trainer.lora),
        finetune.rank,
        finetune.lora_scale(),
    )
    .map_err(|e| anyhow!("{e}"))?;
    println!(
        "oxidize-train llm dpo: steps={} mean_loss={:.4} -> {}",
        run.report.steps,
        run.report.mean_loss,
        args.output.display()
    );
    Ok(())
}

fn run_ppo(args: PpoArgs) -> Result<()> {
    let prompts = load_prompts_file(&args.prompts).map_err(|e| anyhow!("{e}"))?;
    if prompts.is_empty() {
        anyhow::bail!("prompts file {} is empty", args.prompts.display());
    }
    let loaded = load_causal_model(
        &args.model,
        args.max_seq_len + args.max_new_tokens + 8,
        args.allow_partial_arch,
    )
    .map_err(|e| anyhow!("{e}"))?;
    println!("oxidize-train llm ppo:");
    println!("{}", loaded.plan.render());
    println!(
        "reward is the frozen model's log-probability of each greedy token (no separate reward checkpoint)"
    );
    let oxidize_finetuning::CausalModel {
        mut model,
        tokenizer,
        ..
    } = loaded;
    let finetune = FinetuneConfig {
        rank: args.rank,
        learning_rate: args.lr,
        epochs: args.epochs,
        max_seq_len: args.max_seq_len,
        seed: args.seed,
        ..FinetuneConfig::default()
    };
    let run = train_ppo_on_model(
        &mut model,
        &prompts,
        |text| tokenizer.encode(text),
        &finetune,
        PpoConfig {
            clip_eps: args.clip_eps,
            ..PpoConfig::default()
        },
        args.max_new_tokens,
    )
    .map_err(|e| anyhow!("{e}"))?;
    export_lora_gguf(
        &args.output,
        std::slice::from_ref(&run.trainer.lora),
        finetune.rank,
        finetune.lora_scale(),
    )
    .map_err(|e| anyhow!("{e}"))?;
    println!(
        "oxidize-train llm ppo: steps={} transitions={} policy={:.4} value={:.4} -> {}",
        run.report.steps,
        run.report.transitions,
        run.report.mean_policy_loss,
        run.report.mean_value_loss,
        args.output.display()
    );
    Ok(())
}
