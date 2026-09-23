use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use oxidize_finetuning::{
    AdapterMerger, DpoConfig, FinetuneConfig, FinetuneError, LoRAAdapter, MergeStrategy, PpoConfig,
    SelfTrainConfig, SelfTrainLoop, SftTrainer, export_lora_gguf, load_adapter_manifest,
    load_causal_model, load_jsonl_dpo, load_jsonl_sft, load_prompts_file,
    manifest_to_lora_adapters, pack_chunks, train_dpo_on_model, train_ppo_on_model,
};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "oxidize-finetuning",
    about = "Fast LoRA / SFT / DPO / PPO / self-regressing fine-tuning for oxidize GGUF models"
)]
struct Cli {
    /// Rayon worker threads (0 = rayon default).
    #[arg(long, global = true, default_value_t = 0)]
    threads: usize,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Supervised fine-tuning with LoRA.
    Sft(SftArgs),
    /// Direct Preference Optimisation (DPO) with LM-head LoRA.
    Dpo(DpoArgs),
    /// Proximal Policy Optimisation against the frozen model's own likelihood.
    Ppo(PpoArgs),
    /// Iterative self-regressing SFT: train, checkpoint, self-dialogue, repeat.
    SelfTrain(SelfTrainArgs),
    /// Merge multiple LoRA adapter GGUF files.
    Merge(MergeArgs),
}

// ---------------------------------------------------------------------------
// SFT
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
struct SftArgs {
    /// Base model GGUF path.
    #[arg(long)]
    model: PathBuf,

    /// JSONL SFT dataset (Alpaca, ShareGPT, or {text}/{messages} rows).
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
    #[arg(long, default_value_t = 128)]
    window: usize,

    /// Optimizer step cadence, in supervised tokens.
    #[arg(long, default_value_t = 256)]
    tokens_per_step: usize,

    /// Disable packing of short examples into full-length chunks.
    #[arg(long, default_value_t = false)]
    no_pack: bool,

    /// Cap on training tokens per epoch (0 = no cap).
    #[arg(long, default_value_t = 0)]
    max_tokens: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long)]
    eval_split: Option<f32>,

    /// Save adapter every N optimizer steps (0 = only at end).
    #[arg(long, default_value_t = 0)]
    checkpoint_every: usize,

    /// Train even when the architecture's attention block is not in the
    /// layer-wise forward (DeepSeek and GLM-DSA MLA).
    #[arg(long, default_value_t = false)]
    allow_partial_arch: bool,
}

// ---------------------------------------------------------------------------
// DPO
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
struct DpoArgs {
    /// Base model GGUF path.
    #[arg(long)]
    model: PathBuf,

    /// JSONL DPO dataset ({prompt, chosen, rejected} rows).
    #[arg(long)]
    data: PathBuf,

    /// Output directory for trained LoRA adapter manifest.
    #[arg(long, default_value = "dpo-out")]
    output: PathBuf,

    /// DPO beta (KL penalty coefficient).
    #[arg(long, default_value_t = 0.1)]
    beta: f32,

    #[arg(long, default_value_t = 16)]
    rank: usize,

    #[arg(long, default_value_t = 2e-4)]
    lr: f32,

    #[arg(long, default_value_t = 1)]
    epochs: usize,

    /// Packed training chunk length.
    #[arg(long, default_value_t = 512)]
    max_seq_len: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Positions per frozen forward window.
    #[arg(long, default_value_t = 64)]
    window: usize,

    /// Train even when MLA attention is not executed by this trainer.
    #[arg(long, default_value_t = false)]
    allow_partial_arch: bool,
}

// ---------------------------------------------------------------------------
// PPO
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
struct PpoArgs {
    /// Base model GGUF path (actor).
    #[arg(long)]
    model: PathBuf,

    /// Output directory for trained LoRA adapter.
    #[arg(long, default_value = "ppo-out")]
    output: PathBuf,

    /// PPO clip epsilon.
    #[arg(long, default_value_t = 0.2)]
    clip_eps: f32,

    #[arg(long, default_value_t = 1)]
    epochs: usize,

    #[arg(long, default_value_t = 16)]
    rank: usize,

    #[arg(long, default_value_t = 1e-5)]
    lr: f32,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// One prompt per line. Each line is rolled out on the frozen model.
    #[arg(long)]
    prompts: PathBuf,

    /// New tokens sampled per prompt (greedy under the current LoRA policy).
    #[arg(long, default_value_t = 8)]
    max_new_tokens: usize,

    /// Prompt context kept before rollout.
    #[arg(long, default_value_t = 256)]
    max_seq_len: usize,

    /// Train even when MLA attention is not executed by this trainer.
    #[arg(long, default_value_t = false)]
    allow_partial_arch: bool,
}

// ---------------------------------------------------------------------------
// Self-train (iterative SFT + self-dialogue)
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
struct SelfTrainArgs {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    dataset: PathBuf,

    #[arg(long, default_value = "self-train-out")]
    output: PathBuf,

    #[arg(long, default_value_t = 3)]
    rounds: usize,

    #[arg(long, default_value_t = 8)]
    prompts_per_round: usize,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    #[arg(long, default_value_t = true)]
    self_critique: bool,

    #[arg(long, default_value_t = 50)]
    checkpoint_every: usize,

    #[arg(long, default_value_t = 16)]
    lora_rank: usize,

    #[arg(long, default_value_t = 2e-4)]
    learning_rate: f32,

    #[arg(long, default_value_t = 1)]
    epochs_per_round: usize,

    #[arg(long, default_value_t = 512)]
    max_seq_len: usize,

    #[arg(long, default_value_t = 128)]
    window: usize,

    #[arg(long, default_value_t = 256)]
    tokens_per_step: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long)]
    eval_split: Option<f32>,

    #[arg(long)]
    resume_from: Option<PathBuf>,

    /// Train even when MLA attention is not executed by this trainer.
    #[arg(long, default_value_t = false)]
    allow_partial_arch: bool,
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
struct MergeArgs {
    /// Adapter GGUF files to merge (repeat for each adapter).
    #[arg(long = "adapter", required = true)]
    adapters: Vec<PathBuf>,

    /// Per-adapter blend weights (repeat to match --adapter count; defaults to equal weights).
    #[arg(long = "weight")]
    weights: Vec<f32>,

    /// Merge strategy: linear, slerp, or ties.
    #[arg(long, default_value = "linear")]
    strategy: String,

    /// Output GGUF path for the merged adapter.
    #[arg(long, default_value = "merged-adapter.gguf")]
    output: PathBuf,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .context("build rayon pool")?;
    } else if let Ok(n) = std::thread::available_parallelism() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n.get())
            .build_global()
            .context("build rayon pool")?;
    }

    match cli.command {
        Command::Sft(args) => run_sft(args),
        Command::Dpo(args) => run_dpo(args),
        Command::Ppo(args) => run_ppo(args),
        Command::SelfTrain(args) => run_self_train(args),
        Command::Merge(args) => run_merge(args),
    }
}

// ---------------------------------------------------------------------------
// SFT implementation (original logic preserved)
// ---------------------------------------------------------------------------

fn run_sft(args: SftArgs) -> Result<()> {
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

    let loaded = load_causal_model(
        &args.model,
        args.max_seq_len.max(args.window) + 8,
        args.allow_partial_arch,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{}", loaded.plan.render());
    let oxidize_finetuning::CausalModel {
        mut model,
        tokenizer,
        ..
    } = loaded;
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
            "oxidize-finetuning sft: checkpointing to {} every {} steps",
            args.output.display(),
            args.checkpoint_every
        );
    }
    println!(
        "oxidize-finetuning sft: model={} arch={:?} layers={} examples={} chunks={} (~{} tokens) eval_chunks={} rank={} window={} tokens/step={}",
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
        "oxidize-finetuning sft: steps={} tokens={} mean_loss={:.4} | {:.2} tok/s over {:.1}s",
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
        println!("oxidize-finetuning sft: eval_loss={:.4}", eval_loss);
    }

    export_lora_gguf(
        &args.output,
        std::slice::from_ref(&trainer.output_lora),
        config.rank,
        config.lora_scale(),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!(
        "oxidize-finetuning sft: wrote adapter to {}",
        args.output.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// DPO stub
// ---------------------------------------------------------------------------

fn run_dpo(args: DpoArgs) -> Result<()> {
    let mut examples = load_jsonl_dpo(&args.data).map_err(|e| anyhow::anyhow!("{e}"))?;
    let loaded = load_causal_model(&args.model, args.max_seq_len + 8, args.allow_partial_arch)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
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
    let dpo = DpoConfig {
        beta: args.beta,
        reference_free: false,
    };
    println!(
        "oxidize-finetuning dpo: model={} examples={} beta={} rank={} — reference is the frozen LM head",
        args.model.display(),
        examples.len(),
        args.beta,
        args.rank,
    );
    let run = train_dpo_on_model(
        &mut model,
        &mut examples,
        |text| tokenizer.encode(text),
        &finetune,
        dpo,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!(
        "oxidize-finetuning dpo: steps={} mean_loss={:.4} ({:.1}s)",
        run.report.steps, run.report.mean_loss, run.report.elapsed_seconds,
    );
    export_lora_gguf(
        &args.output,
        std::slice::from_ref(&run.trainer.lora),
        finetune.rank,
        finetune.lora_scale(),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!(
        "oxidize-finetuning dpo: wrote adapter to {}",
        args.output.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// PPO
// ---------------------------------------------------------------------------

fn run_ppo(args: PpoArgs) -> Result<()> {
    let prompts = load_prompts_file(&args.prompts).map_err(|e| anyhow::anyhow!("{e}"))?;
    if prompts.is_empty() {
        anyhow::bail!("prompts file {} is empty", args.prompts.display());
    }
    let loaded = load_causal_model(
        &args.model,
        args.max_seq_len + args.max_new_tokens + 8,
        args.allow_partial_arch,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{}", loaded.plan.render());
    println!(
        "oxidize-finetuning ppo: reward is the frozen model's log-probability of each greedy token (no separate reward checkpoint)"
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
    let ppo = PpoConfig {
        clip_eps: args.clip_eps,
        ..PpoConfig::default()
    };
    let run = train_ppo_on_model(
        &mut model,
        &prompts,
        |text| tokenizer.encode(text),
        &finetune,
        ppo,
        args.max_new_tokens,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!(
        "oxidize-finetuning ppo: steps={} transitions={} policy={:.4} value={:.4} entropy={:.4} kl={:.4} ({:.1}s)",
        run.report.steps,
        run.report.transitions,
        run.report.mean_policy_loss,
        run.report.mean_value_loss,
        run.report.mean_entropy,
        run.report.mean_kl,
        run.report.elapsed_seconds,
    );
    export_lora_gguf(
        &args.output,
        std::slice::from_ref(&run.trainer.lora),
        finetune.rank,
        finetune.lora_scale(),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!(
        "oxidize-finetuning ppo: wrote adapter to {}",
        args.output.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Self-train implementation
// ---------------------------------------------------------------------------

fn run_self_train(args: SelfTrainArgs) -> Result<()> {
    let finetune = FinetuneConfig {
        rank: args.lora_rank,
        learning_rate: args.learning_rate,
        epochs: args.epochs_per_round,
        max_seq_len: args.max_seq_len,
        window: args.window,
        tokens_per_step: args.tokens_per_step.max(1),
        seed: args.seed,
        ..FinetuneConfig::default()
    };
    let self_cfg = SelfTrainConfig {
        rounds: args.rounds,
        prompts_per_round: args.prompts_per_round,
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        self_critique: args.self_critique,
        checkpoint_every_steps: args.checkpoint_every,
        seed: args.seed,
        ..SelfTrainConfig::default()
    };

    let loaded = load_causal_model(
        &args.model,
        args.max_seq_len.max(args.window) + args.max_new_tokens + 8,
        args.allow_partial_arch,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{}", loaded.plan.render());
    let oxidize_finetuning::CausalModel {
        mut model,
        tokenizer,
        ..
    } = loaded;
    let eos = tokenizer.special_tokens().eos.unwrap_or(0);

    let examples = load_jsonl_sft(&args.dataset).map_err(|e| anyhow::anyhow!("{e}"))?;
    let encode = |text: &str| -> Vec<u32> { tokenizer.encode(text) };
    let decode = |ids: &[u32]| -> std::result::Result<String, FinetuneError> {
        tokenizer
            .decode(ids)
            .map_err(|error| FinetuneError::Model(format!("tokenizer decode failed: {error:?}")))
    };

    let split = args.eval_split.unwrap_or(0.0).clamp(0.0, 0.5);
    let eval_count = ((examples.len() as f32) * split).round() as usize;
    let (train_examples, eval_examples) = if eval_count > 0 && examples.len() > eval_count {
        let (a, b) = examples.split_at(examples.len() - eval_count);
        (a.to_vec(), b.to_vec())
    } else {
        (examples, Vec::new())
    };

    let mut eval_chunks = Vec::new();
    if !eval_examples.is_empty() {
        let mut eval_copy = eval_examples.clone();
        SftTrainer::tokenize_examples(&mut eval_copy, encode, finetune.max_seq_len)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        eval_chunks = pack_chunks(&eval_copy, finetune.max_seq_len, eos, finetune.pack);
    }

    if let Some(resume) = &args.resume_from {
        println!(
            "oxidize-finetuning self-train: resuming adapter from {}",
            resume.display()
        );
    }

    println!(
        "oxidize-finetuning self-train: model={} seed_examples={} rounds={} prompts/round={} critique={} output={}",
        args.model.display(),
        train_examples.len(),
        args.rounds,
        args.prompts_per_round,
        args.self_critique,
        args.output.display(),
    );

    let loop_ = SelfTrainLoop::new(finetune, self_cfg, &args.output);
    let report = loop_
        .run(
            &mut model,
            &train_examples,
            &eval_chunks,
            encode,
            decode,
            eos,
            args.resume_from.as_deref(),
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!(
        "oxidize-finetuning self-train: finished {} rounds; final checkpoint={} synthetic={}",
        report.rounds.len(),
        report.final_checkpoint.display(),
        report.synthetic_dataset.display(),
    );
    for r in &report.rounds {
        println!(
            "  round {}: loss={:.4} eval={} synthetic=+{} ({:.1}s)",
            r.round,
            r.train_loss,
            r.eval_loss
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "n/a".into()),
            r.synthetic_added,
            r.elapsed_seconds,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Merge implementation
// ---------------------------------------------------------------------------

fn run_merge(args: MergeArgs) -> Result<()> {
    let strategy = match args.strategy.to_lowercase().as_str() {
        "linear" => MergeStrategy::Linear,
        "slerp" => MergeStrategy::Slerp,
        "ties" => MergeStrategy::Ties { density: 0.5 },
        other => {
            anyhow::bail!(
                "unknown merge strategy {:?}; expected linear, slerp, or ties",
                other
            )
        }
    };

    // Normalise weights: if none provided use equal weights; otherwise must
    // match adapter count.
    let weights: Vec<f32> = if args.weights.is_empty() {
        let w = 1.0 / args.adapters.len() as f32;
        vec![w; args.adapters.len()]
    } else {
        anyhow::ensure!(
            args.weights.len() == args.adapters.len(),
            "--weight count ({}) must match --adapter count ({})",
            args.weights.len(),
            args.adapters.len()
        );
        args.weights.clone()
    };

    println!(
        "oxidize-finetuning merge: strategy={} adapters={} output={}",
        args.strategy,
        args.adapters.len(),
        args.output.display(),
    );

    // Load and validate all manifests upfront so we fail fast before merging.
    let mut manifests = Vec::with_capacity(args.adapters.len());
    for path in &args.adapters {
        manifests.push(
            load_adapter_manifest(path)
                .map_err(|e| anyhow::anyhow!("load adapter manifest {}: {e}", path.display()))?,
        );
    }

    // Each manifest may contain multiple LoRAAdapter entries (one per target
    // layer). We merge position-by-position: the first entry in manifest[0]
    // is merged with the first entry in manifest[1], etc.
    let rank = manifests[0].rank;
    let alpha_scale = manifests[0].alpha_scale;

    // Convert all manifests to Vec<LoRAAdapter>.
    let all_adapters: Vec<Vec<LoRAAdapter>> = manifests
        .into_iter()
        .map(manifest_to_lora_adapters)
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let n_entries = all_adapters[0].len();
    anyhow::ensure!(
        all_adapters.iter().all(|v| v.len() == n_entries),
        "all adapter manifests must contain the same number of adapter entries"
    );

    // Merge entry by entry.
    let mut merged_adapters: Vec<LoRAAdapter> = Vec::with_capacity(n_entries);

    for entry_idx in 0..n_entries {
        let mut merger = AdapterMerger::new(strategy.clone());
        for (adapter_idx, adapters) in all_adapters.iter().enumerate() {
            let ad = adapters[entry_idx].clone();
            let w = weights[adapter_idx];
            println!(
                "  {} entry[{}]={:?} (weight={:.4})",
                args.adapters[adapter_idx].display(),
                entry_idx,
                ad.target,
                w,
            );
            merger = merger.add(ad, w);
        }
        let merged = merger
            .merge()
            .map_err(|e| anyhow::anyhow!("merge entry {entry_idx}: {e}"))?;
        merged_adapters.push(merged);
    }

    export_lora_gguf(&args.output, &merged_adapters, rank, alpha_scale)
        .map_err(|e| anyhow::anyhow!("export merged adapter: {e}"))?;

    println!(
        "oxidize-finetuning merge: wrote merged adapter to {}",
        args.output.display()
    );
    Ok(())
}
