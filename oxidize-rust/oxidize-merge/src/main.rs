use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use oxidize_merge::{MergeMethod, MergeOptions, MergeRecipe, MissingTensorPolicy, merge_models};

const DEFAULT_MAX_SHARD_GIB: u64 = 5;

#[derive(Debug, Parser)]
#[command(
    name = "oxidize-merge",
    about = "Merge two HuggingFace SafeTensors checkpoints with linear or SLERP blending"
)]
struct Args {
    #[arg(
        long,
        help = "First model (SafeTensors file or HuggingFace model directory)"
    )]
    a: PathBuf,
    #[arg(
        long,
        help = "Second model (SafeTensors file or HuggingFace model directory)"
    )]
    b: PathBuf,
    #[arg(
        long,
        help = "Output path: .safetensors file or directory for sharded output"
    )]
    output: PathBuf,
    #[arg(
        long,
        value_enum,
        default_value_t = CliMergeMethod::Slerp,
        help = "Blend method: linear or slerp"
    )]
    method: CliMergeMethod,
    #[arg(
        long,
        value_enum,
        help = "Preset merge recipe (overrides per-category weights unless --t is set)"
    )]
    preset: Option<CliPreset>,
    #[arg(
        long,
        help = "Global blend weight t in [0, 1] toward model B (overrides preset category weights)"
    )]
    t: Option<f32>,
    #[arg(
        long,
        default_value_t = 0.3,
        help = "Blend weight for attention tensors toward model B"
    )]
    attention_t: f32,
    #[arg(
        long,
        default_value_t = 0.5,
        help = "Blend weight for MLP / expert tensors toward model B"
    )]
    mlp_t: f32,
    #[arg(
        long,
        default_value_t = 0.4,
        help = "Blend weight for all other float tensors toward model B"
    )]
    other_t: f32,
    #[arg(
        long,
        value_enum,
        default_value_t = CliMissingPolicy::Error,
        help = "Policy when a tensor exists in only one checkpoint"
    )]
    missing: CliMissingPolicy,
    #[arg(
        long,
        default_value_t = DEFAULT_MAX_SHARD_GIB,
        help = "Maximum shard size in GiB for directory output"
    )]
    max_shard_gib: u64,
    #[arg(long, help = "Validate tensor compatibility without writing output")]
    dry_run: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum CliMergeMethod {
    Linear,
    Slerp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum CliPreset {
    KimiK275,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum CliMissingPolicy {
    Error,
    A,
    B,
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    if let Some(t) = args.t
        && !(0.0..=1.0).contains(&t)
    {
        anyhow::bail!("--t must be in [0, 1]");
    }
    for (label, value) in [
        ("attention_t", args.attention_t),
        ("mlp_t", args.mlp_t),
        ("other_t", args.other_t),
    ] {
        if !(0.0..=1.0).contains(&value) {
            anyhow::bail!("--{label} must be in [0, 1]");
        }
    }

    let recipe = build_recipe(&args);
    let report = merge_models(MergeOptions {
        model_a: args.a,
        model_b: args.b,
        output: args.output,
        method: match args.method {
            CliMergeMethod::Linear => MergeMethod::Linear,
            CliMergeMethod::Slerp => MergeMethod::Slerp,
        },
        recipe,
        missing: match args.missing {
            CliMissingPolicy::Error => MissingTensorPolicy::Error,
            CliMissingPolicy::A => MissingTensorPolicy::A,
            CliMissingPolicy::B => MissingTensorPolicy::B,
        },
        max_shard_bytes: args.max_shard_gib.saturating_mul(1024 * 1024 * 1024),
        dry_run: args.dry_run,
    })?;

    if report.dry_run {
        println!(
            "Dry run: would blend {} tensors, copy {} from A, copy {} from B -> {}",
            report.merged_tensors,
            report.copied_from_a,
            report.copied_from_b,
            report.output.display()
        );
    } else {
        println!(
            "Merged {} tensors ({} copied from A, {} copied from B) -> {}",
            report.merged_tensors,
            report.copied_from_a,
            report.copied_from_b,
            report.output.display()
        );
    }
    Ok(())
}

fn build_recipe(args: &Args) -> MergeRecipe {
    if let Some(t) = args.t {
        return MergeRecipe::uniform(t);
    }
    if let Some(CliPreset::KimiK275) = args.preset {
        return MergeRecipe::kimi_k275();
    }
    MergeRecipe {
        attention_t: args.attention_t,
        mlp_t: args.mlp_t,
        other_t: args.other_t,
        default_t: None,
    }
}
