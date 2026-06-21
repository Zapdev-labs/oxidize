mod quantization;
mod run;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use oxidize_prune::mask::SparsityPattern;
use oxidize_prune::wanda::WandaOptions;

use crate::run::ConvertOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum CliPruneMethod {
    Wanda,
    Magnitude,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum CliSparsityPattern {
    Unstructured,
    N2of4,
    N4of8,
}

impl From<CliSparsityPattern> for SparsityPattern {
    fn from(p: CliSparsityPattern) -> Self {
        match p {
            CliSparsityPattern::Unstructured => SparsityPattern::Unstructured,
            CliSparsityPattern::N2of4 => SparsityPattern::N2of4,
            CliSparsityPattern::N4of8 => SparsityPattern::N4of8,
        }
    }
}

#[derive(Debug, Parser, Clone)]
#[command(
    name = "oxidize-convert",
    about = "Convert HuggingFace SafeTensors (file or model directory) to GGUF, optionally pruning and joint-quantizing in one pass"
)]
struct Args {
    #[arg(long, help = "Input SafeTensors file or HuggingFace model directory")]
    input: PathBuf,
    #[arg(long, help = "Output GGUF file")]
    output: PathBuf,
    #[arg(long, help = "Model architecture override, such as llama or qwen2")]
    arch: Option<String>,
    #[arg(long, help = "Optional config.json path")]
    config: Option<PathBuf>,
    #[arg(long, help = "Keep original HuggingFace tensor names")]
    no_hf_names: bool,
    #[arg(
        long,
        value_parser = quantization::parse_target,
        help = "Quantize tensors while converting, such as Q4_K_M or Q8_0"
    )]
    target: Option<oxidize_core::gguf::GgufQuantizationType>,
    /// Prune linear weights in the freshly-converted GGUF before the
    /// final quantization pass. Requires `--prune-calibration` for Wanda.
    #[arg(long, value_enum)]
    prune: Option<CliPruneMethod>,
    /// L2-norms cache from the calibration runner (Wanda only).
    #[arg(long)]
    prune_calibration: Option<PathBuf>,
    /// Sparsity fraction in [0, 1) for the prune pass.
    #[arg(long, default_value_t = 0.5)]
    prune_sparsity: f32,
    /// Sparsity pattern for the prune pass.
    #[arg(long, value_enum, default_value_t = CliSparsityPattern::Unstructured)]
    prune_pattern: CliSparsityPattern,
    /// Re-quantize the survivors to this type after pruning (overrides
    /// `--target` if both are set).
    #[arg(long, value_parser = quantization::parse_target)]
    prune_joint_quantize: Option<oxidize_core::gguf::GgufQuantizationType>,
}

impl From<Args> for ConvertOptions {
    fn from(args: Args) -> Self {
        Self {
            input: args.input,
            output: args.output.clone(),
            arch: args.arch,
            config: args.config,
            map_hf_tensor_names: !args.no_hf_names,
            target: args.target,
        }
    }
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    // Phase 1: SafeTensors → GGUF. If --prune is set, write the
    // intermediate to <output>.prerun.gguf; otherwise write directly
    // to the final output.
    let convert_opts: ConvertOptions = args.clone().into();
    let prune_active = args.prune.is_some();
    let final_output = convert_opts.output.clone();
    let intermediate_output = if prune_active {
        let mut p = final_output.clone();
        let stem = p
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".to_string());
        p.set_file_name(format!("{stem}.prerun.gguf"));
        Some(p)
    } else {
        None
    };
    let convert_output = intermediate_output
        .clone()
        .unwrap_or_else(|| final_output.clone());
    let convert_opts = ConvertOptions {
        output: convert_output,
        ..convert_opts
    };
    let summary = run::convert(convert_opts)?;
    println!(
        "Converted {} tensors -> {}",
        summary.tensor_count,
        summary.output.display()
    );

    // Phase 2 (optional): Wanda / magnitude prune.
    if let Some(method) = args.prune {
        let pattern: SparsityPattern = args.prune_pattern.into();
        let joint = args.prune_joint_quantize.or(args.target);
        let intermediate = intermediate_output
            .as_ref()
            .expect("prune_active implies intermediate_output is Some");
        let opts = WandaOptions {
            input: intermediate.clone(),
            output: final_output.clone(),
            calibration: args.prune_calibration,
            sparsity: args.prune_sparsity,
            pattern,
            joint_quantize: joint,
            keep_names: Vec::new(),
            dry_run: false,
            print_timings: true,
        };
        match method {
            CliPruneMethod::Wanda => {
                let report = oxidize_prune::wanda::wanda_prune(opts)?;
                println!(
                    "Wanda-pruned {} of {} tensors -> {}",
                    report.pruned_tensors,
                    report.total_tensors,
                    report.output.display()
                );
            }
            CliPruneMethod::Magnitude => {
                let report = oxidize_prune::wanda::magnitude_prune(opts)?;
                println!(
                    "Magnitude-pruned {} of {} tensors -> {}",
                    report.pruned_tensors,
                    report.total_tensors,
                    report.output.display()
                );
            }
        }
        // Clean up the intermediate file.
        let _ = std::fs::remove_file(intermediate);
    }
    Ok(())
}
