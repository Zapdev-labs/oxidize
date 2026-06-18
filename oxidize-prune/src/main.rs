pub mod filter;
pub mod gguf_copy;
pub mod mask;
pub mod wanda;
pub mod writer;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use oxidize_core::gguf::GgufQuantizationType;

use crate::filter::PruneFilter;
use crate::gguf_copy::PruneOptions;
use crate::mask::SparsityPattern;
use crate::wanda::{WandaOptions, magnitude_prune, wanda_prune};

/// Pruning method selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PruneMethod {
    /// Tensor-name substring filtering. Preserves the original
    /// byte-identical tensors; this is the fast path from
    /// `oxidize-prune` pre-Wanda.
    NameFilter,
    /// Wanda: per-output-row pruning by `|W| · ‖X‖_2` with calibration
    /// (Sun et al. 2023, ICLR 2024 — `arxiv:2306.11695`).
    Wanda,
    /// Magnitude: per-output-row pruning by `|W|` (Han et al. 2015,
    /// with the per-row comparison group from Wanda Table 7).
    Magnitude,
}

#[derive(Debug, Parser)]
#[command(
    name = "oxidize-prune",
    about = "Copy a GGUF, optionally pruning weights by Wanda, magnitude, or tensor-name filtering"
)]
struct Args {
    #[arg(long, help = "Input GGUF file")]
    input: PathBuf,
    #[arg(long, help = "Output GGUF file")]
    output: PathBuf,
    /// Pruning method.
    #[arg(
        long,
        value_enum,
        default_value_t = CliPruneMethod::NameFilter,
        help = "Pruning method: name-filter (substring match), wanda (calibrated), or magnitude"
    )]
    method: CliPruneMethod,
    #[arg(long, help = "Keep only tensors whose names contain this text (name-filter only)")]
    keep: Vec<String>,
    #[arg(long, help = "Drop tensors whose names contain this text (name-filter only)")]
    drop: Vec<String>,
    #[arg(
        long,
        help = "L2-norms cache from the calibration runner (Wanda only)"
    )]
    calibration: Option<PathBuf>,
    #[arg(
        long,
        default_value_t = 0.5,
        help = "Sparsity fraction in [0, 1) for Wanda / magnitude"
    )]
    sparsity: f32,
    #[arg(
        long,
        value_enum,
        default_value_t = CliSparsityPattern::Unstructured,
        help = "Sparsity pattern: unstructured | n2of4 | n4of8"
    )]
    pattern: CliSparsityPattern,
    #[arg(
        long,
        help = "Re-quantize the survivors to this GGUF type (e.g. Q4_K_M). Default: preserve original."
    )]
    joint_quantize: Option<String>,
    #[arg(
        long,
        help = "Tensor names (substring) that should never be pruned. Default: token_embd, output, rope, norm."
    )]
    keep_name: Vec<String>,
    #[arg(
        long,
        help = "Print selected and removed tensors without writing output"
    )]
    dry_run: bool,
    #[arg(long, help = "Print per-phase timings (dequant/mask/requant) to stderr")]
    timing: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum CliPruneMethod {
    NameFilter,
    Wanda,
    Magnitude,
}

impl From<CliPruneMethod> for PruneMethod {
    fn from(m: CliPruneMethod) -> Self {
        match m {
            CliPruneMethod::NameFilter => PruneMethod::NameFilter,
            CliPruneMethod::Wanda => PruneMethod::Wanda,
            CliPruneMethod::Magnitude => PruneMethod::Magnitude,
        }
    }
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

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    let method: PruneMethod = args.method.into();
    let pattern: SparsityPattern = args.pattern.into();
    match method {
        PruneMethod::NameFilter => {
            let filter = PruneFilter::new(args.keep, args.drop);
            let summary = gguf_copy::prune_gguf(PruneOptions {
                input: args.input,
                output: args.output,
                filter,
                dry_run: args.dry_run,
            })?;
            for name in &summary.removed {
                println!("drop {name}");
            }
            for name in &summary.kept {
                println!("keep {name}");
            }
            if !summary.dry_run {
                println!(
                    "Pruned {} of {} tensors -> {}",
                    summary.removed.len(),
                    summary.total,
                    summary.output.display()
                );
            }
            Ok(())
        }
        PruneMethod::Magnitude => {
            let joint = match args.joint_quantize.as_deref() {
                Some(s) => Some(parse_qtype(s)?),
                None => None,
            };
            let report = magnitude_prune(WandaOptions {
                input: args.input,
                output: args.output,
                calibration: None,
                sparsity: args.sparsity,
                pattern,
                joint_quantize: joint,
                keep_names: args.keep_name,
                dry_run: args.dry_run,
                print_timings: args.timing,
            })?;
            println!(
                "Magnitude-pruned {} of {} tensors{} -> {}",
                report.pruned_tensors,
                report.total_tensors,
                if report.dry_run { " (dry run)" } else { "" },
                report.output.display()
            );
            Ok(())
        }
        PruneMethod::Wanda => {
            let joint = match args.joint_quantize.as_deref() {
                Some(s) => Some(parse_qtype(s)?),
                None => None,
            };
            if let (Some(calib), false) = (args.calibration.as_ref(), args.dry_run) {
                let cache = wanda::load_l2_norms_cache(calib)?;
                // `validate_calibration` only inspects the GGUF header (tensor
                // names + dims). Memory-map the model so only the header pages
                // fault in — `std::fs::read` here would pull the entire 50–100+
                // GB file into RAM and OOM on large models.
                let mapped = oxidize_core::gguf::load_mapped_gguf(&args.input)
                    .map_err(|e| anyhow::anyhow!(e))?;
                wanda::validate_calibration(&cache, mapped.bytes())?;
            }
            let report = wanda_prune(WandaOptions {
                input: args.input,
                output: args.output,
                calibration: args.calibration,
                sparsity: args.sparsity,
                pattern,
                joint_quantize: joint,
                keep_names: args.keep_name,
                dry_run: args.dry_run,
                print_timings: args.timing,
            })?;
            println!(
                "Wanda-pruned {} of {} tensors{} -> {}",
                report.pruned_tensors,
                report.total_tensors,
                if report.dry_run { " (dry run)" } else { "" },
                report.output.display()
            );
            Ok(())
        }
    }
}

fn parse_qtype(s: &str) -> Result<GgufQuantizationType> {
    let normalized = s.to_ascii_uppercase().replace('-', "_");
    let qtype = match normalized.as_str() {
        "F32" => GgufQuantizationType::F32,
        "F16" => GgufQuantizationType::F16,
        "BF16" => GgufQuantizationType::BF16,
        "Q4_0" => GgufQuantizationType::Q4_0,
        "Q4_1" => GgufQuantizationType::Q4_1,
        "Q5_0" => GgufQuantizationType::Q5_0,
        "Q5_1" => GgufQuantizationType::Q5_1,
        "Q8_0" => GgufQuantizationType::Q8_0,
        "Q2_K" => GgufQuantizationType::Q2_K,
        "Q3_K_S" => GgufQuantizationType::Q3_K_S,
        "Q3_K_M" => GgufQuantizationType::Q3_K_M,
        "Q3_K_L" => GgufQuantizationType::Q3_K_L,
        "Q4_K_S" => GgufQuantizationType::Q4_K_S,
        "Q4_K_M" => GgufQuantizationType::Q4_K_M,
        "Q5_K_S" => GgufQuantizationType::Q5_K_S,
        "Q5_K_M" => GgufQuantizationType::Q5_K_M,
        "Q6_K" => GgufQuantizationType::Q6_K,
        "IQ1_S" => GgufQuantizationType::IQ1_S,
        "IQ1_M" => GgufQuantizationType::IQ1_M,
        "IQ3_S" => GgufQuantizationType::IQ3_S,
        "IQ4_XS" => GgufQuantizationType::IQ4_XS,
        other => anyhow::bail!("unknown quantization type: {other}"),
    };
    Ok(qtype)
}
