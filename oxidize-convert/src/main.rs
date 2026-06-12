use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use oxidize_core::gguf::GgufQuantizationType;
use oxidize_core::safetensors_to_gguf::{SafetensorsToGgufConfig, convert_safetensors_to_gguf};

#[derive(Debug, Parser)]
#[command(
    name = "oxidize-convert",
    about = "Convert HuggingFace SafeTensors (file or model directory) to GGUF"
)]
struct Args {
    /// Input SafeTensors file (.safetensors) or HuggingFace model directory
    #[arg(long)]
    input: PathBuf,
    /// Output GGUF file (.gguf)
    #[arg(long)]
    output: PathBuf,
    /// Model architecture (e.g. llama, qwen2). Overrides config.json / SafeTensors metadata.
    #[arg(long)]
    arch: Option<String>,
    /// Optional path to config.json (default: <input>/config.json for directories)
    #[arg(long)]
    config: Option<PathBuf>,
    /// Keep original HuggingFace tensor names instead of mapping to GGUF names
    #[arg(long)]
    no_hf_names: bool,
    /// Quantize tensors while converting (e.g. Q4_K_M, Q8_0)
    #[arg(long)]
    target: Option<String>,
}

fn parse_target(s: &str) -> anyhow::Result<GgufQuantizationType> {
    match s.to_ascii_uppercase().as_str() {
        "Q4_K_M" => Ok(GgufQuantizationType::Q4_K_M),
        "Q4_K_S" => Ok(GgufQuantizationType::Q4_K_S),
        "Q4_0" => Ok(GgufQuantizationType::Q4_0),
        "Q8_0" => Ok(GgufQuantizationType::Q8_0),
        "Q6_K" => Ok(GgufQuantizationType::Q6_K),
        "F16" => Ok(GgufQuantizationType::F16),
        "F32" => Ok(GgufQuantizationType::F32),
        other => anyhow::bail!("unsupported --target quantization: {other}"),
    }
}

fn run(args: Args) -> Result<()> {
    let count = convert_safetensors_to_gguf(
        &args.input,
        &args.output,
        &SafetensorsToGgufConfig {
            arch_override: args.arch,
            map_hf_tensor_names: !args.no_hf_names,
            config_path: args.config,
            target_quantization: args
                .target
                .as_deref()
                .map(parse_target)
                .transpose()?,
        },
    )?;
    println!("Converted {} tensors → {}", count, args.output.display());
    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}
