use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
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
}

fn run(args: Args) -> Result<()> {
    let count = convert_safetensors_to_gguf(
        &args.input,
        &args.output,
        &SafetensorsToGgufConfig {
            arch_override: args.arch,
            map_hf_tensor_names: !args.no_hf_names,
            config_path: args.config,
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
