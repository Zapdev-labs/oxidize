use std::path::PathBuf;

use anyhow::Result;
use oxidize_core::gguf::GgufQuantizationType;
use oxidize_core::safetensors_to_gguf::{SafetensorsToGgufConfig, convert_safetensors_to_gguf};

#[derive(Debug)]
pub struct ConvertOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    pub arch: Option<String>,
    pub config: Option<PathBuf>,
    pub map_hf_tensor_names: bool,
    pub target: Option<GgufQuantizationType>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ConvertSummary {
    pub output: PathBuf,
    pub tensor_count: usize,
}

pub fn convert(options: ConvertOptions) -> Result<ConvertSummary> {
    let count = convert_safetensors_to_gguf(
        &options.input,
        &options.output,
        &SafetensorsToGgufConfig {
            arch_override: options.arch,
            map_hf_tensor_names: options.map_hf_tensor_names,
            config_path: options.config,
            target_quantization: options.target,
        },
    )?;
    Ok(ConvertSummary {
        output: options.output,
        tensor_count: count,
    })
}
