use std::fs;
use std::path::Path;

use serde::Serialize;

use crate::error::{FinetuneError, Result};
use crate::lora::LoRAAdapter;

#[derive(Debug, Serialize)]
struct LoRAExportManifest {
    rank: usize,
    alpha_scale: f32,
    adapters: Vec<LoRAExportEntry>,
}

#[derive(Debug, Serialize)]
struct LoRAExportEntry {
    target: String,
    in_dim: usize,
    out_dim: usize,
    lora_a: Vec<f32>,
    lora_b: Vec<f32>,
}

pub fn export_lora_gguf(
    out_dir: impl AsRef<Path>,
    adapters: &[LoRAAdapter],
    rank: usize,
    alpha_scale: f32,
) -> Result<()> {
    let dir = out_dir.as_ref();
    fs::create_dir_all(dir).map_err(|e| FinetuneError::Model(e.to_string()))?;
    let manifest = LoRAExportManifest {
        rank,
        alpha_scale,
        adapters: adapters
            .iter()
            .map(|a| LoRAExportEntry {
                target: format!("{:?}", a.target),
                in_dim: a.in_dim,
                out_dim: a.out_dim,
                lora_a: a.a.clone(),
                lora_b: a.b.clone(),
            })
            .collect(),
    };
    let json =
        serde_json::to_string_pretty(&manifest).map_err(|e| FinetuneError::Model(e.to_string()))?;
    fs::write(dir.join("adapter_manifest.json"), json)
        .map_err(|e| FinetuneError::Model(e.to_string()))?;
    Ok(())
}
