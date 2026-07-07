use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::FinetuneConfig;
use crate::error::{FinetuneError, Result};
use crate::lora::{LoRAAdapter, LoRATarget};

#[derive(Debug, Serialize, Deserialize)]
pub struct LoRAExportManifest {
    pub rank: usize,
    pub alpha_scale: f32,
    pub adapters: Vec<LoRAExportEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoRAExportEntry {
    pub target: String,
    pub in_dim: usize,
    pub out_dim: usize,
    pub lora_a: Vec<f32>,
    pub lora_b: Vec<f32>,
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

pub fn adapter_manifest_path(path: impl AsRef<Path>) -> std::path::PathBuf {
    let path = path.as_ref();
    if path.is_dir() {
        path.join("adapter_manifest.json")
    } else {
        path.to_path_buf()
    }
}

pub fn load_adapter_manifest(path: impl AsRef<Path>) -> Result<LoRAExportManifest> {
    let json_path = adapter_manifest_path(path);
    let text = fs::read_to_string(&json_path)
        .map_err(|e| FinetuneError::Model(format!("read {}: {e}", json_path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| FinetuneError::Model(format!("parse {}: {e}", json_path.display())))
}

fn target_from_name(s: &str) -> Result<LoRATarget> {
    match s {
        "OutputHead" => Ok(LoRATarget::OutputHead),
        "AttentionQ" => Ok(LoRATarget::AttentionQ),
        "AttentionV" => Ok(LoRATarget::AttentionV),
        "FfnGate" => Ok(LoRATarget::FfnGate),
        "FfnUp" => Ok(LoRATarget::FfnUp),
        other => Err(FinetuneError::Adapter(format!("unknown LoRA target: {other:?}"))),
    }
}

pub fn manifest_to_lora_adapters(manifest: LoRAExportManifest) -> Result<Vec<LoRAAdapter>> {
    let cfg = FinetuneConfig {
        rank: manifest.rank,
        alpha: manifest.alpha_scale * manifest.rank as f32,
        ..FinetuneConfig::default()
    };
    manifest
        .adapters
        .into_iter()
        .map(|e| {
            let target = target_from_name(&e.target)?;
            let mut ad = LoRAAdapter::new(target, e.in_dim, e.out_dim, &cfg);
            ad.a = e.lora_a;
            ad.b = e.lora_b;
            Ok(ad)
        })
        .collect()
}

/// Load the first adapter entry (LM-head LoRA) from a checkpoint directory.
pub fn load_output_head_lora(path: impl AsRef<Path>) -> Result<LoRAAdapter> {
    let manifest = load_adapter_manifest(path)?;
    let adapters = manifest_to_lora_adapters(manifest)?;
    adapters
        .into_iter()
        .find(|a| a.target == LoRATarget::OutputHead)
        .ok_or_else(|| FinetuneError::Adapter("checkpoint has no OutputHead adapter".into()))
}
