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
                target: a.target.name().to_owned(),
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

pub fn manifest_to_lora_adapters(manifest: LoRAExportManifest) -> Result<Vec<LoRAAdapter>> {
    let rank = manifest.rank;
    if rank == 0 {
        return Err(FinetuneError::Adapter(
            "LoRA manifest rank must be greater than zero".into(),
        ));
    }
    let cfg = FinetuneConfig {
        rank,
        alpha: manifest.alpha_scale * rank as f32,
        ..FinetuneConfig::default()
    };
    manifest
        .adapters
        .into_iter()
        .map(|e| {
            let target = LoRATarget::from_name(&e.target).ok_or_else(|| {
                FinetuneError::Adapter(format!("unknown LoRA target: {:?}", e.target))
            })?;
            let expected_a = rank
                .checked_mul(e.in_dim)
                .ok_or_else(|| FinetuneError::Adapter("LoRA A shape overflows usize".into()))?;
            let expected_b = e
                .out_dim
                .checked_mul(rank)
                .ok_or_else(|| FinetuneError::Adapter("LoRA B shape overflows usize".into()))?;
            if e.lora_a.len() != expected_a || e.lora_b.len() != expected_b {
                return Err(FinetuneError::Adapter(format!(
                    "invalid LoRA tensor lengths for {}: A={} expected {}, B={} expected {}",
                    e.target,
                    e.lora_a.len(),
                    expected_a,
                    e.lora_b.len(),
                    expected_b
                )));
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_rejects_mismatched_adapter_shapes() {
        let manifest = LoRAExportManifest {
            rank: 2,
            alpha_scale: 1.0,
            adapters: vec![LoRAExportEntry {
                target: LoRATarget::OutputHead.name().to_owned(),
                in_dim: 3,
                out_dim: 4,
                lora_a: vec![0.0; 5],
                lora_b: vec![0.0; 8],
            }],
        };

        let error = manifest_to_lora_adapters(manifest).expect_err("invalid A shape must fail");
        assert!(error.to_string().contains("invalid LoRA tensor lengths"));
    }

    #[test]
    fn target_names_round_trip() {
        let targets = [
            LoRATarget::OutputHead,
            LoRATarget::AttentionQ,
            LoRATarget::AttentionV,
            LoRATarget::FfnGate,
            LoRATarget::FfnUp,
        ];
        for target in targets {
            assert_eq!(LoRATarget::from_name(target.name()), Some(target));
        }
    }

    #[test]
    fn manifest_rejects_zero_rank() {
        let manifest = LoRAExportManifest {
            rank: 0,
            alpha_scale: 1.0,
            adapters: Vec::new(),
        };
        let error = manifest_to_lora_adapters(manifest).expect_err("zero rank must fail");
        assert!(error.to_string().contains("rank must be greater than zero"));
    }
}
