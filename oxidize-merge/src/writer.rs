use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use safetensors::tensor::{Dtype, TensorView};

#[derive(Debug, Clone)]
pub struct OutputTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

pub(crate) enum MergeWriter {
    Single {
        path: PathBuf,
        tensors: Vec<OutputTensor>,
        metadata: BTreeMap<String, String>,
    },
    Sharded(Box<ShardWriter>),
}

impl MergeWriter {
    pub fn new(output: &Path, max_shard_bytes: u64, metadata: BTreeMap<String, String>) -> Result<Self> {
        if output.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            return Ok(Self::Single {
                path: output.to_path_buf(),
                tensors: Vec::new(),
                metadata,
            });
        }
        fs::create_dir_all(output)?;
        Ok(Self::Sharded(Box::new(ShardWriter::new(
            output,
            max_shard_bytes,
            metadata,
        )?)))
    }

    pub fn push(&mut self, tensor: OutputTensor) -> Result<()> {
        match self {
            Self::Single { tensors, .. } => {
                tensors.push(tensor);
                Ok(())
            }
            Self::Sharded(writer) => writer.push(tensor),
        }
    }

    pub fn finish(self) -> Result<usize> {
        match self {
            Self::Single {
                path,
                tensors,
                metadata,
            } => {
                if tensors.is_empty() {
                    bail!("no tensors were written");
                }
                write_safetensors_file(&path, &tensors, &metadata)?;
                Ok(tensors.len())
            }
            Self::Sharded(writer) => writer.finish(),
        }
    }
}

pub(crate) struct ShardWriter {
    output_dir: PathBuf,
    max_shard_bytes: u64,
    metadata: BTreeMap<String, String>,
    current_shard: Vec<OutputTensor>,
    current_bytes: u64,
    shard_index: usize,
    weight_map: BTreeMap<String, String>,
    total_tensors: usize,
}

impl ShardWriter {
    fn new(
        output_dir: &Path,
        max_shard_bytes: u64,
        metadata: BTreeMap<String, String>,
    ) -> Result<Self> {
        if max_shard_bytes == 0 {
            bail!("max shard size must be greater than zero");
        }
        Ok(Self {
            output_dir: output_dir.to_path_buf(),
            max_shard_bytes,
            metadata,
            current_shard: Vec::new(),
            current_bytes: 0,
            shard_index: 0,
            weight_map: BTreeMap::new(),
            total_tensors: 0,
        })
    }

    fn push(&mut self, tensor: OutputTensor) -> Result<()> {
        let tensor_bytes = tensor.data.len() as u64;
        if !self.current_shard.is_empty()
            && self.current_bytes.saturating_add(tensor_bytes) > self.max_shard_bytes
        {
            self.flush_shard()?;
        }
        self.current_bytes = self.current_bytes.saturating_add(tensor_bytes);
        self.current_shard.push(tensor);
        Ok(())
    }

    fn finish(mut self) -> Result<usize> {
        if !self.current_shard.is_empty() {
            self.flush_shard()?;
        }
        if self.weight_map.is_empty() {
            bail!("no tensors were written");
        }

        let total_shards = self.shard_index;
        let mut final_weight_map = BTreeMap::new();
        for (tensor_name, shard_name) in self.weight_map {
            let updated = shard_name.replace("of-?????", &format!("of-{total_shards:05}"));
            if updated != shard_name {
                let old = self.output_dir.join(&shard_name);
                let new = self.output_dir.join(&updated);
                if old.exists() {
                    fs::rename(&old, &new)?;
                }
            }
            final_weight_map.insert(tensor_name, updated);
        }

        let index_path = self.output_dir.join("model.safetensors.index.json");
        let index = serde_json::json!({
            "metadata": self.metadata,
            "weight_map": final_weight_map,
        });
        let mut file = File::create(&index_path)
            .with_context(|| format!("failed to create {}", index_path.display()))?;
        file.write_all(serde_json::to_string_pretty(&index)?.as_bytes())?;
        Ok(self.total_tensors)
    }

    fn flush_shard(&mut self) -> Result<()> {
        let shard_name = format!("model-{:05}-of-?????.safetensors", self.shard_index);
        let shard_path = self.output_dir.join(&shard_name);
        write_safetensors_file(&shard_path, &self.current_shard, &self.metadata)?;

        for tensor in &self.current_shard {
            self.weight_map
                .insert(tensor.name.clone(), shard_name.clone());
            self.total_tensors += 1;
        }

        self.shard_index += 1;
        self.current_shard.clear();
        self.current_bytes = 0;
        Ok(())
    }
}

fn write_safetensors_file(
    path: &Path,
    tensors: &[OutputTensor],
    metadata: &BTreeMap<String, String>,
) -> Result<()> {
    let mut views = BTreeMap::new();
    for tensor in tensors {
        let view = TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data)
            .with_context(|| format!("failed to build tensor view for {}", tensor.name))?;
        views.insert(tensor.name.clone(), view);
    }
    let meta = if metadata.is_empty() {
        None
    } else {
        Some(metadata.iter().map(|(k, v)| (k.clone(), v.clone())).collect::<HashMap<_, _>>())
    };
    let bytes = safetensors::tensor::serialize(&views, &meta)
        .context("failed to serialize safetensors shard")?;
    let mut file = File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    file.write_all(&bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_single_shard_file() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("merged.safetensors");
        let mut writer = MergeWriter::new(&out, u64::MAX, BTreeMap::new()).unwrap();
        writer
            .push(OutputTensor {
                name: "a".to_owned(),
                dtype: Dtype::F32,
                shape: vec![2],
                data: vec![0, 0, 128, 63, 0, 0, 0, 64],
            })
            .unwrap();
        let count = writer.finish().unwrap();
        assert_eq!(count, 1);
        assert!(out.exists());
    }
}
