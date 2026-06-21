use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;
use serde_json::Value;

/// Merge per-shard metadata, erroring on conflicting values for the same key
/// rather than silently letting a later shard overwrite an earlier one.
fn merge_metadata(
    into: &mut BTreeMap<String, String>,
    from: BTreeMap<String, String>,
) -> Result<()> {
    for (k, v) in from {
        match into.get(&k) {
            Some(existing) if *existing != v => {
                bail!("conflicting metadata for key {k:?}: {existing:?} vs {v:?}");
            }
            _ => {
                into.insert(k, v);
            }
        }
    }
    Ok(())
}

/// Reject shard names that are not a plain file name within the model
/// directory (absolute paths, parent escapes, or nested directories), so a
/// malicious index JSON cannot read arbitrary files via `dir.join(name)`.
fn validate_shard_name(name: &str) -> Result<()> {
    let p = Path::new(name);
    let mut components = p.components();
    match (components.next(), components.next()) {
        (Some(std::path::Component::Normal(_)), None) => Ok(()),
        _ => bail!("invalid shard name {name:?} in weight index (must be a plain file name)"),
    }
}

#[derive(Debug)]
pub struct MappedShard {
    mmap: Mmap,
    tensors: BTreeMap<String, TensorRef>,
}

fn map_readonly(file: &File) -> std::io::Result<Mmap> {
    // SAFETY: read-only mapping; caller owns the `Mmap` for its lifetime.
    unsafe { Mmap::map(file) }
}

impl MappedShard {
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let mmap =
            map_readonly(&file).with_context(|| format!("failed to mmap {}", path.display()))?;
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| anyhow!("failed to parse SafeTensors {}: {e:?}", path.display()))?;
        let mut tensors = BTreeMap::new();
        for (name, view) in st.tensors() {
            let relative_offset = view.data().as_ptr() as usize - mmap.as_ptr() as usize;
            tensors.insert(
                name.to_string(),
                TensorRef {
                    name: name.to_string(),
                    shape: view.shape().to_vec(),
                    dtype: view.dtype(),
                    shard_path: path.to_path_buf(),
                    absolute_offset: relative_offset,
                    size_bytes: view.data().len(),
                },
            );
        }
        Ok(Self { mmap, tensors })
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("tensor {name} missing from shard"))?;
        Ok(&self.mmap[info.absolute_offset..info.absolute_offset + info.size_bytes])
    }
}

#[derive(Debug, Clone)]
pub struct TensorRef {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub shard_path: PathBuf,
    pub absolute_offset: usize,
    pub size_bytes: usize,
}

#[derive(Debug)]
pub struct ModelIndex {
    pub root: PathBuf,
    pub tensors: BTreeMap<String, TensorRef>,
    pub metadata: BTreeMap<String, String>,
}

impl ModelIndex {
    pub fn open(path: &Path) -> Result<Self> {
        if path.is_file() {
            return Self::from_single_file(path);
        }
        if path.is_dir() {
            return Self::from_directory(path);
        }
        bail!(
            "model path {} is neither a file nor a directory",
            path.display()
        )
    }

    fn from_single_file(path: &Path) -> Result<Self> {
        let shard = MappedShard::open(path)?;
        let tensors = shard.tensors;
        let metadata = read_file_metadata(path)?;
        Ok(Self {
            root: path.parent().unwrap_or(path).to_path_buf(),
            tensors,
            metadata,
        })
    }

    fn from_directory(dir: &Path) -> Result<Self> {
        let index_path = find_weight_index(dir)?;
        if let Some(index_path) = index_path {
            return Self::from_weight_index(dir, &index_path);
        }

        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            bail!("no .safetensors files found in {}", dir.display());
        }

        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for shard_path in paths {
            let shard = MappedShard::open(&shard_path)?;
            for (name, info) in shard.tensors {
                if tensors.contains_key(&name) {
                    bail!("duplicate tensor {name} in directory {}", dir.display());
                }
                tensors.insert(name, info);
            }
            merge_metadata(&mut metadata, read_file_metadata(&shard_path)?)?;
        }
        Ok(Self {
            root: dir.to_path_buf(),
            tensors,
            metadata,
        })
    }

    fn from_weight_index(dir: &Path, index_path: &Path) -> Result<Self> {
        let index_raw = std::fs::read_to_string(index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;
        let index: Value =
            serde_json::from_str(&index_raw).context("invalid safetensors index JSON")?;
        let mut metadata = BTreeMap::new();
        if let Some(meta) = index.get("metadata").and_then(|v| v.as_object()) {
            for (k, v) in meta {
                if let Some(s) = v.as_str() {
                    metadata.insert(k.clone(), s.to_owned());
                }
            }
        }
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("weight index missing weight_map"))?;

        let mut shard_cache: BTreeMap<String, MappedShard> = BTreeMap::new();
        let mut tensors = BTreeMap::new();
        for (tensor_name, shard_name_val) in weight_map {
            let shard_name = shard_name_val
                .as_str()
                .ok_or_else(|| anyhow!("weight_map entry for {tensor_name} is not a string"))?;
            if !shard_cache.contains_key(shard_name) {
                validate_shard_name(shard_name)?;
                let shard_path = dir.join(shard_name);
                shard_cache.insert(shard_name.to_owned(), MappedShard::open(&shard_path)?);
                merge_metadata(&mut metadata, read_file_metadata(&shard_path)?)?;
            }
            let shard = shard_cache.get(shard_name).unwrap();
            let info = shard
                .tensors
                .get(tensor_name)
                .ok_or_else(|| anyhow!("tensor {tensor_name} missing from shard {shard_name}"))?
                .clone();
            tensors.insert(tensor_name.clone(), info);
        }
        Ok(Self {
            root: dir.to_path_buf(),
            tensors,
            metadata,
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }
}

pub struct ShardCache {
    shards: BTreeMap<PathBuf, MappedShard>,
}

impl ShardCache {
    pub fn new() -> Self {
        Self {
            shards: BTreeMap::new(),
        }
    }

    pub fn tensor_bytes(&mut self, tensor: &TensorRef) -> Result<&[u8]> {
        if !self.shards.contains_key(&tensor.shard_path) {
            let shard = MappedShard::open(&tensor.shard_path)?;
            self.shards.insert(tensor.shard_path.clone(), shard);
        }
        self.shards
            .get(&tensor.shard_path)
            .unwrap()
            .tensor_bytes(&tensor.name)
    }
}

impl Default for ShardCache {
    fn default() -> Self {
        Self::new()
    }
}

fn find_weight_index(dir: &Path) -> Result<Option<PathBuf>> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".safetensors.index.json"))
        })
        .collect();
    candidates.sort();
    Ok(candidates.into_iter().next())
}

fn read_file_metadata(path: &Path) -> Result<BTreeMap<String, String>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mmap = map_readonly(&file).with_context(|| format!("failed to mmap {}", path.display()))?;
    if mmap.len() < 8 {
        return Ok(BTreeMap::new());
    }
    let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    if 8 + header_len > mmap.len() {
        return Ok(BTreeMap::new());
    }
    let header_json: Value = serde_json::from_slice(&mmap[8..8 + header_len])
        .context("failed to parse safetensors header JSON")?;
    let Some(meta_obj) = header_json.get("__metadata__").and_then(|v| v.as_object()) else {
        return Ok(BTreeMap::new());
    };
    Ok(meta_obj
        .iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
        .collect())
}

pub fn is_blendable(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;
    use std::io::Write;

    fn write_test_safetensors(path: &Path, name: &str, values: &[f32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensor = TensorView::new(Dtype::F32, vec![values.len()], &bytes).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(name.to_owned(), tensor);
        let st = safetensors::tensor::serialize(&tensors, &None).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&st).unwrap();
    }

    #[test]
    fn opens_single_file_model() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        write_test_safetensors(&path, "weight", &[1.0, 2.0, 3.0]);
        let index = ModelIndex::open(&path).unwrap();
        assert_eq!(index.tensors.len(), 1);
        assert!(index.tensors.contains_key("weight"));
    }
}
