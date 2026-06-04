use crate::conversion::map_hf_tensor_name;
use crate::gguf::{GgufMetadataArray, GgufMetadataType, GgufMetadataValue};
use anyhow::{Context, Result, anyhow, bail};
use safetensors::tensor::{Dtype, SafeTensors};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct SafetensorsToGgufConfig {
    pub arch_override: Option<String>,
    pub map_hf_tensor_names: bool,
    pub config_path: Option<PathBuf>,
}

impl Default for SafetensorsToGgufConfig {
    fn default() -> Self {
        Self {
            arch_override: None,
            map_hf_tensor_names: true,
            config_path: None,
        }
    }
}

#[derive(Debug)]
struct OutputTensor {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: u32,
    data: Vec<u8>,
}

/// Convert a single SafeTensors file or a HuggingFace model directory to GGUF v3.
pub fn convert_safetensors_to_gguf(
    input: &Path,
    output: &Path,
    config: &SafetensorsToGgufConfig,
) -> Result<usize> {
    let (tensors, st_meta, config_dir) = load_all_tensors(input)?;
    let arch = resolve_architecture(config, &st_meta, config_dir.as_deref(), input)?;

    let mut metadata = build_base_metadata(&st_meta, &arch, input);
    let auto_config = config_dir.as_ref().map(|d| d.join("config.json"));
    let cfg_path = config
        .config_path
        .as_ref()
        .or(auto_config.as_ref());
    if let Some(cfg_path) = cfg_path.filter(|p| p.is_file()) {
        merge_hf_config_metadata(&mut metadata, &arch, cfg_path)?;
    }

    let output_tensors = build_output_tensors(&tensors, config.map_hf_tensor_names)?;
    let gguf_bytes = write_gguf(3, &metadata, &output_tensors, 32)?;
    std::fs::write(output, &gguf_bytes)
        .with_context(|| format!("failed to write {}", output.display()))?;
    Ok(output_tensors.len())
}

fn resolve_architecture(
    config: &SafetensorsToGgufConfig,
    st_meta: &BTreeMap<String, String>,
    config_dir: Option<&Path>,
    input: &Path,
) -> Result<String> {
    if let Some(arch) = &config.arch_override {
        return Ok(arch.clone());
    }
    if let Some(dir) = config_dir {
        let cfg_path = dir.join("config.json");
        if cfg_path.is_file() {
            if let Ok(arch) = read_arch_from_hf_config(&cfg_path) {
                return Ok(arch);
            }
        }
    }
    Ok(st_meta
        .get("model_type")
        .or_else(|| st_meta.get("architecture"))
        .cloned()
        .unwrap_or_else(|| default_arch_from_path(input)))
}

fn default_arch_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_owned()
}

fn read_arch_from_hf_config(path: &Path) -> Result<String> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let json: Value = serde_json::from_str(&raw).context("invalid config.json")?;
    let arch = json
        .get("model_type")
        .or_else(|| json.pointer("/text_config/model_type"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned())
        .ok_or_else(|| anyhow!("config.json missing model_type"))?;
    Ok(normalize_hf_arch(&arch))
}

fn normalize_hf_arch(model_type: &str) -> String {
    match model_type.to_ascii_lowercase().as_str() {
        "qwen2" | "qwen2_moe" | "qwen2moe" => "qwen2".to_owned(),
        "qwen3" | "qwen3_moe" => "qwen3".to_owned(),
        "qwen3_5" | "qwen35" => "qwen35".to_owned(),
        "llama" | "mistral" | "gemma" | "phi" | "phi3" | "mixtral" => model_type.to_owned(),
        other => other.to_owned(),
    }
}

fn load_all_tensors(
    input: &Path,
) -> Result<(Vec<(String, Dtype, Vec<usize>, Vec<u8>)>, BTreeMap<String, String>, Option<PathBuf>)> {
    if input.is_file() {
        let (tensors, meta) = load_safetensors_file(input)?;
        return Ok((tensors, meta, None));
    }
    if !input.is_dir() {
        bail!("input path {} is neither a file nor a directory", input.display());
    }

    let index_path = find_weight_index(input)?;
    let mut st_meta = BTreeMap::new();
    let mut out: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();

    if let Some(index_path) = index_path {
        let index_raw = std::fs::read_to_string(&index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;
        let index: Value = serde_json::from_str(&index_raw).context("invalid weight index JSON")?;
        if let Some(meta) = index.get("metadata").and_then(|v| v.as_object()) {
            for (k, v) in meta {
                if let Some(s) = v.as_str() {
                    st_meta.insert(k.clone(), s.to_owned());
                }
            }
        }
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("weight index missing weight_map"))?;

        let mut shard_cache: BTreeMap<String, Vec<(String, Dtype, Vec<usize>, Vec<u8>)>> =
            BTreeMap::new();
        for (tensor_name, shard_name) in weight_map {
            let shard_name = shard_name.as_str().ok_or_else(|| {
                anyhow!("weight_map entry for {tensor_name} is not a string")
            })?;
            let shard_path = input.join(shard_name);
            if !shard_cache.contains_key(shard_name) {
                let (tensors, meta) = load_safetensors_file(&shard_path)?;
                st_meta.extend(meta);
                shard_cache.insert(shard_name.to_owned(), tensors);
            }
            let shard_tensors = shard_cache.get(shard_name).unwrap();
            let found = shard_tensors
                .iter()
                .find(|(n, ..)| n == tensor_name)
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "tensor {tensor_name} not found in shard {}",
                        shard_path.display()
                    )
                })?;
            out.push(found);
        }
        return Ok((out, st_meta, Some(input.to_path_buf())));
    }

    let mut paths: Vec<PathBuf> = std::fs::read_dir(input)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        bail!("no .safetensors files found in {}", input.display());
    }
    for path in paths {
        let (tensors, meta) = load_safetensors_file(&path)?;
        st_meta.extend(meta);
        for tensor in tensors {
            if out.iter().any(|(n, ..)| n == &tensor.0) {
                bail!("duplicate tensor {} in directory {}", tensor.0, input.display());
            }
            out.push(tensor);
        }
    }
    Ok((out, st_meta, Some(input.to_path_buf())))
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

fn load_safetensors_file(
    path: &Path,
) -> Result<(Vec<(String, Dtype, Vec<usize>, Vec<u8>)>, BTreeMap<String, String>)> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    // SAFETY: read-only mapping; file handle kept alive for the mapping's lifetime.
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .with_context(|| format!("failed to mmap {}", path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| anyhow!("failed to parse SafeTensors: {e:?}"))?;
    let meta = read_safetensors_metadata(&mmap)?;
    let mut tensors = Vec::with_capacity(st.len());
    for (name, view) in st.tensors() {
        tensors.push((
            name.to_owned(),
            view.dtype(),
            view.shape().to_vec(),
            view.data().to_vec(),
        ));
    }
    Ok((tensors, meta))
}

fn read_safetensors_metadata(mmap: &[u8]) -> Result<BTreeMap<String, String>> {
    if mmap.len() < 8 {
        return Ok(BTreeMap::new());
    }
    let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    if 8 + header_len > mmap.len() {
        bail!("safetensors header length exceeds file size");
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

fn build_base_metadata(
    st_meta: &BTreeMap<String, String>,
    arch: &str,
    input_path: &Path,
) -> BTreeMap<String, GgufMetadataValue> {
    let mut meta: BTreeMap<String, GgufMetadataValue> = BTreeMap::new();
    meta.insert(
        "general.architecture".to_owned(),
        GgufMetadataValue::String(arch.to_owned()),
    );

    if let Some(name) = st_meta
        .get("model_name")
        .or_else(|| st_meta.get("name"))
        .cloned()
        .or_else(|| {
            input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_owned())
        })
    {
        meta.insert("general.name".to_owned(), GgufMetadataValue::String(name));
    }

    for (key, value) in st_meta {
        if matches!(key.as_str(), "model_type" | "architecture" | "model_name" | "name") {
            continue;
        }
        meta.insert(
            format!("general.safetensors.{key}"),
            GgufMetadataValue::String(value.clone()),
        );
    }
    meta
}

fn merge_hf_config_metadata(
    meta: &mut BTreeMap<String, GgufMetadataValue>,
    arch: &str,
    config_path: &Path,
) -> Result<()> {
    let raw = std::fs::read_to_string(config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let json: Value = serde_json::from_str(&raw).context("invalid config.json")?;
    let cfg = json
        .get("text_config")
        .filter(|v| v.is_object())
        .unwrap_or(&json);

    let prefix = |suffix: &str| format!("{arch}.{suffix}");
    let insert_u32 = |meta: &mut BTreeMap<_, _>, key: &str, field: &str| {
        if let Some(v) = cfg.get(field).and_then(json_u32) {
            meta.insert(key.to_owned(), GgufMetadataValue::Uint32(v));
        }
    };
    let insert_f32 = |meta: &mut BTreeMap<_, _>, key: &str, field: &str| {
        if let Some(v) = cfg.get(field).and_then(json_f32) {
            meta.insert(key.to_owned(), GgufMetadataValue::Float32(v));
        }
    };

    insert_u32(meta, &prefix("embedding_length"), "hidden_size");
    insert_u32(meta, &prefix("block_count"), "num_hidden_layers");
    insert_u32(meta, &prefix("feed_forward_length"), "intermediate_size");
    insert_u32(meta, &prefix("attention.head_count"), "num_attention_heads");
    insert_u32(meta, &prefix("attention.head_count_kv"), "num_key_value_heads");
    insert_u32(meta, &prefix("vocab_size"), "vocab_size");
    insert_u32(
        meta,
        &prefix("context_length"),
        "max_position_embeddings",
    );
    insert_f32(
        meta,
        &prefix("attention.layer_norm_rms_epsilon"),
        "rms_norm_eps",
    );
    insert_f32(meta, &prefix("rope.freq_base"), "rope_theta");
    insert_u32(meta, &prefix("attention.sliding_window"), "sliding_window");
    insert_u32(meta, &prefix("expert_count"), "num_experts");
    insert_u32(meta, &prefix("expert_used_count"), "num_experts_per_tok");

    if let Some(model_type) = cfg.get("model_type").and_then(|v| v.as_str()) {
        meta.insert(
            "general.architecture".to_owned(),
            GgufMetadataValue::String(normalize_hf_arch(model_type)),
        );
    }
    Ok(())
}

fn json_u32(v: &Value) -> Option<u32> {
    v.as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .or_else(|| v.as_i64().and_then(|n| u32::try_from(n).ok()))
}

fn json_f32(v: &Value) -> Option<f32> {
    v.as_f64().map(|n| n as f32)
}

fn build_output_tensors(
    tensors: &[(String, Dtype, Vec<usize>, Vec<u8>)],
    map_hf_names: bool,
) -> Result<Vec<OutputTensor>> {
    let mut out: Vec<OutputTensor> = Vec::with_capacity(tensors.len());
    for (name, dtype, shape, raw_data) in tensors {
        let output_name = if map_hf_names {
            map_hf_tensor_name(name)
        } else {
            name.clone()
        };
        let dimensions: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        let (ggml_type, data) = match dtype {
            Dtype::F32 => (0_u32, raw_data.clone()),
            Dtype::F16 => (1_u32, raw_data.clone()),
            Dtype::BF16 => (0_u32, bf16_to_f32(raw_data)),
            other => bail!("unsupported SafeTensors dtype {other:?} in tensor {name}"),
        };
        out.push(OutputTensor {
            name: output_name,
            dimensions,
            ggml_type,
            data,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<u8> {
    assert!(bytes.len() % 2 == 0, "BF16 buffer length must be even");
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn write_gguf(
    version: u32,
    metadata: &BTreeMap<String, GgufMetadataValue>,
    tensors: &[OutputTensor],
    alignment: u64,
) -> Result<Vec<u8>> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("invalid GGUF alignment: {alignment}");
    }

    let mut relative_offsets: Vec<u64> = Vec::with_capacity(tensors.len());
    let mut cursor: u64 = 0;
    for tensor in tensors {
        cursor = align_up(cursor, alignment)?;
        relative_offsets.push(cursor);
        cursor = cursor
            .checked_add(tensor.data.len() as u64)
            .ok_or_else(|| anyhow!("tensor data offset overflow"))?;
    }

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    for (key, value) in metadata {
        write_string(&mut out, key);
        write_metadata_value(&mut out, value)?;
    }

    for (tensor, &rel_offset) in tensors.iter().zip(relative_offsets.iter()) {
        write_string(&mut out, &tensor.name);
        out.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dim in &tensor.dimensions {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&tensor.ggml_type.to_le_bytes());
        out.extend_from_slice(&rel_offset.to_le_bytes());
    }

    pad_to(&mut out, alignment)?;
    let data_start = out.len() as u64;

    for (tensor, &rel_offset) in tensors.iter().zip(relative_offsets.iter()) {
        let target = (data_start + rel_offset) as usize;
        if out.len() < target {
            out.resize(target, 0);
        }
        out.extend_from_slice(&tensor.data);
        pad_to(&mut out, alignment)?;
    }

    Ok(out)
}

fn write_metadata_value(out: &mut Vec<u8>, value: &GgufMetadataValue) -> Result<()> {
    let vtype = metadata_type(value);
    out.extend_from_slice(&(vtype as u32).to_le_bytes());
    write_metadata_payload(out, value, vtype)
}

fn write_metadata_payload(
    out: &mut Vec<u8>,
    value: &GgufMetadataValue,
    vtype: GgufMetadataType,
) -> Result<()> {
    match (vtype, value) {
        (GgufMetadataType::Uint8, GgufMetadataValue::Uint8(v)) => out.push(*v),
        (GgufMetadataType::Int8, GgufMetadataValue::Int8(v)) => out.push(*v as u8),
        (GgufMetadataType::Uint16, GgufMetadataValue::Uint16(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Int16, GgufMetadataValue::Int16(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Uint32, GgufMetadataValue::Uint32(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Int32, GgufMetadataValue::Int32(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Float32, GgufMetadataValue::Float32(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Bool, GgufMetadataValue::Bool(v)) => out.push(u8::from(*v)),
        (GgufMetadataType::String, GgufMetadataValue::String(v)) => write_string(out, v),
        (GgufMetadataType::Array, GgufMetadataValue::Array(arr)) => {
            write_metadata_array(out, arr)?
        }
        (GgufMetadataType::Uint64, GgufMetadataValue::Uint64(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Int64, GgufMetadataValue::Int64(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Float64, GgufMetadataValue::Float64(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        _ => bail!("metadata type mismatch"),
    }
    Ok(())
}

fn write_metadata_array(out: &mut Vec<u8>, arr: &GgufMetadataArray) -> Result<()> {
    out.extend_from_slice(&(arr.element_type as u32).to_le_bytes());
    out.extend_from_slice(&(arr.values.len() as u64).to_le_bytes());
    for value in &arr.values {
        write_metadata_payload(out, value, arr.element_type)?;
    }
    Ok(())
}

fn metadata_type(value: &GgufMetadataValue) -> GgufMetadataType {
    match value {
        GgufMetadataValue::Uint8(_) => GgufMetadataType::Uint8,
        GgufMetadataValue::Int8(_) => GgufMetadataType::Int8,
        GgufMetadataValue::Uint16(_) => GgufMetadataType::Uint16,
        GgufMetadataValue::Int16(_) => GgufMetadataType::Int16,
        GgufMetadataValue::Uint32(_) => GgufMetadataType::Uint32,
        GgufMetadataValue::Int32(_) => GgufMetadataType::Int32,
        GgufMetadataValue::Float32(_) => GgufMetadataType::Float32,
        GgufMetadataValue::Bool(_) => GgufMetadataType::Bool,
        GgufMetadataValue::String(_) => GgufMetadataType::String,
        GgufMetadataValue::Array(_) => GgufMetadataType::Array,
        GgufMetadataValue::Uint64(_) => GgufMetadataType::Uint64,
        GgufMetadataValue::Int64(_) => GgufMetadataType::Int64,
        GgufMetadataValue::Float64(_) => GgufMetadataType::Float64,
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn pad_to(out: &mut Vec<u8>, alignment: u64) -> Result<()> {
    let aligned = align_up(out.len() as u64, alignment)? as usize;
    out.resize(aligned, 0);
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or_else(|| anyhow!("alignment overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::parse_gguf;
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;
    use std::io::Write;

    fn write_test_safetensors(path: &Path, tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) {
        let views: HashMap<String, TensorView> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (
                    name.to_string(),
                    TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        let bytes = safetensors::tensor::serialize(&views, &None).unwrap();
        let mut f = File::create(path).unwrap();
        f.write_all(&bytes).unwrap();
    }

    fn tmp_path(suffix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("oxidize-st2gguf-{nanos}{suffix}"))
    }

    #[test]
    fn maps_hf_tensor_names() {
        let input = tmp_path(".safetensors");
        let output = tmp_path(".gguf");
        let data = vec![0u8; 64];
        write_test_safetensors(
            &input,
            &[(
                "model.layers.0.self_attn.q_proj.weight",
                Dtype::F32,
                vec![4, 4],
                data,
            )],
        );
        convert_safetensors_to_gguf(
            &input,
            &output,
            &SafetensorsToGgufConfig {
                arch_override: Some("llama".into()),
                ..Default::default()
            },
        )
        .unwrap();
        let parsed = parse_gguf(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(parsed.tensor_infos[0].name, "blk.0.attn_q.weight");
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }

    #[test]
    fn reads_config_json_metadata() {
        let dir = std::env::temp_dir().join(format!(
            "oxidize-st2gguf-cfg-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("model.safetensors");
        let output = tmp_path(".gguf");
        write_test_safetensors(&input, &[("weight", Dtype::F32, vec![2, 2], vec![0u8; 16])]);
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"llama","hidden_size":128,"num_hidden_layers":2,"num_attention_heads":4,"vocab_size":256}"#,
        )
        .unwrap();
        convert_safetensors_to_gguf(&dir, &output, &SafetensorsToGgufConfig::default()).unwrap();
        let parsed = parse_gguf(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(
            parsed.metadata.get("llama.embedding_length"),
            Some(&GgufMetadataValue::Uint32(128))
        );
        assert_eq!(
            parsed.metadata.get("llama.block_count"),
            Some(&GgufMetadataValue::Uint32(2))
        );
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_file(output);
    }
}
